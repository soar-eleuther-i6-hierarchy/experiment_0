"""
The detector x confound AUROC grid and its statistics — the reusable, checkpoint-free
toolkit behind retrieval scoring.

Labels enter ONLY here (AUROC scoring + the dispersion readout); the detectors in
`scoring.core.detectors` never saw them. Everything is threshold-free: each cell asks
"does detector D rank a random recovered is-a pair above a random recovered class-C
pair?" — an AUROC, which is prevalence-independent, so the thin confound columns are
handled by honest wide CIs, not by tuning.

Statistics invariants realised here:
  - orientation is fixed in `scoring.core.registry` — the scorer reads it, never argmaxes over sign;
  - AUROC CIs are computed on the logit scale with an [clamp, 1-clamp] guard so perfect
    separation gives a finite interval inside [0, 1];
  - the within-seed CI is a cluster bootstrap BY FEATURE (resample features, form pairs
    only among sampled features), never a pair bootstrap;
  - the two controls (random scalar, shuffled label) must land at AUROC ~0.5.

The trained driver that feeds real checkpoint activations through this toolkit lives in
`scoring.trained.retrieval`.
"""

from __future__ import annotations

import logging
import math

import torch

logger = logging.getLogger(__name__)

from scoring.core.detectors import DetectorInputs
from scoring.core.registry import (
    CONSTANTS,
    POSITIVE_LABEL,
    SCORED_COLUMNS,
    SYMMETRIC_DETECTORS,
)
from toygen import labels

DT = torch.float64

# The held-out detector draw is offset from the training seed so its tokens are disjoint
# from what the SAE trained on. Non-zero by construction.
HELD_OUT_SEED_OFFSET = 10_000


# --------------------------------------------------------------------------
# reduce to the recovered latents / build the pair universe
# --------------------------------------------------------------------------
def reduce_to_recovered(held_acts: torch.Tensor, oriented: torch.Tensor, raw: torch.Tensor,
                        match: torch.Tensor, recovered: torch.Tensor,
                        h: torch.Tensor | None = None, b_dec: torch.Tensor | None = None,
                        tokens: torch.Tensor | None = None, vocab: int = 0
                        ) -> tuple[list[int], DetectorInputs, dict[int, int]]:
    """Gather the matched latent's columns/rows for every recovered feature.

    A feature is in the universe iff it is recovered and has a match (>=0). The returned
    `DetectorInputs` is indexed by recovered-feature POSITION k (0..R-1); `index_map`
    maps the true feature id -> k.
    """
    feats = [int(f) for f in range(int(match.shape[0]))
             if bool(recovered[f]) and int(match[f]) >= 0]
    idx = torch.tensor([int(match[f]) for f in feats], dtype=torch.long)
    di = DetectorInputs(
        acts_rec=held_acts[:, idx], W_unit=oriented[idx], W_raw=raw[idx],
        h=h, b_dec=b_dec, tokens=tokens, vocab=vocab,
    )
    return feats, di, {f: k for k, f in enumerate(feats)}


def pair_frame(recovered_feats: list[int], pair_labels: torch.Tensor
               ) -> tuple[list[tuple[int, int]], torch.Tensor]:
    """All ordered off-diagonal pairs (positions), with the single class INDEX per pair.

    Pairs are POSITIONS into `recovered_feats`; the label is read straight off
    `pair_labels[feat_a, feat_b]` (the only place truth enters the pipeline).

    Returns one class INDEX per pair (not a bitmask): every ordered pair carries exactly
    one class. Self-pairs (a == b) are EXCLUDED — the answer-key diagonal is 0, which
    equals the is_a index, so a self-pair would be miscounted as is_a. Use `class_members`
    to test membership.
    """
    R = len(recovered_feats)
    pairs, ys = [], []
    for a in range(R):
        for b in range(R):
            if a == b:
                continue
            pairs.append((a, b))
            ys.append(int(pair_labels[recovered_feats[a], recovered_feats[b]]))
    return pairs, torch.tensor(ys, dtype=torch.long)


def class_members(y_label: torch.Tensor, name: str) -> torch.Tensor:
    """Boolean mask over pairs: which ones carry the `name` class."""
    return y_label == labels._index(name)


def split_scores(vals: torch.Tensor, y_label: torch.Tensor,
                 pos_name: str, col_name: str
                 ) -> tuple[torch.Tensor, torch.Tensor]:
    """Split per-pair scores into the positive and negative populations of one cell.

    Single-label is mutually exclusive: a pair carries exactly one class, so the positive
    (`pos_name`) and negative (`col_name`) populations are disjoint by construction — no
    overlap to exclude.
    """
    return vals[class_members(y_label, pos_name)], vals[class_members(y_label, col_name)]


# --------------------------------------------------------------------------
# AUROC + logit CI
# --------------------------------------------------------------------------
def _tie_averaged_ranks(x: torch.Tensor) -> torch.Tensor:
    """1-based ranks with tied values sharing their mean rank (scipy `rankdata` semantics)."""
    order = x.argsort()
    xs = x[order]
    _, inv, counts = torch.unique(xs, return_inverse=True, return_counts=True)
    csum = torch.cumsum(counts, 0).double()
    starts = csum - counts.double()
    group_mean = (starts + 1.0 + csum) / 2.0            # mean of (start+1 .. csum)
    ranks = torch.empty(x.numel(), dtype=DT)
    ranks[order] = group_mean[inv]
    return ranks


def auroc(scores_pos: torch.Tensor, scores_neg: torch.Tensor) -> float:
    """AUROC of positives vs negatives, honouring the FROZEN orientation (never argmaxed).

    Mann-Whitney U form (tie-averaged ranks) — identical to sklearn's `roc_auc_score` but
    torch-only, so the scorer needs no extra dependency in the measurement env. NaN-scored
    pairs are dropped (undefined cells). Higher score == the detector calls it is-a-like; if
    the positives genuinely score lower, the AUROC is < 0.5 and that is reported, not
    flipped. Returns NaN if either class is empty after the NaN drop.
    """
    pos = scores_pos[~torch.isnan(scores_pos)].double()
    neg = scores_neg[~torch.isnan(scores_neg)].double()
    n_pos, n_neg = pos.numel(), neg.numel()
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = _tie_averaged_ranks(torch.cat([pos, neg]))
    r_pos = ranks[:n_pos].sum()
    return float((r_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def logit_ci(auroc_value: float, n_pos: int, n_neg: int, clamp: float,
             z: float = 1.96) -> tuple[float, float]:
    """95% CI for AUROC on the logit scale (Hanley-McNeil SE), back-transformed to [0,1].

    Clamping to [clamp, 1-clamp] keeps the logit finite at perfect separation, so the
    interval stays inside (0, 1) instead of blowing past 1.
    """
    if not math.isfinite(auroc_value) or n_pos <= 0 or n_neg <= 0:
        return (float("nan"), float("nan"))
    a = min(max(auroc_value, clamp), 1.0 - clamp)
    q1 = a / (2.0 - a)
    q2 = 2.0 * a * a / (1.0 + a)
    var = (a * (1.0 - a) + (n_pos - 1) * (q1 - a * a) + (n_neg - 1) * (q2 - a * a)) / (n_pos * n_neg)
    se = math.sqrt(max(var, 0.0))
    logit = math.log(a / (1.0 - a))
    se_logit = se / (a * (1.0 - a))
    lo = 1.0 / (1.0 + math.exp(-(logit - z * se_logit)))
    hi = 1.0 / (1.0 + math.exp(-(logit + z * se_logit)))
    return (lo, hi)


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def _split_scores(mat: torch.Tensor, pairs: list[tuple[int, int]], y_label: torch.Tensor,
                  pos_name: str, col_name: str) -> tuple[torch.Tensor, torch.Tensor]:
    pa = torch.tensor([a for a, _ in pairs], dtype=torch.long)
    pb = torch.tensor([b for _, b in pairs], dtype=torch.long)
    vals = mat[pa, pb].double()                          # vectorized gather
    return split_scores(vals, y_label, pos_name, col_name)


def auroc_matrix(detectors: dict[str, torch.Tensor], pairs: list[tuple[int, int]],
                 y_label: torch.Tensor, columns: tuple[str, ...]) -> dict[str, dict[str, dict]]:
    """AUROC(D, C) for every detector x column, with n_pos/n_neg/logit-CI/n_dropped.

    Class membership is by single-label equality: every pair carries exactly one class, so
    it contributes to exactly one column (its own). The per-cell logit CI assumes pair
    INDEPENDENCE and is anticonservative for clustered pairs (~F features); the reportable
    across-seed interval comes from `aggregate_seeds` (Student-t), not a single cell's CI.
    (`cluster_bootstrap_auroc` gives a feature-resampling CI if a per-seed cell CI is needed.)
    """
    out: dict[str, dict[str, dict]] = {}
    for det, mat in detectors.items():
        symmetric = det in SYMMETRIC_DETECTORS
        # The positive (is_a) effective n depends only on the detector, not the column, so
        # hoist it out of the column loop. is_a is directional (its reverse is `reversed`),
        # so a symmetric detector does not double-count it — but we still route it through
        # _effective_n for symmetric detectors to stay consistent.
        n_pos_ci_hoist = _effective_n(mat, pairs, y_label, POSITIVE_LABEL) if symmetric else None
        out[det] = {}
        for col in columns:
            pos, neg = _split_scores(mat, pairs, y_label, POSITIVE_LABEL, col)
            n_dropped = int(torch.isnan(pos).sum() + torch.isnan(neg).sum())
            a = auroc(pos, neg)
            n_pos = int((~torch.isnan(pos)).sum())
            n_neg = int((~torch.isnan(neg)).sum())
            # A symmetric detector gives (a,b) and (b,a) identical values, so a symmetric
            # negative class double-counts each independent comparison; the CI's honest n is
            # the UNORDERED count. The AUROC point estimate is unaffected. The effective
            # n is stored (n_pos_ci/n_neg_ci) so aggregate_seeds' clamp uses the same honest n.
            if symmetric:
                n_pos_ci = n_pos_ci_hoist
                n_neg_ci = _effective_n(mat, pairs, y_label, col)
            else:
                n_pos_ci, n_neg_ci = n_pos, n_neg
            # N-AWARE clamp: a saturated cell (AUROC 1.0) must not be clamped to 1-1e-6 and
            # blow the logit CI out to ~(0, 1). Clamp each cell to [1/(2n), 1-1/(2n)] from its
            # own smaller class (the same rule aggregate_seeds applies on the logit mean), so a
            # perfect-separation cell reports a finite, n-tight interval instead of a degenerate
            # one that silently cancels the symmetric-n correction.
            n_ci = (max(1, min(int(n_pos_ci), int(n_neg_ci)))
                    if (n_pos_ci is not None and n_neg_ci is not None) else 1)
            cell_clamp = max(CONSTANTS["auroc_clamp"], 1.0 / (2.0 * n_ci))
            lo, hi = logit_ci(a, n_pos_ci, n_neg_ci, cell_clamp)
            out[det][col] = {"auroc": a, "n_pos": n_pos, "n_neg": n_neg,
                             "n_pos_ci": n_pos_ci, "n_neg_ci": n_neg_ci,
                             "ci_lo": lo, "ci_hi": hi, "n_dropped": n_dropped}
    return out


def _effective_n(mat: torch.Tensor, pairs: list[tuple[int, int]], y_label: torch.Tensor,
                 name: str) -> int:
    """Unordered count of finite-scored pairs in a class — the independent-comparison n for
    a symmetric detector (collapses (a,b) and (b,a) that share a value AND a class)."""
    member = class_members(y_label, name).tolist()
    seen = set()
    for (a, b), inside in zip(pairs, member):
        if inside and not math.isnan(float(mat[a, b])):
            seen.add(frozenset((a, b)))
    return len(seen)


def _effective_n_mask(mat: torch.Tensor, pairs: list[tuple[int, int]],
                      mask: torch.Tensor) -> int:
    """Unordered count of finite-scored pairs selected by a boolean `mask` over pairs —
    the mask-based twin of `_effective_n` (which keys off a single class NAME). Needed for
    property-vs-rest, whose negative population is "everything else", not one class."""
    seen = set()
    for (a, b), inside in zip(pairs, mask.tolist()):
        if inside and not math.isnan(float(mat[a, b])):
            seen.add(frozenset((a, b)))
    return len(seen)


def property_vs_rest_grid(detectors: dict[str, torch.Tensor], pairs: list[tuple[int, int]],
                          y_label: torch.Tensor, columns: tuple[str, ...],
                          label_masks: dict[str, torch.Tensor] | None = None
                          ) -> dict[str, dict[str, dict]]:
    """AUROC(D, C-vs-rest) for every detector x column: column C = positives, EVERYTHING
    ELSE off-diagonal = negatives.

    The property-vs-rest generalization of `auroc_matrix` (which locks the positive to
    `is_a`). Each column is scored as its own class against the union of all other classes,
    so a detector that isolates, say, `frequency` shows up on the `frequency` column even
    though it is useless for `is_a`. Orientation stays FROZEN (never argmaxed): an AUROC
    below 0.5 is a real INVERTED isolator and is reported as-is — the both-tails cascade
    downstream reads it, so flipping it here would hide half the signal.

    A generative column's positives come from the single-label `y_label` (`class_members`).
    A column named in `label_masks` instead takes its positives from the given boolean mask
    over `pairs` — for the LATENT-side columns (`absorbed`, `merged`) whose truth OVERLAPS a
    generative class (an absorbed edge is also an is_a edge), so they cannot live in the
    single-label `y_label`. Their "rest" is genuinely everything-not-that-mask, is_a pairs
    included — the correct property-vs-rest semantics.

    NaN/empty-column discipline: a column with 0 positive (or 0 negative) pairs yields an
    all-NaN cell (auroc/ci NaN, n_pos or n_neg 0) and is logged once — never a crash. This
    is the SUCCESS path for an over-parameterized SAE with no `absorbed`/`merged` edges,
    not a failure. The per-cell schema matches `auroc_matrix` (auroc/n_pos/n_neg/n_pos_ci/
    n_neg_ci/ci_lo/ci_hi/n_dropped) so `aggregate_seeds` consumes either grid unchanged.
    """
    label_masks = label_masks or {}
    pa = torch.tensor([a for a, _ in pairs], dtype=torch.long)
    pb = torch.tensor([b for _, b in pairs], dtype=torch.long)
    # Column membership is detector-independent; compute the masks once and log empties once
    # (not F times inside the detector loop). A column in `label_masks` uses its explicit mask;
    # otherwise the single-label y_label decides membership.
    col_masks: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    for col in columns:
        if col in label_masks:
            pos_mask = label_masks[col].to(torch.bool)
        else:
            pos_mask = class_members(y_label, col)
        neg_mask = ~pos_mask                         # everything else (pairs are all off-diagonal)
        if int(pos_mask.sum()) == 0:
            logger.warning("property_vs_rest_grid: column %r has 0 positive pairs — NaN cell", col)
        elif int(neg_mask.sum()) == 0:
            logger.warning("property_vs_rest_grid: column %r has 0 negative pairs — NaN cell", col)
        col_masks[col] = (pos_mask, neg_mask)

    out: dict[str, dict[str, dict]] = {}
    for det, mat in detectors.items():
        symmetric = det in SYMMETRIC_DETECTORS
        vals_all = mat[pa, pb].double()              # vectorized gather, once per detector
        out[det] = {}
        for col in columns:
            pos_mask, neg_mask = col_masks[col]
            pos, neg = vals_all[pos_mask], vals_all[neg_mask]
            n_dropped = int(torch.isnan(pos).sum() + torch.isnan(neg).sum())
            a = auroc(pos, neg)
            n_pos = int((~torch.isnan(pos)).sum())
            n_neg = int((~torch.isnan(neg)).sum())
            # Symmetric detectors give (a,b) and (b,a) identical values, so BOTH the column
            # and its rest can double-count independent comparisons; the honest CI n is the
            # UNORDERED count on each side. The AUROC point estimate is unaffected.
            if symmetric:
                n_pos_ci = _effective_n_mask(mat, pairs, pos_mask)
                n_neg_ci = _effective_n_mask(mat, pairs, neg_mask)
            else:
                n_pos_ci, n_neg_ci = n_pos, n_neg
            # N-aware clamp from the smaller class (same rule as auroc_matrix/aggregate_seeds),
            # so a saturated cell reports a finite, n-tight CI instead of a degenerate one.
            n_ci = (max(1, min(int(n_pos_ci), int(n_neg_ci)))
                    if (n_pos_ci is not None and n_neg_ci is not None) else 1)
            cell_clamp = max(CONSTANTS["auroc_clamp"], 1.0 / (2.0 * n_ci))
            lo, hi = logit_ci(a, n_pos_ci, n_neg_ci, cell_clamp)
            out[det][col] = {"auroc": a, "n_pos": n_pos, "n_neg": n_neg,
                             "n_pos_ci": n_pos_ci, "n_neg_ci": n_neg_ci,
                             "ci_lo": lo, "ci_hi": hi, "n_dropped": n_dropped}
    return out


# --------------------------------------------------------------------------
# within-seed cluster bootstrap BY FEATURE
# --------------------------------------------------------------------------
def cluster_bootstrap_auroc(detector_scalar: torch.Tensor, recovered_feats: list[int],
                            y_label_fn, column: str, n_boot: int = 200,
                            rng_seed: int = 0) -> tuple[float, float]:
    """Percentile CI for one cell by resampling FEATURES (never pairs).

    Each replicate draws R features with replacement and forms pairs among the sampled
    MULTISET (a feature drawn twice contributes its pairs twice — the with-replacement
    multiplicity is what makes this a bootstrap and not a 63%-subsample; using `set()`
    undercounts the variance and gives a ~32%-too-narrow CI). `y_label_fn(pos_a, pos_b) ->
    scalar class name` supplies the truth (positions 0..R-1). Returns (nan, nan) for R < 2
    (no off-diagonal pairs / `torch.randint` would reject an empty range).
    """
    R = len(recovered_feats)
    if R < 2:
        return (float("nan"), float("nan"))
    # Lazily-filled ordered-pair label code (-2 unfilled, 1 is_a, 0 column, -1 other).
    # y_label_fn is called only for SAMPLED feature pairs (preserving the feature-resampling
    # contract) and each (i,j) at most once (cached across replicates), then bucketed vectorized.
    lab = torch.full((R, R), -2, dtype=torch.long)

    def _fill(uniq: list[int]) -> None:
        for a in uniq:
            for b in uniq:
                if a != b and int(lab[a, b]) == -2:
                    # Single-label: each pair carries exactly one class name; is_a -> 1,
                    # the column -> 0, anything else -> -1 (ignored by this cell).
                    nm = y_label_fn(a, b)
                    lab[a, b] = 1 if nm == POSITIVE_LABEL else (0 if nm == column else -1)

    g = torch.Generator().manual_seed(int(rng_seed))
    aurocs: list[float] = []
    for _ in range(n_boot):
        sample = torch.randint(0, R, (R,), generator=g)          # WITH replacement (multiset)
        _fill(torch.unique(sample).tolist())
        sub = detector_scalar[sample][:, sample]                 # [R,R] multiset values
        sub_lab = lab[sample][:, sample]
        finite = ~torch.isnan(sub)                               # excludes diagonal + same-feature
        pos = sub[(sub_lab == 1) & finite]
        neg = sub[(sub_lab == 0) & finite]
        if pos.numel() and neg.numel():
            a = auroc(pos, neg)
            if math.isfinite(a):
                aurocs.append(a)
    if not aurocs:
        return (float("nan"), float("nan"))
    t = torch.tensor(aurocs, dtype=DT)                           # torch.quantile sorts internally
    return (float(torch.quantile(t, 0.025)), float(torch.quantile(t, 0.975)))


# --------------------------------------------------------------------------
# label-free percentile ensemble
# --------------------------------------------------------------------------
def component_percentiles(detectors: dict[str, torch.Tensor],
                          pairs: list[tuple[int, int]]) -> dict[str, torch.Tensor]:
    """Per-detector percentile rank of each pair against ONE global pool (all recovered
    pairs, unlabeled). Percentile = fraction of finite scores <= this pair's score. NaN
    where the pair's score is NaN."""
    pa = torch.tensor([a for a, _ in pairs], dtype=torch.long)
    pb = torch.tensor([b for _, b in pairs], dtype=torch.long)
    out: dict[str, torch.Tensor] = {}
    for name, mat in detectors.items():
        vec = mat[pa, pb].double()                       # [n_pairs], vectorized gather
        finite = ~torch.isnan(vec)
        pct = torch.full_like(vec, float("nan"))
        fv = vec[finite]
        if fv.numel():
            # percentile = fraction of finite scores <= this pair's score, O(n log n)
            sorted_fv, _ = torch.sort(fv)
            pct[finite] = torch.searchsorted(sorted_fv, fv, right=True).double() / fv.numel()
        out[name] = pct
    return out


def min_percentile_ensemble(detectors: dict[str, torch.Tensor],
                            pairs: list[tuple[int, int]]) -> torch.Tensor:
    """Per-pair min across the DEFINED detector percentiles: high only if every gate that
    has an opinion ranks it is-a-like. NaN-aware — a pair is NaN only if EVERY detector is
    undefined for it (mirrors `mean_percentile_ensemble`); a single undefined gate must not
    poison an otherwise-agreeing pair (which would silently shrink and bias the ensemble)."""
    comps = component_percentiles(detectors, pairs)
    stacked = torch.stack([comps[k] for k in detectors], dim=0)
    all_nan = torch.isnan(stacked).all(dim=0)
    filled = torch.where(torch.isnan(stacked), torch.full_like(stacked, float("inf")), stacked)
    out = filled.min(dim=0).values
    return torch.where(all_nan, torch.full_like(out, float("nan")), out)


def mean_percentile_ensemble(detectors: dict[str, torch.Tensor],
                             pairs: list[tuple[int, int]]) -> torch.Tensor:
    """Robustness check: per-pair mean across detector percentiles (nan-aware)."""
    comps = component_percentiles(detectors, pairs)
    stacked = torch.stack([comps[k] for k in detectors], dim=0)
    return torch.nanmean(stacked, dim=0)


# --------------------------------------------------------------------------
# controls — must sit at AUROC ~0.5
# --------------------------------------------------------------------------
def random_scalar_control(pairs: list[tuple[int, int]], y_label: torch.Tensor,
                          column: str, rng: torch.Generator) -> float:
    """A random per-pair scalar cannot separate any class -> AUROC ~0.5 (CI-machinery check)."""
    scores = torch.rand(len(pairs), generator=rng, dtype=DT)
    pos, neg = split_scores(scores, y_label, POSITIVE_LABEL, column)
    return auroc(pos, neg)


def shuffled_label_control(detector_scalar: torch.Tensor, pairs: list[tuple[int, int]],
                           y_label: torch.Tensor, column: str, rng: torch.Generator) -> float:
    """Permute annotations on the SAME recovered universe -> a real detector's signal must
    vanish (AUROC ~0.5). Catches a pipeline that manufactures signal from the label map."""
    perm = torch.randperm(y_label.numel(), generator=rng)
    shuffled = y_label[perm]
    vals = torch.tensor([float(detector_scalar[a, b]) for (a, b) in pairs], dtype=DT)
    pos, neg = split_scores(vals, shuffled, POSITIVE_LABEL, column)
    return auroc(pos, neg)


# --------------------------------------------------------------------------
# held-out seed / dispersion-conditioned readout / redundancy
# --------------------------------------------------------------------------
def held_out_sample_seed(train_seed: int, offset: int = HELD_OUT_SEED_OFFSET) -> int:
    """The disjoint sampling seed for the held-out detector draw. offset==0 is refused
    (it would collapse held-out onto in-sample, defeating the held-out firewall)."""
    if offset == 0:
        raise ValueError("held_out offset must be non-zero: a 0 offset reuses the training draw")
    return int(train_seed) + int(offset)


def dispersion_split(isa_pairs: list[tuple[int, int]], r_disp: dict[int, float],
                     detector_scalar: torch.Tensor,
                     neg_pairs: list[tuple[int, int]] | None = None) -> dict:
    """Median-split the is-a pairs by the CHILD's r_disp and report each half.

    r_disp is child-direction dispersion (uses truth g) — a READOUT DIAGNOSTIC, never a
    detector. With `neg_pairs` given, each half's value is the is-a-vs-neg AUROC; without
    it, the mean oriented detector score over the half (a monotone summary).
    """
    disps = torch.tensor([r_disp[c] for _, c in isa_pairs], dtype=DT)
    median = float(disps.median())

    def _summary(half: list[tuple[int, int]]) -> float:
        if not half:
            return float("nan")
        if neg_pairs is not None:
            pos = torch.tensor([float(detector_scalar[a, b]) for a, b in half], dtype=DT)
            neg = torch.tensor([float(detector_scalar[a, b]) for a, b in neg_pairs], dtype=DT)
            return auroc(pos, neg)
        vals = torch.tensor([float(detector_scalar[a, b]) for a, b in half], dtype=DT)
        vals = vals[~torch.isnan(vals)]
        return float(vals.mean()) if vals.numel() else float("nan")

    high = [(p, c) for (p, c) in isa_pairs if r_disp[c] >= median]
    low = [(p, c) for (p, c) in isa_pairs if r_disp[c] < median]
    return {"high": _summary(high), "low": _summary(low), "median": median,
            "n_high": len(high), "n_low": len(low)}


def redundancy_map(detectors: dict[str, torch.Tensor], pairs: list[tuple[int, int]],
                   y_label: torch.Tensor, columns: tuple[str, ...],
                   grid: dict | None = None) -> dict:
    """Pairwise detector rank-correlation + each detector's marginal added AUROC per column.

    Marginal = AUROC(D on C) - max over the OTHER detectors of AUROC(other on C): a near-zero
    marginal is measured domination (exploratory). When no other detector has a defined
    AUROC for a column, the marginal is NaN (`no comparison`), NOT a fabricated `a - 0.5`.
    `grid` may be passed in to avoid recomputing the AUROC matrix.
    """
    names = list(detectors)
    comps = component_percentiles(detectors, pairs)
    rank_corr: dict[str, dict[str, float]] = {}
    for a in names:
        rank_corr[a] = {}
        for b in names:
            va, vb = comps[a], comps[b]
            m = (~torch.isnan(va)) & (~torch.isnan(vb))
            rank_corr[a][b] = float(_spearman(va[m], vb[m])) if int(m.sum()) > 2 else float("nan")

    grid = auroc_matrix(detectors, pairs, y_label, columns) if grid is None else grid
    marginal: dict[str, dict[str, float]] = {}
    for det in names:
        marginal[det] = {}
        for col in columns:
            others = [grid[o][col]["auroc"] for o in names if o != det
                      and math.isfinite(grid[o][col]["auroc"])]
            a = grid[det][col]["auroc"]
            if not others or not math.isfinite(a):
                marginal[det][col] = float("nan")        # no valid comparison -> undefined
            else:
                marginal[det][col] = a - max(others)
    return {"rank_corr": rank_corr, "marginal_auroc": marginal}


def _spearman(a: torch.Tensor, b: torch.Tensor) -> float:
    """Spearman rank correlation with TIE-AVERAGED ranks (scipy `spearmanr` semantics).

    `argsort().argsort()` breaks ties by position, which fabricates a correlation for the
    per-parent broadcast detectors (constant across a parent's columns → many ties). Tie
    averaging fixes that; a constant (zero-variance) vector carries no rank information and
    returns NaN, not a spurious ±1. Precondition: NaN-free inputs (callers filter).
    """
    if a.numel() < 2 or torch.isnan(a).any() or torch.isnan(b).any():
        return float("nan")
    ra = _tie_averaged_ranks(a)
    rb = _tie_averaged_ranks(b)
    ra = ra - ra.mean()
    rb = rb - rb.mean()
    da, db = ra.norm(), rb.norm()
    if float(da) == 0.0 or float(db) == 0.0:
        return float("nan")
    return float((ra @ rb) / (da * db))


# --------------------------------------------------------------------------
# class-balanced pooled ensemble (macro-average is-a-vs-negatives)
# --------------------------------------------------------------------------
def _ensemble_pooled_auroc(ens: torch.Tensor, pairs: list[tuple[int, int]],
                           y_label: torch.Tensor) -> dict:
    """is-a vs the scored negative classes, weighted EQUALLY per class — DETERMINISTIC.

    The macro-average of the per-class is-a-vs-C AUROCs: each scored negative class C
    contributes AUROC(is_a, all finite pairs in C), and the reported `auroc` is the unweighted
    mean over classes. This gives every class equal weight WITHOUT discarding data — the old
    subsample-to-the-thinnest-class approach threw away ~all of the wide classes (topical ~12
    capped every class to 12) and, being a single `manual_seed(0)` draw, made the pooled metric
    a bare point estimate with ~0.02 draw-to-draw sd. A macro-average is equal-weight, uses every
    pair, and has no RNG, so it rests on the whole surviving pool (effect size + n, never a lossy draw).

    Returns `{auroc (macro-mean), per_class {C: auroc}, n_pos, n_neg (total finite negs used),
    n_classes, n_dropped}`.
    """
    pos_mask = class_members(y_label, POSITIVE_LABEL)
    pos_all = torch.tensor([float(v) for v in ens[pos_mask]], dtype=DT)
    pos = pos_all[~torch.isnan(pos_all)]
    # Grouped by class INDEX; each pair carries exactly one class, so the is_a positives and
    # every negative class are disjoint by construction (no is_a pair can also be a negative).
    neg_by_cls: dict[str, list[float]] = {}
    for col in SCORED_COLUMNS:
        if col == POSITIVE_LABEL:
            continue
        m = class_members(y_label, col)
        vals = [float(v) for v in ens[m] if not math.isnan(float(v))]
        if vals:
            neg_by_cls[col] = vals
    # n_dropped counts NaN pairs from the population that FEEDS this metric (is_a + scored negs),
    # NOT the whole universe: a NaN in an out-of-scope (exploratory) class was never eligible
    # here, so counting it would overstate the loss this number labels.
    eligible = pos_mask.clone()
    for col in SCORED_COLUMNS:
        eligible |= class_members(y_label, col)
    n_dropped = int(sum(1 for v in ens[eligible] if math.isnan(float(v))))
    n_neg_total = sum(len(v) for v in neg_by_cls.values())
    nan_loo = {"min": float("nan"), "max": float("nan"),
               "dropped_for_min": None, "dropped_for_max": None}
    if not neg_by_cls or pos.numel() == 0:
        return {"auroc": float("nan"), "per_class": {}, "per_class_n": {}, "loo_range": nan_loo,
                "n_pos": int(pos.numel()), "n_neg": n_neg_total,
                "n_classes": len(neg_by_cls), "n_dropped": n_dropped}
    per_class: dict[str, float] = {}
    per_class_n: dict[str, int] = {}
    for name, v in neg_by_cls.items():        # keyed by class NAME
        per_class[name] = auroc(pos, torch.tensor(v, dtype=DT))
        per_class_n[name] = len(v)
    finite = {k: a for k, a in per_class.items() if math.isfinite(a)}
    macro = sum(finite.values()) / len(finite) if finite else float("nan")
    # Leave-one-class-out sensitivity: recompute the macro dropping each finite class in turn, so a
    # single thin/noisy class that swings the equal-weight mean is VISIBLE in the report (a thin
    # class carries full 1/n_classes weight, and one flipped pair in it can move the macro across the bar).
    loo = dict(nan_loo)
    if len(finite) >= 2:
        names = list(finite)
        loo_vals = {drop: (sum(a for k, a in finite.items() if k != drop) / (len(finite) - 1))
                    for drop in names}
        lo_name = min(loo_vals, key=loo_vals.get)   # dropping this class LOWERS the macro most
        hi_name = max(loo_vals, key=loo_vals.get)
        loo = {"min": loo_vals[lo_name], "max": loo_vals[hi_name],
               "dropped_for_min": lo_name, "dropped_for_max": hi_name}
    return {"auroc": macro, "per_class": per_class, "per_class_n": per_class_n, "loo_range": loo,
            "n_pos": int(pos.numel()), "n_neg": n_neg_total,
            "n_classes": len(neg_by_cls), "n_dropped": n_dropped}


def _ensemble_macro_ci(ens: torch.Tensor, y_label: torch.Tensor, n_boot: int = 1000,
                       rng_seed: int = 0) -> tuple[float, float]:
    """Stratified-bootstrap 95% CI for the macro-mean ensemble AUROC (kept SEPARATE from the
    RNG-free `_ensemble_pooled_auroc` so that point estimate stays deterministic and RNG-free).

    The macro-mean is an equal-weight mean over per-class is-a-vs-C AUROCs, so a THIN class (e.g.
    topical ~12) carries full 1/n_classes weight and dominates the variance — a bare point estimate
    0.84 a hair above the 0.80 bar is not quotable without bounding that variance (effect size +
    n). Each iteration resamples the shared is_a positives ONCE (preserving the cross-class
    correlation — the same is_a pairs are scored against every negative class) and each class's
    negatives independently, recomputes every finite per-class AUROC and their macro-mean, and the CI
    is the 2.5/97.5 percentile over `n_boot` resamples. Deterministic via a fixed generator.
    """
    pos_mask = class_members(y_label, POSITIVE_LABEL)
    pos_all = torch.tensor([float(v) for v in ens[pos_mask]], dtype=DT)
    pos = pos_all[~torch.isnan(pos_all)]
    # Grouped by class INDEX; each pair carries exactly one class, so the is_a positives and
    # every negative class are disjoint by construction (no is_a pair can also be a negative).
    neg_by_cls: dict[str, list[float]] = {}
    for col in SCORED_COLUMNS:
        if col == POSITIVE_LABEL:
            continue
        m = class_members(y_label, col)
        vals = [float(v) for v in ens[m] if not math.isnan(float(v))]
        if vals:
            neg_by_cls[col] = vals
    neg_tensors = [torch.tensor(v, dtype=DT) for v in neg_by_cls.values() if v]
    if pos.numel() == 0 or not neg_tensors:
        return (float("nan"), float("nan"))
    gen = torch.Generator().manual_seed(int(rng_seed))
    npos = int(pos.numel())
    macros: list[float] = []
    for _ in range(int(n_boot)):
        pos_b = pos[torch.randint(npos, (npos,), generator=gen)]
        per = []
        for negt in neg_tensors:
            nn = int(negt.numel())
            a = auroc(pos_b, negt[torch.randint(nn, (nn,), generator=gen)])
            if math.isfinite(a):
                per.append(a)
        if per:
            macros.append(sum(per) / len(per))
    if not macros:
        return (float("nan"), float("nan"))
    macros.sort()
    lo = macros[int(0.025 * len(macros))]
    hi = macros[min(len(macros) - 1, int(0.975 * len(macros)))]
    return (float(lo), float(hi))


# --------------------------------------------------------------------------
# across-seed aggregation
# --------------------------------------------------------------------------
# Student-t 97.5% critical values by df (=n_seeds-1). The across-seed CI has only n_seeds
# replicates, so the multiplier is t_{.975,df}, NOT the normal 1.96 (its df->inf limit): at n=5
# that is 2.7764 vs 1.96, a 29%-too-narrow interval. Hardcoded (dependency-free — aggregate_seeds
# may run in an env without scipy) through df=30, covering the reportable 15-20-seed regime; beyond
# df=30 the z-limit is within ~4% so 1.96 is an acceptable fallback.
_T_CRIT_975: dict[int, float] = {
    1: 12.7062, 2: 4.3027, 3: 3.1824, 4: 2.7764, 5: 2.5706, 6: 2.4469, 7: 2.3646, 8: 2.3060,
    9: 2.2622, 10: 2.2281, 11: 2.2010, 12: 2.1788, 13: 2.1604, 14: 2.1448, 15: 2.1314, 16: 2.1199,
    17: 2.1098, 18: 2.1009, 19: 2.0930, 20: 2.0860, 21: 2.0796, 22: 2.0739, 23: 2.0687, 24: 2.0639,
    25: 2.0595, 26: 2.0555, 27: 2.0518, 28: 2.0484, 29: 2.0452, 30: 2.0423,
}


def _t_crit_975(n_seeds: int) -> float:
    """Two-sided 97.5% Student-t multiplier for `n_seeds` replicates (df = n_seeds-1). Falls back
    to the normal 1.96 for df<1 (undefined spread) or df>30 (within ~4% of the z-limit)."""
    return _T_CRIT_975.get(int(n_seeds) - 1, 1.96)


def aggregate_seeds(reports: list[dict]) -> dict:
    """Across-seed AUROC per cell on the logit scale + a Student-t CI (t_{.975,n-1}, not z:
    at n=5 the multiplier is 2.7764). Preliminary at 3-5 seeds; the reportable regime is ~15-20."""
    if not reports:
        return {}
    # Guard: refuse a report set with a DUPLICATED train_seed — a checkpoint_dirname collision (or
    # a re-scored duplicate) would otherwise be aggregated as if independent, halving the effective n
    # and fabricating a too-tight across-seed CI. Lenient on reports that carry no seed (old schema):
    # the guard fires only on an actual duplicate among the seeds that ARE present.
    seeds = [r["meta"]["train_seed"] for r in reports
             if isinstance(r.get("meta"), dict) and r["meta"].get("train_seed") is not None]
    if len(set(seeds)) < len(seeds):
        dupes = sorted({s for s in seeds if seeds.count(s) > 1})
        raise ValueError(f"aggregate_seeds: non-distinct train_seed(s) {dupes} in the report set — "
                         "seeds must be distinct (a checkpoint_dirname collision or a duplicate report)")
    dets = list(reports[0]["grid"].keys())
    cols = list(reports[0]["grid"][dets[0]].keys())
    agg: dict[str, dict[str, dict]] = {}
    clamp = CONSTANTS["auroc_clamp"]
    for det in dets:
        agg[det] = {}
        for col in cols:
            cells = [r["grid"][det][col] for r in reports
                     if math.isfinite(r["grid"][det][col]["auroc"])]
            if not cells:
                agg[det][col] = {"mean": float("nan"), "ci_lo": float("nan"),
                                 "ci_hi": float("nan"), "median": float("nan"), "n_seeds": 0}
                continue
            vals = [c["auroc"] for c in cells]
            # SAMPLE-SIZE-AWARE clamp: a seed with finite n cannot legitimately act as
            # AUROC 1-1e-6 (logit ±13.8) and dominate the logit-mean. Clamp each seed to
            # [1/(2n), 1-1/(2n)] from its own smaller class, so a single saturated seed can't
            # swamp the others (one saturated seed must not dominate the across-seed CI).
            logits = []
            for c in cells:
                # Prefer the CI-basis (effective) n so a symmetric detector's clamp matches
                # its honest CI. Missing sample sizes -> flat clamp (NEVER default to n=1,
                # which would force cl=0.5 and collapse every AUROC to a fabricated 0.5 null).
                np_ = c.get("n_pos_ci", c.get("n_pos"))
                nn_ = c.get("n_neg_ci", c.get("n_neg"))
                if np_ is None or nn_ is None:
                    cl = clamp
                else:
                    n = max(1, min(int(np_), int(nn_)))
                    cl = max(clamp, 1.0 / (2.0 * n))
                a = min(max(c["auroc"], cl), 1.0 - cl)
                logits.append(math.log(a / (1.0 - a)))
            mean_l = sum(logits) / len(logits)
            sd = (sum((x - mean_l) ** 2 for x in logits) / max(len(logits) - 1, 1)) ** 0.5
            se = sd / math.sqrt(len(logits))
            t_crit = _t_crit_975(len(logits))          # df = n_seeds-1, not the normal 1.96
            svals = sorted(vals)
            median = svals[len(svals) // 2] if len(svals) % 2 else \
                0.5 * (svals[len(svals) // 2 - 1] + svals[len(svals) // 2])
            if sd == 0.0 and len(logits) > 1:
                # All seeds clamped to the SAME logit (every seed saturated, e.g. AUROC 1.0 on H2/H4):
                # the between-seed variance is 0 and a Student-t interval collapses to [mean, mean], a
                # fabricated zero-width CI. Fall back to the ENVELOPE of the per-seed cell CIs (each cell
                # carries its own finite-n logit CI) — the aggregate cannot honestly claim more certainty
                # than any single seed already has.
                lo_env = min((c["ci_lo"] for c in cells if c.get("ci_lo") is not None),
                             default=_sigmoid(mean_l))
                hi_env = max((c["ci_hi"] for c in cells if c.get("ci_hi") is not None),
                             default=_sigmoid(mean_l))
                ci_lo, ci_hi = lo_env, hi_env
            else:
                ci_lo, ci_hi = _sigmoid(mean_l - t_crit * se), _sigmoid(mean_l + t_crit * se)
            agg[det][col] = {"mean": _sigmoid(mean_l), "ci_lo": ci_lo, "ci_hi": ci_hi,
                             "median": median, "n_seeds": len(vals)}
    return agg
