"""
The detector-by-confound AUROC grid and its statistics — the reusable, checkpoint-free
toolkit behind retrieval scoring.

Labels enter only here; detectors in `scoring.core.detectors` never see them. Threshold-free:
each cell asks "does detector D rank a random is-a pair above a random class-C pair?" — an
AUROC, prevalence-independent so thin columns get wide CIs rather than tuning.

Statistics rules enforced here:
  - orientation is fixed in `scoring.core.registry`, never flipped to maximize score;
  - AUROC CIs use the logit scale with a [clamp, 1-clamp] guard for finite intervals;
  - within-seed CI is a cluster bootstrap BY FEATURE, never a bootstrap over pairs;
  - the two controls (random scalar, shuffled label) must land at AUROC ~0.5.

The trained driver feeding real checkpoint activations through this toolkit lives in
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

    A feature is in the universe only if it is recovered and has a match (>=0). The returned
    `DetectorInputs` is indexed by recovered-feature POSITION k (0..R-1); `index_map` maps the
    true feature id to k.
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
    """All ordered off-diagonal pairs (as positions), each with its single class INDEX.

    Pairs are POSITIONS into `recovered_feats`; label comes from `pair_labels[feat_a, feat_b]`
    (the only place truth enters the pipeline). Self-pairs are excluded: the diagonal label 0
    equals is_a's index, so a self-pair would be miscounted. Use `class_members` for membership.
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

    Classes are mutually exclusive, so positive and negative populations never overlap.
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
    """AUROC of positives vs negatives, using the FROZEN orientation (never flipped).

    Computed via the Mann-Whitney U form with tie-averaged ranks (matches sklearn's
    `roc_auc_score`, torch-only). NaN-scored pairs are dropped. An AUROC < 0.5 is reported
    as-is, not flipped. Returns NaN if either class is empty after the drop.
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
    """95% CI for AUROC, computed on the logit scale (Hanley-McNeil SE) then mapped back to [0,1].

    Clamping the AUROC to [clamp, 1-clamp] keeps the logit finite at perfect separation, so the
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
    """AUROC(D, C) for every detector-by-column cell, with n_pos/n_neg/logit-CI/n_dropped.

    Each pair contributes to exactly one column. The per-cell logit CI assumes independent
    pairs, so it's too narrow for clustered pairs; the reportable across-seed interval comes
    from `aggregate_seeds` instead (`cluster_bootstrap_auroc` gives a per-seed CI if needed).
    """
    out: dict[str, dict[str, dict]] = {}
    for det, mat in detectors.items():
        symmetric = det in SYMMETRIC_DETECTORS
        # is_a effective n depends only on the detector, so compute once outside the column loop.
        n_pos_ci_hoist = _effective_n(mat, pairs, y_label, POSITIVE_LABEL) if symmetric else None
        out[det] = {}
        for col in columns:
            pos, neg = _split_scores(mat, pairs, y_label, POSITIVE_LABEL, col)
            n_dropped = int(torch.isnan(pos).sum() + torch.isnan(neg).sum())
            a = auroc(pos, neg)
            n_pos = int((~torch.isnan(pos)).sum())
            n_neg = int((~torch.isnan(neg)).sum())
            # Symmetric detectors double-count (a,b)/(b,a); CI's honest n is the unordered count (point estimate unaffected).
            if symmetric:
                n_pos_ci = n_pos_ci_hoist
                n_neg_ci = _effective_n(mat, pairs, y_label, col)
            else:
                n_pos_ci, n_neg_ci = n_pos, n_neg
            # N-aware clamp: saturated cells (AUROC 1.0) clamp to [1/(2n), 1-1/(2n)] using the smaller class, for a finite n-tight interval.
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
    """Unordered count of finite-scored pairs in a class — the independent-comparison n for a
    symmetric detector (collapses (a,b) and (b,a), which share both a value and a class)."""
    member = class_members(y_label, name).tolist()
    seen = set()
    for (a, b), inside in zip(pairs, member):
        if inside and not math.isnan(float(mat[a, b])):
            seen.add(frozenset((a, b)))
    return len(seen)


def _effective_n_mask(mat: torch.Tensor, pairs: list[tuple[int, int]],
                      mask: torch.Tensor) -> int:
    """Unordered count of finite-scored pairs selected by a boolean `mask` — the mask-based
    version of `_effective_n`, needed for property-vs-rest's "everything else" negative class."""
    seen = set()
    for (a, b), inside in zip(pairs, mask.tolist()):
        if inside and not math.isnan(float(mat[a, b])):
            seen.add(frozenset((a, b)))
    return len(seen)


def property_vs_rest_grid(detectors: dict[str, torch.Tensor], pairs: list[tuple[int, int]],
                          y_label: torch.Tensor, columns: tuple[str, ...],
                          label_masks: dict[str, torch.Tensor] | None = None
                          ) -> dict[str, dict[str, dict]]:
    """AUROC(D, C-vs-rest) for every detector-by-column cell: column C's pairs are the
    positives, everything else off-diagonal is the negatives.

    Generalizes `auroc_matrix` (which always uses `is_a` as positive) so a detector that
    isolates, say, `frequency` shows up on the `frequency` column too. Orientation stays
    FROZEN — an AUROC below 0.5 is a real inverted isolator, reported as-is.

    A generative column's positives come from `y_label` (`class_members`); a column named in
    `label_masks` instead takes positives from the given boolean mask, needed for latent-side
    columns (`absorbed`, `merged`) whose truth overlaps a generative class.

    Empty-column handling: 0 positive or negative pairs yields an all-NaN cell, logged once —
    the success path for an SAE with no such edges, not a failure.
    """
    label_masks = label_masks or {}
    pa = torch.tensor([a for a, _ in pairs], dtype=torch.long)
    pb = torch.tensor([b for _, b in pairs], dtype=torch.long)
    # Column membership is detector-independent; compute masks once (not per detector) and log empties once.
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
            # Symmetric detectors double-count (a,b)/(b,a) on both sides; honest CI n is the unordered count each side.
            if symmetric:
                n_pos_ci = _effective_n_mask(mat, pairs, pos_mask)
                n_neg_ci = _effective_n_mask(mat, pairs, neg_mask)
            else:
                n_pos_ci, n_neg_ci = n_pos, n_neg
            # N-aware clamp from the smaller class (same rule as auroc_matrix), for a finite n-tight CI.
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
    MULTISET — a feature drawn twice contributes its pairs twice, which is what makes this a
    real bootstrap rather than a 63% subsample. `y_label_fn(pos_a, pos_b)` returns the class
    name for a pair of positions. Returns (nan, nan) for R < 2.
    """
    R = len(recovered_feats)
    if R < 2:
        return (float("nan"), float("nan"))
    # Ordered-pair label code, filled lazily (-2 unfilled/1 is_a/0 column/-1 other); y_label_fn called once per sampled pair, cached across replicates.
    lab = torch.full((R, R), -2, dtype=torch.long)

    def _fill(uniq: list[int]) -> None:
        for a in uniq:
            for b in uniq:
                if a != b and int(lab[a, b]) == -2:
                    # Each pair carries exactly one class: is_a -> 1, the column -> 0, else -1 (ignored).
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
    """Per-detector percentile rank of each pair against ONE global pool (all recovered pairs,
    unlabeled). Percentile = fraction of finite scores <= this pair's score. NaN where the
    pair's own score is NaN."""
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


# --------------------------------------------------------------------------
# controls — must sit at AUROC ~0.5
# --------------------------------------------------------------------------
def random_scalar_control(pairs: list[tuple[int, int]], y_label: torch.Tensor,
                          column: str, rng: torch.Generator) -> float:
    """A random per-pair scalar cannot separate any class, so AUROC ~0.5 (checks the CI machinery)."""
    scores = torch.rand(len(pairs), generator=rng, dtype=DT)
    pos, neg = split_scores(scores, y_label, POSITIVE_LABEL, column)
    return auroc(pos, neg)


def shuffled_label_control(detector_scalar: torch.Tensor, pairs: list[tuple[int, int]],
                           y_label: torch.Tensor, column: str, rng: torch.Generator) -> float:
    """Permute the labels on the SAME recovered universe; a real detector's signal must then
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
    """The disjoint sampling seed for the held-out detector draw. An offset of 0 is refused: it
    would make the held-out draw identical to the in-sample one, defeating the firewall."""
    if offset == 0:
        raise ValueError("held_out offset must be non-zero: a 0 offset reuses the training draw")
    return int(train_seed) + int(offset)


def dispersion_split(isa_pairs: list[tuple[int, int]], r_disp: dict[int, float],
                     detector_scalar: torch.Tensor,
                     neg_pairs: list[tuple[int, int]] | None = None) -> dict:
    """Split the is-a pairs at the median of the CHILD's r_disp and report each half.

    r_disp is child-direction dispersion (uses truth g) — a readout diagnostic, never a
    detector. If `neg_pairs` is given, each half's value is the is-a-vs-neg AUROC; otherwise it
    is the mean oriented detector score over the half (a monotone summary).
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
    """Pairwise detector rank-correlation, plus each detector's marginal added AUROC per column.

    Marginal = AUROC(D on C) - max over other detectors' AUROC(other on C); near-zero means
    this detector is dominated. NaN (not a fake `a - 0.5`) when no comparison is possible.
    Pass `grid` to reuse an already-computed AUROC matrix.
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
                marginal[det][col] = float("nan")        # no valid comparison, so undefined
            else:
                marginal[det][col] = a - max(others)
    return {"rank_corr": rank_corr, "marginal_auroc": marginal}


def _spearman(a: torch.Tensor, b: torch.Tensor) -> float:
    """Spearman rank correlation with TIE-AVERAGED ranks (matches scipy's `spearmanr`).

    Plain `argsort().argsort()` breaks ties by position, fabricating correlation for the
    per-parent broadcast detectors (many ties). A zero-variance vector returns NaN, not a
    spurious +/-1. Requires NaN-free inputs (callers filter first).
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
# across-seed aggregation
# --------------------------------------------------------------------------
# Student-t 97.5% critical values by df (n_seeds-1); hardcoded so aggregate_seeds needs no scipy, covers df<=30, falls back to 1.96 beyond that.
_T_CRIT_975: dict[int, float] = {
    1: 12.7062, 2: 4.3027, 3: 3.1824, 4: 2.7764, 5: 2.5706, 6: 2.4469, 7: 2.3646, 8: 2.3060,
    9: 2.2622, 10: 2.2281, 11: 2.2010, 12: 2.1788, 13: 2.1604, 14: 2.1448, 15: 2.1314, 16: 2.1199,
    17: 2.1098, 18: 2.1009, 19: 2.0930, 20: 2.0860, 21: 2.0796, 22: 2.0739, 23: 2.0687, 24: 2.0639,
    25: 2.0595, 26: 2.0555, 27: 2.0518, 28: 2.0484, 29: 2.0452, 30: 2.0423,
}


def _t_crit_975(n_seeds: int) -> float:
    """Two-sided 97.5% Student-t multiplier for `n_seeds` replicates (df = n_seeds - 1). Falls
    back to the normal 1.96 for df < 1 (undefined spread) or df > 30 (within ~4% of the z-limit)."""
    return _T_CRIT_975.get(int(n_seeds) - 1, 1.96)


def aggregate_seeds(reports: list[dict]) -> dict:
    """Across-seed AUROC per cell, averaged on the logit scale, with a Student-t CI (t_{.975,n-1},
    not z: at n=5 the multiplier is 2.7764). Treat 3-5 seeds as preliminary; the reportable regime
    is ~15-20 seeds."""
    if not reports:
        return {}
    # Guard: refuse repeated train_seed(s) — a duplicate would be aggregated as independent, giving a too-tight CI. Reports with no seed (old schema) are allowed.
    seeds = [r["meta"]["train_seed"] for r in reports
             if isinstance(r.get("meta"), dict) and r["meta"].get("train_seed") is not None]
    if len(set(seeds)) < len(seeds):
        dupes = sorted({s for s in seeds if seeds.count(s) > 1})
        raise ValueError(f"aggregate_seeds: non-distinct train_seed(s) {dupes} in the report set — "
                         "seeds must be distinct (a checkpoint_dirname collision or a duplicate report)")
    # Guard: refuse reports that disagree on (config, variant, k) — mixing two experiments would silently combine incomparable runs. Reports with no meta (old schema) are allowed.
    sigs = {(m.get("config"), m.get("variant"), m.get("k"))
            for r in reports if isinstance((m := r.get("meta")), dict)}
    if len(sigs) > 1:
        raise ValueError(f"aggregate_seeds: reports disagree on (config, variant, k) {sorted(sigs)} — "
                         "a --out directory mixing two experiments; aggregate each run separately")
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
            # Sample-size-aware clamp: clamp each seed to [1/(2n), 1-1/(2n)] from its smaller class, so one saturated seed can't swamp the across-seed CI.
            logits = []
            for c in cells:
                # Prefer the CI-basis (effective) n so symmetric detectors' clamp matches their CI; never default to n=1, which would collapse AUROC to a fake 0.5.
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
                # All seeds clamped to the same logit (e.g. every seed saturated): variance is 0, so fall back to the envelope of per-seed CIs rather than a fake zero-width interval.
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
