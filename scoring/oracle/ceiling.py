"""
Stage-0 survival-Δ — the clean-dictionary ceiling and its distance from the trained metric.

A low trained AUROC is ambiguous: the METRIC may be weak, or training may have DESTROYED a
signal that was there in the clean geometry. Stage-0 resolves it. The clean dictionary is the
world's own truth — true feature f's "latent" is f itself, decoder `g[f]`, activation `A[:,f]` —
so scoring the detectors on it gives the ceiling each metric could reach with a perfect SAE.
`Δ = clean − trained` then attributes the loss: a high Δ on a cell whose clean AUROC is high
means the signal existed clean but training erased it (SAE fault); Δ≈0 with a low clean means
the metric was weak even clean (metric fault).

Firewall: the clean dict is TRUTH (A, g), used ONLY to compute the ceiling the trained metric is
compared against — never as a detector input on the trained side. Both stages enumerate the SAME
recovered pairs on the same held-out draw. BUT a detector NaNs undefined cells independently on each
side (a shared-token child underpowered on the trained dict may be well-supported on the clean one),
so the SCORED (non-NaN) subset can differ between clean and trained even though the pair universe is
identical. `survival_delta` therefore carries each side's n_pos/n_neg and a `same_support` flag; a Δ
across a support mismatch (`same_support=False`) is not a clean metric-vs-SAE attribution and must be
read with that caveat, not as "same pairs both sides".
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Sequence

import torch

from scoring.core.detectors import DetectorInputs, compute_all
from scoring.core.grid import auroc_matrix
from scoring.core.registry import CONSTANTS as _REGISTRY_CONSTANTS

if TYPE_CHECKING:                       # WorldBundle is used only as a type hint here
    from scoring.core.world import WorldBundle

_TINY = 1e-12


def oracle_encode(h: torch.Tensor, g: torch.Tensor, realized_l0: float) -> torch.Tensor:
    """The perfect-DECODER SAE's activations: a linear encoder tied to unit-`g` + a JumpReLU gate.

    This is the honest Stage-0 ceiling — what a perfect-GEOMETRY SAE (decoder rows = the true
    directions) could reconstruct through a REAL encoding step, NOT the oracle coefficients `A`.
    The encoder is the standard tied init `W_enc = g_unit`, so a child's projection carries a real
    component of its PARENT's coefficient via the designed overlap cos(g_child,g_parent)=α (the
    α-leakage). This is the whole point: `acts = A` gives exact containment (every is-a child fires
    ⟺ its parent does → coverage_R ≡ 1, a degenerate ceiling that inverted the H5 headline), whereas
    the oracle ENCODER leaks α and gives an honest, non-degenerate ceiling (~0.78 is-a coverage).

        g_unit = g / ‖g‖ ;  proj = h @ g_unitᵀ  (linear, tied, NO per-token pseudo-inverse) ;
        θ = uniform JumpReLU threshold s.t. mean row-L0 == realized_l0 ;  acts = proj · 1[proj > θ].

    `realized_l0` MUST be the trained SAE's realized L0 on real `h` (scoring.core.recovery.realized_l0),
    NOT the nominal top-k: the ceiling is only comparable to the trained metric if both are read at
    the same sparsity. A uniform (single-θ) gate mirrors the checkpoint's inference-time scalar-threshold gate, whose
    threshold is uniform across latents.
    """
    target = float(realized_l0)
    if not (target > 0) or not math.isfinite(target):
        raise ValueError(f"oracle_encode: realized_l0 must be a positive finite number, got {target}")
    g_unit = g.double() / g.double().norm(dim=1, keepdim=True).clamp_min(_TINY)
    proj = h.double() @ g_unit.transpose(0, 1)              # [n, F] linear pre-activation
    n, F = proj.shape
    # A JumpReLU gate is NON-NEGATIVE, and "firing" everywhere downstream is `acts > 0`
    # (fire_thresh=0), so ONLY positive projections can fire. Calibrate θ over the POSITIVE
    # projections so θ stays strictly positive and the achieved row-L0 actually tracks the target.
    # (Selecting θ over all entries let θ drift ≤ 0 once the target exceeded the positive budget,
    # silently plateauing the achieved L0 — a landmine precisely for the dense/pathological
    # checkpoints Stage-0 exists to diagnose.)
    pos = proj[proj > 0]                                    # [P] the fireable projections
    k = int(round(target * n))                              # total firing entries wanted (== l0·n)
    P = int(pos.numel())
    if k >= P:
        raise ValueError(
            f"oracle_encode: mean L0={target:.2f} needs {k} firing entries but only {P} projections "
            f"are positive (~{P / max(n, 1):.1f}/row); a non-negative JumpReLU gate cannot reach it "
            f"(the tied-unit-g encoder fires ≲ F/2 latents/row). The checkpoint is denser than this "
            f"oracle encoder can mirror — do not silently plateau the ceiling.")
    theta = torch.kthvalue(pos, P - k + 1).values           # k-th largest POSITIVE value (> 0)
    return proj * (proj > theta)


def clean_detector_inputs(bundle: "WorldBundle", feats: list[int],
                          realized_l0: float) -> DetectorInputs:
    """The perfect-DECODER detector inputs for the recovered features, on the held-out draw.

    Decoder rows are the true directions `g[f]`; activations come from `oracle_encode` (the tied
    JumpReLU encoder at the trained SAE's `realized_l0`) — NOT the oracle coefficients `A`. Using
    `A` gave every is-a child exact containment (coverage_R ≡ 1, degenerate) and inverted the
    survival-Δ headline; the oracle encoder leaks the designed α and yields the honest ceiling.
    """
    idx = torch.tensor(feats, dtype=torch.long)             # copy (parity with reduce_to_recovered)
    g_sel = bundle.g[idx]                                    # [R, D] true concept directions
    W_unit = g_sel / g_sel.norm(dim=1, keepdim=True).clamp_min(_TINY)
    acts = oracle_encode(bundle.h, bundle.g, realized_l0)[:, idx]   # [n, R] honest oracle acts
    return DetectorInputs(
        acts_rec=acts, W_unit=W_unit, W_raw=g_sel,
        h=bundle.h, b_dec=torch.zeros(bundle.g.shape[1], dtype=bundle.g.dtype),
        tokens=bundle.tokens, vocab=bundle.cfg.vocab,
    )


# Detectors with NO valid Stage-0 clean ceiling under the ALPHA-ENCODER oracle. The tied
# unit-g encoder (no least-squares inverse) does not reconstruct h (FVU > 1), so a
# reconstruction-ablation detector on it measures noise, not a ceiling — it is dropped in the
# legacy single-oracle path (its trained AUROC is still reported in the trained grid). Under
# per-detector routing recon_2a is sent to the true-A regime (real A@g reconstruction) instead,
# so it IS kept there — see `_excluded_detectors`. Firing/geometry detectors are unaffected.
NO_CLEAN_CEILING: tuple[str, ...] = ("recon_2a",)


# Per-detector oracle firing regime: each detector's ground-truth ceiling must be read off the
# firing that makes ITS property faithful, not one blanket oracle.
#   alpha_encoder — α-leaked tied-g JumpReLU encoder (honest containment ceiling; true A gives
#                   coverage ≡ 1, a degenerate ceiling). token_freq_survival MUST stay here: on
#                   true A token-bound pairs fire exactly on their ids, with no frequency breakdown.
#   true_A        — the oracle coefficients A (exact firing structure + the only faithful
#                   reconstruction, since h = A@g + noise; least-squares would give min-norm coeffs,
#                   not the semantic features, because F > D).
#   agnostic      — geometry only (reads W_unit = unit-g, identical under either firing regime).
DETECTOR_ORACLE_REGIME: dict[str, str] = {
    "coverage_R": "alpha_encoder",
    "asymmetry_R": "alpha_encoder",
    "pmi": "alpha_encoder",
    "token_freq_survival": "alpha_encoder",
    "sibling_redundancy": "true_A",
    "joint_child_J": "true_A",
    "joint_child_mass": "true_A",
    "outdegree": "true_A",
    "recon_2a": "true_A",
    "s_res": "agnostic",
}


def _excluded_detectors(routing: dict[str, str] | None) -> set[str]:
    """Which detectors have NO valid clean ceiling and must be dropped from the grid.

    Legacy (routing None): the fixed `NO_CLEAN_CEILING` set (recon_2a on the alpha encoder).
    Routed: only the `NO_CLEAN_CEILING` detectors whose routed regime is still `alpha_encoder` —
    recon_2a routed to `true_A` reconstructs via A@g and is kept.
    """
    if routing is None:
        return set(NO_CLEAN_CEILING)
    return {d for d in NO_CLEAN_CEILING if routing.get(d, "alpha_encoder") == "alpha_encoder"}


def clean_detectors_merged(bundle: "WorldBundle", feats: list[int], realized_l0: float,
                           routing: dict[str, str], constants: dict | None = None
                           ) -> dict[str, torch.Tensor]:
    """Per-detector clean ceilings: run `compute_all` on BOTH the alpha-encoder and the true-A
    firing regimes, then assemble one `{name: matrix}` picking each detector from its routed regime.

    Both regimes are given FULL `DetectorInputs` (same unit/raw decoders = g, h, b_dec=0, tokens) —
    only the activations differ: the alpha-encoder's honest gated projections vs the oracle
    coefficients A. `agnostic` detectors (s_res) read only W_unit, identical either way, so they are
    taken from the alpha bundle. The recon_2a residual under true-A is noise + UNRECOVERED-feature
    energy (A@g omits features outside `feats`), i.e. the honest generative ceiling — NOT distance
    from a perfect reconstruction; its survival-Δ must be read that way.
    """
    constants = _REGISTRY_CONSTANTS if constants is None else constants
    idx = torch.tensor(feats, dtype=torch.long)
    g_sel = bundle.g[idx]
    W_unit = g_sel / g_sel.norm(dim=1, keepdim=True).clamp_min(_TINY)
    common = dict(
        W_unit=W_unit, W_raw=g_sel, h=bundle.h,
        b_dec=torch.zeros(bundle.g.shape[1], dtype=bundle.g.dtype),
        tokens=bundle.tokens, vocab=bundle.cfg.vocab,
    )
    alpha_acts = oracle_encode(bundle.h, bundle.g, realized_l0)[:, idx]
    trueA_acts = bundle.A[:, idx]
    alpha_det = compute_all(DetectorInputs(acts_rec=alpha_acts, **common), constants)
    trueA_det = compute_all(DetectorInputs(acts_rec=trueA_acts, **common), constants)
    by_regime = {"alpha_encoder": alpha_det, "true_A": trueA_det, "agnostic": alpha_det}
    out: dict[str, torch.Tensor] = {}
    for name in alpha_det:                                  # all DETECTORS, in registry order
        regime = routing.get(name, "alpha_encoder")
        out[name] = by_regime[regime][name]
    return out


def _clean_detectors(bundle: "WorldBundle", feats: list[int], realized_l0: float,
                     constants: dict, routing: dict[str, str] | None) -> dict[str, torch.Tensor]:
    """The clean detector matrices, minus those with no valid ceiling under the active oracle.

    routing None -> the single alpha-encoder oracle (legacy); a routing dict -> per-detector
    regimes (`clean_detectors_merged`). Regime-aware exclusion drops only detectors whose routed
    firing cannot yield a ceiling (recon_2a on the alpha encoder)."""
    if routing is None:
        detectors = compute_all(clean_detector_inputs(bundle, feats, realized_l0), constants)
    else:
        detectors = clean_detectors_merged(bundle, feats, realized_l0, routing, constants)
    excluded = _excluded_detectors(routing)
    return {d: m for d, m in detectors.items() if d not in excluded}


def clean_grid(bundle: "WorldBundle", feats: list[int], pairs: list[tuple[int, int]],
               y_label: torch.Tensor, columns: Sequence[str],
               constants: dict, realized_l0: float,
               routing: dict[str, str] | None = None) -> dict[str, dict[str, dict]]:
    """Stage-0 AUROC grid: the detectors on the perfect-decoder oracle dictionary, over the
    recovered pairs, read at the trained SAE's `realized_l0`.

    `routing=None` (default) reads every detector off the single alpha-encoder oracle (legacy).
    A `routing` dict (e.g. `DETECTOR_ORACLE_REGIME`) reads each detector off its property-correct
    firing regime and keeps recon_2a (routed to true-A)."""
    if not feats:
        return {}
    detectors = _clean_detectors(bundle, feats, realized_l0, constants, routing)
    return auroc_matrix(detectors, pairs, y_label, columns)


def degenerate_clean_detectors(bundle: "WorldBundle", feats: list[int],
                               pairs: list[tuple[int, int]], constants: dict,
                               realized_l0: float, tol: float = 1e-9,
                               routing: dict[str, str] | None = None) -> list[str]:
    """Clean detectors whose scored values are constant within `tol` across the recovered pairs.

    A near-constant detector column produces a meaningless AUROC (~0.5 from ties), so its Stage-0
    ceiling must be flagged DEGENERATE, not read as "the metric is weak even clean". With the
    oracle ENCODER the co-firing detectors are non-degenerate by construction (α-leakage spreads
    them), so this is a guard: it surfaces any residual collapse instead of silently scoring 0.5.
    `routing` selects the same single/per-detector oracle as `clean_grid`. Returns the sorted list
    of degenerate detector names."""
    if not feats or not pairs:
        return []
    detectors = _clean_detectors(bundle, feats, realized_l0, constants, routing)
    return _constant_scored_detectors(detectors, pairs, tol)


def _constant_scored_detectors(detectors: dict[str, torch.Tensor],
                               pairs: list[tuple[int, int]], tol: float = 1e-9) -> list[str]:
    """Pure core of `degenerate_clean_detectors`: names of detectors whose scored ceiling over
    `pairs` cannot produce a meaningful AUROC — either FEWER than 2 finite values (undefined /
    all-NaN column: the MAXIMAL degeneracy) or a finite range below `tol` (a constant column → a
    meaningless ~0.5 from ties). Both cases are flagged (an all-NaN detector is strictly more broken
    than a constant one, so it must not be silently exempted). Sorted."""
    degen: list[str] = []
    for det, mat in detectors.items():
        vals = torch.tensor([float(mat[p, c]) for (p, c) in pairs], dtype=torch.float64)
        vals = vals[~torch.isnan(vals)]
        if vals.numel() < 2 or float(vals.max() - vals.min()) < tol:
            degen.append(det)
    return sorted(degen)


def _support_status(cc: dict, tc: dict, min_retention: float
                    ) -> tuple[str, int, int, int, int]:
    """Classify how the trained-side scored support compares to the clean side.

    Each side NaNs undefined cells independently, so the scored pair counts can differ even on
    the SAME pair universe — and a Δ computed across a collapsed trained denominator ("nothing
    changed on the 30% of pairs still scorable") must not read as a clean "survived". Returns
    (`status`, n_pos_clean, n_neg_clean, n_pos_trained, n_neg_trained) where status is:
      - "unknown"  : a side carries NO support keys at all — support is truly absent, NOT "same".
      - "same"     : identical n on both sides.
      - "collapsed": the smaller-class retention min(n_trained/n_clean) fell below `min_retention`.
      - "mismatch" : n differ but retention is within tolerance.
    """
    have_c = ("n_pos" in cc) or ("n_neg" in cc)
    have_t = ("n_pos" in tc) or ("n_neg" in tc)
    np_c, nn_c = int(cc.get("n_pos", 0)), int(cc.get("n_neg", 0))
    np_t, nn_t = int(tc.get("n_pos", 0)), int(tc.get("n_neg", 0))
    if not (have_c and have_t):
        return "unknown", np_c, nn_c, np_t, nn_t
    if np_c == np_t and nn_c == nn_t:
        return "same", np_c, nn_c, np_t, nn_t
    ret_pos = (np_t / np_c) if np_c > 0 else 1.0
    ret_neg = (nn_t / nn_c) if nn_c > 0 else 1.0
    status = "collapsed" if min(ret_pos, ret_neg) < min_retention else "mismatch"
    return status, np_c, nn_c, np_t, nn_t


def survival_delta(clean: dict, trained_grid: dict, columns: Sequence[str],
                   min_support_retention: float = 0.8,
                   degenerate_detectors: Sequence[str] | None = None,
                   clean_worked_floor: float = 0.55
                   ) -> dict[str, dict[str, dict]]:
    """Per (detector, column): the clean/trained AUROCs, their Δ, the inversion flag, and an
    explicit SUPPORT STATUS. `delta = clean − trained`.

    Detectors are taken from `trained_grid` (what actually ran) and columns from `columns`.
    `delta` is NaN whenever either endpoint is NaN — an undefined ceiling or trained value
    cannot yield a real attribution.

    `inverted` requires a REAL flip: `trained < 0.5 AND clean > 0.5`. A destroyed cell erased the
    signal (trained ≈ 0.5, `inverted=False`); an inverted cell had a clean signal that training
    FLIPPED so the negatives now outscore the positives (a stronger phenomenon than erasure). A low
    clean with a low trained is NOT an inversion (there was no signal to flip), so it stays False.

    `support_status` (see `_support_status`) guards the attribution: a Δ on a `collapsed` cell rests
    on a small surviving trained denominator and must NOT be read as a clean "survived"; a `mismatch`
    is a mild count difference; `unknown` means the support was not reported. `same_support` is the
    strict `support_status == "same"` — False on unknown/mismatch/collapsed. Cells needing a caveat
    are collected by `stage0_caveats` and surfaced on the run_retrieval report so the headline reader
    sees them without auditing every cell.

    `clean_worked` gates the SURVIVED reading: a Δ≈0 means "survived" ONLY if the clean metric
    actually WORKED — a clean ceiling at chance (never separated) with a trained value also at chance
    is a Δ≈0 that reads as "survived" but is really "never worked" (e.g. a pooled clean 0.5239 = sub-
    cells 0.056 & 1.000). A cell counts as `clean_worked` iff `clean >= clean_worked_floor` (0.55,
    the primary gate — a fixed floor is required to catch a LARGE-n near-chance value whose CI is
    tight) AND, when the clean cell carries a finite CI, `clean ci_lo > 0.5` (a conservative add-on
    that only ever tightens the floor, for small-n cells whose point clears 0.55 but whose CI dips to
    chance). Cells that did not work are surfaced by `stage0_caveats` as `clean_never_worked`.
    """
    degen = set(degenerate_detectors or ())
    out: dict[str, dict[str, dict]] = {}
    for det in trained_grid:
        out[det] = {}
        for col in columns:
            cc = clean.get(det, {}).get(col, {})
            tc = trained_grid.get(det, {}).get(col, {})
            c = cc.get("auroc", float("nan"))
            t = tc.get("auroc", float("nan"))
            delta = (c - t) if (math.isfinite(c) and math.isfinite(t)) else float("nan")
            status, np_c, nn_c, np_t, nn_t = _support_status(cc, tc, min_support_retention)
            # SURVIVED floor: clean must clear the fixed floor, and (when a finite CI is present) its
            # lower bound must clear chance. A missing/NaN CI falls back to the point floor alone.
            cl_lo = cc.get("ci_lo")
            ci_ok = (cl_lo is None or not isinstance(cl_lo, (int, float))
                     or not math.isfinite(cl_lo) or cl_lo > 0.5)
            clean_worked = bool(math.isfinite(c) and c >= clean_worked_floor and ci_ok)
            out[det][col] = {
                "clean": c, "trained": t, "delta": delta,
                "inverted": bool(math.isfinite(t) and math.isfinite(c) and t < 0.5 and c > 0.5),
                "n_pos_clean": np_c, "n_neg_clean": nn_c,
                "n_pos_trained": np_t, "n_neg_trained": nn_t,
                "support_status": status, "same_support": status == "same",
                # a degenerate CLEAN ceiling (constant/undefined column) is a meaningless ~0.5 —
                # mark every cell so its clean/delta is not read as a real attribution (silent-fail #1)
                "degenerate": det in degen,
                # a Δ≈0 is "survived" only if the clean metric worked (else "never worked", not survived)
                "clean_worked": clean_worked,
            }
    return out


def stage0_caveats(survival: dict) -> list[dict]:
    """The Stage-0 cells whose Δ needs a caveat, for surfacing on the top-level report.

    Returns `[{detector, column, status}, ...]` for every cell whose CLEAN ceiling is `degenerate`,
    OR whose clean metric did not work (`clean_worked` False → `clean_never_worked`), OR whose
    `support_status` is not "same" (collapsed / mismatch / unknown) — the cells where a bare Δ is not
    a clean metric-vs-SAE attribution. Precedence: `degenerate` (a constant clean ceiling voids the Δ
    entirely) > `clean_never_worked` (a clean ceiling at chance makes a Δ≈0 "never worked", not
    "survived") > the support status. Detector/column order follows the survival dict.
    """
    out: list[dict] = []
    for det, cols in survival.items():
        for col, cell in cols.items():
            if cell.get("degenerate"):
                out.append({"detector": det, "column": col, "status": "degenerate"})
            elif not cell.get("clean_worked", True):
                out.append({"detector": det, "column": col, "status": "clean_never_worked"})
            elif cell.get("support_status", "unknown") != "same":
                out.append({"detector": det, "column": col,
                            "status": cell.get("support_status", "unknown")})
    return out


# A ceiling read off the exact oracle coefficients A (the `true_A` regime) is IDEALIZED: with an
# overcomplete dictionary (F > D) no linear encoder can recover A, and even a perfect-GEOMETRY SAE
# projects through α-overlapping directions and cannot reach it. So a `true_A`-routed cell's
# Δ = clean − trained is NOT pure training-erasure — it also folds in the unrecoverable-A gap — and
# its magnitude is NOT comparable to an `alpha_encoder` (realistic-encoder) cell's Δ.
IDEALIZED_REGIMES: tuple[str, ...] = ("true_A",)
REGIME_CAVEAT: str = (
    "Ceilings are per-detector: alpha_encoder cells use a realistic gated encoder (achievable), "
    "true_A cells use the exact oracle coefficients A (IDEALIZED — unrecoverable when F>D, and "
    "unreachable even by a perfect-geometry SAE). Read a true_A cell's delta as 'gap from the "
    "idealized firing/recon oracle', not 'training erased a real signal', and do NOT compare delta "
    "magnitudes across the two regimes.")


def annotate_regime(survival: dict, routing: dict[str, str] | None) -> dict:
    """Stamp every survival cell with the oracle regime its ceiling was read from and whether that
    regime is idealized. In-place, and returns `survival` for chaining.

    `routing=None` (legacy single alpha-encoder oracle) marks every cell `alpha_encoder`,
    `idealized=False` — so a reader never has to know which path produced the grid.
    """
    for det, cols in survival.items():
        regime = "alpha_encoder" if routing is None else routing.get(det, "alpha_encoder")
        idealized = regime in IDEALIZED_REGIMES
        for cell in cols.values():
            cell["oracle_regime"] = regime
            cell["oracle_idealized"] = idealized
    return survival
