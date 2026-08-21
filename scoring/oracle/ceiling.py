"""
Stage-0 clean-oracle ceiling — one of the harness's TWO standalone readouts.

The scorer reports two INDEPENDENT grids, NOT a subtraction:
  * this ORACLE (clean) grid — "can each metric identify its property at its own IDEAL?" The clean
    dictionary is the world's own truth (decoder `g[f]`, an oracle-encoder / true-A / probe firing per
    detector), so scoring the detectors on it is the ceiling each metric could reach with a perfect SAE.
  * the TRAINED grid (`scoring.core.grid` via `retrieval.run_retrieval`) — "can it retrieve the property
    from real SAE latents?"
Per-detector oracles (α-encoder / true-A / probe) made a single-reference `Δ = clean − trained` invalid
(different, sometimes UNATTAINABLE, oracles per detector), so the survival-Δ object is gone. The reader
compares the two grids directly: a high oracle cell with a low trained cell = the metric works at its
ideal but the SAE loses it. `annotate_clean_grid` stamps each oracle cell with `worked` / `degenerate` /
`oracle_regime` / `oracle_idealized` so the oracle grid stands alone; `stage0_caveats` surfaces its
degenerate / never-worked cells.

Firewall: the clean dict is TRUTH (A, g), used ONLY as the ceiling readout — never as a detector input
on the trained side. Both grids enumerate the SAME recovered pairs on the same held-out draw.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Sequence

import torch

from scoring.core.detectors import DetectorInputs, compute_all, s_res_cosine, s_res_probe
from scoring.core.grid import auroc_matrix
from scoring.core.registry import CONSTANTS as _REGISTRY_CONSTANTS

if TYPE_CHECKING:                       # WorldBundle is used only as a type hint here
    from scoring.core.world import WorldBundle

_TINY = 1e-12


class OracleEncodeInfeasible(ValueError):
    """The oracle encoder cannot produce a Stage-0 ceiling for this draw (bad realized_l0, or the
    checkpoint is denser than a tied-unit-g JumpReLU can mirror). A SUBCLASS of ValueError so existing
    `except ValueError` callers (calibrate) still catch it, but narrow enough that run_retrieval can
    catch ONLY this — not a compute_all config error (bad s_res_mode / missing inputs), which must
    surface loudly instead of being mislabeled `stage0_unavailable`."""


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
        raise OracleEncodeInfeasible(
            f"oracle_encode: realized_l0 must be a positive finite number, got {target}")
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
        raise OracleEncodeInfeasible(
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
                           routing: dict[str, str], constants: dict | None = None,
                           s_res_ceiling: str = "cosine") -> dict[str, torch.Tensor]:
    """Per-detector clean ceilings: run `compute_all` on BOTH the alpha-encoder and the true-A
    firing regimes, then assemble one `{name: matrix}` picking each detector from its routed regime.

    Both regimes are given FULL `DetectorInputs` (same unit/raw decoders = g, h, b_dec=0, tokens) —
    only the activations differ: the alpha-encoder's honest gated projections vs the oracle
    coefficients A. The recon_2a residual under true-A is noise + UNRECOVERED-feature energy (A@g omits
    features outside `feats`), i.e. the honest generative ceiling — read its survival-Δ that way.

    `s_res_ceiling` selects the s_res ceiling. s_res is set EXPLICITLY here (never taken from
    compute_all, which is called with s_res_mode="skip") so no cosine is computed-then-discarded:
      "probe" = a LIKE-FOR-LIKE probe trained on the alpha-encoder ORACLE firing over the true-g
                decoders -- the ONLY ceiling that may feed a reported cell (run_retrieval passes this).
      "cosine" (default) = the cheap analytic geometry oracle, a DIAGNOSTIC ceiling for unit tests and
                the degeneracy check ONLY; it is a cross-metric reference, never a reported s_res.
    "probe" trains R probes on the ceiling; the default keeps unit tests cheap and probe-free.
    """
    constants = _REGISTRY_CONSTANTS if constants is None else constants
    # Only s_res may be routed 'agnostic': its ceiling is set EXPLICITLY below (probe/cosine), so the
    # by_regime['agnostic'] = alpha_det fallback is never read for it. ANY other agnostic-routed detector
    # would silently take alpha_encoder numbers while stamped 'agnostic' — a latent mislabel. Fail loudly.
    # Explicit raise, NOT `assert` (which `python -O` strips): a mis-routed detector would silently
    # ship alpha_encoder numbers stamped 'agnostic', so this invariant must hold even under -O.
    if not all(d == "s_res" for d, r in (routing or {}).items() if r == "agnostic"):
        raise RuntimeError(
            "only s_res may use the 'agnostic' regime (any other would silently get alpha_encoder numbers)")
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
    # s_res_mode="skip": neither regime's compute_all s_res is ever used (s_res is agnostic, set
    # explicitly below) -- skipping it avoids computing-then-discarding a cosine s_res on the probe
    # ceiling path, so no cosine is ever produced unless the cosine ceiling is explicitly requested.
    alpha_det = compute_all(DetectorInputs(acts_rec=alpha_acts, **common), constants, s_res_mode="skip")
    trueA_det = compute_all(DetectorInputs(acts_rec=trueA_acts, **common), constants, s_res_mode="skip")
    by_regime = {"alpha_encoder": alpha_det, "true_A": trueA_det, "agnostic": alpha_det}
    out: dict[str, torch.Tensor] = {}
    for name in alpha_det:                                  # all DETECTORS, in registry order
        regime = routing.get(name, "alpha_encoder")
        out[name] = by_regime[regime][name]
    # s_res is set EXPLICITLY (never taken from compute_all): the probe on the α-encoder oracle firing
    # (like-for-like reported ceiling) or, only when the cosine ceiling is explicitly asked for, the
    # analytic geometry oracle. The report path always passes "probe"; "cosine" is a diagnostic ceiling.
    out["s_res"] = (s_res_probe(alpha_acts, bundle.h, W_unit, constants) if s_res_ceiling == "probe"
                    else s_res_cosine(W_unit))
    return out


def _clean_detectors(bundle: "WorldBundle", feats: list[int], realized_l0: float,
                     constants: dict, routing: dict[str, str] | None,
                     s_res_ceiling: str = "cosine") -> dict[str, torch.Tensor]:
    """The clean detector matrices, minus those with no valid ceiling under the active oracle.

    routing None -> the single alpha-encoder oracle (legacy); a routing dict -> per-detector
    regimes (`clean_detectors_merged`). Regime-aware exclusion drops only detectors whose routed
    firing cannot yield a ceiling (recon_2a on the alpha encoder). `s_res_ceiling` is forwarded to the
    merged path (probe-on-oracle vs cosine s_res ceiling)."""
    if routing is None:
        detectors = compute_all(clean_detector_inputs(bundle, feats, realized_l0), constants)
    else:
        detectors = clean_detectors_merged(bundle, feats, realized_l0, routing, constants,
                                           s_res_ceiling=s_res_ceiling)
    excluded = _excluded_detectors(routing)
    return {d: m for d, m in detectors.items() if d not in excluded}


def clean_grid(bundle: "WorldBundle", feats: list[int], pairs: list[tuple[int, int]],
               y_label: torch.Tensor, columns: Sequence[str],
               constants: dict, realized_l0: float,
               routing: dict[str, str] | None = None,
               s_res_ceiling: str = "cosine") -> dict[str, dict[str, dict]]:
    """Stage-0 AUROC grid: the detectors on the perfect-decoder oracle dictionary, over the
    recovered pairs, read at the trained SAE's `realized_l0`.

    `routing=None` (default) reads every detector off the single alpha-encoder oracle (legacy).
    A `routing` dict (e.g. `DETECTOR_ORACLE_REGIME`) reads each detector off its property-correct
    firing regime and keeps recon_2a (routed to true-A). `s_res_ceiling="probe"` makes the s_res
    ceiling a like-for-like probe-on-oracle (default "cosine" keeps the cheap analytic ceiling)."""
    if not feats:
        return {}
    detectors = _clean_detectors(bundle, feats, realized_l0, constants, routing, s_res_ceiling)
    return auroc_matrix(detectors, pairs, y_label, columns)


def degenerate_clean_detectors(bundle: "WorldBundle", feats: list[int],
                               pairs: list[tuple[int, int]], constants: dict,
                               realized_l0: float, tol: float = 1e-9,
                               routing: dict[str, str] | None = None,
                               s_res_ceiling: str = "cosine") -> list[str]:
    """Clean detectors whose scored values are constant within `tol` across the recovered pairs.

    A near-constant detector column produces a meaningless AUROC (~0.5 from ties), so its Stage-0
    ceiling must be flagged DEGENERATE, not read as "the metric is weak even clean". With the
    oracle ENCODER the co-firing detectors are non-degenerate by construction (α-leakage spreads
    them), so this is a guard: it surfaces any residual collapse instead of silently scoring 0.5.
    `routing`/`s_res_ceiling` MUST match `clean_grid`'s so the degeneracy flag is computed on the
    SAME ceiling that is reported (else s_res is checked on cosine while the grid shows the probe).
    Returns the sorted list of degenerate detector names."""
    if not feats or not pairs:
        return []
    detectors = _clean_detectors(bundle, feats, realized_l0, constants, routing, s_res_ceiling)
    return _constant_scored_detectors(detectors, pairs, tol)


def clean_grid_and_degenerate(bundle: "WorldBundle", feats: list[int],
                              pairs: list[tuple[int, int]], y_label: torch.Tensor,
                              columns: Sequence[str], constants: dict, realized_l0: float,
                              routing: dict[str, str] | None = None,
                              s_res_ceiling: str = "cosine", tol: float = 1e-9
                              ) -> tuple[dict[str, dict[str, dict]], list[str]]:
    """The Stage-0 clean grid AND its degenerate-detector list, computed from ONE `_clean_detectors`
    pass. Prefer this over calling `clean_grid` + `degenerate_clean_detectors` separately: it (a)
    guarantees the degeneracy flag is read off the SAME ceiling as the grid (so `s_res_ceiling`
    cannot disagree between them) and (b) does not train the probe ceilings twice per seed."""
    if not feats:
        return {}, []
    detectors = _clean_detectors(bundle, feats, realized_l0, constants, routing, s_res_ceiling)
    grid = auroc_matrix(detectors, pairs, y_label, columns)
    degenerate = _constant_scored_detectors(detectors, pairs, tol) if pairs else []
    return grid, degenerate


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


def stage0_caveats(clean_grid: dict) -> list[dict]:
    """The Stage-0 ORACLE cells whose ceiling reading needs a caveat (clean-only — there is no
    clean-vs-trained comparison object any more; the oracle grid and the trained grid are two
    standalone readouts). Returns `[{detector, column, status}, ...]` for every clean cell that is
    `degenerate` (a constant/undefined column → a meaningless ~0.5, not a real ceiling) or did not
    work (`never_worked` — the metric never clears chance even at its own ideal). `degenerate` takes
    precedence. Reads the flags stamped by `annotate_clean_grid`. Detector/column order follows the grid.
    """
    out: list[dict] = []
    for det, cols in clean_grid.items():
        for col, cell in cols.items():
            if cell.get("degenerate"):
                out.append({"detector": det, "column": col, "status": "degenerate"})
            elif not cell.get("worked", True):
                out.append({"detector": det, "column": col, "status": "never_worked"})
    return out


# A ceiling read off the exact oracle coefficients A (the `true_A` regime) is IDEALIZED: with an
# overcomplete dictionary (F > D) no linear encoder can recover A, and even a perfect-GEOMETRY SAE
# projects through α-overlapping directions and cannot reach it. So a `true_A`-routed oracle cell is
# an UNATTAINABLE ideal, not a ceiling any SAE could reach — read it as characterization, and do not
# compare an idealized oracle cell against an `alpha_encoder` (achievable) one as if on one scale.
IDEALIZED_REGIMES: tuple[str, ...] = ("true_A",)
REGIME_CAVEAT: str = (
    "Oracle ceilings are per-detector: alpha_encoder cells use a realistic gated encoder (achievable); "
    "true_A cells use the exact oracle coefficients A (IDEALIZED — unrecoverable when F>D, unreachable "
    "even by a perfect-geometry SAE). Read a true_A oracle cell as an UNATTAINABLE ideal (characterization "
    "of the metric), NOT a ceiling any SAE could hit, and do not compare it against an alpha_encoder cell "
    "as if the two oracles were on one scale.")


def annotate_regime(survival: dict, routing: dict[str, str] | None,
                    s_res_ceiling: str = "cosine") -> dict:
    """Stamp every survival cell with the oracle regime its ceiling was read from and whether that
    regime is idealized. In-place, and returns `survival` for chaining.

    `routing=None` (legacy single alpha-encoder oracle) marks every cell `alpha_encoder`,
    `idealized=False`. When `s_res_ceiling="probe"`, s_res is stamped `alpha_encoder_probe` (its
    ceiling is a probe on the α-encoder oracle firing — NOT geometry-agnostic), not the routing's
    "agnostic" label, so the per-cell regime does not misdescribe the s_res ceiling. Still
    `idealized=False` (the α-encoder probe ceiling is achievable, not the unrecoverable-A ideal).
    """
    for det, cols in survival.items():
        regime = "alpha_encoder" if routing is None else routing.get(det, "alpha_encoder")
        if det == "s_res" and s_res_ceiling == "probe":
            regime = "alpha_encoder_probe"
        idealized = regime in IDEALIZED_REGIMES
        for cell in cols.values():
            cell["oracle_regime"] = regime
            cell["oracle_idealized"] = idealized
    return survival


def annotate_clean_grid(clean_grid: dict, degenerate: Sequence[str],
                        routing: dict[str, str] | None = None,
                        s_res_ceiling: str = "cosine", worked_floor: float = 0.55) -> dict:
    """Stamp each ORACLE (clean) grid cell with the characterization that says whether its ceiling
    reading is REAL — so the oracle grid stands alone as "can the metric identify the property at its
    ideal?", without any clean-vs-trained subtraction. In-place; returns `clean_grid`.

      worked            : the metric clears chance at its ideal — `auroc >= worked_floor` (0.55) AND,
                          when a finite CI is present, `ci_lo > 0.5` (a near-chance large-n value with a
                          tight CI, or a clears-floor-but-CI-dips small-n value, is NOT `worked`).
      degenerate        : the detector is in `degenerate` (a constant/undefined column → meaningless ~0.5).
      oracle_regime /   : stamped by `annotate_regime` (which oracle firing produced this ceiling; and
      oracle_idealized    whether it is the unattainable true-A ideal). s_res_ceiling='probe' labels
                          s_res `alpha_encoder_probe`.
    """
    degen = set(degenerate or ())
    for det, cols in clean_grid.items():
        for cell in cols.values():
            a = cell.get("auroc", float("nan"))
            ci_lo = cell.get("ci_lo")
            ci_ok = (ci_lo is None or not isinstance(ci_lo, (int, float))
                     or not math.isfinite(ci_lo) or ci_lo > 0.5)
            cell["worked"] = bool(isinstance(a, (int, float)) and math.isfinite(a)
                                  and a >= worked_floor and ci_ok)
            cell["degenerate"] = det in degen
    annotate_regime(clean_grid, routing, s_res_ceiling=s_res_ceiling)
    return clean_grid
