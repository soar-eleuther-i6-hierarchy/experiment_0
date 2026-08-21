"""
Stage-1 mathematical validation of the detector machinery. NO AUROC.

This is the oracle stage of the two-stage redesign: a PREFILTER that confirms each detector EQUATION
is correctly implemented and non-degenerate on PURE inputs, before the trained stage measures whether
it retrieves anything from real SAE latents. It answers "is the machinery correct?", never "how well
does the metric rank?" — so there is no ceiling, no survival-Δ, no AUROC here.

Two pure firing regimes feed `compute_all`:
  * ``true_A``        — the oracle coefficients A (exact firing structure + the faithful
                        reconstruction, since h = A@g + noise).
  * ``alpha_encoder`` — the honest gated encoder: a tied unit-g JumpReLU gate selects the firing
                        support, then a per-token RIDGE least-squares on that support supplies the
                        magnitudes (reconstructing with the raw decoder g). The ridge shrinkage is the
                        KNOWN_BUGS 4.4 fix: the old tied-projection magnitudes could not reconstruct h
                        (FVU≈0.86–1.36); the gated least-squares solution reaches FVU≈0.14 while leaving
                        the JumpReLU firing STRUCTURE essentially unchanged (is_a coverage 0.828 vs the
                        tied 0.829, saturation 0.03 — still honest, non-degenerate). Only the two
                        magnitude detectors (recon_2a, joint_child_mass) see the improved values.

Two checks per detector: FINITENESS (≥2 finite scored values, not an all-NaN column) and NON-DEGENERACY
(`_constant_scored_detectors` — a constant column gives a meaningless ~0.5 AUROC downstream). A detector
that is degenerate in one regime by design (coverage_R ≡ 1 for is_a on true_A) but healthy in the other
is fine; the battery flags a detector broken in BOTH.

A third check validates the s_res PROBE machinery against the analytic geometry: on TRUE firing + TRUE
geometry the probe s_res (`probe_true_g`) must track the closed-form cosine s_res (`cosine_g`). The
`sres_*` knobs are frozen once so this agreement holds on a held-out draw.

This module also HOUSES the salvage the trained side never needed but Stage-1 depends on — `oracle_encode`,
`OracleEncodeInfeasible`, `_constant_scored_detectors` — relocated here from the (Phase-5-deleted)
`ceiling.py`. Firewall: the pure regimes use TRUTH (A, g); they validate machinery, never score the
trained SAE.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

from scoring.core.detectors import (DetectorInputs, compute_all, s_res_cosine,
                                    s_res_probe)
from scoring.core.registry import CONSTANTS as _REGISTRY_CONSTANTS

if TYPE_CHECKING:                       # WorldBundle is used only as a type hint
    from scoring.core.world import WorldBundle

_TINY = 1e-12

# Ridge shrinkage for the gated least-squares magnitudes (KNOWN_BUGS 4.4). Fixed, NOT tuned to any SAE:
# small enough not to bias the well-conditioned per-token support solve, large enough to keep it stable.
RIDGE_LAMBDA = 1e-4

# Below this many defined off-diagonal pairs a probe-vs-cosine correlation is meaningless (a handful of
# points clears any floor by chance), so `s_res_calibration` reports NaN instead — the gate fails loud.
_MIN_CALIB_PAIRS = 100


class OracleEncodeInfeasible(ValueError):
    """The oracle encoder cannot produce a valid firing for this draw (bad realized_l0, or the
    checkpoint is denser than a tied-unit-g JumpReLU can mirror). A SUBCLASS of ValueError so existing
    ``except ValueError`` callers still catch it, but narrow enough that a caller can catch ONLY this —
    not a compute_all config error (bad s_res_mode / missing inputs), which must surface loudly."""


def _gate_support(h: torch.Tensor, g: torch.Tensor, realized_l0: float
                  ) -> tuple[torch.Tensor, torch.Tensor]:
    """The tied unit-g JumpReLU gate: `proj = h @ g_unitᵀ`, a uniform threshold θ over the POSITIVE
    projections chosen so mean row-L0 == realized_l0. Returns (support_mask [n, F], g_unit [F, D]).

    A JumpReLU gate is NON-NEGATIVE, so only positive projections can fire; θ is calibrated over the
    positive projections so it stays strictly positive and the achieved L0 tracks the target (selecting
    θ over all entries let it drift ≤0 once the target exceeded the positive budget, silently plateauing
    the L0 — a landmine for exactly the dense checkpoints this stage exists to diagnose)."""
    target = float(realized_l0)
    if not (target > 0) or not math.isfinite(target):
        raise OracleEncodeInfeasible(
            f"oracle_encode: realized_l0 must be a positive finite number, got {target}")
    g_unit = g.double() / g.double().norm(dim=1, keepdim=True).clamp_min(_TINY)
    proj = h.double() @ g_unit.transpose(0, 1)             # [n, F] linear pre-activation
    n, F = proj.shape
    pos = proj[proj > 0]                                   # [P] the fireable projections
    k = int(round(target * n))                             # total firing entries wanted (== l0·n)
    P = int(pos.numel())
    if k <= 0:
        raise OracleEncodeInfeasible(
            f"oracle_encode: mean L0={target:.3g} rounds to k=0 firing entries over n={n} tokens — a "
            f"(near-)dead SAE the gate cannot represent. Fail loud, don't request a 0-th quantile.")
    if k >= P:
        raise OracleEncodeInfeasible(
            f"oracle_encode: mean L0={target:.2f} needs {k} firing entries but only {P} projections "
            f"are positive (~{P / max(n, 1):.1f}/row); a non-negative JumpReLU gate cannot reach it "
            f"(the tied-unit-g encoder fires ≲ F/2 latents/row). The checkpoint is denser than this "
            f"oracle encoder can mirror — do not silently plateau the ceiling.")
    theta = torch.kthvalue(pos, P - k + 1).values          # k-th largest POSITIVE value (> 0)
    return proj > theta, g_unit


def oracle_encode(h: torch.Tensor, g: torch.Tensor, realized_l0: float,
                  lam: float = RIDGE_LAMBDA) -> torch.Tensor:
    """The honest perfect-GEOMETRY SAE activations: a tied unit-g JumpReLU gate for the firing SUPPORT,
    then a per-token RIDGE least-squares on that support for the MAGNITUDES (reconstructing with the raw
    decoder g). `acts[t, S] = argmin_a ‖h_t − a·g_S‖² + λ‖a‖²`, zero off the support S.

    The gate is the standard tied init `W_enc = g_unit`, so a child's support carries its PARENT (the
    designed overlap cos(g_child, g_parent)=α), giving an honest, non-degenerate firing structure — NOT
    the exact oracle coefficients A (which give coverage_R ≡ 1, a degenerate ceiling). The ridge solve is
    the KNOWN_BUGS 4.4 fix: the earlier tied-projection magnitudes reconstructed h at FVU≈0.86–1.36
    (unit-g coefficients against raw-g decoders — a scale mismatch), so a reconstruction detector on them
    measured noise; the gated least-squares magnitudes reach FVU≈0.14 (achievable, not the F>D-idealised
    true-A 0.027). The support is the gate's, so the firing structure — hence the co-firing detectors — is
    unchanged; only the magnitude detectors (recon_2a, joint_child_mass) see the corrected values. A ridge
    coefficient that comes out ≤0 on the gated support simply does not fire (`acts > 0`), the honest
    JumpReLU semantics — measured ~5% of gated entries, with no effect on is_a coverage.

    `realized_l0` MUST be the trained SAE's realized L0 on real h (not the nominal top-k): the two sides
    are only comparable read at the same sparsity. Raises `OracleEncodeInfeasible` on a non-positive /
    unreachable L0 rather than silently flooring.
    """
    support, _ = _gate_support(h, g, realized_l0)
    gd = g.double()
    n, F = support.shape
    acts = torch.zeros(n, F, dtype=torch.float64, device=gd.device)
    hd = h.double()
    for t in range(n):
        S = support[t].nonzero(as_tuple=True)[0]
        if S.numel() == 0:
            continue
        Gs = gd[S]                                         # [k, D] raw decoder rows on the support
        gram = Gs @ Gs.transpose(0, 1)                     # [k, k]
        gram = gram + lam * torch.eye(S.numel(), dtype=torch.float64, device=gd.device)
        acts[t, S] = torch.linalg.solve(gram, Gs @ hd[t])  # [k] ridge least-squares magnitudes
    return acts


def reconstruction_fvu(h: torch.Tensor, acts: torch.Tensor, g: torch.Tensor) -> float:
    """Fraction of variance unexplained reconstructing h with the RAW decoder g from `acts`
    (`h_hat = acts @ g`). The magnitude figure of merit for the α-encoder: tied projection ≈0.86,
    gated ridge least-squares ≈0.14."""
    h_hat = acts.double() @ g.double()
    err = h.double() - h_hat
    return float((err ** 2).sum() / (h.double() ** 2).sum().clamp_min(_TINY))


def _constant_scored_detectors(detectors: dict[str, torch.Tensor],
                               pairs: list[tuple[int, int]], tol: float = 1e-9) -> list[str]:
    """Names of detectors whose scored values over `pairs` cannot produce a meaningful AUROC — either
    FEWER than 2 finite values (undefined / all-NaN column: the MAXIMAL degeneracy) or a finite range
    below `tol` (a constant column → a meaningless ~0.5 from ties). Both are flagged (an all-NaN
    detector is strictly more broken than a constant one, so it must not be silently exempted). Sorted."""
    degen: list[str] = []
    for det, mat in detectors.items():
        vals = torch.tensor([float(mat[p, c]) for (p, c) in pairs], dtype=torch.float64)
        vals = vals[torch.isfinite(vals)]                  # drop NaN AND ±inf: an inf-emitting detector
        if vals.numel() < 2 or float(vals.max() - vals.min()) < tol:   # bug is broken, not "wide-range"
            degen.append(det)
    return sorted(degen)


# --------------------------------------------------------------------------
# the validation battery
# --------------------------------------------------------------------------
def _pure_inputs(bundle: "WorldBundle", feats: list[int], acts: torch.Tensor) -> DetectorInputs:
    """DetectorInputs on the recovered features with a given activation matrix and the TRUE decoders."""
    idx = torch.tensor(feats, dtype=torch.long)
    g_sel = bundle.g[idx]
    W_unit = g_sel / g_sel.norm(dim=1, keepdim=True).clamp_min(_TINY)
    return DetectorInputs(
        acts_rec=acts, W_unit=W_unit, W_raw=g_sel, h=bundle.h,
        b_dec=torch.zeros(bundle.g.shape[1], dtype=bundle.g.dtype),
        tokens=bundle.tokens, vocab=bundle.cfg.vocab,
    )


def _detector_validity(detectors: dict[str, torch.Tensor], pairs: list[tuple[int, int]],
                       tol: float) -> dict[str, dict]:
    """Per detector over `pairs`: finite-value count + fraction, finiteness (≥2), degeneracy
    (constant/all-NaN). `finite_frac` makes a near-total breakage VISIBLE in the report — a detector
    that is finite on only a handful of pairs still clears the ≥2 `finite` gate (many detectors are
    legitimately sparse — a dead-parent NaN, an unmeasurable rare-bucket), so the gate stays lenient
    but the fraction surfaces a detector that has quietly gone NaN almost everywhere."""
    degen = set(_constant_scored_detectors(detectors, pairs, tol))
    n_pairs = len(pairs)
    out: dict[str, dict] = {}
    for det, mat in detectors.items():
        vals = torch.tensor([float(mat[p, c]) for (p, c) in pairs], dtype=torch.float64)
        n_finite = int(torch.isfinite(vals).sum())         # finite := not NaN and not ±inf
        out[det] = {"n_finite": n_finite, "finite": n_finite >= 2, "degenerate": det in degen,
                    "finite_frac": n_finite / n_pairs if n_pairs else 0.0}
    return out


def machinery_report(bundle: "WorldBundle", feats: list[int], pairs: list[tuple[int, int]],
                     realized_l0: float, constants: dict | None = None,
                     tol: float = 1e-9, lam: float = RIDGE_LAMBDA) -> dict:
    """Run `compute_all` on BOTH pure regimes (true_A, alpha_encoder) over the recovered features and
    report each detector's finiteness + non-degeneracy per regime, plus the α-encoder reconstruction FVU.

    A detector PASSES the battery when it is finite AND non-degenerate in AT LEAST ONE regime — a
    detector that is degenerate in one regime BY DESIGN (coverage_R ≡ 1 for is_a on true_A) but healthy in
    the other is correct machinery; only a detector broken in BOTH regimes is a machinery failure.
    """
    constants = _REGISTRY_CONSTANTS if constants is None else constants
    idx = torch.tensor(feats, dtype=torch.long)
    alpha_full = oracle_encode(bundle.h, bundle.g, realized_l0, lam=lam)   # full dictionary
    alpha_acts = alpha_full[:, idx]
    trueA_acts = bundle.A[:, idx]
    regimes = {
        "true_A": compute_all(_pure_inputs(bundle, feats, trueA_acts), constants),
        "alpha_encoder": compute_all(_pure_inputs(bundle, feats, alpha_acts), constants),
    }
    per_regime = {name: _detector_validity(dets, pairs, tol) for name, dets in regimes.items()}
    detectors = list(regimes["true_A"].keys())
    passed = {
        det: any(per_regime[r][det]["finite"] and not per_regime[r][det]["degenerate"]
                 for r in per_regime)
        for det in detectors
    }
    return {
        "per_regime": per_regime,
        "passed": passed,
        "all_passed": all(passed.values()),
        "failed": sorted(d for d, ok in passed.items() if not ok),
        # reconstruction is a FULL-dictionary property of the encoder (KNOWN_BUGS 4.4 target ~0.14):
        # score the WHOLE oracle_encode output against all g, NOT the recovered-feature slice (a subset
        # would drop the support latents outside `feats` and spuriously inflate the FVU).
        "alpha_encoder_fvu": reconstruction_fvu(bundle.h, alpha_full, bundle.g),
    }


def _corr(x: torch.Tensor, y: torch.Tensor, kind: str) -> float:
    if x.numel() < 2:
        return float("nan")                                # undefined, NOT a real zero correlation
    if kind == "spearman":
        x = x.argsort().argsort().double()
        y = y.argsort().argsort().double()
    x = x.double() - x.double().mean()
    y = y.double() - y.double().mean()
    return float((x @ y) / (x.norm() * y.norm() + _TINY))


def s_res_calibration(bundle: "WorldBundle", feats: list[int], constants: dict | None = None,
                      device: str | None = None) -> dict:
    """Validate the s_res PROBE machinery against the analytic geometry: on TRUE firing + TRUE geometry
    the probe s_res (`probe_true_g`) must track the closed-form cosine s_res (`cosine_g`). Both use TRUTH
    only, so this is pure machinery-validation (no SAE). Reports Pearson/Spearman + mean|diff| over the
    off-diagonal pairs where both are defined. The caller freezes the `sres_*` knobs to the setting that
    holds this agreement on a held-out draw.

    `device` (e.g. "cuda") moves the probe inputs there before training — `train_probe` draws its
    permutations on CPU deterministically (the same sampled tokens on any device), so the probe fits the
    SAME data, just faster. The Adam optimisation runs on-device, so CPU and GPU numbers agree to
    optimisation tolerance (well within the calibration floor), NOT bit-for-bit."""
    constants = _REGISTRY_CONSTANTS if constants is None else constants
    idx = torch.tensor(feats, dtype=torch.long)
    g_sel = bundle.g[idx]
    g_unit = g_sel / g_sel.norm(dim=1, keepdim=True).clamp_min(_TINY)
    A_rec = bundle.A[:, idx]
    h = bundle.h
    if device is not None:
        g_unit, A_rec, h = g_unit.to(device), A_rec.to(device), h.to(device)
    cosine_g = s_res_cosine(g_unit)
    probe_true_g = s_res_probe(A_rec, h, g_unit, constants, label_acts=A_rec)
    R = cosine_g.shape[0]
    offdiag = ~torch.eye(R, dtype=torch.bool, device=cosine_g.device)
    cv, pv = cosine_g[offdiag], probe_true_g[offdiag]
    ok = torch.isfinite(cv) & torch.isfinite(pv)           # drop NaN AND ±inf pairs
    cv, pv = cv[ok], pv[ok]
    enough = cv.numel() >= _MIN_CALIB_PAIRS
    return {
        "n_pairs": int(cv.numel()),
        "pearson": _corr(cv, pv, "pearson") if enough else float("nan"),
        "spearman": _corr(cv, pv, "spearman") if enough else float("nan"),
        "mean_abs_diff": float((cv - pv).abs().mean()) if enough else float("nan"),
    }
