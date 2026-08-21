"""
Stage 1: mathematical validation of the detector machinery. No AUROC.

Prefilters that each detector equation is implemented correctly and non-degenerate on
PURE inputs, before the trained stage measures retrieval on real SAE latents. Answers
"is the machinery correct?", not "how well does the metric rank?".

Two pure firing regimes feed `compute_all`:
  * ``true_A``        — oracle coefficients A: exact firing structure, faithful
                        reconstruction (h = A@g + noise).
  * ``alpha_encoder`` — an honest gated encoder: a tied unit-g JumpReLU gate picks the
                        firing support, a per-token ridge least-squares supplies the
                        magnitudes. Only the magnitude detectors (recon_2a,
                        joint_child_mass) see the improved values.

Each detector is checked for finiteness (≥2 finite values) and non-degeneracy
(`_constant_scored_detectors`); a detector degenerate in one regime by design but
healthy in the other is fine — only failure in BOTH regimes flags it.

A third check validates the s_res PROBE against the analytic cosine s_res on TRUE
firing + TRUE geometry.

Also houses `oracle_encode`, `OracleEncodeInfeasible`, `_constant_scored_detectors`.
Firewall: the pure regimes use TRUTH (A, g) only; they never score the trained SAE.
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

# Ridge shrinkage for the gated least-squares magnitudes. Fixed, not tuned to any SAE.
RIDGE_LAMBDA = 1e-4

# Below this many defined off-diagonal pairs, `s_res_calibration` reports NaN rather than a meaningless correlation.
_MIN_CALIB_PAIRS = 100


class OracleEncodeInfeasible(ValueError):
    """The oracle encoder cannot produce a valid firing for this draw.

    Raised when realized_l0 is bad, or the checkpoint is denser than a tied-unit-g
    JumpReLU can mirror. Subclasses ValueError so existing ``except ValueError``
    callers still catch it, but is narrow enough to catch ONLY this without also
    swallowing a compute_all config error."""


def _gate_support(h: torch.Tensor, g: torch.Tensor, realized_l0: float
                  ) -> tuple[torch.Tensor, torch.Tensor]:
    """Tied unit-g JumpReLU gate: pick the firing support at a target sparsity.

    Computes `proj = h @ g_unitᵀ` and a single uniform threshold θ over the POSITIVE
    projections, chosen so mean row-L0 == realized_l0. Returns (support_mask [n, F],
    g_unit [F, D]).

    θ is calibrated over positive projections only (a JumpReLU gate is non-negative),
    so it stays strictly positive and tracks the target; thresholding over all entries
    would let θ drift ≤ 0 on dense checkpoints and silently plateau the L0."""
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
            f"oracle_encode: mean L0={target:.3g} rounds to k=0 firing entries over n={n} tokens, "
            f"i.e. a (near-)dead SAE the gate cannot represent. Failing loud rather than requesting a "
            f"0-th quantile.")
    if k >= P:
        raise OracleEncodeInfeasible(
            f"oracle_encode: mean L0={target:.2f} needs {k} firing entries but only {P} projections "
            f"are positive (~{P / max(n, 1):.1f}/row), which a non-negative JumpReLU gate cannot reach "
            f"(the tied-unit-g encoder fires at most ~F/2 latents/row). The checkpoint is denser than "
            f"this oracle encoder can mirror; refusing to silently plateau the ceiling.")
    theta = torch.kthvalue(pos, P - k + 1).values          # k-th largest POSITIVE value (> 0)
    return proj > theta, g_unit


def oracle_encode(h: torch.Tensor, g: torch.Tensor, realized_l0: float,
                  lam: float = RIDGE_LAMBDA) -> torch.Tensor:
    """Honest perfect-geometry SAE activations from a gated ridge least-squares.

    Two steps: (1) a tied unit-g JumpReLU gate picks which latents fire (support S);
    (2) a per-token ridge least-squares on that support sets magnitudes so the
    reconstruction with the raw decoder g is good: `acts[t, S] = argmin_a
    ‖h_t − a·g_S‖² + λ‖a‖²`, zero off the support.

    The tied gate init means a child's support carries its parent (designed overlap
    cos(g_child, g_parent)=α), giving an honest, non-degenerate firing structure —
    unlike the exact oracle coefficients A, which give a degenerate coverage_R ≡ 1
    ceiling. The ridge solve fixes tied-projection magnitudes reconstructing h poorly;
    only the magnitude detectors (recon_2a, joint_child_mass) see the corrected values.

    `realized_l0` MUST be the trained SAE's realized L0 on real h, not the nominal
    top-k. Raises `OracleEncodeInfeasible` on a non-positive or unreachable L0.
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
    """Fraction of variance unexplained when reconstructing h with the RAW decoder g
    (`h_hat = acts @ g`). The magnitude figure of merit for the alpha-encoder: tied
    projection ≈ 0.86, gated ridge least-squares ≈ 0.14."""
    h_hat = acts.double() @ g.double()
    err = h.double() - h_hat
    return float((err ** 2).sum() / (h.double() ** 2).sum().clamp_min(_TINY))


def _constant_scored_detectors(detectors: dict[str, torch.Tensor],
                               pairs: list[tuple[int, int]], tol: float = 1e-9) -> list[str]:
    """Names of detectors whose scored values over `pairs` cannot yield a meaningful AUROC.

    Two failure cases, both flagged (sorted):
      * fewer than 2 finite values — an undefined / all-NaN column (maximal degeneracy).
      * a finite range below `tol` — a constant column, which gives a meaningless ~0.5
        AUROC from ties.
    An all-NaN detector is strictly more broken than a constant one, so it is never
    silently exempted."""
    degen: list[str] = []
    for det, mat in detectors.items():
        vals = torch.tensor([float(mat[p, c]) for (p, c) in pairs], dtype=torch.float64)
        vals = vals[torch.isfinite(vals)]                  # drop NaN and ±inf (e.g. an inf-emitting detector)
        if vals.numel() < 2 or float(vals.max() - vals.min()) < tol:
            degen.append(det)
    return sorted(degen)


# --------------------------------------------------------------------------
# the validation battery
# --------------------------------------------------------------------------
def _pure_inputs(bundle: "WorldBundle", feats: list[int], acts: torch.Tensor) -> DetectorInputs:
    """Build DetectorInputs on the recovered features from a given activation matrix and
    the TRUE decoders."""
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
    """Per-detector validity over `pairs`: finite-value count and fraction, finiteness
    (≥2), and degeneracy (constant / all-NaN).

    `finite_frac` surfaces near-total breakage that still clears the lenient ≥2
    `finite` gate (many detectors are legitimately sparse, e.g. a dead-parent NaN)."""
    degen = set(_constant_scored_detectors(detectors, pairs, tol))
    n_pairs = len(pairs)
    out: dict[str, dict] = {}
    for det, mat in detectors.items():
        vals = torch.tensor([float(mat[p, c]) for (p, c) in pairs], dtype=torch.float64)
        n_finite = int(torch.isfinite(vals).sum())         # finite = not NaN and not ±inf
        out[det] = {"n_finite": n_finite, "finite": n_finite >= 2, "degenerate": det in degen,
                    "finite_frac": n_finite / n_pairs if n_pairs else 0.0}
    return out


def machinery_report(bundle: "WorldBundle", feats: list[int], pairs: list[tuple[int, int]],
                     realized_l0: float, constants: dict | None = None,
                     tol: float = 1e-9, lam: float = RIDGE_LAMBDA) -> dict:
    """Run `compute_all` on BOTH pure regimes (true_A, alpha_encoder) over the recovered
    features and report each detector's finiteness and non-degeneracy per regime, plus the
    alpha-encoder reconstruction FVU.

    A detector PASSES the battery when it is finite AND non-degenerate in AT LEAST ONE
    regime. A detector that is degenerate in one regime by design (coverage_R ≡ 1 for is_a
    on true_A) but healthy in the other is correct machinery; only a detector broken in
    BOTH regimes is a machinery failure.
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
        # Score the WHOLE oracle_encode output against all g, not the recovered-feature slice, to avoid inflating FVU.
        "alpha_encoder_fvu": reconstruction_fvu(bundle.h, alpha_full, bundle.g),
    }


def _corr(x: torch.Tensor, y: torch.Tensor, kind: str) -> float:
    if x.numel() < 2:
        return float("nan")                                # undefined, not a real zero correlation
    if kind == "spearman":
        x = x.argsort().argsort().double()
        y = y.argsort().argsort().double()
    x = x.double() - x.double().mean()
    y = y.double() - y.double().mean()
    return float((x @ y) / (x.norm() * y.norm() + _TINY))


def s_res_calibration(bundle: "WorldBundle", feats: list[int], constants: dict | None = None,
                      device: str | None = None) -> dict:
    """Validate the s_res PROBE machinery against the analytic geometry.

    On TRUE firing + TRUE geometry, the probe s_res (`probe_true_g`) must track the
    closed-form cosine s_res (`cosine_g`) — pure machinery-validation, no SAE involved.
    Reports Pearson, Spearman, and mean|diff| over the off-diagonal pairs where both
    are defined.

    `device` moves the probe inputs there before training; permutations are drawn
    deterministically on CPU regardless, so CPU and GPU numbers agree to optimisation
    tolerance, not bit-for-bit."""
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
    ok = torch.isfinite(cv) & torch.isfinite(pv)           # drop NaN and ±inf pairs
    cv, pv = cv[ok], pv[ok]
    enough = cv.numel() >= _MIN_CALIB_PAIRS
    return {
        "n_pairs": int(cv.numel()),
        "pearson": _corr(cv, pv, "pearson") if enough else float("nan"),
        "spearman": _corr(cv, pv, "spearman") if enough else float("nan"),
        "mean_abs_diff": float((cv - pv).abs().mean()) if enough else float("nan"),
    }
