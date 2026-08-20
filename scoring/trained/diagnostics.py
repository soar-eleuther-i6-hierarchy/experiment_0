"""Fallback-vs-exact diagnostic for the joint-child coverage J(p).

`detectors.joint_child_J` reports the capped sum of forward coverages -- an UPPER BOUND on the
true joint coverage. This compares it against the exact reverse-support
`r_supp(p) = P(>=1 kept child fires | parent fires)`, computed from the in-memory firing matrix,
so the fallback's tightness can be measured on the toy. A DIAGNOSTIC only -- never a scored
detector, never a member of the frozen `DETECTORS` grid.
"""
from __future__ import annotations

import torch

from scoring.core.detectors import _forward_F, cofiring, coverage_R, edge_mask
from scoring.core.grid import _spearman

DT = torch.float64
_NAN = float("nan")


def j_fallback_vs_exact(di, constants: dict) -> dict:
    """Per-parent fallback J(p) vs exact r_supp(p), and their agreement.

    Recomputes `Fm`/`cofire`/`fire`/`F_mat`/`em` exactly as `compute_all` does (only
    `di.acts_rec` is needed), so the fallback mirrors `joint_child_J`. Dead parents are NaN on
    both sides. Returns per-parent lists plus `mean_gap`/`max_gap`/`spearman`/`n_parents` over
    the parents finite on both sides.
    """
    eps = constants["coverage_eps"]
    Fm = di.acts_rec > constants["fire_thresh"]
    cofire, fire, _N = cofiring(Fm)
    R_mat = coverage_R(cofire, fire, eps)
    F_mat = _forward_F(cofire, fire, eps)
    em = edge_mask(R_mat, fire, constants["edge_tau"], constants["min_fire_count"],
                   cofire=cofire, min_joint=constants["min_joint"])
    R = int(Fm.shape[1])

    # fallback J(p): mirror joint_child_J (capped forward-coverage sum; dead parent -> NaN)
    contrib = torch.where(em, F_mat, torch.zeros_like(F_mat))
    j_fb = contrib.sum(dim=1).clamp(max=1.0)
    j_fb = torch.where(fire > 0, j_fb, torch.full_like(j_fb, _NAN)).double()

    # exact r_supp(p) = |{tokens: >=1 kept child fires AND parent fires}| / fire_p
    r_supp = torch.full((R,), _NAN, dtype=DT)
    for p in range(R):
        if float(fire[p]) <= 0.0:
            continue
        kids = em[p].nonzero(as_tuple=True)[0]
        if kids.numel() == 0:
            r_supp[p] = 0.0
            continue
        any_child = Fm[:, kids].any(dim=1)
        r_supp[p] = (any_child & Fm[:, p]).double().sum() / fire[p].double()

    both = (~torch.isnan(j_fb)) & (~torch.isnan(r_supp))
    gap = j_fb - r_supp
    fin = gap[both]
    spear = _spearman(j_fb[both], r_supp[both]) if int(both.sum()) >= 2 else _NAN
    return {
        "j_fallback": j_fb.tolist(),
        "r_supp_exact": r_supp.tolist(),
        "gap": gap.tolist(),
        "mean_gap": float(fin.mean()) if fin.numel() else _NAN,
        "max_gap": float(fin.max()) if fin.numel() else _NAN,
        "spearman": spear,
        "n_parents": int(both.sum()),
    }
