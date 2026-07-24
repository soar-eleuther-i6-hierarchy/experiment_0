"""
Energy shares + exact joint-child coverage (metrics_todo.md T2; landscape
Appendix A-B and §C.1 Leg 2/3).

Computed from the stage-01 accumulators (schema v2):

    Share_energy(c, p) = sum_x f_p(x)^2 * 1[f_c(x)>0] / sum_x f_p(x)^2
        fraction of the parent's activation ENERGY spent on tokens where this
        one child fires. One child with share ~1 = rename/duplicate candidate;
        a healthy partition has many children with small shares. NOT a
        partition: shares sum > 1 when siblings co-fire.

    R_supp(p) = |{x : f_p>0 and >=1 child fires}| / |{x : f_p>0}|
        EXACT support version of joint-child coverage. Replaces the
        min(1, sum F) shortcut, which double-counts co-firing children and
        saturates near 1 for any parent with many children (the number the
        old reports showed at 0.92-0.97 regardless of structure).

    R_mass(p) = sum_x f_p^2 * 1[>=1 child fires] / sum_x f_p^2
        energy version; the superparent signature is reverse coverage ~1 for
        many children while R_mass ~ 0 (parent's mass lives where no child is).

Union here is over ALL next-block children (streamable in one pass); the
kept-children-only union depends on the edge set and is computed in the
second pass (run_second_pass.py).
"""

from __future__ import annotations

import torch

_EPS = 1e-12


def share_energy(energy_cofire: torch.Tensor, energy_total: torch.Tensor) -> torch.Tensor:
    """[P, C] energy share per (parent, child). Zero-energy parents get 0."""
    return energy_cofire.double() / energy_total.double().clamp(min=_EPS).unsqueeze(1)


def r_supp(union_count: torch.Tensor, fire_p: torch.Tensor) -> torch.Tensor:
    """[P] exact support joint-child coverage."""
    return union_count.double() / fire_p.double().clamp(min=1.0)


def r_mass(union_energy: torch.Tensor, energy_total: torch.Tensor) -> torch.Tensor:
    """[P] energy-weighted joint-child coverage."""
    return union_energy.double() / energy_total.double().clamp(min=_EPS)
