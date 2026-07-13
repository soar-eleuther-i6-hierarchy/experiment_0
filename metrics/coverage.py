"""
Metric 1 - Activation coverage, three legs (project plan Exp 0, bullet 1).

For a candidate edge parent p -> child c, from co-firing counts:

    reverse  R = cofire(p,c) / fire(c) = P(parent fires | child fires)
        "child contained in parent" - the classic edge criterion.

    forward  F = cofire(p,c) / fire(p) = P(child fires | parent fires)
        "how much of the parent this one child accounts for".

    joint-child J(p) = P(at least one kept child fires | parent fires)
        "children together account for the parent". Exact J needs the union
        count over tokens (computed in the streaming pass); the closed-form
        upper bound min(1, sum_c F) is provided as a fallback.

Known blind spot (why the other metrics exist): coverage only sees
co-OCCURRENCE. A child that happens to fire inside a very frequent parent
gets a high R without any semantic or reconstructive relationship.
"""

from __future__ import annotations

import torch


def coverage_legs(
    cofire: torch.Tensor,      # [P, C] co-firing counts
    fire_p: torch.Tensor,      # [P]    parent firing counts
    fire_c: torch.Tensor,      # [C]    child firing counts
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (R, F), both [P, C]. Zero-fire features get coverage 0."""
    cofire = cofire.double()
    R = cofire / fire_c.double().clamp(min=1.0).unsqueeze(0)
    F = cofire / fire_p.double().clamp(min=1.0).unsqueeze(1)
    return R, F


def keep_edges(
    R: torch.Tensor,           # [P, C] reverse coverage
    fire_p: torch.Tensor,
    fire_c: torch.Tensor,
    tau: float,
    min_fire: int,
) -> torch.Tensor:
    """Boolean [P, C] edge mask: R >= tau, both endpoints fire often enough."""
    keep = R >= tau
    keep[fire_p < min_fire, :] = False
    keep[:, fire_c < min_fire] = False
    return keep


def joint_child_coverage_upper(
    F: torch.Tensor,           # [P, C] forward coverage
    edge_mask: torch.Tensor,   # [P, C] kept edges
) -> torch.Tensor:
    """Upper bound on J(p): sum of forward coverages of kept children, capped at 1.

    Children can co-fire, so the sum double-counts; the streaming union count
    (see 01_cache_stats.py) gives the exact value.
    """
    return (F * edge_mask).sum(dim=1).clamp(max=1.0)


def joint_child_coverage_exact(
    union_count: torch.Tensor,  # [P] tokens where parent fires AND >=1 kept child fires
    fire_p: torch.Tensor,       # [P]
) -> torch.Tensor:
    """Exact J(p) = union_count / fire(p)."""
    return union_count.double() / fire_p.double().clamp(min=1.0)
