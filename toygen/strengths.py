"""
The firing and strength model of the normal toy generator.

Two designed choices live here, both deliberately flat:
  - firing rate p_k walks the forest downward, p_child = p_parent * p_edge;
  - every feature is equally loud (mean active strength = q * sqrt(E0)), so a feature's
    residual energy varies only through its firing rate, never a designed loudness
    ladder.

`topic_rates` is the per-topic firing profile used by the topical confounds; it
marginalises back to exactly the feature's overall rate.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from .spec import ToyConfig
from .tree import Tree

DT = torch.float64


@dataclass
class StrengthSpec:
    p: torch.Tensor              # [F] firing rate per feature
    mean_strength: torch.Tensor  # [F] mean active strength, flat at q * sqrt(E0)
    topic_prior: torch.Tensor    # [Z] uniform prior over document topics


def firing_rates(tree: Tree) -> torch.Tensor:
    """Firing rate p_k, walking the forest downward: p_child = p_parent * p_edge."""
    p = torch.zeros(tree.F, dtype=DT)
    for k in tree.topology_ordering:
        par = tree.parent_of(k)
        p[k] = tree.root_p[k] if par is None else p[par] * tree.p_edge_of(k)
    return p


def topic_rates(p_i: float | torch.Tensor, kappa: float, z_i: int | None,
                pi: torch.Tensor) -> torch.Tensor:
    """Per-topic firing rate that marginalises back to `p_i` under a uniform prior.

    The `- pi[z_i]` centring makes the marginal exact when `pi` is uniform — which is
    what `build_strengths` always constructs. Without it the rate integrates to
    `p_i (1 + kappa pi)` while still looking like a valid probability.
    """
    Z = pi.numel()
    if z_i is None or kappa == 0.0:
        return torch.full((Z,), float(p_i), dtype=DT)
    onehot = torch.zeros(Z, dtype=DT)
    onehot[z_i] = 1.0
    return float(p_i) * (1.0 + kappa * (onehot - pi))


def build_strengths(cfg: ToyConfig, tree: Tree) -> StrengthSpec:
    """Firing rates, a flat loudness, and a uniform topic prior.

    mean_strength = q * sqrt(E0) with q = 1 / sqrt(1 + strength_spread^2), identical
    for every feature: the toy has no designed energy ladder, so the only energy
    spread is the one feature firing already creates.
    """
    p = firing_rates(tree)
    q = 1.0 / math.sqrt(1.0 + cfg.strength_spread ** 2)
    mean_strength = torch.full((tree.F,), q * math.sqrt(cfg.E0), dtype=DT)
    topic_prior = torch.full((cfg.Z,), 1.0 / cfg.Z, dtype=DT)
    return StrengthSpec(p=p, mean_strength=mean_strength, topic_prior=topic_prior)
