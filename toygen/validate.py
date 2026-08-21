"""
Feasibility guards for a built toy.

Reject any config whose sampled world would silently disagree with the ground-truth answer
key. Each check raises `ValueError` at build time, so an infeasible config fails loudly
instead of producing a wrong `truth.pt` without erroring. These are structural checks only;
the old energy-feasibility constraints are gone.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from .spec import ToyConfig
from .strengths import firing_rates, topic_rates

if TYPE_CHECKING:
    from .tree import Tree

DT = torch.float64


def validate_config(cfg: ToyConfig, tree: Tree) -> None:
    """Raise `ValueError` if the built tree/config is infeasible; return None if OK.

    Called at the end of `build_tree`. Checks only the structural feasibility that the sampler
    and the answer key both rely on.
    """
    # Basic config sanity -- fail fast on degenerate values before any per-feature work (e.g. Z=0 would divide by zero below).
    if cfg.Z < 1:
        raise ValueError(f"Z (number of topics) must be >= 1 (got {cfg.Z})")
    if cfg.vocab < cfg.n_bind_ids:
        raise ValueError(
            f"vocab ({cfg.vocab}) must be >= n_bind_ids ({cfg.n_bind_ids}); the "
            f"token-bound id set would fall outside the vocabulary and never fire")
    if cfg.doc_len <= 0:
        raise ValueError(f"doc_len must be > 0 (got {cfg.doc_len})")
    if cfg.D <= 0:
        raise ValueError(f"D (activation dimension) must be > 0 (got {cfg.D})")

    if cfg.noise_sigma <= 0:
        raise ValueError(
            f"noise_sigma must be > 0 (got {cfg.noise_sigma}); a zero-noise world "
            f"gives the reconstruction metric a zero residual to divide by")

    if not (0.0 < cfg.strength_spread < 0.5):
        raise ValueError(
            f"strength_spread must be in (0, 0.5) (got {cfg.strength_spread}); "
            f"outside it the firing-strength floor goes non-positive")

    # Per-edge firing probability and per-feature loading bounds, keeping the eps_p / eps_alpha safety margins.
    for k in range(tree.F):
        if tree.parent_of(k) is None:
            continue
        pe = tree.p_edge_of(k)
        if pe > 1.0 - cfg.eps_p:
            raise ValueError(
                f"child_p_edge must be <= 1 - eps_p = {1.0 - cfg.eps_p:.4f}; "
                f"feature {k} has p_edge {pe}")
        a = tree.alpha_of(k)
        if a > 1.0 - cfg.eps_alpha:
            raise ValueError(
                f"alpha must be <= 1 - eps_alpha = {1.0 - cfg.eps_alpha:.4f}; "
                f"feature {k} has alpha {a}")

    # Exclusive siblings split one draw by p_edge, so per-edge probs must sum to <= 1 -- above that the sampler truncates the realized rate while firing_rates keeps the nominal one. Non-exclusive parents carry no such budget and are skipped.
    for par, kids in tree.children.items():
        if not tree.exclusive.get(par, False) or not kids:
            continue
        tot = sum(tree.p_edge_of(c) for c in kids)
        if tot > 1.0 + 1e-9:
            detail = ", ".join(f"{c}:{tree.p_edge_of(c):.3f}" for c in kids)
            raise ValueError(
                f"sibling budget: exclusive parent {par} has {len(kids)} children "
                f"with sum p_edge = {tot:.4f} > 1 (children p_edge -> {detail}); the "
                f"sampler would truncate the last sibling's realized rate while "
                f"firing_rates keeps the nominal rate, so truth.pt would disagree "
                f"with the sampled world")

    # Topical admissibility: per-topic firing rate must stay in [0, 1], or the "averages back to exactly p_i" guarantee breaks silently (`rand < rate` just never/always fires).
    p = firing_rates(tree)
    pi = torch.full((cfg.Z,), 1.0 / cfg.Z, dtype=DT)
    for k in range(tree.F):
        if tree.kappa[k] <= 0.0:
            continue
        rates = topic_rates(float(p[k]), tree.kappa[k], tree.topic[k], pi)
        if float(rates.min()) < 0.0 or float(rates.max()) > 1.0:
            raise ValueError(
                f"topical admissibility: feature {k} has per-topic firing rates "
                f"outside [0, 1] (min {float(rates.min()):.4f}, "
                f"max {float(rates.max()):.4f}); reduce kappa or increase Z")

    # Token-bound admissibility: firing rate can't exceed the top-n_bind_ids id set's Zipf mass, or the sampler silently caps the realized rate while firing_rates keeps the nominal p_k.
    if any(tree.token_bound[k] for k in range(tree.F)):
        ranks = torch.arange(1, cfg.vocab + 1, dtype=DT)
        w = ranks ** (-cfg.zipf_s)
        bind_mass = float(w[:cfg.n_bind_ids].sum() / w.sum())
        for k in range(tree.F):
            if not tree.token_bound[k]:
                continue
            if float(p[k]) > bind_mass + 1e-9:
                raise ValueError(
                    f"token-bound admissibility: feature {k} has firing rate "
                    f"p = {float(p[k]):.4f} above the top-{cfg.n_bind_ids} Zipf mass "
                    f"{bind_mass:.4f}; the sampler would cap the realized rate below "
                    f"p, so truth.pt would disagree with the sampled world "
                    f"(raise n_bind_ids or lower the token-bound rate)")
