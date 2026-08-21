"""
Feature directions, built in constructive order.

The order matters and cannot be inverted: `u` is drawn first and `g` is built from it.
Deriving `u` from `g` while also drawing `u` under a coherence bound is circular, and
the two per-feature steps must interleave: orthogonalising `u_k` against its ancestor
span needs `g` for every ancestor, which the earlier features have already built.

The `Lambda` returned here is read off the construction, not `U^T G` (the two differ
once siblings are involved); the toygen tests pin this Gram-vs-Lambda distinction.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from .spec import ToyConfig
from .tree import Tree

DT = torch.float64


@dataclass
class Geometry:
    u: torch.Tensor            # [F, D] residual directions
    g: torch.Tensor            # [F, D] concept directions
    Lam: torch.Tensor          # [F, F] constructive change of basis
    coherence: float           # realised max |cos(u_a, u_b)| after orthogonalisation
    # build_directions is best-effort: it returns the lowest of 8 draws rather than
    # raising, so the F/D sweep survives even when the target coherence is missed.
    # coherence_ok = (coherence <= max_unrelated_cos): a SOFT "target met" flag, NOT a
    # health check. At F ≫ D it is EXPECTED False — the repel (`_repel`) only pushes pairs
    # already ABOVE the cap, so realised coherence lands just ABOVE the cap (on `full`:
    # cap 0.12→0.156, 0.30→0.316) — it overshoots the cap, which is exactly why the flag is
    # False. LOWERING the cap gives cleaner unrelated directions, RAISING it does the opposite
    # (KNOWN_BUGS 5.1). A False flag means "the cap is tighter than the F≫D packing allows in
    # 60 repel steps", not "the geometry is bad".
    coherence_ok: bool


def _repel(x: torch.Tensor, max_unrelated_cos: float, steps: int = 60) -> torch.Tensor:
    """Push unit vectors apart until pairwise coherence approaches `max_unrelated_cos`.

    Plain rejection sampling is hopeless at F >> D — random unit vectors in R^D have
    max pairwise |cos| around sqrt(2 log F^2 / D), which already exceeds a tight max_unrelated_cos.
    A few steps of repulsion gets much closer, and the realised value is reported
    rather than asserted.
    """
    x = x / x.norm(dim=1, keepdim=True)
    for _ in range(steps):
        gram = x @ x.T
        gram.fill_diagonal_(0.0)
        over = gram.abs() > max_unrelated_cos
        if not over.any():
            break
        push = (gram * over.to(DT)) @ x
        x = x - 0.10 * push
        x = x / x.norm(dim=1, keepdim=True)
    return x


def build_directions(cfg: ToyConfig, tree: Tree, seed: int | None = None) -> Geometry:
    gen = torch.Generator().manual_seed(cfg.seed if seed is None else seed)
    F, D = tree.F, cfg.D

    # `max_unrelated_cos` is a target, not a hard bound: orthogonalising against ancestor
    # spans moves the vectors, so the realised coherence isn't controllable from the draw
    # (and a tight bound is unreachable by rejection at F >> D; see DESIGN.md). Best-effort
    # over 8 draws, keep the lowest, and record whether the target was met in `coherence_ok`.
    best: Geometry | None = None
    for _ in range(8):
        # --- step 1: draw low-coherence seed directions -----------------------
        ut = torch.randn(F, D, generator=gen, dtype=DT)
        ut = _repel(ut, cfg.max_unrelated_cos)

        u = torch.zeros(F, D, dtype=DT)
        g = torch.zeros(F, D, dtype=DT)
        Lam = torch.zeros(F, F, dtype=DT)

        # --- step 2: build u and g together, in topological order -------------
        # For each feature (parents before children): orthogonalise its drawn
        # direction against its ancestors' concept directions to get u_k, then
        # form g_k from u_k plus its parent's direction.
        for k in tree.topology_ordering:
            anc = sorted(tree.ancestors[k])
            if anc:
                basis = torch.linalg.qr(torch.stack([g[a] for a in anc], dim=1))[0]
                r = ut[k] - basis @ (basis.T @ ut[k])
            else:
                r = ut[k].clone()
            if float(r.norm()) < 1e-8:
                raise ValueError(f"u_{k} undefined: the drawn direction lies in its ancestor span")
            u[k] = r / r.norm()

            # w_k loads on the immediate parent's residual direction, weight 1.
            par = tree.parent_of(k)
            a_k = float(tree.alpha_of(k))
            if par is None or a_k == 0.0:
                g[k] = u[k]                                   # the w_k = 0 branch
                Lam[k, k] = 1.0
            else:
                w = u[par]                                    # ||w|| = 1, single unit loading
                g[k] = a_k * w + math.sqrt(1.0 - a_k ** 2) * u[k]
                Lam[par, k] = a_k
                Lam[k, k] = math.sqrt(1.0 - a_k ** 2)

        gram = u @ u.T
        gram.fill_diagonal_(0.0)
        coherence = float(gram.abs().max())
        cand = Geometry(u=u, g=g, Lam=Lam, coherence=coherence, coherence_ok=coherence <= cfg.max_unrelated_cos)
        if coherence <= cfg.max_unrelated_cos:
            return cand
        if best is None or coherence < best.coherence:
            best = cand

    assert best is not None
    return best
