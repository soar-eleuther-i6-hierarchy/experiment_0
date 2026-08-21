"""
Write a sampled World and its ground truth to `truth.pt`.

Only ground truth is written — directions, coefficients, the corpus draw, the
containment / is-a edges, the single-label answer key, firing rates and the designed alpha
loadings — and nothing a metric could read as a shortcut.

The activations `h` and `noise` are not stored (they are `[n, D]` and dominate the
file). Instead the full `config` and the `sample_seed` are saved, and the noise draw
is seeded, so `sample_world` regenerates the exact same world — including noise — from
`truth.pt` alone. `coherence_ok` records whether the geometry hit its coherence TARGET — a soft flag,
expected False at F≫D (see `geometry.Geometry`), not a health check.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import torch

from .geometry import Geometry
from .labels import pair_label
from .sample import World
from .spec import ToyConfig
from .strengths import StrengthSpec
from .tree import Tree

DT = torch.float64


def write_toy(out_dir: str | Path, cfg: ToyConfig, tree: Tree,
              strengths: StrengthSpec, geo: Geometry, world: World,
              seed: int | None = None) -> Path:
    """Write the ground truth for one sampled world to `truth.pt` in `out_dir`.

    `CONT` is every direct parent->child edge; `ISA` is that subset with alpha > 0
    (the firing_only edges have alpha == 0 and stay out). Edges are stored (parent, child), never child->parent. `seed` is the sampling seed passed to `sample_world`; it is stored (with the full `config`) so the exact world can be regenerated. Pass the same value used to sample `world`, or leave it None if the caller relied on `cfg.seed`.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    cont = [(p, c) for c in range(tree.F) for p, _, _ in tree.parents.get(c, [])]
    isa = [(p, c) for (p, c) in cont if tree.alpha_of(c) > 0.0]
    alpha_span = torch.tensor([tree.alpha_of(k) for k in range(tree.F)], dtype=DT)

    torch.save({
        "config_name": cfg.name,
        "config": dataclasses.asdict(cfg),
        "sample_seed": seed,
        "g": geo.g, "u": geo.u, "Lam": geo.Lam,
        "coherence": geo.coherence, "coherence_ok": geo.coherence_ok,
        "A": world.A, "Atilde": world.Atilde,
        "tokens": world.tokens, "docs": world.docs, "topics": world.topics,
        "CONT": cont, "ISA": isa,
        "pair_label": pair_label(tree),   # the answer key: one class index per ordered pair
        "p": strengths.p, "alpha_span": alpha_span,
    }, out / "truth.pt")
    return out
