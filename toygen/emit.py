"""
Write a sampled World and its ground truth to `truth.pt`.

Only ground truth is written — directions, coefficients, the corpus draw, the
containment / is-a edges, the pair-label matrix, firing rates and the designed alpha
loadings — everything a later SAE-training and analysis pass needs, and nothing a
metric could read as a shortcut.
"""

from __future__ import annotations

from pathlib import Path

import torch

from .geometry import Geometry
from .labels import pair_label_matrix
from .sample import World
from .spec import ToyConfig
from .strengths import StrengthSpec
from .tree import Tree

DT = torch.float64


def write_toy(out_dir: str | Path, cfg: ToyConfig, tree: Tree,
              strengths: StrengthSpec, geo: Geometry, world: World) -> Path:
    """Write the ground truth for one sampled world to `truth.pt` in `out_dir`.

    `CONT` is every direct parent->child edge; `ISA` is that subset with alpha > 0
    (the firing_only edges have alpha == 0 and stay out). Edges are stored (parent,
    child), never child->parent.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    cont = [(p, c) for c in range(tree.F) for p, _, _ in tree.parents.get(c, [])]
    isa = [(p, c) for (p, c) in cont if tree.alpha_of(c) > 0.0]
    alpha_span = torch.tensor([tree.alpha_of(k) for k in range(tree.F)], dtype=DT)

    torch.save({"config_name": cfg.name, "g": geo.g, "u": geo.u, "Lam": geo.Lam, "coherence": geo.coherence, "A": world.A, "Atilde": world.Atilde, "tokens": world.tokens, "docs": world.docs, "topics": world.topics, "CONT": cont, "ISA": isa, "pair_labels": pair_label_matrix(tree), "p": strengths.p, "alpha_span": alpha_span,}, out / "truth.pt")
    return out
