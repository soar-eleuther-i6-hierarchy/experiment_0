"""
In-block (same-level) directed edges (in_block_edges.py; config.IN_BLOCK_BLOCKS).

Hierarchy need not respect the Matryoshka block boundaries: two features in the
SAME block can stand in a parent/child (refinement) or duplicate relation. Unlike
the cross-block graph, a block gives no ordering to fix edge direction or forbid
cycles, so we derive both from coverage asymmetry.

For a within-block co-firing matrix `cofire[C, C]` (symmetric) and firing counts
`fire[C]`, define reverse coverage

    R[i, j] = P(i fires | j fires) = cofire[i, j] / fire[j].

If child j is contained in parent i then R[i, j] ≈ 1 (j almost always co-fires
with i) while R[j, i] ≪ 1. So:

    parent_of[i, j]  (i is parent of j)  iff  R[i, j] ≥ τ  AND  R[j, i] < τ
    duplicate[i, j]  (co-extensive)      iff  R[i, j] ≥ τ  AND  R[j, i] ≥ τ

both restricted to i≠j and to pairs with enough support. `parent_of` is
antisymmetric by construction (if R[j,i] < τ then j→i cannot also hold), so the
in-block graph is acyclic; co-extensive pairs are renames/splits, reported
separately and NEVER drawn as an edge (that is what would create 2-cycles).
"""

from __future__ import annotations

import torch


def directed_coverage(
    cofire: torch.Tensor,      # [C, C] within-block co-firing counts (symmetric)
    fire: torch.Tensor,        # [C]    per-feature firing counts
    tau: float,
    min_fire: int,
    min_joint: int,
) -> dict[str, torch.Tensor]:
    """Directed within-block edges + duplicate flags.

    Returns {"R", "parent_of", "duplicate"}:
      R[i, j]          = P(i | j)                                  [C, C] float
      parent_of[i, j]  = i is parent of j (asymmetric containment) [C, C] bool
      duplicate[i, j]  = i, j co-extensive (rename/split, no edge) [C, C] bool
    """
    cofire = cofire.double()
    fire = fire.double()
    C = fire.shape[0]
    R = cofire / fire.clamp(min=1.0).unsqueeze(0)      # divide column j by fire[j]

    eye = torch.eye(C, dtype=torch.bool, device=R.device)
    support = (
        (cofire >= min_joint)
        & (fire.unsqueeze(0) >= min_fire)              # child j fires enough
        & (fire.unsqueeze(1) >= min_fire)              # parent i fires enough
        & ~eye
    )
    ge = R >= tau
    parent_of = ge & ~ge.T & support                  # j⊆i but not i⊆j
    duplicate = ge & ge.T & support                   # co-extensive both ways
    return {"R": R, "parent_of": parent_of, "duplicate": duplicate}


def duplicate_pairs(duplicate: torch.Tensor) -> list[tuple[int, int]]:
    """Unordered co-extensive pairs (upper triangle of the symmetric flag)."""
    ij = torch.nonzero(torch.triu(duplicate, diagonal=1), as_tuple=False)
    return [(int(i), int(j)) for i, j in ij.tolist()]
