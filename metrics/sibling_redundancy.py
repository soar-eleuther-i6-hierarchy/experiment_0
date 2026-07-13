"""
Metric 3 - Child diversity / sibling redundancy, Tree-SAE style
(project plan Exp 0, bullet 3).

If a parent's children are a REAL refinement, they should partition the
parent's firing set: each child claims a different slice, so siblings rarely
co-fire. If instead the children are the same feature split into near-copies
("feature splitting in disguise"), they co-fire massively.

For each parent p with kept children S = {c1..ck}, we report the mean pairwise
sibling Jaccard:

    J(ci, cj) = cofire(ci, cj) / (fire(ci) + fire(cj) - cofire(ci, cj))

    redundancy(p) = mean over pairs (ci, cj) in S

~0   -> children are diverse (healthy refinement)
~1   -> children are duplicates (splitting)
"""

from __future__ import annotations

import torch


def sibling_redundancy(
    edge_mask: torch.Tensor,     # [P, C] kept edges for one block pair
    child_cofire: torch.Tensor,  # [C, C] within-block co-firing counts of the child block
    fire_c: torch.Tensor,        # [C]    child firing counts
    max_children: int = 512,     # guard: superparents have thousands of children;
                                 # we subsample pairs via the top-firing children
) -> dict[int, dict]:
    """Per-parent redundancy. Returns {parent_local: {redundancy, n_children}}."""
    fire_c = fire_c.double()
    out: dict[int, dict] = {}
    for p in torch.nonzero(edge_mask.sum(dim=1) >= 2).flatten().tolist():
        kids = torch.nonzero(edge_mask[p]).flatten()
        if kids.numel() > max_children:
            top = torch.argsort(fire_c[kids], descending=True)[:max_children]
            kids = kids[top]
        cf = child_cofire[kids][:, kids].double()          # [k, k]
        fi = fire_c[kids]                                   # [k]
        union = fi.unsqueeze(0) + fi.unsqueeze(1) - cf
        jac = cf / union.clamp(min=1.0)
        k = kids.numel()
        offdiag = ~torch.eye(k, dtype=torch.bool)
        out[p] = {
            "redundancy": float(jac[offdiag].mean()),
            "n_children": int(edge_mask[p].sum()),
        }
    return out
