"""The ground-truth pair-label table.

Labels are per ordered pair `(a, b)`: "a is the candidate parent, b the candidate child".
`is_a`, `firing_only`, `transitive`, `reversed` are directional; `sibling`, `superparent`,
`frequency`, `topical` are symmetric. Each pair carries exactly one class -- classes can't
overlap by construction, and `pair_label` enforces this with a disjointness check.

Only data-side properties appear here (never absorption/splitting/merging, which are
dictionary-side). Every label is read off the generator's declared structure, never
inferred from co-firing statistics, so the answer key is independent of the metrics under test.
"""

from __future__ import annotations

import torch

from .tree import Tree

LABELS: tuple[str, ...] = (
    "is_a",         # (a, b) is a direct containment edge with parent->child overlap alpha > 0
    "firing_only",  # a direct containment edge with alpha == 0: nested firing, orthogonal direction
    "transitive",   # b is a strict descendant of a but not its direct child
    "sibling",      # a and b share a direct parent
    "superparent",  # a or b is a declared superparent (high firing rate, no children)
    "frequency",    # a and b are both token-bound on a shared high-frequency id set
    "topical",      # a and b are both topic-modulated on the same topic
    "unrelated",    # the null: no declared property holds
    "reversed",     # the flip of a directed ancestry pair — b is related to a, wrong way round
)


def label_name(i: int) -> str:
    return LABELS[i]


def _index(name: str) -> int:
    return LABELS.index(name)


def pair_label(tree: Tree) -> torch.Tensor:
    """`[F, F]` int8 answer key: the single class index of each ordered pair.

    Cell (a, b) holds `i` where the pair's class is `LABELS[i]`. The diagonal holds the
    `_UNSET` sentinel `-1`, deliberately neither index-0 `is_a` nor `unrelated` -- so code
    that forgets to mask self-pairs can't misread one. `unrelated` is assigned off-diagonal
    wherever nothing else applies; `assign` raises if a cell is claimed by two classes.
    """
    F = tree.F
    _UNSET = -1
    y = torch.full((F, F), _UNSET, dtype=torch.int8)

    def assign(a: int, b: int, name: str) -> None:
        idx = _index(name)
        cur = int(y[a, b])
        if cur != _UNSET and cur != idx:
            raise ValueError(
                f"pair ({a}, {b}) matches two classes: {LABELS[cur]} and {name} — the "
                f"relationship classes must be non-intersecting"
            )
        y[a, b] = idx

    # symmetric coincidence confounds
    for a in range(F):
        for b in range(F):
            if a == b:
                continue
            if ("topical" in tree.tags[a] and "topical" in tree.tags[b]
                    and tree.topic[a] is not None and tree.topic[a] == tree.topic[b]):
                assign(a, b, "topical")
            if tree.token_bound[a] and tree.token_bound[b]:
                # all token-bound features share one high-frequency id set, so any such pair is a frequency confound
                assign(a, b, "frequency")

    # siblings: any two features sharing a direct parent (exclusive or not)
    for kids in tree.children.values():
        for i, x in enumerate(kids):
            for z in kids[i + 1:]:
                assign(x, z, "sibling")
                assign(z, x, "sibling")

    # superparent: both orderings of every pair touching a declared superparent, since its high base rate contaminates the pair either way
    for a in range(F):
        if "superparent" not in tree.tags[a]:
            continue
        for b in range(F):
            if a != b:
                assign(a, b, "superparent")
                assign(b, a, "superparent")

    # declared ancestry (forward) and its reverse. is_a vs firing_only is decided per-edge (alpha > 0), not per-feature -- one feature can sit on both kinds of edge.
    for b in range(F):
        parent_alpha = {p: alpha for p, _, alpha in tree.parents.get(b, [])}
        for a in tree.ancestors[b]:
            if a in parent_alpha:
                name = "is_a" if float(parent_alpha[a]) > 0.0 else "firing_only"
            else:
                name = "transitive"
            assign(a, b, name)
            assign(b, a, "reversed")

    # the null: every off-diagonal pair with no declared property becomes `unrelated`. The diagonal stays at `_UNSET` (-1), so code that forgets to mask self-pairs can't mistake one for an is_a positive.
    off_diag = ~torch.eye(F, dtype=torch.bool)
    y[(y == _UNSET) & off_diag] = _index("unrelated")
    y.fill_diagonal_(_UNSET)
    return y
