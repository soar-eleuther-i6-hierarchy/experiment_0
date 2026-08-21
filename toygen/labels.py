"""The ground-truth pair-label table.

Annotations are per ORDERED pair `(a, b)`, read as "a is the candidate parent, b the
candidate child". Direction matters: `is_a`, `firing_only`, `transitive` and `reversed`
are antisymmetric; `sibling`, `superparent`, `frequency` and `topical` are symmetric.

Each pair carries EXACTLY ONE class. The relationship classes are non-intersecting by
construction: every confound feature is created for a single archetype and sits as a
parentless, childless root, so no ordered pair can satisfy two class predicates.
`pair_label` enforces this with a disjointness check — if a pair ever matched two
classes it raises rather than silently picking one.

Only DATA-SIDE properties appear here. Absorption, splitting and merging are
dictionary-side — induced by training and measured against the learned latents, never
injected. Every annotation is read off the generator's declared structure (`CONT`, the
topic and token-binding tags), never inferred from co-firing statistics, so the answer
key is independent of the metrics under test.
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

    Class `LABELS[i]` -> the cell holds index `i`. The diagonal is the `_UNSET` sentinel `-1` (a
    self-pair is not a relation of any class: NOT index-0 is_a — so a consumer that forgets to mask
    self-pairs can't read a self-pair as an is_a positive — and NOT `unrelated` either, which is
    reserved for the genuine off-diagonal nulls). `unrelated` is assigned off-diagonal exactly where
    no other class applies. Each predicate is checked independently and
    written through `assign`, which RAISES if a cell is claimed by two different classes
    — the non-intersecting invariant the single-label key depends on.
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
                # all token-bound features share one high-frequency id set, so any such
                # ordered pair is a frequency confound
                assign(a, b, "frequency")

    # siblings: any two features sharing a direct parent (exclusive or not)
    for kids in tree.children.values():
        for i, x in enumerate(kids):
            for z in kids[i + 1:]:
                assign(x, z, "sibling")
                assign(z, x, "sibling")

    # superparent: both orderings of every pair touching a declared superparent (it fires
    # on most tokens, so both directions are contaminated by its base rate)
    for a in range(F):
        if "superparent" not in tree.tags[a]:
            continue
        for b in range(F):
            if a != b:
                assign(a, b, "superparent")
                assign(b, a, "superparent")

    # declared ancestry (forward direction) and its reverse. `is_a` vs `firing_only` is
    # decided by the PER-EDGE overlap of the specific a->b containment edge (alpha > 0),
    # not a feature-level alpha — a feature may sit on both kinds of edge.
    for b in range(F):
        parent_alpha = {p: alpha for p, _, alpha in tree.parents.get(b, [])}
        for a in tree.ancestors[b]:
            if a in parent_alpha:
                name = "is_a" if float(parent_alpha[a]) > 0.0 else "firing_only"
            else:
                name = "transitive"
            assign(a, b, name)
            assign(b, a, "reversed")

    # the null: every off-diagonal pair with no declared property becomes `unrelated`. The diagonal
    # is left at the `_UNSET` sentinel (-1): a self-pair has no relation class — keeping it -1 (never
    # index-0 is_a, and distinct from the off-diagonal `unrelated` nulls) means a consumer that
    # forgets to mask self-pairs cannot mistake one for an is_a positive.
    off_diag = ~torch.eye(F, dtype=torch.bool)
    y[(y == _UNSET) & off_diag] = _index("unrelated")
    y.fill_diagonal_(_UNSET)
    return y
