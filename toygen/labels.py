"""
The ground-truth pair label table.

One label per ORDERED pair `(a, b)`, read as "a is the candidate parent, b the
candidate child". Direction matters: `is_a`, `firing_only` and `transitive` are
antisymmetric, the rest are symmetric.

Only DATA-SIDE properties appear here. Absorption, splitting and merging are
dictionary-side — they are induced by training and labelled by a reference standard
against the learned latents, never injected (spec section 1).

Every label is read off the generator's declared structure (`CONT`, `ISA`, the topic
and token-binding tags), never inferred from statistics. Inferring them from co-firing
would make the answer key a restatement of the metrics under test.
"""

from __future__ import annotations

import torch

from .tree import Tree

LABELS: tuple[str, ...] = (
    "is_a",              # (p,c) in ISA: a direct CONT edge with alpha_{p->c} > 0
    "firing_only",       # a direct CONT edge with alpha == 0: nested firing, orthogonal direction
    "transitive",        # b is a strict descendant of a but not its direct child (H2)
    "sibling",           # share a direct parent, mutually exclusive
    "superparent",       # a is a declared superparent (high rate, no children)
    "frequency",         # G: both token-bound on a shared high-frequency id set
    "topical",           # H: both topic-modulated on the same topic
    "unrelated",         # C: the null
    # The flip side of a directed ancestry pair: b really is related to a, just the
    # wrong way round. Appended (so earlier indices are stable) after the blind test
    # suite proved the label set incomplete: with only the nine above, the reverse of
    # an is_a edge had to fall into `unrelated`, which both breaks the null's symmetry
    # and — worse — makes an orientation error indistinguishable from an ordinary
    # false positive. 
    "reversed",
)

_UNLABELLED = -1


def label_name(i: int) -> str:
    return LABELS[i]


def _index(name: str) -> int:
    return LABELS.index(name)


def pair_label_matrix(tree: Tree) -> torch.Tensor:
    """[F, F] int8 of indices into LABELS; the diagonal is -1.

    Precedence is fixed and total: the first matching rule wins, and every
    off-diagonal pair reaches a rule. Confound pairs are resolved BEFORE the
    `unrelated` fallback — otherwise the hardest negatives would be scored as easy
    ones, and the false-positive rate would be measured against the wrong population.
    """
    F = tree.F
    M = torch.full((F, F), _index("unrelated"), dtype=torch.int8)

    is_a = _index("is_a")
    firing_only = _index("firing_only")
    trans = _index("transitive")
    sib = _index("sibling")
    sup = _index("superparent")
    freq = _index("frequency")
    top = _index("topical")

    # --- symmetric confounds (lowest precedence of the declared ones) ------
    for a in range(F):
        for b in range(F):
            if a == b:
                continue
            ta, tb = tree.tags[a], tree.tags[b]
            if "topical" in ta and "topical" in tb and tree.topic[a] is not None \
                    and tree.topic[a] == tree.topic[b]:
                M[a, b] = top
            elif tree.token_bound[a] and tree.token_bound[b]:
                # all token-bound features share one high-frequency id set, so any
                # such ordered pair is a frequency confound
                M[a, b] = freq

    # --- siblings ----------------------------------------------------------
    # Any two features sharing a direct parent, exclusive or not. Exclusivity is a
    # swept design choice (spec 2 B's DISAGREE row, where non-exclusive siblings start
    # to read as is-a to coverage), not a precondition for being a sibling: gating on
    # it would drop the broad parent's non-exclusive children into `unrelated`, putting
    # genuine co-hyponyms in the null class and biasing the negative population the
    # false-positive rate is measured against.
    for par, kids in tree.children.items():
        for i, x in enumerate(kids):
            for y in kids[i + 1:]:
                M[x, y] = sib
                M[y, x] = sib

    # --- superparent: every pair TOUCHING a declared superparent ------------
    # Symmetric, deliberately. The superparent fires on most tokens, so BOTH
    # orderings are contaminated by its base rate: with it as the candidate child,
    # P(superparent | other) is just as spuriously high as the forward direction.
    # Labelling only one direction would put the reverse in `unrelated` and score a
    # base-rate false positive as an easy true negative.
    for a in range(F):
        if "superparent" not in tree.tags[a]:
            continue
        for b in range(F):
            if a != b:
                M[a, b] = sup
                M[b, a] = sup

    # --- ancestry (highest precedence: declared structure wins) ------------
    # The tiers above only avoid colliding with ancestry because the generator builds
    # every confound archetype as a parentless, childless root. That is an invariant
    # of tree.build_tree, not of this function, so assert it: a future confound family
    # that also participates in containment would otherwise have its cell silently
    # overwritten here, with no test-visible signal.
    declared = {a for a in range(F)
                if "superparent" in tree.tags[a]
                or tree.token_bound[a] or tree.kappa[a] > 0}
    for a in declared:
        assert not tree.ancestors[a] and not tree.descendents[a], (
            f"confound feature {a} participates in CONT; the label precedence in this "
            f"function assumes confound archetypes are isolated roots"
        )

    rev = _index("reversed")
    for b in range(F):
        direct = {p for p, _, _ in tree.parents.get(b, [])}
        for a in tree.ancestors[b]:
            M[a, b] = is_a if (a in direct and tree.alpha_of(b) > 0.0) else (
                firing_only if a in direct else trans)
            M[b, a] = rev

    M.fill_diagonal_(_UNLABELLED)
    return M
