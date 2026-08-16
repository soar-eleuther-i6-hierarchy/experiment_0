"""
Builds the feature forest (the `Tree`) from a ToyConfig: the parent/child edges,
their transitive closure, and the per-feature tags.

Only DIRECT parent-child edges are stored in `parents`; `ancestors` / `descendents`
are the transitive closure of those edges. The distinction matters: a
grandparent-grandchild pair is contained (so it passes coverage) but is NOT a direct
edge — it is labelled `transitive` and scored on its own, apart from real edges.

Every feature has a single parent (the forest is a set of independent trees, no
multi-parenting), which keeps each child's firing rate a clean product down its one
chain: p_child = p_parent * p_edge.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .spec import ToyConfig


@dataclass
class Tree:
    F: int
    parents: dict[int, list[tuple[int, float, float]]]   # child -> [(parent, p_edge, alpha_span)]
    children: dict[int, list[int]]
    exclusive: dict[int, bool]
    ancestors: dict[int, set[int]]
    descendents: dict[int, set[int]]
    topology_ordering: list[int]
    root_p: dict[int, float]                             # roots only
    # --- tags -------------------------------------------------------------
    kind: dict[int, str] = field(default_factory=dict)          # "discrete" | "manifold"
    kappa: dict[int, float] = field(default_factory=dict)       # topic modulation, 0 = off
    topic: dict[int, int | None] = field(default_factory=dict)
    token_bound: dict[int, bool] = field(default_factory=dict)
    tags: dict[int, set[str]] = field(default_factory=dict)

    def parent_of(self, k: int) -> int | None:
        ps = self.parents.get(k)
        return ps[0][0] if ps else None

    def p_edge_of(self, k: int) -> float:
        ps = self.parents.get(k)
        return ps[0][1] if ps else 1.0
    
    def alpha_of(self, k: int) -> float:
        ps = self.parents.get(k)
        return ps[0][2] if ps else 0.0


def _close(parents: dict[int, list[tuple[int, float, float]]], F: int
) -> tuple[dict[int, set[int]], dict[int, set[int]], list[int]]:
    """Transitive closure of the direct-edge relation, plus a topological order."""
    anc = {k: set() for k in range(F)}
    for k in range(F):
        stack = [p for p, _, _ in parents.get(k, [])]
        while stack:
            a = stack.pop()
            if a not in anc[k]:
                anc[k].add(a)
                stack += [p for p, _, _ in parents.get(a, [])]
    desc = {k: set() for k in range(F)}
    for k in range(F):
        for a in anc[k]:
            desc[a].add(k)
    depth = {k: 0 for k in range(F)}
    topo = sorted(range(F), key=lambda k: (len(anc[k]), k))
    for k in topo:
        ps = parents.get(k, [])
        depth[k] = 1 + max((depth[p] for p, _, _ in ps), default=-1) if ps else 0
    topo = sorted(range(F), key=lambda k: (depth[k], k))
    return anc, desc, topo


def build_tree(cfg: ToyConfig) -> Tree:
    parents: dict[int, list[tuple[int, float, float]]] = {}
    children: dict[int, list[int]] = {}
    exclusive: dict[int, bool] = {}
    root_p: dict[int, float] = {}
    kind: dict[int, str] = {}
    kappa: dict[int, float] = {}
    topic: dict[int, int | None] = {}
    token_bound: dict[int, bool] = {}
    tags: dict[int, set[str]] = {}

    nxt = 0

    def new(tag: str = "backbone") -> int:
        nonlocal nxt
        k = nxt
        nxt += 1
        kind[k] = "discrete"
        kappa[k] = 0.0
        topic[k] = None
        token_bound[k] = False
        tags[k] = {tag}
        children[k] = []
        return k

    # --- backbone forest --------------------------------------------------
    edge_i = 0
    frontier = []
    for _ in range(cfg.n_roots):
        r = new()
        root_p[r] = cfg.root_p
        frontier.append(r)
    for _ in range(cfg.depth):
        nxt_frontier = []
        for p in frontier:
            exclusive[p] = cfg.exclusive_siblings
            for _ in range(cfg.branching):
                c = new()
                a = 0.0 if (edge_i % cfg.alpha_zero_every == 0) else cfg.alpha
                edge_i += 1
                parents[c] = [(p, cfg.child_p_edge, a)]
                children[p].append(c)
                nxt_frontier.append(c)
        frontier = nxt_frontier

    # --- confounds --------------------------------------------------------
    # The firing rates below are deliberately literal: they define the archetypes
    # rather than parameterising a sweep. Counts and composition are config
    # fields while the defining rates are not, and should not be swept without changing
    # what the archetype means.
    if cfg.confounds:
        # superparent: an always-on childless feature, plus a legitimately broad parent
        # as its foil (without the foil, out-degree separates nothing and scores 100%
        # trivially).
        for _ in range(cfg.n_superparent):
            s = new("superparent")
            root_p[s] = 0.85
        for _ in range(cfg.n_broad_parent):
            b = new("broad_parent")
            root_p[b] = 0.60
            exclusive[b] = False
            for _ in range(cfg.broad_children):
                c = new("broad_child")
                parents[c] = [(b, cfg.child_p_edge, cfg.broad_alpha)]
                children[b].append(c)

        # frequency-coincidence pairs: token-bound features that co-fire only because
        # they share high-frequency token ids, with no declared edge between them.
        for i in range(cfg.n_token_bound_pairs):
            for j in (0, 1):
                k = new("token_bound")
                root_p[k] = 0.05
                token_bound[k] = True
                tags[k].add(f"tokpair{i}")

        # topical pairs: co-occur because they share a document topic z — marginally
        # dependent, but conditionally independent once z is known.
        for i in range(cfg.n_topical_pairs):
            z = i % cfg.Z
            for j in (0, 1):
                k = new("topical")
                root_p[k] = 0.04
                kappa[k] = cfg.kappa
                topic[k] = z
                tags[k].add(f"toppair{i}")

        # manifold features: continuous-strength (graded, not on/off) features.
        for _ in range(cfg.n_manifold):
            k = new("manifold")
            kind[k] = "manifold"
            root_p[k] = 0.06

    F = nxt
    ancestors, descendents, topology_ordering = _close(parents, F)

    return Tree(
        F=F, parents=parents, children=children, exclusive=exclusive,
        ancestors=ancestors, descendents=descendents, topology_ordering=topology_ordering,
        root_p=root_p, kind=kind, kappa=kappa, topic=topic, token_bound=token_bound, tags=tags,
    )
