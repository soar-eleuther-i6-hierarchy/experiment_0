"""
Synthetic ground-truth "toy" for calibrating the five hierarchy metrics.

A metric earns its place only if it BOTH recovers a known parent-child tree AND
rejects pathological edges we inject on purpose. Real gemma-2-2b caches have no ground truth, so we build a
tiny world where we know the answer, and emit the *same cached statistics*
`run_metrics.analyse_pair` reads from `exp0_stats.pt`. The five metrics then run
unchanged (same functions, same config thresholds) and we score them.

World layout (one block pair: parent block of P features -> child block of C):

  Genuine tree (healthy hierarchy):
    parents 0..4  each own a DISJOINT set of 4 children (locals 0..19).
    A child fires on a subset of tokens; its parent fires whenever it does
    (reverse coverage 1); siblings fire on different tokens (low redundancy);
    both contribute real decoder mass (reconstruction gain high); children fire
    across all frequency buckets (frequency survival ~1).
    -> every metric should PASS these.

  Injected pathologies:
    (A) SUPERPARENT  (parent 7): fires on ~90% of ALL tokens with a TINY
        activation, so it co-fires with every child (reverse coverage ~0.9 ->
        huge out-degree) but contributes ~nothing to reconstruction.
        -> out-degree flags it; reconstruction rejects its edges.
    (B) FREQUENCY-COINCIDENCE edge (parent 6 -> child 20): parent 6 only
        co-fires with child 20 on a couple of HIGH-FREQUENCY token ids; on rare
        tokens child 20 fires alone. Reverse coverage passes and the parent even
        contributes real decoder mass on those tokens (reconstruction passes),
        but the edge lives on frequent tokens.
        -> only token-frequency control rejects it.
    (C) FEATURE SPLITTING (parent 5 -> children 24,25,26): three near-duplicate
        children firing on the SAME tokens with the SAME direction. Coverage and
        reconstruction happily accept each edge.
        -> only sibling redundancy flags the split.

Modeling choice: we set the SAE residual error directly to isotropic gaussian
noise (independent of the feature directions). Then the per-token ablation gain
  g_f = 2 a_f <d_f, err> + a_f^2 ||d_f||^2
reduces in expectation to a_f^2 (unit-norm directions), i.e. gains are driven by
activation magnitude. That is exactly the knob that separates a real refinement
(strong activation) from a frequency-riding superparent (tiny activation), so the
reconstruction metric is exercised on the property it is supposed to test.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

# ---- world dimensions ------------------------------------------------------
# D_MODEL is set by the rank rule, not by the statistics-only metrics -- those
# read counts and are indifferent to it. `sres_rank_check` asks whether the
# parent decoder is among the top-k probe correlations over ALL D features, so it
# is only measuring parenthood if unrelated directions are roughly orthogonal.
# Random unit vectors in d dims correlate by about sqrt(2/(pi*d)): at d=16 that
# is 0.20 with a worst pair at 0.73, and unrelated features then outrank a true
# parent on crowding alone. Measured at d=16, three genuine edges failed with the
# CHILD at rank 0 and the parent pushed to rank 6-8 by features it has nothing to
# do with -- a fact about 42 directions in 16 dimensions, not about the metric.
# gemma sits at d=2304 (about 0.02), which no toy can reach; d=64 (about 0.10)
# is enough for the true parent to clear the noise floor.
D_MODEL = 64

# Residual-error energy per token, held FIXED as D_MODEL changes. Scaling the
# noise per-dimension instead would tie the reconstruction metric's denominator
# (err_sum_c) to a constant chosen for the probe geometry, so raising D_MODEL
# would quietly shrink every relative gain. The two knobs are independent
# properties of the world and are kept independent here.
RESID_ERR_ENERGY = 0.64

P = 15          # parent-block features
C = 33          # child-block features
N_FREQ_BUCKETS = 3

# ---- ground-truth structure (parent-local -> list of child-local) ----------
GENUINE_TREE: dict[int, list[int]] = {
    0: [0, 1, 2, 3],
    1: [4, 5, 6, 7],
    2: [8, 9, 10, 11],
    3: [12, 13, 14, 15],
    4: [16, 17, 18, 19],
}
SPLIT_PARENT = 5
SPLIT_CHILDREN = [24, 25, 26]      # near-duplicates -> pathology (C)
FREQ_PARENT = 6
FREQ_CHILD = 20                    # frequency-coincidence -> pathology (B)
SUPERPARENT = 7                    # fires ~everywhere -> pathology (A)

# --- pathologies (D)-(F): added to calibrate the metric rows that had no toy --
# (D) ABSORPTION. The child took over the parent's direction, so on the child's
#     own tokens the parent is SILENT -- the operational signature of an absorbed
#     feature. Truth: (8, 21) is a real refinement. Coverage cannot see it, since
#     R = cofire / fire_c = 0. This is a NEGATIVE control: it exists to
#     demonstrate the blind spot the properties matrix already claims, rather
#     than to be caught.
ABSORB_PARENT = 8
ABSORB_CHILD = 21
# (E) TOPICAL CO-OCCURRENCE. Both features are driven by a shared latent topic
#     and are conditionally INDEPENDENT given it: neither refines the other.
#     Coverage, reconstruction, frequency control and PMI all pass, because each
#     tests a different confound and none of them tests this one. The second
#     negative control -- the open column in the matrix.
TOPIC_PARENT = 9
TOPIC_CHILD = 22
# (G) COMPOSITION. One child latent encodes the conjunction of two component
#     concepts ("red triangle"): it fires exactly where both components fire, so
#     EACH component contains it (R = 1) and both edges clear every gate, yet
#     neither is a taxonomic refinement. The third negative control -- like
#     topic, nothing in the battery tests atomicity; out-degree sees only the
#     symptom (a multi-parented child).
COMP_PARENTS = (10, 11)
COMP_CHILD = 23
# (H) MULTI-PARENTING. Child 31 has ONE true parent (12) and one intruding
#     broad parent (13) that co-fires on all of the child's tokens and on many
#     others. Both edges clear every gate; the child's in-degree of 2 is the
#     symptom, and only the out-degree metric reads in-degree at all.
MULTI_TRUE_PARENT = 12
MULTI_INTRUDER_PARENT = 13
MULTI_CHILD = 31
# (I) CROSS-BLOCK SIBLINGS. Parent-block latent 14 and child-block latent 32
#     fire on exactly the same tokens: two names for one concept straddling the
#     block boundary. Coverage reads the pair as an edge (R = 1 both ways);
#     the joint-child energy share ~ 1 is the rename/duplicate signature.
COEXT_PARENT = 14
COEXT_CHILD = 32
# (F) WITHIN-BLOCK structure, for metric 7 (in_block_edges). Same block, so no
#     block ordering fixes the direction: it has to come out of the coverage
#     asymmetry. 28 fires only inside 27 (directed edge); 29 and 30 fire on
#     exactly the same tokens (co-extensive -> a duplicate, never an edge).
IN_BLOCK_PARENT = 27
IN_BLOCK_CHILD = 28
IN_BLOCK_DUP = (29, 30)

# ---- token-id frequency design (drives the frequency buckets) --------------
FREQ_IDS = [0, 1]                  # high-frequency ids (bucket 0)
MID_IDS = list(range(2, 12))       # mid ids (bucket 1)
RARE_BASE = 100                    # rare ids start here, one per use (bucket 2)


@dataclass
class ToyLabels:
    """Ground-truth edge labels, in parent-local / child-local space."""
    genuine: set[tuple[int, int]] = field(default_factory=set)
    superparent_edges: set[tuple[int, int]] = field(default_factory=set)
    freq_edges: set[tuple[int, int]] = field(default_factory=set)
    split_children: set[int] = field(default_factory=set)
    superparent_parents: set[int] = field(default_factory=set)
    split_parents: set[int] = field(default_factory=set)
    # A real edge that coverage cannot propose, because the absorbed child fires
    # where its parent does not. Kept apart from `genuine` on purpose: scoring it
    # as a miss would penalise metrics 2-9 for a candidate they never receive.
    absorbed_edges: set[tuple[int, int]] = field(default_factory=set)
    # Not an edge at all -- two features under a shared topic. Kept apart from
    # the pathologies for the opposite reason: no metric here is expected to
    # reject it, so counting it as a false positive would misattribute a known
    # gap in the battery to the metric that happens to pass it.
    topical_edges: set[tuple[int, int]] = field(default_factory=set)
    # Same status as topical_edges: component -> composed-child pairs that pass
    # every gate because nothing tests atomicity. Not scored as false positives.
    composition_edges: set[tuple[int, int]] = field(default_factory=set)
    # The intruding parent of the multi-parented child: passes every gate; the
    # detection is the child's in-degree, read only by the out-degree metric.
    multiparent_edges: set[tuple[int, int]] = field(default_factory=set)
    # A cross-block co-extensive pair (two names for one concept): coverage
    # proposes it as an edge; the energy share ~ 1 is the duplicate signature.
    coextensive_edges: set[tuple[int, int]] = field(default_factory=set)
    # Within-block ground truth, in child-local space.
    in_block_edges: set[tuple[int, int]] = field(default_factory=set)
    in_block_duplicates: set[tuple[int, int]] = field(default_factory=set)


class _Gen:
    """Deterministic token accumulator. Every append is one corpus token."""

    def __init__(self, seed: int) -> None:
        self.g = torch.Generator().manual_seed(seed)
        self.rows: list[dict[int, float]] = []   # feature-local -> activation
        self.tok_ids: list[int] = []
        self._rare = RARE_BASE

    def _u(self, lo: float, hi: float) -> float:
        return float(torch.rand(1, generator=self.g) * (hi - lo) + lo)

    def rare_id(self) -> int:
        self._rare += 1
        return self._rare

    def add(self, feats: dict[int, float], tok_id: int) -> None:
        self.rows.append(feats)
        self.tok_ids.append(tok_id)


def _bucket_id_for_genuine(gen: _Gen, i: int) -> int:
    """Spread genuine firings across buckets so survival is testable (~1)."""
    r = i % 4
    if r == 0:
        return FREQ_IDS[i % len(FREQ_IDS)]
    if r == 1:
        return MID_IDS[i % len(MID_IDS)]
    return gen.rare_id()


def build_world(
    seed: int = 0,
    events_per_child: int = 40,
    n_background: int = 1200,
    superparent_rate: float = 0.90,
) -> tuple[dict, ToyLabels]:
    """Generate the toy corpus and reduce it to `analyse_pair`-shaped stats.

    Returns (stats, labels). `stats` mirrors the single-pair slice of
    `exp0_stats.pt`: fire_count, cofire, g_parent_sum, err_sum_c, g_child_sum,
    cofire_by_bucket, fire_c_by_bucket, within_cofire, buckets, total_tokens,
    and the schema-v2 energy/union accumulators energy_cofire, energy_total,
    union_count, union_energy.
    """
    gen = _Gen(seed)
    labels = ToyLabels(
        superparent_parents={SUPERPARENT},
        split_parents={SPLIT_PARENT},
        split_children=set(SPLIT_CHILDREN),
    )

    def parent_act() -> float:
        return gen._u(1.0, 2.0)

    def child_act() -> float:
        return gen._u(0.8, 1.6)

    # (1) genuine tree: each child fires on its own tokens; parent fires too.
    for parent, kids in GENUINE_TREE.items():
        for c in kids:
            labels.genuine.add((parent, c))
            for i in range(events_per_child):
                gen.add({parent: parent_act(), P + c: child_act()},
                        _bucket_id_for_genuine(gen, i))

    # (2C) feature splitting: the 3 split children fire TOGETHER on shared tokens.
    for i in range(events_per_child):
        feats = {SPLIT_PARENT: parent_act()}
        shared = child_act()                       # identical activation -> duplicate
        for c in SPLIT_CHILDREN:
            feats[P + c] = shared
        gen.add(feats, _bucket_id_for_genuine(gen, i))

    # (2B) frequency coincidence: child 20 co-fires with parent 6 on the single
    # most-frequent token id (solidly bucket 0), and fires ALONE on rare tokens.
    for _ in range(60):                            # frequent co-fires (bucket 0)
        gen.add({FREQ_PARENT: gen._u(1.0, 1.8), P + FREQ_CHILD: child_act()},
                FREQ_IDS[0])
    for _ in range(30):                            # rare solo firings (bucket 2)
        gen.add({P + FREQ_CHILD: child_act()}, gen.rare_id())
    labels.freq_edges.add((FREQ_PARENT, FREQ_CHILD))

    # (2D) absorption: the parent fires on its own tokens, the absorbed child on
    # tokens where the parent is SILENT. They never co-fire, so cofire = 0 and no
    # candidate edge is ever proposed -- which is the point being demonstrated.
    for i in range(60):
        gen.add({ABSORB_PARENT: parent_act()}, _bucket_id_for_genuine(gen, i))
    for i in range(25):
        gen.add({P + ABSORB_CHILD: child_act()}, _bucket_id_for_genuine(gen, i))
    labels.absorbed_edges.add((ABSORB_PARENT, ABSORB_CHILD))

    # (2E) topical co-occurrence: on a shared "topic" token the parent always
    # fires and the child fires 60% of the time, drawn independently of the
    # parent given the topic. Reverse coverage is therefore 1.0 -- the child
    # never fires off-topic -- while neither feature refines the other.
    for i in range(60):
        feats = {TOPIC_PARENT: parent_act()}
        if i % 5 < 3:                              # 36 of 60 -> clears MIN_JOINT
            feats[P + TOPIC_CHILD] = child_act()
        gen.add(feats, _bucket_id_for_genuine(gen, i))
    labels.topical_edges.add((TOPIC_PARENT, TOPIC_CHILD))

    # (2G) composition: the two component parents co-fire on shared tokens, and
    # the composed child fires exactly on that intersection (36 of 60 clears
    # MIN_JOINT for both edges); each component also fires alone, so the two
    # components are neither duplicates nor parent/child of each other.
    for i in range(60):
        feats = {p: parent_act() for p in COMP_PARENTS}
        if i % 5 < 3:                              # 36 of 60 -> clears MIN_JOINT
            feats[P + COMP_CHILD] = child_act()
        gen.add(feats, _bucket_id_for_genuine(gen, i))
    for i in range(20):                            # solo firings, one component each
        gen.add({COMP_PARENTS[i % 2]: parent_act()}, _bucket_id_for_genuine(gen, i))
    for p in COMP_PARENTS:
        labels.composition_edges.add((p, COMP_CHILD))

    # (2H) multi-parenting: the child fires only with its true parent, but the
    # intruding broad parent is present on every one of those tokens and on 60
    # more of its own, so both edges pass coverage, reconstruction and the
    # frequency control; the child's in-degree of 2 is what out-degree reports.
    for i in range(40):
        gen.add({MULTI_TRUE_PARENT: parent_act(),
                 MULTI_INTRUDER_PARENT: parent_act(),
                 P + MULTI_CHILD: child_act()}, _bucket_id_for_genuine(gen, i))
    for i in range(60):                            # the intruder is broad
        gen.add({MULTI_INTRUDER_PARENT: parent_act()},
                _bucket_id_for_genuine(gen, i))
    for i in range(20):                            # true parent also fires alone
        gen.add({MULTI_TRUE_PARENT: parent_act()}, _bucket_id_for_genuine(gen, i))
    labels.genuine.add((MULTI_TRUE_PARENT, MULTI_CHILD))
    labels.multiparent_edges.add((MULTI_INTRUDER_PARENT, MULTI_CHILD))

    # (2I) cross-block siblings: identical firing sets and identical activation,
    # one latent in each block -- a duplicate straddling the boundary.
    for i in range(40):
        shared = child_act()
        gen.add({COEXT_PARENT: shared, P + COEXT_CHILD: shared},
                _bucket_id_for_genuine(gen, i))
    labels.coextensive_edges.add((COEXT_PARENT, COEXT_CHILD))

    # (2F) within-block structure, in the CHILD block. 28 fires only on a subset
    # of 27's tokens (asymmetric containment -> a directed edge), 29 and 30 fire
    # on identical tokens (symmetric -> a duplicate, not an edge).
    for i in range(80):
        feats = {P + IN_BLOCK_PARENT: child_act()}
        if i < 35:
            feats[P + IN_BLOCK_CHILD] = child_act()
        gen.add(feats, _bucket_id_for_genuine(gen, i))
    a, b = IN_BLOCK_DUP
    for i in range(40):
        shared = child_act()
        gen.add({P + a: shared, P + b: shared}, _bucket_id_for_genuine(gen, i))
    labels.in_block_edges.add((IN_BLOCK_PARENT, IN_BLOCK_CHILD))
    labels.in_block_duplicates.add(IN_BLOCK_DUP)

    # (3) background tokens on frequent ids (make FREQ_IDS actually frequent).
    for i in range(n_background):
        gen.add({}, FREQ_IDS[i % len(FREQ_IDS)])

    # (2A) superparent overlay: tiny activation on ~superparent_rate of tokens.
    n = len(gen.rows)
    mask = torch.rand(n, generator=gen.g) < superparent_rate
    for i in range(n):
        if bool(mask[i]):
            gen.rows[i][SUPERPARENT] = gen._u(0.015, 0.03)

    return _reduce(gen), labels


def _reduce(gen: _Gen) -> dict:
    """Turn the token list into feats/resid/W_dec then into cached statistics."""
    from metrics.reconstruction import per_token_ablation_gain
    from metrics.token_control import frequency_buckets

    import config as C_cfg  # noqa: N814  (thresholds only; block ranges unused here)

    n = len(gen.rows)
    D = P + C
    feats = torch.zeros(n, D, dtype=torch.float64)
    for i, row in enumerate(gen.rows):
        for f, a in row.items():
            feats[i, f] = a
    tok_ids = torch.tensor(gen.tok_ids, dtype=torch.long)

    # unit-norm decoder directions; residual error = isotropic gaussian noise.
    W_dec = torch.randn(D, D_MODEL, generator=gen.g, dtype=torch.float64)
    W_dec = W_dec / W_dec.norm(dim=1, keepdim=True).clamp(min=1e-8)
    err_scale = (RESID_ERR_ENERGY / D_MODEL) ** 0.5      # E||err||^2 = RESID_ERR_ENERGY
    resid_err = err_scale * torch.randn(n, D_MODEL, generator=gen.g, dtype=torch.float64)
    err = (resid_err * resid_err).sum(dim=1)                       # [n]

    fired = (feats > C_cfg.FIRE_THRESHOLD).double()               # [n, D]
    g = per_token_ablation_gain(feats, resid_err, W_dec)          # [n, D]

    fp = fired[:, :P]                                             # [n, P]
    fc = fired[:, P:]                                             # [n, C]
    gp = g[:, :P]
    gc = g[:, P:]

    fire_count = fired.sum(dim=0)                                 # [D]
    cofire = fp.T @ fc                                            # [P, C]
    g_parent_sum = gp.T @ fc                                      # [P, C]
    err_sum_c = fc.T @ err                                        # [C]
    g_child_sum = (fc * gc).sum(dim=0)                            # [C]
    within_cofire = fc.T @ fc                                     # [C, C]

    # energy / joint-child-union accumulators (schema v2). Computed by the SAME
    # production function so the toy's numbers match exp0_stats.pt exactly; feed
    # the RAW activation slices (accumulate_pair_extras squares + thresholds them).
    from collect_statistics import accumulate_pair_extras

    extras = {
        "energy_cofire": torch.zeros(P, C, dtype=torch.float64),
        "union_count": torch.zeros(P, dtype=torch.float64),
        "union_energy": torch.zeros(P, dtype=torch.float64),
        "energy_total": torch.zeros(P, dtype=torch.float64),
    }
    accumulate_pair_extras(extras, feats[:, :P], feats[:, P:], C_cfg.FIRE_THRESHOLD)

    # frequency buckets from corpus token counts, then per-bucket accumulators.
    vocab = int(tok_ids.max()) + 1
    token_counts = torch.zeros(vocab, dtype=torch.float64)
    token_counts.scatter_add_(0, tok_ids, torch.ones(n, dtype=torch.float64))
    buckets = frequency_buckets(token_counts, C_cfg.FREQ_HIGH_MASS, C_cfg.FREQ_MID_MASS)
    tok_bucket = buckets[tok_ids]                                # [n]

    K = N_FREQ_BUCKETS
    cofire_by_bucket = torch.zeros(K, P, C, dtype=torch.float64)
    fire_c_by_bucket = torch.zeros(K, C, dtype=torch.float64)
    for k in range(K):
        sel = (tok_bucket == k).double().unsqueeze(1)            # [n, 1]
        cofire_by_bucket[k] = fp.T @ (fc * sel)
        fire_c_by_bucket[k] = (fc * sel).sum(dim=0)

    return {
        "P": P,
        "C": C,
        "total_tokens": n,
        "fire_count": fire_count,          # [P + C]
        "fire_p": fire_count[:P],
        "fire_c": fire_count[P:],
        "cofire": cofire,
        "g_parent_sum": g_parent_sum,
        "err_sum_c": err_sum_c,
        "g_child_sum": g_child_sum,
        "within_cofire": within_cofire,
        "energy_cofire": extras["energy_cofire"],
        "energy_total": extras["energy_total"],
        "union_count": extras["union_count"],
        "union_energy": extras["union_energy"],
        "cofire_by_bucket": cofire_by_bucket,
        "fire_c_by_bucket": fire_c_by_bucket,
        "buckets": buckets,
        "token_counts": token_counts,
        # --- per-token view, for the four functions the reduced statistics
        # cannot reach. `resid` is the residual stream the SAE decomposes,
        # x = x_hat + err, which is exactly what stage 03 trains its probes on;
        # building it here rather than in the caller keeps the toy the single
        # definition of the world. `fired` is the [n, D] boolean the probe target
        # and the parent-conditioned masks are read off.
        "resid": feats @ W_dec + resid_err,        # [n, d_model]
        "fired": fired.bool(),                     # [n, D]
        "W_dec": W_dec,                            # [D, d_model]
        "tok_ids": tok_ids,                        # [n]
    }
