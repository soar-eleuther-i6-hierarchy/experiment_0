"""
Synthetic ground-truth "toy" for calibrating the five Exp 0 metrics.

The project plan (Exp 0, "How we decide which metric works") asks us to keep the
metrics that BOTH recover a known parent-child tree AND reject pathological edges
we inject on purpose. Real gemma-2-2b caches have no ground truth, so we build a
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
D_MODEL = 16
P = 8           # parent-block features
C = 32          # child-block features
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
    cofire_by_bucket, fire_c_by_bucket, within_cofire, buckets, total_tokens.
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
    resid_err = 0.20 * torch.randn(n, D_MODEL, generator=gen.g, dtype=torch.float64)
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
        "cofire_by_bucket": cofire_by_bucket,
        "fire_c_by_bucket": fire_c_by_bucket,
        "buckets": buckets,
        "token_counts": token_counts,
    }
