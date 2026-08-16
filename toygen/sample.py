"""
The sampler: documents, topics, token ids, coefficients, activations.

Structural rules enforced token-wise, not on average:
  - containment  (A1): a child can only fire where its parent fires
  - exclusivity       : mutually exclusive siblings never co-fire

Both are drawn so the marginals still come out at their designed values — exclusive
children share one uniform draw partitioned by their `p_edge`, which is exact as long
as the sibling budget holds — each parent's child edge-probs sum to <= 1 (spec 0.7-5).

Manifold features draw a continuous, bounded strength with the same mean and spread as
the discrete features — a smooth graded axis rather than an on/off switch.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from .geometry import Geometry
from .spec import ToyConfig
from .strengths import StrengthSpec, topic_rates
from .tree import Tree

DT = torch.float64


@dataclass
class World:
    A: torch.Tensor            # [n, F] true coefficients
    Atilde: torch.Tensor       # [n, F] healthy coefficients (= A @ Lam.T)
    h: torch.Tensor            # [n, D] activations
    noise: torch.Tensor        # [n, D]
    tokens: torch.Tensor       # [n] token ids
    docs: torch.Tensor         # [n] document index
    topics: torch.Tensor       # [n] topic of the token's document
    token_sets: dict[int, torch.Tensor]


def _zipf_probs(vocab: int, s: float) -> torch.Tensor:
    r = torch.arange(1, vocab + 1, dtype=DT)
    w = r ** (-s)
    return w / w.sum()


def _strength(n: int, mean_strength: float, cv: float, kind: str, gen: torch.Generator) -> torch.Tensor:
    """Positive strength with mean `mean_strength` and sd `cv*mean_strength`, exactly.

    Discrete features use a lognormal (right-skewed, strictly positive) with matched
    first two moments. Manifold features use a *uniform* draw with the same two
    moments, giving the smooth bounded axis C-tile needs; the positivity of its lower
    endpoint requires cv < 1/sqrt(3).

    Lognormal rather than Gamma because `torch.distributions.Gamma` does not accept an
    explicit `generator`, and the only sampler that does is `torch._standard_gamma` —
    a private API with no stability guarantee. `torch.randn` is public and seedable.
    """
    if n == 0:
        return torch.zeros(0, dtype=DT)
    if kind == "manifold":
        b = cv * mean_strength * (12.0 ** 0.5)
        a = mean_strength - b / 2.0
        if a <= 0:
            raise ValueError(f"manifold feature needs cv < 1/sqrt(3); got cv={cv}")
        return a + b * torch.rand(n, generator=gen, dtype=DT)
    sig2 = math.log(1.0 + cv ** 2)
    mean_log = math.log(mean_strength) - sig2 / 2.0
    z = torch.randn(n, generator=gen, dtype=DT)
    return torch.exp(mean_log + math.sqrt(sig2) * z)


def sample_world(cfg: ToyConfig, tree: Tree, strengths: StrengthSpec, geo: Geometry, n_tokens: int, seed: int | None = None) -> World:
    # offset from the geometry stream so the two draws are independent
    gen = torch.Generator().manual_seed((cfg.seed if seed is None else seed) + 1000)
    F = tree.F

    if n_tokens <= 0:
        raise ValueError(f"n_tokens must be positive; got {n_tokens}")
    n_docs = math.ceil(n_tokens / cfg.doc_len)
    n = n_tokens
    docs = torch.arange(n, dtype=torch.long) // cfg.doc_len
    doc_topic = torch.randint(0, cfg.Z, (n_docs,), generator=gen)
    topics = doc_topic[docs]

    probs = _zipf_probs(cfg.vocab, cfg.zipf_s)
    tokens = torch.multinomial(probs, n, replacement=True, generator=gen)

    # All token-bound features share the top-frequency ids, so their co-firing lives
    # entirely in the top frequency bucket. Per-pair disjoint id sets
    # do not work: under Zipf the mass of ids 2i, 2i+1 falls below the design firing
    # rate within a few pairs, and the realised rate is then silently capped by the
    # ids rather than set by p_k.
    high_ids = torch.arange(cfg.n_bind_ids, dtype=torch.long)
    token_sets: dict[int, torch.Tensor] = {
        k: high_ids for k in range(F) if tree.token_bound[k]
    }

    fires = torch.zeros(n, F, dtype=torch.bool)

    def root_fire(k: int) -> torch.Tensor:
        p_k = float(strengths.p[k])
        if tree.token_bound[k]:
            ids = token_sets[k]
            inside = torch.isin(tokens, ids)
            frac = float(inside.double().mean())
            r = min(1.0, p_k / max(frac, 1e-12))
            return inside & (torch.rand(n, generator=gen, dtype=DT) < r)
        if tree.kappa[k] > 0.0:
            rates = topic_rates(p_k, tree.kappa[k], tree.topic[k], strengths.topic_prior)
            return torch.rand(n, generator=gen, dtype=DT) < rates[topics]
        return torch.rand(n, generator=gen, dtype=DT) < p_k

    for k in tree.topology_ordering:
        par = tree.parent_of(k)
        if par is None:
            fires[:, k] = root_fire(k)
            continue
        if tree.exclusive.get(par, False):
            continue                                      # handled per-parent below
        fires[:, k] = fires[:, par] & (torch.rand(n, generator=gen, dtype=DT) < tree.p_edge_of(k))

    # Exclusive sibling groups: one uniform draw per token, partitioned by p_edge.
    # Iterate `tree.topology_ordering`, NOT `tree.children` — this loop reads `fires[:, par]`, so a parent that is itself an exclusive child must already have been resolved. Dict insertion order happens to give that today, but only incidentally; topological order makes it structural.
    for par in tree.topology_ordering:
        kids = tree.children.get(par, [])
        if not tree.exclusive.get(par, False) or not kids:
            continue
        u = torch.rand(n, generator=gen, dtype=DT)
        lo = torch.zeros(n, dtype=DT)
        for c in kids:
            hi = lo + tree.p_edge_of(c)
            fires[:, c] = fires[:, par] & (u >= lo) & (u < hi)
            lo = hi

    A = torch.zeros(n, F, dtype=DT)
    for k in range(F):
        idx = fires[:, k].nonzero(as_tuple=True)[0]
        if idx.numel():
            A[idx, k] = _strength(idx.numel(), float(strengths.mean_strength[k]), cfg.strength_spread, tree.kind[k], gen)

    Atilde = A @ geo.Lam.T
    noise = cfg.noise_sigma * torch.randn(n, cfg.D, generator=gen, dtype=DT)
    h = A @ geo.g + noise
    return World(A=A, Atilde=Atilde, h=h, noise=noise, tokens=tokens, docs=docs, topics=topics, token_sets=token_sets)
