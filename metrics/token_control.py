"""
Metric 5 - Token-frequency-controlled coverage (project plan Exp 0, bullet 5;
idea from Tokenized SAEs).

Hypothesis under test: poly-parenting / superparents are driven by
high-frequency tokens (spaces, punctuation, "the"). If an edge's coverage is
carried by those tokens, conditioning them away should destroy it; a genuine
semantic edge should survive.

We bucket token ids by their share of corpus MASS (not rank): bucket 0 =
most frequent ids covering the top ~50% of all tokens, bucket 1 = next ~40%,
bucket 2 = the long tail. The streaming pass accumulates fire/cofire counts
per bucket; here we recompute reverse coverage per bucket and report a
survival ratio:

    survival(p, c) = R_low+mid(p, c) / R_all(p, c)

    ~1  -> edge holds on rare tokens too (genuine)
    ~0  -> edge exists only on frequent tokens (frequency capture)
"""

from __future__ import annotations

import torch


def frequency_buckets(
    token_counts: torch.Tensor,   # [vocab] corpus counts per token id
    high_mass: float = 0.50,
    mid_mass: float = 0.40,
) -> torch.Tensor:
    """Assign every token id a bucket 0 (high) / 1 (mid) / 2 (low) by cumulative mass."""
    counts = token_counts.double()
    order = torch.argsort(counts, descending=True)
    csum = torch.cumsum(counts[order], dim=0) / counts.sum().clamp(min=1.0)
    bucket_sorted = torch.full_like(order, 2)
    bucket_sorted[csum <= high_mass + mid_mass] = 1
    bucket_sorted[csum <= high_mass] = 0
    buckets = torch.empty_like(order)
    buckets[order] = bucket_sorted
    return buckets


def frequency_controlled_coverage(
    cofire_by_bucket: torch.Tensor,   # [K, P, C] co-firing counts per bucket
    fire_c_by_bucket: torch.Tensor,   # [K, C]    child firing counts per bucket
    edge_mask: torch.Tensor,          # [P, C]    kept edges (from all-token coverage)
    min_fire_low: int = 5,
) -> dict[str, torch.Tensor]:
    """Per-edge reverse coverage per bucket + survival on non-frequent tokens.

    survival[p, c] = R restricted to buckets 1+2 (mid+low), divided by R_all.
    Edges whose child barely fires outside bucket 0 (fewer than min_fire_low
    tokens) are marked untestable (survival = nan) rather than failed.
    """
    cf = cofire_by_bucket.double()
    fc = fire_c_by_bucket.double()

    R_all = cf.sum(0) / fc.sum(0).clamp(min=1.0).unsqueeze(0)          # [P, C]
    cf_rest = cf[1:].sum(0)                                            # [P, C]
    fc_rest = fc[1:].sum(0)                                            # [C]
    R_rest = cf_rest / fc_rest.clamp(min=1.0).unsqueeze(0)             # [P, C]

    survival = R_rest / R_all.clamp(min=1e-12)
    survival = survival.clamp(max=1.5)  # noise guard: tiny denominators
    untestable = (fc_rest < min_fire_low).unsqueeze(0).expand_as(survival)
    survival = torch.where(untestable, torch.nan, survival)
    survival = torch.where(edge_mask, survival, torch.nan)

    return {"R_all": R_all, "R_rest": R_rest, "survival": survival}
