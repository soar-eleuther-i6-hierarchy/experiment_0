"""
Sparsity-sweep aggregation — the calibration curve "metric AUROC vs measured dispersion".

Absorption (the thing that erases decoder geometry) is driven by the SAE's sparsity: a tighter
top-k forces the SAE to fold child concepts into their parents. Train the same world at several
`k`, and each metric traces a dose-response against the measured dispersion. This module collapses
a list of `run_retrieval` reports (each carrying `meta.k`, `dispersion_mean`, and
`grid[det][col].auroc`) into that curve, so a geometry metric can be declared reliable below a
dispersion threshold — the one thing that transfers to a real SAE (dispersion is decoder-only).
"""

from __future__ import annotations

import math
from collections import defaultdict
from itertools import permutations

import torch

from scoring.retrieval import _spearman

DT = torch.float64


def _cell_auroc(report: dict, detector: str, column: str) -> float:
    return report.get("grid", {}).get(detector, {}).get(column, {}).get("auroc", float("nan"))


def sweep_curve(reports: list[dict], detector: str, column: str) -> list[dict]:
    """Group reports by `k`, average dispersion + the cell AUROC across seeds, sort by `k`.

    Returns `[{"k", "dispersion", "auroc", "n_seeds"}, ...]` ascending in `k`. `n_seeds` counts
    the reports at that `k`; averages ignore NaN values (a NaN cell doesn't poison the mean).
    """
    groups: dict[int, list[tuple[float, float]]] = defaultdict(list)
    for r in reports:
        k = r.get("meta", {}).get("k")
        if k is None:                        # a report with no sparsity can't sit on the k-axis
            continue                         # (run_retrieval drops meta.k if the checkpoint lacks it)
        groups[int(k)].append((r.get("dispersion_mean", float("nan")),
                               _cell_auroc(r, detector, column)))

    def _finite(xs: list[float]) -> list[float]:
        return [x for x in xs if x is not None and math.isfinite(x)]

    def _mean(xs: list[float]) -> float:
        f = _finite(xs)
        return sum(f) / len(f) if f else float("nan")

    curve = []
    for k in sorted(groups):
        pts = groups[k]
        # n_seeds = reports at this k; n_finite = how many actually backed the averaged AUROC
        # (so a mostly-NaN k isn't read as well-powered).
        curve.append({"k": k,
                      "dispersion": _mean([d for d, _ in pts]),
                      "auroc": _mean([a for _, a in pts]),
                      "n_seeds": len(pts),
                      "n_finite": len(_finite([a for _, a in pts]))})
    return curve


def dispersion_reliability(reports: list[dict], detector: str, column: str) -> dict:
    """Dose-response across ALL (k, seed) points: Spearman(dispersion, cell AUROC).

    Computed over the individual points (not the seed-averaged curve), so it reflects the full
    spread. A geometry metric on its target column should be strongly NEGATIVE (more dispersion →
    worse AUROC). Returns `{"spearman", "n_points", "n_k_levels", "n_dropped"}`.

    `n_points` OVERSTATES independence: the points cluster by `k` (a sweep of 4 k-levels × N
    seeds is 4 clusters, not 4N independent draws), so `n_k_levels` — the count of distinct `k`
    that survived — is the honest scale for any significance claim, and a naive n=`n_points`
    test on this clustered null over-rejects. `n_dropped` counts points excluded because their
    dispersion or AUROC was non-finite (a whole k-level can silently vanish, and the low-k end
    the calibration depends on is exactly where support is thinnest). Spearman is NaN for <2
    points or a zero-variance (flat) detector — never a tie-fabricated correlation (`_spearman`
    is tie-safe).
    """
    disps, aurs, ks = [], [], []
    n_dropped = 0
    for r in reports:
        d = r.get("dispersion_mean", float("nan"))
        a = _cell_auroc(r, detector, column)
        if d is not None and a is not None and math.isfinite(d) and math.isfinite(a):
            disps.append(float(d))
            aurs.append(float(a))
            k = r.get("meta", {}).get("k")
            if k is not None:
                ks.append(int(k))
        else:
            n_dropped += 1
    n = len(disps)
    n_k_levels = len(set(ks))
    # PRIMARY statistic: Spearman over the per-k SEED-AVERAGED points (n = n_k_levels), because the
    # (k,seed) points cluster by k — the all-points rho is pulled toward whichever k-levels kept more
    # seeds and is not a consistent "dose-response across k". `perm_p` is its EXACT two-sided
    # permutation p (enumerate all k-level label permutations); with only ~4 k-levels the floor is
    # 2/k! (≈0.083 at k=4), which is the honest inferential ceiling the all-points n would hide.
    curve = sweep_curve(reports, detector, column)
    ck = [(p["dispersion"], p["auroc"]) for p in curve
          if math.isfinite(p["dispersion"]) and math.isfinite(p["auroc"])]
    spearman_by_k, perm_p = _by_k_spearman_and_perm_p(ck)
    if n < 2:
        return {"spearman": float("nan"), "n_points": n, "n_k_levels": n_k_levels,
                "n_dropped": n_dropped, "spearman_by_k": spearman_by_k, "perm_p": perm_p}
    rho = _spearman(torch.tensor(disps, dtype=DT), torch.tensor(aurs, dtype=DT))
    return {"spearman": rho, "n_points": n, "n_k_levels": n_k_levels, "n_dropped": n_dropped,
            "spearman_by_k": spearman_by_k, "perm_p": perm_p}


# Exact k!-enumeration is only cheap for few k-levels (8! = 40k perms ≈ 8s; 9! ≈ 73s; 10!+ hangs).
# At or below this many levels we enumerate exactly; above it we fall back to a fixed-seed
# Monte-Carlo permutation test so a wider sweep degrades gracefully instead of blocking the run.
_PERM_EXACT_MAX = 7
_PERM_MC_SAMPLES = 20_000


def _by_k_spearman_and_perm_p(points: list[tuple[float, float]]) -> tuple[float, float]:
    """Spearman over per-k points + its two-sided permutation p-value.

    `points` = the finite (dispersion, auroc) per-k averaged points. Returns (spearman_by_k,
    perm_p). Both are NaN for < 3 k-levels (a perfect |ρ| over 2 points is meaningless) or a flat
    detector (zero-variance AUROC → no dose-response). `perm_p` = fraction of AUROC-label
    permutations whose |Spearman| ≥ |observed|: EXACT (all k!) for k ≤ `_PERM_EXACT_MAX`, else a
    deterministic fixed-seed Monte-Carlo estimate over `_PERM_MC_SAMPLES` draws (the exact
    enumeration is O(k!) and would hang a wide sweep). Both forms include the identity permutation,
    so `perm_p` is never a fabricated 0.
    """
    m = len(points)
    if m < 3:
        return float("nan"), float("nan")
    d = torch.tensor([p[0] for p in points], dtype=DT)
    a = torch.tensor([p[1] for p in points], dtype=DT)
    rho = _spearman(d, a)
    if not math.isfinite(rho):
        return float("nan"), float("nan")
    obs = abs(rho)
    if m <= _PERM_EXACT_MAX:
        perms = [list(pm) for pm in permutations(range(m))]
    else:
        g = torch.Generator().manual_seed(0)                 # deterministic MC fallback
        perms = [torch.randperm(m, generator=g).tolist() for _ in range(_PERM_MC_SAMPLES)]
        perms.append(list(range(m)))                         # always include the identity
    hits = sum(1 for pm in perms
               if math.isfinite(r := _spearman(d, a[pm])) and abs(r) >= obs - 1e-12)
    return rho, hits / len(perms)


def sweep_table(reports: list[dict], cells: list[tuple[str, str]]) -> dict:
    """A `sweep_curve` per requested `(detector, column)` cell, keyed by the tuple."""
    return {(det, col): sweep_curve(reports, det, col) for det, col in cells}
