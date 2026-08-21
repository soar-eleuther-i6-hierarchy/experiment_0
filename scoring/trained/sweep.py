"""
Sparsity-sweep aggregation — the calibration curve "metric AUROC vs measured dispersion".

Absorption is driven by the SAE's sparsity: a tighter top-k folds child concepts into parents.
Training the same world at several `k` traces a dose-response between each metric and the
measured dispersion. This module collapses a list of `run_retrieval` reports into that curve, so
a geometry metric can be declared reliable below a dispersion threshold.
"""

from __future__ import annotations

import math
from collections import defaultdict
from itertools import permutations

import torch

from scoring.core.grid import _spearman

DT = torch.float64


def _cell_auroc(report: dict, detector: str, column: str) -> float:
    return report.get("grid", {}).get(detector, {}).get(column, {}).get("auroc", float("nan"))


def sweep_curve(reports: list[dict], detector: str, column: str) -> list[dict]:
    """Group reports by `k`, average dispersion + the cell AUROC across seeds, sort by `k`.

    Returns `[{"k", "dispersion", "auroc", "n_seeds"}, ...]` ascending in `k`; averages ignore NaN.
    """
    groups: dict[int, list[tuple[float, float]]] = defaultdict(list)
    for r in reports:
        k = r.get("meta", {}).get("k")
        if k is None:                        # a report with no sparsity can't sit on the k-axis
            continue
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
        # n_finite = how many actually backed the averaged AUROC (so a mostly-NaN k isn't misread as well-powered)
        curve.append({"k": k,
                      "dispersion": _mean([d for d, _ in pts]),
                      "auroc": _mean([a for _, a in pts]),
                      "n_seeds": len(pts),
                      "n_finite": len(_finite([a for _, a in pts]))})
    return curve


def dispersion_reliability(reports: list[dict], detector: str, column: str) -> dict:
    """Dose-response across all (k, seed) points: Spearman(dispersion, cell AUROC).

    Computed over individual points, so it reflects the full spread. A geometry metric on its
    target column should be strongly negative (more dispersion, worse AUROC). Returns
    `{"spearman", "n_points", "n_k_levels", "n_dropped"}`.

    `n_points` overstates independence since points cluster by `k`, so `n_k_levels` is the honest
    scale for significance. `n_dropped` counts points excluded for non-finite dispersion/AUROC.
    Spearman is NaN for <2 points or a flat detector.
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
    # Primary statistic: Spearman over per-k seed-averaged points, since all-points rho is pulled
    # toward whichever k-levels kept more seeds. `perm_p` is its exact two-sided permutation p.
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


# Exact k!-enumeration is only cheap for few k-levels; above this we fall back to Monte-Carlo.
_PERM_EXACT_MAX = 7
_PERM_MC_SAMPLES = 20_000


def _by_k_spearman_and_perm_p(points: list[tuple[float, float]]) -> tuple[float, float]:
    """Spearman over per-k points + its two-sided permutation p-value.

    Returns (spearman_by_k, perm_p), both NaN for <3 k-levels or a flat detector. `perm_p` is the
    fraction of label permutations with |Spearman| >= |observed|: exact for k <= `_PERM_EXACT_MAX`,
    else a fixed-seed Monte-Carlo estimate. Always includes the identity permutation.
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
