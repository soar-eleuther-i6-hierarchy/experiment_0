"""
Greedy Boolean cascade over the property-vs-rest grid.

For each column, forward-select percentile-threshold filters (one per detector, both tails
tried) that isolate that class among survivors, using F1 as the greedy objective. Thresholds are
distribution-free percentiles from one global pool.

For is_a the readout adds enrichment over base rate and a precision against the hard negatives
(structurally confusable classes {transitive, reversed, firing_only}), since a pooled vs-rest
precision hides exactly those confusions.

Inputs are only the 10 firewalled detectors, the answer-key labels, and the latent masks for
absorbed/merged — no other truth enters the rule. The is_a rule is the deployment cascade.
"""

from __future__ import annotations

import logging
from typing import NamedTuple

import torch

from scoring.core.grid import class_members, component_percentiles
from scoring.core.registry import POSITIVE_LABEL

logger = logging.getLogger(__name__)

# Distribution-free percentile grid for the filter thresholds (one global pool).
PCT_LEVELS: tuple[float, ...] = (0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90)
MAX_STEPS = 5          # at most one filter per detector
MIN_F1_GAIN = 0.005    # stop when the best addition adds less than this to F1
# is_a's structurally confusable classes — the hard negatives a pooled vs-rest precision hides.
HARD_NEGATIVES: tuple[str, ...] = ("transitive", "reversed", "firing_only")

_SKIP_ZERO_POS = "0 positive pairs"


class _Candidate(NamedTuple):
    """One greedy-step candidate filter."""
    f1: float
    precision: float
    recall: float
    n_surv: int
    detector: str
    level: float
    tail: str
    mask: torch.Tensor


def _target_mask(y_label: torch.Tensor, target: str,
                 label_masks: dict[str, torch.Tensor] | None) -> torch.Tensor:
    """Positive mask for `target`: a supplied latent mask (absorbed/merged) takes precedence,
    otherwise the generative answer-key class."""
    if label_masks and target in label_masks:
        return label_masks[target].to(torch.bool)
    return class_members(y_label, target)


def _filter_masks(percentiles: dict[str, torch.Tensor]) -> dict[tuple[str, float, str], torch.Tensor]:
    """Every candidate (detector, level, tail) keep-mask. A NaN percentile is excluded from the
    keep-set in both tails."""
    masks: dict[tuple[str, float, str], torch.Tensor] = {}
    for det, pct in percentiles.items():
        defined = ~torch.isnan(pct)
        for lvl in PCT_LEVELS:
            masks[(det, lvl, ">")] = defined & (pct >= lvl)
            masks[(det, lvl, "<")] = defined & (pct <= lvl)
    return masks


def _prf1(pos_mask: torch.Tensor, surv: torch.Tensor) -> tuple[float, float, float, int]:
    """(f1, precision, recall, n_surv) of the survivor set against the target positives."""
    tp = int((pos_mask & surv).sum())
    n_surv = int(surv.sum())
    n_pos = int(pos_mask.sum())
    prec = tp / n_surv if n_surv else 0.0
    rec = tp / n_pos if n_pos else 0.0
    f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) else 0.0
    return f1, prec, rec, n_surv


def _hard_negative_readout(surv: torch.Tensor, pos_mask: torch.Tensor, y_label: torch.Tensor,
                           hard_negatives: tuple[str, ...]) -> dict:
    """Precision of is_a restricted to {is_a} ∪ hard-negatives among survivors, plus per-class
    leak counts — answers "of is_a or a confusable cousin, what fraction is is_a?"."""
    tp = int((pos_mask & surv).sum())
    leak: dict[str, int] = {}
    hard_surv = 0
    for nm in hard_negatives:
        c = int((class_members(y_label, nm) & surv).sum())
        leak[nm] = c
        hard_surv += c
    denom = tp + hard_surv
    prec = tp / denom if denom else float("nan")
    return {"precision": prec, "n_isa_surv": tp, "n_hard_surv": hard_surv,
            "hard_negative_leak": leak}


def greedy_cascade(detectors: dict[str, torch.Tensor], pairs: list[tuple[int, int]],
                   y_label: torch.Tensor, target: str = POSITIVE_LABEL,
                   hard_negatives: tuple[str, ...] | None = None,
                   label_masks: dict[str, torch.Tensor] | None = None,
                   percentiles: dict[str, torch.Tensor] | None = None) -> dict:
    """Greedy forward-selection of percentile filters (one per detector, both tails tried)
    maximizing F1 of `target` among the surviving pairs.

    Returns the trajectory + final rule + base-rate + enrichment, plus the hard-negative readout
    when `hard_negatives` is given. A 0-positive target returns a `{"skipped": ...}` marker (a
    success, not a crash). `percentiles` may be passed in to avoid recomputing per column.
    """
    pos_mask = _target_mask(y_label, target, label_masks)
    n_pos = int(pos_mask.sum())
    if n_pos == 0:
        # A 0-positive target is a success, not a failure; the skip guard lives here so a direct
        # caller is as safe as one going through cascade_grid.
        logger.warning("greedy_cascade: target %r has 0 positive pairs — skipped", target)
        return {"target": target, "skipped": _SKIP_ZERO_POS, "n_pos": 0}

    if percentiles is None:
        percentiles = component_percentiles(detectors, pairs)
    masks = _filter_masks(percentiles)
    neg_mask = ~pos_mask
    n_neg = int(neg_mask.sum())
    n_pairs = len(pairs)
    base_rate = n_pos / n_pairs if n_pairs else float("nan")

    surv = torch.ones(n_pairs, dtype=torch.bool)
    used: set[str] = set()
    base_f1, base_prec, base_rec, base_surv = _prf1(pos_mask, surv)
    steps = [{"rule": "(all pairs)", "precision": base_prec, "survival": base_rec,
              "neg_leak": 1.0 if n_neg else float("nan"), "f1": base_f1,
              "n_surv": base_surv, "n_nan_excluded": 0}]
    prev_f1 = base_f1
    for _ in range(MAX_STEPS):
        best: _Candidate | None = None
        for (det, lvl, tail), m in masks.items():
            if det in used:                          # one filter per detector
                continue
            cand = surv & m
            n_surv = int(cand.sum())
            if n_surv == 0:                          # a wipe-out filter is never useful — reject explicitly
                continue
            f1, prec, rec, _ = _prf1(pos_mask, cand)
            if best is None or f1 > best.f1:
                best = _Candidate(f1, prec, rec, n_surv, det, lvl, tail, cand)
        if best is None or best.f1 - prev_f1 < MIN_F1_GAIN:
            break
        # How many pre-filter survivors this detector couldn't score (dropped, not thresholded out).
        n_nan = int((surv & torch.isnan(percentiles[best.detector])).sum())
        surv = best.mask
        used.add(best.detector)
        prev_f1 = best.f1
        neg_leak = int((neg_mask & surv).sum()) / n_neg if n_neg else float("nan")
        steps.append({"rule": f"{best.detector} pct{best.tail}{best.level:.2f}",
                      "precision": best.precision, "survival": best.recall,
                      "neg_leak": neg_leak, "f1": best.f1, "n_surv": best.n_surv,
                      "n_nan_excluded": n_nan})

    final = steps[-1]
    enrichment = (final["precision"] / base_rate) if base_rate else float("nan")
    n_metrics = len(steps) - 1
    result = {
        "target": target, "steps": steps,
        "final_rule": " AND ".join(s["rule"] for s in steps[1:]) or "(none)",
        "base_rate": base_rate, "enrichment": enrichment,
        "final_precision": final["precision"], "final_survival": final["survival"],
        "n_metrics": n_metrics, "trivial": n_metrics == 0,
    }
    if hard_negatives:
        result["hard_negative"] = _hard_negative_readout(surv, pos_mask, y_label, hard_negatives)
    return result


def cascade_grid(detectors: dict[str, torch.Tensor], pairs: list[tuple[int, int]],
                 y_label: torch.Tensor, columns: tuple[str, ...],
                 label_masks: dict[str, torch.Tensor] | None = None) -> dict:
    """Greedy cascade for every column. A 0-positive column is skipped by `greedy_cascade` itself
    (a success, not a crash); the is_a column carries the hard-negative readout. Percentiles are
    computed once and shared across columns."""
    percentiles = component_percentiles(detectors, pairs)
    out: dict[str, dict] = {}
    for col in columns:
        hn = HARD_NEGATIVES if col == POSITIVE_LABEL else None
        out[col] = greedy_cascade(detectors, pairs, y_label, col, hard_negatives=hn,
                                  label_masks=label_masks, percentiles=percentiles)
    return out
