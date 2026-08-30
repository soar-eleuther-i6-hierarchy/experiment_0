"""Oracle score characterization for a single-property toy world (Stage-1 oracle read).

Given a built `WorldBundle` and its recovered-feature list, compute every detector's
per-ordered-pair score on the GROUND-TRUTH latents (`bundle.A`) and TRUE geometry, with
s_res in cosine mode -- the metric's analytic ideal, no SAE involved. Group the scores by
each pair's single ground-truth class so a toy's per-class score distribution reads directly.

This is the oracle half of the single-property program: on perfect latents there is no
absorption or capacity pressure, so an elevated score on a class the metric has no reason to
respond to is a FORMULA flaw, not an SAE artifact. We characterize how the scores separate
the property space -- per-class distributions + precision against the genuine `unrelated`
null -- rather than a property-vs-rest AUROC (which would need injected false-positive
classes and contaminate the toy).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch

from scoring.core.detectors import compute_all
from scoring.core.grid import pair_frame
from scoring.core.registry import CONSTANTS, DETECTOR_SIGN, DETECTORS, SYMMETRIC_DETECTORS
from scoring.oracle.validate_metrics import pure_inputs
from toygen import labels

if TYPE_CHECKING:                       # WorldBundle is used only as a type hint
    from scoring.core.world import WorldBundle

_NULL_CLASS = "unrelated"
_MIN_TABLE_N = 10           # classes with fewer pairs are censored from the summary table
_NULL_QUANTILE = 0.90       # precision-vs-null cut: fraction of a class above the null's p90


def oracle_scores(bundle: "WorldBundle", feats: list[int], constants: dict | None = None,
                  s_res_mode: str = "cosine") -> dict[str, dict[str, torch.Tensor]]:
    """`{detector: {class_name: 1-D tensor of per-pair scores}}` on the oracle latents.

    Scores come from `compute_all` on the TRUE coefficients (`bundle.A`) and TRUE geometry;
    `s_res` defaults to cosine (the analytic ideal, as the metric is defined in the oracle
    read). Pairs are grouped by their single ground-truth class; NaN (undefined) cells are
    kept, so a class's finite fraction can be read alongside its distribution. Classes absent
    from a pure toy come back as empty tensors, not missing keys.
    """
    constants = CONSTANTS if constants is None else constants
    idx = torch.tensor(feats, dtype=torch.long)
    acts = bundle.A[:, idx]
    dets = compute_all(pure_inputs(bundle, feats, acts), constants, s_res_mode=s_res_mode)
    pairs, y = pair_frame(feats, bundle.pair_labels)
    pa = torch.tensor([a for a, _ in pairs], dtype=torch.long)
    pb = torch.tensor([b for _, b in pairs], dtype=torch.long)
    out: dict[str, dict[str, torch.Tensor]] = {}
    for det in DETECTORS:
        vals = dets[det][pa, pb]
        out[det] = {name: vals[y == labels._index(name)] for name in labels.LABELS}
    return out


def _finite(vals: torch.Tensor) -> torch.Tensor:
    return vals[torch.isfinite(vals)]


def _q(vals: torch.Tensor, q: float) -> float:
    return float(torch.quantile(vals.double(), q)) if vals.numel() else float("nan")


def summarize(scores: dict[str, dict[str, torch.Tensor]], null_class: str = _NULL_CLASS,
              min_n: int = _MIN_TABLE_N, null_quantile: float = _NULL_QUANTILE) -> list[dict]:
    """Per-(detector, class) stats rows from `oracle_scores` output.

    Each row carries n, finite count, median, p10/p90 of the FINITE scores, and
    `prec_vs_null` = fraction of the class's finite scores strictly above the null class's
    finite p90 (a separation against the GENUINE null, not an injected-negative AUROC).
    Classes with fewer than `min_n` pairs are flagged `censored` (kept for the raw dump,
    excluded from any trusted reading), matching the N<10 reporting rule.
    """
    rows: list[dict] = []
    for det in DETECTORS:
        by_class = scores[det]
        null_fin = _finite(by_class.get(null_class, torch.empty(0)))
        null_p90 = _q(null_fin, null_quantile)
        for name in labels.LABELS:
            vals = by_class[name]
            fin = _finite(vals)
            n = int(vals.numel())
            prec = (float((fin > null_p90).double().mean())
                    if fin.numel() and null_p90 == null_p90 else float("nan"))
            rows.append({
                "detector": det, "class": name, "n": n, "n_finite": int(fin.numel()),
                "median": _q(fin, 0.5), "p10": _q(fin, 0.10), "p90": _q(fin, 0.90),
                "prec_vs_null": prec, "censored": n < min_n,
            })
    return rows


def symmetric_detectors(s_res_mode: str | None) -> set[str]:
    """The detectors symmetric in (parent, child) for THIS read.

    `pmi` is always symmetric (registry). `s_res` is symmetric ONLY in cosine mode -- there it
    is a Gram matrix `W @ W.T`, so `reversed` equals `is_a` by construction and a high
    `reversed` score is EXPECTED, not a finding. In probe mode (the trained read) `s_res` is a
    directional probe margin and stays out of this set.
    """
    sym = set(SYMMETRIC_DETECTORS)
    if s_res_mode == "cosine":
        sym.add("s_res")
    return sym


def _reversed_note(det: str, symmetric: set[str]) -> str:
    """How a high `reversed` score should read for this detector, given the read's symmetry set."""
    if det in symmetric:
        return "symmetric: high-on-reversed EXPECTED"
    sign = DETECTOR_SIGN.get(det, 1)
    return f"directional (sign {sign:+d}): high-on-reversed is a FINDING"


def _fmt(x: float) -> str:
    return "--" if x != x else f"{x:.3f}"


def format_scores_md(scores: dict[str, dict[str, torch.Tensor]], summary: list[dict],
                     meta: dict) -> str:
    """Render the oracle score dump as a readable per-detector Markdown table."""
    present = sorted({r["class"] for r in summary if r["n"] > 0},
                     key=lambda c: labels.LABELS.index(c))
    L = [f"# Oracle scores — {meta['config']} (seed {meta['seed']}, "
         f"n_tokens {meta['n_tokens']})", "",
         f"World: F={meta['F']}  true_L0={meta['true_l0']:.2f}  n_pairs={meta['n_pairs']}",
         f"Classes present: {', '.join(present)}", "",
         "_Scores on ORACLE latents (true A + true g), s_res=cosine (the analytic ideal). "
         "`prec_vs_null` = fraction of the class's finite scores above the `unrelated` null's "
         f"p{int(_NULL_QUANTILE * 100)}. Classes with N<{_MIN_TABLE_N} are censored._", ""]
    symmetric = symmetric_detectors(meta.get("s_res_mode"))
    by = {(r["detector"], r["class"]): r for r in summary}
    for det in DETECTORS:
        L += ["", f"## {det} — {_reversed_note(det, symmetric)}", "",
              "| class | n | n_finite | median | p10 | p90 | prec_vs_null | flag |",
              "|---|---|---|---|---|---|---|---|"]
        for name in labels.LABELS:
            r = by[(det, name)]
            if r["n"] == 0:
                continue
            if r["censored"]:
                flag = "censored N<%d" % _MIN_TABLE_N
            elif name == "reversed":
                flag = "expected (symmetric)" if det in symmetric else "FINDING if high"
            else:
                flag = ""
            L.append(f"| {name} | {r['n']} | {r['n_finite']} | {_fmt(r['median'])} "
                     f"| {_fmt(r['p10'])} | {_fmt(r['p90'])} | {_fmt(r['prec_vs_null'])} | {flag} |")
    return "\n".join(L) + "\n"


def write_score_reports(scores: dict[str, dict[str, torch.Tensor]], summary: list[dict],
                        meta: dict, out: Path) -> tuple[Path, Path]:
    """Write oracle_scores.npz (per-class per-detector raw scores + a JSON meta/summary
    sidecar entry) and oracle_scores.md (the readable stats table) into `out`."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    npz_path, md_path = out / "oracle_scores.npz", out / "oracle_scores.md"
    arrays: dict[str, np.ndarray] = {}
    for det in DETECTORS:
        for name in labels.LABELS:
            arrays[f"{det}__{name}"] = scores[det][name].detach().cpu().numpy().astype(np.float32)
    # `__meta__`/`__summary__` ride along as 0-d JSON strings so the npz is self-describing.
    arrays["__meta__"] = np.array(json.dumps(meta))
    arrays["__summary__"] = np.array(json.dumps(summary))
    np.savez_compressed(npz_path, **arrays)
    md_path.write_text(format_scores_md(scores, summary, meta), encoding="utf-8")
    return npz_path, md_path
