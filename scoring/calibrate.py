"""Oracle threshold-calibration harness.

Runs the clean-oracle ceiling (a perfect-decoder dictionary, `stage0.clean_grid`) across
configs/seeds WITHOUT any trained SAE, and tabulates the per-(detector, column) ceiling AUROCs
-- so thresholds can be chosen from what a perfect dictionary achieves, in analysis, rather than
hand-picked. The ceiling depends only on the toy world's truth (`g`, `A`, tokens) and a chosen target L0 (the config's true L0).
"""
from __future__ import annotations

import dataclasses
import math

from scoring.registry import CONSTANTS, DETECTORS, SCORED_COLUMNS
from scoring.stage0 import clean_grid
from scoring.harness import regenerate_world
from scoring.retrieval import pair_frame
from toygen import spec, strengths
from toygen.world import resolve_config


def oracle_ceiling_grid(config_name: str, seed: int, n_tokens: int, target_l0: float | None = None, overrides: dict | None = None) -> dict:
    """Per-(detector, column) clean-oracle ceiling AUROC grid for one config + seed.

    Rebuilds the same world the trained SAE would see (geometry + sampling both from `seed`),
    uses an identity match over all features (`feats = range(F)`), and grades the oracle
    dictionary at the config's true L0. No SAE is loaded.
    """
    cfg = spec.replace(resolve_config(config_name, **(overrides or {})), seed=seed)
    bundle = regenerate_world(dataclasses.asdict(cfg), sample_seed=seed, n_tokens=n_tokens)
    feats = list(range(bundle.tree.F))
    pairs, y_label = pair_frame(feats, bundle.pair_labels)
    l0 = strengths.target_l0(bundle.tree) if target_l0 is None else target_l0
    return clean_grid(bundle, feats, pairs, y_label, SCORED_COLUMNS, CONSTANTS, l0)


def aggregate_ceilings(grids: list[dict]) -> dict:
    """Mean/std ceiling AUROC across seeds per (detector, column), skipping NaN cells."""
    out: dict[str, dict[str, dict]] = {}
    dets: set[str] = set().union(*[set(g.keys()) for g in grids]) if grids else set()
    for det in dets:
        out[det] = {}
        cols: set[str] = set().union(*[set(g.get(det, {}).keys()) for g in grids])
        for col in cols:
            vals = [g[det][col]["auroc"] for g in grids
                    if det in g and col in g[det] and not math.isnan(g[det][col]["auroc"])]
            if vals:
                m = sum(vals) / len(vals)
                sd = (sum((v - m) ** 2 for v in vals) / len(vals)) ** 0.5
                out[det][col] = {"mean_ceiling": m, "std_ceiling": sd, "n_seeds": len(vals)}
            else:
                out[det][col] = {"mean_ceiling": float("nan"), "std_ceiling": float("nan"),
                                 "n_seeds": 0}
    return out


def _markdown_table(agg_ceilings: dict) -> str:
    """Markdown table of clean-oracle ceiling AUROCs (mean over seeds) per detector x column."""
    head = "| detector | " + " | ".join(SCORED_COLUMNS) + " |"
    sep = "|" + "---|" * (len(SCORED_COLUMNS) + 1)
    rows = []
    for det in DETECTORS:
        row = agg_ceilings.get(det, {})
        cells = []
        for c in SCORED_COLUMNS:
            m = row.get(c, {}).get("mean_ceiling")
            cells.append(f"{m:.3f}" if isinstance(m, float) and not math.isnan(m) else "-")
        rows.append(f"| {det} | " + " | ".join(cells) + " |")
    return "\n".join([head, sep, *rows])


def main() -> None:
    import argparse
    import json
    from pathlib import Path

    ap = argparse.ArgumentParser(description="Oracle clean-ceiling calibration (per detector x column).")
    ap.add_argument("--config", default="full", choices=sorted(spec.CONFIGS))
    ap.add_argument("--n-superparent", type=int, default=None)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--n-tokens", type=int, default=200_000)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    overrides = ({"n_superparent": args.n_superparent}
                 if args.n_superparent is not None else {})
    grids = []
    for s in args.seeds:
        try:
            grids.append(oracle_ceiling_grid(args.config, s, args.n_tokens, overrides=overrides))
        except ValueError as e:
            raise SystemExit(
                f"oracle ceiling failed for seed {s}: {e}\n"
                f"(an over-large --n-superparent can push the config's true L0 past the oracle's "
                f"positive-projection budget; use a realistic confound-powering count.)")
    agg = aggregate_ceilings(grids)
    report = {"config": args.config, "overrides": overrides, "seeds": args.seeds,
              "n_tokens": args.n_tokens, "ceilings": agg}
    if args.out is not None:
        args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(_markdown_table(agg))
    print("\n(clean-oracle ceilings -- what a perfect dictionary achieves; choose thresholds in analysis)")


if __name__ == "__main__":
    main()
