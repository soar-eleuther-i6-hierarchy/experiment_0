"""Score a set of trained toy-SAE checkpoints end to end and aggregate across seeds.

For each seed the driver runs the three scorers on one checkpoint:

  * ``run_recovery``   — Hungarian feature matching + realized-L0 / architecture provenance;
  * ``run_retrieval``  — the property-vs-rest relationship-retrieval AUROC grid (each
                          generative class vs the rest, scored by the 10 firewalled detectors);
  * ``run_absorption`` — the absorption / decoder-multiplicity / composition decomposition.

Every per-seed report is written to ``<out>/`` as JSON, then ``aggregate_seeds`` combines the
retrieval reports into across-seed AUROC point estimates with Student-t confidence intervals.
A compact summary is printed for quick reading; the JSON files hold the full detail.

Usage (from the experiment_0 directory)::

    python -m scoring.run_scoring \
        --ckpt-glob "/path/checkpoints/seed{seed}/full-matryoshka-k13-x4-pow-sp5" \
        --seeds 0 1 2 3 4 --out outputs_local/final
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from scoring.trained.absorption import run_absorption
from scoring.trained.recovery import run_recovery
from scoring.core.registry import DETECTORS, LATENT_COLUMNS
from scoring.core.grid import aggregate_seeds
from scoring.trained.retrieval import run_retrieval
from toygen import labels

# The property-vs-rest grid columns: the generative classes plus the latent-side ones.
_GRID_COLS = tuple(labels.LABELS) + LATENT_COLUMNS


def score_seeds(ckpt_glob: str, seeds: list[int], out: Path, n_tokens: int) -> dict:
    """Run all three scorers on every seed's checkpoint, save each report, and return the
    per-seed reports plus the across-seed aggregate. ``ckpt_glob`` must contain ``{seed}``."""
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    recovery, retrieval, absorption = {}, {}, {}
    for s in seeds:
        ckpt = ckpt_glob.format(seed=s)
        print(f"[seed {s}] scoring {ckpt} ...", flush=True)
        recovery[s] = run_recovery(ckpt, n_tokens=n_tokens)
        (out / f"recovery_seed{s}.json").write_text(json.dumps(recovery[s], indent=2))
        retrieval[s] = run_retrieval(ckpt, n_tokens=n_tokens)
        (out / f"retrieval_seed{s}.json").write_text(json.dumps(retrieval[s], indent=2))
        absorption[s] = run_absorption(ckpt, n_tokens=n_tokens)
        (out / f"absorption_seed{s}.json").write_text(json.dumps(absorption[s], indent=2))
        print(f"[seed {s}] done ({time.time() - t0:.0f}s elapsed)", flush=True)

    aggregate = aggregate_seeds([retrieval[s] for s in seeds])
    (out / "aggregate_seeds.json").write_text(json.dumps(aggregate, indent=2))
    return {"recovery": recovery, "retrieval": retrieval, "absorption": absorption,
            "aggregate": aggregate, "seeds": seeds}


def print_summary(res: dict) -> None:
    """Print a compact human-readable summary of the scored seeds."""
    seeds, agg = res["seeds"], res["aggregate"]
    print("\n" + "=" * 88)
    print(f"SCORING SUMMARY — {len(seeds)} seeds")
    print("=" * 88)

    print("\n[1] per-seed provenance")
    for s in seeds:
        rec, ret = res["recovery"][s], res["retrieval"][s]
        print(f"  seed{s}: L0={rec.get('realized_l0'):.2f} arch={rec.get('architecture')} "
              f"n_recovered={ret.get('n_recovered')}")

    print("\n[2] across-seed property-vs-rest AUROC (mean; each column vs the rest)")
    print("  " + " " * 20 + "".join(f"{c[:7]:>8}" for c in _GRID_COLS))
    for det in DETECTORS:
        row = agg.get(det, {})
        cells = "".join(
            (f"{m:>8.2f}" if isinstance((m := row.get(c, {}).get("mean")), float) and m == m
             else f"{'--':>8}")
            for c in _GRID_COLS)
        print(f"  {det:20s}{cells}")

    print("\n[3] absorption decomposition + split readout per seed")
    for s in seeds:
        ab = res["absorption"][s]
        sp = res["retrieval"][s].get("split_readout", {})
        print(f"  seed{s}: counts={ab['counts']} by_relation={ab['absorbed_by_relation']} "
              f"n_split={sp.get('n_split', '--')}")

    print("\n[4] is_a deployment cascade per seed (greedy both-tails Boolean rule)")
    for s in seeds:
        isa = res["retrieval"][s].get("cascade", {}).get("is_a", {})
        if "skipped" in isa or not isa:
            print(f"  seed{s}: {isa.get('skipped', 'no cascade')}")
            continue
        hn = isa.get("hard_negative", {})
        trivial = "  (no filter found — base-rate only)" if isa.get("trivial") else ""
        print(f"  seed{s}: prec={isa['final_precision']:.2f} survival={isa['final_survival']:.2f} "
              f"enrich={isa['enrichment']:.1f}x hard_neg_prec={hn.get('precision', float('nan')):.2f} "
              f"n_metrics={isa['n_metrics']}{trivial}")
        print(f"          rule: {isa['final_rule']}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Score trained toy-SAE checkpoints across seeds.")
    ap.add_argument("--ckpt-glob", required=True,
                    help="checkpoint path containing a literal {seed} placeholder")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--out", type=Path, default=Path("outputs_local/final"))
    ap.add_argument("--n-tokens", type=int, default=200_000)
    args = ap.parse_args()

    res = score_seeds(args.ckpt_glob, args.seeds, args.out, args.n_tokens)
    print_summary(res)
    print(f"\nreports written to {args.out}")


if __name__ == "__main__":
    main()
