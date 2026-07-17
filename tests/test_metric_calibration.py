"""
Calibrate the five Exp 0 metrics on the synthetic ground-truth toy.

For each metric we know, by construction, which edges it SHOULD keep and which
pathology it SHOULD catch (see tests/toy_world.py). We run the production metric
functions with the production thresholds from config.py and check:

    Metric 1  coverage          recovers 100% of the genuine tree edges.
    Metric 2  reconstruction    rejects the superparent's edges, keeps genuine.
    Metric 3  sibling redundancy flags the feature-split parent, spares healthy.
    Metric 4  out-degree        identifies the superparent, spares genuine.
    Metric 5  frequency control rejects the frequency-coincidence edge, keeps
                                genuine.

Each metric gets a pass/fail plus a decision MARGIN (how decisively it separated
the two classes); the scorecard is ranked by margin. Run directly for the
printed report, or under pytest for the assertions.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

# allow `python3 tests/test_metric_calibration.py` from the experiment_0 root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config as C  # noqa: E402
from metrics import (  # noqa: E402
    coverage_legs,
    degree_stats,
    edge_reconstruction_condition,
    find_superparents,
    frequency_controlled_coverage,
    keep_edges,
    sibling_redundancy,
)
from tests.toy_world import build_world  # noqa: E402


def _run_metrics(stats):
    """Mirror run_metrics.analyse_pair's calls, on the toy's single block pair."""
    fire_p = stats["fire_p"]
    fire_c = stats["fire_c"]
    cofire = stats["cofire"]

    R, F = coverage_legs(cofire, fire_p, fire_c)
    edge_mask = keep_edges(R, fire_p, fire_c, C.EDGE_TAU, C.MIN_FIRE_COUNT)

    recon = edge_reconstruction_condition(
        stats["err_sum_c"], stats["g_parent_sum"], stats["g_child_sum"], C.RECON_REL_GAIN_MIN
    )
    fcov = frequency_controlled_coverage(
        stats["cofire_by_bucket"], stats["fire_c_by_bucket"], edge_mask
    )
    deg = degree_stats(edge_mask)
    superparents = find_superparents(
        edge_mask, fire_p, stats["total_tokens"],
        C.SUPERPARENT_OUTDEG_FRAC, C.SUPERPARENT_FIRE_FRAC,
    )
    sib = sibling_redundancy(edge_mask, stats["within_cofire"], fire_c)
    return {
        "R": R, "F": F, "edge_mask": edge_mask,
        "recon": recon, "fcov": fcov, "deg": deg,
        "superparents": superparents, "sib": sib,
    }


def _score(stats, labels, m) -> list[dict]:
    """One scorecard row per metric: pass/fail on its designated job + margin."""
    edge_mask = m["edge_mask"]
    kept = {(int(p), int(c)) for p, c in torch.nonzero(edge_mask).tolist()}
    genuine = labels.genuine
    rows: list[dict] = []

    # --- Metric 1: coverage recovers the genuine tree -----------------------
    recovered = genuine & kept
    rows.append({
        "metric": "1. coverage (edge set)",
        "job": "recover genuine tree edges",
        "pass": recovered == genuine,
        "detail": f"{len(recovered)}/{len(genuine)} genuine edges kept; "
                  f"edge set also holds {len(kept) - len(recovered)} non-genuine "
                  f"(that is what metrics 2-5 must prune)",
        "margin": len(recovered) / max(len(genuine), 1),
    })

    # --- Metric 2: reconstruction rejects superparent, keeps genuine --------
    passes = m["recon"]["passes"] & edge_mask
    pgain = m["recon"]["parent_gain"]
    sp_edges = [(p, c) for (p, c) in kept if p in labels.superparent_parents]
    sp_rejected = sum(not bool(passes[p, c]) for p, c in sp_edges)
    gen_kept = sum(bool(passes[p, c]) for p, c in (genuine & kept))
    sp_max_gain = max((float(pgain[p, c]) for p, c in sp_edges), default=0.0)
    gen_min_gain = min((float(pgain[p, c]) for p, c in (genuine & kept)), default=0.0)
    rows.append({
        "metric": "2. reconstruction",
        "job": "reject superparent edges, keep genuine",
        "pass": sp_rejected == len(sp_edges) and gen_kept == len(genuine & kept),
        "detail": f"{sp_rejected}/{len(sp_edges)} superparent edges rejected, "
                  f"{gen_kept}/{len(genuine & kept)} genuine kept "
                  f"(parent-gain: genuine>={gen_min_gain:.2f}, superparent<={sp_max_gain:.4f}, "
                  f"thr={C.RECON_REL_GAIN_MIN})",
        "margin": gen_min_gain / max(sp_max_gain, 1e-9),
    })

    # --- Metric 3: sibling redundancy flags the split parent ----------------
    sib = m["sib"]
    split_red = sib.get(labels.split_parents and next(iter(labels.split_parents)), {}).get("redundancy", 0.0)
    healthy_reds = [v["redundancy"] for p, v in sib.items() if p not in labels.split_parents]
    healthy_max = max(healthy_reds, default=0.0)
    flagged = split_red >= C.SIBLING_REDUNDANCY_FLAG
    healthy_ok = healthy_max < C.SIBLING_REDUNDANCY_FLAG
    rows.append({
        "metric": "3. sibling redundancy",
        "job": "flag feature-split parent, spare healthy",
        "pass": flagged and healthy_ok,
        "detail": f"split parent redundancy={split_red:.2f} "
                  f"({'flagged' if flagged else 'MISSED'}); "
                  f"healthy parents max={healthy_max:.2f} (thr={C.SIBLING_REDUNDANCY_FLAG})",
        "margin": split_red / max(healthy_max, 1e-9),
    })

    # --- Metric 4: out-degree finds the superparent -------------------------
    found = {sp["parent_local"] for sp in m["superparents"]}
    rows.append({
        "metric": "4. out-degree / superparent",
        "job": "identify superparent, spare genuine parents",
        "pass": found == labels.superparent_parents,
        "detail": f"detected superparents {sorted(found)} "
                  f"(truth {sorted(labels.superparent_parents)}); "
                  f"Gini={m['deg']['outdeg_gini']:.3f}, "
                  f"top-1 share={100 * m['deg']['top1_edge_share']:.0f}%",
        "margin": 1.0 if found == labels.superparent_parents else 0.0,
    })

    # --- Metric 5: frequency control rejects the coincidence edge -----------
    survival = m["fcov"]["survival"]
    freq_surv = [float(survival[p, c]) for (p, c) in labels.freq_edges if (p, c) in kept]
    freq_rejected = sum(s < C.FREQ_SURVIVAL_MIN for s in freq_surv)
    gen_surv = [float(survival[p, c]) for (p, c) in (genuine & kept)
                if not torch.isnan(survival[p, c])]
    gen_kept_freq = sum(s >= C.FREQ_SURVIVAL_MIN for s in gen_surv)
    rows.append({
        "metric": "5. frequency control",
        "job": "reject frequency-coincidence edge, keep genuine",
        "pass": freq_rejected == len(freq_surv) and gen_kept_freq == len(gen_surv),
        "detail": f"{freq_rejected}/{len(freq_surv)} freq edges rejected "
                  f"(survival={[round(s, 2) for s in freq_surv]}); "
                  f"{gen_kept_freq}/{len(gen_surv)} genuine survive "
                  f"(min genuine survival={min(gen_surv, default=float('nan')):.2f}, "
                  f"thr={C.FREQ_SURVIVAL_MIN})",
        "margin": (min(gen_surv, default=0.0) / max(max(freq_surv, default=1e-9), 1e-9)),
    })
    return rows


def calibrate(seed: int = 0):
    stats, labels = build_world(seed=seed)
    m = _run_metrics(stats)
    rows = _score(stats, labels, m)
    return stats, labels, rows


def _render(rows) -> str:
    ranked = sorted(rows, key=lambda r: (-int(r["pass"]), -r["margin"]))
    L = ["# Exp 0 - metric calibration on synthetic ground truth", ""]
    L.append("Each metric is graded on the pathology it is meant to catch, using "
             "the production thresholds in `config.py`. Margin = how decisively the "
             "metric separated the two classes (higher is better).")
    L.append("")
    L.append("| rank | metric | job | verdict | margin | detail |")
    L.append("|---|---|---|---|---|---|")
    for i, r in enumerate(ranked, 1):
        v = "PASS" if r["pass"] else "**FAIL**"
        mg = ">1000x" if r["margin"] >= 1000 else f"{r['margin']:.1f}x"
        L.append(f"| {i} | {r['metric']} | {r['job']} | {v} | {mg} | {r['detail']} |")
    L.append("")
    n_pass = sum(r["pass"] for r in rows)
    L.append(f"**{n_pass}/{len(rows)} metrics calibrated.** "
             + ("All five recover the genuine tree and reject their injected "
                "pathology on this toy." if n_pass == len(rows)
                else "Some metrics did not separate cleanly - see FAIL rows."))
    return "\n".join(L)


def main():
    _, _, rows = calibrate()
    md = _render(rows)
    print("\n" + md + "\n")
    out_md = C.OUT_DIR / "toy_calibration.md"
    out_json = C.OUT_DIR / "toy_calibration.json"
    out_md.write_text(md)
    out_json.write_text(json.dumps(
        [{k: (v if not isinstance(v, float) or v != float("inf") else "inf")
          for k, v in r.items()} for r in rows], indent=2))
    print(f"[calib] wrote {out_md}")
    print(f"[calib] wrote {out_json}")
    if not all(r["pass"] for r in rows):
        raise SystemExit("[calib] FAILED: not all metrics recovered ground truth")


# ---- pytest entry points ---------------------------------------------------
def test_all_metrics_calibrate():
    _, _, rows = calibrate()
    failed = [r["metric"] for r in rows if not r["pass"]]
    assert not failed, f"metrics failed calibration: {failed}"


def test_coverage_recovers_tree():
    _, _, rows = calibrate()
    assert rows[0]["pass"], rows[0]["detail"]


def test_reconstruction_rejects_superparent():
    _, _, rows = calibrate()
    assert rows[1]["pass"], rows[1]["detail"]


if __name__ == "__main__":
    main()
