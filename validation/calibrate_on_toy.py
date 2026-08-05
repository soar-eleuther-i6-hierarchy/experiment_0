"""
Calibrate the five hierarchy metrics on the synthetic ground-truth toy.

For each metric we know, by construction, which edges it SHOULD keep and which
pathology it SHOULD catch (see validation/toy_world.py). We run the production metric
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

# allow `python3 validation/calibrate_on_toy.py` from the experiment_0 root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config as C  # noqa: E402
from metrics import (  # noqa: E402
    coverage_legs,
    degree_stats,
    edge_reconstruction_condition,
    find_superparents,
    frequency_controlled_coverage,
    independence_scores,
    joint_child_coverage_upper,
    keep_edges,
    r_mass,
    r_supp,
    share_energy,
    sibling_redundancy,
)
from metrics.coverage import joint_child_coverage_exact  # noqa: E402  (not in metrics.__all__)
from validation.toy_world import build_world  # noqa: E402

# One child holding >= this fraction of the parent's energy flags a feature split
# (matches run_metrics.py's n_share_energy_ge_09 gate).
SPLIT_SHARE_MIN = 0.9


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

    # --- metrics reading the schema-v2 energy / joint-child-union accumulators
    null = independence_scores(cofire, fire_p, fire_c, stats["total_tokens"], C.MIN_JOINT)
    joint_upper = joint_child_coverage_upper(F, edge_mask)
    rs = r_supp(stats["union_count"], fire_p)
    jce = joint_child_coverage_exact(stats["union_count"], fire_p)
    rm = r_mass(stats["union_energy"], stats["energy_total"])
    share = share_energy(stats["energy_cofire"], stats["energy_total"])
    return {
        "R": R, "F": F, "edge_mask": edge_mask,
        "recon": recon, "fcov": fcov, "deg": deg,
        "superparents": superparents, "sib": sib,
        "pmi": null["pmi"], "pmi_valid": null["valid"],
        "joint_upper": joint_upper, "r_supp": rs, "joint_exact": jce,
        "r_mass": rm, "share": share,
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

    sp_parent = next(iter(labels.superparent_parents))
    gen_parents = sorted({p for (p, _) in genuine})

    # --- Metric 6: independence null (PMI) ranks genuine above base rate -----
    pmi, pmi_valid = m["pmi"], m["pmi_valid"]
    gen_pmi = [float(pmi[p, c]) for (p, c) in genuine if bool(pmi_valid[p, c])]
    sp_pmi = [float(pmi[sp_parent, c]) for c in range(stats["C"])
              if bool(pmi_valid[sp_parent, c])]
    gen_min_pmi = min(gen_pmi, default=float("nan"))
    sp_max_pmi = max(sp_pmi, default=0.0)
    rows.append({
        "metric": "6. independence null (PMI)",
        "job": "rank genuine edges above base-rate/superparent co-firing",
        "pass": bool(gen_pmi) and bool(sp_pmi) and gen_min_pmi > sp_max_pmi,
        "detail": f"min genuine PMI={gen_min_pmi:.2f} > max superparent PMI={sp_max_pmi:.2f}; "
                  f"base-rate confound only - topical co-occurrence is not in this toy "
                  f"(needs a model-based null)",
        "margin": gen_min_pmi / max(sp_max_pmi, 1e-9),
    })

    # --- Metric 7: joint-child coverage (support) - genuine covered, not SP --
    rs, jce, ju = m["r_supp"], m["joint_exact"], m["joint_upper"]
    gen_rs = [float(rs[p]) for p in gen_parents]
    sp_rs = float(rs[sp_parent])
    exact_eq = bool(torch.allclose(rs, jce))
    upper_ok = bool((ju >= jce - 1e-9).all())
    rows.append({
        "metric": "7. joint-child coverage (support)",
        "job": "children cover a genuine parent's firing, not a superparent's",
        "pass": bool(gen_rs) and min(gen_rs) > sp_rs and exact_eq and upper_ok,
        "detail": f"genuine R_supp>={min(gen_rs, default=float('nan')):.2f} vs "
                  f"superparent={sp_rs:.2f}; r_supp==joint_child_coverage_exact: {exact_eq}; "
                  f"upper>=exact for all parents: {upper_ok} "
                  f"(covers r_supp, joint_child_coverage_exact, joint_child_coverage_upper)",
        "margin": min(gen_rs, default=0.0) / max(sp_rs, 1e-9),
    })

    # --- Metric 8: joint-child coverage (energy-weighted) -------------------
    rm = m["r_mass"]
    gen_rm = [float(rm[p]) for p in gen_parents]
    sp_rm = float(rm[sp_parent])
    rows.append({
        "metric": "8. joint-child coverage (energy)",
        "job": "energy-weighted child coverage separates genuine from superparent",
        "pass": bool(gen_rm) and min(gen_rm) > sp_rm,
        "detail": f"genuine R_mass>={min(gen_rm, default=float('nan')):.2f} vs "
                  f"superparent={sp_rm:.2f}",
        "margin": min(gen_rm, default=0.0) / max(sp_rm, 1e-9),
    })

    # --- Metric 9: energy concentration flags the feature-split parent ------
    split_parent = next(iter(labels.split_parents))
    max_share = m["share"].max(dim=1).values          # [P]
    split_share = float(max_share[split_parent])
    gen_share_max = max((float(max_share[p]) for p in gen_parents), default=0.0)
    rows.append({
        "metric": "9. energy concentration (share_energy)",
        "job": "flag feature-split parent (a child holds >=90% of its energy)",
        "pass": split_share >= SPLIT_SHARE_MIN and gen_share_max < split_share,
        "detail": f"split parent max child share={split_share:.2f} (thr={SPLIT_SHARE_MIN}); "
                  f"genuine parents max={gen_share_max:.2f}",
        "margin": split_share / max(gen_share_max, 1e-9),
    })
    return rows


def calibrate(seed: int = 0):
    stats, labels = build_world(seed=seed)
    m = _run_metrics(stats)
    rows = _score(stats, labels, m)
    return stats, labels, rows


def _render(rows) -> str:
    ranked = sorted(rows, key=lambda r: (-int(r["pass"]), -r["margin"]))
    # Site-wide page (no layer): one level down from the site root.
    L = [C.nav_html(depth=1, current="outputs/toy_calibration.html"), "",
         "# Exp 0 - metric calibration on synthetic ground truth", ""]
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
             + ("Every scorecard row recovers the genuine tree and rejects its "
                "injected pathology on this toy." if n_pass == len(rows)
                else "Some metrics did not separate cleanly - see FAIL rows."))
    L.append("")
    L.append("These rows cover **13/13 statistics-only metric functions**. The 4 "
             "per-token functions (`train_probe`, `sres_rank_check`, "
             "`negative_parent_composition`, `parent_conditioned_redundancy`) need "
             "per-token residuals/masks from the token cache and are calibrated in "
             "Tier 2, not on these reduced statistics.")
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
