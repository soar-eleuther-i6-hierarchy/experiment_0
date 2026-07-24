"""
Stage 02 - Run the five Exp 0 metrics on the cached statistics.

Loads outputs/exp0_stats.pt (from cache_stats.py) and, for every adjacent
block pair, treats the metrics as competing measurements of the same candidate
edge set:

    1. Activation coverage (reverse R / forward F / joint-child J)  -> edge set
    2. Reconstruction condition (Tree-SAE)  -> does the edge improve recon?
    3. Sibling redundancy                   -> are the children real refinements?
    4. Out-degree distribution              -> superparents / poly-parenting
    5. Token-frequency-controlled coverage  -> does the edge survive on rare tokens?

The edge SET comes from metric 1 (reverse coverage >= EDGE_TAU, both endpoints
firing >= MIN_FIRE_COUNT) - the crude warm-up criterion. Metrics 2-5 then grade
those edges. Writes a JSON blob (machine-readable) and a Markdown digest.

Run:
    cd experiment_0 && python3 run_metrics.py
Output:
    outputs/metrics_report.json
    outputs/metrics_report.md
"""

from __future__ import annotations

import json
import math

import torch

import config as C
import sae_utils as U
from metrics import (
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


def _f(x) -> float:
    return float(x)


def json_safe(obj):
    """Recursively replace NaN/inf floats with None so json.dumps emits valid
    JSON (bare NaN tokens break strict parsers, e.g. JS on the Pages site).
    Applied at serialization time only — markdown rendering happens before."""
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    if isinstance(obj, dict):
        return {k: json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [json_safe(v) for v in obj]
    return obj


def _nanmean(t: torch.Tensor) -> float:
    v = t[~torch.isnan(t)]
    return float(v.mean()) if v.numel() else float("nan")


def _clip(s: str, n: int = 40) -> str:
    s = (s or "").strip()
    return s if len(s) <= n else s[: n - 1] + "…"


def analyse_pair(stats, p_blk, c_blk, labels=None, legacy_guards=False):
    """One block pair through every metric.

    legacy_guards=True reproduces the pre-tranche edge criterion (no
    min_joint) — used only for regression checks against old runs.
    """
    labels = labels or {}
    key = f"{p_blk}->{c_blk}"
    fire = stats["fire_count"].double()
    total = int(stats["total_tokens"])
    p0, p1 = C.BLOCK_RANGES[p_blk]
    c0, c1 = C.BLOCK_RANGES[c_blk]
    fire_p = fire[p0:p1]
    fire_c = fire[c0:c1]

    cofire = stats["cofire"][key].double()

    # --- Metric 1: coverage + edge set --------------------------------------
    # R = P(parent | child) ("reverse"), F = P(child | parent) ("forward");
    # landscape names: Cont(c,p) and Share_freq(c,p).
    R, F = coverage_legs(cofire, fire_p, fire_c)
    min_joint = 0 if legacy_guards else C.MIN_JOINT
    edge_mask = keep_edges(
        R, fire_p, fire_c, C.EDGE_TAU, C.MIN_FIRE_COUNT,
        cofire=cofire, min_joint=min_joint,
    )
    # edges the old criterion kept but the joint-support guard kills (reported)
    edge_mask_nojoint = keep_edges(R, fire_p, fire_c, C.EDGE_TAU, C.MIN_FIRE_COUNT)
    n_dropped_min_joint = int((edge_mask_nojoint & ~edge_mask).sum())
    joint_upper = joint_child_coverage_upper(F, edge_mask)  # [P]
    n_edges = int(edge_mask.sum())

    # --- Independence null (PMI / Dev) --------------------------------------
    null = independence_scores(cofire, fire_p, fire_c, total, C.MIN_JOINT)
    pmi = null["pmi"]
    edge_pmi = pmi[edge_mask]
    edge_pmi_ok = edge_pmi[~torch.isnan(edge_pmi)]
    # PMI ~ 0 = co-firing at chance level given base rates (C-freq artifact)
    n_chance_level = int((edge_pmi_ok < 0.5).sum())

    # --- Energy share + exact joint-child (schema v2 caches) ----------------
    joint_child = None
    if "union_count" in stats:
        e_cof = stats["energy_cofire"][key].double()
        e_tot = stats["energy_total"][key].double()
        rs = r_supp(stats["union_count"][key].double(), fire_p)
        rm = r_mass(stats["union_energy"][key].double(), e_tot)
        S = share_energy(e_cof, e_tot)
        has_edges = edge_mask.any(dim=1)
        max_share = S.max(dim=1).values
        joint_child = {                      # JSON-safe scalar summaries only
            "r_supp_mean": _nanmean(rs[has_edges]) if n_edges else float("nan"),
            "r_mass_mean": _nanmean(rm[has_edges]) if n_edges else float("nan"),
            # rename/duplicate candidates: one child holds ~all parent energy
            "n_share_energy_ge_09": int((max_share[has_edges] >= 0.9).sum()) if n_edges else 0,
        }

    # --- Metric 4: out-degree / superparents --------------------------------
    deg = degree_stats(edge_mask)
    superparents = find_superparents(
        edge_mask, fire_p, total, C.SUPERPARENT_OUTDEG_FRAC, C.SUPERPARENT_FIRE_FRAC
    )

    # --- Metric 2: reconstruction condition ---------------------------------
    recon = edge_reconstruction_condition(
        stats["err_sum_c"][c_blk].double(),
        stats["g_parent_sum"][key].double(),
        stats["g_child_sum"][c_blk].double(),
        C.RECON_REL_GAIN_MIN,
    )
    recon_pass = recon["passes"] & edge_mask
    n_recon_pass = int(recon_pass.sum())

    # --- Metric 5: token-frequency-controlled coverage ----------------------
    fcov = frequency_controlled_coverage(
        stats["cofire_by_bucket"][key].double(),
        stats["fire_c_by_bucket"][c_blk].double(),
        edge_mask,
    )
    survival = fcov["survival"]                               # [P, C] nan off-edge/untestable
    surv_vals = survival[~torch.isnan(survival)]
    n_freq_driven = int((survival < C.FREQ_SURVIVAL_MIN).sum())
    n_testable = int(surv_vals.numel())

    # --- Metric 3: sibling redundancy ---------------------------------------
    # GLOBAL Jaccard only — confounded for Matryoshka (landscape Rev. 2.1): it
    # scores co-firing anywhere, not disjointness within the parent's support.
    # This is a cheap diagnostic, NOT the splitting verdict; the corrected
    # parent-conditioned Jaccard is in the stage-03 second pass. Reported so
    # both numbers are auditable, but must not be read as "flagged splitting".
    sib = {}
    sib_summary = None
    if c_blk in C.SIBLING_BLOCKS:
        sib = sibling_redundancy(edge_mask, stats["within_cofire"][c_blk].double(), fire_c)
        if sib:
            reds = [v["redundancy"] for v in sib.values()]
            sib_summary = {
                "method": "global_jaccard_confounded",   # NOT the verdict — see second_pass
                "n_parents_scored": len(reds),
                "mean_redundancy": _f(sum(reds) / len(reds)),
                "n_over_global_threshold": int(sum(r >= C.SIBLING_REDUNDANCY_FLAG for r in reds)),
            }

    # --- top edges (by reverse coverage) with every metric attached ---------
    top_edges = []
    if n_edges:
        pidx, cidx = torch.nonzero(edge_mask, as_tuple=True)
        order = torch.argsort(R[pidx, cidx], descending=True)[:15]
        for j in order.tolist():
            pi, ci = int(pidx[j]), int(cidx[j])
            gp, gc = p0 + pi, c0 + ci
            s = survival[pi, ci]
            pm = pmi[pi, ci]
            top_edges.append(
                {
                    "parent": gp,
                    "child": gc,
                    "reverse_cov": _f(R[pi, ci]),
                    "forward_cov": _f(F[pi, ci]),
                    "pmi": None if torch.isnan(pm) else _f(pm),
                    "recon_parent_gain": _f(recon["parent_gain"][pi, ci]),
                    "recon_child_gain": _f(recon["child_gain"][ci]),
                    "recon_pass": bool(recon["passes"][pi, ci]),
                    "freq_survival": None if torch.isnan(s) else _f(s),
                    "parent_redundancy": (sib.get(pi, {}) or {}).get("redundancy"),
                    "parent_label": C.feature_label(gp, labels),
                    "child_label": C.feature_label(gc, labels),
                    "parent_url": C.npedia_url(gp),
                    "child_url": C.npedia_url(gc),
                }
            )

    return {
        "pair": key,
        "n_candidate_edges": n_edges,
        # support-guard accounting (never silently drop — landscape Rev. 2)
        "n_dropped_min_joint": n_dropped_min_joint,
        "n_pairs_below_min_joint": null["n_excluded"],
        "independence_null": {
            "mean_edge_pmi": _nanmean(edge_pmi_ok) if n_edges else float("nan"),
            "n_chance_level": n_chance_level,           # edges with PMI < 0.5
            "frac_chance_level": _f(n_chance_level / n_edges) if n_edges else 0.0,
        },
        "joint_child": joint_child,
        "degree": {
            "n_parents_with_children": deg["n_parents_with_children"],
            "n_children_with_parent": deg["n_children_with_parent"],
            "n_multi_parented": deg["n_multi_parented"],
            "poly_frac": _f(deg["poly_frac"]),
            "top1_edge_share": _f(deg["top1_edge_share"]),
            "outdeg_gini": _f(deg["outdeg_gini"]),
            "max_outdeg": int(deg["outdeg"].max()) if n_edges else 0,
        },
        "joint_child_cov_mean": _nanmean(joint_upper[edge_mask.any(dim=1)]) if n_edges else float("nan"),
        "reconstruction": {
            "n_pass": n_recon_pass,
            "frac_pass": _f(n_recon_pass / n_edges) if n_edges else 0.0,
        },
        "freq_control": {
            "n_testable": n_testable,
            "mean_survival": _nanmean(surv_vals) if n_testable else float("nan"),
            "n_freq_driven": n_freq_driven,
            "frac_freq_driven": _f(n_freq_driven / n_testable) if n_testable else 0.0,
        },
        "sibling_redundancy": sib_summary,
        "n_superparents": len(superparents),                      # outdeg-only flag (§C.4)
        "n_superparents_strict": sum(sp["strict"] for sp in superparents),  # old AND gate
        "superparents": [
            {
                **sp,
                "parent_global": C.BLOCK_RANGES[p_blk][0] + sp["parent_local"],
                "label": C.feature_label(C.BLOCK_RANGES[p_blk][0] + sp["parent_local"], labels),
                "url": C.npedia_url(C.BLOCK_RANGES[p_blk][0] + sp["parent_local"]),
            }
            for sp in superparents[:10]
        ],
        "top_edges": top_edges,
    }


def to_markdown(report) -> str:
    # Jekyll renders this file to .html on GitHub Pages; the raw HTML passes
    # through and gives the report the same back-to-index button as the
    # generated dashboards, so navigation is consistent across the site.
    L = [C.BACK_LINK_HTML, "", "# Exp 0 - metrics report", ""]
    L.append(C.scope_line(report["total_tokens"], n_docs=report["config"].get("n_docs")))
    L.append("")
    for pr in report["pairs"]:
        d = pr["degree"]
        L.append(f"## Block pair {pr['pair']}  -  {pr['n_candidate_edges']} candidate edges")
        L.append("")
        L.append(f"- **Out-degree**: {d['n_parents_with_children']} parents, "
                 f"{d['n_children_with_parent']} children, {d['n_multi_parented']} multi-parented "
                 f"(PolyFrac {100 * d['poly_frac']:.1f}%); "
                 f"top-1 parent holds {100 * d['top1_edge_share']:.1f}% of edges, "
                 f"Gini {d['outdeg_gini']:.3f}, max out-degree {d['max_outdeg']}.")
        L.append(f"- **Superparents** (out-degree flag): {pr['n_superparents']} "
                 f"({pr['n_superparents_strict']} also pass the old fire-rate AND-gate)"
                 + (f" — e.g. feature {pr['superparents'][0]['parent_global']} "
                    f"_{_clip(pr['superparents'][0].get('label'), 60)}_: "
                    f"{pr['superparents'][0]['outdeg']} children, fires on "
                    f"{100 * pr['superparents'][0]['fire_frac']:.1f}% of tokens" if pr["superparents"] else ""))
        nl = pr["independence_null"]
        L.append(f"- **Independence null**: mean edge PMI {nl['mean_edge_pmi']:.2f}; "
                 f"{nl['n_chance_level']} edges ({100 * nl['frac_chance_level']:.1f}%) at chance "
                 f"level (PMI < 0.5). {pr['n_dropped_min_joint']} edges dropped by the "
                 f"joint-support guard (n_joint < {C.MIN_JOINT}).")
        rec = pr["reconstruction"]
        L.append(f"- **Recon-ablation contribution filter** (Tree-SAE-inspired baseline): "
                 f"{rec['n_pass']}/{pr['n_candidate_edges']} edges pass "
                 f"({100 * rec['frac_pass']:.1f}%).")
        fq = pr["freq_control"]
        ms = fq["mean_survival"]
        L.append(f"- **Frequency control**: mean survival {ms:.3f} over {fq['n_testable']} testable edges; "
                 f"{fq['n_freq_driven']} ({100 * fq['frac_freq_driven']:.1f}%) are frequency-driven "
                 f"(survival < {C.FREQ_SURVIVAL_MIN}).")
        sb = pr["sibling_redundancy"]
        if sb:
            L.append(f"- **Sibling redundancy** (global Jaccard — confounded proxy, not the "
                     f"splitting verdict; the Rev. 2.1 parent-conditioned version is in the "
                     f"stage-03 second pass): mean {sb['mean_redundancy']:.3f} over "
                     f"{sb['n_parents_scored']} parents; {sb['n_over_global_threshold']} over the "
                     f"{C.SIBLING_REDUNDANCY_FLAG} global threshold.")
        else:
            L.append("- **Sibling redundancy**: n/a (child block not in SIBLING_BLOCKS).")
        jc2 = pr.get("joint_child")
        if jc2:
            L.append(f"- **Joint-child (exact union, parents with edges)**: "
                     f"R_supp mean {jc2['r_supp_mean']:.3f}, R_mass mean {jc2['r_mass_mean']:.3f}; "
                     f"{jc2['n_share_energy_ge_09']} parents with one child holding >=90% of their "
                     f"energy (rename candidates).")
        jc = pr["joint_child_cov_mean"]
        L.append(f"- **Joint-child coverage** (min(1, ΣF) upper bound — saturates when children "
                 f"co-fire, kept only for contrast with the exact union): "
                 + (f"{jc:.3f}" if not math.isnan(jc) else "n/a") + ".")
        if pr["top_edges"]:
            L.append("")
            L.append("| parent -> child | R | F | PMI | recon P/C gain | recon? | surv | sib | parent label | child label |")
            L.append("|---|---|---|---|---|---|---|---|---|---|")
            for e in pr["top_edges"][:8]:
                surv = "-" if e["freq_survival"] is None else f"{e['freq_survival']:.2f}"
                red = "-" if e["parent_redundancy"] is None else f"{e['parent_redundancy']:.2f}"
                pm = "-" if e.get("pmi") is None else f"{e['pmi']:.2f}"
                L.append(f"| {e['parent']} -> {e['child']} | {e['reverse_cov']:.2f} | "
                         f"{e['forward_cov']:.2f} | {pm} | {e['recon_parent_gain']:.2f}/{e['recon_child_gain']:.2f} | "
                         f"{'Y' if e['recon_pass'] else 'n'} | {surv} | {red} | "
                         f"{_clip(e.get('parent_label'))} | {_clip(e.get('child_label'))} |")
        L.append("")
    return "\n".join(L)


def main():
    if not C.EXP0_STATS_PATH.exists():
        raise SystemExit(f"[02] missing {C.EXP0_STATS_PATH} - run cache_stats.py first")
    print(f"[02] loading {C.EXP0_STATS_PATH}")
    stats = torch.load(C.EXP0_STATS_PATH, weights_only=False)
    pairs = stats["pairs"]

    labels = C.load_feature_labels()
    if not labels:
        print("[02] note: outputs/feature_labels.json not found - run fetch_labels.py "
              "to include feature descriptions")
    per_pair = [analyse_pair(stats, p, c, labels) for (p, c) in pairs]
    report = {
        "config": stats["config"],
        "total_tokens": int(stats["total_tokens"]),
        "pairs": per_pair,
        "block_ranges": C.BLOCK_RANGES,
    }

    C.METRICS_MD_PATH.write_text(to_markdown(report))          # md handles NaN itself
    C.METRICS_JSON_PATH.write_text(json.dumps(json_safe(report), indent=2))
    print(f"[02] wrote {C.METRICS_JSON_PATH}")
    print(f"[02] wrote {C.METRICS_MD_PATH}")
    for pr in per_pair:
        print(f"[02]   {pr['pair']}: {pr['n_candidate_edges']} edges, "
              f"{pr['reconstruction']['n_pass']} recon-pass, "
              f"{pr['n_superparents']} superparents")


if __name__ == "__main__":
    main()
