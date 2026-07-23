"""
Visualise Exp 0's five-metric results as self-contained interactive HTML
(plotly.js embedded, works offline - same style as the example sankey script).

Reads the cached statistics in outputs/exp0_stats.pt, recomputes the full
per-block-pair distributions (the JSON report keeps only summaries, not the
distributions), then writes:

    outputs/metrics_dashboard.html    aggregate dashboard: filter funnel + distributions
    outputs/superparent_sankey.html   one superparent's fan-out to its children

    outputs/toy_calibration.html      (--calibration) metric scorecard + the
                                      genuine-vs-pathological separation each
                                      metric achieves on the synthetic toy
    outputs/qualitative_check.html    (--qualitative) surviving vs rejected real
                                      edges with Neuronpedia labels, colour-coded

Run:
    pip install plotly
    python visualize.py                  # real gemma-2-2b dashboards (needs exp0_stats.pt)
    python visualize.py --calibration    # synthetic-toy calibration (needs no cache)
    python visualize.py --qualitative    # real survivor-vs-rejected label table
                                         # (needs outputs/qualitative_check.json)
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import torch
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import config as C
from metrics import (
    coverage_legs,
    edge_reconstruction_condition,
    find_superparents,
    frequency_controlled_coverage,
    keep_edges,
    sibling_redundancy,
)

# House palette (same as the example sankey)
PURPLE, BLUE, TEAL = "#7C22CE", "#2196F3", "#0EA5A4"
PAIR_COLORS = [PURPLE, BLUE, TEAL]
GREEN, GREY, RED = "#22C55E", "#CBD5E1", "#EF4444"
INK = "#5A6B7B"
FONT = dict(family="DejaVu Sans Mono, Courier New, monospace", size=11, color=INK)


def binned_bar(values, color, name, rng=None, nbins=40, log_x=False):
    """Pre-bin an array with numpy and return a go.Bar, so millions of raw
    values are never embedded in the HTML (keeps the file small)."""
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return None
    x = np.log10(np.clip(values, 1, None)) if log_x else values
    counts, edges = np.histogram(x, bins=nbins, range=rng)
    centers = 0.5 * (edges[:-1] + edges[1:])
    if log_x:
        centers = 10 ** centers
    return go.Bar(x=centers, y=counts, marker_color=color, name=name,
                  legendgroup=name, showlegend=False, opacity=0.65)


def compute_pair(stats, p_blk, c_blk):
    """Everything the plots need for one block pair."""
    key = f"{p_blk}->{c_blk}"
    fire = stats["fire_count"].double()
    total = int(stats["total_tokens"])
    p0, p1 = C.BLOCK_RANGES[p_blk]
    c0, c1 = C.BLOCK_RANGES[c_blk]
    fire_p, fire_c = fire[p0:p1], fire[c0:c1]

    cofire = stats["cofire"][key].double()
    R, F = coverage_legs(cofire, fire_p, fire_c)
    edge_mask = keep_edges(R, fire_p, fire_c, C.EDGE_TAU, C.MIN_FIRE_COUNT)
    n_edges = int(edge_mask.sum())

    # Filter funnel: how many edges survive each metric.
    recon = edge_reconstruction_condition(
        stats["err_sum_c"][c_blk].double(),
        stats["g_parent_sum"][key].double(),
        stats["g_child_sum"][c_blk].double(),
        C.RECON_REL_GAIN_MIN,
    )
    n_recon = int((recon["passes"] & edge_mask).sum())

    fcov = frequency_controlled_coverage(
        stats["cofire_by_bucket"][key].double(),
        stats["fire_c_by_bucket"][c_blk].double(),
        edge_mask,
    )
    survival = fcov["survival"]
    surv_vals = survival[~torch.isnan(survival)]
    n_survive = int((survival >= C.FREQ_SURVIVAL_MIN).sum())

    outdeg = edge_mask.sum(dim=1)
    outdeg = outdeg[outdeg > 0].double()

    sib = {}
    if c_blk in C.SIBLING_BLOCKS:
        sib = sibling_redundancy(edge_mask, stats["within_cofire"][c_blk].double(), fire_c)
    redundancy = [v["redundancy"] for v in sib.values()]

    superparents = find_superparents(
        edge_mask, fire_p, total, C.SUPERPARENT_OUTDEG_FRAC, C.SUPERPARENT_FIRE_FRAC
    )

    return {
        "key": key,
        "n_edges": n_edges,
        "n_recon": n_recon,
        "n_survive": n_survive,
        "surv_vals": surv_vals.numpy(),
        "outdeg": outdeg.numpy(),
        "redundancy": redundancy,
        "superparents": superparents,
        "R": R,
        "F": F,
        "cofire": cofire,
        "edge_mask": edge_mask,
        "recon_pass": recon["passes"],
        "survival": survival,
    }


def build_dashboard(pairs_data):
    fig = make_subplots(
        rows=3,
        cols=2,
        subplot_titles=(
            "Filter funnel: edges surviving each metric (log scale)",
            "Share of candidate edges surviving each metric (%)",
            "Out-degree distribution (log-log)",
            "Frequency-survival distribution",
            "Sibling-redundancy distribution",
            "Superparents: child coverage x firing rate",
        ),
        vertical_spacing=0.10,
        horizontal_spacing=0.10,
    )

    stages = ["candidate", "improves recon", "survives freq"]
    for pd_ in pairs_data:
        col = PAIR_COLORS[pairs_data.index(pd_)]
        name = pd_["key"]
        # (1,1) count funnel
        fig.add_bar(
            x=stages, y=[pd_["n_edges"], pd_["n_recon"], pd_["n_survive"]],
            name=name, marker_color=col, legendgroup=name, row=1, col=1,
        )
        # (1,2) percentages
        e = max(pd_["n_edges"], 1)
        fig.add_bar(
            x=["improves recon %", "survives freq %"],
            y=[100 * pd_["n_recon"] / e, 100 * pd_["n_survive"] / e],
            name=name, marker_color=col, legendgroup=name, showlegend=False, row=1, col=2,
        )
        # (2,1) out-degree (log-binned)
        b = binned_bar(pd_["outdeg"], col, name, nbins=30, log_x=True)
        if b:
            fig.add_trace(b, row=2, col=1)
        # (2,2) frequency survival
        b = binned_bar(pd_["surv_vals"], col, name, rng=(0.0, 1.5), nbins=40)
        if b:
            fig.add_trace(b, row=2, col=2)
        # (3,1) sibling redundancy
        b = binned_bar(pd_["redundancy"], col, name, rng=(0.0, 1.0), nbins=30)
        if b:
            fig.add_trace(b, row=3, col=1)
        # (3,2) superparents
        sp = pd_["superparents"]
        if sp:
            fig.add_trace(
                go.Scatter(
                    x=[s["outdeg_frac"] for s in sp],
                    y=[s["fire_frac"] for s in sp],
                    mode="markers", marker=dict(color=col, size=12, line=dict(width=1, color="white")),
                    text=[f"feature {C.BLOCK_RANGES[int(name.split('->')[0])][0] + s['parent_local']}" for s in sp],
                    hovertemplate="%{text}<br>child coverage %{x:.2f}<br>firing rate %{y:.2f}<extra></extra>",
                    name=name, legendgroup=name, showlegend=False,
                ),
                row=3, col=2,
            )

    fig.add_vline(x=C.FREQ_SURVIVAL_MIN, line=dict(color="#EF4444", width=1, dash="dash"), row=2, col=2)
    fig.add_vline(x=C.SIBLING_REDUNDANCY_FLAG, line=dict(color="#EF4444", width=1, dash="dash"), row=3, col=1)

    fig.update_yaxes(type="log", row=1, col=1)
    fig.update_yaxes(type="log", row=2, col=1)  # number of parents
    fig.update_yaxes(type="log", row=2, col=2)
    fig.update_xaxes(type="log", title_text="children per parent (log)", row=2, col=1)
    fig.update_xaxes(title_text="survival", row=2, col=2)
    fig.update_xaxes(title_text="redundancy", row=3, col=1)
    fig.update_xaxes(title_text="child coverage (fraction of block)", row=3, col=2)
    fig.update_yaxes(title_text="firing rate", row=3, col=2)

    fig.update_layout(
        title=dict(text="Exp 0 - grading coverage edges with the five metrics "
                        f"({pairs_data[0]['n_edges']:,} -> {pairs_data[-1]['n_edges']:,} candidate edges)",
                   x=0.01, xanchor="left", font=dict(size=14, color=INK)),
        barmode="group", bargap=0.15, font=FONT,
        paper_bgcolor="white", plot_bgcolor="#FbFcFd",
        width=1200, height=1350, margin=dict(l=60, r=40, t=90, b=50),
        legend=dict(orientation="h", y=1.05, x=0.5, xanchor="center"),
    )
    return fig


def build_superparent_sankey(stats, pd_, p_blk, c_blk, top_n=25):
    """One superparent's flow to its top_n children; colour separates edges that
    survive (reconstruction AND frequency) from frequency-captured ones."""
    sp = pd_["superparents"]
    if not sp:
        return None
    parent_local = sp[0]["parent_local"]
    p0 = C.BLOCK_RANGES[p_blk][0]
    c0 = C.BLOCK_RANGES[c_blk][0]
    gp = p0 + parent_local

    kids = torch.nonzero(pd_["edge_mask"][parent_local]).flatten()
    cof = pd_["cofire"][parent_local, kids]
    order = torch.argsort(cof, descending=True)[:top_n]
    kids = kids[order]

    labels = [f"B{p_blk}:{gp} (fires {100 * sp[0]['fire_frac']:.0f}%)"]
    node_colors = [PURPLE]
    source, target, value, link_colors = [], [], [], []
    for i, ci in enumerate(kids.tolist(), start=1):
        gc = c0 + ci
        passes = bool(pd_["recon_pass"][parent_local, ci])
        s = pd_["survival"][parent_local, ci]
        survives = (not torch.isnan(s)) and float(s) >= C.FREQ_SURVIVAL_MIN
        real = passes and survives
        labels.append(f"B{c_blk}:{gc}")
        node_colors.append(GREEN if real else GREY)
        link_colors.append("rgba(34,197,94,0.55)" if real else "rgba(203,213,225,0.5)")
        source.append(0)
        target.append(i)
        value.append(float(pd_["cofire"][parent_local, ci]))

    fig = go.Figure(
        go.Sankey(
            arrangement="snap",
            node=dict(label=labels, color=node_colors, pad=18, thickness=14, line=dict(width=0)),
            link=dict(source=source, target=target, value=value, color=link_colors,
                      hovertemplate="%{source.label} -> %{target.label}<br>"
                                    "co-fire: %{value}<extra></extra>"),
        )
    )
    n_real = sum(1 for c in node_colors[1:] if c == GREEN)
    fig.update_layout(
        title=dict(text=f"Superparent B{p_blk}:{gp} -> top {len(kids)} children  "
                        f"(green = survives recon & frequency: {n_real}/{len(kids)})",
                   x=0.01, xanchor="left", font=dict(size=13, color=INK)),
        font=FONT, paper_bgcolor="white", plot_bgcolor="white",
        width=1150, height=680, margin=dict(l=50, r=90, t=70, b=40),
    )
    return fig


# ---------------------------------------------------------------------------
# Calibration dashboard (synthetic ground-truth toy from tests/)
# ---------------------------------------------------------------------------
def _calibration_data():
    """Run the five metrics on the toy and split each metric's per-edge score
    into the class it should KEEP (genuine) vs the class it should REJECT."""
    from tests.test_metric_calibration import _render, _run_metrics, _score
    from tests.toy_world import build_world

    stats, labels = build_world()
    m = _run_metrics(stats)
    rows = _score(stats, labels, m)

    edge_mask = m["edge_mask"]
    kept = {(int(p), int(c)) for p, c in torch.nonzero(edge_mask).tolist()}
    genuine = labels.genuine & kept
    sp_edges = [(p, c) for (p, c) in kept if p in labels.superparent_parents]
    freq_edges = [(p, c) for (p, c) in labels.freq_edges if (p, c) in kept]

    pgain = m["recon"]["parent_gain"]
    survival = m["fcov"]["survival"]
    outdeg = edge_mask.sum(dim=1)
    sib = m["sib"]
    genuine_parents = sorted({p for p, _ in genuine} - labels.superparent_parents)

    return {
        "rows": rows,
        "md": _render(rows),
        # (metric 2) reconstruction parent-gain, log scale
        "recon_keep": [float(pgain[p, c]) for p, c in genuine],
        "recon_reject": [float(pgain[p, c]) for p, c in sp_edges],
        "recon_thr": C.RECON_REL_GAIN_MIN,
        # (metric 5) frequency survival
        "freq_keep": [float(survival[p, c]) for p, c in genuine
                      if not torch.isnan(survival[p, c])],
        "freq_reject": [float(survival[p, c]) for p, c in freq_edges],
        "freq_thr": C.FREQ_SURVIVAL_MIN,
        # (metric 3) sibling redundancy
        "sib_keep": [sib[p]["redundancy"] for p in genuine_parents if p in sib],
        "sib_reject": [v["redundancy"] for p, v in sib.items() if p in labels.split_parents],
        "sib_thr": C.SIBLING_REDUNDANCY_FLAG,
        # (metric 4) out-degree
        "deg_keep": [int(outdeg[p]) for p in genuine_parents],
        "deg_reject": [int(outdeg[p]) for p in labels.superparent_parents],
        "deg_thr": C.SUPERPARENT_OUTDEG_FRAC * stats["C"],
    }


def _strip(fig, row, col, keep, reject, log_y=False, rng=None):
    """Two jittered point clouds: green = should-keep, red = should-reject."""
    jit = np.random.default_rng(0)
    for vals, color, name, xc in ((keep, GREEN, "genuine (keep)", 0),
                                  (reject, RED, "pathology (reject)", 1)):
        if not vals:
            continue
        x = xc + jit.uniform(-0.12, 0.12, size=len(vals))
        fig.add_trace(
            go.Scatter(
                x=x, y=vals, mode="markers",
                marker=dict(color=color, size=11, line=dict(width=1, color="white"), opacity=0.85),
                name=name, legendgroup=name, showlegend=(row == 2 and col == 1),
                hovertemplate=f"{name}<br>%{{y:.4g}}<extra></extra>",
            ),
            row=row, col=col,
        )
    fig.update_xaxes(tickvals=[0, 1], ticktext=["genuine", "pathology"],
                     range=[-0.5, 1.5], row=row, col=col)
    if log_y:
        fig.update_yaxes(type="log", row=row, col=col)
    if rng:
        fig.update_yaxes(range=rng, row=row, col=col)


def build_calibration_dashboard(data):
    ranked = sorted(data["rows"], key=lambda r: (-int(r["pass"]), -r["margin"]))
    fig = make_subplots(
        rows=3, cols=2,
        specs=[[{"type": "table", "colspan": 2}, None], [{}, {}], [{}, {}]],
        subplot_titles=(
            "",
            "Metric 2 - reconstruction parent-gain (log)",
            "Metric 5 - frequency survival",
            "Metric 3 - sibling redundancy",
            "Metric 4 - out-degree (children per parent)",
        ),
        vertical_spacing=0.09, horizontal_spacing=0.12,
        row_heights=[0.30, 0.35, 0.35],
    )

    # (1) scorecard table
    def vcell(r):
        return "PASS" if r["pass"] else "FAIL"
    fig.add_trace(
        go.Table(
            columnwidth=[36, 150, 200, 50, 55, 380],
            header=dict(
                values=["#", "metric", "job", "verdict", "margin", "detail"],
                fill_color="#EEF2F6", align="left",
                font=dict(color=INK, size=11), height=26,
            ),
            cells=dict(
                values=[
                    [i for i in range(1, len(ranked) + 1)],
                    [r["metric"] for r in ranked],
                    [r["job"] for r in ranked],
                    [vcell(r) for r in ranked],
                    [">1000x" if r["margin"] >= 1000 else f"{r['margin']:.1f}x" for r in ranked],
                    [r["detail"] for r in ranked],
                ],
                align="left", height=42,
                font=dict(color=[["#166534" if r["pass"] else "#991B1B" for r in ranked]
                                 if j == 3 else INK for j in range(6)], size=10),
                fill_color=[["white"] * len(ranked) if j != 3 else
                            ["#DCFCE7" if r["pass"] else "#FEE2E2" for r in ranked]
                            for j in range(6)],
            ),
        ),
        row=1, col=1,
    )

    # (2) separation strips, one per graded metric. The dashed threshold is a
    # line trace (not add_hline, which trips over the Table subplot in plotly).
    def thr(row, col, y):
        fig.add_trace(
            go.Scatter(x=[-0.5, 1.5], y=[y, y], mode="lines", hoverinfo="skip",
                       line=dict(color=RED, width=1, dash="dash"), showlegend=False),
            row=row, col=col,
        )

    _strip(fig, 2, 1, data["recon_keep"], data["recon_reject"], log_y=True)
    thr(2, 1, data["recon_thr"])
    _strip(fig, 2, 2, data["freq_keep"], data["freq_reject"], rng=[-0.1, 1.6])
    thr(2, 2, data["freq_thr"])
    _strip(fig, 3, 1, data["sib_keep"], data["sib_reject"], rng=[-0.1, 1.1])
    thr(3, 1, data["sib_thr"])
    _strip(fig, 3, 2, data["deg_keep"], data["deg_reject"], log_y=True)
    thr(3, 2, data["deg_thr"])

    fig.update_yaxes(title_text="parent-gain", row=2, col=1)
    fig.update_yaxes(title_text="survival", row=2, col=2)
    fig.update_yaxes(title_text="redundancy", row=3, col=1)
    fig.update_yaxes(title_text="out-degree", row=3, col=2)

    n_pass = sum(r["pass"] for r in data["rows"])
    fig.update_layout(
        title=dict(text=f"Exp 0 - metric calibration on synthetic ground truth "
                        f"({n_pass}/{len(data['rows'])} metrics recover the tree & reject their pathology)  "
                        f"<span style='color:{RED}'>- - dashed = config threshold</span>",
                   x=0.01, xanchor="left", font=dict(size=14, color=INK)),
        font=FONT, paper_bgcolor="white", plot_bgcolor="#FbFcFd",
        width=1200, height=1250, margin=dict(l=60, r=40, t=90, b=50),
        legend=dict(orientation="h", y=1.03, x=0.5, xanchor="center"),
    )
    return fig


def run_calibration():
    data = _calibration_data()
    fig = build_calibration_dashboard(data)
    out = C.OUT_DIR / "toy_calibration.html"
    fig.write_html(str(out), include_plotlyjs=True)
    print(f"saved: {out}")


# ---------------------------------------------------------------------------
# Qualitative dashboard (real gemma edges + Neuronpedia labels, colour-coded)
# ---------------------------------------------------------------------------
# One tint per verdict: survivors green, the three rejected categories in
# distinct warm/purple tints so a glance separates "kept" from "why-killed".
CAT_STYLE = {
    "survivor":          ("survivor",       "#DCFCE7", "#166534"),
    "reject:superparent": ("superparent",   "#EDE9FE", "#5B21B6"),
    "reject:freq-driven": ("freq-driven",   "#FEE2E2", "#991B1B"),
    "reject:no-recon":    ("no-recon",      "#FEF3C7", "#92400E"),
}
CAT_ORDER = ["survivor", "reject:superparent", "reject:freq-driven", "reject:no-recon"]


def _clip(s, n=64):
    s = (s or "").strip()
    return s if len(s) <= n else s[: n - 1] + "…"


def build_qualitative_dashboard(report):
    pairs = list(report.keys())
    fig = make_subplots(
        rows=len(pairs), cols=1,
        specs=[[{"type": "table"}] for _ in pairs],
        subplot_titles=[f"Block pair {k}" for k in pairs],
        vertical_spacing=0.04,
    )

    for ri, key in enumerate(pairs, start=1):
        rows = sorted(report[key], key=lambda r: (CAT_ORDER.index(r["category"]),
                                                   -r["reverse_cov"]))
        edge, verdict, Rv, recon, surv, plab, clab = ([] for _ in range(7))
        vcolor, rowfill = [], []
        for r in rows:
            short, tint, tcol = CAT_STYLE[r["category"]]
            edge.append(f"{r['parent']} → {r['child']}")
            verdict.append(short)
            Rv.append(f"{r['reverse_cov']:.2f}")
            recon.append("Y" if r["recon_pass"] else "n")
            surv.append("-" if r["freq_survival"] is None else f"{r['freq_survival']:.2f}")
            plab.append(_clip(r.get("parent_label")))
            clab.append(_clip(r.get("child_label")))
            vcolor.append(tcol)
            rowfill.append(tint)

        ncol = 7
        fig.add_trace(
            go.Table(
                columnwidth=[74, 90, 34, 40, 40, 250, 250],
                header=dict(
                    values=["edge", "verdict", "R", "recon", "surv",
                            "parent label (Neuronpedia)", "child label (Neuronpedia)"],
                    fill_color="#EEF2F6", align="left",
                    font=dict(color=INK, size=11), height=26,
                ),
                cells=dict(
                    values=[edge, verdict, Rv, recon, surv, plab, clab],
                    align="left", height=24,
                    font=dict(size=10,
                              color=[[INK] * len(edge) if j != 1 else vcolor for j in range(ncol)]),
                    fill_color=[rowfill for _ in range(ncol)],
                    line=dict(color="white", width=1),
                ),
            ),
            row=ri, col=1,
        )

    n_surv = sum(1 for k in pairs for r in report[k] if r["category"] == "survivor")
    n_rej = sum(1 for k in pairs for r in report[k] if r["category"] != "survivor")
    fig.update_layout(
        title=dict(text="Exp 0 - qualitative agreement on real gemma-2-2b  "
                        f"(<span style='color:#166534'>{n_surv} survivors</span> vs "
                        f"<span style='color:#991B1B'>{n_rej} rejected</span>; "
                        "read the labels: survivors should be related, rejected should look like artifacts)",
                   x=0.01, xanchor="left", font=dict(size=13, color=INK)),
        font=FONT, paper_bgcolor="white",
        width=1200, height=200 + 640 * len(pairs), margin=dict(l=30, r=30, t=70, b=30),
    )
    return fig


def run_qualitative():
    path = C.OUT_DIR / "qualitative_check.json"
    if not path.exists():
        raise SystemExit(f"missing {path} - run qualitative_check.py first")
    report = json.loads(path.read_text())
    fig = build_qualitative_dashboard(report)
    out = C.OUT_DIR / "qualitative_check.html"
    fig.write_html(str(out), include_plotlyjs=True)
    print(f"saved: {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--calibration", action="store_true",
                    help="visualise the synthetic-toy metric calibration (no cache needed)")
    ap.add_argument("--qualitative", action="store_true",
                    help="visualise real survivor-vs-rejected edges (needs qualitative_check.json)")
    args = ap.parse_args()
    if args.calibration:
        run_calibration()
        return
    if args.qualitative:
        run_qualitative()
        return

    if not C.EXP0_STATS_PATH.exists():
        raise SystemExit(f"missing {C.EXP0_STATS_PATH} - run cache_stats.py first")
    stats = torch.load(C.EXP0_STATS_PATH, weights_only=False)
    pairs = stats["pairs"]
    pairs_data = [compute_pair(stats, p, c) for (p, c) in pairs]

    dash = build_dashboard(pairs_data)
    dash_path = C.OUT_DIR / "metrics_dashboard.html"
    dash.write_html(str(dash_path), include_plotlyjs=True)
    print(f"saved: {dash_path}")

    # Sankey for the first pair that has a clear superparent (B0->B1: feature 15 fires ~99%).
    for (p, c), pd_ in zip(pairs, pairs_data):
        if pd_["superparents"]:
            sk = build_superparent_sankey(stats, pd_, p, c)
            sk_path = C.OUT_DIR / "superparent_sankey.html"
            sk.write_html(str(sk_path), include_plotlyjs=True)
            print(f"saved: {sk_path}")
            break


if __name__ == "__main__":
    main()
