"""
Visualise the hierarchy-metric results as self-contained interactive HTML
(plotly.js embedded, works offline - same style as the example sankey script).

Reads the cached statistics in outputs/exp0_stats.pt, recomputes the full
per-block-pair distributions (the JSON report keeps only summaries, not the
distributions), then writes:

    outputs/metrics_dashboard.html    aggregate dashboard: filter funnel + distributions
    outputs/superparent_sankey.html   one superparent's fan-out to its children

    outputs/toy_calibration.html      (--calibration) metric scorecard + the
                                      genuine-vs-pathological separation each
                                      metric achieves on the synthetic toy
    outputs/trained_toy_calibration.html (--trained-calibration) Tier-2 scorecard:
                                      edge recovery against the known tree, plus
                                      the Matryoshka nesting check when
                                      outputs/block_tree_alignment.json exists
    outputs/qualitative_dashboard.html (--qualitative) surviving vs rejected real
                                      edges with Neuronpedia labels, colour-coded

Run:
    pip install plotly
    python3 -m reporting.visualize                # real gemma-2-2b dashboards (needs exp0_stats.pt)
    python3 -m reporting.visualize --calibration  # synthetic-toy calibration (needs no cache)
    python3 -m reporting.visualize --qualitative  # real survivor-vs-rejected label table
                                                  # (needs outputs/qualitative_check.json)
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
import plotly.graph_objects as go
from plotly.offline import get_plotlyjs
from plotly.subplots import make_subplots

import config as C
# The block structure comes from the file being graded, never from this module:
# config.BLOCK_RANGES is gemma's 32768 latents in 5 blocks, and slicing a PCFG
# dictionary (1792 in 8) at those boundaries returns a full set of plausible
# numbers computed from the wrong features. run_metrics owns the rule; importing
# it keeps one definition rather than a second copy that can drift.
from run_metrics import source_structure
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
AMBER = "#F59E0B"               # 4th pair (B3->B4), enabled with EXP0_B3B4=1
PAIR_COLORS = [PURPLE, BLUE, TEAL, AMBER]
GREEN, GREY, RED = "#22C55E", "#CBD5E1", "#EF4444"
GREEN_DARK = "#15803D"          # readable green for heading text on white


def page_subtitle(stats=None):
    """Context line for any page that describes ONE layer.

    Every per-layer HTML lands in outputs/layer_NN/ with no on-page clue which
    layer it is, so state it up front together with the run knobs a reader
    needs to interpret the numbers.
    """
    tokens = stats.get("total_tokens") if stats is not None else None
    cfg = stats.get("config") if stats is not None else None
    # Plotly does not decode HTML entities, so config.scope_line uses literal glyphs.
    return C.scope_line(tokens, bold=("<b>", "</b>"), config=cfg)


def plotly_asset(page_dir):
    """Ensure OUT_DIR/assets/plotly.min.js exists; return its page-relative src.

    Embedding the 4.6 MB plotly bundle in every page (`include_plotlyjs=True`)
    made each dashboard ~4.8 MB, so every regeneration added ~70 MB of new blobs
    to git. One shared copy keeps the pages ~150 KB and still works offline and
    on GitHub Pages, which cannot serve an LFS pointer (see .gitattributes).
    The src is computed per page rather than from a level count, so it stays
    right under EXP0_OUT, where the run dir is not named `outputs`.
    """
    dest = C.OUT_DIR / "assets" / "plotly.min.js"
    if not dest.exists():
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(get_plotlyjs(), encoding="utf-8")
    return Path(os.path.relpath(dest, page_dir)).as_posix()


def write_page(fig, path, up=2):
    """Write a plotly figure with the site nav bar on top.

    Every page is reachable only by a deep link, so each carries its own nav
    (`config.nav_html`); `up` is its distance to the site root. Which layer and
    which page this is are read off the output path, so callers never have to
    repeat what the filename already says.

    Injected after write_html because plotly gives no layout slot for it — the
    same slot also carries the <script> tag for the shared plotly bundle that
    `include_plotlyjs=False` leaves out.
    """
    path = Path(path)
    fig.write_html(str(path), include_plotlyjs=False)
    layer, page = _page_identity(path)
    head = (
        f"{C.nav_html(depth=up, layer=layer, page=page, current=_site_path(path))}\n"
        "<script>window.PlotlyConfig = {MathJaxConfig: 'local'};</script>\n"
        f'<script charset="utf-8" src="{plotly_asset(path.parent)}"></script>'
    )
    html = path.read_text().replace("<body>", "<body>\n" + head, 1)
    path.write_text(html)


def _page_identity(path):
    """(layer, page-file) for a written page, read off its own path.

    A page under layer_NN/ describes that layer; anything else is site-wide and
    gets no Page row. It can still keep the page KIND, though: from a run
    directory's dashboard the layer pills should land on each gemma layer's
    dashboard, not on its index. `page` only names one of the five per-layer
    kinds -- the calibration pages are not among them, and marking one would
    point every pill at a file no layer has.
    """
    name = Path(path).name
    kind = name if name in {f for f, _ in C.NAV_PAGES} else None
    parent = Path(path).parent.name
    if parent.startswith("layer_"):
        try:
            return int(parent.split("_")[1]), kind
        except ValueError:
            pass
    return None, kind


def _site_path(path):
    """This page's path from the site root, for highlighting the global row.

    The page's position in the OUTPUT TREE, not on disk: a source with its own
    run directory (outputs/pcfg/) is site-wide too -- it describes no gemma layer
    -- and the old fixed `outputs/<file>` gave it a path matching no nav entry,
    so its own link never lit up. Deriving it from OUT_DIR rather than the repo
    root keeps a scratch run under EXP0_OUT identical to the published page:
    where the file was written is not what the page IS.
    """
    path = Path(path)
    if path.parent.name.startswith("layer_"):
        return None
    return "outputs/" + Path(os.path.relpath(path.resolve(), C.OUT_DIR)).as_posix()


def scope_subtitle(text):
    """Context line for a page that is NOT tied to one layer.

    Multi-layer and layer-independent pages need the same up-front statement of
    what they cover, so a reader never has to guess the scope from the URL.
    """
    return f"<b>{text}</b>"


def _titled(name, desc, stats=None, subtitle=None):
    """Three-line page header, identical on every generated page.

        1. name      what page this is, so a reader arriving from a link knows
        2. desc      one line on what the figure shows
        3. subtitle  the scope: which layer(s), and the knobs behind the numbers
    """
    sub = subtitle if subtitle is not None else page_subtitle(stats)
    return (f"<span style='font-size:17px;color:{PURPLE}'><b>{name}</b></span>"
            f"<br><span style='font-size:12.5px'>{desc}</span>"
            f"<br><span style='font-size:11.5px;color:{INK}'>{sub}</span>")
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
    ranges, sibling_blocks = source_structure(stats)
    p0, p1 = ranges[p_blk]
    c0, c1 = ranges[c_blk]
    fire_p, fire_c = fire[p0:p1], fire[c0:c1]

    cofire = stats["cofire"][key].double()
    R, F = coverage_legs(cofire, fire_p, fire_c)
    edge_mask = keep_edges(R, fire_p, fire_c, C.EDGE_TAU, C.MIN_FIRE_COUNT,
                           cofire=cofire, min_joint=C.MIN_JOINT)
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
    if c_blk in sibling_blocks:
        sib = sibling_redundancy(edge_mask, stats["within_cofire"][c_blk].double(), fire_c)
    redundancy = [v["redundancy"] for v in sib.values()]

    superparents = find_superparents(
        edge_mask, fire_p, total, C.SUPERPARENT_OUTDEG_FRAC, C.SUPERPARENT_FIRE_FRAC
    )

    return {
        "key": key,
        # First global feature index of each endpoint's block, carried alongside
        # the tensors so every downstream plot labels features from the SAME
        # structure these numbers were computed with.
        "p0": p0,
        "c0": c0,
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


def _wrap(text, width=28):
    """Soft-wrap a label onto <br> lines so long descriptions stay readable
    inside plotly hover boxes / sankey nodes."""
    words, lines, cur = text.split(), [], ""
    for w in words:
        if cur and len(cur) + 1 + len(w) > width:
            lines.append(cur)
            cur = w
        else:
            cur = f"{cur} {w}".strip()
    if cur:
        lines.append(cur)
    return "<br>".join(lines)


def build_dashboard(pairs_data, labels=None, stats=None):
    labels = labels or {}
    fig = make_subplots(
        rows=3,
        cols=2,
        subplot_titles=(
            "Edges passing each metric (log scale) — baseline vs strict",
            "Share of candidate edges passing each metric (%)",
            "Out-degree CCDF: P(children per parent ≥ x)",
            "Frequency-survival distribution",
            "Sibling-redundancy distribution",
            "Superparents: child coverage x firing rate",
        ),
        vertical_spacing=0.10,
        horizontal_spacing=0.10,
    )

    stages = ["candidate", "improves recon", "survives freq", "PMI > 0", "pass S_res"]
    for i, pd_ in enumerate(pairs_data):
        # Cycle: the palette holds gemma's four pairs, and a dictionary with more
        # blocks has more pairs (the PCFG SAE's 8 blocks give 7). Repeating a
        # colour is survivable; running off the end of the list is not.
        col = PAIR_COLORS[i % len(PAIR_COLORS)]
        name = pd_["key"]
        n_pmi = pd_.get("n_pmi")
        n_sres = pd_.get("n_sres")
        # (1,1) count of edges passing each metric (independent counts, not strict
        # nesting; recon/freq are the lenient baseline, PMI/S_res the strict tests)
        fig.add_bar(
            x=stages,
            y=[pd_["n_edges"], pd_["n_recon"], pd_["n_survive"], n_pmi, n_sres],
            name=name, marker_color=col, legendgroup=name, row=1, col=1,
        )
        # (1,2) percentages — all metrics, so the strict S_res collapse is visible
        e = max(pd_["n_edges"], 1)
        pmi_pct = 100 * n_pmi / e if n_pmi is not None else None
        sres_pct = 100 * n_sres / e if n_sres is not None else None
        fig.add_bar(
            x=["improves recon %", "survives freq %", "PMI > 0 %", "pass S_res %"],
            y=[100 * pd_["n_recon"] / e, 100 * pd_["n_survive"] / e, pmi_pct, sres_pct],
            name=name, marker_color=col, legendgroup=name, showlegend=False, row=1, col=2,
        )
        # (2,1) out-degree CCDF: P(outdeg >= x) — a heavy right tail = superparents
        od = torch.tensor(pd_["outdeg"]).sort(descending=True).values
        if od.numel():
            n = od.numel()
            ccdf = (torch.arange(1, n + 1).double() / n)
            fig.add_trace(
                go.Scatter(x=od.numpy(), y=ccdf.numpy(), mode="lines",
                           line=dict(color=col, width=2, shape="hv"),
                           name=name, legendgroup=name, showlegend=False),
                row=2, col=1,
            )
            fig.update_xaxes(type="log", row=2, col=1)
            fig.update_yaxes(type="log", row=2, col=1)
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
            p_base = pd_["p0"]
            fig.add_trace(
                go.Scatter(
                    x=[s["outdeg_frac"] for s in sp],
                    y=[s["fire_frac"] for s in sp],
                    mode="markers", marker=dict(color=col, size=12, line=dict(width=1, color="white")),
                    text=[f"feature {p_base + s['parent_local']}<br>"
                          f"{_wrap(C.feature_label(p_base + s['parent_local'], labels))}"
                          for s in sp],
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
        # Title and legend both live above the plot area: pin them to separate
        # paper-space rows (title on top, legend under it) or they overlap.
        title=dict(text=_titled(
                       "Metrics dashboard",
                       "Grading coverage edges with the five metrics "
                       f"({pairs_data[0]['n_edges']:,} to {pairs_data[-1]['n_edges']:,} candidate edges "
                       "across the block pairs)",
                       stats),
                   x=0.01, xanchor="left", yref="container", y=0.985, yanchor="top",
                   font=dict(size=14, color=INK)),
        barmode="group", bargap=0.15, font=FONT,
        paper_bgcolor="white", plot_bgcolor="#FbFcFd",
        width=1200, height=1400, margin=dict(l=60, r=40, t=200, b=50),
        legend=dict(orientation="h", y=1.012, yanchor="bottom", x=0.01, xanchor="left"),
    )
    return fig


def _superparent_sankey_trace(stats, pd_, p_blk, c_blk, top_n=25, feat_labels=None):
    """Build the Sankey trace + title for one block pair's top superparent.

    Returns (trace, title) or None when the pair has no superparent.
    Shared by the single-pair figure and the stacked all-pairs figure.
    """
    feat_labels = feat_labels or {}
    sp = pd_["superparents"]
    if not sp:
        return None
    parent_local = sp[0]["parent_local"]
    p0, c0 = pd_["p0"], pd_["c0"]
    gp = p0 + parent_local

    kids = torch.nonzero(pd_["edge_mask"][parent_local]).flatten()
    cof = pd_["cofire"][parent_local, kids]
    order = torch.argsort(cof, descending=True)[:top_n]
    kids = kids[order]

    labels = [f"B{p_blk}:{gp} {C.feature_label(gp, feat_labels)} "
              f"(fires {100 * sp[0]['fire_frac']:.0f}%)"]
    node_colors = [PURPLE]
    source, target, value, link_colors = [], [], [], []
    for i, ci in enumerate(kids.tolist(), start=1):
        gc = c0 + ci
        if "sres_pass" in pd_:                        # second-pass refinement verdict
            real = bool(pd_["sres_pass"][parent_local, ci])
        else:                                         # fallback: weak baseline + freq
            passes = bool(pd_["recon_pass"][parent_local, ci])
            s = pd_["survival"][parent_local, ci]
            real = passes and (not torch.isnan(s)) and float(s) >= C.FREQ_SURVIVAL_MIN
        labels.append(f"B{c_blk}:{gc} {C.feature_label(gc, feat_labels)}")
        node_colors.append(GREEN if real else GREY)
        link_colors.append("rgba(34,197,94,0.55)" if real else "rgba(203,213,225,0.5)")
        source.append(0)
        target.append(i)
        value.append(float(pd_["cofire"][parent_local, ci]))

    trace = go.Sankey(
        arrangement="snap",
        node=dict(label=labels, color=node_colors, pad=18, thickness=14, line=dict(width=0)),
        link=dict(source=source, target=target, value=value, color=link_colors,
                  hovertemplate="%{source.label} -> %{target.label}<br>"
                                "co-fire: %{value}<extra></extra>"),
    )
    n_real = sum(1 for c in node_colors[1:] if c == GREEN)
    # Rendered as a panel heading: bold, and the survivor count coloured by
    # whether anything survived at all (green if some, red if none).
    count_color = GREEN_DARK if n_real else RED
    title = (f"<b>B{p_blk} → B{c_blk}</b>"
             f"<span style='color:{INK}'>　 superparent </span>"
             f"<b>B{p_blk}:{gp}</b>"
             f"<span style='color:{INK}'> → top {len(kids)} children 　 pass S_res (genuine refinement): </span>"
             f"<b><span style='color:{count_color}'>{n_real}/{len(kids)}</span></b>")
    return trace, title


def build_superparent_sankey(stats, pd_, p_blk, c_blk, top_n=25, feat_labels=None):
    """One superparent's flow to its top_n children; colour separates edges that
    survive (reconstruction AND frequency) from frequency-captured ones."""
    built = _superparent_sankey_trace(stats, pd_, p_blk, c_blk, top_n, feat_labels)
    if built is None:
        return None
    trace, title = built
    fig = go.Figure(trace)
    _add_sres_legend(fig)
    fig.update_layout(
        title=dict(text=title, x=0.01, xanchor="left", font=dict(size=13, color=INK)),
        font=FONT, paper_bgcolor="white", plot_bgcolor="white", showlegend=True,
        legend=dict(orientation="h", y=1.02, x=0.01),
        width=1150, height=680, margin=dict(l=50, r=90, t=90, b=40),
    )
    return fig


def _add_sres_legend(fig):
    """Two off-canvas markers so the green/grey link colouring gets a real legend
    box (Sankey links themselves don't populate the legend)."""
    fig.add_trace(go.Scatter(x=[None], y=[None], mode="markers",
                             marker=dict(size=12, color=GREEN),
                             name="pass S_res (genuine refinement)"))
    fig.add_trace(go.Scatter(x=[None], y=[None], mode="markers",
                             marker=dict(size=12, color=GREY),
                             name="fail S_res (co-fires only)"))
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)


def build_all_superparent_sankeys(stats, pairs, pairs_data, top_n=25, feat_labels=None):
    """Every block pair's top superparent, stacked in ONE figure.

    Stacking keeps a single embedded plotly.js bundle, so the file stays the
    size of one dashboard instead of one per pair.
    """
    built = []
    for (p, c), pd_ in zip(pairs, pairs_data):
        got = _superparent_sankey_trace(stats, pd_, p, c, top_n, feat_labels)
        if got is not None:
            built.append((p, c, *got))
    if not built:
        return None

    n = len(built)
    panel_h = 660                      # px per panel
    title_gap = 0.055                  # fraction of figure height reserved per heading
    fig = go.Figure()
    annotations = []
    for i, (p, c, trace, title) in enumerate(built):
        top = 1.0 - i / n
        bottom = 1.0 - (i + 1) / n
        trace.domain = dict(x=[0, 1], y=[bottom, top - title_gap])
        fig.add_trace(trace)
        annotations.append(dict(
            text=title, x=0.0, xref="paper", xanchor="left",
            y=top - 0.006, yref="paper", yanchor="top",
            showarrow=False, align="left",
            font=dict(size=14, color=PURPLE),
            bgcolor="#F6F3FE", bordercolor="#E3DAFB", borderwidth=1, borderpad=8,
        ))

    _add_sres_legend(fig)
    fig.update_layout(
        title=dict(text=_titled(
                       "Superparent fan-out",
                       "One high-firing feature adopting most of the next block, "
                       "shown for every block pair (green = the edge passes S_res, "
                       "the genuine-refinement test)",
                       stats),
                   x=0.005, xanchor="left", yref="container", y=0.985, yanchor="top",
                   font=dict(size=14, color=INK)),
        annotations=annotations, showlegend=True,
        legend=dict(orientation="h", y=1.0, x=0.30, yref="container"),
        font=FONT, paper_bgcolor="white", plot_bgcolor="white",
        width=1250, height=160 + panel_h * n, margin=dict(l=50, r=90, t=150, b=40),
    )
    return fig


# ---------------------------------------------------------------------------
# Calibration dashboard (synthetic ground-truth toy from validation/)
# ---------------------------------------------------------------------------
def _calibration_data():
    """Run the five metrics on the toy and split each metric's per-edge score
    into the class it should KEEP (genuine) vs the class it should REJECT."""
    from validation.calibrate_on_synthetic_toy import _render, _run_metrics, _score
    from validation.toy_world import build_world

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
        title=dict(text=_titled(
                       "Metric calibration",
                       f"{n_pass}/{len(data['rows'])} metrics recover the true tree and reject their "
                       f"injected pathology　·　"
                       f"<span style='color:{RED}'>- - dashed = config threshold</span>",
                       subtitle=scope_subtitle("No layer: synthetic toy, layer-independent")
                       + "　·　applies to every layer　·　production metrics and thresholds, unchanged"),
                   x=0.01, xanchor="left", yref="container", y=0.985, yanchor="top",
                   font=dict(size=14, color=INK)),
        font=FONT, paper_bgcolor="white", plot_bgcolor="#FbFcFd",
        width=1200, height=1280, margin=dict(l=60, r=40, t=170, b=50),
        legend=dict(orientation="h", y=1.008, yanchor="bottom", x=0.01, xanchor="left"),
    )
    return fig


def run_calibration():
    data = _calibration_data()
    fig = build_calibration_dashboard(data)
    out = C.OUT_DIR / "toy_calibration.html"
    write_page(fig, out, up=1)          # lives in outputs/
    print(f"saved: {out}")


# ---------------------------------------------------------------------------
# Tier 2 dashboard: metrics on a Matryoshka SAE actually trained on the toy.
# Mirrors the Tier-1 scorecard, but the question is edge recovery vs a known
# tree, not pathology separation.
# ---------------------------------------------------------------------------
CAL2_STYLE = {                       # (label, fill, ink)
    "recovered":                  ("recovered ✓",           "#DCFCE7", "#166534"),
    "missed: child not learned":  ("missed (SAE gap)",       "#FEF3C7", "#92400E"),
    "missed":                     ("missed",                 "#FEE2E2", "#991B1B"),
    "spurious":                   ("spurious (false pos)",   "#FEE2E2", "#991B1B"),
}
CAL2_ORDER = ["recovered", "missed: child not learned", "missed", "spurious"]


# A scorecard where every row says PASS is the same survivorship problem the error
# log has: it reads as reassurance. Some rows here are not tests at all -- feature
# recovery is a fact about the SAE, and feature splitting is a caveat on how
# generously the nesting check was read. Grading them would claim a verdict nobody
# measured, so they get a third state and are visibly not scored.
def _verdict_text(r):
    return "noted" if not r.get("graded", True) else ("PASS" if r["pass"] else "FAIL")


def _verdict_ink(r):
    return "#9AA7B3" if not r.get("graded", True) else ("#166534" if r["pass"] else "#991B1B")


def _verdict_fill(r):
    return "#F1F5F9" if not r.get("graded", True) else ("#DCFCE7" if r["pass"] else "#FEE2E2")


def trained_scorecard(d, align):
    """Tier 2's checks as scored rows, in the shape Tier 1 uses.

    A verdict alone is not calibration: a check that passed by a hair is not
    evidence, it is luck that has not run out. So every row carries the same
    `margin` idea as the Tier-1 scorecard — how far this check was from failing —
    and the margins here are small integers because the toy is small. That is the
    honest reading, not a defect of the display.

    The nesting rows come from block_tree_alignment.json when it exists.
    `calibrate_on_trained_toy` indexes by ground truth and never by block, on
    purpose: mixing them would confound "is the metric right?" with "did Matryoshka
    order the features right?". Those are two questions and they get two rows.
    """
    tp, fp2, fn = d["true_positives"], d["false_positives"], d["false_negatives"]
    F = d["n_features"]
    n_gap = sum(1 for r in d["edge_rows"] if "not learned" in r["category"])
    testable = (tp + fn) - n_gap

    rows = [
        {"check": "1. precision — no invented edges",
         "job": "keep nothing the tree does not contain",
         "pass": fp2 == 0,
         "value": f"{d['precision']:.2f}",
         "margin": f"{fp2} false positives",
         "detail": f"{tp} edges kept, {fp2} of them absent from the true tree"},
        {"check": "2. recall against the SAE's ceiling",
         "job": "recover every edge whose endpoints the SAE actually learned",
         "pass": tp == testable,
         "value": f"{tp}/{testable}",
         "margin": f"{testable - tp} testable misses",
         "detail": f"raw recall {d['recall']:.2f} ({tp}/{tp + fn}); {n_gap} of the "
                   f"{fn} misses are edges whose child the SAE never learned, so no "
                   f"metric could have found them"},
        {"check": "3. feature recovery (the ceiling itself)",
         "job": "state what the SAE gave the metrics to work with",
         "graded": False,
         "value": f"{d['n_recovered_features']}/{F}",
         "margin": "—",
         "detail": f"{F - d['n_recovered_features']} true features were never learned. "
                   f"This bounds recall from above: it is a fact about the SAE, so it is "
                   f"reported and not graded"},
    ]

    if align:
        gaps = [r["child_block"] - r["parent_block"]
                for r in align["edge_rows"] if r["child_block"] is not None]
        n_t, n_ok = align["n_testable"], align["n_respected"]
        splits = align.get("split_features") or {}
        rows += [
            {"check": "4. Matryoshka nesting respects the tree",
             "job": "parent lands in an earlier block than its children",
             "pass": n_ok == n_t and n_t > 0,
             "value": f"{n_ok}/{n_t}",
             "margin": f"min gap {min(gaps)} block" + ("" if min(gaps) == 1 else "s"),
             "detail": f"block gaps {sorted(gaps)}; mean parent block "
                       f"{align['mean_parent_block']:.1f} vs child "
                       f"{align['mean_child_block']:.1f}. The narrowest edge clears by "
                       f"{min(gaps)} — the claim holds, but not by much at its tightest"},
            {"check": "5. feature splitting (caveat on 4)",
             "job": "say how generously check 4 was read",
             "graded": False,
             "value": f"{len(splits)} split",
             "margin": "—",
             "detail": ("no true feature was recovered by more than one latent, so check 4 "
                        "had no choice to make" if not splits
                        else "recovered by >1 latent: "
                             + ", ".join(f"feature {k} in blocks {v}" for k, v in splits.items())
                             + ". Check 4 takes the EARLIEST block for each feature, which is "
                               "the reading most favourable to the architecture — a later "
                               "choice could turn a respected edge into a violation. Not a "
                               "metric failure, but it is why 6/6 is weaker than it looks")},
        ]
    return rows


def build_trained_calibration_dashboard(d, align=None):
    """Tier 2 in Tier 1's shape: a scored card, then the evidence behind it.

    Not the layer dashboard's shape. That one plots distributions over millions of
    candidate edges, where a CCDF means something. Here there are nine true edges and
    twenty features — a distribution over nine points is decoration. The question is
    also different in kind: the layer pages ask what happened to a population, both
    calibration tiers ask whether each check did its job where the answer is known.
    """
    sc = trained_scorecard(d, align)
    rows = sorted(d["edge_rows"], key=lambda r: (CAL2_ORDER.index(r["category"]),
                                                 r["parent"], r["child"]))
    have_align = bool(align)

    fig = make_subplots(
        rows=3, cols=2,
        specs=[[{"type": "table", "colspan": 2}, None],
               [{"type": "table"}, {}],
               [{"type": "table"}, {}]],
        subplot_titles=("", "Edge recovery vs the known tree",
                        "Matryoshka nesting: parent block vs child block",
                        "Nesting, edge by edge", "Where the tree landed in the blocks"),
        vertical_spacing=0.075, horizontal_spacing=0.08,
        row_heights=[0.34, 0.33, 0.33],
    )

    # (1) the scorecard — same six columns as Tier 1, same verdict colouring
    fig.add_trace(
        go.Table(
            columnwidth=[36, 190, 210, 50, 92, 420],
            header=dict(values=["#", "check", "job", "verdict", "margin", "detail"],
                        fill_color="#EEF2F6", align="left",
                        font=dict(color=INK, size=11), height=26),
            cells=dict(
                values=[
                    list(range(1, len(sc) + 1)),
                    [r["check"] for r in sc],
                    [r["job"] for r in sc],
                    [_verdict_text(r) for r in sc],
                    [r["margin"] for r in sc],
                    [r["detail"] for r in sc],
                ],
                align="left", height=54,
                font=dict(size=10,
                          color=[[_verdict_ink(r) for r in sc] if j == 3 else INK
                                 for j in range(6)]),
                fill_color=[["white"] * len(sc) if j != 3 else
                            [_verdict_fill(r) for r in sc] for j in range(6)],
            ),
        ), row=1, col=1,
    )

    # (2,1) edge recovery, unchanged in content — the evidence for rows 1-3
    edge, verdict, note, vcol, fill = [], [], [], [], []
    for r in rows:
        label, tint, ink = CAL2_STYLE[r["category"]]
        edge.append(r["edge"])
        verdict.append(label)
        note.append("kept by all metrics, matches the tree" if r["category"] == "recovered"
                    else "child feature never learned by the SAE" if "not learned" in r["category"]
                    else "metric kept an edge not in the tree" if r["category"] == "spurious"
                    else "in the tree but not kept")
        vcol.append(ink)
        fill.append(tint)
    fig.add_trace(
        go.Table(
            columnwidth=[70, 150, 300],
            header=dict(values=["edge (parent → child)", "verdict", "why"],
                        fill_color="#EEF2F6", align="left",
                        font=dict(color=INK, size=11), height=26),
            cells=dict(values=[edge, verdict, note], align="left", height=26,
                       font=dict(size=10, color=[[INK] * len(edge), vcol, [INK] * len(edge)]),
                       fill_color=[fill, fill, fill], line=dict(color="white", width=1)),
        ), row=2, col=1,
    )

    if have_align:
        nb = len(align["block_ranges"])
        tested = [r for r in align["edge_rows"] if r["child_block"] is not None]

        # (2,2) the nesting claim as a picture. Every point must sit ABOVE the
        # dotted diagonal; a violation would be a point below it, which is easier
        # to disbelieve than the word "respected" in a cell.
        fig.add_trace(go.Scatter(x=[0, nb - 1], y=[0, nb - 1], mode="lines",
                                 line=dict(color="#CBD5E1", width=1, dash="dot"),
                                 hoverinfo="skip", showlegend=False), row=2, col=2)
        fig.add_trace(
            go.Scatter(
                x=[r["parent_block"] for r in tested],
                y=[r["child_block"] for r in tested],
                mode="markers+text",
                text=[r["edge"].replace(" -> ", "→") for r in tested],
                textposition="top center", textfont=dict(size=9, color="#9AA7B3"),
                marker=dict(size=13, line=dict(width=1, color="white"),
                            color=["#166534" if r["verdict"] == "respected" else "#991B1B"
                                   for r in tested]),
                hovertemplate="%{text}<br>parent B%{x} → child B%{y}<extra></extra>",
                showlegend=False), row=2, col=2)
        fig.update_xaxes(title_text="parent block", range=[-0.7, nb - 0.3], dtick=1,
                         row=2, col=2)
        fig.update_yaxes(title_text="child block", range=[-0.7, nb + 0.3], dtick=1,
                         row=2, col=2)

        # (3,1) the same six edges as a table, plus the three that cannot be tested
        av, ap, ac, aw, afill, aink = [], [], [], [], [], []
        for r in align["edge_rows"]:
            ok = r["verdict"] == "respected"
            untest = r["child_block"] is None
            av.append(r["edge"].replace(" -> ", " → "))
            ap.append("-" if r["parent_block"] is None else f"B{r['parent_block']}")
            ac.append("-" if untest else f"B{r['child_block']}")
            aw.append("untestable — child never learned" if untest
                      else f"respected (gap {r['child_block'] - r['parent_block']})" if ok
                      else "VIOLATED")
            afill.append("#F1F5F9" if untest else "#DCFCE7" if ok else "#FEE2E2")
            aink.append("#9AA7B3" if untest else "#166534" if ok else "#991B1B")
        fig.add_trace(
            go.Table(
                columnwidth=[70, 50, 50, 220],
                header=dict(values=["edge", "parent", "child", "verdict"],
                            fill_color="#EEF2F6", align="left",
                            font=dict(color=INK, size=11), height=26),
                cells=dict(values=[av, ap, ac, aw], align="left", height=26,
                           font=dict(size=10, color=[[INK] * len(av)] * 3 + [aink]),
                           fill_color=[afill] * 4, line=dict(color="white", width=1)),
            ), row=3, col=1,
        )

        # (3,2) parents early, children late — the nesting claim in aggregate
        fb = {int(k): v for k, v in align["first_block_of_feature"].items()}
        parents = sorted({p for p, _ in d["true_edges"]})
        children = sorted({c for _, c in d["true_edges"]})
        for name, feats, col in (("parents", parents, "#7C22CE"),
                                 ("children", children, "#F59E0B")):
            fig.add_bar(x=list(range(nb)),
                        y=[sum(1 for f in feats if fb.get(f) == b) for b in range(nb)],
                        name=name, marker_color=col, row=3, col=2)
        fig.update_xaxes(title_text="earliest Matryoshka block holding the feature",
                         dtick=1, row=3, col=2)
        fig.update_yaxes(title_text="features", row=3, col=2)
        fig.update_layout(barmode="group",
                          legend=dict(orientation="h", x=0.62, y=0.055,
                                      bgcolor="rgba(255,255,255,0.7)"))

    tp, fp2, fn = d["true_positives"], d["false_positives"], d["false_negatives"]
    F = d["n_features"]
    nest = ""
    if have_align:
        nest = (f"　·　<b><span style='color:#166534'>nesting {align['n_respected']}"
                f"/{align['n_testable']}</span></b>")
    fig.update_layout(
        title=dict(text=_titled(
            "Trained-toy calibration (Tier 2)",
            f"metrics run on a Matryoshka SAE trained on Bussmann's tree　·　"
            f"<b><span style='color:#166534'>precision {d['precision']:.2f}</span></b>, "
            f"<b>recall {d['recall']:.2f}</b> "
            f"({tp}/{tp + fn} edges, {fp2} false positives){nest}",
            subtitle=scope_subtitle("No layer: Bussmann toy, a Matryoshka SAE trained on it")
            + f"　·　SAE recovered {d['n_recovered_features']}/{F} true features　·　"
              "the misses are edges whose child the SAE never learned, not metric failures"),
            x=0.01, xanchor="left", yref="container", y=0.99, yanchor="top",
            font=dict(size=14, color=INK)),
        font=FONT, paper_bgcolor="white", plot_bgcolor="white",
        width=1180, height=1180 if have_align else 780,
        margin=dict(l=40, r=40, t=150, b=30),
    )
    return fig


def run_trained_calibration():
    path = C.OUT_DIR / "trained_toy_calibration.json"
    if not path.exists():
        raise SystemExit(f"missing {path} - run validation/calibrate_on_trained_toy.py first")
    # Optional: the lateral control's output. Absent on a checkout that has not run
    # validation.block_tree_alignment, and the page degrades to the recovery half
    # rather than failing -- the two are separate questions and separate scripts.
    ap = C.OUT_DIR / "block_tree_alignment.json"
    align = json.loads(ap.read_text()) if ap.exists() else None
    if align is None:
        print("note: outputs/block_tree_alignment.json not found - "
              "run `python3 -m validation.block_tree_alignment` for the nesting panels")
    fig = build_trained_calibration_dashboard(json.loads(path.read_text()), align)
    out = C.OUT_DIR / "trained_toy_calibration.html"
    write_page(fig, out, up=1)
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
        edge, verdict, Rv, Fv, gain, recon, surv, plab, clab = ([] for _ in range(9))
        vcolor, rowfill = [], []
        for r in rows:
            short, tint, tcol = CAT_STYLE[r["category"]]
            # Plain text: plotly table cells render HTML as literal text, so a
            # link here would print its own markup. The clickable Neuronpedia
            # links live in the markdown report instead.
            edge.append(f"{r['parent']} → {r['child']}")
            verdict.append(short)
            Rv.append(f"{r['reverse_cov']:.2f}")
            Fv.append(f"{r['forward_cov']:.2f}")
            gain.append(f"{r['recon_parent_gain']:.2f}")
            recon.append("Y" if r["recon_pass"] else "n")
            surv.append("-" if r["freq_survival"] is None else f"{r['freq_survival']:.2f}")
            plab.append(_clip(r.get("parent_label")))
            clab.append(_clip(r.get("child_label")))
            vcolor.append(tcol)
            rowfill.append(tint)

        cols = [edge, verdict, Rv, Fv, gain, recon, surv, plab, clab]
        ncol = len(cols)
        fig.add_trace(
            go.Table(
                columnwidth=[80, 88, 32, 32, 44, 40, 40, 226, 226],
                header=dict(
                    values=["edge", "verdict", "R", "F", "gain", "recon", "surv",
                            "parent label (Neuronpedia)", "child label (Neuronpedia)"],
                    fill_color="#EEF2F6", align="left",
                    font=dict(color=INK, size=11), height=26,
                ),
                cells=dict(
                    values=cols,
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
        title=dict(text=_titled(
                       "Qualitative agreement",
                       f"<span style='color:#166534'>{n_surv} survivors</span> vs "
                       f"<span style='color:#991B1B'>{n_rej} rejected</span>, read against Neuronpedia "
                       "labels: survivors should be semantically related, rejected should look like artifacts"
                       "<br><span style='font-size:11px'>R = reverse coverage　·　F = forward coverage　·　"
                       "gain = parent's reconstruction gain　·　surv = frequency survival　·　"
                       "the markdown report has the same rows with clickable Neuronpedia links</span>"),
                   x=0.01, xanchor="left", yref="container", y=0.985, yanchor="top",
                   font=dict(size=13, color=INK)),
        font=FONT, paper_bgcolor="white",
        width=1200, height=270 + 640 * len(pairs), margin=dict(l=30, r=30, t=140, b=30),
    )
    return fig


def run_qualitative():
    path = C.RUN_DIR / "qualitative_check.json"
    if not path.exists():
        raise SystemExit(f"missing {path} - run python3 -m validation.qualitative_check first")
    report = json.loads(path.read_text())
    fig = build_qualitative_dashboard(report)
    # Named to mirror metrics_dashboard/metrics_report: writing this as
    # qualitative_check.html would shadow Jekyll's render of qualitative_check.md,
    # leaving the text report unreachable on the published site.
    out = C.RUN_DIR / "qualitative_dashboard.html"
    write_page(fig, out, up=2)          # lives in outputs/layer_NN/
    print(f"saved: {out}")


ORANGE = "#D98A3D"


def build_in_block_dashboard(report):
    """In-block (same-level) edges dashboard from in_block_edges.json. Two panels,
    both with an explicit legend: (1) per-block edge/duplicate/gate counts,
    (2) the in-block superparents' out-degree x firing rate."""
    blocks = report["blocks"]
    names = [f"B{b['block']} ({b['n_features']})" for b in blocks]
    fig = make_subplots(
        rows=2, cols=1, vertical_spacing=0.16,
        subplot_titles=(
            "Same-level relations per block: directed edges, duplicates, and how "
            "many survive PMI then S_res (log)",
            "In-block superparents: out-degree x firing rate (bubble = children)",
        ),
    )
    series = [
        ("directed edges", GREY, [b["n_edges"] for b in blocks]),
        ("survive PMI > 0", PURPLE, [b["n_after_pmi"] for b in blocks]),
        ("pass S_res (genuine)", GREEN, [(b["sres"]["n_pass"] if b.get("sres") else 0) for b in blocks]),
        ("duplicate pairs (rename/split)", ORANGE, [b["n_duplicates"] for b in blocks]),
    ]
    for nm, col, y in series:
        fig.add_bar(x=names, y=y, name=nm, marker_color=col, legendgroup=nm, row=1, col=1)
    fig.update_yaxes(type="log", row=1, col=1)

    # (2) superparents across all blocks, sized by out-degree
    any_sp = False
    for bi, b in enumerate(blocks):
        sps = b["superparents"]
        if not sps:
            continue
        any_sp = True
        fig.add_trace(
            go.Scatter(
                x=[s["outdeg_frac"] for s in sps], y=[s["fire_frac"] for s in sps],
                mode="markers",
                marker=dict(color=PAIR_COLORS[bi % len(PAIR_COLORS)], size=[12 + s["outdeg"] / 12 for s in sps],
                            line=dict(width=1, color="white")),
                text=[f"B{b['block']}:{s['global']} {_wrap(s.get('label', ''))}<br>"
                      f"{s['outdeg']} children, fires {100*s['fire_frac']:.0f}%" for s in sps],
                hovertemplate="%{text}<extra></extra>",
                name=f"B{b['block']} superparents", legendgroup=f"sp{b['block']}",
            ), row=2, col=1)
    if not any_sp:
        fig.add_annotation(text="no in-block superparents", row=2, col=1,
                           showarrow=False, font=dict(color=INK))
    fig.update_xaxes(title_text="out-degree fraction of block", row=2, col=1)
    fig.update_yaxes(title_text="firing rate", row=2, col=1)

    fig.update_layout(
        title=dict(text=_titled(
            "In-block (same-level) edges",
            "Directed parent→child WITHIN a block (asymmetric containment); "
            "co-extensive pairs are duplicates, never edges.",
            stats={"total_tokens": report["total_tokens"]},
            subtitle=page_subtitle({"total_tokens": report["total_tokens"]})),
            x=0.01, xanchor="left", font=dict(size=15, color=PURPLE)),
        font=FONT, paper_bgcolor="white", plot_bgcolor="white",
        barmode="group", showlegend=True,
        legend=dict(orientation="h", y=-0.08, x=0.01),
        width=1150, height=820, margin=dict(l=60, r=60, t=140, b=80),
    )
    return fig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-block", action="store_true",
                    help="visualise in-block same-level edges (needs in_block_edges.json)")
    ap.add_argument("--calibration", action="store_true",
                    help="visualise the synthetic-toy metric calibration (no cache needed)")
    ap.add_argument("--qualitative", action="store_true",
                    help="visualise real survivor-vs-rejected edges (needs qualitative_check.json)")
    ap.add_argument("--trained-calibration", action="store_true",
                    help="Tier-2: edge recovery on a trained toy SAE (needs trained_toy_calibration.json)")
    args = ap.parse_args()
    if args.calibration:
        run_calibration()
        return
    if args.qualitative:
        run_qualitative()
        return
    if args.trained_calibration:
        run_trained_calibration()
        return
    if args.in_block:
        if not C.IN_BLOCK_PATH.exists():
            raise SystemExit(f"missing {C.IN_BLOCK_PATH} - run in_block_edges.py first")
        fig = build_in_block_dashboard(json.loads(C.IN_BLOCK_PATH.read_text()))
        path = C.RUN_DIR / "in_block_dashboard.html"
        write_page(fig, path, up=2)
        print(f"saved: {path}")
        return

    if not C.EXP0_STATS_PATH.exists():
        raise SystemExit(C.missing_stats_msg())
    stats = torch.load(C.EXP0_STATS_PATH, weights_only=False)
    pairs = stats["pairs"]
    pairs_data = [compute_pair(stats, p, c) for (p, c) in pairs]

    # Attach per-edge S_res verdicts (second pass) so the Sankey colours by genuine
    # refinement, not the weak run_metrics contribution filter. Absent -> Sankey
    # falls back to the recon+frequency colouring (older caches / no second pass).
    if C.SECOND_PASS_PATH.exists():
        second = json.loads(C.SECOND_PASS_PATH.read_text())
        for pd_ in pairs_data:
            sp_entry = second.get(pd_["key"], {}).get("sres", {})
            p0, c0 = pd_["p0"], pd_["c0"]
            mat = torch.zeros_like(pd_["edge_mask"], dtype=torch.bool)
            for e in sp_entry.get("edges", []):
                if e["pass"]:
                    mat[e["parent"] - p0, e["child"] - c0] = True
            pd_["sres_pass"] = mat
            pd_["n_pmi"] = sp_entry.get("n_edges_scored")   # PMI>0 shortlist size
            pd_["n_sres"] = sp_entry.get("n_pass")          # pass probe-S_res

    feat_labels = C.load_feature_labels()
    if not feat_labels:
        print("note: outputs/feature_labels.json not found - run fetch_labels.py "
              "to show descriptions instead of bare indices")

    dash = build_dashboard(pairs_data, feat_labels, stats=stats)
    dash_path = C.RUN_DIR / "metrics_dashboard.html"
    write_page(dash, dash_path, up=2)
    print(f"saved: {dash_path}")

    # One Sankey per block pair that has a superparent, stacked in a single file.
    sk = build_all_superparent_sankeys(stats, pairs, pairs_data, feat_labels=feat_labels)
    if sk is None:
        print("note: no superparent found in any block pair - skipping sankey")
    else:
        # Sankeys only: _add_sres_legend puts two off-canvas Scatter markers in
        # the same figure, so len(sk.data) reported two panels more than there are.
        n_panels = sum(1 for t in sk.data if t.type == "sankey")
        sk_path = C.RUN_DIR / "superparent_sankey.html"
        write_page(sk, sk_path, up=2)
        print(f"saved: {sk_path}  ({n_panels} block pair{'s' if n_panels != 1 else ''})")


if __name__ == "__main__":
    main()
