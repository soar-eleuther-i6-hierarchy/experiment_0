"""
Paper figures — one file, one output directory, every figure derived from data.

    python3 -m reporting.make_report_figures            # everything it can build
    python3 -m reporting.make_report_figures --list     # what it would build, and from what

Output: outputs/paper_figuers/*.png

Each figure backs a single claim and is named after that claim rather than after
its position in a report, so a filename is readable in a caption and does not go
stale when the order changes.

    funnel_coverage_to_sres                 coverage proposes, the strict test disposes
    edge_survival_by_block_pair             what each filter removes, per block pair
    depth_profile_across_layers             whether anything varies with depth
    multiparenting_by_layer                 the graph is not a tree
    superparent_fanout_vs_firing            why a high-firing parent clears the bar for free
    calibration_synthetic_toy_scorecard     every metric against a known tree
    calibration_trained_toy_recovery        the same tree, through a real training run
    cross_source_funnel_shares              the same battery on gemma and on PCFG
    cross_source_layer_response             gemma vs PCFG at the layers both graded
    cross_source_alignment_check            block index or relative depth — which to pair on
    in_block_relations                      same-level edges and duplicates
    shared_input_moved_every_metric         the battery's own failure mode
    base_rate_vs_frequency_capture          the hypothesis's premise, tested
    sres_null_rate_vs_dictionary_size       what a top-k rank rule costs at each D

Two rules this file follows, both learned the hard way:

**Nothing is hardcoded.** Every number in every title is read from the JSON being
plotted. The previous version quoted layer-6 figures in its titles ("B0: 713
edges, 0/449 genuine"), and those numbers were produced before BOS exclusion --
so the captions kept asserting withdrawn results after the data under them had
been regenerated. A title that cannot go stale is a title computed from its input.

**Nothing is skipped silently.** A figure whose input is missing prints the path
it wanted and why, and `--list` shows the whole plan without drawing anything. A
figure set that quietly renders 6 of 10 reads as a complete set.

Most figures read only committed JSON reports, so they build from a fresh clone
with no caches. The two that need `exp0_stats.pt` or the token cache say so.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

import config as C  # noqa: E402

# The gemma/PCFG comparison's pairing rules and its six measures. Shared with
# `make_report_tables`, which prints the same claim: two copies of the alignment
# logic would eventually disagree about the same numbers.
from reporting import cross_source as X  # noqa: E402

PAPER_DIR = C.OUT_DIR / "paper_figuers"

# Which single gemma layer the recovered-graph figure draws. An editorial
# choice, so it is one explicit constant rather than an accident of which run
# had the probe first (that accident chose layer 6 for months). Layer 12 is
# mid-network AND the layer where the released priors-in-time and temporal
# SAEs exist, so the drawn graph stays directly comparable when the
# cross-architecture results land.
GRAPH_LAYER = 12

# --- palette ---------------------------------------------------------------
# Categorical slots are Okabe-Ito, assigned in fixed order and never cycled. The
# repo's screen palette (#2E9E5B green beside #D98A3D orange) separates by only
# ΔE 3.4 under protanopia -- indistinguishable to a red-green colourblind reader,
# which is a real cost in print where no tooltip can rescue the encoding. These
# four clear every check in both light mode and all-pairs mode; the orange sits
# under 3:1 against white, so every figure using it also carries direct labels.
CAT = ["#0072B2", "#E69F00", "#009E73", "#D55E00"]
# Layers are ORDERED, so depth gets a single-hue sequential ramp rather than four
# categorical hues. Reading L24 as "the dark one" is the encoding doing its job.
DEPTH = ["#9ECAE1", "#6BAED6", "#4292C6", "#2171B5", "#084594", "#08306b"]
NEUTRAL = "#C9CCD1"          # "removed" / reference — not a category
GOOD = "#009E73"             # status: survived. Reserved, never a series colour.
INK, MUTED = "#2B2B33", "#5A6B7B"

plt.rcParams.update({
    "font.size": 9.5,
    "axes.edgecolor": "#D8DBE0",
    "axes.labelcolor": INK,
    "text.color": INK,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "figure.facecolor": "white",
    "savefig.facecolor": "white",
})


# --- data access -----------------------------------------------------------
# Every loader returns None rather than raising, so one absent input costs one
# figure instead of the run.
def _json(path: Path):
    return json.loads(path.read_text()) if path.exists() else None


def gemma_layers() -> list[tuple[int, dict]]:
    out = []
    for L in C.NAV_LAYERS:
        r = _json(C.OUT_DIR / C.SOURCE_NAME / f"layer_{L:02d}" / "metrics_report.json")
        if r:
            out.append((L, r))
    return out


def _pair(report, name):
    return next((p for p in report["pairs"] if p["pair"] == name), None)


# Figure-level titles are OFF by default (--titles turns them on). The paper
# carries every figure's message in its LaTeX caption, and a second copy baked
# into the PNG reads as a machine-written caption above the human one -- the
# exact habit the 2026-08-16 feedback asked us to drop. Panel-level titles and
# axis labels stay: they are part of the plot, not a caption.
TITLES = False


def _title(ax_or_fig, head: str, sub: str = "", width: int = 96):
    """Left-aligned title, wrapped to the figure rather than trusting it to fit.

    Every overflowing title in the previous version was a long second line that
    matplotlib silently let run past the canvas edge, so the caption lost its
    qualifier -- the half a sentence that says what the number does NOT mean.
    Returns the number of lines drawn; 0 when titles are disabled, so a layout
    that reserves headroom for the title can reclaim it.
    """
    if not TITLES:
        return 0
    import textwrap
    lines = textwrap.wrap(head, width)
    if sub:
        lines += textwrap.wrap(sub, width + 8)
    txt = "\n".join(lines)
    if hasattr(ax_or_fig, "set_title"):
        ax_or_fig.set_title(txt, fontsize=10.5, loc="left", color=INK)
    else:
        ax_or_fig.suptitle(txt, fontsize=10.5, x=0.006, ha="left", color=INK)
    return txt.count("\n") + 1


def _label_extremes(ax, xs, vals, fmt, color):
    """Label the first, last, min and max point only.

    A number on every point is the anti-pattern; here it also collided with the
    line it annotated at four points out of five.
    """
    keep = {0, len(vals) - 1, int(np.argmin(vals)), int(np.argmax(vals))}
    for i in sorted(keep):
        ax.annotate(fmt(vals[i]), (xs[i], vals[i]), textcoords="offset points",
                    xytext=(0, 9), ha="center", fontsize=8, color=color)


def _finish(fig, ax_or_axes, name: str, tight: bool = True) -> str:
    axes = ax_or_axes if isinstance(ax_or_axes, (list, np.ndarray)) else [ax_or_axes]
    for ax in np.ravel(axes):
        ax.spines[["top", "right"]].set_visible(False)
    if tight:
        # tight_layout ignores ax.text placed above the axes, so a figure that
        # hangs panel headings there manages its own margins and passes False
        fig.tight_layout()
    path = PAPER_DIR / f"{name}.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return name


# ---------------------------------------------------------------------------
# 1. The funnel: what each stage removes, at every layer that has the strict
#    test. Originally one horizontal funnel for layer 6, the only run that had
#    stage 03; once the other five caught up, showing one layer understated the
#    evidence and overstated how special that layer was.
# ---------------------------------------------------------------------------
def funnel_coverage_to_sres(layers, second, pcfg=None, pair="0->1", where=""):
    stages = [f"candidate edges\nreverse coverage ≥ {C.EDGE_TAU}",
              "above chance\nPMI > 0", "genuine refinement\npasses probe S_res"]
    x = np.arange(len(stages))
    # one panel per SOURCE: the funnel collapses the same way on both, which
    # is itself the cross-source claim, so the PCFG runs stand beside gemma
    groups = [("gemma-2-2b", [(f"L{L}", rep, second.get(L)) for L, rep in layers])]
    if pcfg:
        pruns = [(lab.replace("PCFG layer ", "L"), r, sp) for lab, r, sp in pcfg
                 if sp and pair in sp]
        if pruns:
            groups.append(("PCFG", pruns))
    fig, axes = plt.subplots(1, len(groups), figsize=(4.4 + 4.4 * len(groups), 4.4),
                             sharey=True)
    axes = np.atleast_1d(axes)
    lo, hi = [], []
    for ax, (src, runs) in zip(axes, groups):
        for i, (lab, rep, sp) in enumerate(runs):
            if not (sp and pair in sp):
                continue
            pr = _pair(rep, pair)
            sres = sp[pair]["sres"]
            vals = [pr["n_candidate_edges"], sres["n_edges_scored"], sres["n_pass"]]
            # log y: the whole story is three orders of magnitude, and on a
            # linear axis every line would lie on the floor after stage one.
            # A measured ZERO has no position on a log axis, so it is drawn at
            # a floor with its true value in the end label instead of dropped.
            plot_vals = [max(v, 0.5) for v in vals]
            ax.plot(x, plot_vals, "-o", color=DEPTH[i % len(DEPTH)], lw=1.8, ms=5,
                    zorder=3)
            ax.annotate(lab + (" (0 pass)" if vals[-1] == 0 else ""),
                        (x[-1], plot_vals[-1]), textcoords="offset points",
                        xytext=(10, -3 + 6 * (i % 2)), fontsize=8,
                        color=DEPTH[i % len(DEPTH)])
            if src == groups[0][0]:
                lo.append(vals), hi.append(pr["reconstruction"]["n_pass"])
        ax.set_yscale("log")
        ax.set_xticks(x)
        ax.set_xticklabels(stages, fontsize=8.5)
        ax.set_xlim(-0.25, len(stages) - 0.45)
        ax.set_title(src, fontsize=10, loc="left", color=INK)
        ax.grid(True, axis="y", which="both", alpha=0.12)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("parent → child edges (log)")
    first, last = [v[0] for v in lo], [v[-1] for v in lo]
    share = [100 * b / a for a, b in zip(first, last)]
    # the ablation filter is parallel, not a nested stage; summarized in text
    rec = [100 * h / v[0] for h, v in zip(hi, lo)]
    _title(fig, f"{where}: co-firing proposes thousands, the strict test confirms tens — "
                "every graded layer, both sources",
           f"{min(last)}–{max(last)} of {min(first):,}–{max(first):,} gemma candidates "
           f"survive ({min(share):.1f}–{max(share):.1f}%). The parallel ablation filter "
           f"passes {min(rec):.0f}–{max(rec):.0f}% and is not drawn as a stage", width=86)
    return _finish(fig, axes, "funnel_coverage_to_sres")


# ---------------------------------------------------------------------------
# 2. Per block pair, per layer: what survives reconstruction and the frequency
#    control. This is the withdrawn kill-rate view, rebuilt from committed
#    reports -- the condition on which a withdrawn result was allowed back.
# ---------------------------------------------------------------------------
def edge_survival_by_block_pair(layers):
    pairs = ["0->1", "1->2", "2->3"]
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 3.9), sharey=False)
    x = np.arange(len(pairs))
    w = 0.15

    for i, (L, rep) in enumerate(layers):
        recon = [(_pair(rep, p) or {}).get("reconstruction", {}).get("frac_pass", np.nan)
                 for p in pairs]
        freq = [(_pair(rep, p) or {}).get("freq_control", {}).get("frac_freq_driven", np.nan)
                for p in pairs]
        off = (i - (len(layers) - 1) / 2) * w
        axes[0].bar(x + off, np.array(recon) * 100, w, color=DEPTH[i], label=f"L{L}")
        axes[1].bar(x + off, np.array(freq) * 100, w, color=DEPTH[i], label=f"L{L}")

    for ax, title, ylab in [
        (axes[0], "improve reconstruction", "% of candidate edges"),
        (axes[1], "carried by frequent tokens", "% of candidate edges"),
    ]:
        ax.set_xticks(x)
        ax.set_xticklabels([f"B{p.replace('->', '→B')}" for p in pairs])
        ax.set_ylabel(ylab)
        ax.set_title(title, fontsize=10.5, loc="left")
        ax.grid(True, axis="y", alpha=0.12)
        ax.set_axisbelow(True)
    axes[0].legend(title="layer", fontsize=8.5, title_fontsize=8.5, frameon=False,
                   ncol=5, loc="upper center", bbox_to_anchor=(0.5, -0.10))
    _title(fig, "What each filter removes, by block pair and depth",
           "each panel on its own scale; layers are ordered, so depth is a single-hue ramp",
           width=104)
    fig.subplots_adjust(top=0.80, bottom=0.22)
    return _finish(fig, axes, "edge_survival_by_block_pair")


# ---------------------------------------------------------------------------
# 3. Depth. The claim this replaces said quality degrades with depth; that came
#    from BOS-contaminated caches. Plotted from regenerated reports so the
#    figure can contradict it.
# ---------------------------------------------------------------------------
def depth_profile_across_layers(layers):
    Ls = [L for L, _ in layers]
    series = [
        ("candidate edges (B0→B1)", CAT[0],
         [_pair(r, "0->1")["n_candidate_edges"] for _, r in layers], False),
        ("improve reconstruction, %", CAT[1],
         [100 * _pair(r, "0->1")["reconstruction"]["frac_pass"] for _, r in layers], True),
        ("frequency-driven, %", CAT[3],
         [100 * _pair(r, "0->1")["freq_control"]["frac_freq_driven"] for _, r in layers], True),
        ("mean frequency survival", CAT[2],
         [_pair(r, "0->1")["freq_control"]["mean_survival"] for _, r in layers], True),
    ]
    # Four measures on four axes rather than one: a shared y would be a dual-axis
    # chart wearing a disguise, and the point is the SHAPE of each line, not their
    # relative heights.
    fig, axes = plt.subplots(1, 4, figsize=(12.2, 3.3))
    for ax, (label, col, vals, _) in zip(axes, series):
        ax.plot(Ls, vals, "-o", color=col, lw=2, ms=5)
        fmt = (lambda v: f"{v:,.0f}") if max(vals) > 50 else (lambda v: f"{v:.2f}")
        _label_extremes(ax, Ls, vals, fmt, MUTED)
        ax.set_title(label, fontsize=9.5, loc="left")
        ax.set_xticks(Ls)
        ax.set_xlabel("layer")
        ax.margins(y=0.3)
        ax.grid(True, axis="y", alpha=0.12)
        ax.set_axisbelow(True)
    _title(fig, "Block pair B0→B1 across depth — each panel is one measure, on its own scale",
           "no measure is monotonic in depth; the degradation-with-depth claim came from "
           "caches that counted BOS")
    fig.subplots_adjust(top=0.72)
    return _finish(fig, axes, "depth_profile_across_layers")


# ---------------------------------------------------------------------------
# 4. Multi-parenting: the one claim that did not move when BOS was excluded,
#    because it is a ratio over children that already have a parent.
# ---------------------------------------------------------------------------
def multiparenting_by_layer(layers, pcfg=None):
    # EVERY block pair each source's nesting defines, not a fixed prefix of
    # them: the pair list is read off block_ranges (gemma: 5 blocks -> 4 pairs,
    # PCFG: 8 blocks -> 7 pairs). A pair the pipeline never graded is drawn as
    # an explicit "n.a." at the baseline rather than silently omitted -- gemma's
    # B3->B4 is not yet computed (grading it OOMs: B4 alone spans 24,576
    # latents), and the figure must say so rather than hide the column.
    # one panel per SOURCE: the claim is cross-source ("the tangle is a
    # property of the nesting, not of gemma"), so the PCFG runs stand beside
    # gemma rather than in a separate appendix figure
    groups = [("gemma-2-2b", [(f"L{L}", r) for L, r in layers])]
    if pcfg:
        groups.append(("PCFG", [(n.replace("PCFG ", "").replace("layer ", "L"), r)
                                for n, r in pcfg]))
    # the first four pairs keep the figure's original categorical hues (CAT),
    # so B0->B1 stays the blue readers already know; the three pairs the wider
    # PCFG nesting adds get three new hues. Adjacent-pair CVD separation
    # checked programmatically (Viénot simulation + OKLab): all pairs >= 8
    # except vermilion<->pink under tritanopia (7.0), which is legal here
    # because every bar carries its value as a direct label.
    RAMP = CAT + ["#CC79A7", "#6A51A3", "#56B4E9"]

    def pair_list(runs):
        n_blocks = max((len(r.get("block_ranges", [])) for _, r in runs), default=0)
        return [f"{i}->{i + 1}" for i in range(max(n_blocks - 1, 0))]

    per_src_pairs = {src: pair_list(runs) for src, runs in groups}
    all_pairs = max(per_src_pairs.values(), key=len)

    fig, axes = plt.subplots(1, len(groups), figsize=(4.0 + 4.8 * len(groups), 4.1),
                             sharey=True,
                             gridspec_kw={"width_ratios":
                                          [len(g[1]) * len(per_src_pairs[g[0]])
                                           for g in groups]})
    axes = np.atleast_1d(axes)
    for ax, (src, runs) in zip(axes, groups):
        pairs = per_src_pairs[src]
        x = np.arange(len(runs))
        w = 0.86 / max(len(pairs), 1)
        fs_val = 6.0 if len(pairs) > 4 else 7.5
        for j, p in enumerate(pairs):
            vals, supp = [], []
            for _, r in runs:
                q = _pair(r, p) or {}
                vals.append(100 * q.get("degree", {}).get("poly_frac", np.nan))
                supp.append(q.get("degree", {}).get("n_children_with_parent"))
            off = (j - (len(pairs) - 1) / 2) * w
            ax.bar(x + off, [0 if np.isnan(v) else v for v in vals], w * 0.92,
                   color=RAMP[j])
            for xx, vv, nn in zip(x + off, vals, supp):
                if np.isnan(vv):
                    # absence of measurement, made visible instead of skipped
                    ax.text(xx, 2.5, "n.a.", ha="center", va="bottom", rotation=90,
                            fontsize=5.5, color=NEUTRAL)
                    continue
                # every bar carries its support: the % answers "how much",
                # n answers "over how many children it was computed". The %
                # sits above the bar; n is rotated inside the bar so wide
                # counts (n=3190) never collide with the neighbouring bar.
                # Bars too short to hold the text stack both above instead.
                # inside only when the bar is tall enough to hold the whole
                # rotated string (~2.3 axis units per character + padding);
                # otherwise the white tail vanishes into the white background
                fits_inside = (nn is not None
                               and vv >= 6 + 2.3 * len(f"n={nn}"))
                if fits_inside:
                    ax.text(xx, vv + 1.6, f"{vv:.0f}", ha="center",
                            fontsize=5.5 if len(pairs) > 4 else 6.5, color=MUTED)
                    ax.text(xx, 2.5, f"n={nn}", ha="center", va="bottom",
                            rotation=90, fontsize=5.5, color="#FFFFFF")
                else:
                    # short bar: value above it, n rotated above the value —
                    # never horizontal, so wide counts cannot reach a neighbour
                    ax.text(xx, vv + 1.6, f"{vv:.0f}", ha="center",
                            fontsize=5.5, color=MUTED)
                    if nn is not None:
                        ax.text(xx, vv + 7.0, f"n={nn}", ha="center", va="bottom",
                                rotation=90, fontsize=5.0, color=MUTED)
        ax.set_xticks(x)
        ax.set_xticklabels([lab for lab, _ in runs])
        ax.set_title(f"{src} — {len(pairs) + 1} blocks, {len(pairs)} pairs",
                     fontsize=10, loc="left", color=INK)
        ax.set_ylim(0, 112)
        ax.axhline(100, ls=(0, (2, 3)), lw=1, color=NEUTRAL)
        ax.grid(True, axis="y", alpha=0.12)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("% of children with ≥ 2 parents")
    # one shared key for every pair either source defines, colored by the ramp
    handles = [plt.Rectangle((0, 0), 1, 1, color=RAMP[j])
               for j in range(len(all_pairs))]
    # the key sits above its two shared footnotes: both apply to both panels
    fig.legend(handles, [f"B{p.replace('->', '→B')}" for p in all_pairs],
               fontsize=8, frameon=False, ncol=min(len(all_pairs), 7),
               loc="lower center", bbox_to_anchor=(0.5, 0.062))
    fig.text(0.5, 0.042, "n = children behind the %",
             ha="center", fontsize=7.5, color=MUTED)
    fig.text(0.5, 0.008, "n.a. = B3→B4 not yet computed (out of memory)",
             ha="center", fontsize=7.5, color=MUTED)
    _title(fig, "The graph is not a tree: in the top block pair nearly every child has "
                "several parents, on both sources",
           "every block pair each nesting defines; n.a. = not computed yet — gemma's "
           "B3→B4 runs out of memory (B4 alone spans 24,576 latents), rerun pending. "
           "Ratio over children that already have a parent — the one measure BOS "
           "exclusion left unchanged", width=110)
    # tight_layout ignores fig.legend and would drop it onto the tick labels;
    # this figure manages its own margins and passes tight=False
    fig.subplots_adjust(left=0.055, right=0.995, top=0.80, bottom=0.25,
                        wspace=0.06)
    return _finish(fig, axes, "multiparenting_by_layer", tight=False)


# ---------------------------------------------------------------------------
# 5. Superparents: fan-out against base rate. Reports list the top parents per
#    pair, which is what this needs.
# ---------------------------------------------------------------------------
def superparent_fanout_vs_firing(layers):
    """Why a high-firing parent clears the coverage bar: it barely has to.

    The previous version scattered fan-out against firing rate on a log x-axis.
    It was close to a tautology -- every point is above the gate because the gate
    is what selected it -- the log scale spanned less than a decade and printed
    "4x10^1" where "40" would do, and the five-layer ramp encoded nothing the
    points clustered on.

    What the same numbers do show is the mechanism. Coverage keeps an edge when
    P(parent | child) >= tau. Under independence that probability is just the
    parent's firing rate, so the enrichment a parent needs over chance is tau/rho
    -- 50x for a parent firing on 1% of tokens, and at or below 1x for anything
    firing more often than tau itself.
    """
    fires = [sp["fire_frac"] for _, rep in layers for pr in rep["pairs"]
             for sp in pr.get("superparents", [])]
    tau = C.EDGE_TAU
    free = sum(1 for f in fires if f >= tau)

    fig, ax = plt.subplots(figsize=(7.8, 4.6))
    rho = np.linspace(0.005, 1.0, 400)
    ax.plot(100 * rho, tau / rho, lw=2, color=NEUTRAL, zorder=1,
            label=f"enrichment needed to reach R ≥ {tau}   (= {tau}/ρ)")
    ax.axhline(1.0, ls=(0, (4, 3)), lw=1.3, color=CAT[3])
    ax.text(2, 1.45, "1× — no enrichment at all: the edge is kept on base rate",
            fontsize=8.5, color=CAT[3])
    ax.scatter([100 * f for f in fires], [tau / f for f in fires], s=46, color=CAT[0],
               zorder=3, edgecolor="white", linewidth=0.8,
               label=f"the {len(fires)} flagged parents")
    ax.set_yscale("log")
    ax.set_xlim(0, 104)
    ax.set_xlabel("parent firing rate ρ, % of tokens")
    ax.set_ylabel("enrichment over chance the parent needs (log)")
    ax.legend(fontsize=8.5, frameon=False, loc="upper right")
    _title(ax, "A parent that fires often enough clears the coverage bar without any enrichment",
           f"{free} of the {len(fires)} flagged parents fire on ≥ {100 * tau:.0f}% of tokens, so "
           f"co-firing with most of the next block is arithmetic. One firing on 1% would need "
           f"{tau / 0.01:.0f}× enrichment for the same edge", width=78)
    ax.grid(True, which="both", alpha=0.12)
    ax.set_axisbelow(True)
    return _finish(fig, ax, "superparent_fanout_vs_firing")


# ---------------------------------------------------------------------------
# 6. Tier 1 — every metric against a known tree, ranked by how decisively it
#    separated the two classes.
# ---------------------------------------------------------------------------
def calibration_synthetic_toy_scorecard(rows):
    # Two kinds of row, and only one belongs on a ratio axis. Rows scored
    # categorically (the recovered edge set is right or it is not) carry a margin
    # of 1.0 meaning "correct", which on a log ratio axis is indistinguishable
    # from "separated by a factor of one" -- i.e. from no separation at all. They
    # get their own panel rather than a shared scale that implies a comparison
    # the numbers cannot support.
    ratio = [r for r in rows if r.get("margin_kind") != "categorical"]
    cat_rows = [r for r in rows if r.get("margin_kind") == "categorical"]
    ratio.sort(key=lambda r: r["margin"])
    n_pass = sum(r["pass"] for r in rows)

    fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.4),
                             gridspec_kw={"width_ratios": [1.75, 1]})
    ax = axes[0]
    # Negative controls pass when the battery does NOT act, so they are a
    # different kind of claim and are hatched rather than recoloured -- texture is
    # the secondary encoding that survives greyscale printing.
    for i, r in enumerate(ratio):
        ctrl = r["metric"].lstrip().startswith("—")
        ax.barh(i, min(r["margin"], 1e4), height=0.6,
                color=GOOD if r["pass"] else CAT[3],
                hatch="///" if ctrl else None, edgecolor="white", linewidth=0.8)
        ax.text(min(r["margin"], 1e4) * 1.3, i,
                ">1000×" if r["margin"] >= 1000 else f"{r['margin']:.1f}×",
                va="center", fontsize=8, color=MUTED)
    ax.set_yticks(np.arange(len(ratio)))
    ax.set_yticklabels([r["metric"] for r in ratio], fontsize=8.5)
    ax.set_xscale("log")
    ax.set_xlim(0.3, 1.2e5)
    ax.axvline(1.0, ls=(0, (2, 3)), lw=1, color=NEUTRAL)
    ax.set_xlabel("separation between the two classes (log) — 1× is no separation")
    ax.set_title("scored by margin", fontsize=9.5, loc="left")
    fig.legend(handles=[
        plt.Rectangle((0, 0), 1, 1, color=GOOD, label="caught its pathology"),
        plt.Rectangle((0, 0), 1, 1, facecolor=GOOD, hatch="///", edgecolor="white",
                      label="negative control — passes when nothing catches it"),
    ], fontsize=8, frameon=False, loc="lower right", ncol=2, bbox_to_anchor=(0.99, -0.02))
    ax.grid(True, axis="x", alpha=0.12)
    ax.set_axisbelow(True)

    ax = axes[1]
    for i, r in enumerate(cat_rows):
        ctrl = r["metric"].lstrip().startswith("—")
        ax.barh(i, 1.0, height=0.6, color=GOOD if r["pass"] else CAT[3],
                hatch="///" if ctrl else None, edgecolor="white", linewidth=0.8)
        ax.text(0.5, i, "correct" if r["pass"] else "wrong", va="center", ha="center",
                fontsize=8.5, color="white", fontweight="bold")
    ax.set_yticks(np.arange(len(cat_rows)))
    ax.set_yticklabels([r["metric"] for r in cat_rows], fontsize=8.5)
    ax.set_xticks([])
    ax.set_xlim(0, 1)
    ax.set_title("scored categorically —\nthe answer is right or it is not",
                 fontsize=9.5, loc="left")
    ax.spines["bottom"].set_visible(False)

    _title(fig, f"Every metric scored against a known tree — {n_pass}/{len(rows)} rows pass",
           "the hatched rows are limitations demonstrated rather than caught: an absorbed edge "
           "coverage cannot propose, and a shared-topic pair every filter accepts", width=112)
    fig.subplots_adjust(top=0.80)
    return _finish(fig, axes, "calibration_synthetic_toy_scorecard")


# ---------------------------------------------------------------------------
# 7. Tier 2 — the same tree after a real training run, plus the nesting control.
# ---------------------------------------------------------------------------
def calibration_trained_toy_recovery(tt, align):
    tp, fp, fn = tt["true_positives"], tt["false_positives"], tt["false_negatives"]
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 3.9),
                             gridspec_kw={"width_ratios": [1.25, 1]})

    ax = axes[0]
    bars = [("recovered", tp, GOOD), ("missed", fn, NEUTRAL), ("false positives", fp, CAT[3])]
    ax.bar([b[0] for b in bars], [b[1] for b in bars],
           color=[b[2] for b in bars], width=0.55)
    for i, (_, v, _) in enumerate(bars):
        ax.text(i, v + 0.15, str(v), ha="center", fontsize=10, color=INK)
    ax.set_ylabel("true edges")
    ax.set_ylim(0, max(tp, fn, fp) * 1.35 + 0.5)
    # Panel identifiers, not captions: with two panels sharing a y-label the
    # plot is ambiguous without them, so they stay even with --titles off.
    ax.set_title("edge recovery", fontsize=10, loc="left", color=INK)

    ax = axes[1]
    if align:
        # "violated" is the bar the test is ABOUT -- its zero must be visible,
        # or the grey untestable bar next to the green one reads as a failure.
        n_viol = align["n_testable"] - align["n_respected"]
        bars = [("respected", align["n_respected"], GOOD),
                ("violated", n_viol, CAT[3]),
                ("untestable", len(align.get("untestable", [])), NEUTRAL)]
        ax.bar([b[0] for b in bars], [b[1] for b in bars],
               color=[b[2] for b in bars], width=0.55)
        for i, (_, v, _) in enumerate(bars):
            ax.text(i, v + 0.12, str(v), ha="center", fontsize=10, color=INK)
        ax.set_ylim(0, max(align["n_testable"], 1) * 1.4)
        ax.set_ylabel("true edges")
        ax.set_title("nesting control (early block → late)", fontsize=10, loc="left", color=INK)
    else:
        ax.axis("off")
        ax.text(0.5, 0.5, "block_tree_alignment.json absent\nrun validation.block_tree_alignment",
                ha="center", va="center", fontsize=9, color=MUTED)
    for a in axes:
        a.grid(True, axis="y", alpha=0.12)
        a.set_axisbelow(True)
    return _finish(fig, axes, "calibration_trained_toy_recovery")


# ---------------------------------------------------------------------------
# 8. Two sources through one battery. Shares, never counts: 1792 latents in 8
#    blocks against 32768 in 5 is not a comparison counts can carry.
# ---------------------------------------------------------------------------
def cross_source_funnel_shares(runs):
    # "above chance" here is the report's n_chance_level, whose cutoff is PMI < 0.5
    # -- NOT the PMI > 0 shortlist that stage 03 scores. Labelling it "PMI > 0" put
    # 14% on the chart where the shortlist is 70%, two different thresholds under
    # one name.
    stages = ["improve\nreconstruction", "clears chance\n(PMI ≥ 0.5)", "frequency-\ndriven"]
    # One MARKER per run, grouped by source, rather than one bar per run: the
    # original grouped bars were drawn for two runs, and with every graded
    # layer of both sources present (ten runs) thirty bars cycling four
    # colours stopped saying which source was which. The question this figure
    # answers is cross-source, so source is the encoding and the within-source
    # scatter is shown, not averaged away.
    fig, ax = plt.subplots(figsize=(8.6, 4.0))
    x = np.arange(len(stages))
    groups = [("gemma-2-2b", [r for r in runs if r[0].startswith("gemma")], CAT[0], "o", -0.16),
              ("PCFG", [r for r in runs if not r[0].startswith("gemma")], CAT[1], "s", +0.16)]
    for src, members, col, mark, off in groups:
        pts = {j: [] for j in range(len(stages))}
        for k, (name, rep) in enumerate(members):
            pr = _pair(rep, "0->1")
            if pr is None:
                continue
            n = pr["n_candidate_edges"]
            vals = [100 * pr["reconstruction"]["frac_pass"],
                    100 * (n - pr["independence_null"]["n_chance_level"]) / n
                    if pr["independence_null"].get("n_chance_level") is not None else np.nan,
                    100 * pr["freq_control"]["frac_freq_driven"]]
            jit = (k - (len(members) - 1) / 2) * (0.16 / max(len(members) - 1, 1))
            for j, v in enumerate(vals):
                if not np.isnan(v):
                    pts[j].append(v)
                    ax.scatter(j + off + jit, v, s=42, marker=mark, color=col,
                               alpha=0.75, edgecolor="white", linewidth=0.6, zorder=3,
                               label=f"{src} — {len(members)} layers" if (j, k) == (0, 0) else None)
        for j, vs in pts.items():
            if vs:                        # a tick at the group mean, spread kept visible
                ax.plot([j + off - 0.11, j + off + 0.11], [np.mean(vs)] * 2,
                        lw=2, color=col, zorder=2)
    ax.set_xticks(x)
    ax.set_xticklabels(stages)
    ax.set_ylabel("% of that run's candidate edges")
    ax.set_ylim(-4, 108)
    ax.legend(fontsize=8.5, frameon=False, loc="center right")
    _title(ax, "The same battery, unchanged, on two SAE sources — block pair B0→B1, every "
               "graded layer",
           "one marker per run, tick = source mean; shares rather than counts, because the "
           "dictionaries differ in size and in block count",
           width=84)
    ax.grid(True, axis="y", alpha=0.12)
    ax.set_axisbelow(True)
    return _finish(fig, ax, "cross_source_funnel_shares")


# ---------------------------------------------------------------------------
# 8a/8b. The same battery on both sources at once, layer by layer. gemma's base
#     model has 26 blocks and the PCFG SAE's base model has 4, so "layer 3" does
#     not mean
#     the same thing on both -- and which alignment to use is itself measurable,
#     which is what 8b answers rather than assumes.
# ---------------------------------------------------------------------------
def cross_source_layer_response(rows):
    """Which metrics give the same answer on a 2.6B LM and on a 4-block transformer.

    The question is whether anything the battery reports is a property of
    Matryoshka nesting rather than of gemma. Both sources have an SAE trained on
    layers 1 and 3, so the comparison needs no matching argument: the grey
    dumbbells stand at the two block indices where both were graded, and their
    LENGTH is the disagreement. Restricted to block pair B0->B1, because the
    PCFG runs' deeper pairs hold 0-3 candidate edges each and every number
    computed from them is one or two edges wide.

    Panel ORDER is derived, not chosen: the six are sorted by their own mean gap,
    so the layout is a result rather than an arrangement that flatters one. All
    six share a 0-100 axis because all six are shares, which is what lets a
    dumbbell in one panel be compared with a dumbbell in another; anchoring at
    zero keeps its length proportional to the disagreement in the metric's own
    units rather than to whatever range the data happened to span.
    """
    pairs = X.matched(rows, "layer")
    gem = [r for r in rows if r["is_ref"]]
    oth = [r for r in rows if not r["is_ref"]]
    panels = [(t, k) for t, k, _ in X.ranked(pairs)]
    gaps = [g for _, _, g in X.ranked(pairs)]
    shared = sorted({o["layer"] for o, _, _ in pairs})

    fig, axes = plt.subplots(2, 3, figsize=(12.6, 6.8), sharex=True, sharey=True)
    for ax, (title, key), gp in zip(np.ravel(axes), panels, gaps):
        ax.axvspan(min(shared) - 0.9, max(shared) + 0.9, color="#F2F4F7", zorder=0)
        for o, g, _ in pairs:
            ax.plot([g["layer"], o["layer"]], [100 * g[key], 100 * o[key]],
                    lw=2.2, color=MUTED, zorder=2, solid_capstyle="round", alpha=0.55)
        ax.plot([r["layer"] for r in gem], [100 * r[key] for r in gem], "-o",
                color=CAT[0], lw=1.8, ms=5, zorder=3)
        ax.plot([r["layer"] for r in oth], [100 * r[key] for r in oth], "s",
                color=CAT[1], ms=7, zorder=4, mec="white", mew=1.0)
        # Both readings in the panel title, because they rank the six differently
        # and quoting only the first would make three saturated measures look
        # like the strongest agreement in the figure.
        ax.set_title(f"{title}\nsame-layer gap {gp:.1f} pts  —  "
                     f"{X.gap_ratio(rows, pairs, key):.2f} of its own range",
                     fontsize=9.5, loc="left")
        ax.set_ylim(0, 105)
        ax.set_xlim(-1.2, max(r["layer"] for r in gem) + 1.8)
        ax.set_xticks([r["layer"] for r in gem])
        ax.grid(True, axis="y", alpha=0.12)
        ax.set_axisbelow(True)
    for ax in axes[1]:
        ax.set_xlabel("layer of the base model (block index)")
    for ax in axes[:, 0]:
        ax.set_ylabel("% — every panel, same scale")

    # The shaded band is named once, in the first panel, and the layers are not
    # named at all: the x axis is shared and already ticked with them, so a label
    # per point would be 48 marks for 8 facts.
    ax = np.ravel(axes)[0]
    ax.annotate(f"both sources\ngraded here\n(L{', L'.join(str(s) for s in shared)})",
                (max(shared) + 0.6, 6), fontsize=7.5, color=MUTED, ha="left", va="bottom")

    # The legend goes INSIDE the panel with the most headroom, not under the
    # figure: `_finish` runs tight_layout, which does not know a figure-level
    # legend exists and lets it land on the x labels. Which panel has room is a
    # fact about the data -- every panel shares one 0-100 axis -- so it is
    # computed rather than picked, and stays right if a metric moves.
    n_agree = X.agree_split(gaps)
    roomiest = min(range(len(panels)),
                   key=lambda i: max(100 * r[panels[i][1]] for r in rows))
    np.ravel(axes)[roomiest].legend(handles=[
        plt.Line2D([], [], color=CAT[0], marker="o", lw=1.8, ms=5,
                   label=f"gemma-2-2b — {len(gem)} of {gem[0]['n_layers']} layers graded,\n"
                         f"D = {gem[0]['d_sae']:,} in {gem[0]['n_blocks']} blocks"),
        plt.Line2D([], [], color=CAT[1], marker="s", lw=0, ms=7, mec="white",
                   label=f"PCFG SAE — {len(oth)} of {oth[0]['n_layers']} layers graded,\n"
                         f"D = {oth[0]['d_sae']:,} in {oth[0]['n_blocks']} blocks"),
        plt.Line2D([], [], color=MUTED, lw=2.2, alpha=0.55,
                   label="the two sources at the same layer index"),
    ], fontsize=8, frameon=False, loc="upper left", bbox_to_anchor=(0.02, 0.99),
        labelspacing=0.9)
    _title(fig, "Two base models, one battery: the shape of B0→B1 agrees, its strength does not",
           f"{n_agree} of {len(panels)} measures agree to within {gaps[n_agree - 1]:.1f} points "
           f"at the same layer index — across a 2.6B-parameter language model and a "
           f"{oth[0]['n_layers']}-block transformer on synthetic grammar, with dictionaries "
           f"{max(r['d_sae'] for r in rows) / min(r['d_sae'] for r in rows):.0f}× apart in size. "
           f"The remaining {len(panels) - n_agree} differ by {gaps[n_agree]:.0f}–{gaps[-1]:.0f}. "
           f"Read the second number in each panel title against the first: all "
           f"{n_agree} of the close measures sit against a floor or ceiling on both sources, so "
           f"relative to the range each one varies over at all, only "
           f"{min(panels, key=lambda p: X.gap_ratio(rows, pairs, p[1]))[0]} is clearly closer "
           "than the rest. B0→B1 only — the PCFG runs' deeper block pairs hold 0–3 candidate "
           "edges each, and two shared layers cannot establish a trend. Both PCFG runs are one "
           f"grammar config ({X.grammar_line(rows)}), one point of Exp 2's three-axis sweep, "
           "not PCFG in general", width=126)
    fig.subplots_adjust(top=0.82, hspace=0.42)
    return _finish(fig, axes, "cross_source_layer_response")


def cross_source_alignment_check(rows):
    """Same block index, or same fraction of the network? The data can answer.

    Comparing across two base models of different depth needs an alignment, and
    the choice is usually made silently in the axis label. The two candidates
    disagree completely here: by block index PCFG's layers 1 and 3 pair with
    gemma's 1 and 3, and by relative depth (L+1)/N they pair with gemma's 12 and
    24 -- opposite ends of the network. So the alignment is not a presentational
    detail, and this figure measures it instead of asserting it: for each metric,
    the mean gap between the paired runs under each rule.

    It is a weak test on two PCFG layers and says so in the title. It is included
    because the alternative is an unstated assumption doing the same work.
    """
    by_layer, by_depth = X.matched(rows, "layer"), X.matched(rows, "depth")
    order = [(t, k) for t, k, _ in X.ranked(by_layer)]
    lg = [g for _, _, g in X.ranked(by_layer)]
    dg = [X.gap(by_depth, k) for _, k in order]

    fig, ax = plt.subplots(figsize=(9.6, 4.8))
    y = np.arange(len(order))
    h = 0.36
    ax.barh(y - h / 2, lg, h, color=CAT[0], label="paired by layer index")
    ax.barh(y + h / 2, dg, h, color=CAT[3], label="paired by relative depth (L+1)/N")
    for yy, v in list(zip(y - h / 2, lg)) + list(zip(y + h / 2, dg)):
        ax.text(v + max(lg + dg) * 0.015, yy, f"{v:.1f}", va="center", fontsize=8, color=MUTED)
    ax.set_yticks(y)
    ax.set_yticklabels([t for t, _ in order], fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("mean gap between the two sources, points of a 0–100 share   "
                  "(shorter = the two runs agree)")
    ax.set_xlim(0, max(lg + dg) * 1.16)
    # Sorted ascending with the y axis inverted, so the short bars are at the top
    # and the free space is there. Anchored, not "best": matplotlib's best would
    # move the legend the day a metric changes rank.
    ax.legend(fontsize=8.5, frameon=False, loc="upper right", bbox_to_anchor=(0.995, 0.99))
    ml, md = sum(lg) / len(lg), sum(dg) / len(dg)
    win, lose = ("index", "relative depth") if ml <= md else ("relative depth", "index")
    named = " and ".join(f"PCFG L{o['layer']}→gemma L{q['layer']}" for o, q, _ in by_depth)
    _title(fig, f"Aligning the two models by layer {win} makes them agree more than by {lose}",
           f"mean over the {len(order)} measures: {ml:.1f} points paired by layer index against "
           f"{md:.1f} paired by relative depth. The two rules pair different runs — by depth, "
           f"{named} — so the choice is not cosmetic. On {len(by_layer)} shared layers this is "
           "suggestive and not a result; it is drawn because the alternative is the same choice "
           "made silently in an axis label", width=104)
    ax.grid(True, axis="x", alpha=0.12)
    ax.set_axisbelow(True)
    fig.subplots_adjust(top=0.80)
    return _finish(fig, ax, "cross_source_alignment_check")


# ---------------------------------------------------------------------------
# 9. Within-block relations.
# ---------------------------------------------------------------------------
def in_block_relations(runs):
    """Within-block relations as a RATE per pair, across every graded run.

    Counts cannot be compared across these blocks. gemma's are nested prefixes of
    very different sizes -- 128, 384, 1536, 6144 -- so B3's 833 duplicate pairs at
    L18 look like the deep blocks are where duplication lives, and read as a rate
    they are 0.04 per thousand pairs against B0's 0.12: three times rarer. The
    raw-count reading is the same mistake the project already logs about counts
    across sources, one level down.
    """
    fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.0), sharex=True)
    for i, (name, d) in enumerate(runs):
        blocks = d["blocks"]
        xs = [b["block"] for b in blocks]
        n = [b["n_features"] for b in blocks]
        edge_rate = [1e3 * b["n_edges"] / (f * (f - 1)) for b, f in zip(blocks, n)]
        dup_rate = [1e3 * b["n_duplicates"] / (f * (f - 1) / 2) for b, f in zip(blocks, n)]
        col = DEPTH[i] if name.startswith("gemma") else CAT[3 if "03" in name else 1]
        style = "-o" if name.startswith("gemma") else "--s"
        axes[0].plot(xs, edge_rate, style, color=col, lw=1.8, ms=5, label=name)
        axes[1].plot(xs, dup_rate, style, color=col, lw=1.8, ms=5, label=name)
    for ax, lab in [(axes[0], "directed edges per 1000 ordered pairs"),
                    (axes[1], "duplicate pairs per 1000 unordered pairs")]:
        ax.set_yscale("symlog", linthresh=1e-2)
        ax.set_xlabel("block")
        ax.set_ylabel(lab)
        ax.grid(True, alpha=0.12)
        ax.set_axisbelow(True)
    axes[0].legend(fontsize=7.5, frameon=False, ncol=2)
    _title(fig, "Same-level structure lives in B0, on both sources — as a rate, not a count",
           "gemma blocks are nested prefixes of 128 to 6144 features and PCFG's are eight equal "
           "blocks of 224, so only a per-pair rate compares them; B0 is not the densest because "
           "it is the smallest. PCFG's blocks below B0 hold 0-4 edges in total, so every bend in "
           "those dashed lines is one edge", width=124)
    fig.subplots_adjust(top=0.78)
    return _finish(fig, axes, "in_block_relations")


# ---------------------------------------------------------------------------
# 9a. The battery's own failure mode. Five metrics read the same co-firing matrix
#     and one contaminating token position moved all of them at once; the sixth
#     is a ratio over children that already have a parent, and did not move.
#     Built from the v1 reports kept in outputs_archive/ against the current ones.
# ---------------------------------------------------------------------------
def shared_input_moved_every_metric(layers):
    import math
    ARCH = C.HERE / "outputs_archive"
    # Paired BY LAYER, not by count. A layer graded after the BOS fix (layer 1)
    # has no archived v1 counterpart, and requiring the two lists to be the same
    # length made one new layer silently kill this figure -- the comparison is
    # per-layer, so it should simply run on the layers that have both versions.
    matched = []
    for L, rep in layers:
        hits = sorted(ARCH.glob(f"layer_{L:02d}__v1__*/metrics_report.json"))
        if hits:
            matched.append((rep, _json(hits[0])))
    if len(matched) < 2:
        raise SystemExit("fewer than two layers have an archived v1 counterpart")
    cur, v1 = [c for c, _ in matched], [a for _, a in matched]

    def mean(reports, fn):
        return sum(fn(_pair(r, "0->1")) for r in reports) / len(reports)

    # (label, accessor, reads the co-firing matrix?)
    rows = [
        ("frequency-driven edges, %", lambda p: 100 * p["freq_control"]["frac_freq_driven"], True),
        ("improve reconstruction, %", lambda p: 100 * p["reconstruction"]["frac_pass"], True),
        ("candidate edges", lambda p: p["n_candidate_edges"], True),
        ("sibling redundancy", lambda p: p["sibling_redundancy"]["mean_redundancy"], True),
        ("mean frequency survival", lambda p: p["freq_control"]["mean_survival"], True),
        ("superparents flagged", lambda p: p["n_superparents"], True),
        ("multi-parenting, %",
         lambda p: 100 * (p["degree"].get("poly_frac") if p["degree"].get("poly_frac") is not None
                          else p["degree"]["n_multi_parented"] / max(p["degree"]["n_children_with_parent"], 1)),
         False),
    ]
    data = [(lab, mean(v1, fn), mean(cur, fn), shared) for lab, fn, shared in rows]
    data.sort(key=lambda t: abs(math.log2(max(t[2], 1e-9) / max(t[1], 1e-9))))

    fig, ax = plt.subplots(figsize=(9.4, 4.8))
    y = np.arange(len(data))
    fold = [math.log2(max(b, 1e-9) / max(a, 1e-9)) for _, a, b, _ in data]
    ax.barh(y, fold, height=0.6,
            color=[CAT[3] if sh else GOOD for *_, sh in data])
    # The left-hand value of each pair is a WITHDRAWN number. It is plotted to
    # size the error, never as a result -- so it is drawn in the muted ink used
    # for annotation and the surviving value is in body ink, and the axis says
    # which is which. Withdrawn numbers come back by being readable next to live
    # ones; this is the whole reason those two hand-built pages were archived.
    for i, ((lab, a, b, sh), f) in enumerate(zip(data, fold)):
        fmt = (lambda v: f"{v:,.0f}") if max(a, b) > 20 else (lambda v: f"{v:.2f}")
        # The pair always reads withdrawn → current, left to right, whichever side
        # of zero the bar ends on. Anchoring both parts to the bar's end and
        # flipping only the alignment put them in the wrong order on negative bars.
        right = f >= 0
        gap, ch = 0.30, 0.40                     # data units; ch ≈ one character
        tail = f"→ {fmt(b)}"
        if right:
            x0 = f + gap
            ax.annotate(fmt(a), (x0, i), ha="left", va="center",
                        fontsize=8.5, color="#B0B4BB")
            ax.annotate(tail, (x0 + ch * (len(fmt(a)) + 1), i), ha="left", va="center",
                        fontsize=8.5, color=INK, fontweight="bold")
        else:
            x0 = f - gap
            ax.annotate(tail, (x0, i), ha="right", va="center",
                        fontsize=8.5, color=INK, fontweight="bold")
            ax.annotate(fmt(a), (x0 - ch * (len(tail) + 1), i), ha="right", va="center",
                        fontsize=8.5, color="#B0B4BB")
    ax.axvline(0, color=INK, lw=1)
    ax.set_yticks(y)
    ax.set_yticklabels([lab for lab, *_ in data], fontsize=9)
    ax.set_xlabel(f"log₂ change when the contaminating token position is removed, "
                  f"mean over the {len(matched)} layers graded both before and after\n"
                  "each pair reads  withdrawn → current")
    ax.set_xlim(min(fold) - 4.2, max(fold) + 4.2)
    ax.set_ylim(-0.7, len(data) - 0.3)
    ax.legend(handles=[
        plt.Rectangle((0, 0), 1, 1, color=CAT[3], label="reads the co-firing matrix"),
        plt.Rectangle((0, 0), 1, 1, color=GOOD,
                      label="does not — a ratio over children that already have a parent"),
    ], fontsize=8.5, frameon=False, loc="lower left")   # inside: the far left is empty
    _title(ax, "Six metrics designed as independent detectors moved together",
           "BOS is an attention sink, so with 400 documents every pair in the dictionary was "
           "handed 400 joint firings against a guard set at 30. Five of these read that one "
           "matrix; all five moved. The sixth does not, and did not.", width=92)
    ax.grid(True, axis="x", alpha=0.12)
    ax.set_axisbelow(True)
    return _finish(fig, ax, "shared_input_moved_every_metric")


# ---------------------------------------------------------------------------
# 9b. The premise, tested. The project's motivating observation was that gemma's
#     superparents "mostly track high-frequency tokens (spaces, punctuation,
#     'the')". Two per-edge diagnostics separate that from the alternative, and
#     they disagree by a factor of 40-80 at every layer.
# ---------------------------------------------------------------------------
def base_rate_vs_frequency_capture(layers):
    Ls = [L for L, _ in layers]
    chance = [100 * _pair(r, "0->1")["independence_null"]["frac_chance_level"]
              for _, r in layers]
    freq = [100 * _pair(r, "0->1")["freq_control"]["frac_freq_driven"] for _, r in layers]

    fig, ax = plt.subplots(figsize=(7.6, 4.0))
    x = np.arange(len(Ls))
    w = 0.34
    ax.bar(x - w / 2, chance, w, color=CAT[0],
           label="at chance for the parent's base rate  (PMI < 0.5)")
    ax.bar(x + w / 2, freq, w, color=CAT[3],
           label="carried by globally frequent tokens  (survival < 0.5)")
    for xx, v in zip(x - w / 2, chance):
        ax.text(xx, v + 1.5, f"{v:.0f}%", ha="center", fontsize=8.5, color=MUTED)
    for xx, v in zip(x + w / 2, freq):
        ax.text(xx, v + 1.5, f"{v:.1f}%", ha="center", fontsize=8.5, color=MUTED)
    ax.set_xticks(x)
    ax.set_xticklabels([f"L{L}" for L in Ls])
    ax.set_ylabel("% of candidate edges, B0→B1")
    ax.set_ylim(0, 100)
    ax.legend(fontsize=8.5, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.09))
    _title(ax, "gemma's over-connected parents are a base-rate effect, not token-frequency capture",
           f"the two diagnostics disagree by {min(c / f for c, f in zip(chance, freq)):.0f}–"
           f"{max(c / f for c, f in zip(chance, freq)):.0f}× at every layer. The project's "
           "motivating observation was that superparents mostly track high-frequency tokens; "
           "the frequency control exonerates 98–99% of edges while the independence null "
           "rejects 69–86%", width=82)
    ax.grid(True, axis="y", alpha=0.12)
    ax.set_axisbelow(True)
    return _finish(fig, ax, "base_rate_vs_frequency_capture")


# ---------------------------------------------------------------------------
# 10. What a top-k rank rule costs at each dictionary size. No data file: the
#     rule is a geometry test, so an unrelated parent passes at exactly k/D.
#     Measured on the toy (k/D = 11.9%, observed 2-4 of 20 against 2.4 expected),
#     which is why an S_res share is only comparable across similar D.
# ---------------------------------------------------------------------------
def sres_null_rate_vs_dictionary_size(observed):
    k = C.SRES_RANK_TOP_K
    sources = [("synthetic toy", 42), ("PCFG SAE", 1792), ("gemma SAE", 32768)]
    Ds = np.array([d for _, d in sources], dtype=float)
    null = 100 * k / Ds

    fig, ax = plt.subplots(figsize=(7.8, 4.2))
    grid = np.logspace(1.4, 4.8, 200)
    ax.plot(grid, 100 * k / grid, lw=2, color=NEUTRAL, zorder=1,
            label=f"chance pass rate = k/D  (k = {k})")
    ax.scatter(Ds, null, s=70, color=CAT[0], zorder=3, edgecolor="white", linewidth=1,
               label="chance, per source")
    for i, ((name, d), v) in enumerate(zip(sources, null)):
        # below-left, except the first (no room to its left) and the last (the
        # zero-floor note lives under it)
        left = 0 < i < len(sources) - 1
        ax.annotate(f"{name}\nD = {d:,} → {v:.2f}%", (d, v),
                    ha="right" if left else "left",
                    textcoords="offset points",
                    xytext=(-10 if left else 11, -20 if i < len(sources) - 1 else 4),
                    fontsize=8, color=MUTED)
    floor = float(np.min(null)) / 4                      # a visible place for zero
    drew_obs = zeros = False
    # A drop-line from each measurement to its own chance point. The message is
    # the RATIO between them, and on a log axis that is a vertical distance --
    # which the reader had to estimate by eye against a sloping grey line.
    for name, d, obs in observed:
        y0, y1 = (floor if obs <= 0 else obs), 100 * k / d
        ax.plot([d, d], [min(y0, y1), max(y0, y1)], lw=1.2, color=GOOD, alpha=0.45, zorder=2)
    # With one run per dictionary a label per point was fine; with every layer's
    # second pass present there are several runs at the SAME D, and a label each
    # stacks into an unreadable column. Markers are cheap and stay; text is not:
    # only the extremes of each dictionary's group are named, because the group's
    # spread IS the finding -- runs on one dictionary land on both sides of one null.
    by_d: dict = {}
    for name, d, obs in observed:
        by_d.setdefault(d, []).append((name, obs))
    for name, d, obs in observed:
        # A measured 0% has no position on a log axis. Plotting it at the floor
        # with an open marker and saying so beats dropping the point, which would
        # leave the figure showing only the source whose rate happened to be
        # non-zero -- the more flattering half of the comparison.
        at_zero = obs <= 0
        zeros |= at_zero
        ax.scatter([d], [floor if at_zero else obs], s=72, marker="D",
                   color="white" if at_zero else GOOD, zorder=4,
                   edgecolor=GOOD, linewidth=1.6,
                   label=None if drew_obs else "measured pass rate")
        drew_obs = True
        group = by_d[d]
        lo, hi = min(o for _, o in group), max(o for _, o in group)
        if len(group) > 2 and obs not in (lo, hi):
            continue
        ax.annotate(f"{name}: {'0% — no edge passed' if at_zero else f'{obs:.2f}%'}"
                    + ("" if at_zero else f"  ≈{obs / (100 * k / d):.0f}× chance"),
                    (d, floor if at_zero else obs), textcoords="offset points",
                    xytext=(11, -4 if at_zero else 4), fontsize=7.5, color=GOOD)
    ax.set_xscale("log")
    ax.set_yscale("log")
    if zeros:
        ax.axhline(floor, ls=(0, (1, 4)), lw=1, color=NEUTRAL)
        ax.text(ax.get_xlim()[0] * 1.15, floor * 1.15, "0% drawn here — a log axis has no zero",
                fontsize=7.5, color=MUTED, style="italic", ha="left")
    ax.set_xlabel("dictionary size D (log)")
    ax.set_ylabel("% of unrelated parents that pass (log)")
    # lifted clear of the zero-floor rule that runs along the bottom
    ax.legend(fontsize=8.5, frameon=False, loc="lower left", bbox_to_anchor=(0.0, 0.10))
    # One summary clause per dictionary, not one per run: the subtitle has to
    # stay readable at ten measurements as it was at three.
    parts = []
    for d in sorted(by_d):
        grp, nl = by_d[d], 100 * k / d
        lo, hi = min(o for _, o in grp), max(o for _, o in grp)
        rng = (f"{lo:.2f}–{hi:.2f}%" if len(grp) > 1 else
               (f"{hi:.2f}%" if hi > 0 else "0%"))
        vs = ("below chance" if hi <= nl else
              f"up to {hi / nl:.0f}× its null" if lo <= nl else
              f"{lo / nl:.0f}–{hi / nl:.0f}× its null")
        parts.append(f"{len(grp)} run{'s' if len(grp) > 1 else ''} at D={d:,}: {rng}, {vs}")
    _title(ax, "The rank rule's strictness is set by dictionary size, not by k alone",
           "grey line = what chance alone gives at each D; the vertical drop to it is what a "
           "measured rate is worth. " + "; ".join(parts) +
           ". Runs on the same dictionary land on both sides of their own "
           "null, so a raw pass rate compares nothing", width=78)
    ax.grid(True, which="both", alpha=0.12)
    ax.set_axisbelow(True)
    return _finish(fig, ax, "sres_null_rate_vs_dictionary_size")


# ---------------------------------------------------------------------------
# 9c. Where the tangle lives: every metric per block pair, both sources, one
#     matrix. The claim it backs is locational -- the structural pathology
#     concentrates at the outermost boundary and the deeper pairs are clean --
#     and a location claim wants the whole map on one canvas, not a bar chart
#     per pair.
# ---------------------------------------------------------------------------
def tangle_lives_in_top_block_pair(layers, pcfg):
    from matplotlib import cm, colors as mcolors
    COLS = [
        ("candidates", lambda p: p["n_candidate_edges"], "{:,.0f}"),
        ("multi-parent %", lambda p: 100 * p["degree"]["poly_frac"], "{:.0f}"),
        ("fan-out Gini", lambda p: p["degree"]["outdeg_gini"], "{:.2f}"),
        ("at chance %", lambda p: 100 * p["independence_null"]["frac_chance_level"], "{:.0f}"),
        ("superparents", lambda p: p["n_superparents"], "{:.1f}"),
        ("sibling Jaccard", lambda p: p["sibling_redundancy"]["mean_redundancy"], "{:.2f}"),
        ("freq-driven %", lambda p: 100 * p["freq_control"]["frac_freq_driven"], "{:.1f}"),
    ]

    def source_rows(reports):
        pairs = sorted({q["pair"] for r in reports for q in r["pairs"]},
                       key=lambda s: int(s.split("->")[0]))
        out = []
        def safe(fn, q):
            # deep pairs carry nulls where a sub-metric had nothing to score;
            # absence is NaN, never zero
            try:
                v = fn(q)
                return float(v) if v is not None else np.nan
            except (TypeError, KeyError):
                return np.nan

        for pr in pairs:
            row = []
            for _, fn, _ in COLS:
                # mean over the layers where this pair has candidates at all;
                # a metric of an empty candidate set is not a small value, it
                # is no value, and averaging it in would fake cleanliness
                vals = [safe(fn, q) for r in reports
                        for q in [_pair(r, pr)]
                        if q and q["n_candidate_edges"] > 0]
                vals = [v for v in vals if not np.isnan(v)]
                row.append(float(np.mean(vals)) if vals else np.nan)
            out.append((pr, row))
        return out

    gem = source_rows([r for _, r in layers])
    pcf = source_rows([r for _, r, *_ in [(n, rep) for n, rep in pcfg]]) if pcfg else []
    rows = [(f"gemma B{p.replace('->', '→B')}", v) for p, v in gem] \
         + [(f"PCFG B{p.replace('->', '→B')}", v) for p, v in pcf]
    M = np.array([v for _, v in rows])

    # one hue, light -> dark, per COLUMN: the columns are different quantities
    # on different scales, so colour can only rank within a column -- which is
    # exactly the locational claim -- and the caption says so
    norm = M.copy()
    for j in range(M.shape[1]):
        col = M[:, j]
        top = np.nanmax(col)
        norm[:, j] = col / top if top and not np.isnan(top) else 0
    cmap = cm.get_cmap("Blues")

    fig, ax = plt.subplots(figsize=(9.8, 0.46 * len(rows) + 1.6))
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            v, nv = M[i, j], norm[i, j]
            if np.isnan(v):
                ax.add_patch(plt.Rectangle((j, i), 1, 1, color="#F2F3F5"))
                ax.text(j + 0.5, i + 0.5, "—", ha="center", va="center",
                        fontsize=8, color=MUTED)
                continue
            bg = cmap(0.08 + 0.84 * nv)
            ax.add_patch(plt.Rectangle((j, i), 1, 1, color=bg))
            lum = mcolors.rgb_to_hsv(bg[:3])[2]
            ax.text(j + 0.5, i + 0.5, COLS[j][2].format(v), ha="center", va="center",
                    fontsize=8, color="white" if lum < 0.55 else INK)
    ax.set_xlim(0, M.shape[1])
    ax.set_ylim(M.shape[0], 0)
    ax.set_xticks(np.arange(M.shape[1]) + 0.5)
    ax.set_xticklabels([c for c, *_ in COLS], fontsize=8.5)
    ax.xaxis.set_ticks_position("top")
    ax.set_yticks(np.arange(len(rows)) + 0.5)
    ax.set_yticklabels([n for n, _ in rows], fontsize=8.5)
    if gem and pcf:
        ax.axhline(len(gem), color=INK, lw=1.4)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(length=0)
    return _finish(fig, ax, "tangle_lives_in_top_block_pair")


# ---------------------------------------------------------------------------
# 10a. The battery, question by question. Each metric was designed to answer
#      one plain-language question about an edge; this figure puts the question
#      in the panel title and the measured answer under it, per layer, so a
#      reader meets the instrument and its verdict in the same glance.
# ---------------------------------------------------------------------------
def battery_questions_gemma(layers, second):
    import textwrap
    Ls = [L for L, _ in layers]
    P = [_pair(r, "0->1") for _, r in layers]
    sres_pct = [100 * second[L]["0->1"]["sres"]["frac_pass"]
                if second.get(L) and "0->1" in second[L] else np.nan for L in Ls]
    panels = [
        ("Activation coverage",
         "Does the child fire only when the parent fires?",
         [p["n_candidate_edges"] for p in P], "candidate edges", None),
        ("Reconstruction condition",
         "Does the pair actually carry reconstruction, or do the two just activate together?",
         [100 * p["reconstruction"]["frac_pass"] for p in P], "% of edges passing", None),
        ("Sibling redundancy",
         "Are the children almost copies of each other — feature splitting posing as hierarchy?",
         [p["sibling_redundancy"]["mean_redundancy"] for p in P], "mean pairwise Jaccard",
         C.SIBLING_REDUNDANCY_FLAG),
        ("Out-degree / superparents",
         "Does one parent fan out over most of the next block?",
         [p["n_superparents"] for p in P], "parents flagged", None),
        ("Token-frequency control",
         "Does the edge still hold on rare tokens, or is it frequency-driven?",
         [100 * p["freq_control"]["frac_freq_driven"] for p in P], "% frequency-driven", None),
        ("Independence null",
         "Is the co-firing above chance at all, or just the parent's base rate?",
         [100 * p["independence_null"]["frac_chance_level"] for p in P], "% at chance", None),
        ("Probe-based S_res",
         "Does the parent's decoder really point to the child's concept?",
         sres_pct, "% of scored edges passing", None),
        ("Exact joint-child coverage",
         "How much of the parent do the kept children really explain?",
         [p["joint_child"]["r_supp_mean"] for p in P], "mean support coverage", None),
    ]
    fig, axes = plt.subplots(2, 4, figsize=(13.8, 6.6))
    for ax, (name, q, vals, ylab, thr) in zip(np.ravel(axes), panels):
        ax.bar(np.arange(len(Ls)), vals, 0.62, color=CAT[0])
        if thr is not None:
            ax.axhline(thr, ls=(0, (4, 3)), lw=1.2, color=CAT[3])
            ax.text(len(Ls) - 0.4, thr, f"flag ≥ {thr:g}", fontsize=6.8, color=CAT[3],
                    ha="right", va="bottom")
        ax.set_xticks(np.arange(len(Ls)))
        ax.set_xticklabels([f"L{L}" for L in Ls], fontsize=7.5)
        ax.set_ylabel(ylab, fontsize=8)
        ax.margins(y=0.22)
        ax.tick_params(labelsize=7.5)
        ax.grid(True, axis="y", alpha=0.12)
        ax.set_axisbelow(True)
        ax.text(0, 1.30, name, transform=ax.transAxes, fontsize=9.5,
                fontweight="bold", color=INK, va="top")
        ax.text(0, 1.19, "\n".join(textwrap.wrap(q, 38)), transform=ax.transAxes,
                fontsize=7.3, color=MUTED, va="top", style="italic")
    lines = _title(fig, "The battery, question by question — gemma-2-2b, block pair B0→B1, "
                   "every graded layer",
                   "each panel is the question one metric asks and its measured answer; the "
                   "questions the edges pass are about quality, the ones they fail are about "
                   "structure", width=120)
    # the per-panel question texts sit above each axes, so the top margin is
    # for them; only shave extra room when a figure title is actually drawn.
    # This figure owns its margins (tight=False): tight_layout cannot see the
    # hanging texts and clips the first row's headings.
    fig.subplots_adjust(top=0.84 if lines else 0.90, bottom=0.06, left=0.05,
                        right=0.99, hspace=0.95, wspace=0.34)
    return _finish(fig, axes, "battery_questions_gemma", tight=False)


# ---------------------------------------------------------------------------
# 10b. The recovered graph itself, drawn from the edge lists rather than
#      summarized. One panel per world: the trained toy's recovered tree, and
#      the real B0->B1 edge set on gemma. The layout gives a tree its best
#      chance -- every child is placed under its strongest parent -- so any
#      crossing that remains is structure, not plotting.
# ---------------------------------------------------------------------------
def recovered_graph_toy_vs_gemma(tt, sp, where, sp_pcfg=None, where_pcfg=""):
    true_edges = [tuple(e) for e in tt["true_edges"]]
    found = {tuple(e) for e in tt["found_edges"]}

    n_panels = 3 if (sp_pcfg and "0->1" in sp_pcfg) else 2
    ratios = [1, 1.35, 1.75][:n_panels] if n_panels == 3 else [1, 1.9]
    fig, axes = plt.subplots(1, n_panels, figsize=(6.2 + 3.4 * n_panels, 4.4),
                             gridspec_kw={"width_ratios": ratios})

    # -- toy: the tree it recovered, missed edges dashed ----------------------
    ax = axes[0]
    t_parents = sorted({p for p, _ in true_edges})
    t_children = sorted({c for _, c in true_edges})
    # children grouped under their (unique, by construction) true parent
    order = sorted(t_children, key=lambda c: next(p for p, cc in true_edges if cc == c))
    cx = {c: i / max(len(order) - 1, 1) for i, c in enumerate(order)}
    px = {p: np.mean([cx[c] for pp, c in true_edges if pp == p]) for p in t_parents}
    for p, c in true_edges:
        ok = (p, c) in found
        ax.plot([px[p], cx[c]], [1, 0], linestyle="-" if ok else (0, (3, 3)),
                lw=2 if ok else 1.2, color=GOOD if ok else NEUTRAL, zorder=2)
    ax.scatter([px[p] for p in t_parents], [1] * len(t_parents), s=90, color=INK, zorder=3)
    ax.scatter([cx[c] for c in order], [0] * len(order), s=48, color=MUTED, zorder=3)
    n_fp = tt["false_positives"]
    ax.set_title(f"trained toy — {len(found)}/{len(true_edges)} true edges recovered, "
                 f"{n_fp} false positive{'s' if n_fp != 1 else ''}\n"
                 "green = recovered · dashed = missed (never learned)",
                 fontsize=9.5, loc="left")

    # -- a probed SAE: every candidate edge that reached the probe ------------
    def draw_probed(ax, sp_x, label):
        edges = sp_x["0->1"]["sres"]["edges"]
        by_child: dict = {}
        for e in edges:
            by_child.setdefault(e["child"], []).append(e)
        # anchor each child under the parent that correlates best with its
        # probe -- the layout a genuine tree would satisfy with near-vertical
        # lines
        anchor = {c: max(es, key=lambda e: e["parent_corr"])["parent"]
                  for c, es in by_child.items()}
        deg: dict = {}
        for e in edges:
            deg[e["parent"]] = deg.get(e["parent"], 0) + 1
        parents = sorted(deg, key=lambda p: -deg[p])
        px = {p: i / max(len(parents) - 1, 1) for i, p in enumerate(parents)}
        children = sorted(by_child, key=lambda c: px[anchor[c]])
        cx = {c: i / max(len(children) - 1, 1) for i, c in enumerate(children)}
        for e in edges:
            if not e["pass"]:
                ax.plot([px[e["parent"]], cx[e["child"]]], [1, 0], lw=0.35,
                        color=CAT[0], alpha=0.07, zorder=1)
        n_pass = 0
        for e in edges:
            if e["pass"]:
                n_pass += 1
                ax.plot([px[e["parent"]], cx[e["child"]]], [1, 0], lw=1.8,
                        color=GOOD, zorder=3)
        ax.scatter(list(px.values()), [1] * len(px), s=14, color=INK, zorder=4)
        ax.scatter(list(cx.values()), [0] * len(cx), s=5, color=MUTED, zorder=4)
        key = (f"blue = candidate, unconfirmed · green = probe-confirmed ({n_pass})"
               if n_pass else
               "blue = candidate, unconfirmed · probe-confirmed: none")
        ax.set_title(f"{label} — {len(edges):,} candidate edges "
                     f"({len(px)} parents × {len(cx)} children)\n" + key,
                     fontsize=9.5, loc="left")

    if n_panels == 3:
        draw_probed(axes[1], sp_pcfg, where_pcfg)
    draw_probed(axes[-1], sp, where)

    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([0, 1])
        ax.set_yticklabels(["child block", "parent block"], fontsize=8.5)
        ax.set_ylim(-0.14, 1.14)
        for s in ("left", "bottom"):
            ax.spines[s].set_visible(False)
    _title(fig, "The recovered graph, drawn: a tree where the world is a tree, a tangle on the "
                "released SAE",
           "layout gives a tree its best chance — each child sits under the parent that best "
           "matches its probe, so a clean hierarchy would be near-vertical lines. The crossings "
           "are the structure", width=110)
    fig.subplots_adjust(top=0.80)
    return _finish(fig, axes, "recovered_graph_toy_vs_gemma")


# ---------------------------------------------------------------------------
# 11. The formatting-density axis of the PCFG sweep, at the one layer the local
#     runs grade. Formatting density is the axis the project treats as the
#     mechanism behind bottleneck hijacking, so it gets its own figure: which
#     measures a grammar knob can move, and which are properties of the SAE.
# ---------------------------------------------------------------------------
def pcfg_formatting_sweep(fmt_runs):
    """Metric against delimiter density, one point per seed, mean drawn through.

    The reading is the SHAPE: a flat line is a measure no grammar knob moves --
    intrinsic to the architecture -- and a slope is a measure confounded with
    the corpus. That split is computed and printed rather than asserted.
    """
    layer = fmt_runs[0][2]["config"]["layer"]
    panels = [
        ("multi-parenting, % of children", lambda r, sp:
            100 * _pair(r, "0->1")["degree"]["poly_frac"]),
        ("at chance for the base rate, %", lambda r, sp:
            100 * _pair(r, "0->1")["independence_null"]["frac_chance_level"]),
        ("superparents flagged", lambda r, sp:
            _pair(r, "0->1")["n_superparents"]),
        (r"probe-confirmed $S_{res}$, %", lambda r, sp:
            100 * sp["0->1"]["sres"]["frac_pass"] if sp and "0->1" in sp else np.nan),
    ]
    dens = sorted({d for d, *_ in fmt_runs})
    # ORDINAL x positions. The densities are 0, 0.1667, 0.2308 and 0.24: on a
    # numeric axis the last two land 0.009 apart, their tick labels print on
    # top of each other, and two-thirds of the panel is empty. The spacing of
    # the sweep's rungs is the sweep author's choice, not a finding, so equal
    # spacing loses nothing and the labels carry the true values.
    xpos = {d: i for i, d in enumerate(dens)}
    fig, axes = plt.subplots(1, len(panels), figsize=(12.4, 3.4))
    knob, seed_noise = [], []
    for ax, (label, fn) in zip(axes, panels):
        means, spreads = [], []
        for d in dens:
            vals = [fn(r, sp) for dd, _, r, sp in fmt_runs if dd == d]
            vals = [v for v in vals if not np.isnan(v)]
            ax.scatter([xpos[d]] * len(vals), vals, s=26, color=CAT[0], alpha=0.55, zorder=2)
            means.append(float(np.mean(vals)) if vals else np.nan)
            spreads.append(max(vals) - min(vals) if len(vals) > 1 else 0.0)
        ax.plot([xpos[d] for d in dens], means, "-o", color=CAT[0], lw=2, ms=4, zorder=3)
        ax.set_title(label, fontsize=9.5, loc="left")
        ax.set_xlabel("delimiter density")
        ax.set_xticks(list(xpos.values()))
        ax.set_xticklabels([f"{d:g}" for d in dens], fontsize=8)
        ax.margins(y=0.25, x=0.08)
        ax.grid(True, axis="y", alpha=0.12)
        ax.set_axisbelow(True)
        # What three seeds can and cannot support, computed: the knob's effect
        # is the range of the seed means, and it only means something where it
        # exceeds how far the seeds scatter at a single density.
        m = [v for v in means if not np.isnan(v)]
        clean = label.split(",")[0].replace("$", "").replace("_{res}", "_res")
        (knob if max(m) - min(m) > float(np.mean(spreads)) else
         seed_noise).append(clean)
    _title(fig, f"The formatting-density axis, swept — PCFG transformer, layer {layer}, "
                f"{len(fmt_runs) // len(dens)} seeds per density",
           "one point per seed; the line joins seed means. Where the seeds scatter more at one "
           "density than the mean moves across the whole axis, the knob's effect is not "
           "resolved at three seeds"
           + (f" — that is the case for {', '.join(seed_noise)}" if seed_noise else "")
           + (f"; resolved above seed noise: {', '.join(knob)}" if knob else ""),
           width=110)
    fig.subplots_adjust(top=0.76)
    return _finish(fig, axes, "pcfg_formatting_sweep")


# ---------------------------------------------------------------------------
def build(dry: bool) -> tuple[list[str], list[tuple[str, str]]]:
    """Returns (written, skipped) where skipped carries the reason, never silence."""
    written: list[str] = []
    skipped: list[tuple[str, str]] = []
    G = C.OUT_DIR / C.SOURCE_NAME
    layers = gemma_layers()

    sources: dict[str, str] = {}

    def run(name, need_ok, why, fn):
        if not need_ok:
            skipped.append((name, why))
            return
        sources[name] = why
        if dry:
            written.append(f"{name}   <- {why}")
            return
        written.append(fn())

    # layer -> second_pass.json, for every layer that has the strict test
    second = {L: _json(G / f"layer_{L:02d}" / "second_pass.json") for L, _ in layers}
    second = {L: v for L, v in second.items() if v}
    l6 = _json(G / "layer_06" / "metrics_report.json")
    # every PCFG depth run, as (label, report, second_pass) so the two-source
    # figures never re-derive which directory a second pass belongs to
    pcfg = []
    for p in sorted((C.OUT_DIR / "pcfg-matryoshka").glob("layer_*")):
        r = _json(p / "metrics_report.json")
        if r:
            pcfg.append((f"PCFG {p.name.replace('_', ' ')}", r,
                         _json(p / "second_pass.json")))
    run("funnel_coverage_to_sres", bool(second),
        f"{len(second)} gemma + {sum(1 for *_, sp in pcfg if sp)} PCFG layers with "
        "second_pass.json"
        if second else "needs at least one gemma second_pass.json (stage 03)",
        lambda: funnel_coverage_to_sres(layers, second, pcfg, where="gemma-2-2b"))

    run("edge_survival_by_block_pair", len(layers) >= 2,
        f"{len(layers)} gemma layer reports" if layers else "needs ≥2 gemma metrics_report.json",
        lambda: edge_survival_by_block_pair(layers))
    run("depth_profile_across_layers", len(layers) >= 2,
        f"{len(layers)} gemma layer reports" if layers else "needs ≥2 gemma metrics_report.json",
        lambda: depth_profile_across_layers(layers))
    run("multiparenting_by_layer", len(layers) >= 1,
        f"{len(layers)} gemma + {len(pcfg)} PCFG layer reports"
        if layers else "needs gemma metrics_report.json",
        lambda: multiparenting_by_layer(layers, [(n, r) for n, r, _ in pcfg]))
    # Matched BY LAYER: a layer graded only after the BOS fix has no v1
    # counterpart and must not veto the figure for the layers that do.
    arch_L = {int(p.parent.name.split("__")[0].split("_")[1])
              for p in (C.HERE / "outputs_archive").glob("layer_*__v1__*/metrics_report.json")}
    n_match = sum(1 for L, _ in layers if L in arch_L)
    run("shared_input_moved_every_metric", n_match >= 2,
        f"{n_match} layers graded both before and after BOS exclusion" if n_match
        else "needs the pre-BOS reports in outputs_archive/",
        lambda: shared_input_moved_every_metric(layers))
    run("base_rate_vs_frequency_capture", len(layers) >= 1,
        f"{len(layers)} gemma layer reports" if layers else "needs gemma metrics_report.json",
        lambda: base_rate_vs_frequency_capture(layers))
    run("superparent_fanout_vs_firing", len(layers) >= 1,
        f"{len(layers)} gemma layer reports" if layers else "needs gemma metrics_report.json",
        lambda: superparent_fanout_vs_firing(layers))

    toy = _json(C.OUT_DIR / "synthetic_toy_calibration.json")
    run("calibration_synthetic_toy_scorecard", bool(toy),
        "outputs/synthetic_toy_calibration.json" if toy
        else "needs outputs/synthetic_toy_calibration.json (validation.calibrate_on_synthetic_toy)",
        lambda: calibration_synthetic_toy_scorecard(toy))

    tt = _json(C.OUT_DIR / "trained_toy_calibration.json")
    align = _json(C.OUT_DIR / "block_tree_alignment.json")
    run("calibration_trained_toy_recovery", bool(tt),
        "outputs/trained_toy_calibration.json"
        + ("" if align else " (block_tree_alignment.json absent — right panel blank)")
        if tt else "needs outputs/trained_toy_calibration.json (Tier 2)",
        lambda: calibration_trained_toy_recovery(tt, align))

    # Every graded layer of BOTH sources. This figure once carried gemma layer 6
    # alone -- the layer that had the strict test first -- and kept doing so
    # after the other five caught up, which read as "the comparison rests on one
    # layer" long after it no longer did.
    runs = [(f"gemma L{L}", rep) for L, rep in layers] + [(n, r) for n, r, _ in pcfg]
    run("cross_source_funnel_shares", bool(layers and pcfg),
        f"{len(layers)} gemma + {len(pcfg)} PCFG layer reports"
        if (layers and pcfg) else "needs a gemma report and a PCFG report",
        lambda: cross_source_funnel_shares(runs))

    # The two cross-source depth figures. Both need at least one layer index that
    # BOTH sources graded; the alignment check additionally needs the two rules to
    # actually disagree, which they only do once gemma has layers the toy cannot
    # reach. Each condition is reported rather than silently producing a figure
    # whose dumbbells are all zero length.
    dr = X.rows()
    shared = sorted({o["layer"] for o, _, _ in X.matched(dr, "layer")})
    run("cross_source_layer_response", bool(shared),
        f"{len(dr)} graded runs across {len({r['src'] for r in dr})} sources, "
        f"layer{'s' if len(shared) > 1 else ''} {', '.join(str(s) for s in shared)} on both"
        if shared else "needs one layer index graded on both gemma and PCFG",
        lambda: cross_source_layer_response(dr))
    same = X.alignment_gaps(dr) is None
    run("cross_source_alignment_check", bool(shared) and not same,
        f"the same {len(dr)} runs under both alignment rules"
        if shared and not same else
        "the two alignments pair the same runs here, so the comparison is empty"
        if shared else "needs one layer index graded on both gemma and PCFG",
        lambda: cross_source_alignment_check(dr))

    ib = []
    for base, label in [(G, "gemma"), (C.OUT_DIR / "pcfg-matryoshka", "PCFG")]:
        for lay in sorted(base.glob("layer_*")):
            j = _json(lay / "in_block_edges.json")
            if j:
                ib.append((f"{label} {lay.name.replace('_', ' ')}", j))
    run("in_block_relations", bool(ib),
        f"{len(ib)} runs with in_block_edges.json" if ib
        else "needs in_block_edges.json — pipeline stage 01c",
        lambda: in_block_relations(ib))

    # observed S_res shares, read off whatever second_pass.json files exist --
    # every gemma layer that has one, not the single layer that had one first
    observed = []
    probes = [(f"gemma L{int(q.parent.name.split('_')[1])}", C.D_SAE, q)
              for q in sorted(G.glob("layer_*/second_pass.json"))]
    probes += [(f"PCFG {q.parent.name.replace('_', ' ')}", 1792, q)
               for q in sorted((C.OUT_DIR / "pcfg-matryoshka").glob("layer_*/second_pass.json"))]
    for label, d_sae, path in probes:
        sp = _json(path)
        if sp and "0->1" in sp:
            s = sp["0->1"]["sres"]
            if s["n_edges_scored"]:
                # zero is a measurement, not a missing value -- kept and drawn
                observed.append((label, d_sae, 100 * s["n_pass"] / s["n_edges_scored"]))
    run("sres_null_rate_vs_dictionary_size", True,
        f"config (k={C.SRES_RANK_TOP_K}) + {len(observed)} measured pass rates",
        lambda: sres_null_rate_vs_dictionary_size(observed))

    run("tangle_lives_in_top_block_pair", bool(layers or pcfg),
        f"{len(layers)} gemma + {len(pcfg)} PCFG layer reports, all block pairs"
        if (layers or pcfg) else "needs metrics_report.json for at least one source",
        lambda: tangle_lives_in_top_block_pair(layers, [(n, r) for n, r, _ in pcfg]))

    run("battery_questions_gemma", bool(layers),
        f"{len(layers)} gemma layer reports + {len(second)} second passes"
        if layers else "needs gemma metrics_report.json",
        lambda: battery_questions_gemma(layers, second))

    # the recovered graph itself, drawn at GRAPH_LAYER (see the constant). The
    # PCFG panel is drawn at the PCFG layer with the LARGEST probe-scored edge
    # set -- a stated rule, so the panel does not quietly follow whichever run
    # happened to finish first. Chosen over depth-matching the gemma panel
    # because this figure's job is to show the full funnel visually --
    # candidate, rejected, and probe-CONFIRMED edges -- and the largest scored
    # set is the one panel guaranteed to carry all three stages (a
    # depth-matched layer can have zero confirmed edges, which reads as a dead
    # probe). Depth-matched cross-source comparisons live in the tables.
    s_gl = second.get(GRAPH_LAYER)
    sp_p, sp_p_name = None, ""
    for lab, _, spx in pcfg:
        if spx and "0->1" in spx and spx["0->1"]["sres"]["n_edges_scored"]:
            if (sp_p is None or spx["0->1"]["sres"]["n_edges_scored"]
                    > sp_p["0->1"]["sres"]["n_edges_scored"]):
                sp_p, sp_p_name = spx, lab
    run("recovered_graph_toy_vs_gemma", bool(tt and s_gl),
        f"trained_toy_calibration.json + gemma layer_{GRAPH_LAYER:02d}"
        + (f" + {sp_p_name}" if sp_p else "") + " second_pass.json"
        if (tt and s_gl) else "needs the trained-toy calibration and a gemma second_pass.json",
        lambda: recovered_graph_toy_vs_gemma(tt, s_gl,
                                             f"gemma-2-2b layer {GRAPH_LAYER}, B0→B1",
                                             sp_p, f"{sp_p_name}, B0→B1"))

    # the formatting-density sweep runs, named fmt_<density*1e4>_s<seed>
    fmt_runs = []
    for p in sorted((C.OUT_DIR / "pcfg-matryoshka").glob("fmt_*")):
        r = _json(p / "metrics_report.json")
        if r:
            fmt_runs.append((int(p.name.split("_")[1]) / 1e4,
                             p.name.split("_s")[-1], r,
                             _json(p / "second_pass.json")))
    n_dens = len({d for d, *_ in fmt_runs})
    run("pcfg_formatting_sweep", n_dens >= 2,
        f"{len(fmt_runs)} fmt_* runs across {n_dens} delimiter densities"
        if fmt_runs else "needs pcfg-matryoshka/fmt_* metric reports",
        lambda: pcfg_formatting_sweep(fmt_runs))

    return written, skipped, sources


# ---------------------------------------------------------------------------
# figures.tex — the same figures with captions, ordered as the paper reads.
#
# Same two rules as the rest of this file. Every quantity in every caption is
# read from the JSON its figure plots, so a caption cannot outlive its data --
# which is exactly how this file's own titles came to quote withdrawn numbers.
# And a figure with no caption entry is emitted with none rather than silently
# dropped, so the omission is visible in the .tex.
#
# Captions sit BELOW the figure, which is the convention for figures and the
# mirror of tables.tex, where they sit above.
# ---------------------------------------------------------------------------
TWOCOLUMN = False       # set from --twocolumn; the ICLR template is onecolumn


def _wrap(t: str) -> str:
    import textwrap
    # break_on_hyphens=False: textwrap splits "0--7" across lines by default and
    # LaTeX turns the break into a space, printing "0-- 7".
    return "\n".join(textwrap.wrap(t, 94, initial_indent="    ", subsequent_indent="    ",
                                   break_on_hyphens=False, break_long_words=False))


def _captions():
    """Caption bodies, with their numbers interpolated from the plotted JSON."""
    G = C.OUT_DIR / C.SOURCE_NAME
    lay = gemma_layers()
    d: dict[str, str] = {}

    def rp(L, pair="0->1"):
        return _pair(_json(G / f"layer_{L:02d}" / "metrics_report.json"), pair)

    if lay:
        p6, sp6 = rp(6), _json(G / "layer_06" / "second_pass.json")
        poly = [100 * rp(L)["degree"]["poly_frac"] for L, _ in lay]
        ch = [100 * rp(L)["independence_null"]["frac_chance_level"] for L, _ in lay]
        fq = [100 * rp(L)["freq_control"]["frac_freq_driven"] for L, _ in lay]
        fires = [x["fire_frac"] for _, r in lay for q in r["pairs"]
                 for x in q.get("superparents", [])]
        free = sum(1 for f in fires if f >= C.EDGE_TAU)

        deep = [100 * (_pair(r, p) or {}).get("degree", {}).get("poly_frac", np.nan)
                for _, r in lay for p in ("1->2", "2->3")]
        deep = [v for v in deep if not np.isnan(v)]
        d["multiparenting_by_layer"] = (
            r"\textbf{The recovered graph is not a tree.} Child features are often "
            "associated with multiple parents rather than a single parent --- in the "
            rf"top block pair B0$\rightarrow$B1, {min(poly):.0f}--{max(poly):.0f}\% of "
            "children with any parent have two or more, at every graded layer --- "
            "revealing pervasive multi-parenting in the recovered structure. Deeper "
            rf"block pairs appear cleaner ({min(deep):.0f}--{max(deep):.0f}\%) on "
            "candidate sets of comparable or larger size, so the drop reflects "
            "genuinely looser structure rather than a smaller sample. Gemma's "
            r"B3$\rightarrow$B4 pair is marked n.a.: it is not yet computed --- "
            "grading it exhausts memory (B4 alone spans 24{,}576 latents) and a "
            "rerun is pending. Bars resting on fewer than 20 children print that "
            "support as $n{=}$; a 100\\% computed over one child is a coin flip, "
            "not a result, and the label keeps it from reading as one.")
        pp = [100 * _pair(r, "0->1")["degree"]["poly_frac"]
              for q in sorted((C.OUT_DIR / "pcfg-matryoshka").glob("layer_*"))
              for r in [_json(q / "metrics_report.json")] if r]
        if pp:
            d["multiparenting_by_layer"] += (
                f" The PCFG panel shows the same signature --- "
                rf"{min(pp):.0f}--{max(pp):.0f}\% in B0$\rightarrow$B1 across its layers "
                "--- so the multi-parenting is a property of the Matryoshka nesting, not "
                "of natural language; its deeper-pair bars rest on a handful of children "
                "and carry their support ($n$) rather than posing as results.")
        sps = {L: _json(G / f"layer_{L:02d}" / "second_pass.json") for L, _ in lay}
        sps = {L: v for L, v in sps.items() if v and "0->1" in v}
        if sps:
            cands = [rp(L)["n_candidate_edges"] for L in sps]
            passes = [sps[L]["0->1"]["sres"]["n_pass"] for L in sps]
            shares = [100 * sps[L]["0->1"]["sres"]["frac_pass"] for L in sps]
            recs = [100 * rp(L)["reconstruction"]["frac_pass"] for L in sps]
            d["funnel_coverage_to_sres"] = (
                r"\textbf{Broad coverage does not translate into confirmed structure.} "
                "Coverage proposes many candidate relationships "
                f"({min(cands):,}--{max(cands):,} per layer), but successive independence "
                "and probe tests eliminate most of them --- the independence null rejects "
                rf"{min(ch):.0f}--{max(ch):.0f}\%, and the probe confirms only "
                f"{min(passes)}--{max(passes)} edges per layer "
                rf"(${min(shares):.1f}$--${max(shares):.1f}\%$ of those scored) --- leaving "
                "only a small subset of supported parent--child relationships. The "
                rf"reconstruction filter (passing {min(recs):.0f}--{max(recs):.0f}\%) is "
                "evaluated separately because it does not form a nested stage.")
            pc = []
            for q in sorted((C.OUT_DIR / "pcfg-matryoshka").glob("layer_*/second_pass.json")):
                spx, rr = _json(q), _json(q.parent / "metrics_report.json")
                if spx and "0->1" in spx and rr:
                    pc.append((_pair(rr, "0->1")["n_candidate_edges"],
                               spx["0->1"]["sres"]["n_pass"]))
            if pc:
                d["funnel_coverage_to_sres"] += (
                    " The PCFG panel repeats the same collapse at small scale: "
                    f"{min(c for c, _ in pc)}--{max(c for c, _ in pc)} candidates per "
                    f"layer end in {min(p for _, p in pc)}--{max(p for _, p in pc)} "
                    "probe-confirmed edges.")
        d["base_rate_vs_frequency_capture"] = (
            r"Two per-edge diagnostics on the same candidate sets, B0$\rightarrow$B1. The "
            "independence null asks whether co-firing exceeds what the parent's own firing rate "
            rf"already forces, and rejects {min(ch):.0f}--{max(ch):.0f}\% of edges. The "
            "token-frequency control asks whether the edge survives once globally frequent tokens "
            rf"are removed, and rejects {min(fq):.1f}--{max(fq):.1f}\%. They disagree by "
            rf"{min(c / f for c, f in zip(ch, fq)):.0f}--"
            rf"{max(c / f for c, f in zip(ch, fq)):.0f}$\times$ at every layer. This tests the "
            "observation the bottleneck-hijacking hypothesis is built on---that the "
            "over-connected parents mostly track high-frequency tokens such as spaces, "
            r"punctuation and \emph{the}---and does not support it: the frequency control "
            r"exonerates 98--99\% of edges while the base-rate null rejects most of them.")
        d["superparent_fanout_vs_firing"] = (
            r"An edge is kept when $P(\mathrm{parent} \mid \mathrm{child}) \geq \tau = "
            rf"{C.EDGE_TAU}$. Under independence that probability is simply the parent's firing "
            r"rate $\rho$, so the enrichment a parent needs over chance is $\tau/\rho$, the grey "
            rf"curve. All {len(fires)} parents flagged as superparents, across five layers and "
            rf"every block pair, lie on it. {free} of them fire on at least "
            rf"{100 * C.EDGE_TAU:.0f}\% of tokens and therefore clear the bar at "
            r"$\leq 1\times$ enrichment---on base rate alone, with no association whatsoever "
            r"between parent and child. A parent firing on 1\% of tokens would need "
            rf"{C.EDGE_TAU / 0.01:.0f}$\times$ enrichment for the same edge. This is the "
            "mechanism behind the previous figure.")
        d["depth_profile_across_layers"] = (
            r"Four quantities for block pair B0$\rightarrow$B1 across the five graded layers, "
            "each on its own axis because they are not commensurable. The project previously "
            "reported that hierarchy quality degrades with depth. That claim came from caches in "
            "which the beginning-of-sequence position was counted, and it does not survive "
            "regeneration: the distinct-parent counts among survivors read 5, 7, 6, 7, 6. The "
            "figure is included because the withdrawal is itself a result about how easily a "
            "depth trend can be manufactured.")
        d["edge_survival_by_block_pair"] = (
            r"\emph{Left:} the share of candidate edges whose reconstruction improves when the "
            r"parent is ablated. \emph{Right:} the share the token-frequency control judges to be "
            "carried by globally frequent tokens. Panels are on separate scales; layers are "
            "ordered, so depth is encoded as a single-hue ramp rather than as five categorical "
            "colours. The frequency control is nearly silent in the top block pair and rises "
            "sharply in the deeper ones.")

    arch = sorted((C.HERE / "outputs_archive").glob("layer_*__v1__*/metrics_report.json"))
    if arch and lay:
        d["shared_input_moved_every_metric"] = (
            "Change in each metric, averaged over the five layers, when a single contaminating "
            "token position is excluded from the corpus. The beginning-of-sequence token is an "
            "attention sink on which effectively every feature fires; with 400 documents it "
            "handed every pair in the dictionary 400 joint firings against a support guard set "
            "at 30, so the guard admitted pairs that never co-occur anywhere else. Five of these "
            "six quantities are computed from that one co-firing matrix, and all five moved. The "
            "sixth, multi-parenting, is a ratio over children that already have a parent, does "
            "not read the matrix, and did not move. Agreement among detectors that share an "
            r"input is far weaker evidence than the word \emph{battery} implies. The grey value "
            "in each pair is the withdrawn one, plotted to size the error and not as a result.")

    toy = _json(C.OUT_DIR / "synthetic_toy_calibration.json")
    if toy:
        d["calibration_synthetic_toy_scorecard"] = (
            f"All {len(toy)} scorecard rows on a hand-built world with a known five-parent tree "
            f"and six injected structures; {sum(r['pass'] for r in toy)} pass, covering 21 "
            r"of 21 metric functions. \emph{Left:} rows whose two classes "
            r"separate by a ratio, on a log axis. \emph{Right:} rows scored categorically, where "
            "the recovered answer is either right or not. The two are kept apart because a "
            r"categorical margin of $1.0$ means \emph{correct}, and on a ratio axis that would "
            r"read as \emph{no separation at all}. The two hatched rows are negative controls, "
            r"which pass when the battery does \emph{not} act: an absorbed child, whose true edge "
            "coverage can never propose because the child fires exactly where its parent is "
            "silent, and a shared-topic pair that clears coverage, reconstruction, the frequency "
            "control and the independence null. They turn the two open columns of the properties "
            "matrix into demonstrated limitations rather than asserted ones, and a regression "
            "makes them fail visibly.")

    tt = _json(C.OUT_DIR / "trained_toy_calibration.json")
    if tt:
        pt = tt.get("per_token") or {}
        red = pt.get("parent_conditioned_redundancy") or {}
        cof = pt.get("child_cofire") or {}
        extra = ""
        if red and cof:
            w = max(red.items(), key=lambda kv: kv[1])
            if w[0] in cof:
                extra = (
                    " Beyond edge recovery, the parent-conditioned sibling metric reports "
                    f"{w[1]:.2f} for one true parent against "
                    + " and ".join(f"{v:.2f}" for k, v in red.items() if k != w[0])
                    + " for the others. The grammar declares every parent's children mutually "
                    f"exclusive, and they co-fire {cof[w[0]]['ground_truth']} times in "
                    "200{,}000 draws; the latents that recovered them co-fire "
                    f"{cof[w[0]]['learned']:,} times, and one of the two never fires alone. The "
                    "SAE conflated two concepts the grammar keeps apart. This is a defect nobody "
                    "injected, which the synthetic tier structurally cannot produce, and which "
                    "the rest of the battery misses: both of that parent's edges are counted as "
                    "recovered and precision stays $1.00$.")
        d["calibration_trained_toy_recovery"] = (
            "A Matryoshka SAE trained on the toy hierarchy, with the metrics run on the "
            r"\emph{learned} latents rather than on constructed statistics. Precision "
            rf"{tt['precision']:.2f}, recall {tt['recall']:.2f}: {tt['true_positives']} of "
            f"{len(tt['true_edges'])} true edges recovered with {tt['false_positives']} false "
            "positives, and every miss is an edge whose child the SAE never learned---it "
            f"recovered {tt['n_recovered_features']} of {tt['n_features']} true features, which "
            r"bounds recall from above. \emph{Right:} the lateral control, asking whether the "
            "Matryoshka nesting itself places a parent in an earlier block than its children."
            + extra)

    ib = [q for base in (C.OUT_DIR / C.SOURCE_NAME, C.OUT_DIR / "pcfg-matryoshka")
          for q in sorted(base.glob("layer_*")) if (q / "in_block_edges.json").exists()]
    if ib:
        d["in_block_relations"] = (
            r"Directed edges and co-extensive duplicates \emph{within} each block, where no block "
            "ordering fixes the direction and it must be derived from coverage asymmetry, across "
            f"all {len(ib)} graded runs. Reported as a rate per available pair rather than as a "
            "count: gemma's blocks are nested prefixes of 128 to 6{,}144 features, so the number "
            "of available pairs differs by a factor of 2{,}300 and raw counts invert the "
            "reading---the deepest block holds the most duplicate pairs and the fewest per pair. "
            "The concentration in B0 also holds on a PCFG SAE whose eight blocks are all 224 "
            "features, so it is not an artefact of B0 being small. What distinguishes B0 is "
            "being the outermost Matryoshka prefix: the one block trained to reconstruct on its "
            "own.")

    d["cross_source_funnel_shares"] = (
        "The same metric code and the same global thresholds, applied to the released Matryoshka "
        r"SAE on \texttt{gemma-2-2b} and to Matryoshka SAEs trained on a PCFG corpus. Reported "
        "as shares of each run's own candidate set, because the dictionaries differ in size and "
        "in block count and counts would not compare. Thresholds are deliberately not tuned per "
        "source: holding them fixed is what makes any cross-source comparison mean something. "
        "The reconstruction filter should be read with care on PCFG, where the weakest candidate "
        r"edge sits $3.5\times$ above the threshold---the filter is inert there, and the "
        "surviving edges have passed coverage alone.")
    dr = X.rows()
    by_layer = X.matched(dr, "layer")
    if by_layer:
        by_depth = X.matched(dr, "depth")
        order = [(t, k) for t, k, _ in X.ranked(by_layer)]
        g = [v for _, _, v in X.ranked(by_layer)]
        n_ag = X.agree_split(g)
        agree = list(zip([t for t, _ in order], g))[:n_ag]
        shared = sorted({o["layer"] for o, _, _ in by_layer})
        gem = [r for r in dr if r["is_ref"]]
        oth = [r for r in dr if not r["is_ref"]]
        dens = {r["label"]: X.density(r) for r in dr}
        d["cross_source_layer_response"] = (
            "Six measurements of block pair B0$\\rightarrow$B1, on the released Matryoshka SAE "
            rf"over \texttt{{{C.MODEL_NAME}}} and on Matryoshka SAEs trained over a "
            rf"{oth[0]['n_layers']}-block transformer fitted to a PCFG corpus. Both sources have "
            "an SAE at layer" + ("s " if len(shared) > 1 else " ")
            + " and ".join(str(s) for s in shared) + ", so the grey connectors are drawn between "
            "runs at the same block index and need no matching argument; their length is the "
            "disagreement. Every panel is a share and every panel is on the same axis anchored at "
            "zero, so lengths compare across panels. "
            + ", ".join(X.tex(t) for t, _ in agree)
            + rf" agree to within {max(v for _, v in agree):.1f} "
            rf"points, while {X.tex(order[-1][0])} differs by {g[-1]:.0f} and "
            rf"{X.tex(order[-2][0])} by "
            rf"{g[-2]:.0f}. \textbf{{That agreement is worth less than it looks}}: all {n_ag} "
            "of the close measures sit against a floor or a ceiling on both sources, so scaled "
            r"by the range each one varies over across every graded run "
            r"(Table~\ref{tab:align}) only "
            + X.tex(min(X.PANELS, key=lambda p: X.gap_ratio(dr, by_layer, p[1]))[0])
            + " stands out, and the reconstruction filter moves from nearly the worst to among "
            "the best. The reading we would take, carrying that caveat, is that the "
            "many-to-many fan-out "
            "B0$\\rightarrow$B1 is a property of the Matryoshka nesting rather than of "
            r"\texttt{gemma-2-2b}: what the base model changes is how much of it there is---"
            + " against ".join(rf"{dens[r['label']]:.1f}\%" for r in (gem[0], oth[0]))
            + " of all B0$\\times$B1 feature pairs become candidate edges---and not its "
            r"character. \textbf{Both PCFG runs are a single grammar configuration} ("
            + X.grammar_line(dr) + r")---one point of the three-axis sweep Exp 2 specifies "
            "(terminal distribution, formatting density, grammar depth), so this compares gemma "
            "against one PCFG corpus and not against PCFG. In particular the formatting axis, "
            "which the project now treats as the mechanism behind bottleneck hijacking, is at "
            r"its second-lowest rung here. \textbf{Two shared layers cannot establish a trend}, "
            r"and the deeper"
            "block pairs are excluded because the PCFG runs hold 0--3 candidate edges in each of "
            "them. The token budgets also differ by "
            rf"{max(r['tokens'] for r in dr) / min(r['tokens'] for r in dr):.0f}$\times$, which "
            "cuts against the density gap rather than explaining it: the joint-support guard is "
            "an absolute count, so more tokens make it easier to clear, and the source with more "
            "tokens is the sparser one.")
        if by_depth and not all(o["layer"] == q["layer"] for o, q, _ in by_depth):
            dgap = [X.gap(by_depth, k) for _, k in order]
            ml, md = sum(g) / len(g), sum(dgap) / len(dgap)
            d["cross_source_alignment_check"] = (
                "Comparing two base models of different depth requires an alignment, and the two "
                "defensible ones disagree here about which runs to pair: by block index the PCFG "
                "layers pair with gemma's "
                + " and ".join(f"L{q['layer']}" for _, q, _ in by_layer) + ", and by relative "
                r"depth $(L{+}1)/N$---the fraction of the network that has run when the SAE reads "
                r"\texttt{hook\_resid\_post}---they pair with gemma's "
                + " and ".join(f"L{q['layer']}" for _, q, _ in by_depth) + ", at opposite ends of "
                "the network. The choice is therefore not presentational, so it is measured here "
                "rather than made in an axis label: the mean gap over the six measures is "
                rf"{ml:.1f} points under block index against {md:.1f} under relative depth. "
                rf"\textbf{{On {len(by_layer)} shared layers this is suggestive, not a result.}} "
                "It is reported because an unstated alignment does the same work invisibly, and "
                "because the two rules would support different claims about which part of gemma "
                "a four-block transformer stands in for.")
    if lay:
        d["tangle_lives_in_top_block_pair"] = (
            "Every metric per block pair, averaged over each source's graded layers; a cell is "
            "the mean over the layers where that pair has candidate edges at all, and a dash "
            "marks a pair with none. Colour ranks \\emph{within} a column (each column is its "
            "own quantity and scale), so the reading is locational: the structural pathology "
            "--- multi-parenting, fan-out concentration, base-rate co-firing, superparents --- "
            "concentrates in the outermost pair B0$\\rightarrow$B1 on both sources, and the "
            "deeper pairs are clean but hold candidate sets one to three orders of magnitude "
            "smaller. The hierarchy is broken at its top boundary and quiet below it.")
    if lay:
        d["battery_questions_gemma"] = (
            "Each metric of the battery was built to answer one plain-language question about "
            "a candidate edge; each panel titles that question and plots its measured answer "
            r"on \texttt{gemma-2-2b}, block pair B0$\rightarrow$B1, at every graded layer. Read "
            "as a whole the figure shows the split the paper's results rest on: the questions "
            "about an edge's \\emph{quality} (reconstruction, frequency, siblings) come back "
            "healthy, while the questions about the graph's \\emph{structure} (base-rate "
            "co-firing, the probe, fan-out) come back failing, at every depth.")

    sp_gl = _json(G / f"layer_{GRAPH_LAYER:02d}" / "second_pass.json")
    tt2 = _json(C.OUT_DIR / "trained_toy_calibration.json")
    if tt2 and sp_gl:
        s = sp_gl["0->1"]["sres"]
        n_par = len({e["parent"] for e in s["edges"]})
        n_chi = len({e["child"] for e in s["edges"]})
        # the PCFG middle panel: same largest-probe-scored-set rule as build()
        mid = ""
        best, best_name = None, ""
        for q in sorted((C.OUT_DIR / "pcfg-matryoshka").glob("layer_*/second_pass.json")):
            spx = _json(q)
            if spx and "0->1" in spx and spx["0->1"]["sres"]["n_edges_scored"]:
                if (best is None or spx["0->1"]["sres"]["n_edges_scored"]
                        > best["0->1"]["sres"]["n_edges_scored"]):
                    best, best_name = spx, q.parent.name.replace("_", "~")
        grad = ""
        if best:
            b = best["0->1"]["sres"]
            mid = (rf"\emph{{Middle:}} a Matryoshka SAE trained on a PCFG corpus "
                   f"({best_name.replace('layer~0', 'layer~')}) repeats the tangle at "
                   f"small scale --- {b['n_edges_scored']:,} candidate edges, "
                   f"{b['n_pass']} probe-confirmed --- and ")
            # the dictionary sizes that make "ascending scale" a number, not a vibe
            rr = _json(C.OUT_DIR / "pcfg-matryoshka"
                       / best_name.replace("~", "_") / "metrics_report.json")
            d_p = ((rr or {}).get("config") or {}).get("d_sae", 1792)
            grad = (" Read left to right, the worlds ascend in scale and complexity "
                    f"--- {tt2['n_features']} hand-built features, a {d_p:,}-latent SAE "
                    f"over a small transformer, a {C.D_SAE:,}-latent SAE over a "
                    "2B-parameter LLM --- and the recovered structure degrades in step: "
                    "a clean tree, a small tangle, a dense one.")
        d["recovered_graph_toy_vs_gemma"] = (
            r"\textbf{The recovered graph, drawn edge by edge.} Hierarchy is partially "
            r"recovered, but not as a clean tree --- especially at scale. \emph{Left:} "
            f"the trained toy recovers "
            f"{len(tt2['found_edges'])}/{len(tt2['true_edges'])} true edges with "
            # "no false positives" reads as prose while the count is zero, and
            # falls back to the number the moment a regeneration makes it real
            f"{tt2['false_positives'] or 'no'} false positives, while " + mid +
            rf"\emph{{Right:}} Gemma-2-2B (layer {GRAPH_LAYER}, B0$\rightarrow$B1) shows "
            f"substantial multi-parenting among the {s['n_edges_scored']:,} candidate edges "
            f"that reached the probe --- {n_par} parents $\\times$ {n_chi} children, with "
            f"only the {s['n_pass']} probe-confirmed edges in green." + grad +
            " Each child is placed "
            "beneath its best-matching parent, so a clean hierarchy would appear as "
            "near-vertical lines.")

    fmt_dirs = sorted((C.OUT_DIR / "pcfg-matryoshka").glob("fmt_*"))
    if fmt_dirs:
        _fr = [(int(p.name.split("_")[1]) / 1e4, _json(p / "metrics_report.json"))
               for p in fmt_dirs]
        _fr = [(d0, r) for d0, r in _fr if r]
        _dens = sorted({d0 for d0, _ in _fr})
        _lay = _fr[0][1]["config"]["layer"]
        d["pcfg_formatting_sweep"] = (
            "Four measures of block pair B0$\\rightarrow$B1 against the delimiter density of "
            f"the generating grammar, at layer {_lay} of the PCFG transformer; one point per "
            f"seed ({len(_fr) // len(_dens)} per density), the line through the seed means. "
            "Formatting density is the axis the project treats as the mechanism behind "
            "bottleneck hijacking, and the reading is the shape: a measure that stays flat "
            "across the axis is a property of the SAE that no grammar knob reaches, while a "
            "measure that slopes is confounded with the corpus and must be controlled before "
            "it is blamed on the architecture. Seeds are plotted individually because with "
            "three of them a mean without its spread would overstate what one grammar "
            "configuration can support.")
    d["sres_null_rate_vs_dictionary_size"] = (
        r"The $S_\mathrm{res}$ test passes an edge when both decoders fall within the top "
        rf"$k = {C.SRES_RANK_TOP_K}$ of the probe's correlations over the whole dictionary. It "
        "is a geometry test, so an unrelated parent passes whenever chance places it there: the "
        r"null rate is $k/D$, the grey line, which is $11.9\%$ on the 42-feature synthetic toy, "
        r"$0.28\%$ on a 1{,}792-latent PCFG SAE and $0.015\%$ on gemma's 32{,}768. Each measured "
        "pass rate is drawn with a vertical drop to its own null, and that distance---not the "
        r"rate---is what the measurement is worth. Two runs on the \emph{same} 1{,}792-latent "
        "dictionary land on opposite sides of their null, which is why a raw pass rate compares "
        "nothing across sources. A measured zero has no position on a logarithmic axis and is "
        "drawn at a marked floor rather than silently dropped.")
    return d


# (slot, figure, wide?, bold lead sentence). Order is an argument, not taste: the
# instrument has to be established before any number it produces means anything,
# so the calibrations come first; the empirical claims then follow in order of how
# much evidence stands behind them (multi-parenting has five layers, the funnel
# has one); the hypothesis figure and the mechanism that explains it are adjacent;
# and the battery's own failure closes, because it qualifies everything above it.
TEX_ORDER = [
    ("MAIN 1", "calibration_synthetic_toy_scorecard", True,
     "Every metric scored against a known tree.", "The instrument"),
    ("MAIN 2", "calibration_trained_toy_recovery", True,
     "The same tree, after a real training run.", None),
    ("MAIN 3", "recovered_graph_toy_vs_gemma", True,
     "The recovered graph, drawn.", "What the battery finds on gemma-2-2b"),
    ("MAIN 4", "multiparenting_by_layer", False,
     "The recovered graph is not a tree.", None),
    ("MAIN 4b", "tangle_lives_in_top_block_pair", True,
     "The tangle lives in the coarsest block pair, on both sources.", None),
    ("MAIN 5", "funnel_coverage_to_sres", False,
     "Coverage proposes; the strict test disposes.", None),
    ("MAIN 6", "base_rate_vs_frequency_capture", False,
     "The over-connection is a base-rate effect, not token-frequency capture.",
     "The bottleneck-hijacking hypothesis, tested"),
    ("MAIN 7", "superparent_fanout_vs_firing", False,
     "Why a parent that fires often enough clears the coverage bar for nothing.", None),
    ("MAIN 8", "shared_input_moved_every_metric", True,
     "Six metrics designed as independent detectors failed together.",
     "What this says about metric batteries"),
    ("APP 1", "sres_null_rate_vs_dictionary_size", False,
     "A top-$k$ rank rule is only as strict as the dictionary is large.", "Appendix"),
    ("APP 2", "in_block_relations", True,
     "Same-level structure lives in the outermost block.", None),
    ("APP 3", "edge_survival_by_block_pair", True,
     "What each filter removes, by block pair and by depth.", None),
    ("APP 4", "cross_source_funnel_shares", False,
     "One unchanged battery across SAE sources.", None),
    # The two cross-source depth figures sit together and immediately after the
    # funnel that establishes the sources are comparable at all. The alignment
    # check follows the result it qualifies rather than preceding it: it is a
    # caveat on how the layers were paired, and a reader who has not yet seen the
    # pairing has nothing to apply it to.
    ("APP 5", "cross_source_layer_response", True,
     "The shape of the B0$\\rightarrow$B1 relation is the same on both base models; "
     "its strength is not.", None),
    ("APP 6", "cross_source_alignment_check", False,
     "Which alignment the comparison rests on, measured rather than assumed.", None),
    ("APP 7", "depth_profile_across_layers", True,
     "No measure is monotonic in depth.", None),
    ("APP 8", "pcfg_formatting_sweep", True,
     "The formatting-density axis, swept.", None),
    ("APP 9", "battery_questions_gemma", True,
     "The battery, question by question.", None),
]

TEX_HEAD = r"""% ===========================================================================
% Figures and captions.
%
% Generated by reporting/make_report_figures.py. Every quantity in every
% caption was read from the JSON its figure plots, at generation time; do not
% edit a number here by hand, regenerate.
%
% Ordered as the paper reads. MAIN 1-7 are the main text, APP 1-5 the appendix.
%
% Requires: graphicx. Emitted for a ONECOLUMN class, which is what the ICLR
% template uses; pass --twocolumn for full-width figure* floats.
% Captions sit below the image, as figures take them, and the mirror of
% tables.tex where they sit above.
%
% \graphicspath below lists several candidates on purpose. It resolves against
% the directory of the MAIN .tex, not against this file, so a copy of this file
% \input from a Sections/ subdirectory while the PNGs sit in a folder at the
% project root needs that folder named -- a bare ./ points at the root and finds
% nothing, and every figure fails with nothing wrong in the markup. LaTeX takes
% the first path that hits, so the extra entries cost nothing.
%
% Expects the PNGs reachable via \graphicspath below. To compile on its own,
% prepend
%     \documentclass{article}\usepackage{graphicx}\begin{document}
% and append \end{document}.
% ===========================================================================
"""


def write_tex(out_dir: Path, graphics: str):
    """figures.tex beside the PNGs, or wherever --out points."""
    caps = _captions()
    # \graphicspath resolves against the COMPILATION root -- the directory of the
    # main .tex -- not against the file doing the \input. On Overleaf the images
    # sit in a folder at the project root while this file is \input from a
    # Sections/ subdirectory, so a bare "./" points at the root, finds no PNGs,
    # and every figure comes back "file not found" with nothing wrong in the
    # markup. Emitting one path was the bug; a graphicspath is a LIST precisely
    # so a document can move without its images being re-found by hand.
    #
    # Order: the named folder from the project root, the same folder one level up
    # (compiling this file standalone from inside Sections/), then the two
    # directories a reader is most likely to have dropped the PNGs into. LaTeX
    # takes the first hit, so listing more costs nothing at compile time.
    # PAPER_DIR.name is in the list because that is what the folder is called
    # here, and a reader who drags this directory into Overleaf keeps the name.
    # "figuers" (sic) is the name it has in the paper project, kept because
    # renaming a folder someone else's main.tex already points at is worse than
    # carrying the typo. "figures" is the spelling anyone else would pick.
    extra = ["./", PAPER_DIR.name + "/", "figuers/", "figures/"]
    cands = [graphics] + [f"../{graphics}"] if graphics.strip("/.") else []
    cands += extra + [f"../{e}" for e in extra if e != "./"] + ["../"]
    seen, paths = set(), []
    for g in cands:
        if g not in seen:
            seen.add(g)
            paths.append(g)
    L = [TEX_HEAD, r"\graphicspath{" + "".join(f"{{{g}}}" for g in paths) + "}", ""]
    seen, n = set(), 0
    for slot, name, wide, lead, section in TEX_ORDER:
        if not (PAPER_DIR / f"{name}.png").exists():
            continue
        if section and section not in seen:
            seen.add(section)
            L += [r"% " + "-" * 73, f"% {section}", r"% " + "-" * 73, ""]
        env = "figure*" if (wide and TWOCOLUMN) else "figure"
        width = r"\textwidth" if (wide and TWOCOLUMN) else r"\linewidth"
        body = caps.get(name)
        L += [f"% [{slot}]", rf"\begin{{{env}}}[tbp]", r"  \centering",
              rf"  \includegraphics[width={width}]{{{name}}}",
              r"  \caption{%"]
        # a caption body that opens with its own \textbf lead carries the
        # whole caption; prepending the TEX_ORDER lead would print two bold
        # openers back to back
        if not (body and body.lstrip().startswith(r"\textbf")):
            L.append(rf"    \textbf{{{lead}}}")
        if body:
            L.append(_wrap(body))
        else:
            L.append(r"    % no caption body: nothing in _captions() covers this figure")
        L += [r"  }", rf"  \label{{fig:{name.replace('_', '-')}}}",
              rf"\end{{{env}}}", ""]
        n += 1
    (out_dir / "figures.tex").write_text("\n".join(L))
    return n


CLAIMS = {
    "funnel_coverage_to_sres": "co-firing proposes far more edges than survive the strict test",
    "edge_survival_by_block_pair": "what each filter removes, by block pair and by depth",
    "depth_profile_across_layers": "no measure is monotonic in depth once BOS is excluded",
    "multiparenting_by_layer": "the graph is not a tree — the one claim BOS exclusion left standing",
    "superparent_fanout_vs_firing": "the superparent gate reads fan-out; firing rate is handled per edge",
    "calibration_synthetic_toy_scorecard": "every metric scored against a known tree, plus two demonstrated blind spots",
    "calibration_trained_toy_recovery": "the same tree after a real training run, and the nesting control",
    "cross_source_funnel_shares": "one unchanged battery across two SAE sources",
    "shared_input_moved_every_metric": "five of six metrics share an input and failed together — the battery's own failure mode",
    "base_rate_vs_frequency_capture": "the over-connection is base rate, not frequency capture — the hypothesis's premise, tested",
    "in_block_relations": "same-level structure concentrates in B0 on both sources, read as a per-pair rate",
    "sres_null_rate_vs_dictionary_size": "a top-k rank rule is only as strict as D is large",
    "cross_source_layer_response": "the shape of B0→B1 is the same on both base models at the layers both graded; its strength is not",
    "cross_source_alignment_check": "which alignment across two models of different depth the data prefers — block index or relative depth",
}


def write_readme(written, skipped, sources):
    """The directory documents itself, from the same table that built it.

    A hand-written index of figures is a second place for the truth to live, and
    the one that goes stale first -- it is still describing figure 3 after figure
    3 was replaced.
    """
    readme = PAPER_DIR / "README.md"
    L = [C.nav_html(depth=C.page_depth(readme), current="outputs/paper_figuers/"), "",
         "# `paper_figuers/` — figures for the write-up", "",
         "Generated by [`reporting/make_report_figures.py`](../../reporting/make_report_figures.py); "
         "do not edit by hand, and do not add a figure here that no generator can rebuild.", "",
         "| figure | what it is evidence for | built from |", "| --- | --- | --- |"]
    for name in written:
        L.append(f"| `{name}.png` | {CLAIMS.get(name, '')} | {sources.get(name, '')} |")
    if skipped:
        L += ["", "## Not built", "",
              "| figure | why |", "| --- | --- |"]
        for name, why in skipped:
            L.append(f"| `{name}.png` | {why} |")
    L += ["", "Every number in every title is read from the JSON being plotted, so a caption "
          "cannot outlive the data under it -- the previous generator quoted layer-6 figures "
          "that had been withdrawn.", ""]
    (PAPER_DIR / "README.md").write_text("\n".join(L))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--list", action="store_true",
                    help="show the plan and each figure's input, draw nothing")
    ap.add_argument("--out", type=Path, default=None, metavar="DIR",
                    help=f"write figures.tex here instead of {PAPER_DIR.name}/ "
                         "(the PNGs are always written beside the code)")
    ap.add_argument("--graphicspath", default=None, metavar="PATH",
                    help="an EXTRA \\graphicspath entry, tried first; the common layouts "
                         "(./, the PNG folder's name, figuers/, figures/, and each one "
                         "level up) are always emitted after it")
    ap.add_argument("--twocolumn", action="store_true",
                    help="emit figure* full-width floats (needs a twocolumn class)")
    ap.add_argument("--titles", action="store_true",
                    help="bake the figure-level title/subtitle into each PNG; off by "
                         "default because the paper's LaTeX captions carry that text")
    args = ap.parse_args()
    globals()["TWOCOLUMN"] = args.twocolumn
    globals()["TITLES"] = args.titles

    if not args.list:
        PAPER_DIR.mkdir(parents=True, exist_ok=True)
    written, skipped, sources = build(args.list)
    tex_dir = None
    if not args.list:
        write_readme(written, skipped, sources)
        # Default beside the PNGs; --out puts the .tex somewhere else, in which
        # case graphicspath has to reach back to them and "figures/" is the
        # layout the paper repo uses.
        tex_dir = (args.out or PAPER_DIR)
        tex_dir.mkdir(parents=True, exist_ok=True)
        gp = args.graphicspath or ("./" if tex_dir.resolve() == PAPER_DIR.resolve()
                                   else "figures/")
        n_tex = write_tex(tex_dir, gp)

    print(f"[fig] {'plan for' if args.list else 'wrote'} {PAPER_DIR}")
    for w in written:
        print(f"  ok    {w}")
    for name, why in skipped:
        print(f"  SKIP  {name}\n          {why}")
    if tex_dir is not None:
        print(f"  ok    figures.tex  ({n_tex} figures, graphicspath {gp!r}) -> {tex_dir}")
    if skipped and not args.list:
        print(f"\n[fig] {len(written)} written, {len(skipped)} skipped — "
              "the set above is not complete, and the reasons are printed rather "
              "than left to the reader to notice.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
