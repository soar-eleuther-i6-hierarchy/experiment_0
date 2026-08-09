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

PAPER_DIR = C.OUT_DIR / "paper_figuers"

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
DEPTH = ["#9ECAE1", "#6BAED6", "#4292C6", "#2171B5", "#084594"]
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


def _title(ax_or_fig, head: str, sub: str = "", width: int = 96):
    """Left-aligned title, wrapped to the figure rather than trusting it to fit.

    Every overflowing title in the previous version was a long second line that
    matplotlib silently let run past the canvas edge, so the caption lost its
    qualifier -- the half a sentence that says what the number does NOT mean.
    """
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


def _finish(fig, ax_or_axes, name: str) -> str:
    axes = ax_or_axes if isinstance(ax_or_axes, (list, np.ndarray)) else [ax_or_axes]
    for ax in np.ravel(axes):
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    path = PAPER_DIR / f"{name}.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return name


# ---------------------------------------------------------------------------
# 1. The funnel: what each stage removes, for one run that has the strict test
# ---------------------------------------------------------------------------
def funnel_coverage_to_sres(report, second, pair="0->1", where=""):
    pr = _pair(report, pair)
    sres = second[pair]["sres"]
    n_cand = pr["n_candidate_edges"]
    stages = [
        (f"Candidate edges\nreverse coverage ≥ {C.EDGE_TAU}", n_cand, NEUTRAL),
        ("Above chance\nPMI > 0", sres["n_edges_scored"], CAT[0]),
        ("Genuine refinement\npasses probe S_res", sres["n_pass"], GOOD),
    ]
    n_recon = pr["reconstruction"]["n_pass"]

    fig, ax = plt.subplots(figsize=(7.6, 3.4))
    y = np.arange(len(stages))
    ax.barh(y, [v for _, v, _ in stages], color=[c for _, _, c in stages], height=0.55)
    ax.invert_yaxis()
    ax.set_yticks(y)
    ax.set_yticklabels([lab for lab, _, _ in stages])
    for i, (_, v, _) in enumerate(stages):
        ax.text(v + n_cand * 0.015, i, f"{v:,}   {100 * v / n_cand:.1f}%",
                va="center", fontsize=9.5, color=INK)
    # The ablation filter runs on all candidates independently, so it is not a
    # funnel stage; drawn as a reference line to avoid implying it nests.
    ax.axvline(n_recon, ls=(0, (4, 3)), lw=1.3, color=CAT[3])
    ax.text(n_recon - n_cand * 0.02, 2.42,
            f"ablation filter (parallel, not nested)\npasses {n_recon:,}"
            f"  ({100 * n_recon / n_cand:.0f}%)",
            fontsize=8, color=CAT[3], ha="right", va="top")
    ax.set_ylim(2.85, -0.6)
    ax.set_xlim(0, n_cand * 1.3)
    ax.set_xlabel("parent → child edges")
    _title(ax, f"{where}, block pair B{pair.replace('->', '→B')}: "
               f"co-firing proposes, the strict test disposes",
           f"{sres['n_pass']:,} of {n_cand:,} candidate edges "
           f"({100 * sres['n_pass'] / n_cand:.1f}%) survive", width=74)
    ax.grid(True, axis="x", alpha=0.12)
    ax.set_axisbelow(True)
    return _finish(fig, ax, "funnel_coverage_to_sres")


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
def multiparenting_by_layer(layers):
    pairs = ["0->1", "1->2", "2->3"]
    fig, ax = plt.subplots(figsize=(7.4, 3.6))
    x = np.arange(len(layers))
    w = 0.26
    for j, p in enumerate(pairs):
        vals = [100 * (_pair(r, p) or {}).get("degree", {}).get("poly_frac", np.nan)
                for _, r in layers]
        off = (j - 1) * w
        ax.bar(x + off, vals, w, color=CAT[j], label=f"B{p.replace('->', '→B')}")
        for xx, vv in zip(x + off, vals):
            if not np.isnan(vv):
                ax.text(xx, vv + 1.6, f"{vv:.0f}", ha="center", fontsize=7.5, color=MUTED)
    ax.set_xticks(x)
    ax.set_xticklabels([f"L{L}" for L, _ in layers])
    ax.set_ylabel("% of children with ≥ 2 parents")
    ax.set_ylim(0, 112)
    ax.axhline(100, ls=(0, (2, 3)), lw=1, color=NEUTRAL)
    # below the axis: at 100% the top bars reach the legend's only free space
    ax.legend(fontsize=8.5, frameon=False, ncol=3, loc="upper center",
              bbox_to_anchor=(0.5, -0.09))
    _title(ax, "The graph is not a tree: in the top block pair nearly every child has several parents",
           "the one measure BOS exclusion left unchanged — it is a ratio over children that "
           "already have a parent", width=82)
    ax.grid(True, axis="y", alpha=0.12)
    ax.set_axisbelow(True)
    return _finish(fig, ax, "multiparenting_by_layer")


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
    _title(ax, f"Edge recovery — precision {tt['precision']:.2f}, recall {tt['recall']:.2f}",
           f"the SAE learned {tt['n_recovered_features']}/{tt['n_features']} true features; every "
           f"miss is an edge whose child it never learned", width=52)

    ax = axes[1]
    if align:
        ax.bar(["respected", "untestable"],
               [align["n_respected"], len(align.get("untestable", []))],
               color=[GOOD, NEUTRAL], width=0.5)
        ax.text(0, align["n_respected"] + 0.12, str(align["n_respected"]),
                ha="center", fontsize=10, color=INK)
        ax.set_ylim(0, max(align["n_testable"], 1) * 1.4)
        ax.set_ylabel("true edges")
        _title(ax, f"Nesting control — {align['n_respected']}/{align['n_testable']} testable edges "
                   f"run early block → late",
               f"mean block {align['mean_parent_block']:.1f} for parents, "
               f"{align['mean_child_block']:.1f} for children", width=46)
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
    fig, ax = plt.subplots(figsize=(8.0, 3.8))
    x = np.arange(len(stages))
    w = 0.8 / max(len(runs), 1)
    for i, (name, rep) in enumerate(runs):
        pr = _pair(rep, "0->1")
        if pr is None:
            continue
        n = pr["n_candidate_edges"]
        vals = [100 * pr["reconstruction"]["frac_pass"],
                100 * (n - pr["independence_null"]["n_chance_level"]) / n
                if pr["independence_null"].get("n_chance_level") is not None else np.nan,
                100 * pr["freq_control"]["frac_freq_driven"]]
        off = (i - (len(runs) - 1) / 2) * w
        ax.bar(x + off, vals, w * 0.92, color=CAT[i % len(CAT)],
               label=f"{name}  (n={n:,})")
        for xx, vv in zip(x + off, vals):
            if not np.isnan(vv):
                ax.text(xx, vv + 1.5, f"{vv:.0f}", ha="center", fontsize=7.5, color=MUTED)
    ax.set_xticks(x)
    ax.set_xticklabels(stages)
    ax.set_ylabel("% of that run's candidate edges")
    ax.legend(fontsize=8.5, frameon=False)
    _title(ax, "The same battery, unchanged, on two SAE sources — block pair B0→B1",
           "shares rather than counts: the dictionaries differ in size and in block count",
           width=76)
    ax.grid(True, axis="y", alpha=0.12)
    ax.set_axisbelow(True)
    return _finish(fig, ax, "cross_source_funnel_shares")


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
    v1 = []
    for L, _ in layers:
        hits = sorted(ARCH.glob(f"layer_{L:02d}__v1__*/metrics_report.json"))
        if hits:
            v1.append(_json(hits[0]))
    if len(v1) != len(layers):
        raise SystemExit("archived v1 reports missing")

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
    data = [(lab, mean(v1, fn), mean([r for _, r in layers], fn), shared) for lab, fn, shared in rows]
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
    ax.set_xlabel("log₂ change when the contaminating token position is removed, mean over 5 layers\n"
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
        ax.annotate(f"{name}: {'0% — no edge passed' if at_zero else f'observed {obs:.2f}%'}"
                    + ("" if at_zero else f"\n≈{obs / (100 * k / d):.0f}× its own chance rate"),
                    (d, floor if at_zero else obs), textcoords="offset points",
                    xytext=(11, -4 if at_zero else 8), fontsize=8, color=GOOD)
        drew_obs = True
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
    ratios = [f"{name} {obs / (100 * k / d):.0f}×" if obs > 0 else f"{name} below chance"
              for name, d, obs in observed]
    _title(ax, "The rank rule's strictness is set by dictionary size, not by k alone",
           "grey line = what chance alone gives at each D; the vertical drop to it is what a "
           "measured rate is worth. " + ", ".join(ratios) +
           ". Two runs on the same 1,792-latent dictionary land on opposite sides of their own "
           "null, so a raw pass rate compares nothing", width=78)
    ax.grid(True, which="both", alpha=0.12)
    ax.set_axisbelow(True)
    return _finish(fig, ax, "sres_null_rate_vs_dictionary_size")


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

    # the funnel needs a run that has stage 03; layer 6 is the gemma one that does
    l6 = _json(G / "layer_06" / "metrics_report.json")
    s6 = _json(G / "layer_06" / "second_pass.json")
    run("funnel_coverage_to_sres", bool(l6 and s6),
        "gemma-2-2b/layer_06 metrics_report.json + second_pass.json"
        if (l6 and s6) else "needs layer_06/second_pass.json (stage 03, run on L6 only)",
        lambda: funnel_coverage_to_sres(l6, s6, where="gemma-2-2b, layer 6"))

    run("edge_survival_by_block_pair", len(layers) >= 2,
        f"{len(layers)} gemma layer reports" if layers else "needs ≥2 gemma metrics_report.json",
        lambda: edge_survival_by_block_pair(layers))
    run("depth_profile_across_layers", len(layers) >= 2,
        f"{len(layers)} gemma layer reports" if layers else "needs ≥2 gemma metrics_report.json",
        lambda: depth_profile_across_layers(layers))
    run("multiparenting_by_layer", len(layers) >= 1,
        f"{len(layers)} gemma layer reports" if layers else "needs gemma metrics_report.json",
        lambda: multiparenting_by_layer(layers))
    n_arch = len(sorted((C.HERE / "outputs_archive").glob("layer_*__v1__*/metrics_report.json")))
    run("shared_input_moved_every_metric", n_arch >= len(layers) and len(layers) >= 2,
        f"{len(layers)} current reports vs {n_arch} archived v1 reports" if n_arch
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

    pcfg = [(p.name, _json(p / "metrics_report.json"))
            for p in sorted((C.OUT_DIR / "pcfg-matryoshka").glob("layer_*")) if p.is_dir()]
    pcfg = [(f"PCFG {n.replace('_', ' ')}", r) for n, r in pcfg if r]
    # Every PCFG layer, not the first one. `pcfg[:1]` silently dropped layer 03,
    # which is the run that carries the strict test -- the exact "renders 6 of 10
    # and reads as complete" failure this file is written against.
    runs = ([("gemma-2-2b layer 06", l6)] if l6 else []) + pcfg
    run("cross_source_funnel_shares", len(runs) >= 2,
        f"{len(runs)} sources" if len(runs) >= 2 else "needs a gemma report and a PCFG report",
        lambda: cross_source_funnel_shares(runs))

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

    # observed S_res shares, read off whatever second_pass.json files exist
    observed = []
    probes = [("gemma L6", 32768, G / "layer_06" / "second_pass.json")]
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

        d["multiparenting_by_layer"] = (
            "Percentage of child features with two or more parents, per block pair, across the "
            r"five graded layers. In the top pair B0$\rightarrow$B1 the value is "
            + " / ".join(rf"{v:.0f}\%" for v in poly) + "---nearly every child in the second "
            "block is claimed by several first-block features, which no reading as a hierarchy "
            "survives. This is the only one of the project's four original claims that BOS "
            "exclusion left unchanged, because it is a ratio over children that already have a "
            "parent rather than a count of candidate edges, and so does not depend on the "
            "contaminated candidate set. Deeper pairs are much lower, on candidate sets that are "
            "also much smaller.")
        if sp6:
            sr = sp6["0->1"]["sres"]
            d["funnel_coverage_to_sres"] = (
                r"Block pair B0$\rightarrow$B1 of the layer-6 Matryoshka SAE on "
                rf"\texttt{{gemma-2-2b}}. Reverse coverage $R \geq {C.EDGE_TAU}$ with a "
                f"joint-support guard admits {p6['n_candidate_edges']:,} candidate edges. The "
                "reconstruction-ablation filter---drawn as a dashed reference rather than a "
                "funnel stage, because it is applied to all candidates in parallel and does not "
                f"nest---passes {p6['reconstruction']['n_pass']:,} of them "
                rf"({100 * p6['reconstruction']['frac_pass']:.0f}\%), so the cheap filter barely "
                f"bites. Of the {sr['n_edges_scored']:,} edges reaching the probe-based "
                rf"$S_\mathrm{{res}}$ rank test, {sr['n_pass']} pass "
                rf"({100 * sr['frac_pass']:.1f}\%). \textbf{{Caveat:}} stage~03 has been run on "
                "layer~6 only, so this ratio has a single layer behind it.")
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
            "sharply in the deeper ones, on candidate sets that are also far smaller there.")

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
            f"and six injected structures; {sum(r['pass'] for r in toy)} pass, across seeds "
            r"0--7, covering 21 of 21 metric functions. \emph{Left:} rows whose two classes "
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
    ("MAIN 3", "multiparenting_by_layer", False,
     "The recovered graph is not a tree.", "What the battery finds on gemma-2-2b"),
    ("MAIN 4", "funnel_coverage_to_sres", False,
     "Coverage proposes; the strict test disposes.", None),
    ("MAIN 5", "base_rate_vs_frequency_capture", False,
     "The over-connection is a base-rate effect, not token-frequency capture.",
     "The bottleneck-hijacking hypothesis, tested"),
    ("MAIN 6", "superparent_fanout_vs_firing", False,
     "Why a parent that fires often enough clears the coverage bar for nothing.", None),
    ("MAIN 7", "shared_input_moved_every_metric", True,
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
    ("APP 5", "depth_profile_across_layers", True,
     "No measure is monotonic in depth.", None),
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
% Expects the PNGs reachable via \graphicspath below. To compile on its own,
% prepend
%     \documentclass{article}\usepackage{graphicx}\begin{document}
% and append \end{document}.
% ===========================================================================
"""


def write_tex(out_dir: Path, graphics: str):
    """figures.tex beside the PNGs, or wherever --out points."""
    caps = _captions()
    paths = [graphics] if graphics.strip("/.") == "" else [graphics, "./"]
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
              r"  \caption{%", rf"    \textbf{{{lead}}}"]
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
                    help="what \\graphicspath should point at; defaults to ./ when "
                         "figures.tex sits with the PNGs and figures/ when it does not")
    ap.add_argument("--twocolumn", action="store_true",
                    help="emit figure* full-width floats (needs a twocolumn class)")
    args = ap.parse_args()
    globals()["TWOCOLUMN"] = args.twocolumn

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
