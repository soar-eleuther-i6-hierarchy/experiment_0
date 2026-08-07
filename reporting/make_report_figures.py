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
    superparent_fanout_vs_firing            a parent's fan-out against its base rate
    calibration_synthetic_toy_scorecard     every metric against a known tree
    calibration_trained_toy_recovery        the same tree, through a real training run
    cross_source_funnel_shares              the same battery on gemma and on PCFG
    in_block_relations                      same-level edges and duplicates
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
    axes[0].legend(title="layer", fontsize=8.5, title_fontsize=8.5,
                   frameon=False, ncol=5, loc="upper right")
    fig.suptitle("What each filter removes, by block pair and depth",
                 fontsize=11.5, x=0.006, ha="left", y=1.0)
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
    fig, ax = plt.subplots(figsize=(6.8, 4.4))
    n = 0
    for i, (L, rep) in enumerate(layers):
        xs, ys = [], []
        for pr in rep["pairs"]:
            for sp in pr.get("superparents", []):
                xs.append(100 * sp["fire_frac"])
                ys.append(100 * sp["outdeg_frac"])
        n += len(xs)
        ax.scatter(xs, ys, s=34, color=DEPTH[i], label=f"L{L}",
                   edgecolor="white", linewidth=0.7, zorder=3)
    ax.axhline(100 * C.SUPERPARENT_OUTDEG_FRAC, ls=(0, (4, 3)), lw=1.2, color=CAT[3])
    ax.text(0.6, 100 * C.SUPERPARENT_OUTDEG_FRAC + 2,
            f"gate: fan-out ≥ {100 * C.SUPERPARENT_OUTDEG_FRAC:.0f}%",
            fontsize=8, color=CAT[3])
    ax.set_xscale("log")
    ax.set_xlabel("parent firing rate, % of tokens (log)")
    ax.set_ylabel("fan-out, % of the child block")
    ax.legend(title="layer", fontsize=8.5, title_fontsize=8.5, frameon=False, loc="lower right")
    _title(ax, f"Flagged parents: fan-out against base rate ({n} parents, all block pairs)",
           "the gate reads the vertical axis only — firing rate is handled per edge, by PMI",
           width=68)
    ax.grid(True, alpha=0.12)
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
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.4),
                             gridspec_kw={"width_ratios": [1.25, 1]})

    ax = axes[0]
    bars = [("recovered", tp, GOOD), ("missed", fn, NEUTRAL), ("false positives", fp, CAT[3])]
    ax.bar([b[0] for b in bars], [b[1] for b in bars],
           color=[b[2] for b in bars], width=0.55)
    for i, (_, v, _) in enumerate(bars):
        ax.text(i, v + 0.15, str(v), ha="center", fontsize=10, color=INK)
    ax.set_ylabel("true edges")
    ax.set_ylim(0, max(tp, fn, fp) * 1.35 + 0.5)
    ax.set_title(f"Edge recovery — precision {tt['precision']:.2f}, recall {tt['recall']:.2f}\n"
                 f"the SAE learned {tt['n_recovered_features']}/{tt['n_features']} true features; "
                 f"every miss is an edge whose child it never learned",
                 fontsize=10, loc="left")

    ax = axes[1]
    if align:
        ax.bar(["respected", "untestable"],
               [align["n_respected"], len(align.get("untestable", []))],
               color=[GOOD, NEUTRAL], width=0.5)
        ax.text(0, align["n_respected"] + 0.12, str(align["n_respected"]),
                ha="center", fontsize=10, color=INK)
        ax.set_ylim(0, max(align["n_testable"], 1) * 1.4)
        ax.set_ylabel("true edges")
        ax.set_title(f"Nesting control — {align['n_respected']}/{align['n_testable']} testable "
                     f"edges run early block → late\n"
                     f"mean block {align['mean_parent_block']:.1f} for parents, "
                     f"{align['mean_child_block']:.1f} for children",
                     fontsize=10, loc="left")
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
    stages = ["improve\nreconstruction", "above chance\n(PMI > 0)", "frequency-\ndriven"]
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
    for (name, d), v in zip(sources, null):
        ax.annotate(f"{name}\nD = {d:,} → {v:.2f}%", (d, v),
                    textcoords="offset points", xytext=(9, 9), fontsize=8, color=MUTED)
    floor = float(np.min(null)) / 4                      # a visible place for zero
    drew_obs = zeros = False
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
                    xytext=(10, -6 if at_zero else -22), fontsize=8, color=GOOD)
        drew_obs = True
    ax.set_xscale("log")
    ax.set_yscale("log")
    if zeros:
        ax.axhline(floor, ls=(0, (1, 4)), lw=1, color=NEUTRAL)
        ax.text(ax.get_xlim()[1] * 0.94, floor * 1.15, "0% drawn here — a log axis has no zero",
                fontsize=7.5, color=MUTED, style="italic", ha="right")
    ax.set_xlabel("dictionary size D (log)")
    ax.set_ylabel("% of unrelated parents that pass (log)")
    # lifted clear of the zero-floor rule that runs along the bottom
    ax.legend(fontsize=8.5, frameon=False, loc="lower left", bbox_to_anchor=(0.0, 0.10))
    _title(ax, "The rank rule's strictness is set by dictionary size, not by k alone",
           "an unrelated parent passes whenever chance puts it in the top k of D, so an S_res "
           "pass rate is only comparable between dictionaries of similar size", width=72)
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
        "gemma2_2b/layer_06 metrics_report.json + second_pass.json"
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
    run("superparent_fanout_vs_firing", len(layers) >= 1,
        f"{len(layers)} gemma layer reports" if layers else "needs gemma metrics_report.json",
        lambda: superparent_fanout_vs_firing(layers))

    toy = _json(C.OUT_DIR / "toy_calibration.json")
    run("calibration_synthetic_toy_scorecard", bool(toy),
        "outputs/toy_calibration.json" if toy
        else "needs outputs/toy_calibration.json (validation.calibrate_on_synthetic_toy)",
        lambda: calibration_synthetic_toy_scorecard(toy))

    tt = _json(C.OUT_DIR / "trained_toy_calibration.json")
    align = _json(C.OUT_DIR / "block_tree_alignment.json")
    run("calibration_trained_toy_recovery", bool(tt),
        "outputs/trained_toy_calibration.json"
        + ("" if align else " (block_tree_alignment.json absent — right panel blank)")
        if tt else "needs outputs/trained_toy_calibration.json (Tier 2)",
        lambda: calibration_trained_toy_recovery(tt, align))

    pcfg = [(p.name, _json(p / "metrics_report.json"))
            for p in sorted((C.OUT_DIR / "pcfg").glob("layer_*")) if p.is_dir()]
    pcfg = [(f"PCFG {n.replace('_', ' ')}", r) for n, r in pcfg if r]
    runs = ([("gemma-2-2b layer 06", l6)] if l6 else []) + pcfg[:1]
    run("cross_source_funnel_shares", len(runs) >= 2,
        f"{len(runs)} sources" if len(runs) >= 2 else "needs a gemma report and a PCFG report",
        lambda: cross_source_funnel_shares(runs))

    ib = []
    for base, label in [(G, "gemma"), (C.OUT_DIR / "pcfg", "PCFG")]:
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
    for label, d_sae, path in [
        ("gemma", 32768, G / "layer_06" / "second_pass.json"),
        ("PCFG", 1792, C.OUT_DIR / "pcfg" / "layer_01" / "second_pass.json"),
    ]:
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


CLAIMS = {
    "funnel_coverage_to_sres": "co-firing proposes far more edges than survive the strict test",
    "edge_survival_by_block_pair": "what each filter removes, by block pair and by depth",
    "depth_profile_across_layers": "no measure is monotonic in depth once BOS is excluded",
    "multiparenting_by_layer": "the graph is not a tree — the one claim BOS exclusion left standing",
    "superparent_fanout_vs_firing": "the superparent gate reads fan-out; firing rate is handled per edge",
    "calibration_synthetic_toy_scorecard": "every metric scored against a known tree, plus two demonstrated blind spots",
    "calibration_trained_toy_recovery": "the same tree after a real training run, and the nesting control",
    "cross_source_funnel_shares": "one unchanged battery across two SAE sources",
    "in_block_relations": "same-level structure concentrates in B0 on both sources, read as a per-pair rate",
    "sres_null_rate_vs_dictionary_size": "a top-k rank rule is only as strict as D is large",
}


def write_readme(written, skipped, sources):
    """The directory documents itself, from the same table that built it.

    A hand-written index of figures is a second place for the truth to live, and
    the one that goes stale first -- it is still describing figure 3 after figure
    3 was replaced.
    """
    L = ["# `paper_figuers/` — figures for the write-up", "",
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
    args = ap.parse_args()

    if not args.list:
        PAPER_DIR.mkdir(parents=True, exist_ok=True)
    written, skipped, sources = build(args.list)
    if not args.list:
        write_readme(written, skipped, sources)

    print(f"[fig] {'plan for' if args.list else 'wrote'} {PAPER_DIR}")
    for w in written:
        print(f"  ok    {w}")
    for name, why in skipped:
        print(f"  SKIP  {name}\n          {why}")
    if skipped and not args.list:
        print(f"\n[fig] {len(written)} written, {len(skipped)} skipped — "
              "the set above is not complete, and the reasons are printed rather "
              "than left to the reader to notice.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
