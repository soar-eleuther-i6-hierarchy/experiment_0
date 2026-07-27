"""
Static proof-figures for the layer-6 team report (REPORT_layer6.md).

Five matplotlib PNGs into figures/ — chosen so each one backs a single claim,
and so they are viewable without a browser (unlike the plotly dashboards):

  fig1_metric_funnel.png     coverage edges -> PMI shortlist -> S_res pass
                             (the "coverage is necessary, not sufficient" story)
  fig2_outdegree_ccdf.png    P(out-degree >= x): the superparent heavy tail
  fig3_energy_decay.png      median E[f^2] per block: blocks are an energy ranking
  fig4_superparent_sres.png  per superparent, fan-out edges colored by S_res verdict
  fig5_in_block.png          within-block same-level edges per block
                             (only when in_block_edges.json is present)

Figure titles quote the layer-6 result numbers, so update them if you
regenerate on other data.

Reads the layer's exp0_stats.pt (edge sets + energy), second_pass.json (S_res per
edge), metrics_report.json (funnel counts), and the token cache (per-feature
energy). Run on the server where the data lives:
    EXP0_OUT=.../outputs_local python make_report_figures.py
Output: <RUN_DIR>/figures/*.png  (also copied to a repo-level figures/ by the
caller for tracking).
"""

from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402

import config as C  # noqa: E402
from metrics import coverage_legs, keep_edges  # noqa: E402
from run_second_pass import TokenCache  # noqa: E402

GREEN, GREY, INK, ACCENT = "#2E9E5B", "#B8BCC4", "#2B2B33", "#7C4DD1"
FIG_DIR = C.RUN_DIR / "figures"


def _edge_mask(stats, p_blk, c_blk):
    key = f"{p_blk}->{c_blk}"
    fire = stats["fire_count"].double()
    p0, p1 = C.BLOCK_RANGES[p_blk]
    c0, c1 = C.BLOCK_RANGES[c_blk]
    fire_p, fire_c = fire[p0:p1], fire[c0:c1]
    cofire = stats["cofire"][key].double()
    R, _ = coverage_legs(cofire, fire_p, fire_c)
    return keep_edges(R, fire_p, fire_c, C.EDGE_TAU, C.MIN_FIRE_COUNT,
                      cofire=cofire, min_joint=C.MIN_JOINT)


def fig_metric_funnel(report, second, pair="0->1"):
    """NESTED funnel: coverage ⊃ PMI>0 shortlist ⊃ S_res pass. The weak ablation
    baseline is a PARALLEL filter (applied to all candidates independently), so it
    is drawn as a dashed reference line, not a funnel stage, to avoid implying it
    nests with the others."""
    pr = next(p for p in report["pairs"] if p["pair"] == pair)
    sres = second[pair]["sres"]
    n_cand = pr["n_candidate_edges"]
    n_recon = pr["reconstruction"]["n_pass"]           # weak baseline (parallel)
    n_short = sres["n_edges_scored"]                   # PMI > 0 shortlist (nested)
    n_sres = sres["n_pass"]                            # genuine refinement (nested)

    labels = [
        f"Coverage edges\n(reverse cov ≥ {C.EDGE_TAU})",
        "Survive PMI > 0\n(above chance)",
        "Pass probe-S_res\n(genuine refinement)",
    ]
    vals = [n_cand, n_short, n_sres]
    cols = [GREY, ACCENT, GREEN]
    fig, ax = plt.subplots(figsize=(8.8, 4.0))
    y = range(len(vals))
    ax.barh(list(y), vals, color=cols, height=0.58)
    ax.invert_yaxis()
    ax.set_yticks(list(y))
    ax.set_yticklabels(labels, fontsize=10)
    for i, v in enumerate(vals):
        ax.text(v + n_cand * 0.012, i, f"{v:,}  ({100*v/n_cand:.1f}%)",
                va="center", fontsize=10, color=INK)
    # weak baseline: dashed parallel reference, NOT a nested stage
    ax.axvline(n_recon, ls="--", lw=1.4, color="#C0504D")
    ax.text(n_recon - n_cand * 0.02, 2.38,
            f"← weak baseline (ablation filter, parallel)\n"
            f"   would pass {n_recon:,} ({100*n_recon/n_cand:.0f}%)",
            fontsize=8.5, color="#C0504D", ha="right", va="top")
    ax.set_ylim(2.75, -0.6)
    ax.set_xlim(0, n_cand * 1.24)
    ax.set_xlabel("number of parent→child edges", fontsize=10)
    ax.set_title(f"Layer 6, block B0→B1: coverage ⊃ PMI ⊃ S_res (nested funnel)\n"
                 f"only {n_sres:,} of {n_cand:,} candidate edges ({100*n_sres/n_cand:.1f}%) are genuine refinements",
                 fontsize=11.5, color=INK)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig1_metric_funnel.png", dpi=140)
    plt.close(fig)


def fig_outdegree_ccdf(stats):
    """CCDF P(outdeg >= x) per block pair — a heavy right tail = superparents."""
    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    pairs = [(0, 1), (1, 2), (2, 3)]
    cols = [ACCENT, "#2E9E5B", "#D98A3D"]
    for (p, c), col in zip(pairs, cols):
        outdeg = _edge_mask(stats, p, c).sum(dim=1)
        outdeg = outdeg[outdeg > 0].sort(descending=True).values.double()
        n = outdeg.numel()
        xs = outdeg.numpy()
        ccdf = (torch.arange(1, n + 1).double() / n).numpy()
        c1 = C.BLOCK_RANGES[c][1] - C.BLOCK_RANGES[c][0]
        ax.step(xs, ccdf, where="post", color=col, lw=2,
                label=f"B{p}→B{c}  (child block = {c1} feats)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("out-degree  x  (children per parent, log)", fontsize=10)
    ax.set_ylabel("P(out-degree ≥ x)", fontsize=10)
    ax.set_title("Out-degree tail: a few parents fan out to almost the whole child block\n"
                 "(the flat-then-cliff shape at the right = superparents)",
                 fontsize=11.5, color=INK)
    ax.legend(fontsize=9, frameon=False)
    ax.grid(True, which="both", alpha=0.15)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig2_outdegree_ccdf.png", dpi=140)
    plt.close(fig)


def fig_energy_decay(cache):
    """Median per-feature energy E[f^2] per block, log-y — the Theorem-2 claim
    that the nested blocks are an energy ranking."""
    e = torch.zeros(C.D_SAE, dtype=torch.float64)
    e.scatter_add_(0, cache.f_feats, cache.f_vals.double() ** 2)
    E = e / max(cache.n_tokens, 1)                     # E[f^2] per feature
    meds, los, his, names = [], [], [], []
    for b, (s, t) in enumerate(C.BLOCK_RANGES):
        vals = E[s:t]
        vals = vals[vals > 0]                          # alive features only
        if vals.numel() == 0:
            continue
        q = torch.quantile(vals, torch.tensor([0.25, 0.5, 0.75], dtype=torch.float64))
        los.append(float(q[0])); meds.append(float(q[1])); his.append(float(q[2]))
        names.append(f"B{b}\n[{s}:{t}]")
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    x = range(len(meds))
    ax.plot(list(x), meds, "-o", color=ACCENT, lw=2, label="median E[f²]")
    ax.fill_between(list(x), los, his, color=ACCENT, alpha=0.15, label="IQR")
    ax.set_yscale("log")
    ax.set_xticks(list(x))
    ax.set_xticklabels(names, fontsize=9)
    ax.set_ylabel("per-feature energy  E[f²]  (log)", fontsize=10)
    fold = meds[0] / meds[-1] if meds[-1] > 0 else float("inf")
    ax.set_title(f"Energy falls monotonically across the nested blocks (≈{fold:.0f}× B0→B4)\n"
                 "early blocks hold the high-energy features — blocks are an energy ranking",
                 fontsize=11.5, color=INK)
    ax.legend(fontsize=9, frameon=False)
    ax.grid(True, axis="y", which="both", alpha=0.15)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig3_energy_decay.png", dpi=140)
    plt.close(fig)


def fig_superparent_sres(report, second, pair="0->1"):
    """Per superparent: how many of its fan-out edges pass S_res (green) vs
    fail (grey). Almost all grey = it co-fires with the block but refines none."""
    pr = next(p for p in report["pairs"] if p["pair"] == pair)
    sps = pr["superparents"]
    edges = second[pair]["sres"]["edges"]
    passed = {}
    for e in edges:
        d = passed.setdefault(e["parent"], [0, 0])
        d[0 if e["pass"] else 1] += 1
    rows = []
    for sp in sps:
        gp = sp["parent_global"]
        npass, nfail = passed.get(gp, [0, 0])
        rows.append((gp, sp["outdeg"], npass, nfail))
    rows.sort(key=lambda r: -r[1])

    fig, ax = plt.subplots(figsize=(8.0, 0.6 * len(rows) + 1.8))
    y = range(len(rows))
    labels = [f"F{gp}\n(fires {100*next(s for s in sps if s['parent_global']==gp)['fire_frac']:.0f}%)"
              for gp, _, _, _ in rows]
    npass = [r[2] for r in rows]
    nscored = [r[2] + r[3] for r in rows]
    ax.barh(list(y), nscored, color=GREY, height=0.6, label="fail S_res (not a refinement)")
    ax.barh(list(y), npass, color=GREEN, height=0.6, label="pass S_res (genuine)")
    ax.invert_yaxis()
    ax.set_yticks(list(y)); ax.set_yticklabels(labels, fontsize=9)
    for i, (gp, od, p_, f_) in enumerate(rows):
        ax.text(p_ + f_ + 4, i, f"{p_}/{p_+f_} pass", va="center", fontsize=9, color=INK)
    ax.set_xlabel("S_res-tested fan-out edges  (a subset of each parent's full out-degree)", fontsize=10)
    ax.set_title("Superparents co-fire with a large share of the child block, but refine none of it\n"
                 "(grey = the edge co-fires yet fails the probe refinement test; 0 pass for all six)",
                 fontsize=11.5, color=INK)
    ax.legend(fontsize=9, frameon=False, loc="lower right")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig4_superparent_sres.png", dpi=140)
    plt.close(fig)


def fig_in_block(inblock):
    """Per block: directed same-level edges, duplicates, and how many survive PMI
    then S_res — log-y because B0 dwarfs B1/B2. The point: in-block structure
    concentrates in the scarce top block."""
    blocks = inblock["blocks"]
    names = [f"B{b['block']}\n({b['n_features']} feats)" for b in blocks]
    edges = [b["n_edges"] for b in blocks]
    dups = [b["n_duplicates"] for b in blocks]
    pmi = [b["n_after_pmi"] for b in blocks]
    sres = [(b["sres"]["n_pass"] if b.get("sres") else 0) for b in blocks]

    import numpy as np
    x = np.arange(len(blocks))
    w = 0.2
    fig, ax = plt.subplots(figsize=(9.2, 4.4))
    ax.bar(x - 1.5 * w, edges, w, color=GREY, label="directed edges")
    ax.bar(x - 0.5 * w, pmi, w, color=ACCENT, label="survive PMI > 0")
    ax.bar(x + 0.5 * w, sres, w, color=GREEN, label="pass S_res")
    ax.bar(x + 1.5 * w, dups, w, color="#D98A3D", label="duplicate pairs")
    ax.set_yscale("symlog")
    ax.set_xticks(x); ax.set_xticklabels(names, fontsize=9)
    ax.set_ylabel("count (symlog)", fontsize=10)
    ax.set_title("In-block (same-level) relations: the top block is a dense internal mesh\n"
                 "B0: 713 edges, 0/449 genuine, 5 superparents; deeper blocks sparse but real "
                 "(B1 1/8, B2 3/65)",
                 fontsize=10, color=INK, pad=12)
    ax.legend(fontsize=9, frameon=False, ncol=2)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig5_in_block.png", dpi=140)
    plt.close(fig)


def main():
    import sys

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    # in-block figure only needs in_block_edges.json — skip the 810 MB stats +
    # 6.3 GB token-cache loads that the cross-block figures require.
    if "--in-block-only" in sys.argv:
        fig_in_block(json.loads(C.IN_BLOCK_PATH.read_text()))
        print(f"[fig] wrote fig5_in_block -> {FIG_DIR}")
        return
    report = json.loads(C.METRICS_JSON_PATH.read_text())
    second = json.loads(C.SECOND_PASS_PATH.read_text())
    stats = torch.load(C.EXP0_STATS_PATH, weights_only=False)
    cache = TokenCache(C.TOKEN_CACHE_DIR)

    fig_metric_funnel(report, second)
    fig_outdegree_ccdf(stats)
    fig_energy_decay(cache)
    fig_superparent_sres(report, second)
    if C.IN_BLOCK_PATH.exists():
        fig_in_block(json.loads(C.IN_BLOCK_PATH.read_text()))
    print(f"[fig] wrote figures -> {FIG_DIR}")


if __name__ == "__main__":
    main()
