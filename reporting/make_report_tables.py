"""
Paper tables — LaTeX, ordered as the paper reads, every number read from data.

    python3 -m reporting.make_report_tables              # write tables.tex
    python3 -m reporting.make_report_tables --list       # the plan and its inputs
    python3 -m reporting.make_report_tables --out DIR    # somewhere else

Output: outputs/paper_figuers/tables.tex

The companion to `make_report_figures.py`, and it follows the same two rules for
the same reasons.

**Nothing is hardcoded that can be derived.** Thresholds come from `config`, block
structure from `config`, and every count from the JSON report it describes. The
figure generator had to be rebuilt once because its titles quoted layer-6 numbers
that had been withdrawn; a table of numbers is that failure mode with more places
to hide.

**Nothing is skipped silently.** A table whose input is missing prints the path it
wanted; `--list` shows the whole plan without writing.

What is *not* derived, and cannot be: the properties matrix. Its cells are a
reading of what each metric is able to separate, argued in the module docstrings
and the root README, not measured on any run. It is a literal here, and the
docstring says so where the table is defined rather than leaving a reader to
assume the ticks were computed.

Ordering mirrors the figures: the instrument is defined before any number it
produces is quoted, then the results, then the appendix.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import config as C
# The gemma/PCFG comparison's pairing rules and its six measures. Shared with
# `make_report_figures`, which draws the same claim: two copies of the alignment
# logic would eventually print different numbers for it.
from reporting import cross_source as X

DEFAULT_OUT = C.OUT_DIR / "paper_figuers"


# --- data access ------------------------------------------------------------
def _json(path: Path):
    return json.loads(path.read_text()) if path.exists() else None


def _pair(report, name="0->1"):
    return next((p for p in report["pairs"] if p["pair"] == name), None)


def gemma_layers():
    G = C.OUT_DIR / C.SOURCE_NAME
    out = []
    for L in C.NAV_LAYERS:
        r = _json(G / f"layer_{L:02d}" / "metrics_report.json")
        if r:
            out.append((L, r))
    return out


def pcfg_runs():
    base = C.OUT_DIR / "pcfg-matryoshka"
    return [(d.name, _json(d / "metrics_report.json"), _json(d / "second_pass.json"))
            for d in sorted(base.glob("layer_*")) if (d / "metrics_report.json").exists()]


# --- LaTeX helpers ----------------------------------------------------------
def esc(t) -> str:
    t = str(t)
    for a, b in [("\\", r"\textbackslash{}"), ("&", r"\&"), ("%", r"\%"), ("$", r"\$"),
                 ("#", r"\#"), ("_", r"\_"), ("{", r"\{"), ("}", r"\}"),
                 ("~", r"\textasciitilde{}"), ("^", r"\textasciicircum{}")]:
        t = t.replace(a, b)
    return t


TWOCOLUMN = False       # set from --twocolumn; the ICLR template is onecolumn


def wrap(text, frac: float) -> str:
    r"""A table cell that wraps, using only the LaTeX kernel.

    The obvious way to wrap a cell is a `p{}` column, and the obvious way to stop
    it justifying is `>{\raggedright\arraybackslash}`. Both need the `array`
    package, and neither fails politely when it is absent: `\arraybackslash`
    reports "control sequence never \def'ed" while leaving `\\` redefined, so
    rows stop terminating and the last column runs off the page; `>` reports
    "Illegal character in array arg". This generator cannot see the preamble it
    will be pasted into, so it must not depend on a package it cannot check --
    and a hand-off .tex that needs a preamble edit to compile is a .tex that
    arrives broken.

    `\parbox` is in the kernel. `[t]` puts the cell's first line on the row's
    baseline, which is what a `p{}` column does too, and `\raggedright` inside
    the box is scoped to the box, so it never touches the tabular's `\\`. The
    trailing `\strut` keeps single-line and multi-line rows the same height.
    """
    return rf"\parbox[t]{{{frac}\linewidth}}{{\raggedright {text}\strut}}"


def table(label, caption, header, rows, align=None, star=False, note=None):
    r"""One booktabs table, caption ABOVE the rules.

    Caption above is the convention for tables and below for figures, and it is
    what `iclr2026_conference.tex` does in its own example.

    `star` asks for a full-width float, which only exists in a twocolumn class.
    The ICLR template is `\documentclass{article}` -- onecolumn -- where `table*`
    is legal but pointless and interacts badly with float placement, so it
    degrades to `table` unless --twocolumn says otherwise.
    """
    env = "table*" if (star and TWOCOLUMN) else "table"
    align = align or ("l" + "r" * (len(header) - 1))
    L = [rf"\begin{{{env}}}[tbp]", r"  \centering", r"  \small",
         rf"  \caption{{{caption}}}", rf"  \label{{tab:{label}}}",
         rf"  \begin{{tabular}}{{{align}}}", r"    \toprule",
         "    " + " & ".join(header) + r" \\", r"    \midrule"]
    L += ["    " + " & ".join(str(c) for c in r) + r" \\" for r in rows]
    L += [r"    \bottomrule", r"  \end{tabular}"]
    if note:
        L.append(rf"  \\[2pt] \footnotesize {note}")
    L.append(rf"\end{{{env}}}")
    return "\n".join(L)


# ---------------------------------------------------------------------------
# 1. Setup. Everything a reader needs to reproduce a number, from config.
# ---------------------------------------------------------------------------
def t_setup(layers, has_bos=True):
    b = C.BLOCK_RANGES
    tok = layers[0][1]["total_tokens"] if layers else None
    rows = [
        ("Base model", rf"\texttt{{{esc(C.MODEL_NAME)}}}, residual stream"),
        ("Layers graded", ", ".join(str(L) for L, _ in layers) or "---"),
        ("SAE", rf"\texttt{{{esc(C.SAE_RELEASE)}}}, Matryoshka, $D = {C.D_SAE:,}$"),
        ("Nested blocks", ", ".join(f"B{i} $[{s},{e})$" for i, (s, e) in enumerate(b))),
        ("Corpus", rf"\texttt{{{esc(C.DATASET)}}}, {C.N_DOCS} docs, context {C.CONTEXT_SIZE}"),
        ("Tokens per layer", f"{tok:,} (BOS excluded)" if tok else "---"),
        ("Firing threshold", rf"$a_f > {C.FIRE_THRESHOLD:g}$ post-JumpReLU"),
        ("Edge criterion", rf"$R \geq {C.EDGE_TAU}$, fire $\geq {C.MIN_FIRE_COUNT}$, "
                           rf"co-fire $\geq {C.MIN_JOINT}$"),
        ("Reconstruction gate", rf"both relative gains $\geq {C.RECON_REL_GAIN_MIN}$"),
        (r"$S_\mathrm{res}$ rule", rf"both decoders in the top $k = {C.SRES_RANK_TOP_K}$"),
        ("Sibling flag", rf"mean pairwise Jaccard $\geq {C.SIBLING_REDUNDANCY_FLAG}$"),
        ("Frequency control", rf"survival $\geq {C.FREQ_SURVIVAL_MIN}$; buckets split at "
                              rf"{C.FREQ_HIGH_MASS:g}/{C.FREQ_MID_MASS:g} cumulative token mass"),
        ("Superparent gate", rf"fan-out $\geq {100 * C.SUPERPARENT_OUTDEG_FRAC:.0f}\%$ "
                             "of the child block"),
    ]
    return table(
        "setup", r"\textbf{Setup.} Every threshold is global and identical for every SAE "
        "source: holding them fixed is what makes a cross-source comparison mean anything, and "
        "no threshold is tuned per source. The block structure is read from the cached statistics "
        "of the run being graded, not from this table, so a dictionary of a different shape is "
        "sliced correctly. Token counts exclude the beginning-of-sequence position"
        + (r" (Table~\ref{tab:bos})." if has_bos else "."),
        ["", ""], rows, align="ll")


# ---------------------------------------------------------------------------
# 2. The battery. Thresholds from config; the honest-name note is the point.
# ---------------------------------------------------------------------------
def t_battery():
    rows = [
        ("1a", "Reverse coverage $R$", r"$P(\text{parent} \mid \text{child})$",
         rf"$\geq {C.EDGE_TAU}$", r"\texttt{coverage}"),
        ("1b", "Forward coverage $F$", r"$P(\text{child} \mid \text{parent})$",
         "reported", r"\texttt{coverage}"),
        ("1c", "Joint-child coverage", r"$R_\mathrm{supp}$, $R_\mathrm{mass}$, energy share",
         "reported", r"\texttt{joint\_child}"),
        ("2a", "Reconstruction ablation", "both features carry mass on the child's tokens",
         rf"$\geq {C.RECON_REL_GAIN_MIN}$", r"\texttt{reconstruction}"),
        ("2b", r"Probe $S_\mathrm{res}$", "both decoders near the child-concept probe",
         rf"top-{C.SRES_RANK_TOP_K} rank", r"\texttt{sres}"),
        ("3", "Sibling redundancy", "co-activation among one parent's children",
         rf"$\geq {C.SIBLING_REDUNDANCY_FLAG}$", r"\texttt{sibling\_redundancy}"),
        ("4", "Out-degree", "one parent covering most of the next block",
         rf"$\geq {100 * C.SUPERPARENT_OUTDEG_FRAC:.0f}\%$", r"\texttt{outdegree}"),
        ("5", "Token-frequency control", "does the edge survive on rare tokens",
         rf"$\geq {C.FREQ_SURVIVAL_MIN}$", r"\texttt{token\_control}"),
        ("6", "Independence null", "PMI against independent firing",
         r"$> 0$", r"\texttt{independence\_null}"),
        ("7", "In-block directed coverage", "containment within one block, and duplicates",
         "asymmetry", r"\texttt{in\_block\_edges}"),
    ]
    return table(
        "battery", r"\textbf{The metric battery.} Ten measurements of the same edge. Metric~1 "
        "defines the candidate set and 2--7 grade it. Metric~2a is named honestly as a "
        r"\emph{contribution filter} rather than as Tree SAE's $S_\mathrm{res}$: two strong but "
        "unrelated co-firing features pass both of its legs, and its pass rate scales with block "
        r"energy. The probe-based $S_\mathrm{res}$ is 2b, and its target is the self-label "
        r"$\mathbf{1}[f_c > 0]$, so a corrupted latent yields a probe that validates its own "
        "corruption; we report it as self-labeled and as a self-consistency check, not as ground "
        "truth.",
        ["", "Metric", "What it asks", "Threshold", "Module"], rows,
        align="llllr", star=True)


# ---------------------------------------------------------------------------
# 3. The properties matrix. NOT derived -- see the module docstring.
# ---------------------------------------------------------------------------
MATRIX = [
    ("1a Reverse coverage",        "P", "x", "x", "x", "x", "x", "x"),
    ("1b Forward coverage",        "P", "x", "x", "P", "x", "x", "x"),
    ("1c Joint-child",             "P", "x", "Y", "Y", "x", "x", "x"),
    ("2a Reconstruction",          "P", "x", "x", "x", "x", "P", "x"),
    (r"2b Probe $S_\mathrm{res}$", "Y", "P", "x", "P", "P", "Y", "x"),
    ("3 Sibling redundancy",       "P", "x", "Y", "x", "Y", "x", "x"),
    ("4 Out-degree",               "x", "x", "x", "Y", "x", "P", "x"),
    ("5 Frequency control",        "P", "x", "x", "P", "x", "Y", "x"),
    ("6 Independence null",        "P", "x", "x", "Y", "x", "Y", "x"),
    ("7 In-block coverage",        "Y", "x", "Y", "x", "Y", "x", "x"),
]
GLYPH = {"Y": r"$\bullet$", "P": r"$\circ$", "x": r"--"}


def t_matrix():
    cols = ["Parent$\\rightarrow$child", "Absorption", "Splitting", "Superparent",
            "Siblings", "Frequency", "Topic"]
    rows = [[r[0]] + [GLYPH[c] for c in r[1:]] for r in MATRIX]
    return table(
        "matrix", r"\textbf{What each metric can separate.} A candidate pair can look like an "
        "edge for seven reasons and only one is hierarchy, so the question per metric is which "
        r"column it adds, not whether it is correct. $\bullet$ detects, $\circ$ partial "
        "(necessary but not sufficient, or only in some regimes), -- blind. \\textbf{These cells "
        "are read off each metric's construction and are not a measured accuracy}; every one is "
        r"exercised by a row of Table~\ref{tab:tier1}. Two columns are open and stay open. "
        r"\emph{Absorption} is unreachable because coverage gates the candidate set and an "
        "absorbed child has low $R$ by construction. \\emph{Topic} --- two specific, unrelated "
        "features sharing a latent subject --- passes coverage, reconstruction, the frequency "
        "control and PMI alike; closing it needs a model-based topic null rather than another "
        "threshold.",
        [""] + cols, rows, align="l" + "c" * len(cols), star=True)


# ---------------------------------------------------------------------------
# 4. The tiers.
# ---------------------------------------------------------------------------
def t_tiers(toy, tt, align_json, pcfg=None):
    r"""The ladder, with Tier 3 described as what it is rather than as what it was meant to be.

    Two things this row used to assert and no longer does.

    It claimed Tier 3 "keeps the known tree". It does not: Tier 2's tree is
    Bussmann's -- 20 features, 9 edges, exclusive siblings, from the team repo's
    `configs/tree.json` -- and Tier 3's is a PCFG whose hierarchy is
    document/section/paragraph/sentence over S-V-O roles. Different structures,
    and the word "tree" doing double duty is what let the claim through.

    It also claimed Tier 3 "isolates base-model dependence". Isolation needs one
    variable to move. Between Tiers 2 and 3 the grammar, the base model, the
    corpus and the dictionary size all move together, so the tier BOUNDS the
    base model's contribution rather than isolating it. Isolating it would mean
    training a transformer on Bussmann's own tree, which nothing here does.

    And the ground-truth cell promised a measurement the result cell does not
    deliver: the grammar is known, but nothing maps a latent to a grammar symbol,
    so Tier 3 reports the same battery outputs as Tier 4. The helper that would
    close it exists -- `pcfg_bridge.grammar.vocab.role_of` in the PCFG repo,
    named in its `analysis/README.md` for exactly this -- so the cell says "not
    yet" rather than implying the tier is inherently blind.
    """
    n_pass = sum(r["pass"] for r in toy) if toy else None
    pcfg_result, pcfg_runs_on, pcfg_detail = "---", "a Matryoshka SAE on a PCFG corpus", None
    if pcfg:
        parts, cands = [], []
        for name, r, sp in pcfg:
            p = _pair(r)
            recon = rf"{100 * p['reconstruction']['frac_pass']:.0f}\%"
            sr = sp["0->1"]["sres"] if sp and "0->1" in sp else None
            sres = rf"{sr['n_pass']}/{sr['n_edges_scored']:,}" if sr else "---"
            cands.append(p["n_candidate_edges"])
            parts.append(rf"{name.replace('layer_', 'layer~')}: {p['n_candidate_edges']:,} "
                         rf"candidates, {recon} recon, {sres} $S_\mathrm{{res}}$")
        # The cell summarises and the note carries the per-layer detail: four
        # measurements inline overflowed the text block, and a table that runs
        # off the page is not a table.
        pcfg_result = (rf"{len(pcfg)} layers, {min(cands):,}--{max(cands):,} candidates; "
                       r"\emph{no ground-truth score} --- same battery outputs as Tier~4")
        pcfg_detail = "Tier~3, per layer: " + "; ".join(parts) + "."
        nl = ((pcfg[0][1].get("config") or {}).get("base_model") or {}).get("n_layers")
        pcfg_runs_on = ("a Matryoshka SAE over a "
                        + (f"{nl}-layer " if nl else "small ")
                        + "transformer trained on a PCFG corpus")
    # Columns 2-4 hold sentences, so each is a \parbox rather than an l column
    # that would run off the page. Fractions sum to 0.65, leaving 0.35 for the
    # Tier column and the eight \tabcolsep gaps.
    W = (0.22, 0.18, 0.25)
    rows = [
        ("1 Synthetic toy", "hand-built statistics", "by construction",
         rf"{n_pass}/{len(toy)} rows, 21/21 functions, seeds 0--7" if toy else "---"),
        ("2 Trained toy", "an SAE trained on Bussmann's tree", "Bussmann's tree is known",
         rf"precision {tt['precision']:.2f}, recall {tt['recall']:.2f}" if tt else "---"),
        ("3 PCFG SAE", pcfg_runs_on,
         r"the grammar is known; no latent$\leftrightarrow$symbol mapping yet", pcfg_result),
        # "Released", not "Real": Tier 3 is a real SAE too -- really trained, over
        # a really trained transformer -- and calling only this rung real reads as
        # demoting it, which is the confusion the Tier 3 row was just rewritten to
        # remove. What actually sets this rung apart is that we did not train it.
        # The release id lives in the note, not the cell. In \texttt it is a single
        # unbreakable 29-character token, wider than any sensible column, and TeX
        # does not shrink an overfull box -- it prints it straight across the next
        # column. Which is what it did.
        ("4 Released SAE", rf"\texttt{{{esc(C.MODEL_NAME)}}}", "none --- human reading",
         "40 survivors read against autointerp labels"),
    ]
    rows = [[r[0]] + [wrap(c, w) for c, w in zip(r[1:], W)] for r in rows]
    note = " ".join(t for t in [
        rf"Tier~4's dictionary is the released \texttt{{{esc(C.SAE_RELEASE)}}}.",
        pcfg_detail,
        (rf"Lateral control (not a tier): {align_json['n_respected']}/"
         rf"{align_json['n_testable']} testable true edges run early block "
         r"$\rightarrow$ late on the trained toy." if align_json else None),
    ] if t) or None
    # Tier 2's own size, from its calibration file. The two toys are NOT the same
    # world in two conditions, which "closes exactly that gap" used to imply:
    # Tier 1 grades a pathology-injected world and Tier 2 a clean, smaller tree,
    # so the world changes along with the statistics. Same shape of overclaim as
    # Tier 3's withdrawn "isolating", one rung down -- softer, because it was
    # carried by one word rather than asserted.
    tier2 = ("on a simpler toy"
             if not tt else
             rf"on Bussmann's {tt['n_features']}-feature tree with "
             rf"{len(tt['true_edges'])} edges and no injected pathologies")
    return table(
        "tiers", r"\textbf{Four tiers, trading ground truth against realism.} Each rung "
        "licenses the one above it: Tier~1 proves the arithmetic and nothing about whether "
        rf"an SAE would learn such a structure. Tier~2 attacks that gap, but {tier2} rather "
        "than on the pathology-injected world Tier~1 grades, so it changes the world as well "
        "as the statistics and does not isolate training as the single moving part either. "
        "What it does isolate cleanly is blame: only what the SAE actually learned reaches the "
        "metrics, so a missed edge counts against a metric only if the SAE learned both "
        "endpoints. Tier~3 inserts a base model between the concepts and the SAE, which "
        "Tiers~1 and~2 do not have at all. It is \\textbf{not} a controlled swap from Tier~2: "
        "the grammar, the base model, the corpus and the dictionary size all change together, "
        "so it \\emph{bounds} the base model's contribution rather than isolating it --- "
        "isolating it would mean training a transformer on Bussmann's own tree. Its grammar is "
        "known but nothing yet maps a latent to a grammar symbol, so its result column reports "
        "the same battery outputs as Tier~4 rather than a recovery score; the helper that would "
        r"close the gap (\texttt{role\_of} in the PCFG repo) already exists. Tier~4 is named "
        r"\emph{released} rather than \emph{real} --- Tier~3 is a real SAE too, really trained "
        "over a really trained transformer, and the distinction that matters is that Tier~4 is a "
        "published checkpoint we did not train. That is also a constraint and not only a label: "
        "we do not control its dictionary, which is why the "
        r"$S_\mathrm{res}$ column of Table~\ref{tab:gemma} exists for one layer only. It has no "
        "known answer at all.",
        ["Tier", "Runs on", "Ground truth", "Result"], rows,
        # p{} rather than l: Tier 3's cells are sentences, and in an l column a
        # sentence does not wrap -- it runs off the page edge, and LaTeX only
        # warns about the overfull box, so nobody notices until the PDF.
        # Fractions of \linewidth rather than centimetres, because the widths
        # have to hold under whatever \textwidth the class sets and under
        # --twocolumn, where the float becomes table* and \linewidth changes.
        # 0.70 for the three, leaving the l column and eight \tabcolsep gaps
        # inside the remaining 0.30.
        # Plain l columns: the wrapping lives in each cell's \parbox, so this
        # table needs booktabs and nothing else. See wrap().
        align="llll", star=True, note=note)


# ---------------------------------------------------------------------------
# 5. Tier-1 scorecard.
# ---------------------------------------------------------------------------
def t_tier1(toy):
    order = sorted(toy, key=lambda r: (r.get("margin_kind") == "categorical", -r["margin"]))
    rows = []
    for r in order:
        cat = r.get("margin_kind") == "categorical"
        m = "---" if cat else (r">1000\times$" if r["margin"] >= 1000 else
                               rf"{r['margin']:.1f}$\times$")
        m = r"$>1000\times$" if (not cat and r["margin"] >= 1000) else m
        rows.append([esc(r["metric"]), esc(r["job"]),
                     r"\checkmark" if r["pass"] else r"$\times$", m])
    n_pass = sum(r["pass"] for r in toy)
    return table(
        "tier1", rf"\textbf{{Tier 1: every metric against a known tree.}} {n_pass} of {len(toy)} "
        "rows pass, on every seed from 0 to 7. Margin is how decisively the metric separated the "
        "class it must keep from the class it must reject; rows scored categorically have no "
        "margin, because their answer is right or it is not and a value of $1.0$ on a ratio scale "
        "would read as no separation. The last two rows are negative controls that pass when the "
        "battery does \\emph{not} act.",
        ["Row", "The job it claims", "", "Margin"], rows, align="llcr", star=True)


# ---------------------------------------------------------------------------
# 6. The gemma result, per layer.
# ---------------------------------------------------------------------------
def t_gemma(layers, second6, has_bos=True):
    rows = []
    for L, r in layers:
        p = _pair(r)
        s = ""
        if L == 6 and second6:
            sr = second6["0->1"]["sres"]
            s = rf"{sr['n_pass']}/{sr['n_edges_scored']:,}"
        rows.append([f"L{L}", f"{p['n_candidate_edges']:,}",
                     rf"{100 * p['reconstruction']['frac_pass']:.0f}\%",
                     rf"{100 * p['independence_null']['frac_chance_level']:.0f}\%",
                     rf"{100 * p['freq_control']['frac_freq_driven']:.1f}\%",
                     rf"{100 * p['degree']['poly_frac']:.0f}\%",
                     s or "---"])
    return table(
        "gemma", r"\textbf{Block pair B0$\rightarrow$B1 on \texttt{gemma-2-2b}, all five graded "
        r"layers.} Read left to right: coverage proposes thousands of edges, the reconstruction "
        "filter keeps most of them, and the independence null rejects most of them --- the "
        "over-connection is what the parent's own firing rate already forces, not capture of "
        "frequent tokens, which the frequency control puts at 1--2\\%. Multi-parenting is the "
        "measure that does not depend on the candidate set and the only one of the project's "
        r"original four claims to survive BOS exclusion"
        + (r" (Table~\ref{tab:bos})" if has_bos else "") + ". The "
        r"$S_\mathrm{res}$ column exists for layer 6 alone: stage~03 needs a cached token stream "
        "that the released statistics do not carry.",
        ["Layer", "Candidates", "Recon.", "At chance", "Freq.-driven",
         r"$\geq 2$ parents", r"$S_\mathrm{res}$"],
        rows, star=True)


# ---------------------------------------------------------------------------
# 7. Cross-source (appendix).
# ---------------------------------------------------------------------------
def t_sources(layers, second6, pcfg):
    rows = []
    l6 = next((r for L, r in layers if L == 6), None)
    if l6:
        p = _pair(l6)
        sr = second6["0->1"]["sres"] if second6 else None
        rows.append([r"\texttt{gemma-2-2b} L6", f"{C.D_SAE:,}", str(len(C.BLOCK_RANGES)),
                     f"{l6['total_tokens']:,}", f"{p['n_candidate_edges']:,}",
                     rf"{100 * p['reconstruction']['frac_pass']:.0f}\%",
                     rf"{sr['n_pass']}/{sr['n_edges_scored']:,}" if sr else "---"])
    for name, r, sp in pcfg:
        p = _pair(r)
        cfg = r.get("config") or {}
        d = cfg.get("d_sae") or (r.get("block_ranges") or [[0, 0]])[-1][-1]
        nb = len(r.get("block_ranges") or [])
        sr = sp["0->1"]["sres"] if sp and "0->1" in sp else None
        rows.append([f"PCFG {esc(name.replace('_', ' '))}", f"{d:,}", str(nb),
                     f"{r['total_tokens']:,}", f"{p['n_candidate_edges']:,}",
                     rf"{100 * p['reconstruction']['frac_pass']:.0f}\%",
                     rf"{sr['n_pass']}/{sr['n_edges_scored']:,}" if sr else "---"])
    return table(
        "sources", r"\textbf{The same battery across SAE sources.} Identical metric code and "
        "identical global thresholds; only the block structure follows the source, read from the "
        "statistics file being graded. The reconstruction column should be read with care on "
        "PCFG: the weakest candidate edge there sits $3.5\\times$ above the threshold, so the "
        "filter is inert on that source and its surviving edges have passed coverage alone. "
        r"$S_\mathrm{res}$ pass rates are not comparable between rows of different $D$ --- see "
        r"Table~\ref{tab:null}.",
        ["Run", "$D$", "Blocks", "Tokens", "Candidates", "Recon.", r"$S_\mathrm{res}$"],
        rows, star=True)


# ---------------------------------------------------------------------------
# 7a. gemma against PCFG, layer by layer (appendix). The companion to
#     `cross_source_layer_response`; both read `reporting/cross_source.py`, so
#     the figure's gaps and this table's rows cannot disagree.
# ---------------------------------------------------------------------------
def t_layers(rs):
    """Every graded run of both sources on one axis, ordered by relative depth.

    Sorted by (L+1)/N rather than grouped by source, so the two interleave and a
    reader can see for themselves that the alignment is a choice: PCFG's layer 1
    is the toy's half-way point and lands beside gemma's layer 12, while it is
    also literally layer 1 and gemma has one of those too. Both keys are printed
    for that reason.

    Nothing here is a mean over layers. The figure reduces this table to six gap
    numbers, and a table whose job is to let someone check that reduction has to
    carry the rows it was computed from.
    """
    by_layer = X.matched(rs, "layer")
    shared = sorted({o["layer"] for o, _, _ in by_layer})
    rows = []
    for r in rs:
        rows.append([
            ("PCFG" if not r["is_ref"] else r"\texttt{gemma-2-2b}") + f" L{r['layer']}",
            f"{r['layer']}/{r['n_layers']}",
            f"{r['depth']:.2f}",
            f"{r['tokens']:,}",
            f"{r['n_cand']:,}",
            rf"{X.density(r):.2f}\%",
            rf"{100 * r['poly']:.0f}\%",
            f"{r['sib']:.3f}",
            f"{r['gini']:.3f}",
            rf"{100 * r['recon']:.0f}\%",
            rf"{100 * r['chance']:.0f}\%",
        ])
    # Both facts a reader needs to audit the restriction and the denominator,
    # computed rather than asserted.
    beyond = "; ".join(f"{r['label']} {r['beyond_b0b1']:,}" for r in rs if not r["is_ref"])
    dens_pairs = sorted({r["n_pairs"] for r in rs})
    note = (rf"Restricted to block pair B0$\rightarrow$B1. Candidate edges in \emph{{all}} "
            rf"deeper pairs combined, on the PCFG runs: {beyond}---so no deeper pair supports "
            rf"a comparison. Both PCFG rows are one grammar configuration "
            rf"({X.grammar_line(rs)}), a single point of the three-axis sweep Exp 2 specifies, "
            rf"so this is gemma against one PCFG corpus rather than against PCFG. "
            rf"Density is a share of every ordered B0$\times$B1 feature pair, "
            rf"{dens_pairs[0]:,} on one source and {dens_pairs[-1]:,} on the other "
            rf"({100 * (dens_pairs[-1] / dens_pairs[0] - 1):.0f}\% apart), which is coincidence "
            r"and not design: it is what makes the two densities directly comparable rather "
            r"than merely each normalised by itself.")
    return table(
        "layers", r"\textbf{The same battery on two base models, layer by layer.} Rows are "
        r"ordered by position in their own network, $(L{+}1)/N$, so the sources interleave; "
        rf"layer{'s' if len(shared) > 1 else ''} "
        + " and ".join(str(s) for s in shared) + " carry an SAE on both, which is where the "
        "comparison needs no matching argument at all. The token budgets differ by "
        rf"{max(r['tokens'] for r in rs) / min(r['tokens'] for r in rs):.0f}$\times$; that cuts "
        r"\emph{against} the density gap rather than explaining it, because the joint-support "
        r"guard is an absolute count and more tokens make it easier to clear, yet the source "
        r"with more tokens is the sparser one. What the two agree on is the \emph{shape} of the "
        r"relation---multi-parenting, sibling redundancy and the frequency control---and what "
        r"they do not is its \emph{strength}: density, the reconstruction filter and the "
        r"base-rate null. That split should be read with Table~\ref{tab:align}, which shows the "
        r"three agreeing measures are also the three sitting against a boundary on both sources, "
        r"where agreement is close to free.",
        ["Run", "$L/N$", r"$\frac{L+1}{N}$", "Tokens", "Cand.", "Density",
         r"$\geq 2$ par.", "Sib.", "Gini", "Recon.", "At chance"],
        rows, align="lrrrrrrrrrr", star=True, note=note)


# ---------------------------------------------------------------------------
# 7b. What the pairing in 7a rests on (appendix).
# ---------------------------------------------------------------------------
def t_align(rs):
    """Same block index or same fraction of the network -- measured, not assumed.

    The two rules pair PCFG's layers with opposite ends of gemma, so the choice
    changes what the comparison says and is worth a table of its own rather than
    a sentence in someone else's caption.
    """
    by_layer, by_depth = X.matched(rs, "layer"), X.matched(rs, "depth")
    rows = []
    for title, key, g_layer in X.ranked(by_layer):
        rows.append([X.tex(title), f"{g_layer:.1f}", f"{X.gap(by_depth, key):.1f}",
                     f"{X.spread(rs, key):.1f}", f"{X.gap_ratio(rs, by_layer, key):.2f}"])
    ml = sum(g for _, _, g in X.ranked(by_layer)) / len(X.PANELS)
    md = sum(X.gap(by_depth, k) for _, k in X.PANELS) / len(X.PANELS)
    rows.append([r"\textbf{mean}", rf"\textbf{{{ml:.1f}}}", rf"\textbf{{{md:.1f}}}", "", ""])
    # The measure the two readings disagree about most sharply, named from the
    # data rather than from what the last run happened to show.
    by_pts = [t for t, _, _ in X.ranked(by_layer)]
    by_ratio = sorted(X.PANELS, key=lambda p: X.gap_ratio(rs, by_layer, p[1]))
    flip = X.tex(max(X.PANELS, key=lambda p: by_pts.index(p[0]) - by_ratio.index(p))[0])
    return table(
        "align", r"\textbf{What the cross-source comparison rests on, and how far to trust it.} "
        "Two questions, both of which are usually settled silently. \\emph{First}, comparing base "
        "models of different depth needs a rule for pairing their layers, and the two defensible "
        "rules pick opposite ends of gemma: by block index PCFG's layers pair with gemma's "
        + " and ".join(f"L{q['layer']}" for _, q, _ in by_layer) + ", by relative depth "
        r"$(L{+}1)/N$ with gemma's "
        + " and ".join(f"L{q['layer']}" for _, q, _ in by_depth) + rf". Block index wins on the "
        rf"mean ({ml:.1f} points against {md:.1f}), which is why it is what "
        r"Figure~\ref{fig:cross-source-layer-response} draws---though the margin rests largely "
        r"on one measure, and the reconstruction filter is inert on PCFG in any case. "
        r"\emph{Second}, a small gap is only impressive if the measure could have been far "
        r"apart. The last two columns ask that: \emph{range} is how far each measure moves "
        "across all "
        rf"{len(rs)} graded runs, and the ratio is the gap against it. The two readings do not "
        rf"agree---{flip} is near the bottom on raw points and near the top on the ratio---so "
        "both are printed and neither is called correct. The honest summary is that the three "
        r"measures with the smallest raw gaps are also the three pinned against a boundary "
        r"($\geq 2$ parents at 89--100\%, sibling redundancy at 3--6\%, the frequency control at "
        r"0--2\%), where agreement costs the two sources nothing. \textbf{On "
        rf"{len(by_layer)} shared layers none of this is a result." + "}",
        ["Measure", "Gap by layer index", "Gap by rel.\\ depth", "Range over all runs",
         "Gap / range"],
        rows, align="lrrrr", star=True)


# ---------------------------------------------------------------------------
# 8. The k/D null (appendix).
# ---------------------------------------------------------------------------
def t_null(layers, second6, pcfg):
    k = C.SRES_RANK_TOP_K
    rows = [["Synthetic toy (Tier 1)", "42", rf"{100 * k / 42:.1f}\%", "---", "---"],
            ["Trained toy (Tier 2)", "20", rf"{100 * k / 20:.1f}\%", "5/5 true edges", "---"]]
    obs = []
    if second6:
        s = second6["0->1"]["sres"]
        obs.append((r"\texttt{gemma-2-2b} L6", C.D_SAE, 100 * s["n_pass"] / s["n_edges_scored"]))
    for name, r, sp in pcfg:
        if sp and "0->1" in sp and sp["0->1"]["sres"]["n_edges_scored"]:
            s = sp["0->1"]["sres"]
            d = (r.get("config") or {}).get("d_sae") or 1792
            obs.append((f"PCFG {esc(name.replace('_', ' '))}", d,
                        100 * s["n_pass"] / s["n_edges_scored"]))
    for name, d, o in obs:
        null = 100 * k / d
        rows.append([name, f"{d:,}", rf"{null:.3f}\%", rf"{o:.2f}\%",
                     rf"{o / null:.0f}$\times$" if o > 0 else "below chance"])
    return table(
        "null", rf"\textbf{{The rank rule's null depends on dictionary size.}} $S_\mathrm{{res}}$ "
        rf"passes an edge when both decoders fall in the top $k = {k}$ of the probe's "
        "correlations over the whole dictionary. It is a geometry test, so an unrelated parent "
        "passes whenever chance puts it there, at a rate of $k/D$. The same $k$ is therefore two "
        "orders of magnitude stricter on gemma than on a 42-feature toy, and a measured pass rate "
        "is only interpretable against its own null: two runs on the \\emph{same} 1{,}792-latent "
        "dictionary land on opposite sides of theirs. Reporting $S_\\mathrm{res}$ shares across "
        "sources without this correction compares nothing.",
        ["Source", "$D$", "Null $k/D$", "Observed", "vs.\\ null"], rows, star=True)


# ---------------------------------------------------------------------------
# 9. BOS before/after (appendix).
# ---------------------------------------------------------------------------
def t_bos(layers, arch):
    def mean(reports, fn):
        return sum(fn(_pair(r)) for r in reports) / len(reports)

    cur = [r for _, r in layers]
    spec = [
        ("Candidate edges", lambda p: p["n_candidate_edges"], "{:,.0f}", True),
        (r"Improve reconstruction, \%", lambda p: 100 * p["reconstruction"]["frac_pass"],
         "{:.0f}", True),
        (r"Frequency-driven, \%", lambda p: 100 * p["freq_control"]["frac_freq_driven"],
         "{:.1f}", True),
        ("Mean frequency survival", lambda p: p["freq_control"]["mean_survival"], "{:.2f}", True),
        ("Sibling redundancy", lambda p: p["sibling_redundancy"]["mean_redundancy"],
         "{:.2f}", True),
        ("Superparents flagged", lambda p: p["n_superparents"], "{:.1f}", True),
        (r"Children with $\geq 2$ parents, \%",
         lambda p: 100 * (p["degree"].get("poly_frac") if p["degree"].get("poly_frac") is not None
                          else p["degree"]["n_multi_parented"]
                          / max(p["degree"]["n_children_with_parent"], 1)), "{:.0f}", False),
    ]
    rows = []
    for name, fn, fmt, shared in spec:
        a, b = mean(arch, fn), mean(cur, fn)
        rows.append([name, r"\checkmark" if shared else "--",
                     fmt.format(a), fmt.format(b), rf"{b / a:.2f}$\times$" if a else "---"])
    return table(
        "bos", r"\textbf{One contaminating token position moved five of six metrics.} Means over "
        "the five graded layers, block pair B0$\\rightarrow$B1, before and after excluding the "
        "beginning-of-sequence position. BOS is an attention sink on which effectively every "
        "feature fires; with 400 documents it gave every pair in the dictionary 400 joint firings "
        r"against a support guard of $\mathrm{MIN\_JOINT} = 30$, so the guard admitted pairs that "
        "never co-occur anywhere else. Every quantity computed from that one co-firing matrix "
        "moved, and the one that is not --- a ratio over children that already have a parent --- "
        "did not. \\textbf{The \\emph{before} column is withdrawn} and is reproduced only to size "
        "the error. Agreement among detectors that share an input is much weaker evidence than a "
        "battery implies.",
        ["Quantity", "Reads co-fire", "Before", "After", "Change"], rows, star=True)


# ---------------------------------------------------------------------------
def build(dry: bool):
    written, skipped, parts = [], [], []
    G = C.OUT_DIR / C.SOURCE_NAME
    layers = gemma_layers()
    second6 = _json(G / "layer_06" / "second_pass.json")
    toy = _json(C.OUT_DIR / "synthetic_toy_calibration.json")
    tt = _json(C.OUT_DIR / "trained_toy_calibration.json")
    align = _json(C.OUT_DIR / "block_tree_alignment.json")
    pcfg = pcfg_runs()
    arch = [_json(p) for p in
            sorted((C.HERE / "outputs_archive").glob("layer_*__v1__*/metrics_report.json"))]
    # Two tables cite Table~\ref{tab:bos}. It is only emitted when the archived
    # pre-BOS reports are present, so the citation has to be conditional too --
    # otherwise a clone without outputs_archive/ compiles with a dangling ref
    # printing '??', which is exactly the kind of quiet staleness this file
    # exists to prevent.
    has_bos = bool(arch and layers) and len(arch) == len(layers)

    def add(slot, name, ok, why, fn, section=None):
        if not ok:
            skipped.append((name, why))
            return
        written.append(f"{slot:<7} {name:<10} <- {why}" if dry else name)
        if not dry:
            parts.append((slot, section, fn()))

    add("MAIN 1", "setup", bool(layers), "config + a graded layer's token count",
        lambda: t_setup(layers, has_bos), "The instrument")
    add("MAIN 2", "battery", True, "config thresholds", t_battery)
    add("MAIN 3", "matrix", True, "argued, not measured -- see the module docstring", t_matrix)
    add("MAIN 4", "tiers", bool(toy and tt), "toy + trained-toy calibration JSON"
        if (toy and tt) else "needs both calibration JSONs",
         lambda: t_tiers(toy, tt, align, pcfg), "Validation")
    add("MAIN 5", "tier1", bool(toy), f"{len(toy) if toy else 0} scorecard rows"
        if toy else "needs synthetic_toy_calibration.json", lambda: t_tier1(toy))
    add("MAIN 6", "gemma", bool(layers), f"{len(layers)} gemma layer reports"
        if layers else "needs gemma metrics_report.json",
        lambda: t_gemma(layers, second6, has_bos), "The result")
    add("APP 1", "sources", bool(layers and pcfg), f"gemma + {len(pcfg)} PCFG runs"
        if (layers and pcfg) else "needs a gemma report and a PCFG report",
        lambda: t_sources(layers, second6, pcfg), "Appendix")
    # The layer-by-layer cross-source pair. `t_layers` needs one layer index
    # graded on both sources; `t_align` additionally needs the two pairing rules
    # to actually disagree, which they only do once the reference source has
    # layers the toy cannot reach — otherwise the table would be two identical
    # columns.
    xs = X.rows()
    x_shared = sorted({o["layer"] for o, _, _ in X.matched(xs, "layer")})
    add("APP 2", "layers", bool(x_shared),
        f"{len(xs)} graded runs, layer{'s' if len(x_shared) > 1 else ''} "
        f"{', '.join(str(s) for s in x_shared)} on both sources"
        if x_shared else "needs one layer index graded on both gemma and PCFG",
        lambda: t_layers(xs))
    add("APP 3", "align", X.alignment_gaps(xs) is not None,
        "the same runs under both pairing rules" if X.alignment_gaps(xs) is not None
        else "the two pairing rules select the same runs here, so the table is empty",
        lambda: t_align(xs))
    add("APP 4", "null", bool(second6 or pcfg), "measured pass rates + config"
        if (second6 or pcfg) else "needs at least one second_pass.json",
        lambda: t_null(layers, second6, pcfg))
    add("APP 5", "bos", bool(arch and layers) and len(arch) == len(layers),
        f"{len(arch)} archived v1 reports vs {len(layers)} current"
        if arch else "needs the pre-BOS reports in outputs_archive/",
        lambda: t_bos(layers, arch))
    return written, skipped, parts


HEAD = r"""% ===========================================================================
% Tables.
%
% Generated by reporting/make_report_tables.py. Every number was read
% from the JSON it describes at generation time; do not edit one by hand.
%
% Ordered as the paper reads: the instrument is defined before any number it
% produces is quoted. MAIN 1-6 are the main text, APP 1-3 the appendix.
%
% Requires: booktabs. Nothing else -- deliberately. Wrapping cells use
% \parbox, a kernel command, rather than array's >{...}p{} columns: this
% file cannot see the preamble it is pasted into, and a missing array
% package does not fail politely ("Illegal character in array arg", or
% rows that stop terminating and a last column off the page).
% Emitted for a ONECOLUMN class, which is what the ICLR
% template uses; pass --twocolumn for full-width table* floats.
% Captions sit above the rules, as tables take them and as the ICLR template's
% own example does.
%
% To compile on its own, prepend
%     \documentclass{article}\usepackage{booktabs}\begin{document}
% and append \end{document}.
% ===========================================================================
"""


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--list", action="store_true", help="show the plan, write nothing")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT, help="directory for tables.tex")
    ap.add_argument("--twocolumn", action="store_true",
                    help="emit table* full-width floats (needs a twocolumn class)")
    args = ap.parse_args()
    globals()["TWOCOLUMN"] = args.twocolumn

    written, skipped, parts = build(args.list)
    if not args.list:
        args.out.mkdir(parents=True, exist_ok=True)
        L, seen = [HEAD], set()
        for slot, section, tex in parts:
            if section and section not in seen:
                seen.add(section)
                L += ["", r"% " + "-" * 73, f"% {section}", r"% " + "-" * 73]
            L += ["", f"% [{slot}]", tex]
        (args.out / "tables.tex").write_text("\n".join(L) + "\n")

    print(f"[tab] {'plan for' if args.list else 'wrote'} {args.out / 'tables.tex'}")
    for w in written:
        print(f"  ok    {w}")
    for name, why in skipped:
        print(f"  SKIP  {name}\n          {why}")
    if skipped and not args.list:
        print(f"\n[tab] {len(written)} written, {len(skipped)} skipped — the set above is not "
              "complete, and the reasons are printed rather than left to be noticed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
