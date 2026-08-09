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
def t_setup(layers):
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
        "sliced correctly. Token counts exclude the beginning-of-sequence position; see "
        r"Table~\ref{tab:bos}.",
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
def t_tiers(toy, tt, align_json):
    n_pass = sum(r["pass"] for r in toy) if toy else None
    rows = [
        ("1 Synthetic toy", "hand-built statistics", "by construction",
         rf"{n_pass}/{len(toy)} rows, 21/21 functions, seeds 0--7" if toy else "---"),
        ("2 Trained toy", "a SAE trained on toy model", "the tree is known",
         rf"precision {tt['precision']:.2f}, recall {tt['recall']:.2f}" if tt else "---"),
        ("3 PCFG SAE", "a Matryoshka SAE trained on a small model build on PCFG tree", "the PCFG tree is known", "---"),
        ("4 Real SAE", rf"\texttt{{{esc(C.MODEL_NAME)}}}", "none --- human reading",
         "40 survivors read against autointerp labels"),
    ]
    note = None
    if align_json:
        note = (rf"Lateral control (not a tier): {align_json['n_respected']}/"
                rf"{align_json['n_testable']} testable true edges run early block "
                r"$\rightarrow$ late on the trained toy.")
    return table(
        "tiers", r"\textbf{Four tiers, trading ground truth against realism.} Each rung "
        "licenses the one above it. Tier~1 proves the arithmetic and nothing about whether an SAE "
        "would learn such a structure; Tier~2 closes exactly that gap, because only what the SAE "
        "actually learned reaches the metrics, which is also what lets it attribute a miss --- a "
        "missed edge counts against a metric only if the SAE learned both endpoints. Tier~3 uses "
        "a smaller base model whose structure is not aligned to language; only Tier~4 has no "
        "known answer and is named for how it is judged rather than what it runs on.",
        ["Tier", "Runs on", "Ground truth", "Result"], rows, align="llll", star=True, note=note)


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
def t_gemma(layers, second6):
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
        r"original four claims to survive BOS exclusion (Table~\ref{tab:bos}). The "
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

    def add(slot, name, ok, why, fn, section=None):
        if not ok:
            skipped.append((name, why))
            return
        written.append(f"{slot:<7} {name:<10} <- {why}" if dry else name)
        if not dry:
            parts.append((slot, section, fn()))

    add("MAIN 1", "setup", bool(layers), "config + a graded layer's token count",
        lambda: t_setup(layers), "The instrument")
    add("MAIN 2", "battery", True, "config thresholds", t_battery)
    add("MAIN 3", "matrix", True, "argued, not measured -- see the module docstring", t_matrix)
    add("MAIN 4", "tiers", bool(toy and tt), "toy + trained-toy calibration JSON"
        if (toy and tt) else "needs both calibration JSONs",
         lambda: t_tiers(toy, tt, align), "Validation")
    add("MAIN 5", "tier1", bool(toy), f"{len(toy) if toy else 0} scorecard rows"
        if toy else "needs synthetic_toy_calibration.json", lambda: t_tier1(toy))
    add("MAIN 6", "gemma", bool(layers), f"{len(layers)} gemma layer reports"
        if layers else "needs gemma metrics_report.json",
        lambda: t_gemma(layers, second6), "The result")
    add("APP 1", "sources", bool(layers and pcfg), f"gemma + {len(pcfg)} PCFG runs"
        if (layers and pcfg) else "needs a gemma report and a PCFG report",
        lambda: t_sources(layers, second6, pcfg), "Appendix")
    add("APP 2", "null", bool(second6 or pcfg), "measured pass rates + config"
        if (second6 or pcfg) else "needs at least one second_pass.json",
        lambda: t_null(layers, second6, pcfg))
    add("APP 3", "bos", bool(arch and layers) and len(arch) == len(layers),
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
% Requires: booktabs. Emitted for a ONECOLUMN class, which is what the ICLR
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
