<style>
.x0nav{position:sticky;top:0;z-index:999;background:#fff;border-bottom:1px solid #E3DAFB;
font:500 13px/1.15 system-ui,-apple-system,"Segoe UI",sans-serif;margin:0 0 14px;}
.x0nav .row{display:flex;flex-wrap:wrap;align-items:center;gap:13px;padding:9px 18px;}
.x0nav .row+.row{border-top:1px solid #F1ECFD;}
.x0nav a{text-decoration:none;color:#5A6B7B;}
.x0nav a:hover{color:#7C22CE;}
.x0nav .brand{font-weight:700;color:#7C22CE;letter-spacing:.2px;}
.x0nav .on{color:#7C22CE;font-weight:700;}
.x0nav .lbl{color:#9AA7B3;font-size:11px;text-transform:uppercase;letter-spacing:.7px;}
.x0nav .pill{border:1px solid #E3DAFB;border-radius:7px;padding:5px 10px;background:#F6F3FE;}
.x0nav .pill.on{background:#7C22CE;color:#fff;border-color:#7C22CE;}
.x0nav .sep{width:1px;height:17px;background:#E3DAFB;}
@media (prefers-color-scheme:dark){
.x0nav{background:#141414;border-bottom-color:#2E2E2E;}
.x0nav .row+.row{border-top-color:#242424;}
.x0nav a{color:#A9B4BF;}
.x0nav .brand,.x0nav a:hover,.x0nav .on{color:#C79BF2;}
.x0nav .pill{background:#1E1830;border-color:#3A2B57;}
.x0nav .pill.on{background:#7C22CE;color:#fff;border-color:#7C22CE;}
.x0nav .sep{background:#2E2E2E;}}
</style><nav class="x0nav"><div class="row"><a class="brand" href="../../">SOAR I-6 · metrics</a><a class="" href="../../">Overview</a><a class="" href="../../outputs/">Results</a><a class="" href="../../outputs/#per-layer">Per layer</a><a class="" href="../../outputs/cross_depth_comparison.html">Cross-depth</a><a class="" href="../../outputs/kill_rates.html">Kill rates</a><a class="" href="../../outputs/toy_calibration.html">Toy calibration</a><a class="" href="../../outputs/trained_toy_calibration.html">Trained toy</a></div><div class="row"><span class="lbl">Layer</span><a class="pill" href="../../outputs/layer_03/metrics_report.html">3</a><a class="pill" href="../../outputs/layer_06/metrics_report.html">6</a><a class="pill" href="../../outputs/layer_12/metrics_report.html">12</a><a class="pill" href="../../outputs/layer_18/metrics_report.html">18</a><a class="pill on" href="../../outputs/layer_24/metrics_report.html">24</a><span class="sep"></span><span class="lbl">Page</span><a class="" href="../../outputs/layer_24/metrics_dashboard.html">Dashboard</a><a class="" href="../../outputs/layer_24/superparent_sankey.html">Superparents</a><a class="" href="../../outputs/layer_24/qualitative_dashboard.html">Qualitative</a><a class="on" href="../../outputs/layer_24/metrics_report.html">Metrics report</a><a class="" href="../../outputs/layer_24/qualitative_check.html">Qualitative report</a></div></nav>

# Exp 0 - metrics report

**Layer 24**　·　gemma-2-2b / 24-res-matryoshka-dc　·　blocks.24.hook_resid_post　·　48,971 tokens over 400 docs　·　edge: reverse coverage ≥ 0.5, both endpoints fire ≥ 20

## Block pair 0->1  -  4940 candidate edges

- **Out-degree**: 97 parents, 384 children, 384 multi-parented; top-1 parent holds 7.8% of edges, Gini 0.570, max out-degree 384.
- **Superparents**: 6 (e.g. feature 32 _numbers within a document_: 384 children, fires on 99.5% of tokens)
- **Reconstruction**: 879/4940 edges improve reconstruction (17.8%).
- **Frequency control**: mean survival 0.517 over 4940 testable edges; 2635 (53.3%) are frequency-driven (survival < 0.5).
- **Sibling redundancy**: mean 0.350 over 92 parents; 0 flagged as splitting (>= 0.5).
- **Joint-child coverage** (upper bound, mean over parents): 0.944.

| parent -> child | R | F | recon P/C gain | recon? | surv | sib | parent label | child label |
|---|---|---|---|---|---|---|---|---|
| 32 -> 336 | 1.00 | 0.03 | 1.24/0.16 | Y | 1.00 | 0.05 | numbers within a document | numbers, especially as part of referenc… |
| 32 -> 365 | 1.00 | 0.01 | -0.00/-0.00 | n | 1.00 | 0.05 | numbers within a document | citations in research papers |
| 32 -> 421 | 1.00 | 0.02 | 0.00/-0.00 | n | 1.00 | 0.05 | numbers within a document | text related to the game World of Warcr… |
| 32 -> 387 | 1.00 | 0.02 | 0.00/0.00 | n | 1.00 | 0.05 | numbers within a document | contractions with the word "not." |
| 32 -> 201 | 1.00 | 0.03 | 1.15/0.02 | Y | 1.00 | 0.05 | numbers within a document | names of people and titles or abbreviat… |
| 32 -> 193 | 1.00 | 0.00 | 1.10/0.00 | n | 1.00 | 0.05 | numbers within a document | language common to legal opinions. |
| 32 -> 502 | 1.00 | 0.11 | 1.88/0.02 | Y | 1.00 | 0.05 | numbers within a document | words and phrases associated with polit… |
| 32 -> 162 | 1.00 | 0.07 | 0.00/0.00 | n | 1.00 | 0.05 | numbers within a document | terminology used in scientific publicat… |

## Block pair 1->2  -  108810 candidate edges

- **Out-degree**: 233 parents, 989 children, 816 multi-parented; top-1 parent holds 0.8% of edges, Gini 0.516, max out-degree 891.
- **Superparents**: 7 (e.g. feature 355 _words related to medical and business processes or programs_: 891 children, fires on 31.8% of tokens)
- **Reconstruction**: 83/108810 edges improve reconstruction (0.1%).
- **Frequency control**: mean survival 0.056 over 105552 testable edges; 104570 (99.1%) are frequency-driven (survival < 0.5).
- **Sibling redundancy**: mean 0.485 over 206 parents; 163 flagged as splitting (>= 0.5).
- **Joint-child coverage** (upper bound, mean over parents): 0.841.

| parent -> child | R | F | recon P/C gain | recon? | surv | sib | parent label | child label |
|---|---|---|---|---|---|---|---|---|
| 314 -> 791 | 1.00 | 0.49 | -0.00/-0.00 | n | - | 0.54 | This neuron seems to be activated by pu… | LaTeX labels within a mathematical docu… |
| 442 -> 791 | 1.00 | 0.30 | -0.00/-0.00 | n | - | 0.53 | instances of the verb "make." | LaTeX labels within a mathematical docu… |
| 418 -> 791 | 1.00 | 0.34 | -0.00/-0.00 | n | - | 0.53 | the word "that" | LaTeX labels within a mathematical docu… |
| 215 -> 791 | 1.00 | 0.76 | -0.00/-0.00 | n | - | 0.55 | phrases that indicate spam or advertisi… | LaTeX labels within a mathematical docu… |
| 483 -> 791 | 1.00 | 0.47 | 0.00/-0.00 | n | - | 0.54 | the letter 'e' followed by a period or … | LaTeX labels within a mathematical docu… |
| 505 -> 791 | 1.00 | 0.11 | -0.00/-0.00 | n | - | 0.49 | language associated with death and prof… | LaTeX labels within a mathematical docu… |
| 286 -> 791 | 1.00 | 0.37 | 0.00/-0.00 | n | - | 0.53 | references to figures and sections with… | LaTeX labels within a mathematical docu… |
| 178 -> 791 | 1.00 | 0.40 | -0.00/-0.00 | n | - | 0.53 | words and symbols related to physics ex… | LaTeX labels within a mathematical docu… |

## Block pair 2->3  -  3695288 candidate edges

- **Out-degree**: 1105 parents, 5343 children, 4450 multi-parented; top-1 parent holds 0.1% of edges, Gini 0.379, max out-degree 4289.
- **Superparents**: 2 (e.g. feature 2038 _first-person narratives, especially where opinions or perso…_: 4289 children, fires on 21.9% of tokens)
- **Reconstruction**: 247/3695288 edges improve reconstruction (0.0%).
- **Frequency control**: mean survival 0.017 over 3248273 testable edges; 3243688 (99.9%) are frequency-driven (survival < 0.5).
- **Sibling redundancy**: mean 0.379 over 1005 parents; 0 flagged as splitting (>= 0.5).
- **Joint-child coverage** (upper bound, mean over parents): 0.886.

| parent -> child | R | F | recon P/C gain | recon? | surv | sib | parent label | child label |
|---|---|---|---|---|---|---|---|---|
| 924 -> 8084 | 1.00 | 0.67 | -0.00/-0.00 | n | - | 0.40 | numbers that are associated with code | equations and references in academic pa… |
| 1177 -> 2850 | 1.00 | 0.55 | 0.00/-0.00 | n | - | 0.39 | code snippets related to networks | mathematical notation |
| 526 -> 3717 | 1.00 | 0.87 | -0.00/-0.00 | n | - | 0.40 | references to figures | LaTeX markup for mathematical formulas |
| 1464 -> 3685 | 1.00 | 0.45 | 0.00/-0.00 | n | - | 0.39 | statistical and research analyses | escape codes |
| 1270 -> 7976 | 1.00 | 0.21 | -0.00/-0.00 | n | - | 0.37 | words in passages discussing feelings, … | LaTeX math delimiters |
| 1464 -> 3717 | 1.00 | 0.45 | 0.00/-0.00 | n | - | 0.39 | statistical and research analyses | LaTeX markup for mathematical formulas |
| 595 -> 3089 | 1.00 | 0.84 | 0.00/0.00 | n | - | 0.40 | the word "most" | LaTeX code snippets |
| 1823 -> 2850 | 1.00 | 0.74 | -0.00/-0.00 | n | - | 0.40 | number comparisons and math questions, … | mathematical notation |
