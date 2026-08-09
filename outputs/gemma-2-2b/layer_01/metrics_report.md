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
.x0nav .gh{display:inline-flex;align-items:center;gap:5px;margin-left:auto;}
.x0nav .gh svg{width:15px;height:15px;fill:currentColor;display:block;}
@media (prefers-color-scheme:dark){
.x0nav{background:#141414;border-bottom-color:#2E2E2E;}
.x0nav .row+.row{border-top-color:#242424;}
.x0nav a{color:#A9B4BF;}
.x0nav .brand,.x0nav a:hover,.x0nav .on{color:#C79BF2;}
.x0nav .pill{background:#1E1830;border-color:#3A2B57;}
.x0nav .pill.on{background:#7C22CE;color:#fff;border-color:#7C22CE;}
.x0nav .sep{background:#2E2E2E;}}
</style><nav class="x0nav"><div class="row"><a class="brand" href="../../../">SOAR I-6 · metrics</a><a class="" href="../../../outputs/">Results</a><a class="" href="../../../outputs/synthetic_toy_calibration.html">Synthetic Toy Calibration</a><a class="" href="../../../outputs/trained_toy_calibration.html">Trained Toy Calibration</a><a class="" href="../../../outputs/pcfg-matryoshka/">pcfg-matryoshka</a><a class="on" href="../../../outputs/gemma-2-2b/">gemma-2-2b</a><a class="gh" href="https://github.com/soar-eleuther-i6-hierarchy/metrics" title="Browse the code on GitHub"><svg viewBox="0 0 16 16" aria-hidden="true"><path d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.012 8.012 0 0 0 16 8c0-4.42-3.58-8-8-8z"/></svg>Code</a></div><div class="row"><span class="lbl">Layer</span><a class="pill on" href="../../../outputs/gemma-2-2b/layer_01/metrics_report.html">1</a><a class="pill" href="../../../outputs/gemma-2-2b/layer_03/metrics_report.html">3</a><a class="pill" href="../../../outputs/gemma-2-2b/layer_06/metrics_report.html">6</a><a class="pill" href="../../../outputs/gemma-2-2b/layer_12/metrics_report.html">12</a><a class="pill" href="../../../outputs/gemma-2-2b/layer_18/metrics_report.html">18</a><a class="pill" href="../../../outputs/gemma-2-2b/layer_24/metrics_report.html">24</a><span class="sep"></span><span class="lbl">Page</span><a class="" href="../../../outputs/gemma-2-2b/layer_01/metrics_dashboard.html">Dashboard</a><a class="" href="../../../outputs/gemma-2-2b/layer_01/superparent_sankey.html">Superparents</a><a class="" href="../../../outputs/gemma-2-2b/layer_01/in_block_dashboard.html">In-block</a><a class="" href="../../../outputs/gemma-2-2b/layer_01/qualitative_dashboard.html">Qualitative</a><a class="on" href="../../../outputs/gemma-2-2b/layer_01/metrics_report.html">Metrics report</a><a class="" href="../../../outputs/gemma-2-2b/layer_01/in_block_edges.html">In-block report</a><a class="" href="../../../outputs/gemma-2-2b/layer_01/qualitative_check.html">Qualitative report</a></div></nav>

# Exp 0 - metrics report

**Layer 1**　·　gemma-2-2b / 1-res-matryoshka-dc　·　blocks.1.hook_resid_post　·　48,571 tokens over 400 docs　·　edge: reverse coverage ≥ 0.5, both endpoints fire ≥ 20

## Block pair 0->1  -  2929 candidate edges

- **Out-degree**: 67 parents, 383 children, 383 multi-parented (PolyFrac 100.0%); top-1 parent holds 13.1% of edges, Gini 0.907, max out-degree 383.
- **Superparents** (out-degree flag): 9 (9 also pass the old fire-rate AND-gate) — e.g. feature 53 _linguistic expressions related to organizing, planning, and…_: 383 children, fires on 99.6% of tokens
- **Independence null**: mean edge PMI 0.28; 2206 edges (75.3%) at chance level (PMI < 0.5). 28 edges dropped by the joint-support guard (n_joint < 30).
- **Recon-ablation contribution filter** (Tree-SAE-inspired baseline): 2522/2929 edges pass (86.1%).
- **Frequency control**: mean survival 1.007 over 2901 testable edges; 21 (0.7%) are frequency-driven (survival < 0.5).
- **Sibling redundancy** (global Jaccard — confounded proxy, not the splitting verdict; the parent-conditioned version is in the stage-03 second pass): mean 0.046 over 47 parents; 0 over the 0.5 global threshold.
- **Joint-child (exact union, parents with edges)**: R_supp mean 0.994, R_mass mean 0.995; 0 parents with one child holding >=90% of their energy (rename candidates).
- **Joint-child coverage** (min(1, ΣF) upper bound — saturates when children co-fire, kept only for contrast with the exact union): 0.437.

| parent -> child | R | F | PMI | recon P/C gain | recon? | surv | sib | parent label | child label |
|---|---|---|---|---|---|---|---|---|---|
| 86 -> 378 | 1.00 | 0.00 | 0.01 | 0.78/-0.00 | n | 1.00 | 0.01 | descriptions of technological or mechan… | mathematical expressions and notation r… |
| 86 -> 309 | 1.00 | 0.00 | 0.01 | 0.94/0.67 | Y | 1.00 | 0.01 | descriptions of technological or mechan… | descriptions of experimental procedures… |
| 126 -> 204 | 1.00 | 0.00 | 0.01 | 0.57/0.68 | Y | 1.00 | 0.01 | expressions of technical or scientific … | the concept of the number one |
| 126 -> 237 | 1.00 | 0.00 | 0.01 | 0.73/3.56 | Y | 1.00 | 0.01 | expressions of technical or scientific … | questions or prompts asking for specifi… |
| 126 -> 238 | 1.00 | 0.00 | 0.01 | 1.47/1.81 | Y | - | 0.01 | expressions of technical or scientific … | examples of equations involving derivat… |
| 126 -> 252 | 1.00 | 0.00 | 0.01 | 0.94/3.53 | Y | 1.00 | 0.01 | expressions of technical or scientific … | references to hierarchical or structure… |
| 126 -> 156 | 1.00 | 0.00 | 0.01 | 0.98/0.86 | Y | 1.00 | 0.01 | expressions of technical or scientific … | sequences of closing brackets or simila… |
| 126 -> 148 | 1.00 | 0.00 | 0.01 | 0.15/0.23 | Y | - | 0.01 | expressions of technical or scientific … | complex numerical relationships and pat… |

## Block pair 1->2  -  1861 candidate edges

- **Out-degree**: 100 parents, 690 children, 329 multi-parented (PolyFrac 47.7%); top-1 parent holds 14.6% of edges, Gini 0.920, max out-degree 271.
- **Superparents** (out-degree flag): 0 (0 also pass the old fire-rate AND-gate)
- **Independence null**: mean edge PMI 2.22; 187 edges (10.0%) at chance level (PMI < 0.5). 294 edges dropped by the joint-support guard (n_joint < 30).
- **Recon-ablation contribution filter** (Tree-SAE-inspired baseline): 371/1861 edges pass (19.9%).
- **Frequency control**: mean survival 0.529 over 1833 testable edges; 894 (48.8%) are frequency-driven (survival < 0.5).
- **Sibling redundancy** (global Jaccard — confounded proxy, not the splitting verdict; the parent-conditioned version is in the stage-03 second pass): mean 0.274 over 68 parents; 0 over the 0.5 global threshold.
- **Joint-child (exact union, parents with edges)**: R_supp mean 1.000, R_mass mean 1.000; 1 parents with one child holding >=90% of their energy (rename candidates).
- **Joint-child coverage** (min(1, ΣF) upper bound — saturates when children co-fire, kept only for contrast with the exact union): 0.561.

| parent -> child | R | F | PMI | recon P/C gain | recon? | surv | sib | parent label | child label |
|---|---|---|---|---|---|---|---|---|---|
| 400 -> 642 | 0.93 | 0.06 | 2.89 | 0.21/0.02 | Y | 1.01 | 0.27 | occurrences of code snippets involving … | snippets of code or text enclosed withi… |
| 303 -> 1846 | 0.92 | 0.12 | 4.37 | -0.00/0.01 | n | - | 0.42 | references to unspecified quantities or… | judgments and related legal final decis… |
| 219 -> 1846 | 0.92 | 0.06 | 3.66 | 0.20/0.01 | n | - | 0.09 | references to self-organization and eme… | judgments and related legal final decis… |
| 236 -> 1846 | 0.91 | 0.06 | 3.62 | -0.01/0.01 | n | - | 0.34 | references to groups of people | judgments and related legal final decis… |
| 219 -> 1791 | 0.90 | 0.09 | 3.64 | 0.23/-0.00 | n | 0.00 | 0.09 | references to self-organization and eme… | structures related to tensor products i… |
| 193 -> 1846 | 0.89 | 0.21 | 4.93 | -0.00/0.01 | n | - | 0.42 | examples of the word "such." | judgments and related legal final decis… |
| 249 -> 1846 | 0.89 | 0.06 | 3.60 | -0.00/0.01 | n | - | 0.41 | references to the word "the" | judgments and related legal final decis… |
| 250 -> 533 | 0.89 | 0.25 | 3.53 | 1.63/0.28 | Y | 0.71 | - | references to the HTML element ' (apost… | references to personal possession or qu… |

## Block pair 2->3  -  16729 candidate edges

- **Out-degree**: 344 parents, 483 children, 277 multi-parented (PolyFrac 57.3%); top-1 parent holds 1.0% of edges, Gini 0.875, max out-degree 166.
- **Superparents** (out-degree flag): 0 (0 also pass the old fire-rate AND-gate)
- **Independence null**: mean edge PMI 4.63; 0 edges (0.0%) at chance level (PMI < 0.5). 2348 edges dropped by the joint-support guard (n_joint < 30).
- **Recon-ablation contribution filter** (Tree-SAE-inspired baseline): 383/16729 edges pass (2.3%).
- **Frequency control**: mean survival 0.047 over 10463 testable edges; 10078 (96.3%) are frequency-driven (survival < 0.5).
- **Sibling redundancy** (global Jaccard — confounded proxy, not the splitting verdict; the parent-conditioned version is in the stage-03 second pass): mean 0.467 over 285 parents; 182 over the 0.5 global threshold.
- **Joint-child (exact union, parents with edges)**: R_supp mean 1.000, R_mass mean 1.000; 90 parents with one child holding >=90% of their energy (rename candidates).
- **Joint-child coverage** (min(1, ΣF) upper bound — saturates when children co-fire, kept only for contrast with the exact union): 0.698.

| parent -> child | R | F | PMI | recon P/C gain | recon? | surv | sib | parent label | child label |
|---|---|---|---|---|---|---|---|---|---|
| 1666 -> 4526 | 1.00 | 0.20 | 5.04 | -0.01/-0.00 | n | - | 0.58 | references to specific observations or … | references to a specific formatting or … |
| 1130 -> 6191 | 1.00 | 0.36 | 5.65 | -0.00/-0.00 | n | - | 0.64 | references to the concept of mathematic… | references to diagrammatic or visual re… |
| 884 -> 7888 | 1.00 | 0.37 | 5.65 | -0.00/-0.00 | n | - | 0.64 | references to ongoing or continuous pro… | references to transcription factors inv… |
| 1294 -> 6165 | 1.00 | 0.01 | 1.96 | 0.00/0.10 | n | - | 0.51 | references to cloud seeding, weather mo… | references to software licensing and li… |
| 582 -> 5659 | 1.00 | 0.44 | 5.86 | 0.00/0.00 | n | - | 0.67 | references to the European Commission's… | representations of byte size information |
| 1714 -> 5659 | 1.00 | 0.03 | 3.06 | -0.00/0.00 | n | - | 0.64 | information related to oxygen levels, o… | representations of byte size information |
| 1261 -> 6191 | 1.00 | 0.23 | 5.21 | -0.00/-0.00 | n | - | 0.64 | references to growth and proliferative … | references to diagrammatic or visual re… |
| 936 -> 5659 | 1.00 | 0.28 | 5.42 | -0.01/0.00 | n | - | 0.63 | the maximum values of relevant geometri… | representations of byte size information |
