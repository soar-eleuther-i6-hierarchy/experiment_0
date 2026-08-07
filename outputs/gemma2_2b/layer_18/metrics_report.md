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
</style><nav class="x0nav"><div class="row"><a class="brand" href="../../../">SOAR I-6 · metrics</a><a class="" href="../../../outputs/">Results</a><a class="" href="../../../outputs/toy_calibration.html">Toy calibration</a><a class="" href="../../../outputs/trained_toy_calibration.html">Trained toy</a><a class="" href="../../../outputs/pcfg/">PCFG</a><a class="on" href="../../../outputs/gemma2_2b/">Gemma2_2b</a></div><div class="row"><span class="lbl">Layer</span><a class="pill" href="../../../outputs/gemma2_2b/layer_03/metrics_report.html">3</a><a class="pill" href="../../../outputs/gemma2_2b/layer_06/metrics_report.html">6</a><a class="pill" href="../../../outputs/gemma2_2b/layer_12/metrics_report.html">12</a><a class="pill on" href="../../../outputs/gemma2_2b/layer_18/metrics_report.html">18</a><a class="pill" href="../../../outputs/gemma2_2b/layer_24/metrics_report.html">24</a><span class="sep"></span><span class="lbl">Page</span><a class="" href="../../../outputs/gemma2_2b/layer_18/metrics_dashboard.html">Dashboard</a><a class="" href="../../../outputs/gemma2_2b/layer_18/superparent_sankey.html">Superparents</a><a class="" href="../../../outputs/gemma2_2b/layer_18/qualitative_dashboard.html">Qualitative</a><a class="on" href="../../../outputs/gemma2_2b/layer_18/metrics_report.html">Metrics report</a><a class="" href="../../../outputs/gemma2_2b/layer_18/qualitative_check.html">Qualitative report</a></div></nav>

# Exp 0 - metrics report

**Layer 18**　·　gemma-2-2b / 18-res-matryoshka-dc　·　blocks.18.hook_resid_post　·　48,571 tokens over 400 docs　·　edge: reverse coverage ≥ 0.5, both endpoints fire ≥ 20

## Block pair 0->1  -  1129 candidate edges

- **Out-degree**: 51 parents, 383 children, 340 multi-parented (PolyFrac 88.8%); top-1 parent holds 33.9% of edges, Gini 0.919, max out-degree 383.
- **Superparents** (out-degree flag): 3 (3 also pass the old fire-rate AND-gate) — e.g. feature 89 _technical or scientific language related to data processing…_: 383 children, fires on 98.9% of tokens
- **Independence null**: mean edge PMI 0.39; 797 edges (70.6%) at chance level (PMI < 0.5). 5 edges dropped by the joint-support guard (n_joint < 30).
- **Recon-ablation contribution filter** (Tree-SAE-inspired baseline): 844/1129 edges pass (74.8%).
- **Frequency control**: mean survival 1.004 over 1129 testable edges; 15 (1.3%) are frequency-driven (survival < 0.5).
- **Sibling redundancy** (global Jaccard — confounded proxy, not the splitting verdict; the parent-conditioned version is in the stage-03 second pass): mean 0.042 over 38 parents; 0 over the 0.5 global threshold.
- **Joint-child (exact union, parents with edges)**: R_supp mean 1.000, R_mass mean 1.000; 9 parents with one child holding >=90% of their energy (rename candidates).
- **Joint-child coverage** (min(1, ΣF) upper bound — saturates when children co-fire, kept only for contrast with the exact union): 0.562.

| parent -> child | R | F | PMI | recon P/C gain | recon? | surv | sib | parent label | child label |
|---|---|---|---|---|---|---|---|---|---|
| 89 -> 263 | 1.00 | 0.02 | 0.01 | 1.95/0.03 | Y | 1.00 | 0.01 | technical or scientific language relate… | mentions of different kinds of governme… |
| 89 -> 132 | 1.00 | 0.02 | 0.01 | 1.30/0.13 | Y | 1.00 | 0.01 | technical or scientific language relate… | hyphens used to connect words |
| 89 -> 249 | 1.00 | 0.05 | 0.01 | 2.14/0.05 | Y | 1.00 | 0.01 | technical or scientific language relate… | references to political parties and ele… |
| 89 -> 261 | 1.00 | 0.07 | 0.01 | 2.68/0.01 | Y | 1.00 | 0.01 | technical or scientific language relate… | passages with emotional and personal re… |
| 89 -> 467 | 1.00 | 0.02 | 0.01 | 2.87/0.03 | Y | 1.00 | 0.01 | technical or scientific language relate… | sentences using the word "I" or "you" i… |
| 89 -> 203 | 1.00 | 0.06 | 0.01 | 2.59/0.01 | n | 1.00 | 0.01 | technical or scientific language relate… | discussions of politics and culture, po… |
| 89 -> 363 | 1.00 | 0.01 | 0.01 | 2.39/0.10 | Y | 1.00 | 0.01 | technical or scientific language relate… | the word "be" in various forms |
| 89 -> 243 | 1.00 | 0.05 | 0.01 | 2.28/0.02 | Y | 1.00 | 0.01 | technical or scientific language relate… | mentions of the Peshekee River in Michi… |

## Block pair 1->2  -  1748 candidate edges

- **Out-degree**: 107 parents, 1450 children, 122 multi-parented (PolyFrac 8.4%); top-1 parent holds 81.2% of edges, Gini 0.967, max out-degree 1420.
- **Superparents** (out-degree flag): 1 (1 also pass the old fire-rate AND-gate) — e.g. feature 304 _words or phrases related to technical manuals, components, …_: 1420 children, fires on 80.9% of tokens
- **Independence null**: mean edge PMI 0.49; 1420 edges (81.2%) at chance level (PMI < 0.5). 54 edges dropped by the joint-support guard (n_joint < 30).
- **Recon-ablation contribution filter** (Tree-SAE-inspired baseline): 648/1748 edges pass (37.1%).
- **Frequency control**: mean survival 0.916 over 1745 testable edges; 194 (11.1%) are frequency-driven (survival < 0.5).
- **Sibling redundancy** (global Jaccard — confounded proxy, not the splitting verdict; the parent-conditioned version is in the stage-03 second pass): mean 0.283 over 65 parents; 0 over the 0.5 global threshold.
- **Joint-child (exact union, parents with edges)**: R_supp mean 1.000, R_mass mean 1.000; 0 parents with one child holding >=90% of their energy (rename candidates).
- **Joint-child coverage** (min(1, ΣF) upper bound — saturates when children co-fire, kept only for contrast with the exact union): 0.254.

| parent -> child | R | F | PMI | recon P/C gain | recon? | surv | sib | parent label | child label |
|---|---|---|---|---|---|---|---|---|---|
| 222 -> 2010 | 0.97 | 0.09 | 3.67 | 0.33/0.14 | Y | 1.00 | 0.04 | code-related terms, particularly those … | the boilerplate language of the Apache … |
| 304 -> 1908 | 0.94 | 0.00 | 0.15 | 0.02/0.17 | Y | 1.01 | 0.01 | words or phrases related to technical m… | the word "However" |
| 275 -> 796 | 0.94 | 0.01 | 1.50 | -0.00/0.00 | n | 0.61 | 0.26 | code-like or markup-like text, identify… | LaTeX commands |
| 304 -> 1293 | 0.94 | 0.07 | 0.15 | 0.02/0.02 | Y | 1.00 | 0.01 | words or phrases related to technical m… | code snippets and programming terms |
| 304 -> 1813 | 0.94 | 0.00 | 0.15 | 0.02/0.35 | Y | 0.98 | 0.01 | words or phrases related to technical m… | the word "Please" |
| 304 -> 1546 | 0.93 | 0.04 | 0.14 | 0.02/0.02 | Y | 0.99 | 0.01 | words or phrases related to technical m… | snippets of computer code, errors, and … |
| 304 -> 1032 | 0.93 | 0.00 | 0.14 | 0.01/0.04 | Y | 1.04 | 0.01 | words or phrases related to technical m… | the word "energy" |
| 304 -> 580 | 0.92 | 0.00 | 0.13 | 0.01/0.21 | Y | 0.98 | 0.01 | words or phrases related to technical m… | the word "risk" and words associated wi… |

## Block pair 2->3  -  4195 candidate edges

- **Out-degree**: 394 parents, 583 children, 178 multi-parented (PolyFrac 30.5%); top-1 parent holds 2.0% of edges, Gini 0.911, max out-degree 82.
- **Superparents** (out-degree flag): 0 (0 also pass the old fire-rate AND-gate)
- **Independence null**: mean edge PMI 4.04; 0 edges (0.0%) at chance level (PMI < 0.5). 529 edges dropped by the joint-support guard (n_joint < 30).
- **Recon-ablation contribution filter** (Tree-SAE-inspired baseline): 408/4195 edges pass (9.7%).
- **Frequency control**: mean survival 0.199 over 3457 testable edges; 2849 (82.4%) are frequency-driven (survival < 0.5).
- **Sibling redundancy** (global Jaccard — confounded proxy, not the splitting verdict; the parent-conditioned version is in the stage-03 second pass): mean 0.278 over 230 parents; 37 over the 0.5 global threshold.
- **Joint-child (exact union, parents with edges)**: R_supp mean 1.000, R_mass mean 1.000; 5 parents with one child holding >=90% of their energy (rename candidates).
- **Joint-child coverage** (min(1, ΣF) upper bound — saturates when children co-fire, kept only for contrast with the exact union): 0.430.

| parent -> child | R | F | PMI | recon P/C gain | recon? | surv | sib | parent label | child label |
|---|---|---|---|---|---|---|---|---|---|
| 600 -> 3712 | 0.98 | 0.07 | 3.98 | 0.00/-0.00 | n | - | 0.35 | source code comments, include statement… | code documentation tags like "summary" |
| 2013 -> 8087 | 0.98 | 0.01 | 1.39 | -0.00/-0.00 | n | - | 0.07 | words related to position and direction | HTML document class declarations |
| 2009 -> 5678 | 0.98 | 0.08 | 4.21 | 0.09/0.02 | Y | 1.02 | 0.07 | text from a song about prejudice | elements of poems, such as question wor… |
| 1781 -> 3090 | 0.98 | 0.31 | 4.85 | 0.21/0.10 | Y | 0.98 | - | words relating to fighting and weight | mentions of a particular Mixed Martial … |
| 880 -> 8103 | 0.97 | 0.06 | 4.31 | 0.11/0.13 | Y | 1.03 | 0.13 | CSS or other code related to colors | HTML color codes |
| 1603 -> 5454 | 0.97 | 0.12 | 4.54 | -0.00/-0.00 | n | - | 0.49 | instances of the word "can" | substrings that match the format "[@B##… |
| 1256 -> 8087 | 0.97 | 0.07 | 3.97 | -0.00/-0.00 | n | - | 0.48 | words related to a crime scene with a f… | HTML document class declarations |
| 806 -> 8087 | 0.97 | 0.05 | 3.59 | -0.00/-0.00 | n | - | 0.36 | text about inventions, patents, and inv… | HTML document class declarations |
