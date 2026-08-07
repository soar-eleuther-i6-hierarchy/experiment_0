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
.x0nav .gh{display:inline-flex;align-items:center;gap:5px;}
.x0nav .gh svg{width:15px;height:15px;fill:currentColor;display:block;}
@media (prefers-color-scheme:dark){
.x0nav{background:#141414;border-bottom-color:#2E2E2E;}
.x0nav .row+.row{border-top-color:#242424;}
.x0nav a{color:#A9B4BF;}
.x0nav .brand,.x0nav a:hover,.x0nav .on{color:#C79BF2;}
.x0nav .pill{background:#1E1830;border-color:#3A2B57;}
.x0nav .pill.on{background:#7C22CE;color:#fff;border-color:#7C22CE;}
.x0nav .sep{background:#2E2E2E;}}
</style><nav class="x0nav"><div class="row"><a class="brand" href="../../../">SOAR I-6 · metrics</a><a class="gh" href="https://github.com/soar-eleuther-i6-hierarchy/metrics" title="Browse the code on GitHub"><svg viewBox="0 0 16 16" aria-hidden="true"><path d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.012 8.012 0 0 0 16 8c0-4.42-3.58-8-8-8z"/></svg>Code</a><span class="sep"></span><a class="" href="../../../outputs/">Results</a><a class="" href="../../../outputs/toy_calibration.html">Toy calibration</a><a class="" href="../../../outputs/trained_toy_calibration.html">Trained toy</a><a class="on" href="../../../outputs/gemma2_2b/">Gemma2_2b</a><a class="" href="../../../outputs/pcfg/">PCFG</a></div><div class="row"><span class="lbl">Layer</span><a class="pill on" href="../../../outputs/gemma2_2b/layer_03/metrics_report.html">3</a><a class="pill" href="../../../outputs/gemma2_2b/layer_06/metrics_report.html">6</a><a class="pill" href="../../../outputs/gemma2_2b/layer_12/metrics_report.html">12</a><a class="pill" href="../../../outputs/gemma2_2b/layer_18/metrics_report.html">18</a><a class="pill" href="../../../outputs/gemma2_2b/layer_24/metrics_report.html">24</a><span class="sep"></span><span class="lbl">Page</span><a class="" href="../../../outputs/gemma2_2b/layer_03/metrics_dashboard.html">Dashboard</a><a class="" href="../../../outputs/gemma2_2b/layer_03/superparent_sankey.html">Superparents</a><a class="" href="../../../outputs/gemma2_2b/layer_03/qualitative_dashboard.html">Qualitative</a><a class="on" href="../../../outputs/gemma2_2b/layer_03/metrics_report.html">Metrics report</a><a class="" href="../../../outputs/gemma2_2b/layer_03/qualitative_check.html">Qualitative report</a></div></nav>

# Exp 0 - metrics report

**Layer 3**　·　gemma-2-2b / 3-res-matryoshka-dc　·　blocks.3.hook_resid_post　·　48,571 tokens over 400 docs　·　edge: reverse coverage ≥ 0.5, both endpoints fire ≥ 20

## Block pair 0->1  -  1971 candidate edges

- **Out-degree**: 60 parents, 381 children, 377 multi-parented (PolyFrac 99.0%); top-1 parent holds 19.3% of edges, Gini 0.917, max out-degree 381.
- **Superparents** (out-degree flag): 5 (5 also pass the old fire-rate AND-gate) — e.g. feature 70 _proper nouns that have mixed upper and lowercase letters or…_: 381 children, fires on 99.3% of tokens
- **Independence null**: mean edge PMI 0.34; 1361 edges (69.1%) at chance level (PMI < 0.5). 17 edges dropped by the joint-support guard (n_joint < 30).
- **Recon-ablation contribution filter** (Tree-SAE-inspired baseline): 1770/1971 edges pass (89.8%).
- **Frequency control**: mean survival 1.040 over 1971 testable edges; 18 (0.9%) are frequency-driven (survival < 0.5).
- **Sibling redundancy** (global Jaccard — confounded proxy, not the splitting verdict; the parent-conditioned version is in the stage-03 second pass): mean 0.059 over 46 parents; 0 over the 0.5 global threshold.
- **Joint-child (exact union, parents with edges)**: R_supp mean 0.999, R_mass mean 1.000; 0 parents with one child holding >=90% of their energy (rename candidates).
- **Joint-child coverage** (min(1, ΣF) upper bound — saturates when children co-fire, kept only for contrast with the exact union): 0.482.

| parent -> child | R | F | PMI | recon P/C gain | recon? | surv | sib | parent label | child label |
|---|---|---|---|---|---|---|---|---|---|
| 70 -> 459 | 1.00 | 0.01 | 0.01 | 1.50/0.10 | Y | 1.00 | 0.01 | proper nouns that have mixed upper and … | the word "work" and any words that coll… |
| 70 -> 352 | 1.00 | 0.00 | 0.01 | 1.95/0.01 | Y | 1.00 | 0.01 | proper nouns that have mixed upper and … | square brackets surrounding numbers and… |
| 70 -> 351 | 1.00 | 0.01 | 0.01 | 1.23/0.11 | Y | 1.00 | 0.01 | proper nouns that have mixed upper and … | mentions of `value` or `values` in vari… |
| 70 -> 346 | 1.00 | 0.00 | 0.01 | 0.74/-0.00 | n | 1.00 | 0.01 | proper nouns that have mixed upper and … | extended ascii characters |
| 70 -> 322 | 1.00 | 0.00 | 0.01 | 1.13/0.10 | Y | 1.00 | 0.01 | proper nouns that have mixed upper and … | the keyword "return" in code |
| 70 -> 316 | 1.00 | 0.01 | 0.01 | 1.18/0.07 | Y | 1.00 | 0.01 | proper nouns that have mixed upper and … | reflexive pronouns. |
| 70 -> 310 | 1.00 | 0.01 | 0.01 | 1.27/0.04 | Y | 1.00 | 0.01 | proper nouns that have mixed upper and … | the keyword "public" in code |
| 70 -> 302 | 1.00 | 0.00 | 0.01 | 0.86/0.13 | Y | 1.00 | 0.01 | proper nouns that have mixed upper and … | citations |

## Block pair 1->2  -  1387 candidate edges

- **Out-degree**: 122 parents, 856 children, 300 multi-parented (PolyFrac 35.0%); top-1 parent holds 57.5% of edges, Gini 0.949, max out-degree 798.
- **Superparents** (out-degree flag): 1 (1 also pass the old fire-rate AND-gate) — e.g. feature 448 _a grab bag of proper nouns including names, places, and cod…_: 798 children, fires on 48.0% of tokens
- **Independence null**: mean edge PMI 0.92; 793 edges (57.2%) at chance level (PMI < 0.5). 213 edges dropped by the joint-support guard (n_joint < 30).
- **Recon-ablation contribution filter** (Tree-SAE-inspired baseline): 632/1387 edges pass (45.6%).
- **Frequency control**: mean survival 1.010 over 1313 testable edges; 110 (8.4%) are frequency-driven (survival < 0.5).
- **Sibling redundancy** (global Jaccard — confounded proxy, not the splitting verdict; the parent-conditioned version is in the stage-03 second pass): mean 0.237 over 72 parents; 10 over the 0.5 global threshold.
- **Joint-child (exact union, parents with edges)**: R_supp mean 1.000, R_mass mean 1.000; 56 parents with one child holding >=90% of their energy (rename candidates).
- **Joint-child coverage** (min(1, ΣF) upper bound — saturates when children co-fire, kept only for contrast with the exact union): 0.197.

| parent -> child | R | F | PMI | recon P/C gain | recon? | surv | sib | parent label | child label |
|---|---|---|---|---|---|---|---|---|---|
| 474 -> 1032 | 0.95 | 0.13 | 4.55 | -0.00/-0.00 | n | - | 0.56 | square brackets, often used in mathemat… | the `aligned` environment tag in LaTeX. |
| 208 -> 1032 | 0.95 | 0.01 | 2.34 | 0.00/-0.00 | n | - | 0.47 | scientific and technical texts, especia… | the `aligned` environment tag in LaTeX. |
| 154 -> 1032 | 0.95 | 0.02 | 2.55 | -0.00/-0.00 | n | - | 0.28 | words from scientific documents related… | the `aligned` environment tag in LaTeX. |
| 269 -> 1032 | 0.95 | 0.01 | 1.53 | 0.01/-0.00 | n | - | 0.25 | sequences of symbols denoting mathemati… | the `aligned` environment tag in LaTeX. |
| 145 -> 1032 | 0.95 | 0.00 | 1.25 | -0.01/-0.00 | n | - | 0.05 | words, terms, and related symbols in so… | the `aligned` environment tag in LaTeX. |
| 448 -> 1032 | 0.95 | 0.00 | 0.69 | 0.06/-0.00 | n | - | 0.02 | a grab bag of proper nouns including na… | the `aligned` environment tag in LaTeX. |
| 370 -> 1032 | 0.95 | 0.04 | 3.30 | 0.01/-0.00 | n | - | 0.10 | code, math equations, and youtube links | the `aligned` environment tag in LaTeX. |
| 199 -> 1032 | 0.95 | 0.01 | 1.53 | 0.01/-0.00 | n | - | 0.07 | technical or scientific terms, includin… | the `aligned` environment tag in LaTeX. |

## Block pair 2->3  -  4379 candidate edges

- **Out-degree**: 289 parents, 2810 children, 436 multi-parented (PolyFrac 15.5%); top-1 parent holds 64.1% of edges, Gini 0.972, max out-degree 2806.
- **Superparents** (out-degree flag): 1 (1 also pass the old fire-rate AND-gate) — e.g. feature 1457 _words used in official documents and scientific publications_: 2806 children, fires on 87.9% of tokens
- **Independence null**: mean edge PMI 1.25; 2812 edges (64.2%) at chance level (PMI < 0.5). 2089 edges dropped by the joint-support guard (n_joint < 30).
- **Recon-ablation contribution filter** (Tree-SAE-inspired baseline): 1786/4379 edges pass (40.8%).
- **Frequency control**: mean survival 0.802 over 3917 testable edges; 833 (21.3%) are frequency-driven (survival < 0.5).
- **Sibling redundancy** (global Jaccard — confounded proxy, not the splitting verdict; the parent-conditioned version is in the stage-03 second pass): mean 0.213 over 163 parents; 25 over the 0.5 global threshold.
- **Joint-child (exact union, parents with edges)**: R_supp mean 1.000, R_mass mean 1.000; 5 parents with one child holding >=90% of their energy (rename candidates).
- **Joint-child coverage** (min(1, ΣF) upper bound — saturates when children co-fire, kept only for contrast with the exact union): 0.351.

| parent -> child | R | F | PMI | recon P/C gain | recon? | surv | sib | parent label | child label |
|---|---|---|---|---|---|---|---|---|---|
| 1457 -> 6942 | 1.00 | 0.00 | 0.13 | 0.08/0.00 | n | 1.00 | 0.02 | words used in official documents and sc… | adjectives and nouns related to current… |
| 1607 -> 6997 | 1.00 | 0.13 | 4.62 | 0.00/-0.00 | n | - | 0.22 | the word "those" | the word "spatial" |
| 1457 -> 6449 | 1.00 | 0.00 | 0.13 | 0.02/0.24 | Y | 1.00 | 0.02 | words used in official documents and sc… | the word "this" |
| 652 -> 6997 | 1.00 | 0.03 | 3.02 | 0.00/-0.00 | n | - | 0.18 | various programming code elements like … | the word "spatial" |
| 1058 -> 6997 | 1.00 | 0.12 | 4.57 | -0.00/-0.00 | n | - | 0.56 | capitalized instances of the word "The"… | the word "spatial" |
| 1032 -> 6997 | 1.00 | 0.94 | 6.62 | -0.00/-0.00 | n | - | 0.56 | the `aligned` environment tag in LaTeX. | the word "spatial" |
| 1665 -> 6997 | 1.00 | 0.18 | 4.94 | 0.00/-0.00 | n | - | 0.56 | names of countries and regions | the word "spatial" |
| 1457 -> 2651 | 1.00 | 0.00 | 0.13 | 0.01/0.20 | Y | 1.00 | 0.02 | words used in official documents and sc… | the word "document" and words near it |
