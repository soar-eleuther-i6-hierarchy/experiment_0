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
</style><nav class="x0nav"><div class="row"><a class="brand" href="../../../">SOAR I-6 · metrics</a><a class="" href="../../../outputs/">Results</a><a class="" href="../../../outputs/toy_calibration.html">Toy calibration</a><a class="" href="../../../outputs/trained_toy_calibration.html">Trained toy</a><a class="" href="../../../outputs/gemma2_2b/">Gemma2_2b</a><a class="on" href="../../../outputs/pcfg/">PCFG</a><a class="gh" href="https://github.com/soar-eleuther-i6-hierarchy/metrics" title="Browse the code on GitHub"><svg viewBox="0 0 16 16" aria-hidden="true"><path d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.012 8.012 0 0 0 16 8c0-4.42-3.58-8-8-8z"/></svg>Code</a></div><div class="row"><span class="lbl">Layer</span><a class="pill" href="../../../outputs/pcfg/layer_01/metrics_report.html">1</a><a class="pill on" href="../../../outputs/pcfg/layer_03/metrics_report.html">3</a><span class="sep"></span><span class="lbl">Page</span><a class="" href="../../../outputs/pcfg/layer_03/metrics_dashboard.html">Dashboard</a><a class="" href="../../../outputs/pcfg/layer_03/superparent_sankey.html">Superparents</a><a class="on" href="../../../outputs/pcfg/layer_03/metrics_report.html">Metrics report</a></div></nav>

# Exp 0 - metrics report

**Layer 3**　·　PCFG toy 4L d_model=448 / pcfg　·　matryoshka_hook_resid_post_L3　·　1,792 latents in 8 blocks　·　1,016,600 tokens over 3400 docs　·　edge: reverse coverage ≥ 0.5, both endpoints fire ≥ 20

## Block pair 0->1  -  781 candidate edges

- **Out-degree**: 33 parents, 145 children, 144 multi-parented (PolyFrac 99.3%); top-1 parent holds 11.5% of edges, Gini 0.954, max out-degree 90.
- **Superparents** (out-degree flag): 6 (6 also pass the old fire-rate AND-gate) — e.g. feature 3 _feature 3_: 90 children, fires on 36.5% of tokens
- **Independence null**: mean edge PMI 0.80; 144 edges (18.4%) at chance level (PMI < 0.5). 30 edges dropped by the joint-support guard (n_joint < 30).
- **Recon-ablation contribution filter** (Tree-SAE-inspired baseline): 743/781 edges pass (95.1%).
- **Frequency control**: mean survival 0.888 over 781 testable edges; 9 (1.2%) are frequency-driven (survival < 0.5).
- **Sibling redundancy** (global Jaccard — confounded proxy, not the splitting verdict; the parent-conditioned version is in the stage-03 second pass): mean 0.056 over 19 parents; 0 over the 0.5 global threshold.
- **Joint-child (exact union, parents with edges)**: R_supp mean 0.804, R_mass mean 0.773; 0 parents with one child holding >=90% of their energy (rename candidates).
- **Joint-child coverage** (min(1, ΣF) upper bound — saturates when children co-fire, kept only for contrast with the exact union): 0.382.

| parent -> child | R | F | PMI | recon P/C gain | recon? | surv | sib | parent label | child label |
|---|---|---|---|---|---|---|---|---|---|
| 154 -> 240 | 1.00 | 0.00 | 0.91 | 0.55/-0.00 | n | 1.00 | 0.03 | feature 154 | feature 240 |
| 46 -> 240 | 1.00 | 0.00 | 0.93 | 0.59/-0.00 | n | 1.00 | 0.03 | feature 46 | feature 240 |
| 18 -> 434 | 1.00 | 0.00 | 1.37 | 0.58/0.01 | n | 1.00 | 0.02 | feature 18 | feature 434 |
| 88 -> 434 | 1.00 | 0.00 | 1.40 | 0.35/0.01 | n | 1.00 | 0.02 | feature 88 | feature 434 |
| 99 -> 240 | 1.00 | 0.00 | 0.89 | 0.49/-0.00 | n | 1.00 | 0.03 | feature 99 | feature 240 |
| 22 -> 434 | 1.00 | 0.00 | 0.82 | 0.37/0.01 | n | 1.00 | 0.02 | feature 22 | feature 434 |
| 144 -> 434 | 1.00 | 0.00 | 1.37 | 0.25/0.01 | n | 1.00 | 0.02 | feature 144 | feature 434 |
| 12 -> 291 | 1.00 | 0.04 | 1.35 | 2.17/0.63 | Y | 1.00 | 0.05 | feature 12 | feature 291 |

## Block pair 1->2  -  2 candidate edges

- **Out-degree**: 2 parents, 2 children, 0 multi-parented (PolyFrac 0.0%); top-1 parent holds 50.0% of edges, Gini 0.991, max out-degree 1.
- **Superparents** (out-degree flag): 0 (0 also pass the old fire-rate AND-gate)
- **Independence null**: mean edge PMI 3.99; 0 edges (0.0%) at chance level (PMI < 0.5). 2 edges dropped by the joint-support guard (n_joint < 30).
- **Recon-ablation contribution filter** (Tree-SAE-inspired baseline): 2/2 edges pass (100.0%).
- **Frequency control**: mean survival 1.004 over 2 testable edges; 0 (0.0%) are frequency-driven (survival < 0.5).
- **Sibling redundancy**: n/a (child block not in SIBLING_BLOCKS).
- **Joint-child (exact union, parents with edges)**: R_supp mean 0.422, R_mass mean 0.386; 0 parents with one child holding >=90% of their energy (rename candidates).
- **Joint-child coverage** (min(1, ΣF) upper bound — saturates when children co-fire, kept only for contrast with the exact union): 0.221.

| parent -> child | R | F | PMI | recon P/C gain | recon? | surv | sib | parent label | child label |
|---|---|---|---|---|---|---|---|---|---|
| 245 -> 490 | 0.64 | 0.33 | 5.12 | 0.19/0.57 | Y | 1.01 | - | feature 245 | feature 490 |
| 254 -> 647 | 0.56 | 0.11 | 2.87 | 0.13/0.39 | Y | 1.00 | - | feature 254 | feature 647 |

## Block pair 2->3  -  0 candidate edges

- **Out-degree**: 0 parents, 0 children, 0 multi-parented (PolyFrac 0.0%); top-1 parent holds 0.0% of edges, Gini 0.000, max out-degree 0.
- **Superparents** (out-degree flag): 0 (0 also pass the old fire-rate AND-gate)
- **Independence null**: mean edge PMI nan; 0 edges (0.0%) at chance level (PMI < 0.5). 4 edges dropped by the joint-support guard (n_joint < 30).
- **Recon-ablation contribution filter** (Tree-SAE-inspired baseline): 0/0 edges pass (0.0%).
- **Frequency control**: mean survival nan over 0 testable edges; 0 (0.0%) are frequency-driven (survival < 0.5).
- **Sibling redundancy**: n/a (child block not in SIBLING_BLOCKS).
- **Joint-child (exact union, parents with edges)**: R_supp mean nan, R_mass mean nan; 0 parents with one child holding >=90% of their energy (rename candidates).
- **Joint-child coverage** (min(1, ΣF) upper bound — saturates when children co-fire, kept only for contrast with the exact union): n/a.

## Block pair 3->4  -  0 candidate edges

- **Out-degree**: 0 parents, 0 children, 0 multi-parented (PolyFrac 0.0%); top-1 parent holds 0.0% of edges, Gini 0.000, max out-degree 0.
- **Superparents** (out-degree flag): 0 (0 also pass the old fire-rate AND-gate)
- **Independence null**: mean edge PMI nan; 0 edges (0.0%) at chance level (PMI < 0.5). 8 edges dropped by the joint-support guard (n_joint < 30).
- **Recon-ablation contribution filter** (Tree-SAE-inspired baseline): 0/0 edges pass (0.0%).
- **Frequency control**: mean survival nan over 0 testable edges; 0 (0.0%) are frequency-driven (survival < 0.5).
- **Sibling redundancy**: n/a (child block not in SIBLING_BLOCKS).
- **Joint-child (exact union, parents with edges)**: R_supp mean nan, R_mass mean nan; 0 parents with one child holding >=90% of their energy (rename candidates).
- **Joint-child coverage** (min(1, ΣF) upper bound — saturates when children co-fire, kept only for contrast with the exact union): n/a.

## Block pair 4->5  -  3 candidate edges

- **Out-degree**: 3 parents, 1 children, 1 multi-parented (PolyFrac 100.0%); top-1 parent holds 33.3% of edges, Gini 0.987, max out-degree 1.
- **Superparents** (out-degree flag): 0 (0 also pass the old fire-rate AND-gate)
- **Independence null**: mean edge PMI 3.93; 0 edges (0.0%) at chance level (PMI < 0.5). 4 edges dropped by the joint-support guard (n_joint < 30).
- **Recon-ablation contribution filter** (Tree-SAE-inspired baseline): 0/3 edges pass (0.0%).
- **Frequency control**: mean survival 0.822 over 3 testable edges; 0 (0.0%) are frequency-driven (survival < 0.5).
- **Sibling redundancy**: n/a (child block not in SIBLING_BLOCKS).
- **Joint-child (exact union, parents with edges)**: R_supp mean 0.739, R_mass mean 0.655; 0 parents with one child holding >=90% of their energy (rename candidates).
- **Joint-child coverage** (min(1, ΣF) upper bound — saturates when children co-fire, kept only for contrast with the exact union): 0.004.

| parent -> child | R | F | PMI | recon P/C gain | recon? | surv | sib | parent label | child label |
|---|---|---|---|---|---|---|---|---|---|
| 943 -> 1240 | 0.58 | 0.00 | 3.91 | 0.09/-0.01 | n | 0.79 | - | feature 943 | feature 1240 |
| 1020 -> 1240 | 0.56 | 0.01 | 4.05 | 0.06/-0.01 | n | 0.87 | - | feature 1020 | feature 1240 |
| 1092 -> 1240 | 0.54 | 0.00 | 3.84 | 0.09/-0.01 | n | 0.81 | - | feature 1092 | feature 1240 |

## Block pair 5->6  -  1 candidate edges

- **Out-degree**: 1 parents, 1 children, 0 multi-parented (PolyFrac 0.0%); top-1 parent holds 100.0% of edges, Gini 0.996, max out-degree 1.
- **Superparents** (out-degree flag): 0 (0 also pass the old fire-rate AND-gate)
- **Independence null**: mean edge PMI 5.03; 0 edges (0.0%) at chance level (PMI < 0.5). 1 edges dropped by the joint-support guard (n_joint < 30).
- **Recon-ablation contribution filter** (Tree-SAE-inspired baseline): 1/1 edges pass (100.0%).
- **Frequency control**: mean survival 1.087 over 1 testable edges; 0 (0.0%) are frequency-driven (survival < 0.5).
- **Sibling redundancy**: n/a (child block not in SIBLING_BLOCKS).
- **Joint-child (exact union, parents with edges)**: R_supp mean 0.205, R_mass mean 0.177; 0 parents with one child holding >=90% of their energy (rename candidates).
- **Joint-child coverage** (min(1, ΣF) upper bound — saturates when children co-fire, kept only for contrast with the exact union): 0.024.

| parent -> child | R | F | PMI | recon P/C gain | recon? | surv | sib | parent label | child label |
|---|---|---|---|---|---|---|---|---|---|
| 1280 -> 1475 | 0.54 | 0.02 | 5.03 | 0.14/0.15 | Y | 1.09 | - | feature 1280 | feature 1475 |

## Block pair 6->7  -  0 candidate edges

- **Out-degree**: 0 parents, 0 children, 0 multi-parented (PolyFrac 0.0%); top-1 parent holds 0.0% of edges, Gini 0.000, max out-degree 0.
- **Superparents** (out-degree flag): 0 (0 also pass the old fire-rate AND-gate)
- **Independence null**: mean edge PMI nan; 0 edges (0.0%) at chance level (PMI < 0.5). 0 edges dropped by the joint-support guard (n_joint < 30).
- **Recon-ablation contribution filter** (Tree-SAE-inspired baseline): 0/0 edges pass (0.0%).
- **Frequency control**: mean survival nan over 0 testable edges; 0 (0.0%) are frequency-driven (survival < 0.5).
- **Sibling redundancy**: n/a (child block not in SIBLING_BLOCKS).
- **Joint-child (exact union, parents with edges)**: R_supp mean nan, R_mass mean nan; 0 parents with one child holding >=90% of their energy (rename candidates).
- **Joint-child coverage** (min(1, ΣF) upper bound — saturates when children co-fire, kept only for contrast with the exact union): n/a.
