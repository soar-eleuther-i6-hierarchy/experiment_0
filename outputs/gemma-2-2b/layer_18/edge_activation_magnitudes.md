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
.x0nav details.dens{flex:1 1 100%;}
.x0nav details.dens summary{cursor:pointer;list-style:none;user-select:none;display:inline-block;}
.x0nav details.dens summary::-webkit-details-marker{display:none;}
.x0nav details.dens summary::after{content:"▸";margin-left:4px;}
.x0nav details.dens[open] summary::after{content:"▾";}
.x0nav details.dens summary:hover{color:#7C22CE;}
.x0nav .drow{display:flex;flex-wrap:wrap;align-items:center;gap:13px;padding-top:9px;}
@media (prefers-color-scheme:dark){
.x0nav{background:#141414;border-bottom-color:#2E2E2E;}
.x0nav .row+.row{border-top-color:#242424;}
.x0nav a{color:#A9B4BF;}
.x0nav .brand,.x0nav a:hover,.x0nav .on{color:#C79BF2;}
.x0nav .pill{background:#1E1830;border-color:#3A2B57;}
.x0nav .pill.on{background:#7C22CE;color:#fff;border-color:#7C22CE;}
.x0nav .sep{background:#2E2E2E;}
.x0nav details.dens summary:hover{color:#C79BF2;}}
</style><nav class="x0nav"><div class="row"><a class="brand" href="../../../">SOAR I-6 · metrics</a><a class="" href="../../../outputs/">Results</a><a class="" href="../../../outputs/synthetic_toy_calibration.html">Synthetic Toy Calibration</a><a class="" href="../../../outputs/trained_toy_calibration.html">Trained Toy Calibration</a><a class="" href="../../../outputs/pcfg-matryoshka/">pcfg-matryoshka</a><a class="on" href="../../../outputs/gemma-2-2b/">gemma-2-2b</a><a class="gh" href="https://github.com/soar-eleuther-i6-hierarchy/metrics" title="Browse the code on GitHub"><svg viewBox="0 0 16 16" aria-hidden="true"><path d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.012 8.012 0 0 0 16 8c0-4.42-3.58-8-8-8z"/></svg>Code</a></div><div class="row"><span class="lbl">Layer</span><a class="pill" href="../../../outputs/gemma-2-2b/layer_01/">1</a><a class="pill" href="../../../outputs/gemma-2-2b/layer_03/">3</a><a class="pill" href="../../../outputs/gemma-2-2b/layer_06/">6</a><a class="pill" href="../../../outputs/gemma-2-2b/layer_12/">12</a><a class="pill on" href="../../../outputs/gemma-2-2b/layer_18/">18</a><a class="pill" href="../../../outputs/gemma-2-2b/layer_24/">24</a><span class="sep"></span><span class="lbl">Page</span><a class="" href="../../../outputs/gemma-2-2b/layer_18/metrics_dashboard.html">Dashboard</a><a class="" href="../../../outputs/gemma-2-2b/layer_18/superparent_sankey.html">Superparents</a><a class="" href="../../../outputs/gemma-2-2b/layer_18/in_block_dashboard.html">In-block</a><a class="" href="../../../outputs/gemma-2-2b/layer_18/qualitative_dashboard.html">Qualitative</a><a class="" href="../../../outputs/gemma-2-2b/layer_18/metrics_report.html">Metrics report</a><a class="" href="../../../outputs/gemma-2-2b/layer_18/in_block_edges.html">In-block report</a><a class="" href="../../../outputs/gemma-2-2b/layer_18/qualitative_check.html">Qualitative report</a></div></nav>

# Edge activation magnitudes

**Layer 18**　·　gemma-2-2b / 18-res-matryoshka-dc　·　blocks.18.hook_resid_post　·　48,571 tokens over 400 docs　·　edge: reverse coverage ≥ 0.5, co-fire ≥ 30, both endpoints fire ≥ 20

Per kept edge: endpoint mean activations on the tokens both fire on. `child/parent` compares the two means; `parent shared/all` is the parent's mean there against its mean over all its firings — well below 1 means the parent goes quiet exactly where the child fires, the regime where the reconstruction contribution filter can fail an active feature.

## Firing values per block

| block | firings | p10 | p50 | p90 |
|---|---|---|---|---|
| B0 | 596,173 | 6.36 | 16.30 | 74.19 |
| B1 | 606,369 | 5.68 | 9.62 | 28.81 |
| B2 | 851,607 | 5.55 | 8.47 | 26.33 |
| B3 | 1,384,694 | 5.54 | 8.16 | 20.50 |
| B4 | 935,193 | 5.53 | 8.07 | 19.94 |

## Kept edges per pair

| pair | edges | child/parent p50 | % child stronger | parent shared/all p50 | % parent halved |
|---|---|---|---|---|---|
| 0->1 | 1129 | 0.54 | 10% | 1.06 | 0% |
| 1->2 | 1748 | 1.00 | 50% | 0.99 | 3% |
| 2->3 | 4195 | 1.02 | 53% | 0.63 | 30% |

### 0->1: parents most weakened on their child's tokens

| parent → child | shared/all | child/parent | parent label | child label |
|---|---|---|---|---|
| 89 → 216 | 0.29 | 0.24 | technical or scientific language relate… | LaTeX math environments |
| 89 → 222 | 0.37 | 0.50 | technical or scientific language relate… | code-related terms, particularly those … |
| 89 → 201 | 0.48 | 0.25 | technical or scientific language relate… | code snippets with certain structural e… |
| 89 → 374 | 0.49 | 0.34 | technical or scientific language relate… | internet URLs beginning with "http" |
| 89 → 163 | 0.50 | 0.10 | technical or scientific language relate… | LaTeX equations and expressions. |
| 89 → 328 | 0.50 | 0.14 | technical or scientific language relate… | scientific references and quantitative … |
| 89 → 442 | 0.51 | 0.27 | technical or scientific language relate… | code indentation |
| 89 → 435 | 0.52 | 0.10 | technical or scientific language relate… | French characters with accents |

### 1->2: parents most weakened on their child's tokens

| parent → child | shared/all | child/parent | parent label | child label |
|---|---|---|---|---|
| 420 → 592 | 0.07 | 1.19 | blank lines | references to figures |
| 420 → 1746 | 0.07 | 1.12 | blank lines | mathematical code, in particular the ke… |
| 420 → 964 | 0.07 | 1.71 | blank lines | legal citations |
| 317 → 592 | 0.18 | 1.07 | identifies months of the year, especial… | references to figures |
| 317 → 1746 | 0.18 | 1.01 | identifies months of the year, especial… | mathematical code, in particular the ke… |
| 317 → 964 | 0.18 | 1.54 | identifies months of the year, especial… | legal citations |
| 420 → 656 | 0.20 | 0.38 | blank lines | legal jargon |
| 374 → 592 | 0.21 | 1.12 | internet URLs beginning with "http" | references to figures |

### 2->3: parents most weakened on their child's tokens

| parent → child | shared/all | child/parent | parent label | child label |
|---|---|---|---|---|
| 1156 → 2287 | 0.23 | 1.83 | comments in code with a star preceeding… | references to religious texts |
| 1156 → 2858 | 0.23 | 1.76 | comments in code with a star preceeding… | equations with delta functions, especia… |
| 1156 → 3007 | 0.23 | 1.11 | comments in code with a star preceeding… | ranges, such as between two numbers or … |
| 1156 → 3349 | 0.23 | 1.61 | comments in code with a star preceeding… | uses common words when making logical a… |
| 1156 → 3413 | 0.23 | 1.30 | comments in code with a star preceeding… | the word "master" in various contexts |
| 1156 → 3732 | 0.23 | 2.91 | comments in code with a star preceeding… | seemingly random letters or numbers, of… |
| 1156 → 3952 | 0.23 | 1.14 | comments in code with a star preceeding… | grammatical errors and suggestions on f… |
| 1156 → 4235 | 0.23 | 1.23 | comments in code with a star preceeding… | legal jargon in court documents |
