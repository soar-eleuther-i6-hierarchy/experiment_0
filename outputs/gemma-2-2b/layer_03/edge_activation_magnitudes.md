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
</style><nav class="x0nav"><div class="row"><a class="brand" href="../../../">SOAR I-6 · metrics</a><a class="" href="../../../outputs/">Results</a><a class="" href="../../../outputs/synthetic_toy_calibration.html">Synthetic Toy Calibration</a><a class="" href="../../../outputs/trained_toy_calibration.html">Trained Toy Calibration</a><a class="" href="../../../outputs/pcfg-matryoshka/">pcfg-matryoshka</a><a class="on" href="../../../outputs/gemma-2-2b/">gemma-2-2b</a><a class="gh" href="https://github.com/soar-eleuther-i6-hierarchy/metrics" title="Browse the code on GitHub"><svg viewBox="0 0 16 16" aria-hidden="true"><path d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.012 8.012 0 0 0 16 8c0-4.42-3.58-8-8-8z"/></svg>Code</a></div><div class="row"><span class="lbl">Layer</span><a class="pill" href="../../../outputs/gemma-2-2b/layer_01/">1</a><a class="pill on" href="../../../outputs/gemma-2-2b/layer_03/">3</a><a class="pill" href="../../../outputs/gemma-2-2b/layer_06/">6</a><a class="pill" href="../../../outputs/gemma-2-2b/layer_12/">12</a><a class="pill" href="../../../outputs/gemma-2-2b/layer_18/">18</a><a class="pill" href="../../../outputs/gemma-2-2b/layer_24/">24</a><span class="sep"></span><span class="lbl">Page</span><a class="" href="../../../outputs/gemma-2-2b/layer_03/metrics_dashboard.html">Dashboard</a><a class="" href="../../../outputs/gemma-2-2b/layer_03/superparent_sankey.html">Superparents</a><a class="" href="../../../outputs/gemma-2-2b/layer_03/in_block_dashboard.html">In-block</a><a class="" href="../../../outputs/gemma-2-2b/layer_03/qualitative_dashboard.html">Qualitative</a><a class="" href="../../../outputs/gemma-2-2b/layer_03/metrics_report.html">Metrics report</a><a class="" href="../../../outputs/gemma-2-2b/layer_03/in_block_edges.html">In-block report</a><a class="" href="../../../outputs/gemma-2-2b/layer_03/qualitative_check.html">Qualitative report</a></div></nav>

# Edge activation magnitudes

**Layer 3**　·　gemma-2-2b / 3-res-matryoshka-dc　·　blocks.3.hook_resid_post　·　48,571 tokens over 400 docs　·　edge: reverse coverage ≥ 0.5, co-fire ≥ 30, both endpoints fire ≥ 20

Per kept edge: endpoint mean activations on the tokens both fire on. `child/parent` compares the two means; `parent shared/all` is the parent's mean there against its mean over all its firings — well below 1 means the parent goes quiet exactly where the child fires, the regime where the reconstruction contribution filter can fail an active feature.

## Firing values per block

| block | firings | p10 | p50 | p90 |
|---|---|---|---|---|
| B0 | 679,452 | 1.63 | 4.97 | 24.36 |
| B1 | 622,767 | 1.47 | 2.76 | 10.16 |
| B2 | 988,106 | 1.44 | 2.49 | 7.54 |
| B3 | 1,105,615 | 1.42 | 2.22 | 6.18 |
| B4 | 806,099 | 1.41 | 2.18 | 6.30 |

## Kept edges per pair

| pair | edges | child/parent p50 | % child stronger | parent shared/all p50 | % parent halved |
|---|---|---|---|---|---|
| 0->1 | 1971 | 0.58 | 13% | 1.07 | 0% |
| 1->2 | 1387 | 0.88 | 39% | 1.11 | 5% |
| 2->3 | 4379 | 0.85 | 39% | 0.99 | 7% |

### 0->1: parents most weakened on their child's tokens

| parent → child | shared/all | child/parent | parent label | child label |
|---|---|---|---|---|
| 17 → 217 | 0.57 | 0.33 | words or phrases related to science and… | mathematical formulas and notation |
| 70 → 461 | 0.61 | 0.37 | proper nouns that have mixed upper and … | mathematical formulas with exponents, a… |
| 52 → 313 | 0.62 | 1.59 | sections from legal and scientific publ… | code snippets and structured data, part… |
| 111 → 184 | 0.62 | 0.58 | code, formatting and markup syntax, and… | uses of the word "first" and subsequent… |
| 70 → 194 | 0.63 | 0.45 | proper nouns that have mixed upper and … | decimal numbers |
| 70 → 325 | 0.64 | 0.27 | proper nouns that have mixed upper and … | computer code |
| 70 → 439 | 0.64 | 0.86 | proper nouns that have mixed upper and … | statistical data from scientific public… |
| 70 → 505 | 0.65 | 0.76 | proper nouns that have mixed upper and … | legal and scientific research article f… |

### 1->2: parents most weakened on their child's tokens

| parent → child | shared/all | child/parent | parent label | child label |
|---|---|---|---|---|
| 209 → 2021 | 0.24 | 0.96 | source code and equations | bold math symbols |
| 209 → 1593 | 0.24 | 1.36 | source code and equations | horizontal lines made of equals signs |
| 223 → 2021 | 0.26 | 0.83 | questions that ask "What are..." or "Wh… | bold math symbols |
| 223 → 954 | 0.26 | 0.92 | questions that ask "What are..." or "Wh… | the phrase "required by" |
| 223 → 1593 | 0.26 | 1.18 | questions that ask "What are..." or "Wh… | horizontal lines made of equals signs |
| 209 → 1678 | 0.27 | 2.04 | source code and equations | LaTeX code |
| 223 → 1678 | 0.28 | 1.89 | questions that ask "What are..." or "Wh… | LaTeX code |
| 209 → 954 | 0.28 | 0.93 | source code and equations | the phrase "required by" |

### 2->3: parents most weakened on their child's tokens

| parent → child | shared/all | child/parent | parent label | child label |
|---|---|---|---|---|
| 1766 → 2404 | 0.12 | 13.21 | the names of days of the week | question delimiters |
| 1766 → 4488 | 0.12 | 1.48 | the names of days of the week | legal citations of cases |
| 1766 → 4750 | 0.12 | 1.45 | the names of days of the week | the phrase "behind the" |
| 1766 → 4815 | 0.12 | 2.83 | the names of days of the week | the word "bottom" |
| 1766 → 5601 | 0.12 | 1.05 | the names of days of the week | asterisks |
| 1766 → 5679 | 0.12 | 1.79 | the names of days of the week | the letters "UT" or "ut" |
| 1766 → 5876 | 0.12 | 1.13 | the names of days of the week | positions or ordinals in mathematical t… |
| 1766 → 6815 | 0.12 | 1.68 | the names of days of the week | math multiple choice questions |
