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
</style><nav class="x0nav"><div class="row"><a class="brand" href="../../../">SOAR I-6 · metrics</a><a class="" href="../../../outputs/">Results</a><a class="" href="../../../outputs/synthetic_toy_calibration.html">Synthetic Toy Calibration</a><a class="" href="../../../outputs/trained_toy_calibration.html">Trained Toy Calibration</a><a class="" href="../../../outputs/pcfg-matryoshka/">pcfg-matryoshka</a><a class="on" href="../../../outputs/gemma-2-2b/">gemma-2-2b</a><a class="gh" href="https://github.com/soar-eleuther-i6-hierarchy/metrics" title="Browse the code on GitHub"><svg viewBox="0 0 16 16" aria-hidden="true"><path d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.012 8.012 0 0 0 16 8c0-4.42-3.58-8-8-8z"/></svg>Code</a></div><div class="row"><span class="lbl">Layer</span><a class="pill" href="../../../outputs/gemma-2-2b/layer_01/">1</a><a class="pill" href="../../../outputs/gemma-2-2b/layer_03/">3</a><a class="pill" href="../../../outputs/gemma-2-2b/layer_06/">6</a><a class="pill on" href="../../../outputs/gemma-2-2b/layer_12/">12</a><a class="pill" href="../../../outputs/gemma-2-2b/layer_18/">18</a><a class="pill" href="../../../outputs/gemma-2-2b/layer_24/">24</a><span class="sep"></span><span class="lbl">Page</span><a class="" href="../../../outputs/gemma-2-2b/layer_12/metrics_dashboard.html">Dashboard</a><a class="" href="../../../outputs/gemma-2-2b/layer_12/superparent_sankey.html">Superparents</a><a class="" href="../../../outputs/gemma-2-2b/layer_12/in_block_dashboard.html">In-block</a><a class="" href="../../../outputs/gemma-2-2b/layer_12/qualitative_dashboard.html">Qualitative</a><a class="" href="../../../outputs/gemma-2-2b/layer_12/metrics_report.html">Metrics report</a><a class="" href="../../../outputs/gemma-2-2b/layer_12/in_block_edges.html">In-block report</a><a class="" href="../../../outputs/gemma-2-2b/layer_12/qualitative_check.html">Qualitative report</a></div></nav>

# Edge activation magnitudes

**Layer 12**　·　gemma-2-2b / 12-res-matryoshka-dc　·　blocks.12.hook_resid_post　·　48,571 tokens over 400 docs　·　edge: reverse coverage ≥ 0.5, co-fire ≥ 30, both endpoints fire ≥ 20

Per kept edge: endpoint mean activations on the tokens both fire on. `child/parent` compares the two means; `parent shared/all` is the parent's mean there against its mean over all its firings — well below 1 means the parent goes quiet exactly where the child fires, the regime where the reconstruction contribution filter can fail an active feature.

## Firing values per block

| block | firings | p10 | p50 | p90 |
|---|---|---|---|---|
| B0 | 629,247 | 3.65 | 9.83 | 39.09 |
| B1 | 556,031 | 3.16 | 5.45 | 16.08 |
| B2 | 753,850 | 3.08 | 4.71 | 13.23 |
| B3 | 1,362,205 | 3.04 | 4.34 | 10.08 |
| B4 | 1,098,834 | 3.03 | 4.25 | 9.57 |

## Kept edges per pair

| pair | edges | child/parent p50 | % child stronger | parent shared/all p50 | % parent halved |
|---|---|---|---|---|---|
| 0->1 | 1473 | 0.56 | 8% | 1.03 | 0% |
| 1->2 | 621 | 0.82 | 32% | 1.07 | 6% |
| 2->3 | 1747 | 0.91 | 39% | 0.76 | 12% |

### 0->1: parents most weakened on their child's tokens

| parent → child | shared/all | child/parent | parent label | child label |
|---|---|---|---|---|
| 44 → 209 | 0.37 | 0.27 | technical terminology related to progra… | terms related to legal proceedings and … |
| 44 → 147 | 0.50 | 0.16 | technical terminology related to progra… | phrases associated with legal terminolo… |
| 44 → 369 | 0.52 | 0.21 | technical terminology related to progra… | references to legal terminology and cas… |
| 44 → 433 | 0.52 | 0.17 | technical terminology related to progra… | references to legal cases and citations… |
| 44 → 402 | 0.52 | 0.16 | technical terminology related to progra… | numbers and related numerical values or… |
| 44 → 287 | 0.53 | 0.14 | technical terminology related to progra… | structured data or code snippets with r… |
| 44 → 394 | 0.54 | 0.22 | technical terminology related to progra… | legal citations and references in judic… |
| 44 → 145 | 0.54 | 0.25 | technical terminology related to progra… | numerical values and their occurrences … |

### 1->2: parents most weakened on their child's tokens

| parent → child | shared/all | child/parent | parent label | child label |
|---|---|---|---|---|
| 370 → 775 | 0.24 | 2.45 | proper names and specific identifiers w… | complex mathematical expressions, parti… |
| 196 → 775 | 0.30 | 2.05 | the word "with" and its variations in d… | complex mathematical expressions, parti… |
| 250 → 1466 | 0.32 | 1.26 | questions and interrogative phrases | mathematical expressions and calculatio… |
| 307 → 775 | 0.32 | 2.48 | expressions of necessity or requirement | complex mathematical expressions, parti… |
| 250 → 1659 | 0.32 | 0.88 | questions and interrogative phrases | legal terminology related to jurisdicti… |
| 250 → 1883 | 0.32 | 1.00 | questions and interrogative phrases | references to citations or legal docume… |
| 250 → 1001 | 0.32 | 1.10 | questions and interrogative phrases | mathematical expressions and notations … |
| 250 → 1052 | 0.33 | 0.83 | questions and interrogative phrases | mathematical symbols and notations used… |

### 2->3: parents most weakened on their child's tokens

| parent → child | shared/all | child/parent | parent label | child label |
|---|---|---|---|---|
| 936 → 2168 | 0.20 | 1.31 | terms related to inventions and their d… | structured programming elements and syn… |
| 936 → 2209 | 0.20 | 1.06 | terms related to inventions and their d… | mathematical expressions and variables … |
| 936 → 3519 | 0.20 | 1.45 | terms related to inventions and their d… | modal verbs |
| 936 → 3671 | 0.20 | 1.68 | terms related to inventions and their d… | mathematical expressions and symbols re… |
| 936 → 3903 | 0.20 | 1.14 | terms related to inventions and their d… | character patterns and formats typicall… |
| 936 → 4094 | 0.20 | 1.83 | terms related to inventions and their d… | mathematical expressions involving vari… |
| 936 → 5145 | 0.20 | 1.01 | terms related to inventions and their d… | proper nouns, specifically names and ti… |
| 936 → 5889 | 0.20 | 1.01 | terms related to inventions and their d… | sections or markers within a structured… |
