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
</style><nav class="x0nav"><div class="row"><a class="brand" href="../../../">SOAR I-6 · metrics</a><a class="" href="../../../outputs/">Results</a><a class="" href="../../../outputs/synthetic_toy_calibration.html">Synthetic Toy Calibration</a><a class="" href="../../../outputs/trained_toy_calibration.html">Trained Toy Calibration</a><a class="" href="../../../outputs/pcfg-matryoshka/">pcfg-matryoshka</a><a class="on" href="../../../outputs/gemma-2-2b/">gemma-2-2b</a><a class="gh" href="https://github.com/soar-eleuther-i6-hierarchy/metrics" title="Browse the code on GitHub"><svg viewBox="0 0 16 16" aria-hidden="true"><path d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.012 8.012 0 0 0 16 8c0-4.42-3.58-8-8-8z"/></svg>Code</a></div><div class="row"><span class="lbl">Layer</span><a class="pill on" href="../../../outputs/gemma-2-2b/layer_01/">1</a><a class="pill" href="../../../outputs/gemma-2-2b/layer_03/">3</a><a class="pill" href="../../../outputs/gemma-2-2b/layer_06/">6</a><a class="pill" href="../../../outputs/gemma-2-2b/layer_12/">12</a><a class="pill" href="../../../outputs/gemma-2-2b/layer_18/">18</a><a class="pill" href="../../../outputs/gemma-2-2b/layer_24/">24</a><span class="sep"></span><span class="lbl">Page</span><a class="" href="../../../outputs/gemma-2-2b/layer_01/metrics_dashboard.html">Dashboard</a><a class="" href="../../../outputs/gemma-2-2b/layer_01/superparent_sankey.html">Superparents</a><a class="" href="../../../outputs/gemma-2-2b/layer_01/in_block_dashboard.html">In-block</a><a class="" href="../../../outputs/gemma-2-2b/layer_01/qualitative_dashboard.html">Qualitative</a><a class="" href="../../../outputs/gemma-2-2b/layer_01/metrics_report.html">Metrics report</a><a class="" href="../../../outputs/gemma-2-2b/layer_01/in_block_edges.html">In-block report</a><a class="" href="../../../outputs/gemma-2-2b/layer_01/qualitative_check.html">Qualitative report</a></div></nav>

# Edge activation magnitudes

**Layer 1**　·　gemma-2-2b / 1-res-matryoshka-dc　·　blocks.1.hook_resid_post　·　48,571 tokens over 400 docs　·　edge: reverse coverage ≥ 0.5, co-fire ≥ 30, both endpoints fire ≥ 20

Per kept edge: endpoint mean activations on the tokens both fire on. `child/parent` compares the two means; `parent shared/all` is the parent's mean there against its mean over all its firings — well below 1 means the parent goes quiet exactly where the child fires, the regime where the reconstruction contribution filter can fail an active feature.

## Firing values per block

| block | firings | p10 | p50 | p90 |
|---|---|---|---|---|
| B0 | 725,235 | 1.30 | 5.12 | 26.00 |
| B1 | 436,246 | 1.14 | 2.03 | 9.30 |
| B2 | 712,190 | 1.11 | 1.77 | 7.33 |
| B3 | 924,468 | 1.13 | 1.90 | 5.82 |
| B4 | 1,024,787 | 1.12 | 1.81 | 5.25 |

## Kept edges per pair

| pair | edges | child/parent p50 | % child stronger | parent shared/all p50 | % parent halved |
|---|---|---|---|---|---|
| 0->1 | 2929 | 0.55 | 25% | 1.00 | 0% |
| 1->2 | 1861 | 1.06 | 55% | 0.91 | 40% |
| 2->3 | 16729 | 0.96 | 44% | 0.45 | 56% |

### 0->1: parents most weakened on their child's tokens

| parent → child | shared/all | child/parent | parent label | child label |
|---|---|---|---|---|
| 96 → 148 | 0.19 | 1.62 | sequences of formatting and style code … | complex numerical relationships and pat… |
| 73 → 219 | 0.21 | 12.89 | references to formal mathematical expre… | references to self-organization and eme… |
| 73 → 193 | 0.21 | 0.92 | references to formal mathematical expre… | examples of the word "such." |
| 17 → 148 | 0.22 | 0.37 | sequences of special characters and for… | complex numerical relationships and pat… |
| 111 → 148 | 0.24 | 1.29 | segments of text that are enclosed by q… | complex numerical relationships and pat… |
| 95 → 219 | 0.32 | 7.30 | references to color spaces, especially … | references to self-organization and eme… |
| 91 → 148 | 0.33 | 2.67 | expressions of nested parentheses | complex numerical relationships and pat… |
| 95 → 281 | 0.41 | 0.65 | references to color spaces, especially … | references to the concept of a "between… |

### 1->2: parents most weakened on their child's tokens

| parent → child | shared/all | child/parent | parent label | child label |
|---|---|---|---|---|
| 403 → 1845 | 0.06 | 1.19 | patterns indicative of structured, form… | references to network-related technolog… |
| 403 → 668 | 0.06 | 1.02 | patterns indicative of structured, form… | references to specific gene names |
| 403 → 678 | 0.06 | 1.10 | patterns indicative of structured, form… | references to the defendant |
| 403 → 1846 | 0.06 | 1.15 | patterns indicative of structured, form… | judgments and related legal final decis… |
| 403 → 2036 | 0.06 | 0.96 | patterns indicative of structured, form… | references to executing commands or scr… |
| 403 → 1647 | 0.06 | 1.93 | patterns indicative of structured, form… | references to filtering or filtering de… |
| 403 → 955 | 0.06 | 1.58 | patterns indicative of structured, form… | calculations involving the variable l |
| 403 → 1791 | 0.06 | 1.40 | patterns indicative of structured, form… | structures related to tensor products i… |

### 2->3: parents most weakened on their child's tokens

| parent → child | shared/all | child/parent | parent label | child label |
|---|---|---|---|---|
| 1338 → 2092 | 0.08 | 0.72 | references to the word "that." | references to property-like entities wi… |
| 1338 → 2179 | 0.08 | 0.98 | references to the word "that." | references to court decisions or rulings |
| 1338 → 2261 | 0.08 | 0.73 | references to the word "that." | references to chest-related medical iss… |
| 1338 → 2310 | 0.08 | 0.70 | references to the word "that." | mathematical expressions and array-base… |
| 1338 → 2474 | 0.08 | 1.15 | references to the word "that." | references to the technique of pattern … |
| 1338 → 2485 | 0.08 | 0.73 | references to the word "that." | references to specific types of transpo… |
| 1338 → 2578 | 0.08 | 1.31 | references to the word "that." | references to measures of diversity or … |
| 1338 → 2595 | 0.08 | 1.06 | references to the word "that." | references to the scientific term "infl… |
