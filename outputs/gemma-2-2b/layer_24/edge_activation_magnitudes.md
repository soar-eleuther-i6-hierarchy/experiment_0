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
</style><nav class="x0nav"><div class="row"><a class="brand" href="../../../">SOAR I-6 · metrics</a><a class="" href="../../../outputs/">Results</a><a class="" href="../../../outputs/synthetic_toy_calibration.html">Synthetic Toy Calibration</a><a class="" href="../../../outputs/trained_toy_calibration.html">Trained Toy Calibration</a><a class="" href="../../../outputs/pcfg-matryoshka/">pcfg-matryoshka</a><a class="on" href="../../../outputs/gemma-2-2b/">gemma-2-2b</a><a class="gh" href="https://github.com/soar-eleuther-i6-hierarchy/metrics" title="Browse the code on GitHub"><svg viewBox="0 0 16 16" aria-hidden="true"><path d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.012 8.012 0 0 0 16 8c0-4.42-3.58-8-8-8z"/></svg>Code</a></div><div class="row"><span class="lbl">Layer</span><a class="pill" href="../../../outputs/gemma-2-2b/layer_01/">1</a><a class="pill" href="../../../outputs/gemma-2-2b/layer_03/">3</a><a class="pill" href="../../../outputs/gemma-2-2b/layer_06/">6</a><a class="pill" href="../../../outputs/gemma-2-2b/layer_12/">12</a><a class="pill" href="../../../outputs/gemma-2-2b/layer_18/">18</a><a class="pill on" href="../../../outputs/gemma-2-2b/layer_24/">24</a><span class="sep"></span><span class="lbl">Page</span><a class="" href="../../../outputs/gemma-2-2b/layer_24/metrics_dashboard.html">Dashboard</a><a class="" href="../../../outputs/gemma-2-2b/layer_24/superparent_sankey.html">Superparents</a><a class="" href="../../../outputs/gemma-2-2b/layer_24/in_block_dashboard.html">In-block</a><a class="" href="../../../outputs/gemma-2-2b/layer_24/qualitative_dashboard.html">Qualitative</a><a class="" href="../../../outputs/gemma-2-2b/layer_24/metrics_report.html">Metrics report</a><a class="" href="../../../outputs/gemma-2-2b/layer_24/in_block_edges.html">In-block report</a><a class="" href="../../../outputs/gemma-2-2b/layer_24/qualitative_check.html">Qualitative report</a></div></nav>

# Edge activation magnitudes

**Layer 24**　·　gemma-2-2b / 24-res-matryoshka-dc　·　blocks.24.hook_resid_post　·　48,571 tokens over 400 docs　·　edge: reverse coverage ≥ 0.5, co-fire ≥ 30, both endpoints fire ≥ 20

Per kept edge: endpoint mean activations on the tokens both fire on. `child/parent` compares the two means; `parent shared/all` is the parent's mean there against its mean over all its firings — well below 1 means the parent goes quiet exactly where the child fires, the regime where the reconstruction contribution filter can fail an active feature.

## Firing values per block

| block | firings | p10 | p50 | p90 |
|---|---|---|---|---|
| B0 | 677,187 | 13.02 | 37.94 | 165.00 |
| B1 | 597,346 | 11.52 | 19.55 | 65.00 |
| B2 | 851,798 | 11.34 | 17.77 | 51.78 |
| B3 | 1,163,676 | 11.30 | 16.95 | 43.09 |
| B4 | 780,092 | 11.32 | 17.28 | 45.78 |

## Kept edges per pair

| pair | edges | child/parent p50 | % child stronger | parent shared/all p50 | % parent halved |
|---|---|---|---|---|---|
| 0->1 | 2273 | 0.45 | 6% | 1.01 | 0% |
| 1->2 | 424 | 0.74 | 26% | 1.47 | 5% |
| 2->3 | 4867 | 0.94 | 43% | 1.03 | 4% |

### 0->1: parents most weakened on their child's tokens

| parent → child | shared/all | child/parent | parent label | child label |
|---|---|---|---|---|
| 54 → 186 | 0.44 | 0.61 | Medical concepts related to arteries, b… | characters and symbols associated with … |
| 49 → 220 | 0.44 | 0.38 | what appears to be the start of a disjo… | references to figures, tables, and supp… |
| 32 → 493 | 0.45 | 1.09 | numbers within a document | mixed English and Russian text related … |
| 32 → 220 | 0.48 | 0.10 | numbers within a document | references to figures, tables, and supp… |
| 32 → 264 | 0.48 | 0.26 | numbers within a document | code-related tokens, especially within … |
| 85 → 156 | 0.50 | 1.58 | code and jargon related to technology a… | number sequences |
| 32 → 395 | 0.52 | 0.20 | numbers within a document | code syntax with underscores and hyphens |
| 32 → 231 | 0.52 | 0.24 | numbers within a document | capitalized words or acronyms separated… |

### 1->2: parents most weakened on their child's tokens

| parent → child | shared/all | child/parent | parent label | child label |
|---|---|---|---|---|
| 387 → 1065 | 0.28 | 0.98 | contractions with the word "not." | math equations |
| 208 → 1114 | 0.33 | 0.63 | instances of the word "There" | math problems involving sequences, and … |
| 208 → 1170 | 0.33 | 0.43 | instances of the word "There" | occurrences of C# (C Sharp) code |
| 276 → 1065 | 0.34 | 0.98 | US legal and code citations. | math equations |
| 400 → 1065 | 0.34 | 0.99 | mentions of spans of time | math equations |
| 393 → 1065 | 0.35 | 0.77 | words used in electrical engineering co… | math equations |
| 505 → 1114 | 0.35 | 1.43 | language associated with death and prof… | math problems involving sequences, and … |
| 505 → 1170 | 0.35 | 0.98 | language associated with death and prof… | occurrences of C# (C Sharp) code |

### 2->3: parents most weakened on their child's tokens

| parent → child | shared/all | child/parent | parent label | child label |
|---|---|---|---|---|
| 1762 → 3200 | 0.21 | 1.03 | technical specifications for camera len… | sentences that describe the authors of … |
| 1762 → 4616 | 0.21 | 1.04 | technical specifications for camera len… | code pertaining to error messages |
| 1762 → 5445 | 0.21 | 0.90 | technical specifications for camera len… | LaTeX, code snippets, and/or data |
| 1762 → 7270 | 0.21 | 1.76 | technical specifications for camera len… | the letter "A" followed by a colon |
| 1762 → 7843 | 0.21 | 0.95 | technical specifications for camera len… | statistical analysis values, particular… |
| 928 → 2597 | 0.26 | 0.92 | code or text broken up into single char… | the description of baseball plays, spec… |
| 928 → 3113 | 0.26 | 1.07 | code or text broken up into single char… | technical documents about elements |
| 928 → 3411 | 0.26 | 1.05 | code or text broken up into single char… | mathematical expressions, especially th… |
