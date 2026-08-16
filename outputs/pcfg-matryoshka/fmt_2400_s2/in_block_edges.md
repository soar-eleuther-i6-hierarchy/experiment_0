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
</style><nav class="x0nav"><div class="row"><a class="brand" href="../../../">SOAR I-6 · metrics</a><a class="" href="../../../outputs/">Results</a><a class="" href="../../../outputs/synthetic_toy_calibration.html">Synthetic Toy Calibration</a><a class="" href="../../../outputs/trained_toy_calibration.html">Trained Toy Calibration</a><a class="on" href="../../../outputs/pcfg-matryoshka/">pcfg-matryoshka</a><a class="" href="../../../outputs/gemma-2-2b/">gemma-2-2b</a><a class="gh" href="https://github.com/soar-eleuther-i6-hierarchy/metrics" title="Browse the code on GitHub"><svg viewBox="0 0 16 16" aria-hidden="true"><path d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.012 8.012 0 0 0 16 8c0-4.42-3.58-8-8-8z"/></svg>Code</a></div><div class="row"><span class="lbl">Layer</span><a class="pill" href="../../../outputs/pcfg-matryoshka/layer_00/in_block_edges.html">0</a><a class="pill" href="../../../outputs/pcfg-matryoshka/layer_01/in_block_edges.html">1</a><a class="pill" href="../../../outputs/pcfg-matryoshka/layer_02/in_block_edges.html">2</a><a class="pill" href="../../../outputs/pcfg-matryoshka/layer_03/in_block_edges.html">3</a></div><div class="row"><details class="dens" open><summary class="lbl">Density</summary><div class="drow"><a class="pill" href="../../../outputs/pcfg-matryoshka/fmt_0000_s0/in_block_edges.html">0.00-s0</a><a class="pill" href="../../../outputs/pcfg-matryoshka/fmt_0000_s1/in_block_edges.html">0.00-s1</a><a class="pill" href="../../../outputs/pcfg-matryoshka/fmt_0000_s2/in_block_edges.html">0.00-s2</a><a class="pill" href="../../../outputs/pcfg-matryoshka/fmt_1667_s0/in_block_edges.html">0.17-s0</a><a class="pill" href="../../../outputs/pcfg-matryoshka/fmt_1667_s1/in_block_edges.html">0.17-s1</a><a class="pill" href="../../../outputs/pcfg-matryoshka/fmt_1667_s2/in_block_edges.html">0.17-s2</a><a class="pill" href="../../../outputs/pcfg-matryoshka/fmt_2308_s0/in_block_edges.html">0.23-s0</a><a class="pill" href="../../../outputs/pcfg-matryoshka/fmt_2308_s1/in_block_edges.html">0.23-s1</a><a class="pill" href="../../../outputs/pcfg-matryoshka/fmt_2308_s2/in_block_edges.html">0.23-s2</a><a class="pill" href="../../../outputs/pcfg-matryoshka/fmt_2400_s0/in_block_edges.html">0.24-s0</a><a class="pill" href="../../../outputs/pcfg-matryoshka/fmt_2400_s1/in_block_edges.html">0.24-s1</a><a class="pill on" href="../../../outputs/pcfg-matryoshka/fmt_2400_s2/in_block_edges.html">0.24-s2</a><span class="sep"></span><span class="lbl">Page</span><a class="" href="../../../outputs/pcfg-matryoshka/fmt_2400_s2/metrics_dashboard.html">Dashboard</a><a class="" href="../../../outputs/pcfg-matryoshka/fmt_2400_s2/superparent_sankey.html">Superparents</a><a class="" href="../../../outputs/pcfg-matryoshka/fmt_2400_s2/in_block_dashboard.html">In-block</a><a class="" href="../../../outputs/pcfg-matryoshka/fmt_2400_s2/metrics_report.html">Metrics report</a><a class="on" href="../../../outputs/pcfg-matryoshka/fmt_2400_s2/in_block_edges.html">In-block report</a></div></details></div></nav>

# In-block (same-level) directed edges

**Layer 2**　·　PCFG toy 4L d_model=448 / pcfg　·　matryoshka_hook_resid_post_L2　·　1,792 latents in 8 blocks　·　656,000 tokens over 2000 docs　·　edge: reverse coverage ≥ 0.5, both endpoints fire ≥ 20

Parent→child *within* a block (asymmetric containment); co-extensive pairs are reported as duplicates (renames/splits), never edges.

## Block B0  (224 features)
- **881** directed edges, **49** duplicate pairs, 811 survive PMI>0; PolyFrac 54%, Gini 0.888.
- **S_res: 22/811** edges are genuine refinements (0 children untestable).
- In-block superparents: 2 (e.g. F94 _feature 94_: 186 children, fires 99%).

| parent → child | R | PMI | parent label | child label |
|---|---|---|---|---|
| 94 → 61 | 1.00 | 0.01 | feature 94 | feature 61 |
| 94 → 152 | 1.00 | 0.01 | feature 94 | feature 152 |
| 94 → 115 | 1.00 | 0.01 | feature 94 | feature 115 |
| 40 → 173 | 1.00 | 1.69 | feature 40 | feature 173 |
| 94 → 172 | 1.00 | 0.01 | feature 94 | feature 172 |
| 94 → 105 | 1.00 | 0.01 | feature 94 | feature 105 |
| 94 → 19 | 1.00 | 0.01 | feature 94 | feature 19 |
| 129 → 27 | 1.00 | 1.70 | feature 129 | feature 27 |

_Duplicate pairs (rename/split candidates):_ 8≈184 (feature 8); 17≈46 (feature 17); 17≈59 (feature 17); 17≈178 (feature 17); 17≈212 (feature 17); 26≈46 (feature 26)

## Block B1  (224 features)
- **30** directed edges, **4** duplicate pairs, 30 survive PMI>0; PolyFrac 62%, Gini 0.959.
- **S_res: 3/30** edges are genuine refinements (0 children untestable).

| parent → child | R | PMI | parent label | child label |
|---|---|---|---|---|
| 420 → 292 | 0.99 | 2.25 | feature 420 | feature 292 |
| 403 → 292 | 0.99 | 2.04 | feature 403 | feature 292 |
| 250 → 292 | 0.99 | 2.55 | feature 250 | feature 292 |
| 372 → 292 | 0.99 | 4.51 | feature 372 | feature 292 |
| 277 → 292 | 0.99 | 5.03 | feature 277 | feature 292 |
| 266 → 394 | 0.99 | 4.79 | feature 266 | feature 394 |
| 266 → 231 | 0.98 | 4.78 | feature 266 | feature 231 |
| 273 → 414 | 0.92 | 2.67 | feature 273 | feature 414 |

_Duplicate pairs (rename/split candidates):_ 224≈414 (feature 224); 231≈415 (feature 231); 266≈415 (feature 266); 310≈414 (feature 310)

## Block B2  (224 features)
- **12** directed edges, **0** duplicate pairs, 12 survive PMI>0; PolyFrac 43%, Gini 0.969.
- **S_res: 2/12** edges are genuine refinements (0 children untestable).

| parent → child | R | PMI | parent label | child label |
|---|---|---|---|---|
| 654 → 482 | 0.96 | 3.97 | feature 654 | feature 482 |
| 520 → 617 | 0.83 | 2.65 | feature 520 | feature 617 |
| 520 → 635 | 0.73 | 2.53 | feature 520 | feature 635 |
| 615 → 545 | 0.70 | 1.89 | feature 615 | feature 545 |
| 612 → 545 | 0.68 | 1.99 | feature 612 | feature 545 |
| 521 → 466 | 0.66 | 2.24 | feature 521 | feature 466 |
| 600 → 466 | 0.62 | 2.13 | feature 600 | feature 466 |
| 520 → 609 | 0.60 | 2.33 | feature 520 | feature 609 |

## Block B3  (224 features)
- **1** directed edges, **0** duplicate pairs, 1 survive PMI>0; PolyFrac 0%, Gini 0.996.
- **S_res: 0/1** edges are genuine refinements (0 children untestable).

| parent → child | R | PMI | parent label | child label |
|---|---|---|---|---|
| 707 → 894 | 0.57 | 3.60 | feature 707 | feature 894 |

## Block B4  (224 features)
- **1** directed edges, **0** duplicate pairs, 1 survive PMI>0; PolyFrac 0%, Gini 0.996.
- **S_res: 0/1** edges are genuine refinements (0 children untestable).

| parent → child | R | PMI | parent label | child label |
|---|---|---|---|---|
| 971 → 1022 | 0.85 | 4.64 | feature 971 | feature 1022 |

## Block B5  (224 features)
- **0** directed edges, **0** duplicate pairs, 0 survive PMI>0; PolyFrac 0%, Gini 0.000.

## Block B6  (224 features)
- **0** directed edges, **0** duplicate pairs, 0 survive PMI>0; PolyFrac 0%, Gini 0.000.

## Block B7  (224 features)
- **0** directed edges, **0** duplicate pairs, 0 survive PMI>0; PolyFrac 0%, Gini 0.000.
