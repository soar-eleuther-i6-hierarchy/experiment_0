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
</style><nav class="x0nav"><div class="row"><a class="brand" href="../../../">SOAR I-6 · metrics</a><a class="" href="../../../outputs/">Results</a><a class="" href="../../../outputs/synthetic_toy_calibration.html">Synthetic Toy Calibration</a><a class="" href="../../../outputs/trained_toy_calibration.html">Trained Toy Calibration</a><a class="" href="../../../outputs/gemma2_2b/">Gemma2_2b</a><a class="on" href="../../../outputs/pcfg/">PCFG</a><a class="gh" href="https://github.com/soar-eleuther-i6-hierarchy/metrics" title="Browse the code on GitHub"><svg viewBox="0 0 16 16" aria-hidden="true"><path d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.012 8.012 0 0 0 16 8c0-4.42-3.58-8-8-8z"/></svg>Code</a></div><div class="row"><span class="lbl">Layer</span><a class="pill on" href="../../../outputs/pcfg/layer_01/in_block_edges.html">1</a><a class="pill" href="../../../outputs/pcfg/layer_03/in_block_edges.html">3</a><span class="sep"></span><span class="lbl">Page</span><a class="" href="../../../outputs/pcfg/layer_01/metrics_dashboard.html">Dashboard</a><a class="" href="../../../outputs/pcfg/layer_01/superparent_sankey.html">Superparents</a><a class="" href="../../../outputs/pcfg/layer_01/in_block_dashboard.html">In-block</a><a class="" href="../../../outputs/pcfg/layer_01/metrics_report.html">Metrics report</a><a class="on" href="../../../outputs/pcfg/layer_01/in_block_edges.html">In-block report</a></div></nav>

# In-block (same-level) directed edges

**Layer 1**　·　PCFG toy 4L d_model=448 / pcfg　·　matryoshka_hook_resid_post_L1　·　1,792 latents in 8 blocks　·　1,016,600 tokens over 3400 docs　·　edge: reverse coverage ≥ 0.5, both endpoints fire ≥ 20

Parent→child *within* a block (asymmetric containment); co-extensive pairs are reported as duplicates (renames/splits), never edges.

## Block B0  (224 features)
- **550** directed edges, **78** duplicate pairs, 550 survive PMI>0; PolyFrac 99%, Gini 0.956.
- **S_res: 15/550** edges are genuine refinements (0 children untestable).
- In-block superparents: 3 (e.g. F153 _feature 153_: 134 children, fires 57%).

| parent → child | R | PMI | parent label | child label |
|---|---|---|---|---|
| 201 → 8 | 1.00 | 1.33 | feature 201 | feature 8 |
| 120 → 8 | 1.00 | 1.41 | feature 120 | feature 8 |
| 13 → 8 | 1.00 | 1.44 | feature 13 | feature 8 |
| 109 → 8 | 0.99 | 1.36 | feature 109 | feature 8 |
| 7 → 8 | 0.99 | 2.91 | feature 7 | feature 8 |
| 31 → 127 | 0.99 | 1.31 | feature 31 | feature 127 |
| 34 → 127 | 0.99 | 1.35 | feature 34 | feature 127 |
| 13 → 127 | 0.99 | 1.43 | feature 13 | feature 127 |

_Duplicate pairs (rename/split candidates):_ 7≈101 (feature 7); 13≈31 (feature 13); 13≈34 (feature 13); 13≈46 (feature 13); 13≈52 (feature 13); 13≈97 (feature 13)

## Block B1  (224 features)
- **0** directed edges, **0** duplicate pairs, 0 survive PMI>0; PolyFrac 0%, Gini 0.000.

## Block B2  (224 features)
- **1** directed edges, **0** duplicate pairs, 1 survive PMI>0; PolyFrac 0%, Gini 0.996.
- **S_res: 0/1** edges are genuine refinements (0 children untestable).

| parent → child | R | PMI | parent label | child label |
|---|---|---|---|---|
| 513 → 453 | 0.67 | 2.40 | feature 513 | feature 453 |

## Block B3  (224 features)
- **1** directed edges, **0** duplicate pairs, 1 survive PMI>0; PolyFrac 0%, Gini 0.996.
- **S_res: 0/1** edges are genuine refinements (0 children untestable).

| parent → child | R | PMI | parent label | child label |
|---|---|---|---|---|
| 708 → 675 | 0.59 | 3.77 | feature 708 | feature 675 |

## Block B4  (224 features)
- **0** directed edges, **0** duplicate pairs, 0 survive PMI>0; PolyFrac 0%, Gini 0.000.

## Block B5  (224 features)
- **0** directed edges, **0** duplicate pairs, 0 survive PMI>0; PolyFrac 0%, Gini 0.000.

## Block B6  (224 features)
- **1** directed edges, **0** duplicate pairs, 1 survive PMI>0; PolyFrac 0%, Gini 0.996.
- **S_res: 1/1** edges are genuine refinements (0 children untestable).

| parent → child | R | PMI | parent label | child label |
|---|---|---|---|---|
| 1407 → 1551 | 0.52 | 4.22 | feature 1407 | feature 1551 |

## Block B7  (224 features)
- **0** directed edges, **0** duplicate pairs, 0 survive PMI>0; PolyFrac 0%, Gini 0.000.
