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
.x0nav details.dens{display:flex;flex-wrap:wrap;align-items:center;gap:13px;}
.x0nav details.dens summary{cursor:pointer;list-style:none;user-select:none;}
.x0nav details.dens summary::-webkit-details-marker{display:none;}
.x0nav details.dens summary::after{content:"▸";margin-left:4px;}
.x0nav details.dens[open] summary::after{content:"▾";}
.x0nav details.dens summary:hover{color:#7C22CE;}
@media (prefers-color-scheme:dark){
.x0nav{background:#141414;border-bottom-color:#2E2E2E;}
.x0nav .row+.row{border-top-color:#242424;}
.x0nav a{color:#A9B4BF;}
.x0nav .brand,.x0nav a:hover,.x0nav .on{color:#C79BF2;}
.x0nav .pill{background:#1E1830;border-color:#3A2B57;}
.x0nav .pill.on{background:#7C22CE;color:#fff;border-color:#7C22CE;}
.x0nav .sep{background:#2E2E2E;}
.x0nav details.dens summary:hover{color:#C79BF2;}}
</style><nav class="x0nav"><div class="row"><a class="brand" href="../../">SOAR I-6 · metrics</a><a class="" href="../../outputs/">Results</a><a class="" href="../../outputs/synthetic_toy_calibration.html">Synthetic Toy Calibration</a><a class="" href="../../outputs/trained_toy_calibration.html">Trained Toy Calibration</a><a class="on" href="../../outputs/pcfg-matryoshka/">pcfg-matryoshka</a><a class="" href="../../outputs/gemma-2-2b/">gemma-2-2b</a><a class="gh" href="https://github.com/soar-eleuther-i6-hierarchy/metrics" title="Browse the code on GitHub"><svg viewBox="0 0 16 16" aria-hidden="true"><path d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.012 8.012 0 0 0 16 8c0-4.42-3.58-8-8-8z"/></svg>Code</a></div><div class="row"><span class="lbl">Layer</span><a class="pill" href="../../outputs/pcfg-matryoshka/layer_00/">0</a><a class="pill" href="../../outputs/pcfg-matryoshka/layer_01/">1</a><a class="pill" href="../../outputs/pcfg-matryoshka/layer_02/">2</a><a class="pill" href="../../outputs/pcfg-matryoshka/layer_03/">3</a></div><div class="row"><details class="dens"><summary class="lbl">Density</summary><a class="pill" href="../../outputs/pcfg-matryoshka/fmt_0000_s0/">0.00-s0</a><a class="pill" href="../../outputs/pcfg-matryoshka/fmt_0000_s1/">0.00-s1</a><a class="pill" href="../../outputs/pcfg-matryoshka/fmt_0000_s2/">0.00-s2</a><a class="pill" href="../../outputs/pcfg-matryoshka/fmt_1667_s0/">0.17-s0</a><a class="pill" href="../../outputs/pcfg-matryoshka/fmt_1667_s1/">0.17-s1</a><a class="pill" href="../../outputs/pcfg-matryoshka/fmt_1667_s2/">0.17-s2</a><a class="pill" href="../../outputs/pcfg-matryoshka/fmt_2308_s0/">0.23-s0</a><a class="pill" href="../../outputs/pcfg-matryoshka/fmt_2308_s1/">0.23-s1</a><a class="pill" href="../../outputs/pcfg-matryoshka/fmt_2308_s2/">0.23-s2</a><a class="pill" href="../../outputs/pcfg-matryoshka/fmt_2400_s0/">0.24-s0</a><a class="pill" href="../../outputs/pcfg-matryoshka/fmt_2400_s1/">0.24-s1</a><a class="pill" href="../../outputs/pcfg-matryoshka/fmt_2400_s2/">0.24-s2</a></details></div></nav>

# `pcfg-matryoshka/` — pcfg-matryoshka

**Layer 0**　·　PCFG toy 4L d_model=448 / pcfg　·　matryoshka_hook_resid_post_L0　·　1,792 latents in 8 blocks　·　1,016,600 tokens over 3400 docs　·　edge: reverse coverage ≥ 0.5, both endpoints fire ≥ 20

One directory per layer, graded by the same battery as every other source in [outputs/](../README.md). The bar above moves between the layers; nothing in `metrics/` differs between them.

| Layer | Pages |
| --- | --- |
| [**0**](layer_00/) | [dashboard](layer_00/metrics_dashboard.html), [superparents](layer_00/superparent_sankey.html), [in-block](layer_00/in_block_dashboard.html), [metrics report](layer_00/metrics_report.html), [in-block report](layer_00/in_block_edges.html) |
| [**1**](layer_01/) | [dashboard](layer_01/metrics_dashboard.html), [superparents](layer_01/superparent_sankey.html), [in-block](layer_01/in_block_dashboard.html), [metrics report](layer_01/metrics_report.html), [in-block report](layer_01/in_block_edges.html) |
| [**2**](layer_02/) | [dashboard](layer_02/metrics_dashboard.html), [in-block](layer_02/in_block_dashboard.html), [metrics report](layer_02/metrics_report.html), [in-block report](layer_02/in_block_edges.html) |
| [**3**](layer_03/) | [dashboard](layer_03/metrics_dashboard.html), [superparents](layer_03/superparent_sankey.html), [in-block](layer_03/in_block_dashboard.html), [metrics report](layer_03/metrics_report.html), [in-block report](layer_03/in_block_edges.html) |

## Formatting sweep

Exp 2, axis (b): four formatting densities × three seeds, each its own run — a separate base model and SAE, all trained and graded at layer 2. The density is the fraction of delimiter tokens the grammar emits; the seed row with the bare hash on the compute node is s0.

| Run | Pages |
| --- | --- |
| [**0.00-s0**](fmt_0000_s0/) | [dashboard](fmt_0000_s0/metrics_dashboard.html), [in-block](fmt_0000_s0/in_block_dashboard.html), [metrics report](fmt_0000_s0/metrics_report.html), [in-block report](fmt_0000_s0/in_block_edges.html) |
| [**0.00-s1**](fmt_0000_s1/) | [dashboard](fmt_0000_s1/metrics_dashboard.html), [in-block](fmt_0000_s1/in_block_dashboard.html), [metrics report](fmt_0000_s1/metrics_report.html), [in-block report](fmt_0000_s1/in_block_edges.html) |
| [**0.00-s2**](fmt_0000_s2/) | [dashboard](fmt_0000_s2/metrics_dashboard.html), [superparents](fmt_0000_s2/superparent_sankey.html), [in-block](fmt_0000_s2/in_block_dashboard.html), [metrics report](fmt_0000_s2/metrics_report.html), [in-block report](fmt_0000_s2/in_block_edges.html) |
| [**0.17-s0**](fmt_1667_s0/) | [dashboard](fmt_1667_s0/metrics_dashboard.html), [superparents](fmt_1667_s0/superparent_sankey.html), [in-block](fmt_1667_s0/in_block_dashboard.html), [metrics report](fmt_1667_s0/metrics_report.html), [in-block report](fmt_1667_s0/in_block_edges.html) |
| [**0.17-s1**](fmt_1667_s1/) | [dashboard](fmt_1667_s1/metrics_dashboard.html), [in-block](fmt_1667_s1/in_block_dashboard.html), [metrics report](fmt_1667_s1/metrics_report.html), [in-block report](fmt_1667_s1/in_block_edges.html) |
| [**0.17-s2**](fmt_1667_s2/) | [dashboard](fmt_1667_s2/metrics_dashboard.html), [superparents](fmt_1667_s2/superparent_sankey.html), [in-block](fmt_1667_s2/in_block_dashboard.html), [metrics report](fmt_1667_s2/metrics_report.html), [in-block report](fmt_1667_s2/in_block_edges.html) |
| [**0.23-s0**](fmt_2308_s0/) | [dashboard](fmt_2308_s0/metrics_dashboard.html), [in-block](fmt_2308_s0/in_block_dashboard.html), [metrics report](fmt_2308_s0/metrics_report.html), [in-block report](fmt_2308_s0/in_block_edges.html) |
| [**0.23-s1**](fmt_2308_s1/) | [dashboard](fmt_2308_s1/metrics_dashboard.html), [in-block](fmt_2308_s1/in_block_dashboard.html), [metrics report](fmt_2308_s1/metrics_report.html), [in-block report](fmt_2308_s1/in_block_edges.html) |
| [**0.23-s2**](fmt_2308_s2/) | [dashboard](fmt_2308_s2/metrics_dashboard.html), [in-block](fmt_2308_s2/in_block_dashboard.html), [metrics report](fmt_2308_s2/metrics_report.html), [in-block report](fmt_2308_s2/in_block_edges.html) |
| [**0.24-s0**](fmt_2400_s0/) | [dashboard](fmt_2400_s0/metrics_dashboard.html), [superparents](fmt_2400_s0/superparent_sankey.html), [in-block](fmt_2400_s0/in_block_dashboard.html), [metrics report](fmt_2400_s0/metrics_report.html), [in-block report](fmt_2400_s0/in_block_edges.html) |
| [**0.24-s1**](fmt_2400_s1/) | [dashboard](fmt_2400_s1/metrics_dashboard.html), [in-block](fmt_2400_s1/in_block_dashboard.html), [metrics report](fmt_2400_s1/metrics_report.html), [in-block report](fmt_2400_s1/in_block_edges.html) |
| [**0.24-s2**](fmt_2400_s2/) | [dashboard](fmt_2400_s2/metrics_dashboard.html), [superparents](fmt_2400_s2/superparent_sankey.html), [in-block](fmt_2400_s2/in_block_dashboard.html), [metrics report](fmt_2400_s2/metrics_report.html), [in-block report](fmt_2400_s2/in_block_edges.html) |
