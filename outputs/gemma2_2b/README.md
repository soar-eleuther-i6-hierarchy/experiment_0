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
</style><nav class="x0nav"><div class="row"><a class="brand" href="../../">SOAR I-6 · metrics</a><a class="" href="../../outputs/">Results</a><a class="" href="../../outputs/toy_calibration.html">Synthetic Toy Calibration</a><a class="" href="../../outputs/trained_toy_calibration.html">Trained Toy Calibration</a><a class="on" href="../../outputs/gemma2_2b/">Gemma2_2b</a><a class="" href="../../outputs/pcfg/">PCFG</a><a class="gh" href="https://github.com/soar-eleuther-i6-hierarchy/metrics" title="Browse the code on GitHub"><svg viewBox="0 0 16 16" aria-hidden="true"><path d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.012 8.012 0 0 0 16 8c0-4.42-3.58-8-8-8z"/></svg>Code</a></div><div class="row"><span class="lbl">Layer</span><a class="pill" href="../../outputs/gemma2_2b/layer_03/">3</a><a class="pill" href="../../outputs/gemma2_2b/layer_06/">6</a><a class="pill" href="../../outputs/gemma2_2b/layer_12/">12</a><a class="pill" href="../../outputs/gemma2_2b/layer_18/">18</a><a class="pill" href="../../outputs/gemma2_2b/layer_24/">24</a></div></nav>

# `gemma2_2b/` — Gemma2_2b

**Layer 3**　·　gemma-2-2b / 3-res-matryoshka-dc　·　blocks.3.hook_resid_post　·　48,571 tokens over 400 docs　·　edge: reverse coverage ≥ 0.5, both endpoints fire ≥ 20

One directory per layer, graded by the same battery as every other source in [outputs/](../README.md). The bar above moves between the layers; nothing in `metrics/` differs between them.

| Layer | Pages |
| --- | --- |
| [**3**](layer_03/) | [dashboard](layer_03/metrics_dashboard.html), [superparents](layer_03/superparent_sankey.html), [in-block](layer_03/in_block_dashboard.html), [qualitative](layer_03/qualitative_dashboard.html), [metrics report](layer_03/metrics_report.html), [qualitative report](layer_03/qualitative_check.html) |
| [**6**](layer_06/) | [dashboard](layer_06/metrics_dashboard.html), [superparents](layer_06/superparent_sankey.html), [in-block](layer_06/in_block_dashboard.html), [qualitative](layer_06/qualitative_dashboard.html), [metrics report](layer_06/metrics_report.html), [qualitative report](layer_06/qualitative_check.html) |
| [**12**](layer_12/) | [dashboard](layer_12/metrics_dashboard.html), [superparents](layer_12/superparent_sankey.html), [in-block](layer_12/in_block_dashboard.html), [qualitative](layer_12/qualitative_dashboard.html), [metrics report](layer_12/metrics_report.html), [qualitative report](layer_12/qualitative_check.html) |
| [**18**](layer_18/) | [dashboard](layer_18/metrics_dashboard.html), [superparents](layer_18/superparent_sankey.html), [in-block](layer_18/in_block_dashboard.html), [qualitative](layer_18/qualitative_dashboard.html), [metrics report](layer_18/metrics_report.html), [qualitative report](layer_18/qualitative_check.html) |
| [**24**](layer_24/) | [dashboard](layer_24/metrics_dashboard.html), [superparents](layer_24/superparent_sankey.html), [in-block](layer_24/in_block_dashboard.html), [qualitative](layer_24/qualitative_dashboard.html), [metrics report](layer_24/metrics_report.html), [qualitative report](layer_24/qualitative_check.html) |

These sat at `outputs/layer_NN/` until 7 August. They moved when a second source was published beside them: with only gemma here, `layer_06` read as a global fact rather than a fact about one model.

Stage 03 (`run_token_metrics.py`) has run on layer 6 only — see [outputs/README.md](../README.md#the-second-pass-has-run-on-layer-6-only) before reading the sibling-redundancy figure on the other four.
