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
@media (prefers-color-scheme:dark){
.x0nav{background:#141414;border-bottom-color:#2E2E2E;}
.x0nav .row+.row{border-top-color:#242424;}
.x0nav a{color:#A9B4BF;}
.x0nav .brand,.x0nav a:hover,.x0nav .on{color:#C79BF2;}
.x0nav .pill{background:#1E1830;border-color:#3A2B57;}
.x0nav .pill.on{background:#7C22CE;color:#fff;border-color:#7C22CE;}
.x0nav .sep{background:#2E2E2E;}}
</style><nav class="x0nav"><div class="row"><a class="brand" href="../../">SOAR I-6 · metrics</a><a class="" href="../../outputs/">Results</a><a class="" href="../../outputs/toy_calibration.html">Toy calibration</a><a class="" href="../../outputs/trained_toy_calibration.html">Trained toy</a><a class="" href="../../outputs/pcfg/">PCFG</a></div><div class="row"><span class="lbl">Layer</span><a class="pill" href="../../outputs/gemma2_2b/layer_03/">3</a><a class="pill" href="../../outputs/gemma2_2b/layer_06/">6</a><a class="pill" href="../../outputs/gemma2_2b/layer_12/">12</a><a class="pill" href="../../outputs/gemma2_2b/layer_18/">18</a><a class="pill" href="../../outputs/gemma2_2b/layer_24/">24</a></div></nav>

# `gemma2_2b/` — google/gemma-2-2b

The five residual-stream layers of `google/gemma-2-2b` graded against its released Matryoshka SAE (32,768 latents in 5 nested blocks [128, 512, 2048, 8192, 32768]). One directory per layer, five pages each; the bar above moves between them.

| Layer | SAE | Pages |
| --- | --- | --- |
| [**3**](layer_03/) | `blocks.3.hook_resid_post` | [dashboard](layer_03/metrics_dashboard.html), [superparents](layer_03/superparent_sankey.html), [qualitative](layer_03/qualitative_dashboard.html), [metrics report](layer_03/metrics_report.html), [qualitative report](layer_03/qualitative_check.html) |
| [**6**](layer_06/) | `blocks.6.hook_resid_post` | [dashboard](layer_06/metrics_dashboard.html), [superparents](layer_06/superparent_sankey.html), [qualitative](layer_06/qualitative_dashboard.html), [metrics report](layer_06/metrics_report.html), [qualitative report](layer_06/qualitative_check.html) |
| [**12**](layer_12/) | `blocks.12.hook_resid_post` | [dashboard](layer_12/metrics_dashboard.html), [superparents](layer_12/superparent_sankey.html), [qualitative](layer_12/qualitative_dashboard.html), [metrics report](layer_12/metrics_report.html), [qualitative report](layer_12/qualitative_check.html) |
| [**18**](layer_18/) | `blocks.18.hook_resid_post` | [dashboard](layer_18/metrics_dashboard.html), [superparents](layer_18/superparent_sankey.html), [qualitative](layer_18/qualitative_dashboard.html), [metrics report](layer_18/metrics_report.html), [qualitative report](layer_18/qualitative_check.html) |
| [**24**](layer_24/) | `blocks.24.hook_resid_post` | [dashboard](layer_24/metrics_dashboard.html), [superparents](layer_24/superparent_sankey.html), [qualitative](layer_24/qualitative_dashboard.html), [metrics report](layer_24/metrics_report.html), [qualitative report](layer_24/qualitative_check.html) |

These sat at `outputs/layer_NN/` until 7 August. They moved when a second source was published beside them: with only gemma here, `layer_06` read as a global fact rather than a fact about one model. Other sources are listed in [outputs/README.md](../README.md).

Stage 03 (`run_token_metrics.py`) has run on layer 6 only — see [outputs/README.md](../README.md#the-second-pass-has-run-on-layer-6-only) before reading the sibling-redundancy figure on the other four.
