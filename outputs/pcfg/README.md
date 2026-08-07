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
</style><nav class="x0nav"><div class="row"><a class="brand" href="../../">SOAR I-6 · metrics</a><a class="" href="../../outputs/">Results</a><a class="" href="../../outputs/toy_calibration.html">Toy calibration</a><a class="" href="../../outputs/trained_toy_calibration.html">Trained toy</a><a class="on" href="../../outputs/pcfg/">PCFG</a></div><div class="row"><span class="lbl">Layer</span><a class="pill" href="../../outputs/layer_03/">3</a><a class="pill" href="../../outputs/layer_06/">6</a><a class="pill" href="../../outputs/layer_12/">12</a><a class="pill" href="../../outputs/layer_18/">18</a><a class="pill" href="../../outputs/layer_24/">24</a></div></nav>

# pcfg

**Layer 1**　·　PCFG toy 4L d_model=448 / pcfg　·　matryoshka_hook_resid_post_L1　·　1,792 latents in 8 blocks　·　1,016,600 tokens over 3400 docs　·　edge: reverse coverage ≥ 0.5, both endpoints fire ≥ 20

The same metric battery as the gemma layers, on an SAE from a different source. Nothing in `metrics/` changed to produce these numbers; the block structure, the model and the dictionary all come from the run's own cached statistics.

| Page | What is on it |
| ---- | ------------- |
| [Dashboard](metrics_dashboard.html) | filter funnel and the per-block-pair distributions |
| [Superparents](superparent_sankey.html) | one superparent's fan-out to its children |
| [Metrics report](metrics_report.html) | the numbers behind the dashboard, as text |

The `exp0_stats.pt` cache and the token cache behind these numbers are not in git -- they are rebuildable from the run directory by `adapters/from_pcfg.py`.
