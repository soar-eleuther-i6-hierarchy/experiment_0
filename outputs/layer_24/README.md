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
</style><nav class="x0nav"><div class="row"><a class="brand" href="../../">SOAR I-6 · metrics</a><a class="" href="../../outputs/">Results</a><a class="" href="../../outputs/cross_depth_comparison.html">Cross-depth</a><a class="" href="../../outputs/kill_rates.html">Kill rates</a><a class="" href="../../outputs/toy_calibration.html">Toy calibration</a><a class="" href="../../outputs/trained_toy_calibration.html">Trained toy</a></div><div class="row"><span class="lbl">Layer</span><a class="pill" href="../../outputs/layer_03/">3</a><a class="pill" href="../../outputs/layer_06/">6</a><a class="pill" href="../../outputs/layer_12/">12</a><a class="pill" href="../../outputs/layer_18/">18</a><a class="pill on" href="../../outputs/layer_24/">24</a><span class="sep"></span><span class="lbl">Page</span><a class="" href="../../outputs/layer_24/metrics_dashboard.html">Dashboard</a><a class="" href="../../outputs/layer_24/superparent_sankey.html">Superparents</a><a class="" href="../../outputs/layer_24/qualitative_dashboard.html">Qualitative</a><a class="" href="../../outputs/layer_24/metrics_report.html">Metrics report</a><a class="" href="../../outputs/layer_24/qualitative_check.html">Qualitative report</a></div></nav>

# Layer 24

The five pages for layer 24 of `google/gemma-2-2b`'s residual stream, graded by the metrics in [`metrics/`](../../metrics/README.md). Use the bar above to move between layers while staying on the same page.

| Page | What is on it |
| ---- | ------------- |
| [Dashboard](metrics_dashboard.html) | filter funnel and the per-block-pair distributions |
| [Superparents](superparent_sankey.html) | one superparent's fan-out to its children |
| [Qualitative](qualitative_dashboard.html) | surviving vs rejected edges, with Neuronpedia labels |
| [Metrics report](metrics_report.html) | the numbers behind the dashboard, as text |
| [Qualitative report](qualitative_check.html) | survivor vs rejected edges read against the labels |

Both reports are also in the repo as `.md`; the `.html` links above are what GitHub Pages renders. The `exp0_stats.pt` cache behind these numbers is not in git -- see [outputs/README.md](../README.md#the-big-caches-are-not-in-git).
