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
</style><nav class="x0nav"><div class="row"><a class="brand" href="../">SOAR I-6 · metrics</a><a class="" href="../outputs/">Results</a><a class="" href="../outputs/cross_depth_comparison.html">Cross-depth</a><a class="" href="../outputs/kill_rates.html">Kill rates</a><a class="on" href="../outputs/toy_calibration.html">Toy calibration</a><a class="" href="../outputs/trained_toy_calibration.html">Trained toy</a></div><div class="row"><span class="lbl">Layer</span><a class="pill" href="../outputs/layer_03/">3</a><a class="pill" href="../outputs/layer_06/">6</a><a class="pill" href="../outputs/layer_12/">12</a><a class="pill" href="../outputs/layer_18/">18</a><a class="pill" href="../outputs/layer_24/">24</a></div></nav>

# Exp 0 - metric calibration on synthetic ground truth

Each metric is graded on the pathology it is meant to catch, using the production thresholds in `config.py`. Margin = how decisively the metric separated the two classes (higher is better).

| rank | metric | job | verdict | margin | detail |
|---|---|---|---|---|---|
| 1 | 5. frequency control | reject frequency-coincidence edge, keep genuine | PASS | >1000x | 1/1 freq edges rejected (survival=[0.0]); 20/20 genuine survive (min genuine survival=1.00, thr=0.5) |
| 2 | 2. reconstruction | reject superparent edges, keep genuine | PASS | 661.4x | 24/24 superparent edges rejected, 20/20 genuine kept (parent-gain: genuine>=3.05, superparent<=0.0046, thr=0.01) |
| 3 | 3. sibling redundancy | flag feature-split parent, spare healthy | PASS | 92.0x | split parent redundancy=1.00 (flagged); healthy parents max=0.01 (thr=0.5) |
| 4 | 6. independence null (PMI) | rank genuine edges above base-rate/superparent co-firing | PASS | 36.1x | min genuine PMI=2.59 > max superparent PMI=0.07; base-rate confound only - topical co-occurrence is not in this toy (needs a model-based null) |
| 5 | 9. energy concentration (share_energy) | flag feature-split parent (a child holds >=90% of its energy) | PASS | 3.7x | split parent max child share=1.00 (thr=0.9); genuine parents max=0.27 |
| 6 | 8. joint-child coverage (energy) | energy-weighted child coverage separates genuine from superparent | PASS | 2.3x | genuine R_mass>=1.00 vs superparent=0.43 |
| 7 | 7. joint-child coverage (support) | children cover a genuine parent's firing, not a superparent's | PASS | 2.3x | genuine R_supp>=1.00 vs superparent=0.43; r_supp==joint_child_coverage_exact: True; upper>=exact for all parents: True (covers r_supp, joint_child_coverage_exact, joint_child_coverage_upper) |
| 8 | 1. coverage (edge set) | recover genuine tree edges | PASS | 1.0x | 20/20 genuine edges kept; edge set also holds 28 non-genuine (that is what metrics 2-5 must prune) |
| 9 | 4. out-degree / superparent | identify superparent, spare genuine parents | PASS | 1.0x | detected superparents [7] (truth [7]); Gini=0.432, top-1 share=50% |

**9/9 metrics calibrated.** Every scorecard row recovers the genuine tree and rejects its injected pathology on this toy.

These rows cover **13/13 statistics-only metric functions**. The 4 per-token functions (`train_probe`, `sres_rank_check`, `negative_parent_composition`, `parent_conditioned_redundancy`) need per-token residuals/masks from the token cache and are calibrated in Tier 2, not on these reduced statistics.