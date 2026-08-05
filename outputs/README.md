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
</style><nav class="x0nav"><div class="row"><a class="brand" href="../">SOAR I-6 · metrics</a><a class="" href="../">Overview</a><a class="on" href="../outputs/">Results</a><a class="" href="../outputs/#per-layer">Per layer</a><a class="" href="../outputs/cross_depth_comparison.html">Cross-depth</a><a class="" href="../outputs/kill_rates.html">Kill rates</a><a class="" href="../outputs/toy_calibration.html">Toy calibration</a><a class="" href="../outputs/trained_toy_calibration.html">Trained toy</a></div><div class="row"><span class="lbl">Layer</span><a class="pill" href="../outputs/layer_03/">3</a><a class="pill" href="../outputs/layer_06/">6</a><a class="pill" href="../outputs/layer_12/">12</a><a class="pill" href="../outputs/layer_18/">18</a><a class="pill" href="../outputs/layer_24/">24</a></div></nav>

# `outputs/` — results

Everything the pipeline produces. Per-layer artifacts live in `layer_NN/`; layer-independent ones
(the toy calibrations, the cross-depth pages) sit directly here.

`assets/plotly.min.js` is the one shared plotly bundle every dashboard links to. Inlining it in each
page instead cost 4.6 MB per file and added ~70 MB of blobs to git on every regeneration; the pages
are ~20 KB now and still open offline. `reporting/visualize.py` writes it automatically if missing.

Link to the `.html` form, not `.md`: GitHub Pages serves `.md` as raw markdown text.

## Across all layers

- [**Cross-depth comparison**](https://soar-eleuther-i6-hierarchy.github.io/metrics/outputs/cross_depth_comparison.html) — the cross-depth story: 4 metric panels, superparent table, the qualitative-agreement collapse.
- [**Kill rates**](https://soar-eleuther-i6-hierarchy.github.io/metrics/outputs/kill_rates.html) — how many edges each metric removes.
- [**Toy calibration scorecard**](https://soar-eleuther-i6-hierarchy.github.io/metrics/outputs/toy_calibration.html) — Tier 1, synthetic ground truth (9/9).
- [**Trained-toy calibration**](https://soar-eleuther-i6-hierarchy.github.io/metrics/outputs/trained_toy_calibration.html) — Tier 2, edge recovery on a Matryoshka SAE trained on Bussmann's tree (precision 1.00, recall 0.67).

## Per layer

Every layer has the same five pages: three interactive dashboards, then the two rendered text
reports behind them. Each layer also has its own landing page — `layer_NN/`, written by
`python3 -m reporting.layer_index` — which is where the nav bar's layer buttons go.

| Layer | Metrics dashboard | Superparent fan-out | Qualitative dashboard | metrics report | qualitative report |
| ----- | ----------------- | ------------------- | --------------------- | -------------- | ------------------ |
| **L3**  | [open](https://soar-eleuther-i6-hierarchy.github.io/metrics/outputs/layer_03/metrics_dashboard.html) | [open](https://soar-eleuther-i6-hierarchy.github.io/metrics/outputs/layer_03/superparent_sankey.html) | [open](https://soar-eleuther-i6-hierarchy.github.io/metrics/outputs/layer_03/qualitative_dashboard.html) | [report](https://soar-eleuther-i6-hierarchy.github.io/metrics/outputs/layer_03/metrics_report.html) | [report](https://soar-eleuther-i6-hierarchy.github.io/metrics/outputs/layer_03/qualitative_check.html) |
| **L6**  | [open](https://soar-eleuther-i6-hierarchy.github.io/metrics/outputs/layer_06/metrics_dashboard.html) | [open](https://soar-eleuther-i6-hierarchy.github.io/metrics/outputs/layer_06/superparent_sankey.html) | [open](https://soar-eleuther-i6-hierarchy.github.io/metrics/outputs/layer_06/qualitative_dashboard.html) | [report](https://soar-eleuther-i6-hierarchy.github.io/metrics/outputs/layer_06/metrics_report.html) | [report](https://soar-eleuther-i6-hierarchy.github.io/metrics/outputs/layer_06/qualitative_check.html) |
| **L12** | [open](https://soar-eleuther-i6-hierarchy.github.io/metrics/outputs/layer_12/metrics_dashboard.html) | [open](https://soar-eleuther-i6-hierarchy.github.io/metrics/outputs/layer_12/superparent_sankey.html) | [open](https://soar-eleuther-i6-hierarchy.github.io/metrics/outputs/layer_12/qualitative_dashboard.html) | [report](https://soar-eleuther-i6-hierarchy.github.io/metrics/outputs/layer_12/metrics_report.html) | [report](https://soar-eleuther-i6-hierarchy.github.io/metrics/outputs/layer_12/qualitative_check.html) |
| **L18** | [open](https://soar-eleuther-i6-hierarchy.github.io/metrics/outputs/layer_18/metrics_dashboard.html) | [open](https://soar-eleuther-i6-hierarchy.github.io/metrics/outputs/layer_18/superparent_sankey.html) | [open](https://soar-eleuther-i6-hierarchy.github.io/metrics/outputs/layer_18/qualitative_dashboard.html) | [report](https://soar-eleuther-i6-hierarchy.github.io/metrics/outputs/layer_18/metrics_report.html) | [report](https://soar-eleuther-i6-hierarchy.github.io/metrics/outputs/layer_18/qualitative_check.html) |
| **L24** | [open](https://soar-eleuther-i6-hierarchy.github.io/metrics/outputs/layer_24/metrics_dashboard.html) | [open](https://soar-eleuther-i6-hierarchy.github.io/metrics/outputs/layer_24/superparent_sankey.html) | [open](https://soar-eleuther-i6-hierarchy.github.io/metrics/outputs/layer_24/qualitative_dashboard.html) | [report](https://soar-eleuther-i6-hierarchy.github.io/metrics/outputs/layer_24/metrics_report.html) | [report](https://soar-eleuther-i6-hierarchy.github.io/metrics/outputs/layer_24/qualitative_check.html) |

## What is in a `layer_NN/` directory

| Artifact | Written by | What it holds |
| -------- | ---------- | ------------- |
| `exp0_stats.pt` | `collect_statistics.py` | ~700 MB of cached statistics: co-firing counts, per-bucket co-firing, per-edge reconstruction sums, within-block sibling co-firing, energy. **Not in git** — see below. |
| `token_cache/` | `collect_statistics.py` | fp16 residuals + sparse latents, so the second pass can train probes without re-running the model. Not in git. |
| `feature_labels.json` | `fetch_labels.py` | all 32768 autointerp descriptions from Neuronpedia's S3 export (~99.9% coverage; ~26 features fall back to `feature <idx>`). |
| `metrics_report.{json,md}` | `run_metrics.py` | per-block-pair summaries and the top edges, annotated with labels. |
| `second_pass.json` | `run_token_metrics.py` | `S_res` verdicts per edge, parent-conditioned sibling redundancy, exact kept-children union. |
| `in_block_edges.{json,md}` | `in_block_edges.py` | same-level directed edges and co-extensive duplicates per block. |
| `qualitative_check.{json,md}` | `validation/qualitative_check.py` | survivor vs rejected edges with both endpoint labels, for human reading. |
| `metrics_dashboard.html`, `superparent_sankey.html`, `qualitative_dashboard.html`, `in_block_dashboard.html` | `reporting/visualize.py` (`--qualitative`, `--in-block`) | the interactive pages linked above. |
| `figures/*.png` | `reporting/make_report_figures.py` | the five static proof-figures, written into each run dir; not tracked in git. |
| `npedia_labels_cache.json` | `validation/qualitative_check.py` | per-feature Neuronpedia API fallback for the handful missing from the bulk export. |

`python3 -m utils.organize_outputs` sorts a run directory into `dashboards/` and `reports/` for browsing;
it leaves the data files the scripts read exactly where they expect them, and is idempotent.

## The big caches are not in git

`outputs/**/*.pt` is gitignored. The ~700 MB-per-layer caches live on the Hub at
[`soar-eleuther-i6-hierarchy/experiment_0-stats`](https://huggingface.co/datasets/soar-eleuther-i6-hierarchy/experiment_0-stats):

```bash
hf download soar-eleuther-i6-hierarchy/experiment_0-stats --repo-type dataset --local-dir outputs/
```

They are loaded with `weights_only=False`. The HTML dashboards **are** tracked and deliberately not
in LFS (`.gitattributes` sets `-filter -diff -merge` on `outputs/**/*.html`), because GitHub Pages
does not resolve LFS objects and would serve the pointer stub instead of the page.

To keep an experimental run away from the published directory, redirect it:
`EXP0_OUT=outputs_local python3 run_metrics.py`.

## How the metrics are validated: three tiers

The same metrics are checked at three tiers of increasing realism. Each tier gives up one guarantee
and gains one dose of reality; a metric we trust has to hold across all three. ("Tier", not "layer",
to avoid confusion with the model's residual-stream layers.)

| Tier | What it is | Ground truth? | What it proves |
| ---- | ---------- | ------------- | -------------- |
| **1. Synthetic** | [`validation/toy_world.py`](../validation/toy_world.py): a known 5-parent tree plus three injected pathologies, reduced to the statistics the metrics read | yes, by construction | the maths is right — 9/9 scorecard rows pass across seeds 0–5, each pathology caught by its intended metric |
| **2. Trained toy** | [`validation/calibrate_on_trained_toy.py`](../validation/calibrate_on_trained_toy.py): a Matryoshka SAE actually trained on Bussmann's tree, metrics run on the *learned* features | yes, the tree is known | the metrics survive a real training run — **precision 1.00, recall 0.67** (6/9 edges, 0 false positives) |
| **3. Real SAE** | [`validation/qualitative_check.py`](../validation/qualitative_check.py) on `gemma-2-2b / NN-res-matryoshka-dc`, read against Neuronpedia labels | no, human judgement stands in | the metrics mean something on a production SAE |

Tier 1 is certain but artificial; Tier 3 is real but has no ground truth; Tier 2 is the bridge that
has both a trained SAE and a known answer.

**What the tiers do not cover.** Tier 1 now scores **9/9 rows across seeds 0–5**, covering all
13 statistics-only metric functions — including the independence null (PMI), joint-child coverage
(support and energy) and energy concentration. Two gaps remain. `S_res` needs per-token
residuals the reduced statistics do not carry, so it is calibrated in Tier 2, not here; **in-block
directed coverage has no known-tree calibration at all**. And because
[`validation/toy_world.py`](../validation/toy_world.py) injects only three pathologies (a
superparent, a feature-split parent, a frequency-coincidence edge), absorption and concept
co-occurrence are still unscored — extending the toy with an absorbed child and a topical confound
is what would let those columns be measured rather than argued from construction.

**Tier 1 detail** — each metric is scored on the job it claims:

| Metric | Job | Result |
| ------ | --- | ------ |
| 1. coverage | recover genuine tree edges | 20/20 kept (plus 28 non-genuine for metrics 2–5 to prune) |
| 2. reconstruction | reject superparent edges, keep genuine | 24/24 superparent edges rejected, 20/20 genuine kept |
| 3. sibling redundancy | flag the feature-split parent, spare healthy ones | split parent 1.00 (flagged), healthy max 0.01 |
| 4. out-degree | identify the superparent, spare genuine parents | detected `[7]`, truth `[7]`; Gini 0.432 |
| 5. frequency control | reject the frequency-coincidence edge | 1/1 rejected (survival 0.0), 20/20 genuine survive |
| 6. independence null (PMI) | rank genuine edges above base-rate co-firing | min genuine PMI 2.59 > max superparent PMI 0.07 |
| 7. joint-child coverage (support) | children cover a genuine parent's firing, not a superparent's | genuine `R_supp` ≥ 1.00 vs superparent 0.43 |
| 8. joint-child coverage (energy) | same, energy-weighted | genuine `R_mass` ≥ 1.00 vs superparent 0.43 |
| 9. energy concentration | flag the feature-split parent (one child holds ≥90% of its energy) | split parent max child share 1.00 (thr 0.9), genuine max 0.27 |

**Tier 2 detail** — on the trained toy the metrics keep **every** edge whose two endpoints the SAE
actually recovered, with **no false positives**. The three missed edges are exactly the three child
features the SAE never learned (it recovered 17 of 20), so 0.67 recall is a limit of the trained SAE,
not of the metrics. The loop:

1. **Rebuild the toy** — Bussmann's compositional tree from the team's
   [`sae-training`](https://github.com/soar-eleuther-i6-hierarchy/sae-training) repo
   (`configs/tree.json`): 3 parents, each with 3 mutually-exclusive children, plus rare features. A
   child only fires when its parent fires — that is the ground-truth hierarchy.
2. **Train a Matryoshka SAE on it** (batch-topk, `k=2`); the checkpoint lands in `outputs/toy_trained/`.
   The SAE sees only activations, never the tree.
3. **Match learned latents to true features** — for each latent, the true feature its decoder points
   at most (cosine ≥ 0.4). This is how we know which latent *is* parent 0.
4. **Run the metrics on the learned latents**, same functions and thresholds as the real pipeline.
5. **Score against the known tree** — precision and recall over true parent→child edges. An edge kept
   on a semantically wrong pair shows up immediately as a false positive.

Steps 3–5 are what separates "the metric failed" from "the SAE failed": a miss only counts against a
metric if the SAE learned both endpoints, which is why per-feature recovery is reported too.

```bash
python3 validation/test_metric_calibration.py                    # Tier 1
python3 -m reporting.visualize --calibration                # Tier 1 dashboard

PYTHONPATH=src python3 validation/calibrate_on_trained_toy.py    # Tier 2 (needs outputs/toy_trained/)
python3 -m reporting.visualize --trained-calibration        # Tier 2 dashboard

python3 -m validation.qualitative_check                     # Tier 3
python3 -m reporting.visualize --qualitative                # Tier 3 dashboard
```

## The finding these outputs support

Reconstruction and frequency control kill the vast majority of coverage edges — B1→B2 at layer 6:
271k candidates, ~0% improve reconstruction, 99.4% frequency-driven — and a handful of high-firing
superparents generate most of the co-occurrence edges. That is the point of the experiment, not a
data error. Semantic agreement with Neuronpedia labels is clean at L3/L6 and collapses by L24.

Static PNG versions of the main claims live in each run dir's `figures/`, rebuilt with
`python3 -m reporting.make_report_figures`.
