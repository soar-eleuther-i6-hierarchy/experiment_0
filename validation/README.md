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
</style><nav class="x0nav"><div class="row"><a class="brand" href="../">SOAR I-6 · metrics</a><a class="" href="../outputs/">Results</a><a class="" href="../outputs/cross_depth_comparison.html">Cross-depth</a><a class="" href="../outputs/kill_rates.html">Kill rates</a><a class="" href="../outputs/toy_calibration.html">Toy calibration</a><a class="" href="../outputs/trained_toy_calibration.html">Trained toy</a></div><div class="row"><span class="lbl">Layer</span><a class="pill" href="../outputs/layer_03/">3</a><a class="pill" href="../outputs/layer_06/">6</a><a class="pill" href="../outputs/layer_12/">12</a><a class="pill" href="../outputs/layer_18/">18</a><a class="pill" href="../outputs/layer_24/">24</a></div></nav>

# `validation/` — metric calibration

This is **calibration**, not a unit-test suite: how a metric gets *scored* rather than eyeballed.
(The directory was called `tests/`, which promised coverage of `metrics/` and delivered a toy-world
generator; `tests/` is now free for real unit tests.)

All three tiers live here. Tiers 1–2 have a ground-truth tree and run offline; **Tier 3 is the odd
one** — it needs the real `exp0_stats.pt`, needs the network for labels, is judged by human reading
rather than against a known answer, and writes a published artifact into `RUN_DIR`. It sits here
because the three tiers are one argument, not because it shares their dependencies.

Full results and the three-tier table: [outputs/README.md](../outputs/README.md#how-the-metrics-are-validated-three-tiers)

| File | Tier | What it does |
| ---- | ---- | ------------ |
| [`toy_world.py`](toy_world.py) | 1 | builds a synthetic world: a known 5-parent tree plus three injected pathologies (superparent, feature-split parent, frequency-coincidence edge), reduced to exactly the statistics the metrics read |
| [`test_metric_calibration.py`](test_metric_calibration.py) | 1 | runs every metric on that world and scores it on the job it claims — **9/9 pass across seeds 0–5**, covering 13/13 statistics-only metric functions |
| [`calibrate_on_trained_toy.py`](calibrate_on_trained_toy.py) | 2 | runs the metrics on a Matryoshka SAE *actually trained* on Bussmann's tree, matches learned latents back to true features, and scores edge recovery — **precision 1.00, recall 0.67** |
| [`qualitative_check.py`](qualitative_check.py) | 3 | on the real `gemma-2-2b` SAE: contrasts survivor vs rejected edges and reads both endpoint labels against Neuronpedia. Also pipeline stage 02b |

```bash
python3 validation/test_metric_calibration.py                    # Tier 1
PYTHONPATH=src python3 validation/calibrate_on_trained_toy.py    # Tier 2, needs outputs/toy_trained/
python3 -m validation.qualitative_check                          # Tier 3, needs exp0_stats.pt + labels
```

## Two checks that are not tiers

The ladder scores the *metrics*. These two ask different questions, and neither belongs on it.

| File | Asks | Result |
| ---- | ---- | ------ |
| [`block_tree_alignment.py`](block_tree_alignment.py) | does the **Matryoshka nesting itself** put a parent in an earlier block than its children? | **6/6 testable edges respected**; mean block 1.7 for parents, 4.5 for children |
| [`test_collect_generic.py`](test_collect_generic.py) | does `collect_statistics.collect()` run on a source that is not gemma? | passes on a 28-feature stub in 3 blocks; no network, no GPU, ~1s |

```bash
python3 -m validation.block_tree_alignment       # needs outputs/toy_trained/
python3 -m validation.test_collect_generic       # needs nothing
```

**Why the alignment check is worth having.** Tier 2 indexes by ground truth rather than by block on
purpose — mixing the two would confound "is the metric right?" with "did Matryoshka order the
features right?", and that separation is what lets it report recall 0.67 as the SAE's ceiling rather
than the metrics'. But nobody then asked the second question, and here it is answerable: the toy has
ten Matryoshka blocks and a known tree. On gemma it is not, because the correct ordering is unknown
and a violation cannot be told apart from a concept we misread.

The answer matters for how the gemma results are read. The nesting is **not** structurally incapable
of producing a hierarchy — on clean ground truth it produces the right one — so the production
failure is about what the distribution does to it, which is what Exp 2 sweeps. The 3 untestable
edges are the same 3 children the SAE never learned. Feature splitting does show up: 3 true features
are recovered by two latents each.

**Why the generic check is worth having.** `collect()` was split out of `main()` so an adapter can
feed it a PCFG transformer or a trained toy instead of gemma. That claim is cheap to make and easy to
break — one `config.py` global left in the accumulation loop silently reintroduces gemma's block
boundaries. The stub run is what keeps it honest. It also runs the umbrella's
`contracts/validate_stats.py` against its own output when that repo is checked out beside this one.

Tier 2 needs a checkpoint in `outputs/toy_trained/`, trained via `sae-training/scripts/train_toy.py`
from the team's [`sae-training`](https://github.com/soar-eleuther-i6-hierarchy/sae-training) repo. It
also reads that repo's `configs/tree.json` for the ground-truth tree, and expects the clone **beside**
`experiment_0/` (`../sae-training/`); set `EXP0_SAE_TRAINING` if yours lives elsewhere.

Both tiers write into [`outputs/`](../outputs/) (`toy_calibration.json`,
`trained_toy_calibration.json`) and have a dashboard: `python3 -m reporting.visualize --calibration`
and `python3 -m reporting.visualize --trained-calibration`.

## Why both

Tier 1 is certain but artificial — it proves the arithmetic is right, and nothing about whether an
SAE would ever learn such a structure. Tier 2 closes exactly that gap: the toy passes through a real
training run first, so only what the SAE actually learned reaches the metrics. That is also what lets
it attribute a miss: a missed edge counts against a metric only if the SAE learned both endpoints
(it recovered 17 of 20 features, and all three misses trace to the three it did not).

## Adding a calibration for a new metric

A new metric earns its place by catching a property no existing metric catches
(see the matrix in the [root README](../README.md#what-each-metric-catches)). So the calibration has
two halves: inject that property into `toy_world.py`, then assert the new metric flags it **and**
that it spares the genuine tree edges. A metric that only fires on the pathology is a detector; one
that also kills healthy edges is a filter with a false-positive problem.
