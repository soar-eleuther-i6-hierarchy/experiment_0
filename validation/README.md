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
</style><nav class="x0nav"><div class="row"><a class="brand" href="../">SOAR I-6 · metrics</a><a class="" href="../outputs/">Results</a><a class="" href="../outputs/toy_calibration.html">Toy calibration</a><a class="" href="../outputs/trained_toy_calibration.html">Trained toy</a><a class="" href="../outputs/pcfg/">PCFG</a></div><div class="row"><span class="lbl">Layer</span><a class="pill" href="../outputs/gemma2_2b/layer_03/">3</a><a class="pill" href="../outputs/gemma2_2b/layer_06/">6</a><a class="pill" href="../outputs/gemma2_2b/layer_12/">12</a><a class="pill" href="../outputs/gemma2_2b/layer_18/">18</a><a class="pill" href="../outputs/gemma2_2b/layer_24/">24</a></div></nav>

# `validation/` — metric calibration

This is **calibration**, not a unit-test suite: how a metric gets *scored* rather than eyeballed.
(The directory was called `tests/`, which promised coverage of `metrics/` and delivered a toy-world
generator. [`tests/`](../tests/) now holds real unit tests.)

All three tiers live here. Tiers 1–2 have a ground-truth tree and run offline; **Tier 3 is the odd
one** — it needs the real `exp0_stats.pt`, needs the network for labels, is judged by human reading
rather than against a known answer, and writes a published artifact into `RUN_DIR`. It sits here
because the three tiers are one argument, not because it shares their dependencies.

Full results and the three-tier table: [outputs/README.md](../outputs/README.md#how-the-metrics-are-validated-three-tiers)

| File | Tier | Runs on | Scored against | What it does |
| ---- | ---- | ------- | -------------- | ------------ |
| [`toy_world.py`](toy_world.py) | 1 | — | — | builds the synthetic world: a known 5-parent tree plus three injected pathologies (superparent, feature-split parent, frequency-coincidence edge), reduced to exactly the statistics the metrics read |
| [`calibrate_on_synthetic_toy.py`](calibrate_on_synthetic_toy.py) | 1 | hand-built statistics | **known tree** | runs every metric on that world and scores it on the job it claims — **9/9 pass across seeds 0–5**, covering 13/13 statistics-only metric functions |
| [`calibrate_on_trained_toy.py`](calibrate_on_trained_toy.py) | 2 | a Matryoshka SAE *actually trained* on that tree | **known tree** | matches learned latents back to true features and scores edge recovery — **precision 1.00, recall 0.67** |
| [`qualitative_check.py`](qualitative_check.py) | 3 | the real `gemma-2-2b` SAE | **nothing — no ground truth** | contrasts survivor vs rejected edges and reads both endpoint labels against Neuronpedia. Also pipeline stage 02b |

**Why Tier 3 is not named `calibrate_*`.** The first two score metrics against an answer we know.
Tier 3 has no such answer: it is judged by reading labels that are themselves model-generated
(Neuronpedia's autointerp). Calling it a calibration would claim a ground truth that does not exist,
so the verb differs on purpose. Its name says *how* it is judged rather than *what it runs on*,
which is the honest emphasis for the one tier where the method of judgement is the caveat.

```bash
python3 validation/calibrate_on_synthetic_toy.py               # Tier 1
PYTHONPATH=src python3 validation/calibrate_on_trained_toy.py  # Tier 2, needs outputs/toy_trained/
python3 -m validation.qualitative_check                        # Tier 3, needs exp0_stats.pt + labels
```

**Both tiers use the same toy** — Bussmann's tree, from `sae-training/configs/tree.json`. What
differs is where the statistics come from: Tier 1 builds them by hand so the tree is exactly right,
Tier 2 reads them off a Matryoshka SAE that had to learn that tree first. The filenames say which.

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

## The lateral control

The three tiers are a ladder along one axis: ground truth traded against realism, all answering
*are the metrics trustworthy?* A control answers a different question — one that has to be closed
before the gemma result means what we say it means.

| File | Question | Result |
| ---- | -------- | ------ |
| [`block_tree_alignment.py`](block_tree_alignment.py) | does the **Matryoshka nesting itself** put a parent in an earlier block than its children? | **6/6 testable edges respected**; mean block 1.7 for parents, 4.5 for children |

```bash
python3 -m validation.block_tree_alignment                       # needs outputs/toy_trained/
python3 -m reporting.visualize --trained-calibration             # renders it onto the Tier-2 page
```

It writes `outputs/block_tree_alignment.json`, which the Tier-2 dashboard picks up if it is there
and skips if it is not — the two are separate questions and separate scripts, but they share a page
because a reader comparing **6/6 respected** against **6/9 recovered** will otherwise assume the two
sixes count different edges. They do not: the three edges the nesting cannot test are exactly the
three whose child the SAE never learned.

The page also states what the 6/6 costs. Three true features were recovered by more than one latent,
and the check takes the **earliest** block for each — the reading most favourable to the
architecture. A later choice could turn a respected edge into a violation. The narrowest respected
edge clears by a single block.

**The objection it closes.** The tiers establish that the metrics are sound; gemma then says the
hierarchy fails. A reader is entitled to reply: *maybe Matryoshka simply cannot produce a coherent
hierarchy, so you have measured a broken architecture rather than discovered anything.* This is the
control for that. On a clean toy with a known tree the nesting produces the **right** ordering, so
the architecture is not structurally incapable — and the production failure is about what the data
distribution does to it, which is exactly what Exp 2 sweeps.

**Why it is answerable here and nowhere else.** Tier 2 indexes by ground truth rather than by block
on purpose: mixing the two would confound "is the metric right?" with "did Matryoshka order the
features right?", and that separation is what lets it report recall 0.67 as the SAE's ceiling rather
than the metrics'. So the second question stayed open. The toy can answer it because it has ten
Matryoshka blocks *and* a known tree. On gemma it cannot be asked at all — the correct ordering is
unknown, so a violation is indistinguishable from a concept we misread.

The 3 untestable edges are the same 3 children the SAE never learned, the same ceiling recall 0.67
reports. Feature splitting does show up: 3 true features are recovered by two latents each.

## What is *not* here

[`../tests/`](../tests/) holds unit tests of the pipeline code — they measure nothing about
hierarchy and score no metric, they guard claims the code makes about itself.
`tests/test_collect_generic.py` is the first: it asserts that `collect_statistics.collect()` still
runs on a source that is not gemma, a property one leftover `config` global in the accumulation loop
would silently break.

The naming follows that split: everything here is `calibrate_*` or a named control, and a `test_`
prefix means a unit test in `../tests/`. Tier 1 used to be `test_metric_calibration.py` — a
calibration wearing a unit-test name, which also invited pytest to collect a file that is not a
pytest test. It is now `calibrate_on_synthetic_toy.py`, parallel to `calibrate_on_trained_toy.py`:
same toy, `synthetic` vs `trained` statistics, ladder legible from the filenames alone.

## Adding a calibration for a new metric

A new metric earns its place by catching a property no existing metric catches
(see the matrix in the [root README](../README.md#what-each-metric-catches)). So the calibration has
two halves: inject that property into `toy_world.py`, then assert the new metric flags it **and**
that it spares the genuine tree edges. A metric that only fires on the pathology is a detector; one
that also kills healthy edges is a filter with a false-positive problem.
