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
.x0nav .gh{display:inline-flex;align-items:center;gap:5px;}
.x0nav .gh svg{width:15px;height:15px;fill:currentColor;display:block;}
@media (prefers-color-scheme:dark){
.x0nav{background:#141414;border-bottom-color:#2E2E2E;}
.x0nav .row+.row{border-top-color:#242424;}
.x0nav a{color:#A9B4BF;}
.x0nav .brand,.x0nav a:hover,.x0nav .on{color:#C79BF2;}
.x0nav .pill{background:#1E1830;border-color:#3A2B57;}
.x0nav .pill.on{background:#7C22CE;color:#fff;border-color:#7C22CE;}
.x0nav .sep{background:#2E2E2E;}}
</style><nav class="x0nav"><div class="row"><a class="brand" href="./">SOAR I-6 · metrics</a><a class="gh" href="https://github.com/soar-eleuther-i6-hierarchy/metrics" title="Browse the code on GitHub"><svg viewBox="0 0 16 16" aria-hidden="true"><path d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.012 8.012 0 0 0 16 8c0-4.42-3.58-8-8-8z"/></svg>Code</a><span class="sep"></span><a class="" href="./outputs/">Results</a><a class="" href="./outputs/toy_calibration.html">Toy calibration</a><a class="" href="./outputs/trained_toy_calibration.html">Trained toy</a><a class="" href="./outputs/gemma2_2b/">Gemma2_2b</a><a class="" href="./outputs/pcfg/">PCFG</a></div></nav>

# experiment_0: Implement Metrics (SOAR I-6)

Grades candidate parent→child edges between the nested blocks of a **Matryoshka SAE** on
`google/gemma-2-2b` (residual stream, layers 3–24). A family of competing metrics decides which
"edges" are real hierarchy and which are frequency, splitting or co-occurrence artifacts.

**Live site:** [soar-eleuther-i6-hierarchy.github.io/metrics](https://soar-eleuther-i6-hierarchy.github.io/metrics/)

| Where to go | What is there |
| ----------- | ------------- |
| [metrics/](metrics/) | every metric: formula, threshold, what it catches, what it is blind to |
| [outputs/](outputs/) | all results — dashboards, reports, per-layer pages, validation tiers |
| [validation/](validation/) | all three calibration tiers: synthetic toy, trained toy, real SAE |

## Install and run

Run everything from the `experiment_0/` directory.

```bash
pip install torch sae_lens datasets plotly numpy matplotlib

python3 run_pipeline.py         # every stage in order, refusing any whose inputs are missing
python3 run_pipeline.py --list  # show the order and what is already satisfied
python3 run_pipeline.py --from 02   # resume after the slow one
```

### The stages

The order is a dependency chain, not a preference. `run_pipeline.py` encodes it and
refuses to run a stage whose inputs are absent — a stage fed a missing input either
crashes obscurely later or, worse, silently reads a stale file from a previous run.

| | Script | Reads | Writes | Note |
| --- | --- | --- | --- | --- |
| **01** | `collect_statistics.py` | model + SAE + corpus | `exp0_stats.pt`, `token_cache/` | the only stage that touches the model; slow |
| 01b | `fetch_labels.py` | Neuronpedia | `feature_labels.json` | display only, optional |
| **02** | `run_metrics.py` | `exp0_stats.pt` | `metrics_report.{json,md}` | the ten-metric battery |
| 02b | `validation/qualitative_check.py` | 01 + 02 | `qualitative_check.json` | Tier 3 |
| **03** | `run_token_metrics.py` | 02 + `token_cache/` + a decoder | `second_pass.json` | `S_res`, parent-conditioned siblings, kept-children union — model-free |
| 04 | `reporting/visualize.py` | 02 | dashboards | |

Stage 03's decoder is the released gemma SAE unless the run dir holds a `w_dec.pt`, which
the non-gemma adapters write; `--w-dec` overrides both.

Stage 03 needs `token_cache/`, which stage 01 writes only when `CACHE_RESIDUALS=1`.

**Not a stage:** [`in_block_edges.py`](in_block_edges.py) builds its own candidate set from
a *within-block* co-firing matrix instead of filtering stage 02's, so it answers the same
question on a different domain and sits nowhere in this chain. It needs 01, not 02.

**Why the filenames are not numbered.** A module whose name starts with a digit cannot be
imported, and `collect_statistics.collect` is imported by the adapters that grade non-gemma
SAEs. The name says what a stage does; `run_pipeline.py` says when it runs.

Individual stages still run directly:

```bash
python3 collect_statistics.py   # cache every statistic the metrics need (slow, needs model+SAE)
python3 fetch_labels.py         # feature labels for the current layer
python3 run_metrics.py          # metrics_report.{json,md}
python3 run_token_metrics.py    # S_res probes + parent-conditioned siblings (model-free)
python3 -m validation.qualitative_check   # Tier 3: survivor-vs-rejected edges vs Neuronpedia labels
python3 -m reporting.visualize  # rebuild the dashboards
```

`EXP0_LAYER` (default 6) selects the layer; everything writes to `outputs/<source>/layer_NN/`.

```bash
python3 collect_statistics.py --docs 16              # quick smoke slice (16 docs instead of 400)
EXP0_LAYER=12 python3 collect_statistics.py          # any layer 0–24
EXP0_DEVICE=cuda:1 python3 collect_statistics.py     # device (default: mps on Mac, cuda on server)
CUDA_VISIBLE_DEVICES=1 python3 collect_statistics.py # pin one GPU on the shared server
EXP0_OUT=my_run python3 run_metrics.py               # redirect all outputs away from the published outputs/
EXP0_RUN=pcfg/layer_01 python3 -m reporting.visualize # a layer of another source
```

### Grading a source that is not gemma

Stages 02, 03 and 04 take the block structure, the model and the dictionary from the stats
file, so an SAE with a different shape needs an adapter and no metric code — see
[`adapters/from_pcfg.py`](../adapters/from_pcfg.py) in the umbrella repo, which also writes
the `token_cache/` stage 03 reads and the `w_dec.pt` it scores S_res with.

`EXP0_RUN` names the output directory: `EXP0_RUN=pcfg/layer_01` publishes at
`outputs/pcfg/layer_01/`, the same source/layer shape as gemma. Any depth
works — every page derives its own distance to the site root and its link to the shared plotly
bundle from its path — but the LAST component decides how the page reads: a directory named
`layer_NN` gets the layer nav, anything else gets the site-wide one. Give the directory an index
with `python3 -m reporting.layer_index --run`, and the source's own index with
`--source <name>`, or those URLs 404 on GitHub Pages. Register the source in `config.SOURCES` —
its label, its layers and which of the five page kinds it actually has — and it appears in the nav
with its own layer row. Pages generated this way mark no layer as current and their headers name the run's
own model and dictionary. First one published: [`outputs/pcfg/layer_01/`](outputs/pcfg/layer_01/README.md).

Each source is one entry in `NAV_GLOBAL` — a place you go, not a row that follows you. The layer
pills belong to gemma, so they appear only inside `outputs/gemma2_2b/`; before that they sat on
every page, offering to "switch layer" from a PCFG dashboard built on a different model entirely.

Listing a source in `NAV_GLOBAL` makes it reachable from anywhere — and makes every already-generated
page's bar wrong, including ten dashboards whose generator needs that layer's ~700 MB cache.
`python3 -m reporting.refresh_nav` re-renders the bar in place instead, deriving each page's
identity from its own path; `--check` reports without writing. It changes navigation and nothing
else, so it is not a way to freshen stale numbers.

The heavy cache (`exp0_stats.pt`, ~700 MB per layer) is not in git — pull it from the Hub instead of
recomputing:

```bash
hf download soar-eleuther-i6-hierarchy/experiment_0-stats --repo-type dataset --local-dir outputs/
```

More entry points (extra dashboards, in-block edges, the validation tiers) are listed in
[metrics/README.md](metrics/README.md) and [outputs/README.md](outputs/README.md).

## What each metric catches

The validation table: **metrics (rows) × properties (columns)**. A candidate pair can look like an edge for seven different reasons. Only one of them is real
hierarchy; the rest are the pathologies the metrics have to separate out. Each metric is a partial
detector, so the question for every metric is not "is it correct" but **which column does it add**.

| Metric | Parent→child | Absorption | Splitting | Superparent | Siblings | Frequency coincidence | Concept co-occurrence |
| ------ | :----------: | :--------: | :-------: | :---------: | :------: | :-------------------: | :-------------------: |
| **1a** Coverage — reverse `R` | ◐ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ |
| **1b** Coverage — forward `F` | ◐ | ✗ | ✗ | ◐ | ✗ | ✗ | ✗ |
| **1c** Joint-child `R_supp` / `R_mass` / energy share | ◐ | ✗ | ✅ | ✅ | ✗ | ✗ | ✗ |
| **2a** Reconstruction ablation (contribution filter) | ◐ | ✗ | ✗ | ✗ | ✗ | ◐ | ✗ |
| **2b** `S_res` probe, rank-scored | ✅ | ◐ | ✗ | ◐ | ◐ | ✅ | ✗ |
| **3** Sibling redundancy | ◐ | ✗ | ✅ | ✗ | ✅ | ✗ | ✗ |
| **4** Out-degree / superparent | ✗ | ✗ | ✗ | ✅ | ✗ | ◐ | ✗ |
| **5** Token-frequency control | ◐ | ✗ | ✗ | ◐ | ✗ | ✅ | ✗ |
| **6** Independence null (PMI / Dev) | ◐ | ✗ | ✗ | ✅ | ✗ | ✅ | ✗ |
| **7** In-block directed coverage | ✅ | ✗ | ✅ | ✗ | ✅ | ✗ | ✗ |

✅ detects this property · ◐ partial — necessary but not sufficient, or only in some regimes · ✗ blind

**The properties.** *Parent→child*: the child is a genuine refinement of the parent. *Absorption*:
the child has absorbed a case from the parent, so the parent goes **silent** exactly where the child
fires. *Splitting*: the "children" are near-copies of one another. *Superparent*: one parent fans out
over most of the next block and fires on a huge share of tokens. *Siblings*: the two features are
co-level (co-hyponyms or co-extensive duplicates), not parent and child. *Frequency coincidence*: the
co-firing is base rate, carried by high-frequency tokens. *Concept co-occurrence*: both features are
specific and genuinely unrelated (`enzyme`, `CT scan`) but share a latent topic (`biology`), so they
co-fire in the same documents.

**Two columns are still open.**

- **Concept co-occurrence is caught by nothing.** `enzyme` → `CT scan` passes coverage (they co-fire),
  passes reconstruction (both carry mass on biology tokens), passes token-frequency control (biology
  tokens are not frequent), and passes PMI (they are *not* independent — the shared topic makes
  PMI > 0). Closing it needs a **model-based null**: a topic model / LDA-style `M` that removes the
  shared concept and asks whether the residuals `a = enzyme − biology`, `b = CT − biology` are still
  dependent. Not implemented in this tranche.
- **Absorption** is only reachable through decoder geometry (`S_res`), and even there the edge never
  arrives: coverage gates the candidate set, and an absorbed child has low `R` by construction, so
  the pair is dropped before any later metric sees it.

Cells are read off each metric's construction (see the module docstrings) together with the
calibrations below — they state what a metric is *able* to separate, not a measured accuracy on the
real SAE.

### Why trust the table: three tiers

Each tier gives up one guarantee and gains one dose of reality; a metric we trust has to hold across
all three. ("Tier", not "layer", to avoid confusion with the model's residual-stream layers.)

| Tier | What it is | Ground truth? | What it proves |
| ---- | ---------- | ------------- | -------------- |
| **1. Synthetic** | a known 5-parent tree plus injected pathologies, reduced to cached stats | yes, by construction | the maths is right — 9/9, each pathology caught by its intended metric |
| **2. Trained toy** | a Matryoshka SAE actually trained on Bussmann's tree; metrics run on the *learned* features | yes, the tree is known | the metrics survive real training — precision 1.00, recall 0.67, 0 false positives |
| **3. Real SAE** | the production `gemma-2-2b` Matryoshka SAE, read against Neuronpedia labels | no, human judgement | the metrics mean something outside a toy |

Tier 1 is certain but artificial; Tier 3 is real but has no ground truth; Tier 2 is the bridge with
both a trained SAE and a known answer.

**Scope.** Tier 1 scores 9/9 rows across seeds 0–5, covering every statistics-only metric —
coverage, joint-child, reconstruction, sibling redundancy, out-degree, token-frequency control and
the independence null. Still uncalibrated: `S_res` (needs per-token residuals, so it is graded in
Tier 2) and in-block directed coverage, whose matrix rows follow from their construction alone. Full
detail, per-metric scorecards and how to run each tier: **[outputs/README.md](outputs/README.md#how-the-metrics-are-validated-three-tiers)**.

## Headline findings

- **Coverage proposes far more edges than survive.** At layer 6 only 512 of 8,156 candidate B0→B1
  edges (6.3%) improve reconstruction; B2→B3 keeps 113 of 4.7M, with 99.9% frequency-driven.
- **The structure is not a tree.** Multi-parenting is near-total (383 of 383 children at layer 6),
  and feature 15 ("technical documentation language") fires on 99.0% of tokens and parents the
  entire B1 block.
- **Semantic quality degrades with depth.** Survivors read as real refinement early
  (L3/L6: "legal citations" → "legal citations"); at layer 24 the 8 survivors collapse onto 2
  parents, one firing on 41.9% of tokens with unrelated children.
- **The metrics themselves hold up.** 9/9 on the synthetic toy (each injected pathology caught by its
  intended metric) and **precision 1.00 / recall 0.67** on a Matryoshka SAE actually trained on
  Bussmann's tree — every miss traced to a feature the SAE never learned, not to a metric.

> Killing 94% to 99.9% of coverage edges is the result, not a failure: the Matryoshka SAE's hierarchy
> claim does not survive any measurement stricter than raw co-firing.
>
> Caveats: the metrics cover the SAE/MLP slice only, and B3→B4 is off by default (memory;
> enable with `EXP0_B3B4=1`).

Full numbers, dashboards and per-layer pages: **[outputs/](outputs/)**.

## Repo layout

Run the pipeline scripts top-to-bottom; each reads the previous one's output
(see each script's `Needs:` / `Writes:` header).

```
metrics/                              the repository
├── config.py                         every threshold, path and layer-derived constant
│
├── collect_statistics.py             ① stream the corpus once, accumulate every statistic
│                                        (GPU, slow — the only stage that touches the model)
├── fetch_labels.py                   ② bulk autointerp labels for the current layer
├── run_metrics.py                    ③ post-process exp0_stats.pt -> metrics_report.{json,md}
├── run_token_metrics.py              ④ model-free token-cache pass: S_res, sibling redundancy
├── in_block_edges.py                 ⑤ same-level (within-block) edges and duplicates
│
├── metrics/                          one file per metric: pure functions over cached
│   │                                 tensors, no model and no IO
│   ├── coverage.py
│   ├── joint_child.py
│   ├── reconstruction.py
│   ├── sres.py
│   ├── sibling_redundancy.py
│   ├── outdegree.py
│   ├── token_control.py
│   ├── independence_null.py
│   └── README.md
│
├── utils/                            infrastructure
│   ├── sae_utils.py                  model and SAE loaders
│   ├── io.py                         the token cache
│   └── organize_outputs.py           tidies a run directory
│
├── reporting/                        everything that produces something to read
│   ├── visualize.py                  the interactive dashboards
│   ├── make_report_figures.py        static proof-figures
│   ├── layer_index.py                landing pages: per layer, --source, --run
│   └── refresh_nav.py                re-render the nav bar in place, no cache needed
│
├── tests/                            guards on claims the code makes about itself
│   ├── test_collect_generic.py       stage 01 runs on a source that is not gemma
│   └── test_dashboards_generic.py    stages 02-04 grade and render one too
│
├── validation/                       the three calibration tiers + a lateral control
│   ├── toy_world.py                  the synthetic ground-truth world
│   ├── calibrate_on_synthetic_toy.py    Tier 1 — synthetic toy
│   ├── calibrate_on_trained_toy.py   Tier 2 — trained toy
│   ├── qualitative_check.py          Tier 3 — real SAE (also pipeline stage 02b)
│   ├── block_tree_alignment.py       lateral control — blocks vs the declared tree
│   └── README.md
│
├── outputs_archive/                  superseded runs, tracked on purpose (~11 MB)
│   ├── README.md                     what is in here and why v1 was withdrawn
│   ├── layer_NN__v1__*/              the five layers before BOS was excluded
│   └── *__v1__*.html                 two hand-built pages, withdrawn not updated
│
└── outputs/                          the published results (~16 MB)
    ├── assets/
    │   └── plotly.min.js             the one bundle every dashboard links to
    ├── toy_calibration.html          Tier-1 scorecard (+ .json, .md)
    ├── trained_toy_calibration.html  Tier-2 scorecard (+ .json)
    ├── toy_trained/                  the Tier-2 checkpoint
    ├── README.md
    ├── gemma2_2b/                    the five gemma layers, grouped by source
    │   └── layer_06/                 identical in layer_03, layer_12, layer_18, layer_24
    │   ├── README.md                 the layer's landing page
    │   ├── metrics_dashboard.html
    │   ├── superparent_sankey.html
    │   ├── qualitative_dashboard.html
    │   ├── metrics_report.json       (+ .md)
    │   ├── qualitative_check.json    (+ .md)
    │   ├── feature_labels.json
    │   └── npedia_labels_cache.json
    └── pcfg/                         a second source, same shape as gemma's
        ├── README.md                 the source index (layer_index --source pcfg)
        └── layer_01/                 the one layer trained so far
            ├── README.md
            ├── metrics_dashboard.html    the pages a source without Neuronpedia
            ├── superparent_sankey.html     labels can produce
            └── metrics_report.json       (+ .md, + second_pass.json)
```

Two directories exist only when you run the pipeline and are deliberately kept out of git:
`outputs/<source>/layer_NN/exp0_stats.pt` (~700 MB per layer, on the Hub instead) and
`outputs/<source>/layer_NN/token_cache/` (rebuildable). `outputs_local/` is where a scratch run goes,
via `EXP0_OUT`.

The team's other repos are **siblings** of this one, not nested inside it:
`../sae-training/` (Tier-2 training, and `configs/tree.json`), `../PCFG/`,
`../MEETING_NOTES/`.

The SAE has `D_SAE = 32768` features in 5 nested blocks with prefix lengths
`[128, 512, 2048, 8192, 32768]` → `B0=[0,128) B1=[128,512) B2=[512,2048) B3=[2048,8192) B4=[8192,32768)`.
Cross-block edges are computed between **adjacent** blocks only.
