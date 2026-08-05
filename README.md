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
</style><nav class="x0nav"><div class="row"><a class="brand" href="./">SOAR I-6 · metrics</a><a class="on" href="./">Overview</a><a class="" href="./outputs/">Results</a><a class="" href="./outputs/#per-layer">Per layer</a><a class="" href="./outputs/cross_depth_comparison.html">Cross-depth</a><a class="" href="./outputs/kill_rates.html">Kill rates</a><a class="" href="./outputs/toy_calibration.html">Toy calibration</a><a class="" href="./outputs/trained_toy_calibration.html">Trained toy</a></div><div class="row"><span class="lbl">Layer</span><a class="pill" href="./outputs/layer_03/">3</a><a class="pill" href="./outputs/layer_06/">6</a><a class="pill" href="./outputs/layer_12/">12</a><a class="pill" href="./outputs/layer_18/">18</a><a class="pill" href="./outputs/layer_24/">24</a></div></nav>

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

python3 collect_statistics.py   # cache every statistic the metrics need (slow, needs model+SAE)
python3 fetch_labels.py         # feature labels for the current layer
python3 run_metrics.py          # metrics_report.{json,md}
python3 run_token_metrics.py    # S_res probes + parent-conditioned siblings (model-free)
python3 -m validation.qualitative_check   # Tier 3: survivor-vs-rejected edges vs Neuronpedia labels
python3 -m reporting.visualize  # rebuild the dashboards
```

`EXP0_LAYER` (default 6) selects the layer; everything writes to `outputs/layer_NN/`.

```bash
python3 collect_statistics.py --docs 16              # quick smoke slice (16 docs instead of 400)
EXP0_LAYER=12 python3 collect_statistics.py          # any layer 0–24
EXP0_DEVICE=cuda:1 python3 collect_statistics.py     # device (default: mps on Mac, cuda on server)
CUDA_VISIBLE_DEVICES=1 python3 collect_statistics.py # pin one GPU on the shared server
EXP0_OUT=my_run python3 run_metrics.py               # redirect all outputs away from the published outputs/
```

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
metrics/                          the repo (git root; was experiment_0)
├── config.py                     every threshold, path and layer-derived constant
│
│   ── pipeline, in order ──      each script's Needs:/Writes: header states its inputs
├── collect_statistics.py         stream the corpus once, accumulate every statistic
│                                 (GPU, slow — the only stage that touches the model)
├── fetch_labels.py               bulk autointerp labels for the current layer
├── run_metrics.py                post-process exp0_stats.pt -> metrics_report.{json,md}
├── run_token_metrics.py          model-free token-cache pass: S_res probes,
│                                 parent-conditioned siblings
├── in_block_edges.py             same-level (within-block) directed edges and duplicates
│
├── metrics/                      one file per metric: pure functions over cached
│   │                             tensors, no model and no IO
│   ├── coverage.py               joint_child.py        reconstruction.py
│   ├── sres.py                   sibling_redundancy.py
│   └── outdegree.py              token_control.py      independence_null.py
│
├── utils/                        infrastructure
│   └── sae_utils.py              model/SAE loaders · io.py token cache
│                                 · organize_outputs.py run-dir tidy
│
├── reporting/                    everything that produces something to read
│   ├── visualize.py              the interactive dashboards
│   ├── make_report_figures.py    static proof-figures
│   └── layer_index.py            each layer's landing page
│
├── validation/                   all three calibration tiers
│   ├── toy_world.py              the synthetic ground-truth world
│   ├── test_metric_calibration.py    Tier 1 — synthetic toy
│   ├── calibrate_on_trained_toy.py   Tier 2 — trained toy
│   └── qualitative_check.py          Tier 3 — real SAE (also pipeline stage 02b)
│
├── outputs/                      everything the pipeline produces (~15 MB)
│   ├── assets/plotly.min.js      the one bundle every dashboard links to
│   ├── cross_depth_comparison.html   kill_rates.html
│   ├── toy_calibration.{html,json,md}    trained_toy_calibration.{html,json}
│   ├── toy_trained/              the Tier-2 checkpoint
│   └── layer_{03,06,12,18,24}/
│       ├── README.md             the layer's landing page
│       ├── metrics_dashboard.html    superparent_sankey.html
│       ├── qualitative_dashboard.html
│       ├── metrics_report.{json,md}  qualitative_check.{json,md}
│       ├── feature_labels.json       npedia_labels_cache.json
│       ├── exp0_stats.pt         NOT in git — on the Hub, see below
│       └── token_cache/          NOT in git — rebuildable
│
└── outputs_local/                gitignored: scratch runs, e.g. EXP0_OUT=outputs_local
```

The team's other repos are **siblings** of this one, not nested inside it:
`../sae-training/` (Tier-2 training, and `configs/tree.json`), `../PCFG/`,
`../MEETING_NOTES/`.

The SAE has `D_SAE = 32768` features in 5 nested blocks with prefix lengths
`[128, 512, 2048, 8192, 32768]` → `B0=[0,128) B1=[128,512) B2=[512,2048) B3=[2048,8192) B4=[8192,32768)`.
Cross-block edges are computed between **adjacent** blocks only.
