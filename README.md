# experiment_0: Implement Metrics (SOAR I-6)

Grades candidate parent→child edges between the nested blocks of a **Matryoshka SAE** on
`google/gemma-2-2b` (residual stream, layers 3–24). A family of competing metrics decides which
"edges" are real hierarchy and which are frequency, splitting or co-occurrence artifacts.

**Live site:** [soar-eleuther-i6-hierarchy.github.io/experiment_0](https://soar-eleuther-i6-hierarchy.github.io/experiment_0/)

| Where to go | What is there |
| ----------- | ------------- |
| [metrics/](metrics/) | every metric: formula, threshold, what it catches, what it is blind to |
| [outputs/](outputs/) | all results — dashboards, reports, per-layer pages, validation tiers |
| [validation/](validation/) | Tier 1 + Tier 2 calibration (synthetic toy, trained toy) |

## Install and run

Run everything from the `experiment_0/` directory.

```bash
pip install torch sae_lens datasets plotly numpy matplotlib

python3 cache_stats.py        # Stage 01: cache every statistic the metrics need (slow, needs model+SAE)
python3 fetch_labels.py       # feature labels for the current layer
python3 run_metrics.py        # Stage 02: metrics_report.{json,md}
python3 run_second_pass.py    # Stage 03: S_res probes + parent-conditioned siblings (model-free)
python3 qualitative_check.py  # survivor-vs-rejected edges vs Neuronpedia labels
python3 visualize.py          # rebuild the dashboards
```

`EXP0_LAYER` (default 6) selects the layer; everything writes to `outputs/layer_NN/`.

```bash
python3 cache_stats.py --docs 16              # quick smoke slice (16 docs instead of 400)
EXP0_LAYER=12 python3 cache_stats.py          # any layer 0–24
EXP0_DEVICE=cuda:1 python3 cache_stats.py     # device (default: mps on Mac, cuda on server)
CUDA_VISIBLE_DEVICES=1 python3 cache_stats.py # pin one GPU on the shared server
EXP0_OUT=my_run python3 run_metrics.py        # redirect all outputs away from the published outputs/
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
| **1. Synthetic** | a known 5-parent tree plus injected pathologies, reduced to cached stats | yes, by construction | the maths is right — 5/5, each pathology caught by its intended metric |
| **2. Trained toy** | a Matryoshka SAE actually trained on Bussmann's tree; metrics run on the *learned* features | yes, the tree is known | the metrics survive real training — precision 1.00, recall 0.67, 0 false positives |
| **3. Real SAE** | the production `gemma-2-2b` Matryoshka SAE, read against Neuronpedia labels | no, human judgement | the metrics mean something outside a toy |

Tier 1 is certain but artificial; Tier 3 is real but has no ground truth; Tier 2 is the bridge with
both a trained SAE and a known answer.

**Scope.** Tiers 1–2 calibrate the original five metrics (1a, 2a, 3, 4, 5). The later additions —
`S_res`, the independence null, joint-child and in-block — are theory-backed and their matrix rows
follow from their construction, but no known-tree calibration exists for them yet. Full detail,
per-metric scorecards and how to run each tier: **[outputs/README.md](outputs/README.md#how-the-metrics-are-validated-three-tiers)**.

## Headline findings

- **Coverage proposes far more edges than survive.** At layer 6 only 512 of 8,156 candidate B0→B1
  edges (6.3%) improve reconstruction; B2→B3 keeps 113 of 4.7M, with 99.9% frequency-driven.
- **The structure is not a tree.** Multi-parenting is near-total (383 of 383 children at layer 6),
  and feature 15 ("technical documentation language") fires on 99.0% of tokens and parents the
  entire B1 block.
- **Semantic quality degrades with depth.** Survivors read as real refinement early
  (L3/L6: "legal citations" → "legal citations"); at layer 24 the 8 survivors collapse onto 2
  parents, one firing on 41.9% of tokens with unrelated children.
- **The metrics themselves hold up.** 5/5 on the synthetic toy (each injected pathology caught by its
  intended metric) and **precision 1.00 / recall 0.67** on a Matryoshka SAE actually trained on
  Bussmann's tree — every miss traced to a feature the SAE never learned, not to a metric.

> Killing 94% to 99.9% of coverage edges is the result, not a failure: the Matryoshka SAE's hierarchy
> claim does not survive any measurement stricter than raw co-firing.
>
> Caveats: the metrics cover the SAE/MLP slice only, and B3→B4 is off by default (memory;
> enable with `EXP0_B3B4=1`).

Full numbers, dashboards and per-layer pages: **[outputs/](outputs/)**.

## Repo layout

```
cache_stats.py        Stage 01  stream the corpus once, accumulate every statistic (GPU, slow)
run_metrics.py        Stage 02  pure post-processing over exp0_stats.pt -> metrics_report.{json,md}
run_second_pass.py    Stage 03  model-free token-cache pass: S_res probes, parent-conditioned siblings
in_block_edges.py               same-level (within-block) directed edges and duplicates
qualitative_check.py            survivor vs rejected edges read against Neuronpedia labels
fetch_labels.py                 bulk autointerp labels for the current layer
visualize.py                    all interactive HTML dashboards
make_report_figures.py          the five static PNGs in figures/
organize_outputs.py             tidy a run dir into dashboards/ + reports/
config.py                       every threshold, path and layer-derived constant
sae_utils.py                    the only model/SAE loaders
metrics/                        one file per metric, pure functions over cached tensors
validation/                     Tier 1 (synthetic toy) and Tier 2 (trained toy) calibration
```

The SAE has `D_SAE = 32768` features in 5 nested blocks with prefix lengths
`[128, 512, 2048, 8192, 32768]` → `B0=[0,128) B1=[128,512) B2=[512,2048) B3=[2048,8192) B4=[8192,32768)`.
Cross-block edges are computed between **adjacent** blocks only.
