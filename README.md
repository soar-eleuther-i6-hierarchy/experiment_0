# experiment_0: Implement Metrics (SOAR I-6)

Grades candidate parent→child edges between the nested blocks of a **Matryoshka SAE** on
`google/gemma-2-2b` (residual stream, layers 3 to 24). Five competing metrics (coverage,
reconstruction, sibling redundancy, out-degree, and token-frequency control) decide which
"edges" are real hierarchy and which are frequency / co-occurrence artifacts.

Each metric is validated twice: against a **synthetic toy with known ground truth** (5/5 pass,
every injected pathology caught by its intended metric) and against **Neuronpedia labels on the
real SAE**. Headline finding: that qualitative agreement is clean early but **degrades with depth**,
and at layer 24 the surviving edges collapse onto a single feature firing on 41.9% of tokens.

**Live site:** [https://soar-eleuther-i6-hierarchy.github.io/experiment_0/](https://soar-eleuther-i6-hierarchy.github.io/experiment_0/)

## Key results

**Coverage proposes far more edges than survive.** At layer 6, only 512 of 8,156 candidate
B0→B1 edges (6.3%) improve reconstruction; the rest are co-firing with nothing behind it.

**Deeper block pairs carry no hierarchy signal at all.** B2→B3 has 4.7M candidate edges, of
which 113 improve reconstruction, and 99.9% are frequency-driven (mean survival 0.015, so the
edges vanish on rare tokens).

**The structure is not a tree.** Multi-parenting is near-total (383 of 383 children have
multiple parents at layer 6), and the busiest parent covers the entire child block: feature 15
("technical documentation language") fires on 99.0% of tokens and parents all 383 B1 features.

**Semantic quality degrades with network depth.** Survivor edges read as genuine refinement
early (L3/L6: "legal citations" → "legal citations"), but at layer 24 the 8 survivors collapse
onto just 2 distinct parents, one firing on 41.9% of tokens with unrelated children.

**Root cause of that collapse: the superparent gate is an AND of two conditions**
(`fan-out >= 30%` AND `fires >= 10%`). Feature 14 clears firing by 4x (41.9%) but its fan-out is
only 21.9%, so it slips through. Superparent thresholds likely need per-layer calibration.

## How the metrics are validated: three tiers

The same five metrics are checked at three tiers of increasing realism. Each tier gives up
one guarantee and gains one dose of reality; a metric we trust has to hold across all three.
("Tier", not "layer", to avoid any confusion with the model's residual-stream layers.)

| Tier                     | What it is                                                                                                                                              | Ground truth?                 | What it proves                                                                                        |
| ------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------- | ----------------------------------------------------------------------------------------------------- |
| **1. Synthetic**   | `tests/toy_world.py`: a known 5-parent tree plus three injected pathologies, reduced to the statistics the metrics read                               | yes, by construction          | the maths is right (5/5 metrics pass across seeds 0–5, each pathology caught by its intended metric) |
| **2. Trained toy** | `tests/calibrate_on_trained_toy.py`: a Matryoshka SAE actually trained on Bussmann's tree (`sae-training`), metrics run on the *learned* features | yes, the tree is known        | the metrics survive a real training run, not just hand-built stats*(in progress)*                     |
| **3. Real SAE**    | `qualitative_check.py` on `gemma-2-2b / 6-res-matryoshka-dc` (layer 6), read against Neuronpedia labels                                             | no, human judgement stands in | the metrics mean something on a production SAE                                                        |

Tier 1 is certain but artificial; Tier 3 is real but has no ground truth; Tier 2 is the bridge
that has both a trained SAE and a known answer.

> Killing 94% to 99.9% of coverage edges is the result, not a failure: the Matryoshka SAE's
> hierarchy claim does not survive any measurement stricter than raw co-firing.
>
> Caveats: the metrics cover the SAE/MLP slice only, and B3→B4 is excluded for memory reasons.

### Across all layers

- [**Cross-depth comparison**](https://soar-eleuther-i6-hierarchy.github.io/experiment_0/outputs/cross_depth_comparison.html): the cross-depth story (4 metric panels, superparent table, qualitative-agreement collapse).
- [**Toy calibration scorecard**](https://soar-eleuther-i6-hierarchy.github.io/experiment_0/outputs/toy_calibration.html): synthetic ground-truth calibration (5/5).
- [**Kill rates**](https://soar-eleuther-i6-hierarchy.github.io/experiment_0/outputs/kill_rates.html): how many edges each metric removes.

### Per layer

Every layer has the same five pages: three interactive dashboards, then the two
rendered text reports behind them.

| Layer         | Metrics dashboard                                                                                        | Superparent fan-out                                                                                       | Qualitative dashboard                                                                                        | metrics report                                                                                          | qualitative report                                                                                         |
| ------------- | -------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------- |
| **L3**  | [open](https://soar-eleuther-i6-hierarchy.github.io/experiment_0/outputs/layer_03/metrics_dashboard.html) | [open](https://soar-eleuther-i6-hierarchy.github.io/experiment_0/outputs/layer_03/superparent_sankey.html) | [open](https://soar-eleuther-i6-hierarchy.github.io/experiment_0/outputs/layer_03/qualitative_dashboard.html) | [report](https://soar-eleuther-i6-hierarchy.github.io/experiment_0/outputs/layer_03/metrics_report.html) | [report](https://soar-eleuther-i6-hierarchy.github.io/experiment_0/outputs/layer_03/qualitative_check.html) |
| **L6**  | [open](https://soar-eleuther-i6-hierarchy.github.io/experiment_0/outputs/layer_06/metrics_dashboard.html) | [open](https://soar-eleuther-i6-hierarchy.github.io/experiment_0/outputs/layer_06/superparent_sankey.html) | [open](https://soar-eleuther-i6-hierarchy.github.io/experiment_0/outputs/layer_06/qualitative_dashboard.html) | [report](https://soar-eleuther-i6-hierarchy.github.io/experiment_0/outputs/layer_06/metrics_report.html) | [report](https://soar-eleuther-i6-hierarchy.github.io/experiment_0/outputs/layer_06/qualitative_check.html) |
| **L12** | [open](https://soar-eleuther-i6-hierarchy.github.io/experiment_0/outputs/layer_12/metrics_dashboard.html) | [open](https://soar-eleuther-i6-hierarchy.github.io/experiment_0/outputs/layer_12/superparent_sankey.html) | [open](https://soar-eleuther-i6-hierarchy.github.io/experiment_0/outputs/layer_12/qualitative_dashboard.html) | [report](https://soar-eleuther-i6-hierarchy.github.io/experiment_0/outputs/layer_12/metrics_report.html) | [report](https://soar-eleuther-i6-hierarchy.github.io/experiment_0/outputs/layer_12/qualitative_check.html) |
| **L18** | [open](https://soar-eleuther-i6-hierarchy.github.io/experiment_0/outputs/layer_18/metrics_dashboard.html) | [open](https://soar-eleuther-i6-hierarchy.github.io/experiment_0/outputs/layer_18/superparent_sankey.html) | [open](https://soar-eleuther-i6-hierarchy.github.io/experiment_0/outputs/layer_18/qualitative_dashboard.html) | [report](https://soar-eleuther-i6-hierarchy.github.io/experiment_0/outputs/layer_18/metrics_report.html) | [report](https://soar-eleuther-i6-hierarchy.github.io/experiment_0/outputs/layer_18/qualitative_check.html) |
| **L24** | [open](https://soar-eleuther-i6-hierarchy.github.io/experiment_0/outputs/layer_24/metrics_dashboard.html) | [open](https://soar-eleuther-i6-hierarchy.github.io/experiment_0/outputs/layer_24/superparent_sankey.html) | [open](https://soar-eleuther-i6-hierarchy.github.io/experiment_0/outputs/layer_24/qualitative_dashboard.html) | [report](https://soar-eleuther-i6-hierarchy.github.io/experiment_0/outputs/layer_24/metrics_report.html) | [report](https://soar-eleuther-i6-hierarchy.github.io/experiment_0/outputs/layer_24/qualitative_check.html) |

> Link to the `.html` form, not `.md`: GitHub Pages serves `.md` as raw markdown text.

## When is an edge a parent → child? The thresholds

Every metric is a matrix over feature pairs with one threshold that turns a number into a
decision. A feature "fires" on a token when its activation exceeds `FIRE_THRESHOLD = 1e-3`
(post-JumpReLU); every matrix below is built on that. All thresholds live in
[config.py](config.py).

**1. Coverage: defines the candidate edge set.** From `cofire[p,c]` (tokens where both fire):

```
R[p,c] = cofire[p,c] / fire_count[c]     reverse: is the child contained in the parent?
F[p,c] = cofire[p,c] / fire_count[p]     forward: how much of the parent the child explains
```

Keep the edge when `R[p,c] ≥ EDGE_TAU = 0.50` **and** both endpoints fire at least
`MIN_FIRE_COUNT = 20` times. (`F` is computed but not part of the accept rule.)

**2. Reconstruction: the real test of parenthood.** Per-token ablation gain
`g_f = 2·a_f·⟨d_f, x−x̂⟩ + a_f²‖d_f‖²`, summed over the child's tokens and divided by the base
error there:

```
parent_gain[p,c] = Σ g_p / Σ err        child_gain[c] = Σ g_c / Σ err
```

Pass when **both** `parent_gain ≥ RECON_REL_GAIN_MIN = 0.01` and `child_gain ≥ 0.01`, ablating
the parent must make the child's reconstruction at least 1% worse.

**3. Frequency control: does the edge survive on rare tokens?** Split tokens into buckets by
cumulative mass (0 = top 50%, 1 = next 40%, 2 = rest):

```
survival[p,c] = R over buckets 1+2  ÷  R over all buckets
```

Pass when `survival ≥ FREQ_SURVIVAL_MIN = 0.50`. Near 0 means the edge lives only on frequent
tokens, a frequency artifact.

**4. Out-degree: flags a superparent (a rejection rule).** With
`fire_rate[p] = fire_count[p] / total_tokens`, flag when **both**
`outdeg[p] / n_children ≥ SUPERPARENT_OUTDEG_FRAC = 0.30` and `fire_rate[p] ≥ SUPERPARENT_FIRE_FRAC = 0.10`.
*(This AND-gate is what R5 above shows can leak at deep layers.)*

**5. Sibling redundancy: flags feature-splitting (a rejection rule).** Mean Jaccard overlap
between a parent's children, `J(i,j) = cofire[i,j] / (fire[i]+fire[j]−cofire[i,j])`; flag when
`redundancy ≥ SIBLING_REDUNDANCY_FLAG = 0.50`.

**Verdict.** An edge is a real parent → child (a *survivor*) when it clears coverage,
reconstruction and frequency control, and its parent is not a superparent:

```
R ≥ 0.5  and  fire_p ≥ 20  and  fire_c ≥ 20          (coverage)
and  parent_gain ≥ 0.01  and  child_gain ≥ 0.01      (reconstruction)
and  survival ≥ 0.5                                   (frequency control)
and  parent not flagged as a superparent
```

The three rejection categories map exactly onto these: **superparent** / **freq-driven**
(`survival < 0.5`) / **no-recon** (`parent_gain < 0.01`).

## Run it yourself

Run from the `experiment_0/` directory.

```bash
pip install torch sae_lens datasets plotly numpy

python3 cache_stats.py        # Stage 01: cache every statistic the metrics need (slow, needs model+SAE)
python3 fetch_labels.py       # feature labels for the current layer
python3 run_metrics.py        # Stage 02: metrics_report.{json,md}
python3 qualitative_check.py  # survivor-vs-rejected edges vs Neuronpedia
python3 visualize.py          # rebuild the dashboards
```

`EXP0_LAYER` (default 6) selects the layer and writes to `outputs/layer_NN/`.

### More options

```bash
# Quick smoke slice: 16 docs instead of 400, just to check the pipeline runs
python3 cache_stats.py --docs 16

# Target a different layer (0 to 24); everything derives from EXP0_LAYER
EXP0_LAYER=12 python3 cache_stats.py

# Pick the device (default: mps on Mac, cuda on the server)
python3 cache_stats.py --device cpu
EXP0_DEVICE=cuda:1 python3 cache_stats.py
CUDA_VISIBLE_DEVICES=1 python3 cache_stats.py   # pin one GPU on the shared server

# Extra dashboards
python3 visualize.py --qualitative     # qualitative_dashboard
python3 visualize.py --calibration     # toy_calibration scorecard
```

**Validation tiers** (see "How the metrics are validated" above):

```bash
# Tier 1: synthetic ground truth (no model, no network)
python3 tests/test_metric_calibration.py

# Tier 2: calibrate on a Matryoshka SAE trained on Bussmann's toy
#   needs a checkpoint in outputs/toy_trained/ (train via sae-training/scripts/train_toy.py)
PYTHONPATH=src python3 tests/calibrate_on_trained_toy.py

# Tier 3 is qualitative_check.py above, on the real gemma-2-2b SAE
```
