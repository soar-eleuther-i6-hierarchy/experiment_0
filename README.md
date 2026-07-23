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

**The metrics themselves are validated, not just asserted.** On a synthetic toy with a known
5-parent tree plus three injected pathologies, all five metrics pass 5/5 across seeds 0 to 5,
each catching the pathology it was designed for.

> Killing 94% to 99.9% of coverage edges is the result, not a failure: the Matryoshka SAE's
> hierarchy claim does not survive any measurement stricter than raw co-firing.
>
> Caveats: the metrics cover the SAE/MLP slice only, and B3→B4 is excluded for memory reasons.

### Main dashboards

- [**Cross-depth comparison**](https://soar-eleuther-i6-hierarchy.github.io/experiment_0/outputs/cross_depth_comparison.html): the cross-depth story (4 metric panels, superparent table, qualitative-agreement collapse).
- [**Toy calibration scorecard**](https://soar-eleuther-i6-hierarchy.github.io/experiment_0/outputs/toy_calibration.html): synthetic ground-truth calibration (5/5).
- [**Qualitative agreement (L6)**](https://soar-eleuther-i6-hierarchy.github.io/experiment_0/outputs/layer_06/qualitative_check.html): survivor-vs-rejected edges read against Neuronpedia labels (every layer has one, see the table below).
- [**Kill rates**](https://soar-eleuther-i6-hierarchy.github.io/experiment_0/outputs/kill_rates.html): how many edges each metric removes.

### Per layer

Every layer has the same four pages: two interactive dashboards, the superparent
fan-out for all block pairs, and the rendered metrics digest.

| Layer | Metrics dashboard | Superparent fan-out | metrics report | Qualitative agreement |
| ----- | ----------------- | ------------------- | -------------- | --------------------- |
| **L3**  | [open](https://soar-eleuther-i6-hierarchy.github.io/experiment_0/outputs/layer_03/metrics_dashboard.html) | [open](https://soar-eleuther-i6-hierarchy.github.io/experiment_0/outputs/layer_03/superparent_sankey.html) | [report](https://soar-eleuther-i6-hierarchy.github.io/experiment_0/outputs/layer_03/metrics_report.html) | [open](https://soar-eleuther-i6-hierarchy.github.io/experiment_0/outputs/layer_03/qualitative_check.html) |
| **L6**  | [open](https://soar-eleuther-i6-hierarchy.github.io/experiment_0/outputs/layer_06/metrics_dashboard.html) | [open](https://soar-eleuther-i6-hierarchy.github.io/experiment_0/outputs/layer_06/superparent_sankey.html) | [report](https://soar-eleuther-i6-hierarchy.github.io/experiment_0/outputs/layer_06/metrics_report.html) | [open](https://soar-eleuther-i6-hierarchy.github.io/experiment_0/outputs/layer_06/qualitative_check.html) |
| **L12** | [open](https://soar-eleuther-i6-hierarchy.github.io/experiment_0/outputs/layer_12/metrics_dashboard.html) | [open](https://soar-eleuther-i6-hierarchy.github.io/experiment_0/outputs/layer_12/superparent_sankey.html) | [report](https://soar-eleuther-i6-hierarchy.github.io/experiment_0/outputs/layer_12/metrics_report.html) | [open](https://soar-eleuther-i6-hierarchy.github.io/experiment_0/outputs/layer_12/qualitative_check.html) |
| **L18** | [open](https://soar-eleuther-i6-hierarchy.github.io/experiment_0/outputs/layer_18/metrics_dashboard.html) | [open](https://soar-eleuther-i6-hierarchy.github.io/experiment_0/outputs/layer_18/superparent_sankey.html) | [report](https://soar-eleuther-i6-hierarchy.github.io/experiment_0/outputs/layer_18/metrics_report.html) | [open](https://soar-eleuther-i6-hierarchy.github.io/experiment_0/outputs/layer_18/qualitative_check.html) |
| **L24** | [open](https://soar-eleuther-i6-hierarchy.github.io/experiment_0/outputs/layer_24/metrics_dashboard.html) | [open](https://soar-eleuther-i6-hierarchy.github.io/experiment_0/outputs/layer_24/superparent_sankey.html) | [report](https://soar-eleuther-i6-hierarchy.github.io/experiment_0/outputs/layer_24/metrics_report.html) | [open](https://soar-eleuther-i6-hierarchy.github.io/experiment_0/outputs/layer_24/qualitative_check.html) |

> Link to the `.html` form, not `.md`: GitHub Pages serves `.md` as raw markdown text.
