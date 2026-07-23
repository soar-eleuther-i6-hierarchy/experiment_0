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

### Main dashboards

- [**Cross-depth comparison**](https://soar-eleuther-i6-hierarchy.github.io/experiment_0/outputs/cross_depth_comparison.html): the cross-depth story (4 metric panels, superparent table, qualitative-agreement collapse).
- [**Toy calibration scorecard**](https://soar-eleuther-i6-hierarchy.github.io/experiment_0/outputs/toy_calibration.html): synthetic ground-truth calibration (5/5).
- [**Qualitative check (L6)**](https://soar-eleuther-i6-hierarchy.github.io/experiment_0/outputs/layer_06/qualitative_check.html): survivor-vs-rejected labelled edge table.
- [**Kill rates**](https://soar-eleuther-i6-hierarchy.github.io/experiment_0/outputs/kill_rates.html): how many edges each metric removes.

### Per layer

| Layer         | Dashboard                                                                                                | Superparent Sankey                                                                                        | metrics_report                                                                                    | qualitative_check                                                                                    |
| ------------- | -------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------- |
| **L3**  | [open](https://soar-eleuther-i6-hierarchy.github.io/experiment_0/outputs/layer_03/metrics_dashboard.html) | [open](https://soar-eleuther-i6-hierarchy.github.io/experiment_0/outputs/layer_03/superparent_sankey.html) | [report](https://soar-eleuther-i6-hierarchy.github.io/experiment_0/outputs/layer_03/metrics_report.html) | [report](https://soar-eleuther-i6-hierarchy.github.io/experiment_0/outputs/layer_03/qualitative_check.html) |
| **L6**  | [open](https://soar-eleuther-i6-hierarchy.github.io/experiment_0/outputs/layer_06/metrics_dashboard.html) | [open](https://soar-eleuther-i6-hierarchy.github.io/experiment_0/outputs/layer_06/superparent_sankey.html) | [report](https://soar-eleuther-i6-hierarchy.github.io/experiment_0/outputs/layer_06/metrics_report.html) | [report](https://soar-eleuther-i6-hierarchy.github.io/experiment_0/outputs/layer_06/qualitative_check.html) |
| **L12** | [open](https://soar-eleuther-i6-hierarchy.github.io/experiment_0/outputs/layer_12/metrics_dashboard.html) | [open](https://soar-eleuther-i6-hierarchy.github.io/experiment_0/outputs/layer_12/superparent_sankey.html) | [report](https://soar-eleuther-i6-hierarchy.github.io/experiment_0/outputs/layer_12/metrics_report.html) | [report](https://soar-eleuther-i6-hierarchy.github.io/experiment_0/outputs/layer_12/qualitative_check.html) |
| **L18** | [open](https://soar-eleuther-i6-hierarchy.github.io/experiment_0/outputs/layer_18/metrics_dashboard.html) | [open](https://soar-eleuther-i6-hierarchy.github.io/experiment_0/outputs/layer_18/superparent_sankey.html) | [report](https://soar-eleuther-i6-hierarchy.github.io/experiment_0/outputs/layer_18/metrics_report.html) | [report](https://soar-eleuther-i6-hierarchy.github.io/experiment_0/outputs/layer_18/qualitative_check.html) |
| **L24** | [open](https://soar-eleuther-i6-hierarchy.github.io/experiment_0/outputs/layer_24/metrics_dashboard.html) | [open](https://soar-eleuther-i6-hierarchy.github.io/experiment_0/outputs/layer_24/superparent_sankey.html) | [report](https://soar-eleuther-i6-hierarchy.github.io/experiment_0/outputs/layer_24/metrics_report.html) | [report](https://soar-eleuther-i6-hierarchy.github.io/experiment_0/outputs/layer_24/qualitative_check.html) |

> Link to the `.html` form, not `.md`, GitHub Pages serves `.md` as raw markdown text.
> Note: at L6, `qualitative_check.html` is the interactive Plotly table (the dashboard linked above); at the other layers it is the rendered report.
