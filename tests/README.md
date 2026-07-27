# `tests/` — metric calibration

There is no unit-test suite here. What lives in this directory is **calibration**: the two tiers of
validation that have a ground-truth answer, so a metric can be scored rather than eyeballed.
Tier 3 (the real SAE, judged against Neuronpedia labels) is [`qualitative_check.py`](../qualitative_check.py)
in the repo root.

← [Back to the main README](../README.md) · full results and the three-tier table: [outputs/README.md](../outputs/README.md#how-the-metrics-are-validated-three-tiers)

| File | Tier | What it does |
| ---- | ---- | ------------ |
| [`toy_world.py`](toy_world.py) | 1 | builds a synthetic world: a known 5-parent tree plus three injected pathologies (superparent, feature-split parent, frequency-coincidence edge), reduced to exactly the statistics the metrics read |
| [`test_metric_calibration.py`](test_metric_calibration.py) | 1 | runs every metric on that world and scores it on the job it claims — **5/5 pass across seeds 0–5**, each pathology caught by its intended metric |
| [`calibrate_on_trained_toy.py`](calibrate_on_trained_toy.py) | 2 | runs the metrics on a Matryoshka SAE *actually trained* on Bussmann's tree, matches learned latents back to true features, and scores edge recovery — **precision 1.00, recall 0.67** |

```bash
python3 tests/test_metric_calibration.py                    # Tier 1
PYTHONPATH=src python3 tests/calibrate_on_trained_toy.py    # Tier 2, needs outputs/toy_trained/
```

Tier 2 needs a checkpoint in `outputs/toy_trained/`, trained via `sae-training/scripts/train_toy.py`
from the team's [`sae-training`](https://github.com/soar-eleuther-i6-hierarchy/sae-training) repo.

Both tiers write into [`outputs/`](../outputs/) (`toy_calibration.json`,
`trained_toy_calibration.json`) and have a dashboard: `visualize.py --calibration` and
`visualize.py --trained-calibration`.

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
