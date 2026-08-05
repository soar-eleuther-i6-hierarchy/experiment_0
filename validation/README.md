# `validation/` — metric calibration

This is **calibration**, not a unit-test suite: how a metric gets *scored* rather than eyeballed.
(The directory was called `tests/`, which promised coverage of `metrics/` and delivered a toy-world
generator; `tests/` is now free for real unit tests.)

All three tiers live here. Tiers 1–2 have a ground-truth tree and run offline; **Tier 3 is the odd
one** — it needs the real `exp0_stats.pt`, needs the network for labels, is judged by human reading
rather than against a known answer, and writes a published artifact into `RUN_DIR`. It sits here
because the three tiers are one argument, not because it shares their dependencies.

← [Back to the main README](../README.md) · full results and the three-tier table: [outputs/README.md](../outputs/README.md#how-the-metrics-are-validated-three-tiers)

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
