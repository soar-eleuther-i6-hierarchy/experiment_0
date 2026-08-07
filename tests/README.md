# `tests/` — unit tests for the pipeline code

Distinct from [`validation/`](../validation/), which scores the *metrics* against known answers.
Nothing here measures hierarchy or grades an edge. These guard claims the **code** makes about
itself — the kind that break silently, with every downstream number still looking reasonable.

| File | Guards | Cost |
| ---- | ------ | ---- |
| [`test_collect_generic.py`](test_collect_generic.py) | `collect_statistics.collect()` runs on a source that is not gemma | no network, no GPU, ~1s |
| [`test_dashboards_generic.py`](test_dashboards_generic.py) | stages 02 → 03 → 04 grade and *render* a source that is not gemma | no network, no GPU, ~13s |
| [`test_calibration_covers_metrics.py`](test_calibration_covers_metrics.py) | every function in `metrics.__all__` is actually called by the Tier-1 calibration | AST only, ~0.1s |

```bash
python3 -m tests.test_collect_generic
python3 -m tests.test_dashboards_generic
python3 -m tests.test_calibration_covers_metrics
```

## Why this one exists

Stage 01 used to load gemma-2-2b, the released Matryoshka SAE and pile-10k, then accumulate
statistics from them, all in one function. `collect()` is the accumulation split out, so an adapter
can feed it a PCFG transformer or a trained toy instead.

That claim is cheap to make and easy to break. One `config` global left in the accumulation
loop — `C.D_SAE`, `C.BLOCK_RANGES`, `utils.sae_utils.block_slice` — reintroduces gemma's 32768
latents in 5 blocks with no symptom at all: a 1792-latent dictionary gets sliced at the wrong
boundaries and every statistic downstream is computed from the wrong columns. Wrong statistics do
not crash. They produce plausible numbers.

So the test builds a stub model, a stub SAE and a config with 28 features in 3 blocks — nothing like
gemma — and asserts the result is a well-formed schema-v2 stats file whose shapes came from the
config it was handed. When the umbrella repo is checked out beside this one it also runs
`contracts/validate_stats.py` against its own output.

## And why the second one

`collect()` being source-agnostic bought nothing while the stages that *read* its output were not.
`run_token_metrics.py` and `reporting/visualize.py` both sliced `config.BLOCK_RANGES` directly, so a
PCFG file (1792 latents in 8 blocks) was graded and drawn against gemma's five — silently for the
first four pairs, because every index is in range there, and only then with an error. Stage 03 also
loaded the *released gemma decoder* to score S_res, whatever dictionary the cache came from.

The second test runs a PCFG-shaped file (8 blocks) through 02 → 03 → 04 and asserts the pages
describe the file they were built from: all seven pairs graded, the header naming that run's model
and dictionary rather than gemma's, S_res computed from the run's own `w_dec.pt`, and no layer pill
lit on a run that is not a gemma layer.
