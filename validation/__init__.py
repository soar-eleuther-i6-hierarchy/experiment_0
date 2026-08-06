"""
Hierarchy-metric calibration harness.

`metrics/__init__.py` promises the five metrics are pure functions over cached
tensors so "the same code runs on the synthetic ground-truth toy in validation/
and the real gemma-2-2b caches". This package IS that toy:

  - `toy_world.py`      builds a small SAE-like world with a KNOWN parent-child
                        tree plus three injected pathologies, and emits exactly
                        the cached statistics `analyse_pair` consumes.
  - `calibrate_on_synthetic_toy.py`
                        runs the five production metrics (same thresholds as
                        config.py) on the toy, checks each catches the pathology
                        it is meant to, and ranks them.

Two more tiers of the same argument live here: `calibrate_on_trained_toy.py`
(Tier 2, the toy passed through a real training run) and `qualitative_check.py`
(Tier 3, the real gemma-2-2b SAE read against Neuronpedia labels — the one tier
with no ground truth, and the only one that needs the cache and the network).

Run:
    cd experiment_0
    python3 validation/calibrate_on_synthetic_toy.py   # prints scorecard, writes report
    pytest validation/                              # if pytest is installed
    python3 -m validation.qualitative_check         # Tier 3
"""
