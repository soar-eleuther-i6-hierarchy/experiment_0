"""
Exp 0 metric calibration harness.

`metrics/__init__.py` promises the five metrics are pure functions over cached
tensors so "the same code runs on the synthetic ground-truth toy in tests/ and
the real gemma-2-2b caches". This package IS that toy:

  - `toy_world.py`      builds a small SAE-like world with a KNOWN parent-child
                        tree plus three injected pathologies, and emits exactly
                        the cached statistics `analyse_pair` consumes.
  - `test_metric_calibration.py`
                        runs the five production metrics (same thresholds as
                        config.py) on the toy, checks each catches the pathology
                        it is meant to, and ranks them.

Run:
    cd experiment_0
    python3 tests/test_metric_calibration.py     # prints scorecard, writes report
    pytest tests/                                # if pytest is installed
"""
