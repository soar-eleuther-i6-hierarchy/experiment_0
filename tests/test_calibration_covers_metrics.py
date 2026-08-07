"""Every metric function must be exercised by the Tier-1 calibration.

The gap this closes was not a wrong number, it was a claim. Tier 1's page said
the four per-token functions were "calibrated in Tier 2"; Tier 2 imports five
functions and none of them is one of those four. So `metrics/sres.py` -- the
strict test, the one that decides which edges survive on gemma -- was graded
against no known answer at all, and the page said otherwise. Nothing failed,
because a metric with no calibration does not complain.

A prose claim about which tier covers what cannot be checked by reading either
tier. This can: it reads the calibration's source and asserts that every metric
function is actually *called* there. Adding a metric without a calibration row
now fails here instead of quietly inheriting the previous sentence's credibility.

Why AST rather than a runtime trace: the calibration trains 20 probes, and a
test that has to run it is a test that gets skipped. Parsing costs milliseconds
and answers the same question -- is this function named as a call?

    python3 -m tests.test_calibration_covers_metrics
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import metrics as M  # noqa: E402

# Not in metrics.__all__, but they are metric functions and they are graded:
# the first two live in coverage.py/reconstruction.py without being re-exported,
# the last two in in_block_edges.py because that module is a stage runner rather
# than part of the pure-function package.
EXTRA = {
    "joint_child_coverage_exact",
    "per_token_ablation_gain",
    "directed_coverage",
    "duplicate_pairs",
}

# The calibration is one file, but the world builder legitimately calls some of
# them while reducing the toy to cached statistics -- that is still coverage.
SOURCES = (
    ROOT / "validation" / "calibrate_on_synthetic_toy.py",
    ROOT / "validation" / "synthetic_toy_world.py",
)


def called_names(paths) -> set[str]:
    """Every bare-name call site across the given files."""
    out: set[str] = set()
    for p in paths:
        for node in ast.walk(ast.parse(p.read_text())):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                out.add(node.func.id)
    return out


def test_calibration_covers_every_metric_function():
    universe = set(M.__all__) | EXTRA
    missing = sorted(universe - called_names(SOURCES))
    assert not missing, (
        f"{len(missing)} metric function(s) are never called by the Tier-1 "
        f"calibration: {missing}\n"
        "Add a scorecard row in validation/calibrate_on_synthetic_toy.py that "
        "asserts what the function is FOR -- that it flags its pathology and "
        "spares the genuine tree. A function with no row is ungraded, whatever "
        "any README says about it."
    )
    return len(universe)


def main() -> int:
    n = test_calibration_covers_every_metric_function()
    print(f"[test] all {n} metric functions are exercised by the Tier-1 calibration")
    return 0


if __name__ == "__main__":
    sys.exit(main())
