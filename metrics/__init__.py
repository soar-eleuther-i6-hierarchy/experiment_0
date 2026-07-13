"""
Exp 0 metric implementations.

Every metric is a pure function over cached statistics (co-firing counts,
per-edge reconstruction sums, ...) so the same code runs on:
  - the synthetic ground-truth toy in tests/ (calibration), and
  - the real gemma-2-2b Matryoshka caches (cache_stats.py outputs).
"""

from .coverage import coverage_legs, joint_child_coverage_upper, keep_edges
from .outdegree import degree_stats, find_superparents
from .reconstruction import edge_reconstruction_condition
from .sibling_redundancy import sibling_redundancy
from .token_control import frequency_buckets, frequency_controlled_coverage

__all__ = [
    "coverage_legs",
    "joint_child_coverage_upper",
    "keep_edges",
    "degree_stats",
    "find_superparents",
    "edge_reconstruction_condition",
    "sibling_redundancy",
    "frequency_buckets",
    "frequency_controlled_coverage",
]
