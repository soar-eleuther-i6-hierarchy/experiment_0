"""
Exp 0 metric implementations.

Every metric is a pure function over cached statistics (co-firing counts,
per-edge reconstruction sums, ...) so the same code runs on:
  - the synthetic ground-truth toy in tests/ (calibration), and
  - the real gemma-2-2b Matryoshka caches (cache_stats.py outputs).
"""

from .coverage import coverage_legs, joint_child_coverage_upper, keep_edges
from .independence_null import independence_scores
from .joint_child import r_mass, r_supp, share_energy
from .outdegree import degree_stats, find_superparents
from .reconstruction import edge_reconstruction_condition
from .sibling_redundancy import parent_conditioned_redundancy, sibling_redundancy
from .sres import negative_parent_composition, sres_rank_check, train_probe
from .token_control import frequency_buckets, frequency_controlled_coverage

__all__ = [
    "coverage_legs",
    "joint_child_coverage_upper",
    "keep_edges",
    "independence_scores",
    "share_energy",
    "r_supp",
    "r_mass",
    "degree_stats",
    "find_superparents",
    "edge_reconstruction_condition",
    "sibling_redundancy",
    "parent_conditioned_redundancy",
    "train_probe",
    "sres_rank_check",
    "negative_parent_composition",
    "frequency_buckets",
    "frequency_controlled_coverage",
]
