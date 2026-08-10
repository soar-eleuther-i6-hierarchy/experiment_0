"""Tier-1 calibration tests for the 6 energy / joint-child metrics.

These were written test-first against a spec, before `validation.synthetic_toy_world`
emitted the four accumulators they consume - `energy_cofire` [P,C],
`energy_total` [P], `union_count` [P], `union_energy` [P]. That gap is now
closed: the toy's `_reduce` emits all four, so every test here passes. They are
kept as the executable contract for those accumulators and metrics - each test
encodes a requirement (a metric separates the fixture it targets, or an exact
invariant holds) that a future edit must not silently break.

The one test that never depended on the new keys (the PMI ranking check) is a
fixture-sanity / regression guard over already-emitted statistics.

Moved from `tests_local/` into the tracked suite so the Tier-1 scorecard's
building blocks run in CI rather than only being AST-parsed by
`tests/test_calibration_covers_metrics.py`.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
import torch

import config as CFG                      # thresholds live here; not the toy's `C`
import validation.synthetic_toy_world as tw   # renamed upstream from validation/toy_world.py; same build_world/P/C/SUPERPARENT API
from metrics import (
    coverage_legs,
    independence_scores,
    joint_child_coverage_upper,
    keep_edges,
    r_mass,
    r_supp,
    share_energy,
)
from metrics.coverage import joint_child_coverage_exact  # not re-exported in metrics.__all__

# The four accumulators the toy must emit for these metrics to be computable.
NEW_STAT_KEYS = ("energy_cofire", "energy_total", "union_count", "union_energy")


def _coverage_and_edges(stats):
    """(F, edge_mask) via the existing coverage/keep-edge path - uses no new keys."""
    R, F = coverage_legs(stats["cofire"], stats["fire_p"], stats["fire_c"])
    edge_mask = keep_edges(
        R, stats["fire_p"], stats["fire_c"], CFG.EDGE_TAU, CFG.MIN_FIRE_COUNT
    )
    return F, edge_mask


def test_reduce_emits_energy_union_keys():
    """_reduce must emit energy_cofire[P,C]/energy_total[P]/union_count[P]/union_energy[P]: catches a forgotten key or a wrong-axis (transposed) accumulator."""
    stats, _ = tw.build_world(seed=0)
    P, C = stats["P"], stats["C"]

    missing = [k for k in NEW_STAT_KEYS if k not in stats]
    assert not missing, f"stats is missing new accumulator keys: {missing}"

    assert tuple(stats["energy_cofire"].shape) == (P, C)
    assert tuple(stats["energy_total"].shape) == (P,)
    assert tuple(stats["union_count"].shape) == (P,)
    assert tuple(stats["union_energy"].shape) == (P,)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_pmi_ranks_genuine_above_superparent(seed):
    """PMI must rank every genuine tree edge strictly above the superparent's base-rate co-firing: catches PMI failing to discount a near-always-on parent (miswired PMI or a broken fixture)."""
    stats, labels = tw.build_world(seed=seed)
    null = independence_scores(
        stats["cofire"], stats["fire_p"], stats["fire_c"],
        stats["total_tokens"], CFG.MIN_JOINT,
    )
    pmi, valid = null["pmi"], null["valid"]

    genuine = [float(pmi[p, c]) for (p, c) in labels.genuine if bool(valid[p, c])]
    superparent = [
        float(pmi[tw.SUPERPARENT, c])
        for c in range(stats["C"])
        if bool(valid[tw.SUPERPARENT, c])
    ]
    assert genuine and superparent, "fixture yielded no valid genuine/superparent pairs to rank"
    assert min(genuine) > max(superparent)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_joint_child_coverage_high_for_genuine_low_for_superparent(seed):
    """Union-based joint-child coverage must be near-complete for a genuine parent yet clearly lower for the superparent whose mass lives where no child fires: catches a transposed union accumulator or coverage that can't tell a complete child-cover from an incomplete one."""
    stats, _ = tw.build_world(seed=seed)
    rs = r_supp(stats["union_count"], stats["fire_p"])

    genuine_vals = [float(rs[g]) for g in sorted(tw.GENUINE_TREE)]
    sp_val = float(rs[tw.SUPERPARENT])
    assert sp_val < 0.95
    assert min(genuine_vals) > sp_val + 0.2


def test_rsupp_equals_joint_child_exact():
    """r_supp and joint_child_coverage_exact are the same union_count/fire_p formula and must agree cell-for-cell: catches the two implementations silently diverging."""
    stats, _ = tw.build_world(seed=0)
    a = r_supp(stats["union_count"], stats["fire_p"])
    b = joint_child_coverage_exact(stats["union_count"], stats["fire_p"])
    assert torch.allclose(a, b)


def test_joint_upper_bounds_exact():
    """The forward-sum upper bound must never fall below the exact union coverage for any parent: catches a bound that is violated (a real bug, since upper is defined to over-count)."""
    stats, _ = tw.build_world(seed=0)
    F, edge_mask = _coverage_and_edges(stats)
    exact = joint_child_coverage_exact(stats["union_count"], stats["fire_p"])
    upper = joint_child_coverage_upper(F, edge_mask)
    assert (upper >= exact - 1e-9).all()


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_r_mass_high_for_genuine_low_for_superparent(seed):
    """Energy-weighted joint-child coverage must separate genuine parents from the superparent, same direction as r_supp: catches union_energy/energy_total swapped or an energy accumulator blind to where the parent's mass lives."""
    stats, _ = tw.build_world(seed=seed)
    rm = r_mass(stats["union_energy"], stats["energy_total"])
    genuine_vals = [float(rm[g]) for g in sorted(tw.GENUINE_TREE)]
    assert min(genuine_vals) > float(rm[tw.SUPERPARENT])


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_share_energy_flags_split_parent(seed):
    """Energy concentration must expose the feature-split parent (one duplicate child holds ~all its energy) while a genuine parent's energy stays spread across disjoint children: catches share_energy missing a dominating duplicate child."""
    stats, _ = tw.build_world(seed=seed)
    S = share_energy(stats["energy_cofire"], stats["energy_total"])
    max_share = S.max(dim=1).values  # [P]

    genuine_max = max(float(max_share[g]) for g in sorted(tw.GENUINE_TREE))
    assert float(max_share[tw.SPLIT_PARENT]) >= 0.9
    assert genuine_max < float(max_share[tw.SPLIT_PARENT])
    assert genuine_max < 0.6
