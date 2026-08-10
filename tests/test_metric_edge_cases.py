"""Edge-case and integration coverage that the formula recomputation in
`test_metric_math.py` and the class-separation grading in
`calibrate_on_synthetic_toy.py` leave untested.

Two gaps this closes:

  - Numeric edge cases: `gini` on degenerate inputs, `negative_parent_composition`
    on an empty negative class, and the NaN ("untestable") branches of
    `frequency_controlled_coverage`. The calibration only ever exercises the
    happy path (all-token coverage, min_fire_low=0), so a broken degenerate
    branch would never show up there.
  - Independent-route integration checks: the streaming accumulators graded
    against a brute-force per-token reference, the probe trainer graded against a
    planted direction, and a couple of invariants (PMI NaN guard, outdeg-only
    superparent flag). These were authored test-first in the local tranche and are
    promoted here so they run in CI, not only against the local scratch suite.

Finally, one test runs the whole Tier-1 scorecard (`calibrate()`), so the
strengthened rows - degree_stats/gini and negative_parent_composition are now
graded, not just printed - are executed in CI rather than only AST-parsed.

    python -m pytest tests/test_metric_edge_cases.py -q
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config as C  # noqa: E402


# ---------------------------------------------------------------------------
# Numeric edge cases
# ---------------------------------------------------------------------------
def test_gini_equal_distribution_is_zero_and_concentrated_is_high():
    """gini must be 0 for a flat distribution and (n-1)/n at maximal concentration."""
    from metrics.outdegree import gini

    assert gini(torch.tensor([1.0, 1.0, 1.0, 1.0])) == pytest.approx(0.0, abs=1e-9)
    # all mass on one of four bins -> the theoretical maximum (n-1)/n = 0.75
    assert gini(torch.tensor([0.0, 0.0, 0.0, 4.0])) == pytest.approx(0.75, abs=1e-9)
    # a middle case, computed by hand: sorted [1,2,3,4] -> 0.25
    assert gini(torch.tensor([4.0, 1.0, 3.0, 2.0])) == pytest.approx(0.25, abs=1e-9)


def test_gini_degenerate_inputs_return_zero():
    """Empty and all-zero tensors have no concentration to measure -> 0.0, not nan/div0."""
    from metrics.outdegree import gini

    assert gini(torch.tensor([])) == 0.0
    assert gini(torch.tensor([0.0, 0.0, 0.0])) == 0.0


def test_negative_parent_composition_fraction_and_empty():
    """Fraction of the negative class on which the parent fires; empty class -> 0.0."""
    from metrics.sres import negative_parent_composition

    neg = torch.tensor([True, True, True, False, False])
    fires_p = torch.tensor([True, False, True, True, False])
    # among the 3 negatives (idx 0,1,2), the parent fires on 2 -> 2/3
    assert negative_parent_composition(neg, fires_p) == pytest.approx(2 / 3)
    # no negatives -> defined as 0.0 (no division by zero)
    empty = torch.zeros(5, dtype=torch.bool)
    assert negative_parent_composition(empty, fires_p) == 0.0


def test_frequency_controlled_coverage_marks_untestable_and_nonedges_nan():
    """survival is nan for a child that barely fires outside bucket 0 (untestable)
    and for any non-edge; a well-supported edge stays finite."""
    from metrics.token_control import frequency_controlled_coverage

    # K=3 buckets, P=2 parents, C=2 children. child 0 fires plenty in buckets
    # 1+2 (testable); child 1 fires almost only in bucket 0 (fc_rest=2 < 5).
    fc = torch.tensor([[20.0, 20.0],   # bucket 0
                       [10.0, 1.0],    # bucket 1
                       [10.0, 1.0]])   # bucket 2
    cof = torch.tensor([
        [[10.0, 10.0], [8.0, 5.0]],    # bucket 0: [P,C]
        [[5.0, 0.0], [4.0, 0.0]],      # bucket 1
        [[5.0, 0.0], [4.0, 0.0]],      # bucket 2
    ])
    edge_mask = torch.tensor([[True, True], [False, True]])
    out = frequency_controlled_coverage(cof, fc, edge_mask, min_fire_low=5)
    surv = out["survival"]

    assert not torch.isnan(surv[0, 0]), "testable, kept edge must be finite"
    assert float(surv[0, 0]) == pytest.approx(1.0)          # R_rest == R_all here
    assert torch.isnan(surv[0, 1]), "child fires < min_fire_low outside bucket 0 -> untestable"
    assert torch.isnan(surv[1, 0]), "non-edge must be nan"
    assert torch.isnan(surv[1, 1]), "untestable child stays nan even where it is an edge"


# ---------------------------------------------------------------------------
# Streaming accumulators vs a brute-force reference (independent route)
# ---------------------------------------------------------------------------
def _make_world(seed=0, n=500, P=4, Cc=6):
    g = torch.Generator().manual_seed(seed)
    feats_p = torch.rand(n, P, generator=g) * (torch.rand(n, P, generator=g) < 0.3)
    feats_c = torch.rand(n, Cc, generator=g) * (torch.rand(n, Cc, generator=g) < 0.2)
    return feats_p.double(), feats_c.double()


def _dense_reference(feats_p, feats_c, thr):
    """Brute-force per-token loop the streaming accumulators are checked against."""
    n, P = feats_p.shape
    Cc = feats_c.shape[1]
    fired_p = feats_p > thr
    fired_c = feats_c > thr
    energy_cofire = torch.zeros(P, Cc, dtype=torch.float64)
    union_count = torch.zeros(P, dtype=torch.float64)
    union_energy = torch.zeros(P, dtype=torch.float64)
    energy_total = (feats_p ** 2).sum(dim=0)
    for t in range(n):
        any_c = bool(fired_c[t].any())
        for p in range(P):
            e = float(feats_p[t, p] ** 2)
            for c in range(Cc):
                if fired_c[t, c]:
                    energy_cofire[p, c] += e
            if fired_p[t, p] and any_c:
                union_count[p] += 1
            if any_c:
                union_energy[p] += e
    return energy_cofire, union_count, union_energy, energy_total


def test_streaming_accumulators_match_dense_reference():
    """Catches transposed matmuls / wrong-axis sums in the stage-01 accumulators
    (energy_cofire, union_count, union_energy) that feed r_supp/r_mass/share_energy."""
    from collect_statistics import accumulate_pair_extras

    feats_p, feats_c = _make_world()
    thr = C.FIRE_THRESHOLD
    ref = _dense_reference(feats_p, feats_c, thr)

    P, Cc = feats_p.shape[1], feats_c.shape[1]
    acc = {
        "energy_cofire": torch.zeros(P, Cc, dtype=torch.float64),
        "union_count": torch.zeros(P, dtype=torch.float64),
        "union_energy": torch.zeros(P, dtype=torch.float64),
        "energy_total": torch.zeros(P, dtype=torch.float64),
    }
    for lo, hi in [(0, 123), (123, 400), (400, 500)]:   # uneven chunks, as the real pass does
        accumulate_pair_extras(acc, feats_p[lo:hi], feats_c[lo:hi], thr)

    for got, want, name in zip(
        (acc["energy_cofire"], acc["union_count"], acc["union_energy"], acc["energy_total"]),
        ref,
        ("energy_cofire", "union_count", "union_energy", "energy_total"),
        strict=True,   # a future key added to only one tuple must error, not silently drop a check
    ):
        assert torch.allclose(got, want, atol=1e-9), f"{name} mismatch"


# ---------------------------------------------------------------------------
# Independence null: NaN guard (no -inf) and excluded count
# ---------------------------------------------------------------------------
def test_pmi_guard_yields_nan_not_neg_inf_and_counts_excluded():
    """Sub-min_joint pairs must be NaN (never log(0) = -inf leaking into means/sorts)
    AND counted, not silently dropped."""
    from metrics.independence_null import independence_scores

    fire_p = torch.tensor([50.0, 50.0])
    fire_c = torch.tensor([40.0])
    cofire = torch.tensor([[0.0], [5.0]])            # both below min_joint=10
    out = independence_scores(cofire, fire_p, fire_c, 1000, min_joint=10)
    assert torch.isnan(out["pmi"]).all()
    assert not torch.isinf(out["pmi"]).any()
    assert int(out["n_excluded"]) == 2


# ---------------------------------------------------------------------------
# Superparent flag: out-degree only, strict (old AND) variant preserved
# ---------------------------------------------------------------------------
def test_superparent_flag_outdeg_only_with_strict_kept():
    """The outdeg-only flag must catch a high-fanout low-fire parent that the old
    AND fire-gate missed; the AND behaviour must survive as the strict variant."""
    from metrics.outdegree import find_superparents

    Cc = 100
    edge_mask = torch.zeros(3, Cc, dtype=torch.bool)
    edge_mask[0, :35] = True    # 35% fan-out, low fire (the case the AND-gate missed)
    edge_mask[1, :40] = True    # classic superparent: high fan-out, high fire
    edge_mask[2, :2] = True     # normal parent
    fire_p = torch.tensor([50.0, 900.0, 100.0])
    sps = find_superparents(edge_mask, fire_p, total_tokens=1000,
                            outdeg_frac=0.30, fire_frac=0.10)
    flagged = {sp["parent_local"] for sp in sps}
    assert flagged == {0, 1}, "outdeg-only flag must catch the low-fire case"
    strict = {sp["parent_local"] for sp in sps if sp["strict"]}
    assert strict == {1}, "strict (old AND) variant must be preserved"


# ---------------------------------------------------------------------------
# Probe-based S_res on a ground-truth residual geometry
# ---------------------------------------------------------------------------
def _toy_geometry(seed=0, d=32, n=4000):
    """Residual world with a KNOWN parent direction and child = parent + spec.

    Returns (resid [n,d], child_mask [n], W_dec [F,d]) where W_dec row 0 is the
    parent decoder (parent dir), row 1 the child decoder (spec residual), 2.. random.
    """
    g = torch.Generator().manual_seed(seed)
    dp = torch.zeros(d); dp[0] = 1.0
    ds = torch.zeros(d); ds[1] = 1.0
    child_mask = torch.zeros(n, dtype=torch.bool)
    child_mask[: n // 8] = True
    parent_only = torch.zeros(n, dtype=torch.bool)
    parent_only[n // 8: n // 2] = True
    resid = 0.3 * torch.randn(n, d, generator=g)
    resid[child_mask] += 2.0 * dp + 2.0 * ds
    resid[parent_only] += 2.0 * dp
    F = 40
    W_dec = torch.randn(F, d, generator=g)
    W_dec = W_dec / W_dec.norm(dim=1, keepdim=True)
    W_dec[0] = dp
    W_dec[1] = ds
    return resid, child_mask, W_dec


def test_probe_recovers_planted_direction():
    """On separable data the probe must be unit-norm and align with the planted
    discriminative direction (dp+ds)/sqrt(2) - catches a trainer that does not
    learn or does not normalize."""
    from metrics.sres import train_probe

    resid, child_mask, _ = _toy_geometry()
    probe = train_probe(resid, child_mask, seed=0)
    assert probe is not None
    assert probe.norm().item() == pytest.approx(1.0, abs=1e-4)
    target = torch.zeros(resid.shape[1]); target[0] = 1.0; target[1] = 1.0
    target = target / target.norm()
    # observed alignment on this seed/env is ~0.86; the 0.7 bar leaves headroom
    # for cross-environment float-reduction drift while a garbage probe (~0) fails.
    assert abs(float(probe @ target)) > 0.7


def test_train_probe_returns_none_without_negatives():
    """The degenerate probe contract: no negatives / no positives / fewer than
    min_neg negatives all yield None (untestable) rather than a garbage direction."""
    from metrics.sres import train_probe

    resid = torch.randn(500, 16)
    assert train_probe(resid, torch.ones(500, dtype=torch.bool)) is None    # no negatives
    assert train_probe(resid, torch.zeros(500, dtype=torch.bool)) is None   # no positives
    few_neg = torch.ones(500, dtype=torch.bool)
    few_neg[:3] = False                                                     # 3 < min_neg=10
    assert train_probe(resid, few_neg) is None
    ok = torch.zeros(500, dtype=torch.bool)
    ok[:100] = True
    assert train_probe(resid, ok) is not None


# The 1/sqrt(2) S_res ceiling (why the rank rule is used instead of a fixed
# threshold) is a geometric identity that holds for ANY unit probe against
# orthonormal decoders, so a test of it grades no metric behaviour and was
# dropped in review. The invariant is documented in metrics/sres.py; train_probe
# is graded by test_probe_recovers_planted_direction above.


# ---------------------------------------------------------------------------
# The full Tier-1 scorecard runs (so the strengthened rows execute in CI)
# ---------------------------------------------------------------------------
def test_synthetic_scorecard_all_rows_pass():
    """Runs calibrate() end to end: every metric must recover its pathology on
    the toy. This is the row-level grading (degree_stats/gini and
    negative_parent_composition are graded, not just printed) that
    tests/test_calibration_covers_metrics.py only AST-parses."""
    from validation.calibrate_on_synthetic_toy import calibrate

    _, _, rows = calibrate()
    failed = [r["metric"] for r in rows if not r["pass"]]
    assert not failed, f"metrics failed calibration: {failed}"
