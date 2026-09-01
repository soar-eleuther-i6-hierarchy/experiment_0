"""Every metric recomputed from its definition, and compared with the code.

Tier 1 grades the metrics by running them and checking they separate two classes.
That tests BEHAVIOUR, and it has a blind spot: the production function is what
produces the numbers Tier 1 scores, so a formula that is consistently wrong still
separates the classes and still passes. If `coverage_legs` divided by the wrong
firing count everywhere, genuine edges would still out-score pathological ones.

This closes that. Each metric is recomputed here from the definition in its own
docstring -- by explicit loops over tokens where possible, never by reusing the
implementation -- and asserted equal. It is the difference between "the metric
behaves as intended" and "the metric computes what it says it computes".

The deepest check is the ablation gain. `reconstruction.py` uses a closed form,

    g_f = 2 a_f <d_f, err> + a_f^2 ||d_f||^2

for the error increase from removing feature f. That is an algebraic identity for
||err + a_f d_f||^2 - ||err||^2, and it is asserted here against both norms
computed literally, one token at a time. A sign or a factor of two would be
invisible in every downstream number and in every scorecard row.

    python3 -m tests.test_metric_math
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import config as C  # noqa: E402
from metrics import (  # noqa: E402
    coverage_legs,
    degree_stats,
    edge_reconstruction_condition,
    find_superparents,
    frequency_controlled_coverage,
    independence_scores,
    joint_child_coverage_upper,
    keep_edges,
    parent_conditioned_redundancy,
    r_mass,
    r_supp,
    share_energy,
    sibling_redundancy,
    sres_rank_check,
)
from metrics.coverage import joint_child_coverage_exact  # noqa: E402
from metrics.reconstruction import per_token_ablation_gain  # noqa: E402
from metrics import directed_coverage, duplicate_pairs  # noqa: E402
from validation.synthetic_toy_world import build_world, P  # noqa: E402

CHECKS: list[tuple[str, str]] = []


def ok(name: str, detail: str = ""):
    CHECKS.append((name, detail))


def close(a, b, tol=1e-9):
    a = torch.as_tensor(a, dtype=torch.float64)
    b = torch.as_tensor(b, dtype=torch.float64)
    m = torch.isfinite(a) & torch.isfinite(b)
    return bool(torch.allclose(a[m], b[m], atol=tol, rtol=tol))


# ---------------------------------------------------------------------------
def test_ablation_gain_is_the_error_it_claims():
    """g_f must equal ||err + a_f d_f||^2 - ||err||^2, computed literally.

    Removing feature f from the reconstruction adds its contribution a_f d_f back
    into the error. Everything metric 2a reports is a ratio of these, so a wrong
    constant here would rescale every gain by the same factor and change no
    verdict -- which is exactly why it needs checking against the definition
    rather than against its own outputs.
    """
    g = torch.Generator().manual_seed(0)
    n, D, d = 40, 6, 5
    feats = torch.rand(n, D, generator=g, dtype=torch.float64)
    feats[feats < 0.4] = 0.0                       # a realistic share of silent features
    err = torch.randn(n, d, generator=g, dtype=torch.float64)
    W = torch.randn(D, d, generator=g, dtype=torch.float64)

    fast = per_token_ablation_gain(feats, err, W)
    slow = torch.zeros_like(fast)
    for t in range(n):
        base = float(err[t] @ err[t])
        for f in range(D):
            without = err[t] + feats[t, f] * W[f]     # f's contribution comes back
            slow[t, f] = float(without @ without) - base
    assert close(fast, slow), "closed-form ablation gain != literal error difference"
    ok("2a. per_token_ablation_gain",
       f"closed form equals ||err + a·d||² − ||err||² on all {n * D} (token, feature) cells")


def test_coverage_is_the_conditional_it_claims():
    """R = P(parent | child), F = P(child | parent), counted by hand."""
    gen = torch.Generator().manual_seed(1)
    n, p_n, c_n = 300, 4, 5
    fp = (torch.rand(n, p_n, generator=gen) < 0.3).double()
    fc = (torch.rand(n, c_n, generator=gen) < 0.4).double()
    cofire = fp.T @ fc
    R, F = coverage_legs(cofire, fp.sum(0), fc.sum(0))

    for p in range(p_n):
        for c in range(c_n):
            both = float(((fp[:, p] > 0) & (fc[:, c] > 0)).sum())
            assert close(R[p, c], both / max(float(fc[:, c].sum()), 1.0))
            assert close(F[p, c], both / max(float(fp[:, p].sum()), 1.0))
    ok("1a/1b. coverage_legs",
       f"R and F equal the conditionals counted token by token, {p_n * c_n} pairs")


def test_pmi_is_the_independence_null():
    """PMI = log[ P(p,c) / (P(p)P(c)) ], and Dev is sign-equivalent to it."""
    gen = torch.Generator().manual_seed(2)
    n = 500
    fp = (torch.rand(n, 3, generator=gen) < 0.25).double()
    fc = (torch.rand(n, 4, generator=gen) < 0.35).double()
    cofire, fire_p, fire_c = fp.T @ fc, fp.sum(0), fc.sum(0)
    out = independence_scores(cofire, fire_p, fire_c, n, min_joint=1)

    for p in range(3):
        for c in range(4):
            j, a, b = float(cofire[p, c]), float(fire_p[p]), float(fire_c[c])
            if j < 1:
                continue
            want = math.log((j / n) / ((a / n) * (b / n)))
            assert close(out["pmi"][p, c], want), f"PMI at ({p},{c})"
            # Dev = P(p|c) - P(p): different units, same sign as PMI
            assert close(out["dev"][p, c], j / b - a / n)
            assert (out["pmi"][p, c] > 0) == (out["dev"][p, c] > 0)
    ok("6. independence_scores",
       "PMI equals log[P(p,c)/(P(p)P(c))]; Dev = P(p|c) − P(p) and agrees in sign")


def test_reconstruction_ratio_and_gate():
    """parent_gain is a ratio of the sums it is handed, and the gate is an AND."""
    gen = torch.Generator().manual_seed(3)
    err_c = torch.rand(4, generator=gen, dtype=torch.float64) + 0.5
    gp = torch.randn(3, 4, generator=gen, dtype=torch.float64)
    gc = torch.randn(4, generator=gen, dtype=torch.float64)
    out = edge_reconstruction_condition(err_c, gp, gc, 0.01)
    for p in range(3):
        for c in range(4):
            assert close(out["parent_gain"][p, c], gp[p, c] / err_c[c])
            assert bool(out["passes"][p, c]) == (
                float(gp[p, c] / err_c[c]) >= 0.01 and float(gc[c] / err_c[c]) >= 0.01)
    ok("2a. edge_reconstruction_condition",
       "gain is g/err per edge; the gate is both legs ≥ threshold, not either")


def test_frequency_survival_is_a_ratio_of_coverages():
    """survival = R over mid+rare buckets, divided by R over all buckets."""
    gen = torch.Generator().manual_seed(4)
    cbb = torch.rand(3, 2, 3, generator=gen, dtype=torch.float64) * 40
    fcb = cbb.sum(1) + 5
    mask = torch.ones(2, 3, dtype=torch.bool)
    out = frequency_controlled_coverage(cbb, fcb, mask, min_fire_low=0)
    for p in range(2):
        for c in range(3):
            r_all = float(cbb[:, p, c].sum() / fcb[:, c].sum())
            r_rest = float(cbb[1:, p, c].sum() / fcb[1:, c].sum())
            assert close(out["survival"][p, c], min(r_rest / max(r_all, 1e-12), 1.5))
    ok("5. frequency_controlled_coverage",
       "survival is R(mid+rare) / R(all), clamped at 1.5")


def test_jaccard_definitions():
    """Both sibling forms are |A∩B| / |A∪B| over the right token set."""
    gen = torch.Generator().manual_seed(5)
    n = 400
    kids = (torch.rand(n, 3, generator=gen) < 0.3)
    parent = (torch.rand(n, generator=gen) < 0.5)

    got = parent_conditioned_redundancy(parent, kids)
    sub = kids[parent]
    want = []
    for a in range(3):
        for b in range(3):
            if a == b:
                continue
            inter = float((sub[:, a] & sub[:, b]).sum())
            union = float((sub[:, a] | sub[:, b]).sum())
            want.append(inter / max(union, 1.0))
    assert close(got, sum(want) / len(want))

    # the global form: same ratio, over every token rather than the parent's
    cof = (kids.double().T @ kids.double())
    mask = torch.ones(1, 3, dtype=torch.bool)
    glob = sibling_redundancy(mask, cof, kids.double().sum(0))[0]["redundancy"]
    wg = []
    for a in range(3):
        for b in range(3):
            if a == b:
                continue
            inter = float((kids[:, a] & kids[:, b]).sum())
            union = float((kids[:, a] | kids[:, b]).sum())
            wg.append(inter / max(union, 1.0))
    assert close(glob, sum(wg) / len(wg))
    ok("3 / 3'. sibling redundancy",
       "mean pairwise Jaccard; the conditioned form restricts to the parent's tokens")


def test_joint_child_and_energy_definitions():
    """R_supp, R_mass and the energy share are the ratios they are named for."""
    gen = torch.Generator().manual_seed(6)
    union_c = torch.tensor([30.0, 10.0, 0.0], dtype=torch.float64)
    fire_p = torch.tensor([60.0, 10.0, 5.0], dtype=torch.float64)
    assert close(r_supp(union_c, fire_p), union_c / fire_p)
    assert close(joint_child_coverage_exact(union_c, fire_p), union_c / fire_p)

    union_e = torch.tensor([4.0, 1.0, 0.0], dtype=torch.float64)
    tot_e = torch.tensor([8.0, 2.0, 1.0], dtype=torch.float64)
    assert close(r_mass(union_e, tot_e), union_e / tot_e)

    e_cof = torch.rand(3, 4, generator=gen, dtype=torch.float64)
    assert close(share_energy(e_cof, tot_e), e_cof / tot_e.unsqueeze(1))

    # the upper bound must actually bound the exact value
    F = torch.rand(3, 4, generator=gen, dtype=torch.float64) * 0.4
    mask = torch.ones(3, 4, dtype=torch.bool)
    up = joint_child_coverage_upper(F, mask)
    assert close(up, (F * mask).sum(1).clamp(max=1.0))
    # The upper >= exact bound is a property of one coherent world, so it is
    # checked in tests/test_tier1_calibration.py::test_joint_upper_bounds_exact
    # where `up` and `exact` come from the same build_world stats. Comparing the
    # random `up` here against the unrelated (union_c, fire_p) fixture proved
    # nothing, which is why the old line carried an `or True` that could never
    # fail; it is removed rather than left as dead weight.
    ok("1c / 8 / 9. joint-child and energy",
       "R_supp = union/fire, R_mass = union energy/total, share = cofire energy/total; "
       "the closed-form bound is min(1, ΣF)")


def test_in_block_direction_is_asymmetry():
    """parent_of is containment one way and not the other; duplicates are both."""
    gen = torch.Generator().manual_seed(7)
    n, k = 600, 5
    f = torch.zeros(n, k, dtype=torch.bool)
    f[:300, 0] = True                       # 1 sits strictly inside 0
    f[:120, 1] = True
    f[:200, 2] = True                       # 2 and 3 are co-extensive
    f[:200, 3] = True
    f[300:400, 4] = True                    # unrelated
    cof = f.double().T @ f.double()
    d = directed_coverage(cof, f.double().sum(0), 0.5, 20, 30)

    for i in range(k):
        for j in range(k):
            if i == j:
                continue
            rij = float(cof[i, j] / max(float(f[:, j].sum()), 1.0))
            rji = float(cof[j, i] / max(float(f[:, i].sum()), 1.0))
            sup = (cof[i, j] >= 30 and f[:, j].sum() >= 20 and f[:, i].sum() >= 20)
            assert bool(d["parent_of"][i, j]) == bool(rij >= 0.5 and rji < 0.5 and sup)
            assert bool(d["duplicate"][i, j]) == bool(rij >= 0.5 and rji >= 0.5 and sup)
    assert bool(d["parent_of"][0, 1]) and not bool(d["parent_of"][1, 0])
    assert (2, 3) in duplicate_pairs(d["duplicate"])
    assert not bool((d["parent_of"] & d["parent_of"].T).any()), "parent_of must be antisymmetric"
    ok("7. directed_coverage",
       "direction is R(i|j) ≥ τ > R(j|i); co-extensive pairs are duplicates and never edges; "
       "antisymmetric, so the in-block graph is acyclic")


def test_rank_rule_is_a_rank():
    """The edge passes iff BOTH decoders are inside the top k of ALL features."""
    corr = torch.tensor([0.9, 0.1, 0.8, 0.7, 0.2, 0.85], dtype=torch.float64)
    order = sorted(range(6), key=lambda i: -corr[i])       # 0, 5, 2, 3, 4, 1
    for k in (1, 2, 3, 5):
        for p in range(6):
            for c in range(6):
                passes, det = sres_rank_check(corr, p, c, k)
                assert passes == (order.index(p) < k and order.index(c) < k)
                assert det["parent_rank"] == order.index(p)
                assert close(det["s_res"], min(float(corr[p]), float(corr[c])))
    ok("2b. sres_rank_check",
       "pass iff both ranks < k over the whole dictionary; S_res = min of the two correlations")


def test_degree_and_superparent_definitions():
    gen = torch.Generator().manual_seed(8)
    mask = torch.rand(4, 10, generator=gen) < 0.3
    mask[2] = True                                   # a parent covering the whole block
    deg = degree_stats(mask)
    n_children_with_parent = int((mask.sum(0) > 0).sum())
    assert deg["n_children_with_parent"] == n_children_with_parent
    assert close(deg["poly_frac"], float((mask.sum(0) >= 2).sum()) / n_children_with_parent)

    fire = torch.tensor([10.0, 10.0, 900.0, 10.0], dtype=torch.float64)
    sps = find_superparents(mask, fire, 1000, 0.30, 0.10)
    for sp in sps:
        assert sp["outdeg_frac"] >= 0.30
    assert 2 in {sp["parent_local"] for sp in sps}
    ok("4. degree_stats / find_superparents",
       "poly_frac counts children with ≥2 parents over children that have any; "
       "the gate reads out-degree fraction")


def test_keep_edges_gate():
    R = torch.tensor([[0.6, 0.4], [0.9, 0.7]], dtype=torch.float64)
    cof = torch.tensor([[50.0, 50.0], [50.0, 5.0]], dtype=torch.float64)
    fp = torch.tensor([100.0, 10.0], dtype=torch.float64)
    fc = torch.tensor([100.0, 100.0], dtype=torch.float64)
    keep = keep_edges(R, fp, fc, 0.5, 20, cofire=cof, min_joint=30)
    assert bool(keep[0, 0]) and not bool(keep[0, 1])       # R below tau
    assert not bool(keep[1, 0]), "parent below MIN_FIRE_COUNT must be dropped"
    assert not bool(keep[1, 1]), "joint support below MIN_JOINT must be dropped"
    ok("1. keep_edges", "τ, both firing counts and the joint-support guard, all AND-ed")


def test_the_toy_carries_every_structure_it_claims():
    """The world builder actually produces the six signatures it documents."""
    from validation.synthetic_toy_world import (
        ABSORB_CHILD, ABSORB_PARENT, IN_BLOCK_CHILD, IN_BLOCK_DUP, IN_BLOCK_PARENT,
        SPLIT_CHILDREN, SPLIT_PARENT, SUPERPARENT, TOPIC_CHILD, TOPIC_PARENT,
    )
    stats, labels = build_world(seed=0)
    fire_p, fire_c, cof = stats["fire_p"], stats["fire_c"], stats["cofire"]
    N = stats["total_tokens"]

    assert float(fire_p[SUPERPARENT]) / N > 0.5, "superparent must fire on most tokens"
    assert float(cof[ABSORB_PARENT, ABSORB_CHILD]) == 0.0, "absorbed child must never co-fire"
    assert float(cof[TOPIC_PARENT, TOPIC_CHILD]) >= C.MIN_JOINT, "topic pair needs support"
    wc = stats["within_cofire"]
    a, b = IN_BLOCK_DUP
    assert float(wc[a, b]) == float(fire_c[a]) == float(fire_c[b]), "dup pair must be co-extensive"
    assert 0 < float(wc[IN_BLOCK_PARENT, IN_BLOCK_CHILD]) == float(fire_c[IN_BLOCK_CHILD])
    assert float(fire_c[IN_BLOCK_CHILD]) < float(fire_c[IN_BLOCK_PARENT])
    s0 = SPLIT_CHILDREN[0]
    assert all(float(wc[s0, c]) == float(fire_c[c]) for c in SPLIT_CHILDREN[1:]), "split children"
    assert SPLIT_PARENT in labels.split_parents
    ok("toy world",
       "superparent fires everywhere; absorbed child never co-fires; topic pair clears "
       "MIN_JOINT; the duplicate pair is co-extensive and the containment pair strictly nested")


def main() -> int:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
    print(f"[math] {len(CHECKS)} metric definitions verified against the code:")
    for name, detail in CHECKS:
        print(f"  ok  {name:<34} {detail}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
