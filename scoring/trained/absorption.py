"""
Dictionary-damage decomposition — Chanin absorption, splitting, composition (merging).

Follows the canonical, null-calibrated definitions from the literature (Chanin arXiv:2409.14507 and
arXiv:2505.11756, Bricken 2023, Bussmann arXiv:2503.17547), measured against the toy's ground-truth
match — the thing that lets us say how much of the damage is each mechanism:

  - ABSORPTION (Chanin): a SPECIFIC child latent absorbs the GENERAL parent's direction. The
    child's matched latent decoder carries the PARENT direction (`cos(W_dec[c*], g_p) > eps`, reported
    as the signed absorbed angle `theta_hat = atan2(parent_comp, child_comp)`, atan2 not arcsin),
    AND the parent latent develops a firing HOLE (`R_P = P(parent latent fires | child fires) < 1`)
    — its "seemingly arbitrary exception cases where it fails to fire". The hole is what distinguishes
    absorption from hedging WITHOUT measuring hedging (hedging keeps the parent firing, R_P ~ 1).
  - DECODER MULTIPLICITY (Chanin calls this "splitting"): the general feature's direction is
    tracked by MULTIPLE latents — `M_P = #{j : |cos(W_dec[j], g_f)| > eps}`, reported as EXCESS over
    the null (`M_P - E[M_P|null]`), never a raw count (a fixed eps manufactures multiplicity in
    exactly the wide-SAE cell where it is predicted). Reported under `decoder_multiplicity`.
  - COMPOSITION / MERGING: a conjunction latent for a co-hyponym pair — `K = cos(W_dec[j*],
    (g_a+g_b)/sqrt(2)) - 1/sqrt(2) > 0` ("red triangle", Bussmann).

`eps` is null-calibrated: the q-quantile of |cos| between random directions IN THE DECODER
SIGNAL SPAN and the trained decoders — the heavy null there, not the R^D null. Report the EXCESS.

FIREWALL: a FIDELITY diagnostic. Uses truth `g`/`A`/match + the containment/sibling TREE structure
(identity side); NEVER `pair_labels` (relationship SCORING labels); never a detector or AUROC input.
The is_a/firing_only split and the sibling candidate pairs are the reporting breakdown only.
"""

from __future__ import annotations

import math
from typing import Sequence

import torch

from scoring.core.recovery import activation_corr

DT = torch.float64
_TINY = 1e-12

ABSORPTION_CONSTANTS: dict[str, float] = {
    "hole_min": 0.15,        # parent recall must drop this far below 1 to count as a Chanin hole
    # R_solo: fraction of PARENT-SOLO tokens the parent latent must still fire on for absorption.
    # Absorption keeps R_solo ~ 0.6-0.7; a conjunction/merging latent ~ 0.
    "solo_min": 0.10,
    "conj_min": 0.05,        # excess conjunction cos (K) above 1/sqrt(2) to count as composition
    # Bonferroni target on the EXPECTED number of chance latents across the whole dictionary
    # (q = 1 - null_target_exceedances/d_sae), not a fixed quantile: a fixed p95 lets ~56 chance
    # latents through (~46% of clean features flagged by noise). At 0.01, E[chance]~0.02.
    "null_target_exceedances": 0.01,
    "n_null_perm": 1000,     # random in-span directions; large so the extreme tail quantile is stable
    # A clean feature scores excess ~1 (its own latent), a k-latent feature ~k-0.4; 1.5 catches a
    # genuine 2-way multiplicity while rejecting a single-latent clean feature.
    "multiplicity_excess_min": 1.5,
}


def _unit(x: torch.Tensor) -> torch.Tensor:
    return x / x.norm(dim=-1, keepdim=True).clamp_min(_TINY)


def _decoder_span_basis(W_dec: torch.Tensor) -> torch.Tensor:
    """Orthonormal basis [r, D] of the trained decoders' row space (the signal span).

    The cosine null must be computed here, not in R^D: decoders concentrate in the N_true-dim
    signal span where random |cos| is far heavier.
    """
    Wu = _unit(W_dec.double())
    _, S, Vh = torch.linalg.svd(Wu, full_matrices=False)
    r = int((S > 1e-6 * S[0]).sum()) if S.numel() else 0
    return Vh[:max(r, 1)]


def null_cos_threshold(W_dec: torch.Tensor, g: torch.Tensor, n_perm: int = 200,
                       q: float = 0.95, seed: int = 0) -> float:
    """eps = the q-quantile of |cos| between random IN-SPAN unit directions and the decoders.

    Deterministic (fixed generator). `g` is accepted for signature parity / future span options;
    the null lives in the decoder span, independent of any single true direction.
    """
    basis = _decoder_span_basis(W_dec)                     # [r, D]
    Wu = _unit(W_dec.double())
    r = basis.shape[0]
    gen = torch.Generator().manual_seed(int(seed))
    z = torch.randn(n_perm, r, generator=gen, dtype=DT)    # [n_perm, r]
    V = _unit(z @ basis)                                   # [n_perm, D] random unit vecs in span
    null_cos = (V @ Wu.transpose(0, 1)).abs()              # [n_perm, d_sae]
    return float(torch.quantile(null_cos.flatten(), q))


def _expected_null_count(W_dec: torch.Tensor, eps: float, n_perm: int = 200,
                         seed: int = 1) -> float:
    """E[ #{j : |cos(v, W_dec[j])| > eps} ] for a random in-span direction v — the chance
    inflation that `parent_multiplicity_excess` subtracts so `excess` recovers the genuine count."""
    basis = _decoder_span_basis(W_dec)
    Wu = _unit(W_dec.double())
    gen = torch.Generator().manual_seed(int(seed))
    z = torch.randn(n_perm, basis.shape[0], generator=gen, dtype=DT)
    V = _unit(z @ basis)
    counts = ((V @ Wu.transpose(0, 1)).abs() > eps).double().sum(dim=1)   # [n_perm]
    return float(counts.mean())


# --------------------------------------------------------------------------
# splitting (parent multiplicity, excess over null)
# --------------------------------------------------------------------------
def parent_multiplicity_excess(g: torch.Tensor, W_dec: torch.Tensor, f: int, eps: float) -> dict:
    """`M_P = #{j : |cos(W_dec[j], g[f])| > eps AND f = argmax_f' |cos(W_dec[j], g[f'])|}`, and
    `excess = M_P - E[M_P|null]`.

    A latent counts toward feature f's multiplicity ONLY if f is that latent's BEST-matching true
    feature. This is essential on an OVERCOMPLETE toy (F>D): the true features are non-orthogonal by
    design (an is-a child sits at cos≈alpha≈0.55 with its parent, above eps), so a naive
    `|cos| > eps` count miscounts a parent's own latent into every child's multiplicity — reporting
    ~100% splitting on a perfectly clean dictionary. Best-match attribution gives each latent to one
    feature, so a clean feature scores M_P=1 (its own latent) and a real split scores >=2.
    """
    Wu = _unit(W_dec.double())
    cosmat = (_unit(g.double()) @ Wu.transpose(0, 1)).abs()   # [F, d_sae]
    best_f = cosmat.argmax(dim=0)                             # [d_sae] each latent's best feature
    M_P = int(((cosmat[f] > eps) & (best_f == f)).sum())
    en = _expected_null_count(W_dec, eps)
    return {"M_P": M_P, "expected_null": en, "excess": M_P - en}


# --------------------------------------------------------------------------
# Chanin absorption (child latent carries parent + parent hole)
# --------------------------------------------------------------------------
def absorption_signals(g: torch.Tensor, W_dec: torch.Tensor, acts: torch.Tensor, A: torch.Tensor,
                       match: torch.Tensor, p: int, c: int, eps: float, constants: dict) -> dict:
    """Chanin absorption for a containment edge parent=p, child=c.

    `absorbed = (child latent carries the RESIDUAL parent direction beyond the null) AND (the parent
    latent has a firing HOLE on the child's tokens) AND (a standalone parent latent survives on
    parent-solo tokens)`. The three gates are, in order:

      - `resid_parent = cos(W_dec[c*], unit(g_p − (g_p·ĝ_c)ĝ_c))` — the child decoder's overlap with
        the part of the parent direction ORTHOGONAL to the child's own direction. The subtraction is
        the fix for the toy's designed is-a overlap `cos(g_c, g_p) = α ≈ 0.55`: the bare
        `cos(dec_c, g_p)` fires on every clean is-a child (α > eps), but resid_dir ⟂ g_c by
        construction, so a clean child decoder (dec_c = g_c) gives `resid_parent = 0` EXACTLY. A
        planted absorbed child at angle θ into the residual gives `resid_parent = sin θ`. This is the
        geometry gate. `parent_component`/`child_component`/`theta_hat` (atan2, signed) are still
        reported as the raw absorbed-angle decomposition.
      - `R_P`: parent recall on child-firing tokens; `hole = R_P < 1 − hole_min` distinguishes
        absorption (a hole) from hedging (parent keeps firing, R_P ~ 1).
      - `R_solo`: parent-latent recall on PARENT-SOLO tokens (parent active, child absent). A merging
        (conjunction) latent has no standalone parent latent → `R_solo ≈ 0`, so `R_solo > solo_min`
        vetoes merging without measuring merging directly.

    A `match[c] < 0` or `match[p] < 0` (child/parent unrecovered) gives `absorbed=False`.
    """
    mc, mp = int(match[c]), int(match[p])
    if mc < 0 or mp < 0:
        return {"parent_component": float("nan"), "child_component": float("nan"),
                "resid_parent": float("nan"), "theta_hat": float("nan"), "R_P": float("nan"),
                "R_solo": float("nan"), "hole": False, "absorbed": False}
    gp_u = _unit(g[p].double())
    gc_u = _unit(g[c].double())
    dec_c = _unit(W_dec[mc].double())
    parent_component = float(dec_c @ gp_u)                         # absorbed-angle numerator (signed, raw)
    child_component = float(dec_c @ gc_u)                          # absorbed-angle denominator (signed)
    theta_hat = math.atan2(parent_component, child_component)
    # Residual-parent projection: parent direction with the child's designed overlap removed. A clean
    # is-a child reads 0 here (resid_dir ⟂ g_c); only a child carrying the parent BEYOND its own
    # designed overlap scores > 0. Degenerate g_p ∥ g_c → no residual to carry.
    resid_dir = gp_u - (gp_u @ gc_u) * gc_u
    if float(resid_dir.norm()) < _TINY:
        resid_parent = 0.0
    else:
        resid_parent = float(dec_c @ (resid_dir / resid_dir.norm().clamp_min(_TINY)))
    child_fires = A[:, c].double() > 0
    n_cf = int(child_fires.sum())
    if n_cf == 0:
        R_P = float("nan"); hole = False
    else:
        R_P = float((acts[child_fires, mp].double() > 0).double().mean())   # parent recall
        hole = R_P < (1.0 - constants["hole_min"])
    parent_solo = (A[:, p].double() > 0) & (~child_fires)         # parent active, child absent
    n_solo = int(parent_solo.sum())
    R_solo = (float((acts[parent_solo, mp].double() > 0).double().mean())
              if n_solo > 0 else float("nan"))
    solo_ok = math.isfinite(R_solo) and R_solo > constants["solo_min"]
    absorbed = bool(resid_parent > eps and hole and solo_ok)
    return {"parent_component": parent_component, "child_component": child_component,
            "resid_parent": resid_parent, "theta_hat": theta_hat, "R_P": R_P,
            "R_solo": R_solo, "hole": hole, "absorbed": absorbed}


# --------------------------------------------------------------------------
# composition / merging (conjunction latent)
# --------------------------------------------------------------------------
def conjunction_strength(g: torch.Tensor, W_dec: torch.Tensor, a: int, b: int) -> dict:
    """`K = max_j cos(W_dec[j], unit(g_a+g_b)) - baseline`; composed iff `K > conj_min`.

    A conjunction latent for (a AND b) points along the normalized SUM of the two true directions.
    The baseline is what a pure SINGLE-feature latent already scores against that sum — for
    ORTHONORMAL features that is exactly `1/sqrt(2)` (the orthonormal-case form), but this toy is
    OVERCOMPLETE (F > D, features non-orthogonal), where a single-feature latent scores
    `sqrt((1+rho)/2) > 1/sqrt(2)`. Hardcoding `1/sqrt(2)` would fabricate composition on positively
    correlated features, so the baseline is computed from the ACTUAL geometry: the larger of the two
    single-feature cosines onto the sum direction. Composition is the EXCESS over that.
    """
    conj = _unit((g[a].double() + g[b].double()))
    cosj = _unit(W_dec.double()) @ conj                    # [d_sae], signed
    jstar = int(torch.argmax(cosj))
    baseline = max(float(conj @ _unit(g[a].double())), float(conj @ _unit(g[b].double())))
    K = float(cosj[jstar]) - baseline
    return {"K": K, "latent": jstar, "baseline": baseline}


# --------------------------------------------------------------------------
# orchestration
# --------------------------------------------------------------------------
def classify_dictionary(g: torch.Tensor, W_dec: torch.Tensor, A: torch.Tensor, acts: torch.Tensor,
                        match: torch.Tensor, matched_corr: torch.Tensor, recovered: torch.Tensor,
                        cont_edges: Sequence[tuple[int, int]],
                        sibling_pairs: Sequence[tuple[int, int]],
                        isa_child: dict[int, bool], constants: dict | None = None) -> dict:
    """Per-mechanism tallies (coexistence expected — NOT a forced single label per feature).

    Absorption per containment edge, decoder multiplicity per feature, composition per sibling pair —
    all null-calibrated. `cont_edges`/`sibling_pairs`/`isa_child` come from the TREE (identity truth);
    `pair_labels` is deliberately NOT a parameter (firewall).
    """
    constants = ABSORPTION_CONSTANTS if constants is None else constants
    F = int(g.shape[0])
    d_sae = int(W_dec.shape[0])
    # Bonferroni-adaptive quantile: target a small EXPECTED chance count across the whole dictionary
    # (not a fixed p95, which is fatally noisy at large d_sae — see ABSORPTION_CONSTANTS). This
    # dictionary-wide null gates BOTH the DECODER-MULTIPLICITY count and the ABSORPTION resid gate.
    #
    # KNOWN LIMIT (open, tracked): the absorption gate reuses this dictionary-wide multiplicity eps,
    # which is over-conservative for a targeted per-EDGE test — on real data eps≈0.35–0.38, a ~20–22°
    # absorbed-angle floor (resid_parent = sin θ̂). Pre-registered shallow cells (e.g. θ*≈19° at
    # ratio_sd=0.8/λ=0.02) can fall just under it and read CLEAN. A principled per-checkpoint floor was
    # attempted (effect-size sin θ*) and REJECTED under review: θ* is not computable for these
    # checkpoints — the SAEs are BatchTopK (not the ReLU+L1 regime
    # that θ* is derived for), and `resolved_config` stores neither `ratio_sd` nor an effective λ, so no
    # per-checkpoint θ* lookup is possible. Unblocking this needs (1) persisting ratio_sd/λ_eff per
    # checkpoint, (2) a θ* derivation for the deployed activation (BatchTopK), and (3) a
    # TRAINED-CLEAN null (an SAE trained on a no-absorption world) to bound the false positives any
    # lowered floor admits — the clean-dict W_dec=g control (resid≡0) cannot see trained-noise leakage.
    q_eff = 1.0 - constants["null_target_exceedances"] / max(d_sae, 1)
    eps = null_cos_threshold(W_dec, g, n_perm=int(constants["n_null_perm"]), q=q_eff)
    expected_null = _expected_null_count(W_dec, eps, n_perm=int(constants["n_null_perm"]))

    absorbed_edges: list[dict] = []
    absorbed_children: set[int] = set()
    # UNCLASSIFIED (catch-all): a containment edge whose absorption gates could NOT be evaluated —
    # `match<0` (unrecovered), `n_cf==0` (child never fires → R_P undefined), or `n_solo==0` (no
    # parent-solo tokens → R_solo undefined). Such an edge must NOT be silently folded into `clean`
    # (a known route to a false headline: a collapsed high-λ SAE drops many edges here).
    unclassified_edges: list[dict] = []
    unclassified_children: set[int] = set()
    # BELOW-RHO gate: an endpoint that was ASSIGNED (Hungarian match>=0) but NOT RECOVERED
    # (matched_corr < rho). On an overcomplete SAE (d_sae>=F) every true feature is assigned, so
    # match>=0 is NOT recovery -- a below-rho latent is an arbitrary weak match, and classifying it
    # as a real mechanism inflates absorbed/multiplicity/composed. Excluded from every mechanism
    # bucket (consistent with `clean`, which was already recovered-gated) and reported separately.
    below_rho_edges: list[dict] = []
    below_rho_children: set[int] = set()
    for (p, c) in cont_edges:
        if not (bool(recovered[int(p)]) and bool(recovered[int(c)])):
            below_rho_children.add(int(c))
            below_rho_edges.append({"parent": int(p), "child": int(c),
                                    "is_a": bool(isa_child.get(int(c), False))})
            continue
        sig = absorption_signals(g, W_dec, acts, A, match, int(p), int(c), eps, constants)
        if sig["absorbed"]:
            absorbed_children.add(int(c))
            absorbed_edges.append({"parent": int(p), "child": int(c),
                                   "theta_hat": sig["theta_hat"], "R_P": sig["R_P"],
                                   "is_a": bool(isa_child.get(int(c), False))})
        elif not math.isfinite(sig["R_P"]) or not math.isfinite(sig["R_solo"]):
            reason = ("no_child_fire" if not math.isfinite(sig["R_P"]) else "no_parent_solo")
            unclassified_children.add(int(c))
            unclassified_edges.append({"parent": int(p), "child": int(c), "reason": reason,
                                       "is_a": bool(isa_child.get(int(c), False))})
    # an edge absorbed via one parent overrides an unclassified/below-rho edge via another
    unclassified_children -= absorbed_children
    below_rho_children -= (absorbed_children | unclassified_children)

    # Inline the multiplicity count with the once-computed eps/expected_null. BEST-MATCH attribution (a latent counts
    # toward feature f only if f is its argmax feature) — essential on the overcomplete non-orthogonal
    # toy, else a parent's latent is miscounted into every correlated child (see
    # parent_multiplicity_excess). The orchestrator shares the null across all F features.
    cosmat = (_unit(g.double()) @ _unit(W_dec.double()).transpose(0, 1)).abs()   # [F, d_sae]
    best_f = cosmat.argmax(dim=0)                                                # [d_sae]
    above = cosmat > eps
    multiplicity_features: list[dict] = []
    for f in range(F):
        if not bool(recovered[f]):        # recovered-gated (parity with absorbed/clean)
            continue
        M_P = int((above[f] & (best_f == f)).sum())
        excess = M_P - expected_null
        if excess >= constants["multiplicity_excess_min"]:
            multiplicity_features.append({"feature": f, "M_P": M_P, "excess": excess})

    composed_pairs: list[dict] = []
    for (a, b) in sibling_pairs:
        if not (bool(recovered[int(a)]) and bool(recovered[int(b)])):   # recovered-gated
            continue
        cs = conjunction_strength(g, W_dec, int(a), int(b))
        if cs["K"] > constants["conj_min"]:
            composed_pairs.append({"a": int(a), "b": int(b), "K": cs["K"], "latent": cs["latent"]})

    multiplicity_set = {d["feature"] for d in multiplicity_features}
    clean = int(sum(1 for f in range(F)
                    if bool(recovered[f]) and f not in absorbed_children
                    and f not in multiplicity_set and f not in unclassified_children))
    return {
        "eps": eps, "expected_null": expected_null,
        "absorbed_edges": absorbed_edges,
        "decoder_multiplicity": multiplicity_features,
        "composed_pairs": composed_pairs,
        "unclassified_edges": unclassified_edges,
        "below_rho_edges": below_rho_edges,
        "counts": {"absorbed": len(absorbed_children),
                   "decoder_multiplicity": len(multiplicity_features),
                   "composed": len(composed_pairs), "clean": clean,
                   "unclassified": len(unclassified_children),
                   "below_rho": len(below_rho_children)},
        "absorbed_by_relation": {
            "is_a": sum(1 for e in absorbed_edges if e["is_a"]),
            "firing_only": sum(1 for e in absorbed_edges if not e["is_a"]),
        },
        "unclassified_by_relation": {
            "is_a": sum(1 for e in unclassified_edges if e["is_a"]),
            "firing_only": sum(1 for e in unclassified_edges if not e["is_a"]),
        },
    }


def latent_pair_masks(classification: dict, feats: list[int],
                      pairs: list[tuple[int, int]]) -> dict[str, torch.Tensor]:
    """Boolean masks over `pairs` (POSITIONS into `feats`) for the latent-side scoring columns.

      absorbed — per ORDERED recovered pair (parent, child) that is a Chanin-absorbed containment
                 edge (`classification['absorbed_edges']`, directional).
      merged   — per SYMMETRIC recovered pair that is a composed / merged sibling conjunction
                 (`classification['composed_pairs']`, stored once as (a,b); marked in BOTH orders).

    `absorbed_edges`/`composed_pairs` carry TRUE feature ids; `feats[pos]` maps a pair position back
    to its true id. FIREWALL: these are truth+trained-derived labels used ONLY for scoring (like
    `pair_labels`) — the 10 detectors never see them.
    """
    absorbed_set = {(int(e["parent"]), int(e["child"])) for e in classification["absorbed_edges"]}
    merged_set: set[tuple[int, int]] = set()
    for e in classification["composed_pairs"]:
        a, b = int(e["a"]), int(e["b"])
        merged_set.add((a, b))
        merged_set.add((b, a))                         # symmetric: both orderings are merged
    ids = [(feats[a], feats[b]) for (a, b) in pairs]
    # CPU masks (the whole toy scoring path is CPU — load_sae pins device="cpu"). If a real
    # non-toy checkpoint ever runs on CUDA, these must move to the detector device before
    # property_vs_rest_grid indexes with them (`.to(vals_all.device)`); moot today.
    absorbed = torch.tensor([pid in absorbed_set for pid in ids], dtype=torch.bool)
    merged = torch.tensor([pid in merged_set for pid in ids], dtype=torch.bool)
    return {"absorbed": absorbed, "merged": merged}


def split_readout(classification: dict, feats: list[int]) -> dict:
    """Per-LATENT split readout (decoder multiplicity) — NOT a pair column.

    `split` is per-FEATURE (a feature's direction tracked by multiple latents), not a relation over a
    pair, so it cannot be a property-vs-rest column; it is reported as a side readout instead: the
    recovered features whose decoder-multiplicity EXCESS cleared the null, with that excess. Scoped to
    the recovered universe `feats` for parity with the grid.
    """
    recovered_ids = {int(f) for f in feats}
    out = [{"feature": int(d["feature"]), "M_P": int(d["M_P"]), "excess": float(d["excess"])}
           for d in classification["decoder_multiplicity"] if int(d["feature"]) in recovered_ids]
    return {"n_split": len(out), "features": out}


def tree_edges_and_siblings(tree) -> tuple[list[tuple[int, int]], list[tuple[int, int]],
                                           dict[int, bool]]:
    """`(cont_edges, sibling_pairs, isa_child)` derived from the TREE (identity truth).

    Shared by `run_absorption` and `run_retrieval` so the two driver reports cannot silently drift if
    the `tree.parents`/`tree.children` schema changes. `cont_edges` are ordered (parent, child)
    containment edges; `isa_child[c]` is True iff child c's edge carries a non-zero overlap (is_a, not
    firing_only); `sibling_pairs` are the unordered co-hyponym pairs.
    """
    cont_edges = [(p, c) for c in range(tree.F) for p, _, _ in tree.parents.get(c, [])]
    isa_child = {c: (tree.alpha_of(c) > 0.0) for _, c in cont_edges}
    sibling_pairs = [(kids[i], kids[j]) for kids in tree.children.values()
                     for i in range(len(kids)) for j in range(i + 1, len(kids))]
    return cont_edges, sibling_pairs, isa_child


def run_absorption(ckpt_dir, n_tokens: int = 200_000, rho: float | None = None) -> dict:
    """Orchestrator (server; needs sae_training): decompose one checkpoint's dictionary damage.

    Not unit-tested (loads a real checkpoint); the pure functions above carry the coverage.
    Uses the SAME in-sample match as run_recovery, then classifies against truth.
    """
    from scoring.trained.loaders import load_sae
    from scoring.core.world import regenerate_world, signed_normalized_decoder
    from scoring.core.registry import CONSTANTS
    from scoring.core.recovery import match_features

    rho = CONSTANTS["rho_star"] if rho is None else rho
    loaded = load_sae(ckpt_dir)
    meta = loaded.meta
    world = regenerate_world(meta["resolved_config"], sample_seed=meta["train_seed"],
                             n_tokens=n_tokens)
    acts = loaded.encode(world.h)
    oriented = signed_normalized_decoder(loaded.W_dec, acts, world.h)
    res = match_features(activation_corr(world.A, acts), world.g, oriented, rho=rho)

    cont_edges, sibling_pairs, isa_child = tree_edges_and_siblings(world.tree)
    report = classify_dictionary(world.g, oriented, world.A, acts, res.match, res.matched_corr,
                                 res.recovered, cont_edges, sibling_pairs, isa_child)
    report["meta"] = {k: meta[k] for k in ("config", "variant", "k", "train_seed", "overrides")
                      if k in meta}
    report["n_recovered"] = int(res.recovered.sum())
    return report


def main() -> None:
    import argparse
    import json
    from pathlib import Path

    ap = argparse.ArgumentParser(description="Chanin absorption / splitting / composition decomposition.")
    ap.add_argument("ckpt", type=Path)
    ap.add_argument("--n-tokens", type=int, default=200_000)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    report = run_absorption(args.ckpt, n_tokens=args.n_tokens)
    text = json.dumps(report, indent=2)
    if args.out is not None:
        args.out.write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
