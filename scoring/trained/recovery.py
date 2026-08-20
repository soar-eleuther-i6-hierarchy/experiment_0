"""
Tier-A recovery driver — score one real checkpoint's feature recovery in both true bases.

Loads the SAE (`scoring.trained.loaders`), regenerates the matching world
(`scoring.core.world`), checks the world invariants, and runs the pure recovery math
(`scoring.core.recovery`). Parallels the retrieval and absorption drivers; not
unit-tested (loads a real checkpoint) — validated on the server.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import torch

from scoring.core.recovery import (
    activation_corr,
    child_direction_dispersion,
    count_dead,
    match_features,
    match_one_to_many,
    per_class_recovery,
    realized_l0,
    recovery_rate_curve,
    reconstruction_fvu,
)
from scoring.core.world import (
    WorldBundle,
    check_world_invariants,
    dispersion_clean_floor,
    regenerate_world,
    signed_normalized_decoder,
)
from scoring.trained.loaders import load_sae
from toygen import labels


def _prefix_depth_rankcorr(meta: dict, bundle: WorldBundle, match: torch.Tensor, recovered: torch.Tensor) -> float | None:
    """Spearman rank correlation of true depth vs the matched latent's matryoshka prefix.

    Diagnostic: does the hierarchy sort itself into the nested prefixes?
    Restricted to RECOVERED features — an unrecovered feature's matched latent is a
    meaningless best-of-a-bad-lot, so it must not enter the rank. Returns None when it
    cannot be computed (no steps recorded, or too few recovered features / no variance).
    """
    steps = meta.get("matryoshka_steps")
    if not steps:
        return None
    cuts = sorted(int(s) for s in steps)

    def prefix_index(j: int) -> int:
        for i, cut in enumerate(cuts):
            if j < cut:
                return i
        return len(cuts) - 1

    depths, pidx = [], []
    for f in range(bundle.tree.F):
        j = int(match[f])
        if j < 0 or not bool(recovered[f]):
            continue
        depths.append(len(bundle.tree.ancestors[f]))
        pidx.append(prefix_index(j))
    if len(depths) < 3 or len(set(depths)) < 2 or len(set(pidx)) < 2:
        return None
    from scipy.stats import spearmanr  # lazy; diagnostic only

    rho, _ = spearmanr(depths, pidx)
    return None if math.isnan(rho) else float(rho)


def _match_confusion(bundle: WorldBundle, match: torch.Tensor, oriented: torch.Tensor, recovered: torch.Tensor) -> dict:
    """Parent/child swap rate over recovered is-a edges (a swap sign-flips asymmetric detectors).

    For each is-a edge with both endpoints recovered, check that each endpoint's matched
    (oriented, unit-norm) decoder row aligns at least as well with its OWN concept
    direction as with the other endpoint's. Report the fraction that fail. Gated on
    `recovered` (not merely assigned): a below-rho Hungarian assignment is a feature the
    SAE never learned, and scoring its orientation is meaningless.
    """
    gn = bundle.g / bundle.g.norm(dim=1, keepdim=True).clamp_min(1e-12)
    checked = swapped = 0
    for p, c in bundle.ISA:
        jp, jc = int(match[p]), int(match[c])
        if jp < 0 or jc < 0 or not bool(recovered[p]) or not bool(recovered[c]):
            continue
        checked += 1
        p_ok = float(oriented[jp] @ gn[p]) >= float(oriented[jp] @ gn[c])
        c_ok = float(oriented[jc] @ gn[c]) >= float(oriented[jc] @ gn[p])
        if not (p_ok and c_ok):
            swapped += 1
    return {"checked": checked, "swapped": swapped,
            "swap_rate": (swapped / checked) if checked else None}


def run_recovery(ckpt_dir: str | Path, n_tokens: int = 200_000, rho: float = 0.5,
                 rhos: tuple[float, ...] = (0.3, 0.5, 0.7),
                 out: str | Path | None = None) -> dict:
    """Score one checkpoint's Tier-A recovery in both the g-basis (A) and u-basis (Atilde).

    Regenerates the matching world (a fresh `n_tokens` draw — see `regenerate_world`),
    loads the SAE, checks the world invariants, then reports recovery curves, per-class
    recovery, splitting, child-direction dispersion (over RECOVERED children only),
    reconstruction FVU, dead latents, and the hierarchy/prefix and swap diagnostics
    (also recovered-only). `A` is primary (labels live in the g-basis); the `Atilde`
    recovery is the hedging diagnostic — the A-vs-Atilde gap is a result. `n_tokens`
    (default 200k) is the matching draw, not the training `world_tokens`; both are reported.
    """
    loaded = load_sae(ckpt_dir)
    meta = loaded.meta
    bundle = regenerate_world(meta["resolved_config"], sample_seed=meta["train_seed"],n_tokens=n_tokens)
    check_world_invariants(bundle)

    acts = loaded.encode(bundle.h)                                    # [n, S]
    oriented = signed_normalized_decoder(loaded.W_dec, acts, bundle.h)
    isa_children = [c for _, c in bundle.ISA]

    report: dict = {
        "checkpoint": str(ckpt_dir),
        "meta": {k: meta[k] for k in
                 ("config", "variant", "k", "d_sae", "F", "train_seed", "world_tokens",
                  "overrides") if k in meta},
        "matching_tokens": n_tokens, "rho": rho,
    }
    match_by_basis: dict[str, torch.Tensor] = {}
    recovered_by_basis: dict[str, torch.Tensor] = {}
    for basis_name, coeff in (("A", bundle.A), ("Atilde", bundle.Atilde)):
        corr = activation_corr(coeff, acts)
        res = match_features(corr, bundle.g, oriented, rho=rho)
        match_by_basis[basis_name] = res.match
        recovered_by_basis[basis_name] = res.recovered
        curve = recovery_rate_curve(corr, rhos=rhos)
        # dispersion over recovered children only: an unrecovered child has match == -1
        # and r_disp == 1.0, which would conflate recovery attrition with genuine
        # absorption/splitting dispersion.
        recovered_children = [c for c in isa_children if bool(res.recovered[c])]
        disp = child_direction_dispersion(oriented, bundle.g, res.match, recovered_children, m=5)
        splits = match_one_to_many(corr, rho=rho)
        split_map = {int(f): ls for f, ls in splits["per_feature"].items() if len(ls) > 1}
        report[basis_name] = {
            "n_recovered": int(res.recovered.sum()),
            "recovery_rate": float(res.recovered.double().mean()),
            "recovery_curve": {str(k): v for k, v in curve.items()},
            "per_class_recovery": per_class_recovery(
                res.recovered, bundle.pair_labels, list(labels.LABELS)),
            "split_count": splits["split_count"],
            "split_map": split_map,
            "n_isa_children": len(isa_children),
            "n_recovered_isa_children": len(recovered_children),
            "dispersion_mean": float(disp.mean()) if len(disp) else None,
            "dispersion_per_child": {int(c): float(v) for c, v in zip(recovered_children, disp)},
        }

    report["fvu"] = reconstruction_fvu(bundle.h, acts, loaded.W_dec, loaded.b_dec)
    report["dead_latents"] = count_dead(acts, thresh=0.0)
    # REALIZED L0 on the ACTUAL h (not a randn probe): the honest sparsity the co-firing
    # detectors and the dispersion axis are read against. Inference gates with a scalar
    # threshold (use_threshold), so measure it rather than assume it equals the nominal k.
    report["realized_l0"] = realized_l0(acts)
    report["architecture"] = loaded.arch
    _floor = dispersion_clean_floor(bundle)
    report["dispersion_clean_floor"] = float(_floor.mean()) if len(_floor) else None
    report["prefix_depth_rankcorr"] = _prefix_depth_rankcorr(
        meta, bundle, match_by_basis["A"], recovered_by_basis["A"])
    report["match_confusion"] = _match_confusion(
        bundle, match_by_basis["A"], oriented, recovered_by_basis["A"])

    if out is not None:
        Path(out).write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description="Tier-A recovery scorer for a toy SAE checkpoint.")
    ap.add_argument("ckpt", type=Path, help="checkpoint dir (holds SAE files + toy_meta.json)")
    ap.add_argument("--n-tokens", type=int, default=200_000)
    ap.add_argument("--rho", type=float, default=0.5)
    ap.add_argument("--out", type=Path, default=None, help="write the JSON report here")
    args = ap.parse_args()
    report = run_recovery(args.ckpt, n_tokens=args.n_tokens, rho=args.rho, out=args.out)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
