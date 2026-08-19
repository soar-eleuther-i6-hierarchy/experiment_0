"""
Tier-A orchestration — regenerate the world a checkpoint trained on, load the SAE,
and run the recovery scorer over both true bases.

`scoring.recovery` is the pure numeric core; this module is the IO/wiring layer:
it rebuilds the exact toy world from a checkpoint's persisted `resolved_config`
(so no activations need to be shipped), loads the trained dictionary across the
5.x -> 6.x boundary, and drives the scorer. The scoring functions are tested
against synthetic oracles; the parts here that need `toygen` (world regeneration,
invariant checks, decoder orientation) are tested off-GPU, and the parts that need
a real checkpoint (`load_sae` / `run_recovery`) are validated on the server.

`sae_lens` is imported LAZILY inside `load_sae` so this module imports and its
world-regeneration path runs in an environment without sae-lens (e.g. the CPU test
env), and only the checkpoint entrypoints require the 6.x measurement environment.
"""

from __future__ import annotations

import dataclasses
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import torch

from toygen import geometry, labels, sample, spec, strengths
from toygen import tree as tree_mod

from scoring.recovery import (
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

_TINY = 1e-12


@dataclass(frozen=True)
class WorldBundle:
    """Everything the recovery scorer needs from one regenerated toy world.

    A / Atilde  [n, F]   true (g-basis) and healthy (u-basis) coefficients
    h           [n, D]   activations the SAE saw
    g / u       [F, D]   concept / residual directions
    Lam         [F, F]   constructive change of basis (g == Lam.T @ u)
    tree                 the built Tree (hierarchy, alphas, firing structure)
    pair_labels [F, F]   int8 single-label answer key per ordered pair (one class INDEX;
                         see toygen.labels)
    p           [F]      firing rate per feature
    CONT / ISA           direct parent->child edges / the alpha>0 (is-a) subset
    cfg                  the ToyConfig the world was built from
    tokens      [n]      token id per row (for the Tier-B frequency detector)
    """

    A: torch.Tensor
    Atilde: torch.Tensor
    h: torch.Tensor
    g: torch.Tensor
    u: torch.Tensor
    Lam: torch.Tensor
    tree: tree_mod.Tree
    pair_labels: torch.Tensor
    p: torch.Tensor
    CONT: list[tuple[int, int]]
    ISA: list[tuple[int, int]]
    cfg: spec.ToyConfig
    tokens: torch.Tensor | None = None


def regenerate_world(resolved_config: dict, sample_seed: int, n_tokens: int) -> WorldBundle:
    """Rebuild the toy world from a checkpoint's persisted `resolved_config`.

    The geometry is drawn from `cfg.seed` (persisted in the config), so it is identical
    to training; the coefficients are a FRESH draw of `n_tokens` under `sample_seed`.
    This is a new iid sample from the SAME world (same geometry + config), NOT a
    token-identical prefix of the training draw — `sample_world` is one sequential RNG
    stream whose shape depends on `n_tokens`, so a smaller draw diverges token-wise from
    the first RNG call. That is exactly right for the recovery match, which is a
    population quantity (activation-correlation) and is unbiased under resampling: pass
    the training `sample_seed` for the matching draw, a disjoint seed for the held-out
    detector draw.

    `resolved_config` is filtered to known `ToyConfig` fields first, so a checkpoint
    written under a FUTURE schema (extra keys) still loads instead of crashing the
    constructor.
    """
    known = {f.name for f in dataclasses.fields(spec.ToyConfig)}
    cfg = spec.ToyConfig(**{k: v for k, v in resolved_config.items() if k in known})

    tree = tree_mod.build_tree(cfg)
    strength = strengths.build_strengths(cfg, tree)
    geo = geometry.build_directions(cfg, tree, seed=cfg.seed)
    world = sample.sample_world(cfg, tree, strength, geo, n_tokens=n_tokens, seed=sample_seed)

    cont = [(p, c) for c in range(tree.F) for p, _, _ in tree.parents.get(c, [])]
    isa = [(p, c) for (p, c) in cont if tree.alpha_of(c) > 0.0]
    return WorldBundle(
        A=world.A, Atilde=world.Atilde, h=world.h,
        g=geo.g, u=geo.u, Lam=geo.Lam, tree=tree,
        pair_labels=labels.pair_label(tree), p=strength.p,
        CONT=cont, ISA=isa, cfg=cfg, tokens=world.tokens,
    )


def dispersion_clean_floor(bundle: WorldBundle, m: int = 5) -> torch.Tensor:
    """The lowest child-direction dispersion any (imperfect) dictionary can reach for THIS
    config: run `child_direction_dispersion` with the TRUE unit-`g` as the decoder and an
    identity match over the is-a children. The trained dispersion must be read against this
    per-config floor (measured ~0.33 on powered-full), NOT a hard-coded 0.21 — the null is a
    property of `(d_sae, D)` and the tree, so it moves with the config.
    """
    gn = bundle.g / bundle.g.norm(dim=1, keepdim=True).clamp_min(_TINY)
    F = bundle.g.shape[0]
    isa_children = [c for _, c in bundle.ISA]
    return child_direction_dispersion(gn, bundle.g, torch.arange(F), isa_children, m=m)


def check_world_invariants(bundle: WorldBundle, atol: float = 1e-6) -> None:
    """Guard the generator's identities before scoring; raise if the world is inconsistent.

    Lemma 1 (`h - noise == Atilde @ u`) reduces to `A @ g == Atilde @ u`, and the change
    of basis to `g == Lam.T @ u`; both hold to ~1e-16 on a clean regeneration. A
    violation means the regenerated world does not match what the SAE trained on, so
    every downstream number would be meaningless — fail loudly instead.
    """
    err1 = float((bundle.A @ bundle.g - bundle.Atilde @ bundle.u).abs().max())
    err2 = float((bundle.g - bundle.Lam.transpose(0, 1) @ bundle.u).abs().max())
    if err1 > atol or err2 > atol:
        raise ValueError(
            f"world invariants violated: |A@g - Atilde@u|={err1:.2e}, "
            f"|g - Lam.T@u|={err2:.2e} (atol={atol:.1e})")


def signed_normalized_decoder(W_dec: torch.Tensor, acts: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
    """L2-normalise decoder rows and orient them with a ground-truth-free sign rule.

    Cosine / S_res read normalised rows (fixes dotting an un-normalised column), and a
    decoder column's sign is arbitrary after training, so orient each row by the sign of
    its co-fire-weighted projection onto the data: `sign_j = sign(sum_i acts[i,j] * (h_i . w_j))`.
    This uses no ground truth, yet is deterministic and invariant to the input row's
    sign, so it can precede any geometry-channel detector.
    """
    hw = h @ W_dec.transpose(0, 1)                       # [n, S] = (h_i . w_j)
    s = (acts * hw).sum(dim=0)                            # [S]
    sign = torch.ones_like(s)
    sign[s < 0] = -1.0
    unit = W_dec / W_dec.norm(dim=1, keepdim=True).clamp_min(_TINY)
    return unit * sign[:, None]


# --------------------------------------------------------------------------
# checkpoint entrypoints (need the 6.x measurement env + a real checkpoint)
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class LoadedSAE:
    encode: Callable[[torch.Tensor], torch.Tensor]
    W_dec: torch.Tensor          # [S, D] decoder rows (one per latent)
    b_dec: torch.Tensor          # [D]
    meta: dict[str, object]
    arch: str | None = None      # sae.cfg.architecture (checkpoint is saved as JumpReLU, not the training-time BatchTopK) — load-bearing for the realized-L0 read


def load_sae(ckpt_dir: str | Path) -> LoadedSAE:
    """Load a trained checkpoint (6.x side) and its `toy_meta.json`.

    `sae_lens` is imported here, not at module top, so the rest of the harness runs
    without it. Returns an `encode` callable (float64 acts), the decoder rows, the
    decoder bias, and the training metadata the scorer regenerates the world from.
    """
    from sae_lens import SAE  # lazy: 6.x measurement environment only

    ckpt = Path(ckpt_dir)
    sae = SAE.load_from_disk(str(ckpt), device="cpu")
    meta = json.loads((ckpt / "toy_meta.json").read_text(encoding="utf-8"))

    def encode(h: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return sae.encode(h.to(torch.float32)).to(torch.float64)

    return LoadedSAE(
        encode=encode,
        W_dec=sae.W_dec.detach().to(torch.float64),
        b_dec=sae.b_dec.detach().to(torch.float64),
        meta=meta,
        arch=_arch_name(sae),
    )


def _arch_name(sae) -> str | None:
    """The saved SAE's architecture as a plain string ("jumprelu"), robust to sae-lens exposing it
    as a str attribute, a bound method/classmethod, or not at all. A checkpoint trained as BatchTopK
    but SAVED as JumpReLU reports "jumprelu" here — the load-bearing fact for the realized-L0 read."""
    a = getattr(sae.cfg, "architecture", None)
    if callable(a):
        try:
            a = a()
        except Exception:
            a = None
    return str(a) if a is not None else type(sae.cfg).__name__


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
    gn = bundle.g / bundle.g.norm(dim=1, keepdim=True).clamp_min(_TINY)
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

    Regenerates the matching world (a fresh `n_tokens` draw from the training world —
    see `regenerate_world`; the correlation match is unbiased under resampling), loads
    the SAE, checks the world invariants, then reports recovery curves, per-class
    recovery, splitting, child-direction dispersion (over RECOVERED children only),
    reconstruction FVU, dead latents, and the hierarchy/prefix and swap diagnostics
    (also recovered-only). `A` is primary (labels live in the g-basis); the `Atilde`
    recovery is the hedging diagnostic — the A-vs-Atilde gap is a result.

    The matching draw uses `n_tokens` (default 200k — enough for a stable population
    correlation), NOT the full training `world_tokens`; both are reported so the reader
    can see the matching sample size against the training size.
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
        report[basis_name] = {
            "n_recovered": int(res.recovered.sum()),
            "recovery_rate": float(res.recovered.double().mean()),
            "recovery_curve": {str(k): v for k, v in curve.items()},
            "per_class_recovery": per_class_recovery(
                res.recovered, bundle.pair_labels, list(labels.LABELS)),
            "split_count": match_one_to_many(corr, cap=3, rho=rho)["split_count"],
            "n_isa_children": len(isa_children),
            "n_recovered_isa_children": len(recovered_children),
            "dispersion_mean": float(disp.mean()) if len(disp) else None,
            "dispersion_per_child": {int(c): float(v) for c, v in zip(recovered_children, disp)},
        }

    report["fvu"] = reconstruction_fvu(bundle.h, acts, loaded.W_dec, loaded.b_dec)
    report["dead_latents"] = count_dead(acts, thresh=0.0)
    # REALIZED L0 on the ACTUAL h (not a randn probe): the honest sparsity the co-firing detectors
    # and the sweep's dispersion axis are read against. JumpReLU's uniform threshold means the
    # nominal top-k is NOT the realized L0 once the checkpoint is saved. `architecture` records
    # that the saved forward pass is a JumpReLU, not the training-time BatchTopK.
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
