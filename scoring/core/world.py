"""
Toy-world regeneration and decoder orientation — the checkpoint-free half of the
recovery / feature-matching harness.

`scoring.core.recovery` is the pure numeric core. This module:
  * rebuilds the exact toy world from a checkpoint's persisted `resolved_config`, so no
    activations need to be shipped, and
  * provides the ground-truth-free decoder-orientation rule the detectors read.

It imports no training dependencies, so it runs in the CPU test environment. The
checkpoint seam that loads a real SAE lives in `scoring.trained.loaders`.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass

import torch

from toygen import geometry, labels, sample, spec, strengths
from toygen import tree as tree_mod

from scoring.core.recovery import child_direction_dispersion

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
    tokens      [n]      token id per row (used by the retrieval-grid frequency detector)
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

    Geometry uses the persisted `cfg.seed` (identical to training); coefficients are a
    FRESH iid draw of `n_tokens` under `sample_seed` — a new sample from the same
    world, unbiased for the recovery match's population estimate. Pass the training
    seed for matching, a disjoint seed for held-out detection.

    `resolved_config` is filtered to known `ToyConfig` fields, so a future-schema
    checkpoint still loads.
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
    """Lowest child-direction dispersion any dictionary can reach for THIS config.

    Runs `child_direction_dispersion` with the TRUE unit-`g` as the decoder and an
    identity match over the is-a children. Read the trained dispersion against this
    per-config floor (measured ~0.33 on powered-full), not a hard-coded 0.21: the null
    depends on `(d_sae, D)` and the tree, so it shifts with the config.
    """
    gn = bundle.g / bundle.g.norm(dim=1, keepdim=True).clamp_min(_TINY)
    F = bundle.g.shape[0]
    isa_children = [c for _, c in bundle.ISA]
    return child_direction_dispersion(gn, bundle.g, torch.arange(F), isa_children, m=m)


def check_world_invariants(bundle: WorldBundle, atol: float = 1e-6) -> None:
    """Check the generator's identities before scoring; raise if the world is inconsistent.

    Two identities must hold: `A @ g == Atilde @ u` (from `h - noise == Atilde @ u`) and
    the change of basis `g == Lam.T @ u`. Both hold to ~1e-16 on a clean regeneration.

    A violation means the regenerated world does not match what the SAE trained on, which
    would make every downstream number meaningless — so fail loudly here instead.
    """
    err1 = float((bundle.A @ bundle.g - bundle.Atilde @ bundle.u).abs().max())
    err2 = float((bundle.g - bundle.Lam.transpose(0, 1) @ bundle.u).abs().max())
    if err1 > atol or err2 > atol:
        raise ValueError(
            f"world invariants violated: |A@g - Atilde@u|={err1:.2e}, "
            f"|g - Lam.T@u|={err2:.2e} (atol={atol:.1e})")


def signed_normalized_decoder(W_dec: torch.Tensor, acts: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
    """L2-normalise decoder rows and orient them with a ground-truth-free sign rule.

    Cosine and S_res detectors need normalised rows, and a decoder row's sign is
    arbitrary after training, so orient each row by the sign of its co-fire-weighted
    projection onto the data: `sign_j = sign(sum_i acts[i,j] * (h_i . w_j))`.

    Uses no ground truth, is deterministic, and invariant to the input row's sign —
    safe to run before any geometry-channel detector.
    """
    hw = h @ W_dec.transpose(0, 1)                       # [n, S] = (h_i . w_j)
    s = (acts * hw).sum(dim=0)                            # [S]
    sign = torch.ones_like(s)
    sign[s < 0] = -1.0
    unit = W_dec / W_dec.norm(dim=1, keepdim=True).clamp_min(_TINY)
    return unit * sign[:, None]
