"""
Train a BatchTopK Matryoshka SAE on the toy generator.

This is the seam between `toygen` (which knows the ground truth) and the SAE training
library that produced the released gemma checkpoint. It deliberately changes as little
as possible about that training path: the SAE class, the loss, the aux loss and the
BatchTopK threshold EMA are all used exactly as the released checkpoint used them. What
we supply is the data and the prefix sizes.

Two things here are not obvious and cost real time if rediscovered:

1. `train_toy_sae` builds its store as `FakeActivationsStore(model, provider,
   sae.cfg.context_size)` — the batch size comes from `context_size`, while the
   trainer's token accounting uses `train_batch_size_tokens`. Set them unequal
   and the run reports a training-token count that never happened. They are tied
   together below and asserted.

2. `FakeActivationsStore.next_batch` returns `model(provider(n))`, i.e. the
   provider is expected to yield feature coefficients and the toy model embeds
   them. Our generator already emits activations `h` in the residual basis it
   constructed, so the model here is the identity — routing `h` through a second,
   unrelated embedding would silently destroy the geometry the ground truth is
   defined against.

Checkpoint variants (`--variant`), which exist to isolate the released
checkpoint's L0 defect rather than inherit it:

  unfolded  the primary artifact. Decoder norms are left alone, so the saved
            `threshold` and the activations it gates are on the same scale.
  folded    reproduces the release: `fold_W_dec_norm()` scales W_enc/b_enc by
            the decoder norms while the BatchTopK threshold EMA is left behind,
            so the shipped threshold under-gates by ~mean(||W_dec||). We
            reproduce this defect on purpose, on data where we know the answer.
  rescaled  the fix upstream SAELens 6.x adopted (`rescale_acts_by_decoder_norm`):
            make top-k selection norm-invariant so that folding is exact. The
            training library is pinned to 5.x and cannot take the flag, so it is
            applied here as a forward-pass patch.

Usage:
    python train_toy.py --config backbone --variant unfolded
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import time
from pathlib import Path

import torch
from torch import nn
from transformer_lens.hook_points import HookedRootModule

from hedging_paper.saes.matryoshka_sae import (
    BatchTopkMatryoshkaSAE,
    MatryoshkaSAEConfig,
    MatryoshkaSAERunnerConfig,
)
from hedging_paper.toy_models.train_toy_sae import train_toy_sae
from toygen import spec

from toygen.world import (
    CONFOUND_OVERRIDES,
    build_world,
    checkpoint_dirname,
    choose_k,
    geometric_prefixes,
    resolve_config,
)

# The world is generated once and sampled from, rather than streamed. The
# generator runs at ~87k tok/s, which would starve the GPU on a dictionary this
# small; a resident world removes the bottleneck entirely.
# Independence is preserved where it matters — the measurement (held-out) world is
# drawn from a disjoint seed, so nothing the SAE trained on is scored.
WORLD_TOKENS = 8_000_000
BATCH_SIZE = 4096


class IdentityModel(HookedRootModule):
    """`train_toy_sae` insists on a model between the provider and the SAE.

    Ours has nothing to do: `toygen` already emits activations. This exists only
    to satisfy the `.eval()` / `__call__` interface the store expects.
    """

    def __init__(self) -> None:
        super().__init__()
        self.setup()

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        return x


def apply_rescale_patch(sae: BatchTopkMatryoshkaSAE) -> None:
    """Backport SAELens 6.x's `rescale_acts_by_decoder_norm` to the 5.x fork.

    6.x multiplies the pre-activations by ||W_dec|| before the top-k selection,
    which makes the selection invariant to decoder norm drift and therefore makes
    `fold_W_dec_norm` an exact no-op on behaviour. 6.x also now *raises* rather
    than folding a TopK SAE without it. The training library cannot take the flag
    (it is built on 5.x internals), so the same transformation is wrapped on here.
    """
    inner = sae.activation_fn

    class RescaledBatchTopK(nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return inner(x * sae.W_dec.norm(dim=-1))

    sae.activation_fn = RescaledBatchTopK()


@torch.no_grad()
def sae_quality(sae, X: torch.Tensor) -> dict:
    """Reconstruction quality of the SAE on an activation batch X [N, D].

    final_mse: mean squared reconstruction error per element.
    explained_variance: 1 - SS_res/SS_tot (R^2-style, over the whole batch).
    dead_feature_frac: fraction of latents that never fire on X.
    realized_l0: mean nonzero latents per token (should track k for an unfolded SAE).
    """
    sae.eval()
    acts = sae.encode(X)
    resid = X - sae.decode(acts)
    ss_tot = (X - X.mean(dim=0)).pow(2).sum().clamp_min(1e-12)
    fired = acts > 0
    return {
        "final_mse": resid.pow(2).mean().item(),
        "explained_variance": (1.0 - resid.pow(2).sum() / ss_tot).item(),
        "dead_feature_frac": (fired.sum(0) == 0).float().mean().item(),
        "realized_l0": fired.float().sum(-1).mean().item(),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="backbone", choices=sorted(spec.CONFIGS))
    ap.add_argument("--variant", default="unfolded",
                    choices=["unfolded", "folded", "rescaled"])
    ap.add_argument("--expansion", type=int, default=4)
    ap.add_argument("--n-steps", type=int, default=4,
                    help="number of nested matryoshka prefixes")
    ap.add_argument("--k", type=int, default=None,
                    help="BatchTopK k; defaults to the world's true L0 round(sum firing_rates). "
                         "A value below that starves the dictionary and is refused.")
    # Powered-confound overrides: raise a confound column's node count so it has enough
    # independent nodes for valid inference (needs >=5). Only 'full' (and
    # any confounds=True recipe) accepts these; resolve_config refuses them otherwise.
    ap.add_argument("--n-superparent", type=int, default=None,
                    help="override the number of always-on superparent nodes (powered run)")
    ap.add_argument("--n-token-bound-pairs", type=int, default=None,
                    help="override the number of shared-token confound pairs")
    ap.add_argument("--n-topical-pairs", type=int, default=None,
                    help="override the number of shared-topic confound pairs")
    ap.add_argument("--n-bind-ids", type=int, default=None,
                    help="override how many top-frequency ids the token-bound features share")
    ap.add_argument("--training-tokens", type=int, default=50_000_000)
    ap.add_argument("--world-tokens", type=int, default=WORLD_TOKENS)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="checkpoints")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        # A silent CPU fallback would train slowly and unintentionally; refuse it.
        raise SystemExit("CUDA unavailable — refusing to train on CPU.")

    # Resolve any powered-confound overrides up front so the exact config that trained
    # is the one persisted for the scorer (which regenerates the world from it). The
    # override set is derived from the single CONFOUND_OVERRIDES source so it cannot
    # drift from what resolve_config accepts; the same dict names the checkpoint dir and
    # is written to toy_meta.json.
    overrides = {name: getattr(args, name) for name in CONFOUND_OVERRIDES
                 if getattr(args, name) is not None}
    # --seed selects an independent world, not merely the SAE init: it seeds the geometry
    # draw (via cfg.seed) and the sampling draw, so each seed is a fresh geometry + fresh
    # Monte-Carlo world. This is what makes across-seed the valid unit of inference:
    # varying only the SAE init would replicate one world,
    # not the pair-level facts the statistics rest on. The scorer regenerates this exact
    # world from the persisted resolved_config (its `seed` is the geometry seed) plus
    # `train_seed` (the sampling seed); its held-out draw uses a disjoint sampling seed.
    cfg = spec.replace(resolve_config(args.config, **overrides), seed=args.seed)

    t0 = time.time()
    # build_world passes `config` through verbatim, so `cfg` is unchanged by this call;
    # the reassignment keeps one authoritative config object (the one persisted below).
    h, tree, cfg = build_world(
        args.config, args.world_tokens, args.seed, device, config=cfg)
    steps, d_sae = geometric_prefixes(tree.F, args.expansion, args.n_steps)
    # Derive k from the world's true L0, not cfg.K -- powered confounds raise the true L0
    # well past the declared K, and a top-k below it trains a structurally starved SAE.
    k = choose_k(tree, args.k)
    print(f"world: {tuple(h.shape)} in {time.time() - t0:.1f}s | F={tree.F} "
          f"-> steps={steps} d_sae={d_sae} k={k}")

    # Seed the global torch RNG for SAE init and batch order (the sae_lens runner's `seed`
    # leaves both inert). After build_world, so it is independent of the world draw.
    torch.manual_seed(args.seed)

    def provider(n: int) -> torch.Tensor:
        idx = torch.randint(0, h.shape[0], (n,), device=h.device)
        return h[idx]

    runner_cfg = MatryoshkaSAERunnerConfig(
        architecture="topk",
        activation_fn_kwargs={"k": k},
        model_name="toy",
        hook_name="toy.residual",
        hook_layer=0,
        d_in=cfg.D,
        d_sae=d_sae,
        expansion_factor=None,
        # Must equal train_batch_size_tokens: the store takes its batch size from
        # context_size while the trainer counts tokens with the other.
        context_size=BATCH_SIZE,
        train_batch_size_tokens=BATCH_SIZE,
        training_tokens=args.training_tokens,
        init_encoder_as_decoder_transpose=True,
        scale_sparsity_penalty_by_decoder_norm=False,
        decoder_heuristic_init=True,
        normalize_sae_decoder=False,
        device=device,
        lr=args.lr,
        lr_warm_up_steps=0,
        lr_decay_steps=0,
        log_to_wandb=False,
        seed=args.seed,
        # matryoshka
        matryoshka_steps=steps,
        include_matryoshka_inner_l1_loss=False,
        use_matryoshka_aux_loss=True,
        skip_outer_loss=False,
        normalize_losses_by_num_matryoshka_steps=False,
    )
    # Not an assert: `python -O` strips asserts, and this guards the exact silent
    # corruption documented at the top of this file — a run that reports a
    # training-token count that never happened.
    if runner_cfg.context_size != runner_cfg.train_batch_size_tokens:
        raise ValueError(
            f"context_size ({runner_cfg.context_size}) must equal "
            f"train_batch_size_tokens ({runner_cfg.train_batch_size_tokens}): "
            f"train_toy_sae takes the store's batch size from context_size while the "
            f"trainer counts tokens with train_batch_size_tokens.")

    sae = BatchTopkMatryoshkaSAE(MatryoshkaSAEConfig.from_sae_runner_config(runner_cfg))
    if args.variant == "rescaled":
        apply_rescale_patch(sae)

    train_toy_sae(
        sae,
        IdentityModel(),
        provider,
        lr=args.lr,
        training_tokens=args.training_tokens,
        train_batch_size_tokens=BATCH_SIZE,
        device=torch.device(device),
        log_to_wandb=False,
    )

    if args.variant in ("folded", "rescaled"):
        sae.fold_W_dec_norm()

    out = Path(args.out) / checkpoint_dirname(
        args.config, args.variant, k, args.expansion, overrides, seed=args.seed)
    sae.save_model_final(out)

    # Final reconstruction quality, measured on the reloaded (saved/deployment) model over an
    # in-distribution batch. Persisted per seed so training quality is tracked without wandb.
    from sae_lens import SAE as _SAE
    eval_idx = torch.randint(0, h.shape[0], (min(131072, h.shape[0]),), device=h.device)
    train_quality = sae_quality(_SAE.load_from_pretrained(str(out), device=device), h[eval_idx])
    print(f"train quality: EV={train_quality['explained_variance'] * 100:.2f}%  "
          f"dead={train_quality['dead_feature_frac'] * 100:.2f}%  "
          f"realized_L0={train_quality['realized_l0']:.2f}  mse={train_quality['final_mse']:.5f}")

    meta = {
        "config": args.config,
        "variant": args.variant,
        "k": k,
        "expansion": args.expansion,
        "n_steps": args.n_steps,
        "matryoshka_steps": steps,
        "d_sae": d_sae,
        "d_in": cfg.D,
        "F": tree.F,
        "training_tokens": args.training_tokens,
        "world_tokens": args.world_tokens,
        "train_seed": args.seed,
        # The exact, fully-resolved config that trained -- the measurement-side scorer
        # rebuilds the world from this (via spec.ToyConfig(**resolved_config)), so it
        # must be the powered config, not just the base name. `overrides` is redundant
        # with it but kept for a quick human read of what was powered.
        "overrides": overrides,
        "resolved_config": dataclasses.asdict(cfg),
        "topk_threshold": float(sae.topk_threshold),
        "W_dec_norm_mean": float(sae.W_dec.norm(dim=-1).mean()),
        "train_quality": train_quality,
    }
    (out / "toy_meta.json").write_text(json.dumps(meta, indent=2))
    print(f"saved {out}\n{json.dumps(meta, indent=2)}")


if __name__ == "__main__":
    main()
