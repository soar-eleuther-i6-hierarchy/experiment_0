"""Train a Matryoshka BatchTopK SAE on the toy generator, using the sae-training backend.

Builds a toy world, feeds its activations to sae_training's MatryoshkaSAE, and saves the
trained dictionary plus the config the scorer needs to rebuild the same world.

Usage:
    python train_toy.py --config full --n-superparent 5 --seed 0
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import time
from pathlib import Path

import torch

from sae_training.architectures.matryoshka import MatryoshkaSAE
from sae_training.config import MatryoshkaSAEConfig
from sae_training.trainer import train_sae
from sae_training.utils import get_wsd_scheduler

from toygen import spec
from toygen.world import (
    CONFOUND_OVERRIDES,
    build_world,
    checkpoint_dirname,
    choose_k,
    geometric_prefixes,
    resolve_config,
)


WORLD_TOKENS = 8_000_000
BATCH_SIZE = 4096


@torch.no_grad()
def sae_quality(sae, X: torch.Tensor) -> dict:
    """Measure SAE reconstruction quality on an activation batch X [N, D].

    final_mse: mean squared reconstruction error per element.
    explained_variance: 1 - SS_res/SS_tot (R^2-style, over the whole batch).
    dead_feature_frac: fraction of latents that never fire on X.
    realized_l0: mean number of nonzero latents per token (should track k).
    """
    sae.eval()
    acts = sae.encode(X, use_threshold=True)[0]      # encode returns (hidden, pre_acts); keep hidden
    x_hat = acts @ sae.W_dec + sae.b_dec             # reconstruct by hand (more robust than a helper)
    resid = X - x_hat
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
    ap.add_argument("--expansion", type=int, default=4)
    ap.add_argument("--n-steps", type=int, default=4, help="number of nested matryoshka prefixes")
    ap.add_argument("--k", type=int, default=None, help="BatchTopK k; defaults to the world's true L0, round(sum of firing_rates). A smaller value starves the dictionary and is refused.")
    # Confound-strength overrides (full config only, node count >=5); resolve_config refuses these otherwise.
    ap.add_argument("--n-superparent", type=int, default=None)
    ap.add_argument("--n-token-bound-pairs", type=int, default=None)
    ap.add_argument("--n-topical-pairs", type=int, default=None)
    ap.add_argument("--n-bind-ids", type=int, default=None)
    ap.add_argument("--training-tokens", type=int, default=50_000_000)
    ap.add_argument("--world-tokens", type=int, default=WORLD_TOKENS)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--randomize-structure", action="store_true",
                    help="draw a seed-varied backbone (ragged branching/depth) instead of the fixed "
                         "lattice, so --seed also varies the tree STRUCTURE; the confound set stays locked.")
    ap.add_argument("--out", default="checkpoints")
    ap.add_argument("--force", action="store_true",
                    help="overwrite an already-completed checkpoint dir (default: refuse — different "
                         "training hparams can resolve to the SAME dirname)")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        raise SystemExit("CUDA unavailable -- refusing to train on CPU.")

    # Resolve confound overrides up front so the saved config matches exactly what trained.
    overrides = {name: getattr(args, name) for name in CONFOUND_OVERRIDES if getattr(args, name) is not None}
    # --seed varies geometry+sampling; the default fixed lattice shares one hierarchy across seeds, while --randomize-structure redraws the backbone per seed (confound set stays locked).
    cfg = spec.replace(resolve_config(args.config, **overrides),
                       seed=args.seed, randomize_structure=args.randomize_structure)

    t0 = time.time()
    h, tree, cfg = build_world(args.config, args.world_tokens, args.seed, device, config=cfg)
    latent_sizes, d_sae = geometric_prefixes(tree.F, args.expansion, args.n_steps)
    if latent_sizes[-1] != d_sae:
        raise ValueError(f"latent_sizes must end at d_sae; got {latent_sizes} vs {d_sae}")
    # Derive k from the world's true L0, not cfg.K — strong confounds can push true L0 above declared K.
    k = choose_k(tree, args.k)
    n_train_steps = args.training_tokens // BATCH_SIZE
    print(f"world: {tuple(h.shape)} in {time.time() - t0:.1f}s | F={tree.F} "
          f"-> latent_sizes={latent_sizes} d_sae={d_sae} k={k} steps={n_train_steps}")

    # Seed SAE init + batch order here (sae_training seeds neither); done after build_world so --seed reproduces both world and fit.
    torch.manual_seed(args.seed)

    sae_cfg = MatryoshkaSAEConfig(
        d_in=cfg.D, d_sae=d_sae, latent_sizes=list(latent_sizes),
        activation_function="batch_topk", k=k, lr=args.lr,
        # Calibrate threshold well before the run ends; the default of 1000 never kicks in on short/smoke runs.
        threshold_start_step=min(1000, max(1, n_train_steps // 2)),
        use_auxk=True,
    )
    sae = MatryoshkaSAE(sae_cfg).to(device)

    opt = torch.optim.Adam(sae.parameters(), lr=args.lr)
    sched = get_wsd_scheduler(opt, n_train_steps)

    batch_gen = torch.Generator(device=h.device).manual_seed(args.seed + 1)

    def loader():
        # Yield plain tensors [B, d_in]; a tuple would break Matryoshka, which has no attn_mask.
        for _ in range(n_train_steps):
            idx = torch.randint(0, h.shape[0], (BATCH_SIZE,), generator=batch_gen, device=h.device)
            yield h[idx]

    out = Path(args.out) / checkpoint_dirname(
        args.config, "matryoshka", k, args.expansion, overrides, seed=args.seed,
        randomize_structure=args.randomize_structure)
    # Don't silently overwrite a completed run — dirname omits training hparams, so different runs can collide (toy_meta.json only exists once a run completes).
    if (out / "toy_meta.json").exists() and not args.force:
        raise SystemExit(
            f"refusing to overwrite the completed checkpoint at {out}: a run with different hparams "
            f"resolves to the same dirname and would replace it. Pass --force to overwrite.")
    out.mkdir(parents=True, exist_ok=True)

    # train_sae runs BatchTopK, EMA-calibrates the threshold, and writes cfg.json + sae_weights.safetensors to save_dir.
    train_sae(sae, loader(), opt, lr_scheduler=sched, device=device,
              max_steps=n_train_steps, save_dir=str(out), grad_clip=1.0, dead_feature_window=200)

    # Final reconstruction quality on the reloaded (deployment) model, saved per seed to track quality without wandb.
    reloaded = MatryoshkaSAE.from_pretrained(str(out), device=device)
    eval_idx = torch.randint(0, h.shape[0], (min(131072, h.shape[0]),), device=h.device)
    train_quality = sae_quality(reloaded, h[eval_idx].to(torch.float32))
    print(f"train quality: EV={train_quality['explained_variance'] * 100:.2f}%  "
          f"dead={train_quality['dead_feature_frac'] * 100:.2f}%  "
          f"realized_L0={train_quality['realized_l0']:.2f}  mse={train_quality['final_mse']:.5f}")

    meta = {
        "config": args.config,
        "variant": "matryoshka",
        "activation_function": "batch_topk",
        "k": k,
        "expansion": args.expansion,
        "n_steps": args.n_steps,
        "matryoshka_steps": list(latent_sizes),
        "d_sae": d_sae,
        "d_in": cfg.D,
        "F": tree.F,
        "training_tokens": args.training_tokens,
        "world_tokens": args.world_tokens,
        "train_seed": args.seed,
        # The fully-resolved config that trained; the scorer rebuilds the world from this.
        "overrides": overrides,
        "resolved_config": dataclasses.asdict(cfg),
        "threshold": float(sae.threshold),
        "train_quality": train_quality,
    }
    (out / "toy_meta.json").write_text(json.dumps(meta, indent=2))
    print(f"saved {out}\n{json.dumps(meta, indent=2)}")


if __name__ == "__main__":
    main()
