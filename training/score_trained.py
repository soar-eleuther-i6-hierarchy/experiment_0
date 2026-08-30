"""Trained-read scoring for a toy checkpoint.

Runs in the toysae env (needs `sae_training` to load the SAE). Loads the checkpoint, matches
learned latents to true features (Hungarian on activation correlation), reduces to the
recovered universe, and computes every detector on a HELD-OUT draw with `s_res` in PROBE mode
(the metric as deployed on trained latents). Emits per-pair scores grouped by ground-truth
class in the SAME npz layout as the oracle read (`{detector}__{class}` + `__meta__`), so the
identical downstream analysis (pr_by_class, FINDINGS) applies with no changes.

Usage (on the server, from ~/toysae):
    python score_trained.py --ckpt checkpoints/only_isa-matryoshka-k29-x4-s0 \
        --out results/only_isa_trained_scores.npz
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from scoring.core.detectors import compute_all
from scoring.core.grid import held_out_sample_seed, pair_frame, reduce_to_recovered
from scoring.core.recovery import activation_corr, match_features
from scoring.core.registry import CONSTANTS, DETECTORS
from scoring.core.world import regenerate_world, signed_normalized_decoder
from scoring.trained.loaders import load_sae
from toygen import labels


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--n-tokens", type=int, default=200_000)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    L = load_sae(args.ckpt)
    meta, W = L.meta, L.W_dec
    rc = meta["resolved_config"]
    train_seed = int(meta["train_seed"])

    # Match learned -> true features on the TRAINING-seed draw.
    inw = regenerate_world(rc, sample_seed=train_seed, n_tokens=args.n_tokens)
    ai = L.encode(inw.h)
    oi = signed_normalized_decoder(W, ai, inw.h)
    res = match_features(activation_corr(inw.A, ai), inw.g, oi, rho=CONSTANTS["rho_star"])

    # Score on a DISJOINT held-out draw.
    ho = regenerate_world(rc, sample_seed=held_out_sample_seed(train_seed), n_tokens=args.n_tokens)
    ah = L.encode(ho.h)
    oh = signed_normalized_decoder(W, ah, ho.h)
    feats, di, _ = reduce_to_recovered(ah, oh, W, res.match, res.recovered,
                                       h=ho.h, b_dec=L.b_dec, tokens=ho.tokens, vocab=ho.cfg.vocab)
    dets = compute_all(di, CONSTANTS, s_res_mode="probe")           # PROBE, as deployed
    pairs, y = pair_frame(feats, ho.pair_labels)
    pa = np.array([p[0] for p in pairs])
    pb = np.array([p[1] for p in pairs])
    yv = y.numpy()

    arrays: dict[str, np.ndarray] = {}
    for det in DETECTORS:
        v = dets[det].numpy()[pa, pb]
        for name in labels.LABELS:
            arrays[f"{det}__{name}"] = v[yv == labels._index(name)].astype(np.float32)

    # Absorption (Bussmann-style, threshold-free): decoder-vs-truth cosine on recovered is-a edges.
    # For a CLEAN child latent, cos(child_decoder, parent_truth) == the designed overlap alpha;
    # ABSORPTION shows as cos ABOVE alpha (the child latent has swallowed the parent direction).
    # cos_child_self ~1 means the child was recovered; a drop signals the child latent drifted.
    g_unit = (inw.g / inw.g.norm(dim=1, keepdim=True).clamp_min(1e-12)).double()

    def _cos(u: torch.Tensor, w: torch.Tensor) -> float:
        return float((u @ w) / (u.norm() * w.norm() + 1e-12))

    isa = labels._index("is_a")
    c2p, cself = [], []
    for (a, b), yy in zip(pairs, yv):
        if yy != isa:
            continue
        p, c = feats[a], feats[b]
        dc = oh[int(res.match[c])].double()                # child's oriented matched decoder
        c2p.append(_cos(dc, g_unit[p]))                    # -> parent truth (absorption if > alpha)
        cself.append(_cos(dc, g_unit[c]))                  # -> own truth (recovery)
    arrays["absorb__cos_child_to_parent"] = np.array(c2p, dtype=np.float32)
    arrays["absorb__cos_child_self"] = np.array(cself, dtype=np.float32)

    # Activation-space absorption (Bussmann Fig 3, the decisive test): child => parent in the
    # data, so an HONEST parent latent fires whenever the child latent fires. P(parent latent |
    # child latent) ~ 1 => no absorption; a drop => the parent latent went silent (absorbed).
    thr = CONSTANTS["fire_thresh"]
    pf_given_c = []
    for (a, b), yy in zip(pairs, yv):
        if yy != isa:
            continue
        Lp, Lc = int(res.match[feats[a]]), int(res.match[feats[b]])
        cf = ah[:, Lc] > thr
        n_c = int(cf.sum())
        pf_given_c.append(float((ah[:, Lp] > thr)[cf].double().mean()) if n_c > 0 else float("nan"))
    arrays["absorb__parent_fires_given_child"] = np.array(pf_given_c, dtype=np.float32)

    R = len(feats)
    F = int(inw.g.shape[0])
    smeta = {
        "config": meta.get("config"), "read": "trained", "s_res_mode": "probe",
        "train_seed": train_seed, "n_tokens": args.n_tokens,
        "F": F, "n_recovered": R, "recovered_frac": R / F,
        "d_sae": meta.get("d_sae"), "k": meta.get("k"),
        "variant": meta.get("variant"), "alpha_designed": rc.get("alpha"),
        "train_quality": meta.get("train_quality"), "n_pairs": len(pairs),
    }
    arrays["__meta__"] = np.array(json.dumps(smeta))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, **arrays)
    print(f"wrote {out} | recovered {R}/{F} ({R / F * 100:.1f}%) | pairs {len(pairs)}")
    for name in ("is_a", "reversed", "unrelated"):
        print(f"  {name}: {int((yv == labels._index(name)).sum())} pairs")


if __name__ == "__main__":
    main()
