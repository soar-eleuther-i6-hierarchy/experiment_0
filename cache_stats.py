"""
Stage 01 - Cache the richer statistics Exp 0's five metrics need.

The warm-up's stage 01 kept only co-firing counts (crude activation coverage).
Exp 0 treats the metrics as competing measurements of the SAME edge, so this
pass streams gemma-2-2b's layer-6 residual through the Matryoshka SAE once and
accumulates, for every adjacent block pair p -> c we compute:

    fire_count[f]                : tokens feature f fired on            [D_SAE]
    cofire[p->c][P, C]           : tokens both p and c fired on
    cofire_by_bucket[p->c][K,P,C]: co-firing split by token-frequency bucket
    fire_c_by_bucket[c][K, C]    : child firing counts per freq bucket

    # reconstruction (Tree-SAE condition), summed over the CHILD's firing tokens
    err_sum_c[c][C]              : base recon error  ||x - x_hat||^2
    g_child_sum[c][C]            : child's own ablation gain g_c
    g_parent_sum[p->c][P, C]     : parent's ablation gain g_p

    # sibling redundancy: within-block child-child co-firing [C, C] (blocks 1-3)
    within_cofire[b][C, C]

Token-frequency buckets need corpus-wide counts, so we tokenize every doc up
front (cheap - no model), build the buckets, then run the single model pass.

Run:
    cd experiment_0 && python3 cache_stats.py            # full (config N_DOCS)
    python3 cache_stats.py --docs 16                     # quick smoke slice
Output:
    outputs/exp0_stats.pt
"""

from __future__ import annotations

import argparse
import time

import torch
from datasets import load_dataset

import config as C
import sae_utils as U
from metrics.reconstruction import per_token_ablation_gain
from metrics.token_control import frequency_buckets


def tokenize_docs(model, texts, ctx):
    """Truncate each doc to `ctx` tokens (BOS prepended). Returns list of 1-D LongTensors."""
    seqs = []
    for t in texts:
        toks = model.to_tokens(t, prepend_bos=C.PREPEND_BOS)[0][:ctx]
        if toks.numel() > 0:
            seqs.append(toks.cpu())
    return seqs


def right_pad(seqs, pad_id, device):
    """Right-pad a list of 1-D token tensors into a [B, maxlen] batch on device."""
    maxlen = max(len(s) for s in seqs)
    batch = torch.full((len(seqs), maxlen), pad_id, dtype=torch.long)
    for i, s in enumerate(seqs):
        batch[i, : len(s)] = s
    return batch.to(device)


def adjacent_pairs():
    pairs = [(k, k + 1) for k in range(C.N_BLOCKS - 1)]
    if not C.INCLUDE_B3_B4:
        pairs = [(p, c) for (p, c) in pairs if not (p == 3 and c == 4)]
    return pairs


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--docs", type=int, default=C.N_DOCS, help="docs to sample")
    ap.add_argument("--device", default=None, help="cpu / mps / cuda (default auto)")
    args = ap.parse_args()

    device = args.device or C.pick_device()
    print(f"[01] layer = {C.LAYER}  ({C.SAE_ID})")
    print(f"[01] device = {device}")
    print(f"[01] block structure:\n{U.human_block_table()}\n")

    model = U.load_model(device)
    sae = U.load_sae(device)
    pad_id = model.tokenizer.pad_token_id
    W_dec = sae.W_dec.detach()                    # [D_SAE, d_model]

    # MPS has no float64; a pass this size stays well within float32's exact-int
    # range for counts, and recon sums are only read as ratios afterwards. On
    # CUDA / CPU we keep float64. (str check so "mps:0" / "cuda:5" both work.)
    acc_dtype = torch.float32 if C.is_mps(device) else torch.float64

    print(f"[01] loading {C.DATASET} (first {args.docs} docs) ...")
    ds = load_dataset(C.DATASET, split=f"train[:{args.docs}]")
    texts = [t for t in ds["text"] if isinstance(t, str) and t.strip()]

    # ---- pre-pass: tokenize once, count token ids, build frequency buckets ----
    print("[01] tokenizing + counting token ids for frequency buckets ...")
    seqs = tokenize_docs(model, texts, C.CONTEXT_SIZE)
    vocab = model.cfg.d_vocab
    token_counts = torch.zeros(vocab, dtype=torch.float64)
    for s in seqs:
        token_counts.scatter_add_(0, s.long(), torch.ones(s.numel(), dtype=torch.float64))
    buckets = frequency_buckets(token_counts, C.FREQ_HIGH_MASS, C.FREQ_MID_MASS)  # [vocab]
    buckets_dev = buckets.to(device)
    K = C.N_FREQ_BUCKETS
    for k in range(K):
        print(f"[01]   bucket {k}: {int((buckets == k).sum())} token ids")

    pairs = adjacent_pairs()
    print(f"[01] block pairs: {pairs}")

    def blk_len(b):
        return C.BLOCK_RANGES[b][1] - C.BLOCK_RANGES[b][0]

    # ---- accumulators --------------------------------------------------------
    fire_count = torch.zeros(C.D_SAE, dtype=acc_dtype, device=device)
    total_tokens = 0

    cofire = {pr: torch.zeros(blk_len(pr[0]), blk_len(pr[1]), dtype=acc_dtype, device=device) for pr in pairs}
    cofire_by_bucket = {
        pr: torch.zeros(K, blk_len(pr[0]), blk_len(pr[1]), dtype=acc_dtype, device=device) for pr in pairs
    }
    g_parent_sum = {pr: torch.zeros(blk_len(pr[0]), blk_len(pr[1]), dtype=acc_dtype, device=device) for pr in pairs}

    # child-block indexed (a block appears as child in exactly one pair here)
    child_blocks = sorted({c for (_, c) in pairs})
    err_sum_c = {b: torch.zeros(blk_len(b), dtype=acc_dtype, device=device) for b in child_blocks}
    g_child_sum = {b: torch.zeros(blk_len(b), dtype=acc_dtype, device=device) for b in child_blocks}
    fire_c_by_bucket = {b: torch.zeros(K, blk_len(b), dtype=acc_dtype, device=device) for b in child_blocks}

    within_cofire = {
        b: torch.zeros(blk_len(b), blk_len(b), dtype=acc_dtype, device=device) for b in C.SIBLING_BLOCKS
    }

    # ---- main pass -----------------------------------------------------------
    t0 = time.time()
    n_batches = (len(seqs) + C.BATCH_DOCS - 1) // C.BATCH_DOCS
    for bi in range(n_batches):
        chunk = seqs[bi * C.BATCH_DOCS : (bi + 1) * C.BATCH_DOCS]
        tokens = right_pad(chunk, pad_id, device)          # [b, seq]

        _, cache = model.run_with_cache(
            tokens, stop_at_layer=C.LAYER + 1, names_filter=C.HOOK_NAME
        )
        resid = cache[C.HOOK_NAME]                          # [b, seq, d_model]

        keep = tokens != pad_id                             # [b, seq]
        resid = resid[keep]                                 # [n, d_model]
        tok_ids = tokens[keep]                              # [n]
        tok_bucket = buckets_dev[tok_ids]                   # [n] in {0..K-1}

        feats = sae.encode(resid)                           # [n, D_SAE]
        x_hat = sae.decode(feats)                           # [n, d_model]
        resid_err = resid - x_hat                           # [n, d_model]
        # Cast to acc_dtype: on CUDA/CPU the accumulators are float64 while the
        # model runs in float32, so `fc.T @ err` below would hit a Double-vs-Float
        # mismatch. (On MPS both are float32, which is why this only bit on CUDA.)
        err = (resid_err * resid_err).sum(dim=1).to(acc_dtype)  # [n] base recon error

        fired = (feats > C.FIRE_THRESHOLD).to(acc_dtype)    # [n, D_SAE]
        g = per_token_ablation_gain(feats, resid_err, W_dec).to(acc_dtype)  # [n, D_SAE]

        fire_count += fired.sum(dim=0)
        total_tokens += fired.shape[0]

        # per-bucket row masks (float [n] selector reused across pairs)
        bucket_sel = [(tok_bucket == k).to(acc_dtype) for k in range(K)]

        for (p, c) in pairs:
            fp = fired[:, U.block_slice(p)]                 # [n, P]
            fc = fired[:, U.block_slice(c)]                 # [n, C]
            gp = g[:, U.block_slice(p)]                     # [n, P]

            cofire[(p, c)] += fp.T @ fc                     # [P, C]
            g_parent_sum[(p, c)] += gp.T @ fc               # [P, C] sum over c-firing tokens of g_p
            for k in range(K):
                fck = fc * bucket_sel[k].unsqueeze(1)       # [n, C] child-fire only on bucket-k tokens
                cofire_by_bucket[(p, c)][k] += fp.T @ fck

        for b in child_blocks:
            fc = fired[:, U.block_slice(b)]                 # [n, C]
            gc = g[:, U.block_slice(b)]                     # [n, C]
            err_sum_c[b] += fc.T @ err                      # [C] sum of base err over c-firing tokens
            g_child_sum[b] += (fc * gc).sum(dim=0)          # [C]
            for k in range(K):
                fire_c_by_bucket[b][k] += (fc * bucket_sel[k].unsqueeze(1)).sum(dim=0)

        for b in C.SIBLING_BLOCKS:
            fb = fired[:, U.block_slice(b)]                 # [n, Cb]
            within_cofire[b] += fb.T @ fb                   # [Cb, Cb]

        if bi % 5 == 0 or bi == n_batches - 1:
            dt = time.time() - t0
            print(
                f"[01] batch {bi + 1}/{n_batches} | tokens={total_tokens} "
                f"| {dt:.1f}s | {total_tokens / max(dt, 1e-6):.0f} tok/s"
            )

    # ---- save ----------------------------------------------------------------
    def pk(d):  # {(p,c): tensor} -> {"p->c": cpu tensor}
        return {f"{p}->{c}": v.cpu() for (p, c), v in d.items()}

    out = {
        "fire_count": fire_count.cpu(),
        "total_tokens": int(total_tokens),
        "token_counts": token_counts,
        "buckets": buckets,
        "pairs": pairs,
        "cofire": pk(cofire),
        "cofire_by_bucket": pk(cofire_by_bucket),
        "g_parent_sum": pk(g_parent_sum),
        "err_sum_c": {b: v.cpu() for b, v in err_sum_c.items()},
        "g_child_sum": {b: v.cpu() for b, v in g_child_sum.items()},
        "fire_c_by_bucket": {b: v.cpu() for b, v in fire_c_by_bucket.items()},
        "within_cofire": {b: v.cpu() for b, v in within_cofire.items()},
        "config": {
            "layer": C.LAYER,
            "sae_release": C.SAE_RELEASE,
            "sae_source": C.SAE_SOURCE,
            "sae_id": C.SAE_ID,
            "matryoshka_steps": C.MATRYOSHKA_STEPS,
            "block_ranges": C.BLOCK_RANGES,
            "fire_threshold": C.FIRE_THRESHOLD,
            "n_docs": args.docs,
            "context_size": C.CONTEXT_SIZE,
            "sibling_blocks": C.SIBLING_BLOCKS,
            "freq_high_mass": C.FREQ_HIGH_MASS,
            "freq_mid_mass": C.FREQ_MID_MASS,
        },
    }
    torch.save(out, C.EXP0_STATS_PATH)
    print(f"\n[01] saved -> {C.EXP0_STATS_PATH}")
    print(f"[01] total tokens: {total_tokens}")
    alive = int((fire_count > 0).sum())
    print(f"[01] alive features: {alive}/{C.D_SAE} ({100 * alive / C.D_SAE:.1f}%)")


if __name__ == "__main__":
    main()
