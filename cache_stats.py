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


def keep_mask(tokens, pad_id):
    """[b, seq] bool: which positions enter the statistics.

    Excludes padding AND position 0: tokenize_docs prepends BOS to every row,
    and gemma-2's residual at BOS is an extreme-norm attention-sink outlier —
    every BOS-firing feature would co-fire there by construction, contaminating
    fire/cofire/energy/recon counts (metrics_todo.md T1).
    """
    keep = tokens != pad_id
    keep[:, 0] = False
    return keep


def count_tokens(seqs, vocab):
    """Corpus token counts for the frequency buckets, skipping each row's
    leading BOS — a guaranteed once-per-doc hit that would otherwise be
    forced toward bucket 0 and skew the bucket boundaries."""
    counts = torch.zeros(vocab, dtype=torch.float64)
    for s in seqs:
        body = s[1:].long()
        if body.numel():
            counts.scatter_add_(0, body, torch.ones(body.numel(), dtype=torch.float64))
    return counts


def accumulate_pair_extras(acc, feats_p, feats_c, thr):
    """T2 accumulators for one (parent, child) block pair, one token chunk.

    acc holds (all in acc["energy_total"].dtype):
        energy_cofire [P, C] : sum over c-firing tokens of f_p^2
                               -> Share_energy(c,p) = energy_cofire / energy_total
        union_count   [P]    : parent-firing tokens where >=1 child fires
                               -> exact R_supp(p) (replaces the min(1, sum F) bound)
        union_energy  [P]    : sum of f_p^2 over tokens where >=1 child fires
                               -> R_mass(p)
        energy_total  [P]    : sum of f_p^2 over all tokens
    """
    dt = acc["energy_total"].dtype
    fired_p = (feats_p > thr).to(dt)
    fired_c = (feats_c > thr).to(dt)
    energy_p = feats_p.to(dt) ** 2                     # [n, P]
    any_c = fired_c.amax(dim=1)                        # [n] 1.0 where >=1 child fires
    acc["energy_cofire"] += energy_p.T @ fired_c       # [P, C]
    acc["union_count"] += fired_p.T @ any_c            # [P]
    acc["union_energy"] += energy_p.T @ any_c          # [P]
    acc["energy_total"] += energy_p.sum(dim=0)         # [P]


class TokenCacheWriter:
    """Streams fp16 residuals + sparse latents to disk shards so stage 03
    (S_res probes, parent-conditioned sibling stats, kept-children unions)
    can run without re-touching the model. Row indices are global positions
    in the SAME kept-token stream the statistics are accumulated over."""

    def __init__(self, cache_dir, flush_tokens=200_000):
        import shutil

        # write into a .tmp dir; finalize() swaps it in — a crash mid-run
        # leaves the previous known-good cache untouched
        self.final_dir = cache_dir
        self.dir = cache_dir.with_name(cache_dir.name + ".tmp")
        if self.dir.exists():
            shutil.rmtree(self.dir)                    # never mix runs
        self.dir.mkdir(parents=True)
        self.flush_tokens = flush_tokens
        self.base = 0
        self.shard = 0
        self._reset()

    def _reset(self):
        self.res, self.rows, self.feats, self.vals, self.buf = [], [], [], [], 0

    def add(self, resid, feats, thr):
        nz = (feats > thr).nonzero(as_tuple=False)     # [k, 2] (row, feat)
        self.res.append(resid.detach().to("cpu", torch.float16))
        self.rows.append((nz[:, 0] + self.base + self.buf).to("cpu", torch.int32))
        self.feats.append(nz[:, 1].to("cpu", torch.int32))
        self.vals.append(feats[nz[:, 0], nz[:, 1]].detach().to("cpu", torch.float16))
        self.buf += resid.shape[0]
        if self.buf >= self.flush_tokens:
            self.flush()

    def flush(self):
        if not self.buf:
            return
        torch.save(
            {
                "resid": torch.cat(self.res),
                "rows": torch.cat(self.rows),
                "feats": torch.cat(self.feats),
                "vals": torch.cat(self.vals),
                "base": self.base,
            },
            self.dir / f"shard_{self.shard:04d}.pt",
        )
        self.base += self.buf
        self.shard += 1
        self._reset()

    def finalize(self, extra_meta):
        import json
        import shutil

        self.flush()
        meta = {"total_tokens": self.base, "n_shards": self.shard, **extra_meta}
        (self.dir / "meta.json").write_text(json.dumps(meta, indent=2))
        if self.final_dir.exists():
            shutil.rmtree(self.final_dir)
        self.dir.rename(self.final_dir)
        return meta


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
    from datasets import load_dataset  # lazy: heavy dep, not needed by the helpers

    ds = load_dataset(C.DATASET, split=f"train[:{args.docs}]")
    texts = [t for t in ds["text"] if isinstance(t, str) and t.strip()]

    # ---- pre-pass: tokenize once, count token ids, build frequency buckets ----
    print("[01] tokenizing + counting token ids for frequency buckets ...")
    seqs = tokenize_docs(model, texts, C.CONTEXT_SIZE)
    vocab = model.cfg.d_vocab
    token_counts = count_tokens(seqs, vocab)          # BOS excluded (T1)
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

    # within-block co-firing for sibling stats (SIBLING_BLOCKS) AND the in-block
    # same-level edge analysis (IN_BLOCK_BLOCKS) — union so B0 gets cached too.
    within_blocks = sorted(set(C.SIBLING_BLOCKS) | set(getattr(C, "IN_BLOCK_BLOCKS", [])))
    within_cofire = {
        b: torch.zeros(blk_len(b), blk_len(b), dtype=acc_dtype, device=device) for b in within_blocks
    }

    # T2: energy shares + exact joint-child unions, one accumulator set per pair
    pair_extras = {
        pr: {
            "energy_cofire": torch.zeros(blk_len(pr[0]), blk_len(pr[1]), dtype=acc_dtype, device=device),
            "union_count": torch.zeros(blk_len(pr[0]), dtype=acc_dtype, device=device),
            "union_energy": torch.zeros(blk_len(pr[0]), dtype=acc_dtype, device=device),
            "energy_total": torch.zeros(blk_len(pr[0]), dtype=acc_dtype, device=device),
        }
        for pr in pairs
    }

    # T7 groundwork: token-level caches for the model-free second pass
    cache_writer = TokenCacheWriter(C.TOKEN_CACHE_DIR) if C.CACHE_RESIDUALS else None

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

        keep = keep_mask(tokens, pad_id)                    # [b, seq] pad + BOS excluded
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

            accumulate_pair_extras(                          # T2: energy + exact unions
                pair_extras[(p, c)],
                feats[:, U.block_slice(p)],
                feats[:, U.block_slice(c)],
                C.FIRE_THRESHOLD,
            )

        for b in child_blocks:
            fc = fired[:, U.block_slice(b)]                 # [n, C]
            gc = g[:, U.block_slice(b)]                     # [n, C]
            err_sum_c[b] += fc.T @ err                      # [C] sum of base err over c-firing tokens
            g_child_sum[b] += (fc * gc).sum(dim=0)          # [C]
            for k in range(K):
                fire_c_by_bucket[b][k] += (fc * bucket_sel[k].unsqueeze(1)).sum(dim=0)

        for b in within_blocks:
            fb = fired[:, U.block_slice(b)]                 # [n, Cb]
            within_cofire[b] += fb.T @ fb                   # [Cb, Cb]

        if cache_writer is not None:
            cache_writer.add(resid, feats, C.FIRE_THRESHOLD)

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
        "schema_version": 2,                    # v2: BOS excluded + T2 extras
        "fire_count": fire_count.cpu(),
        "total_tokens": int(total_tokens),
        "token_counts": token_counts,
        "buckets": buckets,
        "pairs": pairs,
        "energy_cofire": pk({pr: v["energy_cofire"] for pr, v in pair_extras.items()}),
        "union_count": pk({pr: v["union_count"] for pr, v in pair_extras.items()}),
        "union_energy": pk({pr: v["union_energy"] for pr, v in pair_extras.items()}),
        "energy_total": pk({pr: v["energy_total"] for pr, v in pair_extras.items()}),
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
            "bos_excluded": True,
            "min_joint": C.MIN_JOINT,
        },
    }
    tmp = C.EXP0_STATS_PATH.with_suffix(".pt.tmp")     # atomic: never clobber a
    torch.save(out, tmp)                               # good stats file mid-write
    tmp.replace(C.EXP0_STATS_PATH)
    if cache_writer is not None:
        meta = cache_writer.finalize({"d_model": int(model.cfg.d_model), "layer": C.LAYER})
        print(f"[01] token cache: {meta['total_tokens']} tokens, {meta['n_shards']} shards -> {C.TOKEN_CACHE_DIR}")
    print(f"\n[01] saved -> {C.EXP0_STATS_PATH}")
    print(f"[01] total tokens: {total_tokens}")
    alive = int((fire_count > 0).sum())
    print(f"[01] alive features: {alive}/{C.D_SAE} ({100 * alive / C.D_SAE:.1f}%)")


if __name__ == "__main__":
    main()
