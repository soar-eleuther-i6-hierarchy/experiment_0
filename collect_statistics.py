"""
Cache the richer statistics the hierarchy metrics need.

The only stage that touches the model. It streams a residual stream through a
Matryoshka SAE once and accumulates, per adjacent block pair, the co-firing /
reconstruction / sibling statistics the metrics later read (the exact tensors and
shapes are documented at the accumulator block in `collect()`). Token-frequency
buckets need corpus-wide counts, so every doc is tokenized up front (no model) to
build the buckets before the single model pass.

Two halves, split so the second can be reused:

  main()      loads gemma-2-2b, the released Matryoshka SAE and pile-10k.
              Everything source-specific lives here.
  collect()   the accumulation, which knows nothing about where the model, the
              SAE or the tokens came from. Its `cfg` argument carries the block
              structure, so an adapter can pass its own instead of gemma's and
              get a stats file run_metrics.py reads unmodified.

tests/test_collect_generic.py exercises collect() on a 28-feature stub, which
is what keeps that reuse honest.

Needs:  model + SAE, GPU
Writes: outputs/layer_NN/exp0_stats.pt  (+ token_cache/ when CACHE_RESIDUALS)
Run:    python3 collect_statistics.py            # full (config N_DOCS)
        python3 collect_statistics.py --docs 16  # quick smoke slice
"""

from __future__ import annotations

import argparse
import time
from datetime import datetime

import torch

import config as C
from utils import sae_utils as U
from utils.io import TokenCacheWriter
from metrics.reconstruction import per_token_ablation_gain
from metrics.token_control import frequency_buckets, local_frequency_buckets


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
    fire/cofire/energy/recon counts.
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
    """Energy / joint-child accumulators for one (parent, child) block pair,
    one token chunk.

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


def adjacent_pairs(cfg=C):
    pairs = [(k, k + 1) for k in range(cfg.N_BLOCKS - 1)]
    if not cfg.INCLUDE_B3_B4:
        pairs = [(p, c) for (p, c) in pairs if not (p == 3 and c == 4)]
    return pairs


@torch.no_grad()
def collect(model, sae, seqs, *, device, cfg=C, out_path=None, extra_config=None, pad_id=None):
    """Accumulate every statistic the metrics need, and save them.

    This is the source-agnostic half of stage 01: it takes a model, an SAE and a
    list of token sequences, and knows nothing about where any of them came from.
    `main()` below feeds it gemma-2-2b + the released Matryoshka SAE + pile-10k;
    an adapter feeds it a PCFG transformer + a locally trained SAE + corpus.bin,
    and gets a stats file the metrics read without modification.

    `cfg` supplies the block structure and thresholds. It defaults to the gemma
    config module, and anything with the same attribute names works — which is
    what lets an adapter pass its own MATRYOSHKA_STEPS / BLOCK_RANGES / D_SAE
    instead of gemma's.

    Args:
        model:        HookedTransformer; activations are read at cfg.HOOK_NAME.
        sae:          anything with .encode / .decode / .W_dec.
        seqs:         list of 1-D LongTensors, one per document, BOS already
                      prepended if the source wants one (position 0 is dropped).
        out_path:     where to write; defaults to cfg.EXP0_STATS_PATH.
        extra_config: merged into the saved `config` dict — provenance the caller
                      knows and this function cannot (n_docs, grammar hash, ...).
        pad_id:       padding id; defaults to the model tokenizer's. Sources with
                      no tokenizer (a PCFG corpus is raw ids) must pass one.
    """
    if pad_id is None:
        pad_id = model.tokenizer.pad_token_id
    W_dec = sae.W_dec.detach()                    # [D_SAE, d_model]

    # MPS has no float64; a pass this size stays well within float32's exact-int
    # range for counts, and recon sums are only read as ratios afterwards. On
    # CUDA / CPU we keep float64. (str check so "mps:0" / "cuda:5" both work.)
    acc_dtype = torch.float32 if C.is_mps(device) else torch.float64

    # ---- pre-pass: count token ids, build frequency buckets ------------------
    vocab = model.cfg.d_vocab
    token_counts = count_tokens(seqs, vocab)          # BOS excluded
    buckets = frequency_buckets(token_counts, cfg.FREQ_HIGH_MASS, cfg.FREQ_MID_MASS)  # [vocab]
    buckets_dev = buckets.to(device)
    K = cfg.N_FREQ_BUCKETS
    for k in range(K):
        print(f"[01]   bucket {k}: {int((buckets == k).sum())} token ids")

    pairs = adjacent_pairs(cfg)
    print(f"[01] block pairs: {pairs}")

    # A block is a SET of feature indices. Matryoshka's happen to be contiguous
    # prefixes, and nothing in the metrics requires that -- every one of them is a
    # matrix product over selected columns. Sources whose groups are not contiguous
    # (the trained toy indexes by which true feature each latent recovered, giving
    # lists like [0, 3, 8]) declare cfg.BLOCK_INDICES instead of BLOCK_RANGES.
    block_indices = getattr(cfg, "BLOCK_INDICES", None)
    if block_indices is not None:
        block_indices = [torch.as_tensor(ix, dtype=torch.long, device=device) for ix in block_indices]

    def blk_len(b):
        if block_indices is not None:
            return int(block_indices[b].numel())
        return cfg.BLOCK_RANGES[b][1] - cfg.BLOCK_RANGES[b][0]

    def blk_slice(b):
        # NOT utils.sae_utils.block_slice: that one reads the gemma config module
        # globally, so it would slice an adapter's dictionary with gemma's ranges.
        # An index tensor selects the same way a slice does, at the cost of a gather
        # instead of a view -- paid only by sources that need it.
        if block_indices is not None:
            return block_indices[b]
        start, end = cfg.BLOCK_RANGES[b]
        return slice(start, end)

    # ---- accumulators --------------------------------------------------------
    fire_count = torch.zeros(cfg.D_SAE, dtype=acc_dtype, device=device)
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

    # A second, parallel set bucketed by frequency WITHIN each window rather than
    # across the corpus. Both are accumulated in the same pass so the two readings
    # are over identical edges and identical tokens — the only difference is what
    # counts as "frequent". Off by default; the extra keys are optional in the
    # stats contract, so a file without them still grades.
    local_freq = getattr(cfg, "LOCAL_FREQ_BUCKETS", False)
    cofire_by_local = (
        {pr: torch.zeros(K, blk_len(pr[0]), blk_len(pr[1]), dtype=acc_dtype, device=device) for pr in pairs}
        if local_freq else None
    )
    fire_c_by_local = (
        {b: torch.zeros(K, blk_len(b), dtype=acc_dtype, device=device) for b in child_blocks}
        if local_freq else None
    )

    # within-block co-firing for sibling stats (SIBLING_BLOCKS) AND the in-block
    # same-level edge analysis (IN_BLOCK_BLOCKS) — union so B0 gets cached too.
    within_blocks = sorted(set(cfg.SIBLING_BLOCKS) | set(getattr(cfg, "IN_BLOCK_BLOCKS", [])))
    within_cofire = {
        b: torch.zeros(blk_len(b), blk_len(b), dtype=acc_dtype, device=device) for b in within_blocks
    }

    # energy shares + exact joint-child unions, one accumulator set per pair
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
    cache_writer = TokenCacheWriter(cfg.TOKEN_CACHE_DIR) if cfg.CACHE_RESIDUALS else None

    # ---- main pass -----------------------------------------------------------
    t0 = time.time()
    n_batches = (len(seqs) + cfg.BATCH_DOCS - 1) // cfg.BATCH_DOCS
    for bi in range(n_batches):
        chunk = seqs[bi * cfg.BATCH_DOCS : (bi + 1) * cfg.BATCH_DOCS]
        tokens = right_pad(chunk, pad_id, device)          # [b, seq]

        _, cache = model.run_with_cache(
            tokens, stop_at_layer=cfg.LAYER + 1, names_filter=cfg.HOOK_NAME
        )
        resid = cache[cfg.HOOK_NAME]                        # [b, seq, d_model]

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

        fired = (feats > cfg.FIRE_THRESHOLD).to(acc_dtype)  # [n, D_SAE]
        g = per_token_ablation_gain(feats, resid_err, W_dec).to(acc_dtype)  # [n, D_SAE]

        fire_count += fired.sum(dim=0)
        total_tokens += fired.shape[0]

        # per-bucket row masks (float [n] selector reused across pairs)
        bucket_sel = [(tok_bucket == k).to(acc_dtype) for k in range(K)]

        # Same positions, bucketed by frequency inside their own window instead of
        # across the corpus. Accumulated in the same pass so the two readings differ
        # in nothing but the definition of "frequent".
        local_sel = None
        if local_freq:
            tok_bucket_local = local_frequency_buckets(
                tokens, keep, vocab, cfg.FREQ_HIGH_MASS, cfg.FREQ_MID_MASS
            )[keep]                                         # [n]
            local_sel = [(tok_bucket_local == k).to(acc_dtype) for k in range(K)]

        for (p, c) in pairs:
            fp = fired[:, blk_slice(p)]                     # [n, P]
            fc = fired[:, blk_slice(c)]                     # [n, C]
            gp = g[:, blk_slice(p)]                         # [n, P]

            cofire[(p, c)] += fp.T @ fc                     # [P, C]
            g_parent_sum[(p, c)] += gp.T @ fc               # [P, C] sum over c-firing tokens of g_p
            for k in range(K):
                fck = fc * bucket_sel[k].unsqueeze(1)       # [n, C] child-fire only on bucket-k tokens
                cofire_by_bucket[(p, c)][k] += fp.T @ fck
                if local_sel is not None:
                    cofire_by_local[(p, c)][k] += fp.T @ (fc * local_sel[k].unsqueeze(1))

            accumulate_pair_extras(                          # energy + exact unions
                pair_extras[(p, c)],
                feats[:, blk_slice(p)],
                feats[:, blk_slice(c)],
                cfg.FIRE_THRESHOLD,
            )

        for b in child_blocks:
            fc = fired[:, blk_slice(b)]                     # [n, C]
            gc = g[:, blk_slice(b)]                         # [n, C]
            err_sum_c[b] += fc.T @ err                      # [C] sum of base err over c-firing tokens
            g_child_sum[b] += (fc * gc).sum(dim=0)          # [C]
            for k in range(K):
                fire_c_by_bucket[b][k] += (fc * bucket_sel[k].unsqueeze(1)).sum(dim=0)
                if local_sel is not None:
                    fire_c_by_local[b][k] += (fc * local_sel[k].unsqueeze(1)).sum(dim=0)

        for b in within_blocks:
            fb = fired[:, blk_slice(b)]                     # [n, Cb]
            within_cofire[b] += fb.T @ fb                   # [Cb, Cb]

        if cache_writer is not None:
            cache_writer.add(resid, feats, cfg.FIRE_THRESHOLD)

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
        "schema_version": 2,                    # v2: BOS excluded + energy/union extras
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
        # Optional: present only when cfg.LOCAL_FREQ_BUCKETS is on. Same shapes as
        # their global counterparts, bucketed within each window instead.
        **(
            {
                "cofire_by_local_bucket": pk(cofire_by_local),
                "fire_c_by_local_bucket": {b: v.cpu() for b, v in fire_c_by_local.items()},
            }
            if local_freq else {}
        ),
        "config": {
            "layer": cfg.LAYER,
            "sae_release": cfg.SAE_RELEASE,
            "sae_source": cfg.SAE_SOURCE,
            "sae_id": cfg.SAE_ID,
            "matryoshka_steps": cfg.MATRYOSHKA_STEPS,
            "block_ranges": cfg.BLOCK_RANGES,
            "fire_threshold": cfg.FIRE_THRESHOLD,
            "context_size": cfg.CONTEXT_SIZE,
            "sibling_blocks": cfg.SIBLING_BLOCKS,
            "freq_high_mass": cfg.FREQ_HIGH_MASS,
            "freq_mid_mass": cfg.FREQ_MID_MASS,
            "bos_excluded": True,
            "min_joint": cfg.MIN_JOINT,
            "local_freq_buckets": bool(local_freq),
            # Present only for sources whose blocks are not contiguous. Readers must
            # prefer it over block_ranges when it is there.
            **({"block_indices": [ix.cpu().tolist() for ix in block_indices]}
               if block_indices is not None else {}),
            **(extra_config or {}),
        },
    }
    out_path = out_path or cfg.EXP0_STATS_PATH
    tmp = out_path.with_suffix(".pt.tmp")               # atomic: never clobber a
    torch.save(out, tmp)                               # good stats file mid-write
    tmp.replace(out_path)
    if cache_writer is not None:
        meta = cache_writer.finalize({"d_model": int(model.cfg.d_model), "layer": cfg.LAYER})
        print(f"[01] token cache: {meta['total_tokens']} tokens, {meta['n_shards']} shards -> {cfg.TOKEN_CACHE_DIR}")
    print(f"\n[01] saved -> {out_path}")
    print(f"[01] total tokens: {total_tokens}")
    alive = int((fire_count > 0).sum())
    print(f"[01] alive features: {alive}/{cfg.D_SAE} ({100 * alive / cfg.D_SAE:.1f}%)")
    return out


def main():
    """Stage 01 for gemma-2-2b: load the model, the released SAE and pile-10k.

    Everything source-specific lives here; the accumulation is in collect().
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--docs", type=int, default=C.N_DOCS, help="docs to sample")
    ap.add_argument("--device", default=None, help="cpu / mps / cuda (default auto)")
    args = ap.parse_args()

    device = args.device or C.pick_device()
    print(f"[01] layer = {C.LAYER}  ({C.SAE_ID})")
    print(f"[01] device = {device}")

    # Stage 01 is what starts a run, so this is where the previous one is put
    # aside. Later stages write into the same directory on purpose.
    stamp = datetime.now().strftime("%Y-%m-%dT%H-%M")
    kept = C.archive_run_dir(stamp)
    if kept is not None:
        print(f"[01] previous run archived -> {kept}")
    print(f"[01] block structure:\n{U.human_block_table()}\n")

    model = U.load_model(device)
    sae = U.load_sae(device)

    print(f"[01] loading {C.DATASET} (first {args.docs} docs) ...")
    from datasets import load_dataset  # lazy: heavy dep, not needed by the helpers

    ds = load_dataset(C.DATASET, split=f"train[:{args.docs}]")
    texts = [t for t in ds["text"] if isinstance(t, str) and t.strip()]

    print("[01] tokenizing + counting token ids for frequency buckets ...")
    seqs = tokenize_docs(model, texts, C.CONTEXT_SIZE)

    collect(model, sae, seqs, device=device, cfg=C, extra_config={"n_docs": args.docs})


if __name__ == "__main__":
    main()
