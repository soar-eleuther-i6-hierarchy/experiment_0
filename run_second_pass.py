"""
Stage 03 - the "second pass": a model-free sweep over the token caches for the
metrics that need per-token detail the stage-01 count matrices cannot provide.
Nothing here touches the model again — it all runs off the token cache (fp16
residuals + sparse latents) written by cache_stats.py:

    1. S_res, probe-based and RANK-scored (metrics/sres.py) on the shortlist
       of edges that survive coverage + the independence null. Self-labeled
       probes — see the circularity caveat in metrics/sres.py.
    2. Parent-conditioned sibling redundancy for flagged superparents
       (landscape Rev. 2.1: Jaccard restricted to the parent's firing set).
    3. Exact joint-child union over the KEPT children only (stage 01 streams
       the all-children union; the kept-children one depends on the edge set,
       which exists only after stage 02).

Run (after cache_stats.py and run_metrics.py, on the server):
    python3 run_second_pass.py                # all pairs
    python3 run_second_pass.py --pairs 0->1   # subset
Output:
    outputs/layer_NN/second_pass.json  (+ "second_pass" key merged into
    metrics_report.json when present)
"""

from __future__ import annotations

import argparse
import json

import torch

import config as C
import sae_utils as U
from metrics import (
    coverage_legs,
    find_superparents,
    independence_scores,
    keep_edges,
    parent_conditioned_redundancy,
    sres_rank_check,
    train_probe,
)

CHUNK = 65_536          # tokens per dense chunk when scanning the sparse cache


# ---------------------------------------------------------------------------
# Token cache access
# ---------------------------------------------------------------------------
class TokenCache:
    """Loads the stage-01 shards once; serves per-feature firing masks and
    row-chunked dense slices of any feature range."""

    def __init__(self, cache_dir):
        meta = json.loads((cache_dir / "meta.json").read_text())
        self.n_tokens = int(meta["total_tokens"])
        self.d_model = int(meta["d_model"])
        res, rows, feats, vals = [], [], [], []
        for i in range(int(meta["n_shards"])):
            sh = torch.load(cache_dir / f"shard_{i:04d}.pt", weights_only=True)
            res.append(sh["resid"])
            rows.append(sh["rows"].long())
            feats.append(sh["feats"].long())
            vals.append(sh["vals"])
        self.resid = torch.cat(res)                       # [N, d] fp16
        rows, feats, vals = torch.cat(rows), torch.cat(feats), torch.cat(vals)
        # feat-sorted view -> O(log) per-feature masks
        order = torch.argsort(feats, stable=True)
        self.f_rows, self.f_feats, self.f_vals = rows[order], feats[order], vals[order]
        self.f_bounds = torch.searchsorted(
            self.f_feats, torch.arange(C.D_SAE + 1, dtype=torch.long)
        )
        # row-sorted view -> chunked dense slices
        order = torch.argsort(rows, stable=True)
        self.r_rows, self.r_feats, self.r_vals = rows[order], feats[order], vals[order]
        self.r_bounds = torch.searchsorted(
            self.r_rows, torch.arange(0, self.n_tokens + CHUNK, CHUNK, dtype=torch.long)
        )

    def feature_rows(self, f: int) -> torch.Tensor:
        lo, hi = int(self.f_bounds[f]), int(self.f_bounds[f + 1])
        return self.f_rows[lo:hi]

    def feature_mask(self, f: int) -> torch.Tensor:
        m = torch.zeros(self.n_tokens, dtype=torch.bool)
        m[self.feature_rows(f)] = True
        return m

    def chunks_dense(self, g0: int, g1: int, values: bool = False):
        """Yield (row_lo, dense [chunk, g1-g0]) for global feature range [g0, g1)."""
        n_chunks = len(self.r_bounds) - 1
        for ci in range(n_chunks):
            lo, hi = int(self.r_bounds[ci]), int(self.r_bounds[ci + 1])
            row_lo = ci * CHUNK
            n = min(CHUNK, self.n_tokens - row_lo)
            if n <= 0:
                break
            r = self.r_rows[lo:hi] - row_lo
            f = self.r_feats[lo:hi]
            sel = (f >= g0) & (f < g1)
            dense = torch.zeros(n, g1 - g0, dtype=torch.float32)
            if values:
                dense[r[sel], f[sel] - g0] = self.r_vals[lo:hi][sel].float()
            else:
                dense[r[sel], f[sel] - g0] = 1.0
            yield row_lo, dense


# ---------------------------------------------------------------------------
# Per-pair work
# ---------------------------------------------------------------------------
def rebuild_edges(stats, p_blk, c_blk):
    """Same edge set + shortlist as run_metrics (single source: metrics/)."""
    key = f"{p_blk}->{c_blk}"
    fire = stats["fire_count"].double()
    total = int(stats["total_tokens"])
    p0, p1 = C.BLOCK_RANGES[p_blk]
    c0, c1 = C.BLOCK_RANGES[c_blk]
    fire_p, fire_c = fire[p0:p1], fire[c0:c1]
    cofire = stats["cofire"][key].double()
    R, _ = coverage_legs(cofire, fire_p, fire_c)
    edge_mask = keep_edges(R, fire_p, fire_c, C.EDGE_TAU, C.MIN_FIRE_COUNT,
                           cofire=cofire, min_joint=C.MIN_JOINT)
    null = independence_scores(cofire, fire_p, fire_c, total, C.MIN_JOINT)
    shortlist = edge_mask & (null["pmi"] > 0.0)
    return edge_mask, shortlist, fire_p, fire_c, R


def sres_for_pair(cache, W_dec, stats, p_blk, c_blk, device):
    """Probe-based S_res over the shortlist. One probe per unique child."""
    _, shortlist, _, fire_c, _ = rebuild_edges(stats, p_blk, c_blk)
    p0, _ = C.BLOCK_RANGES[p_blk]
    c0, _ = C.BLOCK_RANGES[c_blk]

    child_locals = torch.nonzero(shortlist.any(dim=0)).flatten()
    child_locals = child_locals[fire_c[child_locals] >= C.MIN_PROBE_POS]
    if child_locals.numel() > C.SRES_MAX_CHILDREN_PER_PAIR:
        print(f"[03] {p_blk}->{c_blk}: capping probes to "
              f"{C.SRES_MAX_CHILDREN_PER_PAIR}/{child_locals.numel()} children (by fire count)")
        top = torch.argsort(fire_c[child_locals], descending=True)
        child_locals = child_locals[top[: C.SRES_MAX_CHILDREN_PER_PAIR]]

    resid = cache.resid.to(device)
    Wd = W_dec.to(device)
    results, n_pass, n_untestable_children = [], 0, 0
    for i, cl in enumerate(child_locals.tolist()):
        gc = c0 + cl
        pos = cache.feature_mask(gc).to(device)
        probe = train_probe(resid, pos, seed=gc,
                            neg_ratio=C.SRES_NEG_RATIO,
                            max_tokens=C.SRES_MAX_PROBE_TOKENS,
                            min_neg=C.SRES_MIN_NEG)
        if probe is None:                                 # too few negatives -> untestable
            n_untestable_children += 1
            continue
        corr = (probe @ Wd.T).cpu()                       # [D_SAE]
        for pl in torch.nonzero(shortlist[:, cl]).flatten().tolist():
            gp = p0 + pl
            ok, detail = sres_rank_check(corr, gp, gc, C.SRES_RANK_TOP_K)
            n_pass += int(ok)
            results.append({"parent": gp, "child": gc, "pass": ok, **detail})
        if (i + 1) % 50 == 0:
            print(f"[03]   {p_blk}->{c_blk}: {i + 1}/{child_locals.numel()} probes")
    return {
        "n_shortlist_edges": int(shortlist.sum()),
        "n_probed_children": int(child_locals.numel()),
        "n_untestable_children": n_untestable_children,   # skipped: < SRES_MIN_NEG negatives
        "n_edges_scored": len(results),
        "n_pass": n_pass,
        "frac_pass": n_pass / len(results) if results else 0.0,
        "edges": results,
    }


def conditioned_redundancy_for_pair(cache, stats, p_blk, c_blk):
    """Parent-conditioned sibling Jaccard for this pair's flagged superparents."""
    edge_mask, _, fire_p, fire_c, _ = rebuild_edges(stats, p_blk, c_blk)
    p0, _ = C.BLOCK_RANGES[p_blk]
    c0, _ = C.BLOCK_RANGES[c_blk]
    total = int(stats["total_tokens"])
    sps = find_superparents(edge_mask, fire_p, total,
                            C.SUPERPARENT_OUTDEG_FRAC, C.SUPERPARENT_FIRE_FRAC)
    out = []
    for sp in sps:
        pl = sp["parent_local"]
        kids = torch.nonzero(edge_mask[pl]).flatten()
        if kids.numel() > 512:                            # cap children per parent so the pairwise Jaccard stays cheap
            top = torch.argsort(fire_c[kids], descending=True)[:512]
            kids = kids[top]
        fires_p = cache.feature_mask(p0 + pl)
        kid_masks = torch.stack([cache.feature_mask(c0 + int(k)) for k in kids], dim=1)
        red = parent_conditioned_redundancy(fires_p, kid_masks)
        out.append({**sp, "parent_global": p0 + pl,
                    "n_kids_scored": int(kids.numel()),
                    "conditioned_redundancy": red})
    return out


def kept_union_for_pair(cache, stats, p_blk, c_blk):
    """Exact R_supp / R_mass over the KEPT children only (chunked scan)."""
    edge_mask, _, fire_p, _, _ = rebuild_edges(stats, p_blk, c_blk)
    key = f"{p_blk}->{c_blk}"
    p0, p1 = C.BLOCK_RANGES[p_blk]
    c0, c1 = C.BLOCK_RANGES[c_blk]
    K = edge_mask.double()                                # [P, C]
    union_count = torch.zeros(p1 - p0, dtype=torch.float64)
    union_energy = torch.zeros(p1 - p0, dtype=torch.float64)
    child_chunks = cache.chunks_dense(c0, c1)
    parent_chunks = cache.chunks_dense(p0, p1, values=True)
    for (_, fc), (_, ep) in zip(child_chunks, parent_chunks):
        any_kept = (fc @ K.T.float() > 0).double()        # [n, P]
        e = (ep.double() ** 2)                            # [n, P]
        union_count += ((e > 0).double() * any_kept).sum(dim=0)
        union_energy += (e * any_kept).sum(dim=0)
    # denominator from stage 01 (over ALL tokens) so r_mass_kept is directly
    # comparable with the all-children r_mass in the stage-02 report
    energy_total = stats["energy_total"][key].double()
    has = edge_mask.any(dim=1)
    r_supp_kept = union_count / fire_p.clamp(min=1.0)
    r_mass_kept = union_energy / energy_total.clamp(min=1e-12)
    return {
        "r_supp_kept_mean": float(r_supp_kept[has].mean()) if has.any() else float("nan"),
        "r_mass_kept_mean": float(r_mass_kept[has].mean()) if has.any() else float("nan"),
        "per_parent": [
            {"parent_global": p0 + int(p), "r_supp_kept": float(r_supp_kept[p]),
             "r_mass_kept": float(r_mass_kept[p])}
            for p in torch.nonzero(has).flatten().tolist()
        ],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", nargs="*", default=None, help='e.g. 0->1 1->2')
    ap.add_argument("--device", default=None)
    ap.add_argument("--skip-sres", action="store_true")
    args = ap.parse_args()

    device = args.device or C.pick_device()
    print(f"[03] layer = {C.LAYER}  device = {device}")
    stats = torch.load(C.EXP0_STATS_PATH, weights_only=False)
    if not (C.TOKEN_CACHE_DIR / "meta.json").exists():
        raise SystemExit(f"[03] no token cache at {C.TOKEN_CACHE_DIR} - rerun cache_stats.py")
    cache = TokenCache(C.TOKEN_CACHE_DIR)
    print(f"[03] token cache: {cache.n_tokens} tokens")

    sae = U.load_sae("cpu")
    W_dec = sae.W_dec.detach().float()                    # [D_SAE, d]

    pairs = stats["pairs"]
    if args.pairs:
        want = set(args.pairs)
        pairs = [pr for pr in pairs if f"{pr[0]}->{pr[1]}" in want]

    from run_metrics import json_safe

    # UPDATE any existing second_pass.json rather than replace it, so a
    # partial --pairs rerun never wipes other pairs' results.
    report = {}
    if C.SECOND_PASS_PATH.exists():
        report = json.loads(C.SECOND_PASS_PATH.read_text())

    for (p, c) in pairs:
        key = f"{p}->{c}"
        print(f"[03] pair {key}")
        entry = {}
        entry["kept_union"] = kept_union_for_pair(cache, stats, p, c)
        entry["superparent_conditioned_redundancy"] = conditioned_redundancy_for_pair(cache, stats, p, c)
        if not args.skip_sres:
            entry["sres"] = sres_for_pair(cache, W_dec, stats, p, c, device)
        report[key] = entry
        # flush after EVERY pair — a crash later never loses finished pairs
        C.SECOND_PASS_PATH.write_text(json.dumps(json_safe(report), indent=2))
        ku = entry["kept_union"]
        print(f"[03]   R_supp_kept mean {ku['r_supp_kept_mean']:.3f} | "
              f"R_mass_kept mean {ku['r_mass_kept_mean']:.3f} | "
              + (f"S_res pass {entry['sres']['n_pass']}/{entry['sres']['n_edges_scored']}"
                 if not args.skip_sres else "S_res skipped"))

    print(f"[03] wrote {C.SECOND_PASS_PATH}")
    if C.METRICS_JSON_PATH.exists():                      # merge for one-stop reading
        full = json.loads(C.METRICS_JSON_PATH.read_text())
        merged = dict(full.get("second_pass") or {})
        merged.update({
            k: {kk: (vv if kk != "sres" else {x: y for x, y in vv.items() if x != "edges"})
                for kk, vv in v.items()}
            for k, v in report.items()
        })
        full["second_pass"] = merged
        C.METRICS_JSON_PATH.write_text(json.dumps(json_safe(full), indent=2))
        print(f"[03] merged summary into {C.METRICS_JSON_PATH}")


if __name__ == "__main__":
    main()
