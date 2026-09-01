"""On-disk token cache: writer (fills it during the model pass) and reader.

The writer streams fp16 residuals + sparse latents to sharded files so the later
model-free passes (S_res probes, parent-conditioned siblings, kept-children
unions) never re-touch the model. The reader loads those shards once and serves
per-feature firing masks and row-chunked dense slices.

Needs:  nothing (pure torch + a cache directory)
Writes: <cache_dir>/shard_*.pt + meta.json  (writer); reads them back (reader)
"""

from __future__ import annotations

import json

import torch

import config as C

CHUNK = 65_536          # tokens per dense chunk when scanning the sparse cache


# ---------------------------------------------------------------------------
# Reader
# ---------------------------------------------------------------------------
class TokenCache:
    """Loads the cache shards once; serves per-feature firing masks and
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

    def feature_vals(self, f: int) -> tuple[torch.Tensor, torch.Tensor]:
        """(rows, activation values) for feature f, row-aligned."""
        lo, hi = int(self.f_bounds[f]), int(self.f_bounds[f + 1])
        return self.f_rows[lo:hi], self.f_vals[lo:hi]

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
# Writer
# ---------------------------------------------------------------------------
class TokenCacheWriter:
    """Streams fp16 residuals + sparse latents to disk shards so the later
    model-free passes (S_res probes, parent-conditioned sibling stats,
    kept-children unions) can run without re-touching the model. Row indices are
    global positions in the SAME kept-token stream the statistics are
    accumulated over."""

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
