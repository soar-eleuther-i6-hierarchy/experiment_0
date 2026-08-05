"""Prove collect() is source-agnostic: run it on a model that is not gemma.

Stage 01 used to be one function that loaded gemma-2-2b, the released Matryoshka
SAE and pile-10k, then accumulated statistics from them. collect() is the second
half of that, split out so an adapter can feed it a PCFG transformer or a trained
toy SAE instead.

This test is what makes "source-agnostic" a checked property rather than a claim.
It builds a tiny stub model, a tiny stub SAE and a config with its own block
structure -- 28 features in 3 blocks, not gemma's 32768 in 5 -- and asserts the
output is a well-formed stats file.

It needs no network, no GPU and no model download; it runs in about a second.

    python3 -m validation.test_collect_generic
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import torch

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from collect_statistics import adjacent_pairs, collect  # noqa: E402

HOOK = "blocks.1.hook_resid_post"
D_MODEL = 8
D_VOCAB = 20
STEPS = [4, 12, 28]  # nothing like gemma's [128, 512, 2048, 8192, 32768]


def block_ranges(steps):
    out, prev = [], 0
    for s in steps:
        out.append((prev, s))
        prev = s
    return out


class StubModel:
    """The slice of HookedTransformer that collect() actually touches."""

    def __init__(self, seed=0):
        g = torch.Generator().manual_seed(seed)
        self.cfg = SimpleNamespace(d_vocab=D_VOCAB, d_model=D_MODEL)
        self.tokenizer = SimpleNamespace(pad_token_id=0)
        # A fixed per-token residual, so the pass is deterministic and every
        # statistic is a function of the token stream alone.
        self._emb = torch.randn(D_VOCAB, D_MODEL, generator=g)

    def run_with_cache(self, tokens, stop_at_layer=None, names_filter=None):
        return None, {HOOK: self._emb[tokens]}


class StubSAE:
    """Encoder/decoder pair with a nested block structure, ReLU-sparse."""

    def __init__(self, d_sae, seed=0):
        g = torch.Generator().manual_seed(seed + 1)
        self.W_enc = torch.randn(D_MODEL, d_sae, generator=g) * 0.5
        self.W_dec = torch.randn(d_sae, D_MODEL, generator=g) * 0.5
        self.b_enc = torch.full((d_sae,), -0.15)

    def encode(self, x):
        return torch.relu(x @ self.W_enc + self.b_enc)

    def decode(self, f):
        return f @ self.W_dec


def make_cfg(out_dir: Path):
    ranges = block_ranges(STEPS)
    return SimpleNamespace(
        LAYER=1,
        HOOK_NAME=HOOK,
        MATRYOSHKA_STEPS=STEPS,
        BLOCK_RANGES=ranges,
        N_BLOCKS=len(ranges),
        D_SAE=STEPS[-1],
        INCLUDE_B3_B4=False,
        FIRE_THRESHOLD=1e-3,
        BATCH_DOCS=4,
        CONTEXT_SIZE=16,
        SIBLING_BLOCKS=[1, 2],
        IN_BLOCK_BLOCKS=[0, 1],
        N_FREQ_BUCKETS=3,
        FREQ_HIGH_MASS=0.50,
        FREQ_MID_MASS=0.40,
        MIN_JOINT=2,
        CACHE_RESIDUALS=False,
        TOKEN_CACHE_DIR=out_dir / "token_cache",
        EXP0_STATS_PATH=out_dir / "exp0_stats.pt",
        SAE_RELEASE="stub",
        SAE_SOURCE="stub",
        SAE_ID=HOOK,
    )


def make_seqs(n_docs=12, length=16, seed=0):
    """Zipf-ish token stream, so the frequency buckets are not all one bucket.

    Token id 0 is reserved for padding and never sampled: keep_mask drops every
    position equal to pad_id, so a real token sharing that id would be silently
    dropped from the statistics.
    """
    g = torch.Generator().manual_seed(seed + 2)
    weights = 1.0 / torch.arange(1, D_VOCAB).float()          # ids 1 .. D_VOCAB-1
    seqs = []
    for _ in range(n_docs):
        ids = torch.multinomial(weights, length, replacement=True, generator=g) + 1
        ids[0] = 1  # stand-in BOS: position 0 is dropped by keep_mask
        seqs.append(ids)
    return seqs


def main() -> int:
    with tempfile.TemporaryDirectory() as td:
        out_dir = Path(td)
        cfg = make_cfg(out_dir)
        model, sae = StubModel(), StubSAE(cfg.D_SAE)
        seqs = make_seqs()

        stats = collect(
            model,
            sae,
            seqs,
            device="cpu",
            cfg=cfg,
            extra_config={"source": "stub", "n_docs": len(seqs)},
            pad_id=0,  # a source with no tokenizer must pass this
        )

        failures = []

        def check(cond, msg):
            if not cond:
                failures.append(msg)

        # --- the block structure came from cfg, not from gemma's config module ---
        saved = stats["config"]
        check(saved["block_ranges"] == cfg.BLOCK_RANGES, "block_ranges did not come from cfg")
        check(saved["matryoshka_steps"] == STEPS, "matryoshka_steps did not come from cfg")
        check(saved["source"] == "stub", "extra_config was not merged into config")
        check(stats["fire_count"].shape == (cfg.D_SAE,), "fire_count is not sized by cfg.D_SAE")

        # --- per-pair accumulators are shaped by cfg's blocks ---
        pairs = adjacent_pairs(cfg)
        check(stats["pairs"] == pairs, f"pairs mismatch: {stats['pairs']} != {pairs}")
        for p, c in pairs:
            key = f"{p}->{c}"
            plen = cfg.BLOCK_RANGES[p][1] - cfg.BLOCK_RANGES[p][0]
            clen = cfg.BLOCK_RANGES[c][1] - cfg.BLOCK_RANGES[c][0]
            check(
                tuple(stats["cofire"][key].shape) == (plen, clen),
                f"cofire[{key}] shape {tuple(stats['cofire'][key].shape)} != ({plen}, {clen})",
            )

        # --- position 0 is dropped from every doc ---
        expected = sum(len(s) - 1 for s in seqs)
        check(
            stats["total_tokens"] == expected,
            f"total_tokens {stats['total_tokens']} != {expected} (BOS should be excluded)",
        )

        # --- the file is on disk and reloads ---
        check(cfg.EXP0_STATS_PATH.exists(), "no stats file was written")
        reloaded = torch.load(cfg.EXP0_STATS_PATH, map_location="cpu", weights_only=False)
        check(reloaded["schema_version"] == 2, "schema_version is not 2")

        # --- and it satisfies the umbrella's contract, if that is checked out ---
        validator = ROOT.parent / "contracts" / "validate_stats.py"
        if validator.exists():
            sys.path.insert(0, str(validator.parent))
            from validate_stats import validate  # noqa: E402

            rep = validate(reloaded)
            check(
                not rep.errors,
                "contract violations: " + "; ".join(rep.errors),
            )
            print(f"[test] contract check ran (mode={rep.mode})")
        else:
            print("[test] contracts/validate_stats.py not found — contract check skipped")

    for f in failures:
        print(f"  FAIL {f}")
    if failures:
        print(f"\n{len(failures)} failure(s)")
        return 1
    print("\ncollect() is source-agnostic: ran on a 28-feature stub, not gemma")
    return 0


if __name__ == "__main__":
    sys.exit(main())
