"""
Exp 0 - Implement Metrics: central configuration.

Exp 0 treats the metrics as *competing measurements of the same edge*
(project plan §3, Exp 0). The edges come from the warm-up task's crude
activation-coverage graph; here we add the richer signals:

    1. Activation coverage, three legs (forward / reverse / joint-child)
    2. Reconstruction condition (Tree SAE)   - pair must improve reconstruction
    3. Child diversity / sibling redundancy  - co-activation among children
    4. Out-degree distribution               - superparent / poly-parenting detector
    5. Token-frequency-controlled coverage   - condition on token frequency

The SAE / model / block structure is identical to the warm-up task
(gemma-2-2b, layer-6 residual Matryoshka SAE). We deliberately reuse the
warm-up's cached statistics where possible so both stages describe the
SAME sampled corpus.
"""

from __future__ import annotations

from pathlib import Path

# ---------------------------------------------------------------------------
# Where the warm-up task lives (its outputs are our starting point)
# ---------------------------------------------------------------------------
# NOTE for collaborators: adjust WARMUP_DIR if your checkout differs.
WARMUP_DIR = Path(
    "/Users/ruqiya/Codeing-repos/Research/eleuther/I-6-Hierarchy-in-SAEs/Warm-up task"
)
WARMUP_ACT_STATS = WARMUP_DIR / "outputs" / "activation_stats.pt"   # stage 01 warm-up
WARMUP_GRAPH = WARMUP_DIR / "outputs" / "parent_child_graph.pt"     # stage 02 warm-up

# ---------------------------------------------------------------------------
# Model + SAE (must match the warm-up exactly)
# ---------------------------------------------------------------------------
MODEL_NAME = "google/gemma-2-2b"
SAE_RELEASE = "gemma-2-2b-res-matryoshka-dc"
SAE_ID = "blocks.6.hook_resid_post"
HOOK_NAME = "blocks.6.hook_resid_post"
LAYER = 6
MODEL_KWARGS = {"center_writing_weights": False}
PREPEND_BOS = True

MATRYOSHKA_STEPS = [128, 512, 2048, 8192, 32768]
D_SAE = 32768


def _block_ranges(steps):
    ranges, prev = [], 0
    for s in steps:
        ranges.append((prev, s))
        prev = s
    return ranges


BLOCK_RANGES = _block_ranges(MATRYOSHKA_STEPS)
N_BLOCKS = len(BLOCK_RANGES)


def block_of(feature_idx: int) -> int:
    for b, (start, end) in enumerate(BLOCK_RANGES):
        if start <= feature_idx < end:
            return b
    raise ValueError(f"feature {feature_idx} out of range [0,{D_SAE})")


# ---------------------------------------------------------------------------
# Dataset (same slice as the warm-up so counts are comparable)
# ---------------------------------------------------------------------------
DATASET = "NeelNanda/pile-10k"
N_DOCS = 400
CONTEXT_SIZE = 128
BATCH_DOCS = 8

# ---------------------------------------------------------------------------
# Thresholds
# ---------------------------------------------------------------------------
FIRE_THRESHOLD = 1e-3     # feature "fires" above this (post-JumpReLU)
EDGE_TAU = 0.5            # reverse-coverage edge criterion (same as warm-up)
MIN_FIRE_COUNT = 20       # rare-feature guard (same as warm-up)

# Which adjacent block pairs to compute. B3->B4 is the 6144 x 24576 monster;
# the warm-up skipped it and we do too (its accumulators dominate memory).
INCLUDE_B3_B4 = False

# --- Metric 2: reconstruction condition ------------------------------------
# An edge passes when ablating the parent hurts reconstruction on the child's
# firing tokens by at least this relative amount (and same for the child).
RECON_REL_GAIN_MIN = 0.01     # >=1% relative error increase = "contributes"

# --- Metric 3: sibling redundancy ------------------------------------------
# Mean pairwise child-child co-activation (Jaccard) above this = feature
# splitting in disguise rather than real refinement.
SIBLING_REDUNDANCY_FLAG = 0.5
# Within-block co-firing matrices are needed for sibling stats. B4's 24576^2
# does not fit in RAM comfortably; we compute B1, B2, B3 only.
SIBLING_BLOCKS = [1, 2, 3]

# --- Metric 4: out-degree / superparents ------------------------------------
SUPERPARENT_OUTDEG_FRAC = 0.30
SUPERPARENT_FIRE_FRAC = 0.10

# --- Metric 5: token-frequency control --------------------------------------
# Token ids ranked by corpus frequency; buckets split by cumulative token MASS:
#   bucket 0 (high) = most frequent ids covering the top HIGH_MASS of tokens
#   bucket 1 (mid)  = next ids up to HIGH_MASS + MID_MASS
#   bucket 2 (low)  = the rest
FREQ_HIGH_MASS = 0.50
FREQ_MID_MASS = 0.40
N_FREQ_BUCKETS = 3
# An edge is "frequency-driven" when its reverse coverage on low+mid tokens
# drops below this fraction of its all-token reverse coverage.
FREQ_SURVIVAL_MIN = 0.5

# ---------------------------------------------------------------------------
# Paths + device
# ---------------------------------------------------------------------------
HERE = Path(__file__).resolve().parent
OUT_DIR = HERE / "outputs"
OUT_DIR.mkdir(exist_ok=True)

EXP0_STATS_PATH = OUT_DIR / "exp0_stats.pt"          # written by cache_stats.py
METRICS_JSON_PATH = OUT_DIR / "metrics_report.json"  # written by run_metrics.py
METRICS_MD_PATH = OUT_DIR / "metrics_report.md"      # written by run_metrics.py

DEVICE_OVERRIDE: str | None = None


def pick_device() -> str:
    import torch

    if DEVICE_OVERRIDE is not None:
        if DEVICE_OVERRIDE == "mps":
            import os

            os.environ.setdefault("TRANSFORMERLENS_ALLOW_MPS", "1")
        return DEVICE_OVERRIDE
    if torch.backends.mps.is_available():
        import os

        os.environ.setdefault("TRANSFORMERLENS_ALLOW_MPS", "1")
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


NEURONPEDIA_BASE = "https://www.neuronpedia.org/gemma-2-2b/6-res-matryoshka-dc"


def npedia_url(feature_idx: int) -> str:
    return f"{NEURONPEDIA_BASE}/{int(feature_idx)}"
