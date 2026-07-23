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

import os
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

# Which transformer layer's residual-stream Matryoshka SAE to analyse.
# Override per run with the EXP0_LAYER env var, e.g. `EXP0_LAYER=12 python3 ...`.
# The matryoshka SAE is released for layers 0-24; the SAE id, hook name, the
# Neuronpedia / dataset "source" name, and the per-layer output dir are all
# derived from this single number so nothing has to be edited by hand per layer.
LAYER = int(os.environ.get("EXP0_LAYER", "6"))
SAE_ID = f"blocks.{LAYER}.hook_resid_post"
HOOK_NAME = SAE_ID
SAE_SOURCE = f"{LAYER}-res-matryoshka-dc"  # e.g. "6-res-matryoshka-dc"
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

# Per-layer artifacts live in outputs/layer_NN/ so runs on different layers never
# clobber each other. Layer-independent artifacts (e.g. the synthetic toy
# calibration) stay directly in OUT_DIR.
RUN_DIR = OUT_DIR / f"layer_{LAYER:02d}"
RUN_DIR.mkdir(parents=True, exist_ok=True)

def scope_line(total_tokens=None, bold=("**", "**"), sep="　·　"):
    """Which layer, and the knobs a reader needs to interpret the numbers.

    Single source for the context line shown on every page and report, so the
    dashboards and the markdown digests can never drift apart. `bold` selects the
    emphasis syntax: ("**", "**") for markdown, ("<b>", "</b>") for HTML.
    """
    b0, b1 = bold
    bits = [f"{b0}Layer {LAYER}{b1}", f"gemma-2-2b / {SAE_SOURCE}", SAE_ID]
    if total_tokens:
        bits.append(f"{int(total_tokens):,} tokens over {N_DOCS} docs")
    bits.append(f"edge: reverse coverage ≥ {EDGE_TAU}, both endpoints fire ≥ {MIN_FIRE_COUNT}")
    return sep.join(bits)


# Back-to-index button, shared by every generated page so navigation is uniform.
# Pages are only reachable by deep link, so each needs its own way back. Emitted
# into the markdown reports too: Jekyll passes raw HTML through when it renders
# them. The href is relative (two levels up from outputs/layer_NN/) so it works
# on GitHub Pages and when the file is opened locally.
BACK_LINK_HTML = (
    '<a href="../../" title="Back to the experiment_0 index" style="position:fixed;'
    'top:14px;right:18px;z-index:999;font:600 13px/1 system-ui,-apple-system,sans-serif;'
    'color:#7C22CE;background:#F6F3FE;border:1px solid #E3DAFB;border-radius:8px;'
    'padding:9px 13px;text-decoration:none">&#8592; Back to index</a>'
)

EXP0_STATS_PATH = RUN_DIR / "exp0_stats.pt"          # written by cache_stats.py
METRICS_JSON_PATH = RUN_DIR / "metrics_report.json"  # written by run_metrics.py
METRICS_MD_PATH = RUN_DIR / "metrics_report.md"      # written by run_metrics.py

# Force a device with the EXP0_DEVICE env var or a script's --device flag:
#   local Mac      -> auto-picks "mps"
#   server (A40)   -> auto-picks "cuda"; pin your assigned GPU with either
#                     CUDA_VISIBLE_DEVICES=<n> (recommended) or EXP0_DEVICE=cuda:<n>
DEVICE_OVERRIDE: str | None = os.environ.get("EXP0_DEVICE")


def pick_device() -> str:
    import torch

    if DEVICE_OVERRIDE:
        if DEVICE_OVERRIDE.startswith("mps"):
            os.environ.setdefault("TRANSFORMERLENS_ALLOW_MPS", "1")
        return DEVICE_OVERRIDE
    if torch.backends.mps.is_available():
        os.environ.setdefault("TRANSFORMERLENS_ALLOW_MPS", "1")
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def is_mps(device: str) -> bool:
    """True for 'mps' (and any 'mps:0' form). MPS has no float64, so accumulators
    fall back to float32 there; CUDA/CPU keep float64."""
    return str(device).startswith("mps")


NEURONPEDIA_BASE = f"https://www.neuronpedia.org/gemma-2-2b/{SAE_SOURCE}"
NEURONPEDIA_API = f"https://www.neuronpedia.org/api/feature/gemma-2-2b/{SAE_SOURCE}/{{}}"

# Bulk autointerp explanation export on S3 (one gzipped JSONL batch / 128 feats).
S3_EXPLANATIONS = (
    "https://neuronpedia-datasets.s3.us-east-1.amazonaws.com/"
    f"v1/gemma-2-2b/{SAE_SOURCE}/explanations/batch-{{}}.jsonl.gz"
)


def npedia_url(feature_idx: int) -> str:
    return f"{NEURONPEDIA_BASE}/{int(feature_idx)}"


# ---------------------------------------------------------------------------
# Feature labels (autointerp descriptions), one per feature.
# Written by fetch_labels.py from the Neuronpedia dataset export; a dict
# {index_str: description}. A handful of features have no export description
# and simply fall back to "feature <idx>".
# ---------------------------------------------------------------------------
FEATURE_LABELS_PATH = RUN_DIR / "feature_labels.json"


def load_feature_labels() -> dict[str, str]:
    """Return {index_str: description}, or {} if fetch_labels.py hasn't run."""
    import json

    if FEATURE_LABELS_PATH.exists():
        return json.loads(FEATURE_LABELS_PATH.read_text())
    return {}


def feature_label(feature_idx: int, labels: dict[str, str] | None = None) -> str:
    """Human-readable label for a global feature index, with a graceful
    fallback for the ~26 features that have no export description."""
    if labels:
        text = labels.get(str(int(feature_idx)))
        if text:
            return text
    return f"feature {int(feature_idx)}"
