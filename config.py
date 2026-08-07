"""
Central configuration for the hierarchy-metric suite.

The suite treats the metrics as *competing measurements of the same edge*. The
edges come from the warm-up task's crude activation-coverage graph; here we
add the richer signals:

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
# Joint-support guard: a child firing MIN_FIRE_COUNT times
# inside a near-always-on parent hits R = 1.0 by chance; requiring a minimum
# co-fire count kills those. Excluded edges are REPORTED, not silently dropped.
# NOTE: any value <= EDGE_TAU * MIN_FIRE_COUNT (= 10) is vacuous — every kept
# edge already satisfies it. 30 matches the warm-up task's guard.
MIN_JOINT = 30

# Which adjacent block pairs to compute. B3->B4 is the 6144 x 24576 monster;
# the warm-up skipped it on the 4 GB GPU. On the A40 it fits (~8-10 GB extra):
# enable with EXP0_B3B4=1. Caveat: B4 is 24576 mostly-rare features, so many
# B3->B4 edges are unsupported (dropped by MIN_FIRE_COUNT) — a thin, noisy pair.
INCLUDE_B3_B4 = os.environ.get("EXP0_B3B4", "0") == "1"

# --- Metric 2a: reconstruction-ablation contribution filter -----------------
# (Tree-SAE-INSPIRED baseline, not the paper's S_res — see metrics/reconstruction.py.)
# An edge passes when ablating the parent hurts reconstruction on the child's
# firing tokens by at least this relative amount (and same for the child).
RECON_REL_GAIN_MIN = 0.01     # >=1% relative error increase = "contributes"

# --- Metric 2b: probe-based S_res (Tree SAE Eq. 5, rank-scored) --------------
# S_res(p,c) = min((d_c*)ᵀ d_c, (d_c*)ᵀ d_p) with d_c* a linear-probe direction
# for the child concept. Scored by the RANK rule (both decoders in the top-k
# probe correlations over all features), not a threshold: healthy pairs have
# d_p ⟂ d_c which caps the min at 1/√2, so thresholds above that reject
# everything. Probes need enough positives to train on.
SRES_RANK_TOP_K = 5           # Tree SAE's operational rule: both in top-5
MIN_PROBE_POS = 50            # min child-firing tokens to train a probe
SRES_MAX_CHILDREN_PER_PAIR = 4000   # cost guard; log when hit
SRES_NEG_RATIO = 4            # negatives sampled per positive
SRES_MAX_PROBE_TOKENS = 20000 # cap on (pos + neg) tokens per probe
SRES_MIN_NEG = 10             # fewer negatives than this -> child untestable (no probe)

# --- Metric 3: sibling redundancy ------------------------------------------
# Mean pairwise child-child co-activation (Jaccard) above this = feature
# splitting in disguise rather than real refinement.
SIBLING_REDUNDANCY_FLAG = 0.5
# Within-block co-firing matrices are needed for sibling stats. B4's 24576^2
# does not fit in RAM comfortably; we compute B1, B2, B3 only.
SIBLING_BLOCKS = [1, 2, 3]

# --- In-block (same-level) edges (in_block_edges.py) ------------------------
# Hierarchy need not respect block boundaries: two features in the SAME block
# can stand in a parent/child (refinement) or duplicate relation. These blocks
# get a within-block directed-edge analysis. collect_statistics caches within-cofire for
# SIBLING_BLOCKS ∪ IN_BLOCK_BLOCKS, so B0 is cached after a rerun; on older caches
# in_block_edges falls back to rebuilding B0 from the token cache. B3/B4 skipped:
# B4's 24576^2 matrix is ~4.8 GB, not worth it.
IN_BLOCK_BLOCKS = [0, 1, 2]

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
# EXP0_OUT redirects ALL outputs to a directory of your choice, so a scratch run
# never touches the git-tracked outputs/ published on GitHub Pages. Default: unchanged.
OUT_DIR = Path(os.environ.get("EXP0_OUT", HERE / "outputs"))
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Results are grouped by SOURCE first, layer second: outputs/gemma2_2b/layer_NN/.
# They used to sit at outputs/layer_NN/, from when gemma was the only source there
# could be -- which stopped being true the moment a PCFG run was published beside
# them, and left the site claiming that "layer 6" is a global fact rather than a
# fact about one model.
SOURCE_NAME = "gemma2_2b"
LAYER_RUN = f"{SOURCE_NAME}/layer_{LAYER:02d}"

# EXP0_RUN names the directory instead, for a run that is not a gemma layer: the
# PCFG SAE is 1792 latents in 8 blocks over its own base model, and `layer_01`
# would both collide with a future gemma layer and misdescribe it. Any depth under
# OUT_DIR works -- every page derives its own distance to the site root from its
# path -- but the LAST component decides how it is read: a directory named
# `layer_NN` gets the layer nav, anything else gets the site-wide one.
RUN_NAME = os.environ.get("EXP0_RUN", LAYER_RUN)
RUN_DIR = OUT_DIR / RUN_NAME
RUN_DIR.mkdir(parents=True, exist_ok=True)
# Layer-independent artifacts (the toy calibrations) stay directly in OUT_DIR.
IS_LAYER_RUN = Path(RUN_NAME).name.startswith("layer_")


def page_depth(path) -> int:
    """Levels from a generated page's directory up to the site root.

    The site root is the repo root, so the answer is simply how deep the file sits
    inside it -- which also covers the pages that are NOT under outputs/ at all:
    the package READMEs, and outputs_archive/. Derived rather than passed, because
    the depth changed for 25 published pages the day results were grouped by
    source, and every caller that had hardcoded 2 was then wrong.

    A page written under EXP0_OUT is outside the repo; there OUT_DIR stands in for
    `outputs/`, so a scratch run produces the same bar as the published page.
    """
    p = Path(path).resolve()
    try:
        return len(p.relative_to(HERE.resolve()).parts) - 1
    except ValueError:
        return len(p.relative_to(OUT_DIR.resolve()).parts)

def scope_line(total_tokens=None, bold=("**", "**"), sep="　·　", n_docs=None, config=None):
    """Which layer, and the knobs a reader needs to interpret the numbers.

    Single source for the context line shown on every page and report, so the
    dashboards and the markdown digests can never drift apart. `bold` selects the
    emphasis syntax: ("**", "**") for markdown, ("<b>", "</b>") for HTML.

    `config` is the stats file's own config block. Pass it and the line describes
    the run that produced the numbers rather than this module's gemma defaults --
    a PCFG page reading "gemma-2-2b" states the wrong model, and a reader has no
    way to tell from the page that it is wrong. Absent, nothing changes: every
    caller that omits it is grading gemma.
    """
    cfg = config or {}
    b0, b1 = bold
    base = cfg.get("base_model") or {}
    model = MODEL_NAME.split("/")[-1]
    if cfg.get("source") == "pcfg":
        model = (f"PCFG toy {base.get('n_layers', '?')}L "
                 f"d_model={base.get('d_model', '?')}")
    bits = [f"{b0}Layer {cfg.get('layer', LAYER)}{b1}",
            f"{model} / {cfg.get('sae_source', SAE_SOURCE)}",
            cfg.get("sae_id", SAE_ID)]
    steps = cfg.get("matryoshka_steps")
    if steps and list(steps) != MATRYOSHKA_STEPS:
        bits.append(f"{steps[-1]:,} latents in {len(steps)} blocks")
    if total_tokens:
        bits.append(f"{int(total_tokens):,} tokens over "
                    f"{n_docs or cfg.get('n_docs') or N_DOCS} docs")
    bits.append(f"edge: reverse coverage ≥ {EDGE_TAU}, both endpoints fire ≥ {MIN_FIRE_COUNT}")
    return sep.join(bits)


# --- site navigation --------------------------------------------------------
# Every page is reachable only by a deep link, so each carries its own nav bar.
# Emitted into the markdown reports too: Jekyll passes raw HTML through when it
# renders them (and serves them as .html, which is why the report links below
# end in .html, not .md). All hrefs are relative to the site root, so the bar
# works on GitHub Pages and when a file is opened straight off disk.
#
# It is `sticky`, not `fixed`: sticky occupies layout space, so the bar can
# never cover the top of a plot the way the old floating back-button did.
NAV_LAYERS = [3, 6, 12, 18, 24]

# (file, label) for the five pages every layer_NN/ has. `file` doubles as the
# identity of "which page am I on", so switching layer lands on the same kind.
NAV_PAGES = [
    ("metrics_dashboard.html", "Dashboard"),
    ("superparent_sankey.html", "Superparents"),
    ("qualitative_dashboard.html", "Qualitative"),
    ("metrics_report.html", "Metrics report"),
    ("qualitative_check.html", "Qualitative report"),
]

# Global links: (path from site root, label). "" is the repo README = the index.
# No "Overview" entry: the brand on the left already links to the site root, and
# two adjacent links to the same place is noise. No "Per layer" either: it went
# to the layer *table*, while the pills in the second row open each layer
# directly, which is strictly better.
# kill_rates.html and cross_depth_comparison.html are gone from this list. Both
# were hand-built with no generator, so neither could be rebuilt by rerunning a
# stage, and both were written against caches that counted BOS. BOS is an
# attention sink -- every feature fires there -- so it manufactured 400 joint
# firings for every pair in the dictionary and defeated the joint-support guard.
# Excluding it inverted the numbers those two pages were built to display: the
# deep-pair reconstruction share, the frequency-driven share, the death rate.
# They now sit in outputs_archive/ carrying a banner. A page that states a
# withdrawn result next to regenerated ones is worse than no page; put them back
# only behind a generator, so a rerun can never leave them stale again.
NAV_GLOBAL = [
    ("outputs/", "Results"),
    ("outputs/toy_calibration.html", "Toy calibration"),
    ("outputs/trained_toy_calibration.html", "Trained toy"),
    # A run, not a layer: the PCFG SAE has its own base model and its own
    # dictionary, so it belongs in the global row rather than among the layer
    # pills. Adding an entry here changes the bar on EVERY page, including ones
    # whose generator needs a 700 MB cache -- see reporting/refresh_nav.py.
    ("outputs/pcfg/", "PCFG"),
]

NAV_CSS = """<style>
.x0nav{position:sticky;top:0;z-index:999;background:#fff;border-bottom:1px solid #E3DAFB;
font:500 13px/1.15 system-ui,-apple-system,"Segoe UI",sans-serif;margin:0 0 14px;}
.x0nav .row{display:flex;flex-wrap:wrap;align-items:center;gap:13px;padding:9px 18px;}
.x0nav .row+.row{border-top:1px solid #F1ECFD;}
.x0nav a{text-decoration:none;color:#5A6B7B;}
.x0nav a:hover{color:#7C22CE;}
.x0nav .brand{font-weight:700;color:#7C22CE;letter-spacing:.2px;}
.x0nav .on{color:#7C22CE;font-weight:700;}
.x0nav .lbl{color:#9AA7B3;font-size:11px;text-transform:uppercase;letter-spacing:.7px;}
.x0nav .pill{border:1px solid #E3DAFB;border-radius:7px;padding:5px 10px;background:#F6F3FE;}
.x0nav .pill.on{background:#7C22CE;color:#fff;border-color:#7C22CE;}
.x0nav .sep{width:1px;height:17px;background:#E3DAFB;}
@media (prefers-color-scheme:dark){
.x0nav{background:#141414;border-bottom-color:#2E2E2E;}
.x0nav .row+.row{border-top-color:#242424;}
.x0nav a{color:#A9B4BF;}
.x0nav .brand,.x0nav a:hover,.x0nav .on{color:#C79BF2;}
.x0nav .pill{background:#1E1830;border-color:#3A2B57;}
.x0nav .pill.on{background:#7C22CE;color:#fff;border-color:#7C22CE;}
.x0nav .sep{background:#2E2E2E;}}
</style>"""


def nav_html(depth: int = 2, layer: int | None = None, page: str | None = None,
             current: str | None = None) -> str:
    """The site nav bar for one page.

    depth    levels from this page's directory up to the site root
             (1 for outputs/x.html, 3 for outputs/gemma2_2b/layer_NN/x.html);
             `page_depth(path)` computes it, and callers should use that rather
             than count by hand
    layer    the layer this page describes, or None for a site-wide page
    page     which of NAV_PAGES this is, so the row can mark it current
    current  this page's own path from the site root, used to highlight its
             entry in the global row (site-wide pages only)

    The second row only appears on a layer page: it is what makes navigation
    two-dimensional — change the layer and stay on the same page, or change the
    page and stay on the same layer.
    """
    # depth 0 is the site root itself (the repo README): "" is not a usable href.
    root = "../" * depth or "./"

    top = [f'<a class="brand" href="{root}">SOAR I-6 · metrics</a>']
    for href, label in NAV_GLOBAL:
        # A directory entry owns every page under it, so a run that publishes a
        # whole directory (outputs/pcfg/) lights up its own nav entry from any of
        # its pages, not only from the index. `outputs/` itself is excluded --
        # it is the site index and prefixes every page there is, so it would be
        # permanently lit next to the entry that is actually current.
        owns = href.endswith("/") and href.count("/") > 1
        here = current is not None and (href == current or
                                        (owns and current.startswith(href)))
        on = "on" if here else ""
        top.append(f'<a class="{on}" href="{root}{href}">{label}</a>')
    rows = ['<div class="row">' + "".join(top) + "</div>"]

    # The layer row is on EVERY page, not just layer pages: without it there is
    # no way into a given layer from the index or a site-wide page. On a layer
    # page a pill keeps the current page kind; elsewhere it opens that layer's
    # dashboard. The "Page" group needs a layer to stay within, so it is the one
    # part that only appears on a layer page.
    # A pill keeps the current page kind when there is one; from anywhere else
    # it opens that layer's index, which is why every layer_NN/ has a README.
    second = ['<span class="lbl">Layer</span>']
    for L in NAV_LAYERS:
        on = " on" if L == layer else ""
        second.append(
            f'<a class="pill{on}" href="{root}outputs/{SOURCE_NAME}/layer_{L:02d}/{page or ""}">{L}</a>'
        )
    if layer is not None:
        second.append('<span class="sep"></span><span class="lbl">Page</span>')
        for f, label in NAV_PAGES:
            on = " on" if f == page else ""     # page=None -> the layer index, nothing marked
            second.append(
                f'<a class="{on.strip()}" href="{root}outputs/{SOURCE_NAME}/layer_{layer:02d}/{f}">{label}</a>'
            )
    rows.append('<div class="row">' + "".join(second) + "</div>")

    return NAV_CSS + '<nav class="x0nav">' + "".join(rows) + "</nav>"

# --- keeping previous runs -------------------------------------------------
# A run writes into a FIXED path (outputs/<source>/layer_NN/) because the published site
# links to it by name -- timestamping that directory would 404 every page. So a
# rerun would otherwise replace the previous run's numbers with no trace. This
# takes a dated copy first. Copy, not move, so the site stays whole even if the
# run dies.
#
# Tracked, not gitignored, on purpose. A superseded run is evidence: when a number
# changes, the version it replaced has to stay citable rather than sit in one
# person's working copy. outputs_local/ was the old home and is ignored, so an
# archive there vanished on a fresh clone -- the one case you need it.
ARCHIVE_DIR = HERE / "outputs_archive"

# Never archived. Two different reasons, both worth stating.
#
# exp0_stats.pt / token_cache / figures: too big to keep per run. The ~700 MB
# cache is on the Hub and the token cache is rebuildable from it.
#
# feature_labels.json / npedia_labels_cache.json: not run output at all. They are
# Neuronpedia's labels for this layer's SAE, a property of the dictionary, and
# nothing we compute touches them -- so an archived copy is byte-identical to the
# live one and archives nothing. Left in, they were 9.8 MB of the archive's 11 MB.
# Safe to drop because the dashboards inline their labels at generation time; no
# archived page reads these files when it is opened.
ARCHIVE_SKIP = ("*.pt", "token_cache", "figures",
                "feature_labels.json", "npedia_labels_cache.json")


def archive_run_dir(stamp: str) -> "Path | None":
    """Copy this layer's current artifacts to ARCHIVE_DIR/layer_NN__<stamp>/.

    `stamp` is passed in rather than read from the clock here, so the caller
    decides the run's identity and every stage of one run can share it.
    Returns the archive path, or None when there is nothing to keep yet.
    """
    import shutil

    if not RUN_DIR.exists() or not any(RUN_DIR.iterdir()):
        return None
    dest = ARCHIVE_DIR / f"{RUN_DIR.name}__{stamp}"
    if dest.exists():
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(RUN_DIR, dest, ignore=shutil.ignore_patterns(*ARCHIVE_SKIP))
    return dest


EXP0_STATS_PATH = RUN_DIR / "exp0_stats.pt"          # written by collect_statistics.py
METRICS_JSON_PATH = RUN_DIR / "metrics_report.json"  # written by run_metrics.py
METRICS_MD_PATH = RUN_DIR / "metrics_report.md"      # written by run_metrics.py

# The ~700 MB/layer stats caches are too big for git, so they are gitignored and
# hosted on the Hub instead. A fresh clone has no exp0_stats.pt, so every consumer
# points at both ways to get one: download it, or recompute it.
HF_STATS_DATASET = "soar-eleuther-i6-hierarchy/experiment_0-stats"


def missing_stats_msg() -> str:
    """Error text for the metric scripts when exp0_stats.pt isn't there yet."""
    return (
        f"missing {EXP0_STATS_PATH}\n"
        f"  download it:  hf download {HF_STATS_DATASET} --repo-type dataset "
        f'--include "{RUN_DIR.name}/*" --local-dir {RUN_DIR.parent}/\n'
        f"  or rebuild it: EXP0_LAYER={LAYER} python3 collect_statistics.py"
    )

# Token-level caches (written by collect_statistics.py when CACHE_RESIDUALS):
# fp16 residuals + sparse latents let run_token_metrics.py train S_res probes and
# compute parent-conditioned sibling stats WITHOUT re-running the model.
CACHE_RESIDUALS = os.environ.get("EXP0_CACHE_RESIDUALS", "1") != "0"
TOKEN_CACHE_DIR = RUN_DIR / "token_cache"
SECOND_PASS_PATH = RUN_DIR / "second_pass.json"      # model-free token-cache pass (S_res + parent-conditioned siblings); written by run_token_metrics.py
IN_BLOCK_PATH = RUN_DIR / "in_block_edges.json"      # written by in_block_edges.py

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
