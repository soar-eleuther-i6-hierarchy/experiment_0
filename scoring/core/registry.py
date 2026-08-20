"""
Scoring configuration for the retrieval scorer. The detector set and their orientation, the scored confound columns, and the numeric constants the detectors read by name. 
"""

from __future__ import annotations

# The ten per-ordered-pair detector scalars, in a fixed order.
DETECTORS: tuple[str, ...] = (
    "coverage_R", "asymmetry_R", "joint_child_J", "pmi", "token_freq_survival",
    "recon_2a", "s_res", "sibling_redundancy", "joint_child_mass", "outdegree",
)

# Orientation: +1 if the raw scalar already reads "higher == more is-a-like", -1 if it must be negated. It must be chosen up front and left alone
DETECTOR_SIGN: dict[str, int] = {
    "coverage_R": 1, "asymmetry_R": 1, "joint_child_J": 1, "pmi": 1,
    "token_freq_survival": 1, "recon_2a": 1, "s_res": 1,
    "sibling_redundancy": -1, "joint_child_mass": 1, "outdegree": -1,
}

# The confound columns scored against is_a 
SCORED_COLUMNS: tuple[str, ...] = (
    "firing_only", "sibling", "superparent", "frequency", "topical",
    "transitive", "reversed", "unrelated",
)

POSITIVE_LABEL: str = "is_a"

# Detectors symmetric in (parent, child). A symmetric negative class counts (a,b) and (b,a) as two identical negatives.
SYMMETRIC_DETECTORS: tuple[str, ...] = ("pmi",)

# The general-purpose is-a detectors in the fusion ensemble.
ENSEMBLE_DETECTORS: tuple[str, ...] = (
    "coverage_R", "asymmetry_R", "pmi", "recon_2a", "s_res", "token_freq_survival",
)

# Numeric knobs the detectors and scorer read by name.
CONSTANTS: dict[str, float] = {
    "fire_thresh": 0.0,        # firing := activation > 0 (BatchTopK nonzero == top-k)
    "edge_tau": 0.5,           # reverse-coverage cut for the inferred edge set
    "min_fire_count": 20,      # both endpoints must fire this often to form an edge
    "min_joint": 30,           # min co-firing tokens for a supported edge
    "recon_rel_gain_min": 0.01,
    "pmi_laplace": 1.0,        # +1 smoothing in the PMI ratio
    "coverage_eps": 1e-6,      # coverage denominator floor
    "r_disp_m": 5,             # top-m competitors for the dispersion readout
    "freq_high_mass": 0.5,     # top corpus-mass cut for the high-frequency token bucket
    "freq_mid_mass": 0.4,      # mid bucket cut
    "n_freq_buckets": 3,
    "freq_min_fire_low": 5,    # below this, the rare-token survival cell is underpowered
    "rho_star": 0.5,           # recovery threshold the recovered universe is built on
    "auroc_clamp": 1e-6,       # clamp AUROC to [clamp, 1-clamp] before the logit CI
    # --- probe s_res, mirror config.py so compute_all is self-contained ---
    "sres_rank_top_k": 5,      # both decoders in the top-k probe correlations (diagnostic rank rule)
    "sres_min_probe_pos": 50,  # min child-firing tokens to train probe, below this the column is NaN
    "sres_neg_ratio": 4,       # negatives sampled per positive
    "sres_max_probe_tokens": 20000,  # cap on (pos + neg) tokens per probe
    "sres_min_neg": 10,        # fewer negatives than this -> child untestable (no probe)
    "sres_steps": 300,         # probe Adam steps (calibration knob)
    "sres_lr": 0.05,           # probe Adam lr (calibration knob)
}