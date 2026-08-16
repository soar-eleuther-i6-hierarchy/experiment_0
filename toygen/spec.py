"""
Settings for building a synthetic toy world.

`ToyConfig` contains knobs that can be tuned in the constructed toy tress such as tree shape, how often features fire, their geometry, their loudness, and (optionally) a battery of distractors. Every value is a design choice made up front, and the rest of `toygen/` just builds the world these configurations describe.

Three ready-made recipes at the bottom:
  - backbone: the clean tree only (is_a, firing_only, sibling, transitive).
  - full: backbone plus the confounds - realistic look-alikes the metrics must not be fooled by.
  - minimal:  a small backbone for quick bring-up.

Each planted property maps to a reported SAE phenomenon (one reference each):
  is_a / sibling / transitive  hierarchical & categorical concept geometry -- Park et al. 2024 (arXiv:2406.01506)
  firing_only                  the orthogonal-geometry null, i.e. is_a's hard negative (contrast to the above)
  superparent / broad_parent   high-density and feature-splitting features -- Bricken et al. 2023 (Towards Monosemanticity)
  token_bound                  single-token / spurious co-activation features -- Bricken et al. 2023
  topical                      co-occurring feature clusters ("lobes") -- Li et al. 2024 (arXiv:2410.19750)
  manifold                     non-linear / circular (graded) features -- Engels et al. 2024 (arXiv:2405.14860)
Dictionary-side properties (absorption, feature splitting, merging) are induced by SAE
training, not built here.
The base activation model (sum of active feature directions) follows Elhage et al. 2022 (arXiv:2209.10652).
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass


@dataclass(frozen=True)
class ToyConfig:
    name: str
    seed: int = 0

    # --- activation space ---------------------------------------------------
    D: int = 128                     # dimension of the activation vectors the SAE sees
    noise_sigma: float = 0.05        # std of the Gaussian noise added to every activation (> 0)
    max_unrelated_cos: float = 0.35  # target cap on the cosine between unrelated direction arrows,
                                     # so random pairs don't accidentally look aligned

    # --- tree shape ---------------------------------------------------------
    n_roots: int = 6                 # independent trees in the forest
    branching: int = 3               # children per parent
    depth: int = 3                   # levels below each root (number of blocks = depth + 1)
    exclusive_siblings: bool = True  # siblings split the parent's tokens instead of overlapping

    # --- firing -------------------------------------------------------------
    root_p: float = 0.18             # a root fires on this fraction of tokens
    child_p_edge: float = 0.32       # P(child fires | parent fires). With exclusive siblings the
                                     # children share the parent's tokens, so the per-parent edge
                                     # probabilities must sum to <= 1 -- i.e. this stays <= 1/branching
    eps_p: float = 0.02              # keep child_p_edge <= 1 - eps so a parent can fire without a given child

    # --- composition (geometry) ---------------------------------------------
    alpha: float = 0.55              # cosine of a child's arrow onto its parent's (the is_a overlap)
    alpha_zero_every: int = 4        # every n-th edge, counted across the whole forest, gets alpha = 0:
                                     # the firing_only cell -- nested firing but orthogonal geometry
    eps_alpha: float = 0.02          # keep alpha <= 1 - eps so the change-of-basis matrix stays invertible

    # --- strength -----------------------------------------------------------
    strength_spread: float = 0.35    # spread of firing strengths (sd / mean); < 0.5 keeps every strength positive
    firing_threshold: float = 0.015  # a feature counts as active only above this strength. Provisional: a trained
                                     # SAE sets its own cutoff, so this is only a stand-in for the analytic path
    K: int = 6                       # target sparsity: mean active features per token (sum of firing rates ~ K)

    # --- strength scale -----------------------------------------------------
    E0: float = 1.0                  # base strength scale; every feature's mean active
                                     # strength is q * sqrt(E0), with no designed ladder

    # --- corpus (only exercised by the frequency / topical confounds) -------
    vocab: int = 5000                # token-id vocabulary size
    doc_len: int = 128               # tokens per document
    freq_high_mass: float = 0.50     # corpus-mass cut point for the high frequency token bucket
    freq_mid_mass: float = 0.40      # cut point for the mid frequency bucket (keep in step with config.py to stay comparable)
    Z: int = 8                       # number of topics (used by the topical confound)
    zipf_s: float = 1.05             # Zipf exponent controlling how skewed token frequencies are

    # --- confounds (enabled by confounds=True; off in backbone / minimal) ---
    confounds: bool = False          # master switch for the whole distractor battery below
    n_superparent: int = 1           # always-on wide parents -- the base-rate confound
    n_broad_parent: int = 1          # genuine wide parents -- the superparent's honest foil, so out-degree isn't trivially decisive
    broad_children: int = 5          # children under each broad parent
    broad_alpha: float = 0.35        # broad children's geometric loading, kept low so their inherited loudness fits the parent's band
    n_token_bound_pairs: int = 6     # feature pairs that co-fire only via shared token ids (frequency coincidence)
    n_topical_pairs: int = 6         # feature pairs lifted together by a shared topic (topical co-occurrence)
    n_manifold: int = 4              # features whose strength is a smooth dial, not on/off (e.g. sentiment), so the metrics don't assume every concept is a clean switch
    kappa: float = 3.0               # topic-modulation strength for topical features
    n_bind_ids: int = 6              # how many top-frequency token ids the token-bound features share; the set must carry enough corpus mass to support their firing rate


def replace(cfg: ToyConfig, **kw) -> ToyConfig:
    """dataclasses.replace, re-exported so callers need not import dataclasses."""
    return dataclasses.replace(cfg, **kw)


def backbone_config() -> ToyConfig:
    """Clean stratum: the backbone tree only (is_a, firing_only, sibling, transitive), no confounds."""
    return ToyConfig(name="backbone", confounds=False)


def full_config() -> ToyConfig:
    """Backbone plus the full confound battery -- the main validation world."""
    return ToyConfig(name="full", confounds=True)


def minimal_config() -> ToyConfig:
    """Smallest world for quick bring-up: a 3-root backbone, no confounds. Still
    exercises is_a and firing_only, but is small enough to train a real SAE on
    quickly. K is lowered to 2 to match the smaller forest's firing mass."""
    return ToyConfig(name="minimal", confounds=False, n_roots=3, K=2)


CONFIGS = {
    "backbone": backbone_config,
    "full": full_config,
    "minimal": minimal_config,
}
