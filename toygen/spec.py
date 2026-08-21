"""
Settings for building a synthetic toy world.

`ToyConfig` is the single knob panel: tree shape, firing rates, geometry, strength, and an
optional confound battery. Two ready-made recipes at the bottom: backbone (clean tree only)
and full (backbone plus realistic confounds).

Each planted property mirrors a real SAE phenomenon:
  is_a / sibling / transitive: hierarchical & categorical concept geometry -- Park et al. 2024
  firing_only: the orthogonal-geometry null / hard negative for is_a
  superparent: always-on high-base-rate distractor -- foil for coverage/out-degree metrics
  broad_parent: a genuine wide parent, i.e. feature-splitting family -- Bricken et al. 2023
  token_bound: single-token / spurious co-activation features -- Bricken et al. 2023
  topical: co-occurring feature clusters ("lobes") -- Li et al. 2024
Dictionary-side properties (absorption, splitting, merging) come from SAE training, not here.
The base activation model follows Elhage et al. 2022.
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
    max_unrelated_cos: float = 0.12  # cap on cosine between unrelated feature directions

    # --- tree shape ---------------------------------------------------------
    n_roots: int = 6                 # independent trees in the forest
    branching: int = 3               # children per parent
    depth: int = 3                   # levels below each root (number of blocks = depth + 1)
    exclusive_siblings: bool = True  # siblings split the parent's tokens instead of overlapping
    randomize_structure: bool = False  # opt-in: draw a seed-varied backbone instead of the fixed lattice; confound counts stay seed-invariant except superparent, which scales with F

    # --- firing -------------------------------------------------------------
    root_p: float = 0.18             # a root fires on this fraction of tokens
    child_p_edge: float = 0.32       # P(child fires | parent fires); with exclusive siblings must stay <= 1/branching
    eps_p: float = 0.02              # keep child_p_edge <= 1 - eps so a parent can fire without a given child

    # --- composition (geometry) ---------------------------------------------
    alpha: float = 0.48              # cosine of a child's direction onto its parent (is_a overlap); siblings share alpha^2 of the parent
    alpha_zero_every: int = 4        # every n-th edge (across the forest) gets alpha = 0: the firing_only cell
    eps_alpha: float = 0.02          # keep alpha <= 1 - eps so the change-of-basis matrix stays invertible

    # --- strength -----------------------------------------------------------
    strength_spread: float = 0.35    # spread of firing strengths (sd / mean); < 0.5 keeps every strength positive
    K: int = 6                       # legacy sparsity hint only; actual k is derived from the tree's true L0 by world.choose_k.

    # --- strength scale -----------------------------------------------------
    E0: float = 1.0                  # base strength scale; every feature's mean active strength is q * sqrt(E0), with no designed ladder

    # --- corpus (only exercised by the frequency / topical confounds) -------
    vocab: int = 5000                # token-id vocabulary size
    doc_len: int = 128               # tokens per document
    freq_high_mass: float = 0.50     # corpus-mass cut point for the high frequency token bucket
    freq_mid_mass: float = 0.40      # cut point for the mid frequency bucket
    Z: int = 8                       # number of topics (used by the topical confound)
    zipf_s: float = 1.05             # Zipf exponent controlling how skewed token frequencies are

    # --- confounds (enabled by confounds=True; off in backbone) ---
    confounds: bool = False          # master switch for the whole distractor battery below
    n_superparent: int = 1           # always-on wide parents -- the base-rate confound
    n_broad_parent: int = 1          # genuine wide parents -- the superparent's honest foil
    broad_children: int = 5          # children under each broad parent
    broad_alpha: float = 0.48        # is_a overlap for a broad parent's children; kept equal to `alpha` to stay above the unrelated ceiling
    n_token_bound_pairs: int = 8     # token-bound pairs co-firing via one shared token-id set; 8 pairs = 16 features = 240 ordered frequency pairs
    n_topical_pairs: int = 12        # feature pairs lifted by a shared topic, round-robin over Z; 12 pairs = 24 features across 8 topic groups
    kappa: float = 7.2               # topic-modulation strength; higher lifts same-topic co-firing (bounded so per-topic rates stay in [0, 1])
    n_bind_ids: int = 2              # top-frequency token ids shared by token-bound features; must stay under the id set's Zipf mass


# --- seed-varied backbone knobs (only read when randomize_structure=True) -------------
# Structure RNG is seeded from cfg.seed + offset + attempt, so draws are reproducible and never collide with the geometry/sampling streams.
STRUCTURE_SEED_OFFSET: int = 9973
# A randomized draw is retried until it clears these floors, so it can't starve a scored class or shrink the dictionary:
F_MIN: int = 120                     # minimum total feature count
MIN_PAIRS_PER_CLASS: int = 5         # minimum ordered pairs per guarded scored class
STRUCTURE_MAX_ATTEMPTS: int = 64     # deterministic retries before giving up (then raises)
# Backbone-derived classes whose pair count moves with structure and must stay well-populated.
GUARDED_STRUCTURE_CLASSES: tuple[str, ...] = ("is_a", "firing_only", "sibling")


def replace(cfg: ToyConfig, **kw) -> ToyConfig:
    """dataclasses.replace, re-exported so callers need not import dataclasses."""
    return dataclasses.replace(cfg, **kw)


def backbone_config() -> ToyConfig:
    """Clean stratum: the backbone tree only (is_a, firing_only, sibling, transitive), no confounds."""
    return ToyConfig(name="backbone", confounds=False)


def full_config() -> ToyConfig:
    """Backbone plus the full confound battery -- the main validation world."""
    return ToyConfig(name="full", confounds=True)


CONFIGS = {
    "backbone": backbone_config,
    "full": full_config,
}
