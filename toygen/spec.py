"""
Settings for building a synthetic toy world.

`ToyConfig` contains knobs that can be tuned in the constructed toy trees such as tree shape, how often features fire, their geometry, their loudness, and (optionally) a battery of distractors. Every value is a design choice made up front, and the rest of `toygen/` just builds the world these configurations describe.

Two ready-made recipes at the bottom:
  - backbone: the clean tree only (is_a, firing_only, sibling, transitive).
  - full: backbone plus the confounds - realistic look-alikes the metrics must not be fooled by.

Each planted property maps to a reported SAE phenomenon (one reference each):
  is_a / sibling / transitive:  hierarchical & categorical concept geometry -- Park et al. 2024 (arXiv:2406.01506)
  firing_only:  the orthogonal-geometry null, i.e. is_a's hard negative (contrast to the above)
  superparent:  the always-on / high-base-rate confound (a childless, high-firing distractor) -- NOT a Bricken term; it is the base-rate foil the coverage / out-degree metrics must not be fooled by
  broad_parent:  a genuine wide parent (feature-splitting family) -- Bricken et al. 2023 (Towards Monosemanticity)
  token_bound:  single-token / spurious co-activation features -- Bricken et al. 2023
  topical:  co-occurring feature clusters ("lobes") -- Li et al. 2024 (arXiv:2410.19750)
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
    max_unrelated_cos: float = 0.12  # cap on the cosine between unrelated feature directions,
                                     # so random pairs don't look accidentally aligned

    # --- tree shape ---------------------------------------------------------
    n_roots: int = 6                 # independent trees in the forest
    branching: int = 3               # children per parent
    depth: int = 3                   # levels below each root (number of blocks = depth + 1)
    exclusive_siblings: bool = True  # siblings split the parent's tokens instead of overlapping
    randomize_structure: bool = False  # opt-in: draw a seed-varied backbone (ragged branching/depth,
                                     # stratified is-a/firing_only, mass-preserving edge probs) instead
                                     # of the fixed lattice. Off = the exact deterministic tree as before.
                                     # The confound battery is regenerated per seed: frequency and topical
                                     # counts are seed-invariant, but the superparent class size scales
                                     # with F (2*(F-1)*n_superparent), so it is NOT identical across seeds.

    # --- firing -------------------------------------------------------------
    root_p: float = 0.18             # a root fires on this fraction of tokens
    child_p_edge: float = 0.32       # P(child fires | parent fires). With exclusive siblings the
                                     # children share the parent's tokens, so the per-parent edge
                                     # probabilities must sum to <= 1 -- i.e. this stays <= 1/branching
    eps_p: float = 0.02              # keep child_p_edge <= 1 - eps so a parent can fire without a given child

    # --- composition (geometry) ---------------------------------------------
    alpha: float = 0.48              # cosine of a child's direction onto its parent (is_a overlap). Siblings share alpha^2 of the parent. See DESIGN.md for the 0.48 choice.
    alpha_zero_every: int = 4        # every n-th edge, counted across the whole forest, gets alpha = 0: the firing_only cell -- nested firing but orthogonal geometry
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
    n_broad_parent: int = 1          # genuine wide parents -- the superparent's honest foil, so out-degree isn't trivially decisive
    broad_children: int = 5          # children under each broad parent
    broad_alpha: float = 0.48        # is_a overlap for the broad parent's children. Held equal to `alpha` so these is_a edges sit above the unrelated ceiling; a lower value would drop them into the null band where the metric could not separate them.
    n_token_bound_pairs: int = 8     # token-bound feature pairs that co-fire only via a shared token-id set. They form one clique (all share the same ids), so 8 pairs = 16 features = 240 ordered frequency pairs, not 16.
    n_topical_pairs: int = 12        # feature pairs lifted by a shared topic. Topics are assigned round-robin (z = i % Z), so with Z = 8 some topics carry more than one pair and form larger groups: 12 pairs = 24 features across 8 topic groups. Raise Z for uniform 2-feature groups.
    kappa: float = 7.2               # topic-modulation strength for the topical confound; higher lifts same-topic co-firing so the confound can propose a false edge. Bounded so per-topic firing rates stay in [0, 1] (validate_config enforces this).
    n_bind_ids: int = 2              # how many top-frequency token ids the token-bound features share. Must stay under the id set's Zipf mass (validate_config enforces this).


# --- seed-varied backbone knobs (only read when randomize_structure=True) -------------
# The structure RNG is seeded from cfg.seed + this offset + attempt, so it is reproducible
# and disjoint from the geometry/sampling streams (both keyed on cfg.seed) without ever
# advancing cfg.seed itself.
STRUCTURE_SEED_OFFSET: int = 9973
# A randomized draw is rejected-and-retried until it clears these floors (guards against a
# shallow/narrow draw starving the scored classes or shrinking the dictionary):
F_MIN: int = 120                     # minimum total feature count
MIN_PAIRS_PER_CLASS: int = 5         # minimum ordered pairs per guarded scored class
STRUCTURE_MAX_ATTEMPTS: int = 64     # deterministic retries before giving up (raises)
# Backbone-derived scored classes whose pair count varies with structure and must stay
# powered for the pilot. (superparent/frequency/topical are fixed-count confounds; unrelated
# is always large; transitive/reversed are allowed to be small.)
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
