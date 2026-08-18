"""
World-building, config resolution, and SAE-sizing helpers for the toy generator.

These are shared by both the trainer and the measurement scorer: the trainer builds a
world to train on, and the scorer rebuilds the exact same world from a checkpoint's
persisted `resolved_config` to score against. Everything here is pure toygen + torch,
with no GPU dependency, so it runs and is tested anywhere.

Matryoshka prefixes are a fixed geometric schedule, independent of the toy's hierarchy:
sizing the SAE's nesting from the toy's own depth would hand it the answer, so whether
the hierarchy sorts into those prefixes stays a measurement, not an input.
"""

from __future__ import annotations

import torch

from . import geometry, sample, spec, strengths
from . import tree as tree_mod


def geometric_prefixes(F: int, expansion: int, n_steps: int = 4) -> tuple[list[int], int]:
    """Matryoshka prefix cutoffs for a dictionary of width `expansion * F`.

    Cutoffs halve down from the full width: e.g. F=240, expansion=4 -> d_sae=960 and
    steps [120, 240, 480, 960]. The last step is always the full width (the outermost
    doll). Depends only on (F, expansion, n_steps), never on the tree's depth.
    """
    if F <= 0:
        raise ValueError(f"F must be positive; got {F}")
    if expansion <= 0:
        raise ValueError(f"expansion must be positive; got {expansion}")
    if n_steps < 1:
        raise ValueError(f"n_steps must be >= 1; got {n_steps}")
    d_sae = expansion * F
    steps = sorted({max(1, round(d_sae / (2 ** i))) for i in range(n_steps)})
    if steps[-1] != d_sae:                       # rounding guard; keep full width last
        steps.append(d_sae)
    return steps, d_sae


# Confound counts a "powered" run may raise (each confound column needs >= 5 independent
# nodes for valid inference). These are the only overridable knobs; resolve_config and
# train_toy both read this tuple.
CONFOUND_OVERRIDES: tuple[str, ...] = (
    "n_superparent", "n_token_bound_pairs", "n_topical_pairs", "n_bind_ids",
)

# Short, stable abbreviations for the checkpoint-dir suffix (sp6-tp8, ...).
_OVERRIDE_ABBR: dict[str, str] = {
    "n_superparent": "sp", "n_token_bound_pairs": "tb",
    "n_topical_pairs": "tp", "n_bind_ids": "bi",
}


def choose_k(tree: tree_mod.Tree, k_override: int | None = None) -> int:
    """The SAE's BatchTopK `k`, derived from the world's true L0 (never a starved value).

    Defaults to `round(target_l0(tree))` -- the mean active features per token, which is
    what the dictionary's top-k must be wide enough to represent. An explicit override is
    honoured only if it does not starve the dictionary: a `k_override` below the derived
    value drops real active features on the average token and reads downstream as recovery
    failure, so it raises rather than silently training a starved SAE. Over-provisioning
    (`k_override >= derived`) is allowed and returned unchanged.
    """
    l0 = strengths.target_l0(tree)
    # round() is round-half-to-even; safe because L0 is a sum of continuous rate products
    # (never exactly x.5).
    derived = round(l0)
    if k_override is None:
        return derived
    if k_override < derived:
        raise ValueError(
            f"k={k_override} starves a world with true L0 ~= {l0:.1f} (derived k={derived}); "
            f"the SAE could not represent an average token. Raise k to >= {derived}, or omit "
            f"--k to derive it from the world.")
    return k_override


def resolve_config(cfg_name: str, **overrides: int) -> spec.ToyConfig:
    """Return the named base config with the given confound counts overridden.

    `overrides` may name only the knobs in `CONFOUND_OVERRIDES`; anything else raises
    (so a typo'd knob fails loudly instead of silently doing nothing). Only the non-None
    overrides are applied. Three mistakes are refused loudly rather than silently
    training the wrong world:
      - a confound-count override against a config whose `confounds` are OFF (the
        generator ignores those counts unless `confounds=True`, so the run would be
        "powered" in name only),
      - a negative count (the generator would clamp it to 0, quietly un-powering), and
      - a bool (an int subclass that would slip past `< 0` and resolve True->1).
    Feasibility of the resulting config is still enforced downstream by `build_tree`.
    """
    if cfg_name not in spec.CONFIGS:
        raise ValueError(
            f"unknown config {cfg_name!r}; known: {sorted(spec.CONFIGS)}")
    unknown = set(overrides) - set(CONFOUND_OVERRIDES)
    if unknown:
        raise ValueError(
            f"unknown override(s) {sorted(unknown)}; overridable: "
            f"{list(CONFOUND_OVERRIDES)}")
    overrides = {name: val for name, val in overrides.items() if val is not None}

    base = spec.CONFIGS[cfg_name]()
    if not overrides:
        return base
    if not base.confounds:
        raise ValueError(
            f"config {cfg_name!r} has confounds disabled, so the overrides "
            f"{sorted(overrides)} would be silently ignored (the generator reads the "
            f"confound counts only when confounds=True); refusing to build a 'powered' "
            f"world that is powered in name only")
    bad = {name: val for name, val in overrides.items()
           if isinstance(val, bool) or val < 0}
    if bad:
        raise ValueError(f"confound counts must be non-negative ints (not bool); got {bad}")
    return spec.replace(base, **overrides)


def checkpoint_dirname(config_name: str, variant: str, k: int, expansion: int,
                       overrides: dict[str, int], seed: int | None = None) -> str:
    """Checkpoint directory name, disambiguated by any active confound overrides and the seed.

    With no overrides this is the legacy stem `{config_name}-{variant}-k{k}-x{expansion}`,
    so existing un-powered checkpoints keep their names. With overrides a deterministic
    `-pow-<tag>` suffix (sorted by key, so insertion order is irrelevant) is appended,
    so two differently-powered runs never collide on disk.

    `seed` (opt-in) appends a terminal `-s{seed}` so two seeds trained into the same `--out`
    directory never collide on disk and silently clobber each other's weights. Default
    `None` reproduces the legacy name, so callers/checkpoints that pre-date this stay resolvable.
    """
    stem = f"{config_name}-{variant}-k{k}-x{expansion}"
    if overrides:
        tag = "-".join(f"{_OVERRIDE_ABBR.get(key, key)}{overrides[key]}"
                       for key in sorted(overrides))
        stem = f"{stem}-pow-{tag}"
    if seed is not None:
        stem = f"{stem}-s{int(seed)}"
    return stem


def build_world(cfg_name: str, n_tokens: int, seed: int, device: str,
                config: spec.ToyConfig | None = None,
                ) -> tuple[torch.Tensor, tree_mod.Tree, spec.ToyConfig]:
    """Generate one toy and return (activations, tree, config).

    float32 on `device`: the generator works in float64 internally, but the SAE trains
    in float32 and an 8M x D double buffer would be pure overhead. Geometry is drawn
    from the config's own seed; the sampling draw uses `seed`, so a disjoint seed gives
    an independent world over the same design.

    `config` lets a caller pass a pre-resolved `ToyConfig` (e.g. a powered variant from
    `resolve_config`) to use verbatim; when omitted the base recipe named by `cfg_name`
    is built, preserving the original call path. A passed `config` must agree with
    `cfg_name` (`config.name == cfg_name`), so a caller can't build one world under
    another world's label.
    """
    if config is not None and config.name != cfg_name:
        raise ValueError(
            f"config.name ({config.name!r}) != cfg_name ({cfg_name!r}); a passed config "
            f"must match the name it is built under, or the world and its label diverge")
    cfg = config if config is not None else spec.CONFIGS[cfg_name]()
    tree = tree_mod.build_tree(cfg)
    strength = strengths.build_strengths(cfg, tree)
    geo = geometry.build_directions(cfg, tree, seed=cfg.seed)
    world = sample.sample_world(cfg, tree, strength, geo, n_tokens=n_tokens, seed=seed)
    return world.h.to(device=device, dtype=torch.float32), tree, cfg
