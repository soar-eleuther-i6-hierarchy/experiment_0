"""
Trained retrieval driver — score one real checkpoint's property-vs-rest AUROC grid.

Matching is on the IN-SAMPLE draw; the detector statistics run on a HELD-OUT draw
(disjoint seed, same geometry). The reusable AUROC/CI toolkit lives in
`scoring.core.grid`; this module only wires a loaded checkpoint through it and writes the
report. Every generative class is scored property-vs-rest (that class = positives vs every
other off-diagonal pair) by the 10 firewalled detectors; the greedy Boolean cascade over
this grid is applied downstream.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import torch

from scoring.core.detectors import compute_all, s_res_variants
from scoring.core.grid import (
    auroc_matrix,
    class_members,
    dispersion_split,
    held_out_sample_seed,
    pair_frame,
    property_vs_rest_grid,
    random_scalar_control,
    redundancy_map,
    reduce_to_recovered,
    shuffled_label_control,
)
from scoring.core.registry import (
    CONSTANTS,
    LATENT_COLUMNS,
    POSITIVE_LABEL,
    SCORED_COLUMNS,
)
from scoring.trained.cascade import cascade_grid
from toygen import labels

# The probe is the SOLE s_res mode that may feed a reported cell (the trained detector). The cheap
# cosine geometry oracle is a diagnostic/calibration variant only, never a reported detector.
_REPORT_S_RES_MODE = "probe"


def run_retrieval(ckpt_dir: str | Path, n_tokens: int = 200_000, rho: float | None = None,
                  out: str | Path | None = None, s_res_diagnostics: bool = False) -> dict:
    """Score one checkpoint's Tier-B retrieval grid (matching in-sample, detectors held-out).

    Not unit-tested (loads a real checkpoint via `scoring.trained.loaders.load_sae`);
    validated on the server.
    """
    # Driver-local imports: `load_sae` pulls the heavy `sae_training` backend, so keep it out of
    # module import (this file must stay importable in the CPU test env). The rest are grouped with
    # it — the module-level toolkit above is all `scoring.core`, so there is no import cycle to dodge.
    from scoring.trained.loaders import load_sae
    from scoring.core.world import regenerate_world, signed_normalized_decoder
    from scoring.core.recovery import (activation_corr, child_direction_dispersion,
                                       match_features, per_class_recovery, realized_l0)

    rho = CONSTANTS["rho_star"] if rho is None else rho
    loaded = load_sae(ckpt_dir)
    meta = loaded.meta
    W_raw = loaded.W_dec

    # 1. Tier-A match on the IN-SAMPLE draw.
    in_world = regenerate_world(meta["resolved_config"], sample_seed=meta["train_seed"],
                                n_tokens=n_tokens)
    acts_in = loaded.encode(in_world.h)
    oriented_in = signed_normalized_decoder(W_raw, acts_in, in_world.h)
    res = match_features(activation_corr(in_world.A, acts_in), in_world.g, oriented_in, rho=rho)

    # 2. HELD-OUT draw for the detector statistics (disjoint seed, same geometry).
    ho_seed = held_out_sample_seed(int(meta["train_seed"]))
    ho = regenerate_world(meta["resolved_config"], sample_seed=ho_seed, n_tokens=n_tokens)
    acts_ho = loaded.encode(ho.h)
    oriented_ho = signed_normalized_decoder(W_raw, acts_ho, ho.h)
    # Realized L0 on the HELD-OUT draw — reported provenance for the sparsity the detectors see.
    ho_realized_l0 = realized_l0(acts_ho)

    feats, di, _ = reduce_to_recovered(
        acts_ho, oriented_ho, W_raw, res.match, res.recovered,
        h=ho.h, b_dec=loaded.b_dec, tokens=ho.tokens, vocab=ho.cfg.vocab)
    # The trained retrieval detector uses the real Tree-SAE probe s_res (one probe per recovered
    # child, self-label); the cheap cosine geometry oracle is a diagnostic only, never reported.
    detectors = compute_all(di, CONSTANTS, s_res_mode=_REPORT_S_RES_MODE)
    pairs, y_label = pair_frame(feats, ho.pair_labels)

    # Latent-side labels (absorbed / merged / split): the Chanin dictionary-damage classification on
    # the SAME in-sample match, joined onto the recovered pair universe. FIREWALL: truth+trained-
    # derived labels used ONLY for scoring — the 10 detectors never see them. Absorption is a property
    # of the dictionary + in-sample firing, so it is measured on the in-sample draw (as run_absorption
    # does), not the held-out detector draw.
    from scoring.trained.absorption import (classify_dictionary, latent_pair_masks, split_readout,
                                            tree_edges_and_siblings)
    cont_edges, sibling_pairs, isa_child = tree_edges_and_siblings(in_world.tree)
    classification = classify_dictionary(in_world.g, oriented_in, in_world.A, acts_in,
                                         res.match, res.matched_corr, res.recovered,
                                         cont_edges, sibling_pairs, isa_child)
    latent_masks = latent_pair_masks(classification, feats, pairs)

    # Property-vs-rest over EVERY generative class + the latent-side columns (each class = positives
    # vs every other off-diagonal pair), scored by the 10 firewalled detectors. Replaces the is_a-
    # locked auroc_matrix: a detector that isolates, say, `frequency` now shows on the frequency
    # column even though it is useless for is_a. Orientation stays frozen (AUROC<0.5 is a reported
    # inverted isolator). The greedy Boolean cascade reads this grid downstream.
    grid_columns = tuple(labels.LABELS) + LATENT_COLUMNS
    grid = property_vs_rest_grid(detectors, pairs, y_label, grid_columns, label_masks=latent_masks)
    per_class_rec = per_class_recovery(res.recovered, ho.pair_labels, list(labels.LABELS))

    # Greedy both-tails Boolean cascade over the grid: per column a readable percentile rule that
    # isolates that class among survivors. is_a's rule is the deployment cascade; its readout carries
    # enrichment-over-base-rate + the hard-negative precision vs {transitive, reversed, firing_only}
    # (a pooled vs-rest AUROC hides those confusable cousins). 0-positive columns are skipped, not
    # crashed. Reads only the firewalled detectors + the answer key (+ latent masks for absorbed/merged).
    cascade = cascade_grid(detectors, pairs, y_label, grid_columns, label_masks=latent_masks)

    # dispersion-conditioned is-a readout: r_disp over the FULL held-out dictionary,
    # indexed by true feature id; a truth-g DIAGNOSTIC on the readout, never a detector.
    r_disp_vec = child_direction_dispersion(oriented_ho, ho.g, res.match, feats,
                                            m=CONSTANTS["r_disp_m"])
    isa_m = class_members(y_label, POSITIVE_LABEL).tolist()
    fo_m = class_members(y_label, "firing_only").tolist()
    isa_pairs = [pr for pr, inside in zip(pairs, isa_m) if inside]
    fo_pairs = [pr for pr, inside in zip(pairs, fo_m) if inside]
    r_disp = {b: float(r_disp_vec[b]) for _, b in isa_pairs}   # b == child position == feats index
    disp = (dispersion_split(isa_pairs, r_disp, detectors["s_res"], neg_pairs=fo_pairs)
            if isa_pairs and fo_pairs else None)
    # Mean over the IS-A CHILDREN (r_disp is already keyed to them), NOT all recovered features
    # (roots are well-recovered and would pull it down) — so the sweep's dispersion axis matches
    # Tier-A's dispersion_mean and the clean floor (~0.33) it is read against.
    dispersion_mean = float(sum(r_disp.values()) / len(r_disp)) if r_disp else float("nan")

    rng = torch.Generator().manual_seed(0)
    controls = {
        "random_scalar_firing_only": random_scalar_control(pairs, y_label, "firing_only", rng),
        "shuffled_s_res_firing_only": shuffled_label_control(
            detectors["s_res"], pairs, y_label, "firing_only", rng),
    }
    report = {
        "checkpoint": str(ckpt_dir),
        "meta": {k: meta[k] for k in ("config", "variant", "k", "train_seed", "overrides")
                 if k in meta},
        "n_recovered": len(feats), "n_pairs": len(pairs),
        "ci_note": ("per-cell grid ci_lo/ci_hi are logit CIs under a pair-INDEPENDENCE "
                    "assumption (anticonservative: pairs derive from ~F features, not n_pairs "
                    "independent draws); the reportable interval is the across-seed Student-t "
                    "from aggregate_seeds, not a single seed's cell CI."),
        "grid_note": ("property-vs-rest: each column = that class as positives vs EVERY other "
                      "off-diagonal pair as negatives, scored by the 10 firewalled detectors; "
                      "orientation frozen (AUROC<0.5 = an inverted isolator, reported as-is). "
                      "Generative columns come from pair_labels; latent columns (absorbed/merged) "
                      "from the absorption classification."),
        "grid": grid, "per_class_recovery": per_class_rec,
        "cascade": cascade,
        "cascade_note": ("greedy forward-selection of percentile filters (one per detector, BOTH "
                         "tails tried at each step) maximizing F1 of the column among survivors; "
                         "distribution-free global-pool thresholds. is_a's rule is the deployment "
                         "cascade — its readout adds enrichment (final precision / base rate) and a "
                         "hard-negative precision vs {transitive, reversed, firing_only}. 0-positive "
                         "columns are skipped."),
        "latent_columns": list(LATENT_COLUMNS),
        # NOT re-reporting the absorption `counts` here: `run_absorption` is the single source of truth
        # for the full decomposition, and the grid's absorbed/merged n_pos already carries the
        # recovered-universe count. Duplicating it would risk a silent cross-report divergence if the
        # two drivers were ever run with different n_tokens/rho. `split_readout` IS reported here — it is
        # the per-latent readout scoped to the recovered `feats`, not part of run_absorption's report.
        "split_readout": split_readout(classification, feats),
        "latent_firewall_note": ("absorbed/merged/split are truth+trained-derived labels (the Chanin "
                                 "classification) used ONLY for scoring — the 10 detectors never see "
                                 "them; resid_parent/K DEFINE these labels but are NOT detectors."),
        "realized_l0": ho_realized_l0, "architecture": loaded.arch,
        "dispersion_mean": dispersion_mean,
        "dispersion_split": disp,
        "redundancy": redundancy_map(detectors, pairs, y_label, SCORED_COLUMNS, grid=grid),
        "redundancy_note": ("marginal_auroc is computed from the property-vs-rest `grid`, so it is each "
                            "detector's added COLUMN-vs-rest AUROC (NOT the legacy is_a-vs-C marginal); "
                            "rank_corr is grid-independent (detector-detector percentile correlation)."),
        "controls": controls,
    }
    # J(p) fallback-vs-exact diagnostic: a readout on joint_child_J's accuracy, never a scored
    # detector (kept out of the DETECTORS grid).
    from scoring.trained.diagnostics import j_fallback_vs_exact
    report["j_supp_diagnostic"] = j_fallback_vs_exact(di, CONSTANTS)

    # OPT-IN s_res 3-variant diagnostic (trains extra probes → off by default so a standard run is
    # one-probe-per-child). Uses ground-truth g/A, so it is a DIAGNOSTIC, never a firewalled detector.
    if s_res_diagnostics:
        _TINY = 1e-12
        idx = torch.tensor(feats, dtype=torch.long)
        g_sel = ho.g[idx]
        g_unit = g_sel / g_sel.norm(dim=1, keepdim=True).clamp_min(_TINY)
        variants = s_res_variants(di.acts_rec, ho.h, di.W_unit, g_unit, ho.A[:, idx], CONSTANTS)

        def _isa_row(mat: torch.Tensor) -> dict:
            # is_a-vs-column diagnostic (auroc_matrix is is_a-locked), over the confound columns.
            row = auroc_matrix({"s": mat}, pairs, y_label, SCORED_COLUMNS)["s"]
            return {c: row.get(c, {}).get("auroc") for c in SCORED_COLUMNS}

        aurocs = {name: _isa_row(mat) for name, mat in variants.items()}

        def _fin(x) -> bool:
            return isinstance(x, float) and math.isfinite(x)
        bias = {c: (aurocs["probe_self_W"][c] - aurocs["probe_true_W"][c])
                if (_fin(aurocs["probe_self_W"][c]) and _fin(aurocs["probe_true_W"][c]))
                else float("nan") for c in SCORED_COLUMNS}
        report["s_res_diagnostics"] = {
            "auroc": aurocs, "self_label_bias": bias,
            "note": ("cosine_g = analytic geometry oracle; probe_true_g calibrates the probe "
                     "machinery (≈ cosine_g once knobs are tuned); probe_self_W = the deployed "
                     "detector; self_label_bias = probe_self_W − probe_true_W isolates the "
                     "self-label circularity (measurable only on the toy)."),
        }
    if out is not None:
        Path(out).write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description="Tier-B retrieval scorer for a toy SAE checkpoint.")
    ap.add_argument("ckpt", type=Path)
    ap.add_argument("--n-tokens", type=int, default=200_000)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--s-res-diagnostics", action="store_true",
                    help="also emit the s_res 3-variant diagnostic (trains extra probes)")
    args = ap.parse_args()
    report = run_retrieval(args.ckpt, n_tokens=args.n_tokens, out=args.out,
                           s_res_diagnostics=args.s_res_diagnostics)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
