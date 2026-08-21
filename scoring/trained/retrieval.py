"""
Tier-B retrieval driver — score one real checkpoint's detector x confound AUROC grid.

Matching is on the IN-SAMPLE draw; the detector statistics run on a HELD-OUT draw
(disjoint seed, same geometry). The reusable AUROC/CI/ensemble toolkit lives in
`scoring.core.grid`; this module only wires a loaded checkpoint through it, adds the
oracle-ceiling survival-Δ (`scoring.oracle.ceiling`), and writes the report.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import torch

from scoring.core.detectors import compute_all, s_res_variants
from scoring.core.grid import (
    _ensemble_macro_ci,
    _ensemble_pooled_auroc,
    auroc_matrix,
    class_members,
    dispersion_split,
    held_out_sample_seed,
    mean_percentile_ensemble,
    min_percentile_ensemble,
    pair_frame,
    random_scalar_control,
    redundancy_map,
    reduce_to_recovered,
    shuffled_label_control,
)
from scoring.core.registry import (
    CONSTANTS,
    ENSEMBLE_DETECTORS,
    POSITIVE_LABEL,
    SCORED_COLUMNS,
)
from toygen import labels

# The probe is the SOLE s_res mode/ceiling that may feed a reported cell. These constants make that a
# single source of truth for the report path (trained detector + probe-on-oracle ceiling); the guard
# below asserts the probe regime was actually stamped, so a silent regression to the cosine geometry
# oracle (a diagnostic-only variant) trips immediately instead of shipping a cross-metric s_res number.
_REPORT_S_RES_MODE = "probe"
_REPORT_S_RES_CEILING = "probe"


def _assert_s_res_is_probe(stage0_grid: dict) -> None:
    """Fail loudly if the reported oracle s_res was not read from the probe ceiling.

    The probe ceiling (`s_res_ceiling="probe"`) stamps every s_res cell `oracle_regime =
    'alpha_encoder_probe'`. Any other label means s_res fell back to the cosine geometry oracle -- a
    diagnostic variant that must never feed a reported cell (it would be a cross-metric ceiling, not
    the like-for-like probe-vs-probe reading the report claims)."""
    # An explicit raise, NOT `assert` — this guards a REPORTED science number, so it must survive
    # `python -O` (which strips asserts); a stripped guard would let a cosine ceiling ship as the probe.
    s_res_cols = stage0_grid.get("s_res")
    if not s_res_cols:
        raise RuntimeError("reported Stage-0 grid is missing s_res (it is agnostic and always kept)")
    for col, cell in s_res_cols.items():
        regime = cell.get("oracle_regime")
        if regime != "alpha_encoder_probe":
            raise RuntimeError(
                f"reported s_res ceiling regressed to {regime!r} at column {col!r}; the report path "
                f"must use the probe ceiling (_REPORT_S_RES_CEILING='probe'), not the cosine oracle")


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
                                       match_features, realized_l0)

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
    # Realized L0 on the HELD-OUT draw — the sparsity the Stage-0 oracle encoder is calibrated to,
    # so the clean ceiling is read at the SAME L0 as the trained metric (not the nominal top-k).
    ho_realized_l0 = realized_l0(acts_ho)

    feats, di, _ = reduce_to_recovered(
        acts_ho, oriented_ho, W_raw, res.match, res.recovered,
        h=ho.h, b_dec=loaded.b_dec, tokens=ho.tokens, vocab=ho.cfg.vocab)
    # The trained retrieval detector uses the real Tree-SAE probe s_res (one probe per recovered
    # child, self-label). The Stage-0 ceiling below ALSO uses the probe (like-for-like, s_res_ceiling
    # =_REPORT_S_RES_CEILING) -- the cheap cosine geometry oracle is a diagnostic only and never a
    # reported ceiling.
    detectors = compute_all(di, CONSTANTS, s_res_mode=_REPORT_S_RES_MODE)
    pairs, y_label = pair_frame(feats, ho.pair_labels)
    all_columns = SCORED_COLUMNS
    grid = auroc_matrix(detectors, pairs, y_label, all_columns)
    per_class_rec = _per_class_recovery(res.recovered, ho.pair_labels, feats)

    # Ensemble vs class-balanced pooled negatives (H6). Over the GENERAL is-a detectors
    # only (ENSEMBLE_DETECTORS) — a per-parent specialist inside the gate vetoes every is-a
    # pair. MEAN-percentile is H6's primary (min is fragile to a single dead gate); the min
    # is reported as a robustness number.
    ens_subset = {d: detectors[d] for d in ENSEMBLE_DETECTORS}
    ens_mean = mean_percentile_ensemble(ens_subset, pairs)
    ens_min = min_percentile_ensemble(ens_subset, pairs)
    ens_h6 = _ensemble_pooled_auroc(ens_mean, pairs, y_label)
    ens_h6["ci_lo"], ens_h6["ci_hi"] = _ensemble_macro_ci(ens_mean, y_label)   # H6 variance bound
    ens_h6_min = _ensemble_pooled_auroc(ens_min, pairs, y_label)

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

    # Stage-0 ORACLE ceiling grid: "can each metric identify its property at its own ideal?" — reported
    # ALONGSIDE the trained grid ("can it retrieve it from SAE latents?"), as two standalone answers.
    # There is no clean-vs-trained subtraction: per-detector oracles (α-encoder / true-A / probe) make a
    # single-reference Δ invalid, so the reader compares the two grids directly. The oracle-ceiling
    # toolkit reads `auroc_matrix` from `scoring.core.grid`, so this driver-local import only defers the
    # (light) module load.
    from scoring.oracle.ceiling import (DETECTOR_ORACLE_REGIME, REGIME_CAVEAT, OracleEncodeInfeasible,
                                        annotate_clean_grid, clean_grid_and_degenerate, stage0_caveats)
    # A held-out draw whose realized L0 is 0 (a fully dead SAE) cannot calibrate the oracle-encoder
    # ceiling — the oracle_encode gate would have no positive projections to threshold. Refuse to
    # fabricate a Stage-0 ceiling in that case rather than emit a garbage 0.5 grid; the recovery
    # side already reports the dead SAE. (A partial-but-positive L0 is fine.)
    stage0, stage0_caveat_cells, stage0_degenerate = {}, [], []
    if ho_realized_l0 > 0 and feats:
        try:
            # Per-detector oracle routing: each detector's ceiling is read off the firing regime that
            # makes its property faithful (containment on the α-encoder, firing-structure/recon on true A).
            # ONE clean-detectors pass yields both the grid and the degenerate set (no double probe-train).
            stage0, stage0_degenerate = clean_grid_and_degenerate(
                ho, feats, pairs, y_label, all_columns, CONSTANTS, ho_realized_l0,
                routing=DETECTOR_ORACLE_REGIME, s_res_ceiling=_REPORT_S_RES_CEILING)
            # Stamp each ORACLE cell with `worked`/`degenerate`/`oracle_regime`/`oracle_idealized` so the
            # clean grid stands alone; `stage0_caveats` then surfaces the degenerate / never-worked cells.
            annotate_clean_grid(stage0, stage0_degenerate, DETECTOR_ORACLE_REGIME,
                                s_res_ceiling=_REPORT_S_RES_CEILING)
            stage0_caveat_cells = stage0_caveats(stage0)
            # Guard: the reported s_res ceiling MUST be the probe, never the cosine geometry oracle. The
            # probe ceiling stamps every s_res cell `alpha_encoder_probe`; anything else means a silent
            # regression to cosine (a cross-metric ceiling) -- fail loudly rather than ship it.
            _assert_s_res_is_probe(stage0)
        except OracleEncodeInfeasible as e:
            # NARROW: catch ONLY the oracle-encoder infeasibility (a too-DENSE held-out draw whose mean
            # L0 needs more firing entries than the tied-unit-g gate can produce), even when
            # realized_l0 > 0. A compute_all config error (bad s_res_mode / missing inputs) raises a
            # bare ValueError and must NOT be caught here — it would be mislabeled `stage0_unavailable`
            # instead of surfacing. Emit an empty Stage-0 grid + a caveat, like the dead-SAE branch.
            stage0, stage0_degenerate = {}, []
            stage0_caveat_cells = [{"detector": "*", "column": "*",
                                    "status": "stage0_unavailable", "reason": str(e)}]

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
        "ensemble_detectors": list(ENSEMBLE_DETECTORS),
        "grid": grid, "per_class_recovery": per_class_rec,
        "stage0_clean_grid": stage0,
        "stage0_caveats": stage0_caveat_cells,
        "stage0_degenerate_detectors": stage0_degenerate,
        "stage0_oracle_regime": dict(DETECTOR_ORACLE_REGIME),
        "stage0_regime_caveat": REGIME_CAVEAT,
        "grids_note": ("TWO standalone readouts, NOT a subtraction: `stage0_clean_grid` = the metric "
                       "at its own oracle ideal (per-detector: α-encoder / true-A / probe; a true-A "
                       "`oracle_idealized` cell is an UNATTAINABLE ideal, read as characterization); "
                       "`grid` = retrieval from the trained SAE latents. Compare them directly; a high "
                       "oracle cell with a low trained cell = the metric works at its ideal but the SAE "
                       "loses it. s_res: oracle = probe on the α-encoder firing, trained = probe on the "
                       "learned dict."),
        "realized_l0": ho_realized_l0, "architecture": loaded.arch,
        "dispersion_mean": dispersion_mean,
        "ensemble_h6": ens_h6, "ensemble_h6_min_robustness": ens_h6_min,
        "dispersion_split": disp,
        "redundancy": redundancy_map(detectors, pairs, y_label, SCORED_COLUMNS, grid=grid),
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
            row = auroc_matrix({"s": mat}, pairs, y_label, all_columns)["s"]
            return {c: row.get(c, {}).get("auroc") for c in all_columns}

        aurocs = {name: _isa_row(mat) for name, mat in variants.items()}

        def _fin(x) -> bool:
            return isinstance(x, float) and math.isfinite(x)
        bias = {c: (aurocs["probe_self_W"][c] - aurocs["probe_true_W"][c])
                if (_fin(aurocs["probe_self_W"][c]) and _fin(aurocs["probe_true_W"][c]))
                else float("nan") for c in all_columns}
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


def _per_class_recovery(recovered: torch.Tensor, pair_labels: torch.Tensor,
                        feats: list[int]) -> dict[str, list[int]]:
    F = pair_labels.shape[0]
    eye = torch.eye(F, dtype=torch.bool)
    both = recovered[:, None] & recovered[None, :]
    out = {}
    for name in labels.LABELS:
        # Single-label equality; mask the diagonal because is_a is index 0 == the diagonal.
        in_cls = (pair_labels == labels._index(name)) & ~eye
        out[name] = [int((in_cls & both).sum()), int(in_cls.sum())]
    return out


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
