"""Stage-1 oracle validation driver: run the mathematical-validation checks on a toy world
and save the result as JSON (machine record) + a readable Markdown report.

Stage 1 asks whether each metric is a well-formed instrument on a perfect dictionary — finiteness,
non-degeneracy, a faithful alpha-encoder reconstruction, and the s_res probe tracking analytic cosine.
It is NOT AUROC (that is Stage 2, scoring.run_scoring).

Usage (from the experiment_0 directory)::

    python -m scoring.run_validate --config full --out outputs_local/final
"""

from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path

import torch

from scoring.core.grid import pair_frame
from scoring.core.registry import DETECTORS
from scoring.core.world import regenerate_world
from scoring.oracle.score_dump import oracle_scores, summarize, write_score_reports
from scoring.oracle.validate_metrics import (
    machinery_report, oracle_encode, reconstruction_fvu, s_res_calibration,
)
from toygen import labels, strengths
from toygen.world import resolve_config

_TINY = 1e-12
FVU_MAX = 0.25            # alpha-encoder reconstruction must beat this to be a faithful stand-in
CALIBRATION_FLOOR = 0.60  # s_res probe vs analytic-cosine Spearman floor


def _reconstruction_compare(bundle, realized_l0: float) -> dict:
    """FVU of the naive tied projection vs the gated ridge-LS encoder on the same firing support."""
    support = oracle_encode(bundle.h, bundle.g, realized_l0)
    g_unit = bundle.g.double() / bundle.g.double().norm(dim=1, keepdim=True).clamp_min(_TINY)
    tied = (bundle.h.double() @ g_unit.T) * (support != 0)
    return {"tied_fvu": reconstruction_fvu(bundle.h, tied, bundle.g),
            "gated_fvu": reconstruction_fvu(bundle.h, support, bundle.g)}


def build_validation_world(config: str = "full", n_tokens: int = 30_000,
                           seed: int | None = None):
    """Build the toy world + recovered-feature list + pair frame once.

    Returned as `(bundle, feats, pairs, sample_seed, true_l0)` so the gate report and the
    oracle score dump can read the SAME world (identical under a fixed seed) instead of
    regenerating it twice.
    """
    rc = dataclasses.asdict(resolve_config(config))
    sample_seed = rc["seed"] if seed is None else seed
    bundle = regenerate_world(rc, sample_seed=sample_seed, n_tokens=n_tokens)
    feats = list(range(bundle.g.shape[0]))
    pairs, _ = pair_frame(feats, bundle.pair_labels)
    true_l0 = float(strengths.firing_rates(bundle.tree).sum())            # the world's true L0
    return bundle, feats, pairs, sample_seed, true_l0


def run_validation(config: str = "full", n_tokens: int = 30_000, seed: int | None = None,
                   device: str | None = None, realized_l0: float | None = None,
                   prebuilt: tuple | None = None) -> dict:
    """Build the world and run the Stage-1 checks; return the full validation report as a dict.

    `prebuilt` accepts a `build_validation_world` tuple so a caller that also dumps oracle
    scores can share one world build; when None the world is built here.
    """
    if prebuilt is None:
        prebuilt = build_validation_world(config, n_tokens, seed)
    bundle, feats, pairs, sample_seed, true_l0 = prebuilt
    if realized_l0 is None:
        realized_l0 = true_l0                                             # the world's true L0
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    machinery = machinery_report(bundle, feats, pairs, realized_l0)
    calibration = s_res_calibration(bundle, feats, device=device)
    reconstruction = _reconstruction_compare(bundle, realized_l0)

    fvu, spearman = machinery["alpha_encoder_fvu"], calibration["spearman"]
    gates = {
        "machinery_all_passed": bool(machinery["all_passed"]),
        "fvu_ok": bool(fvu == fvu and fvu < FVU_MAX),                       # NaN-safe
        "calibration_ok": bool(spearman == spearman and spearman >= CALIBRATION_FLOOR),
    }
    gates["overall_pass"] = all(gates.values())
    return {
        # Derived from the built world (not the passed args) so a prebuilt-world caller
        # can't make the report claim a config/token-count it wasn't computed on.
        "meta": {"config": bundle.cfg.name, "F": bundle.g.shape[0], "D": bundle.g.shape[1],
                 "n_tokens": int(bundle.A.shape[0]), "n_pairs": len(pairs), "seed": sample_seed,
                 "realized_l0": realized_l0, "device": device},
        "machinery": machinery,
        "s_res_calibration": calibration,
        "reconstruction": reconstruction,
        "gates": gates,
    }


def format_report_md(report: dict) -> str:
    """Render the validation report as a readable Markdown gate report."""
    m, rep, cal = report["meta"], report["machinery"], report["s_res_calibration"]
    rec, g = report["reconstruction"], report["gates"]
    yes = lambda ok: "yes" if ok else "NO"
    verdict = "PASS" if g["overall_pass"] else "FAIL"
    L = [f"# Stage 1 — oracle validation ({m['config']}): {verdict}", "",
         f"World: F={m['F']}  D={m['D']}  n_tokens={m['n_tokens']}  pairs={m['n_pairs']}  "
         f"seed={m['seed']}  realized_L0={m['realized_l0']:.2f}", "",
         "## Gates", "",
         "| gate | value | pass |", "|---|---|---|",
         f"| machinery (every detector finite + non-degenerate in >=1 regime) | all_passed={rep['all_passed']} | {yes(g['machinery_all_passed'])} |",
         f"| alpha-encoder reconstruction FVU < {FVU_MAX} | {rep['alpha_encoder_fvu']:.3f} | {yes(g['fvu_ok'])} |",
         f"| s_res probe vs cosine Spearman >= {CALIBRATION_FLOOR} | {cal['spearman']:.3f} (n_pairs={cal['n_pairs']}) | {yes(g['calibration_ok'])} |"]
    if rep["failed"]:
        L.append(f"\n**Failed detectors:** {', '.join(rep['failed'])}")

    L += ["", "## [1] machinery gate (per detector, per regime)", "",
          "| detector | true_A finite | true_A degenerate | alpha finite | alpha degenerate | PASS |",
          "|---|---|---|---|---|---|"]
    pr = rep["per_regime"]
    for d in DETECTORS:
        a, b = pr["true_A"][d], pr["alpha_encoder"][d]
        L.append(f"| {d} | {a['finite']} | {a['degenerate']} "
                 f"| {b['finite']} | {b['degenerate']} "
                 f"| {'PASS' if rep['passed'][d] else 'FAIL'} |")

    # domain coverage: finite (non-NaN) fraction per ground-truth class, per regime.
    classes = list(labels.LABELS)
    L += ["", "## domain coverage — finite fraction per ground-truth class", "",
          "_Where each metric is DEFINED (non-NaN), not where it is correct — that is Stage 2. "
          "1.00 on a class it measures + low on classes it can't = healthy abstention; "
          "a low value on a class it SHOULD score = breakage._"]
    for regime in ("true_A", "alpha_encoder"):
        L += ["", f"### {regime}", "",
              "| detector | " + " | ".join(classes) + " |",
              "|" + "---|" * (len(classes) + 1)]
        for d in DETECTORS:
            fbc = pr[regime][d]["finite_by_class"]
            cells = " | ".join(f"{fbc[c]:.2f}" if fbc[c] == fbc[c] else "--" for c in classes)
            L.append(f"| {d} | {cells} |")

    L += ["", "## [2] s_res probe calibration (probe vs analytic cosine, truth-only)", "",
          f"- n_pairs = {cal['n_pairs']}",
          f"- Spearman = {cal['spearman']:.3f}  (floor {CALIBRATION_FLOOR})",
          f"- Pearson = {cal['pearson']:.3f}",
          f"- mean|diff| = {cal['mean_abs_diff']:.3f}",
          "", "## [3] alpha-encoder reconstruction", "",
          f"- tied projection FVU = {rec['tied_fvu']:.3f}",
          f"- gated ridge-LS FVU  = {rec['gated_fvu']:.3f}  (target ~0.14)", ""]
    return "\n".join(L) + "\n"


def write_reports(report: dict, out: Path) -> tuple[Path, Path]:
    """Write stage1_validation.json + stage1_validation.md into `out`; return the two paths."""
    out.mkdir(parents=True, exist_ok=True)
    json_path, md_path = out / "stage1_validation.json", out / "stage1_validation.md"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    md_path.write_text(format_report_md(report), encoding="utf-8")
    return json_path, md_path


def main() -> None:
    ap = argparse.ArgumentParser(description="Run Stage-1 oracle validation on a toy world.")
    ap.add_argument("--config", default="full")
    ap.add_argument("--n-tokens", type=int, default=30_000)
    ap.add_argument("--seed", type=int, default=None, help="sample seed; default = the config's seed")
    ap.add_argument("--realized-l0", type=float, default=None,
                    help="gate calibration target L0; default = the world's true L0")
    ap.add_argument("--out", type=Path, default=Path("outputs_local/final"))
    ap.add_argument("--device", default=None, help="cuda|cpu; default auto")
    ap.add_argument("--scores", action="store_true",
                    help="also dump oracle per-pair metric scores by class "
                         "(oracle_scores.npz + .md), s_res in cosine mode")
    args = ap.parse_args()

    prebuilt = (build_validation_world(args.config, args.n_tokens, args.seed)
                if args.scores else None)
    report = run_validation(args.config, args.n_tokens, args.seed, args.device,
                            args.realized_l0, prebuilt=prebuilt)
    print(format_report_md(report))
    json_path, md_path = write_reports(report, args.out)
    verdict = "PASS" if report["gates"]["overall_pass"] else "FAIL"
    print(f"stage-1 validation: {verdict}  ->  {json_path}  |  {md_path}")

    if args.scores:
        bundle, feats, pairs, sample_seed, true_l0 = prebuilt
        scores = oracle_scores(bundle, feats)                            # s_res cosine (oracle read)
        summary = summarize(scores)
        meta = {"config": bundle.cfg.name, "seed": sample_seed,
                "n_tokens": int(bundle.A.shape[0]), "F": int(bundle.g.shape[0]),
                "true_l0": true_l0, "n_pairs": len(pairs), "s_res_mode": "cosine"}
        npz_path, sc_md = write_score_reports(scores, summary, meta, args.out)
        print(f"oracle scores  ->  {npz_path}  |  {sc_md}")


if __name__ == "__main__":
    main()
