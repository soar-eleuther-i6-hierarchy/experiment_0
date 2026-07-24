"""
Tidy the run directory for browsing: sort generated artifacts into subfolders.

The analysis pipeline (cache_stats / run_metrics / run_second_pass / in_block_edges
/ visualize / make_report_figures) writes everything *flat* into RUN_DIR. This helper
moves the browsable artifacts into `dashboards/` and `reports/` while leaving the data
files the scripts read (exp0_stats.pt, feature_labels.json, token_cache/) exactly where
they expect them — so it never breaks a rerun.

Idempotent: safe to run repeatedly, and again after regenerating any artifact.

    python3 organize_outputs.py            # tidy RUN_DIR
    python3 organize_outputs.py --dry-run  # show what would move
"""

from __future__ import annotations

import argparse
import shutil

import config as C

# artifact basename -> destination subdir (relative to RUN_DIR).
# Anything not listed (exp0_stats.pt, feature_labels.json, token_cache/, figures/,
# ab_400docs/, pre_b3b4/, this script's data deps) is left untouched.
DASHBOARDS = [
    "metrics_dashboard.html",
    "superparent_sankey.html",
    "in_block_dashboard.html",
    "qualitative_dashboard.html",
]
REPORTS = [
    "metrics_report.md", "metrics_report.json",
    "second_pass.json",
    "in_block_edges.md", "in_block_edges.json",
    "qualitative_check.md", "qualitative_check.json",
]
LAYOUT = {"dashboards": DASHBOARDS, "reports": REPORTS}


def organize(run_dir, dry_run: bool) -> None:
    moved = kept = 0
    for subdir, names in LAYOUT.items():
        dest_dir = run_dir / subdir
        for name in names:
            src = run_dir / name
            if not src.exists():
                continue
            dest = dest_dir / name
            if src.resolve() == dest.resolve():          # already in place
                kept += 1
                continue
            if dry_run:
                print(f"[org] would move {name} -> {subdir}/")
                moved += 1
                continue
            dest_dir.mkdir(parents=True, exist_ok=True)
            shutil.move(str(src), str(dest))             # overwrites stale dest
            print(f"[org] moved {name} -> {subdir}/")
            moved += 1
    print(f"[org] {'would move' if dry_run else 'moved'} {moved} file(s); "
          f"{kept} already tidy. Data files (exp0_stats.pt, feature_labels.json, "
          f"token_cache/, figures/) left in place.")


INDEX = """\
# Run outputs — where to look

Read order (top = start here):

1. `reports/metrics_report.md` — the metrics per block pair (the main read).
2. `reports/second_pass.json` — the model-free pass: S_res (genuine refinement)
   + parent-conditioned sibling redundancy.
3. `reports/in_block_edges.md` — same-level (within-block) parent->child edges.
4. `reports/qualitative_check.md` — surviving vs rejected edges, with labels.

Folders:
- `dashboards/` — interactive HTML (a separate render step, `visualize.py`),
  one per report above.
- `figures/`    — static PNGs for the write-up (`make_report_figures.py`).
- `reports/`    — the `.md` / `.json` metric outputs.
- root files (`exp0_stats.pt`, `feature_labels.json`, `token_cache/`) are the
  inputs the scripts read; they stay put so reruns keep working.
"""


def write_index(run_dir, dry_run: bool) -> None:
    """Drop a short INDEX.md so anyone opening the run dir knows where to start."""
    if dry_run:
        print("[org] would write INDEX.md")
        return
    (run_dir / "INDEX.md").write_text(INDEX)
    print("[org] wrote INDEX.md")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    organize(C.RUN_DIR, args.dry_run)
    write_index(C.RUN_DIR, args.dry_run)


if __name__ == "__main__":
    main()
