"""
Write the landing page for each layer: outputs/layer_NN/README.md.

Without one, outputs/layer_NN/ is a 404 on GitHub Pages -- the directory holds
five pages but nothing that answers "show me layer 3". That made the layer
pills in the nav bar unable to point at a layer as such; they had to pick one
of its pages. This closes that: every layer is now addressable on its own.

Needs nothing (no model, no cache, no stats) -- the page list is config.
Writes: outputs/layer_NN/README.md for every layer in config.NAV_LAYERS

A run that is not a gemma layer (EXP0_RUN=pcfg) has the same problem and gets the
same treatment via --run, with two differences: its page list is whatever it
actually produced -- a PCFG SAE has no Neuronpedia labels, so no qualitative pages
-- and its heading comes from the run's own report rather than from this module's
gemma constants.

Run:
    python3 -m reporting.layer_index          # all layers in NAV_LAYERS
    python3 -m reporting.layer_index --layer 6
    EXP0_RUN=pcfg python3 -m reporting.layer_index --run
"""
from __future__ import annotations

import argparse
import json

import config as C

BLURB = {
    "metrics_dashboard.html": "filter funnel and the per-block-pair distributions",
    "superparent_sankey.html": "one superparent's fan-out to its children",
    "qualitative_dashboard.html": "surviving vs rejected edges, with Neuronpedia labels",
    "metrics_report.html": "the numbers behind the dashboard, as text",
    "qualitative_check.html": "survivor vs rejected edges read against the labels",
}


def render(layer: int) -> str:
    """The layer landing page: nav bar, then its five pages."""
    # page=None marks nothing in the Page group -- this page is the index, not
    # one of the five.
    L = [C.nav_html(depth=2, layer=layer), "", f"# Layer {layer:02d}", ""]
    L.append(
        f"The five pages for layer {layer} of `{C.MODEL_NAME}`'s residual stream, graded by the "
        f"metrics in [`metrics/`](../../metrics/README.md). Use the bar above to move between "
        f"layers while staying on the same page."
    )
    L += ["", "| Page | What is on it |", "| ---- | ------------- |"]
    for f, label in C.NAV_PAGES:
        L.append(f"| [{label}]({f}) | {BLURB.get(f, '')} |")
    L += [
        "",
        "Both reports are also in the repo as `.md`; the `.html` links above are what "
        "GitHub Pages renders. The `exp0_stats.pt` cache behind these numbers is not in git "
        f"-- see [outputs/README.md](../README.md#the-big-caches-are-not-in-git).",
        "",
    ]
    return "\n".join(L)


def render_run(run: str) -> str:
    """The landing page for a run that is not a gemma layer.

    Lists the pages this run actually produced rather than the fixed five: the
    qualitative pair needs Neuronpedia labels, which exist for gemma's dictionary
    and for no other. Promising a reader a page that is not there is worse than
    a shorter table.
    """
    run_dir = C.OUT_DIR / run
    report = run_dir / "metrics_report.json"
    if not report.is_file():
        raise SystemExit(f"[index] no {report} -- run run_metrics.py first")
    r = json.loads(report.read_text())
    cfg = r.get("config") or {}

    L = [C.nav_html(depth=2, current=f"outputs/{run}/"), "",
         f"# {run}", ""]
    L.append(C.scope_line(r.get("total_tokens"), n_docs=cfg.get("n_docs"), config=cfg))
    L += ["", "The same metric battery as the gemma layers, on an SAE from a different source. "
              "Nothing in `metrics/` changed to produce these numbers; the block structure, the "
              "model and the dictionary all come from the run's own cached statistics.", ""]
    L += ["| Page | What is on it |", "| ---- | ------------- |"]
    for f, label in C.NAV_PAGES:
        if (run_dir / f).exists() or (run_dir / f.replace(".html", ".md")).exists():
            L.append(f"| [{label}]({f}) | {BLURB.get(f, '')} |")
    L += ["", "The `exp0_stats.pt` cache and the token cache behind these numbers are not in git "
              "-- they are rebuildable from the run directory by `adapters/from_pcfg.py`.", ""]
    return "\n".join(L)


def write(layer: int) -> None:
    path = C.OUT_DIR / f"layer_{layer:02d}" / "README.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render(layer))
    print(f"[index] wrote {path}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--layer", type=int, help="one layer instead of all of NAV_LAYERS")
    ap.add_argument("--run", action="store_true",
                    help="write the index for EXP0_RUN's directory instead of a gemma layer")
    args = ap.parse_args()
    if args.run:
        path = C.RUN_DIR / "README.md"
        path.write_text(render_run(C.RUN_NAME))
        print(f"[index] wrote {path}")
        return
    for layer in ([args.layer] if args.layer is not None else C.NAV_LAYERS):
        write(layer)


if __name__ == "__main__":
    main()
