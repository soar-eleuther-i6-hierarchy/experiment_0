"""
Write the landing page for each layer: outputs/layer_NN/README.md.

Without one, outputs/layer_NN/ is a 404 on GitHub Pages -- the directory holds
five pages but nothing that answers "show me layer 3". That made the layer
pills in the nav bar unable to point at a layer as such; they had to pick one
of its pages. This closes that: every layer is now addressable on its own.

Needs nothing (no model, no cache, no stats) -- the page list is config.
Writes: outputs/layer_NN/README.md for every layer in config.NAV_LAYERS

Run:
    python3 -m reporting.layer_index          # all layers in NAV_LAYERS
    python3 -m reporting.layer_index --layer 6
"""
from __future__ import annotations

import argparse

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


def write(layer: int) -> None:
    path = C.OUT_DIR / f"layer_{layer:02d}" / "README.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render(layer))
    print(f"[index] wrote {path}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--layer", type=int, help="one layer instead of all of NAV_LAYERS")
    args = ap.parse_args()
    for layer in ([args.layer] if args.layer is not None else C.NAV_LAYERS):
        write(layer)


if __name__ == "__main__":
    main()
