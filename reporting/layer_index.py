"""
Write the landing page for each layer: outputs/layer_NN/README.md.

Without one, the layer directory is a 404 on GitHub Pages -- it holds
per-layer pages but nothing that answers "show me layer 3". That made the layer
pills in the nav bar unable to point at a layer as such; they had to pick one
of its pages. This closes that: every layer is now addressable on its own.

Needs nothing (no model, no cache, no stats) -- the page list is config.
Writes: outputs/<source>/layer_NN/README.md for every layer in config.NAV_LAYERS
        (--source: the source directory's own index; --run: a non-layer run's)

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
import pathlib
import re

import config as C

BLURB = {
    "metrics_dashboard.html": "filter funnel and the per-block-pair distributions",
    "superparent_sankey.html": "one superparent's fan-out to its children",
    "in_block_dashboard.html": "same-level edges and duplicates inside one block, where no block ordering fixes direction",
    "qualitative_dashboard.html": "surviving vs rejected edges, with Neuronpedia labels",
    "metrics_report.html": "the numbers behind the dashboard, as text",
    "qualitative_check.html": "survivor vs rejected edges read against the labels",
}


def render(layer: int) -> str:
    """The layer landing page: nav bar, then the pages this source has."""
    # page=None marks nothing in the Page group -- this page is the index, not
    # one of the pages themselves.
    path = C.OUT_DIR / C.SOURCE_NAME / f"layer_{layer:02d}" / "README.md"
    depth = C.page_depth(path)
    # Relative links, computed from the page's depth rather than written out:
    # grouping results by source moved every layer page one level down, and every
    # hand-counted `../` in here pointed somewhere else afterwards.
    to_root, to_outputs = "../" * depth, "../" * (depth - 1)
    L = [C.nav_html(depth=depth, layer=layer,
                    current=f"outputs/{C.SOURCE_NAME}/layer_{layer:02d}/"), "",
         f"# Layer {layer:02d}", ""]
    L.append(
        f"The {len(C.NAV_PAGES)} pages for layer {layer} of `{C.MODEL_NAME}`'s residual stream, graded by the "
        f"metrics in [`metrics/`]({to_root}metrics/README.md). Use the bar above to move between "
        f"layers while staying on the same page."
    )
    L += ["", "| Page | What is on it |", "| ---- | ------------- |"]
    for f, label in C.NAV_PAGES:
        L.append(f"| [{label}]({f}) | {BLURB.get(f, '')} |")
    L += [
        "",
        "Both reports are also in the repo as `.md`; the `.html` links above are what "
        "GitHub Pages renders. The `exp0_stats.pt` cache behind these numbers is not in git "
        f"-- see [outputs/README.md]({to_outputs}README.md#the-big-caches-are-not-in-git).",
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

    # A run whose directory IS a layer (pcfg/layer_01) marks that pill, exactly as
    # a gemma layer page does. The refresher derives the same thing from the path;
    # they must not disagree.
    m = re.fullmatch(r"layer_(\d+)", pathlib.Path(run).name)
    L = [C.nav_html(depth=C.page_depth(run_dir / "README.md"),
                    layer=int(m.group(1)) if m else None,
                    page=None, current=f"outputs/{run}/"), "",
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


def render_source(source: str | None = None) -> str:
    """The landing page for a source directory: outputs/<source>/README.md.

    Same reason every layer has one -- a directory with no index is a 404 on
    GitHub Pages, and this one is where the nav's brand and the results index
    now send a reader looking for "the gemma results".
    """
    source = source or C.SOURCE_NAME
    cfg = C.SOURCES[source]
    src_dir = C.OUT_DIR / source
    path = src_dir / "README.md"
    depth = C.page_depth(path)
    to_outputs = "../" * (depth - 1)

    # What this source IS comes from a graded layer's own report, not from
    # config: only gemma's model and dictionary are constants in this repo.
    scope, model = "", C.MODEL_NAME
    for layer in cfg["layers"]:
        report = src_dir / f"layer_{layer:02d}" / "metrics_report.json"
        if report.is_file():
            r = json.loads(report.read_text())
            scope = C.scope_line(r.get("total_tokens"),
                                 n_docs=(r.get("config") or {}).get("n_docs"),
                                 config=r.get("config"))
            break

    L = [C.nav_html(depth=depth, current=f"outputs/{source}/"), "",
         f"# `{source}/` — {cfg['label']}", ""]
    if scope:
        L += [scope, ""]
    L.append(
        f"One directory per layer, graded by the same battery as every other source in "
        f"[outputs/]({to_outputs}README.md). The bar above moves between the layers; nothing "
        f"in `metrics/` differs between them."
    )
    L += ["", "| Layer | Pages |", "| --- | --- |"]
    for layer in cfg["layers"]:
        d = src_dir / f"layer_{layer:02d}"
        pages = ", ".join(f"[{label.lower()}](layer_{layer:02d}/{f})"
                          for f, label in cfg["pages"] if (d / f).exists()
                          or (d / f.replace(".html", ".md")).exists())
        L.append(f"| [**{layer}**](layer_{layer:02d}/) | {pages or '_not graded yet_'} |")
    if source == C.SOURCE_NAME:
        L += [
            "",
            "These sat at `outputs/layer_NN/` until 7 August. They moved when a second source was "
            "published beside them: with only gemma here, `layer_06` read as a global fact rather "
            f"than a fact about one model.",
            "",
            f"Stage 03 (`run_token_metrics.py`) has run on layer 6 only — see "
            f"[outputs/README.md]({to_outputs}README.md#the-second-pass-has-run-on-layer-6-only) "
            "before reading the sibling-redundancy figure on the other four.",
        ]
    L.append("")
    return "\n".join(L)


def write(layer: int) -> None:
    path = C.OUT_DIR / C.SOURCE_NAME / f"layer_{layer:02d}" / "README.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render(layer))
    print(f"[index] wrote {path}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--layer", type=int, help="one layer instead of all of NAV_LAYERS")
    ap.add_argument("--run", action="store_true",
                    help="write the index for EXP0_RUN's directory instead of a gemma layer")
    ap.add_argument("--source", nargs="?", const=C.SOURCE_NAME, metavar="NAME",
                    help="write outputs/<source>/README.md, the index over its layers "
                         f"(default {C.SOURCE_NAME}; any key of config.SOURCES)")
    args = ap.parse_args()
    if args.source:
        if args.source not in C.SOURCES:
            raise SystemExit(f"[index] unknown source {args.source!r}; "
                             f"known: {', '.join(C.SOURCES)}")
        path = C.OUT_DIR / args.source / "README.md"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(render_source(args.source))
        print(f"[index] wrote {path}")
        return
    if args.run:
        path = C.RUN_DIR / "README.md"
        path.write_text(render_run(C.RUN_NAME))
        print(f"[index] wrote {path}")
        return
    for layer in ([args.layer] if args.layer is not None else C.NAV_LAYERS):
        write(layer)


if __name__ == "__main__":
    main()
