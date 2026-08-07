"""
Rewrite the nav bar in every generated page, without regenerating the pages.

The bar is baked into each artifact rather than served from one place -- there is
no layout to put it in, because these are plotly HTML files and Jekyll-rendered
markdown. That is fine until the bar itself changes: adding one entry to
`config.NAV_GLOBAL` makes every page that does not have it wrong, and ten of them
are dashboards whose generator needs the ~700 MB stats cache for that layer. The
choice was between a site with two different nav bars on it and a 3.5 GB download
to change a link.

This is the third option. The nav block is a self-delimiting region -- NAV_CSS
opens with `<style>` and the bar closes with `</nav>` -- and every input to
`config.nav_html` is derivable from the file's own path, exactly as it is at
generation time. So the bar can be re-rendered in place, and the rest of the file
is not touched.

It is not a substitute for regenerating a page: it changes navigation and nothing
else. If the NUMBERS on a page are stale, this will not tell you, and a page whose
bar is fresh looks maintained. Rerun the stage.

    python3 -m reporting.refresh_nav            # every page under OUT_DIR
    python3 -m reporting.refresh_nav --check    # report what would change, write nothing

Includes outputs_archive/. Those pages are withdrawn, and the banner on each says
so -- but a withdrawn page whose every nav link 404s is not "clearly historical",
it is broken. Their pills are never marked current, so the bar navigates without
claiming the archived page is the live one.
"""

from __future__ import annotations

import argparse
import re

import config as C

# NAV_CSS ... </nav>, the whole injected region. Non-greedy so a page holding two
# (none do, but a future one might) is handled one at a time.
NAV_RE = re.compile(r"<style>\s*\.x0nav\{.*?</nav>", re.DOTALL)


def identity(path):
    """(depth, layer, page, current) for one file -- what its generator passed.

    Mirrors reporting.visualize's `_page_identity` / `_site_path` and
    layer_index's calls. Derived from the path rather than stored, which is why
    this can run over a page nobody can rebuild.

    Every file under the repo is fair game, not only `outputs/`: the package
    READMEs carry the same bar, and so does every archived page -- and those were
    the ones left with 295 dead links when the layer directories moved.
    """
    depth = C.page_depth(path)
    try:
        rel = path.resolve().relative_to(C.OUT_DIR.resolve())
        under_outputs = True
    except ValueError:                      # a page outside outputs/, e.g. a package README
        rel = path.resolve().relative_to(C.HERE.resolve())
        under_outputs = False
    if depth == 0:
        # The site index itself: the brand link's target rather than any nav
        # entry, so nothing is marked current.
        return 0, None, None, None

    parent = rel.parts[-2] if len(rel.parts) > 1 else ""
    is_index = path.name == "README.md"

    # `page` is which of the five per-layer page kinds this is -- it drives where
    # the layer pills point, so it must name a file that exists in every layer.
    # The calibration pages are not one of the five: marking them would send
    # every pill to a 404. Reports are markdown Jekyll serves as .html, hence the
    # rename.
    page = path.name.replace(".md", ".html")
    page = None if is_index or page not in {f for f, _ in C.NAV_PAGES} else page

    # Exactly `layer_NN`, so an ARCHIVED copy -- layer_24__v1__2026-08-06T17-45 --
    # is not marked as the current layer 24. It is a withdrawn page; its banner
    # says so, and lighting the pill would say the opposite.
    layer = int(parent.split("_")[1]) if re.fullmatch(r"layer_\d+", parent) else None

    if not under_outputs:
        return depth, layer, page, None
    # Site-wide: a run directory, or a page directly in outputs/. layer=None, so
    # the Page row is not drawn; the pills still carry `page` and stay on this
    # kind of page when you jump to a gemma layer.
    # `.md` -> `.html` here too: Jekyll serves the markdown under the .html name,
    # which is the name the nav entries carry, so toy_calibration.md must
    # highlight the same entry its rendered form does.
    served = rel.with_suffix(".html").as_posix() if rel.suffix == ".md" else rel.as_posix()
    current = f"outputs/{rel.parent.as_posix()}/" if is_index and len(rel.parts) > 1 else \
              "outputs/" if is_index else \
              f"outputs/{served}"
    return depth, layer, page, current


def refresh(path, write=True):
    """Replace the file's nav block. Returns True if it changed."""
    text = path.read_text(encoding="utf-8")
    if not NAV_RE.search(text):
        return False
    depth, layer, page, current = identity(path)
    nav = C.nav_html(depth=depth, layer=layer, page=page, current=current)
    new = NAV_RE.sub(lambda _: nav, text, count=1)
    if new == text:
        return False
    if write:
        path.write_text(new, encoding="utf-8")
    return True


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true", help="report, write nothing")
    args = ap.parse_args()

    # Every page on the site, not only outputs/: the package READMEs carry the
    # bar too, and so does outputs_archive/, whose links all pointed at the old
    # layer paths.
    changed = []
    for path in sorted(C.HERE.rglob("*")):
        if path.suffix not in (".html", ".md") or not path.is_file():
            continue
        if any(part in (".git", "node_modules", "__pycache__") for part in path.parts):
            continue
        if refresh(path, write=not args.check):
            changed.append(path.relative_to(C.HERE))

    verb = "would update" if args.check else "updated"
    for p in changed:
        print(f"[nav] {verb} {p}")
    print(f"[nav] {verb} {len(changed)} page(s)")
    # --check is a report, not a gate: exit 1 so it can be one if anyone wants it.
    return 1 if (args.check and changed) else 0


if __name__ == "__main__":
    raise SystemExit(main())
