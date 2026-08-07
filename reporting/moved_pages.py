"""
Keep the pre-move URLs alive: outputs/layer_NN/* -> outputs/<source>/layer_NN/*.

Results are grouped by source now. That is the right shape once there is more than
one source, but it moved 25 pages plus 5 directory URLs that were published, cited
in meeting notes and opened from other people's bookmarks — and a 404 is the worst
possible answer to a link someone was given in a meeting.

Each stub is a `<meta http-equiv="refresh">` page that also says where it went in
text, for a reader whose browser does not follow it. They are generated, not
hand-written: a hand-built page is exactly what this repo has already had to
withdraw once, and these have to be deletable by rerunning nothing.

    python3 -m reporting.moved_pages           # write the stubs
    python3 -m reporting.moved_pages --list    # what it would write
    python3 -m reporting.moved_pages --remove  # take them down again

They are not permanent. When the old links have stopped being followed — a year is
generous for a research site — delete `outputs/layer_NN/` with --remove.
"""

from __future__ import annotations

import argparse
import shutil

import config as C

# Everything a layer directory used to serve, plus the directory itself (README.md,
# which is what Jekyll renders for outputs/layer_NN/).
MOVED_FILES = [f for f, _ in C.NAV_PAGES] + ["README.md"]

STUB = """<!DOCTYPE html>
<meta charset="utf-8">
<meta http-equiv="refresh" content="0; url={target}">
<title>Moved — {old}</title>
<link rel="canonical" href="{target}">
<p style="font:15px/1.5 system-ui,sans-serif;padding:2rem">
This page moved to <a href="{target}">{new}</a>.
Results are grouped by source now: gemma's layers live under
<code>outputs/{source}/</code>, beside the other sources.
</p>
"""

# No `redirect_to:` front matter: that needs the jekyll-redirect-from plugin
# enabled in _config.yml, and inert config that looks functional is worse than
# none. Jekyll passes raw HTML through, so the same meta refresh works here, and
# the text below is the fallback for anything that ignores it.
MD_STUB = """<meta http-equiv="refresh" content="0; url={target}">

# Moved

`outputs/layer_{layer:02d}/` is now [`outputs/{source}/layer_{layer:02d}/`]({target}).

Results are grouped by source: gemma's five layers sit under `outputs/{source}/`,
beside the PCFG run and anything published later.
"""


def stub_paths(layer: int):
    """(old path, relative target) for every URL this layer used to serve."""
    old_dir = C.OUT_DIR / f"layer_{layer:02d}"
    for name in MOVED_FILES:
        # ../<source>/layer_NN/<name> -- relative, so it works on the Pages site,
        # on a fork's site, and from a local checkout alike.
        target = f"../{C.SOURCE_NAME}/layer_{layer:02d}/" + ("" if name == "README.md" else name)
        yield old_dir / name, target


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--list", action="store_true", help="show what would be written")
    ap.add_argument("--remove", action="store_true", help="delete the stub directories")
    args = ap.parse_args()

    if args.remove:
        for layer in C.NAV_LAYERS:
            d = C.OUT_DIR / f"layer_{layer:02d}"
            if d.exists():
                shutil.rmtree(d)
                print(f"[moved] removed {d}")
        return 0

    n = 0
    for layer in C.NAV_LAYERS:
        for path, target in stub_paths(layer):
            n += 1
            if args.list:
                print(f"[moved] {path.relative_to(C.OUT_DIR)} -> {target}")
                continue
            path.parent.mkdir(parents=True, exist_ok=True)
            if path.suffix == ".md":
                path.write_text(MD_STUB.format(target=target, layer=layer,
                                               source=C.SOURCE_NAME))
            else:
                path.write_text(STUB.format(target=target, old=path.name,
                                            new=target, source=C.SOURCE_NAME))
    print(f"[moved] {'would write' if args.list else 'wrote'} {n} redirect(s) "
          f"for {len(C.NAV_LAYERS)} layers")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
