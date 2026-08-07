"""Every link and asset on the generated site resolves to a file that exists.

This is the guard the blank-pages entry in `research-log/ERROR_LOG.md` is left
open on. Fifteen published pages rendered their nav bar above an empty page for
two days because one `<script src>` pointed a directory short after the results
were regrouped under `outputs/<source>/`. Nothing failed: the bundle 404s, plotly
never loads, and the nav is plain HTML so the page still looks maintained.

A page whose asset link is dead is not detectably different from a working one by
reading either the page or the generator. It is only detectable by resolving the
link, which is what this does.

Two conventions it has to know about, because the site relies on both:

**Jekyll serves `foo.md` at `foo.html`.** A nav entry points at
`metrics_report.html`, and only `metrics_report.md` is in the repo. That link is
correct; a checker that did not know this would report every report page broken.

**A directory link resolves to its `README.md`.** `outputs/pcfg/` is how the nav
addresses a source, and GitHub Pages serves that directory's README.

    python3 -m tests.test_site_links            # fails on the first dead target
    python3 -m tests.test_site_links --list     # print every link it resolved
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# href="..." / src="..." in HTML, and [text](target) in markdown.
HTML_REF = re.compile(r'(?:href|src)="([^"]+)"')
MD_REF = re.compile(r"\[[^\]]*\]\(([^)\s]+)(?:\s+\"[^\"]*\")?\)")

SKIP_SCHEME = ("http://", "https://", "mailto:", "data:", "//", "#", "javascript:")


def targets(path: Path) -> list[str]:
    text = path.read_text(errors="ignore")
    refs = HTML_REF.findall(text)
    if path.suffix == ".md":
        refs += MD_REF.findall(text)
    out = []
    for r in refs:
        r = r.split("#", 1)[0].strip()          # drop the anchor; we check the file
        if not r or r.startswith(SKIP_SCHEME):
            continue
        out.append(r)
    return out


def resolves(base: Path, ref: str) -> bool:
    """Does `ref`, written on the page at `base`, name a file that exists?"""
    p = (base.parent / ref).resolve()
    if p.exists():
        return True
    # Jekyll serves markdown under .html, and a directory under its README.
    if p.suffix == ".html" and p.with_suffix(".md").exists():
        return True
    if ref.endswith("/") or not p.suffix:
        return (p / "README.md").exists() or (p / "index.html").exists()
    return False


def walk() -> list[Path]:
    pages = []
    for base in (ROOT / "outputs", ROOT / "outputs_archive"):
        if base.exists():
            pages += [p for p in base.rglob("*") if p.suffix in (".html", ".md")]
    pages += [p for p in ROOT.glob("*.md")]
    return sorted(pages)


def check() -> list[tuple[Path, str]]:
    return [(p, r) for p in walk() for r in targets(p) if not resolves(p, r)]


def test_every_site_link_resolves():
    broken = check()
    assert not broken, (
        f"{len(broken)} dead link(s)/asset(s) on the generated site:\n"
        + "\n".join(f"  {p.relative_to(ROOT)}  ->  {r}" for p, r in broken[:40])
        + "\nA page whose asset 404s still renders its nav bar, so this is not "
          "visible by opening it."
    )


def main() -> int:
    pages = walk()
    n_refs = sum(len(targets(p)) for p in pages)
    if "--list" in sys.argv:
        for p in pages:
            for r in targets(p):
                mark = "ok  " if resolves(p, r) else "DEAD"
                print(f"  {mark} {p.relative_to(ROOT)}  ->  {r}")
    broken = check()
    if broken:
        for p, r in broken:
            print(f"[links] DEAD  {p.relative_to(ROOT)}  ->  {r}")
        print(f"[links] {len(broken)} dead of {n_refs} references across {len(pages)} pages")
        return 1
    print(f"[links] all {n_refs} references across {len(pages)} pages resolve")
    return 0


if __name__ == "__main__":
    sys.exit(main())
