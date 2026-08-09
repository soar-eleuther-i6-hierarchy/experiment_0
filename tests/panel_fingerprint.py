"""What each dashboard's panels are, read out of the source.

The captions in `reporting/visualize.py` are the one part of the generated site
that cannot keep itself honest. Every NUMBER in them is read from JSON at
generation time, so a number cannot outlive its data. The SENTENCES cannot do
that: change what a panel plots -- swap an axis, add a sixth filter, reorder the
grid -- and the numbers update themselves while the prose goes on describing the
old panel, with nothing in the code noticing.

That is the same failure this repo has already had twice in prose that no test
could reach: figure titles that quoted withdrawn layer-6 results, and "14/14
across seeds 0-7" surviving in seven files after the seed sweep it described
stopped existing. Both were caught by a person reading, months late.

So this module extracts the panel structure statically -- no data, no cache, no
plotting -- and `test_captions_match_panels.py` pins it. The pin does not check
that a caption is *correct*; nothing can. It guarantees that a change to the
panels cannot land without a human being sent back to the caption that describes
them.

Static, via `ast`, for two reasons: building the figures would need
`exp0_stats.pt`, which is not in git, so the guard would be unrunnable in exactly
the clone where review happens; and what we want to detect is an edit to the
source, which is the thing the source segment records.
"""

from __future__ import annotations

import ast
from pathlib import Path

VISUALIZE = Path(__file__).resolve().parents[1] / "reporting" / "visualize.py"


def _norm(text: str) -> str:
    """Collapse whitespace so reflowing a line is not reported as a change.

    Line breaks inside a long title move whenever the surrounding code is
    reindented, and a guard that fires on reindentation is a guard people learn
    to re-pin without reading.
    """
    return " ".join(text.split())


def panel_sources(path: Path = VISUALIZE) -> dict[str, str]:
    """{builder function name: normalised source of its panel titles}.

    `subplot_titles=` is usually a literal tuple, in which case its source *is*
    the titles. Two builders pass a name or a comprehension instead, and for a
    bare name the argument text would never change however much the titles did --
    so the assignments that build that name are folded in as well.
    """
    src = path.read_text()
    tree = ast.parse(src)
    out: dict[str, str] = {}
    for fn in (n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)):
        for call in ast.walk(fn):
            if not (isinstance(call, ast.Call)
                    and getattr(call.func, "id", "") == "make_subplots"):
                continue
            kw = next((k for k in call.keywords if k.arg == "subplot_titles"), None)
            if kw is None:
                continue
            parts = [ast.get_source_segment(src, kw.value) or ""]
            if isinstance(kw.value, ast.Name):
                target = kw.value.id
                for node in ast.walk(fn):
                    if isinstance(node, ast.Assign) and any(
                            isinstance(t, ast.Name) and t.id == target for t in node.targets):
                        parts.append(ast.get_source_segment(src, node) or "")
                    elif (isinstance(node, ast.AugAssign)
                          and isinstance(node.target, ast.Name)
                          and node.target.id == target):
                        parts.append(ast.get_source_segment(src, node) or "")
            out[fn.name] = _norm(" ".join(parts))
    return out
