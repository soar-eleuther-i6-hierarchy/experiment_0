"""The captions cannot be checked for truth, so the panels are pinned instead.

Read `tests/panel_fingerprint.py` first; it explains why this exists.

The short version: in `reporting/visualize.py` every number inside a caption is
read from JSON at generation time and so cannot go stale, but the sentences
around those numbers are hand-written English. Change what a panel plots and the
numbers follow the data while the prose keeps describing the panel that used to
be there. No amount of deriving fixes that, because the thing that went wrong is
a claim about meaning.

What this test does about it is narrow and worth stating plainly: **it does not
check that any caption is correct.** It cannot. It fails when the panel structure
moves, which sends whoever moved it back to the caption. That is the whole
mechanism -- a tripwire, not a proof.

The counterpart guard is `test_every_caption_builder_is_wired`, which catches the
other direction: a page that grew a caption builder nobody connected to it.
"""

from __future__ import annotations

import hashlib
import re
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from panel_fingerprint import VISUALIZE, panel_sources  # noqa: E402

# sha256[:16] of the normalised panel-title source of each dashboard builder,
# as it stood when its caption was last read against it.
#
# WHEN THIS FAILS: do not re-pin first. Open the builder, look at what changed,
# then open its `captions_*()` and check every sentence still describes the panel
# it names. Re-pin only after that -- re-pinning without reading is the same as
# deleting the test, and it is the easier of the two to do by accident.
EXPECTED = {
    "build_dashboard": "ced8575ee9d06670",
    "build_calibration_dashboard": "2555919128e62357",
    "build_trained_calibration_dashboard": "abdcf3ccede1b582",
    "build_qualitative_dashboard": "4af54320d4270f07",
    "build_in_block_dashboard": "87b810c72878b199",
}

# builder -> the caption function whose prose describes it
CAPTIONED_BY = {
    "build_dashboard": "captions_dashboard",
    "build_calibration_dashboard": "captions_calibration",
    "build_trained_calibration_dashboard": "captions_trained_calibration",
    "build_qualitative_dashboard": "captions_qualitative",
    "build_in_block_dashboard": "captions_in_block",
    # the Sankey page has no subplot_titles to pin, but it does have prose
    "build_all_superparent_sankeys": "captions_sankey",
}


def _digest(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def test_no_dashboard_lost_its_pin():
    """A new dashboard must be pinned, and a removed one un-pinned, deliberately."""
    found = set(panel_sources())
    assert found == set(EXPECTED), (
        f"dashboards with panel titles: {sorted(found)}\n"
        f"pinned in EXPECTED          : {sorted(EXPECTED)}\n"
        "A new dashboard needs a caption builder and an entry here; a removed one "
        "needs both taken out."
    )


@pytest.mark.parametrize("builder", sorted(EXPECTED))
def test_panels_unchanged_since_captions_were_written(builder):
    got = _digest(panel_sources()[builder])
    assert got == EXPECTED[builder], (
        f"\nThe panels of {builder}() have changed.\n\n"
        f"Its captions are written in {CAPTIONED_BY[builder]}() and describe what each "
        "panel shows. They contain hand-written sentences, which no generator can "
        "keep true.\n\n"
        "Re-read that function against the new panels, fix any sentence that now "
        f"describes the old figure, then set\n    \"{builder}\": \"{got}\",\nin "
        f"{Path(__file__).name}.\n"
    )


@pytest.mark.parametrize("builder,caption_fn", sorted(CAPTIONED_BY.items()))
def test_every_caption_builder_is_wired(builder, caption_fn):
    """Both functions exist, and the page that draws one passes the other.

    Guards the failure the pin cannot see: a dashboard whose captions were never
    connected still renders, just with no captions, and looks finished.
    """
    src = VISUALIZE.read_text()
    for name in (builder, caption_fn):
        assert re.search(rf"^def {name}\(", src, re.M), f"{name}() is gone from visualize.py"
    assert re.search(rf"captions={caption_fn}\(|captions=\s*\n?\s*{caption_fn}\(", src), (
        f"{caption_fn}() is defined but never passed to write_page(), so the page "
        f"{builder}() draws would publish with no captions."
    )
