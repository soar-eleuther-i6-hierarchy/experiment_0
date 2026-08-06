# outputs_archive/

Superseded results, kept so a withdrawn number stays citable.

`outputs/layer_NN/` is a fixed path — the published site links to it by name, so timestamping it
would 404 every page. That means a rerun overwrites the previous run in place. `config.ARCHIVE_DIR`
takes a dated copy first, into this directory, before the overwrite happens.

This directory is **tracked in git, not ignored**. That is the point: when a number changes, the
version it replaced has to survive in the repository rather than in one person's working copy.
`outputs_local/` was the old home for this and is gitignored, so an archive there vanished on a fresh
clone — precisely the one case where you need it.

Nothing here is linked from the site nav. These pages are evidence, not reading material.

## What is in here

### `layer_NN__v1__2026-08-06T17-45/`

The five gemma-2-2b layers as they read before BOS was excluded from the co-firing statistics.
Distinguishable at a glance by the token count: **48,971** here, **48,571** in `outputs/`. The
difference is exactly 400 — one BOS position per document, 400 documents.

### `kill_rates__v1__*.html`, `cross_depth_comparison__v1__*.html`

Two hand-built pages with no generator, withdrawn rather than updated. Both were built to display the
fractions that the BOS exclusion inverted. Each carries a banner at the top stating that its numbers
are withdrawn and why.

## What was touched, and what was not

The numbers, tables and figures in every archived file are exactly as they were published. Two kinds
of edit were made, both to the chrome around them:

- **Nav links.** Every archived page's nav pointed at `outputs/kill_rates.html` and
  `outputs/cross_depth_comparison.html`, which no longer exist there. Those links were repointed at
  the archived copies in this directory, so navigating inside the archive still works. Nothing was
  removed — a dead link loses more than it preserves.
- **Banners.** The two withdrawn pages got a banner at the top saying their numbers are withdrawn.

Two file kinds are deliberately *not* archived, via `config.ARCHIVE_SKIP`. `exp0_stats.pt`,
`token_cache/` and `figures/` are too large — the cache lives on the Hub. `feature_labels.json` and
`npedia_labels_cache.json` are not run output at all: they are Neuronpedia's labels for the layer's
SAE, identical in v1 and v2 because nothing we compute touches them. They were 9.8 MB of an 11 MB
archive, byte-identical to the tracked copies in `outputs/`. The archived dashboards inline their
labels at generation time, so they still read correctly without them.

## Why v1 was wrong

BOS is an attention sink: every feature fires there. With `PREPEND_BOS = True` and 400 documents,
every parent/child pair in the dictionary accumulated 400 joint firings — against a `MIN_JOINT = 30`
support guard. The guard exists to kill pairs whose co-firing is coincidence, and BOS handed every
pair enough co-firing to clear it. So the guard passed everything, and the metrics downstream graded
a candidate set that should never have existed.

The wider methodological point: five of the six metrics read the same co-firing matrix. They were
designed as independent detectors that would fail independently, and one contaminated token position
defeated them together. Agreement between them is weaker evidence than it looks.

## Reading an archived page

Do not compare a v1 page against a v2 page metric by metric and treat the delta as a finding. The two
were computed over different token sets. The delta is the contamination, not a result.
