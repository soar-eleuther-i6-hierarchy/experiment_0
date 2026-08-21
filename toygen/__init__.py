"""
Synthetic generator for toy SAE-training worlds.

Builds a declared containment forest and turns it into activations plus ground truth:
`spec -> tree -> geometry -> strengths -> sample`. Every pair carries a ground-truth label
(from the graph, never from statistics). The world is fully determined by its `ToyConfig` +
seed, so training and scoring regenerate it deterministically via `world.regenerate_world`
rather than persisting the (huge) coefficient matrices.

A "toy" is one `ToyConfig`; the generator itself is written once. See `spec.CONFIGS`
for the shipped configs.
"""
