"""
Synthetic generator for toy SAE-training worlds.

Builds a declared containment forest and turns it into activations plus ground truth:
`spec -> tree -> geometry -> strengths -> sample`. Every pair gets a ground-truth label from
the graph, never from statistics. A world is fully determined by its `ToyConfig` + seed, so
training and scoring can regenerate it deterministically instead of storing the (huge)
coefficient matrices.

A "toy" is one `ToyConfig`; the generator itself is written once. See `spec.CONFIGS` for the
shipped configs.
"""
