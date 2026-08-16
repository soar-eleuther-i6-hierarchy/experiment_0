"""
Synthetic generator for toy SAE-training worlds.

Builds a declared containment forest and turns it into activations plus ground truth:
`spec -> tree -> geometry -> strengths -> sample -> emit(truth)`. Every pair carries a
ground-truth label (from the graph, never from statistics), and the directions,
coefficients and firing rates are all written out for later SAE training and analysis.

A "toy" is one `ToyConfig`; the generator itself is written once. See `spec.CONFIGS`
for the shipped configs.
"""
