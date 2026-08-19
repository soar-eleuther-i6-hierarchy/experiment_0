"""SAE dictionary-evaluation harness for the toy generator.

Scores a trained toy SAE against the generator's ground truth: feature recovery,
relationship-retrieval AUROCs, the Stage-0 clean-ceiling survival map, and the
absorption / decoder-multiplicity / composition decomposition. Kept separate from the
world generator in `toygen` and the trainer in `training`.
"""
