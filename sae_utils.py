"""
Shared loaders + tiny helpers for Exp 0's stage scripts.

Mirrors the warm-up task's sae_utils so both stages describe the SAME model /
SAE / block structure. Keeps "how do I get the model / SAE / block masks" in one
spot so cache_stats.py and run_metrics.py stay readable.
"""

from __future__ import annotations

import torch

import config as C


def load_sae(device: str | None = None):
    """Load the released Matryoshka SAE (layer 6 residual stream)."""
    from sae_lens import SAE

    device = device or C.pick_device()
    out = SAE.from_pretrained(release=C.SAE_RELEASE, sae_id=C.SAE_ID, device=device)
    sae = out[0] if isinstance(out, tuple) else out
    sae = sae.to(device)
    sae.eval()
    return sae


def load_model(device: str | None = None):
    """Load gemma-2-2b the way the SAE expects (no processing + its kwargs)."""
    from sae_lens import HookedSAETransformer

    device = device or C.pick_device()
    model = HookedSAETransformer.from_pretrained_no_processing(
        C.MODEL_NAME, device=device, **C.MODEL_KWARGS
    )
    model.eval()
    return model


def block_slice(block: int) -> slice:
    start, end = C.BLOCK_RANGES[block]
    return slice(start, end)


def block_masks(device: str = "cpu") -> list[torch.Tensor]:
    """Boolean mask per block over the 32768 feature axis (length = N_BLOCKS)."""
    masks = []
    for start, end in C.BLOCK_RANGES:
        m = torch.zeros(C.D_SAE, dtype=torch.bool, device=device)
        m[start:end] = True
        masks.append(m)
    return masks


def human_block_table() -> str:
    lines = ["block | range | n_features"]
    for b, (s, e) in enumerate(C.BLOCK_RANGES):
        lines.append(f"B{b}    | [{s}:{e}) | {e - s}")
    return "\n".join(lines)
