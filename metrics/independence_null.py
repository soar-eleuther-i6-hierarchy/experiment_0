"""
Independence-null deviation: PMI + Dev (metrics_todo.md T3; landscape
Appendix A-L / A-M).

Raw reverse coverage rewards frequent parents: a parent firing on ~all tokens
has R ~ 1 against every child by base rate alone. Score the surprise over the
independent-firing null instead:

    PMI(p, c) = log[ P(p and c) / (P(p) P(c)) ] = log[ n_joint * N / (n_p * n_c) ]
        ~0 for a base-rate-only "parent" even when R ~ 1.

    Dev(p, c) = R(p, c) - rho_p        (rho_p = parent firing rate)
        sign-equivalent to PMI; kept because it reads in coverage units.

Support guard (landscape Rev. 2): pairs with n_joint < min_joint are NaN (not
-inf, which would poison means/sorts) and are COUNTED, not silently dropped.

Scope note: this controls the C-freq confound only. Topical co-occurrence
(mg -> medical text) passes both PMI and Dev; that needs the model-based
null M' (out of scope for this tranche).
"""

from __future__ import annotations

import torch


def independence_scores(
    cofire: torch.Tensor,       # [P, C] co-firing counts
    fire_p: torch.Tensor,       # [P]
    fire_c: torch.Tensor,       # [C]
    total_tokens: int,
    min_joint: int = 10,
) -> dict:
    """Returns {"pmi" [P,C], "dev" [P,C], "valid" [P,C] bool, "n_excluded" int}."""
    cofire = cofire.double()
    fire_p = fire_p.double()
    fire_c = fire_c.double()
    N = float(max(total_tokens, 1))

    expected = fire_p.unsqueeze(1) * fire_c.unsqueeze(0) / N          # [P, C]
    pmi = torch.log(cofire.clamp(min=1e-300) / expected.clamp(min=1e-300))
    R = cofire / fire_c.clamp(min=1.0).unsqueeze(0)
    dev = R - (fire_p / N).unsqueeze(1)

    valid = cofire >= min_joint
    nan = torch.tensor(float("nan"), dtype=torch.float64)
    pmi = torch.where(valid, pmi, nan)
    dev = torch.where(valid, dev, nan)
    return {
        "pmi": pmi,
        "dev": dev,
        "valid": valid,
        "n_excluded": int((~valid).sum()),
    }
