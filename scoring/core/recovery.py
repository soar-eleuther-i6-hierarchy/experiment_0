"""
Recovery scoring: did the SAE learn the toy's true features at all?

This is the pure numeric core of the recovery gate. Given the true per-token
coefficients and the learned latent activations, it matches learned latents to true
features, decides which features are recovered, and characterises the failure modes
(splitting, absorption/dispersion, dead latents, reconstruction loss).

Everything downstream is conditioned on what this module reports, because a metric
computed on a feature the SAE never learned is meaningless.

Shapes (n tokens, F true features, S learned latents, D activation dim):
  true_coeff  [n, F]   a column is one true feature's coefficient over tokens
  learned     [n, S]   a column is one latent's activation over tokens
  g / W_dec   [F, D] / [S, D]   concept directions / decoder rows (one per latent)
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from toygen import labels


_TIEBREAK_EPS = 1e-6
_TINY = 1e-12


def activation_corr(true_coeff: torch.Tensor, learned_acts: torch.Tensor) -> torch.Tensor:
    """Pearson correlation of every true feature against every learned latent.

    Returns `[F, S]` where entry `[i, j] = corr(true_coeff[:, i], learned_acts[:, j])`
    over the n rows, clipped to `[-1, 1]`.

    A zero-variance column on either side (a dead/constant latent, or a true feature
    that never fired in this draw) has undefined Pearson correlation. We set it to 0.0
    rather than NaN, so its `1 - corr` matching cost is 1 and it is never matched by
    accident.
    """
    tc = true_coeff - true_coeff.mean(dim=0, keepdim=True) # [n, F]
    lc = learned_acts - learned_acts.mean(dim=0, keepdim=True) # [n, S]
    std_t = tc.pow(2).sum(dim=0).sqrt()          # [F]
    std_l = lc.pow(2).sum(dim=0).sqrt()          # [S]
    cov = tc.transpose(0, 1) @ lc                 # [F, S] sum of centred products
    denom = std_t[:, None] * std_l[None, :]       # [F, S]
    corr = torch.zeros_like(cov)
    nz = denom > 0                                # zero variance on either side -> stays 0.0
    corr[nz] = cov[nz] / denom[nz]
    return corr.clamp(-1.0, 1.0)


@dataclass(frozen=True)
class MatchResult:
    """One-to-one assignment of learned latents to true features.

    match         [F] long   assigned latent index per true feature, -1 if unassigned
    matched_corr  [F] float  correlation at the matched latent, -inf where unassigned
    recovered     [F] bool   matched_corr >= rho
    """

    match: torch.Tensor
    matched_corr: torch.Tensor
    recovered: torch.Tensor


def match_features(corr: torch.Tensor, g: torch.Tensor, W_dec: torch.Tensor,
                   rho: float = 0.5) -> MatchResult:
    """One-to-one Hungarian match of true features to learned latents on `1 - corr`.

    Minimises total `1 - corr` (maximises total activation correlation). Ties break
    toward decoder alignment `cos(W_dec[j], g[i])`, a tiny cost perturbation so
    correlation still dominates. When `S < F`, surplus features are left unassigned
    (`match == -1`, `recovered == False`); a feature is recovered when its matched
    correlation clears `rho`.
    """
    from scipy.optimize import linear_sum_assignment

    F, S = corr.shape
    gn = g / g.norm(dim=1, keepdim=True).clamp_min(_TINY) # [F, D]
    wn = W_dec / W_dec.norm(dim=1, keepdim=True).clamp_min(_TINY) # [S, D]
    geom = gn @ wn.transpose(0, 1)                # [F, S] cos(g_i, W_dec[j])
    cost = (1.0 - corr) - _TIEBREAK_EPS * geom    # [F, S] lower = better; higher cos wins ties
    rows, cols = linear_sum_assignment(cost.detach().cpu().numpy())

    match = torch.full((F,), -1, dtype=torch.long, device=corr.device) # [F]
    matched_corr = torch.full((F,), float("-inf"), dtype=corr.dtype, device=corr.device) # [F]
    for r, c in zip(rows.tolist(), cols.tolist()):
        match[r] = c
        matched_corr[r] = corr[r, c]
    recovered = matched_corr >= rho
    return MatchResult(match=match, matched_corr=matched_corr, recovered=recovered)


def recovery_rate_curve(corr: torch.Tensor, rhos: tuple[float, ...] = (0.3, 0.5, 0.7)) -> dict[float, float]:
    """Recovery-rate curve: fraction of true features whose best latent clears each rho.

    Uses `max_j corr[i, j]` directly instead of the Hungarian assignment, so it is a
    cheap upper-bound estimate of the recovery rate (how many features *could* be
    recovered at each threshold). Non-increasing in rho by construction.
    """
    best = corr.max(dim=1).values  # [F]
    return {rho: float((best >= rho).double().mean()) for rho in rhos}


def match_one_to_many(corr: torch.Tensor, rho: float = 0.5) -> dict[str, int | dict[int, list[int]]]:
    """Splitting sensitivity: for each true feature, every latent it claims above rho.

    Returns `{"per_feature": {i: [latent, ...]}, "split_count": int}`. Each feature's
    list holds ALL latents with `corr >= rho`, best first, uncapped. `split_count` is
    the number of features covered by more than one such latent — the main
    feature-splitting signal.
    """
    per_feature: dict[int, list[int]] = {}
    split_count = 0
    for i in range(corr.shape[0]):
        row = corr[i]  # [S]
        qualifying = (row >= rho).nonzero(as_tuple=True)[0] # latents with corr >= rho
        if qualifying.numel() == 0:
            per_feature[i] = []
            continue
        # stable sort so exact-corr ties resolve deterministically (ascending latent index)
        order = torch.argsort(row[qualifying], descending=True, stable=True)
        latents = qualifying[order].tolist()
        per_feature[i] = latents
        if len(latents) > 1:
            split_count += 1
    return {"per_feature": per_feature, "split_count": split_count}


def per_class_recovery(recovered: torch.Tensor, pair_labels: torch.Tensor, label_names: list[str]) -> dict[str, tuple[int, int]]:
    """Per-label recovery counts: (# ordered pairs with BOTH endpoints recovered, # total).

    Downstream AUROC is conditioned on recovered pairs. Some classes (e.g. firing-only)
    recover less often than is-a, so a column can look good simply because its hard cases
    were dropped as unrecovered. Reporting the attrition per class keeps the two effects
    from being confused. A pair counts as recovered only if both endpoints are.
    """
    F = pair_labels.shape[0]
    eye = torch.eye(F, dtype=torch.bool, device=pair_labels.device) # mask self-edges (diagonal)
    both_recovered = recovered[:, None] & recovered[None, :]   # [F, F]
    out: dict[str, tuple[int, int]] = {}
    for name in label_names:
        in_class = (pair_labels == labels._index(name)) & ~eye
        total = int(in_class.sum())
        both = int((in_class & both_recovered).sum())
        out[name] = (both, total)
    return out


def child_direction_dispersion(W_dec: torch.Tensor, g: torch.Tensor, match: torch.Tensor, children: list[int], m: int = 5) -> torch.Tensor:
    """Child-direction dispersion r_disp: how cleanly each child got its own latent.

    For child `c`, over the `m` latents nearest to `g_c` by `|cos|`, the fraction of its
    true-direction energy landing on latents other than its own match:

        r_disp(c) = sum_{j in top-m, j != match[c]} max(cos(W_dec[j], g_c), 0)^2
                    / sum_{j in top-m}              max(cos(W_dec[j], g_c), 0)^2

    Captures dispersion = absorption + splitting together (not parent-specific). The
    top-m restriction avoids full-dictionary extreme-value inflation; a zero
    denominator yields 0.0, never NaN.
    """
    wn = W_dec / W_dec.norm(dim=1, keepdim=True).clamp_min(_TINY)   # [S, D]
    gn = g / g.norm(dim=1, keepdim=True).clamp_min(_TINY)           # [F, D]
    out = torch.zeros(len(children), dtype=W_dec.dtype, device=W_dec.device)
    for pos, c in enumerate(children):
        cos = wn @ gn[c]  # [S, D] @ [D] --> [S]
        # Stable sort (not topk) so |cos| ties resolve deterministically; rank by |cos|, negatives clamped to 0 below.
        top = torch.sort(cos.abs(), descending=True, stable=True).indices[:min(m, cos.numel())]
        pos_energy = cos[top].clamp_min(0.0).pow(2) # max(cos, 0)^2 over the top-m
        den = float(pos_energy.sum())
        if den <= 0.0:
            continue                                               # stays 0.0
        mc = int(match[c])
        own = torch.zeros_like(pos_energy)
        if mc >= 0:
            own[top == mc] = pos_energy[top == mc]
        out[pos] = (pos_energy.sum() - own.sum()) / den
    return out


def reconstruction_fvu(h: torch.Tensor, acts: torch.Tensor, W_dec: torch.Tensor, b_dec: torch.Tensor | None = None) -> float:
    """Fraction of variance unexplained by the SAE reconstruction `acts @ W_dec (+ b_dec)`.

    Computes `||h - h_hat||^2 / ||h - mean(h)||^2`: 0 is perfect, 1 matches the
    constant-mean baseline, and > 1 is worse than just predicting the mean. On a clean
    oracle it bottoms out at the injected noise fraction.
    """
    h_hat = acts @ W_dec # [n, S] @ [S, D] -> [n, D]
    if b_dec is not None:
        h_hat = h_hat + b_dec
    resid = (h - h_hat).pow(2).sum()
    total = (h - h.mean(dim=0, keepdim=True)).pow(2).sum()
    return float(resid / total.clamp_min(_TINY))


def count_dead(acts: torch.Tensor, thresh: float = 0.0) -> int:
    """Number of latents that never exceed `thresh` on any token (never fire)."""
    return int((acts.max(dim=0).values <= thresh).sum())


def realized_l0(acts: torch.Tensor) -> float:
    """Realized L0: mean per-token count of firing latents, on the ACTUAL activations.

    `acts` [n, S] must be the SAE's real activations, not a randn probe (which
    overstates L0 by ~30x since a uniform threshold fires more on higher-norm input).
    This is the honest sparsity to read the co-firing detectors and sweep x-axis
    against — the nominal top-k is not the realized L0 once the checkpoint is saved.
    """
    return float((acts > 0).double().sum(dim=1).mean())
