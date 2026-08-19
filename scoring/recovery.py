"""
Tier-A recovery scoring — did the SAE learn the toy's true features at all?

Pure numeric core of the recovery gate: given the
true per-token coefficients and the learned latent activations, match learned
latents to true features, decide which are recovered, and characterise the
failure modes (splitting, absorption/dispersion, dead latents, reconstruction
loss). A metric result on a feature the SAE never learned is meaningless, so
everything downstream is conditioned on what this module reports.

Deliberately dependency-light — torch always, scipy only lazily inside `match_features`
(the Hungarian assignment); everything else here is torch-only, no `sae_lens`, no `toygen`,
no CUDA — so it is tested against checkpoint-free synthetic oracles and reused on
either side of the 5.x/6.x boundary. The orchestration that feeds it a real
checkpoint and a regenerated world lives in `scoring.harness`.

Shapes (n tokens, F true features, S learned latents, D activation dim):
  true_coeff  [n, F]   a column is one true feature's coefficient over tokens
  learned     [n, S]   a column is one latent's activation over tokens
  g / W_dec   [F, D] / [S, D]   concept directions / decoder rows (one per latent)
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from toygen import labels

# The geometric tie-break must never overtake the activation-correlation signal:
# real corr gaps are O(0.1+), so a 1e-6 nudge only decides exact ties.
_TIEBREAK_EPS = 1e-6
_TINY = 1e-12


def activation_corr(true_coeff: torch.Tensor, learned_acts: torch.Tensor) -> torch.Tensor:
    """Pearson correlation of every true feature against every learned latent.

    Returns `[F, S]` with entry `[i, j] = corr(true_coeff[:, i], learned_acts[:, j])`
    over the n rows. Correlation, not decoder cosine, so the geometry channel stays
    clean for S_res later.

    Degeneracy guard: the result is clipped to `[-1, 1]`, and any
    zero-variance column on EITHER side — a dead/constant latent or a true feature
    that never fired in this draw — has undefined Pearson correlation and is set to
    `0.0` (never NaN), so its `1 - corr` cost is `1` and it is never spuriously matched.
    """
    tc = true_coeff - true_coeff.mean(dim=0, keepdim=True)
    lc = learned_acts - learned_acts.mean(dim=0, keepdim=True)
    std_t = tc.pow(2).sum(dim=0).sqrt()          # [F]
    std_l = lc.pow(2).sum(dim=0).sqrt()          # [S]
    cov = tc.transpose(0, 1) @ lc                 # [F, S] sum of centred products
    denom = std_t[:, None] * std_l[None, :]       # [F, S]
    corr = torch.zeros_like(cov)
    nz = denom > 0                                # zero-variance either side -> stays 0.0
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
    """Hungarian match of true features to learned latents on `1 - corr`.

    Minimises total `1 - corr` (so it maximises total activation correlation),
    one-to-one. Ties in correlation are broken toward the latent whose decoder row
    aligns better with the true concept direction — `cos(W_dec[j], g[i])` — applied
    as a tiny cost perturbation so correlation still strictly dominates.

    With `S < F` the surplus `F - S` true features are left unassigned
    (`match == -1`, `recovered == False`). A feature is recovered iff its matched
    correlation clears `rho` (report the curve over rho elsewhere; 0.5 is primary).
    """
    # scipy is imported LAZILY (only Hungarian matching needs it) so that importing this module —
    # e.g. for `activation_corr` / `child_direction_dispersion`, which many callers use WITHOUT
    # matching — never requires scipy. A hard top-level import blocked collection of every test
    # module in the import chain (scoring.recovery/harness/stage0/absorption) in any scipy-less env.
    from scipy.optimize import linear_sum_assignment

    F, S = corr.shape
    gn = g / g.norm(dim=1, keepdim=True).clamp_min(_TINY)
    wn = W_dec / W_dec.norm(dim=1, keepdim=True).clamp_min(_TINY)
    geom = gn @ wn.transpose(0, 1)                # [F, S] cos(g_i, W_dec[j])
    cost = (1.0 - corr) - _TIEBREAK_EPS * geom    # lower = better; higher cos preferred
    rows, cols = linear_sum_assignment(cost.detach().cpu().numpy())

    match = torch.full((F,), -1, dtype=torch.long, device=corr.device)
    matched_corr = torch.full((F,), float("-inf"), dtype=corr.dtype, device=corr.device)
    for r, c in zip(rows.tolist(), cols.tolist()):
        match[r] = c
        matched_corr[r] = corr[r, c]
    recovered = matched_corr >= rho
    return MatchResult(match=match, matched_corr=matched_corr, recovered=recovered)


def recovery_rate_curve(corr: torch.Tensor,
                        rhos: tuple[float, ...] = (0.3, 0.5, 0.7)) -> dict[float, float]:
    """Functional-recovery CDF points: fraction of true features whose best latent clears rho.

    Uses `max_j corr[i, j]` directly (independent of the Hungarian assignment), so it
    reports how many features *could* be recovered at each threshold. Monotone
    non-increasing in rho by construction.
    """
    best = corr.max(dim=1).values                 # [F]
    return {rho: float((best >= rho).double().mean()) for rho in rhos}


def match_one_to_many(corr: torch.Tensor, cap: int = 3,
                      rho: float = 0.5) -> dict[str, int | dict[int, list[int]]]:
    """Splitting sensitivity: the up-to-`cap` best latents each true feature claims above rho.

    Returns `{"per_feature": {i: [latent, ...]}, "split_count": int}` where a feature's
    list holds its top-`cap` latents with `corr >= rho` (best first), and `split_count`
    is the number of features covered by more than one such latent — the load-bearing
    feature-splitting signal.
    """
    per_feature: dict[int, list[int]] = {}
    split_count = 0
    for i in range(corr.shape[0]):
        row = corr[i]
        qualifying = (row >= rho).nonzero(as_tuple=True)[0]
        if qualifying.numel() == 0:
            per_feature[i] = []
            continue
        # stable so exact-corr ties resolve to a deterministic (ascending-index) order
        order = torch.argsort(row[qualifying], descending=True, stable=True)
        latents = qualifying[order][:cap].tolist()
        per_feature[i] = latents
        if len(latents) > 1:
            split_count += 1
    return {"per_feature": per_feature, "split_count": split_count}


def per_class_recovery(recovered: torch.Tensor, pair_labels: torch.Tensor,
                       label_names: list[str]) -> dict[str, tuple[int, int]]:
    """Per-label recovery: (# ordered pairs with BOTH endpoints recovered, # total) by class.

    A Tier-B AUROC is conditioned on recovered pairs, and containment-only recovers
    less often than is-a, so a column can look good simply because its hard cases were
    unrecovered. This reports the attrition per class so the two can't be confused.
    The diagonal is skipped; a pair counts as recovered only if both its
    endpoints are.
    """
    F = pair_labels.shape[0]
    eye = torch.eye(F, dtype=torch.bool, device=pair_labels.device)
    both_recovered = recovered[:, None] & recovered[None, :]   # [F, F]
    out: dict[str, tuple[int, int]] = {}
    for name in label_names:
        # Single-label equality: each ordered pair carries exactly one class INDEX. The
        # diagonal is masked because is_a is index 0 AND the diagonal is 0, so an unmasked
        # is_a count would swallow every self-pair.
        in_class = (pair_labels == labels._index(name)) & ~eye
        total = int(in_class.sum())
        both = int((in_class & both_recovered).sum())
        out[name] = (both, total)
    return out


def child_direction_dispersion(W_dec: torch.Tensor, g: torch.Tensor,
                               match: torch.Tensor, children: list[int],
                               m: int = 5) -> torch.Tensor:
    """Child-direction dispersion r_disp — how cleanly each child got its own latent.

    For child `c`: over the `m` latents nearest to `g_c` by `|cos|`, the fraction of
    the child's true-direction energy that lands on latents OTHER than its own match:

        r_disp(c) = sum_{j in top-m, j != match[c]} max(cos(W_dec[j], g_c), 0)^2
                    / sum_{j in top-m}             max(cos(W_dec[j], g_c), 0)^2

    This is dispersion = absorption + splitting: it does not require the
    leaked energy to sit on the parent latent, so read it as cleanliness of the child's
    own recovery, not parent-specific absorption. The top-`m` restriction is required —
    over the full dictionary the average competitor is an extreme-value cosine, and an
    unrestricted ratio would read high even for a clean match. Denominator zero
    (no positively-aligned latent in the top-m) yields `0.0`, never NaN.
    """
    wn = W_dec / W_dec.norm(dim=1, keepdim=True).clamp_min(_TINY)   # [S, D]
    gn = g / g.norm(dim=1, keepdim=True).clamp_min(_TINY)           # [F, D]
    out = torch.zeros(len(children), dtype=W_dec.dtype, device=W_dec.device)
    for pos, c in enumerate(children):
        cos = wn @ gn[c]                                            # [S]
        # stable sort (not topk) so |cos| ties at the m-th cutoff resolve deterministically,
        # keeping r_disp reproducible when latents are near-duplicates.
        top = torch.sort(cos.abs(), descending=True, stable=True).indices[:min(m, cos.numel())]
        pos_energy = cos[top].clamp_min(0.0).pow(2)                 # max(cos,0)^2 over top-m
        den = float(pos_energy.sum())
        if den <= 0.0:
            continue                                               # stays 0.0
        mc = int(match[c])
        own = torch.zeros_like(pos_energy)
        if mc >= 0:
            own[top == mc] = pos_energy[top == mc]
        out[pos] = (pos_energy.sum() - own.sum()) / den
    return out


def reconstruction_fvu(h: torch.Tensor, acts: torch.Tensor, W_dec: torch.Tensor,
                       b_dec: torch.Tensor | None = None) -> float:
    """Fraction of variance unexplained by the SAE reconstruction `acts @ W_dec (+ b_dec)`.

    `||h - h_hat||^2 / ||h - mean(h)||^2`: 0 is perfect, 1 matches the constant-mean
    baseline, > 1 is worse than the mean. On a clean oracle it bottoms out at the
    injected noise fraction.
    """
    h_hat = acts @ W_dec
    if b_dec is not None:
        h_hat = h_hat + b_dec
    resid = (h - h_hat).pow(2).sum()
    total = (h - h.mean(dim=0, keepdim=True)).pow(2).sum()
    return float(resid / total.clamp_min(_TINY))


def count_dead(acts: torch.Tensor, thresh: float = 0.0) -> int:
    """Number of latents that never exceed `thresh` on any token (never fire)."""
    return int((acts.max(dim=0).values <= thresh).sum())


def realized_l0(acts: torch.Tensor) -> float:
    """Realized L0 = the mean per-token count of firing latents, on the ACTUAL activations.

    `acts` [n, S] must be the SAE's activations on the REAL residual `h` (what the SAE saw), not a
    randn probe: a JumpReLU uses a uniform threshold, so a higher-norm input fires strictly more
    latents and a randn probe (‖·‖≈11 vs real h ‖·‖≈3.8) overstates L0 by ~30x. This is the honest
    sparsity the co-firing detectors and the sweep x-axis must be read against — the nominal top-k
    is not the realized L0 once the checkpoint is saved as a JumpReLU.
    """
    return float((acts > 0).double().sum(dim=1).mean())
