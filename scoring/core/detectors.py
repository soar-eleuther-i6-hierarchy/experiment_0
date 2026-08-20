"""
detectors — the ten per-ordered-pair scalars, from SAE outputs ONLY.

Each detector reads the held-out activations and the (oriented, unit) decoders of the
recovered latents and produces an `[R, R]` matrix whose entry `[p, c]` scores the
ordered pair parent=p, child=c. LABELS NEVER ENTER HERE — the
firewall keeps relationship truth on the scoring side (`scoring.core.grid`).

Conventions (see `scoring.core.registry`):
  - firing := activation > 0 (BatchTopK's nonzero set is its top-k);
  - the diagonal is NaN (no self-pairs);
  - the co-firing detectors (coverage_R, asymmetry_R, pmi) carry the fixed
    smoothing (coverage +1e-6, PMI +1 Laplace), so a zero-fire endpoint gives a finite
    smoothed value, by design — NOT a hidden imputation;
  - the energy / reconstruction / frequency detectors (recon_2a, joint_child_mass,
    token_freq_survival) and the per-parent graph detectors have no fixed
    smoothing, so their genuine 0/0 cells (never-firing child, dead parent, underpowered
    bucket) are NaN — never a silently-imputed finite value, never +/-inf;
  - `compute_all` applies each detector's frozen sign so higher == more is-a-like.

The scalars are implemented directly, not via `metrics/`, whose zero-denominator clamping
would hide the undefined cells the scorer must drop - WILL FIX THIS IN LATER COMMITS
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from scoring.core.registry import DETECTOR_SIGN, DETECTORS

DT = torch.float64
_NAN = float("nan")


@dataclass(frozen=True)
class DetectorInputs:
    """SAE-side inputs for the recovered latents (R of them), over n held-out tokens.

    acts_rec [n, R]  float activations of the matched latent per recovered feature
    W_unit   [R, D]  oriented, L2-normalized decoder rows (geometry channel)
    W_raw    [R, D]  raw decoder rows (reconstruction channel)
    h        [n, D]  the residual activations the SAE saw (recon channel)
    b_dec    [D]     decoder bias (recon channel)
    tokens   [n]     token id per row (frequency channel)
    vocab            token-id vocabulary size
    """

    acts_rec: torch.Tensor
    W_unit: torch.Tensor
    W_raw: torch.Tensor
    h: torch.Tensor | None = None
    b_dec: torch.Tensor | None = None
    tokens: torch.Tensor | None = None
    vocab: int = 0


def _nan_diag(m: torch.Tensor) -> torch.Tensor:
    m = m.clone()
    m.fill_diagonal_(_NAN)
    return m


def _broadcast_parent(vec: torch.Tensor, R: int) -> torch.Tensor:
    """A per-parent vector -> an [R, R] matrix constant across columns, NaN diagonal."""
    return _nan_diag(vec.reshape(R, 1).expand(R, R).clone())


def cofiring(Fm: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, int]:
    """(cofire[R,R], fire[R], N) from a boolean firing mask `Fm` [n, R]."""
    F = Fm.double()
    cofire = F.transpose(0, 1) @ F
    fire = F.sum(dim=0)
    return cofire, fire, int(Fm.shape[0])


def coverage_R(cofire: torch.Tensor, fire: torch.Tensor, eps: float) -> torch.Tensor:
    """Reverse coverage R(p|c) = cofire[p,c] / fire[c] = P(parent fires | child fires).

    CONDITIONAL denominator (no additive eps): a child that never fires (fire[c]==0) is
    UNDEFINED (0/0) -> NaN, never a fabricated 0 that would rank a dead child as "never
    contained". `eps` is retained in the signature for back-compat but no longer perturbs a
    defined cell.
    """
    fire_c = fire.reshape(1, -1)
    denom = torch.where(fire_c > 0, fire_c, torch.full_like(fire_c, _NAN))
    return _nan_diag(cofire / denom)


def _forward_F(cofire: torch.Tensor, fire: torch.Tensor, eps: float) -> torch.Tensor:
    """Forward coverage F(p|c) = cofire[p,c] / fire[p] = P(child fires | parent fires).

    Same conditional denominator as `coverage_R`: a dead parent (fire[p]==0) is NaN
    (consistent with `joint_child_J`); no additive eps perturbs a live parent.
    """
    fire_p = fire.reshape(-1, 1)
    denom = torch.where(fire_p > 0, fire_p, torch.full_like(fire_p, _NAN))
    return cofire / denom


def asymmetry_R(R_mat: torch.Tensor) -> torch.Tensor:
    """R(p|c) - R(c|p): >0 for a directed containment, ~0 for symmetric co-firing."""
    return _nan_diag(R_mat - R_mat.transpose(0, 1))


def pmi(cofire: torch.Tensor, fire: torch.Tensor, N: int, laplace: float) -> torch.Tensor:
    """Smoothed PMI = log((cofire+l)*N / ((fire_p+l)(fire_c+l))). Symmetric."""
    num = (cofire + laplace) * float(N)
    den = (fire.reshape(-1, 1) + laplace) * (fire.reshape(1, -1) + laplace)
    return _nan_diag(torch.log(num / den))


def s_res(W_unit: torch.Tensor) -> torch.Tensor:
    """Decoder-geometry overlap = cos(d_c, d_p) between oriented unit decoders.

    For an is-a edge this is cos(g_c, g_p) = alpha_c * sqrt(1 - alpha_p^2); 0 for the
    orthogonal-geometry (firing_only) null. Symmetric, so orientation-blind by design.
    """
    W = W_unit.double()
    return _nan_diag(W @ W.transpose(0, 1))


def edge_mask(R_mat: torch.Tensor, fire: torch.Tensor, tau: float, min_fire: int,
              cofire: torch.Tensor | None = None, min_joint: int = 0) -> torch.Tensor:
    """Inferred edge set: R(p|c) >= tau, both endpoints fire >= min_fire, and (when
    `cofire`/`min_joint` are given) at least `min_joint` co-firing tokens. Bool [R,R].

    The joint-support gate kills chance edges — a child firing `min_fire` times inside a
    near-always-on parent reaches R=1 with no evidence beyond base rate (this is the
    `metrics.coverage.keep_edges` guard, MIN_JOINT=30). The NaN diagonal of `R_mat`
    compares False, so self-edges are excluded automatically.
    """
    keep = R_mat >= tau                                  # NaN >= tau -> False
    enough = fire >= min_fire
    keep = keep & enough.reshape(-1, 1) & enough.reshape(1, -1)
    if cofire is not None and min_joint > 0:
        keep = keep & (cofire >= min_joint)
    return keep


def outdegree(em: torch.Tensor, fire: torch.Tensor, N: int) -> torch.Tensor:
    """Per-parent out-degree (# kept children), broadcast across columns. Raw (+); the
    frozen sign (-1) is applied in `compute_all` so a wide superparent scores LOW. A dead
    parent (never fires on this draw) is evidence-free -> NaN, not a confident 0."""
    R = em.shape[0]
    outdeg = em.double().sum(dim=1)
    outdeg = torch.where(fire > 0, outdeg, torch.full_like(outdeg, _NAN))
    return _broadcast_parent(outdeg, R)


def joint_child_J(F_mat: torch.Tensor, em: torch.Tensor, fire: torch.Tensor) -> torch.Tensor:
    """Per-parent J(p) = sum_c forward-coverage(p,c) over kept children, capped at 1.
    A dead parent (never fires) is undefined -> NaN."""
    R = F_mat.shape[0]
    contrib = torch.where(em, F_mat, torch.zeros_like(F_mat))
    J = contrib.sum(dim=1).clamp(max=1.0)
    J = torch.where(fire > 0, J, torch.full_like(J, _NAN))
    return _broadcast_parent(J, R)


def joint_child_mass(acts_rec: torch.Tensor, Fm: torch.Tensor, em: torch.Tensor) -> torch.Tensor:
    """Per-parent r_mass(p) = sum_t f_p^2 * 1[>=1 kept child fires] / sum_t f_p^2.

    A dead parent (zero activation energy) gives 0/0 -> NaN (never 0 or 0.5).
    """
    R = acts_rec.shape[1]
    energy = (acts_rec.double() ** 2)                    # [n, R]
    energy_total = energy.sum(dim=0)                     # [R]
    r_mass = torch.full((R,), _NAN, dtype=DT)
    for p in range(R):
        kids = em[p].nonzero(as_tuple=True)[0]
        if float(energy_total[p]) <= 0.0:
            continue                                     # dead parent -> NaN
        if kids.numel() == 0:
            r_mass[p] = 0.0                              # live parent, no kept children
            continue
        any_child = Fm[:, kids].any(dim=1)               # [n]
        r_mass[p] = float(energy[any_child, p].sum() / energy_total[p])
    return _broadcast_parent(r_mass, R)


def sibling_redundancy(em: torch.Tensor, cofire: torch.Tensor, fire: torch.Tensor) -> torch.Tensor:
    """Per-parent mean pairwise sibling Jaccard over its kept children, broadcast.

    Jaccard(i,j) = cofire[i,j] / (fire[i]+fire[j]-cofire[i,j]). Parents with <2 kept
    children are undefined -> NaN. Raw (high == redundant); frozen sign (-1) flips it so
    a redundant/splitting parent scores LOW.
    """
    R = em.shape[0]
    red = torch.full((R,), _NAN, dtype=DT)
    for p in range(R):
        kids = em[p].nonzero(as_tuple=True)[0]
        k = int(kids.numel())
        if k < 2:
            continue
        cf = cofire[kids][:, kids]                       # [k, k] child co-firing
        fk = fire[kids]
        union = fk.reshape(-1, 1) + fk.reshape(1, -1) - cf
        jac = torch.where(union > 0, cf / union, torch.full_like(cf, _NAN))
        iu = torch.triu_indices(k, k, offset=1)          # upper triangle = each sibling pair once
        vals = jac[iu[0], iu[1]]
        vals = vals[~torch.isnan(vals)]
        if vals.numel():
            red[p] = float(vals.mean())
    return _broadcast_parent(red, R)


def recon_2a(acts_rec: torch.Tensor, h: torch.Tensor, W_raw: torch.Tensor,
             b_dec: torch.Tensor, Fm: torch.Tensor) -> torch.Tensor:
    """Reconstruction-ablation parent gain.

    Over tokens where the child fires: parent_gain[p,c] = sum g[t,p] / sum ||err[t]||^2,
    where err = h - (acts @ W_raw + b_dec) and g[t,f] = 2 a_f <d_f, err_t> + a_f^2 ||d_f||^2.
    A never-firing child gives a 0 denominator -> NaN (undefined, dropped downstream).
    """
    a = acts_rec.double()
    W = W_raw.double()
    x_hat = a @ W + (b_dec.double() if b_dec is not None else 0.0)
    err = h.double() - x_hat                             # [n, D]
    dot = err @ W.transpose(0, 1)                        # [n, R] = <d_f, err_t>
    dnorm2 = (W ** 2).sum(dim=1)                         # [R]
    g = 2.0 * a * dot + (a ** 2) * dnorm2.reshape(1, -1)  # [n, R] per-token ablation gain
    Fc = Fm.double()                                     # [n, R] child-firing indicator
    numer = g.transpose(0, 1) @ Fc                       # [R(parent), R(child)]
    err_sq = (err ** 2).sum(dim=1)                       # [n]
    denom_c = err_sq @ Fc                                # [R(child)]
    out = numer / torch.where(denom_c > 0, denom_c, torch.full_like(denom_c, _NAN)).reshape(1, -1)
    return _nan_diag(out)


def token_freq_survival(Fm: torch.Tensor, tokens: torch.Tensor, vocab: int,
                        min_joint: int, high_mass: float, mid_mass: float,
                        min_fire_low: int) -> torch.Tensor:
    """Frequency-controlled coverage survival.

    survival(p,c) = R(p|c) over rare-token buckets {1,2} / R(p|c) over all buckets.
    An edge that survives on rare tokens (~1) is a real relationship; a frequency-driven
    coincidence collapses (~0). NaN where unmeasurable: the child never co-fires with the
    parent (R_all==0), fires too rarely (< min_fire_low), or has too few JOINT co-fires
    (< min_joint) to trust the ratio. NOT gated on being a kept edge: the shared-token
    confound caps reverse-coverage at ~0.18 (it co-fires only on shared top-frequency ids),
    so an edge-mask gate would NaN exactly the shared-token pairs this detector must score.
    """
    counts = torch.bincount(tokens, minlength=vocab).double()
    order = torch.argsort(counts, descending=True)
    cum = torch.cumsum(counts[order], dim=0) / counts.sum().clamp_min(1.0)
    bucket_sorted = torch.full((counts.numel(),), 2, dtype=torch.long)
    bucket_sorted[cum <= high_mass] = 0
    bucket_sorted[(cum > high_mass) & (cum <= high_mass + mid_mass)] = 1
    id_bucket = torch.empty_like(bucket_sorted)
    id_bucket[order] = bucket_sorted
    tok_bucket = id_bucket[tokens]                       # [n]

    Ffloat = Fm.double()
    rest = (tok_bucket > 0)                              # buckets 1 and 2 (rare)
    Fm_rest = Fm & rest.reshape(-1, 1)
    cofire_all = Ffloat.transpose(0, 1) @ Ffloat
    fire_all = Ffloat.sum(dim=0)
    Fr = Fm_rest.double()
    cofire_rest = Fr.transpose(0, 1) @ Fr
    fire_rest = Fr.sum(dim=0)

    R_all = cofire_all / fire_all.clamp_min(1.0).reshape(1, -1)
    R_rest = cofire_rest / fire_rest.clamp_min(1.0).reshape(1, -1)
    survival = (R_rest / R_all.clamp_min(1e-12)).clamp(max=1.5)

    # Support floor on TOTAL child firing (not the rare-bucket count): a shared-token
    # child fires almost entirely in bucket 0, so gating on fire_rest would NaN it — but a
    # zero rare-bucket rate is the SIGNAL (survival -> 0), not missing data. The min_joint
    # floor on the actual co-fire count kills chance pairs (a 1-cofire ratio is not a
    # measurement), mirroring edge_mask's joint-support guard without requiring R >= tau.
    undefined = ((fire_all.reshape(1, -1) < min_fire_low) | (R_all <= 0)
                 | (cofire_all < min_joint))
    survival = torch.where(undefined, torch.full_like(survival, _NAN), survival)
    return _nan_diag(survival)


# --------------------------------------------------------------------------
# the handoff
# --------------------------------------------------------------------------
def compute_all(inputs: DetectorInputs, constants: dict) -> dict[str, torch.Tensor]:
    """Every detector as an oriented [R, R] matrix (higher == is-a-like), NaN diagonal.

    Applies each detector's frozen `DETECTOR_SIGN` exactly once. Any residual +/-inf is
    coerced to NaN so downstream means/sorts see only finite-or-NaN.
    """
    if inputs.h is None or inputs.b_dec is None or inputs.tokens is None:
        missing = [n for n in ("h", "b_dec", "tokens") if getattr(inputs, n) is None]
        raise ValueError(
            f"compute_all needs {missing} (recon_2a / token_freq_survival read them); "
            f"reduce_to_recovered must be given h/b_dec/tokens before scoring")
    eps = constants["coverage_eps"]
    Fm = inputs.acts_rec > constants["fire_thresh"]
    cofire, fire, N = cofiring(Fm)
    R_mat = coverage_R(cofire, fire, eps)
    F_mat = _forward_F(cofire, fire, eps)
    em = edge_mask(R_mat, fire, constants["edge_tau"], constants["min_fire_count"],
                   cofire=cofire, min_joint=constants["min_joint"])

    raw = {
        "coverage_R": R_mat,
        "asymmetry_R": asymmetry_R(R_mat),
        "joint_child_J": joint_child_J(F_mat, em, fire),
        "pmi": pmi(cofire, fire, N, constants["pmi_laplace"]),
        "token_freq_survival": token_freq_survival(
            Fm, inputs.tokens, inputs.vocab, constants["min_joint"],
            constants["freq_high_mass"], constants["freq_mid_mass"],
            constants["freq_min_fire_low"]),
        "recon_2a": recon_2a(inputs.acts_rec, inputs.h, inputs.W_raw, inputs.b_dec, Fm),
        "s_res": s_res(inputs.W_unit),
        "sibling_redundancy": sibling_redundancy(em, cofire, fire),
        "joint_child_mass": joint_child_mass(inputs.acts_rec, Fm, em),
        "outdegree": outdegree(em, fire, N),
    }

    out: dict[str, torch.Tensor] = {}
    for name in DETECTORS:
        m = raw[name] * float(DETECTOR_SIGN[name])
        m = torch.where(torch.isinf(m), torch.full_like(m, _NAN), m)
        out[name] = _nan_diag(m)
    return out
