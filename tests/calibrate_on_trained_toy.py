"""Ground-truth calibration on Bussmann's toy, with a *trained* SAE.

`toy_world.py` hands the five metrics hand-built statistics: it proves the maths
but not that the metrics survive a real training run. This closes that gap:

  1. rebuild the toy hierarchy from the team's repo (sae-training/configs/tree.json),
  2. load the Matryoshka SAE we trained on it (outputs/toy_trained/, GPU 3),
  3. match each learned latent to the true feature it recovered (decoder cosine),
  4. cache the statistics the metrics read, computed from the LEARNED latents,
  5. run the five production metrics and compare the edges they keep against the
     known parent->child tree.

So this is calibration on ground truth AND on trained features. If the metrics
recover the tree here, they are trustworthy on the real gemma-2-2b SAE.

Run:
    PYTHONPATH=src python3 tests/calibrate_on_trained_toy.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
from safetensors.torch import load_file

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

CKPT = ROOT / "outputs" / "toy_trained"
TREE = ROOT / "sae-training" / "configs" / "tree.json"

from metrics import (                                             # noqa: E402
    coverage_legs, keep_edges, edge_reconstruction_condition,
    frequency_controlled_coverage, frequency_buckets,
)
from metrics.reconstruction import per_token_ablation_gain        # noqa: E402


# --------------------------------------------------------------------------
# toy hierarchy: read the team's Tree config directly. We reimplement the tiny
# sampler here so drawing samples needs none of the repo's heavy deps.
# --------------------------------------------------------------------------
class Node:
    def __init__(self, d, nxt):
        self.p = d["active_prob"]
        self.readout = d.get("is_read_out", True)
        self.excl = d.get("mutually_exclusive_children", False)
        self.idx = nxt[0] if self.readout else None
        if self.readout:
            nxt[0] += 1
        self.children = [Node(c, nxt) for c in d.get("children", [])]


def build_tree():
    return Node(json.loads(TREE.read_text()), [0])


def true_edges(tree) -> set[tuple[int, int]]:
    """Parent->child edges over READ-OUT features (hidden children have no index)."""
    edges = set()

    def walk(n):
        if n.idx is not None:
            for c in n.children:
                if c.idx is not None:
                    edges.add((n.idx, c.idx))
        for c in n.children:
            walk(c)

    walk(tree)
    return edges


def n_features(tree) -> int:
    m = [0]

    def walk(n):
        if n.readout:
            m[0] += 1
        for c in n.children:
            walk(c)

    walk(tree)
    return m[0]


def sample(tree, n, F, gen) -> torch.Tensor:
    """[n, F] binary ground-truth activations, honouring the two structural rules:
    a child can fire only if its parent fires, and exclusive siblings never co-fire.
    """
    out = torch.zeros(n, F)

    def rec(node, mask):
        fire = mask & (torch.rand(n, generator=gen) < node.p)
        if node.idx is not None:
            out[fire, node.idx] = 1.0
        if node.excl and node.children:
            probs = torch.tensor([c.p for c in node.children])
            pick = torch.multinomial(probs, n, replacement=True, generator=gen)
            for i, c in enumerate(node.children):
                rec(c, fire & (pick == i))
        else:
            for c in node.children:
                rec(c, fire)

    rec(tree, torch.ones(n, dtype=torch.bool))
    return out


# --------------------------------------------------------------------------
# the trained SAE
# --------------------------------------------------------------------------
def load_sae():
    return load_file(str(CKPT / "sae_weights.safetensors")), json.loads((CKPT / "cfg.json").read_text())


def encode(w, x, cfg):
    """Replicate the trained SAE's activation. The toy SAE uses batch_topk with no
    saved inference threshold, so we apply the same batch-wide top-(k*n) selection
    the architecture uses at train time; relu alone would leave the codes near-zero.
    """
    pre = (x - w["b_dec"]) @ w["W_enc"] + w["b_enc"]
    act = cfg.get("activation_function", "relu")
    if act == "relu":
        return torch.relu(pre)
    post = torch.relu(pre)
    k = cfg["k"]
    if act == "topk":
        top = post.topk(k, dim=-1)
        out = torch.zeros_like(post)
        return out.scatter_(-1, top.indices, top.values)
    # batch_topk: keep the k*n largest post-relu activations across the whole batch
    flat = post.flatten()
    keep = min(k * post.shape[0], flat.numel())
    top = flat.topk(keep)
    out = torch.zeros_like(flat)
    out.scatter_(-1, top.indices, top.values)
    return out.reshape_as(post)


def match_latents(w, true_dirs):
    """Each latent -> the true feature its decoder points at (-1 if none)."""
    W = w["W_dec"] / w["W_dec"].norm(dim=1, keepdim=True).clamp(min=1e-8)
    T = true_dirs / true_dirs.norm(dim=1, keepdim=True).clamp(min=1e-8)
    cos = W @ T.T
    best = cos.argmax(dim=1)
    best[cos.max(dim=1).values < 0.4] = -1
    return best


def main():
    torch.manual_seed(0)
    gen = torch.Generator().manual_seed(0)
    tree = build_tree()
    truth = true_edges(tree)
    F = n_features(tree)
    print(f"toy: {F} read-out features, {len(truth)} true parent->child edges")
    print(f"  true edges: {sorted(truth)}")

    w, cfg = load_sae()
    true_dirs = torch.eye(F)                          # identity embedding, per the paper
    match = match_latents(w, true_dirs)
    recovered = {int(t) for t in match if t >= 0}
    print(f"\ntrained Matryoshka SAE: recovered {len(recovered)}/{F} true features")

    n = 200_000
    gt = sample(tree, n, F, gen)                       # [n, F] ground-truth firings
    x = gt @ true_dirs                                 # model input
    acts = encode(w, x, cfg)                           # LEARNED latent activations
    resid = x - (acts @ w["W_dec"] + w["b_dec"])

    fired = (acts > 1e-3).double()
    g = per_token_ablation_gain(acts.double(), resid.double(), w["W_dec"].double())
    err = (resid.double() ** 2).sum(dim=1)

    true_parents = sorted({p for p, _ in truth})
    true_children = sorted({c for _, c in truth})
    m = match.tolist()
    parent_lat = [i for i, t in enumerate(m) if t in true_parents]
    child_lat = [i for i, t in enumerate(m) if t in true_children]
    print(f"latents matched to parents: {len(parent_lat)}, to children: {len(child_lat)}")
    if not parent_lat or not child_lat:
        print("SAE did not recover both parents and children; stop.")
        return

    fp = fired[:, parent_lat]
    fc = fired[:, child_lat]
    cofire = fp.T @ fc
    fire_p, fire_c = fp.sum(0), fc.sum(0)

    R, _ = coverage_legs(cofire, fire_p, fire_c)
    edge_mask = keep_edges(R, fire_p, fire_c, 0.5, 20)
    recon = edge_reconstruction_condition(
        fc.T @ err, g[:, parent_lat].T @ fc, (fc * g[:, child_lat]).sum(0), 0.01)

    token_ids = fired.argmax(dim=1).long()             # token-like id per row
    vocab = int(token_ids.max()) + 1
    counts = torch.zeros(vocab, dtype=torch.float64)
    counts.scatter_add_(0, token_ids, torch.ones(n, dtype=torch.float64))
    buckets = frequency_buckets(counts, 0.5, 0.4)[token_ids]
    cbb = torch.zeros(3, fp.shape[1], fc.shape[1], dtype=torch.float64)
    fcb = torch.zeros(3, fc.shape[1], dtype=torch.float64)
    for k in range(3):
        sel = (buckets == k).double().unsqueeze(1)
        cbb[k] = fp.T @ (fc * sel)
        fcb[k] = (fc * sel).sum(0)
    fcov = frequency_controlled_coverage(cbb, fcb, edge_mask)

    survivors = edge_mask & recon["passes"] & (fcov["survival"] >= 0.5)

    found = set()
    for pi in range(survivors.shape[0]):
        for ci in range(survivors.shape[1]):
            if survivors[pi, ci]:
                found.add((m[parent_lat[pi]], m[child_lat[ci]]))

    tp, fp_, fn = found & truth, found - truth, truth - found
    prec = len(tp) / max(len(found), 1)
    rec = len(tp) / max(len(truth), 1)
    print("\n=== calibration on the TRAINED toy ===")
    print(f"recovered edges: {sorted(found)}")
    print(f"true positives : {len(tp)} / {len(truth)}   false pos: {len(fp_)}   false neg: {len(fn)}")
    if fn:
        print(f"  missed edges : {sorted(fn)}")
    if fp_:
        print(f"  spurious     : {sorted(fp_)}")
    print(f"precision {prec:.2f}   recall {rec:.2f}")
    print(f"VERDICT: {'PASS' if prec >= 0.8 and rec >= 0.8 else 'NEEDS WORK'}")

    # per-edge verdict rows for the dashboard, ordered parent then child
    def edge_row(e):
        p, c = e
        cat = "recovered" if e in found else ("missed: child not learned"
              if c not in recovered else "missed")
        return {"edge": f"{p} -> {c}", "parent": p, "child": c,
                "found": e in found, "category": cat}

    result = {
        "n_features": F,
        "n_recovered_features": len(recovered),
        "recovered_features": sorted(recovered),
        "true_edges": sorted(truth),
        "found_edges": sorted(found),
        "true_positives": len(tp), "false_positives": len(fp_), "false_negatives": len(fn),
        "precision": prec, "recall": rec,
        "cfg": cfg,
        "edge_rows": [edge_row(e) for e in sorted(truth)] +
                     [{"edge": f"{p} -> {c}", "parent": p, "child": c, "found": True,
                       "category": "spurious"} for (p, c) in sorted(fp_)],
        "missed_children": sorted({c for _, c in fn if c not in recovered}),
    }
    out = ROOT / "outputs" / "trained_toy_calibration.json"
    out.write_text(json.dumps(result, indent=2))
    print(f"wrote {out}")
    return result


if __name__ == "__main__":
    main()
