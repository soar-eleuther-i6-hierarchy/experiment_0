"""
Bulk-fetch every feature's explanation for the gemma-2-2b layer-6 Matryoshka SAE
and write a single {index: description} map to outputs/feature_labels.json.

Source: Neuronpedia's public dataset export on S3 (256 gzipped JSONL batches,
128 features each = all 32768 features). This is the same label text the
Neuronpedia API returns, but pulled in one sweep instead of one HTTP call per
feature - so it is faster, complete, and works offline once cached.

    https://neuronpedia-datasets.s3.us-east-1.amazonaws.com/
        v1/gemma-2-2b/<LAYER>-res-matryoshka-dc/explanations/batch-{i}.jsonl.gz

The layer is taken from config (EXP0_LAYER env var); labels land in that layer's
outputs/layer_NN/feature_labels.json. Each JSONL line has (among other fields)
"index" and "description"; we keep only those two.

Run:
    cd experiment_0
    python3 fetch_labels.py                 # all 256 batches -> feature_labels.json
    python3 fetch_labels.py --batches 4     # smoke: first 4 batches only
    EXP0_LAYER=12 python3 fetch_labels.py   # labels for a different layer
"""

from __future__ import annotations

import argparse
import gzip
import json
import urllib.request

import config as C

BASE = C.S3_EXPLANATIONS
N_BATCHES = 256  # 256 * 128 == 32768 == D_SAE
LABELS_PATH = C.FEATURE_LABELS_PATH


def fetch_batch(i: int) -> dict[str, str]:
    """Return {index_str: description} for one batch, or {} on failure."""
    url = BASE.format(i)
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "refusal-lens-exp0"})
        with urllib.request.urlopen(req, timeout=30) as r:
            raw = gzip.decompress(r.read())
    except Exception as e:  # network / missing batch
        print(f"  batch-{i}: fetch failed ({type(e).__name__})")
        return {}
    out = {}
    for line in raw.decode("utf-8").splitlines():
        if not line.strip():
            continue
        d = json.loads(line)
        out[str(d["index"])] = (d.get("description") or "").strip()
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--batches", type=int, default=N_BATCHES,
                    help="how many batches to fetch (default all 256)")
    args = ap.parse_args()

    labels: dict[str, str] = {}
    for i in range(args.batches):
        labels.update(fetch_batch(i))
        if (i + 1) % 32 == 0 or i + 1 == args.batches:
            print(f"[labels] {i + 1}/{args.batches} batches, {len(labels)} features")

    LABELS_PATH.write_text(json.dumps(labels, ensure_ascii=False))
    print(f"[labels] wrote {LABELS_PATH}  ({len(labels)} features, "
          f"expected {C.D_SAE})")


if __name__ == "__main__":
    main()
