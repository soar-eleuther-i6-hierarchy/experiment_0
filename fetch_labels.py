"""
Bulk-fetch every feature's explanation for the current layer's gemma-2-2b
Matryoshka SAE and write a single {index: description} map to
outputs/layer_NN/feature_labels.json.

Source: Neuronpedia's public dataset export on S3 (gzipped JSONL batches). This
is the same label text the Neuronpedia API returns, but pulled in one sweep
instead of one HTTP call per feature - faster, complete, and offline once cached.

    https://neuronpedia-datasets.s3.us-east-1.amazonaws.com/
        v1/gemma-2-2b/<LAYER>-res-matryoshka-dc/explanations/batch-{i}.jsonl.gz

The **number of batches varies per layer** (e.g. layer 6 has 256, layer 12 has
515 - some layers carry more than one explanation set), so we DON'T assume a
fixed count: we list the actual batch keys from the S3 bucket first, then fetch
them all. (Assuming 256 previously under-fetched layers with more batches - layer
12 came back with only ~half its features.)

The layer is taken from config (EXP0_LAYER env var); labels land in that layer's
outputs/layer_NN/feature_labels.json. Each JSONL line has (among other fields)
"index" and "description"; we keep only those two. If a feature appears in more
than one batch (duplicate explanation sets), the last one wins.

Run:
    cd experiment_0
    python3 fetch_labels.py                 # all batches for the layer
    python3 fetch_labels.py --batches 4     # smoke: first 4 listed batches only
    EXP0_LAYER=12 python3 fetch_labels.py   # labels for a different layer
"""

from __future__ import annotations

import argparse
import gzip
import json
import re
import urllib.parse
import urllib.request

import config as C

BUCKET = "https://neuronpedia-datasets.s3.us-east-1.amazonaws.com/"
PREFIX = f"v1/gemma-2-2b/{C.SAE_SOURCE}/explanations/batch-"
LABELS_PATH = C.FEATURE_LABELS_PATH
_UA = {"User-Agent": "refusal-lens-exp0"}


def list_batch_keys() -> list[str]:
    """List every batch-*.jsonl.gz key under the layer's explanations prefix,
    following S3 pagination (a single listing returns at most 1000 keys)."""
    keys: list[str] = []
    token = None
    while True:
        url = f"{BUCKET}?list-type=2&prefix={PREFIX}"
        if token:
            url += f"&continuation-token={urllib.parse.quote(token)}"
        req = urllib.request.Request(url, headers=_UA)
        with urllib.request.urlopen(req, timeout=30) as r:
            xml = r.read().decode("utf-8")
        keys += re.findall(r"<Key>([^<]+\.jsonl\.gz)</Key>", xml)
        m = re.search(r"<NextContinuationToken>([^<]+)</NextContinuationToken>", xml)
        if re.search(r"<IsTruncated>true</IsTruncated>", xml) and m:
            token = m.group(1)
        else:
            break
    return keys


def fetch_batch(key: str) -> dict[str, str]:
    """Return {index_str: description} for one batch key, or {} on failure."""
    try:
        req = urllib.request.Request(BUCKET + key, headers=_UA)
        with urllib.request.urlopen(req, timeout=30) as r:
            raw = gzip.decompress(r.read())
    except Exception as e:  # network / missing batch
        print(f"  {key.split('/')[-1]}: fetch failed ({type(e).__name__})")
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
    ap.add_argument("--batches", type=int, default=None,
                    help="cap on how many listed batches to fetch (default: all)")
    args = ap.parse_args()

    keys = list_batch_keys()
    print(f"[labels] layer {C.LAYER}: {len(keys)} batch files on S3")
    if args.batches is not None:
        keys = keys[: args.batches]

    labels: dict[str, str] = {}
    for i, key in enumerate(keys):
        labels.update(fetch_batch(key))
        if (i + 1) % 32 == 0 or i + 1 == len(keys):
            print(f"[labels] {i + 1}/{len(keys)} batches, {len(labels)} features")

    LABELS_PATH.write_text(json.dumps(labels, ensure_ascii=False))
    print(f"[labels] wrote {LABELS_PATH}  ({len(labels)} features, "
          f"expected {C.D_SAE})")


if __name__ == "__main__":
    main()
