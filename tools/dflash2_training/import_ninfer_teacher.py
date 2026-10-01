#!/usr/bin/env python3
"""Import exact NInfer DFlash teacher arrays into the training dataset format.

The C++ teacher path records the actual loaded .ninfer target:
- after-layer residual taps as raw BF16 bits,
- exact greedy label,
- stable target top16 ids and FP32 logits.

This importer renormalizes those top16 logits into log-probabilities and writes the same .pt
sequence blobs consumed by train.py. The final unused predictor row is padded so the blob keeps the
same N-row tensor convention as the Hugging Face recorder.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch


def read_meta(prefix: Path) -> dict:
    with open(str(prefix) + ".json", encoding="utf-8") as f:
        return json.load(f)


def read_i32(prefix: Path, suffix: str) -> np.ndarray:
    return np.fromfile(str(prefix) + suffix, dtype="<i4")


def read_f32(prefix: Path, suffix: str) -> np.ndarray:
    return np.fromfile(str(prefix) + suffix, dtype="<f4")


def load_record(prefix: Path) -> tuple[dict, dict]:
    meta = read_meta(prefix)
    n = int(meta["token_count"])
    p = int(meta["predictor_count"])
    hidden = int(meta["hidden_size"])
    layers = [int(x) for x in meta["target_layer_ids"]]
    topk = int(meta["top_k"])
    fused_width = hidden * len(layers)
    if n != p + 1 or topk <= 0 or not layers:
        raise ValueError(f"{prefix}: invalid teacher metadata")

    ids_np = read_i32(prefix, ".tokens.i32")
    labels_np = read_i32(prefix, ".labels.i32")
    top_ids_np = read_i32(prefix, ".top_ids.i32")
    top_logits_np = read_f32(prefix, ".top_logits.f32")
    features_np = np.fromfile(str(prefix) + ".features.bf16", dtype="<u2")

    if ids_np.size != n or labels_np.size != p:
        raise ValueError(f"{prefix}: token/label array size mismatch")
    if top_ids_np.size != p * topk or top_logits_np.size != p * topk:
        raise ValueError(f"{prefix}: top-k array size mismatch")
    if features_np.size != p * fused_width:
        raise ValueError(f"{prefix}: feature array size mismatch")

    ids = torch.from_numpy(ids_np.astype(np.int32, copy=False)).to(torch.int32)
    labels = torch.from_numpy(labels_np.astype(np.int32, copy=False)).to(torch.int32)
    top_ids = torch.from_numpy(
        top_ids_np.reshape(p, topk).astype(np.int32, copy=False)).to(torch.int32)
    top_logits = torch.from_numpy(
        top_logits_np.reshape(p, topk).astype(np.float32, copy=False))
    top_lp = torch.log_softmax(top_logits, dim=-1).to(torch.float16)

    # Preserve BF16 payload bits from the target. torch.Tensor.view(dtype) reinterprets the
    # uint16 storage; clone first because NumPy owns the original file-backed allocation.
    fused_bits = torch.from_numpy(
        features_np.reshape(p, fused_width).copy()).to(torch.uint16)
    fused = fused_bits.view(torch.bfloat16)

    # train.py intentionally never trains on the last token's nonexistent next-token label, but
    # its held-out acceptance helper precomputes DFlash context tensors over len(ids). Pad one
    # inert row to retain that existing shape contract.
    fused = torch.cat(
        [fused, torch.zeros((1, fused_width), dtype=torch.bfloat16)], dim=0)
    label_pad = labels[-1:] if labels.numel() else torch.zeros((1,), dtype=torch.int32)
    labels = torch.cat([labels, label_pad], dim=0)
    if p:
        top_ids = torch.cat([top_ids, top_ids[-1:].clone()], dim=0)
        top_lp = torch.cat([top_lp, top_lp[-1:].clone()], dim=0)
    else:
        top_ids = torch.zeros((1, topk), dtype=torch.int32)
        top_lp = torch.full((1, topk), -float("inf"), dtype=torch.float16)

    blob = {
        "ids": ids,
        "fused": fused,
        "label": labels,
        "top_ids": top_ids,
        "top_lp": top_lp,
        "target_layer_ids": layers,
        "teacher_source": "ninfer-target",
        "teacher_distribution": "top16-renormalized",
        "tap_semantics": meta.get("tap_semantics", "after-layer-residual"),
    }
    return meta, blob


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--prefix", required=True,
                    help="teacher prefix; PREFIX.manifest.json is preferred, PREFIX.json is single")
    ap.add_argument("--out", required=True, help="training dataset directory")
    ap.add_argument("--default-split", default="train")
    args = ap.parse_args()

    prefix = Path(args.prefix)
    manifest_path = Path(str(prefix) + ".manifest.json")
    records: list[dict]
    if manifest_path.exists():
        with open(manifest_path, encoding="utf-8") as f:
            outer = json.load(f)
        records = list(outer["records"])
        base = manifest_path.parent
    else:
        records = [{
            "name": prefix.name,
            "prefix": prefix.name,
            "kind": "corpus",
            "topic": "prose",
            "split": args.default_split,
            "gen_start": 0,
        }]
        base = prefix.parent

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    converted = []
    common_layers = None
    topk = None

    for index, record in enumerate(records):
        item_prefix = base / record["prefix"]
        meta, blob = load_record(item_prefix)
        name = str(record.get("name") or f"seq-{index:06d}")
        # Keep filenames safe while preserving the human name in the manifest.
        filename = f"seq-{index:06d}.pt"
        blob.update({
            "name": name,
            "kind": str(record.get("kind", "corpus")),
            "topic": str(record.get("topic", "prose")),
            "split": str(record.get("split", args.default_split)),
            "gen_start": int(record.get("gen_start", 0)),
        })
        torch.save(blob, out / filename)

        layers = [int(x) for x in meta["target_layer_ids"]]
        this_topk = int(meta["top_k"])
        if common_layers is None:
            common_layers = layers
            topk = this_topk
        elif common_layers != layers or topk != this_topk:
            raise ValueError("teacher records disagree on tap layers or top-k width")

        converted.append({
            "name": name,
            "file": filename,
            "kind": blob["kind"],
            "topic": blob["topic"],
            "split": blob["split"],
            "n": int(blob["ids"].numel()),
            "gen_start": blob["gen_start"],
        })
        print(f"[{index+1}/{len(records)}] {name}: {converted[-1]['n']} tokens")

    with open(out / "manifest.json", "w", encoding="utf-8") as f:
        json.dump({
            "version": 1,
            "teacher_source": "ninfer-target",
            "distribution": "top16-renormalized",
            "target_layer_ids": common_layers or [],
            "topk": topk or 0,
            "sequences": converted,
        }, f, indent=2)
    print(f"wrote {len(converted)} NInfer teacher sequences to {out}")


if __name__ == "__main__":
    main()
