#!/usr/bin/env python3
"""Convert native NInfer DFlash teacher files (.ndft) into the dataset used by train.py.

The native recorder stores exact target residual taps and target-head top-16 *scores*.  train.py
applies softmax to the supplied top-K values, so logits and log-probabilities are equivalent up to
an additive per-row normalization constant.
"""
from __future__ import annotations

import argparse
import json
import struct
from pathlib import Path

import torch


MAGIC = b"NINFDFT1"
HEADER = struct.Struct("<IIII")  # version, feature_rows, layer_count, top_k
CHUNK = struct.Struct("<II")     # token_count, chunk_begin


def read_exact(handle, size: int) -> bytes:
    data = handle.read(size)
    if len(data) != size:
        raise EOFError(f"truncated teacher file: wanted {size}, got {len(data)}")
    return data


def read_teacher(path: Path) -> dict:
    with path.open("rb") as f:
        if read_exact(f, len(MAGIC)) != MAGIC:
            raise ValueError(f"{path}: invalid magic")
        version, feature_rows, layer_count, top_k = HEADER.unpack(read_exact(f, HEADER.size))
        if version != 1 or feature_rows <= 0 or layer_count <= 0 or top_k <= 0:
            raise ValueError(f"{path}: unsupported header")
        layers = list(struct.unpack(
            f"<{layer_count}I", read_exact(f, 4 * layer_count)))

        ids_parts = []
        pos_parts = []
        fused_parts = []
        top_id_parts = []
        top_score_parts = []
        expected_begin = 0

        while True:
            prefix = f.read(CHUNK.size)
            if not prefix:
                break
            if len(prefix) != CHUNK.size:
                raise EOFError(f"{path}: truncated chunk header")
            tokens, begin = CHUNK.unpack(prefix)
            if tokens <= 0 or begin != expected_begin:
                raise ValueError(
                    f"{path}: non-contiguous chunk begin={begin}, expected={expected_begin}")

            ids = torch.frombuffer(
                bytearray(read_exact(f, tokens * 4)), dtype=torch.int32).clone()
            positions = torch.frombuffer(
                bytearray(read_exact(f, tokens * 4)), dtype=torch.int32).clone()
            fused = torch.frombuffer(
                bytearray(read_exact(f, tokens * feature_rows * 2)),
                dtype=torch.bfloat16).clone().reshape(tokens, feature_rows)
            top_ids = torch.frombuffer(
                bytearray(read_exact(f, tokens * top_k * 4)),
                dtype=torch.int32).clone().reshape(tokens, top_k)
            top_scores = torch.frombuffer(
                bytearray(read_exact(f, tokens * top_k * 4)),
                dtype=torch.float32).clone().reshape(tokens, top_k)

            ids_parts.append(ids)
            pos_parts.append(positions)
            fused_parts.append(fused)
            top_id_parts.append(top_ids)
            top_score_parts.append(top_scores)
            expected_begin += tokens

    if not ids_parts:
        raise ValueError(f"{path}: contains no chunks")
    ids = torch.cat(ids_parts)
    positions = torch.cat(pos_parts)
    fused = torch.cat(fused_parts)
    top_ids = torch.cat(top_id_parts)
    top_scores = torch.cat(top_score_parts)
    return {
        "ids": ids,
        "positions": positions,
        "fused": fused,
        "label": top_ids[:, 0].clone(),
        "top_ids": top_ids,
        # train.py only uses softmax(top_lp), so raw target scores preserve the same truncated
        # distribution while avoiding a 248k-vocabulary logsumexp in the recorder.
        "top_lp": top_scores.to(torch.float16),
        "target_layer_ids": layers,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", required=True, help="directory containing teacher-*.ndft")
    ap.add_argument("--out", required=True)
    ap.add_argument("--topic", default="prose")
    ap.add_argument("--kind", default="corpus")
    ap.add_argument("--holdout-every", type=int, default=10,
                    help="every Nth sequence is held out; 0 disables automatic holdout")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    source = Path(args.input)
    files = sorted(source.glob("teacher-*.ndft"))
    if args.limit:
        files = files[:args.limit]
    if not files:
        raise SystemExit(f"no teacher-*.ndft files under {source}")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    manifest = []
    layer_ids = None

    for index, path in enumerate(files):
        blob = read_teacher(path)
        if layer_ids is None:
            layer_ids = blob["target_layer_ids"]
        elif blob["target_layer_ids"] != layer_ids:
            raise ValueError(f"{path}: target layer ids differ from previous files")

        name = f"ninfer-{index:06d}"
        split = "heldout" if args.holdout_every and index % args.holdout_every == 0 else "train"
        blob.update({
            "name": name,
            "kind": args.kind,
            "topic": args.topic,
            "split": split,
            "gen_start": 0,
        })
        torch.save(blob, out / f"{name}.pt")
        manifest.append({
            "name": name,
            "kind": args.kind,
            "topic": args.topic,
            "split": split,
            "n": int(blob["ids"].numel()),
            "gen_start": 0,
            "source": path.name,
        })
        print(f"[{index+1}/{len(files)}] {path.name} -> {name} "
              f"{manifest[-1]['n']} tokens {split}", flush=True)

    with (out / "manifest.json").open("w", encoding="utf-8") as f:
        json.dump({
            "version": 1,
            "source": "ninfer-native-teacher-v1",
            "target_layer_ids": layer_ids,
            "topk": 16,
            "sequences": manifest,
        }, f, indent=2)
    print(f"wrote {len(manifest)} sequences to {out}", flush=True)


if __name__ == "__main__":
    main()
