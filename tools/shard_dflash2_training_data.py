#!/usr/bin/env python3
"""Pack per-sequence DFlash2 training records into mmap-friendly shard files.

tools/train_dflash2.py already understands this layout. Sharding changes only storage: sequence
boundaries, split labels and tensor values are preserved exactly.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

import torch

FIELDS = ("fused", "ids", "label", "top_ids", "top_lp")


def tensor_bytes(t: torch.Tensor) -> int:
    return t.numel() * t.element_size()


def load_record(path: str) -> dict[str, torch.Tensor]:
    raw = torch.load(path, map_location="cpu", weights_only=False)
    out = {}
    for field in FIELDS:
        value = raw.get(field)
        if not isinstance(value, torch.Tensor):
            raise ValueError(f"{path}: missing tensor {field}")
        out[field] = value.contiguous()
    n = out["ids"].shape[0]
    for field in FIELDS:
        if out[field].shape[0] != n:
            raise ValueError(f"{path}: {field} first dimension differs from ids")
    return out


def flush(out_dir: str, index: int, pending: list[tuple[dict, dict[str, torch.Tensor]]]) -> None:
    if not pending:
        return
    bundle = {}
    for field in FIELDS:
        bundle[field] = torch.cat([blob[field] for _, blob in pending], dim=0).contiguous()
    path = os.path.join(out_dir, f"shard-{index:04d}.pt")
    temp = path + ".tmp"
    torch.save(bundle, temp)
    os.replace(temp, path)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", required=True, help="directory containing manifest.json and .pt files")
    ap.add_argument("--out", required=True, help="output directory for shard files + manifest")
    ap.add_argument("--max-shard-gib", type=float, default=2.5)
    ap.add_argument("--copy-meta", action="store_true",
                    help="copy non-.pt sidecar files from the source directory")
    ap.add_argument("--delete-source", action="store_true",
                    help="delete source per-sequence .pt files only after all shards/manifest exist")
    args = ap.parse_args()

    if args.max_shard_gib <= 0:
        raise SystemExit("--max-shard-gib must be positive")
    src = os.path.abspath(os.path.expanduser(args.data))
    out_dir = os.path.abspath(os.path.expanduser(args.out))
    if src == out_dir:
        raise SystemExit("--out must differ from --data")
    os.makedirs(out_dir, exist_ok=True)

    with open(os.path.join(src, "manifest.json"), encoding="utf-8") as handle:
        manifest = json.load(handle)
    if manifest.get("sharded"):
        raise SystemExit("input manifest is already sharded")

    maximum = int(args.max_shard_gib * (1 << 30))
    shard = 0
    pending: list[tuple[dict, dict[str, torch.Tensor]]] = []
    pending_bytes = 0
    rewritten: list[dict] = []
    source_files: list[str] = []

    def emit() -> None:
        nonlocal shard, pending, pending_bytes
        if not pending:
            return
        flush(out_dir, shard, pending)
        offset = 0
        for meta, blob in pending:
            n = int(blob["ids"].shape[0])
            row = dict(meta)
            row["shard"] = shard
            row["offset"] = offset
            row["n"] = n
            rewritten.append(row)
            offset += n
        print(f"shard-{shard:04d}.pt: {len(pending)} sequences, {offset:,} positions, "
              f"{pending_bytes / (1 << 30):.2f} GiB logical", flush=True)
        shard += 1
        pending = []
        pending_bytes = 0

    for meta in manifest.get("sequences", []):
        name = meta["name"]
        path = os.path.join(src, name + ".pt")
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        blob = load_record(path)
        size = sum(tensor_bytes(blob[field]) for field in FIELDS)
        if pending and pending_bytes + size > maximum:
            emit()
        pending.append((meta, blob))
        pending_bytes += size
        source_files.append(path)
    emit()

    out_manifest = dict(manifest)
    out_manifest["sharded"] = True
    out_manifest["shard_count"] = shard
    out_manifest["source_data"] = src
    out_manifest["sequences"] = rewritten
    manifest_path = os.path.join(out_dir, "manifest.json")
    temp = manifest_path + ".tmp"
    with open(temp, "w", encoding="utf-8") as handle:
        json.dump(out_manifest, handle, indent=2)
        handle.write("\n")
    os.replace(temp, manifest_path)

    if args.copy_meta:
        for child in Path(src).iterdir():
            if child.name == "manifest.json" or child.suffix == ".pt" or not child.is_file():
                continue
            shutil.copy2(child, os.path.join(out_dir, child.name))

    if args.delete_source:
        # Manifest and all shard files exist before destructive cleanup.
        for path in source_files:
            os.remove(path)

    print(f"wrote {shard} shards and {len(rewritten)} sequence entries to {out_dir}")


if __name__ == "__main__":
    main()
