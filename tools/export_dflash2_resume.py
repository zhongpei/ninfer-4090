#!/usr/bin/env python3
"""Export a servable DFlash2 checkpoint from train_dflash2.py resume state.

Adapted from TandemLLM/tools/export_resume.py (AGPL-3.0-only).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time

import torch

from tools.dflash2_training.module import load_weights
from tools.train_dflash2 import export

FROZEN_PREFIXES = ("candidate_selector.",)


def merge(base: dict[str, torch.Tensor], trained: dict[str, torch.Tensor]) -> tuple[dict, list[str]]:
    out = dict(base)
    for name, value in trained.items():
        if name not in base:
            raise SystemExit(f"resume state carries unknown tensor {name}")
        if tuple(value.shape) != tuple(base[name].shape):
            raise SystemExit(
                f"{name}: resume shape {tuple(value.shape)} != base {tuple(base[name].shape)}")
        out[name] = value
    from_base = sorted(name for name in base if name not in trained)
    return out, from_base


def sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 22), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--state", required=True)
    ap.add_argument("--base", required=True, help="DFlash2 checkpoint the training run started from")
    ap.add_argument("--out", required=True)
    ap.add_argument("--allow-untrained", action="store_true",
                    help="permit non-selector tensors to fall back to the base checkpoint")
    args = ap.parse_args()

    started = time.perf_counter()
    blob = torch.load(os.path.expanduser(args.state), map_location="cpu",
                      weights_only=False, mmap=True)
    trained = blob.get("weights")
    if not isinstance(trained, dict):
        raise SystemExit("resume state has no weights mapping")

    base = load_weights(os.path.expanduser(args.base), device="cpu")
    weights, from_base = merge(base, trained)
    unexpected = [
        name for name in from_base
        if not any(name.startswith(prefix) for prefix in FROZEN_PREFIXES)
    ]
    if unexpected and not args.allow_untrained:
        raise SystemExit(
            "non-selector tensors are missing from the resume state; "
            f"use --allow-untrained only if intentional: {unexpected[:8]}")

    block = int(blob.get("block") or 0)
    export(weights, os.path.expanduser(args.base), os.path.expanduser(args.out), block=block)
    model_path = os.path.join(os.path.expanduser(args.out), "model.safetensors")
    digest = sha256(model_path)
    manifest = {
        "format": "ninfer-dflash2-trained-v1",
        "source_state": os.path.abspath(os.path.expanduser(args.state)),
        "base": os.path.abspath(os.path.expanduser(args.base)),
        "tag": blob.get("tag"),
        "block": block,
        "step": blob.get("step"),
        "steps_total": blob.get("steps_total"),
        "lr": blob.get("lr"),
        "train": blob.get("train"),
        "best": blob.get("best"),
        "fingerprint": blob.get("fingerprint"),
        "from_base": from_base,
        "sha256": digest,
        "exported": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    with open(os.path.join(os.path.expanduser(args.out), "MANIFEST.json"), "w",
              encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
        handle.write("\n")
    print(f"wrote {args.out}; sha256={digest[:16]} "
          f"in {time.perf_counter() - started:.1f}s")


if __name__ == "__main__":
    main()
