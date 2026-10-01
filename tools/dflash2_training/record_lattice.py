#!/usr/bin/env python3
"""Record DFlash2 candidate lattices from held-out teacher data for offline tree sweeps."""
from __future__ import annotations

import argparse
import json
import random

import torch
import torch.nn.functional as F

from tools.dflash2_training.model import (
    DFlash2Module, load_config, load_weight_tensors, target_embedding_and_head,
)
from tools.dflash2_training.train import load_data, run_blocks


@torch.inference_mode()
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--drafter", required=True)
    ap.add_argument("--target", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--block", type=int, default=16)
    ap.add_argument("--samples", type=int, default=256)
    ap.add_argument("--per-sequence", type=int, default=8)
    ap.add_argument("--split", default="heldout")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    cfg, snapshot = load_config(args.drafter)
    module = DFlash2Module(
        cfg, load_weight_tensors(snapshot, dtype=torch.bfloat16, device=args.device)
    ).to(args.device)
    embed, head = target_embedding_and_head(args.target, args.device)
    data = [s for s in load_data(args.data, "cpu", args.device)
            if s.split == args.split]
    if not data:
        raise SystemExit(f"no {args.split} samples")

    rng = random.Random(args.seed)
    written = 0
    with open(args.out, "w", encoding="utf-8") as out:
        for sample in data:
            lo = max(1, sample.gen_start - 1) if sample.kind == "gen" else 1
            hi = len(sample) - args.block
            if hi <= lo:
                continue
            choices = list(range(lo, hi + 1))
            rng.shuffle(choices)
            for anchor in choices[:args.per_sequence]:
                anchors = torch.tensor([anchor], dtype=torch.long, device=args.device)
                pred = run_blocks(module, embed, sample, anchors, args.device, args.block)[0]
                logits = F.linear(pred.to(head.dtype), head).float()
                cand, unary = module.unary_candidates(logits)
                scores = module.lattice(pred, cand, unary, int(sample.ids[anchor]))
                target = sample.label[anchor:anchor + args.block - 1].tolist()
                out.write(json.dumps({
                    "sample": sample.name,
                    "anchor_position": anchor,
                    "anchor": int(sample.ids[anchor]),
                    "candidates": cand.cpu().tolist(),
                    "scores": scores.cpu().tolist(),
                    "target_tokens": [int(x) for x in target],
                }) + "\n")
                written += 1
                if written >= args.samples:
                    print(f"wrote {written} lattice rows to {args.out}")
                    return
    print(f"wrote {written} lattice rows to {args.out}")


if __name__ == "__main__":
    main()
