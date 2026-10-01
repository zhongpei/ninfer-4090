#!/usr/bin/env python3
"""Convert a NInfer DFlash teacher trace into the dataset consumed by train.py.

The trace is emitted by --dflash-teacher-dump from the *actual deployed .ninfer target*. Runtime
stores BF16 full-vocabulary logits to keep collection simple and exact; this converter computes the
target top-K distribution offline on CPU and stitches committed speculative rounds by request id and
execution frontier.

This intentionally accepts only the linear chain trace produced by the current runtime collector.
Tree/lookup takeover are rejected at startup while tracing so branch rows after a mismatch cannot
enter training data.
"""
from __future__ import annotations

import argparse
import json
import random
import struct
from dataclasses import dataclass
from pathlib import Path

import torch


HEADER = struct.Struct("<8s6I")
RECORD = struct.Struct("<IQII")
MAGIC = b"NIFTRC1\0"
RECORD_MAGIC = 0x31434552


@dataclass
class Segment:
    request_id: int
    frontier: int
    ids: torch.Tensor
    fused: torch.Tensor
    label: torch.Tensor
    top_ids: torch.Tensor
    top_lp: torch.Tensor

    @property
    def n(self) -> int:
        return int(self.ids.numel())


def read_exact(f, n: int) -> bytes:
    data = f.read(n)
    if len(data) != n:
        raise EOFError(f"truncated teacher trace: wanted {n} bytes, got {len(data)}")
    return data


def tensor_from_bytes(data: bytes, dtype: torch.dtype, shape) -> torch.Tensor:
    # bytearray gives torch a writable buffer and clone detaches the returned tensor from it.
    return torch.frombuffer(bytearray(data), dtype=dtype).clone().reshape(shape)


def parse_trace(path: Path, topk: int) -> tuple[dict, dict[int, list[Segment]]]:
    requests: dict[int, list[Segment]] = {}
    with path.open("rb") as f:
        raw = read_exact(f, HEADER.size)
        magic, version, token_domain, feature_rows, physical_rows, physical_width, reserved = (
            HEADER.unpack(raw)
        )
        if magic != MAGIC or version != 1 or reserved != 0:
            raise ValueError("unsupported NInfer teacher trace header")
        if not (0 < token_domain <= physical_rows) or physical_width <= 0 or feature_rows <= 0:
            raise ValueError("invalid NInfer teacher trace geometry")
        k = min(topk, token_domain)

        while True:
            marker = f.read(4)
            if not marker:
                break
            if len(marker) != 4:
                raise EOFError("truncated teacher record marker")
            rec_magic = struct.unpack("<I", marker)[0]
            if rec_magic != RECORD_MAGIC:
                raise ValueError(f"invalid teacher record marker 0x{rec_magic:08x}")
            request_id, frontier, rows = struct.unpack("<QII", read_exact(f, 16))
            if rows == 0 or rows > physical_width:
                raise ValueError(f"invalid teacher record row count {rows}")

            ids = tensor_from_bytes(read_exact(f, rows * 4), torch.int32, (rows,)).long()
            labels = tensor_from_bytes(read_exact(f, rows * 4), torch.int32, (rows,)).long()
            fused = tensor_from_bytes(
                read_exact(f, feature_rows * rows * 2),
                torch.bfloat16,
                (rows, feature_rows),
            )
            logits = tensor_from_bytes(
                read_exact(f, physical_rows * rows * 2),
                torch.bfloat16,
                (rows, physical_rows),
            )
            valid_logits = logits[:, :token_domain].float()
            logp = torch.log_softmax(valid_logits, dim=-1)
            top_lp, top_ids = torch.topk(logp, k, dim=-1)
            del logits, valid_logits, logp

            requests.setdefault(request_id, []).append(
                Segment(
                    request_id=request_id,
                    frontier=frontier,
                    ids=ids,
                    fused=fused,
                    label=labels,
                    top_ids=top_ids.to(torch.int32),
                    top_lp=top_lp.to(torch.float16),
                )
            )

    meta = {
        "version": 1,
        "token_domain": token_domain,
        "feature_rows": feature_rows,
        "physical_rows": physical_rows,
        "physical_width": physical_width,
    }
    return meta, requests


def trim_segment(seg: Segment, begin: int) -> Segment:
    return Segment(
        request_id=seg.request_id,
        frontier=seg.frontier + begin,
        ids=seg.ids[begin:],
        fused=seg.fused[begin:],
        label=seg.label[begin:],
        top_ids=seg.top_ids[begin:],
        top_lp=seg.top_lp[begin:],
    )


def stitch(request_id: int, segments: list[Segment]) -> list[dict]:
    segments = sorted(segments, key=lambda x: (x.frontier, x.n))
    sequences: list[dict] = []
    parts: list[Segment] = []
    expected: int | None = None

    def flush():
        nonlocal parts, expected
        if not parts:
            return
        sequences.append(
            {
                "request_id": request_id,
                "frontier": parts[0].frontier,
                "ids": torch.cat([x.ids for x in parts]),
                "fused": torch.cat([x.fused for x in parts], dim=0),
                "label": torch.cat([x.label for x in parts]),
                "top_ids": torch.cat([x.top_ids for x in parts], dim=0),
                "top_lp": torch.cat([x.top_lp for x in parts], dim=0),
            }
        )
        parts = []
        expected = None

    for original in segments:
        seg = original
        if expected is None:
            parts = [seg]
            expected = seg.frontier + seg.n
            continue

        if seg.frontier < expected:
            overlap = expected - seg.frontier
            if overlap >= seg.n:
                continue
            seg = trim_segment(seg, overlap)

        if seg.frontier != expected:
            flush()
            parts = [seg]
            expected = seg.frontier + seg.n
            continue

        parts.append(seg)
        expected += seg.n

    flush()
    return sequences


def parse_taps(text: str) -> list[int]:
    taps = [int(x) for x in text.split(",") if x.strip()]
    if not taps or taps != sorted(set(taps)):
        raise ValueError("--taps must be a strictly increasing comma-separated list")
    return taps


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("trace")
    ap.add_argument("--out", required=True)
    ap.add_argument("--topk", type=int, default=64)
    ap.add_argument("--taps", default="5,19,33,47,61")
    ap.add_argument("--min-len", type=int, default=16)
    ap.add_argument("--holdout-fraction", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=20261001)
    args = ap.parse_args()
    if args.topk <= 0 or args.min_len < 2:
        raise SystemExit("--topk must be positive and --min-len >= 2")
    if not 0.0 <= args.holdout_fraction < 1.0:
        raise SystemExit("--holdout-fraction must be in [0,1)")

    taps = parse_taps(args.taps)
    meta, requests = parse_trace(Path(args.trace), args.topk)
    hidden = meta["feature_rows"] // len(taps)
    if hidden * len(taps) != meta["feature_rows"]:
        raise SystemExit(
            f"feature_rows={meta['feature_rows']} is not divisible by {len(taps)} taps"
        )

    stitched: list[dict] = []
    for request_id, segments in requests.items():
        stitched.extend(stitch(request_id, segments))
    stitched = [x for x in stitched if int(x["ids"].numel()) >= args.min_len]
    stitched.sort(key=lambda x: (x["request_id"], x["frontier"]))
    if not stitched:
        raise SystemExit("no stitched teacher sequence satisfies --min-len")

    rng = random.Random(args.seed)
    held_count = int(round(len(stitched) * args.holdout_fraction))
    held_indices = set(rng.sample(range(len(stitched)), held_count)) if held_count else set()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = []
    for index, item in enumerate(stitched):
        name = f"ninfer-{item['request_id']:08d}-{item['frontier']:09d}"
        split = "heldout" if index in held_indices else "train"
        blob = {
            "ids": item["ids"].to(torch.int32),
            "fused": item["fused"].to(torch.bfloat16),
            "label": item["label"].to(torch.int32),
            "top_ids": item["top_ids"].to(torch.int32),
            "top_lp": item["top_lp"].to(torch.float16),
            "name": name,
            "kind": "gen",
            "topic": "ninfer",
            "split": split,
            "gen_start": 0,
            "target_layer_ids": taps,
            "source": "ninfer-trace",
        }
        torch.save(blob, out_dir / f"{name}.pt")
        manifest.append(
            {
                "name": name,
                "kind": "gen",
                "topic": "ninfer",
                "split": split,
                "n": int(blob["ids"].numel()),
                "gen_start": 0,
                "request_id": int(item["request_id"]),
                "frontier": int(item["frontier"]),
            }
        )

    with (out_dir / "manifest.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "version": 1,
                "source": "ninfer-trace",
                "trace": str(Path(args.trace)),
                "target_layer_ids": taps,
                "hidden_size": hidden,
                "topk": min(args.topk, meta["token_domain"]),
                "token_domain": meta["token_domain"],
                "sequences": manifest,
            },
            f,
            indent=2,
        )
    print(
        f"converted {len(manifest)} stitched sequences "
        f"({sum(x['n'] for x in manifest)} rows) -> {out_dir}",
        flush=True,
    )


if __name__ == "__main__":
    main()
