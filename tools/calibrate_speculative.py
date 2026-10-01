#!/usr/bin/env python3
"""Calibrate NInfer Stair/Tree-Stair costs from real CLI runs.

The wide drafter stays fixed. For each rung this tool makes its cost artificially dominant in the
router, measures decode time / speculative rounds, takes the median, normalizes to the cheapest
rung, and writes a JSON profile consumable by --spec-profile.

No CUDA profiler is required and the profile measures the complete round economics seen by the
runtime (wide draft + target verify + commit), so draft_cost is written as 0 by default.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import subprocess
import sys
from pathlib import Path


RUNG_WIDTHS = (3, 7, 11, 15)


def metric(text: str, label: str) -> str:
    pattern = rf"summary\s+{re.escape(label)}\s+([^\r\n]+)"
    match = re.search(pattern, text)
    if not match:
        raise RuntimeError(f"missing CLI summary metric: {label}")
    return match.group(1).strip()


def parse_rate(value: str) -> float:
    match = re.search(r"([0-9]+(?:\.[0-9]+)?)\s*tok/s", value)
    if not match:
        raise RuntimeError(f"invalid rate: {value!r}")
    return float(match.group(1))


def parse_int(value: str) -> int:
    match = re.search(r"([0-9]+)", value)
    if not match:
        raise RuntimeError(f"invalid integer metric: {value!r}")
    return int(match.group(1))


def forced_costs(index: int) -> str:
    values = ["1000000"] * len(RUNG_WIDTHS)
    values[index] = "1"
    return ",".join(values)


def one_run(args, prompt: str, rung_index: int, rep: int) -> dict:
    width = RUNG_WIDTHS[rung_index]
    command = [
        args.binary,
        args.model,
        "--prompt", prompt,
        "--max-new", str(args.max_new),
        "--max-context", str(args.max_context),
        "--kv-dtype", args.kv_dtype,
        "--spec", "dflash2",
        "--draft-tokens", "15",
        "--spec-router", "stair",
        "--spec-stair-widths", ",".join(map(str, RUNG_WIDTHS)),
        "--spec-stair-costs", forced_costs(rung_index),
        "--spec-stair-draft-cost", "0",
        "--spec-stair-prior-weight", "0",
        "--spec-stair-warmup", "0",
        "--spec-stair-probe-period", "0",
        "--spec-stair-margin", "0",
        "--greedy",
        "--no-thinking",
    ]
    if args.no_cuda_graph:
        command.append("--no-cuda-graph")
    if args.tree:
        command += [
            "--spec-tree", "lattice",
            "--spec-tree-nodes", "15",
            "--spec-tree-spine", str(args.tree_spine),
        ]
    if args.extra:
        command += args.extra

    proc = subprocess.run(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if proc.returncode != 0:
        raise RuntimeError(
            f"rung={width} rep={rep} failed ({proc.returncode})\n"
            + "\n".join(proc.stderr.splitlines()[-20:])
        )
    stderr = proc.stderr
    generated = parse_int(metric(stderr, "generated tokens"))
    speed = parse_rate(metric(stderr, "decode speed"))
    backend = "dflash2"
    rounds = parse_int(metric(stderr, f"{backend} rounds"))
    decoded = max(0, generated - 1)
    if decoded == 0 or speed <= 0 or rounds <= 0:
        raise RuntimeError(
            f"invalid calibration run: generated={generated} speed={speed} rounds={rounds}"
        )
    decode_seconds = decoded / speed
    round_ms = 1000.0 * decode_seconds / rounds

    tree_rounds = 0
    if args.tree:
        try:
            tree_rounds = parse_int(metric(stderr, "tree rounds"))
        except RuntimeError:
            tree_rounds = 0

    return {
        "width": width,
        "rep": rep,
        "generated": generated,
        "decode_tok_s": speed,
        "rounds": rounds,
        "tree_rounds": tree_rounds,
        "round_ms": round_ms,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--binary", default="./build-ninja/apps/ninfer")
    ap.add_argument("--model", required=True)
    source = ap.add_mutually_exclusive_group(required=True)
    source.add_argument("--prompt")
    source.add_argument("--prompt-file")
    ap.add_argument("--out", required=True)
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--max-new", type=int, default=768)
    ap.add_argument("--max-context", type=int, default=8192)
    ap.add_argument("--kv-dtype", default="int8")
    ap.add_argument("--tree", action="store_true")
    ap.add_argument("--tree-spine", type=int, default=7)
    ap.add_argument("--no-cuda-graph", action="store_true")
    ap.add_argument("--extra", action="append", default=[],
                    help="one additional CLI argument; repeat for flag/value pairs")
    args = ap.parse_args()

    if args.reps < 1:
        raise SystemExit("--reps must be positive")
    prompt = args.prompt
    if args.prompt_file:
        prompt = Path(args.prompt_file).read_text(encoding="utf-8")
    assert prompt is not None

    runs: list[dict] = []
    medians: list[float] = []
    for index, width in enumerate(RUNG_WIDTHS):
        samples = []
        for rep in range(1, args.reps + 1):
            row = one_run(args, prompt, index, rep)
            runs.append(row)
            samples.append(row["round_ms"])
            print(
                f"width={width:2d} rep={rep} round={row['round_ms']:.3f} ms "
                f"decode={row['decode_tok_s']:.2f} tok/s rounds={row['rounds']}",
                flush=True,
            )
        median = statistics.median(samples)
        medians.append(median)
        print(f"width={width:2d} median={median:.3f} ms", flush=True)

    baseline = min(medians)
    relative = [value / baseline for value in medians]
    profile = {
        "version": 1,
        "kind": "ninfer-tree-stair" if args.tree else "ninfer-chain-stair",
        "model": os.path.abspath(args.model),
        "kv_dtype": args.kv_dtype,
        "max_context": args.max_context,
        "draft_tokens": 15,
        "widths": list(RUNG_WIDTHS),
        "verify_costs": relative,
        # round_ms already includes the one wide draft; avoid double-counting it in the router.
        "draft_cost": 0.0,
        "round_ms": medians,
        "repetitions": args.reps,
        "runs": runs,
    }
    Path(args.out).write_text(json.dumps(profile, indent=2) + "\n", encoding="utf-8")
    print("profile:", args.out)
    print("--spec-profile", args.out)


if __name__ == "__main__":
    main()
