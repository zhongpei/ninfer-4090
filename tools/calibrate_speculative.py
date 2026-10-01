#!/usr/bin/env python3
"""Calibrate NInfer Stair/Tree-Stair costs from a real-text sweep CSV.

For each fixed rung, infer mean round latency from:
    committed_tokens_per_round = 1 + accepted / rounds
    round_ms = committed_tokens_per_round / decode_tok_s * 1000

When --draft-cost-ms is supplied, that common wide-drafter cost is subtracted before normalizing
the verify staircase and emitted separately in the same relative units. Without it, the measured
total round cost is used directly and --spec-stair-draft-cost=0 is recommended.

For the 24GB b16 Tree-Stair path, use dflash2-tree-realtext.ps1 rows:
    tree15_n3, tree15_n7, tree15_n11, tree15_n15
so every rung contains the same wide DFlash2 proposal cost.
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path


def parse_csv(path: Path):
    rows = []
    with path.open(newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            rows.append(row)
    return rows


def as_float(row, key):
    value = row.get(key, "")
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def rung_latency(rows, label: str):
    values = []
    for row in rows:
        if row.get("config") != label:
            continue
        speed = as_float(row, "decode_tok_s")
        rounds = as_float(row, "rounds")
        accepted = as_float(row, "accepted")
        if speed is None or rounds is None or accepted is None or speed <= 0 or rounds <= 0:
            continue
        committed_per_round = 1.0 + accepted / rounds
        values.append(committed_per_round / speed * 1000.0)
    if not values:
        raise ValueError(f"no valid measurements for config {label!r}")
    return {
        "samples": len(values),
        "median_round_ms": statistics.median(values),
        "min_round_ms": min(values),
        "max_round_ms": max(values),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("csv")
    ap.add_argument("--widths", default="3,7,11,15")
    ap.add_argument(
        "--labels",
        default="tree15_n3,tree15_n7,tree15_n11,tree15_n15",
        help="comma-separated sweep config names corresponding to --widths",
    )
    ap.add_argument(
        "--draft-cost-ms",
        type=float,
        default=0.0,
        help="measured common wide-drafter latency; zero treats measured round cost as one table",
    )
    ap.add_argument("--gpu", default="RTX 4090")
    ap.add_argument("--kv", default="int8")
    ap.add_argument("--context", default="8192")
    ap.add_argument("--model", default="")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    widths = [int(x) for x in args.widths.split(",") if x]
    labels = [x for x in args.labels.split(",") if x]
    if len(widths) != 4 or len(labels) != 4:
        raise SystemExit("NInfer Stair runtime currently requires exactly four widths/labels")
    if widths != sorted(set(widths)) or widths[0] <= 0 or widths[-1] > 15:
        raise SystemExit("--widths must be four strictly increasing values in [1,15]")
    if args.draft_cost_ms < 0:
        raise SystemExit("--draft-cost-ms must be nonnegative")

    rows = parse_csv(Path(args.csv))
    measurements = {str(w): rung_latency(rows, label) for w, label in zip(widths, labels)}
    total_ms = [measurements[str(w)]["median_round_ms"] for w in widths]

    if args.draft_cost_ms:
        verify_ms = [max(1.0e-6, x - args.draft_cost_ms) for x in total_ms]
        baseline = verify_ms[0]
        costs = [x / baseline for x in verify_ms]
        draft_cost = args.draft_cost_ms / baseline
        cost_kind = "verify_after_common_draft_subtraction"
    else:
        baseline = total_ms[0]
        costs = [x / baseline for x in total_ms]
        draft_cost = 0.0
        cost_kind = "total_round_cost"

    profile = {
        "version": 1,
        "source_csv": str(Path(args.csv)),
        "gpu": args.gpu,
        "kv": args.kv,
        "context": args.context,
        "model": args.model,
        "widths": widths,
        "labels": labels,
        "measurements": measurements,
        "draft_cost_ms": args.draft_cost_ms,
        "cost_kind": cost_kind,
        "verify_costs": costs,
        "draft_cost": draft_cost,
        "cli": {
            "spec_stair_widths": ",".join(str(x) for x in widths),
            "spec_stair_costs": ",".join(f"{x:.6f}" for x in costs),
            "spec_stair_draft_cost": round(draft_cost, 6),
        },
    }

    print(
        "--spec-stair-widths "
        + profile["cli"]["spec_stair_widths"]
        + " --spec-stair-costs "
        + profile["cli"]["spec_stair_costs"]
        + " --spec-stair-draft-cost "
        + f"{draft_cost:.6f}"
    )
    for width, total, cost in zip(widths, total_ms, costs):
        print(f"K{width}: median_round_ms={total:.3f} normalized_cost={cost:.6f}")

    if args.out:
        with Path(args.out).open("w", encoding="utf-8") as f:
            json.dump(profile, f, indent=2)
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
