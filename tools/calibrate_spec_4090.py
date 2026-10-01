#!/usr/bin/env python3
"""Derive a 24GB Tree-Stair verify-cost table from dflash2-tree-realtext.ps1 CSV output.

The calibration deliberately compares b16_tree3/7/11/15: the neural draft cost is therefore held
constant and only the target-tree budget changes.  The reported round latency is reconstructed as

    generated_tokens / decode_tok_s / speculative_rounds

and normalized to the smallest measured tree budget.  It is a served-cost table (verify + fixed
round overhead), which is exactly the quantity the runtime router needs for A/B selection.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path


def rows(path: Path):
    with path.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(line for line in f if line.count(",") >= 5)
        for row in reader:
            if not row.get("config") or row.get("decode_tok_s") in ("", "FAILED", None):
                continue
            yield row


def number(row, name):
    try:
        return float(row[name])
    except (KeyError, TypeError, ValueError):
        return math.nan


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("csv", nargs="+", type=Path)
    ap.add_argument("--widths", default="3,7,11,15")
    ap.add_argument("--json", type=Path, default=None)
    args = ap.parse_args()

    widths = [int(x) for x in args.widths.split(",") if x]
    samples = defaultdict(list)
    hashes = defaultdict(set)
    for path in args.csv:
        for row in rows(path):
            label = row["config"]
            for width in widths:
                if label != f"b16_tree{width}":
                    continue
                speed = number(row, "decode_tok_s")
                generated = number(row, "generated")
                rounds = number(row, "rounds")
                if speed > 0 and generated > 0 and rounds > 0:
                    samples[width].append(1000.0 * generated / speed / rounds)
                if row.get("sha256"):
                    hashes[width].add(row["sha256"])

    missing = [w for w in widths if not samples[w]]
    if missing:
        raise SystemExit(
            "missing fixed-b16 tree rows for widths: " + ",".join(map(str, missing)))

    medians = {w: statistics.median(samples[w]) for w in widths}
    base = medians[widths[0]]
    relative = {w: medians[w] / base for w in widths}
    payload = {
        "schema": "ninfer-spec-tree-stair-v1",
        "widths": widths,
        "round_ms_median": [medians[w] for w in widths],
        "relative_costs": [relative[w] for w in widths],
        "samples": [len(samples[w]) for w in widths],
        "output_hash_variants": [len(hashes[w]) for w in widths],
        "cli": {
            "spec_stair_widths": ",".join(map(str, widths)),
            "spec_stair_costs": ",".join(f"{relative[w]:.6f}" for w in widths),
        },
    }
    print(json.dumps(payload, indent=2))
    print("\nCLI:")
    print("  --spec-stair-widths " + payload["cli"]["spec_stair_widths"])
    print("  --spec-stair-costs " + payload["cli"]["spec_stair_costs"])
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
