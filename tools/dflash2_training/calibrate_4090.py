#!/usr/bin/env python3
"""Convert forced-rung RTX 4090 sweep CSV into a Stair cost profile.

The companion PowerShell sweep keeps the DFlash2 neural proposal fixed at K15 and forces exactly
one target-verification rung per arm. Because the wide proposal cost is common to every arm, this
tool treats measured total seconds/round as the router's effective cost and recommends
--spec-stair-draft-cost 0. That avoids double-counting the common drafter cost.

It also enforces the most important correctness gate for greedy A/B: all successful arms must emit
the same SHA256 for a repetition. A mismatch stops calibration instead of producing a fast but
non-equivalent profile.
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path


WIDTHS = (3, 7, 11, 15)


def load_rows(paths: list[str]) -> list[dict]:
    rows: list[dict] = []
    for name in paths:
        with open(name, newline="", encoding="utf-8-sig") as f:
            reader = csv.DictReader(line for line in f if not line.startswith("    "))
            for row in reader:
                if not row.get("config") or row.get("decode_tok_s") == "FAILED":
                    continue
                try:
                    row["_rep"] = int(row["rep"])
                    row["_seconds_per_round"] = float(row["seconds_per_round"])
                    row["_decode_tok_s"] = float(row["decode_tok_s"])
                except (KeyError, TypeError, ValueError):
                    continue
                rows.append(row)
    return rows


def verify_hashes(rows: list[dict]) -> None:
    by_rep: dict[int, set[str]] = defaultdict(set)
    for row in rows:
        digest = (row.get("sha256") or "").strip().upper()
        if digest:
            by_rep[row["_rep"]].add(digest)
    bad = {rep: hashes for rep, hashes in by_rep.items() if len(hashes) > 1}
    if bad:
        detail = ", ".join(f"rep {rep}: {len(hashes)} hashes" for rep, hashes in sorted(bad.items()))
        raise SystemExit("greedy output hash mismatch across forced Stair rungs: " + detail)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("csv", nargs="+", help="CSV output from dflash2-stair-cost-calibration.ps1")
    ap.add_argument("--out", default="", help="optional JSON profile output")
    ap.add_argument("--min-reps", type=int, default=3)
    args = ap.parse_args()

    rows = load_rows(args.csv)
    if not rows:
        raise SystemExit("no successful calibration rows found")
    verify_hashes(rows)

    medians: dict[int, float] = {}
    speeds: dict[int, float] = {}
    counts: dict[int, int] = {}
    for width in WIDTHS:
        label = f"rung{width}"
        values = [r["_seconds_per_round"] for r in rows
                  if r["config"] == label and r["_seconds_per_round"] > 0]
        toks = [r["_decode_tok_s"] for r in rows
                if r["config"] == label and r["_decode_tok_s"] > 0]
        if len(values) < args.min_reps:
            raise SystemExit(f"{label}: need at least {args.min_reps} successful reps, got {len(values)}")
        medians[width] = statistics.median(values)
        speeds[width] = statistics.median(toks)
        counts[width] = len(values)

    baseline = medians[WIDTHS[0]]
    costs = [medians[w] / baseline for w in WIDTHS]
    profile = {
        "version": 1,
        "hardware": "RTX 4090 / sm_89 calibration",
        "method": "forced Stair rung, K15 drafter fixed, median total seconds/round",
        "widths": list(WIDTHS),
        "verify_costs": costs,
        "draft_cost": 0.0,
        "median_seconds_per_round": [medians[w] for w in WIDTHS],
        "median_decode_tok_s": [speeds[w] for w in WIDTHS],
        "repetitions": [counts[w] for w in WIDTHS],
        "greedy_hash_gate": "passed",
        "cli": (
            "--spec-router stair "
            f"--spec-stair-widths {','.join(map(str, WIDTHS))} "
            f"--spec-stair-costs {','.join(f'{x:.6f}' for x in costs)} "
            "--spec-stair-draft-cost 0"
        ),
    }

    print(json.dumps(profile, indent=2))
    if args.out:
        path = Path(args.out)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(profile, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
