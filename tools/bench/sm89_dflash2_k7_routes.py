"""Qualify the RTX 4090 (sm_89) DFlash2 K7 kernel-route candidates.

This is intentionally a local hardware qualification tool, not a production autotuner.  It drives
bench/ops/q8_dflash2_schedule_bench at the four physical K7 extents C1/C2/C4/C8 => T8/T16/T32/T64,
requires the benchmark to report sm=89, retains the raw cold-cache spread, and emits a compact JSON
artifact showing the measured winner versus the currently routed schedule for attention input and
SwiGLU.

The output is evidence for editing the sm_89-only tables.  It never rewrites C++ route tables by
itself: a noisy one-run winner is not a safe production boundary.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import sys

EXTENTS = (8, 16, 32, 64)
CONCURRENCY = {8: 1, 16: 2, 32: 4, 64: 8}
SECTION = re.compile(r"^# gpu=(?P<gpu>.+?) sm=(?P<sm>\d+)\s+(?P<title>q8 dflash2 .+?), cold,")
ROW = re.compile(
    r"^\s*(?P<t>8|16|32|64)\s+.*?\s(?P<public>[0-9]+(?:\.[0-9]+)?)\s+"
    r"(?P<routed>[A-Za-z0-9_.]+)\s+(?P<winner>[A-Za-z0-9_]+)\s*$"
)


def parse_sweep(stdout: str) -> tuple[str | None, dict[str, dict[int, dict]]]:
    sections: dict[str, dict[int, dict]] = {}
    current = None
    gpu_name = None
    for line in stdout.splitlines():
        header = SECTION.match(line)
        if header:
            if header.group("sm") != "89":
                raise RuntimeError(
                    f"sm89 qualification requires RTX Ada compute capability 8.9, got sm={header.group('sm')}"
                )
            gpu_name = header.group("gpu")
            title = header.group("title")
            current = (
                "attn_input"
                if "attn_input" in title
                else "linear_swiglu"
                if "linear_swiglu" in title
                else None
            )
            if current is not None:
                sections[current] = {}
            continue
        row = ROW.match(line)
        if not row or current is None:
            continue
        tokens = int(row.group("t"))
        sections[current][tokens] = {
            "client_concurrency": CONCURRENCY[tokens],
            "columns": tokens,
            "public_op_median_us": float(row.group("public")),
            "routed_schedule": row.group("routed"),
            "measured_winner": row.group("winner"),
        }

    required = {"attn_input", "linear_swiglu"}
    if set(sections) != required or any(set(rows) != set(EXTENTS) for rows in sections.values()):
        raise RuntimeError("incomplete sweep parse")
    return gpu_name, sections


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exe", type=Path, required=True,
                        help="built q8_dflash2_schedule_bench executable")
    parser.add_argument("--out", type=Path, required=True,
                        help="new JSON output path")
    parser.add_argument("--gpu", type=int, default=0,
                        help="physical CUDA_VISIBLE_DEVICES ordinal")
    parser.add_argument("--repeat", type=int, default=31)
    parser.add_argument("--warmup", type=int, default=5)
    args = parser.parse_args(argv)

    if not args.exe.is_file():
        parser.error("--exe must name the built q8_dflash2_schedule_bench")
    if args.out.exists():
        parser.error("--out already exists")
    if args.gpu < 0 or args.repeat < 9 or args.warmup < 1:
        parser.error("gpu must be nonnegative; repeat >=9 and warmup >=1")

    command = [
        str(args.exe.resolve()),
        "--tokens", ",".join(map(str, EXTENTS)),
        "--repeat", str(args.repeat),
        "--warmup", str(args.warmup),
        "--spread",
    ]
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    completed = subprocess.run(command, capture_output=True, text=True, env=env)
    raw_path = args.out.with_suffix(args.out.suffix + ".log")
    raw_path.write_text(completed.stdout + completed.stderr, encoding="utf-8")
    if completed.returncode != 0:
        raise RuntimeError(f"schedule sweep failed with exit code {completed.returncode}; see {raw_path}")

    try:
        gpu_name, sections = parse_sweep(completed.stdout)
    except RuntimeError as error:
        raise RuntimeError(f"{error}; see {raw_path}") from error

    report = {
        "schema_version": 1,
        "artifact_type": "ninfer_sm89_dflash2_k7_route_qualification",
        "gpu": gpu_name,
        "compute_capability": 89,
        "cold_cache": True,
        "repeat": args.repeat,
        "warmup": args.warmup,
        "extents": list(EXTENTS),
        "results": {
            op: [rows[t] for t in EXTENTS]
            for op, rows in sections.items()
        },
        "raw_log": str(raw_path.resolve()),
        "decision_rule": (
            "Use this as candidate evidence only. Retune the sm89-only C++ route table after "
            "the winner clears its own min..p95 spread in repeated runs and the public Op matches "
            "the selected schedule."
        ),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
