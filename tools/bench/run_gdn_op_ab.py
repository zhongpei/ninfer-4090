"""Cold-L2 GDN micro A/B screen; no 27B artifact and no model upload.

Uses the existing gated_delta_net Op benchmark with CUDA Graph replay.
Op timings are a screening signal, never standalone production qualification.
"""
from __future__ import annotations

import argparse
import csv
from io import StringIO
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys

from tools.bench.run_upstream_route_matrix import CLEAN_ENV, streaming_sha256, summary_values


def positive_list(value: str) -> list[int]:
    try:
        numbers = [int(x) for x in value.split(",")]
    except ValueError as exc:
        raise ValueError("expected comma separated integers") from exc
    if not numbers or any(x <= 0 for x in numbers) or len(numbers) != len(set(numbers)):
        raise ValueError("shapes must be positive and distinct")
    return numbers


def parse_row(stdout: str, heads: int, tokens: int) -> dict:
    reader = csv.DictReader(StringIO(stdout))
    required = {"state_form", "normalization", "value_heads", "tokens", "batch",
                "dtype", "state_dim", "qk_heads", "execution", "cache",
                "median_us", "p95_us", "graph_nodes"}
    if not required <= set(reader.fieldnames or []):
        raise ValueError("missing GDN microbenchmark CSV columns")
    rows = list(reader)
    if len(rows) != 1:
        raise ValueError("exactly one running-state row is required")
    r = rows[0]
    if (r["state_form"] != "running" or r["normalization"] != "fused"
            or r["dtype"] != "BF16" or int(r["state_dim"]) != 128
            or int(r["qk_heads"]) != 16
            or r["execution"] != "cuda_graph" or r["cache"] != "cold_l2"
            or int(r["value_heads"]) != heads or int(r["tokens"]) != tokens
            or int(r["batch"]) != 1 or int(r["graph_nodes"]) < 1):
        raise ValueError("GDN geometry or cold-L2 graph contract changed")
    median, tail = float(r["median_us"]), float(r["p95_us"])
    if (not math.isfinite(median) or not math.isfinite(tail)
            or median <= 0 or tail < median):
        raise ValueError("invalid GDN time distribution")
    return {"median_us": median, "p95_us": tail, "graph_nodes": int(r["graph_nodes"])}


def compare(rows: list[dict], pairs: int, attributed_fraction: float | None = None) -> dict:
    a = {r["pair"]: r["data"] for r in rows if r["arm"] == "baseline"}
    b = {r["pair"]: r["data"] for r in rows if r["arm"] == "candidate"}
    if len(a) != pairs or len(b) != pairs:
        raise ValueError("missing or duplicate paired measurements")
    gain = [100 * (a[i]["median_us"] / b[i]["median_us"] - 1) for i in range(pairs)]
    tail = [100 * (b[i]["p95_us"] / a[i]["p95_us"] - 1) for i in range(pairs)]
    base_times = [a[i]["median_us"] for i in range(pairs)]
    candidate_times = [b[i]["median_us"] for i in range(pairs)]
    projected = None
    if attributed_fraction is not None:
        ratio = statistics.median(base_times) / statistics.median(candidate_times)
        projected = 100 * (1 / (1 - attributed_fraction + attributed_fraction / ratio) - 1)
    return {
        "baseline_us": summary_values(base_times),
        "candidate_us": summary_values(candidate_times),
        "paired_speedup_pct": summary_values(gain),
        "paired_p95_regression_pct": summary_values(tail),
        "estimated_model_gain_pct": projected,
        "performance_screen": statistics.median(gain) >= 2 and
                              summary_values(gain)["p05"] >= -5 and
                              statistics.median(tail) <= 5,
        "production_qualified": False,
    }


def run_arm(args, heads: int, tokens: int, pair: int, order: int, arm: str) -> dict:
    tag = f"h{heads}-t{tokens}-p{pair:02d}-{order}-{arm}"
    folder = args.out / tag
    folder.mkdir()
    command = [
        str(args.exe.resolve()), "--running", "--qk-heads", "16",
        "--value-heads", str(heads), "--tokens", str(tokens),
        "--warmup", str(args.warmup), "--repeat", str(args.repeat),
        "--flush-mib", str(args.flush_mib), "--csv",
    ]
    env = os.environ.copy()
    for key in CLEAN_ENV:
        env.pop(key, None)
    env.update({
        "NINFER_DEVICE_ROUTE_MODE": "off",
        "NINFER_GDN_TWO_STAGE": "0",
        "NINFER_GDN_TWO_STAGE_NUMERICS": "exact",
        "NINFER_GDN_EXACT_PREFETCH": "1" if arm == "candidate" else "0",
    })
    if args.cuda_visible_devices is not None:
        env["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices
    (folder / "invocation.json").write_text(json.dumps({
        "command": command, "route_env": {k: env[k] for k in CLEAN_ENV if k in env},
        "cuda_visible_devices": env.get("CUDA_VISIBLE_DEVICES"),
    }, indent=2) + "\n")
    try:
        process = subprocess.run(command, env=env, capture_output=True, text=True,
                                 timeout=args.timeout, check=False)
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(f"benchmark timed out: {tag}") from exc
    (folder / "stdout.csv").write_text(process.stdout)
    (folder / "stderr.log").write_text(process.stderr)
    if process.returncode:
        raise RuntimeError(f"benchmark failed {tag} ({process.returncode})")
    result = {"heads": heads, "tokens": tokens, "pair": pair, "order": order,
              "arm": arm, "data": parse_row(process.stdout, heads, tokens)}
    (folder / "record.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exe", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--heads", default="32,48")
    parser.add_argument("--tokens", default="32,64,65,128,256,1024")
    parser.add_argument("--pairs", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeat", type=int, default=50)
    parser.add_argument("--flush-mib", type=int, default=256)
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--cuda-visible-devices")
    parser.add_argument("--attributed-fraction", type=float,
                        help="measured fraction of request time spent in GDN (0..1)")
    args = parser.parse_args(argv)
    if not args.exe.is_file() or args.out.exists():
        parser.error("executable must exist and --out directory must not exist")
    if args.pairs < 3 or args.warmup < 0 or args.repeat <= 0 or args.flush_mib <= 0 or args.timeout <= 0:
        parser.error("invalid benchmark settings")
    if args.attributed_fraction is not None and (
        not math.isfinite(args.attributed_fraction) or
        not 0 <= args.attributed_fraction <= 1
    ):
        parser.error("--attributed-fraction must be a measured value in [0,1]")
    if args.cuda_visible_devices is not None and (
        not args.cuda_visible_devices or
        any(not x.isdigit() for x in args.cuda_visible_devices.split(","))
    ):
        parser.error("--cuda-visible-devices expects numeric GPU ordinals")
    try:
        heads, tokens = positive_list(args.heads), positive_list(args.tokens)
    except ValueError as exc:
        parser.error(str(exc))
    if any(h % 16 for h in heads):
        parser.error("value heads must be a multiple of 16")
    args.out.mkdir(parents=True)
    records = []
    try:
        with (args.out / "records.jsonl").open("w") as journal:
            for h in heads:
                for t in tokens:
                    for pair in range(args.pairs):
                        order = ("baseline", "candidate") if pair % 2 == 0 else (
                            "candidate", "baseline")
                        for index, arm in enumerate(order):
                            record = run_arm(args, h, t, pair, index, arm)
                            records.append(record)
                            journal.write(json.dumps(record) + "\n")
                            journal.flush()
        comparisons = []
        for h in heads:
            for t in tokens:
                subset = [r for r in records if r["heads"] == h and r["tokens"] == t]
                comparisons.append({
                    "heads": h, "tokens": t,
                    **compare(subset, args.pairs, args.attributed_fraction),
                })
        report = {
            "schema": "ninfer.gdn-op-ab.v1",
            "objective": "cold-L2 CUDA Graph latency in microseconds",
            "executable": str(args.exe.resolve()),
            "exe_sha256": streaming_sha256(args.exe),
            "comparisons": comparisons,
            "numerical_gate": "not_run", "production_qualified": False,
        }
        (args.out / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
        table = ["# GDN exact prefetch Op A/B screen (no model loading)", "",
                 "| H_v | Tokens | Base median (µs) | Candidate median (µs) | Paired gain | P05 gain | P95 change | Screen |",
                 "|---:|---:|---:|---:|---:|---:|---:|---|"]
        for row in comparisons:
            table.append(
                f"| {row['heads']} | {row['tokens']} | "
                f"{row['baseline_us']['median']:.3f} | {row['candidate_us']['median']:.3f} "
                f"| {row['paired_speedup_pct']['median']:+.2f}% | "
                f"{row['paired_speedup_pct']['p05']:+.2f}% | "
                f"{row['paired_p95_regression_pct']['median']:+.2f}% | "
                f"{'pass' if row['performance_screen'] else 'reject'} |"
            )
        table += ["", "This is a cheap cold-L2 Op screen; it does NOT approve production routing.",
                  "The measured resident 27B workload remains the end-to-end authority.", ""]
        (args.out / "summary.md").write_text("\n".join(table), encoding="utf-8")
        print(f"Wrote {args.out / 'summary.json'}")
        return 0
    except Exception as exc:
        (args.out / "ERROR.txt").write_text(f"{type(exc).__name__}: {exc}\n")
        raise


if __name__ == "__main__":
    sys.exit(main())
