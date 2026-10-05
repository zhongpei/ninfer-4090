#!/usr/bin/env python3
"""Run long-history KV perplexity across storage formats and compare against INT8.

Each arm invokes ninfer-perplexity in fixed-depth mode. For depth D and tail N the evaluator
builds the real prefix [0,D) into Main KV and scores only [D,D+N), so the reported delta isolates
quality drift caused by a long quantized cache instead of repeatedly resetting a short window.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import subprocess
from typing import Any

DEFAULT_DTYPES = ("int8", "fp8", "rk8v4", "rk4v4-e8")
DEFAULT_DEPTHS = (4096, 16384, 32768, 65536, 131072)
BASELINE = "int8"


def csv_text(value: str) -> list[str]:
    out = [part.strip() for part in value.split(",") if part.strip()]
    if not out or len(out) != len(set(out)):
        raise argparse.ArgumentTypeError("expected unique comma-separated values")
    return out


def csv_uints(value: str) -> list[int]:
    try:
        out = [int(part) for part in value.split(",") if part]
    except ValueError as error:
        raise argparse.ArgumentTypeError("expected comma-separated positive integers") from error
    if not out or any(x <= 0 for x in out) or out != sorted(set(out)):
        raise argparse.ArgumentTypeError("depths must be positive, unique and increasing")
    return out


def find_report(directory: Path) -> Path:
    reports = list(directory.rglob("report.json"))
    if len(reports) != 1:
        raise RuntimeError(f"expected exactly one report.json under {directory}, found {len(reports)}")
    return reports[0]


def run_arm(args: argparse.Namespace, dtype: str) -> dict[str, Any]:
    arm_dir = args.out / dtype
    arm_dir.mkdir(parents=True, exist_ok=False)
    command = [
        str(args.exe.resolve()),
        str(args.model.resolve()),
        "--corpus", str(args.corpus.resolve()),
        "--depths", ",".join(map(str, args.depths)),
        "--tail", str(args.tail),
        "--context", str(max(args.context, args.depths[-1] + args.tail)),
        "--stride", str(args.stride),
        "--device", str(args.device),
        "--kv-dtype", dtype,
        "--output", str(arm_dir.resolve()),
        "--log-level", args.log_level,
    ]
    if args.quick:
        command.append("--quick")
    (arm_dir / "command.json").write_text(json.dumps(command, indent=2) + "\n", encoding="utf-8")
    completed = subprocess.run(command, text=True, capture_output=True)
    (arm_dir / "stdout.log").write_text(completed.stdout, encoding="utf-8")
    (arm_dir / "stderr.log").write_text(completed.stderr, encoding="utf-8")
    result: dict[str, Any] = {
        "dtype": dtype,
        "returncode": completed.returncode,
        "command": command,
    }
    if completed.returncode != 0:
        result["status"] = "failed"
        return result
    report_path = find_report(arm_dir)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    result.update({
        "status": "ok",
        "report": str(report_path),
        "artifact": report.get("artifact"),
        "execution": report.get("execution"),
        "resources": report.get("resources"),
        "depths": report.get("depths", []),
        "overall": report.get("overall"),
    })
    return result


def index_depths(arm: dict[str, Any]) -> dict[int, dict[str, Any]]:
    return {
        int(row["prefix_depth"]): row
        for row in arm.get("depths", [])
        if "prefix_depth" in row
    }


def build_comparison(arms: list[dict[str, Any]]) -> dict[str, Any]:
    by_dtype = {arm["dtype"]: arm for arm in arms}
    baseline = by_dtype.get(BASELINE)
    if baseline is None or baseline.get("status") != "ok":
        raise RuntimeError("INT8 baseline must complete before deltas can be computed")
    base_depths = index_depths(baseline)
    rows: list[dict[str, Any]] = []
    for dtype, arm in by_dtype.items():
        if arm.get("status") != "ok":
            continue
        depth_map = index_depths(arm)
        for depth, candidate in depth_map.items():
            base = base_depths.get(depth)
            if base is None:
                continue
            if int(candidate.get("scored_tokens", 0)) != int(base.get("scored_tokens", 0)):
                raise RuntimeError(f"{dtype}@{depth} scored-token coverage differs from INT8")
            if int(candidate.get("stream_count", 0)) != int(base.get("stream_count", 0)):
                raise RuntimeError(f"{dtype}@{depth} stream coverage differs from INT8")
            mean_nll = float(candidate["mean_nll"])
            ppl = float(candidate["perplexity"])
            base_nll = float(base["mean_nll"])
            base_ppl = float(base["perplexity"])
            rows.append({
                "dtype": dtype,
                "prefix_depth": depth,
                "stream_count": int(candidate["stream_count"]),
                "scored_tokens": int(candidate["scored_tokens"]),
                "mean_nll": mean_nll,
                "perplexity": ppl,
                "delta_mean_nll_vs_int8": mean_nll - base_nll,
                "delta_perplexity_vs_int8": ppl - base_ppl,
                "delta_perplexity_percent_vs_int8":
                    (ppl / base_ppl - 1.0) * 100.0 if base_ppl > 0 else math.nan,
            })
    rows.sort(key=lambda row: (row["prefix_depth"], row["dtype"]))
    return {"baseline": BASELINE, "rows": rows}


def write_markdown(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# Long-history KV perplexity matrix",
        "",
        "Baseline: int8. Each row scores only the requested tail after building the full prefix.",
        "",
        "| depth | KV | streams | scored | PPL | dNLL vs INT8 | dPPL % vs INT8 |",
        "|---:|---|---:|---:|---:|---:|---:|",
    ]
    for row in summary["comparison"]["rows"]:
        lines.append(
            f"| {row['prefix_depth']} | {row['dtype']} | {row['stream_count']} | "
            f"{row['scored_tokens']} | {row['perplexity']:.8f} | "
            f"{row['delta_mean_nll_vs_int8']:+.8e} | "
            f"{row['delta_perplexity_percent_vs_int8']:+.5f}% |"
        )
    failures = [arm for arm in summary["arms"] if arm.get("status") != "ok"]
    if failures:
        lines += ["", "## Failed arms", ""]
        for arm in failures:
            lines.append(f"- {arm['dtype']}: exit {arm['returncode']}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exe", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument(
        "--corpus",
        type=Path,
        default=Path("eval/corpora/perplexity-1m/manifest.json"),
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--dtypes", type=csv_text, default=list(DEFAULT_DTYPES))
    parser.add_argument("--depths", type=csv_uints, default=list(DEFAULT_DEPTHS))
    parser.add_argument("--tail", type=int, default=2048)
    parser.add_argument("--context", type=int, default=4096)
    parser.add_argument("--stride", type=int, default=2048)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--log-level", default="warning")
    args = parser.parse_args(argv)

    if not args.exe.is_file() or not args.model.is_file() or not args.corpus.is_file():
        parser.error("--exe, --model and --corpus must exist")
    if args.out.exists():
        parser.error("--out must be a new directory")
    if BASELINE not in args.dtypes:
        parser.error("INT8 must be included because it is the comparison baseline")
    if args.tail <= 0:
        parser.error("--tail must be positive")
    if args.context < 2 or args.stride <= 0 or args.stride >= args.context:
        parser.error("context/stride must satisfy context>=2 and 1<=stride<context")

    args.out.mkdir(parents=True)
    arms = [run_arm(args, dtype) for dtype in args.dtypes]
    summary: dict[str, Any] = {
        "schema_version": 1,
        "artifact_type": "ninfer_long_history_kv_perplexity_matrix",
        "protocol": {
            "description": "build prefix [0,D), score only [D,D+tail)",
            "depths": args.depths,
            "tail_tokens": args.tail,
            "corpus": str(args.corpus.resolve()),
            "quick": args.quick,
        },
        "arms": arms,
    }
    summary["comparison"] = build_comparison(arms)
    (args.out / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    write_markdown(args.out / "summary.md", summary)
    print((args.out / "summary.md").read_text(encoding="utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
