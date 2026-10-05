"""Run long-history KV-cache perplexity qualification across storage formats.

Each arm invokes ninfer-perplexity in fixed-depth mode. At depth D the evaluator materializes the
complete prefix [0,D) into the selected Main KV representation and scores only the following tail.
The comparison therefore measures quality drift caused by a *long encoded cache*, rather than the
old truncated 4K-window protocol which resets KV near every scored token.

The JSON/Markdown summary uses the first dtype as the baseline (INT8 by default) and reports
delta-NLL and perplexity ratio at every depth with identical scored-token coverage.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import shutil
import subprocess
import time


DEFAULT_DTYPES = ("int8", "fp8", "rk8v4", "rk4v4-e8")


def csv_items(value: str) -> list[str]:
    out = [item.strip() for item in value.split(",") if item.strip()]
    if not out or len(out) != len(set(out)):
        raise argparse.ArgumentTypeError("comma-separated unique values required")
    return out


def csv_depths(value: str) -> list[int]:
    try:
        out = [int(item) for item in csv_items(value)]
    except ValueError as error:
        raise argparse.ArgumentTypeError("depths must be integers") from error
    if any(depth <= 0 for depth in out) or out != sorted(out):
        raise argparse.ArgumentTypeError("depths must be positive and increasing")
    return out


def run_arm(args: argparse.Namespace, dtype: str, root: Path) -> dict:
    arm_dir = root / dtype
    if arm_dir.exists():
        shutil.rmtree(arm_dir)
    arm_dir.mkdir(parents=True)
    result_dir = arm_dir / "result"

    command = [
        str(args.exe.resolve()),
        str(args.model.resolve()),
        "--device", str(args.device),
        "--kv-dtype", dtype,
        "--depths", ",".join(map(str, args.depths)),
        "--tail", str(args.tail),
        "--context", str(max(4096, args.depths[-1] + args.tail)),
        "--output", str(result_dir.resolve()),
        "--log-level", args.log_level,
    ]
    if args.corpus is not None:
        command += ["--corpus", str(args.corpus.resolve())]
        if args.quick:
            command.append("--quick")
    else:
        command += ["--text", str(args.text.resolve())]
    if args.no_prefill_a8:
        command.append("--no-prefill-a8")
    if args.prefill_cublas:
        command.append("--prefill-cublas")
        if args.no_prefill_cublas_projections:
            command.append("--no-prefill-cublas-projections")

    (arm_dir / "command.json").write_text(json.dumps(command, indent=2) + "\n", encoding="utf-8")
    started = time.perf_counter()
    completed = subprocess.run(command, text=True, capture_output=True, check=False)
    (arm_dir / "stdout.log").write_text(completed.stdout, encoding="utf-8")
    (arm_dir / "stderr.log").write_text(completed.stderr, encoding="utf-8")
    if completed.returncode != 0:
        raise RuntimeError(f"{dtype}: ninfer-perplexity exited {completed.returncode}")

    report_path = result_dir / "report.json"
    if not report_path.is_file():
        raise RuntimeError(f"{dtype}: report.json was not produced")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("schema_version", 0) < 3:
        raise RuntimeError(f"{dtype}: evaluator lacks fixed-depth report schema")
    execution = report.get("execution", {})
    if execution.get("protocol") != "fixed-depth-long-history":
        raise RuntimeError(f"{dtype}: evaluator did not run fixed-depth protocol")

    depths = {}
    for row in report.get("depths", []):
        depth = int(row["prefix_depth"])
        depths[depth] = {
            "stream_count": int(row["stream_count"]),
            "scored_tokens": int(row["scored_tokens"]),
            "total_nll": float(row["total_nll"]),
            "mean_nll": float(row["mean_nll"]),
            "perplexity": float(row["perplexity"]),
        }
    if not depths:
        raise RuntimeError(f"{dtype}: no requested depth had score coverage")
    return {
        "dtype": dtype,
        "seconds": time.perf_counter() - started,
        "depths": depths,
        "overall": report["overall"],
        "report": str(report_path.resolve()),
    }


def compare(arms: list[dict]) -> dict:
    baseline = arms[0]
    base_depths = baseline["depths"]
    rows = []
    for depth in sorted(base_depths):
        base = base_depths[depth]
        for arm in arms:
            if depth not in arm["depths"]:
                raise RuntimeError(f"{arm['dtype']}: missing baseline depth {depth}")
            row = arm["depths"][depth]
            if (row["scored_tokens"], row["stream_count"]) != (
                base["scored_tokens"], base["stream_count"]
            ):
                raise RuntimeError(
                    f"{arm['dtype']}: depth {depth} coverage differs from {baseline['dtype']}"
                )
            delta_nll = row["mean_nll"] - base["mean_nll"]
            ppl_ratio = math.exp(delta_nll)
            rows.append(
                {
                    "depth": depth,
                    "dtype": arm["dtype"],
                    **row,
                    "delta_mean_nll_vs_baseline": delta_nll,
                    "ppl_ratio_vs_baseline": ppl_ratio,
                    "ppl_change_percent_vs_baseline": (ppl_ratio - 1.0) * 100.0,
                }
            )
    return {
        "schema_version": 1,
        "artifact_type": "ninfer_long_context_kv_perplexity_comparison",
        "baseline": baseline["dtype"],
        "arms": arms,
        "rows": rows,
    }


def markdown(summary: dict) -> str:
    lines = [
        "# Long-context KV perplexity comparison",
        "",
        f"Baseline: **{summary['baseline']}**",
        "",
        "| Prefix depth | KV | Streams | Scored tokens | Mean NLL | PPL | ΔNLL | PPL change |",
        "|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary["rows"]:
        lines.append(
            f"| {row['depth']} | {row['dtype']} | {row['stream_count']} | "
            f"{row['scored_tokens']} | {row['mean_nll']:.8f} | {row['perplexity']:.8f} | "
            f"{row['delta_mean_nll_vs_baseline']:+.8f} | "
            f"{row['ppl_change_percent_vs_baseline']:+.4f}% |"
        )
    lines += [
        "",
        "Interpretation: positive ΔNLL / PPL change means worse next-token likelihood than the",
        "baseline after encoding the same complete prefix in that KV format. Negative values mean",
        "lower measured NLL; treat small negative changes as measurement/numerical variation unless",
        "they repeat across independent corpus domains.",
        "",
    ]
    return "\n".join(lines)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exe", required=True, type=Path)
    parser.add_argument("--model", required=True, type=Path)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--corpus", type=Path)
    source.add_argument("--text", type=Path)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--depths", type=csv_depths, default=[8192, 32768, 65536, 131072, 196608, 258048])
    parser.add_argument("--tail", type=int, default=2048)
    parser.add_argument("--dtypes", type=csv_items, default=list(DEFAULT_DTYPES))
    parser.add_argument("--log-level", default="warning")
    parser.add_argument("--no-prefill-a8", action="store_true")
    parser.add_argument("--prefill-cublas", action="store_true")
    parser.add_argument("--no-prefill-cublas-projections", action="store_true")
    args = parser.parse_args(argv)

    if not args.exe.is_file() or not args.model.is_file():
        parser.error("--exe and --model must exist")
    if args.corpus is not None and not args.corpus.is_file():
        parser.error("--corpus must exist")
    if args.text is not None and not args.text.is_file():
        parser.error("--text must exist")
    if args.quick and args.corpus is None:
        parser.error("--quick requires --corpus")
    if args.tail <= 0:
        parser.error("--tail must be positive")
    if args.out.exists():
        parser.error("--out must be a new directory")
    args.out.mkdir(parents=True)

    arms = []
    for dtype in args.dtypes:
        print(f"[kv-ppl] {dtype}", flush=True)
        arms.append(run_arm(args, dtype, args.out))
    summary = compare(arms)
    (args.out / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    table = markdown(summary)
    (args.out / "summary.md").write_text(table, encoding="utf-8")
    print(table)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
