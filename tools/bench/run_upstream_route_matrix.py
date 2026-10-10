"""Interleaved RTX 4090 route A/B with existing ninfer_bench JSON output.

Measures isolated prefill; NOT an Agent-serving or correctness qualification.
The reference arm is repeated beside each candidate, AB/BA alternating, so GPU
thermal drift is less likely to fabricate a winner. Never edits production config.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
from typing import Any

DEFAULT_CASES: dict[str, dict[str, str]] = {
    "baseline": {"NINFER_DEVICE_ROUTE_MODE": "off"},
    "prompt_fast": {
        "NINFER_DEVICE_ROUTE_MODE": "builtin",
        "NINFER_DEVICE_ROUTE_ONLY": "attn_prompt_fast",
    },
    "gdn_two_stage": {
        "NINFER_DEVICE_ROUTE_MODE": "builtin",
        "NINFER_DEVICE_ROUTE_ONLY": "gdn_two_stage/h32,gdn_two_stage/h48",
    },
    "t2_upstream": {
        "NINFER_DEVICE_ROUTE_MODE": "builtin",
        "NINFER_DEVICE_ROUTE_ONLY": "t2_a16",
        "NINFER_DEVICE_ROUTE_OVERRIDES": "t2_a16=upstream",
    },
    "sm_wave": {
        "NINFER_DEVICE_ROUTE_MODE": "builtin",
        "NINFER_DEVICE_ROUTE_ONLY": "prefill_align",
        "NINFER_DEVICE_ROUTE_OVERRIDES": "prefill_align=on",
    },
    "all_candidates": {
        "NINFER_DEVICE_ROUTE_MODE": "builtin",
        "NINFER_DEVICE_ROUTE_OVERRIDES": "t2_a16=upstream;prefill_align=on",
    },
}
CLEAN_ENV = (
    "NINFER_DEVICE_ROUTE_MODE", "NINFER_DEVICE_PROFILE_PATH", "NINFER_DEVICE_PROFILES",
    "NINFER_DEVICE_ROUTE_ONLY", "NINFER_DEVICE_ROUTE_OVERRIDES",
    "NINFER_PROMPT_FAST", "NINFER_GDN_TWO_STAGE", "NINFER_PREFILL_ALIGN",
)


def read_cases(path: Path | None) -> dict[str, dict[str, str]]:
    if path is None:
        return DEFAULT_CASES
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or "baseline" not in payload:
        raise ValueError("cases JSON requires an object with a baseline arm")
    cases = {}
    for name, variables in payload.items():
        if not name.replace("_", "").replace("-", "").isalnum() or not isinstance(variables, dict):
            raise ValueError(f"invalid case name or env mapping: {name!r}")
        if any(not isinstance(k, str) or not k.startswith("NINFER_") or
               not isinstance(v, str) for k, v in variables.items()):
            raise ValueError(f"invalid environment override in {name}")
        cases[name] = variables
    return cases


def percentile(samples: list[float], q: float) -> float:
    ordered = sorted(samples)
    point = (len(ordered) - 1) * q
    lower = math.floor(point)
    upper = math.ceil(point)
    return ordered[lower] * (upper - point) + ordered[upper] * (point - lower) if lower != upper else ordered[lower]


def summary_values(samples: list[float]) -> dict[str, float] | None:
    if not samples:
        return None
    return {
        "median": statistics.median(samples),
        "p05": percentile(samples, .05),
        "p95": percentile(samples, .95),
        "min": min(samples),
        "max": max(samples),
    }


def streaming_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def benchmark_command(args: argparse.Namespace, report: Path) -> list[str]:
    command = [
        str(args.exe.resolve()), "--weights", str(args.model.resolve()),
        "--device", str(args.device), "--max-ctx", str(args.max_context),
        "-p", args.prompts, "-r", "1", "--warmup", "0",
        "--kv-dtype", args.kv_dtype, "--prefill-chunk", args.prefill_chunk,
        "--output", "json", "--output-file", str(report),
    ]
    if args.spec != "none":
        command += ["--spec", args.spec, "--draft-tokens", str(args.draft_tokens)]
    if args.prefill_cublas:
        command.append("--prefill-cublas")
    return command


def run_one(args: argparse.Namespace, case: str, override: dict[str, str],
            pair: int, iteration: int, warmup: bool) -> dict[str, Any]:
    tag = f"{'warmup' if warmup else 'pair'}-{pair:02d}-{iteration:02d}-{case}"
    base = args.out / tag
    base.mkdir()
    report = base / "bench.json"
    command = benchmark_command(args, report)
    env = os.environ.copy()
    for key in CLEAN_ENV:
        env.pop(key, None)
    env.update(override)
    env["NINFER_DEVICE_ROUTE_TRACE"] = "1"
    (base / "invocation.json").write_text(json.dumps({
        "command": command, "route_env": {k: env[k] for k in CLEAN_ENV if k in env},
        "case": case, "pair": pair, "warmup": warmup,
    }, indent=2) + "\n", encoding="utf-8")
    with (base / "stdout.log").open("wb") as stdout, (base / "stderr.log").open("wb") as stderr:
        try:
            outcome = subprocess.run(command, env=env, stdout=stdout, stderr=stderr,
                                     timeout=args.timeout, check=False)
        except subprocess.TimeoutExpired as e:
            raise RuntimeError(f"benchmark timeout for {tag}, see {base}") from e
    if outcome.returncode != 0 or not report.is_file():
        raise RuntimeError(f"benchmark failed {tag}: exit {outcome.returncode}, see {base}")
    raw = json.loads(report.read_text(encoding="utf-8"))
    tests = raw.get("tests", [])
    if not isinstance(tests, list) or not tests:
        raise RuntimeError(f"{tag}: no benchmark tests in report")
    metrics: dict[str, float] = {}
    for test in tests:
        label = str(test["label"])
        value = test.get("prefill_tok_s_mean")
        if value is None or not math.isfinite(float(value)) or float(value) <= 0:
            raise RuntimeError(f"{tag}: missing/invalid prefill_tok_s_mean for {label}")
        if label in metrics:
            raise RuntimeError(f"{tag}: duplicate test label {label}")
        metrics[label] = float(value)
    memory = raw.get("memory") or {}
    config = raw.get("config") or {}
    record = {
        "case": case, "pair": pair, "warmup": warmup,
        "report": str(report.relative_to(args.out)), "metrics": metrics,
        "resolved_chunk": config.get("prefill_chunk"),
        "runtime_reservation_bytes": memory.get("runtime_reservation_bytes"),
        "workspace_capacity_bytes": (memory.get("workspace") or {}).get("capacity_bytes"),
    }
    (base / "record.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    return record


def calculate(rows: list[dict[str, Any]], name: str, pairs: int) -> dict[str, Any]:
    baseline = {r["pair"]: r for r in rows if r["case"] == "baseline" and not r["warmup"]}
    candidate = {r["pair"]: r for r in rows if r["case"] == name and not r["warmup"]}
    if len(baseline) != pairs or len(candidate) != pairs:
        raise RuntimeError(f"missing baseline/candidate measurements for {name}")
    labels = sorted(baseline[0]["metrics"])
    results = {}
    for label in labels:
        ratios = []
        speeds = []
        refs = []
        for pair in range(pairs):
            a, b = baseline[pair], candidate[pair]
            if set(a["metrics"]) != set(b["metrics"]):
                raise RuntimeError(f"output benchmark label mismatch for {name}, pair {pair}")
            av = a["metrics"][label]
            bv = b["metrics"][label]
            refs.append(av)
            speeds.append(bv)
            ratios.append(100.0 * (bv / av - 1.0))
        results[label] = {
            "baseline_tok_s": summary_values(refs),
            "candidate_tok_s": summary_values(speeds),
            "paired_improvement_pct": summary_values(ratios),
        }
    return {
        "case": name,
        "tests": results,
        "resolved_chunk_values": sorted({str(r["resolved_chunk"]) for r in candidate.values()}),
        "runtime_reservation_peak_bytes": max(
            [int(r["runtime_reservation_bytes"]) for r in candidate.values()
             if r["runtime_reservation_bytes"] is not None], default=None),
        "workspace_capacity_peak_bytes": max(
            [int(r["workspace_capacity_bytes"]) for r in candidate.values()
             if r["workspace_capacity_bytes"] is not None], default=None),
        "performance_only_screen": all(
            m["paired_improvement_pct"]["median"] >= 2
            and m["paired_improvement_pct"]["p05"] >= -5 for m in results.values()
        ),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exe", type=Path, required=True, help="ninfer_bench binary")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True, help="new output directory")
    parser.add_argument("--cases-json", type=Path, help="optional case-name -> NINFER_* env map")
    parser.add_argument("--cases", default="prompt_fast,gdn_two_stage,t2_upstream,sm_wave,all_candidates")
    parser.add_argument("--pairs", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--max-context", type=int, default=32768)
    parser.add_argument("--prompts", default="1024,4096,16384,32768")
    parser.add_argument("--prefill-chunk", default="1024")
    parser.add_argument("--kv-dtype", default="int8")
    parser.add_argument("--spec", choices=("none", "mtp", "dflash", "dflash2"), default="dflash2")
    parser.add_argument("--draft-tokens", type=int, default=7)
    parser.add_argument("--prefill-cublas", action="store_true")
    args = parser.parse_args(argv)
    if not args.exe.is_file() or not args.model.is_file():
        parser.error("--exe/--model must refer to existing files")
    if args.out.exists():
        parser.error("--out already exists; never overwrite historical measurements")
    if args.pairs < 3 or args.warmup < 0 or args.timeout < 1 or args.max_context < 1024:
        parser.error("pairs >= 3, warmup >= 0, timeout >= 1, context >= 1024 required")
    cases = read_cases(args.cases_json)
    selected = [item.strip() for item in args.cases.split(",") if item.strip()]
    if len(set(selected)) != len(selected) or not selected or any(x not in cases or x == "baseline" for x in selected):
        parser.error("--cases must be unique, available non-baseline case names")
    args.out.mkdir(parents=True)
    gpu = {}
    try:
        p = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,driver_version,memory.total",
             "--format=csv,noheader", "-i", str(args.device)],
            capture_output=True, text=True, timeout=10, check=False)
        gpu = {"nvidia_smi": p.stdout.strip(), "returncode": p.returncode}
    except (FileNotFoundError, subprocess.TimeoutExpired):
        gpu = {"nvidia_smi": "not available"}
    records: list[dict[str, Any]] = []
    try:
        for case in selected:
            for warmup in range(args.warmup):
                for index, arm in enumerate(("baseline", case)):
                    run_one(args, arm, cases[arm], warmup, index, True)
            for pair in range(args.pairs):
                ordered = ("baseline", case) if pair % 2 == 0 else (case, "baseline")
                for index, arm in enumerate(ordered):
                    records.append(run_one(args, arm, cases[arm], pair, index, False))
            # One candidate's baseline records should not be reused for other candidates.
            results = [x for x in records if x["case"] in ("baseline", case)]
            # Assign the baseline from the last n pairs for this candidate, not earlier arms.
            # Reused pair numbers from earlier candidates are deliberately excluded.
            result = calculate(results[-2 * args.pairs:], case, args.pairs)
            (args.out / f"summary-{case}.json").write_text(json.dumps(result, indent=2) + "\n")
        reports = [json.loads((args.out / f"summary-{case}.json").read_text()) for case in selected]
        manifest = {
            "schema": "ninfer.route-ab.v1", "hardware": gpu,
            "benchmark_scope": "isolated ninfer_bench prefill, not serving quality or latency",
            "model": str(args.model.resolve()),
            "model_size_bytes": args.model.stat().st_size,
            "model_sha256": None,  # model files can be >20 GB; do not hide expensive hashing
            "exe": str(args.exe.resolve()), "exe_sha256": streaming_sha256(args.exe),
            "settings": {"prompts": args.prompts, "prefill_chunk": args.prefill_chunk,
                         "kv_dtype": args.kv_dtype, "spec": args.spec, "draft_tokens": args.draft_tokens,
                         "max_context": args.max_context, "pairs": args.pairs, "warmup": args.warmup},
            "cases": {k: cases[k] for k in ["baseline", *selected]},
            "candidates": reports,
            "quality_gate": "not executed",
            "serving_gate": "not executed",
            "promotion": "not approved: requires user local correctness and serving measurements",
        }
        (args.out / "summary.json").write_text(json.dumps(manifest, indent=2) + "\n")
        table = ["# RTX 4090 route A/B — prefill only", "",
                 "| Arm | Prompt | Median Δ tok/s | P05 paired Δ | P95 paired Δ | Perf-only screen |",
                 "|---|---|---:|---:|---:|---|"]
        for case in reports:
            for label, metric in case["tests"].items():
                ratio = metric["paired_improvement_pct"]
                table.append(f"| {case['case']} | {label} | {ratio['median']:+.2f}% "
                             f"| {ratio['p05']:+.2f}% | {ratio['p95']:+.2f}% "
                             f"| {'pass' if case['performance_only_screen'] else 'fail'} |")
        table += ["", "These results are **NOT** permission to enable production routes.",
                  "Run numerical oracle, perplexity, state/checkpoint and Agent serving gates separately.",
                  "All raw logs and exact invocation environments are in per-run directories.", ""]
        (args.out / "summary.md").write_text("\n".join(table), encoding="utf-8")
    except Exception as exc:
        (args.out / "ERROR.txt").write_text(str(exc) + "\n")
        raise
    print(f"Saved {args.out / 'summary.json'} and {args.out / 'summary.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
