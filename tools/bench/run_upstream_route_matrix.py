"""RTX 4090 resident-weight route A/B matrix for ninfer_bench.

One benchmark process, one model upload, fresh per-arm Engine/Program/CUDA Graph.
Baseline/candidate observations are interleaved AB/BA in the *same resident session*.
No route is promoted from this performance-only screen.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import selectors
import statistics
import subprocess
import sys
import time
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
        # Exact numerical contract: no approximate kernel is selected.
    },
    "gdn_two_stage_approx": {
        "NINFER_DEVICE_ROUTE_MODE": "builtin",
        "NINFER_DEVICE_ROUTE_ONLY": "gdn_two_stage/h32,gdn_two_stage/h48",
        "NINFER_GDN_TWO_STAGE_NUMERICS": "approx",
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
        "NINFER_DEVICE_ROUTE_ONLY": (
            "attn_prompt_fast,gdn_two_stage/h32,gdn_two_stage/h48,t2_a16,prefill_align"
        ),
        "NINFER_DEVICE_ROUTE_OVERRIDES": "t2_a16=upstream;prefill_align=on",
        # Does not switch on approximate GDN.
    },
}
CLEAN_ENV = (
    "NINFER_DEVICE_ROUTE_MODE", "NINFER_DEVICE_PROFILE_PATH", "NINFER_DEVICE_PROFILES",
    "NINFER_DEVICE_ROUTE_ONLY", "NINFER_DEVICE_ROUTE_OVERRIDES",
    "NINFER_PROMPT_FAST", "NINFER_GDN_TWO_STAGE", "NINFER_GDN_TWO_STAGE_NUMERICS",
    "NINFER_PREFILL_ALIGN", "NINFER_DEVICE_ROUTE_TRACE",
)


def read_cases(path: Path | None) -> dict[str, dict[str, str]]:
    if path is None:
        return DEFAULT_CASES
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or "baseline" not in payload:
        raise ValueError("cases JSON requires an object with a baseline arm")
    result: dict[str, dict[str, str]] = {}
    for case, values in payload.items():
        if not isinstance(case, str) or not case or not case.replace("_", "").replace("-", "").isalnum():
            raise ValueError(f"invalid arm name: {case!r}")
        if not isinstance(values, dict) or any(key not in CLEAN_ENV or
                not isinstance(v, str) or len(v) > 4096 for key, v in values.items()):
            raise ValueError(f"invalid route environment for {case}")
        result[case] = values
    if result["baseline"].get("NINFER_DEVICE_ROUTE_MODE") != "off":
        raise ValueError("baseline must explicitly disable device profiles")
    return result


def percentile(samples: list[float], q: float) -> float:
    ordered = sorted(samples)
    point = (len(ordered) - 1) * q
    lower, upper = math.floor(point), math.ceil(point)
    return (ordered[lower] * (upper - point) + ordered[upper] * (point - lower)
            if lower != upper else ordered[lower])


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


def benchmark_command(args: argparse.Namespace) -> list[str]:
    result = [
        str(args.exe.resolve()), "--resident-session",
        "--weights", str(args.model.resolve()),
        "--device", str(args.device), "--max-ctx", str(args.max_context),
        "-p", args.prompts, "-r", str(args.repetitions),
        "--warmup", str(args.arm_warmup),
        "--kv-dtype", args.kv_dtype, "--prefill-chunk", args.prefill_chunk,
        "--spec", "dflash2", "--draft-tokens", str(args.draft_tokens),
        "--output", "json",
    ]
    if args.prefill_cublas:
        result.append("--prefill-cublas")
    return result


def build_plan(selected: list[str], pairs: int, warmup: int) -> list[dict[str, Any]]:
    """Exactly one unique baseline + candidate per pair, never across experiments."""
    plan = []
    for case in selected:
        for warm in range(warmup):
            for order, arm in enumerate(("baseline", case)):
                plan.append(dict(experiment=case, case=arm, pair=warm, order=order, warmup=True))
        for pair in range(pairs):
            order = ("baseline", case) if pair % 2 == 0 else (case, "baseline")
            for index, arm in enumerate(order):
                plan.append(dict(experiment=case, case=arm, pair=pair,
                                 order=index, warmup=False))
    for arm in plan:
        prefix = "warmup" if arm["warmup"] else "pair"
        arm["tag"] = (f"{prefix}-{arm['experiment']}-{arm['pair']:02d}-"
                      f"{arm['order']:02d}-{arm['case']}")
    if len({arm["tag"] for arm in plan}) != len(plan):
        raise ValueError("duplicate route matrix arm identifier")
    return plan


def calculate(rows: list[dict[str, Any]], name: str, pairs: int) -> dict[str, Any]:
    baseline = {r["pair"]: r for r in rows if r["case"] == "baseline" and not r["warmup"]}
    candidate = {r["pair"]: r for r in rows if r["case"] == name and not r["warmup"]}
    if len(baseline) != pairs or len(candidate) != pairs:
        raise RuntimeError(f"missing baseline/candidate measurements for {name}")
    labels = sorted(baseline[0]["metrics"])
    results = {}
    for label in labels:
        ratios, speeds, refs = [], [], []
        for pair in range(pairs):
            a, b = baseline[pair], candidate[pair]
            if set(a["metrics"]) != set(b["metrics"]):
                raise RuntimeError(f"benchmark test label mismatch {name}, pair {pair}")
            av, bv = a["metrics"][label], b["metrics"][label]
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
        "quality_gate": "blocked_approx_state_semantics" if "approx" in name else "not_executed",
    }


class JsonlReader:
    """Bounded nonblocking reads: per-arm deadline includes partial JSON packets."""
    def __init__(self, stream):
        self.fd = stream.fileno()
        self.selector = selectors.DefaultSelector()
        self.selector.register(self.fd, selectors.EVENT_READ)
        self.buffer = bytearray()

    def close(self):
        self.selector.close()

    def receive(self, timeout: int) -> dict[str, Any]:
        deadline = time.monotonic() + timeout
        while True:
            end = self.buffer.find(b"\n")
            if end >= 0:
                raw = bytes(self.buffer[:end])
                del self.buffer[:end + 1]
                payload = json.loads(raw)
                if not isinstance(payload, dict):
                    raise RuntimeError("invalid resident JSONL response")
                return payload
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("resident route session response timeout")
            if not self.selector.select(remaining):
                raise TimeoutError("resident route session response timeout")
            chunk = os.read(self.fd, 65536)
            if not chunk:
                raise RuntimeError("resident benchmark terminated without response")
            self.buffer.extend(chunk)
            if len(self.buffer) > 12 * 1024 * 1024:
                raise RuntimeError("resident benchmark JSONL response exceeded 12 MiB")


def validated_measurement(packet: dict[str, Any], expected_id: str) -> dict[str, Any]:
    if packet.get("id") != expected_id or packet.get("event") != "measurement" or packet.get("ok") is not True:
        raise RuntimeError(f"wrong/failed resident response for {expected_id}: {str(packet)[:400]}")
    if packet.get("model_load_count") != 1:
        raise RuntimeError("resident route matrix reloaded the model")
    report = packet.get("report")
    if not isinstance(report, dict) or (report.get("residency") or {}).get("model_load_count") != 1:
        raise RuntimeError("resident route arm lacks verified single-load evidence")
    tests = report.get("tests")
    if not isinstance(tests, list) or not tests:
        raise RuntimeError("resident route arm has no benchmark tests")
    metrics: dict[str, float] = {}
    for test in tests:
        label = str(test["label"])
        value = test.get("prefill_tok_s_mean")
        if label in metrics or value is None or not math.isfinite(float(value)) or float(value) <= 0:
            raise RuntimeError(f"invalid prefill throughput for {label}")
        metrics[label] = float(value)
    memory = report.get("memory") or {}
    config = report.get("config") or {}
    return {
        "metrics": metrics,
        "resolved_chunk": config.get("prefill_chunk"),
        "runtime_reservation_bytes": memory.get("runtime_reservation_bytes"),
        "workspace_capacity_bytes": (memory.get("workspace") or {}).get("capacity_bytes"),
        "program_create_seconds": (report.get("residency") or {}).get("program_create_seconds"),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exe", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True, help="new output directory")
    parser.add_argument("--cases-json", type=Path)
    parser.add_argument("--cases", default="prompt_fast,gdn_two_stage,t2_upstream,sm_wave,all_candidates")
    parser.add_argument("--pairs", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=1, help="discarded complete baseline/candidate pairs before measured pairs")
    parser.add_argument("--arm-warmup", type=int, default=1, help="in-Engine warmup repetitions; avoids timing graph capture")
    parser.add_argument("--repetitions", type=int, default=1, help="measured repetitions per fresh Engine")
    parser.add_argument("--route-trace", action="store_true", help="diagnostic only; distorted host timings")
    parser.add_argument("--timeout", type=int, default=900, help="deadline per resident response including initial model upload")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--max-context", type=int, default=32768)
    parser.add_argument("--prompts", default="1024,4096,16384,32768")
    parser.add_argument("--prefill-chunk", default="1024")
    parser.add_argument("--kv-dtype", default="int8")
    parser.add_argument("--spec", choices=("dflash2",), default="dflash2",
                        help="resident owner currently supports only DFlash2 Full")
    parser.add_argument("--draft-tokens", type=int, default=7)
    parser.add_argument("--prefill-cublas", action="store_true")
    args = parser.parse_args(argv)

    if not args.exe.is_file() or not args.model.is_file():
        parser.error("--exe and --model must be existing files")
    if args.out.exists():
        parser.error("--out must not exist; old evidence is immutable")
    if args.pairs < 3 or args.warmup < 0 or args.arm_warmup < 1 or args.repetitions < 1 or args.timeout < 1:
        parser.error("pairs>=3, warmup>=0, arm-warmup>=1, repetitions>=1, timeout>=1")
    if args.max_context < 1024 or args.device < 0 or not 1 <= args.draft_tokens <= 15:
        parser.error("unsupported context/device/draft token settings")

    cases = read_cases(args.cases_json)
    selected = [name.strip() for name in args.cases.split(",") if name.strip()]
    if not selected or len(set(selected)) != len(selected) or any(
            c == "baseline" or c not in cases for c in selected):
        parser.error("cases must be distinct available non-baseline names")
    plan = build_plan(selected, args.pairs, args.warmup)
    command = benchmark_command(args)
    environment = os.environ.copy()
    for key in CLEAN_ENV:
        environment.pop(key, None)
    args.out.mkdir(parents=True)
    (args.out / "plan.json").write_text(json.dumps({
        "schema": "ninfer.route-resident-plan.v1", "cases": cases,
        "runs": plan, "command": command,
        "scope": "single artifact materialization, fresh CUDA Graph/Program per arm",
    }, indent=2) + "\n", encoding="utf-8")
    gpu = {}
    try:
        p = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version,memory.total",
                            "--format=csv,noheader", "-i", str(args.device)],
                           capture_output=True, text=True, timeout=10, check=False)
        gpu = {"nvidia_smi": p.stdout.strip(), "returncode": p.returncode}
    except (FileNotFoundError, subprocess.TimeoutExpired):
        gpu = {"nvidia_smi": "not available"}

    records = []
    reports = []
    process = None
    started = time.monotonic()
    try:
        with (args.out / "session-stderr.log").open("wb") as stderr, (
                args.out / "records.jsonl").open("w", encoding="utf-8") as journal:
            process = subprocess.Popen(
                command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                stderr=stderr, env=environment, bufsize=0)
            assert process.stdin is not None and process.stdout is not None
            reader = JsonlReader(process.stdout)
            try:
                ready = reader.receive(args.timeout)
                if ready.get("event") != "ready" or ready.get("model_load_count") != 1 or not ready.get("ok"):
                    raise RuntimeError("resident model did not initialize exactly once")
                (args.out / "residency.json").write_text(json.dumps(ready, indent=2) + "\n")
                for idx, item in enumerate(plan):
                    tag, case, experiment = item["tag"], item["case"], item["experiment"]
                    work_dir = args.out / tag
                    work_dir.mkdir()
                    overrides = dict(cases[case])
                    if args.route_trace:
                        overrides["NINFER_DEVICE_ROUTE_TRACE"] = "1"
                    (work_dir / "invocation.json").write_text(json.dumps({
                        "command": command, "route_env": overrides, **item,
                    }, indent=2) + "\n")
                    packet = json.dumps({"id": tag, "route_env": overrides}) + "\n"
                    process.stdin.write(packet.encode("utf-8"))
                    process.stdin.flush()
                    response = reader.receive(args.timeout)
                    measurement = validated_measurement(response, tag)
                    report = response["report"]
                    (work_dir / "bench.json").write_text(json.dumps(report, indent=2) + "\n")
                    record = dict(item)
                    record.update(measurement)
                    record["report"] = str((work_dir / "bench.json").relative_to(args.out))
                    (work_dir / "record.json").write_text(json.dumps(record, indent=2) + "\n")
                    records.append(record)
                    journal.write(json.dumps(record) + "\n")
                    journal.flush()  # Retain all completed arms even after a CUDA failure.
                    print(f"[resident {idx + 1}/{len(plan)}] {tag}; loads=1; "
                          f"program_init={measurement['program_create_seconds']:.3f}s", flush=True)

                process.stdin.write((json.dumps({"id": "__stop__", "stop": True}) + "\n").encode())
                process.stdin.flush()
                bye = reader.receive(args.timeout)
                if bye.get("event") != "bye" or bye.get("model_load_count") != 1 or not bye.get("ok"):
                    raise RuntimeError("resident benchmark shutdown failed or weights reloaded")
                process.stdin.close()
                if process.wait(timeout=20) != 0:
                    raise RuntimeError(f"resident benchmark exited {process.returncode}")
            finally:
                reader.close()

        for case in selected:
            rows = [record for record in records if record["experiment"] == case]
            result = calculate(rows, case, args.pairs)
            (args.out / f"summary-{case}.json").write_text(json.dumps(result, indent=2) + "\n")
            reports.append(result)

        summary = {
            "schema": "ninfer.route-ab.v2", "scope": "single resident model per full AB matrix; fresh Program/Graph per arm",
            "model": str(args.model.resolve()), "model_size_bytes": args.model.stat().st_size,
            "model_sha256": None, "exe": str(args.exe.resolve()),
            "exe_sha256": streaming_sha256(args.exe), "hardware": gpu,
            "model_load_count": 1, "program_creation_count": len(plan),
            "seconds_total": time.monotonic() - started,
            "quality_gate": "not_executed",
            "serving_gate": "not_executed",
            "settings": {"prompts": args.prompts, "prefill_chunk": args.prefill_chunk,
                         "kv_dtype": args.kv_dtype, "spec": args.spec,
                         "draft_tokens": args.draft_tokens, "max_context": args.max_context,
                         "pairs": args.pairs, "warmup": args.warmup,
                         "arm_warmup": args.arm_warmup, "repetitions": args.repetitions},
            "cases": {name: cases[name] for name in ["baseline", *selected]},
            "candidates": reports,
            "promotion": "not approved; requires numerical, memory and serving correctness",
        }
        (args.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        table = ["# RTX 4090 resident-weight route A/B — prefill only", "",
                 "**Model materializations: 1; fresh Program/CUDA Graph per arm.**", "",
                 "| Arm | Prompt | Median Δ tok/s | P05 paired Δ | P95 paired Δ | Performance screen |",
                 "|---|---|---:|---:|---:|---|"]
        for case in reports:
            for label, value in case["tests"].items():
                ratio = value["paired_improvement_pct"]
                table.append(f"| {case['case']} | {label} | {ratio['median']:+.2f}% "
                             f"| {ratio['p05']:+.2f}% | {ratio['p95']:+.2f}% "
                             f"| {'pass' if case['performance_only_screen'] else 'fail'} |")
        table += ["", "A performance screen is not a correctness gate. GDN approx is explicitly unqualified.",
                  "This is a serial single-request prefill benchmark, not concurrent Agent serving.", ""]
        (args.out / "summary.md").write_text("\n".join(table), encoding="utf-8")
        print(f"Saved {args.out / 'summary.json'} and {args.out / 'summary.md'}")
        return 0
    except Exception as exc:
        (args.out / "ERROR.txt").write_text(f"{type(exc).__name__}: {exc}\n")
        raise
    finally:
        if process is not None and process.poll() is None:
            process.kill()
            process.wait(timeout=20)


if __name__ == "__main__":
    sys.exit(main())
