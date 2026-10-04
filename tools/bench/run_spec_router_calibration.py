"""Public Engine AB/BA measurement; never fabricates routing profile coverage."""
from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import subprocess
import threading
import time
from pathlib import Path

from tools.dflash2_training.ab_suite import select_workloads


def percentile(values, percentile):
    ordered = sorted(values)
    if not ordered:
        raise ValueError("no latency samples")
    position = (len(ordered) - 1) * percentile / 100
    lo = math.floor(position)
    hi = math.ceil(position)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (position - lo)


def frontier_bucket(frontier):
    if not 1 <= frontier <= 32768:
        raise ValueError("execution frontier outside calibration domain")
    return "1-1024" if frontier <= 1024 else "1025-8192" if frontier <= 8192 else "8193-32768"


def output_identity(request):
    return {k: request[k] for k in ("generated_token_ids", "content", "reasoning",
                                    "tool_calls", "finish_reason", "matched_stop_string")}


def validate_report(report):
    if report.get("artifact_type") != "ninfer_calibration_measurement" or report.get("schema_version") != 1:
        raise ValueError("unsupported measurement report")
    for key, expected in (("kv", "int8"), ("graph", True), ("cache", True), ("greedy", True),
                          ("presence_penalty", 0), ("frequency_penalty", 0), ("proposal_head", "full")):
        if report.get(key) != expected:
            raise ValueError("incompatible measurement configuration: " + key)
    requests = report["requests"]
    clients = report["client_concurrency"]
    if not 1 <= clients <= report["engine_concurrency"] <= 8:
        raise ValueError("invalid client/engine concurrency")
    if report["repeats"] != len(report["wave_wall_ns"]):
        raise ValueError("missing repeat timing")
    expected = {(rep, slot) for rep in range(report["repeats"]) for slot in range(clients)}
    actual = [(r["repeat"], r["slot"]) for r in requests]
    if len(actual) != len(expected) or set(actual) != expected or not expected:
        raise ValueError("missing or duplicate requests")
    for request in requests:
        output_identity(request)
        if not request["generated_token_ids"] or request["latency_ns"] <= 0:
            raise ValueError("incomplete request")
        # None, ContextCapacity, Cancelled are not qualifying terminal responses.
        if request["finish_reason"] not in (1, 3, 4):
            raise ValueError("incomplete or failed terminal response")
    if not report["rounds"] or any(t <= 0 for t in report["wave_wall_ns"]):
        raise ValueError("missing round or end-to-end measurements")
    k = report["draft_tokens"]
    if k not in (0, 7, 11, 15):
        raise ValueError("unsupported fixed action")
    expected_event = {"draft_tokens": k, "verify_width": k + 1,
                      "proposal_width": k + 1 if k else 0,
                      "backend": 3 if k else 0, "neural_drafter_executed": bool(k)}
    for event in report["rounds"]:
        if any(event.get(key) != value for key, value in expected_event.items()):
            raise ValueError("observer action/physical width/proposal/skip mismatch")
        if not 1 <= event["active_batch"] <= report["engine_concurrency"]:
            raise ValueError("invalid actual batch")
        frontier_bucket(event["max_execution_frontier"])
        if event["elapsed_ns"] <= 0 or event["committed_tokens"] <= 0:
            raise ValueError("unsettled round")


def metrics(report):
    validate_report(report)
    latencies = [r["latency_ns"] / 1e6 for r in report["requests"]]
    buckets = {}
    round_times = {}
    for event in report["rounds"]:
        key = (event["active_batch"], frontier_bucket(event["max_execution_frontier"]))
        cell = buckets.setdefault(key, {"rounds": 0, "committed_tokens": 0, "elapsed_ns": 0})
        round_times.setdefault(key, []).append(event["elapsed_ns"])
        cell["rounds"] += 1
        cell["committed_tokens"] += event["committed_tokens"]
        cell["elapsed_ns"] += event["elapsed_ns"]
    cells = []
    for (batch, context), value in sorted(buckets.items()):
        k = report["draft_tokens"]
        cells.append({"active_batch": batch, "frontier_bucket": context, **value,
                      "draft_tokens": k, "verify_width": k + 1,
                      "proposal_width": k + 1 if k else 0, "backend": 3 if k else 0,
                      "neural_drafter_executed": bool(k),
                      "round_p50_ns": percentile(round_times[(batch, context)], 50),
                      "round_p95_ns": percentile(round_times[(batch, context)], 95),
                      "round_tokens_per_second": value["committed_tokens"] * 1e9 / value["elapsed_ns"]})
    return {"end_to_end_tokens_per_second": sum(len(r["generated_token_ids"]) for r in report["requests"])
            * 1e9 / sum(report["wave_wall_ns"]),
            "request_p50_ms": percentile(latencies, 50), "request_p95_ms": percentile(latencies, 95),
            "actual_batch_frontier_cells": cells}


def qualify_pairs(pairs):
    """Every correctness sample participates; performance ratios stay paired."""
    speedups, p95_ratios, failures = [], [], []
    canonical_output = None
    candidate_action = None
    for pair_index, (baseline, candidate) in enumerate(pairs):
        validate_report(baseline)
        validate_report(candidate)
        for field in ("model", "device", "client_concurrency", "engine_concurrency",
                      "max_context", "kv_capacity", "max_tokens", "kv", "graph", "cache",
                      "greedy", "presence_penalty", "frequency_penalty", "proposal_head", "repeats"):
            if baseline[field] != candidate[field]:
                raise ValueError("unpaired configuration: " + field)
        if baseline["draft_tokens"] != 0:
            raise ValueError("comparison baseline must be target-only")
        base_ids = {(r["repeat"], r["slot"]): output_identity(r) for r in baseline["requests"]}
        cand_ids = {(r["repeat"], r["slot"]): output_identity(r) for r in candidate["requests"]}
        if candidate_action is None:
            candidate_action = candidate["draft_tokens"]
        if candidate["draft_tokens"] != candidate_action:
            raise ValueError("candidate action changed between pairs")
        if canonical_output is None:
            canonical_output = base_ids
        if base_ids != canonical_output:
            failures.append({"pair": pair_index, "reason": "cross-pair baseline instability"})
        if base_ids != cand_ids:
            failures.append({"pair": pair_index, "reason": "full output mismatch"})
        for arm in (baseline, candidate):
            identities = [output_identity(r) for r in arm["requests"]]
            if any(identity != identities[0] for identity in identities[1:]):
                failures.append({"pair": pair_index, "reason": "repeat/slot instability"})
        base_metrics, cand_metrics = metrics(baseline), metrics(candidate)
        speedups.append(cand_metrics["end_to_end_tokens_per_second"] / base_metrics["end_to_end_tokens_per_second"])
        p95_ratios.append(cand_metrics["request_p95_ms"] / base_metrics["request_p95_ms"])
    if not pairs:
        raise ValueError("no paired measurements")
    correct = not failures
    return {"correct": correct, "failures": failures,
            "paired_speedups": speedups, "paired_p95_ratios": p95_ratios,
            "median_speedup": statistics.median(speedups) if correct else None,
            "qualified_performance": correct and all(s > 1 for s in speedups)
            and statistics.median(speedups) >= 1.02 and all(p <= 1.05 for p in p95_ratios)}


def run_measurement(command, directory, gpu, sample_seconds):
    directory.mkdir()
    stop = threading.Event()
    samples, sampler_errors = [], []

    def sample():
        while not stop.is_set():
            try:
                proc = subprocess.run(["nvidia-smi", "-i", str(gpu),
                                       "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                                      capture_output=True, text=True, timeout=5, check=True)
                samples.append({"monotonic_ns": time.monotonic_ns(), "used_mib": int(proc.stdout.strip())})
            except (OSError, ValueError, subprocess.SubprocessError) as error:
                sampler_errors.append(str(error))
                return
            stop.wait(sample_seconds)

    thread = threading.Thread(target=sample, daemon=True)
    thread.start()
    try:
        with (directory / "stdout.log").open("wb") as stdout, (directory / "stderr.log").open("wb") as stderr:
            result = subprocess.run(command, stdout=stdout, stderr=stderr, check=False)
    finally:
        stop.set()
        thread.join()
    memory = {"kind": "device_memory_observed_lower_bound", "gpu": str(gpu),
              "sample_interval_seconds": sample_seconds, "samples": samples,
              "peak_used_mib": max((s["used_mib"] for s in samples), default=None),
              "errors": sampler_errors}
    (directory / "memory.json").write_text(json.dumps(memory, indent=2) + "\n")
    (directory / "command.json").write_text(json.dumps(command, indent=2) + "\n")
    if result.returncode:
        raise RuntimeError(f"measurement failed ({result.returncode}): {directory / 'stderr.log'}")
    report = json.loads((directory / "measurement.json").read_text())
    validate_report(report)
    return report


def csv_ints(value, allowed):
    result = [int(item) for item in value.split(",")]
    if not result or len(set(result)) != len(result) or any(item not in allowed for item in result):
        raise argparse.ArgumentTypeError("invalid or duplicate selections")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exe", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--workloads", default="all")
    parser.add_argument("--concurrency", type=lambda v: csv_ints(v, range(1, 9)), default=[1, 2, 4, 8])
    parser.add_argument("--draft-tokens", type=lambda v: csv_ints(v, (7, 11, 15)), default=[7, 11, 15])
    parser.add_argument("--pairs", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--max-context", type=int, default=8192)
    parser.add_argument("--kv-capacity", type=int, default=8192)
    parser.add_argument("--engine-concurrency", type=int, default=8)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--nvml-device", required=True, help="physical index/UUID for nvidia-smi; distinct from CUDA ordinal")
    parser.add_argument("--memory-sample-seconds", type=float, default=0.05)
    parser.add_argument("--cooldown", type=float, default=2.0)
    args = parser.parse_args(argv)
    if args.pairs < 2 or args.repeats < 2 or args.max_tokens < 2 or args.memory_sample_seconds <= 0 or args.cooldown < 0:
        parser.error("need >=2 pairs/repeats, >=2 tokens and positive memory interval")
    if not 2 <= args.max_context <= 32768 or not 2 <= args.kv_capacity <= 262144:
        parser.error("max context must be 2..32768 and KV capacity 2..262144")
    if not args.model.is_file() or not args.exe.is_file():
        parser.error("explicit existing artifact and executable are required")
    if not max(args.concurrency) <= args.engine_concurrency <= 8:
        parser.error("engine concurrency must cover clients and be <=8")
    workloads = select_workloads(args.workloads)
    args.out.mkdir(parents=True, exist_ok=False)
    environment = {
        "cuda_visible_device_ordinal": args.device,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "nvml_physical_device": args.nvml_device,
        "memory_sampling": "observed device-wide lower bound; Engine logical peaks reported separately",
    }
    try:
        hardware = subprocess.run(
            ["nvidia-smi", "-i", args.nvml_device,
             "--query-gpu=name,uuid,driver_version,memory.total", "--format=csv"],
            capture_output=True, text=True, timeout=5, check=True)
        environment["hardware"] = hardware.stdout
    except (OSError, subprocess.SubprocessError) as error:
        environment["hardware_query_error"] = str(error)
    (args.out / "environment.json").write_text(json.dumps(environment, indent=2) + "\n")
    summary = {"artifact_type": "ninfer_calibration_comparison", "schema_version": 1,
               "comparisons": [], "coverage_note": "actual event cells only; no routing profile generated"}
    for workload in workloads:
        prompt_file = args.out / (workload.name + ".prompt")
        prompt_file.write_text(workload.prompt, encoding="utf-8")
        for clients in args.concurrency:
            for k in args.draft_tokens:
                pairs = []
                for pair in range(args.pairs):
                    reports = {}
                    order = (0, k) if pair % 2 == 0 else (k, 0)
                    for position, action in enumerate(order):
                        directory = args.out / f"{workload.name}-C{clients}-K{k}-p{pair}-o{position}-a{action}"
                        command = [str(args.exe.resolve()), "--model", str(args.model.resolve()),
                                   "--prompt-file", str(prompt_file.resolve()),
                                   "--output", str((directory / "measurement.json").resolve()),
                                   "--draft-tokens", str(action), "--client-concurrency", str(clients),
                                   "--engine-concurrency", str(args.engine_concurrency),
                                   "--repeats", str(args.repeats), "--max-tokens", str(args.max_tokens),
                                   "--max-context", str(args.max_context), "--kv-capacity", str(args.kv_capacity),
                                   "--device", str(args.device)]
                        reports[action] = run_measurement(command, directory, args.nvml_device,
                                                          args.memory_sample_seconds)
                        if args.cooldown:
                            time.sleep(args.cooldown)
                    pairs.append((reports[0], reports[k]))
                entry = {"workload": workload.name, "client_concurrency": clients, "draft_tokens": k,
                         **qualify_pairs(pairs),
                         "runs": [{"baseline": metrics(a), "candidate": metrics(b)} for a, b in pairs]}
                summary["comparisons"].append(entry)
                (args.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return 0 if all(e["correct"] for e in summary["comparisons"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
