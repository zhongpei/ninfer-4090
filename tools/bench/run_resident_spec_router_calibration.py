"""Measure resident-K15 actions through the public native Engine benchmark.

Priming must be selected explicitly. Prompt variants are UTF8 <workload>.txt files;
actual batch/frontier coverage always comes from round observations. Use
--proposal-compute selected to measure physically narrowed drafts; default full
preserves the original full-width proposal controls and measurement schema.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import subprocess
import sys
import threading
import time
from pathlib import Path

from tools.bench.calibrated_router_profile import (
    UPPERS, ROUND_FIELDS, REQUEST_FIELDS, analyse_record, boolean, build_profile,
    fields, integer, physical_proposal_width, text, unique_object, validate_identity,
)
from tools.dflash2_training.ab_suite import select_workloads


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def read_json(path):
    return json.loads(path.read_bytes(), object_pairs_hook=unique_object,
                      parse_constant=lambda value: (_ for _ in ()).throw(
                          ValueError("nonfinite JSON value: " + value)))


def selections(value, allowed):
    try:
        values = [int(item) for item in value.split(",")]
    except ValueError as error:
        raise argparse.ArgumentTypeError("integer list required") from error
    if not values or len(values) != len(set(values)) or any(v not in allowed for v in values):
        raise argparse.ArgumentTypeError("invalid or duplicate selections")
    return values


def suite_label(value):
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", value):
        raise argparse.ArgumentTypeError("suite label must contain only ASCII letters, digits, _, . or -")
    return value


def constant_profile(identity, action, proposal_compute="full"):
    validate_identity(identity)
    physical_proposal_width(action, proposal_compute)
    profile = {"schema_version": 1, "artifact_type": "ninfer_spec_router_profile",
               "identity": identity,
               "cells": [{"active_batch": batch, "frontier_upper": upper, "draft_tokens": action}
                         for batch in range(1, 9) for upper in UPPERS],
               "provenance": {"purpose": "measurement_control", "qualified": False,
                              "resident_startup_draft_tokens": 15}}
    if proposal_compute == "selected":
        profile.update(schema_version=2, proposal_compute="selected")
    return profile


def execute_native(command, directory, gpu, sample_seconds):
    """Retain stdout/stderr/command and device-wide sampled lower bounds on failure too."""
    directory.mkdir(parents=True, exist_ok=False)
    write_json(directory / "command.json", command)
    stop = threading.Event()
    samples, errors = [], []

    def sampler():
        while True:
            try:
                result = subprocess.run(["nvidia-smi", "-i", str(gpu),
                                         "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                                        capture_output=True, text=True, timeout=5, check=True)
                samples.append({"monotonic_ns": time.monotonic_ns(), "used_mib": int(result.stdout.strip())})
            except (OSError, ValueError, subprocess.SubprocessError) as error:
                errors.append(str(error))
                return
            if stop.wait(sample_seconds):
                return

    thread = threading.Thread(target=sampler, daemon=True)
    thread.start()
    process = None
    failure = None
    try:
        with (directory / "stdout.log").open("wb") as stdout, (directory / "stderr.log").open("wb") as stderr:
            process = subprocess.run(command, stdout=stdout, stderr=stderr, check=False)
    except (OSError, subprocess.SubprocessError) as error:
        failure = error
    finally:
        stop.set()
        thread.join()
        write_json(directory / "memory.json",
                   {"kind": "device_memory_observed_lower_bound", "gpu": str(gpu),
                    "sample_interval_seconds": sample_seconds, "samples": samples,
                    "peak_used_mib": max((s["used_mib"] for s in samples), default=None),
                    "errors": errors,
                    "scope": "device-wide sampled use, distinct from Engine logical/allocator peaks"})
    if failure:
        raise RuntimeError(f"native execution failed: {directory}: {failure}") from failure
    if process.returncode:
        raise RuntimeError(f"native exit {process.returncode}: {directory / 'stderr.log'}")


def validate_priming(priming, enabled, action, proposal_compute="full"):
    expected_proposal_width = physical_proposal_width(action, proposal_compute)
    if not isinstance(priming, dict) or type(priming.get("enabled")) is not bool or priming["enabled"] != enabled:
        raise ValueError("native priming selection mismatch")
    if not enabled:
        fields(priming, ("enabled",), "disabled priming")
        return
    fields(priming, ("enabled", "requested_output_tokens", "request", "rounds",
                     "wave_wall_ns", "settled_committed_tokens"), "priming")
    if integer(priming["requested_output_tokens"], "primer budget") != 16:
        raise ValueError("primer must request 16 output-limit tokens")
    request = priming["request"]
    fields(request, REQUEST_FIELDS, "primer response")
    if (integer(request["finish_reason"], "primer finish") != 1
            or len(request["generated_token_ids"]) != 16):
        raise ValueError("primer did not complete 16 output-limit tokens")
    for token in request["generated_token_ids"]:
        integer(token, "primer token", 0, (1 << 31) - 1)
    for key in ("content", "reasoning"):
        text(request[key], "primer " + key)
    integer(request["latency_ns"], "primer latency", 1, (1 << 64) - 1)
    integer(priming["wave_wall_ns"], "primer wall", 1, (1 << 64) - 1)
    if request["latency_ns"] > priming["wave_wall_ns"]:
        raise ValueError("primer request latency exceeds settled priming wall")
    events = priming["rounds"]
    if not isinstance(events, list) or not events:
        raise ValueError("primer has no settled round evidence")
    commits = 0
    for index, event in enumerate(events):
        fields(event, ROUND_FIELDS, "primer round")
        if integer(event["round_index"], "primer round_index") != index:
            raise ValueError("primer round indices not continuous")
        if integer(event["active_batch"], "primer batch") != 1:
            raise ValueError("primer must be one request")
        for key in ("draft_tokens", "verify_width", "proposal_width", "backend"):
            integer(event[key], "primer " + key)
        boolean(event["neural_drafter_executed"], "primer neural_drafter_executed")
        if (event["draft_tokens"] != action or event["verify_width"] != action + 1
                or event["proposal_width"] != expected_proposal_width or event["backend"] != 3
                or event["neural_drafter_executed"] != bool(action)):
            raise ValueError("primer resident physical action mismatch")
        commits += integer(event["committed_tokens"], "primer commits", 0, action + 1)
        integer(event["elapsed_ns"], "primer round elapsed", 1, (1 << 64) - 1)
        integer(event["max_execution_frontier"], "primer frontier", 1, 32768)
    if commits != 15 or integer(priming["settled_committed_tokens"], "settled primer commits") != commits:
        raise ValueError("primer settle accounting mismatch")


def validate_wrapper(wrapper, identity, action, configuration, prime, proposal_compute="full"):
    fields(wrapper, ("artifact_type", "schema_version", "record", "resources", "priming"), "native wrapper")
    if wrapper["artifact_type"] != "ninfer_resident_router_measurement" or type(wrapper["schema_version"]) is not int or wrapper["schema_version"] != 1:
        raise ValueError("unsupported native resident measurement wrapper")
    analysis = analyse_record(wrapper["record"], identity, action, proposal_compute)
    if wrapper["record"]["configuration"] != configuration:
        raise ValueError("native request configuration differs from driver command")
    resources = wrapper["resources"]
    if not isinstance(resources, dict) or not isinstance(resources.get("memory"), dict):
        raise ValueError("missing Engine resource evidence")
    for key in ("workspace_logical_peak_bytes", "workspace_allocator_peak_bytes", "runtime_reservation_bytes"):
        integer(resources["memory"].get(key), key, 1, (1 << 64) - 1)
    if resources.get("routing_counters_scope") != "published_engine_snapshot_including_priming":
        raise ValueError("cumulative routing snapshot scope not declared")
    validate_priming(wrapper["priming"], prime, action, proposal_compute)
    return analysis


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exe", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--workloads", default="all")
    parser.add_argument("--prompt-dir", type=Path)
    parser.add_argument("--workload-label-prefix", type=suite_label, required=True,
                        help="campaign suite name; comparison labels are SUITE/workload/Crequested (not actual batch)")
    parser.add_argument("--concurrency", type=lambda v: selections(v, range(1, 9)), default=[1, 2, 4, 8])
    parser.add_argument("--draft-tokens", type=lambda v: selections(v, (7, 11, 15)), default=[7, 11, 15])
    parser.add_argument("--proposal-compute", choices=("full", "selected"), default="full")
    parser.add_argument("--pairs", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--max-context", type=int, default=32768)
    parser.add_argument("--kv-capacity", type=int, default=32768)
    parser.add_argument("--engine-concurrency", type=int, default=8)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--nvml-device", required=True)
    parser.add_argument("--memory-sample-seconds", type=float, default=0.05)
    parser.add_argument("--cooldown", type=float, default=2)
    priming = parser.add_mutually_exclusive_group(required=True)
    priming.add_argument("--prime-prefix", dest="prime", action="store_true")
    priming.add_argument("--no-prime-prefix", dest="prime", action="store_false")
    args = parser.parse_args(argv)
    if (args.pairs < 2 or args.repeats < 2 or not 2 <= args.max_tokens <= 32768
            or not 2 <= args.max_context <= 32768 or not 2 <= args.kv_capacity <= 262144
            or not max(args.concurrency) <= args.engine_concurrency <= 8
            or not math.isfinite(args.cooldown) or not math.isfinite(args.memory_sample_seconds)
            or args.cooldown < 0 or args.memory_sample_seconds <= 0):
        parser.error("invalid measurement counts, context/capacity/concurrency or timing options")
    if not args.exe.is_file() or not args.model.is_file():
        parser.error("explicit existing native executable and artifact are required")
    workloads = select_workloads(args.workloads)
    args.out.mkdir(parents=True, exist_ok=False)
    environment = {"cuda_visible_device_ordinal": args.device,
                   "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                   "nvml_physical_device": args.nvml_device, "prime_prefix": args.prime,
                   "proposal_compute": args.proposal_compute,
                   "workload_label_prefix": args.workload_label_prefix,
                   "requested_max_context": args.max_context, "requested_kv_capacity": args.kv_capacity,
                   "sampling": "greedy, zero presence/frequency penalties, thinking off",
                   "timing": "generation waves exclude priming; Engine counters remain cumulative snapshots"}
    document = None
    try:
        hardware = subprocess.run(["nvidia-smi", "-i", args.nvml_device,
                                   "--query-gpu=name,uuid,driver_version,memory.total", "--format=csv"],
                                  capture_output=True, text=True, timeout=5, check=True)
        environment["hardware"] = hardware.stdout
    except (OSError, subprocess.SubprocessError) as error:
        environment["hardware_query_error"] = str(error)
    write_json(args.out / "environment.json", environment)
    base = [str(args.exe.resolve()), "--model", str(args.model.resolve()),
            "--engine-concurrency", str(args.engine_concurrency), "--max-context", str(args.max_context),
            "--kv-capacity", str(args.kv_capacity), "--device", str(args.device)]
    try:
        seed_dir = args.out / "identity"
        seed_path = seed_dir / "native.json"
        execute_native([*base, "--identity-only", "--draft-tokens", "15", "--output", str(seed_path.resolve())],
                       seed_dir, args.nvml_device, args.memory_sample_seconds)
        seed = read_json(seed_path)
        fields(seed, ("artifact_type", "schema_version", "identity"), "identity wrapper")
        if seed["artifact_type"] != "ninfer_routing_identity" or type(seed["schema_version"]) is not int or seed["schema_version"] != 1:
            raise ValueError("native did not publish routing identity")
        identity = seed["identity"]
        validate_identity(identity)
        if (identity["max_concurrency"] != args.engine_concurrency or identity["max_context"] != args.max_context
                or identity["resolved_kv_capacity"] < args.kv_capacity or identity["proposal_head"] != "full"):
            raise ValueError("native startup identity differs from requested configuration")
        controls_dir = args.out / "controls"
        controls_dir.mkdir()
        controls = {}
        for action in (0, 7, 11, 15):
            controls[action] = controls_dir / f"action-{action}.json"
            write_json(controls[action], constant_profile(identity, action, args.proposal_compute))
        prompts_dir = args.out / "prompts"
        prompts_dir.mkdir()
        document = {"schema_version": 1, "artifact_type": "ninfer_resident_router_measurements",
                    "identity": identity, "comparisons": []}
        if args.proposal_compute == "selected":
            document.update(schema_version=2, proposal_compute="selected")
        for workload in workloads:
            prompt = ((args.prompt_dir / (workload.name + ".txt")).read_bytes().decode("utf-8")
                      if args.prompt_dir else workload.prompt)
            text(prompt, "workload prompt", True)
            prompt_path = prompts_dir / (workload.name + ".txt")
            prompt_path.write_bytes(prompt.encode("utf-8"))
            for clients in args.concurrency:
                configuration = {"prompt": prompt, "max_tokens": args.max_tokens, "client_concurrency": clients,
                                 "repeats": args.repeats,
                                 "sampling": {"temperature": 0, "presence_penalty": 0, "frequency_penalty": 0}}
                for candidate in args.draft_tokens:
                    comparison = {"workload": f"{args.workload_label_prefix}/{workload.name}/C{clients}",
                                  "candidate_action": candidate, "pairs": []}
                    document["comparisons"].append(comparison)
                    for pair in range(args.pairs):
                        order = [0, candidate] if pair % 2 == 0 else [candidate, 0]
                        records = {}
                        for position, action in enumerate(order):
                            directory = args.out / "runs" / f"{workload.name}-C{clients}-K{candidate}-p{pair}-o{position}-a{action}"
                            report_path = directory / "native.json"
                            command = [*base, "--prompt-file", str(prompt_path.resolve()),
                                       "--output", str(report_path.resolve()), "--draft-tokens", str(action),
                                       "--spec-router-profile", str(controls[action].resolve()),
                                       "--client-concurrency", str(clients), "--repeats", str(args.repeats),
                                       "--max-tokens", str(args.max_tokens)]
                            if args.prime:
                                command.append("--prime-prefix")
                            execute_native(command, directory, args.nvml_device, args.memory_sample_seconds)
                            wrapper = read_json(report_path)
                            analysis = validate_wrapper(wrapper, identity, action, configuration, args.prime,
                                                        args.proposal_compute)
                            write_json(directory / "actual-cell-statistics.json", analysis)
                            records[action] = wrapper["record"]
                            if args.cooldown:
                                time.sleep(args.cooldown)
                        comparison["pairs"].append({"pair_id": str(pair), "order": order,
                                                    "baseline": records[0], "candidate": records[candidate]})
                        write_json(args.out / "measurements.json", document)
        path = args.out / "measurements.json"
        raw = path.read_bytes()
        profile = build_profile(document, {"path": str(path.resolve()), "sha256": hashlib.sha256(raw).hexdigest()})
        summary = {"artifact_type": "ninfer_resident_calibration_summary", "schema_version": 1,
                   "identity": identity, "cells": profile["cells"], "provenance": profile["provenance"],
                   "prime_prefix": args.prime, "proposal_compute": args.proposal_compute}
        write_json(args.out / "qualification-summary.json", summary)
        correct = all(not any("output" in reason or "instability" in reason
                              for reason in comparison["reasons"])
                      for comparison in profile["provenance"]["comparisons"])
        return 0 if correct else 1
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as error:
        write_json(args.out / "failure.json", {"error": str(error)})
        if document is not None:
            write_json(args.out / "measurements.json", document)
        print(f"resident calibration: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
