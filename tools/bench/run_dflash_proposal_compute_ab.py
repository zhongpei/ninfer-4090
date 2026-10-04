"""Paired full/selected proposal computation with the SAME resident K15 and selected action.

K7/K11 isolate physical drafter work. K0 (both skip) and K15 (both full) are A/A
controls. This experiment never manufactures an Auto profile or an Auto speedup.
All timings exclude Engine startup; complete raw responses, rounds and resource
reports are retained. Run as python -m tools.bench.run_dflash_proposal_compute_ab.
"""
from __future__ import annotations

import argparse
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

from tools.bench.calibrated_router_profile import fields, validate_identity
from tools.bench.run_resident_spec_router_calibration import (
    constant_profile, execute_native, read_json, selections, validate_wrapper, write_json,
)
from tools.dflash2_training.ab_suite import select_workloads


def qualify_compute_pairs(pairs, action):
    if action not in (0, 7, 11, 15) or type(action) is not int:
        raise ValueError("unsupported action")
    if not isinstance(pairs, list) or len(pairs) < 2:
        raise ValueError("at least two retained AB/BA pairs required")
    orientations, ids, signatures, cells = set(), set(), set(), set()
    reasons, ratios, latencies = [], [], []
    for pair in pairs:
        fields(pair, ("pair_id", "order", "full", "selected"), "compute pair")
        if pair["pair_id"] in ids:
            raise ValueError("duplicate pair id")
        ids.add(pair["pair_id"])
        order = tuple(pair["order"])
        if order not in (("full", "selected"), ("selected", "full")):
            raise ValueError("invalid physical compute order")
        orientations.add(order)
        for mode in ("full", "selected"):
            run = pair[mode]
            signatures.add((run["configuration_key"], run["response_key"]))
            cells.add(tuple(run["cell"]) if run["cell"] else None)
            if not run["stable_response"]:
                reasons.append("repeat/slot output instability")
        ratios.append(pair["selected"]["throughput"] / pair["full"]["throughput"])
        latencies.append(pair["selected"]["request_p95_ns"] / pair["full"]["request_p95_ns"])
    if len(orientations) != 2:
        raise ValueError("both full/selected and selected/full orders are required")
    if len(signatures) != 1:
        reasons.append("complete output or configuration mismatch across retained arms/pairs")
    correct = not reasons
    covered = len(cells) == 1 and None not in cells
    control = action in (0, 15)
    qualified = (correct and covered and not control and all(r > 1 for r in ratios)
                 and statistics.median(ratios) >= 1.02 and all(r <= 1.05 for r in latencies))
    return {"correct": correct, "failures": sorted(set(reasons)), "is_aa_control": control,
            "paired_speedups": ratios, "paired_p95_ratios": latencies,
            "median_speedup": statistics.median(ratios) if correct else None,
            "common_dominant_actual_cell": list(next(iter(cells))) if covered else None,
            "qualified_compute_speedup": qualified,
            "qualification": "all complete outputs match; AB and BA; every pair faster; median >=1.02; "
                             "every paired p95 <=1.05; common >=95% actual batch/frontier cell; not an A/A control"}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("exe", "model", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--nvml-device", required=True)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--workloads", default="all")
    parser.add_argument("--prompt-dir", type=Path)
    parser.add_argument("--actions", type=lambda v: selections(v, (0, 7, 11, 15)), default=[0, 7, 11, 15])
    parser.add_argument("--concurrency", type=lambda v: selections(v, range(1, 9)), default=[1, 2, 4, 8])
    parser.add_argument("--engine-concurrency", type=int, default=8)
    parser.add_argument("--max-context", type=int, default=32768)
    parser.add_argument("--kv-capacity", type=int, default=32768)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--pairs", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--cooldown", type=float, default=2)
    parser.add_argument("--memory-sample-seconds", type=float, default=.05)
    prime = parser.add_mutually_exclusive_group(required=True)
    prime.add_argument("--prime-prefix", dest="prime", action="store_true")
    prime.add_argument("--no-prime-prefix", dest="prime", action="store_false")
    args = parser.parse_args(argv)
    import math
    if (args.pairs < 2 or args.repeats < 2 or not 2 <= args.max_tokens <= args.max_context <= 32768
            or not args.max_context <= args.kv_capacity <= 262144
            or not max(args.concurrency) <= args.engine_concurrency <= 8
            or not 0 <= args.device <= 255 or not math.isfinite(args.cooldown) or args.cooldown < 0
            or not math.isfinite(args.memory_sample_seconds) or args.memory_sample_seconds <= 0):
        parser.error("invalid bounds, counts or sampling interval")
    if not args.exe.is_file() or not args.model.is_file():
        parser.error("existing native executable and artifact required")
    workloads = select_workloads(args.workloads)
    args.out.mkdir(parents=True, exist_ok=False)
    summary = {"artifact_type": "ninfer_dflash_proposal_compute_ab", "schema_version": 1,
               "timing_scope": __doc__.strip(), "comparisons": [], "correct": False,
               "prime_prefix": args.prime}
    base = [str(args.exe.resolve()), "--model", str(args.model.resolve()),
            "--engine-concurrency", str(args.engine_concurrency), "--max-context", str(args.max_context),
            "--kv-capacity", str(args.kv_capacity), "--device", str(args.device)]
    try:
        environment = {"cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                       "cuda_visible_device_ordinal": args.device, "nvml_physical_device": args.nvml_device}
        try:
            environment["hardware"] = subprocess.run(
                ["nvidia-smi", "-i", args.nvml_device,
                 "--query-gpu=name,uuid,driver_version,memory.total", "--format=csv"],
                capture_output=True, text=True, timeout=5, check=True).stdout
        except (OSError, subprocess.SubprocessError) as error:
            environment["hardware_query_error"] = str(error)
        write_json(args.out / "environment.json", environment)
        directory = args.out / "identity"
        path = directory / "native.json"
        execute_native([*base, "--identity-only", "--draft-tokens", "15", "--output", str(path.resolve())],
                       directory, args.nvml_device, args.memory_sample_seconds)
        seed = read_json(path)
        fields(seed, ("schema_version", "artifact_type", "identity"), "identity wrapper")
        if type(seed["schema_version"]) is not int or seed["schema_version"] != 1 or seed["artifact_type"] != "ninfer_routing_identity":
            raise ValueError("unsupported identity wrapper")
        identity = seed["identity"]
        validate_identity(identity)
        if (identity["max_context"] != args.max_context or identity["max_concurrency"] != args.engine_concurrency
                or identity["resolved_kv_capacity"] < args.kv_capacity or identity["proposal_head"] != "full"):
            raise ValueError("public startup identity differs from requested configuration")
        summary["identity"] = identity
        controls = {}
        (args.out / "controls").mkdir()
        for action in args.actions:
            for mode in ("full", "selected"):
                path = args.out / "controls" / f"{mode}-k{action}.json"
                write_json(path, constant_profile(identity, action, mode))
                controls[mode, action] = path
        (args.out / "prompts").mkdir()
        for workload in workloads:
            prompt = ((args.prompt_dir / (workload.name + ".txt")).read_bytes().decode("utf-8")
                      if args.prompt_dir else workload.prompt)
            if not prompt:
                raise ValueError("empty workload prompt")
            prompt_path = args.out / "prompts" / (workload.name + ".txt")
            prompt_path.write_bytes(prompt.encode("utf-8"))
            for clients in args.concurrency:
                config = {"prompt": prompt, "max_tokens": args.max_tokens, "client_concurrency": clients,
                          "repeats": args.repeats, "sampling": {"temperature": 0, "presence_penalty": 0, "frequency_penalty": 0}}
                for action in args.actions:
                    comparison = {"workload": workload.name, "requested_clients": clients,
                                  "selected_draft_tokens": action, "pairs": []}
                    summary["comparisons"].append(comparison)
                    for index in range(args.pairs):
                        order = ["full", "selected"] if index % 2 == 0 else ["selected", "full"]
                        pair = {"pair_id": str(index), "order": order}
                        for mode in order:
                            directory = args.out / "runs" / f"{workload.name}-C{clients}-K{action}-p{index}-{mode}"
                            path = directory / "native.json"
                            command = [*base, "--prompt-file", str(prompt_path.resolve()),
                                       "--output", str(path.resolve()), "--draft-tokens", str(action),
                                       "--spec-router-profile", str(controls[mode, action].resolve()),
                                       "--client-concurrency", str(clients), "--repeats", str(args.repeats),
                                       "--max-tokens", str(args.max_tokens)]
                            if args.prime:
                                command.append("--prime-prefix")
                            execute_native(command, directory, args.nvml_device, args.memory_sample_seconds)
                            raw = read_json(path)
                            analysis = validate_wrapper(raw, identity, action, config, args.prime, mode)
                            write_json(directory / "analysis.json", analysis)
                            pair[mode] = analysis
                            if args.cooldown:
                                time.sleep(args.cooldown)
                        comparison["pairs"].append(pair)
                        write_json(args.out / "comparison.json", summary)
                    comparison["qualification"] = qualify_compute_pairs(comparison["pairs"], action)
                    write_json(args.out / "comparison.json", summary)
        summary["correct"] = bool(summary["comparisons"]) and all(
            item["qualification"]["correct"] for item in summary["comparisons"])
        write_json(args.out / "comparison.json", summary)
        # A correct negative performance result is a completed experiment, NOT a promotion.
        return 0 if summary["correct"] else 1
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as error:
        write_json(args.out / "failure.json", {"error": str(error)})
        write_json(args.out / "comparison.json", summary)
        print("proposal compute A/B: " + str(error), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
