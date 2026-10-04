"""Build an audited resident-K15 routing profile from paired public Engine measurements.

Input schema1: ninfer_resident_router_measurements {identity, comparisons}.
Each comparison is {workload, candidate_action, pairs}; each pair is
{pair_id, order: [0,K] or [K,0], baseline, candidate}. A record contains the complete
public identity, requested_action, configuration {prompt, max_tokens,
client_concurrency, repeats, sampling {temperature, presence_penalty,
frequency_penalty}}, requests (native complete responses), raw rounds and
wave_wall_ns. Each raw round also has a collector-owned round_index, continuous
from zero in record order; it is not a public observer field. No standalone-K measurement or client-concurrency coverage is inferred.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import re
import statistics
import struct
from collections import defaultdict
from pathlib import Path

UPPERS = (1024, 8192, 32768)
ACTIONS = (0, 7, 11, 15)
CACHE_FIELDS = ("enabled", "extra_device_state_slots", "host_state_slots",
                "host_kv_capacity_bytes", "max_private_continuations", "max_shared_prefixes",
                "max_long_anchors_per_continuation", "max_cache_markers_per_request")
EXECUTION_FLAGS = ("lm_head_q4", "lm_head_q6", "embedding_q4", "embedding_q6",
                   "gdn_state_fp16", "mlp_a8_decode", "prefill_a8", "prefill_cublas",
                   "prefill_cublas_projections", "mtp_experts_q4", "enable_vision")
EXECUTION_FIELDS = (*EXECUTION_FLAGS, "vision_residency", "vision_max_merged_tokens",
                    "rope_scaling_factor", "rope_scaling_original_context")
IDENTITY_FIELDS = ("artifact_id", "prefill_signature", "hardware_class", "backend",
                   "kv_storage", "startup_draft_tokens", "proposal_head", "use_cuda_graph",
                   "max_concurrency", "max_context", "prefill_chunk", "resolved_kv_capacity",
                   "context_cache", "execution_options")
REQUEST_FIELDS = ("repeat", "slot", "latency_ns", "prompt_tokens", "finish_reason",
                  "prefix_reuse_path", "reused_prompt_tokens", "content", "reasoning",
                  "matched_stop_string", "tool_calls", "generated_token_ids")
ROUND_FIELDS = ("round_index", "active_batch", "max_execution_frontier", "verify_width", "draft_tokens",
                "proposal_width", "backend", "neural_drafter_executed",
                "committed_tokens", "elapsed_ns")


def fields(value, expected, label):
    if not isinstance(value, dict) or set(value) != set(expected):
        raise ValueError(f"{label}: fields do not match")


def integer(value, label, low=0, high=(1 << 32) - 1):
    if type(value) is not int or not low <= value <= high:
        raise ValueError(f"{label}: integer outside {low}..{high}")
    return value


def finite(value, label):
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError(f"{label}: finite number required")
    return value


def boolean(value, label):
    if type(value) is not bool:
        raise ValueError(f"{label}: bool required")


def text(value, label, nonempty=False):
    if not isinstance(value, str) or (nonempty and not value):
        raise ValueError(f"{label}: string required")


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def validate_identity(identity):
    """Exact field/type mapping of runtime routing_profile_identity_json."""
    fields(identity, IDENTITY_FIELDS, "identity")
    for key in ("artifact_id", "prefill_signature", "hardware_class"):
        text(identity[key], key, True)
    if not re.fullmatch("[0-9a-f]{32}", identity["artifact_id"]):
        raise ValueError("artifact_id: expected artifact directory UUID")
    if (identity["backend"] != "dflash2" or identity["kv_storage"] != "int8"
            or identity["startup_draft_tokens"] != 15
            or type(identity["startup_draft_tokens"]) is not int):
        raise ValueError("identity: resident DFlash2 INT8 K15 required")
    if identity["proposal_head"] not in ("full", "optimized"):
        raise ValueError("identity: invalid proposal head")
    boolean(identity["use_cuda_graph"], "use_cuda_graph")
    if not identity["use_cuda_graph"]:
        raise ValueError("identity: graph-on required")
    integer(identity["max_concurrency"], "max_concurrency", 1, 8)
    for key in ("max_context", "prefill_chunk", "resolved_kv_capacity"):
        integer(identity[key], key, 1)
    cache = identity["context_cache"]
    fields(cache, CACHE_FIELDS, "context_cache")
    boolean(cache["enabled"], "context_cache.enabled")
    if not cache["enabled"]:
        raise ValueError("identity: cache-on required")
    for key in CACHE_FIELDS[1:]:
        integer(cache[key], "context_cache." + key, high=(1 << 64) - 1
                if key == "host_kv_capacity_bytes" else (1 << 32) - 1)
    execution = identity["execution_options"]
    fields(execution, EXECUTION_FIELDS, "execution_options")
    for key in EXECUTION_FLAGS:
        boolean(execution[key], "execution_options." + key)
    if execution["vision_residency"] not in ("resident", "overlay"):
        raise ValueError("identity: invalid vision residency")
    for key in ("vision_max_merged_tokens", "rope_scaling_original_context"):
        integer(execution[key], key, 1)
    scale = finite(execution["rope_scaling_factor"], "rope_scaling_factor")
    try:
        scale32 = struct.unpack("f", struct.pack("f", scale))[0]
    except (OverflowError, struct.error) as error:
        raise ValueError("rope_scaling_factor: outside FP32") from error
    if not math.isfinite(scale32) or scale32 <= 0:
        raise ValueError("rope_scaling_factor: positive finite FP32 required")
    normalized = copy.deepcopy(identity)
    normalized["execution_options"]["rope_scaling_factor"] = scale32
    return canonical(normalized)


def percentile(values, percent):
    ordered = sorted(values)
    position = (len(ordered) - 1) * percent / 100
    lo, hi = math.floor(position), math.ceil(position)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (position - lo)


def cell_of(event):
    frontier = integer(event["max_execution_frontier"], "max_execution_frontier", 1, 32768)
    return (event["active_batch"], next(upper for upper in UPPERS if frontier <= upper))


def analyse_record(record, identity, action):
    fields(record, ("identity", "requested_action", "configuration", "requests",
                    "rounds", "wave_wall_ns"), "record")
    if validate_identity(record["identity"]) != validate_identity(identity):
        raise ValueError("record identity does not match public startup identity")
    if integer(record["requested_action"], "requested_action") != action:
        raise ValueError("requested action does not match paired arm")
    config = record["configuration"]
    fields(config, ("prompt", "max_tokens", "client_concurrency", "repeats", "sampling"), "configuration")
    text(config["prompt"], "prompt", True)
    clients = integer(config["client_concurrency"], "client_concurrency", 1, identity["max_concurrency"])
    repeats = integer(config["repeats"], "repeats", 2)
    budget = integer(config["max_tokens"], "max_tokens", 2, identity["max_context"])
    sampling = config["sampling"]
    fields(sampling, ("temperature", "presence_penalty", "frequency_penalty"), "sampling")
    for key, value in sampling.items():
        if finite(value, key) != 0:
            raise ValueError("only greedy zero-penalty calibration is supported")
    normalized_config = copy.deepcopy(config)
    normalized_config["sampling"] = dict.fromkeys(sampling, 0)
    config_key = canonical(normalized_config)
    walls = record["wave_wall_ns"]
    if not isinstance(walls, list) or len(walls) != repeats:
        raise ValueError("missing wave timings")
    for wall in walls:
        integer(wall, "wave_wall_ns", 1, (1 << 64) - 1)
    requests = record["requests"]
    if not isinstance(requests, list) or len(requests) != clients * repeats:
        raise ValueError("missing requests")
    outputs = {}
    response_keys = []
    latencies = []
    token_count = 0
    for request in requests:
        fields(request, REQUEST_FIELDS, "request")
        rep = integer(request["repeat"], "repeat", 0, repeats - 1)
        slot = integer(request["slot"], "slot", 0, clients - 1)
        if (rep, slot) in outputs:
            raise ValueError("duplicate request")
        latency = integer(request["latency_ns"], "latency_ns", 1, (1 << 64) - 1)
        if latency > walls[rep]:
            raise ValueError("latency exceeds enclosing wave")
        latencies.append(latency)
        prompt_tokens = integer(request["prompt_tokens"], "prompt_tokens", 1, identity["max_context"])
        integer(request["prefix_reuse_path"], "prefix_reuse_path")
        integer(request["reused_prompt_tokens"], "reused_prompt_tokens", 0, prompt_tokens)
        finish = integer(request["finish_reason"], "finish_reason")
        if finish not in (1, 3, 4):
            raise ValueError("incomplete terminal response")
        tokens = request["generated_token_ids"]
        if not isinstance(tokens, list) or not 1 <= len(tokens) <= budget:
            raise ValueError("invalid complete output budget")
        if finish == 1 and len(tokens) != budget:
            raise ValueError("output-limit response is incomplete")
        for token in tokens:
            integer(token, "token id", 0, (1 << 31) - 1)
        for key in ("content", "reasoning"):
            text(request[key], key)
        if request["matched_stop_string"] is not None:
            text(request["matched_stop_string"], "matched_stop_string")
        if not isinstance(request["tool_calls"], list):
            raise ValueError("tool_calls must be an array")
        for call in request["tool_calls"]:
            fields(call, ("name", "arguments_json"), "tool call")
            text(call["name"], "tool name")
            text(call["arguments_json"], "tool arguments")
        response = {key: request[key] for key in ("generated_token_ids", "content", "reasoning",
                                                 "tool_calls", "finish_reason", "matched_stop_string")}
        response["prompt_tokens"] = prompt_tokens
        key = canonical(response)
        outputs[(rep, slot)] = key
        response_keys.append(key)
        token_count += len(tokens)
    events = record["rounds"]
    if not isinstance(events, list) or not events:
        raise ValueError("no raw settled rounds")
    cell_stats = {}
    committed = 0
    for round_index, event in enumerate(events):
        fields(event, ROUND_FIELDS, "round")
        if integer(event["round_index"], "round_index") != round_index:
            raise ValueError("round_index must be unique, continuous and in collector order")
        batch = integer(event["active_batch"], "active_batch", 1, clients)
        cell = cell_of(event)
        if event["max_execution_frontier"] > identity["max_context"]:
            raise ValueError("frontier exceeds identity context")
        for key in ("draft_tokens", "verify_width", "proposal_width", "backend"):
            integer(event[key], key)
        boolean(event["neural_drafter_executed"], "neural_drafter_executed")
        if (event["draft_tokens"] != action or event["verify_width"] != action + 1
                or event["proposal_width"] != (16 if action else 0)
                or event["backend"] != 3 or event["neural_drafter_executed"] != bool(action)):
            raise ValueError("resident-K15 physical action contract mismatch")
        commit = integer(event["committed_tokens"], "committed_tokens", 0, batch * (action + 1))
        elapsed = integer(event["elapsed_ns"], "elapsed_ns", 1, (1 << 64) - 1)
        committed += commit
        stats = cell_stats.setdefault(cell, {"rounds": 0, "elapsed_ns": 0, "committed_tokens": 0})
        stats["rounds"] += 1
        stats["elapsed_ns"] += elapsed
        stats["committed_tokens"] += commit
    elapsed = sum(cell["elapsed_ns"] for cell in cell_stats.values())
    wall = sum(walls)
    # The final commit/statistics publication may finish after the request waiter. Full
    # round and request wave timings are distinct boundaries, with no enclosure relation.
    if committed == 0:
        raise ValueError("run has no committed decode tokens")
    if all(r["finish_reason"] == 1 for r in requests) and committed != token_count - len(requests):
        raise ValueError("decode commit/first-token accounting mismatch")
    dominant = max(cell_stats, key=lambda cell: cell_stats[cell]["elapsed_ns"])
    dominant_ns = cell_stats[dominant]["elapsed_ns"]
    coverage = dominant if dominant_ns * 100 >= elapsed * 95 else None
    stats = [{"active_batch": cell[0], "frontier_upper": cell[1], **value,
              "round_elapsed_fraction": value["elapsed_ns"] / elapsed}
             for cell, value in sorted(cell_stats.items())]
    return {"configuration_key": config_key,
            "response_key": canonical([[rep, slot, key] for (rep, slot), key in sorted(outputs.items())]),
            "stable_response": len(set(response_keys)) == 1,
            "tokens": token_count, "wave_wall_ns": wall, "full_round_elapsed_ns": elapsed,
            "throughput": token_count * 1e9 / wall, "request_p95_ns": percentile(latencies, 95),
            "cell": coverage, "dominant_cell": dominant, "dominant_fraction": dominant_ns / elapsed,
            "cells": stats}


def analyse_comparison(comparison, identity):
    fields(comparison, ("workload", "candidate_action", "pairs"), "comparison")
    text(comparison["workload"], "workload", True)
    action = integer(comparison["candidate_action"], "candidate_action")
    if action not in (7, 11, 15):
        raise ValueError("comparison candidate action must be 7/11/15")
    audit = {"workload": comparison["workload"], "candidate_action": action,
             "qualified": False, "reasons": [], "pairs": []}
    pairs = comparison["pairs"]
    if not isinstance(pairs, list):
        raise ValueError("pairs must be an array")
    analysed = []
    ids = set()
    orientations = set()
    baseline_signatures = set()
    all_cells = set()
    valid_records = 0
    for pair in pairs:
        fields(pair, ("pair_id", "order", "baseline", "candidate"), "pair")
        text(pair["pair_id"], "pair_id", True)
        if pair["pair_id"] in ids:
            audit["reasons"].append("duplicate pair_id")
        ids.add(pair["pair_id"])
        order = pair["order"]
        if order not in ([0, action], [action, 0]) or any(type(a) is not int for a in order):
            audit["reasons"].append("invalid AB/BA order")
            orientation = None
        else:
            orientation = order[0] == 0
            orientations.add(orientation)
        entry = {"pair_id": pair["pair_id"], "order": order}
        arm_data = {}
        for arm, selected in (("baseline", 0), ("candidate", action)):
            try:
                run = analyse_record(pair[arm], identity, selected)
                arm_data[arm] = run
                valid_records += 1
                all_cells.add(run["cell"])
                entry[arm] = {key: run[key] for key in ("dominant_cell", "dominant_fraction", "cells",
                                                      "tokens", "wave_wall_ns", "full_round_elapsed_ns",
                                                      "throughput", "request_p95_ns")}
                if not run["stable_response"]:
                    audit["reasons"].append(arm + " repeat/slot output instability")
                if arm == "baseline":
                    baseline_signatures.add((run["configuration_key"], run["response_key"]))
            except (ValueError, TypeError, KeyError, OverflowError) as error:
                entry[arm] = {"error": str(error)}
                audit["reasons"].append(arm + ": " + str(error))
        if len(arm_data) == 2:
            base, candidate = arm_data["baseline"], arm_data["candidate"]
            if base["configuration_key"] != candidate["configuration_key"]:
                audit["reasons"].append("paired configuration mismatch")
            if base["response_key"] != candidate["response_key"]:
                audit["reasons"].append("complete paired output mismatch")
            speedup = candidate["throughput"] / base["throughput"]
            p95_ratio = candidate["request_p95_ns"] / base["request_p95_ns"]
            entry["speedup"], entry["p95_ratio"] = speedup, p95_ratio
            if speedup <= 1:
                audit["reasons"].append("retained pair does not beat target-only")
            if p95_ratio > 1.05:
                audit["reasons"].append("retained pair p95 regression exceeds 5%")
        analysed.append((pair["pair_id"], orientation, arm_data))
        audit["pairs"].append(entry)
    if len(pairs) < 2 or orientations != {True, False}:
        audit["reasons"].append("at least two retained pairs including AB and BA required")
    if len(baseline_signatures) > 1:
        audit["reasons"].append("cross-pair baseline output/configuration instability")
    cell = next(iter(all_cells)) if len(all_cells) == 1 and None not in all_cells else None
    if cell is None:
        audit["reasons"].append("no common >=95% dominant actual batch/frontier cell")
    speedups = [entry["speedup"] for entry in audit["pairs"] if "speedup" in entry]
    if len(speedups) == len(pairs) and speedups:
        audit["median_speedup"] = statistics.median(speedups)
        if audit["median_speedup"] < 1.02:
            audit["reasons"].append("median paired improvement below 2%")
    audit["qualified"] = not audit["reasons"] and valid_records == 2 * len(pairs)
    audit["cell"] = {"active_batch": cell[0], "frontier_upper": cell[1]} if cell else None
    return {"audit": audit, "cell": cell, "analysed": analysed,
            "baseline_signatures": baseline_signatures,
            "pattern": tuple(sorted(((pid, orientation) for pid, orientation, _ in analysed), key=lambda item: item[0]))}


def build_profile(document, source=None):
    fields(document, ("schema_version", "artifact_type", "identity", "comparisons"), "input")
    if type(document["schema_version"]) is not int or document["schema_version"] != 1:
        raise ValueError("unsupported input schema")
    if document["artifact_type"] != "ninfer_resident_router_measurements":
        raise ValueError("resident router measurements required")
    identity = document["identity"]
    validate_identity(identity)
    if not isinstance(document["comparisons"], list):
        raise ValueError("comparisons must be an array")
    comparisons = [analyse_comparison(c, identity) for c in document["comparisons"]]
    groups = defaultdict(list)
    for comparison in comparisons:
        if comparison["cell"]:
            groups[comparison["cell"]].append(comparison)
    choices = {}
    selection_audit = []
    for cell, group in sorted(groups.items()):
        workloads = {c["audit"]["workload"] for c in group}
        baseline_by_workload = defaultdict(set)
        for c in group:
            baseline_by_workload[c["audit"]["workload"]].update(c["baseline_signatures"])
        conflicting = [w for w, signatures in baseline_by_workload.items() if len(signatures) > 1]
        cell_audit = {"active_batch": cell[0], "frontier_upper": cell[1],
                      "required_workloads": sorted(workloads), "candidates": [],
                      "selected_action": 0, "baseline_conflicts": sorted(conflicting)}
        scores = {}
        for action in (7, 11, 15):
            members = [c for c in group if c["audit"]["candidate_action"] == action]
            reasons = []
            represented = [c["audit"]["workload"] for c in members]
            if set(represented) != workloads or len(represented) != len(set(represented)):
                reasons.append("missing or duplicate representative workload")
            if any(not c["audit"]["qualified"] for c in members):
                reasons.append("one or more representative workloads not qualified")
            if members and len({c["pattern"] for c in members}) != 1:
                reasons.append("workload pair_id/order patterns do not align")
            merged = []
            if not reasons and not conflicting and members:
                for pid, orientation in members[0]["pattern"]:
                    arms = {}
                    for arm in ("baseline", "candidate"):
                        runs = [next(data[arm] for pair_id, direction, data in c["analysed"]
                                     if pair_id == pid and direction == orientation) for c in members]
                        tokens, wall = sum(r["tokens"] for r in runs), sum(r["wave_wall_ns"] for r in runs)
                        arms[arm] = {"tokens": tokens, "wave_wall_ns": wall,
                                     "throughput": tokens * 1e9 / wall}
                    merged.append({"pair_id": pid, "baseline_first": orientation, **arms})
                scores[action] = statistics.median(p["candidate"]["throughput"] for p in merged)
            cell_audit["candidates"].append({"draft_tokens": action, "reasons": reasons,
                                             "paired_merged_throughput": merged,
                                             "median_candidate_throughput": scores.get(action)})
        if scores:
            best = max(scores.values())
            choices[cell] = min(action for action, score in scores.items() if score >= best * 0.99)
            cell_audit["selected_action"] = choices[cell]
        selection_audit.append(cell_audit)
    raw_hash = hashlib.sha256(json.dumps(document, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    provenance = {
        "input": source or {"canonical_input_sha256": raw_hash},
        "coverage_rule": "Both arms of every retained run must share one dominant actual cell with "
                         ">=95% of full-round elapsed time. E2E uses complete waves; minority cells "
                         "are audit evidence and never coverage.",
        "correctness_rule": "All retained pairs/repeats/slots and cross-candidate baselines must match "
                            "complete responses and public configuration.",
        "qualification_thresholds": {"minimum_dominant_round_fraction": 0.95,
                                     "minimum_paired_median_speedup": 1.02,
                                     "maximum_paired_request_p95_ratio": 1.05,
                                     "relative_candidate_throughput_tie": 0.01,
                                     "minimum_retained_pairs": 2},
        "comparisons": [c["audit"] for c in comparisons],
        "selection": selection_audit,
        "uncovered_action": 0,
    }
    return {"schema_version": 1, "artifact_type": "ninfer_spec_router_profile",
            "identity": copy.deepcopy(identity),
            "cells": [{"active_batch": batch, "frontier_upper": upper,
                       "draft_tokens": choices.get((batch, upper), 0)}
                      for batch in range(1, 9) for upper in UPPERS],
            "provenance": provenance}


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON field: " + key)
        result[key] = value
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    raw = args.input.read_bytes()
    document = json.loads(raw, object_pairs_hook=unique_object, parse_constant=lambda value: (_ for _ in ()).throw(
        ValueError("nonfinite JSON number: " + value)))
    profile = build_profile(document, {"path": str(args.input.resolve()),
                                       "sha256": hashlib.sha256(raw).hexdigest()})
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(profile, stream, indent=2, allow_nan=False)
        stream.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
