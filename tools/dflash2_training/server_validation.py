"""Fail-closed validation of server A/B measurements (no inference dependencies).

Generated strings are compared exactly, including whitespace. Transport ids and timing
metadata are archived but are not generated-output equivalence fields. Null/missing
text is represented by an empty string only for the optional text channels; a response
with no generated content in any supported channel is never a successful sample.
"""
from __future__ import annotations

import hashlib
import json
import math
from collections import Counter, defaultdict
from typing import Any, Iterable


def json_bytes(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
                      allow_nan=False).encode("utf-8")


def digest(value: Any) -> str:
    return hashlib.sha256(json_bytes(value)).hexdigest()


def nearest_rank(values: Iterable[float], quantile: float) -> float | None:
    if not 0.0 < quantile <= 1.0:
        raise ValueError("quantile must be in (0, 1]")
    data = sorted(float(v) for v in values)
    if any(not math.isfinite(v) or v < 0 for v in data):
        raise ValueError("latencies must be finite and nonnegative")
    return data[math.ceil(quantile * len(data)) - 1] if data else None


def canonical_response(response: dict, *, thinking: str = "off") -> dict:
    if thinking not in ("off", "model-default"):
        raise ValueError("unsupported thinking policy")
    if not isinstance(response, dict) or "error" in response:
        raise ValueError("response is not a successful JSON object")
    choices = response.get("choices")
    if not isinstance(choices, list) or len(choices) != 1:
        raise ValueError("expected exactly one choice")
    choice = choices[0]
    if not isinstance(choice, dict) or not isinstance(choice.get("message"), dict):
        raise ValueError("choice lacks an assistant message")
    message = choice["message"]
    if message.get("role") != "assistant":
        raise ValueError("response message role must be assistant")
    channels = {}
    for key in ("content", "reasoning_content", "refusal"):
        value = message.get(key)
        if value is not None and not isinstance(value, str):
            raise ValueError(f"{key} must be text or null")
        channels[key] = "" if value is None else value
    finish = choice.get("finish_reason")
    if finish not in ("stop", "length", "tool_calls", "content_filter"):
        raise ValueError("missing/unsupported terminal finish_reason")
    # These benchmark requests advertise no tools. Silently ignoring an unexpected
    # tool response would repeat the old content-only hashing bug.
    if message.get("tool_calls") not in (None, []) or message.get("function_call") is not None:
        raise ValueError("unrequested tool/function call in text-only benchmark")
    if finish == "tool_calls":
        raise ValueError("unrequested tool_calls finish reason")
    if thinking == "off":
        if channels["reasoning_content"]:
            raise ValueError("server generated reasoning with --thinking off")
        if not channels["content"]:
            raise ValueError("empty content in no-thinking text workload")
    elif not any(channels.values()):
        raise ValueError("empty response in every generated channel")
    return {**channels, "finish_reason": finish}


def completion_tokens(response: dict) -> int:
    usage = response.get("usage")
    value = usage.get("completion_tokens") if isinstance(usage, dict) else None
    if type(value) is not int or value <= 0:
        raise ValueError("missing/nonpositive integer usage.completion_tokens")
    return value


def first_difference(left: dict, right: dict) -> dict | None:
    for key in ("content", "reasoning_content", "refusal", "finish_reason"):
        a, b = left.get(key), right.get(key)
        if a == b:
            continue
        result = {"field": key}
        if isinstance(a, str) and isinstance(b, str):
            pos = next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)))
            result.update(char_offset=pos, base_length=len(a), candidate_length=len(b),
                          base_excerpt=a[max(0, pos - 24):pos + 48],
                          candidate_excerpt=b[max(0, pos - 24):pos + 48])
        return result
    return None


def inspect_run(run: dict, expected: set[tuple[str, int]], *, minimum_repeats: int = 2) -> dict:
    rows = run.get("rows", [])
    counts = Counter((r.get("workload"), r.get("rep")) for r in rows)
    missing = sorted(expected - counts.keys())
    extra = sorted(counts.keys() - expected, key=repr)
    duplicates = [list(key) for key, n in counts.items() if n != 1]
    errors, valid = [], {}
    if not expected:
        errors.append({"reason": "empty expected workload set"})
    for row in rows:
        key = (row.get("workload"), row.get("rep"))
        output = row.get("output")
        if row.get("ok") is not True or not isinstance(output, dict):
            errors.append({"key": list(key), "reason": row.get("error", "missing full response")})
            continue
        # Always recompute from actual output, never trust an imported SHA alone.
        try:
            if set(output) != {"content", "reasoning_content", "refusal", "finish_reason"}:
                raise ValueError("not a canonical complete response")
            fake = {"choices": [{"message": {"role": "assistant", **{
                k: v for k, v in output.items() if k != "finish_reason"}},
                "finish_reason": output["finish_reason"]}]}
            canonical_response(fake, thinking=run.get("thinking", "off"))
            actual = digest(output)
            if actual != row.get("sha256"):
                raise ValueError("stored response SHA does not match full response")
            if type(row.get("completion_tokens")) is not int or row["completion_tokens"] <= 0:
                raise ValueError("invalid completion token count")
        except (ValueError, TypeError) as exc:
            errors.append({"key": list(key), "reason": str(exc)})
            continue
        valid[key] = row
    by_workload = defaultdict(list)
    for key, row in valid.items():
        by_workload[key[0]].append(row)
    unstable, insufficient = [], []
    for name in sorted({w for w, _ in expected}):
        samples = by_workload[name]
        if len(samples) < minimum_repeats:
            insufficient.append(name)
        if len({r["sha256"] for r in samples}) > 1:
            unstable.append(name)
    complete = not (missing or extra or duplicates or errors or run.get("error"))
    return {"complete": complete, "stable": complete and not unstable and not insufficient,
            "missing": [list(k) for k in missing], "unexpected": [list(k) for k in extra],
            "duplicates": duplicates, "errors": errors, "unstable_workloads": unstable,
            "insufficient_repeat_workloads": insufficient}


def compare_runs(base: dict, candidate: dict, expected: set[tuple[str, int]]) -> dict:
    a, b = inspect_run(base, expected), inspect_run(candidate, expected)
    lookup_a = {(r.get("workload"), r.get("rep")): r for r in base.get("rows", [])}
    lookup_b = {(r.get("workload"), r.get("rep")): r for r in candidate.get("rows", [])}
    differences = []
    for key in sorted(expected):
        x, y = lookup_a.get(key, {}), lookup_b.get(key, {})
        if not isinstance(x.get("output"), dict) or not isinstance(y.get("output"), dict):
            differences.append({"key": list(key), "field": "missing_full_response"})
        else:
            difference = first_difference(x["output"], y["output"])
            if difference:
                differences.append({"key": list(key), **difference})
    config_match = all(base.get(k) == candidate.get(k) for k in (
        "concurrency", "thinking", "cache_mode", "cuda_graph", "engine_concurrency",
        "max_context", "kv_capacity", "kv_dtype"))
    exact = a["complete"] and b["complete"] and not differences and config_match
    stable = a["stable"] and b["stable"]
    reasons = []
    if not a["complete"] or not b["complete"]:
        reasons.append("incomplete_or_failed_requests")
    if not config_match:
        reasons.append("configuration_mismatch")
    if differences:
        reasons.append("full_response_difference")
    if a["unstable_workloads"]:
        reasons.append("baseline_self_unstable")
    if b["unstable_workloads"]:
        reasons.append("candidate_self_unstable")
    if a["insufficient_repeat_workloads"] or b["insufficient_repeat_workloads"]:
        reasons.append("insufficient_self_repeats")
    x, y = base.get("tok_s"), candidate.get("tok_s")
    valid_speed = all(type(v) in (int, float) and math.isfinite(v) and v > 0 for v in (x, y))
    ratio = y / x if valid_speed else None
    if not valid_speed:
        reasons.append("invalid_throughput")
    qualified = exact and stable and valid_speed
    return {"exact": exact, "self_stable": stable, "qualified": qualified,
            "diagnostic_speedup": ratio, "qualified_speedup": ratio if qualified else None,
            "reasons": reasons, "differences": differences,
            "baseline_validation": a, "candidate_validation": b}


COUNTER_PATHS = {
    "speculative_rounds": ("rounds",),
    "accepted_tokens": ("accepted_tokens",),
    "drafted_tokens": ("drafted_tokens",),
    "tree_rounds": ("tree", "rounds"),
    "tree_fallback_rounds": ("tree", "fallback_rounds"),
    "tree_nodes": ("tree", "nodes"),
    "tree_accepted_drafts": ("tree", "accepted_drafts"),
    "head_skip_rounds": ("lookup", "head_skip_rounds"),
    "lookup_rounds": ("lookup", "rounds"),
}


def runtime_counters(events: list[dict], expected_requests: int, thinking: str) -> dict:
    """A fresh server/run has no client warmups: all Chat request_done rows belong here.

    Never substitute zero for absent/malformed telemetry. Startup engine warmup does
    not go through HTTP request_done; matching the expected request count is required.
    """
    done = [e for e in events if e.get("event") == "request_done"]
    ids = [(e.get("server_instance_id"), e.get("request", {}).get("request_id")) for e in done]
    errors = []
    if len(done) != expected_requests or len(set(ids)) != expected_requests:
        errors.append("runtime_request_count_or_identity_mismatch")
    if any(i[0] is None or i[1] is None for i in ids):
        errors.append("missing_runtime_request_identity")
    if len({i[0] for i in ids}) != 1:
        errors.append("mixed_or_missing_server_instances")
    if thinking == "off" and any(e.get("request", {}).get("enable_thinking") is not False for e in done):
        errors.append("runtime_thinking_policy_mismatch")
    for e in done:
        sampling = e.get("request", {}).get("sampling", {})
        if any(type(sampling.get(k)) not in (float, int) or sampling[k] != 0 for k in (
                "temperature", "presence_penalty", "frequency_penalty")):
            errors.append("runtime_sampler_mismatch")
            break
    sums = {}
    for name, path in COUNTER_PATHS.items():
        values = []
        for event in done:
            value = event.get("speculative")
            for part in path:
                value = value.get(part) if isinstance(value, dict) else None
            if type(value) is not int or value < 0:
                values = []
                errors.append("missing_or_invalid_counter:" + name)
                break
            values.append(value)
        sums[name] = sum(values) if len(values) == expected_requests and not errors else None
    # Avoid partially-qualified counters when any metadata is untrustworthy.
    if errors:
        sums = {k: None for k in COUNTER_PATHS}
    real, fallback = sums["tree_rounds"], sums["tree_fallback_rounds"]
    share = real / (real + fallback) if real is not None and real + fallback > 0 else None
    return {"status": "complete" if not errors else "unqualified", "errors": sorted(set(errors)),
            "request_done_count": len(done), "counters": sums, "tree_round_share": share}
