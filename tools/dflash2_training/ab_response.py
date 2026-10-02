"""Fail-closed response and runtime-evidence checks for the Linux server A/B runner.

Response equality is not a numerical oracle. Failed or unstable runs retain their diagnostic
measurements, but cannot qualify a performance improvement. No logits/tokens are rounded here.
"""
from __future__ import annotations

import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
                      allow_nan=False)


def response_signature(response: dict) -> dict:
    """Compare output semantics, not generated request IDs, timestamps or timing/usage metadata.

    Keep tool argument bytes and types intact; only the provider-generated tool-call ID is
    excluded. Missing/null textual channels are empty channels, not permission to omit reasoning.
    """
    choices = response.get("choices")
    if not isinstance(choices, list) or len(choices) != 1:
        raise ValueError("expected exactly one non-streaming Chat Completions choice")
    choice = choices[0]
    if not isinstance(choice, dict):
        raise ValueError("choice must be an object")
    message = choice.get("message")
    finish = choice.get("finish_reason")
    if not isinstance(message, dict) or message.get("role") != "assistant":
        raise ValueError("missing assistant message")
    if not isinstance(finish, str) or not finish:
        raise ValueError("missing terminal finish_reason")
    result: dict[str, Any] = {"role": "assistant", "finish_reason": finish}
    for channel in ("content", "reasoning_content", "refusal"):
        value = message.get(channel)
        if value is not None and not isinstance(value, str):
            raise ValueError(f"{channel} must be text or null")
        result[channel] = "" if value is None else value
    calls = message.get("tool_calls")
    if calls is None:
        calls = []
    if not isinstance(calls, list):
        raise ValueError("tool_calls must be an array")
    result["tool_calls"] = []
    for call in calls:
        if not isinstance(call, dict) or call.get("type") != "function":
            raise ValueError("invalid tool-call type")
        function = call.get("function")
        if (not isinstance(function, dict) or not isinstance(function.get("name"), str)
                or not function["name"]):
            raise ValueError("invalid tool-call function")
        if not isinstance(function.get("arguments"), str):
            raise ValueError("tool arguments must be their original JSON string")
        result["tool_calls"].append({key: value for key, value in call.items() if key != "id"})
    if not any(result[key] for key in ("content", "reasoning_content", "refusal", "tool_calls")):
        raise ValueError("empty complete response; an empty content hash is not evidence")
    return result


def response_record(response: dict, thinking: str) -> dict:
    if thinking not in ("off", "model-default"):
        raise ValueError("unknown thinking policy")
    signature = response_signature(response)
    if thinking == "off" and signature["reasoning_content"]:
        raise ValueError("no-thinking trial returned reasoning_content")
    usage = response.get("usage")
    count = usage.get("completion_tokens") if isinstance(usage, dict) else None
    if type(count) is not int or count <= 0:
        raise ValueError("missing or invalid usage.completion_tokens")
    prompt = usage.get("prompt_tokens")
    if type(prompt) is not int or prompt < 0:
        raise ValueError("missing or invalid usage.prompt_tokens")
    return {"signature": signature,
            "sha256": hashlib.sha256(canonical_json(signature).encode("utf-8")).hexdigest(),
            "completion_tokens": count, "prompt_tokens": prompt,
            "content_bytes": len(signature["content"].encode("utf-8")),
            "reasoning_bytes": len(signature["reasoning_content"].encode("utf-8"))}


def first_difference(left: Any, right: Any) -> dict | None:
    """Return an exact structural/character mismatch, never claim it is a token/Op location."""
    if canonical_json(left) == canonical_json(right):
        return None
    if isinstance(left, dict) and isinstance(right, dict):
        for field in sorted(set(left) | set(right)):
            if field not in left or field not in right:
                return {"field": field, "reason": "missing_field"}
            diff = first_difference(left[field], right[field])
            if diff is not None:
                return {**diff, "field": field + ("." + diff["field"] if diff.get("field") else "")}
    if isinstance(left, str) and isinstance(right, str):
        index = next((i for i, (a, b) in enumerate(zip(left, right)) if a != b),
                     min(len(left), len(right)))
        return {"reason": "text", "character_offset": index,
                "left_length": len(left), "right_length": len(right),
                "left_excerpt": left[max(0, index - 24):index + 40],
                "right_excerpt": right[max(0, index - 24):index + 40]}
    return {"reason": "value_or_type", "left": left, "right": right}


def nearest_rank(values: list[float], probability: float) -> float | None:
    if not 0 < probability <= 1:
        raise ValueError("percentile probability must be in (0, 1]")
    if any(not math.isfinite(value) or value < 0 for value in values):
        raise ValueError("latency must be finite and nonnegative")
    return sorted(values)[math.ceil(probability * len(values)) - 1] if values else None


def trial_integrity(rows: list[dict], expected: set[tuple[str, int]]) -> dict:
    keys = [(row.get("workload"), row.get("rep")) for row in rows]
    counts = Counter(keys)
    missing = sorted(expected - set(counts))
    unexpected = sorted(set(counts) - expected, key=str)
    duplicates = sorted((key for key, count in counts.items() if count != 1), key=str)
    invalid = []
    for row in rows:
        reason = None
        if row.get("ok") is not True:
            reason = row.get("error", "request_failed")
        elif not isinstance(row.get("signature"), dict) or not row.get("sha256"):
            reason = "missing_full_response_evidence"
        else:
            try:
                raw = row.get("response")
                if not isinstance(raw, dict):
                    raise ValueError("missing_raw_response")
                if canonical_json(response_signature(raw)) != canonical_json(row["signature"]):
                    raise ValueError("raw_response_signature_mismatch")
                digest = hashlib.sha256(canonical_json(row["signature"]).encode("utf-8")).hexdigest()
                if digest != row["sha256"]:
                    raise ValueError("signature_hash_mismatch")
            except (ValueError, TypeError) as exc:
                reason = str(exc)
        if reason:
            invalid.append({"workload": row.get("workload"), "rep": row.get("rep"),
                            "reason": reason})
    return {"passed": bool(expected) and not (missing or unexpected or duplicates or invalid),
            "missing": missing, "unexpected": unexpected, "duplicates": duplicates,
            "invalid": invalid}


def repeat_stability(rows: list[dict], workload_names: list[str]) -> dict:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        if row.get("ok") and row.get("signature") is not None:
            grouped[row["workload"]].append(row)
    findings = []
    for name in workload_names:
        group = grouped[name]
        if len(group) < 2:
            findings.append({"workload": name, "reason": "insufficient_repeats"})
            continue
        for row in group[1:]:
            diff = first_difference(group[0]["signature"], row["signature"])
            if diff is not None:
                findings.append({"workload": name, "reason": "self_repeat_mismatch",
                                 "reference_rep": group[0].get("rep"), "rep": row.get("rep"),
                                 "difference": diff})
                break
    return {"passed": bool(workload_names) and not findings, "findings": findings}


def compare_responses(base: list[dict], candidate: list[dict],
                      expected: set[tuple[str, int]]) -> dict:
    bcheck, ccheck = trial_integrity(base, expected), trial_integrity(candidate, expected)
    bmap = {(r.get("workload"), r.get("rep")): r for r in base}
    cmap = {(r.get("workload"), r.get("rep")): r for r in candidate}
    mismatches = []
    for key in sorted(expected):
        b, c = bmap.get(key, {}), cmap.get(key, {})
        if not b.get("ok") or not c.get("ok"):
            continue  # integrity reports both-sided failures, including zero successful requests
        diff = first_difference(b.get("signature"), c.get("signature"))
        if diff is not None:
            mismatches.append({"workload": key[0], "rep": key[1], "difference": diff})
    return {"passed": bcheck["passed"] and ccheck["passed"] and not mismatches,
            "base_integrity": bcheck, "candidate_integrity": ccheck, "mismatches": mismatches}


COUNTER_PATHS = {
    "rounds": ("rounds",), "drafted_tokens": ("drafted_tokens",),
    "accepted_tokens": ("accepted_tokens",),
    "tree_rounds": ("tree", "rounds"), "tree_fallback_rounds": ("tree", "fallback_rounds"),
    "tree_nodes": ("tree", "nodes"), "tree_accepted_drafts": ("tree", "accepted_drafts"),
    "lookup_rounds": ("lookup", "rounds"),
    "head_skip_rounds": ("lookup", "head_skip_rounds"),
}


def runtime_evidence(path: Path, expected_requests: int) -> dict:
    """Aggregate actual request_done events only, never infer execution from startup flags.

    The runtime uses internal numeric request IDs; do not join them to protocol UUIDs or infer
    per-workload counters from completion order. Preserve the log for later explicit correlation.
    """
    totals = {key: 0 for key in COUNTER_PATHS}
    ids = set()
    errors = []
    done = 0
    try:
        with path.open(encoding="utf-8") as stream:
            for line in stream:
                event = json.loads(line)
                if not isinstance(event, dict):
                    raise ValueError("runtime log event must be an object")
                if event.get("event") in ("request_error", "request_rejected"):
                    errors.append(event.get("event"))
                if event.get("event") != "request_done":
                    continue
                done += 1
                request = event.get("request", {})
                if not isinstance(request, dict):
                    raise ValueError("runtime log request must be an object")
                rid = (event.get("server_instance_id"), request.get("request_id"))
                if None in rid or rid in ids:
                    errors.append("invalid_or_duplicate_request_id")
                ids.add(rid)
                for key, fields in COUNTER_PATHS.items():
                    value: Any = event.get("speculative")
                    for field in fields:
                        value = value.get(field) if isinstance(value, dict) else None
                    if type(value) is not int or value < 0:
                        errors.append("invalid_counter:" + key)
                    else:
                        totals[key] += value
    except (OSError, ValueError, TypeError) as exc:
        errors.append(str(exc))
    denominator = totals["tree_rounds"] + totals["tree_fallback_rounds"]
    return {"complete": done == expected_requests and expected_requests > 0 and not errors,
            "request_done_count": done, "errors": errors, **totals,
            "tree_execution_fraction": totals["tree_rounds"] / denominator if denominator else None}
