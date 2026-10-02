"""Machine metrics, provenance and strict qualification for the CLI A/B suite.

Old human logs remain readable for diagnosis; they cannot certify full precision or
next-token equivalence. A malformed machine record never falls back silently.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path

PREFIX = "NINFER_METRICS_JSON "
COUNTERS = ("generated", "decoded", "prompt_tokens", "rounds", "drafted", "accepted",
            "tree_rounds", "tree_fallback", "tree_nodes", "tree_accepted", "lookup_queries",
            "lookup_hits", "lookup_rounds", "lookup_drafted", "lookup_accepted", "head_skip_rounds",
            "workspace_peak_bytes", "runtime_reservation_bytes")
ANSI = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
NUMBER = r"([0-9]+(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?)([kKMGT]?)"
SCALES = {"": 1.0, "k": 1e3, "K": 1e3, "M": 1e6, "G": 1e9, "T": 1e12}


def positive(value) -> bool:
    return type(value) in (int, float) and math.isfinite(value) and value > 0


def rounded_metric(stderr: str, pattern: str) -> float | None:
    """Parse old pretty output, retaining its rounded nature in the caller's source tag."""
    pattern = pattern.replace(r"([\d.]+)", NUMBER).replace(r"(\d+)", NUMBER)
    found = re.search(pattern, ANSI.sub("", stderr))
    if found is None:
        return None
    value = float(found.group(1)) * SCALES[found.group(2)]
    return value if math.isfinite(value) else None


def metrics_from_log(stderr: str, patterns: dict[str, str]) -> tuple[dict, str, list[str]]:
    records = [line[len(PREFIX):] for line in stderr.splitlines() if line.startswith(PREFIX)]
    if not records:
        return {k: rounded_metric(stderr, v) for k, v in patterns.items()}, "pretty-rounded", ["missing-machine-metrics"]
    if len(records) != 1:
        return {}, "invalid-machine", ["duplicate-machine-metrics"]
    try:
        def reject_constant(text):
            raise ValueError("non-finite JSON constant: " + text)
        data = json.loads(records[0], parse_constant=reject_constant)
        if not isinstance(data, dict) or data.get("schema") != 1:
            raise ValueError("unsupported metrics schema")
        if any(type(data.get(k)) is not int or data[k] < 0 for k in COUNTERS):
            raise ValueError("missing or invalid integer counter")
        ids = data.get("token_ids")
        if (not isinstance(ids, list) or not ids or len(ids) != data["generated"]
                or any(type(t) is not int or t < 0 for t in ids)):
            raise ValueError("missing or invalid generated token sequence")
        if data["decoded"] != len(ids) - 1 or type(data.get("finish_reason")) is not int:
            raise ValueError("invalid decoded count or finish reason")
        if not positive(data.get("decode_tok_s")) or not positive(data.get("decode_seconds")):
            raise ValueError("nonpositive decode time or throughput")
        if not math.isclose(data["decode_tok_s"], data["decoded"] / data["decode_seconds"], rel_tol=1e-12):
            raise ValueError("inconsistent decoded-token throughput")
        for key in ("temperature", "presence_penalty", "frequency_penalty"):
            if type(data.get(key)) not in (int, float) or data[key] != 0:
                raise ValueError("unexpected sampler policy: " + key)
        for key in ("prefill_seconds", "acceptance_pct", "tok_per_round"):
            value = data.get(key)
            if value is not None and (type(value) not in (int, float) or not math.isfinite(value) or value < 0):
                raise ValueError("invalid numeric metric: " + key)
        return data, "machine-v1", []
    except (TypeError, ValueError, KeyError) as exc:
        return {}, "invalid-machine", [str(exc)]


def token_difference(a: dict, b: dict) -> dict | None:
    left, right = a.get("token_ids"), b.get("token_ids")
    if not isinstance(left, list) or not isinstance(right, list):
        return {"reason": "missing-token-evidence"}
    if left == right:
        return None
    index = next((i for i, pair in enumerate(zip(left, right)) if pair[0] != pair[1]), min(len(left), len(right)))
    return {"reason": "token-divergence", "token_index": index,
            "base_token": left[index] if index < len(left) else None,
            "candidate_token": right[index] if index < len(right) else None,
            "common_generated_prefix": left[:index],
            "base_length": len(left), "candidate_length": len(right)}


def row_identity(row: dict) -> str:
    return json.dumps([row.get("stdout_sha256"), row.get("token_ids"), row.get("finish_reason")], separators=(",", ":"))


def file_identity(path: Path) -> dict:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return {"path": str(path.resolve()), "size_bytes": path.stat().st_size, "sha256": h.hexdigest()}


def qualify_workload(rows: list[dict], base: str, candidate: str, pairs: int) -> dict:
    expected = {(p, arm) for p in range(pairs) for arm in (base, candidate)}
    actual = [(r.get("pair"), r.get("arm")) for r in rows]
    issues = []
    if not expected or set(actual) != expected or len(actual) != len(set(actual)):
        issues.append({"reason": "missing-or-duplicate-run"})
    mapping = {(r.get("pair"), r.get("arm")): r for r in rows}
    for row in rows:
        if row.get("returncode") != 0 or row.get("stdout_bytes", 0) <= 0:
            issues.append({"reason": "failed-or-empty-run", "pair": row.get("pair"), "arm": row.get("arm")})
        if row.get("metrics_source") != "machine-v1" or row.get("metric_errors"):
            issues.append({"reason": "unqualified-metrics", "pair": row.get("pair"), "arm": row.get("arm")})
    for pair in range(pairs):
        a, b = mapping.get((pair, base), {}), mapping.get((pair, candidate), {})
        if a.get("stdout_sha256") != b.get("stdout_sha256"):
            issues.append({"reason": "output-sha", "pair": pair})
        difference = token_difference(a, b)
        if difference:
            issues.append({"pair": pair, **difference})
        if a.get("finish_reason") != b.get("finish_reason"):
            issues.append({"reason": "finish-reason", "pair": pair})
    for arm in (base, candidate):
        if len({row_identity(r) for r in rows if r.get("arm") == arm}) > 1:
            issues.append({"reason": "self-repeat-unstable", "arm": arm})
    return {"passed": not issues, "mismatches": issues,
            "repeatability": "checked" if pairs >= 2 else "not-measured"}
