#!/usr/bin/env python3
"""Alternating multi-scenario A/B harness for NInfer speculative decoding.

Design references:
- TandemLLM block_ab: alternate states, discard the first pair, judge per-workload and pooled,
  and treat output-token differences as correctness failures.
- Spec-Bench: same hardware/environment, report speedup and mean accepted tokens.
- DFlash public evals: cover reasoning/math, code, chat/prose and prompt-copy workloads, and
  distinguish latency-oriented single-request tests from throughput-oriented serving tests.

This harness is intentionally CLI-first so it can exercise the exact NInfer artifact/runtime stack
without adding a Python inference dependency. A separate server matrix may consume the JSONL output.
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import math
import os
import re
import statistics
import subprocess
import sys
import time
from pathlib import Path


METRICS = {
    "decode_tok_s": r"decode speed\s+([\d.]+) tok/s",
    "generated": r"generated tokens\s+(\d+)",
    "rounds": r"\S+ rounds\s+(\d+)",
    "drafted": r"\S+ drafted tokens\s+(\d+)",
    "accepted": r"\S+ accepted tokens\s+(\d+)",
    "acceptance_pct": r"\S+ acceptance rate\s+([\d.]+)%",
    "tok_per_round": r"\S+ acceptance length\s+([\d.]+) tok/round",
    "tree_rounds": r"tree rounds\s+(\d+)",
    "tree_fallback": r"tree fallback rounds\s+(\d+)",
    "tree_nodes": r"tree nodes\s+(\d+)",
    "tree_accepted": r"tree accepted drafts\s+(\d+)",
    "lookup_queries": r"lookup queries\s+(\d+)",
    "lookup_hits": r"lookup hits\s+(\d+)",
    "lookup_rounds": r"lookup rounds\s+(\d+)",
    "lookup_drafted": r"lookup drafted tokens\s+(\d+)",
    "lookup_accepted": r"lookup accepted tokens\s+(\d+)",
}


@dataclasses.dataclass(frozen=True)
class Workload:
    name: str
    prompt: str
    max_new: int = 512
    max_context: int = 8192
    weight: float = 1.0
    tags: tuple[str, ...] = ()


@dataclasses.dataclass(frozen=True)
class Arm:
    name: str
    args: tuple[str, ...]
    compare_to: str


def repeated_context() -> str:
    block = (
        "A cache entry contains key, timestamp, owner, checksum, and payload. "
        "When the checksum fails, discard the entry and rebuild it from the owner. "
    )
    return (block * 90) + (
        "\nNow continue the operational runbook in the same repetitive style for another "
        "thirty steps. Preserve the field names exactly."
    )


def long_context() -> str:
    paragraph = (
        "The service accepts a request, validates quota, chooses a proxy route, records a trace id, "
        "opens an upstream connection, verifies the response, and updates accounting. "
    )
    return (paragraph * 160) + (
        "\nSummarize the architecture, identify three failure modes, and propose a deterministic "
        "recovery sequence without losing the trace id."
    )


WORKLOADS = (
    Workload(
        "prose",
        "Explain how a paged KV cache reduces memory fragmentation for concurrent LLM inference. "
        "Use concrete examples and compare it with reserving maximum context per request.",
        tags=("prose", "short-context"),
    ),
    Workload(
        "chat",
        "A user says: 'My API becomes unstable only when I raise traffic above the configured "
        "rate limit.' Reply as a systems engineer: explain retry storms, exponential backoff, "
        "jitter, and per-client rate distribution.",
        tags=("chat", "short-context"),
    ),
    Workload(
        "reasoning",
        "A warehouse has 17 rows. Row i contains 3i+2 boxes. Every fifth box is inspected, and "
        "one quarter of inspected boxes require a second inspection. Derive the total box count "
        "and the expected number of inspection operations step by step.",
        tags=("reasoning", "math"),
    ),
    Workload(
        "code",
        "Implement a production-quality Python function that merges overlapping half-open integer "
        "intervals. Include type hints, edge cases, examples, and complexity analysis.",
        tags=("code",),
    ),
    Workload(
        "structured",
        "Return ONLY valid JSON with keys summary, risks, mitigations, and metrics. Analyze a "
        "speculative-decoding deployment where proposal acceptance falls as context grows. "
        "metrics must be an object with exactly four numeric fields.",
        tags=("structured", "json"),
    ),
    Workload(
        "lookup-repeat",
        repeated_context(),
        max_new=768,
        max_context=16384,
        weight=1.5,
        tags=("lookup", "repetition", "long-context"),
    ),
    Workload(
        "long-context",
        long_context(),
        max_new=768,
        max_context=32768,
        weight=1.5,
        tags=("long-context",),
    ),
)


ARMS = (
    Arm("baseline", (), "baseline"),
    Arm("dflash2-k7", ("--spec", "dflash2", "--draft-tokens", "7"), "baseline"),
    Arm("dflash2-k11", ("--spec", "dflash2", "--draft-tokens", "11"), "baseline"),
    Arm("dflash2-k15", ("--spec", "dflash2", "--draft-tokens", "15"), "baseline"),
    Arm(
        "tree7",
        ("--spec", "dflash2", "--draft-tokens", "7",
         "--spec-tree", "lattice", "--spec-tree-nodes", "7", "--spec-tree-spine", "5"),
        "dflash2-k7",
    ),
    Arm(
        "tree11",
        ("--spec", "dflash2", "--draft-tokens", "11",
         "--spec-tree", "lattice", "--spec-tree-nodes", "11", "--spec-tree-spine", "7"),
        "dflash2-k11",
    ),
    Arm(
        "tree15",
        ("--spec", "dflash2", "--draft-tokens", "15",
         "--spec-tree", "lattice", "--spec-tree-nodes", "15", "--spec-tree-spine", "7"),
        "dflash2-k15",
    ),
    Arm(
        "tree15-stair",
        ("--spec", "dflash2", "--draft-tokens", "15",
         "--spec-tree", "lattice", "--spec-tree-nodes", "15", "--spec-tree-spine", "7",
         "--spec-router", "stair", "--spec-stair-widths", "3,7,11,15"),
        "tree15",
    ),
    Arm(
        "lookup-replace",
        ("--spec", "dflash2", "--draft-tokens", "15",
         "--lookup-ngram", "5", "--lookup-strategy", "vote", "--lookup-dflash", "replace",
         "--lookup-min-support", "1", "--lookup-min-confidence", "0.55",
         "--lookup-base-drafts", "7", "--lookup-deep-after", "2", "--lookup-deep-drafts", "15"),
        "dflash2-k15",
    ),
    Arm(
        "lookup-skip",
        ("--spec", "dflash2", "--draft-tokens", "15",
         "--lookup-ngram", "5", "--lookup-strategy", "vote", "--lookup-dflash", "skip",
         "--lookup-min-support", "1", "--lookup-min-confidence", "0.55",
         "--lookup-base-drafts", "7", "--lookup-deep-after", "2", "--lookup-deep-drafts", "15"),
        "dflash2-k15",
    ),
)


def parse_metric(stderr: str, key: str) -> float | None:
    m = re.search(METRICS[key], stderr)
    if not m:
        return None
    return float(m.group(1))


def run_once(args, arm: Arm, workload: Workload, pair: int, order: int) -> dict:
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = out_dir / f"{workload.name}__{arm.name}__p{pair:02d}o{order}"
    stdout_path = stem.with_suffix(".out.txt")
    stderr_path = stem.with_suffix(".err.log")

    cmd = [
        str(args.exe), str(args.model),
        "--prompt", workload.prompt,
        "--max-new", str(workload.max_new),
        "--max-context", str(workload.max_context),
        "--kv-dtype", args.kv_dtype,
        "--greedy", "--no-thinking", "--raw-output",
        *arm.args,
    ]
    env = os.environ.copy()
    env["NINFER_AB_PAIR"] = str(pair)
    env["NINFER_AB_WORKLOAD"] = workload.name
    env["NINFER_AB_ARM"] = arm.name

    started = time.perf_counter()
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env)
    wall_s = time.perf_counter() - started
    stdout_path.write_bytes(proc.stdout)
    stderr_path.write_bytes(proc.stderr)

    stderr = proc.stderr.decode("utf-8", errors="replace")
    row = {
        "pair": pair,
        "order": order,
        "arm": arm.name,
        "compare_to": arm.compare_to,
        "workload": workload.name,
        "tags": list(workload.tags),
        "weight": workload.weight,
        "returncode": proc.returncode,
        "wall_s": wall_s,
        "stdout_sha256": hashlib.sha256(proc.stdout).hexdigest(),
        "stdout_bytes": len(proc.stdout),
        "stderr_path": str(stderr_path),
        "stdout_path": str(stdout_path),
        "command": cmd,
    }
    for key in METRICS:
        row[key] = parse_metric(stderr, key)
    return row


def ordered_pair(base: Arm, cand: Arm, pair: int) -> tuple[Arm, Arm]:
    # Balanced AB/BA order suppresses temperature/clock drift better than always A then B.
    return (base, cand) if pair % 2 == 0 else (cand, base)


def median(values):
    vals = [float(x) for x in values if x is not None and math.isfinite(float(x))]
    return statistics.median(vals) if vals else None


def paired_values(rows: list[dict], workload: str, base: str, cand: str, discard: int):
    by_pair: dict[int, dict[str, dict]] = {}
    for row in rows:
        if row["workload"] != workload or row["pair"] < discard:
            continue
        if row["arm"] not in (base, cand):
            continue
        by_pair.setdefault(row["pair"], {})[row["arm"]] = row
    return [
        (pair, pair_rows[base], pair_rows[cand])
        for pair, pair_rows in sorted(by_pair.items())
        if base in pair_rows and cand in pair_rows
    ]


def output_gate(pairs: list[tuple[int, dict, dict]]) -> dict:
    mismatches = []
    for pair, base, cand in pairs:
        if base["returncode"] != 0 or cand["returncode"] != 0:
            mismatches.append({"pair": pair, "reason": "nonzero-exit",
                               "base_rc": base["returncode"], "cand_rc": cand["returncode"]})
        elif base["stdout_sha256"] != cand["stdout_sha256"]:
            mismatches.append({"pair": pair, "reason": "output-sha",
                               "base": base["stdout_sha256"], "cand": cand["stdout_sha256"]})
    return {"passed": not mismatches, "mismatches": mismatches}


def perf_summary(pairs: list[tuple[int, dict, dict]]) -> dict:
    ratios = []
    deltas = []
    accepted = []
    tpr = []
    for _, base, cand in pairs:
        b = base.get("decode_tok_s")
        c = cand.get("decode_tok_s")
        if b and c and b > 0:
            ratios.append(c / b)
            deltas.append(c - b)
        if cand.get("acceptance_pct") is not None:
            accepted.append(cand["acceptance_pct"])
        if cand.get("tok_per_round") is not None:
            tpr.append(cand["tok_per_round"])
    med_ratio = median(ratios)
    return {
        "pairs": len(pairs),
        "median_speedup": med_ratio,
        "median_delta_tok_s": median(deltas),
        "median_acceptance_pct": median(accepted),
        "median_tok_per_round": median(tpr),
        "pair_speedups": ratios,
        # Tandem-style conservative resolution: all retained pairs must have the same sign and
        # the median gain must clear a small practical threshold.
        "resolved_better": bool(ratios) and all(x > 1.0 for x in ratios)
                           and med_ratio is not None and med_ratio >= 1.02,
        "resolved_worse": bool(ratios) and all(x < 1.0 for x in ratios)
                          and med_ratio is not None and med_ratio <= 0.98,
    }


def pooled(results: dict, workloads: tuple[Workload, ...]) -> dict:
    total_weight = 0.0
    log_speed = 0.0
    used = []
    for w in workloads:
        item = results.get(w.name)
        if not item:
            continue
        speed = item["performance"].get("median_speedup")
        if speed is None or speed <= 0:
            continue
        total_weight += w.weight
        log_speed += w.weight * math.log(speed)
        used.append(w.name)
    return {
        "weighted_geomean_speedup": math.exp(log_speed / total_weight) if total_weight else None,
        "weight": total_weight,
        "workloads": used,
    }


def judge(rows: list[dict], workloads: tuple[Workload, ...], arms: tuple[Arm, ...],
          discard: int) -> dict:
    by_name = {a.name: a for a in arms}
    comparisons = {}
    for cand in arms:
        if cand.name == "baseline":
            continue
        base = by_name[cand.compare_to]
        per = {}
        for w in workloads:
            pairs = paired_values(rows, w.name, base.name, cand.name, discard)
            per[w.name] = {
                "output": output_gate(pairs),
                "performance": perf_summary(pairs),
            }
        comparisons[cand.name] = {
            "base": base.name,
            "workloads": per,
            "pooled": pooled(per, workloads),
            "correct": all(x["output"]["passed"] for x in per.values()),
            "resolved_worse_workloads": [
                w for w, x in per.items() if x["performance"]["resolved_worse"]
            ],
        }
    return comparisons


def select_workloads(names: str) -> tuple[Workload, ...]:
    if not names or names == "all":
        return WORKLOADS
    wanted = {x.strip() for x in names.split(",") if x.strip()}
    out = tuple(w for w in WORKLOADS if w.name in wanted)
    missing = wanted - {w.name for w in out}
    if missing:
        raise SystemExit("unknown workloads: " + ",".join(sorted(missing)))
    return out


def select_arms(names: str) -> tuple[Arm, ...]:
    if not names or names == "all":
        return ARMS
    wanted = {x.strip() for x in names.split(",") if x.strip()}
    by_name = {a.name: a for a in ARMS}
    missing = wanted - set(by_name)
    if missing:
        raise SystemExit("unknown arms: " + ",".join(sorted(missing)))
    # Pull comparator arms in automatically so every selected experiment is a real A/B.
    expanded = set(wanted)
    changed = True
    while changed:
        changed = False
        for name in list(expanded):
            base = by_name[name].compare_to
            if base not in expanded:
                expanded.add(base)
                changed = True
    return tuple(a for a in ARMS if a.name in expanded)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--exe", type=Path, default=Path("build-ninja/apps/ninfer"))
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--out", default="profiles/ab-suite")
    ap.add_argument("--pairs", type=int, default=4)
    ap.add_argument("--discard", type=int, default=1)
    ap.add_argument("--cooldown", type=float, default=5.0)
    ap.add_argument("--kv-dtype", default="int8")
    ap.add_argument("--workloads", default="all")
    ap.add_argument("--arms", default="all")
    ap.add_argument("--json", default="")
    ap.add_argument("--summary", default="")
    args = ap.parse_args()

    if args.pairs <= args.discard:
        raise SystemExit("--pairs must exceed --discard")
    workloads = select_workloads(args.workloads)
    arms = select_arms(args.arms)
    by_name = {a.name: a for a in arms}

    rows = []
    # Each candidate is paired with its own comparator. We deliberately do not run one global
    # baseline and reuse it across the whole matrix because thermal/clock state would then differ.
    for workload in workloads:
        for cand in arms:
            if cand.name == "baseline":
                continue
            if cand.compare_to not in by_name:
                continue
            base = by_name[cand.compare_to]
            for pair in range(args.pairs):
                first, second = ordered_pair(base, cand, pair)
                for order, arm in enumerate((first, second)):
                    print(f"[ab] {workload.name} pair={pair+1}/{args.pairs} "
                          f"order={order+1} {arm.name}", flush=True)
                    row = run_once(args, arm, workload, pair, order)
                    rows.append(row)
                    print(
                        f"     rc={row['returncode']} tok/s={row.get('decode_tok_s')} "
                        f"accept={row.get('acceptance_pct')} sha={row['stdout_sha256'][:12]}",
                        flush=True,
                    )
                    if args.cooldown > 0:
                        time.sleep(args.cooldown)

    result = {
        "version": 1,
        "model": str(args.model),
        "exe": str(args.exe),
        "kv_dtype": args.kv_dtype,
        "pairs": args.pairs,
        "discard": args.discard,
        "workloads": [dataclasses.asdict(w) for w in workloads],
        "arms": [dataclasses.asdict(a) for a in arms],
        "runs": rows,
    }
    result["comparisons"] = judge(rows, workloads, arms, args.discard)

    json_path = Path(args.json or (Path(args.out) / "ab-results.json"))
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    lines = [
        "# NInfer speculative A/B matrix",
        "",
        f"- model: `{args.model}`",
        f"- KV: `{args.kv_dtype}`",
        f"- pairs: {args.pairs}, discarded first: {args.discard}",
        "",
        "| candidate | baseline | correct | pooled speedup | resolved worse workloads |",
        "|---|---|---:|---:|---|",
    ]
    for name, cmp in result["comparisons"].items():
        speed = cmp["pooled"]["weighted_geomean_speedup"]
        speed_text = "" if speed is None else f"{speed:.3f}x"
        worse = ", ".join(cmp["resolved_worse_workloads"]) or "-"
        lines.append(
            f"| {name} | {cmp['base']} | {'PASS' if cmp['correct'] else 'FAIL'} | "
            f"{speed_text} | {worse} |"
        )
    lines += ["", "## Per workload", ""]
    for name, cmp in result["comparisons"].items():
        lines += [f"### {name} vs {cmp['base']}", "",
                  "| workload | exact | speedup | accept % | tok/round | verdict |",
                  "|---|---:|---:|---:|---:|---|"]
        for w, item in cmp["workloads"].items():
            p = item["performance"]
            speed = p["median_speedup"]
            verdict = ("better" if p["resolved_better"] else
                       "worse" if p["resolved_worse"] else "unresolved")
            speed_text = "" if speed is None else f"{speed:.3f}x"
            accept_text = (
                "" if p["median_acceptance_pct"] is None
                else f"{p['median_acceptance_pct']:.2f}"
            )
            tpr_text = (
                "" if p["median_tok_per_round"] is None
                else f"{p['median_tok_per_round']:.3f}"
            )
            lines.append(
                f"| {w} | {'PASS' if item['output']['passed'] else 'FAIL'} | "
                f"{speed_text} | {accept_text} | {tpr_text} | {verdict} |"
            )
        lines.append("")
    summary_path = Path(args.summary or (Path(args.out) / "ab-summary.md"))
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {json_path}")
    print(f"wrote {summary_path}")

    # Correctness is a hard gate. Performance regressions are findings, not harness failures.
    if any(not cmp["correct"] for cmp in result["comparisons"].values()):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
