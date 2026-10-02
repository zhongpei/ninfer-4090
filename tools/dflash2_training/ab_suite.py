#!/usr/bin/env python3
"""Alternating CLI A/B with complete run identities and strict machine evidence.

Built-in prompts are synthetic regression workloads, not task-quality benchmarks.
The target implementation is never changed to make this qualification pass.
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
import time
from pathlib import Path

from tools.dflash2_training.cli_validation import (
    file_identity, metrics_from_log, positive, qualify_workload, rounded_metric,
)

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
    "head_skip_rounds": r"lookup head-skip rounds\s+(\d+)",
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


# Preserve the report's exact prompt bytes and capacities for reproducibility.
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
    Workload("lookup-repeat", repeated_context(), max_new=768, max_context=16384,
             weight=1.5, tags=("lookup", "repetition", "long-context")),
    Workload("long-context", long_context(), max_new=768, max_context=32768,
             weight=1.5, tags=("long-context",)),
)

ARMS = (
    Arm("baseline", (), "baseline"),
    Arm("dflash2-k7", ("--spec", "dflash2", "--draft-tokens", "7"), "baseline"),
    Arm("dflash2-k11", ("--spec", "dflash2", "--draft-tokens", "11"), "baseline"),
    Arm("dflash2-k15", ("--spec", "dflash2", "--draft-tokens", "15"), "baseline"),
    Arm("tree7", ("--spec", "dflash2", "--draft-tokens", "7", "--spec-tree", "lattice",
                  "--spec-tree-nodes", "7", "--spec-tree-spine", "5"), "dflash2-k7"),
    Arm("tree11", ("--spec", "dflash2", "--draft-tokens", "11", "--spec-tree", "lattice",
                   "--spec-tree-nodes", "11", "--spec-tree-spine", "7"), "dflash2-k11"),
    Arm("tree15", ("--spec", "dflash2", "--draft-tokens", "15", "--spec-tree", "lattice",
                   "--spec-tree-nodes", "15", "--spec-tree-spine", "7"), "dflash2-k15"),
    Arm("tree15-stair", ("--spec", "dflash2", "--draft-tokens", "15", "--spec-tree", "lattice",
                         "--spec-tree-nodes", "15", "--spec-tree-spine", "7", "--spec-router",
                         "stair", "--spec-stair-widths", "3,7,11,15"), "tree15"),
    Arm("lookup-replace", ("--spec", "dflash2", "--draft-tokens", "15", "--lookup-ngram", "5",
                           "--lookup-strategy", "vote", "--lookup-dflash", "replace",
                           "--lookup-min-support", "1", "--lookup-min-confidence", "0.55",
                           "--lookup-base-drafts", "7", "--lookup-deep-after", "2",
                           "--lookup-deep-drafts", "15"), "dflash2-k15"),
    Arm("lookup-skip", ("--spec", "dflash2", "--draft-tokens", "15", "--lookup-ngram", "5",
                        "--lookup-strategy", "vote", "--lookup-dflash", "skip",
                        "--lookup-min-support", "1", "--lookup-min-confidence", "0.55",
                        "--lookup-base-drafts", "7", "--lookup-deep-after", "2",
                        "--lookup-deep-drafts", "15"), "dflash2-k15"),
)


def parse_metric(stderr: str, key: str) -> float | None:
    metrics, _, _ = metrics_from_log(stderr, METRICS)
    return metrics.get(key)


def run_once(args, arm: Arm, workload: Workload, pair: int, order: int,
             comparison: str | None = None) -> dict:
    comparison = comparison or arm.name
    run_id = f"{comparison}__{workload.name}__p{pair:02d}__o{order}__{arm.name}"
    if any(not re.fullmatch(r"[A-Za-z0-9_-]+", part) for part in (comparison, workload.name, arm.name)):
        raise ValueError("unsafe run identity")
    directory = Path(args.out) / "runs" / run_id
    directory.mkdir(parents=True, exist_ok=False)
    stdout_path, stderr_path = directory / "stdout.txt", directory / "stderr.log"
    cmd = [str(args.exe), str(args.model), "--prompt", workload.prompt,
           "--max-new", str(workload.max_new), "--max-context", str(workload.max_context),
           "--kv-dtype", args.kv_dtype, "--greedy", "--no-thinking", "--raw-output",
           "--presence-penalty", "0", "--frequency-penalty", "0", "--seed", "0", *arm.args]
    if getattr(args, "no_cuda_graph", False):
        cmd.append("--no-cuda-graph")
    env = os.environ.copy()
    env.update(NINFER_AB_METRICS="1", NINFER_AB_PAIR=str(pair),
               NINFER_AB_WORKLOAD=workload.name, NINFER_AB_ARM=arm.name)
    started = time.perf_counter()
    error = None
    try:
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env,
                              timeout=getattr(args, "timeout", 600.0))
        stdout, stderr, returncode = proc.stdout, proc.stderr, proc.returncode
    except subprocess.TimeoutExpired as exc:
        stdout, stderr, returncode = exc.stdout or b"", exc.stderr or b"", -1
        error = "timeout"
    except OSError as exc:
        stdout, stderr, returncode, error = b"", str(exc).encode(), -1, str(exc)
    wall_s = time.perf_counter() - started
    stdout_path.write_bytes(stdout)
    stderr_path.write_bytes(stderr)
    metrics, source, errors = metrics_from_log(stderr.decode("utf-8", errors="replace"), METRICS)
    # Metric records are untrusted input: they cannot override run identity, process status or SHA.
    row = {key: metrics.get(key) for key in METRICS}
    for key in ("token_ids", "finish_reason", "decoded", "decode_seconds", "prefill_seconds",
                "workspace_peak_bytes", "runtime_reservation_bytes"):
        row[key] = metrics.get(key)
    row.update(run_id=run_id, comparison=comparison, pair=pair, order=order,
               arm=arm.name, compare_to=arm.compare_to, workload=workload.name,
               tags=list(workload.tags), weight=workload.weight, returncode=returncode,
               wall_s=wall_s, error=error, stdout_sha256=hashlib.sha256(stdout).hexdigest(),
               stdout_bytes=len(stdout), stderr_sha256=hashlib.sha256(stderr).hexdigest(),
               stdout_path=str(stdout_path), stderr_path=str(stderr_path), command=cmd,
               metrics_source=source, metric_errors=errors,
               prompt_sha256=hashlib.sha256(workload.prompt.encode()).hexdigest())
    (directory / "run.json").write_text(json.dumps(row, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return row


def ordered_pair(base: Arm, cand: Arm, pair: int) -> tuple[Arm, Arm]:
    return (base, cand) if pair % 2 == 0 else (cand, base)


def median(values):
    values = [float(v) for v in values if type(v) in (float, int) and math.isfinite(v)]
    return statistics.median(values) if values else None


def paired_values(rows, workload, base, cand, discard):
    by_pair = {}
    for row in rows:
        if row["workload"] != workload or row["pair"] < discard or row.get("comparison", cand) != cand:
            continue
        if row["arm"] in (base, cand):
            by_pair.setdefault(row["pair"], {})[row["arm"]] = row
    return [(i, r[base], r[cand]) for i, r in sorted(by_pair.items()) if base in r and cand in r]


def output_gate(pairs):
    # Compatibility helper; the main path additionally enforces identities, tokens and A/A.
    bad = [{"pair": i, "reason": "exit-or-output"} for i, a, b in pairs
           if a["returncode"] != 0 or b["returncode"] != 0 or a["stdout_sha256"] != b["stdout_sha256"]]
    return {"passed": bool(pairs) and not bad, "mismatches": bad}


def perf_summary(pairs):
    usable = [(a, b) for _, a, b in pairs if positive(a.get("decode_tok_s")) and positive(b.get("decode_tok_s"))]
    ratios = [b["decode_tok_s"] / a["decode_tok_s"] for a, b in usable]
    complete = bool(pairs) and len(usable) == len(pairs)
    return {"pairs": len(pairs), "measured_pairs": len(usable), "complete": complete,
            "median_speedup": median(ratios), "pair_speedups": ratios,
            "median_delta_tok_s": median([b["decode_tok_s"] - a["decode_tok_s"] for a, b in usable]),
            "median_acceptance_pct": median([b.get("acceptance_pct") for _, b in usable]),
            "median_tok_per_round": median([b.get("tok_per_round") for _, b in usable]),
            "resolved_better": complete and len(ratios) >= 3 and min(ratios) > 1 and median(ratios) >= 1.02,
            "resolved_worse": complete and len(ratios) >= 3 and max(ratios) < 1 and median(ratios) <= 0.98}


def pooled(results, workloads):
    used, missing, weighted = [], [], []
    for w in workloads:
        item = results.get(w.name, {})
        p = item.get("performance", {})
        speed = p.get("median_speedup")
        if not positive(speed) or not p.get("complete"):
            missing.append(w.name)
        else:
            used.append(w.name)
            weighted.append((w.weight, math.log(speed)))
    complete = bool(used) and not missing
    diagnostic = math.exp(sum(w * s for w, s in weighted) / sum(w for w, _ in weighted)) if complete else None
    qualified = complete and all(results[w.name].get("qualified_performance", False) for w in workloads)
    return {"weighted_geomean_speedup": diagnostic if qualified else None,
            "diagnostic_geomean_speedup": diagnostic, "complete_coverage": complete,
            "qualified": qualified, "workloads": used, "missing_workloads": missing,
            "weight": sum(w for w, _ in weighted)}


def judge(rows, workloads, arms, discard, expected_pairs=None):
    comparisons = {}
    for cand in arms:
        if cand.name == "baseline":
            continue
        per = {}
        for w in workloads:
            selected = [r for r in rows if r["workload"] == w.name
                        and r.get("comparison", cand.name) == cand.name
                        and r["arm"] in (cand.compare_to, cand.name)]
            count = expected_pairs if expected_pairs is not None else max((r["pair"] for r in selected), default=-1) + 1
            output = qualify_workload(selected, cand.compare_to, cand.name, count)
            pairs = paired_values(selected, w.name, cand.compare_to, cand.name, discard)
            performance = perf_summary(pairs)
            complete = performance["complete"] and performance["pairs"] == count - discard
            qualified = output["passed"] and count >= 2 and complete
            if not qualified:
                performance.update(resolved_better=False, resolved_worse=False)
            per[w.name] = {"output": output, "performance": performance, "qualified_performance": qualified}
        comparisons[cand.name] = {"base": cand.compare_to, "workloads": per,
            "pooled": pooled(per, workloads), "correct": bool(per) and all(x["output"]["passed"] for x in per.values()),
            "resolved_worse_workloads": [w for w, x in per.items() if x["performance"]["resolved_worse"]]}
    return comparisons


def select_workloads(names):
    if not names or names == "all":
        return WORKLOADS
    wanted = {x.strip() for x in names.split(",") if x.strip()}
    out = tuple(w for w in WORKLOADS if w.name in wanted)
    missing = wanted - {w.name for w in out}
    if missing:
        raise SystemExit("unknown workloads: " + ",".join(sorted(missing)))
    return out


def select_arms(names):
    if not names or names == "all":
        return ARMS
    wanted = {x.strip() for x in names.split(",") if x.strip()}
    by_name = {a.name: a for a in ARMS}
    missing = wanted - set(by_name)
    if missing:
        raise SystemExit("unknown arms: " + ",".join(sorted(missing)))
    expanded = set(wanted)
    while True:
        new = expanded | {by_name[name].compare_to for name in expanded}
        if new == expanded:
            break
        expanded = new
    return tuple(a for a in ARMS if a.name in expanded)


def report_markdown(result):
    lines = ["# CLI qualification", "", "Ratios from failed gates are diagnostic only; no partial-coverage pooled score.", "",
             "| candidate | base | exact gate | qualified pooled | diagnostic pooled | missing workloads |",
             "|---|---|---|---:|---:|---|"]
    def fmt(x):
        return "—" if x is None else f"{x:.3f}x"
    for name, c in result["comparisons"].items():
        p = c["pooled"]
        lines.append(f"| {name} | {c['base']} | {'PASS' if c['correct'] else 'FAIL'} | {fmt(p['weighted_geomean_speedup'])} | {fmt(p['diagnostic_geomean_speedup'])} | {', '.join(p['missing_workloads']) or '-'} |")
    for name, c in result["comparisons"].items():
        lines += ["", f"## {name} vs {c['base']}", "", "| workload | exact gate | diagnostic ratio | performance eligible | verdict |",
                  "|---|---|---:|---|---|"]
        for w, x in c["workloads"].items():
            p = x["performance"]
            verdict = "better" if p["resolved_better"] else "worse" if p["resolved_worse"] else "unresolved"
            lines.append(f"| {w} | {'PASS' if x['output']['passed'] else 'FAIL'} | {fmt(p['median_speedup'])} | {x['qualified_performance']} | {verdict} |")
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--exe", type=Path, default=Path("build-ninja/apps/ninfer"))
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("profiles/ab-suite"))
    ap.add_argument("--pairs", type=int, default=4)
    ap.add_argument("--discard", type=int, default=1)
    ap.add_argument("--cooldown", type=float, default=5.0)
    ap.add_argument("--timeout", type=float, default=600.0)
    ap.add_argument("--kv-dtype", default="int8")
    ap.add_argument("--workloads", default="all")
    ap.add_argument("--arms", default="all")
    ap.add_argument("--aa", action="store_true")
    ap.add_argument("--compare-baseline", action="store_true", help="qualify complete combinations against target-only")
    ap.add_argument("--no-cuda-graph", action="store_true")
    ap.add_argument("--json", default="")
    ap.add_argument("--summary", default="")
    args = ap.parse_args()
    if args.discard < 0 or args.pairs <= args.discard:
        ap.error("--pairs must exceed --discard >= 0")
    if not math.isfinite(args.cooldown) or args.cooldown < 0 or not positive(args.timeout):
        ap.error("invalid cooldown or timeout")
    workloads, arms = select_workloads(args.workloads), select_arms(args.arms)
    if args.aa:
        arms = (ARMS[0], Arm("baseline-aa", (), "baseline"))
    elif args.compare_baseline:
        arms = tuple(dataclasses.replace(a, compare_to="baseline") for a in arms)
    if not workloads or not any(a.name != "baseline" for a in arms):
        ap.error("A/B requires workloads and a candidate")
    args.exe, args.model = args.exe.resolve(), args.model.resolve()
    if not args.exe.is_file() or not os.access(args.exe, os.X_OK) or not args.model.is_file():
        ap.error("--exe must be executable and --model an existing artifact")
    args.out.mkdir(parents=True, exist_ok=True)
    # Lock this evidence set before any process runs; never reuse another run's files.
    journal = (args.out / "runs.jsonl").open("x", encoding="utf-8")
    rows = []
    by_name = {a.name: a for a in arms}
    settings = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    result = {"version": 2, "settings": settings, "exe_identity": file_identity(args.exe),
              "model_identity": file_identity(args.model), "runs": rows,
              "cuda_visible_devices": os.getenv("CUDA_VISIBLE_DEVICES"),
              "workloads": [dataclasses.asdict(w) for w in workloads],
              "arms": [dataclasses.asdict(a) for a in arms]}
    interrupted = False
    try:
        for w in workloads:
            for cand in arms:
                if cand.name == "baseline":
                    continue
                for pair in range(args.pairs):
                    for order, arm in enumerate(ordered_pair(by_name[cand.compare_to], cand, pair)):
                        print(f"[ab] {cand.name} {w.name} pair={pair} order={order} {arm.name}", flush=True)
                        row = run_once(args, arm, w, pair, order, comparison=cand.name)
                        rows.append(row)
                        journal.write(json.dumps(row, allow_nan=False) + "\n")
                        journal.flush()
                        if args.cooldown:
                            time.sleep(args.cooldown)
    except KeyboardInterrupt:
        interrupted = True
    finally:
        journal.close()
        result["interrupted"] = interrupted
        result["comparisons"] = judge(rows, workloads, arms, args.discard, args.pairs)
        for target, text in ((Path(args.json or args.out / "ab-results.json"), json.dumps(result, indent=2, allow_nan=False) + "\n"),
                             (Path(args.summary or args.out / "ab-summary.md"), report_markdown(result))):
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(text, encoding="utf-8")
    print(f"wrote {args.out / 'ab-results.json'}", flush=True)
    if interrupted:
        raise SystemExit(130)
    if not result["comparisons"] or any(not c["correct"] for c in result["comparisons"].values()):
        raise SystemExit(2)

if __name__ == "__main__":
    main()
