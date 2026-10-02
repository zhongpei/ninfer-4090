#!/usr/bin/env python3
"""Isolated, paired server A/B with full-response and repeat-stability gates.

Each comparison/concurrency/pair/side owns a new server and request-log directory. No sampler
rule or numerical tolerance is changed. A FAIL keeps diagnostic throughput but publishes no
qualified speedup. --arms baseline performs independent baseline/baseline stability trials.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import dataclasses
import datetime as dt
import json
import math
import os
import platform
import signal
import socket
import statistics
import subprocess
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path

from tools.dflash2_training.ab_response import (
    canonical_json, compare_responses, nearest_rank, repeat_stability,
    response_record, runtime_evidence, trial_integrity,
)


def request_json(url: str, payload: dict | None = None, timeout: float = 300.0) -> dict:
    data = None if payload is None else canonical_json(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"},
                                 method="GET" if payload is None else "POST")
    # These tests own a loopback server; an unrelated HTTP_PROXY must not intercept the test.
    with urllib.request.build_opener(urllib.request.ProxyHandler({})).open(req, timeout=timeout) as res:
        result = json.loads(res.read())
    if not isinstance(result, dict):
        raise ValueError("HTTP JSON response must be an object")
    return result


def wait_ready(port: int, proc: subprocess.Popen, timeout: float,
               expected_model: str | None = None) -> str:
    deadline = time.monotonic() + timeout
    base = f"http://127.0.0.1:{port}"
    last = ""
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"server exited during startup rc={proc.returncode}")
        try:
            if request_json(base + "/health", timeout=2).get("status") == "ok":
                model = str(request_json(base + "/v1/models", timeout=5)["data"][0]["id"])
                if expected_model is not None and model != expected_model:
                    raise RuntimeError("port belongs to a different server instance")
                if proc.poll() is None:
                    return model
        except (OSError, ValueError, KeyError, IndexError) as exc:
            last = str(exc)
        time.sleep(0.25)
    raise TimeoutError("server readiness timed out: " + last)


def one_request(base: str, model_id: str, prompt: str, max_tokens: int, timeout: float,
                *, thinking: str = "off", seed: int = 7,
                submitted_at: float | None = None) -> dict:
    started = time.perf_counter()
    payload = {"model": model_id, "messages": [{"role": "user", "content": prompt}],
               "max_tokens": max_tokens, "temperature": 0, "presence_penalty": 0,
               "frequency_penalty": 0, "seed": seed, "stream": False}
    row = {"ok": False, "request": payload,
           "client_queue_s": max(0.0, started - submitted_at) if submitted_at is not None else 0.0}
    try:
        response = request_json(base + "/v1/chat/completions", payload, timeout)
        completed = time.perf_counter()
        row["response"] = response  # retain reasoning, tools, usage and finish reason even on invalid output
        row.update(response_record(response, thinking))
        row["ok"] = True
    except Exception as exc:
        completed = time.perf_counter()
        row["error"] = f"{type(exc).__name__}: {exc}"
        if isinstance(exc, urllib.error.HTTPError):
            row["http_status"] = exc.code
            row["error_body"] = exc.read().decode("utf-8", errors="replace")
    row["latency_s"] = completed - started
    row["client_total_s"] = row["client_queue_s"] + row["latency_s"]
    return row


def run_level(base: str, model_id: str, workloads, concurrency: int, repeats: int,
              timeout: float, *, thinking: str = "off", seed: int = 7,
              max_new: int = 512) -> dict:
    jobs = [(w, rep) for rep in range(repeats) for w in workloads]
    started = time.perf_counter()
    rows = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = {}
        for workload, rep in jobs:
            submitted = time.perf_counter()
            future = pool.submit(one_request, base, model_id, workload.prompt,
                                 min(workload.max_new, max_new), timeout,
                                 thinking=thinking, seed=seed, submitted_at=submitted)
            futures[future] = (workload.name, rep)
        for future in concurrent.futures.as_completed(futures):
            name, rep = futures[future]
            row = future.result()
            row.update({"workload": name, "rep": rep})
            rows.append(row)
    wall = time.perf_counter() - started  # raw-response serialization/disk writes happen later
    good = [r for r in rows if r["ok"]]
    tokens = sum(r["completion_tokens"] for r in good)
    latency = [r["latency_s"] for r in good]
    total_latency = [r["client_total_s"] for r in good]
    expected = {(w.name, rep) for w in workloads for rep in range(repeats)}
    return {"concurrency": concurrency, "wall_s": wall, "requests": len(rows),
            "ok": len(good), "errors": len(rows) - len(good), "completion_tokens": tokens,
            "tok_s": tokens / wall if wall > 0 else None,
            "latency_median_s": statistics.median(latency) if latency else None,
            "latency_p95_s": nearest_rank(latency, 0.95),
            "client_total_p95_s": nearest_rank(total_latency, 0.95),
            "latency_samples": len(latency), "percentile_method": "nearest-rank",
            "integrity": trial_integrity(rows, expected),
            "stability": repeat_stability(rows, [w.name for w in workloads]),
            "rows": sorted(rows, key=lambda r: (r["rep"], r["workload"]))}


def stop_server(proc: subprocess.Popen) -> None:
    if proc.poll() is not None:
        return
    try:
        if os.name == "posix":
            os.killpg(proc.pid, signal.SIGTERM)  # only our start_new_session process group
        elif hasattr(signal, "CTRL_BREAK_EVENT"):
            proc.send_signal(signal.CTRL_BREAK_EVENT)
        else:
            proc.terminate()
        proc.wait(timeout=20)
    except (ProcessLookupError, OSError, subprocess.TimeoutExpired):
        if proc.poll() is None:
            if os.name == "posix":
                os.killpg(proc.pid, signal.SIGKILL)
            else:
                proc.kill()
            proc.wait(timeout=10)


def server_command(args, arm, level: int, model_id: str, request_log: Path) -> list[str]:
    capacity = level if args.server_capacity == "level" else int(args.server_capacity)
    command = [str(args.serve), str(args.model), "--host", "127.0.0.1", "--port", str(args.port),
               "--model-id", model_id, "--max-context", str(args.max_context),
               "--kv-capacity", str(args.kv_capacity or args.max_context),
               "--max-concurrency", str(capacity), "--kv-dtype", args.kv_dtype,
               "--request-log-jsonl", str(request_log), "--log-stats-interval-ms", "1000",
               "--greedy", "--presence-penalty", "0", "--frequency-penalty", "0"]
    if args.thinking == "off":
        command.append("--no-thinking")
    if args.prefix_reuse == "off":
        command.append("--no-prefix-reuse")
    if args.cuda_graph == "off":
        command.append("--no-cuda-graph")
    return command + list(arm.args)


def run_trial(args, arm, workloads, comparison: str, level: int, pair: int, side: str) -> dict:
    directory = args.out / comparison / f"c{level}-p{pair:02d}-{side}-{arm.name}"
    directory.mkdir(parents=True, exist_ok=False)
    request_log = directory / "requests.jsonl"
    model_id = "ab-" + uuid.uuid4().hex
    command = server_command(args, arm, level, model_id, request_log)
    result = {"comparison": comparison, "arm": arm.name, "side": side, "pair": pair,
              "concurrency": level, "command": command, "directory": str(directory), "rows": []}
    process = None
    try:
        # Fail before loading a model instead of measuring an existing production server.
        with socket.socket() as probe:
            if os.name == "posix":
                # Match the server bind policy: TIME_WAIT from our previous trial is not a
                # live listener. SO_REUSEPORT is deliberately not enabled.
                probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            probe.bind(("127.0.0.1", args.port))
        with (directory / "server.log").open("wb") as log:
            spawn = {"start_new_session": True} if os.name == "posix" else {
                "creationflags": subprocess.CREATE_NEW_PROCESS_GROUP}
            process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, **spawn)
            try:
                wait_ready(args.port, process, args.startup_timeout, model_id)
                result.update(run_level(f"http://127.0.0.1:{args.port}", model_id, workloads,
                                        level, args.repeats, args.request_timeout,
                                        thinking=args.thinking, seed=args.seed, max_new=args.max_new))
            finally:
                stop_server(process)
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        # Preserve completed trial evidence and always stop the owned server, including Ctrl-C.
        if process is not None:
            stop_server(process)
        result["runtime"] = runtime_evidence(request_log, len(workloads) * args.repeats)
        with (directory / "raw-responses.jsonl").open("w", encoding="utf-8") as stream:
            for row in result["rows"]:
                stream.write(canonical_json(row) + "\n")
        (directory / "trial.json").write_text(json.dumps(result, ensure_ascii=False, indent=2,
                                                        allow_nan=False) + "\n", encoding="utf-8")
    return result


def evaluate(trials: list[dict], experiments, workloads, levels: list[int], pairs: int,
             repeats: int, discard: int) -> dict:
    expected = {(w.name, rep) for w in workloads for rep in range(repeats)}
    comparisons = {}
    for name, base, candidate in experiments:
        result_levels = []
        for level in levels:
            group = [t for t in trials if t["comparison"] == name and t["concurrency"] == level]
            keys = [(t["pair"], t["side"]) for t in group]
            expected_trials = {(p, s) for p in range(pairs) for s in ("base", "candidate")}
            complete = len(keys) == len(expected_trials) and set(keys) == expected_trials
            index = {(t["pair"], t["side"]): t for t in group}
            gates, ratios = [], []
            for pair in range(pairs):
                b, c = index.get((pair, "base"), {}), index.get((pair, "candidate"), {})
                gate = compare_responses(b.get("rows", []), c.get("rows", []), expected)
                gates.append({"pair": pair, **gate})  # discard applies to performance ONLY
                bs, cs = b.get("tok_s"), c.get("tok_s")
                if pair >= discard and bs and cs and bs > 0 and cs > 0 and math.isfinite(bs) and math.isfinite(cs):
                    ratios.append(cs / bs)
            stability = {}
            for side in ("base", "candidate"):
                rows = [r for t in group if t["side"] == side for r in t.get("rows", [])]
                stability[side] = repeat_stability(rows, [w.name for w in workloads])
            exact = complete and all(g["passed"] for g in gates)
            stable = all(s["passed"] for s in stability.values())
            runtime_complete = complete and all(t.get("runtime", {}).get("complete") for t in group)
            optimization_observed = True
            if "--spec-tree" in candidate.args and "lattice" in candidate.args:
                optimization_observed = any(t.get("runtime", {}).get("tree_rounds", 0) > 0
                                            for t in group if t["side"] == "candidate" and t["pair"] >= discard)
            elif "--lookup-dflash" in candidate.args and "skip" in candidate.args:
                optimization_observed = any(t.get("runtime", {}).get("head_skip_rounds", 0) > 0
                                            for t in group if t["side"] == "candidate" and t["pair"] >= discard)
            gate_ok = exact and stable and runtime_complete and not any(t.get("error") for t in group)
            measurable = len(ratios) == pairs - discard
            usage_equal = complete and all(g["usage_equal"] for g in gates)
            eligible = (gate_ok and measurable and optimization_observed and usage_equal
                        and base.name != candidate.name)
            median = statistics.median(ratios) if ratios else None
            status = "baseline_control" if gate_ok and base.name == candidate.name else "unqualified"
            if eligible:
                status = "insufficient_pairs" if len(ratios) < 3 else "unresolved"
                if len(ratios) >= 3 and all(r > 1 for r in ratios) and median >= 1.02:
                    status = "consistent_gain"
                elif len(ratios) >= 3 and all(r < 1 for r in ratios) and median <= 0.98:
                    status = "consistent_regression"
            result_levels.append({"concurrency": level, "complete": complete,
                                  "exact": exact, "stable": stable, "gate_passed": gate_ok,
                                  "runtime_complete": runtime_complete, "usage_equal": usage_equal,
                                  "optimization_observed": optimization_observed,
                                  "stability": stability, "response_checks": gates,
                                  "diagnostic_pair_speedups": ratios,
                                  "diagnostic_median_speedup": median,
                                  "qualified_speedup": median if eligible and len(ratios) >= 3 else None,
                                  "performance_status": status,
                                  "runtime_by_trial": [{"pair": t["pair"], "side": t["side"],
                                                        **t.get("runtime", {})} for t in group]})
        comparisons[name] = {"base": base.name, "candidate": candidate.name, "levels": result_levels}
    return comparisons


def save_result(path: Path, result: dict) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                         encoding="utf-8")
    temporary.replace(path)


def main() -> None:
    from tools.dflash2_training.ab_suite import select_arms, select_workloads

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--serve", type=Path, default=Path("build/apps/ninfer-serve"))
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--arms", default="baseline,dflash2-k15,tree15,tree15-stair,lookup-skip")
    ap.add_argument("--workloads", default="all")
    ap.add_argument("--concurrency", default="1,2,4,8")
    ap.add_argument("--server-capacity", choices=("level", "1", "2", "4", "8"), default="8",
                    help="startup capacity, distinct from offered client concurrency")
    ap.add_argument("--pairs", type=int, default=4)
    ap.add_argument("--discard", type=int, default=1)
    ap.add_argument("--repeats", type=int, default=2)
    ap.add_argument("--thinking", choices=("off", "model-default"), default="off")
    ap.add_argument("--prefix-reuse", choices=("on", "off"), default="on")
    ap.add_argument("--cuda-graph", choices=("on", "off"), default="on")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--port", type=int, default=18080)
    ap.add_argument("--startup-timeout", type=float, default=180.0)
    ap.add_argument("--request-timeout", type=float, default=300.0)
    ap.add_argument("--cooldown", type=float, default=8.0)
    ap.add_argument("--kv-dtype", default="int8")
    ap.add_argument("--max-context", type=int, default=32768)
    ap.add_argument("--kv-capacity", type=int, default=None)
    ap.add_argument("--max-new", type=int, default=512)
    args = ap.parse_args()
    try:
        levels = [int(x) for x in args.concurrency.split(",")]
    except ValueError:
        ap.error("--concurrency requires comma-separated integers")
    if not levels or len(set(levels)) != len(levels) or any(not 1 <= c <= 8 for c in levels):
        ap.error("distinct client concurrency levels must be in [1,8]")
    if not 0 <= args.discard < args.pairs or args.repeats < 2:
        ap.error("need pairs > discard >= 0 and repeats >= 2 for self-stability")
    if args.server_capacity != "level" and int(args.server_capacity) < max(levels):
        ap.error("server capacity must cover offered client concurrency")
    if not 1 <= args.port <= 65535 or min(args.max_context, args.max_new) <= 0:
        ap.error("invalid port/context/output limit")
    if args.kv_capacity is not None and args.kv_capacity <= 0:
        ap.error("--kv-capacity must be positive")
    if not 0 <= args.seed <= (1 << 64) - 1 or not all(math.isfinite(v) for v in
                               (args.cooldown, args.startup_timeout, args.request_timeout)):
        ap.error("invalid seed or non-finite duration")
    if args.cooldown < 0 or min(args.startup_timeout, args.request_timeout) <= 0:
        ap.error("invalid cooldown or timeout")
    args.serve, args.model = args.serve.resolve(), args.model.resolve()
    if not args.serve.is_file() or not os.access(args.serve, os.X_OK) or not args.model.is_file():
        ap.error("provide an executable --serve and existing --model artifact")
    arms, workloads = select_arms(args.arms), select_workloads(args.workloads)
    by_name = {a.name: a for a in arms}
    experiments = [(a.name, by_name[a.compare_to], a) for a in arms if a.name != "baseline"]
    if not experiments and "baseline" in by_name:
        experiments = [("baseline-self", by_name["baseline"], by_name["baseline"])]
    if not workloads or not experiments:
        ap.error("at least one workload and experiment are required")
    if len(workloads) * args.repeats < max(levels):
        ap.error("offered job count cannot fill requested concurrency; increase --repeats")
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    args.out = (args.out or Path("profiles") / ("ab-server-" + stamp + "-" + uuid.uuid4().hex[:8])).resolve()
    if args.out.exists() and (not args.out.is_dir() or any(args.out.iterdir())):
        ap.error("--out must be absent or empty; existing evidence is never overwritten")
    args.out.mkdir(parents=True, exist_ok=True)
    result = {"version": 2, "status": "running", "config": {k: str(v) if isinstance(v, Path) else v
                                                              for k, v in vars(args).items()},
              "environment": {"python": platform.python_version(), "platform": platform.platform(),
                              "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES")},
              "workloads": [dataclasses.asdict(w) for w in workloads],
              "arms": [dataclasses.asdict(a) for a in arms], "trials": []}
    path = args.out / "server-results.json"
    save_result(path, result)
    try:
        for name, base, candidate in experiments:
            for level in levels:
                for pair in range(args.pairs):
                    # Use side, not arm name: baseline-self has the same arm on both sides.
                    sides = ("base", "candidate") if pair % 2 == 0 else ("candidate", "base")
                    for side in sides:
                        arm = base if side == "base" else candidate
                        print(f"[ab] {name} C{level} pair={pair} {side}={arm.name}", flush=True)
                        trial = run_trial(args, arm, workloads, name, level, pair, side)
                        result["trials"].append(trial)
                        save_result(path, result)
                        if args.cooldown:
                            time.sleep(args.cooldown)
        result["status"] = "complete"
    except KeyboardInterrupt:
        result["status"] = "interrupted"
    finally:
        result["comparisons"] = evaluate(result["trials"], experiments, workloads, levels,
                                           args.pairs, args.repeats, args.discard)
        save_result(path, result)
    lines = ["# Isolated server A/B", "", "Full response + self-stability gate; no numerical tolerance.",
             "Throughput = completion usage / request-level wall time. Archive writes excluded.",
             "Diagnostics are not validated speedup claims. Runtime counts are trial aggregates.", "",
             "| comparison | C | exact | self stable | qualified speedup | diagnostic ratio | status |",
             "|---|---:|---|---|---:|---:|---|"]
    for name, comparison in result["comparisons"].items():
        for row in comparison["levels"]:
            valid, diagnostic = row["qualified_speedup"], row["diagnostic_median_speedup"]
            vtext, dtext = (f"{valid:.3f}x" if valid is not None else "—"), (f"{diagnostic:.3f}x" if diagnostic is not None else "—")
            lines.append(f"| {name} | {row['concurrency']} | {row['exact']} | {row['stable']} | "
                         f"{vtext} | {dtext} | {row['performance_status']} |")
    (args.out / "server-summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {path}", flush=True)
    passed = result["status"] == "complete" and all(row["gate_passed"]
              for c in result["comparisons"].values() for row in c["levels"])
    raise SystemExit(130 if result["status"] == "interrupted" else (0 if passed else 2))


if __name__ == "__main__":
    main()
