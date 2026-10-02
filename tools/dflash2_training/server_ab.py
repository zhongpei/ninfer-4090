#!/usr/bin/env python3
"""Paired, isolated server A/B with full-response and self-repeat qualification.

One fresh server per comparison/pair/client-concurrency/arm. Engine concurrency
stays fixed. Startup is excluded from throughput; all measured HTTP exchanges and
client queueing are included. Raw archives are written after the timed request batch.
A failed exactness or repeatability gate never produces a qualified speedup.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
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
from pathlib import Path
from types import SimpleNamespace

from tools.dflash2_training.ab_suite import select_arms, select_workloads
from tools.dflash2_training.cli_validation import file_identity
from tools.dflash2_training.server_validation import (
    canonical_response, compare_runs, completion_tokens, digest, inspect_run,
    json_bytes, nearest_rank, runtime_counters,
)


def http_exchange(url: str, payload: dict | None, timeout: float) -> tuple[int, dict, bytes]:
    # This harness only talks to the loopback server it launched. Do not route it
    # through HTTP_PROXY/HTTPS_PROXY, common on the deployment host.
    request = urllib.request.Request(
        url, data=None if payload is None else json_bytes(payload),
        headers={"Content-Type": "application/json", "Connection": "close"},
        method="GET" if payload is None else "POST")
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        with opener.open(request, timeout=timeout) as response:
            return response.status, dict(response.headers), response.read()
    except urllib.error.HTTPError as error:
        return error.code, dict(error.headers), error.read()


def request_json(url: str, payload: dict | None = None, timeout: float = 300.0) -> dict:
    status, _, raw = http_exchange(url, payload, timeout)
    if status != 200:
        raise RuntimeError(f"HTTP {status}: {raw[:256]!r}")
    return json.loads(raw)


def wait_ready(port: int, proc: subprocess.Popen, timeout: float) -> str:
    deadline = time.monotonic() + timeout
    base = f"http://127.0.0.1:{port}"
    last = "no response"
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"server exited during startup rc={proc.returncode}")
        try:
            if request_json(base + "/health", timeout=2).get("status") == "ok":
                models = request_json(base + "/v1/models", timeout=5)
                model_id = models["data"][0]["id"]
                if not isinstance(model_id, str) or not model_id:
                    raise ValueError("empty model id")
                if proc.poll() is not None:
                    raise RuntimeError("own server exited after readiness")
                return model_id
        except (OSError, ValueError, KeyError, IndexError, RuntimeError) as exc:
            last = str(exc)
        time.sleep(0.2)
    raise TimeoutError("server readiness timed out: " + last)


def one_request(base: str, model_id: str, prompt: str, max_tokens: int, timeout: float,
                *, thinking: str = "off", submitted_at: float | None = None) -> dict:
    started = time.perf_counter()
    submitted_at = started if submitted_at is None else submitted_at
    payload = {"model": model_id, "messages": [{"role": "user", "content": prompt}],
               "max_tokens": max_tokens, "temperature": 0, "presence_penalty": 0,
               "frequency_penalty": 0, "seed": 0, "stream": False}
    row = {"ok": False, "request": payload, "client_queue_s": started - submitted_at,
           "_raw": b"", "response_headers": {}, "http_status": None}
    try:
        status, headers, raw = http_exchange(base + "/v1/chat/completions", payload, timeout)
        row.update(http_status=status, response_headers=headers, _raw=raw)
        # Capture complete wire body, including JSON/API errors, before validating it.
        if status != 200:
            raise ValueError(f"HTTP {status}")
        response = json.loads(raw)
        output = canonical_response(response, thinking=thinking)
        tokens = completion_tokens(response)
        row.update(ok=True, response=response, output=output, sha256=digest(output),
                   completion_tokens=tokens,
                   content_bytes=len(output["content"].encode("utf-8")),
                   reasoning_bytes=len(output["reasoning_content"].encode("utf-8")))
    except (OSError, ValueError, KeyError, TypeError) as exc:
        row["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        finished = time.perf_counter()
        row["latency_s"] = finished - started
        row["e2e_latency_s"] = finished - submitted_at
    return row


def run_level(base: str, model_id: str, workloads, concurrency: int, repeats: int,
              timeout: float, *, thinking: str = "off", max_tokens: int = 512) -> dict:
    jobs = [(w.name, w.prompt, max_tokens, rep) for rep in range(repeats) for w in workloads]
    rows = []
    started = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = {}
        for name, prompt, count, rep in jobs:
            submitted = time.perf_counter()
            future = pool.submit(one_request, base, model_id, prompt, count, timeout,
                                 thinking=thinking, submitted_at=submitted)
            futures[future] = name, rep
        for future in concurrent.futures.as_completed(futures):
            name, rep = futures[future]
            try:
                row = future.result()
            except Exception as exc:
                row = {"ok": False, "error": f"worker failure: {exc}", "_raw": b""}
            row.update(workload=name, rep=rep)
            rows.append(row)
    wall = time.perf_counter() - started
    good = [r for r in rows if r["ok"]]
    latencies = [r["latency_s"] for r in good]
    e2e = [r["e2e_latency_s"] for r in good]
    tokens = sum(r["completion_tokens"] for r in good)
    return {"concurrency": concurrency, "thinking": thinking, "wall_s": wall,
            "requests": len(rows), "ok": len(good), "errors": len(rows) - len(good),
            "completion_tokens": tokens, "tok_s": tokens / wall if wall else None,
            "latency_median_s": statistics.median(latencies) if latencies else None,
            "latency_p95_s": nearest_rank(latencies, 0.95),
            "e2e_latency_p95_s": nearest_rank(e2e, 0.95),
            "rows": sorted(rows, key=lambda r: (r["rep"], r["workload"]))}


def stop_server(proc: subprocess.Popen) -> None:
    if proc.poll() is not None:
        return
    try:
        if os.name == "posix":
            os.killpg(proc.pid, signal.SIGTERM)
        else:
            proc.terminate()
        proc.wait(timeout=20)
    except subprocess.TimeoutExpired:
        if os.name == "posix":
            os.killpg(proc.pid, signal.SIGKILL)
        else:
            proc.kill()
        proc.wait(timeout=10)
    except ProcessLookupError:
        pass


def assert_free_port(port: int) -> None:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        # A previous owned server may leave TIME_WAIT sockets, not a listener.
        # REUSEADDR permits that reuse but does not share a live listening socket.
        if os.name == "posix":
            probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            probe.bind(("127.0.0.1", port))
        except OSError as exc:
            raise RuntimeError(f"port {port} already in use; refusing to use an existing server") from exc


def build_command(args, arm, port: int, request_log: Path) -> list[str]:
    command = [str(args.serve), str(args.model), "--host", "127.0.0.1", "--port", str(port),
               "--max-context", str(args.max_context), "--kv-capacity", str(args.kv_capacity),
               "--max-concurrency", str(args.engine_concurrency), "--kv-dtype", args.kv_dtype,
               "--request-log-jsonl", str(request_log), "--log-stats-interval-ms", "0"]
    if args.thinking == "off":
        command.append("--no-thinking")
    if args.cache_mode == "disabled":
        command.append("--no-prefix-reuse")
    if args.no_cuda_graph:
        command.append("--no-cuda-graph")
    command.extend(arm.args)
    return command


def write_json(path: Path, value: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_bytes(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False).encode() + b"\n")
    tmp.replace(path)


def archive_rows(directory: Path, rows: list[dict]) -> None:
    for row in rows:
        # Independent ordinal is used so duplicate response/job ids cannot overwrite evidence.
        stem = f"response-{row['archive_index']:05d}"
        raw_file = directory / (stem + ".body")
        raw = row.pop("_raw", b"")
        raw_file.write_bytes(raw)
        row["raw_response_path"] = str(raw_file)
        row["raw_response_sha256"] = hashlib.sha256(raw).hexdigest()


def collect_runtime(path: Path, expected_requests: int, thinking: str) -> dict:
    try:
        events = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if any(not isinstance(e, dict) for e in events):
            raise ValueError("non-object runtime event")
        return runtime_counters(events, expected_requests, thinking)
    except (OSError, ValueError, TypeError) as exc:
        return {"status": "unqualified", "errors": [str(exc)], "counters": None,
                "tree_round_share": None}


def run_arm(args, arm, comparison: str, pair: int, order: int, level: int, directory: Path,
            workloads) -> dict:
    directory.mkdir(parents=True, exist_ok=False)
    request_log = directory / "runtime.requests.jsonl"
    command = build_command(args, arm, args.port, request_log)
    result = {"comparison": comparison, "pair": pair, "order": order, "arm": arm.name,
              "concurrency": level, "thinking": args.thinking, "cache_mode": args.cache_mode,
              "cuda_graph": not args.no_cuda_graph, "engine_concurrency": args.engine_concurrency,
              "max_context": args.max_context, "kv_capacity": args.kv_capacity,
              "kv_dtype": args.kv_dtype, "command": command, "rows": [], "tok_s": None}
    proc = None
    try:
        assert_free_port(args.port)
        with (directory / "server.log").open("wb") as log:
            try:
                proc = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                        start_new_session=os.name == "posix", env=os.environ.copy())
                model_id = wait_ready(args.port, proc, args.startup_timeout)
                result.update(run_level(f"http://127.0.0.1:{args.port}", model_id, workloads,
                                        level, args.repeats, args.request_timeout,
                                        thinking=args.thinking, max_tokens=args.max_tokens))
            finally:
                if proc is not None:
                    stop_server(proc)
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
    # Archives and runtime-log processing are deliberately outside the level timer.
    for i, row in enumerate(result["rows"]):
        row["archive_index"] = i
    archive_rows(directory, result["rows"])
    result["runtime"] = collect_runtime(request_log, len(workloads) * args.repeats, args.thinking)
    result["server_returncode"] = proc.returncode if proc is not None else None
    write_json(directory / "run.json", result)
    return result


def summarize(runs: list[dict], arms, workloads, levels: list[int], pairs: int, discard: int,
              repeats: int) -> dict:
    expected = {(w.name, rep) for w in workloads for rep in range(repeats)}
    report = {}
    for candidate in arms:
        if candidate.name == "baseline":
            continue
        name = candidate.name
        levels_out = []
        for level in levels:
            selected = [r for r in runs if r["comparison"] == name and r["concurrency"] == level]
            paired = []
            # Detect run identity collisions before any dictionary overwrites records.
            ids = [(r["pair"], r["arm"]) for r in selected]
            identities_ok = len(ids) == len(set(ids)) == pairs * 2
            for pair in range(pairs):
                members = [r for r in selected if r["pair"] == pair]
                mapping = {r["arm"]: r for r in members}
                a = mapping.get(candidate.compare_to, {"rows": [], "error": "missing baseline run"})
                b = mapping.get(candidate.name, {"rows": [], "error": "missing candidate run"})
                judged = compare_runs(a, b, expected)
                telemetry_ok = all(r.get("runtime", {}).get("status") == "complete" for r in (a, b))
                if not telemetry_ok:
                    judged["reasons"].append("runtime_telemetry_unqualified")
                    judged["qualified"] = False
                    judged["qualified_speedup"] = None
                if not identities_ok:
                    judged["reasons"].append("missing_or_duplicate_run_identity")
                    judged["qualified"] = False
                    judged["qualified_speedup"] = None
                judged.update(pair=pair, performance_retained=pair >= discard,
                              base_runtime=a.get("runtime"), candidate_runtime=b.get("runtime"))
                paired.append(judged)
            # Stability must hold across fresh-process pairs too, not just two rows in one run.
            cross_pair_unstable = {}
            for arm_name in (candidate.compare_to, candidate.name):
                bad = []
                for workload in workloads:
                    hashes = {digest(row["output"]) for run in selected if run["arm"] == arm_name
                              for row in run.get("rows", []) if row.get("ok")
                              and row.get("workload") == workload.name and isinstance(row.get("output"), dict)}
                    if len(hashes) > 1:
                        bad.append(workload.name)
                cross_pair_unstable[arm_name] = bad
            complete = len(selected) == pairs * 2 and identities_ok
            gate = (complete and all(p["qualified"] for p in paired)
                    and not any(cross_pair_unstable.values()))
            diagnostic = [p["diagnostic_speedup"] for p in paired if p["performance_retained"]
                          and p["diagnostic_speedup"] is not None]
            measured = len(diagnostic) == pairs - discard
            qualified = gate and measured
            coverage = {}
            for key in ("tree_rounds", "tree_fallback_rounds", "head_skip_rounds"):
                values = [(p.get("candidate_runtime") or {}).get("counters") for p in paired
                          if p["performance_retained"]]
                coverage[key] = (sum(v[key] for v in values)
                                 if values and all(isinstance(v, dict) and type(v.get(key)) is int
                                                   for v in values) else None)
            levels_out.append({"concurrency": level, "qualification": "PASS" if qualified else "FAIL",
                               "exact": complete and all(p["exact"] for p in paired),
                               "cross_pair_unstable": cross_pair_unstable, "pairs": paired,
                               "candidate_route_coverage": coverage,
                               "diagnostic_median_speedup": statistics.median(diagnostic) if diagnostic else None,
                               "qualified_median_speedup": statistics.median(diagnostic) if qualified else None})
        report[name] = {"base": candidate.compare_to, "levels": levels_out}
    return report


def select_test_arms(names: str, aa: bool):
    if not aa:
        return select_arms(names)
    baseline = next(a for a in select_arms("baseline") if a.name == "baseline")
    return (baseline, SimpleNamespace(name="baseline-aa", compare_to="baseline", args=tuple(baseline.args)))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--serve", type=Path, default=Path("build-ninja/apps/ninfer-serve"))
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("profiles/ab-server"))
    ap.add_argument("--compare-baseline", action="store_true", help="direct target-only control for every selected candidate")
    ap.add_argument("--aa", action="store_true", help="baseline-only A/A repeatability control; overrides --arms")
    ap.add_argument("--arms", default="baseline,dflash2-k15,tree15,tree15-stair,lookup-skip")
    ap.add_argument("--workloads", default="prose,code,long-context")
    ap.add_argument("--concurrency", default="1,2,4,8")
    ap.add_argument("--engine-concurrency", type=int, default=8)
    ap.add_argument("--repeats", type=int, default=2)
    ap.add_argument("--pairs", type=int, default=4)
    ap.add_argument("--discard", type=int, default=1)
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--thinking", choices=("off", "model-default"), default="off")
    ap.add_argument("--cache-mode", choices=("disabled", "enabled"), default="enabled")
    ap.add_argument("--no-cuda-graph", action="store_true")
    ap.add_argument("--port", type=int, default=18080)
    ap.add_argument("--startup-timeout", type=float, default=180.0)
    ap.add_argument("--request-timeout", type=float, default=300.0)
    ap.add_argument("--cooldown", type=float, default=0.0)
    ap.add_argument("--kv-dtype", default="int8")
    ap.add_argument("--max-context", type=int, default=32768)
    ap.add_argument("--kv-capacity", type=int, default=32768)
    args = ap.parse_args()
    arms, workloads = select_test_arms(args.arms, args.aa), select_workloads(args.workloads)
    if args.compare_baseline:
        arms = tuple(SimpleNamespace(name=a.name, compare_to="baseline", args=tuple(a.args)) for a in arms)
    try:
        levels = [int(x) for x in args.concurrency.split(",")]
    except ValueError:
        ap.error("--concurrency must contain integers")
    if (not levels or len(set(levels)) != len(levels) or min(levels) < 1
            or max(levels) > args.engine_concurrency or not 1 <= args.engine_concurrency <= 8):
        ap.error("client concurrency must be unique levels in [1, engine-concurrency <= 8]")
    if (args.repeats < 2 or args.pairs < 1 or not 0 <= args.discard < args.pairs
            or args.max_tokens < 1 or args.max_context < 1 or args.kv_capacity < 1
            or not 1 <= args.port <= 65535 or not workloads
            or not any(a.name != "baseline" for a in arms)):
        ap.error("need >=2 repeats, pairs > discard >=0, positive capacities, workloads and candidates")
    if (not math.isfinite(args.cooldown) or args.cooldown < 0
            or any(not math.isfinite(x) or x <= 0 for x in (args.startup_timeout, args.request_timeout))):
        ap.error("timeouts must be finite/positive and cooldown finite/nonnegative")
    args.serve, args.model = args.serve.resolve(), args.model.resolve()
    if not args.serve.is_file() or not os.access(args.serve, os.X_OK) or not args.model.is_file():
        ap.error("--serve must be executable and --model an existing artifact")
    args.out.mkdir(parents=True, exist_ok=True)
    # Exclusive journal reserves a new result set; never append stale evidence or delete old data.
    journal = (args.out / "runs.jsonl").open("x", encoding="utf-8")
    params = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    metadata = {"version": 2, "settings": params, "platform": platform.platform(),
                "python": platform.python_version(), "cuda_visible_devices": os.getenv("CUDA_VISIBLE_DEVICES"),
                "model_identity": file_identity(args.model), "executable_identity": file_identity(args.serve),
                "workloads": [w.name for w in workloads],
                "timing": "level includes client queue + HTTP + parsing, excludes startup and archive I/O",
                "equivalence": "complete generated response, not token/logit equality",
                "arms": [{"name": a.name, "compare_to": a.compare_to, "args": list(a.args)} for a in arms]}
    write_json(args.out / "manifest.json", metadata)
    by_name = {a.name: a for a in arms}
    runs = []
    interrupted = False
    try:
        for cand in arms:
            if cand.name == "baseline":
                continue
            for level in levels:
                for pair in range(args.pairs):
                    members = (by_name[cand.compare_to], cand)
                    if pair % 2:
                        members = tuple(reversed(members))
                    for order, arm in enumerate(members):
                        directory = args.out / f"{cand.name}__c{level}__p{pair:02d}__o{order}__{arm.name}"
                        print(f"[server-ab] {cand.name} C{level} pair={pair} order={order} {arm.name}", flush=True)
                        result = run_arm(args, arm, cand.name, pair, order, level, directory, workloads)
                        runs.append(result)
                        journal.write(json.dumps(result, ensure_ascii=False, allow_nan=False) + "\n")
                        journal.flush()
                        if args.cooldown:
                            time.sleep(args.cooldown)
    except KeyboardInterrupt:
        interrupted = True
    finally:
        journal.close()
        comparisons = summarize(runs, arms, workloads, levels, args.pairs, args.discard, args.repeats)
        write_json(args.out / "server-results.json", {**metadata, "interrupted": interrupted,
                                                     "runs": runs, "comparisons": comparisons})
    lines = ["# Server qualification", "", "Raw ratios are diagnostic, never release-qualified after any gate failure.", "",
             "| candidate | base | concurrency | qualification | qualified speedup | diagnostic ratio | actual tree / fallback / head-skip |",
             "|---|---|---:|---|---:|---:|---|"]
    for name, comparison in comparisons.items():
        for level in comparison["levels"]:
            qualified = level["qualified_median_speedup"]
            diagnostic = level["diagnostic_median_speedup"]
            q = "—" if qualified is None else f"{qualified:.3f}x"
            d = "—" if diagnostic is None else f"{diagnostic:.3f}x"
            coverage = level["candidate_route_coverage"]
            route = " / ".join("unknown" if coverage[k] is None else str(coverage[k])
                               for k in ("tree_rounds", "tree_fallback_rounds", "head_skip_rounds"))
            lines.append(f"| {name} | {comparison['base']} | {level['concurrency']} | {level['qualification']} | {q} | {d} | {route} |")
    (args.out / "server-summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {args.out / 'server-results.json'}", flush=True)
    if interrupted:
        raise SystemExit(130)
    if not comparisons or any(x["qualification"] != "PASS" for c in comparisons.values() for x in c["levels"]):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
