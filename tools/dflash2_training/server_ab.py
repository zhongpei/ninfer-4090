#!/usr/bin/env python3
"""Server-side speculative decoding A/B: serial latency and concurrent throughput.

Each arm starts a fresh ninfer-serve process so baseline/candidate do not share adaptive state,
prefix caches, or allocator history. Every concurrency level runs the same prompt set under greedy
sampling. The harness reports request latency, aggregate completion-token throughput, error rate,
and per-prompt output hashes.

NInfer currently supports max_concurrency <= 8, so the default matrix is 1,2,4,8 rather than the
1,4,8,16,32 matrix commonly used by DFlash/SGLang benchmarks.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import statistics
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path

from tools.dflash2_training.ab_suite import ARMS, WORKLOADS, select_arms, select_workloads


def request_json(url: str, payload: dict | None = None, timeout: float = 300.0) -> dict:
    data = None if payload is None else json.dumps(payload).encode()
    req = urllib.request.Request(
        url, data=data,
        headers={"Content-Type": "application/json"},
        method="GET" if payload is None else "POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as res:
        return json.loads(res.read())


def wait_ready(port: int, proc: subprocess.Popen, timeout: float) -> str:
    deadline = time.monotonic() + timeout
    base = f"http://127.0.0.1:{port}"
    last = ""
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"server exited during startup rc={proc.returncode}")
        try:
            health = request_json(base + "/health", timeout=2)
            if health.get("status") == "ok":
                models = request_json(base + "/v1/models", timeout=5)
                return str(models["data"][0]["id"])
        except Exception as exc:  # readiness polling
            last = str(exc)
        time.sleep(0.5)
    raise TimeoutError("server readiness timed out: " + last)


def one_request(base: str, model_id: str, prompt: str, max_tokens: int, timeout: float) -> dict:
    payload = {
        "model": model_id,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0,
        "stream": False,
    }
    started = time.perf_counter()
    try:
        out = request_json(base + "/v1/chat/completions", payload, timeout)
        elapsed = time.perf_counter() - started
        text = out["choices"][0]["message"].get("content") or ""
        usage = out.get("usage") or {}
        completion = int(usage.get("completion_tokens") or 0)
        return {
            "ok": True,
            "latency_s": elapsed,
            "completion_tokens": completion,
            "sha256": hashlib.sha256(text.encode()).hexdigest(),
            "text_bytes": len(text.encode()),
        }
    except Exception as exc:
        return {"ok": False, "latency_s": time.perf_counter() - started, "error": str(exc)}


def run_level(base: str, model_id: str, workloads, concurrency: int, repeats: int,
              timeout: float) -> dict:
    jobs = []
    for rep in range(repeats):
        for w in workloads:
            jobs.append((w.name, w.prompt, min(w.max_new, 512), rep))

    started = time.perf_counter()
    rows = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = {
            pool.submit(one_request, base, model_id, prompt, max_new, timeout):
                (name, rep)
            for name, prompt, max_new, rep in jobs
        }
        for future in concurrent.futures.as_completed(futures):
            name, rep = futures[future]
            row = future.result()
            row.update({"workload": name, "rep": rep})
            rows.append(row)
    wall = time.perf_counter() - started

    good = [r for r in rows if r["ok"]]
    tokens = sum(r["completion_tokens"] for r in good)
    lat = [r["latency_s"] for r in good]
    return {
        "concurrency": concurrency,
        "wall_s": wall,
        "requests": len(rows),
        "ok": len(good),
        "errors": len(rows) - len(good),
        "completion_tokens": tokens,
        "tok_s": tokens / wall if wall > 0 else 0.0,
        "latency_median_s": statistics.median(lat) if lat else None,
        "latency_p95_s": sorted(lat)[max(0, int(0.95 * len(lat)) - 1)] if lat else None,
        "rows": rows,
    }


def stop_server(proc: subprocess.Popen) -> None:
    if proc.poll() is not None:
        return
    try:
        proc.terminate()
        proc.wait(timeout=20)
    except Exception:
        proc.kill()
        proc.wait(timeout=10)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--serve", type=Path, default=Path("build-ninja/apps/ninfer-serve"))
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--out", default="profiles/ab-server")
    ap.add_argument("--arms", default="baseline,dflash2-k15,tree15,tree15-stair,lookup-skip")
    ap.add_argument("--workloads", default="prose,chat,reasoning,code,lookup-repeat")
    ap.add_argument("--concurrency", default="1,2,4,8")
    ap.add_argument("--repeats", type=int, default=2)
    ap.add_argument("--port", type=int, default=18080)
    ap.add_argument("--startup-timeout", type=float, default=180.0)
    ap.add_argument("--request-timeout", type=float, default=300.0)
    ap.add_argument("--cooldown", type=float, default=8.0)
    ap.add_argument("--kv-dtype", default="int8")
    ap.add_argument("--max-context", type=int, default=32768)
    args = ap.parse_args()

    # select_arms recursively pulls every comparator dependency, so selecting tree15-stair also
    # keeps tree15 -> dflash2-k15 -> baseline and every reported speedup has a real control.
    arms = select_arms(args.arms)
    workloads = select_workloads(args.workloads)
    levels = [int(x) for x in args.concurrency.split(",") if x.strip()]
    if not levels or max(levels) > 8:
        raise SystemExit("server A/B concurrency must be within NInfer's current [1,8] limit")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    all_results = {}
    for index, arm in enumerate(arms):
        port = args.port + index
        log_path = out / f"{arm.name}.server.log"
        request_log = out / f"{arm.name}.requests.jsonl"
        cmd = [
            str(args.serve), str(args.model),
            "--host", "127.0.0.1", "--port", str(port),
            "--max-context", str(args.max_context),
            "--max-concurrency", str(max(levels)),
            "--kv-dtype", args.kv_dtype,
            "--request-log-jsonl", str(request_log),
            "--log-stats-interval-ms", "1000",
            *arm.args,
        ]
        print("[server]", arm.name, " ".join(cmd), flush=True)
        log = open(log_path, "wb")
        proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=os.environ.copy())
        try:
            model_id = wait_ready(port, proc, args.startup_timeout)
            base = f"http://127.0.0.1:{port}"
            arm_rows = []
            for level in levels:
                print(f"  concurrency={level}", flush=True)
                arm_rows.append(
                    run_level(base, model_id, workloads, level, args.repeats,
                              args.request_timeout)
                )
                if args.cooldown:
                    time.sleep(args.cooldown)
            all_results[arm.name] = {
                "compare_to": arm.compare_to,
                "command": cmd,
                "levels": arm_rows,
            }
        finally:
            stop_server(proc)
            log.close()
        if args.cooldown:
            time.sleep(args.cooldown)

    # Compare same workload/rep hashes and aggregate throughput at each concurrency.
    comparisons = {}
    for arm in arms:
        if arm.name == "baseline" or arm.compare_to not in all_results:
            continue
        base = all_results[arm.compare_to]
        cand = all_results[arm.name]
        levels_out = []
        for b, c in zip(base["levels"], cand["levels"]):
            b_hash = {(r["workload"], r["rep"]): r.get("sha256") for r in b["rows"] if r["ok"]}
            c_hash = {(r["workload"], r["rep"]): r.get("sha256") for r in c["rows"] if r["ok"]}
            keys = sorted(set(b_hash) | set(c_hash))
            mismatches = [k for k in keys if b_hash.get(k) != c_hash.get(k)]
            levels_out.append({
                "concurrency": c["concurrency"],
                "base_tok_s": b["tok_s"],
                "candidate_tok_s": c["tok_s"],
                "speedup": c["tok_s"] / b["tok_s"] if b["tok_s"] > 0 else None,
                "base_latency_median_s": b["latency_median_s"],
                "candidate_latency_median_s": c["latency_median_s"],
                "errors": c["errors"],
                "exact": not mismatches,
                "mismatches": [{"workload": x[0], "rep": x[1]} for x in mismatches],
            })
        comparisons[arm.name] = {"base": arm.compare_to, "levels": levels_out}

    result = {
        "version": 1,
        "model": str(args.model),
        "kv_dtype": args.kv_dtype,
        "workloads": [w.name for w in workloads],
        "results": all_results,
        "comparisons": comparisons,
    }
    (out / "server-results.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    lines = [
        "# NInfer server speculative A/B",
        "",
        "| candidate | base | concurrency | exact | tok/s speedup | median latency base -> cand | errors |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for name, comp in comparisons.items():
        for row in comp["levels"]:
            speed = "" if row["speedup"] is None else f"{row['speedup']:.3f}x"
            lat = (
                f"{row['base_latency_median_s']:.3f}s -> {row['candidate_latency_median_s']:.3f}s"
                if row["base_latency_median_s"] is not None
                and row["candidate_latency_median_s"] is not None else ""
            )
            lines.append(
                f"| {name} | {comp['base']} | {row['concurrency']} | "
                f"{'PASS' if row['exact'] else 'FAIL'} | {speed} | {lat} | {row['errors']} |"
            )
    (out / "server-summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {out / 'server-results.json'}")
    print(f"wrote {out / 'server-summary.md'}")

    if any(not row["exact"] or row["errors"]
           for comp in comparisons.values() for row in comp["levels"]):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
