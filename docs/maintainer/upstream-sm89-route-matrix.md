# Unified RTX 4090 route experiment (opt-in)

Source: `iamwavecut/ninfer-all@e1debb45108d74aee199816d942e3bb8145d0feb`.
Target: `zhongpei/ninfer-4090` / sm_89. This imports route infrastructure and
select kernel **candidates**, not a blanket replacement of custom FP8, DFlash2,
paged-KV or context-cache code.

## Available switches

- `NINFER_DEVICE_ROUTE_MODE=off` (default): existing production mapping, **no upstream routing**.
- `NINFER_DEVICE_ROUTE_MODE=builtin`: bundled upstream RTX 4090 route profile (exact 128 SM device only).
- `NINFER_DEVICE_ROUTE_MODE=file` + `NINFER_DEVICE_PROFILE_PATH=/absolute/path.json`: use external version-2 profile; mismatched GPU identity is rejected.
- `NINFER_DEVICE_ROUTE_MODE=auto`: use that file if available, else the built-in 4090 profile. **This is profile auto-selection, not online benchmarking.**
- `NINFER_DEVICE_ROUTE_ONLY=attn_prompt_fast,gdn_two_stage/h48`: restrict the active profile to named keys for isolation; an empty list leaves all keys available.
- `NINFER_DEVICE_ROUTE_OVERRIDES='attn_prompt_fast=off;gdn_two_stage/h48=on'`: override complete width ranges after profile selection. A JSON profile supports width-specific bands.
- `NINFER_PROMPT_FAST=0|1`: force old/fast prompt kernel where implemented.
- `NINFER_GDN_TWO_STAGE=0|1`: force recurrent/two-stage GDN, for prefill width >=16.
- `NINFER_PREFILL_ALIGN=0|1`: enable wave occupancy alignment of fixed INT8-family prompt chunks. This only reduces the requested physical chunk; auto chunk remains unchanged.

The upstream reference contains 65 **route keys**. This is not 65 activated new
kernels. Routing is meaningful only for ops with a local candidate hook.
Currently: INT8/rk8v4 fast prompt attention, h32/h48 two-stage GDN and
whole-SM wave-aligned fixed prompt chunk. Other keys are retained as reference
data for future qualified kernels; they do not override local dispatches.

**Important**: the bundled device profile selects candidates built for another
fork and is therefore *not qualified* for this checkout. Never use `builtin` or
`auto` in production before the correctness and performance gates below.
The default remains `off`.

## RTX 4090 test matrix

Use the *same binary, same model, same driver*, idle GPU, same
`--max-context`, `--kv-dtype`, Graph settings and DFlash2 K.

1. Compile with `-DCMAKE_CUDA_ARCHITECTURES=89`; run the existing operator
   conformance tests before any performance comparison.
2. `off`: baseline. Repeat 5 warmups and 10+ timed AB/BA paired runs.
3. Isolated GDN: `NINFER_DEVICE_ROUTE_MODE=builtin NINFER_DEVICE_ROUTE_ONLY=gdn_two_stage/h48`.
   Test prompt widths 15/16/17/64/128/1024/8192 and full GDN state recovery;
   include FP16/FP32 state, separated input/output, rewrite checkpoints and
   chunk-boundary equivalence.
4. Isolated attention: `NINFER_DEVICE_ROUTE_ONLY=attn_prompt_fast`.
   Test int8/rk8v4, 1K/4K/16K/32K/131K prompts, KV pages and
   FP32/FP16 numerical reference, long-context PPL and needle retrieval.
5. Isolated wave alignment: with fixed `--prefill-chunk` of 1024, 1536,
   2048, 3072, 4096 and 8192 run `NINFER_PREFILL_ALIGN=0` and `=1`.
   Compare resolved physical chunk, reservation, workspace and end-to-end
   throughput. Auto chunk must be unaffected.
6. Combine only **passing** candidates. Compare `off`, isolated winners and
   combined winners at concurrency 1/2/4/8, DFlash2 K7 and K15, with actual
   Agent request streams.

Guardrails:

- Correctness first: numerical oracle, long-context perplexity, exact prefix
  recovery, and greedy output checks. Kernel numerical differences may move
  near-tied greedy decisions; investigate every mismatch, not only aggregate PPL.
- Report short/long `TTFT`, prompt tok/s, decode tok/s, request p50/p95,
  acceptance rate, GPU memory *peak*, Graph capture allocation and retries.
- Give each pair the same prompt and GPU context; alternate AB and BA orders,
  discard warmup and calculate median and p95 across >=10 pairs.
- Promote a candidate only if there is a real end-to-end benefit under at least
  one named workload, no unexpected quality regression, <=5% request-p95
  regression for unaffected workloads, and no memory or startup-capacity loss.
- If a candidate fails, leave it off. An unsupported upstream route name must
  fall back to the locally compiled path, never silently pretend to be selected.
- Caution: a new GDN prefill algorithm can change floating-point summation order.
  Its test must cover boundary/state-image invariants, not just model output.

The local test runner `tools/bench/run_upstream_route_matrix.py` writes an
audit manifest for each route configuration without requiring GPU in CI.

Example profile:
```json
{
  "schema": "ninfer.device-route-profiles",
  "schema_version": 2,
  "devices": [{
    "hardware_class": "nvidia-geforce-rtx-4090-sm89",
    "multiprocessors": 128,
    "origin": "operator local AB",
    "routes": {
      "attn_prompt_fast": [[2147483647, "on"]],
      "gdn_two_stage/h48": [[2147483647, "off"]],
      "gdn_two_stage/h32": [[2147483647, "off"]],
      "prefill_align": [[2147483647, "off"]]
    }
  }]
}
```

## Running the isolated route A/B locally

Build from this PR branch on the RTX 4090 host (clean Release, `sm_89`):

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES=89 -DNINFER_BUILD_BENCHMARKS=ON -DBUILD_TESTING=ON
cmake --build build -j 8 --target ninfer_bench
python3 -m unittest tools.bench.test_upstream_route_matrix -v

python3 -m tools.bench.run_upstream_route_matrix \
  --exe ./build/bench/ninfer_bench \
  --model /absolute/path/to/Ternary-Bonsai-2-27B.ninfer \
  --out profiles/bench/upstream-sm89-first-pass \
  --device 0 --max-context 32768 --kv-dtype int8 \
  --spec dflash2 --draft-tokens 7 \
  --prefill-chunk 1024 --prompts 1024,4096,16384,32768 \
  --cases prompt_fast,gdn_two_stage,t2_upstream,sm_wave,all_candidates \
  --warmup 1 --pairs 10
```

For `rk8v4` change `--kv-dtype` to the local CLI's supported
spelling and use a *new* `--out` directory. If testing longer context, set
`--max-context` accordingly, keep the model and numeric flags identical.

This script restarts the benchmark binary for each arm: it runs a full
`ninfer_bench` JSON test sequence, not a microkernel-only loop. Each
candidate's baseline is taken from its own adjacent AB/BA pairs. See
`summary.md`, `summary.json`, and the raw reports and invocation JSON.
A `performance_only_screen: true` is **not** production qualification.
You must separately check numerical and serving gates.

A custom case matrix can be supplied via `--cases-json cases.json`.
For example, only enable the candidate T2 A16 route on a local profile:

```json
{
  "baseline": {"NINFER_DEVICE_ROUTE_MODE": "off"},
  "t2_only": {
    "NINFER_DEVICE_ROUTE_MODE": "builtin",
    "NINFER_DEVICE_ROUTE_ONLY": "t2_a16",
    "NINFER_DEVICE_ROUTE_OVERRIDES": "t2_a16=upstream"
  }
}
```

Then add `--cases t2_only --cases-json cases.json`. The script emits reports
without altering the serving configuration.

**To qualify an Agent-serving deployment**, conduct a separate open-loop
client A/B against two otherwise identical server configurations: fixed
32K/131K histories, 1/2/4/8 active requests, Qwen tool calls,
re-ask/regenerate/edited-last-turn branches, and 512-token continuation.
Capture TTFT p50/p95, wall requests/s, final token/s, draft acceptance,
prompt cache hit depth, 429/5xx, GPU peak memory and exact greedy outputs.
A candidate that wins isolated prefill but regresses these actual scenarios
should remain opt-in.

## Route selection diagnostics

For a separate **diagnostic** run, add `--route-trace` to the matrix script
with only one candidate and three pairs in a new output directory. The runtime
prints lines such as:

```text
ninfer route applied device=0 key=attn_prompt_fast width=1 schedule=on
ninfer route applied device=0 key=gdn_two_stage/h48 width=1024 schedule=on
ninfer route applied device=0 key=t2_a16 width=8 schedule=upstream
```

This trace is limited to the first observation of a device/key/schedule.
**Do not enable it during timing qualification**, because the extra host locks
and logging can distort throughput. A route profile's loaded key count is
not evidence that its kernels were executed. Use the actual trace and the
reported `config` field to verify selected routes and resolved chunk capacity.
