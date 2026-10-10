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

### GDN correctness gate and fast-math opt-in

The upstream two-stage algorithm does not satisfy the current exact FP32
state/segment-continuation contract: it replaces token-wise recurrence with
BF16/TF32 matrix blocks and a different reduction order. Simply changing a
tolerance would invalidate the existing context-cache/checkpoint contract.

- **Default:** `NINFER_GDN_TWO_STAGE_NUMERICS=exact` (or unset). Even if the
  device profile requests two-stage, the exact recurrent implementation is used.
- **Experiment only:** set `NINFER_GDN_TWO_STAGE_NUMERICS=approx` alongside
  `NINFER_GDN_TWO_STAGE=1` or a profile route. This **does not** pass exact
  state/chunk-split correctness and must not be considered production safe.
- For misaligned BF16 Q/K/V/output views, fast two-stage is now rejected at
  dispatch and the existing recurrent kernel executes instead. Test this with
  shifted +2/+4/+8 byte pointers and CUDA compute-sanitizer.
- Distinguish the actual route from a loaded profile key in benchmark reports.
  `NINFER_DEVICE_ROUTE_TRACE=1` only shows the requested profile; it cannot
  prove that the guarded accelerated Kernel ran.

The next qualification milestone is to develop a chunk-parallel alternative
with an exact state transition or prove a separate approximate-state cache
identity and quality contract. Until then, 8–10% prefill uplift measured on
the old route is an *experimental upper bound*, not a promotable speedup.

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

## Local retest: correctness first, model-resident AB/BA second

This PR now includes a model-resident `ninfer_bench --resident-session` JSONL
protocol. **Only one model materialization/upload occurs per entire matrix**:
the owner holds the 27B weights on GPU while each arm makes and destroys its
own Engine/Program/CUDA Graph. The previous Python tool started and exited
`ninfer_bench` for every arm, reloading the model every time. The new runner
does not do that. Program planning/capture may still occur once per arm;
the report records `program_create_seconds` separately from generation.

The protocol deliberately **never** swaps a CUDA Kernel beneath an active
Graph or concurrent request. The existing `ResidentModelSession` supports
only single-GPU DFlash2 Full at present; don't claim support for other
speculative backends. Each arm's environment is isolated and restored. The
GDN and Prompt Fast environment overrides no longer cache the very first
arm's value for the lifetime of the process.

### 0. Compile and validate the host-side resident protocol

```bash
git fetch origin pull/26/head:pr26-sm89
git switch pr26-sm89
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES=89 \
  -DNINFER_BUILD_BENCHMARKS=ON -DBUILD_TESTING=ON
cmake --build build -j8 --target ninfer_bench ninfer_gated_delta_net_test
python3 -m unittest tools.bench.test_upstream_route_matrix -v
python3 -m py_compile tools/bench/run_upstream_route_matrix.py
git diff --check
```

The Python tests include a **real long-running fake benchmark process**. They
assert exactly one process/model owner across 24 interleaved sample arms,
unique per-case baseline directories, explicit 5-key combination scoping,
response/load-count validation and fail-closed generated-token identity.

### 1. GDN exact correctness gates (must pass)

First exercise the original/default route, then request Two-stage in the
**exact** numerical mode. Both must preserve full-vs-split byte-exact
FP32 state and BF16 output at 59+5 and 123+5, FP16 state conversions, and
shifted-input views.

```bash
# Baseline: should pass unchanged.
env -u NINFER_GDN_TWO_STAGE -u NINFER_GDN_TWO_STAGE_NUMERICS \
  ./build/tests/ninfer_gated_delta_net_test

# Force requested route but insist on existing exact FP32 state semantics:
# the safe recurrent route is selected (NOT the approximate fast algorithm).
NINFER_GDN_TWO_STAGE=1 NINFER_GDN_TWO_STAGE_NUMERICS=exact \
  ./build/tests/ninfer_gated_delta_net_test

# Check illegal memory accesses with shifted BF16 views (must pass).
NINFER_GDN_TWO_STAGE=1 NINFER_GDN_TWO_STAGE_NUMERICS=exact \
  compute-sanitizer --tool memcheck --error-exitcode=99 \
  ./build/tests/ninfer_gated_delta_net_test

# Actually select the fast candidate, then exercise its alignment fallback.
# This test is allowed to pass even though aligned approximate GDN still fails
# the bitwise state-splitting correctness contract.
NINFER_GDN_TWO_STAGE_NUMERICS=approx \
  compute-sanitizer --tool memcheck --error-exitcode=99 \
  ./build/tests/ninfer_gated_delta_net_test --unaligned-fast-only
```

**Fast GDN experiment (not a green gate):** With
`NINFER_GDN_TWO_STAGE_NUMERICS=approx` and a selected two-stage route,
alignment-invalid Q/K/V/output views now fall back to the original safe
recurrent kernel. Aligned fast kernels still retain the upstream BF16/TF32
chunk algorithm, which **does not guarantee exact FP32 state and segmentation
identity**. Capture and report the exact failures; do NOT loosen the existing
oracle or assert that the +8–10% prefill uplift is qualified.

```bash
NINFER_GDN_TWO_STAGE=1 NINFER_GDN_TWO_STAGE_NUMERICS=approx \
  ./build/tests/ninfer_gated_delta_net_test \
  > /tmp/gdn-approx-quality.log 2>&1
# Nonzero exit with aligned split-state mismatches remains a correctness blocker.
```

### 2. Smoke test the resident process (single model load)

Choose a **new** output directory. Change the artifact path below to
the exact local 27B file.

```bash
python3 -m tools.bench.run_upstream_route_matrix \
  --exe ./build/bench/ninfer_bench \
  --model /opt/ninfer-4090/Ternary-Bonsai-2-27B-ninfer-v3.ninfer \
  --out profiles/bench/sm89-resident-smoke \
  --device 0 --max-context 32768 --kv-dtype int8 \
  --spec dflash2 --draft-tokens 7 --prefill-chunk 1024 \
  --prompts 1024,4096,16384,32768 \
  --cases gdn_two_stage \
  --warmup 1 --arm-warmup 1 --pairs 3
```

Inspect `residency.json` (`model_load_count: 1`), every
`pair-*/bench.json` (`residency.model_load_count: 1`,
`residency.artifact_bytes_read_this_arm: 0`,
`residency.weight_bytes_uploaded_this_arm: 0` and
`generated_token_hashes`), `records.jsonl`, `summary.json` and
`session-stderr.log`. The owner remains alive across all six measured arms;
a fresh Program and graph is intentionally created for each arm.
All completed arms are journaled immediately. A per-candidate
`summary-CANDIDATE.json` is written as soon as its last paired arm finishes.
If a later CUDA candidate fails, the process terminates rather than silently
reusing a poisoned device. Successful earlier results survive in the output
directory; `plan.json`, `records.jsonl` and `ERROR.txt` distinguish
completed arms from the unfinished remainder.

### 3. Separate *experimental* GDN throughput from correctness

```bash
python3 -m tools.bench.run_upstream_route_matrix \
  --exe ./build/bench/ninfer_bench \
  --model /opt/ninfer-4090/Ternary-Bonsai-2-27B-ninfer-v3.ninfer \
  --out profiles/bench/sm89-gdn-fast-experiment \
  --device 0 --max-context 32768 --kv-dtype int8 \
  --spec dflash2 --draft-tokens 7 --prefill-chunk 1024 \
  --prompts 1024,4096,16384,32768 \
  --cases gdn_two_stage_approx \
  --warmup 1 --arm-warmup 1 --pairs 10
```

This explicitly opts into approximate state semantics. The report contains
a `blocked_approx_state_semantics` quality gate regardless of throughput.
Even identical output-token digests cannot override the prior failing
full-vs-split FP32 state oracle.

### 4. Expanded experiments (optional after correctness)

To compare the other candidates in the same **resident** process:

```bash
python3 -m tools.bench.run_upstream_route_matrix \
  --exe ./build/bench/ninfer_bench \
  --model /opt/ninfer-4090/Ternary-Bonsai-2-27B-ninfer-v3.ninfer \
  --out profiles/bench/sm89-full-route-matrix \
  --device 0 --max-context 32768 --kv-dtype int8 \
  --spec dflash2 --draft-tokens 7 --prefill-chunk 1024 \
  --prompts 1024,4096,16384,32768 \
  --cases prompt_fast,gdn_two_stage,t2_upstream,sm_wave,all_candidates \
  --warmup 1 --arm-warmup 1 --pairs 10
```

For longer **generated** outputs, include combined prompt+decode tests:
`--max-context 65536 --prompt-gen '4096,256;32768,256'`. Those tests
compare entire 257-token greedily generated continuations through a
64-bit hash in each arm. A hash mismatch is an automatic **failure** of
the output identity gate; a match is necessary but not sufficient for
FP32 state/checkpoint correctness.

Run a separate `--kv-dtype rk8v4` or `fp8` matrix if those storages
are part of production. Verify the route's actual execution, not only
that a key appears in the bundled device profile. All-candidate mode is
restricted to the intended five keys; it never loads all 65 upstream
configurations.

### 5. Required promotion gates

| Stage | Checks | Result required |
|---|---|---|
| Correctness | CUDA memcheck; full vs 59+5, 123+5 state/output; FP64 reference; graph replay | Pass |
| Generated tokens | Every paired AB/BA generated ID digest, including mixed pp+tg | Match |
| Quality | 4K/32K/131K perplexity and long-range needle/tool-call retrieval | No unacceptable degradation |
| Performance | 1K/4K/16K/32K, at least 10 paired orders with independent warmup | Stable >=2% end-to-end benefit |
| Memory | Real peak allocations, Program workspace capacity, GPU 24 GiB budget | No capacity regression |
| Agent serving | DFlash2 K7/K15, C1/C2/C4/C8, branch/prefix caching, TTFT and request p95 | No critical regression |

The runner is **serial prefill/optional decode**, not a concurrent Agent
serving test. It records the count of artifact materializations, not the
number of independently constructed Programs. P95 request latency,
long-session cache correctness, and 131K multi-request behavior need
separate real-server qualification before promoting any Kernel.

For diagnostic route inspection run a new output directory with
`--route-trace`, but **do not** collect performance evidence with tracing
enabled. The loaded route key is not proof that a guarded candidate kernel
ran.

A custom case matrix can be supplied via `--cases-json cases.json`,
where the `baseline` route must explicitly set
`NINFER_DEVICE_ROUTE_MODE=off`. For example:

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

The reported `performance_only_screen` is NOT a decision to
enable a candidate in production. No new fast GDN route is yet qualified.
