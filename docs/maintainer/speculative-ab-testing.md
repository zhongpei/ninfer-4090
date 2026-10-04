# Speculative decoding A/B test matrix

This suite is the release/performance validation layer for the DFlash2, tree, Stair and lookup changes. It is deliberately separate from unit tests: it measures the real NInfer executable and server on the target GPU.

## Method

The protocol combines four useful practices from existing speculative-decoding projects:

1. **TandemLLM-style alternating A/B pairs.** Baseline and candidate are interleaved AB/BA rather than all baseline runs followed by all candidate runs. The first pair is normally discarded from performance statistics; every pair remains part of the correctness gate. Results are reported per workload and as a weighted pool. A greedy output mismatch is a hard correctness failure, not a performance data point.
2. **Spec-Bench-style workload diversity.** Do not infer speculative quality from one prompt. Measure natural-language, chat, reasoning/math, code, structured output, repetition/copy and long-context workloads on the same hardware/runtime build.
3. **DFlash-style acceptance telemetry.** Record target throughput together with acceptance rate, accepted tokens/round and the tree/lookup counters. A faster result without acceptance data cannot explain why it improved.
4. **Serving throughput separately from serial latency.** A method can improve C1 latency and still lose at larger batches. The server suite therefore measures C1/C2/C4/C8 independently.

The default benchmark is greedy because output SHA equality is then a strong exactness gate. Both CLI and server comparisons explicitly set presence/frequency penalties to zero; model generation presets must not silently disable tree execution. Positive-temperature tree execution is supported when presence/frequency penalties are zero and requires a separate seeded sampling check.

## Current RTX 4090 qualification

The historical 2026-10-02 qualification used CUDA 12.8 on RTX 4090 with
`Ternary-Bonsai-2-27B-ninfer-v3.ninfer`, INT8 KV, and the `prose,lookup-repeat`
workloads. Run the same small correctness matrix with an explicit artifact path:

```bash
python3.11 -m tools.dflash2_training.ab_suite \
  --exe build/apps/ninfer --model /path/to/Ternary-Bonsai-2-27B-ninfer-v3.ninfer \
  --out /tmp/ninfer-ab --kv-dtype int8 \
  --arms baseline,dflash2-k15,tree15,tree15-stair,lookup-skip \
  --workloads prose,lookup-repeat --pairs 1 --discard 0 --cooldown 0
```

All 16 CLI executions completed successfully and both tree arms executed real
tree rounds. Chain and lookup comparisons passed exact output equality on both
workloads. Fixed-tree and Tree-Stair comparisons passed on `lookup-repeat` but
failed on `prose`; the harness correctly exited with status 2. That run did not qualify the exact-output release gate for this artifact and
establishes no performance improvement.

At the first fixed-tree divergence, ordinary execution produced BF16 logits
`18.0/18.0` for token IDs `25/318`, while tree execution produced
`17.875/18.0`. Both argmax decisions were correct for their represented logits.
A linear spine reproduced the same divergence, so branching was not required.
An additional presence-penalty 1.5 chain comparison also diverged at a one-ULP
BF16 boundary. Independent sampling, branch-state and KV oracles passed, but
they do not establish bitwise stability of the complete target forward across
execution routes. The specific source of the target-logit differences remains
unresolved; neither these observations nor successful generation waive the
exact-output gate.

The 2026-10-03 chain repair qualification uses the explicit artifact
`/opt/ninfer-4090/Ternary-Bonsai-2-27B-ninfer-v3.ninfer`, RTX 4090, CUDA 12.8,
INT8 KV, greedy sampling, zero penalties, cache on and CUDA Graphs on. The
original chat CLI K15 comparison now matches all 512 generated token IDs at
context 8192. The original chat public Engine fixture also matches target-only
across fresh, retained and restored prefixes, including
`Root → PrivateTurnClosure`, with explicitly resolved zero penalties. Final 2026-10-04 switching acceptance
also passes terminal/cancellation, page/ring continuation, context boundaries,
actual-batch membership changes, cache restoration and zero-commit release.

Two fresh-process AB/BA server pairs pass complete response equality on all
seven workloads and both repeats for K7/K11/K15 at C1/C2/C4/C8. HTTP
responses do not expose token IDs; exact token comparison is supplied by the
CLI/Engine fixtures and remains an explicit check in the native actual-batch
measurement matrix. The first
query-width-dependent differences were in attention partition/reduction and
GDN normalization/control projection; prefix segmentation also exposed a T2A8
FFN accumulation-order difference. Independent represented-input mathematical
oracles and targeted width regressions now pass without relaxing their criteria.

These results close the original chain and server reuse failures within this
scope. Tree, Stair, lookup, nonzero penalties and stochastic cross-route output
are not qualified by this campaign. Existing HTTP throughput results lack
per-round actual batch/frontier observation and do not qualify a calibrated
policy. Fixed-route measurement and automatic switching remain separate gates.
Local evidence is in `profiles/bench/consistency-2026-10-03/`, with final server
responses under `server-final/` and numerical/CLI fixtures under `chain/`.

The final repaired binary passes all 84 native fixed-route comparisons
(336 measurements), with exact token IDs and complete repeated responses.
Seventy-one comparisons qualify paired performance; see
[measured throughput and limitations](../performance.md#rtx-4090-sm_89-chain-qualification).
Resident-K15 calibration covers actual batch/frontier cells separately. Final
automatic comparisons use true target-only, standalone fixed K7 and calibrated
Auto in fresh unprimed processes, with identical context/KV budgets and natural
repeat reuse. K7 is selected as the uniform fixed reference because it qualifies
27 of the 28 workload/concurrency combinations, rather than selecting a
different comparator after seeing each automatic result.

## Workloads

| workload | purpose |
|---|---|
| prose | ordinary explanatory generation |
| chat | conversational systems answer |
| reasoning | multi-step arithmetic/reasoning |
| code | structured code generation |
| structured | JSON-constrained output |
| lookup-repeat | highly repetitive long prompt where suffix lookup should help |
| long-context | low-repetition long prompt to expose context-dependent verification cost |

`lookup-repeat` and `long-context` carry higher pooled weights because they exercise the two behaviors most likely to differ from short-prompt microbenchmarks.

For publication-quality results, the same harness can later be fed external Spec-Bench/DFlash datasets; the built-ins are deterministic smoke/regression workloads and require no network or dataset download.

## Arms and comparators

| candidate | comparator |
|---|---|
| dflash2-k7/k11/k15 | target-only baseline |
| tree7/tree11/tree15 | chain DFlash2 at the same K |
| tree15-stair | fixed tree15 |
| lookup-replace/lookup-skip | chain DFlash2 K15 |

This matters: comparing tree15 only to target-only would combine the DFlash gain and the tree gain, so it would not tell us whether the tree itself paid.

## CLI test

```powershell
python -m tools.dflash2_training.ab_suite --model X:\models\qwen3_8_27b_dflash2.ninfer --pairs 4 --discard 1 --cooldown 5 --kv-dtype int8
```

Outputs include `ab-results.json`, `ab-summary.md`, and per-run stdout/stderr files under the selected output directory.

The runner records external wall time plus NInfer's own decode metrics. It exits nonzero on any greedy output mismatch, execution failure, missing pair, or empty comparison. Server comparisons require every expected request to succeed on both baseline and candidate. A throughput regression is recorded as a finding, not converted into a correctness failure.

The conservative performance label is:

- **resolved better**: every retained paired run is faster and median speedup is at least 1.02x;
- **resolved worse**: every retained paired run is slower and median speedup is at most 0.98x;
- otherwise **unresolved**.

This prevents a single noisy median from being reported as a win.

## Server test

```powershell
python -m tools.dflash2_training.server_ab --model X:\models\qwen3_8_27b_dflash2.ninfer --concurrency 1,2,4,8
```

For every arm the harness launches a fresh server, waits on `/health`, discovers the public model id through `/v1/models`, then sends the same greedy Chat Completions workload set.

It reports aggregate completion-token throughput, median/p95 request latency, request errors, per-workload/per-repetition SHA equality, and speedup at each concurrency.

A new server per arm prevents prefix cache, adaptive routing history and allocator history from silently leaking from A into B.

## 4090 profiles

```powershell
.\scripts\sweeps\dflash2-ab-matrix.ps1 -Profile smoke
.\scripts\sweeps\dflash2-ab-matrix.ps1 -Profile core
.\scripts\sweeps\dflash2-ab-matrix.ps1 -Profile full
```

`core` executes a full INT8 CLI matrix, focused FP8/RK8V4 comparisons on prose/code/long-context, and an INT8 server C1/C2/C4/C8 matrix.

`full` increases pair count/cooldown, runs the broad workload set for INT8/FP8/RK8V4 and runs server concurrency on all three practical 24 GB KV formats.

The suite intentionally does not include BF16 KV in the default 24 GB matrix. It may be requested manually when the selected artifact/context capacity fits, but it should not cause the standard 4090 validation to fail from a configuration outside the intended memory envelope.

## Linux Stair calibration

The 4090 cost calibration also has a native Bash entry point:

```bash
export CUDA_VISIBLE_DEVICES=0
export NINFER_DFLASH2_MODEL=/models/qwen3_8_27b_dflash2.ninfer

bash scripts/sweeps/dflash2-stair-cost-calibration.sh \
  > profiles/sweeps/staircost.csv

python3 -m tools.dflash2_training.calibrate_4090 \
  profiles/sweeps/staircost.csv \
  --out profiles/4090-stair.json
```

This keeps calibration and the subsequent A/B matrix on the same Linux build, CUDA driver and GPU.

## Reading the result

Do not optimize on one number. For each arm check, in order:

1. exact output gate;
2. errors/fallbacks;
3. speedup per workload;
4. accepted tokens/round and acceptance rate;
5. long-context behavior;
6. C1 vs C8 server behavior;
7. KV-format sensitivity;
8. weighted pooled speedup.

A candidate that wins prose but resolves worse on code or C8 should remain an A/B option rather than becoming the default.


## Windows compatibility

The PowerShell wrapper remains available for development machines, but Linux/Bash is the canonical deployment benchmark path.
