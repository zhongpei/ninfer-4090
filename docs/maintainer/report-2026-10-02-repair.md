# Repair and qualification of the 2026-10-02 report

## Status and scope

The source report used `4dc309ee5db03b5daa0127302967c2441eb916c3`,
`Ternary-Bonsai-2-27B-ninfer-v3.ninfer`, INT8 group64 KV, greedy and zero
presence/frequency penalties. It reports 504 CLI calls and 616 server requests:
CLI output qualification passed 25/63 scenarios; complete-response server
qualification passed 0/20 comparison groups. Successful execution and passing
local operator oracles did not establish full-model route equivalence.

This change addresses one confirmed numerical-policy dependency and the report's
measurement/evidence defects. **It does not declare the artifact qualified, explain
all ULP divergences, or publish a verified speedup.** The specific upstream op at
the captured chain/tree/spine divergence still requires target-hardware diagnosis.

## Runtime change

`causal_attention_split_capacity` previously clipped H24/KV4 split capacity to an
occupancy grid target computed from batch size. Device active-split selection is
bounded by this capacity, so changing the batch can change reduction partitioning
for the same query. The cap is removed; planning and launch use the existing B1
capacity calculation. The host policy regression tests B1–B8, both head geometries,
several token widths and context boundaries, and BF16/INT8/FP8/RK8V4.

This fixes that dependency **at a fixed token width and execution envelope**, not
all differences between tensor widths, prompt/small-T kernels, graph envelopes,
projection paths or recurrent-state evolution. More batched scratch and CTAs may
be needed. Measure startup reservation, peak workspace, VRAM and performance on the
24 GB device before qualification. No sampler tolerance, tie-breaking workaround,
second drafter, model conversion or enlarged tree family is introduced.

## CLI evidence version 2

The harness opts in to `NINFER_AB_METRICS=1`. The executable then emits one
`NINFER_METRICS_JSON` record on stderr after generation. It carries full precision
seconds/rates, integer counters, generated token IDs, terminal reason, sampler
policy and memory usage. Ordinary CLI execution is unchanged without this variable.
Rebuild the CLI: an older executable lacking the record is diagnostic-only.

Generated token sequences, stdout bytes and terminal reasons must agree. The
first different generated-token index and common generated prefix are retained;
this is **not** a captured full rendered prompt or layer-state snapshot. All pairs,
including the first pair discarded from timing, participate in correctness and
self-repeat checks. The harness records command, prompt hash, model/executable
SHA256 and individual file hashes. Each comparison/workload/pair/order/arm has an
exclusive directory. Rerunning into an existing evidence set fails instead of
overwriting another comparison's control logs. Timeouts retain partial output.

Legacy `1.xxk tok/s` and other SI/exponent strings can be read but are labeled
`pretty-rounded`. Missing or malformed machine evidence never becomes a qualified
result. Missing performance scenarios block the pooled score; they are not silently
removed from the denominator. Reports separate diagnostic and qualified ratios.

`--aa` is a target-only repeatability control. `--compare-baseline` compares every
selected strategy directly with target-only, in addition to the default incremental
comparison graph. Lookup-versus-chain agreement alone is not target-only agreement.

## Server evidence version 2

For each comparison, pair, client concurrency and arm, a fresh owned process is
started, checked for readiness and stopped. Pairs alternate AB/BA. Engine concurrency
stays fixed at eight; client levels do not change its startup capacity. Prefix reuse
stays enabled by default; `--cache-mode disabled` is a separate diagnostic configuration.
No model-default/no-thinking results are pooled together.

Full raw HTTP bodies and headers are archived after the timed request batch. The
generated equivalence fields are `content`, `reasoning_content`, `refusal` and
`finish_reason`; whitespace is not normalized. Transport IDs and timestamps are not
generated output. Text-only requests reject unexpected tool calls. No-thinking
requires nonempty content and no reasoning; model-default may return reasoning but
cannot pass with all generated channels empty. This is response equivalence, not
token/logit equality. HTTP or schema errors remain failures with archived bodies.

Every expected request must occur exactly once. Both arms must be internally stable
across repeats and fresh-process pairs. Baseline instability is reported separately;
it cannot be attributed entirely to a candidate. Runtime request_done records must
match the owned instance and expected count, with the requested sampler/thinking
policy and complete counters. Actual tree/fallback/head-skip coverage is shown;
missing telemetry is unknown, never silently zero. Failed gates yield no qualified
speedup. Median/p95 use observed request timing; p95 uses nearest rank. Client queue
latency is separately recorded. Aggregate throughput includes queueing, HTTP and
parsing, but excludes startup and archive disk writes.

The protocol intentionally does not force output identity or disable production
caches to obtain PASS. Nor does a passed response gate score answer quality.

## Linux reproduction

Run these commands in a checkout of this PR. The shell wrapper does not build,
modify the deployment checkout, stop an existing service or query CI. It refuses
an occupied benchmark port. Each fresh server uses the next port starting at
`NINFER_AB_PORT` (default 18080); the full range must fit below 65536. Use
different available ranges for simultaneous or immediately consecutive experiments
to avoid listeners and TCP TIME_WAIT from another experiment. Choose one idle GPU and avoid concurrent host-heavy
benchmarks; the wrapper records the environment but does not lock GPU clocks.

```bash
export CUDA_VISIBLE_DEVICES=0
export NINFER_BUILD_DIR=/opt/ninfer-4090/build
export NINFER_DFLASH2_MODEL=/opt/ninfer-4090/Ternary-Bonsai-2-27B-ninfer-v3.ninfer

# Rebuild the selected checkout's binaries before using these entry points.
# One baseline-only server A/A experiment, then chain-focused direct CLI evidence.
NINFER_AB_PORT=18080 bash scripts/sweeps/speculative-report-recheck.sh aa
bash scripts/sweeps/speculative-report-recheck.sh cli

# Incremental server comparisons, and complete combinations against target-only.
NINFER_AB_PORT=19080 bash scripts/sweeps/speculative-report-recheck.sh ab
NINFER_AB_PORT=20080 bash scripts/sweeps/speculative-report-recheck.sh direct
```

The build directory must contain binaries rebuilt from the PR, not the frozen
old deployment binary. Native policy check after a CUDA-enabled build:

```bash
./build/tests/ninfer_softmax_attention_test --split-capacity-only
```

Adjust the executable path to the actual build output. No GPU inference is claimed
by that host policy test. The focused CPU protocol/serializer suite is:

```bash
python3 -m unittest discover -s tests/report_regression -v
```

The standalone serializer fixture uses an available `g++`; other tests need only
the standard Python library. HTTP/CLI subprocess fixtures are explicitly not model
inference. CUDA build, native attention execution, the original full regression
suite, real-model A/A and A/B and 24 GB peak memory remain separate validations.

## Post-merge RTX 4090 qualification (2026-10-02)

PR #14 merged as `784d625e`. Post-merge validation repaired legacy test fixtures
in `ab05bb0` and isolated each fresh server instance's port in `e377d58`.
The production C++ implementation stayed at the PR merge version. The qualification
below used the frozen contents of `e377d58`; no sampling or correctness gate was
relaxed.

Environment: two RTX 4090 24 GB devices, driver 610.57.04, CUDA compiler/runtime
12.8, Release `sm_89`, 96 logical CPUs on two sockets, and Python 3.11.15.
The explicit model was `/opt/ninfer-4090/Ternary-Bonsai-2-27B-ninfer-v3.ninfer`
(9,520,051,456 bytes). The existing build was rebuilt with
`cmake --build build -j`, including `--split-compile=8` for the heavy CUDA unit;
it was not reconfigured.

All four reproduction modes used seven workloads, four AB/BA pairs, and discarded
only the first pair from performance. Servers used two repeats per workload,
512 maximum generated tokens, client concurrency 1/2/4/8, engine concurrency 8,
context/KV capacity 32768, INT8 KV, greedy with zero presence/frequency penalties,
thinking off, prefix reuse enabled, CUDA Graphs enabled, and two-second cooldowns.
GPU0 ran CLI/direct and GPU1 ran native/AA/incremental AB; host resources were
shared and GPU clocks were not locked.

| Check | Execution | Qualification |
|---|---|---|
| Parallel incremental build | 184/184 default targets, exit 0 | PASS |
| Full Python regression | 143/143, 35.13 seconds | PASS |
| Host split-capacity policy | B1–B8 | PASS |
| Native attention | default, DFlash2, NVFP4, K8V4 | all PASS |
| CLI direct-to-baseline | 224/224 calls successful | 11/28 workload comparisons; 0/4 overall candidates |
| Baseline A/A | 32 processes, 448/448 requests successful | all four concurrency levels PASS |
| Incremental server A/B | 128 processes, 1792/1792 requests successful | 0/16 comparisons |
| Server direct-to-baseline | 128 processes, 1792/1792 requests successful | 0/16 comparisons |

The formal matrices completed 4256 successful generation calls, excluding health
probes and rejected diagnostic attempts. Server run identities and ports were
unique; telemetry was complete, with no interrupted matrix, launch, port, HTTP or
OOM failure. A/B wrapper exit 2 denotes failed qualification, not failed execution.
Baseline A/A median ratios were 0.999/1.002/0.999/1.003 at C1/C2/C4/C8.

CLI qualified workloads were K7: reasoning/structured/lookup-repeat; K11:
prose/lookup-repeat; K15 and lookup-skip: prose/reasoning/lookup-repeat. All other
17 workload comparisons had generated-token divergence, first appearing at
zero-based index 16–195. Every candidate diverged on chat at index 16. Overall
CLI diagnostic geometric-mean ratios were 3.348/3.412/3.256/3.304 respectively;
none is an overall qualified speedup.

The following server ratios are **diagnostic only**; every row failed qualification.

| Candidate | Control | C1 | C2 | C4 | C8 |
|---|---|---:|---:|---:|---:|
| chain K15 | target-only | 2.511 | 1.751 | 1.170 | 0.903 |
| tree15 | target-only | 2.246 | 1.744 | 1.072 | 0.885 |
| tree15 + Stair | target-only | 2.272 | 1.725 | 1.168 | 0.857 |
| lookup-skip | target-only | 2.478 | 1.779 | 1.181 | 0.894 |
| chain K15 | target-only (incremental matrix) | 2.539 | 1.771 | 1.226 | 0.927 |
| tree15 | chain K15 | 0.808 | 0.955 | 0.935 | 0.997 |
| tree15 + Stair | tree15 | 1.036 | 0.995 | 0.986 | 1.010 |
| lookup-skip | chain K15 | 0.995 | 1.006 | 0.935 | 0.985 |

Both server matrices had full-response differences in all 64 pairs and candidate
self-repeat instability in 60 pairs. Direct target-only controls had no self-repeat
failure. Incremental controls had 44 self-repeat failures; these controls include
chain/tree and must not be confused with the passing target-only A/A experiment.
Actual counters confirmed tree/Stair/lookup execution. Direct tree15 tree/fallback
rounds were 4032/12, 517/4061, 335/4481 and 328/4482 by concurrency; direct
lookup/head-skip rounds were 261/36/3/6. The counts do not attribute kernel costs.

GPU0 direct and GPU1 AB each reached an observed 14243 MiB (about 13.91 GiB) in
one-second NVML sampling, without OOM. This is a sampled lower bound, not a strict
instantaneous peak or a filled-32768-context capacity qualification. All formal
server startup workspace capacities/peaks were 163127296 bytes; baseline runtime
reservation was 3857350912 bytes and candidate reservations about 7.45 GiB.
Against the old `4dc309e` startup anchors, the five matching arms had identical
memory fields; engine configuration differed only in the logging interval.
Unchanged planned capacity does not establish zero extra CTAs or execution cost.
There is no comparable old whole-device peak measurement.

The initial fixed-port direct attempt completed only five processes/70 requests;
123 launches were rejected during port reuse. It is superseded by the complete
formal retry. A temporary unique-port driver omitted the readiness-port change;
it was stopped and its results were not used. The formal fix was protected by a
CPU fresh-process test with address reuse disabled, RED to GREEN, and the full
143-test independent regression.

The remaining blockers are speculative output divergence and candidate server
self-repeat instability. The specific upstream operator has not been attributed;
native oracle and baseline A/A passes do not establish full-model route equivalence.
C8 direct diagnostic throughput was about 10–14% below target-only, while C1 was
faster. These strategy comparisons do not establish PR14's net effect versus old
code: no old/new binaries were compared under an identical repaired harness.

Unaffected checks reuse the previous 146 applicable CTest passes and seven skips;
they were not rerun as a full CTest suite in this round. MoE and DFlash1 real-model
qualification still lacks corresponding artifacts. CI was not queried.

Local evidence is retained under `profiles/bench/pr14-2026-10-02/`: `report.md`,
`summary.json`, `final-audit.json`, `regression/`, and the CLI/server result directories.
The complete local archive is `profiles/bench/pr14-2026-10-02.zip`. These generated
logs and archives are excluded from Git by the repository's existing ignore policy.

## Development priority

Keep fixed DFlash2 chain as the correctness investigation baseline. Retain lookup
skip as an explicit repeated-content experiment and preserve tree/Stair controls
without enabling them by default or increasing tree size. Use the generated-token
mismatch evidence to fix the same-prefix first operator/state divergence next.
The initial report remains unqualified until that work is validated on the artifact.
