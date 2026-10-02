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
an occupied benchmark port. Choose one idle GPU and avoid concurrent host-heavy
benchmarks; the wrapper records the environment but does not lock GPU clocks.

```bash
export CUDA_VISIBLE_DEVICES=0
export NINFER_BUILD_DIR=/opt/ninfer-4090/build
export NINFER_DFLASH2_MODEL=/opt/ninfer-4090/Ternary-Bonsai-2-27B-ninfer-v3.ninfer

# Rebuild the selected checkout's binaries before using these entry points.
# One baseline-only server A/A experiment, then chain-focused direct CLI evidence.
bash scripts/sweeps/speculative-report-recheck.sh aa
bash scripts/sweeps/speculative-report-recheck.sh cli

# Incremental server comparisons, and complete combinations against target-only.
bash scripts/sweeps/speculative-report-recheck.sh ab
bash scripts/sweeps/speculative-report-recheck.sh direct
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

## Development priority

Keep fixed DFlash2 chain as the correctness investigation baseline. Retain lookup
skip as an explicit repeated-content experiment and preserve tree/Stair controls
without enabling them by default or increasing tree size. Use the generated-token
mismatch evidence to fix the same-prefix first operator/state divergence next.
The initial report remains unqualified until that work is validated on the artifact.
