# Speculative decoding A/B test matrix

This is a release-qualification and attribution harness, not an answer-quality benchmark or an
independent numerical oracle. Linux is the deployment path. Correctness failures remain failures;
none of the checks below changes target logits, sampling, token tie-breaking or inference kernels.

## Current qualification and scope

The owner-supplied **2026-10-02 interim report** tested `4dc309ee5db03b5daa0127302967c2441eb916c3`
on dense `Ternary-Bonsai-2-27B-ninfer-v3.ninfer`, RTX 4090, CUDA 12.8 and INT8 group64 KV.
It reports 336 successful no-thinking server requests but **all 20 full-response comparison groups
failed**. Baseline itself differed in three long-context repeat groups at C2/C4/C8. Consequently,
neither candidate throughput ratios nor apparent slowdowns qualify as equivalent-output
performance results. The report's 504-run CLI matrix was still running; no final CLI result is
inferred here. MoE, DFlash1, other checkpoints and other KV formats were not established by that run.

The earlier small CLI check completed 16 executions on prose/lookup-repeat: chain/lookup matched
on those two workloads, while fixed tree and Tree-Stair failed on prose. At a captured first
fixed-tree divergence, the represented BF16 logits for tokens 25/318 were 18.0/18.0 in ordinary
execution and 17.875/18.0 in tree execution. A branch-free spine reproduced the difference;
argmax was correct for both represented inputs. A chain check with presence penalty 1.5 showed a
similar boundary. **These observations do not attribute every failure to rounding or identify a
specific offending Op.** Independent Op/state/KV oracle passes do not prove whole-model equality.

The fixes to this harness address confirmed evidence loss and experimental contamination in the
old `server_ab.py`. They do **not** repair or qualify the unresolved model-level numerical
route/repeat instability. Those remain release blockers requiring actual model/GPU reproduction.

## Comparison contract

Both harnesses use greedy generation with explicit zero presence/frequency penalties. Server
requests also specify a fixed seed and default to `--thinking off`. `--thinking model-default`
is a separate, explicitly recorded experiment; it must not be pooled with no-thinking runs.
This suite makes no stochastic-sampling equivalence claim.

CLI comparisons use the existing target-only, DFlash2 K7/K11/K15, same-K tree, fixed-tree versus
Tree-Stair, and chain-versus-lookup controls. Selecting a candidate includes its comparator
ancestors; one global baseline is not reused across unrelated pairs. CLI pair discard affects
performance only. Existing CLI qualification runs need not be interrupted to adopt server fixes.

The built-in prose, chat, reasoning, code, JSON, lookup-repeat and long-context prompts are
**synthetic regression workloads**, not a public task score. The long-context fixture includes
repetition; a large max-context setting is a capacity, not the actual tokenized prompt length.
Record actual prompt usage. Passing a JSON prompt's equality check does not establish JSON validity,
code correctness or answer accuracy. External-dataset coverage is not claimed.

## Linux CLI entry

```bash
python3.11 -m tools.dflash2_training.ab_suite \
  --exe ./build/apps/ninfer \
  --model /opt/ninfer-4090/Ternary-Bonsai-2-27B-ninfer-v3.ninfer \
  --out /tmp/ninfer-cli-check --kv-dtype int8 \
  --arms baseline,dflash2-k15,tree15,tree15-stair,lookup-skip \
  --workloads prose,lookup-repeat --pairs 1 --discard 0 --cooldown 0
```

The existing `scripts/sweeps/dflash2-ab-matrix.sh smoke|core|full|server` matrix remains available.
Use a fresh `NINFER_AB_OUT` for each campaign. The PowerShell wrapper remains a development entry,
not the authoritative Linux deployment workflow. Never use output lengths or speeds from failed
comparisons as validated algorithm improvements.

## Isolated server A/B

```bash
export CUDA_VISIBLE_DEVICES=0
python3.11 -m tools.dflash2_training.server_ab \
  --serve ./build/apps/ninfer-serve \
  --model /opt/ninfer-4090/Ternary-Bonsai-2-27B-ninfer-v3.ninfer \
  --out /tmp/ninfer-server-qualified-run \
  --arms baseline,dflash2-k15,tree15,tree15-stair,lookup-skip \
  --workloads all --concurrency 1,2,4,8 --server-capacity 8 \
  --thinking off --prefix-reuse on --cuda-graph on \
  --kv-dtype int8 --max-context 32768 --kv-capacity 32768 \
  --pairs 4 --discard 1 --repeats 2
```

Each **comparison / client concurrency / pair / side** owns a fresh server, unique public model ID,
request log and raw-response archive. AB/BA order alternates between pairs. No C1-to-C8 cache or
engine-scope Stair learning crosses trials. Within a trial, production prefix reuse and adaptive
learning remain enabled unless explicitly overridden. Startup warmup remains the runtime's own.
Only the spawned server process group is stopped; an existing listener is rejected, not reused or
killed. Output directories must be new or empty to prevent overwriting evidence.

`--server-capacity 8` is the default and is **not** replaced with C1 when one client runs.
`--server-capacity level` is an explicit different experiment. The client job count must be at
least the largest offered concurrency; use more repeats for a narrow workload set. That check
ensures enough jobs exist, not that all runtime decode rounds are full batches.

### Complete-response and self-stability gate

The response signature includes content, reasoning_content, finish_reason, refusal, role and tool
function calls. Tool argument strings and whitespace are preserved exactly. Only generated
request/timestamp metadata and provider-generated tool-call IDs are outside output semantics;
missing/null text channels normalize to the empty channel. The complete raw response is retained
before validation. All-empty output, malformed choices, missing terminal reasons and missing or
invalid completion usage fail. Reasoning-only answers can be valid in model-default mode, but are
never validated by an empty content hash; nonempty reasoning in off mode fails.

Every expected request on **both** sides must succeed exactly once. Missing or duplicate rows,
missing full-response evidence and hash/signature disagreement fail closed. Every pair, including
the performance-discarded first pair, participates in correctness. Independently, each side must
repeat the same full response for each workload within and across fresh trial processes.
Even when A and B agree pairwise, a jointly unstable baseline/candidate cannot pass self-stability.
At least two repetitions per workload are required.

Mismatch reports identify the field and first **character** difference (with excerpts); this is
not a token index, a logit dump or an Op attribution. No numerical epsilon or text stripping is used.

### Runtime evidence and honest speedups

The runner reads the runtime's `request_done` JSONL records after process shutdown. It records real
tree rounds, fallback rounds, nodes, accepted drafts, lookup rounds and head-skip rounds. Missing
logs, malformed counters, duplicated internal IDs and incomplete completion records fail evidence
qualification. Startup and aggregate throughput events do not count as measured requests.
These are **trial totals**: the internal numeric request ID is not guessed to equal the protocol
UUID, and completion order is not used to fabricate workload attribution.

`tree_execution_fraction = tree_rounds / (tree_rounds + tree_fallback_rounds)` describes mixed
execution. An arm labelled tree may mostly run chain fallback. No measured tree/head-skip work
means that optimization was not exercised; its flag alone cannot establish speedup.

The result retains `diagnostic_pair_speedups` and `diagnostic_median_speedup`, including failed
runs for diagnosis. **`qualified_speedup` is null** unless the complete-response, self-stability,
completeness and runtime-evidence gates pass, all retained paired measurements exist, the selected
optimization executed in measured pairs, and at least three performance pairs remain.
`consistent_gain/regression` additionally requires every retained paired ratio to have the same
sign and median gain/loss to reach 2%; this is a practical heuristic, **not a confidence interval**.
No aggregate performance claim overrides a failing workload or unstable baseline. A passing
performance gate alone is still not an answer-quality score.

### Timing and artifacts

Throughput is the sum of usage.completion_tokens divided by the whole request-level wall interval,
including dispatch, queueing, prefill, generation, response parsing and client validation. Model
load/readiness and **archive disk writes** are outside that interval. Per-request latency starts
when a worker begins HTTP work; client thread-pool waiting and total client time are reported
separately. p95 is nearest-rank `ceil(0.95*n)`: with 14 samples it is the 14th sample, not the 13th.
There is no client streaming TTFT measurement in this non-streaming harness.

`server-results.json` is checkpointed after each completed trial. Each trial directory contains
`server.log`, `requests.jsonl`, `raw-responses.jsonl` and `trial.json`. Config, Python/platform,
CUDA visibility, actual commands and full response usage remain available. Runtime startup logs
also contain GPU and memory metadata. Interrupted/unstarted trials are not successes. A nonzero
exit denotes gate/execution failure; exit 130 denotes interruption.

## Baseline-first Linux reproduction

Use the small diagnostic driver before another 504-run campaign:

```bash
export NINFER_PYTHON=python3.11
export NINFER_SERVE_EXE=/opt/ninfer-4090/build/apps/ninfer-serve
export CUDA_VISIBLE_DEVICES=0
bash scripts/sweeps/dflash2-server-repro.sh \
  /opt/ninfer-4090/Ternary-Bonsai-2-27B-ninfer-v3.ninfer \
  /tmp/ninfer-repro-20261002
```

It runs mixed prose/long-context jobs sequentially on the selected GPU, covering baseline-self
with production settings, no prefix reuse, no graphs, neither, and startup capacity 1 separately.
It then restores production settings for the candidate comparisons. Each condition is separately
labelled. A correctness FAIL does not hide the other diagnostic conditions; their exit codes are
saved in `exit-status.tsv`. An interrupt stops the campaign. One pair is intentionally insufficient
for a performance claim. No production server settings, GPU clocks or production process are modified.

Interpretation: a pass only with graphs off narrows a graph/envelope question; a pass only without
prefix reuse narrows a reuse/materialization question. Neither establishes an Op bug by itself,
and neither qualifies production settings. A failure even in baseline-only fresh C1 trials must
be resolved before attributing all differences to speculative acceptance.

## Qualification follow-through

Keep model artifact, GPU, KV format, thinking, penalties, build and execution flags explicit.
Do not tune Stair costs or promote tree/lookup defaults based on failed output comparisons.
The actual attention code selects split/grid geometry using batch, token width and execution
window, while tree full-attention visits nodes with width one. That is a concrete route difference
to investigate, not proof of the reported logit divergence. Next numerical evidence must compare
the same represented inputs, prefix state and position, then localize the first differing Op and
check its contract/oracle. Do not replace that investigation with changed tie-breaking or tolerance.

The default 24 GB matrix is not a guarantee that every C8/context/artifact fits. An OOM is retained
as a failed configuration, not silently skipped or rewritten to a shorter context. BF16 KV can be
requested explicitly when memory permits. MoE/DFlash1 require their own matching artifacts.

The existing Linux Stair cost tool remains available **after** correctness qualification:

```bash
mkdir -p profiles/sweeps
bash scripts/sweeps/dflash2-stair-cost-calibration.sh > profiles/sweeps/staircost.csv
python3.11 -m tools.dflash2_training.calibrate_4090 profiles/sweeps/staircost.csv \
  --out profiles/4090-stair.json
```

## CPU regression tests

```bash
python3.11 -m unittest discover -s tests/tools -p test_server_ab_response.py -v
```

The test includes a local HTTP fixture exercising readiness, fresh process ownership, complete
response archiving and rapid Linux restarts. It loads no model and establishes no GPU inference
correctness. The same test is registered with CTest via the configured Python interpreter.
