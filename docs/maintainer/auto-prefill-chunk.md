# VRAM-aware automatic prefill chunk

Status: **experimental, opt-in through `--prefill-chunk auto`**. Fixed numeric chunks keep their
existing behavior.

## Contract

Auto prefill separates two quantities that were previously one constant:

- **physical chunk capacity**: the largest startup-selected prefill shape whose persistent/workspace
  reservation fits the device;
- **service chunk**: the amount of prompt work charged as one scheduler quantum. When decode work is
  runnable, Auto caps one prefill execution unit at 2,048 tokens so the existing decode/prefill
  alternation remains responsive.

Startup tries these physical rungs from largest to smallest:

```text
8192, 6144, 4096, 3072, 2048, 1536, 1024
```

The search runs after model weights are resident and uses the same SequencePlanner /
SequenceCapacityCurve that final Program allocation uses. It does not allocate trial Programs.

### KV capacity is protected

A larger prefill chunk is allowed to consume only memory that the historical 1,024-token startup
would have left unused.

The selector first resolves the 1,024-token baseline and records its Main-KV page count. Every
larger rung must still fit **that exact page count**. With `--kv-capacity auto`, the configured
automatic KV headroom is also preserved. With explicit KV capacity, Auto additionally keeps a
256 MiB transient/driver margin while trying enlarged rungs; if that margin rejects every enlarged
rung, it falls back to the already-valid 1,024-token baseline rather than breaking startup.

Therefore `--prefill-chunk auto` must not exchange context capacity for prompt throughput.

## Runtime scheduling

When no decode batch is runnable, the Program may use the complete startup-selected physical chunk.
For example, a 16K fresh prompt may execute as two 8K units if the machine qualified 8K.

When decode is runnable beside a long prefill, Engine asks Program for at most:

```text
min(resolved physical chunk, 2048)
```

prompt tokens in that service unit. The existing scheduler then returns to decode before another
prefill unit. The Program still owns workspace sized for the maximum physical chunk; runtime
adaptation only narrows live extents and never reallocates CUDA storage.

Admission **service** accounting uses the 2,048-token service quantum in Auto mode. If an isolated
prefill executes a larger physical unit, its returned `service_work_quanta` consumes the matching
number of precomputed scheduler quanta, so fairness does not change just because several quanta
were fused into one GPU call.

Context-cache pricing remains separate: `PrefillWork.chunks` describes the actual physical
prefill machine work and therefore uses the startup-selected physical chunk, not the scheduler
service quantum. This separation prevents Auto from making retained-prefix/rebuild economics look
more expensive merely because the scheduler can preempt long prompt work more often.

Capture/rewrite boundaries remain hard segmentation points.

## Calibrated DFlash2

Auto prefill may be used with calibrated DFlash2 routing. The routing identity is built **after**
the physical prefill chunk resolves and records that resolved numeric chunk. A profile therefore
still cannot be reused across a startup that resolves to a different physical chunk; normal
identity validation rejects it.

## Qualification

First establish isolated prefill throughput and resolved memory size:

```bash
python3 -m tools.bench.run_prefill_chunk_matrix \
  --exe ./build/bench/ninfer_bench \
  --model /absolute/path/to/model.ninfer \
  --out profiles/bench/prefill-chunk-matrix \
  --device 0 --max-context 32768 --kv-dtype int8 \
  --spec dflash2 --draft-tokens 7
```

The matrix compares fixed 1024/1536/2048/3072/4096/6144/8192 against Auto and records the chunk
Engine actually selected, runtime reservation, workspace capacity and prompt throughput.

For serving qualification, run the existing long-prompt cohort harness twice with the same
artifact/configuration, once with fixed 1024 and once with Auto (or reproduce the same matrix with
your normal server harness). Use C2/C4/C8 and a nontrivial output budget so an earlier admitted
request is decoding while a later request is still prefilling; that is the condition that exercises
the 2K runtime interleave cap rather than only the isolated maximum chunk. Retain request logs and
compare complete greedy responses between the two arms.

Production promotion requires:

1. exact greedy responses against fixed-1024 for the same requests;
2. unchanged resolved KV capacity for Auto versus fixed-1024;
3. improved prompt throughput/TTFT for the target workload;
4. no unacceptable decode latency or throughput regression while prefill and decode coexist;
5. device-memory peak inside the deployment margin.

The feature is deliberately not the default until those RTX 4090 measurements are complete.

## RTX 4090 measurement, 2026-10-05 (PR #18)

Decision: keep Auto opt-in and retain the fixed-1024 default. Auto successfully selected 8192
without reducing explicit KV capacity. Its long-prompt gains were modest and its memory reservation
was substantially larger; the largest safe chunk was not the fastest at every prompt length.

The specified `ninfer_bench` target initially failed to compile because the new runtime helper used
an unqualified `execution::Parameters` type. Commit `49217d4` qualified it as
`models::qwen3_5::execution::Parameters`; the same target then built successfully. That code was used
for the complete matrix below. The commit also corrected the local Python 3.11 path in `AGENTS.md`.

Environment: device 0, NVIDIA GeForce RTX 4090 (24 GB), driver 610.57.04, CUDA compiler 12.8.61,
`sm_89`, Python 3.11.15. The command above used the explicit artifact
`/opt/ninfer-4090/Ternary-Bonsai-2-27B-ninfer-v3.ninfer`, INT8 group64 KV, DFlash2 K7, CUDA Graphs,
context 32768, and prompts 1024/4096/16384/32768. Each arm had one warmup and three measured
repetitions. All eight arms completed successfully. This measures isolated single-request prefill;
no decode workload was included.

Mean prefill throughput in tokens/s (higher is better):

| Requested chunk | 1K prompt | 4K prompt | 16K prompt | 32K prompt |
|---|---:|---:|---:|---:|
| 1024 | 3603.2 | 3306.5 | 2865.6 | 2440.8 |
| 1536 | 3533.3 | 3294.6 | 2968.4 | 2453.3 |
| 2048 | 3524.1 | 3438.7 | 2894.9 | 2474.5 |
| 3072 | 3507.8 | 3394.3 | 2931.4 | 2474.4 |
| 4096 | 3487.5 | 3388.5 | 2913.2 | 2486.0 |
| 6144 | 3502.3 | 3547.6 | 2908.2 | 2447.1 |
| 8192 | 3487.9 | 3550.4 | 2919.3 | 2511.6 |
| Auto -> 8192 | 3472.6 | 3478.6 | 2927.5 | 2498.5 |

Relative to fixed 1024 in this matrix, Auto changed throughput by -3.6%, +5.2%, +2.2% and +2.4%
at the four prompt lengths. These are observed means from one sequential matrix, not repeated
interleaved A/B evidence; the reports retain standard deviations. Fixed 1536 had the highest
observed 16K throughput, whereas fixed 8192 had the highest 4K and 32K throughput.

| Startup resource | Fixed 1024 | Auto -> 8192 |
|---|---:|---:|
| Main KV capacity (tokens / page groups) | 32768 / 512 | 32768 / 512 |
| Runtime reservation (MiB) | 1883.0 | 3242.5 |
| Workspace capacity (MiB) | 176.8 | 1116.3 |

Reservation and workspace are reported capacities, not measured peak VRAM. The baseline is fixed
1024 in the same post-merge binary; the pre-PR commit was not independently timed. This pass did
not test `kv-capacity auto`, pressure-driven rung fallback, serving context-cache reservations,
C2/C4/C8 decode interleave, greedy response equality or CI. It therefore supports successful
startup and isolated prefill performance at this configuration, not default promotion or a claim
about concurrent decode latency.

Committed evidence: [matrix summary](../../profiles/bench/prefill-chunk-matrix/summary.json).
Its eight per-arm JSON reports are committed beside it and include the memory/configuration and
timing statistics. Absolute paths in reports identify the measurement host; the relative links
here locate the committed evidence.
