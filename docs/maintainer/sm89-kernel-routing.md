# RTX 4090 (sm_89) kernel-route qualification

NInfer supports both sm_86 and sm_89, but schedule boundaries measured on one architecture are not
assumed to be valid on the other. The build now defines `NINFER_SM86` or `NINFER_SM89` in
addition to the shared compatibility macro, and DFlash2 Q8 attention/SwiGLU own separate route
catalogs.

The initial sm_89 catalog deliberately starts from the last qualified sm_86 mapping. It is a
candidate baseline, not a claim that the boundaries are optimal on Ada.

## Why K7 gets a focused sweep

For DFlash2 K7, the physical proposal/verify extents at client batches C1/C2/C4/C8 are exactly:

| client batch | columns |
|---:|---:|
| 1 | 8 |
| 2 | 16 |
| 4 | 32 |
| 8 | 64 |

Those four widths therefore dominate the production K7 path. On RTX 4090, Ada has a materially
different SM count and cache hierarchy from the RTX 3090 measurements that established the
existing tables. The first qualification pass keeps every existing candidate executable and adds
one Ada occupancy candidate without routing production traffic to it:

- T8/T16 use four K-split warps with a four-CTA/SM launch bound;
- T32 raises the K-split/MMA launch bound to four CTAs/SM;
- T64 attention keeps the 32x64 BK128 tile but raises its launch bound to four CTAs/SM;
- T64 SwiGLU keeps the 64x64 BK128 tile but raises its launch bound to three CTAs/SM.

These candidates change launch geometry/register budget only; stored weights and arithmetic stay
on the same Q8 path. They are measured cold at exactly the K7 widths before any production route
is changed.

Build the schedule benchmark and run:

```bash
cmake --build build -j --target ninfer_q8_dflash2_schedule_bench
python3 -m tools.bench.sm89_dflash2_k7_routes \
  --exe build/bench/ninfer_q8_dflash2_schedule_bench \
  --out profiles/bench/sm89-dflash2-k7-routes.json \
  --gpu 0 --repeat 31 --warmup 5
```

The driver requires the benchmark header to report `sm=89`, stores the full min/median/p95 log,
and writes one compact JSON row for each K7 extent and each of the two hot Ops.

Do not promote a one-run winner mechanically. A production route change should satisfy all of:

1. the same winner is observed in repeated cold sweeps;
2. its margin clears the candidates' own min..p95 spread;
3. the public Op timing agrees with the selected schedule rather than hiding a second launch;
4. the real DFlash2 K7 end-to-end output gate remains exact;
5. end-to-end throughput improves for the concurrency whose physical extent was changed.

This keeps architecture-specific tuning measurable while avoiding a runtime autotuner or per-request
hardware probing.

## RTX 4090 measurement, 2026-10-05 (PR #17)

Decision: retain the production routes. The new `sm89_occ_k7` candidate did not establish a
consistent advantage over the inherited routes. PR #17 adds benchmark candidates and separate
architecture catalogs; it does not establish a production throughput improvement.

The specified benchmark target built successfully on the merged code at `020d704`. Device 0 was
an NVIDIA GeForce RTX 4090 (24 GB), driver 610.57.04, with CUDA compiler 12.8.61 and `sm_89`.
The command above used `/home/fofo/.local/bin/python3.11`, 31 timed repetitions and five warmups.
This was one cold-cache sweep of the synthetic 27B Q8 fixtures at T8/T16/T32/T64.

Median kernel time in microseconds (lower is better):

| Extent / K7 concurrency | Attention production | Attention new candidate | SwiGLU production | SwiGLU new candidate |
|---|---:|---:|---:|---:|
| T8 / C1 | 56.3 | 59.4 | 258.0 | 257.1 |
| T16 / C2 | 58.4 | 57.3 | 262.1 | 261.1 |
| T32 / C4 | 73.7 | 73.7 | 269.3 | 269.3 |
| T64 / C8 | 81.9 | 81.9 | 337.9 | 343.0 |

The small apparent T8/T16 gains overlap the measured min..p95 distributions. At T64, an existing
alternative, `mma_r32_c64_k128`, measured 298.0 us for SwiGLU versus 337.9 us for the production
route (11.8% less time); their distributions were 295.9..299.1 and 334.8..341.0 us. This is a
candidate for further qualification, not a route promotion. Attention's existing C32 alternative
also measured 78.9 us at T64 versus production's 81.9 us, with overlapping distributions.

The comparison measures the original production kernels and candidate kernels in the same binary;
the pre-PR commit was not independently rebuilt and timed. Build and timing completion do not
qualify numerical correctness. No independent oracle, repeated sweep, real-model output gate,
end-to-end throughput test or CI check was run in this pass.

Committed evidence: [compact report](../../profiles/bench/sm89-dflash2-k7-routes.json) and
[full min/median/p95 log](../../profiles/bench/sm89-dflash2-k7-routes.json.log).
