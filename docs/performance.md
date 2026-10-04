# Single-GPU serving performance

> **Read the hardware label before the numbers.** This index and the per-model pages it links to
> carry measurements from different GPUs, and they must not be read against each other:
>
> | section | hardware | measured by |
> |---|---|---|
> | [RTX 4090 (`sm_89`) chain qualification](#rtx-4090-sm_89-chain-qualification) below | RTX 4090, `sm_89`, CUDA 12.8 | this fork |
> | [RTX 3090 (`sm_86`) findings](#rtx-3090-sm_86-findings-this-fork) below | RTX 3090, `sm_86`, CUDA 12.8 | this fork |
> | [Vision residency on RTX 3090](performance/qwen3.8-27b.md#vision-residency-on-rtx-3090-groupwise-int-sm_86) | RTX 3090, `sm_86` | this fork |
> | Every per-model page under `performance/` | **RTX 5090, `sm_120a`, CUDA 13.1** | upstream |
>
> The upstream campaign is kept because it is the only corpus-scale evidence published for these
> artifact profiles, and all of its tested revisions are reachable in this fork's history. It is
> **not** a statement about how this fork performs on an RTX 3090. The two parts answer different
> questions: upstream's tables characterise the artifact profiles, and this fork's measurements
> characterise `sm_86` kernel behaviour — where upstream's inherited route boundaries were wrong by
> 12–41%.

## RTX 4090 (`sm_89`) chain qualification

The final 2026-10-04 fixed-route campaign passes all 84 correctness
comparisons and qualifies 71 performance comparisons across seven workloads,
C1/C2/C4/C8 and K7/K11/K15. The subsequent
[resident DFlash2 qualification report](performance/dflash2-4090-2026-10-04.md)
completes full/selected calibration and held-out automatic-route comparisons:
18 real-model cases pass, all 720 validation requests match exact outputs, and
each GPU loads its model once. Fixed K7 is the default recommendation for that
matrix, with a Fixed K11 exception for single-request medium-context reuse.
Auto Selected is correct but does not qualify as faster than the best fixed
policy. Earlier server measurements below are historical evidence and do not
qualify the final routing profile.

The 2026-10-03 campaign uses the explicit local
`Ternary-Bonsai-2-27B-ninfer-v3.ninfer` artifact, INT8 KV, greedy zero-penalty
sampling, cache on and CUDA Graphs on. The original K15 chat CLI case now
matches all 512 target-only token IDs at context 8192. Public Engine checks also
match fresh and restored continuation, including `Root → PrivateTurnClosure`.
Seven-workload server AB/BA comparisons pass complete responses and repeated
output for K7/K11/K15 at C1/C2/C4/C8. HTTP does not expose token IDs; the
native actual-batch measurement matrix additionally compares exact token IDs.
See the [current qualification scope](maintainer/speculative-ab-testing.md#current-rtx-4090-qualification).

Two retained fresh-process server pairs measure K15 C1 at 2.567× and 2.601×
target-only end-to-end throughput. HTTP request p95 changes from 6.840 to 4.274
seconds and from 6.887 to 4.238 seconds. Server startup context and KV capacity
are 32768, engine concurrency is eight, and requests have a 512-token output
budget with thinking disabled. C1 denotes client concurrency. These reports
lack actual per-round batch/frontier measurements and do not qualify a routing
profile. K15 C8's diagnostic median throughput ratio is approximately 0.920×,
so the C1 result must not be generalized to all concurrent workloads.

The final native public-Engine campaign uses the repaired persistent GDN
transition, the explicit artifact above, INT8 KV, cache and graphs enabled,
greedy zero-penalty sampling, a 512-token output budget, context/KV capacity
32768 and startup concurrency eight. Two AB/BA pairs per comparison yield
336 fresh-process measurements. All 84 comparisons match exact token IDs,
complete responses and repeated output. Of these, 71 satisfy every paired
throughput gain, median gain of at least 2%, and paired request-p95 regression
of no more than 5%.

| Fixed action | C1 | C2 | C4 | C8 |
|---|---:|---:|---:|---:|
| K7 | 7/7 | 7/7 | 7/7 | 6/7 |
| K11 | 7/7 | 7/7 | 6/7 | 3/7 |
| K15 | 7/7 | 7/7 | 4/7 | 3/7 |

Fractions count qualified workloads, not actual batch coverage. Observer events
cover actual batches 1–8 and the ≤1024 and 1025–8192 frontier bands; no
8193–32768 cell occurs in the original seven workloads. Each action loads its
own fixed-width drafter. These results therefore cannot select an action whose
startup K15 drafter remains resident.

Across the seven workloads, median aggregate end-to-end throughput is:

| Requested concurrency | Target-only | Fixed K7 | Fixed K11 | Fixed K15 |
|---|---:|---:|---:|---:|
| C1 | 96.9 | 254.0 | 248.4 | 255.0 |
| C2 | 183.2 | 448.5 | 367.7 | 339.2 |
| C4 | 338.1 | 596.1 | 448.7 | 417.5 |
| C8 | 573.1 | 683.9 | 546.5 | 456.0 |

Units are generated tokens/s across concurrent requests. Candidate columns
are medians of each workload's two retained measurements; target-only is the
median across the paired baseline arms. These descriptive medians include
performance failures and do not replace per-comparison paired ratios. K7's
median paired gains across workloads are approximately 163%, 144%, 76% and
18% at C1/C2/C4/C8. Chat C1 measures 97.04 → 247.45 tokens/s with K7,
and request p95 falls from 5.313 to 2.083 seconds. Chat C8 K15 instead
loses 37% throughput; long-context C8 K7 loses 4%. The final policy must
retain target-only where no action qualifies. Evidence, actual round times,
commits, request p50/p95 and memory samples are retained in
`profiles/bench/consistency-2026-10-04/perf/fixed-final/`.

| C1 workload | Target-only tokens/s | K7 tokens/s | Paired gain | Request p95, target → K7 (s) |
|---|---:|---:|---:|---:|
| prose | 96.63 | 253.97 | +162.8% | 5.331 → 2.031 |
| chat | 97.04 | 247.45 | +155.0% | 5.313 → 2.083 |
| reasoning | 97.34 | 445.42 | +357.6% | 5.291 → 1.178 |
| code | 97.64 | 349.83 | +258.3% | 5.267 → 1.472 |
| structured | 97.96 | 219.86 | +124.4% | 2.988 → 1.346 |
| lookup-repeat | 88.99 | 384.62 | +332.2% | 6.126 → 1.699 |
| long-context | 83.14 | 160.44 | +93.0% | 6.848 → 3.884 |

These gains compare correct fixed DFlash2 against true target-only. They do
not establish a final automatic-route gain. The retained 2026-10-03 fixed
reports describe an earlier binary and are excluded from final calibration.

Numerical repairs have nonuniform Op costs. Fixed attention partitions and
high/residual query quantization, fixed-MMA GDN norm/control arithmetic, and
canonical group-64 T2A8 accumulation pass their independent mathematical
oracles and width regressions. Some short-width operations are slower than the
old inconsistent routes. Op microbenchmarks do not establish an end-to-end
policy improvement. Actual-batch calibration, resident maximum-drafter action
costs and automatic-route performance remain separate qualification gates.
Local commands, outputs and hardware records are retained under
`profiles/bench/consistency-2026-10-03/`.

Final long-context integration subsequently exposed an additional GDN
counterexample: a T1019 prefill uses a 59-token recurrent tail after 960
chunked tokens, while T1024 uses only 64-token chunks. Identical represented
inputs and initial states produce different common-prefix BF16 outputs;
target-only endpoint continuation then diverges at generated index two.
The repair uses the same FP32 recurrent transition at every positive width,
within one persistent kernel with tiled operand loading. Independent final
integration now passes endpoint continuation, context-boundary switching,
membership changes, stop/cancellation and zero-commit state release. Native
C4/C8 checks match every token and complete response against true target-only,
observe actual batches 1–8 and all four actions, and record 12 route switches
with no fixed fallback. Evidence is in
`profiles/bench/consistency-2026-10-03/integration-final-gnp/` and
`profiles/bench/consistency-2026-10-04/integration-final-native/`.
The final fixed measurements above include this repair. The current K15/C1
ring-wrap and context-tail continuation regression also passes. Resident-K15
calibration is running; no final automatic-route gain is claimed.

The final calibrated startup at concurrency eight, context 32768 and KV
capacity 131072 prepares and first-launches all 160 graph definitions and
executables for actual widths 1/8/12/16 and batches 1–8. Its observed graph
preparation device-memory increase is 1,684,013,056 bytes, within the 3 GiB
allowance. Runtime reservation is 11,341,939,456 bytes and logical workspace
peak is 185,393,152 bytes. These allocation measurements establish capacity;
throughput and sampled whole-device memory are recorded separately per arm.

Long-context resident-K15 action calibration additionally observes an actual
10,301-token input. All pairs pass complete token/response and repeat checks.
With a 768-token budget, K7 qualifies for actual B1–B6; its paired gains are
133% at B1 and 50% at B4. B7/B8 initially lack ≥95% common-cell elapsed
attribution, despite observed rounds at those batches. A focused 1536-token
budget supplement ends naturally at 1527 generated tokens and qualifies K7
at B7/B8 with gains of 20.2%/19.8%, while K11/K15 fail performance gates.
All eight high-context cells now have eligible K7 evidence. These gains compare
actions against resident-K15 action0, which skips the drafter; they are not
Auto-versus-true-None gains. The sampled device-wide peak in every tail arm is
17,949 MiB. Evidence is in
`profiles/bench/consistency-2026-10-04/perf/resident-long-final/` and
`profiles/bench/consistency-2026-10-04/perf/resident-long-tail-final/`.

The A16 compound Conv repair passes its independent FP64 oracle, reducing
the original B2/T1 relative-L2 counterexample from 0.003274 to 0.001694
against the unchanged 0.00315 limit. Whole-Op timings remain mixed: T1 and
T1024 are about 14% slower than the old BF16 staging path.

The `prefill_a8=false` checks exposed two additional public T2 A16 width
counterexamples: K15 diverged at generated index 16, and target-only Root
prefill segmentation diverged at index 177. Canonical KW4 accumulation now
passes independent FP64 qualification and exact width/segmentation checks,
including the actual Root projection subweights. None, fixed K15 and
calibrated target-only each pass four complete 512-token responses with
explicit zero penalties, cache and graphs enabled: 6144 token IDs match,
with Root, PrivateTurnClosure and reuse-off checks. Calibrated target-only
actually uses width one and skips the drafter without fixed fallback.
Cold-cache public Linear timings at N34816/K5120 change by +21.6% at T1,
+2.1% at T16 and −15.8% at T50. These are Op costs, not end-to-end gains;
the default campaign uses the unchanged T2 A8 route. Final automatic-switching
correctness passes; resident-drafter calibration and end-to-end automatic-route
performance remain pending.

## RTX 3090 (`sm_86`) findings — this fork

**DFlash2 measured on this fork (RTX 3090, Qwen3.8-27B groupwise-int, `--kv-dtype int8`,
`--draft-tokens 7`, greedy, 96 new tokens).** Text: 20.0% acceptance, 2.38 tok/round.
Vision (`--vision`, the committed `image_chart` fixture): 85.7% acceptance, 7.00 tok/round,
and byte-identical output to the non-speculative vision run. These are single-prompt smoke
numbers, not a campaign; the [Vision residency](performance/qwen3.8-27b.md#vision-residency-on-rtx-3090-groupwise-int-sm_86)
and per-model pages remain the measured corpus results.

**Historical RTX 3090 measurements found deterministic output within each configuration,
but different output across execution routes.** Measured 2026-09-09 by
hashing the generated text of 23 configurations x 3 repetitions
(`scripts/sweeps/dflash2-draft-tokens-realtext.ps1`, `content_sha256`):

- **every configuration reproduced its own output exactly**, 23 of 23, three runs each;
- the width-1 greedy path produced a hash matched by no speculative configuration;
- **DFlash2 and MTP do not match each other**, and DFlash2's output varies with the draft count —
  eight distinct outputs across k = 1..12.

Verification evaluates k+1 columns in one pass while plain decode evaluates one. Different
arithmetic routes can change a near-tie argmax, but the output hashes alone do not locate the
first differing operation or state transition. These historical measurements do not establish
cross-route equivalence or qualify the current RTX 4090 implementation.

The RTX 4090 DFlash2 release gate requires exact greedy output against target-only execution,
including prefix reuse, before performance selection. It does not require every private
floating-point intermediate to be bit-identical. See the [target consistency
contract](maintainer/dflash.md#greedy-target-consistency) and [A/B qualification
status](maintainer/speculative-ab-testing.md#current-rtx-4090-qualification).

*An earlier version of this section claimed DFlash2 and MTP "produce the same output as each other".
That was inferred from one prompt and is wrong; the hashes above are the measurement.*

**Acceptance on realistic text, which the committed corpus cannot measure.** 27B DFlash2, INT8 KV,
greedy, 256 generated tokens of the model's own prose, medians of three. `bench/fixtures/bench_corpus.ids`
reports a flat 100% at every draft count because it is a curated bank tiled to length; these are the
numbers to quote instead:

| `--draft-tokens` | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 10 | 12 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| acceptance | 80.9% | 67.6% | 51.7% | 49.1% | 42.1% | 33.1% | 28.2% | 31.0% | 23.2% | 19.5% |
| tok/round | 1.81 | 2.35 | 2.55 | 2.95 | 3.07 | 2.97 | 2.93 | 3.45 | 3.27 | 3.27 |
| decode tok/s | 47.3 | 55.2 | **58.6** | 58.3 | 57.2 | 51.0 | 50.5 | 47.5 | 39.7 | 39.3 |

Acceptance falls steadily as the window widens, from 80.9% at k=1 to 19.5% at k=12, with one
uptick against the trend at k=7 -> 8 (28.2% -> 31.0%); tokens-per-round plateaus near 3.0 from
k=4 — which is the real reason to recommend four rather than more. MTP3 reaches 58.3 tok/s at 56.0%
acceptance, and MTP3 with `--lm-head-draft` is the fastest configuration measured at 62.9 tok/s.

**Route boundaries are measured here, not inherited.** Two maintainer benches time every
schedule of an Op at the same column count, cold, so a boundary can be chosen from data rather
than from whichever table happened to be live:
`bench/ops/q4_q5_attn_input_schedule_bench.cu` and
`bench/ops/q4_linear_swiglu_schedule_bench.cu`. Both take an explicit per-schedule domain,
because a kernel handed more columns than it supports either corrupts memory (q4_q5
parent_split_fixed past 12) or silently does one column's work at a constant cost and appears to
win everywhere (q4 swiglu gemv_pair, registered for a single column). Cold is the production
regime for these: the Q4/Q5 attention projection alone streams ~35 MB of weights per call
against the 3090's 6 MB of L2.

The Q4/Q5 attention-input table had been dead code since the catch-up -- `resolve_plan` was a
hardcoded chain of upstream's sm_120 boundaries -- and routing through the measured table
exposed a `switch` fallthrough that ran a second kernel over the first. Public-Op cost at T=16
fell 502.8 -> 226.3 us and at T=32 500.7 -> 217.1 us. Q4 SwiGLU's SmallTTiled bound returned to
24, where the tiled kernel and the 40-wide pair tile actually cross (6% at T=25, 14% at T=32).

Concurrent decode extents are what the sm_86 kernel routes are selected for. A decode round covers
`concurrency x (draft window + 1)` token columns, and above the single-token point the cost of a
route is set by its padded tile width rather than by the live column count. Selecting the narrowest
tile that still covers each extent is worth 40% at C4 and C8; C1, whose four columns already sit on
the exact-T routes, is unchanged.

**The upstream DFlash2 corpus numbers are still not reproduced here.** The upstream catch-up
brought the DFlash2 speculative backend (`--spec dflash2 --draft-tokens 7`, Qwen3.8-27B only), and
upstream publishes DFlash2 corpus numbers for their own hardware. Those are deliberately not
added to the findings in this section: only single-prompt smoke numbers exist on a 3090 so
far, and pasting another architecture's corpus results beside them would read as agreement that
has not been measured. DFlash2 rows will be added here once measured on a 3090.

### Handing prefill GEMMs to cuBLAS (`--prefill-cublas`, opt-in)

**Result.** Prompt processing is 43-83% faster again, for +0.156% perplexity. The weights are
materialised as int8 with one scale per row and the GEMM is handed to cuBLAS, which runs this card's
shapes about twice as fast as this fork's own integer mainloop can. Measured on one RTX 3090,
Qwen3.8-27B groupwise-int, `--kv-dtype int8`, both arms on the same build and card in one session:

| prefill tok/s | 1k | 4k | 16k | 51k |
|---|---:|---:|---:|---:|
| shipped integer-activation routes | 1,666.6 | 1,634.4 | 1,503.9 | 1,240.8 |
| **`--prefill-cublas --prefill-chunk 4096`** | **2,379.8** | **2,988.7** | **2,609.0** | **1,904.1** |
| change | +42.8% | +82.9% | +73.5% | +53.5% |

Decode is untouched (45.1 against 45.7 tok/s, the route is gated to wide token counts), and the
workspace grows from 155.6 MiB to 661.2 MiB.

**It is off by default because it is a quality trade, not a free win.** cuBLAS reduces over the
whole of K, so it cannot see a scale per 64 columns: the weight carries one scale per row and the
activations one per token. Perplexity on `ninfer-ppl-1m-v1` (quick preset, kv int8) against 4.343155
for the integer route:

| | perplexity | change |
|---|---:|---:|
| MLP and out_proj on the route | 4.346284 | +0.072% |
| plus the attention and GDN input projections | 4.349944 | +0.156% |

The second step is `--no-prefill-cublas-projections` to decline. For scale, this fork has accepted a
trade at +0.082% and rejected one at +0.69%.

**Two things had to be true for the quality to land there.** The weight side is cheap because int8
buys four more bits per code while only losing per-group scale granularity: measured on the real
artifact it costs 9.7e-3 to 1.1e-2 relative L2 (the median per projection kind; the worst single
tensor is 1.5e-2), against the 9e-3 to 2.0e-2 this fork already accepts from per-group activation
quantisation (`tools/w4_row_scale_error.cpp`, using the route's own row scale with channel
equalisation off). An earlier figure of 8e-3 to 1.0e-2 came from a clipping search the route cannot
perform, and was the best case rather than the route. The activation side was
*not* cheap until the channels were equalised -- activation outliers concentrate in a few input
channels, so a token's absmax is set by those and everything else is quantised against far too
large a step. Folding a per-channel scale out of the activations and into the weights is exact
(`X[j,t]/s[j]` with `W[i,j]*s[j]` leaves the product unchanged) and took the cost from +0.474% to
+0.072%.

**Chunk size is the knob.** The dequantise pass is weight-sized and the GEMM token-sized, so the
route wants a large `--prefill-chunk` and the two settings belong together. The trade-off table,
including what each chunk costs a concurrent decode lane, is in
`src/ops/linear_swiglu/q4cublas/w4_cublas_prefill.h`.

### Integer activations for every registered prefill projection

**Result.** Prompt processing is 22-29% faster than the previous state at every context length:
the attention and GDN projections now reach the s8 tensor cores that only the MLP pair used, and
the schedule they share stopped re-streaming the weight matrix once per 128 tokens. Measured on one RTX 3090 at 315 W, Qwen3.8-27B groupwise-int, `--kv-dtype int8`,
`ninfer_bench -r 3 --warmup 1`, all three arms on the same build and card:

| prefill tok/s | 1k | 4k | 16k | 51k |
|---|---:|---:|---:|---:|
| `--no-prefill-a8` (every projection A16) | 1,008.5 | 995.4 | 946.9 | 836.4 |
| previous state (integer MLP only, 128-token tile) | 1,303.2 | 1,277.5 | 1,195.1 | 1,022.9 |
| every registered projection, 128-token tile | 1,529.8 | 1,493.3 | 1,382.3 | 1,153.3 |
| **every registered projection, widened tile** | **1,684.9** | **1,648.5** | **1,521.8** | **1,242.7** |
| change against the previous state | +29.3% | +29.0% | +27.3% | +21.5% |

Three routes were added, in the order their Nsight share justified — a 4,096-token prefill spends
749 ms of 3,280 ms in the GDN input projection, 346 ms in the two output projections and 209 ms in
the attention input projection:

| route | shape | per-Op, T=1024 | end to end, pp4096 |
|---|---|---|---|
| attention o_proj and GDN out_proj | Q5 [5120,6144] LinearAdd | 1,097.7 → 827.4 us | +3.0% |
| GDN input projection | Q4 [4096,5120] + Q5 [12288,5120] | — | +10.1% |
| attention input projection | Q4 + Q5 [7168,5120] | — | +1.6% |

They share one schedule (`src/ops/common/rowsplit_a8_mma.cuh`): the 128x128 tile the MLP routes
measured, with the codec and the output mapping as template parameters. Both input projections are
two parents over one activation, so each quantises its input once for all its launches — O(K*T)
against the GEMMs' O(N*K*T).

**Quality.** Quick corpus, same build: 4.342982 with `--no-prefill-a8` against **4.343155** with
every integer route on, **+0.004%** — inside run-to-run noise, and inside the +0.05% this fork
requires before a lossy route is default-on. Relative L2 against the FP64 oracle is 0.010-0.020
across every destination range, against the 0.04 allowance A8 activation compute is held to.
Decode is untouched (45.8 against 45.6 tok/s at tg128): the routes admit only full 128-column
prefill tiles, so decode and partial chunks stay on A16.

**The token tile, and why it was hiding.** Ablating the probe at the gate_up shape, T=512, gives
the decomposition that explains the whole schedule -- each row removes one cost and keeps the MMA
count identical:

| | us | |
|---|---:|---|
| complete kernel | 2,068 | |
| minus the activation and weight scale reads | 1,918 | the rescale's loads are ~7% |
| minus the per-group rescale entirely | 1,856 | the rescale is ~10% |
| minus the A-fragment shared reads | 1,423 | **assembling A is ~21%** |
| minus the B-fragment reads as well | 1,424 | B is free: one 128-bit load per n-tile |
| minus the MMAs, keeping every load | 1,377 | |
| minus the MMAs *and* every shared read | 1,378 | **streaming alone is 67% of the kernel** |

So these kernels were never compute-bound, and what generates the traffic is the token tile: a
block covering BN tokens re-streams the entire weight matrix once per column block, eight times
over at the production chunk of 1,024. Widening it is worth more than everything else attempted:

| tile (rows x tokens) | us at T=1024 | TOP/s |
|---|---:|---:|
| 128 x 128 | 4,265 | 85.6 |
| 128 x 256 | 3,702 | 98.6 |
| **64 x 512** | **3,060** | **119.3** |

A grid swizzle to let L2 serve the repeated passes measured within 0.5% of nothing: the passes have
to be removed, not cached. gate_up is the exception that proves the rule -- its gate/up pairing
fixes the row tile at 128 and so caps it at 256 tokens, where the tile it gains does not pay for
the operand assembly it would lose, so it keeps its own kernel.

**What is still missing, measured against a tuned kernel rather than against a microbenchmark.**
`tools/int8_gemm_reference_probe.cu` runs cuBLAS's own int8 GEMM (IMMA, TN, s32 accumulate) over
these shapes on this card. It solves a strictly easier problem -- 8-bit weights, no unpack, no
per-group scale, no fused epilogue -- and it reads *twice* the weight bytes our int4 codes do, so
it is an upper bound rather than a like-for-like rival. At T=1024:

| shape | cuBLAS int8 | this fork's W4A8 | ratio |
|---|---:|---:|---:|
| gate_up 34816x5120 | 237.5 TOP/s | 95.5 | 2.5x |
| down 5120x17408 | 176.1 TOP/s | 100.1 | 1.8x |
| out_proj 5120x6144 | 176.7 TOP/s | 93.3 | 1.9x |

That reading also corrects the ablation above: cuBLAS finishes the entire gate_up GEMM in 1,537 us,
less than the 1,711 us our schedule spends on streaming *alone* at the same shape and tile. The
streaming is not a hardware floor, it is our streaming -- a 2-stage pipeline over 16 warps with the
A fragments assembled from per-lane 2-byte loads. Everything cheap has now been tried against it:
pipeline depth 2 -> 4 is worth 3.9%, interleaving the B loads with the MMAs -1.9%, occupancy 6%,
grid swizzle 0.5%, operand layout ~17% (and that one needs a repack this engine does not allow).
What is left is not a knob but a mainloop: register-level double buffering, `ldmatrix`, and a
swizzled shared layout, which is what CUTLASS-class kernels -- and the Marlin kernel the vLLM
stacks use -- are built out of.

**What did not pay, so nobody re-walks it.** `TODO.md`'s long-standing explanation for these
kernels running at ~30% of the INT8 ceiling — 124 registers holding the SM to 16 of 48 warps — is
not what costs the time. `tools/w4a8_rowsplit_probe.cu` measures the alternatives at the gate_up
shape, T=512:

- **Doubling occupancy buys 6%.** A 64x128 tile compiles to 55 registers and genuinely runs 2
  blocks per SM (32 of 48 warps, confirmed with `cudaOccupancyMaxActiveBlocksPerMultiprocessor`):
  1,890 → 1,785 us, against the 40% `ncu` estimated.
- **Smaller tiles past that lose badly**: 128x64 at 63 registers measures 2,844 us, 1.5x worse.
- **Layout is worth nothing without a repack.** Streaming the weights with `cp.async` in their
  RowSplit order, packed nibbles in shared and the scales as an async ring — everything the
  fragment-order probe won except permuting the weights — measures 1,726 us against production's
  1,701. `tools/w4a8_real_weight_probe.cu` puts the fully repacked layout at 1,566 us with
  per-group scales, so the whole layout lever is ~8%, and a permuted copy of the 27B's two MLP
  matrices is ~9.7 GB on a 24 GB card. The 1,415 us headline needs per-token activation scales,
  whose relative L2 reaches 12.9% on outlier-heavy inputs against 0.9-2.0% per group.

### Small-T tensor-core kernels for verify and cohort decode

**Result.** Qwen3.8-27B MTP3 decode is 1.5x faster at C1 and 1.7x at C8 than v0.9.1, with
perplexity bit-identical. Measured before/after on one rented RTX 3090 in one session: Linux,
CUDA 12.8, 350 W, INT8 KV, MTP3 with the optimized draft head, CUDA Graphs, greedy unless noted.

| workload | v0.9.1 | small-T kernels |
|---|---:|---:|
| thinking-off chat, C1 (their 8 prompts, C × 1000 / mean TPOT) | 74.6 tok/s | **113.2 tok/s** |
| thinking-off chat, C1, temperature 0.7 | 72.6 tok/s | **112.8 tok/s** |
| thinking-off chat, C8 | 268.6 tok/s | **460.4 tok/s** |
| reasoning cohort C1 / C2 / C4 / C8 decode | 63 / 89 / 137 / 221 | **93 / 168 / 271 / 376** |
| `ninfer_bench` plain / MTP3 | 39.98 / 53.96 | **47.15 / 85.29** |
| perplexity, quick corpus | 4.342425 | 4.342425 |

The thinking-off prompts are
[syv-ai/qwen38-27b-rtx3090](https://github.com/syv-ai/qwen38-27b-rtx3090)'s
`bench/prompts_real.jsonl`. That project serves the same model on the same card through patched
vLLM, and reports 111-124 tok/s at C1 and 407.3 at C8 with the same metric. Their card is capped
at 250 W and they report 5-8% run-to-run spread, so C1 is parity and C8 a lead, measured on
different machines.

**Why the round cost too much.** An MTP3 verify is four token columns and a C8 cohort round is 32.
The profile attributing an MTP3 round against a plain decode step
(`scripts/sweeps/decode-step-profile.sh 27b-decode 27b-decode-mtp3`) had the round at 38.2 ms
against a 24.8 ms step, about 1.5x. vLLM/Marlin pays 1.14x for the same round on this card. The
extra time sat in four families: Q4 gate_up, the Q5 residual projections, and the GDN and
attention input projections. At four columns they ran either SIMT split kernels, whose cost grows
with every column, or a Q4 small-T MMA that was instruction-bound. The round is now 25.7 ms
against a 22.3 ms step, 1.15x.

**What changed**, cold single-kernel medians in us, from the schedule benches
(`bench/ops/*_schedule_bench.cu`):

| Op (27B shape) | T | v0.9.1 route | now |
|---|---:|---:|---:|
| Q4 gate_up, 34816 × 5120 | 4 | 200.7 | **120.8** |
| | 32 | 444.4 (c40 tile) | **~205** |
| Q5 down, 5120 × 17408 | 4 | 99.3 (split2) | **82.9** |
| | 32 | 289.8 (best MMA tile) | **144.4** |
| Q5 out, 5120 × 6144 | 4 | 38.9 (split2) | **34.8** |
| | 32 | 105.5 (best MMA tile) | **55.3** |
| GDN input, Q4 4096 + Q5 12288 rows | 4 | 100.4 (independent) | **78.8** |
| | 32 | 263.2 (grouped c32) | **153.6** |
| attention input, Q4 7168 + Q5 7168 rows | 4 | 90.1 (parent split) | **68.6** |
| | 32 | 191.5 (r32/c32) | **135.2** |

- **Bank conflicts.** The Q4 small-T MMA staged its code rows 256 B apart, an 8-way shared-memory
  bank conflict on every A load. Padding the rows was worth 9.6% at C1 on its own.
- **Permuted k order.** The MMA's k slots may be any permutation of a group, provided A and B agree.
  Giving each lane sixteen contiguous k makes its A operand one 64-bit code load, decoded to bf16
  by a magic-bias `byte_perm` with no int-to-float, and its B operand two 128-bit loads with no
  `ldmatrix`.
- **A Q5 counterpart** (`q5_small_t_mma.cuh`) folds the high-bit plane into the same decode. It
  replaces split2 from T=3 and the MMA tiles to 32 columns, for the residual projections, GDN
  value/z and attention gate/value.
- **Wide extents share activation slabs** (`ops/common/small_t_layout.cuh`). At 32 columns and one
  16-row tile per CTA, gate_up re-read ~713 MB of staged activations per call against 89 MB of
  weights. Two to four row tiles per CTA now share each staged slab, and at 24 columns each warp
  feeds one B fragment to two tiles.
- **The fused GDN conv projection is gone.** Batch-1 widths 1-3 and 5-6 of the GDN conv
  forms ran a fused SIMT projection that lost at every width: 95.2 against 80.9 us at T=1, and
  162.8 against 85.0 at T=5. At T=5 that was 4.8 ms of a four-draft-token round. Removing it made
  plain decode 5% faster, and four or five draft tokens stopped costing more than they return.

**Measured and rejected**, so nobody re-runs them:

| idea | result |
|---|---|
| split-K across CTAs for the 5120-row Q5 shapes | 0-2.5% at narrow T; not worth a workspace |
| magic-bias decode alone, before the k permutation | neutral in situ |
| 2- and 3-stage cp.async rings for the narrow Q4 tile | slower; occupancy is the latency hiding |
| deeper rings through dynamic shared memory (to 4 stages, 99 KB) | slower at every shape: occupancy loss outweighs the hidden latency |
| two tiles per warp at 16 and 32 columns | slower (kept only at 24, where it wins 14%) |
| `cp.async ... L2::256B` prefetch hint on weight loads | neutral |
| a 40-65K-row prefix of the frequency-sorted draft head | acceptance 62.7% -> 56.5% at 40,960 rows; net 4% slower |
| drafting with the full output head | +2% acceptance for 4 ms more per round |
| programmatic dependent launch to hide kernel ramp | needs sm_90; compiled out on `sm_86` |

What is left at C1 is mostly ramp-up and drain at the ~500 kernel boundaries of a round (the
Q5 residual kernels reach 68-82% of their streaming floor, gate_up 89%).

**What is left at C8 is not tensor-core rate, though this page said so for a cycle.** The claim was
that gate_up at T=32 runs at about 68% of the card's measured bf16 MMA peak -- 205 us against a
139 us tensor floor -- which reads as a kernel most of the way to saturating its tensor cores.
Counters disagree. `ncu` on the T=32 gate_up kernel, Windows RTX 3090 at 315 W:

| | BF16 small-T | int8 small-T |
|---|---:|---:|
| tensor pipe active | **38.3%** | 21.0% |
| L1/TEX throughput | 36.4% | 73.5% |
| DRAM throughput | 39.4% | 42.7% |
| warp occupancy | 30.9% | 45.4% |

Nothing is near saturation in the BF16 column: tensor, L1 and DRAM all sit within three points of
each other around 38%, at 31% occupancy, which is the signature of a latency-bound kernel rather
than one limited by any unit. The 68% figure compares against a computed floor, not against a
measured pipe, and the two do not agree.

That distinction decided a real experiment. An int8 tensor-core route for this Op was built on the
strength of the old reading, since s8 MMA is about 4.7x the bf16 rate on this hardware. It
delivered 4-6%, not 4.7x, and the right column above says why: it did exactly what a denser MMA
should, halving tensor-pipe pressure from 38.3% to 21.0%, but that pipe was never the constraint,
and the cost of getting there landed on L1 at 73.5%. See
[the quality-trade notes](maintainer/quality-trade-experiments.md) and
`src/ops/linear/q4/q4_small_t_mma_i8.cuh`. Work aimed at C8 should target operand movement and
occupancy; a wider tile that puts more work in flight is the lever the counters actually point at.

**Reproduce.** Build both trees with `-DNINFER_BUILD_APPS=ON -DNINFER_BUILD_BENCHMARKS=ON` and run
each harness against both `ninfer-serve` binaries on one card, interleaved in one sitting:
`tools/bench/run_qwen38_replayssm_cohort_sweep.py` for the cohort, and for the thinking-off numbers

```bash
python tools/bench/run_chat_decode.py --model models/qwen3_8_27b.ninfer \
  --prompts prompts_real.jsonl --concurrency 1 --reps 2 --out chat-decode \
  --arm base=/path/to/old/ninfer-serve --arm new=./build/apps/ninfer-serve
```

with the eight prompts of syv-ai/qwen38-27b-rtx3090's `bench/prompts_real.jsonl` (that file is
theirs and is not vendored here). `--concurrency 8` gives the C8 row. Both report decode as
C × 1000 / mean TPOT from the server's own request log.

### Recommended configurations (RTX 3090, Qwen3.8-27B)

Every flag below is measured elsewhere in this file or in
[quality-trade-experiments.md](maintainer/quality-trade-experiments.md). Nothing here is a guess.

**Fastest at one stream, when context beyond ~130K is not needed:**

```
--spec dflash2 --draft-tokens 7 --lm-head-draft --prefill-cublas --prefill-chunk 4096 --kv-dtype rk8v4 --embedding-q4 --gdn-state-fp16
```

Prefill about 1.7x and decode about 1.39x against the previous defaults. Costs +0.156% perplexity
from the cuBLAS route, +0.083% from rk8v4, and nothing measurable from the other two.

**Longest context, still fast:**

```
--spec mtp --draft-tokens 3 --lm-head-draft --prefill-cublas --prefill-chunk 2048 --kv-dtype rk8v4 --embedding-q4 --lm-head-q6 --gdn-state-fp16
```

**200,000 tokens of context, verified by loading it.** MTP rather than DFlash2 because DFlash2's
draft weights and its refusal of `--lm-head-q6` cost about 65K tokens between them -- the same
configuration on DFlash2 loads at 130K and fails at 150K (needing 4.79 GB against 4.46 free) and at
200K (6.09 GB against 4.45). Chunk 2048 rather than 4096 because the larger chunk costs ~300 MiB of
workspace for its last 4% of prefill, and at this context that is the binding constraint.

**Why these flags and not others:**

| flag | what it buys | what it costs |
|---|---|---|
| `--kv-dtype rk8v4` | 23% less KV than int8 — 26,112 B/token against 33,792 | +0.083% perplexity, ~1% decode |
| `--embedding-q4` | +24.3K tokens of context | nothing measurable (−0.062%, inside noise) |
| `--lm-head-q6` | +12.9K tokens of context | +0.012%; **incompatible with DFlash/DFlash2** |
| `--gdn-state-fp16` | halves the host state image, modest C8 gain | nothing measurable |
| `--prefill-cublas` | prefill 1.63x–1.83x | +0.156% perplexity, ~500 MiB workspace |
| `--lm-head-q4` | +24.3K tokens | **+0.69%** — larger than every KV format; not recommended |
| `--kv-dtype nvfp4` | 45% less KV than int8 | +0.36% perplexity, 12% slower decode |

**With vision**, add `--vision --vision-residency overlay`: the overlay residency keeps the tower in
host memory and borrows device memory per image, which is what preserves the context budget above.
Note that `--lm-head-q4` was silently skipped under overlay vision before 2026-09-14; `--lm-head-q6`
and `--embedding-q4` transcode in the materializer and are captured from the final bytes, so they
compose with it correctly.

### Choosing a speculative backend by concurrency (RTX 3090, Qwen3.8-27B)

**DFlash2 at K=7 is faster than MTP3 wherever it fits, and it stops fitting at C8.** Aggregate
decode tok/s through the serving route, thinking off, greedy, decode time taken from the server's
request log rather than a wall clock:

| C | MTP3 | DFlash2 K=7 | change | tokens/round, MTP3 / DFlash2 |
|---:|---:|---:|---:|---|
| 1 | 135.0 | **187.1** | +38.6% | 3.51 / 5.57 |
| 2 | 238.2 | **313.5** | +31.6% | 3.51 / 5.82 |
| 4 | 387.4 | **406.2** | +4.9% | 3.54 / 5.63 |
| 8 | **522.8** | does not fit | — | 3.53 / — |

The lead shrinks with concurrency because speculation pays by making a multi-column round cost about
one sweep of the weights, and batching already amortises that sweep across lanes, so the baseline
catches up. Acceptance itself does not degrade: tokens per round holds flat at every level.

What stops it at C8 is memory. The DFlash2 artifact carries the draft model — 18.3 GiB of weights
against 16.7 — and the runtime reservation then needs 4.64 GB where 3.78 GB remains. A 4096-token KV
still needs 4.24 GB; only 2048 fits, which is too little for eight streams to do useful work.

**K is workload-dependent.** Swept on realistic generation, mean of a reasoning, a code and a
summarisation prompt: K = 3/4/5/6/7/8/9/10/12 gives 128.0/146.0/159.0/169.6/**172.3**/163.7/163.3/
160.2/159.2 tok/s. The peak is 7, matching the published shape `E = (1 - a^(K+1))/(1 - a)` and the
5–8 band reported for EAGLE-class drafters. But the mean hides real spread — the reasoning prompt
keeps improving to K=12 (204.5) while summarisation collapses there (119.6) — so a deployment
serving one kind of work should sweep its own K. A synthetic corpus is no guide: `bench_corpus.ids`
picks K=15, which loses on real generation.

**DFlash2 also forfeits `--lm-head-q4/q6`**, which DFlash's candidate top-k cannot read
(`src/models/qwen3_5/load/storage_trades.cpp`). Between the extra weights and the lost head trade it
costs about 65K tokens of context on this card -- 130K against 200K, both verified by loading -- so
the choice between it and MTP3 is a speed/context trade, not a free upgrade.

### Choosing a KV format (RTX 3090, Qwen3.8-27B)

All six SM86 KV formats, measured on Qwen3.8-27B.

| KV profile | Bytes/token | KV at 2,048 tokens | Perplexity | vs `bf16` | Decode at 32K depth |
|---|---:|---:|---:|---:|---:|
| `bf16` | 65,536 | 128.00 MiB | 4.343225 | — | 32.50 tok/s |
| `int8` | 33,792 | 66.00 MiB | 4.343263 | +0.0009% | **33.86 tok/s** |
| `fp8` | 33,024 | 64.50 MiB | 4.347181 | +0.0911% | 30.13 tok/s |
| `rk8v4` | 26,112 | 51.00 MiB | 4.346811 | +0.0826% | 33.54 tok/s |
| `k8v4` | 25,728 | 50.25 MiB | 4.347596 | +0.1006% | 28.61 tok/s |
| `nvfp4` | **18,432** | **36.00 MiB** | 4.358924 | +0.3615% | 29.62 tok/s |

Perplexity is `ninfer-perplexity` on the fixed `ninfer-ppl-1m-v1` corpus, `--quick`, context/stride
4096/2048, 261,167 scored tokens. Decode is 128 timed steps on top of a 32,768-token prefill, no
speculation; attention re-reads the whole cache each step, so a format's cost only shows at depth.
See [the README](../README.md#choosing-a-kv-format) for the fuller writeup and recommendations.

## Published coverage

Published measurements use one NVIDIA GeForce RTX 5090 through NInfer's public HTTP serving route.
Choose a model below for its detailed results, run conditions, output limitations, and reproduction
commands. These are recorded historical measurements; a model/backend being supported does not
mean every workload or concurrency has a published measurement.

Read the [measurement and publication rules](performance/methodology.md) for workload definitions,
metric formulas, statistics, comparison requirements, and the standard result-page format.

Each cell links to the relevant result section. "Not published" describes measurement coverage,
not product support. C is configured request concurrency; K is the number of draft tokens.

| Model / weights | MTP0 context profile | Single-request speculative decode | Corpus makespan | MTP3 decode saturation |
|---|---|---|---|---|
| Qwen3.6-27B / `groupwise-int` | [8K–256K](performance/qwen3.6-27b.md#no-speculation-context-profile) | [MTP3](performance/qwen3.6-27b.md#single-request-speculative-decode) | Not published | [C=1, 2, 4, 8](performance/qwen3.6-27b.md#decode-saturation) |
| Qwen3.6-27B / `nvfp4` | [8K–256K](performance/qwen3.6-27b.md#no-speculation-context-profile) | [MTP3](performance/qwen3.6-27b.md#single-request-speculative-decode) | Not published | [C=1, 2, 4, 8](performance/qwen3.6-27b.md#decode-saturation) |
| Qwen3.6-35B-A3B / `groupwise-int` | [8K–256K](performance/qwen3.6-35b-a3b.md#no-speculation-context-profile) | [MTP3; DFlash K=7 stochastic/greedy](performance/qwen3.6-35b-a3b.md#single-request-speculative-decode) | [MTP3 C=1, 2, 4, 8; DFlash C=1](performance/qwen3.6-35b-a3b.md#corpus-makespan) | [C=1, 2, 4, 8](performance/qwen3.6-35b-a3b.md#decode-saturation) |
| Qwen3.8-27B / `groupwise-int` | [8K–256K](performance/qwen3.8-27b.md#no-speculation-context-profile) | [MTP3; DFlash2 K=7](performance/qwen3.8-27b.md#single-request-speculative-decode) | [MTP3 C=1, 2, 4, 8; DFlash2 C=1](performance/qwen3.8-27b.md#corpus-makespan) | Not published |
| Qwen3.8-27B / `nvfp4` | [8K–256K](performance/qwen3.8-27b.md#no-speculation-context-profile) | [MTP3; DFlash2 K=7](performance/qwen3.8-27b.md#single-request-speculative-decode) | [MTP3 C=1, 2, 4, 8; DFlash2 C=1](performance/qwen3.8-27b.md#corpus-makespan) | [C=1, 2, 4, 8](performance/qwen3.8-27b.md#decode-saturation) |

Qwen3.8 and Qwen3.6-35B-A3B C=1 corpus points also supply their single-request phase tables.
The Qwen3.6-27B NVFP4 MTP3 phase table comes from a corpus C=1 point whose full makespan is
not published here. The Qwen3.8 NVFP4 saturation reports retain configuration and
values but no tested Git revision; the model page records that provenance limitation.

## Reading the results

| Question | Metric to use |
|---|---|
| How fast is prompt processing or an individual decode phase? | Prefill phase, Server TTFT, Decode phase |
| How long does the full fixed request set take? | Corpus makespan, Corpus decode, Requests/s |
| What aggregate decode rate is sustained at a full batch? | Steady decode |

These rates use different time boundaries. Server TTFT is an internal phase sum; external
streaming TTFT has its [own benchmark contract](../tools/bench/ttft/README.md). Stochastic runs
can generate different token totals even with the same prompts and seeds. Output-limit and
repetition samples remain labeled in the measured corpus; throughput alone does not establish
successful task completion. See the [35B termination and anomalies](performance/qwen3.6-35b-a3b.md#termination-and-anomalies)
and [Qwen3.8 DFlash2 outcomes](performance/qwen3.8-27b.md#dflash2-completion-outcomes).

## Related references

- [Serving benchmark runners](../tools/bench/README.md#serving-corpus-benchmark): usage and local report files.
- [Engine and Op benchmarks](../bench/README.md): their separate measurement scopes and commands.
- [Capability evaluation](../eval/README.md): evaluation workflow; published scores live in the
  [model cards](README.md#model-artifacts), with a [README summary](../README.md#evaluation).
- [Perplexity](perplexity.md): offline causal-scoring measurement and comparison rules.

Model pages are the detailed result authority. README and model-card performance tables are
excerpts linked to those pages; update them together when replacing an applicable measurement.
