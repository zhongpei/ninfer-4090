# RTX 4090 native FP8 KV attention

Status: **opt-in through `--kv-dtype fp8`**. Bonsai 27B qualification on 2026-10-05
selects INT8 as the sm89 product default. Native-FP8 passes 25 focused numerical cases and has
measured quality through a 258048-token history. It improves decode on the tested 29K synthetic
input while reducing prefill throughput. See the [four-format report](../performance/bonsai-kv-4090-2026-10-05.md)
for measurements and their limits.

## Why this path exists

RTX 4090 is `sm_89`. Ada Tensor Cores natively execute:

```text
mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32
```

The previous NInfer compatibility path treated FP8 KV as storage only: K widened to BF16 before QK
and V widened to FP16 before PV. That discarded the hardware's native E4M3 contraction while saving
only a small amount of cache memory versus INT8.

The sm89 path keeps persistent K/V codes in their row-scaled E4M3 representation.

## Native QK

Each BF16 query row is Hadamard-rotated exactly like the cache producer, represented as main and residual E4M3 terms with independent
FP32 scales, and contracted directly against cached E4M3 K codes:

```text
Q_bf16 -> Hadamard -> Q_main_e4m3 + q_main_scale
                         + Q_residual_e4m3 + q_residual_scale
K_cache = K_e4m3 + k_scale

score = (MMA(Q_main_e4m3, K_e4m3) * q_main_scale
       + MMA(Q_residual_e4m3, K_e4m3) * q_residual_scale) * k_scale
```

The score is then masked/scaled and fed to the existing online-softmax/split reducer contract.

## Native PV with row-scaled V

A V cache row has its own scale, so directly multiplying `P_fp8 * V_code_fp8` would be wrong.
For one key tile the kernel instead forms:

```text
A[k] = P[k] * v_scale[k]
```

and represents A as main and residual E4M3 terms per query row/tile:

```text
A ~= a_main_scale * A_main_e4m3 + a_residual_scale * A_residual_e4m3
V = v_scale * V_e4m3

sum(P * V) ~= a_main_scale * MMA(A_main_e4m3, V_e4m3)
            + a_residual_scale * MMA(A_residual_e4m3, V_e4m3)
```

Softmax maxima and denominators continue to use the unquantized FP32 probabilities. Only the PV
Tensor Core operand is quantized. The residual term reduces transient quantization error while
keeping native FP8 contraction. These are private arithmetic details, not new public storage
boundaries; qualification still uses the independent FP64 attention oracle and the existing
FP8 criterion. The depth-conditioned perplexity protocol measures the real model consequences
of the stored KV representation and execution arithmetic.

## Unified sm89 route

On sm89, `Fp8E4M3Row256` no longer falls back to the legacy widened prompt kernel. K/V are
published once through the canonical paged-cache append Op, and all query widths are segmented into
the native small-T implementation. This gives decode, speculative verify and causal-scoring
prefill the same FP8 arithmetic.

The implementation uses the same arithmetic for wide prompts and small query widths.
The measured prefill penalty is recorded in the report; a specialized wide-prompt tile would need
its own numerical and performance qualification.

sm86 retains the legacy widened implementation.

## Long-history quality gate

Use the fixed-depth evaluator, not ordinary 4K sliding-window perplexity:

```bash
python3 -m tools.bench.run_kv_long_context_perplexity \
  --exe ./build/apps/ninfer-perplexity \
  --model /absolute/path/Ternary-Bonsai-2-27B-NInfer-v3.ninfer \
  --text /absolute/path/long-evaluation-stream.txt \
  --depths 8192,32768,65536,131072,196608,258048 \
  --tail 2048 \
  --dtypes int8,fp8,rk8v4,rk4v4-e8 \
  --out profiles/perplexity/bonsai-kv-long
```

INT8 is the default baseline. Evaluate the per-depth `delta_mean_nll_vs_baseline` and
`ppl_change_percent_vs_baseline`; do not use only the aggregate row.

Local RTX 4090 measurements in the linked report are authoritative for the tested workload;
they do not qualify other architectures or extended YaRN contexts.

The input must reach every requested depth. The bundled corpus has approximately 64K tokens per
stream; for deeper measurements explicitly construct a longer stream and record its source order.
The matrix runner rejects missing depths or unequal stream/target coverage rather than producing
a partial comparison.
