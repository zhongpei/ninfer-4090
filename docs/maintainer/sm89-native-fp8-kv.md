# RTX 4090 native FP8 KV attention

Status: **experimental, opt-in through `--kv-dtype fp8`**. INT8 remains the product baseline until
the long-history quality and throughput gates are run on Bonsai 27B.

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

Each BF16 query row is Hadamard-rotated exactly like the cache producer, quantized to E4M3 with one
FP32 query scale, and contracted directly against cached E4M3 K codes:

```text
Q_bf16 -> Hadamard -> Q_e4m3 + q_scale
K_cache = K_e4m3 + k_scale

raw_score = MMA(Q_e4m3, K_e4m3)
score     = raw_score * q_scale * k_scale
```

The score is then masked/scaled and fed to the existing online-softmax/split reducer contract.

## Native PV with row-scaled V

A V cache row has its own scale, so directly multiplying `P_fp8 * V_code_fp8` would be wrong.
For one key tile the kernel instead forms:

```text
A[k] = P[k] * v_scale[k]
```

and quantizes A per query row:

```text
A ~= a_scale * A_e4m3
V = v_scale * V_e4m3

sum(P * V) ~= a_scale * MMA(A_e4m3, V_e4m3)
```

Softmax maxima and denominators continue to use the unquantized FP32 probabilities. Only the PV
Tensor Core operand is quantized. That extra probability-side E4M3 boundary is an intentional
quality/performance trade and is why this change ships together with a depth-conditioned
perplexity protocol.

## Unified sm89 route

On sm89, `Fp8E4M3Row256` no longer falls back to the legacy widened prompt kernel. K/V are
published once through the canonical paged-cache append Op, and all query widths are segmented into
the native small-T implementation. This gives decode, speculative verify and causal-scoring
prefill the same FP8 arithmetic.

The first implementation chooses correctness/one implementation over a specialized wide-prompt
tile. Wide prompt throughput may therefore need a later native FP8 prompt kernel after the quality
gate establishes that the arithmetic is acceptable.

sm86 retains the legacy widened implementation.

## Long-history quality gate

Use the fixed-depth evaluator, not ordinary 4K sliding-window perplexity:

```bash
python3 -m tools.bench.run_kv_long_context_perplexity \
  --exe ./build/apps/ninfer-perplexity \
  --model /absolute/path/Ternary-Bonsai-2-27B-NInfer-v3.ninfer \
  --corpus eval/corpora/perplexity-1m/manifest.json \
  --depths 8192,32768,65536,131072,196608,258048 \
  --tail 2048 \
  --dtypes int8,fp8,rk8v4,rk4v4-e8 \
  --out profiles/perplexity/bonsai-kv-long
```

INT8 is the default baseline. Evaluate the per-depth `delta_mean_nll_vs_baseline` and
`ppl_change_percent_vs_baseline`; do not use only the aggregate row.

No quality or performance result is claimed by this implementation PR. Local RTX 4090 measurement
is authoritative.
