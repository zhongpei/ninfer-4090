# RTX 4090 Ternary + YaRN merge provenance

This fork's first combined line is based on:

- `iamwavecut/ninfer-3090@franken/v0.11` at `0cfc651ee78555de1ae3258bd7a922ea8f43f54f`.
  This is the execution/runtime base and supplies Ternary Bonsai 2 support, NInfer v3,
  Hadamard-rotated `t2_g128_fp16`, MTP/DFlash2, Vision and the current Qwen3.8 path.
- `alanthinker/ninfer-4090-yarn`. The port keeps its corrected YaRN semantics rather than
  cherry-picking the old runtime: cache positions stay absolute, RoPE positions are scaled,
  prefill and decode use the same double-precision rounding, MTP AR positions are scaled, and
  multimodal `[T,3]` MRoPE is transformed elementwise.
- `sergiuszm/ninfer-4090@rtx4090-port`. The E8 KV kernel family was imported from this line:
  H64 rotation, packed 4-bit K/V, E8 lattice projection for `rk4v4-e8`, E8-root coding for
  `rk2v4-e8`, and inverse value rotation after attention.

## Product contract for v1

- GPU target: RTX 4090 / `sm_89` (default CMake architecture).
- Primary model: `WaveCut/Ternary-Bonsai-2-27B-NInfer-v3`.
- Native context: up to 262,144 tokens; MTP and DFlash2 are available.
- Extended context: YaRN-style linear RoPE scaling; MTP is supported.
- DFlash/DFlash2 + YaRN: rejected at startup. The drafter currently aliases proposal RoPE
  positions with physical KV positions, so scaling only the former requires a later position-domain
  split.
- Causal-attention execution envelope: 786,432 tokens; the intended first acceptance targets are
  512K and 658,176.
- Preferred long-context KV mode: `rk4v4-e8`.

## YaRN mapping

For absolute RoPE position `p`, native threshold `N`, and factor `f >= 1`:

```
p' = p                                      , p <= N
p' = N + round((p - N) / f)                , p > N
```

KV/cache addresses always use `p`, never `p'`. Device and host paths evaluate the quotient in
double precision before adding `N` so a token reaches the same RoPE position through prefill,
ordinary decode, MTP verify, MTP AR, prefix reuse, and Vision MRoPE.

## Follow-up

DFlash2 + YaRN is deliberately not part of v1. The required follow-up is to split the drafter's
single proposal-position domain into independent cache positions and RoPE positions, propagate both
through proposal/verify and CUDA Graph capture, then remove the startup guard.
