# NInfer-4090: Ternary Bonsai 2 and YaRN

[中文（默认）](README.md) · **English**

NInfer-4090 is a local C++/CUDA inference engine targeting **NVIDIA GeForce RTX 4090 (24 GB, `sm_89`)**.
Its primary model is [WaveCut/Ternary-Bonsai-2-27B-NInfer-v3](https://huggingface.co/WaveCut/Ternary-Bonsai-2-27B-NInfer-v3).
It supports ternary weights, MTP/DFlash2 speculative decoding, paged KV, prefix reuse, Vision,
and YaRN-style position scaling for extended contexts.

The current recommendation is **INT8 KV + Fixed K7 by default for native contexts; MTP3 + YaRN with `rk4v4-e8` as the preferred option for extended contexts**.
The full Fixed K7 concurrency matrix uses INT8 KV, native RoPE, and Vision disabled; the focused report below separately compares other KV formats at C1.
Auto Selected has passed the applicable correctness checks, but has not demonstrated a repeatable gain of at least 2%
over the best fixed policy, so it remains opt-in.

This document summarizes capabilities and existing measurements as of **2026-10-05**.
Historical tests describe the versions on which they ran; this documentation rewrite did not rerun them.

## Capabilities and execution

| Capability | Current scope |
|---|---|
| Models | `Qwen3_5ForCausalLM` and `Qwen3_5MoeForCausalLM`; Qwen3.6/3.8 and Bonsai artifacts with the same architecture use the same execution path |
| Artifacts | v3 `.ninfer`; converters choose representations and the loader binds the actual weights |
| Speculative decoding | MTP, DFlash, DFlash2; corresponding weights and resources must be present in the artifact |
| Input | Text, images, video; multimodal input requires Vision |
| Serving | OpenAI Chat Completions, Responses Core, Anthropic Messages, streaming output, and tool-call representation |
| Concurrency | 1–8 active requests fixed at startup, bounded FIFO ingress, one compact decode batch per round |
| Caching | Shared paged KV, compatible-prefix reuse, device/host checkpoints, and continuations |
| Offline scoring | `ninfer-perplexity` computes perplexity through the public Engine CausalScoring path |

The primary execution model uses one GPU and one resident model. Generation and offline scoring use the native public Engine;
there is no Python model-inference route. Clients execute tool calls. Active requests are not preempted,
and maximum concurrency and capacities are fixed at startup.
See [Engine architecture](docs/maintainer/engine-architecture.md) and [serving protocols](docs/serving.md).

<a id="quick-start"></a>

## Build and quick start

### Build

Requirements are CMake 3.28+, CUDA 12.8+, a C++20 compiler, and an NVIDIA driver.
Linux builds also require Ninja, pkg-config, FFmpeg development libraries
(libavformat/libavcodec 60+, libavutil 58+, libswscale 7+), and libcurl 7.85+.
The repository includes the remaining pinned header dependencies. Windows source builds use VS 2022 and vcpkg.
The latest full build was verified on Linux, GCC 13, CUDA 12.8.61, Release, `sm_89`; it does not establish a current Windows qualification.

In a source directory that **has not been configured**, run:

```bash
cmake --preset release -DCMAKE_CUDA_ARCHITECTURES=89
cmake --build build -j
```

If `build/` is already configured correctly, run only the build command. Applications are in `build/apps/`:

- `ninfer`: command-line generation;
- `ninfer-serve`: HTTP serving;
- `ninfer-perplexity`: offline perplexity evaluation.

For tests and benchmarks, use a new development configuration:

```bash
cmake --preset dev -DCMAKE_CUDA_ARCHITECTURES=89 \
  -DPython3_EXECUTABLE=/absolute/path/to/python3.11
cmake --build build -j
```

Both `release` and `dev` use `build/`; switching presets changes the build options.
Python 3.11 is used for testing, conversion, and evaluation tools, not as a replacement for the C++/CUDA generation engine.

### Prepare the model

Obtain the `.ninfer` file from the [Bonsai 2 v3 artifact page](https://huggingface.co/WaveCut/Ternary-Bonsai-2-27B-NInfer-v3).
The commands below assume that it is in the repository root:

```text
Ternary-Bonsai-2-27B-ninfer-v3.ninfer
```

Select an explicit model path rather than relying on directory order.
Vision, MTP, and DFlash2 require their components in the artifact.
See [weight conversion](docs/weight-conversion.md) for custom models and mixed representations.

### Command-line generation

```bash
./build/apps/ninfer ./Ternary-Bonsai-2-27B-ninfer-v3.ninfer \
  --prompt 'Explain speculative decoding briefly.' --device 0 \
  --max-context 32768 --kv-capacity 32768 --prefill-chunk 1024 --kv-dtype int8 \
  --spec dflash2 --draft-tokens 7 --spec-router fixed \
  --greedy --presence-penalty 0 --frequency-penalty 0 --no-thinking --max-new 512
```

Answer content goes to stdout; reasoning and diagnostics go to stderr.
`--greedy` overrides temperature only; set both penalties explicitly when zero penalties are required.
Use `./build/apps/ninfer --help` for the exact options and [CLI usage](docs/cli.md) for input formats.

<a id="enable-fixed-k7-or-auto-selected"></a>

## Fixed K7 and Auto Selected

### Fixed K7 serving

This INT8 KV configuration corresponds to the E8 startup settings in the resident/Auto measurements
and supports up to eight active requests:

```bash
./build/apps/ninfer-serve ./Ternary-Bonsai-2-27B-ninfer-v3.ninfer \
  --model-id bonsai2-27b --host 127.0.0.1 --port 8080 --device 0 \
  --max-context 32768 --kv-capacity 131072 --max-concurrency 8 \
  --prefill-chunk 1024 --kv-dtype int8 \
  --spec dflash2 --draft-tokens 7 --spec-router fixed \
  --device-state-slots 8 --host-state-slots 8 --host-kv-mib 8192 \
  --max-private-continuations 16 --max-shared-prefixes 8 \
  --max-long-anchors-per-continuation 2 --max-cache-markers-per-request 4 \
  --greedy --presence-penalty 0 --frequency-penalty 0 \
  --no-thinking --default-max-tokens 512
```

The service is available at `http://127.0.0.1:8080/v1`. For example:

```bash
curl http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"bonsai2-27b","messages":[{"role":"user","content":"Hello"}],"max_tokens":128}'
```

`--max-context` is the per-request limit; `--kv-capacity` is the shared pool capacity, not an equal allocation per lane.
C8 means maximum active requests or the client concurrency specified by a measurement;
actual round batches can still be 1–8. Waiting requests can enter at a safe boundary after an active request finishes.

### Optional Auto Selected

Auto requires a calibration profile matching the actual artifact, GPU, and startup configuration.
Schema 3 defaults to K7, with K0/K11 overrides only for qualified cells.
K15 remains the maximum resident drafter and an independent comparator, not an Auto override.
The 2026-10-05 E8 full/selected profiles contain no overrides; they do not demonstrate a concurrent policy faster than Fixed K7.

Generate a new profile with an existing development build. This command uses two idle GPUs to run separate single-GPU jobs:

```bash
python3.11 -m tools.bench.run_dflash_gpu_campaign \
  --model /absolute/path/to/Ternary-Bonsai-2-27B-ninfer-v3.ninfer --build-dir build \
  --out profiles/bench/dflash-gpu-campaign --gpus 0,1 \
  --concurrency 1,2,4,8 --pairs 2 --repeats 2 \
  --max-tokens 512 --max-context 32768 --kv-capacity 131072
```

Replace the speculative decoding options in the serving command above with:

```bash
--spec dflash2 --draft-tokens 15 --spec-router calibrated \
--spec-router-profile profiles/bench/dflash-gpu-campaign/auto-E8-selected.json
```

Keep the remaining startup settings consistent with profile generation.
Selected compute is enabled by `"proposal_compute": "selected"` in the profile;
there is no production CLI option named `--proposal-compute selected`.
Changes to the artifact, KV format, cache capacities, concurrency, or execution settings require a matching new profile.
The serving profile above cannot be loaded directly by the single-command CLI, which disables context caching.
See [calibrated routing](docs/maintainer/speculative-routing.md#calibrated-dflash2-chain).

<a id="choosing-a-kv-format"></a>

## KV, Vision, and extended contexts

Generation accepts `bf16`, `int8`, `fp8`, `rk8v4`, `rk4v4`, `rk4v4-e8`, `rk2v4-e8`, `nvfp4`, and `k8v4`.
These select KV storage; they do not imply that RTX 4090 can execute Blackwell-only weight/activation kernels.
On sm89 builds, Engine, CLI, Serving, and inference benchmarks consistently default to INT8 when the KV option is omitted; other architectures retain their existing defaults.
native-FP8, RK8V4, and RK4V4-E8 remain explicit choices. See the [Bonsai four-format report](docs/performance/bonsai-kv-4090-2026-10-05.md) for the decision.
The choices below distinguish measured recommendations from the current extended-context product design:

| Goal | Configuration | Evidence scope |
|---|---|---|
| Measured native-context throughput | INT8 KV + Fixed K7 | DFlash2 real-model matrices below |
| Long input with decode dominating | `fp8` (sm89 native-FP8) | At a synthetic 29K input: decode +16.5%, prefill −30.8%; not a general throughput advantage |
| Reduce native-context KV storage | `rk8v4` or `rk4v4-e8` | At equal capacity, payload is about 24.2% / 48.5% smaller than INT8; see the four-format report for quality and C1 throughput |
| Beyond the native 262144 tokens | MTP3 + YaRN + `rk4v4-e8` | Supported path; INT8 matrices do not replace full capacity and quality qualification at 512K/658176 |
| Image/video input | `--vision --vision-residency overlay` | Requires Vision components; the latest DFlash2 performance matrices disable Vision |

Vision overlay keeps the tower in host memory and borrows device memory per image;
`--vision-max-merged` bounds merged tokens per media item.
`--kv-capacity auto` sizes the shared cache after weights and runtime reservations,
leaving 1024 MiB of sizing headroom; it does not guarantee that every configuration fits in VRAM.

### YaRN extended-context example

This is a **512K acceptance-target configuration**, not a measured throughput or quality guarantee:

```bash
./build/apps/ninfer-serve ./Ternary-Bonsai-2-27B-ninfer-v3.ninfer \
  --model-id bonsai2-27b --device 0 --max-concurrency 1 \
  --max-context 524288 --kv-capacity auto --kv-dtype rk4v4-e8 \
  --spec mtp --draft-tokens 3 --lm-head-draft --gdn-state-fp16 \
  --rope-scaling-factor 2.12 --rope-scaling-original-context 262144 \
  --vision --vision-residency overlay --vision-max-merged 12288
```

Position `p` is unchanged up to the native threshold `N`; beyond it, the mapping is `N + round((p-N)/factor)`.
Only RoPE positions are scaled; physical KV addresses remain absolute.
**Enabling DFlash/DFlash2 together with YaRN fails at startup**; use MTP for extended contexts.
The causal-attention execution envelope is 786432 tokens, with initial acceptance targets of 512K and 658176.
An execution envelope is not a model-quality or VRAM-capacity guarantee.
See [position scaling and compressed KV design](docs/maintainer/rtx4090-ternary-yarn-port.md).

## RTX 4090 measurements

The following tables use different configurations and test versions.
Their best numbers cannot be combined into a claim about one configuration.
Throughput sums generated tokens across concurrent requests;
end-to-end, decode, and individual kernel timings retain their respective measurement scopes.

### Bonsai 2 fixed-route matrix (2026-10-04)

Configuration: Bonsai 2 v3, INT8 KV, greedy, zero penalties, cache and CUDA Graphs enabled,
512-token output budget, context/KV capacity 32768, Engine concurrency 8.
The campaign covers seven workloads, C1/C2/C4/C8, K7/K11/K15, with two AB/BA pairs per comparison.
All 84 comparisons match token IDs, complete responses, and repeated output; 71 meet the performance qualification gates.

| Client concurrency | Target-only | Fixed K7 | Fixed K11 | Fixed K15 |
|---|---:|---:|---:|---:|
| C1 | 96.9 | 254.0 | 248.4 | 255.0 |
| C2 | 183.2 | 448.5 | 367.7 | 339.2 |
| C4 | 338.1 | 596.1 | 448.7 | 417.5 |
| C8 | 573.1 | 683.9 | 546.5 | 456.0 |

Units: aggregate end-to-end tokens/s. These are descriptive medians across workloads, including comparisons that failed performance gates.
K7 qualifies on 7/7, 7/7, 7/7, and 6/7 workloads at C1/C2/C4/C8 respectively;
long-context C8 K7 is approximately 4% slower than target-only.
The K7 recommendation is therefore not a guarantee of improvement on every workload.
See the [complete matrix and qualification rules](docs/performance.md#rtx-4090-sm_89-chain-qualification).

### Resident and Auto (2026-10-04 to 2026-10-05)

Resident measurements use Bonsai 2, INT8 KV, A8 prefill/chunk 1024, native RoPE,
greedy zero-penalty sampling, cache and Graphs enabled, and maximum context 32768.
KV capacity is 32768 at E1 and 131072 at E8.
Each GPU uploads the model once; each job uses independent mutable state.
Short/medium/long validation prompts contain 54/4554/11706 tokens; all 720 validation requests generate 512 tokens.

The highest aggregate rate in this matrix is **816.7 tokens/s**, at C8 with a medium prompt and a later reuse wave.
For one request reusing the medium prompt, Fixed K11 reaches **347.2** versus K7's **329.8 tokens/s**.
These are workload-specific results, not minimum performance guarantees.
See the [resident report and resource accounting](docs/performance/dflash2-4090-2026-10-04.md).

The schema-3 follow-up reports geometric-mean speedups over backend None:

| Scope | Fixed K7 | Auto Full | Auto Selected |
|---|---:|---:|---:|
| Single, cold | 2.094× | 1.995× | 2.103× |
| Single, warm | 2.731× | 2.588× | 2.771× |
| Concurrent, cold | 1.433× | 1.287× | 1.419× |
| Concurrent, warm | 1.740× | 1.492× | 1.728× |

Cold means the first unprimed wave; concurrent requests within it can still reuse a prefix.
Warm means a later wave with observed natural reuse.
Auto Selected approaches K7, but no held-out condition clears the gates:
at least 2% improvement over the best fixed policy, faster in every retained pair, and at most 5% p95 regression.
This follow-up reused the preceding correctness evidence and did not rerun full regressions or CI.
See the [Auto K7 follow-up](docs/performance/dflash2-auto-k7-4090-2026-10-05.md).

### T64 SwiGLU route optimization (PR #19, 2026-10-05)

Only `sm_89` DFlash2 T64 changes from `R64/C64/BK128` to the existing `R32/C64/BK128`;
other widths and paths remain unchanged.
RTX 4090 cold-cache measurements use 5 warmups and 31 measured runs:

| Metric | Previous R64 | New R32 | Change |
|---|---:|---:|---:|
| Median individual SwiGLU kernel time | 337.1 μs | 298.0 μs | Approximately 11.6% less time |
| Fixed K7 / C8 end-to-end throughput | 639.69 tokens/s | 644.46 tokens/s | Approximately +0.75% |

The end-to-end test uses the same Bonsai artifact, INT8 KV, context 8192, KV capacity 32768,
512-token output, and two repeats.
The baseline reuses the same build with the pre-merge route object;
all 8192 output token IDs match, and 339 rounds cover actual batches 1–8.
This is one short A/B comparison, establishing a directional result rather than a stable statistically significant gain.
See [PR #19](https://github.com/zhongpei/ninfer-4090/pull/19).
Local evidence is under `profiles/bench/pr19-t64-2026-10-05/`, including numerical logs,
kernel measurements, A/B JSON reports, and `summary.json`.

### Bonsai 27B four-format KV comparison (2026-10-05)

After the native-FP8 fixes in PR #24, **25/25** numerical cases pass against an independent FP64 attention oracle, and the scoring comparison tool has **9 passing tests**. Error limits were not relaxed.
Quality tests materialize the complete KV history before scoring the next 2048 tokens: four domains at 8K/32K and a concatenated stream at depths 64K/128K/192K/258048.
See the [four-format test report](docs/performance/bonsai-kv-4090-2026-10-05.md) for methodology, per-depth PPL, memory, and limitations.

The table gives medians of three repetitions with an actual 29141-token prompt, C1 Fixed K7, and 256 output tokens.
The task requests consecutive integers and achieves 100% draft acceptance; this synthetic workload does not establish ordinary conversation throughput.

| KV | Prefill (k tok/s) | Decode (tok/s) | 32768-token KV payload |
|---|---:|---:|---:|
| INT8 | 2.60 | 415.7 | 1056 MiB |
| native-FP8 | 1.80 | 484.1 | 1032 MiB |
| RK8V4 | 2.56 | 404.7 | 800 MiB |
| RK4V4-E8 | 2.56 | 405.5 | 544 MiB |

native-FP8 improves decode by about 16.5% at the longer input but reduces prefill by about 30.8%; at an actual 6847-token prompt, decode is about 3.6% slower.
Four-domain PPL changes versus INT8 at 8K/32K are +0.018%/+0.051% for native-FP8, +0.027%/+0.060% for RK8V4, and +0.411%/+0.247% for RK4V4-E8.

### Early Qwen3.8 INT8 compatibility results (2026-08-15)

This is a **historical Qwen3.8-27B INT8 artifact baseline**, not a Bonsai or current-build measurement.
Configuration: RTX 4090, CUDA 12.8.93, INT8 KV, MTP3 + LM-head draft, greedy,
prefix reuse disabled, 1024 output tokens per request.

| Concurrency | End-to-end tokens/s | Decode tokens/s | Mean TTFT | Peak VRAM |
|---|---:|---:|---:|---:|
| C1 | 102.13 | 103.35 | 112 ms | 18250 MiB |
| C2 | 162.46 | 165.74 | 160 ms | 18562 MiB |
| C4 | 193.49 | 198.76 | 295 ms | 19184 MiB |
| C8 | 299.82 | 315.09 | 644 ms | 20708 MiB |

C1–C4 use an 8K KV pool; C8 uses 16K. Every request completes its specified output length.
See the [early qualification record](docs/rtx-4090-early.md).

<a id="evaluation"></a>

## Correctness, quality, and verification status

| Test record | Passing scope | Limitations |
|---|---|---|
| PR #19 full build | `cmake --build build -j`, 512 build steps completed, exit 0 | Linux / CUDA 12.8.61 / Release / sm_89; CI not queried |
| PR #19 T64 numerics | Q8 A16 DFlash2 `[34816,5120] → [17408]`; eager and CUDA Graph execution, workspace/guard/input-read-only checks pass | Independent FP64 oracle; relative-L2 0.002881/0.003077 below 0.0033; T64 only |
| Fixed-route campaign | 84/84 correctness comparisons pass, actual batches 1–8 | Specified Bonsai/INT8/native-context matrix |
| Resident real-model suite | 18/18 cases pass, 9 each for full/selected | Version executed on 2026-10-04 |
| Resident ownership suite | 5/5, including exclusive execution, state isolation, and teardown checks | Does not establish shared mutable state across Engines |
| Held-out validation | 720 requests match None in token IDs, complete responses, and repeated output | Two measurement orders; p95 is descriptive with small samples |
| CPU / Python regressions | Report suite 69 passed; benchmark suite 106 passed plus 2 supplementary native CLI checks passed; native CPU 3/3 | Supplement removed the original 2 skips; not a repository-wide rerun in this documentation task |
| GDN state repair | Endpoint continuation, context-boundary switching, membership changes, stop/cancel, zero-commit; native C4/C8 token/response equality | Existing integration reports |
| A16 Conv repair | B2/T1 FP64 relative-L2 improves from 0.003274 to 0.001694, below the original 0.00315 limit | T1/T1024 whole-Op approximately 14% slower; the repair has a cost |
| T2 A16 repair | FP64 and width/segmentation checks pass; 6144 output token IDs match | Public Linear T1/T16/T50 costs change by approximately +21.6%/+2.1%/−15.8%; not end-to-end gains |
| Early Engine integration (2026-10-02) | Five separate real-model tests pass: loading, prefix, CausalScore, Vision workspace, DFlash2 | Vision workspace is not an image-request or quality qualification; tests requiring matching MoE/DFlash artifacts remain unverified |

Correctness and performance are separate conclusions.
See [4090 qualification](docs/performance.md#rtx-4090-sm_89-chain-qualification) for existing state-repair and A16 evidence,
and the [resident checks](docs/performance/dflash2-4090-2026-10-04.md#correctness-and-execution-checks)
for CPU, real-model, and ownership details.
PR #19 tests only the new T64 route; old paths were compared in source, without old-function tests or the full runtime suite.

The 2026-10-02 stage record also retains combined initial CTest and failed-test retry evidence:
146 passed / 7 skipped, Python 87 passed, and an overall A/B output-gate failure.
This was not a fresh run passing all 153 tests.
Later fixed/resident matrices re-established the exact-output evidence listed above after state and numerical repairs.
These historical records remain locally in `profiles/bench/final-ab-2026-10-02/final-summary.json`;
they are not a current-version full-suite pass claim.

**Not qualified by the data here:** Bonsai quality/performance with `rk2v4-e8`, C8 throughput and task accuracy across the four KV formats,
full capacity and quality at YaRN 512K/658176, the latest DFlash2 matrix with Vision enabled,
and other hardware or the current Windows version.
Perplexity, accuracy, and VRAM numbers from other models or historical hardware do not substitute for these results.

To measure quality, use the fixed corpus:

```bash
./build/apps/ninfer-perplexity ./Ternary-Bonsai-2-27B-ninfer-v3.ninfer \
  --corpus eval/corpora/perplexity-1m/manifest.json --quick --kv-dtype int8
```

Keep the artifact, corpus, context, stride, and execution settings fixed, changing only the variable under comparison.
Use the scoring application's own `--help` for its KV choices; do not assume they exactly match generation options.
See [perplexity methodology](docs/perplexity.md) and [capability evaluation](eval/README.md).

## Tests, measurements, and documentation

Development builds provide the following test entry points.
These are reproduction commands, not a claim that this documentation task ran the full suite:

```bash
ctest --test-dir build --output-on-failure
cmake --build build -j --target ninfer_linear_swiglu_q8_a16_test
./build/tests/ninfer_linear_swiglu_q8_a16_test
```

The standard SwiGLU test above covers multiple shapes and widths;
PR #19 qualification used a temporary T64-only entry point.
Numerical Ops use independent FP32/FP64 oracles, while exact codecs use exact oracles.
Kernel benchmarks establish performance only at their Op scope.
For end-to-end measurements, use a public Engine benchmark or protocol requests,
recording actual batches, configuration, and comparison order.

| Document | Contents |
|---|---|
| [Documentation map](docs/README.md) | User guides and maintainer references |
| [CLI](docs/cli.md) / [HTTP serving](docs/serving.md) | Input, options, streaming, protocols, continuations |
| [Tests](tests/README.md) / [Benchmarks](bench/README.md) | Build, test, and measurement entry points and metrics |
| [4090 performance record](docs/performance.md#rtx-4090-sm_89-chain-qualification) | Qualification gates, throughput, repairs, limitations |
| [Weight conversion](docs/weight-conversion.md) | Artifacts, recipes, representations, optional components |
| [Op development](docs/maintainer/op-development.md) | Mathematical semantics, independent oracles, performance qualification |

The current executable `--help` is the direct reference for option names and availability.
Tests cover only their stated versions and scopes; checks that were not run or queried are not treated as passes.

## Origins, contributions, and license

This project builds on the engine from [Neroued/ninfer](https://github.com/Neroued/ninfer) and the subsequent Ternary execution line,
combining the [YaRN port](https://github.com/alanthinker/ninfer-4090-yarn)
and [4090 compressed KV implementation](https://github.com/sergiuszm/ninfer-4090).
Upstream schedules targeting RTX 5090 / `sm_120a` require measurement on Ada before adoption.

See [CONTRIBUTORS.md](CONTRIBUTORS.md) for credits and [PR_POLICY.md](PR_POLICY.md) for contribution rules.
The license is **GNU AGPL-3.0-only**; see [LICENSE](LICENSE).
