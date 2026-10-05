# NInfer-4090: Ternary Bonsai 2 and YaRN

[中文（默认）](README.md) · **English**

NInfer-4090 is a local C++/CUDA inference engine targeting **NVIDIA GeForce RTX 4090 (24 GB, `sm_89`)**.
Its primary model is [WaveCut/Ternary-Bonsai-2-27B-NInfer-v3](https://huggingface.co/WaveCut/Ternary-Bonsai-2-27B-NInfer-v3).
It supports ternary weights, MTP/DFlash2 speculative decoding, paged KV, prefix reuse, Vision,
and YaRN-style position scaling for extended contexts.

Recommended configurations: **FP8 KV + DFlash2 Fixed K7 for single-request long-input decode; INT8 KV + Fixed K7 for general chat and concurrent serving; MTP3 + YaRN + `rk4v4-e8` for extended contexts.**
Performance depends on input length, output length, concurrency, and prefix reuse. Measurement scopes are described below.

## Capabilities and execution

| Capability | Current scope |
|---|---|
| Models | `Qwen3_5ForCausalLM` and `Qwen3_5MoeForCausalLM`; Qwen3.6/3.8 and Bonsai artifacts with the same architecture use the same execution path |
| Artifacts | `.ninfer` model files |
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

In a source directory that **has not been configured**, run:

```bash
cmake --preset release -DCMAKE_CUDA_ARCHITECTURES=89
cmake --build build -j
```

If `build/` is already configured correctly, run only the build command. Applications are in `build/apps/`:

- `ninfer`: command-line generation;
- `ninfer-serve`: HTTP serving;
- `ninfer-perplexity`: offline perplexity evaluation.

See [build instructions](docs/maintainer/build-system.md) for development, test, and benchmark configurations.

### Prepare the model

Obtain the `.ninfer` file from the [Bonsai model page](https://huggingface.co/WaveCut/Ternary-Bonsai-2-27B-NInfer-v3).
The commands below assume that it is in the repository root:

```text
Ternary-Bonsai-2-27B-ninfer-v3.ninfer
```

Select an explicit model path rather than relying on directory order.
Vision, MTP, and DFlash2 require their components in the artifact.
See [weight conversion](docs/weight-conversion.md) for custom models and mixed representations.

### Single-request long-input decode

This configuration uses FP8 KV and native RoPE, with Vision disabled:

```bash
./build/apps/ninfer ./Ternary-Bonsai-2-27B-ninfer-v3.ninfer \
  --messages /absolute/path/to/messages.json --device 0 \
  --max-context 32768 --kv-capacity 32768 --prefill-chunk 1024 --kv-dtype fp8 \
  --spec dflash2 --draft-tokens 7 --spec-router fixed \
  --greedy --presence-penalty 0 --frequency-penalty 0 --no-thinking --max-new 256
```

Provide your conversation in `messages.json`; see [CLI usage](docs/cli.md) for its format. This configuration suits long inputs when single-request decode speed is the main goal. Prefill remains slower than measured INT8, and FP8 lacks a full concurrent performance comparison.
Size context and KV capacity for actual needs; larger capacities increase reserved resources.

### General command-line generation

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

## HTTP serving

### Fixed K7 serving

This INT8 KV configuration supports up to eight active requests:

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
Actual round batches vary with active requests. Waiting requests can enter at a safe boundary after an active request finishes.

### Optional automatic routing

Automatic routing requires a calibration profile matching the model, GPU, KV format, capacities, and concurrency. It has not demonstrated a stable gain over the recommended Fixed K7, so fixed routing remains the default recommendation.
To use automatic routing, replace the speculative decoding options in the serving command:

```bash
--spec dflash2 --draft-tokens 15 --spec-router calibrated \
--spec-router-profile /absolute/path/to/auto-selected.json
```

See [calibrated routing](docs/maintainer/speculative-routing.md#calibrated-dflash2-chain) for profile generation and use. A serving profile cannot be loaded directly by the single-command CLI with context caching disabled.

<a id="choosing-a-kv-format"></a>

## KV, Vision, and extended contexts

Generation accepts `bf16`, `int8`, `fp8`, `rk8v4`, `rk4v4`, `rk4v4-e8`, `rk2v4-e8`, `nvfp4`, and `k8v4`.
These select KV storage; they do not imply that RTX 4090 can execute Blackwell-only weight/activation kernels.
On sm89 builds, Engine, CLI, Serving, and inference benchmarks consistently default to INT8 when the KV option is omitted; other architectures retain their existing defaults.
native-FP8, RK8V4, and RK4V4-E8 remain explicit choices. See the [Bonsai four-format report](docs/performance/bonsai-kv-4090-2026-10-05.md) for the decision.
The choices below distinguish measured recommendations from the current extended-context product design:

| Goal | Configuration | Evidence scope |
|---|---|---|
| General chat and concurrent serving | INT8 KV + Fixed K7 | Real-model measurements across multiple workloads and concurrency 1–8 |
| Single-request long-input decode | `fp8` + DFlash2 Fixed K7 | 513.7 tok/s decode on a synthetic 29K input; prefill and concurrency limits below |
| Reduce native-context KV storage | `rk8v4` or `rk4v4-e8` | At equal capacity, payload is about 24.2% / 48.5% smaller than INT8; see the four-format report for quality and C1 throughput |
| Beyond the native 262144 tokens | MTP3 + YaRN + `rk4v4-e8` | Supported path; Full capacity and quality beyond 512K require verification |
| Image/video input | `--vision --vision-residency overlay` | Requires Vision components; the listed DFlash2 performance measurements disable Vision |

Vision overlay keeps the tower in host memory and borrows device memory per image;
`--vision-max-merged` bounds merged tokens per media item.
`--kv-capacity auto` sizes the shared cache after weights and runtime reservations,
leaving 1024 MiB of sizing headroom; it does not guarantee that every configuration fits in VRAM.

### YaRN extended-context example

This example sets a 512K context; full quality and capacity qualification at that length is not complete:

```bash
./build/apps/ninfer-serve ./Ternary-Bonsai-2-27B-ninfer-v3.ninfer \
  --model-id bonsai2-27b --device 0 --max-concurrency 1 \
  --max-context 524288 --kv-capacity auto --kv-dtype rk4v4-e8 \
  --spec mtp --draft-tokens 3 --lm-head-draft --gdn-state-fp16 \
  --rope-scaling-factor 2.12 --rope-scaling-original-context 262144 \
  --vision --vision-residency overlay --vision-max-merged 12288
```

**Enabling DFlash/DFlash2 together with YaRN fails at startup**; use MTP for extended contexts.
The causal-attention execution limit is 786432 tokens; usable length depends on VRAM and model quality.
See [position scaling and compressed KV](docs/maintainer/rtx4090-ternary-yarn-port.md).

## RTX 4090 measurements

### Single-request FP8 decode

Bonsai 27B, DFlash2 Fixed K7, native RoPE, chunk 1024, greedy zero penalties, Vision disabled, and 256 output tokens. Values are medians of three runs per input.

| Actual input tokens | Context / KV capacity | Prefill (k tok/s) | Decode (tok/s) |
|---:|---:|---:|---:|
| 6847 | 8192 / 8192 | 2.62 | 573.3 |
| 29141 | 32768 / 32768 | 2.16 | 513.7 |

The input requests consecutive integers and achieves 100% draft acceptance; these rates do not describe ordinary chat. Decode speed also does not establish total request time. See the [FP8 performance report](docs/performance/bonsai-fp8-pr25-diagnosis.md) for numerics, per-depth PPL, and methodology.

### INT8 concurrent serving

Bonsai 27B, INT8 KV, caching and CUDA Graphs enabled, context/KV capacity 32768, up to eight active requests, and a 512-token output budget. Values are median aggregate end-to-end throughput across seven workloads, in tokens/s.

| Client concurrency | Without speculation | DFlash2 Fixed K7 |
|---:|---:|---:|
| 1 | 96.9 | 254.0 |
| 2 | 183.2 | 448.5 |
| 4 | 338.1 | 596.1 |
| 8 | 573.1 | 683.9 |

These summarize workloads rather than guarantee minimum performance per request. K7 is about 4% slower on one long-context, concurrency-eight workload, so evaluate fixed routing on actual traffic. Measurements with one request reusing a medium-length input also show K11 ahead of K7. See the [serving performance report](docs/performance.md#rtx-4090-sm_89-chain-qualification) for configurations and scope.

<a id="evaluation"></a>

## Quality evaluation and tests

FP8 passes independent attention numerical checks and completes fixed-history PPL measurements at depths 8192 through 258048. PPL changes are mixed and do not establish lossless quality on all tasks. See the [KV format report](docs/performance/bonsai-kv-4090-2026-10-05.md) for other formats' quality and memory comparisons.
Full quality at a 512K extended context, FP8 concurrent throughput, and speculative decoding performance with Vision enabled require verification for the actual configuration.

Evaluate perplexity:

```bash
./build/apps/ninfer-perplexity ./Ternary-Bonsai-2-27B-ninfer-v3.ninfer \
  --corpus eval/corpora/perplexity-1m/manifest.json --quick --kv-dtype int8
```

Keep the model, corpus, context, and execution settings consistent when comparing results. See [perplexity methodology](docs/perplexity.md) and [capability evaluation](eval/README.md).

After completing a development build, run tests:

```bash
ctest --test-dir build --output-on-failure
```

Test and benchmark entry points and detailed results are linked below.

## Documentation

| Document | Contents |
|---|---|
| [Documentation map](docs/README.md) | User guides and maintainer references |
| [CLI](docs/cli.md) / [HTTP serving](docs/serving.md) | Inputs, options, streaming, protocols, and continuations |
| [Tests](tests/README.md) / [Benchmarks](bench/README.md) | Build, test, and measurement commands |
| [4090 performance](docs/performance.md#rtx-4090-sm_89-chain-qualification) | Throughput, measurement conditions, and limits |
| [Weight conversion](docs/weight-conversion.md) | Artifacts, representations, and optional components |

The corresponding executable's `--help` is the reference for option names and availability.

## Origins, contributions, and license

This project builds on the engine from [Neroued/ninfer](https://github.com/Neroued/ninfer) and the subsequent Ternary execution line,
combining the [YaRN port](https://github.com/alanthinker/ninfer-4090-yarn)
and [4090 compressed KV implementation](https://github.com/sergiuszm/ninfer-4090).
Upstream schedules targeting RTX 5090 / `sm_120a` require measurement on Ada before adoption.

See [CONTRIBUTORS.md](CONTRIBUTORS.md) for credits and [PR_POLICY.md](PR_POLICY.md) for contribution rules.
The license is **GNU AGPL-3.0-only**; see [LICENSE](LICENSE).
