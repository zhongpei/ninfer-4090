# NInfer-4090：Ternary Bonsai 2 与 YaRN

**中文** · [English](README.en.md)

NInfer-4090 是面向 **NVIDIA GeForce RTX 4090（24 GB，`sm_89`）** 的 C++/CUDA 本地推理引擎。
主要模型为 [WaveCut/Ternary-Bonsai-2-27B-NInfer-v3](https://huggingface.co/WaveCut/Ternary-Bonsai-2-27B-NInfer-v3)，
支持三值权重、MTP/DFlash2 推测解码、分页 KV、前缀复用、Vision 和 YaRN 风格的长上下文位置缩放。

推荐配置：**单请求长输入解码用 FP8 KV + DFlash2 Fixed K7；通用对话与并发服务用 INT8 KV + Fixed K7；扩展上下文用 MTP3 + YaRN + `rk4v4-e8`。**
性能取决于输入长度、输出长度、并发和前缀复用；各配置的测量范围见下文。

## 能力与执行方式

| 能力 | 当前范围 |
|---|---|
| 模型 | `Qwen3_5ForCausalLM`、`Qwen3_5MoeForCausalLM`；同架构的 Qwen3.6/3.8 与 Bonsai 工件沿用相同执行路径 |
| 工件 | `.ninfer` 模型文件 |
| 推测解码 | MTP、DFlash、DFlash2；需要工件内对应的权重和资源 |
| 输入 | 文本、图像、视频；多模态输入需要启用 Vision |
| 服务 | OpenAI Chat Completions、Responses Core、Anthropic Messages，支持流式输出和工具调用表示 |
| 并发 | 启动时固定 1–8 个活动请求，有界 FIFO 等待队列，每轮组成紧凑 decode batch |
| 缓存 | 共享分页 KV、兼容前缀复用、设备/主机 checkpoint 与 continuation |
| 离线评分 | `ninfer-perplexity` 通过公共 Engine 的 CausalScoring 路径计算困惑度 |

主要运行方式是一张 GPU、一个常驻模型。生成与离线评分都走原生公共 Engine，没有 Python 模型推理路线。
工具调用交给客户端执行。活动请求不被抢占，最大并发和容量在启动时确定。
详见 [Engine 架构](docs/maintainer/engine-architecture.md) 和 [服务协议](docs/serving.md)。

<a id="quick-start"></a>

## 构建与快速启动

### 构建

需要 CMake 3.28+、CUDA 12.8+、C++20 编译器和 NVIDIA 驱动。
Linux 构建还需要 Ninja、pkg-config、FFmpeg 开发库（libavformat/libavcodec 60+、libavutil 58+、libswscale 7+）及 libcurl 7.85+。
仓库包含其余固定版本的头文件依赖。Windows 源码构建使用 VS 2022 与 vcpkg。

在**尚未配置的源码目录**执行：

```bash
cmake --preset release -DCMAKE_CUDA_ARCHITECTURES=89
cmake --build build -j
```

已有正确配置的 `build/` 直接执行构建即可。应用位于 `build/apps/`：

- `ninfer`：命令行生成；
- `ninfer-serve`：HTTP 服务；
- `ninfer-perplexity`：离线困惑度评估。

开发构建、测试和 benchmark 配置见 [构建说明](docs/maintainer/build-system.md)。

### 准备模型

从 [Bonsai 模型页面](https://huggingface.co/WaveCut/Ternary-Bonsai-2-27B-NInfer-v3)
获取 `.ninfer` 文件。下面的命令假设它位于仓库根目录：

```text
Ternary-Bonsai-2-27B-ninfer-v3.ninfer
```

使用明确的模型路径，不依赖目录中的文件顺序。Vision、MTP、DFlash2 是否可启用取决于工件包含的组件。
自定义或混合表示的模型见 [权重转换](docs/weight-conversion.md)。

### 单请求长输入解码

以下配置使用 FP8 KV、原生 RoPE，不启用 Vision：

```bash
./build/apps/ninfer ./Ternary-Bonsai-2-27B-ninfer-v3.ninfer \
  --messages /absolute/path/to/messages.json --device 0 \
  --max-context 32768 --kv-capacity 32768 --prefill-chunk 1024 --kv-dtype fp8 \
  --spec dflash2 --draft-tokens 7 --spec-router fixed \
  --greedy --presence-penalty 0 --frequency-penalty 0 --no-thinking --max-new 256
```

`messages.json` 是用户提供的对话输入，格式见 [CLI 使用说明](docs/cli.md)。该配置适合长输入、以解码速度为主要目标的单请求场景；prefill 仍慢于已测 INT8，尚无 FP8 的完整并发性能比较。
上下文和 KV 容量按实际需求设置，较大容量会增加预留资源。

### 通用命令行生成

```bash
./build/apps/ninfer ./Ternary-Bonsai-2-27B-ninfer-v3.ninfer \
  --prompt 'Explain speculative decoding briefly.' --device 0 \
  --max-context 32768 --kv-capacity 32768 --prefill-chunk 1024 --kv-dtype int8 \
  --spec dflash2 --draft-tokens 7 --spec-router fixed \
  --greedy --presence-penalty 0 --frequency-penalty 0 --no-thinking --max-new 512
```

回答写入 stdout，推理文本和诊断写入 stderr。`--greedy` 只覆盖 temperature；需要零惩罚时显式设置两个 penalty。
完整参数以 `./build/apps/ninfer --help` 为准，输入格式见 [CLI 使用说明](docs/cli.md)。

<a id="enable-fixed-k7-or-auto-selected"></a>

## HTTP 服务

### Fixed K7 服务

下面的 INT8 KV 配置可同时激活最多八个请求：

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

服务地址为 `http://127.0.0.1:8080/v1`。例如：

```bash
curl http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"bonsai2-27b","messages":[{"role":"user","content":"Hello"}],"max_tokens":128}'
```

`--max-context` 是每个请求的上限，`--kv-capacity` 是共享池容量，不按 lane 平均分配。
实际每轮 batch 随活动请求数量变化。请求完成后，等待请求可在安全边界进入执行。

### 可选自动路由

自动路由需要与模型、GPU、KV、容量和并发设置匹配的校准文件。目前没有证明它稳定快于推荐的 Fixed K7，默认建议使用固定策略。
需要自动路由时，在服务命令中替换推测解码参数：

```bash
--spec dflash2 --draft-tokens 15 --spec-router calibrated \
--spec-router-profile /absolute/path/to/auto-selected.json
```

校准文件的生成与使用见 [校准路由说明](docs/maintainer/speculative-routing.md#calibrated-dflash2-chain)。服务的校准文件不能直接用于关闭 context cache 的单命令 CLI。

<a id="choosing-a-kv-format"></a>

## KV、Vision 与长上下文

生成程序接受 `bf16`、`int8`、`fp8`、`rk8v4`、`rk4v4`、`rk4v4-e8`、`rk2v4-e8`、`nvfp4`、`k8v4`。
这是 KV 存储选择，不表示 RTX 4090 可执行 Blackwell 专用的权重/激活 kernel。
sm89 构建的 Engine、CLI、Serving 和 inference benchmark 在省略 KV 选项时统一使用 INT8；其他架构保持原有默认值。
native-FP8、RK8V4 和 RK4V4-E8 均可显式选择。选择依据见 [Bonsai 四格式报告](docs/performance/bonsai-kv-4090-2026-10-05.md)。
下面的选择依据分别是实测推荐与当前长上下文产品设计：

| 使用目标 | 配置 | 证据范围 |
|---|---|---|
| 通用对话与并发服务 | INT8 KV + Fixed K7 | 有多工作负载、1–8并发的真实模型测量 |
| 单请求长输入解码 | `fp8` + DFlash2 Fixed K7 | 29K合成输入decode 513.7 tok/s；prefill与并发限制见下文 |
| 原生上下文节省 KV | `rk8v4` 或 `rk4v4-e8` | 同容量 payload 比 INT8 少约 24.2% / 48.5%；质量与 C1 吞吐见四格式专项报告 |
| 超过原生 262144 tokens | MTP3 + YaRN + `rk4v4-e8` | 支持路径；512K以上的完整容量与质量仍需验证 |
| 图像/视频输入 | `--vision --vision-residency overlay` | 需要 Vision 组件；所列 DFlash2 性能测量关闭 Vision |

Vision overlay 将视觉塔保留在主机内存，处理图像时借用设备内存；`--vision-max-merged` 限制单项媒体的 merged tokens。
`--kv-capacity auto` 在权重和运行时预留之后自动定共享缓存容量，保留 1024 MiB 的 sizing headroom；它不保证任意配置都能装入显存。

### YaRN 长上下文启动示例

以下示例设置512K上下文；该长度的完整质量与容量验证尚未完成：

```bash
./build/apps/ninfer-serve ./Ternary-Bonsai-2-27B-ninfer-v3.ninfer \
  --model-id bonsai2-27b --device 0 --max-concurrency 1 \
  --max-context 524288 --kv-capacity auto --kv-dtype rk4v4-e8 \
  --spec mtp --draft-tokens 3 --lm-head-draft --gdn-state-fp16 \
  --rope-scaling-factor 2.12 --rope-scaling-original-context 262144 \
  --vision --vision-residency overlay --vision-max-merged 12288
```

**DFlash/DFlash2 与 YaRN 同时启用会在启动时报错**；扩展上下文使用 MTP。
因果注意力执行上限为786432 tokens；实际可用长度受显存和模型质量限制。
详见 [位置缩放与压缩 KV 说明](docs/maintainer/rtx4090-ternary-yarn-port.md)。

## RTX 4090 实测性能

### 单请求 FP8 解码

Bonsai 27B、DFlash2 Fixed K7、原生RoPE、chunk 1024、greedy零惩罚、关闭Vision，输出256 tokens。每个输入测量三次，表中为中位数。

| 实际输入 tokens | Context / KV容量 | Prefill（k tok/s） | Decode（tok/s） |
|---:|---:|---:|---:|
| 6847 | 8192 / 8192 | 2.62 | 573.3 |
| 29141 | 32768 / 32768 | 2.16 | 513.7 |

输入要求连续输出整数，draft接受率为100%；这些数字不代表普通对话吞吐。解码速度也不等于请求总耗时。数值、逐深度PPL和测量方法见 [FP8性能报告](docs/performance/bonsai-fp8-pr25-diagnosis.md)。

### INT8 并发服务

Bonsai 27B、INT8 KV、缓存与CUDA Graph开启，context/KV容量32768、最大活动请求数8，输出预算512 tokens。以下为七种工作负载的端到端总吞吐中位数，单位tokens/s。

| 客户端并发 | 无推测解码 | DFlash2 Fixed K7 |
|---:|---:|---:|
| 1 | 96.9 | 254.0 |
| 2 | 183.2 | 448.5 |
| 4 | 338.1 | 596.1 |
| 8 | 573.1 | 683.9 |

这些数值是跨工作负载的汇总，不是每个请求的最低性能。长上下文、8并发的一个工作负载中K7慢约4%，因此固定策略仍须按实际业务评估。单请求中等输入复用的测量还显示K11快于K7；详细配置与适用范围见 [服务性能报告](docs/performance.md#rtx-4090-sm_89-chain-qualification)。

<a id="evaluation"></a>

## 质量评估与测试

FP8通过独立attention数值检查，并完成8192至258048深度的固定历史PPL测量；PPL变化有正有负，不能视为所有任务无损。其他KV格式的质量与内存比较见 [KV格式报告](docs/performance/bonsai-kv-4090-2026-10-05.md)。
512K扩展上下文的完整质量、FP8并发吞吐、启用Vision后的推测解码性能，仍需针对实际配置验证。

评估困惑度：

```bash
./build/apps/ninfer-perplexity ./Ternary-Bonsai-2-27B-ninfer-v3.ninfer \
  --corpus eval/corpora/perplexity-1m/manifest.json --quick --kv-dtype int8
```

比较时保持模型、语料、上下文和执行设置一致。方法见 [困惑度评估](docs/perplexity.md) 与 [能力评估](eval/README.md)。

开发构建完成后运行测试：

```bash
ctest --test-dir build --output-on-failure
```

测试、benchmark入口及详细结果见下列文档。

## 文档

| 文档 | 内容 |
|---|---|
| [文档导航](docs/README.md) | 用户指南与维护者参考 |
| [CLI](docs/cli.md) / [HTTP服务](docs/serving.md) | 输入、参数、流式输出、协议与continuation |
| [测试](tests/README.md) / [Benchmarks](bench/README.md) | 构建、测试与测量命令 |
| [4090性能](docs/performance.md#rtx-4090-sm_89-chain-qualification) | 吞吐、测量条件与限制 |
| [权重转换](docs/weight-conversion.md) | 工件、表示和可选组件 |

参数名称和可用选项以对应可执行程序的 `--help` 为准。

## 来源、贡献与许可证

本项目基于 [Neroued/ninfer](https://github.com/Neroued/ninfer) 的引擎与后续 Ternary 执行线，
融合 [YaRN 移植](https://github.com/alanthinker/ninfer-4090-yarn) 与
[4090 压缩 KV 实现](https://github.com/sergiuszm/ninfer-4090)。
上游面向 RTX 5090 / `sm_120a` 的调度需在 Ada 上实测后采用。

贡献者名单见 [CONTRIBUTORS.md](CONTRIBUTORS.md)，贡献约定见 [PR_POLICY.md](PR_POLICY.md)。
许可证为 **GNU AGPL-3.0-only**，见 [LICENSE](LICENSE)。
