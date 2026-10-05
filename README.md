# NInfer-4090：Ternary Bonsai 2 与 YaRN

**中文** · [English](README.en.md)

NInfer-4090 是面向 **NVIDIA GeForce RTX 4090（24 GB，`sm_89`）** 的 C++/CUDA 本地推理引擎。
主要模型为 [WaveCut/Ternary-Bonsai-2-27B-NInfer-v3](https://huggingface.co/WaveCut/Ternary-Bonsai-2-27B-NInfer-v3)，
支持三值权重、MTP/DFlash2 推测解码、分页 KV、前缀复用、Vision 和 YaRN 风格的长上下文位置缩放。

当前实测推荐是：**原生上下文使用 Fixed K7；扩展上下文使用 MTP3 + YaRN，并优先考虑 `rk4v4-e8`。**
Fixed K7 的性能结论来自 INT8 KV、原生 RoPE、关闭 Vision 的测试配置；不能直接用于其他 KV 或长上下文配置。
Auto Selected 已通过对应正确性验证，但尚未证明比最佳 Fixed 策略稳定快至少 2%，因此仍为可选功能。

本文汇总截至 **2026-10-05** 的功能和已有测量。历史测试说明其执行版本的结果，本次文档重写没有重新运行这些测试。

## 能力与执行方式

| 能力 | 当前范围 |
|---|---|
| 模型 | `Qwen3_5ForCausalLM`、`Qwen3_5MoeForCausalLM`；同架构的 Qwen3.6/3.8 与 Bonsai 工件沿用相同执行路径 |
| 工件 | v3 `.ninfer`；转换器选择表示，加载器绑定实际权重 |
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
本轮完整构建验证环境为 Linux、GCC 13、CUDA 12.8.61、Release、`sm_89`；不代表本轮验证了 Windows。

在**尚未配置的源码目录**执行：

```bash
cmake --preset release -DCMAKE_CUDA_ARCHITECTURES=89
cmake --build build -j
```

已有正确配置的 `build/` 直接执行构建即可。应用位于 `build/apps/`：

- `ninfer`：命令行生成；
- `ninfer-serve`：HTTP 服务；
- `ninfer-perplexity`：离线困惑度评估。

需要测试和 benchmark 时，在新的开发配置中使用：

```bash
cmake --preset dev -DCMAKE_CUDA_ARCHITECTURES=89 \
  -DPython3_EXECUTABLE=/absolute/path/to/python3.11
cmake --build build -j
```

`release` 与 `dev` 使用同一 `build/`，切换 preset 会修改构建选项。
Python 3.11 用于测试、转换和评估工具；不会在模型生成时替代 C++/CUDA 引擎。

### 准备模型

从 [Bonsai 2 v3 工件页面](https://huggingface.co/WaveCut/Ternary-Bonsai-2-27B-NInfer-v3)
获取 `.ninfer` 文件。下面的命令假设它位于仓库根目录：

```text
Ternary-Bonsai-2-27B-ninfer-v3.ninfer
```

使用明确的模型路径，不依赖目录中的文件顺序。Vision、MTP、DFlash2 是否可启用取决于工件包含的组件。
自定义或混合表示的模型见 [权重转换](docs/weight-conversion.md)。

### 命令行生成

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

## Fixed K7 与 Auto Selected

### Fixed K7 服务

下面的 INT8 KV 配置与 resident/Auto 测量的 E8 启动设置对应，可同时激活最多八个请求：

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
C8 表示最大活动请求数或指定测试的客户端并发；实际每轮 batch 仍可能为 1–8。
请求完成后，等待请求可在安全边界进入执行。

### 可选 Auto Selected

Auto 使用与实际工件、GPU 和启动配置匹配的校准文件。schema 3 默认选择 K7，只有合格单元才覆盖为 K0/K11；
K15 保留作最大常驻 drafter 和独立比较项，不是 Auto override。
2026-10-05 的 E8 full/selected profile 均没有 override，不能据此声称 Auto 已找到比 Fixed K7 更快的并发策略。

使用已有开发构建生成新 profile；以下命令使用两张空闲 GPU 分别运行单 GPU 作业：

```bash
python3.11 -m tools.bench.run_dflash_gpu_campaign \
  --model /absolute/path/to/Ternary-Bonsai-2-27B-ninfer-v3.ninfer --build-dir build \
  --out profiles/bench/dflash-gpu-campaign --gpus 0,1 \
  --concurrency 1,2,4,8 --pairs 2 --repeats 2 \
  --max-tokens 512 --max-context 32768 --kv-capacity 131072
```

在上面的服务命令中，将推测解码参数替换为：

```bash
--spec dflash2 --draft-tokens 15 --spec-router calibrated \
--spec-router-profile profiles/bench/dflash-gpu-campaign/auto-E8-selected.json
```

其余启动设置保持与生成 profile 时一致。Selected compute 由 profile 中的 `"proposal_compute": "selected"` 选择，
生产 CLI 没有 `--proposal-compute selected` 参数。工件、KV、缓存容量、并发或执行设置变化后需要匹配的新 profile。
上述服务 profile 不直接用于关闭 context cache 的单命令 CLI。
详见 [校准路由](docs/maintainer/speculative-routing.md#calibrated-dflash2-chain)。

<a id="choosing-a-kv-format"></a>

## KV、Vision 与长上下文

生成程序接受 `bf16`、`int8`、`fp8`、`rk8v4`、`rk4v4`、`rk4v4-e8`、`rk2v4-e8`、`nvfp4`、`k8v4`。
这是 KV 存储选择，不表示 RTX 4090 可执行 Blackwell 专用的权重/激活 kernel。
下面的选择依据分别是实测推荐与当前长上下文产品设计：

| 使用目标 | 配置 | 证据范围 |
|---|---|---|
| 已测原生上下文吞吐 | INT8 KV + Fixed K7 | 本文 DFlash2 真实模型矩阵 |
| 原生上下文节省 KV | `rk8v4` 或 `rk4v4-e8` | 已有实现；本文 INT8 性能表不覆盖它们 |
| 超过原生 262144 tokens | MTP3 + YaRN + `rk4v4-e8` | 支持路径；512K/658176 的完整容量与质量验收不能由 INT8 矩阵代替 |
| 图像/视频输入 | `--vision --vision-residency overlay` | 需要 Vision 组件；本文最新 DFlash2 性能矩阵关闭 Vision |

Vision overlay 将视觉塔保留在主机内存，处理图像时借用设备内存；`--vision-max-merged` 限制单项媒体的 merged tokens。
`--kv-capacity auto` 在权重和运行时预留之后自动定共享缓存容量，保留 1024 MiB 的 sizing headroom；它不保证任意配置都能装入显存。

### YaRN 长上下文启动示例

下面是 **512K 验收目标配置**，不是已测吞吐或质量保证：

```bash
./build/apps/ninfer-serve ./Ternary-Bonsai-2-27B-ninfer-v3.ninfer \
  --model-id bonsai2-27b --device 0 --max-concurrency 1 \
  --max-context 524288 --kv-capacity auto --kv-dtype rk4v4-e8 \
  --spec mtp --draft-tokens 3 --lm-head-draft --gdn-state-fp16 \
  --rope-scaling-factor 2.12 --rope-scaling-original-context 262144 \
  --vision --vision-residency overlay --vision-max-merged 12288
```

位置 `p` 不超过原生阈值 `N` 时保持不变；超过阈值后使用 `N + round((p-N)/factor)`。
缩放只作用于 RoPE 位置，KV 物理地址保持绝对位置。
**DFlash/DFlash2 与 YaRN 同时启用会在启动时报错**；扩展上下文使用 MTP。
因果注意力执行上限为 786432 tokens，最初验收目标为 512K 和 658176；执行上限不等于模型质量或显存容量保证。
详见 [位置缩放与压缩 KV 设计](docs/maintainer/rtx4090-ternary-yarn-port.md)。

## RTX 4090 实测性能

以下表格分别来自不同配置和测试版本，不能将各表中的最佳数字拼成一个配置的性能。
吞吐为所有并发请求生成 tokens 的总和；端到端、decode 和单 kernel 耗时采用各自的测量口径。

### Bonsai 2 Fixed-route 矩阵（2026-10-04）

配置：Bonsai 2 v3、INT8 KV、greedy、零惩罚、缓存与 CUDA Graph 开启，512-token 输出预算，
context/KV capacity 32768，Engine concurrency 8；七种工作负载、C1/C2/C4/C8、K7/K11/K15，每项两组 AB/BA。
84 项比较全部匹配 token IDs、完整响应和重复输出；71 项通过性能资格门槛。

| 客户端并发 | Target-only | Fixed K7 | Fixed K11 | Fixed K15 |
|---|---:|---:|---:|---:|
| C1 | 96.9 | 254.0 | 248.4 | 255.0 |
| C2 | 183.2 | 448.5 | 367.7 | 339.2 |
| C4 | 338.1 | 596.1 | 448.7 | 417.5 |
| C8 | 573.1 | 683.9 | 546.5 | 456.0 |

单位：aggregate end-to-end tokens/s；数值为跨工作负载的描述性中位数，包含未通过性能门槛的比较。
K7 在 C1/C2/C4/C8 分别有 7/7、7/7、7/7、6/7 个工作负载合格；长上下文 C8 的 K7 比 target-only 慢约 4%。
因此“推荐 K7”不是所有场景都更快的保证。
[完整矩阵与资格规则](docs/performance.md#rtx-4090-sm_89-chain-qualification)。

### Resident 与 Auto（2026-10-04 至 2026-10-05）

Resident 测试使用 Bonsai 2、INT8 KV、A8 prefill/chunk 1024、native RoPE、greedy 零惩罚、缓存与 Graph 开启，
最大上下文 32768；E1 的 KV capacity 为 32768，E8 为 131072。每个 GPU 只上传一次模型，每个作业使用独立可变状态。
短/中/长验证 prompt 为 54/4554/11706 tokens，720 个验证请求各生成 512 tokens。

该矩阵测得最高 aggregate rate 为 **816.7 tokens/s**，来自 C8、中等 prompt、后续复用波次；
单请求中等 prompt 复用时，Fixed K11 为 **347.2**、K7 为 **329.8 tokens/s**。
这些是指定工作负载的结果，不是所有请求的最低性能。
[Resident 报告及资源记录](docs/performance/dflash2-4090-2026-10-04.md)。

schema-3 follow-up 对 backend None 的几何平均加速如下：

| 范围 | Fixed K7 | Auto Full | Auto Selected |
|---|---:|---:|---:|
| 单请求，cold | 2.094× | 1.995× | 2.103× |
| 单请求，warm | 2.731× | 2.588× | 2.771× |
| 并发，cold | 1.433× | 1.287× | 1.419× |
| 并发，warm | 1.740× | 1.492× | 1.728× |

Cold 是未预热的首波请求，同波并发仍可能复用前缀；warm 是后续观察到自然复用的波次。
Auto Selected 接近 K7，但没有 held-out 条件通过“超过最佳 Fixed 至少 2%、每个保留 pair 更快、p95 回退不超过 5%”的门槛。
该 follow-up 复用了前一轮正确性结果，没有重新运行完整回归或 CI。
[Auto K7 follow-up](docs/performance/dflash2-auto-k7-4090-2026-10-05.md)。

### T64 SwiGLU 路由优化（PR #19，2026-10-05）

只将 `sm_89` DFlash2 的 T64 从 `R64/C64/BK128` 改为已有的 `R32/C64/BK128`；其他宽度与路径保持一致。
RTX 4090 cold-cache 测量采用 5 次 warmup、31 次测量：

| 指标 | 原 R64 | 新 R32 | 变化 |
|---|---:|---:|---:|
| 单 SwiGLU kernel 中位耗时 | 337.1 μs | 298.0 μs | 耗时降低约 11.6% |
| Fixed K7 / C8 端到端吞吐 | 639.69 tokens/s | 644.46 tokens/s | 约 +0.75% |

端到端测试使用同一 Bonsai 工件、INT8 KV、context 8192、KV capacity 32768、512-token 输出、两次重复。
基线复用相同构建，仅使用合并前的路由对象；8192 个输出 token IDs 完全一致，339 轮覆盖实际 batch 1–8。
只有一次短 A/B，收益是方向性结果，不能宣称稳定显著提升。
[PR #19](https://github.com/zhongpei/ninfer-4090/pull/19)；
本地证据目录为 `profiles/bench/pr19-t64-2026-10-05/`，含数值日志、kernel 测量、A/B JSON 与 `summary.json`。

### 早期 Qwen3.8 INT8 兼容性结果（2026-08-15）

这是 **Qwen3.8-27B INT8 工件的历史基线**，不是 Bonsai 或当前构建的性能。
配置：RTX 4090、CUDA 12.8.93、INT8 KV、MTP3 + LM-head draft、greedy、前缀复用关闭，每请求 1024 输出 tokens。

| 并发 | End-to-end tokens/s | Decode tokens/s | Mean TTFT | Peak VRAM |
|---|---:|---:|---:|---:|
| C1 | 102.13 | 103.35 | 112 ms | 18250 MiB |
| C2 | 162.46 | 165.74 | 160 ms | 18562 MiB |
| C4 | 193.49 | 198.76 | 295 ms | 19184 MiB |
| C8 | 299.82 | 315.09 | 644 ms | 20708 MiB |

C1–C4 使用 8K KV pool，C8 使用 16K；所有请求完成指定输出长度。
[早期测试记录](docs/rtx-4090-early.md)。

<a id="evaluation"></a>

## 正确性、质量与验证状态

| 测试记录 | 通过范围 | 限制 |
|---|---|---|
| PR #19 完整编译 | `cmake --build build -j`，512 个构建步骤完成，exit 0 | Linux / CUDA 12.8.61 / Release / sm_89；未查询 CI |
| PR #19 T64 数值 | Q8 A16 DFlash2 `[34816,5120] → [17408]`；普通与 CUDA Graph 执行、workspace/guard/输入只读检查通过 | FP64 独立 oracle；relative-L2 0.002881/0.003077，小于 0.0033；仅 T64 |
| Fixed-route campaign | 84/84 正确性比较通过，实际 batch 1–8 | 指定 Bonsai/INT8/native-context 矩阵 |
| Resident real-model | 18/18 用例通过，full/selected 各 9 个场景 | 2026-10-04 的执行版本 |
| Resident ownership | 5/5，通过独占执行、状态隔离及 teardown 检查 | 不代表跨 Engine 共享可变状态 |
| Held-out validation | 720 个请求的 token IDs、完整响应与重复结果匹配 None | 两个测量顺序；p95 为小样本描述值 |
| CPU / Python 回归 | report suite 69 passed；benchmark suite 106 passed + 2 项 native CLI 补测通过；native CPU 3/3 | 补测消除了原 2 项 skip；不是本次重新运行的全仓库测试 |
| GDN 状态修复 | endpoint continuation、context 边界切换、成员变化、stop/cancel、zero-commit；native C4/C8 token/响应一致 | 见已有 integration 报告 |
| A16 Conv 修复 | B2/T1 的 FP64 relative-L2 从 0.003274 降至 0.001694，小于原门限 0.00315 | T1/T1024 whole-Op 约慢 14%，不是无代价优化 |
| T2 A16 修复 | FP64、宽度/分段一致性通过；6144 个输出 token IDs 匹配 | public Linear 的 T1/T16/T50 成本分别约 +21.6%/+2.1%/−15.8%，不等于端到端收益 |
| 早期 Engine 集成（2026-10-02） | loading、prefix、CausalScore、Vision workspace、DFlash2 五个真实模型测试分别通过 | Vision workspace 不等于实际图像请求或质量验证；MoE/DFlash 的对应工件测试未验证 |

正确性和性能是独立结论。已有状态修复和 A16 的证据见
[4090 qualification](docs/performance.md#rtx-4090-sm_89-chain-qualification)，
CPU、real-model 与 ownership 明细见 [Resident 报告](docs/performance/dflash2-4090-2026-10-04.md#correctness-and-execution-checks)。
PR #19 按范围只测试新增 T64 路由，旧路径仅比对源码，没有运行旧功能测试或完整运行时套件。

2026-10-02 的阶段记录还保留了 CTest 初跑与失败项复测合计 146 passed / 7 skipped、Python 87 passed，
以及整体 A/B 输出门禁失败的结果；这不是一次全新的 153 项全通过。
后续 Fixed/Resident 矩阵在状态与数值修复后重新建立了上面列出的精确输出证据。
这些历史记录保留在本地 `profiles/bench/final-ab-2026-10-02/final-summary.json`，不作为当前版本的完整测试通过声明。

**尚未由本文数据完成验证的范围：** Bonsai 的 `rk4v4-e8`/`rk2v4-e8` 全面质量与性能、
YaRN 512K/658176 的完整容量与质量、开启 Vision 的最新 DFlash2 矩阵，以及其他硬件和 Windows 当前版本。
不使用其他模型或历史硬件的困惑度、准确率、显存数字替代这些结论。

需要测量质量时，可使用固定语料：

```bash
./build/apps/ninfer-perplexity ./Ternary-Bonsai-2-27B-ninfer-v3.ninfer \
  --corpus eval/corpora/perplexity-1m/manifest.json --quick --kv-dtype int8
```

比较时保持工件、语料、context、stride 和执行设置一致，只改变待比较项。
评分程序的 KV 选项以自己的 `--help` 为准，不能直接假定与生成程序完全相同。
[困惑度方法](docs/perplexity.md) · [能力评估](eval/README.md)。

## 测试、测量与文档

开发构建的测试入口如下；这是复现命令，不表示本次文档工作运行了完整套件：

```bash
ctest --test-dir build --output-on-failure
cmake --build build -j --target ninfer_linear_swiglu_q8_a16_test
./build/tests/ninfer_linear_swiglu_q8_a16_test
```

上面的 SwiGLU 标准测试覆盖多种形状和宽度；PR #19 验收使用了仅 T64 的临时入口。
数值 Op 用独立 FP32/FP64 oracle，精确编码用精确 oracle；kernel benchmark 只证明对应 Op 的性能。
端到端建议使用公共 Engine benchmark 或协议请求测量，并记录实际 batch、配置和比较顺序。

| 文档 | 内容 |
|---|---|
| [文档地图](docs/README.md) | 用户指南与维护者参考 |
| [CLI](docs/cli.md) / [HTTP 服务](docs/serving.md) | 输入、参数、流式输出、协议与 continuation |
| [测试](tests/README.md) / [Benchmark](bench/README.md) | 构建、测试、测量入口与指标 |
| [4090 性能记录](docs/performance.md#rtx-4090-sm_89-chain-qualification) | 资格条件、吞吐、修复与限制 |
| [权重转换](docs/weight-conversion.md) | 工件、recipe、表示和可选组件 |
| [Op 开发](docs/maintainer/op-development.md) | 数学语义、独立 oracle 与性能资格 |

当前可执行程序的 `--help` 是参数名称与可用选项的直接依据。测试只覆盖其标注的版本与范围，未运行或未查询的检查不视为通过。

## 来源、贡献与许可证

本项目基于 [Neroued/ninfer](https://github.com/Neroued/ninfer) 的引擎与后续 Ternary 执行线，
融合 [YaRN 移植](https://github.com/alanthinker/ninfer-4090-yarn) 与
[4090 压缩 KV 实现](https://github.com/sergiuszm/ninfer-4090)。
上游面向 RTX 5090 / `sm_120a` 的调度需在 Ada 上实测后采用。

贡献者名单见 [CONTRIBUTORS.md](CONTRIBUTORS.md)，贡献约定见 [PR_POLICY.md](PR_POLICY.md)。
许可证为 **GNU AGPL-3.0-only**，见 [LICENSE](LICENSE)。
