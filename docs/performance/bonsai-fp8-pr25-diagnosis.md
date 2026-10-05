# PR #25 FP8 性能诊断与修复

**结论：已修复当前可复现的 split 调度性能退步。29K 输入 decode 三次中位数为513.7 tok/s，比修复前PR25的449.3提高14.34%，比PR24历史484.1提高6.11%；推荐在本报告测得的长输入场景采用修复后的FP8方案。** 数值专项通过，六深度PPL完成且有小幅双向变化，不能宣称质量无损或所有场景均有收益。本次只测试新方案及其修复候选，没有运行或checkout旧代码。缺少旧方案同类counter及NCU硬件计数器，不能严格证明哪项历史改动是唯一原因。

## 顺序、实施和代码状态

按先文档、后实施、再调查的顺序完成。采集前本文记录了目标、假设、范围和验收条件；调查结束后更新为结果与复现说明。

初始诊断阶段生产代码固定为 `main@36d1238`，未修改生产计算。修复阶段只修改下文两处私有split调度及对应边界测试；量化、默认KV和生产NVTX均未修改。现有decode标记、attention benchmark的profiler API边界及Nsight原生CSV导出已经足够。

新增 `tools/bench/summarize_nsys_trace.py` 与 `tools/bench/test_summarize_nsys_trace.py`，并更新 `bench/README.md`。工具按精确的 projected NVTX 名称选取 decode 的首尾范围，聚合 kernel 的调用、耗时、grid/block 和资源；GPU 忙碌时间使用区间并集，避免重叠重复计数。拒绝切断 GPU event 的范围与多 device/context 输入。工具要求一次 capture 只有一个待分析请求。

Main Review 后由一个 verifier 在固定生产代码状态采集。工具的两个行为检查、Python 3.11 py_compile、真实 Nsight CSV smoke 解析均通过，两个既有 benchmark target 编译通过。正式 CLI 采集也成功验证了 projected decode 范围解析。没有创建提交或推送；本次代码与文档改动留在工作区。

## 环境与测量范围

硬件为两张 NVIDIA GeForce RTX 4090 24 GB，测试前均为空闲。正式CLI测量使用GPU 0，修复后的PPL使用GPU 1；CUDA/NVCC 12.8.61，Release、sm89，Nsight Systems 2024.6.2、Nsight Compute 2025.1，Python 3.11.15。

显式工件为 `/opt/ninfer-4090/Ternary-Bonsai-2-27B-ninfer-v3.ninfer`。CLI 复用 [单向验收报告](bonsai-fp8-pr25-4090-2026-10-05.md)中的 messages-32k command.json：prompt 29141、context/KV capacity 32768、FP8、DFlash2 Fixed K7、greedy、零惩罚、关闭 thinking、256-token 输出。旧 FP8 仅复用历史中位数 484.1 tok/s；它不是同场 A/B。

初始诊断复用了已有数值和PPL证据；修复后在最终代码上重新完成native FP8数值专项、六次CLI及六深度PPL。没有执行全仓库套件、CI、旧方案或其他KV格式。

## 1. 修复前正常计时：退步可以复现

| 重复 | prefill，k tok/s | decode，tok/s | acceptance |
|---:|---:|---:|---:|
| 1 | 1.88 | 448.5 | 100% |
| 2 | 1.93 | 449.5 | 100% |
| 3 | 1.93 | 449.4 | 100% |

三次均退出 0，完成 256 tokens，accepted/drafted 为 223/223。新中位数为 **449.4 tok/s**，与前次新方案的 449.3 接近；相对旧历史中位数 484.1 低 **7.17%**。此结果支持退步可复现，不提供改动级因果。

## 2. 完整 decode 时间线

Nsight Systems 使用 cuda,nvtx 与 CUDA Graph node 跟踪，禁用 CPU sampling。原生 CSV 导出后选取 `ninfer:decode`。范围核对得到 32 个 decode、32 个 dflash round、512 次 native attention（32 × 16）；29 个 prefill range 均未与 decode window 重叠，未发现 prefill 泄漏。

| 指标 | 结果 |
|---|---:|
| projected decode 首尾跨度 | 643.174 ms |
| GPU event 区间并集 | 566.992 ms |
| 未被 GPU trace 覆盖的区间 | 76.181 ms |
| native attention 累计时间 | 152.745 ms |
| native attention 次数 / 平均 | 512 / 298.330 µs |
| native attention / GPU busy | 26.94% |
| 三个主要 t2 i8 kernel 累计时间 | 244.310 ms |
| 三个主要 t2 i8 kernel / GPU busy | 43.10% |

Attention 是最大单个 kernel，但不是 decode 唯一成本。三个三值权重 kernel 的累计时间分别为 112.576、90.012、41.722 ms；没有旧 counter，不能据当前占比说它们造成了历史回退。

每轮 16 次 attention，单轮累计约 4.772–4.776 ms，没有发现明显轮间抖动。未追踪的 76.181 ms 可能涉及 host/control/Graph 调度或 trace 缺口，不能单独确认 CPU 瓶颈。

带 profiling 的应用显示约 397.5 tok/s，低于正常计时。Graph node 采集有扰动，以上时间用于当前成本归属，不能代替普通吞吐，也不能与旧无 profiling 的数字直接比较。

## 3. 公共 attention Op：主要成本在主内核

独立公共 append-and-attend 使用 H24/KV4、D256、B1、T8、context29141、Graph、warm cache，预热5次、测量61次。中位数 **333.824 µs**，最小333.792 µs，p95为334.848 µs，完整 Graph 含4个节点。

独立一次 Nsight Systems 采集的分解如下；该样本不与61次正常计时混作同一测量：

| 部分 | 时间 | 占四个 kernel 累计时间 |
|---|---:|---:|
| KV append | 2.048 µs | 0.62% |
| Q prepare | 1.824 µs | 0.55% |
| native attention | 323.639 µs | 97.21% |
| reduce output | 5.408 µs | 1.62% |

真实生成 decode 中，Q prepare 的512次累计为779.374 µs，仅相当于主attention累计时间的 **0.51%**。因此不能把主要成本归给Q预处理；这些单向数据也不能计算它相对旧fused实现的净收益。

公共Op输入为独立非均匀fixture，生产KV来自真实模型；两者工作负载不同。不能直接用333.824 µs与真实生成的298.330 µs差异判断内核回退。

## 4. 绝对分区产生16倍工作量差异

修复前planner在代表性29141-key窗口的每个KV head分配如下。真实生成随着round增长，窗口不是固定29141；这个表是静态代表点，不是每个CTA的实测耗时。

| split数量 / 每KV head | keys / split | Bc32 tiles / split |
|---:|---:|---:|
| 16 | 128 | 4 |
| 16 | 512 | 16 |
| 16 | 1024 | 32 |
| 1 | 2048 | 64 |
| 1 | 469 | 15 |

有50个active splits，最长与最短工作量为 **64/4 = 16倍**。生产Graph envelope为32768，观察到grid为 `4×51×1`；代表点每KV head另有1个inactive split。

绝对边界有助于保持prefix reduction分区稳定，但当前递增跨度同时引入明显负载差异。这支持“工作分配存在长尾风险”的假设，尚未测得CTA级时间或SM调度，不能将16倍tile差异直接等同为16倍执行时间。

## 5. 资源明确限制理论驻留

Nsight Systems记录实际主attention launch为384 threads/block、134 registers/thread、static shared29120 B、dynamic shared40960 B。CUDA device probe得到RTX4090每SM有65536 registers、102400 B shared、1536 threads，最多24 blocks。

| 资源 | 每CTA需求 | 每SM能力 | 仅由该资源计算的上限 |
|---|---:|---:|---:|
| registers | 至少134 × 384 = 51456 | 65536 | 1 CTA |
| shared memory | 29120 + 40960 = 70080 B | 102400 B | 1 CTA |
| threads | 384 | 1536 | 4 CTA |

因此当前T8内核的综合理论上限是 **最多1 CTA/SM**。这不是实际occupancy测量；资源分配粒度、launch属性与scheduler还可能形成其他限制。寄存器和shared各自都已足以排除2 CTA驻留，所以只减少一项资源也不必然增加驻留数。没有旧内核的同类counter，不能断言双缓冲是唯一原因。

## 6. 初始诊断的假设结论与建议

| 假设 | 调查结论 |
|---|---|
| attention是重要decode热点 | 支持：最大单kernel，占GPU busy约27%；但不是唯一成本 |
| 绝对split工作分配不均 | 确认静态分配差异16倍；性能长尾仍需测量验证 |
| resource pressure限制驻留 | 确认理论上限1 CTA/SM；实际occupancy未测得 |
| pipeline重叠不足、transpose/barrier等待高 | 未决：缺硬件counter，不能从总时间推断 |
| Q prepare是主要新增成本 | 不支持：仅为attention时间约0.5% |
| host/Graph/control空隙是重要瓶颈 | 未决：存在未追踪区间，但来源未确认 |

后续建议先验证 **保持绝对分区稳定性前提下的工作量均衡**，并联合检查register/shared资源，而不是优先删除Q prepare或residual。任何调优都应单独改变一个因素，先通过数值oracle，再测完整public attention和真实生成；microbenchmark收益不能替代端到端结果。

若要解释pipeline/transpose/barrier，需要先具备NCU硬件计数器访问条件。本次NCU返回 `ERR_NVGPUCTRPERM`；`sudo -n true` 返回 `sudo: a password is required`。没有修改系统权限、索取密码或设置自动重试。因此硬件等待类型与实际occupancy仍为未验证项。后续受控调度实验补充了可修复成本的性能证据，但没有消除这一counter限制。

## 性能修复：受控实验与正式实现

用户要求继续定位并修复退步后，仅在新方案上进行单因素实验，未checkout或运行PR24。同场修复前PR25公共Op基线与前次诊断吻合。实验确认当前绝对split的长尾是主要可修复成本，而不是Q prepare的额外launch。

| 实验 | 29K/T8 | 40K/T8 | 决策 |
|---|---:|---:|---|
| 修复前PR25 | 334.848 µs | 344.064 µs | 同场基线 |
| 768-key中段，提前进入4096-key尾段 | 183.296 µs | 645.120 µs | 弃用：将长尾移到40K |
| 保持原59392-key尾段起点 | 183.296 µs | 397.312 µs | 仍有40K回退，不能验收 |
| 同一最终分区，仅反转物理CTA与逻辑split映射 | 183.296 µs | 275.456 µs | 保留，并验证其他代表点 |

第一项单因素将29K最长split由64 tiles缩到24 tiles，主Op延迟降低45.26%。第二项受控映射实验使40K延迟从397.312降至275.456 µs，说明grid映射对该Ada工作负载有实质影响。没有硬件CTA时间counter，不能进一步量化实际SM排队过程或宣称CUDA固定执行block顺序；此次证据是保持同一数学分区下的实际性能变化。

正式实现只有两个生产文件：`small_t.cuh`的绝对分区与`small_t_fp8_sm89.cuh`的映射。跨度为128/512/768/2048/4096 keys，在2048/10240/34816/59392切换；768中段覆盖32个split，随后12个2048 split，4096尾段保持原起点。786432 keys需要254 splits，未扩大256槽位或每CTA64页的上限。物理CTA y映射到逻辑split `gridDim.y - 1 - blockIdx.y`，partial存储和reduce合并仍使用逻辑ID。

warp、Bc32、双缓冲、KV存储、Q/P main+residual与FP32 softmax均未改动，没有新增公开配置或旧路径fallback。

### 最终公共Op结果

均为H24/KV4、D256、B1、append、Graph、warm cache、预热5次、61次测量，表中为中位数。同场基线是修复前PR25，不是重新测试PR24。

| Context | T | 修复前 µs | 修复后 µs | 延迟变化 |
|---:|---:|---:|---:|---:|
| 6847 | 1 | 67.584 | 67.584 | 0% |
| 6847 | 8 | 81.920 | 81.920 | 0% |
| 29141 | 1 | 274.304 | 146.432 | −46.62% |
| 29141 | 8 | 334.848 | 183.296 | −45.26% |
| 34816 | 8 | 342.016 | 194.560 | −43.11% |
| 40960 | 8 | 344.064 | 275.456 | −19.94% |
| 65536 | 8 | 676.864 | 527.264 | −22.10% |

29K/T8 public Op workspace从10007040增加到11394048 bytes；40K/T8从11195904增加到13573632 bytes。这是更多partial槽位的成本，不是持久KV增大。实际程序资源由最终生成/评分报告另行记录。

### 数值验证与验收状态

`ninfer_softmax_attention_test --native-fp8-only`完整一轮通过；新增两条跨分区边界用例复用独立FP64 oracle，并检验Graph、ragged/reordered batch和逐列T1 bitwise一致性。门限未放宽。现有host容量检查和编译期覆盖检查通过；未另做786432-key的数GB GPU oracle。

前一候选的一轮数值测试在性能反例出现后被中止（exit143），不计为通过。最终代码经过完整专项，避免用旧状态测试结果验收新状态。产品、benchmark及perplexity已构建。

### 最终真实生成

复用相同工件、messages、FP8、C1 Fixed K7和输出条件，每个输入三次正常计时。下表吞吐均为中位数；历史数字只用于比较，没有重新运行PR24。

| 输入tokens | 修复后prefill k tok/s | 修复后decode tok/s（范围） | 修复前PR25 decode | PR24历史decode | 相对修复前 / PR24 |
|---:|---:|---:|---:|---:|---:|
| 6847 | 2.62 | 573.3（573.0–573.4） | 572.8 | 560.9 | +0.09% / +2.21% |
| 29141 | 2.16 | 513.7（513.1–513.8） | 449.3 | 484.1 | +14.34% / +6.11% |

修复前PR25 prefill为2.54/1.94 k tok/s，PR24历史为2.57/1.80 k tok/s。短输入decode变化很小，不能据此宣称有实质提升。六次均退出0、完成256输出、accepted/drafted为223/223、100% acceptance、7.97 tokens/round，结束原因为output-limit。GPU workspace peak均为139.5 MiB，KV仍为258.0 MiB / 1.01 GiB。

### 最终固定深度PPL

同一评分输入、目标区间和执行配置；逐stream/window核对input、target、prefix_depth、position_base及scored_tokens，与PR24历史和修复前PR25完全一致，execution配置也一致。quick为4 streams、16384 targets；long为1 stream、8192 targets，合计24576 targets。

| 深度 | PR24历史PPL | 修复前PR25 PPL | 修复后PPL | 相对PR24 |
|---:|---:|---:|---:|---:|
| 8192 | 4.882174 | 4.884220 | 4.884220 | +0.0419% |
| 32768 | 3.571380 | 3.571077 | 3.570569 | −0.0227% |
| 65536 | 8.592169 | 8.605957 | 8.614935 | +0.2650% |
| 131072 | 21.386692 | 21.373408 | 21.377437 | −0.0433% |
| 196608 | 1.830166 | 1.829590 | 1.829221 | −0.0517% |
| 258048 | 1.267433 | 1.268088 | 1.266505 | −0.0732% |

六深度均完成，无NaN、OOM或fallback；没有放宽oracle标准。PPL变化有正有负，64K相对修复前增加约0.1043%，不能称为质量无损，也没有事后设置质量阈值。评分workspace_capacity仍为524460032 bytes，quick/long KV payload仍为1149763584/8589410304 bytes。仅为该固定输入上的质量证据，不覆盖其他模型、batch或扩展上下文。

正式CLI证据为 `profiles/bench/pr25-fp8-fix-2026-10-05/final-validation/messages-{8k,32k}/rep{1,2,3}/`中的command.json、status.json及run.log；PPL证据为 `profiles/perplexity/bonsai-kv-pr25-fix-2026-10-05/{quick,long}/fp8/fp8/result/report.json`及同次命令与日志。

修复证据位于 `profiles/bench/pr25-fp8-fix-2026-10-05/`：`baseline*.csv`、`balanced*.csv`、`final*.csv`、`reversed*.csv`、`public-summary.json`、`native-reversed.log`与`final-cli32k/`。这组文件记录已弃用候选和最终方案，不能混用。

## 复现与证据

正常CLI完整参数保存在 `profiles/bench/pr25-fp8-diagnosis-2026-10-05/campaign/cli/messages-32k/rep{1,2,3}/command.json`。

时间线可用以下方式采集同一CLI命令，随后按 [benchmark文档](../../bench/README.md)导出并汇总：

```bash
nsys profile --trace=cuda,nvtx --cuda-graph-trace=node \
  --sample=none --cpuctxsw=none -o profiles/nsys/pr25-decode \
  ./build/apps/ninfer /opt/ninfer-4090/Ternary-Bonsai-2-27B-ninfer-v3.ninfer \
  --messages profiles/perplexity/bonsai-kv-pr24-2026-10-05/workload/messages-32k.json \
  --max-context 32768 --kv-capacity 32768 --prefill-chunk 1024 \
  --max-new 256 --device 0 --kv-dtype fp8 --spec dflash2 \
  --draft-tokens 7 --spec-router fixed --greedy \
  --presence-penalty 0 --frequency-penalty 0 --no-thinking --log-level info

./build/bench/ninfer_causal_softmax_attention_bench \
  --entry append --geometry d256-h24-kv4 --kv-dtype fp8 \
  --batch 1 --tokens 8 --context 29141 --execution graph --cache warm \
  --warmup 5 --repeat 61
```

公共Op使用同参数的 `--profile` 模式进行单调用profiling，setup与预热在profiler边界外。NCU尝试日志保留实际参数及权限错误；未生成有效硬件counter报告。

本地证据根目录为 `profiles/bench/pr25-fp8-diagnosis-2026-10-05/`：

- `tool-*`：构建、Python检查、真实CSV smoke与输出。
- `campaign/cli/summary.json`：三次正常计时。
- `campaign/nsys/decode-summary.json`、`decode-kernel-detail.json`、`decode-gaps.json`、`decode-window-scope-check.json`：decode汇总、kernel、空隙、范围核对。
- `campaign/nsys/cli-generate-node.nsys-rep`及原生CSV：完整时间线。
- `campaign/attention/append-h24-t8-context29141-graph-warm.csv`：正常公共Op测量。
- `campaign/attention/public-profile-node.nsys-rep`及原生CSV：单次公共Op分解。
- `campaign/attention/split-distribution-29141.json`、`device-attrs.log`、`resource-residency-analysis.json`：分区及资源证据。
- `campaign/attention/append-h24-t8-context29141-sm89-attn.ncu.log`：NCU权限阻塞。

此次未运行旧方案、旧counter采集、C8、其他KV、其他硬件、全仓库套件或CI。修复后的代表性公共Op、真实生成和六深度PPL已完成；硬件等待类型和历史退步的唯一归因仍有上述限制。全部改动留在工作区，未创建提交或推送。

## 同类调度问题扩展排查

目标是判断当前sm89 attention是否还有与本次修复相同的可修复退步。范围包括共享split planner、其他KV格式的small-T路径，以及FP8不同query宽度、batch和Graph envelope。不会把不均匀跨度、较高资源需求或物理block顺序单独判定为bug。

先只读核对调用、分区和已有证据；仅对可能改变修复决策的候选运行公共Op聚焦实验。确认问题需要可观察的错误或受控性能证据。性能候选须保留数学契约，覆盖其影响的边界，并在实际修改后通过独立oracle。若没有足够证据，则报告为未验证假设，不移植本次FP8分区或grid映射到其他路线。当前基线为已提交的 `8c76e89`；不运行PR24或其他历史实现。

源码调查找到两个候选，尚无性能结论：

- FP8宽Graph envelope：普通profile在32767之后覆盖整个capacity（`src/models/qwen3_5/program/planning/graph_profiles.cpp`），launcher按max_visible_keys选择gridY，kernel按实际window过滤inactive split。例如实际32768而容量786432时，约62个active split与254个grid槽位并存。现有反向映射使inactive槽位位于物理grid前部。先用同一输入、不同合法envelope测量，避免把不同context的数学工作量混入对照。
- NVFP4/K8V4：host upper bound保留旧分段最高split数，而device按实际window选active数。它保证容量安全，但可能发射更多inactive CTA，暂未发现数值错误。本轮优先验证FP8；不将其结论推广至其他KV格式。

最低成本实施是让已有公共Op benchmark显式接受较宽execution envelope，并记录该参数；不修改产品调度或数学。宽envelope测量由单一worker执行，另一个verifier只测已冻结生产代码的相邻分区边界。确认额外开销后再选择生产修复，保持absolute logical split、partial槽位及reduce顺序。此次新增排查尚未提交。


### 扩展排查的聚焦结果

当前生产代码仍为8c76e89，仅benchmark增加了 `--max-visible-keys` 参数，生产调度没有改动。Release sm89、CUDA12.8.61、RTX4090，公共FP8 append、H24/KV4、D256、Graph、warm、identity mapping，预热5次、31次计时。普通边界测量使用GPU0，宽envelope使用GPU1；不能混作同场绝对延迟比较。

| 分区边界 | B1前后 µs | B8前后 µs | 判断 |
|---:|---:|---:|---|
| 10240 | 83.968 → 83.968 | 447.488 → 450.432 | B8 +0.66%，无明显跳变 |
| 34816 | 194.496 → 193.536 | 1317.888 → 1320.960 | B8复测+0.23%；首轮异常未复现 |
| 59392 | 278.528 → 278.400 | 1909.600 → 1923.008 | B8 +0.70%，无明显跳变 |

每个边界使用context分别为边界−8、边界+8，T8的实际最大window还包含8个query列。34816的B8首轮曾下降7.45%，复测后消失，不作为可修复缺陷。29K的T2/T16亦完成：B1为153.600/362.496 µs，B8为899.072/2249.728 µs；不同宽度本来有不同工作量，不能据此计算改进收益或推断端到端吞吐。T16捕获7节点，T2捕获4节点。

同一个context34824、T8输入（实际可见34832）在B1下的合法宽envelope对照：

| max_visible_keys | 延迟中位数 µs | 相对精确envelope | workspace bytes |
|---:|---:|---:|---:|
| 34832 | 194.560 | — | 12979200 |
| 131072 | 195.584 | +0.53% | 18725376 |
| 786432 | 196.608 | +1.05% | 50428416 |

B8补测同一输入的精确envelope与786432上界：1320.96 → 1340.42 µs（+1.47%），workspace由103833600增至403427328 bytes，两点均成功且无OOM。证据为 `envelope/b8-exact.csv` 与 `envelope/b8-wide786k.csv`。

**扩展结论：在本轮测得的边界、宽度、batch与宽envelope场景，没有确认另一个需要立即修复的性能退化。** B1/B8下inactive CTA确有小幅成本，暂不支持修改生产调度。workspace明显增加，但宽envelope承诺图可服务更长输入，减少容量需要改变Graph分段契约；不是可以直接删掉的内存浪费。KV逻辑/物理capacity依法扩到envelope上限，不绕过公共Op校验；初始化和计时仍只使用真实可见输入。

源码审计没有确认另一个数值bug。INT8有自己的绝对planner，不应照搬FP8改动；NVFP4/K8V4的overlaunch仍是未实测候选，本轮没有测试这些KV格式，也没有对其修改。完整命令、CSV和日志位于 `profiles/bench/pr25-fp8-similar-audit-2026-10-05/`，使用带identity后缀的正式边界CSV与 `envelope/`；早期探索行和失败的非法容量尝试不是通过证据。

benchmark构建、过小上界拒绝、超过公开上限拒绝及新增help参数检查通过。原生产native FP8数值证据继续有效；本轮未重跑PPL、真实生成、旧实现、全套测试或CI。新增benchmark和报告留工作区，未提交或推送。
