# TandemLLM adoption status for 24GB GPUs

This document records the final scope of the TandemLLM-derived work on the RTX 4090 path. The goal
is not source-level feature parity with a 128GB GB10 system. The production target is one or two
24GB Ada cards, so persistent VRAM is treated as a first-class constraint.

## Adopted

| Area | NInfer implementation | 24GB impact |
|---|---|---|
| adaptive speculative width | fixed vs Stair A/B, explicit measured cost rungs | host-only state |
| runtime speculative tree | DFlash2 lattice tree, hybrid attention + GDN/ReplaySSM correctness | <=16-row replay buffers |
| Tree-Stair | one b16 drafter, dynamic 3/7/11/15 node target cuts | no second drafter |
| GPU tree control | device tree build, greedy path walk, dynamic accepted-path gather | transient only |
| lookup drafting | recent/vote, local/process/static sources, replace/head-skip/deep | host history only |
| lookup persistence | bounded suffix snapshot across clean restarts | host/disk only |
| router persistence | request or Engine lifetime, optional disk snapshot | host/disk only |
| DFlash2 training | CE + truncated target-K KL, b8/b16, lastN/all, resume/export | offline |
| native teacher | exact loaded-.ninfer taps + prepared target-head top-16 | offline transient |
| tree calibration | lattice recording and offline budget sweeps | offline |

All new runtime policies remain explicit parameters. Defaults preserve the old chain/recent/request-
local behavior.

## Recommended 24GB runtime profile

One b16 DFlash2 checkpoint is resident. Do not keep a b8 checkpoint beside it.

```bash
ninfer-serve MODEL.ninfer \
  --spec dflash2 --draft-tokens 15 \
  --spec-tree lattice --spec-tree-nodes 15 --spec-tree-spine 7 \
  --spec-router stair \
  --spec-stair-widths 3,7,11,15 \
  --spec-router-scope engine \
  --spec-router-state /var/lib/ninfer/spec-router.state \
  --lookup-ngram 5 --lookup-strategy vote --lookup-dflash skip \
  --lookup-persistent-tokens 262144 \
  --lookup-persistent-path /var/lib/ninfer/lookup-history.bin
```

The one wide drafter pays its proposal cost once. Tree-Stair changes only the active target tree
prefix inside the already-reserved <=16-row physical allocation.

## Native DFlash training loop

For Ternary Bonsai / T2 / mixed Q5 artifacts, collect supervision from the artifact that will
actually serve:

```bash
ninfer-cli MODEL.ninfer \
  --prompt "$(cat corpus.txt)" \
  --spec dflash2 --draft-tokens 15 \
  --dflash-teacher-out train/native-teacher \
  --max-new 1

python -m tools.dflash2_training.import_ninfer_teacher \
  --input train/native-teacher --out train/ninfer-data

python -m tools.dflash2_training.train \
  --drafter BASE_DFLASH2 \
  --target HF_BASE_MATCHING_THE_ARTIFACT \
  --data train/ninfer-data \
  --out train/dflash2-b16 \
  --block 16 --train last1 --optimizer adamw8bit
```

The native recorder uses the exact target residual taps and exact prepared NInfer output head to
produce hard labels and top-16 teacher scores. The standalone trainer still uses the matching base
checkpoint's differentiable embedding/head; this must correspond to the artifact vocabulary/base
model.

## Deliberately not adopted as the default 24GB path

### Resident b8 + b16 dual drafter

Tandem can use separate narrow/wide drafter checkpoints. NInfer instead keeps one b16 checkpoint and
cuts target work. A second resident drafter increases the memory floor for a benefit that the
single-wide policy can approximate without the extra weights.

### Persistent 24/32-row tree allocation

Offline tools can evaluate 23/31-node trees, but the production 24GB path stops at 15 draft nodes /
16 target rows. Larger trees require larger target replay buffers and are deferred until measured
16-row verification leaves enough benefit to justify the memory and kernel work.

### Blackwell NVFP4/TMA/SM121 kernels

Those implementations are hardware-specific to GB10/Blackwell and are not copied into the sm_89
runtime. NInfer keeps its own T2/Q4/Q5/Q8 and Ada small-M routes.

### Static-corpus mmap

The static suffix corpus remains a bounded host vector loaded at startup. On the target machine
class (256GB host RAM) this is not a GPU-memory constraint. Cross-restart process history and router
state were prioritized because they change serving behavior directly. An mmap storage backend can
be added later without changing the lookup contract.

### Sampling tree and merged lookup+DFlash tree

Tree v1 remains raw-greedy C1; sampled requests and lookup takeover fall back to exact existing
paths. Lookup and DFlash are not merged into one branch tree in this 24GB closure. Both remain
possible future throughput work, but neither is required to keep the current runtime exact.

## Remaining performance work is measurement-driven

After these PRs, further speculative work should be justified by RTX 4090 measurements rather than
Tandem's GB10 numbers. In particular:

- measure K7/K11/K15 chain vs tree;
- calibrate target verify costs by context/KV profile;
- train the native-teacher b16 checkpoint and compare accepted tokens/round;
- only consider >16 rows or sampled/merged trees if those results show material remaining headroom.

The final architecture therefore optimizes for a low VRAM floor first and keeps every larger
Tandem-style optimization behind evidence rather than making it a permanent resident cost.
