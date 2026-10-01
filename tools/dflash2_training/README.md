# DFlash2 training tools

This directory contains the NInfer-side adaptation of TandemLLM's DFlash2 distillation workflow.

For the closest match to a deployed quantized/Ternary target, prefer the native exporter over the
Hugging Face recorder:

```bash
ninfer-teacher out/bonsai2-target.ninfer \
  --input prompts.jsonl --out train/ninfer-teacher \
  --devices 0,1 --kv-dtype int8 \
  --layers 5,19,33,47,61 --top-k 16

python -m tools.dflash2_training.train \
  --drafter BASE_DFLASH2 --target /models/Qwen3.8-27B \
  --data train/ninfer-teacher --out train/dflash2-b16 \
  --block 16 --train last1 --optimizer adamw8bit
```

`ninfer-teacher` executes the actual `.ninfer` target kernels and records their target taps,
argmax and exact top-16 log-probabilities. The Python `--target` directory still supplies the
borrowed embedding/lm-head tensors used by the differentiable DFlash2 reference module; the
teacher distribution itself comes from the deployed artifact.

Main commands:

```bash
# 1. Record target taps and top-K teacher distributions.
python -m tools.dflash2_training.record_teacher --model MODEL --input prompts.jsonl --out DATA

# 2. Fine-tune a b16 drafter (15 proposals).
python -m tools.dflash2_training.train \
  --drafter BASE_DFLASH2 --target MODEL --data DATA --out DFLASH2_B16 \
  --block 16 --train last1 --optimizer adamw8bit

# 3a. Build an ordinary Qwen3.8 NInfer artifact.
python -m tools.dflash2_training.export_ninfer \
  --model MODEL --drafter DFLASH2_B16 --out qwen38-b16.ninfer

# 3b. Build the Ternary Bonsai artifact.
python -m tools.dflash2_training.export_ninfer \
  --model MODEL --ternary Ternary-Bonsai-2-27B-PQ2_0.gguf \
  --drafter DFLASH2_B16 --out bonsai-b16.ninfer

# 4. Record candidate lattices and sweep tree budgets offline.
python -m tools.dflash2_training.record_lattice \
  --drafter DFLASH2_B16 --target MODEL --data DATA --out lattice.jsonl
python -m tools.dflash2_training.calibrate_tree lattice.jsonl
```

See `docs/maintainer/dflash2-training.md` for the tap contract, loss, block-16 rationale, memory
strategy, NInfer conversion and why branch-aware runtime tree verification is a separate target
state transaction.
