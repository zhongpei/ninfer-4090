# DFlash2 training tools

This directory contains the NInfer-side adaptation of TandemLLM's DFlash2 distillation workflow.

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

# 5. Gate and convert a trained b16 drafter. Thresholds are operator policy.
python -m tools.dflash2_training.production_gate \
  --model MODEL --drafter DFLASH2_B16 --out qwen38-b16.ninfer \
  --min-best 2.0 --min-improvement 0.1

# Optional exact greedy-output gate against the same target without speculation:
python -m tools.dflash2_training.production_gate \
  --model MODEL --drafter DFLASH2_B16 --out qwen38-b16.ninfer \
  --ninfer ./build-ninja/apps/ninfer --baseline-artifact qwen38-target.ninfer

# RTX 4090 Stair-cost conversion after running the forced-rung PowerShell sweep:
python -m tools.dflash2_training.calibrate_4090 profiles/sweeps/staircost.csv \
  --out profiles/4090-stair.json
```

See `docs/maintainer/dflash2-training.md` for the tap contract, loss, block-16 rationale, memory
strategy, NInfer conversion and why branch-aware runtime tree verification is a separate target
state transaction.


## Native NInfer teacher loop

For a quantized/Ternary target, prefer collecting teacher distributions from the actual artifact:

```bash
ninfer MODEL.ninfer \
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

The runtime recorder is disabled by default and uses only phase-reused workspace. Target-head
top-16 is produced one token at a time, so teacher collection does not scale resident GPU memory
with the corpus/prefill length.
