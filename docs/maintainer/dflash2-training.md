# DFlash2 training and tree calibration

This repository includes a standalone training path adapted from TandemLLM. The goal is not to
train a language model. It fine-tunes the DFlash2 proposal model against the exact target
distribution that NInfer will verify.

The pipeline is deliberately split into four stages so each expensive asset can be reused:

```text
HF target
   |
   +-- record_teacher.py
   |      ids + 5 target taps + target argmax + target top-K
   v
recorded dataset
   |
   +-- train.py
   |      CE(target argmax) + KL(target top-K || draft)
   v
DFlash2 companion checkpoint
   |
   +-- export_ninfer.py
   v
.ninfer artifact
```

A separate lattice path records DFlash2 candidate tables and evaluates Tandem-style speculative
trees offline before any branch-aware target runtime is added.

## 1. Record target data

### Preferred: record the actual loaded `.ninfer` target

For Ternary Bonsai / mixed T2+Q5 work, the strongest teacher is the artifact NInfer actually
serves, not an assumed BF16/HF target. The CLI can record the five DFlash target taps and the
prepared target head's stable top-16 during root prefill:

```bash
ninfer-cli out/bonsai2-dflash2-b16.ninfer \
  --prompt "$(cat corpus.txt)" \
  --spec dflash2 --draft-tokens 15 \
  --dflash-teacher-out train/native-teacher \
  --max-context 8192 --max-new 1
```

Teacher recording requires context-cache reuse to be disabled so every file starts at token zero.
The single-request CLI already uses that root-prefill behavior. The recorder is an offline path:
it computes target-head top-16 one token at a time and writes host files, so it intentionally favors
fidelity and bounded VRAM over serving throughput.

Convert the resulting `.ndft` files into the dataset consumed by `train.py`:

```bash
python -m tools.dflash2_training.import_ninfer_teacher \
  --input train/native-teacher \
  --out train/ninfer-data \
  --topic code \
  --holdout-every 10
```

The native file records:

```text
ids
absolute positions
5 x target residual taps
target-head top-16 global token ids
target-head top-16 FP32 scores
```

The top-16 scores come from the exact prepared NInfer head (including the artifact's actual
weight representation). The importer stores them in the trainer's `top_lp` field; only a
per-row softmax is used by the truncated KL objective, so raw scores and normalized
log-probabilities differ only by an irrelevant additive constant.

**Embedding/head compatibility:** the standalone PyTorch trainer still borrows embedding/head
weights from `--target`. For a native-teacher run, point `--target` at the HF/base checkpoint
that corresponds to the `.ninfer` artifact. The hard labels and soft top-16 distribution come
from the actual NInfer target; the HF/base embedding/head supplies the differentiable training
projection. If a future artifact intentionally changes the vocabulary embedding/head independently
of that base checkpoint, export of those prepared tensors would be required for a fully
self-contained trainer.

### Alternative: Hugging Face teacher recorder

Prepare JSONL, one object per source sequence:

```json
{"name":"code-001","kind":"gen","topic":"code","text":"Review this C++ function..."}
{"name":"doc-001","kind":"corpus","topic":"prose","text":"Long public text..."}
```

Then:

```bash
python -m tools.dflash2_training.record_teacher \
  --model /models/Qwen3.8-27B \
  --input prompts.jsonl \
  --out train/dflash2-data \
  --taps 5,19,33,47,61 \
  --topk 64 \
  --generate-new 256 \
  --device-map auto
```

`device-map auto` is useful on the 2x4090 + large-host-RAM setup: collection needs the full
target, but training does not.

The recorder uses Transformers `output_hidden_states`. Element 0 is the embedding residual and
element L is the residual entering decoder layer L, so taps 5/19/33/47/61 match the DFlash2
checkpoint contract.

Two sequence types are useful:

- **gen**: prompt plus the target's own greedy continuation. This matches serving distribution.
- **corpus**: teacher-forced public/code text. This is much cheaper and broadens coverage.

Held-out samples are marked in `manifest.json` and must not be used for optimizer steps.

## 2. Fine-tune the drafter

A 24GB card should start with a partial mode:

```bash
python -m tools.dflash2_training.train \
  --drafter /models/z-lab-Qwen3.8-27B-DFlash2 \
  --target /models/Qwen3.8-27B \
  --data train/dflash2-data \
  --out train/dflash2-b16 \
  --block 16 \
  --train last1 \
  --optimizer adamw8bit \
  --steps 10000 \
  --resume train/dflash2-b16.resume.pt
```

Training modes:

| mode | moving tensors | use |
|---|---|---|
| `fc` | fusion projection + outer norms | cheapest probe |
| `last1` / `last2` | fusion + final N DFlash2 layers | 24GB-friendly starting point |
| `all` | all backbone tensors except selector codebooks | full fine-tune; needs more memory |

The selector codebooks remain frozen by default. They are large sparse-lookup tables and their
gradient density is poor at normal batch sizes.

The trainer can use bitsandbytes `PagedAdamW8bit`; choose `--optimizer adamw` when
bitsandbytes is unavailable.

### Objective

For a block anchored at p, row 0 carries the already-known anchor. Rows 1..K are trained against:

```text
CE(draft_logits[j], target_argmax[p+j])
+ kl_weight * KL(target_topK[p+j] || draft_logits[j])
```

The hard CE term is exactly the greedy speculative acceptance event. The top-K soft term preserves
the target's uncertainty instead of overfitting only one token.

### Block 16

The DFlash2 architecture is not intrinsically limited to eight rows. `--block 16` trains 15 draft
positions and exports `dflash_config.block_size=16`. NInfer already admits DFlash2
`--draft-tokens 1..15`, so a b16 checkpoint can serve K3/K7/K11/K15 from one resident drafter.

## 3. Convert to NInfer

The trainer exports a normal companion checkpoint:

```text
config.json
model.safetensors
```

For the current Ternary Bonsai source, keep the Qwen3.8 HF directory as the primary
configuration/resource source and pass the GGUF as the recipe's named `ternary` source:

```bash
python -m tools.dflash2_training.export_ninfer \
  --model /models/Qwen3.8-27B \
  --ternary /models/Ternary-Bonsai-2-27B-PQ2_0.gguf \
  --drafter train/dflash2-b16 \
  --components text,dflash2 \
  --out out/bonsai2-dflash2-b16.ninfer
```

For an ordinary Qwen3.8 safetensors source:

```bash
python -m tools.dflash2_training.export_ninfer \
  --model /models/Qwen3.8-27B \
  --drafter train/dflash2-b16 \
  --recipe qwen3_8_27b \
  --components text,dflash2 \
  --out out/qwen38-dflash2-b16.ninfer
```

The existing converter maps the companion names directly to:

```text
dflash2/feature_projection
dflash2/layers/*/attention_conv/*
dflash2/layers/*/mlp_conv/*
dflash2/candidate_selector/*
```

No new artifact format is introduced.

## 4. Gate the converted NInfer artifact

Training loss and Hugging Face-side acceptance are not the release gate for a T2/Q5/Ternary
target. After conversion, compare the released and custom companions through the actual NInfer
runtime:

```bash
python -m tools.dflash2_training.ninfer_gate \
  --ninfer build-ninja/apps/ninfer \
  --artifact released=out/bonsai2-released-dflash2.ninfer \
  --artifact custom=out/bonsai2-dflash2-b16.ninfer \
  --prompts train/heldout-prompts.jsonl \
  --out results/dflash2-ninfer-gate.csv \
  --draft-tokens 15 --tree
```

The gate records decode throughput, accepted tokens, tokens/round, tree statistics and the output
SHA256 for every prompt. This is where conversion/quantization/runtime effects are judged; the HF
teacher-side metric remains a training signal.

## 5. Runtime A/B

Combine the trained b16 checkpoint with PR #4/#5 controls:

```bash
ninfer-serve out/bonsai2-dflash2-b16.ninfer \
  --spec dflash2 --draft-tokens 15 \
  --spec-router stair \
  --lookup-ngram 5 --lookup-strategy vote --lookup-dflash skip
```

Useful arms are:

1. released DFlash2 fixed K;
2. trained b16 fixed K;
3. trained b16 + Stair;
4. trained b16 + lookup replace;
5. trained b16 + lookup head-skip/deep;
6. trained b16 + Stair + lookup.

Keep the target artifact, prompt set, KV format and sampling profile fixed.

## 6. Tree planning

TandemLLM also gets additional acceptance by verifying multiple branches in one target pass. NInfer
now carries the offline part of that work:

```bash
python -m tools.dflash2_training.record_lattice \
  --drafter train/dflash2-b16 \
  --target /models/Qwen3.8-27B \
  --data train/dflash2-data \
  --out train/lattice.jsonl \
  --block 16

python -m tools.dflash2_training.calibrate_tree \
  train/lattice.jsonl --budgets 3,7,11,15,23,31
```

`tree_policy.py` constructs a greedy spine first and then spends the remaining budget best-first
on alternative nodes by cumulative path probability.

### Why runtime tree verification is not switched on in this PR

The target is not a plain transformer. NInfer must keep Paged KV, GDN/ReplaySSM, continuation
StateImage, partial commit, stochastic acceptance and CUDA Graph topology consistent. A branch
therefore needs a parent-indexed version of all target recurrent state, followed by path-only
commit and discard.

Implementing a tree as if only KV mattered would be incorrect. This PR imports the proposal and
calibration side now and leaves runtime tree verification for the branch-aware state transaction
change, instead of exposing a flag whose output/state contract is incomplete.

## Provenance

The training objective, tap contract, block-16 strategy, held-out accepted-tokens/block gate, and
lattice/tree-planning approach are adapted from 0xBakeer/TandemLLM at commit
`c11aecaa7ba767642409ff51d90d48c79c572d3a`. The repository is AGPL-3.0-only.
