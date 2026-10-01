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

## Native NInfer teacher data

The Hugging Face recorder remains useful for broad data generation, but a quantized/Ternary NInfer
target does not necessarily have the same target distribution as its BF16 source. Use
`ninfer-teacher` when training the drafter that will actually ship:

```bash
ninfer-teacher out/bonsai2-target.ninfer \
  --input prompts.jsonl --out train/ninfer-teacher \
  --devices 0,1 --max-context 8192 --kv-dtype int8 \
  --layers 5,19,33,47,61 --top-k 16
```

For N input tokens the exporter records N-1 predictor rows. Each row contains:

- the input token at that predictor;
- raw BF16 bits for the selected target-layer residual taps;
- actual target argmax;
- top-16 ids;
- exact top-16 log-probabilities computed from the represented dense BF16 target logits.

The exporter uses `EnginePurpose::CausalScoring` with an explicit teacher capability. It owns
fresh StateImage/KV state, does not enter the serving context cache, and reuses one bounded pinned
host tap buffer per prefill chunk. Generation engines do not reserve this teacher workspace.

Input JSONL accepts either `{"text":"..."}` or `{"tokens":[...]}`, plus optional
`name/kind/topic/split`. The output manifest/raw-array format is consumed directly by
`train.py`.

The differentiable trainer still needs the source target embedding/lm-head tensors through
`--target`. The *teacher labels/taps* come from the real `.ninfer` artifact; this is the
important part for matching T2/Q5/Ternary target behavior.

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

## 4. Runtime A/B

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

## Tree planning

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
