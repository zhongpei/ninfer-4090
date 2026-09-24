# Ternary Bonsai 2 27B, NInfer v3 artifact

A [NInfer](https://github.com/Neroued/ninfer) **v3** artifact of PrismML's
[Ternary Bonsai 2 27B](https://huggingface.co/prism-ml/Ternary-Bonsai-2-27B-gguf) for the
**franken line** of NInfer-3090: [iamwavecut/ninfer-3090](https://github.com/iamwavecut/ninfer-3090),
branch `franken/v0.11`. That line is ashalliants' v0.11.0 plus ternary weights and
Hadamard-rotated checkpoints; stock NInfer builds refuse this file (`t2_g128_fp16` and the
`hadamard_signs` Use auxiliary are franken additions).

This revision needs `franken/v0.11` at
[`3fb3d33e`](https://github.com/iamwavecut/ninfer-3090/commit/3fb3d33e) or later, and DFlash2
through its proposal head (`--spec dflash2 --lm-head-draft`) needs
[`ce02e2cf`](https://github.com/iamwavecut/ninfer-3090/commit/ce02e2cf) or later; the numbers
below are from [`897f89a3`](https://github.com/iamwavecut/ninfer-3090/commit/897f89a3).
Its DFlash2 adapter's output and down projections are Q4 and reach the residual through a path
older builds lack: builds from
[`1e9e01e1`](https://github.com/iamwavecut/ninfer-3090/commit/1e9e01e1) on load the text, Vision
and MTP profiles but refuse `--spec dflash2` at startup with `q4 linear: unsupported shape`.
On those builds the previous revision,
[8beb07b](https://huggingface.co/WaveCut/Ternary-Bonsai-2-27B-NInfer-v3/tree/8beb07b18ed45f937871a6be2614952980322e6f),
runs every profile. Builds older than `1e9e01e1` refuse both at startup with
`embedding: unsupported table qtype`.

## What is inside

| component | representation |
|---|---|
| text projections (64 layers), output head and token embedding | `t2_g128_fp16`: PrismML's ternary rows imported without rounding (2-bit codes, one fp16 scale per 128 columns), Hadamard-rotated; every projection's Use carries the sign vector of its input width, and the engine restores each gathered token row with the hidden-width signs |
| GDN A/B controls, norms, convolution, `A_log`, `dt_bias` | BF16/FP32, restored from llama.cpp's exporter conventions (grouped value heads, `w` instead of `1 + w`, `A_log` from `-exp(A_log)`) |
| MTP head | [ProCreations/Ternary-Bonsai-2-27B-MTP](https://huggingface.co/ProCreations/Ternary-Bonsai-2-27B-MTP): Qwen3.8-27B's head fine-tuned on frozen Bonsai 2 features, Q8 |
| DFlash2 adapter | [ProCreations/Ternary-Bonsai-2-27B-DFlash2](https://huggingface.co/ProCreations/Ternary-Bonsai-2-27B-DFlash2): z-lab's adapter trained against Bonsai 2; Q4, with its fused query/key/value projection in Q8 |
| proposal head | the output head's own `t2_g128_fp16` rows for the 131,072 most frequent tokens of NInfer's token ranking, copied byte for byte with the head's rotation |
| Vision tower | vanilla Qwen3.8-27B (PrismML's mmproj carries the same weights) |
| chat template | NInfer's pinned `qwen3_8.jinja` |

With `--lm-head-draft` the drafters score their candidates on the proposal head instead of the
full one. MTP decodes 4% faster through it and accepts the same share of its drafts; DFlash2
gains nothing measurable, so its recommended profile leaves the flag off. The head is loaded only
with the flag.

One `.ninfer` file of 9,520,051,456 bytes (8.87 GiB), next to its conversion report,
`SHA256SUMS` and `NOTICE`.

Conversion command, from the franken tree (add `--device cuda` to encode the Q8 parts on a GPU):

```bash
python3 -m tools.convert \
  --model Qwen3.8-27B \
  --recipe bonsai2_27b_ternary \
  --source ternary=Ternary-Bonsai-2-27B-PQ2_0.gguf \
  --source mtp=Ternary-Bonsai-2-27B-MTP/model_mtp.safetensors \
  --source dflash2=Ternary-Bonsai-2-27B-DFlash2 \
  --components text,vision,mtp,dflash2 \
  --resource chat_template.jinja=tools/chat_templates/qwen3_8.jinja \
  --proposal \
  --name bonsai2-27b \
  --out Ternary-Bonsai-2-27B-ninfer-v3.ninfer
```

## Running

Build the `franken/v0.11` branch (see its README). Every profile below uses the same base command;
add the flags of the row you want:

```bash
ninfer-serve Ternary-Bonsai-2-27B-ninfer-v3.ninfer --model-id bonsai2-27b \
  --max-context 198400 --kv-capacity 198400 --kv-dtype rk8v4 --gdn-state-fp16
```

Device memory with one request lane and a 198,400-token `rk8v4` KV cache (`nvidia-smi`, the whole
process):

| configuration | add flags | weights | VRAM in use |
|---|---|---:|---:|
| text only | none | 6.70 GiB | 12.13 GiB |
| Vision, tower resident | `--vision --vision-residency resident --vision-max-merged 12288` | 6.97 GiB | 12.86 GiB |
| Vision, tower in host memory (overlay) | `--vision --vision-residency overlay --vision-max-merged 8192` | 6.70 GiB | 12.16 GiB |
| MTP | `--spec mtp --draft-tokens 3` | 7.12 GiB | 12.86 GiB |
| MTP + Vision overlay | `--spec mtp --draft-tokens 3 --vision --vision-residency overlay --vision-max-merged 12288` | 7.12 GiB | 12.89 GiB |
| MTP through the proposal head | `--spec mtp --draft-tokens 3 --lm-head-draft` | 7.28 GiB | 13.02 GiB |
| MTP through the proposal head + Vision overlay | `--spec mtp --draft-tokens 3 --lm-head-draft --vision --vision-residency overlay --vision-max-merged 12288` | 7.28 GiB | 13.05 GiB |
| DFlash2 | `--spec dflash2 --draft-tokens 7` | 7.99 GiB | 13.61 GiB |
| DFlash2 + Vision overlay | `--spec dflash2 --draft-tokens 7 --vision --vision-residency overlay --vision-max-merged 12288` | 7.99 GiB | 13.64 GiB |

The MTP head costs 0.73 GiB (0.42 GiB of weights plus the KV and graphs of its attention layer),
the proposal head another 0.16 GiB when `--lm-head-draft` loads it, the DFlash2 adapter 1.48 GiB,
and the Vision tower 0.73 GiB resident or 0.03 GiB in overlay. In
overlay the tower lives in pinned host memory, and an image borrows device memory for its encode
from the output head, the token table and the loaded drafter, which are restored afterwards. The
ternary head and table alone are too small for the 12,288-token window, so overlay without MTP or
DFlash2 takes `--vision-max-merged 8192` (or the resident tower); with either drafter loaded,
12,288 works.

The fastest single-stream profile is DFlash2 with seven drafts and the Vision tower in overlay;
MTP runs best with `--lm-head-draft`:

```bash
ninfer-serve Ternary-Bonsai-2-27B-ninfer-v3.ninfer --model-id bonsai2-27b \
  --max-context 198400 --kv-capacity 198400 --kv-dtype rk8v4 --gdn-state-fp16 \
  --spec dflash2 --draft-tokens 7 \
  --vision --vision-residency overlay --vision-max-merged 12288
```

### How much context fits on a 24 GiB card

With DFlash2, the Vision tower in overlay and `rk8v4` KV, the model's whole 262,144-token window
fits with room to spare. Let the engine size the cache:

```bash
ninfer-serve Ternary-Bonsai-2-27B-ninfer-v3.ninfer --model-id bonsai2-27b \
  --max-context 262144 --kv-capacity auto --kv-dtype rk8v4 --gdn-state-fp16 \
  --spec dflash2 --draft-tokens 7 \
  --vision --vision-residency overlay --vision-max-merged 12288
```

| `--max-concurrency` | KV cache from `--kv-capacity auto` | VRAM in use |
|---:|---|---:|
| 1 | 262,144 tokens, the whole window; 8.4 GiB stay free | 15.19 GiB |
| 2 | 519,744 tokens shared by the two requests | 21.75 GiB |
| 4 | 461,248 tokens shared by the four requests | 20.91 GiB |

`auto` takes the largest cache that leaves 1 GiB of headroom; one lane stops at the window. Each
`rk8v4` token costs 26,112 bytes of KV. With MTP instead of DFlash2 one lane leaves 9.0 GiB free,
and text alone 9.9 GiB.

## Results

One RTX 3090 at its full 350 W power limit, one request at a time, the validation flags of both
revisions (198,400-token window, `rk8v4` KV, Vision in overlay, four lanes). This revision on
`franken/v0.11` 897f89a3 ran interleaved with the previous one on `7826626d`, on the same card,
in two rounds; decode is each suite's generated tokens over its decode time. MTP runs through the
proposal head (`--lm-head-draft`) on this revision and through the full head on the previous one.

| profile | revision | decode, short chat | decode, GSM8K answers | tokens per step, GSM8K | VRAM in use |
|---|---|---:|---:|---:|---:|
| no speculation | this | 91.4 tok/s | 90.5 tok/s | 1 | 12.6 GiB |
| | previous | 87.5 tok/s | 86.4 tok/s | 1 | 12.6 GiB |
| MTP, three drafts | this | 199.1 tok/s | 212.2 tok/s | 3.1 | 13.5 GiB |
| | previous | 183.9 tok/s | 196.2 tok/s | 3.1 | 13.4 GiB |
| DFlash2, seven drafts | this | 237.2 tok/s | 295.1 tok/s | 4.7 | 14.5 GiB |
| | previous | 224.6 tok/s | 283.8 tok/s | 4.7 | 14.5 GiB |

Prefill runs at 1,730 tok/s on prompts of about 1,000 tokens, 1,794 at 8K, 1,505 at 32K and 1,234 at
64K (1,698, 1,769, 1,496 and 1,230 before). With MTP, decode holds 176 tok/s at 8K of context, 137
at 32K and 116 at 64K (165, 122 and 105 before); per speculative step, DFlash2 is 3% faster up to 8K
of context and 7 to 9% faster at 32K and 64K.

Quality on a fixed 1,179-item slice, greedy decoding with no sampling penalties, against the
previous revision on its engine:

| suite | items | this revision, MTP | this revision, DFlash2 | previous revision, MTP | previous revision, DFlash2 |
|---|---:|---:|---:|---:|---:|
| GSM8K | 200 | 95.5 | 96.5 | 96.0 | 95.0 |
| MMLU-Pro | 280 | 73.2 | 73.6 | 74.3 | 73.9 |
| HumanEval | 164 | 90.9 | 90.9 | 90.2 | 90.2 |
| MGSM, Russian | 250 | 90.4 | 90.0 | 90.8 | 90.0 |
| Global-MMLU, Russian | 285 | 78.6 | 79.3 | 77.5 | 79.3 |
| pooled | 1,179 | 84.4 | 84.7 | 84.5 | 84.5 |

Under MTP the two revisions disagree on 27 items, 13 now solved and 14 now missed (sign test p =
1.0); under DFlash2 on 29, 16 now solved and 13 now missed (p = 0.71). That is the churn small
numeric changes cause on long reasoning chains: greedy speculation keeps the target's own tokens,
but the int8 heads and the width of each verify step move its logits by rounding. GSM8K scores 95.5
under MTP and 96.5 under DFlash2 against 94.5 for llama.cpp on the source GGUF.

Perplexity (`ninfer-perplexity --quick`, `ninfer-ppl-1m-v1`, 4,096-token windows at a 2,048 stride,
`rk8v4`) is 5.630: 7.820 WikiText, 8.663 PG-19, 7.903 Chinese Wikipedia, 1.861 code (5.630 before;
the int8 heads move it by 0.004%). The dense Qwen3.8-27B NInfer artifact measures 4.346 under the
same settings: the ternary projections cost about 30% perplexity at 2.125 bits per weight, against
4.25 and 5.25 in the dense artifact.

### Long context and concurrent requests

A retrieval check plants three codes at a third, two thirds and nine tenths of a prose document,
asks for all three, then for the last one alone. With DFlash2 on one lane of the 262,144-token
window the model returns all three, in order, at 32K, 131K and 250K tokens, and the last one each
time. The 250K document prefills at 613 tok/s, where attention over the growing cache
dominates, and the answer still decodes at 71 tok/s.

Several requests at once, on this revision's engine with the Vision tower in overlay (32 GSM8K
questions, 512 new tokens each; generated tokens over wall time, prompts included):

| profile | 1 at a time | 2 at once | 4 at once |
|---|---:|---:|---:|
| MTP, three drafts | 183 tok/s | 285 tok/s | 442 tok/s |
| DFlash2, seven drafts | 258 tok/s | 323 tok/s | 424 tok/s |
| no speculation | 86 tok/s | 151 tok/s | 275 tok/s |

With four requests in flight MTP overtakes DFlash2, whose eight tokens per verified request make
each batched step twice as wide as MTP's four.

