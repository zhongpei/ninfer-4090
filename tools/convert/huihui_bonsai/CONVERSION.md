# Huihui GGUF to NInfer Conversion Guide

## Purpose

`convert.py` replaces the text portion of an existing NInfer artifact with the
Huihui PrismML GGUF. It does not rebuild or overwrite the complete model. The
source NInfer artifact retains:

- the vision tower and vision resources;
- MTP and DFlash2 weights and configuration;
- the tokenizer, chat template, and generation configuration; and
- every unselected tensor, directory binding, and execution use.

The converter regenerates:

- the `text/` text tower;
- the `proposal/` proposal head, when the source artifact contains one; and
- the Hadamard-sign auxiliary used by the text tower.

The `PQ2_0` tensors in the Huihui GGUF are imported directly. A small number
of rotated matrices use llama.cpp `Q2_K`/`Q3_K`; the converter decodes them
with llama.cpp-compatible rules and materializes them into the existing
NInfer `t2_g128_fp16` objects.

## Inputs

The default paths on this machine are:

```text
source artifact: /opt/ninfer-4090/Ternary-Bonsai-2-27B-ninfer-v3.ninfer
GGUF:            /opt/llama.cpp-Ternary-Bonsai-2-27B/models/Huihui-Qwen3.8-27B-abliterated-Ternary-Bonsai-PQ2_0.gguf
```

Use `--base-artifact` and `--gguf` to select different paths. The source
artifact and GGUF must describe the same Qwen3.8-27B Dense geometry.

## Conversion

Run the non-writing structural validation first:

```bash
/opt/minicoda3/bin/python3 tools/convert/huihui_bonsai/convert.py \
  --validate-only --device cpu
```

After confirming that GPU0 is free, run the full conversion. This example
writes the result beside the source NInfer model in the repository root:

```bash
CUDA_VISIBLE_DEVICES=0 /opt/minicoda3/bin/python3 \
  tools/convert/huihui_bonsai/convert.py \
  --base-artifact /opt/ninfer-4090/Ternary-Bonsai-2-27B-ninfer-v3.ninfer \
  --gguf /opt/llama.cpp-Ternary-Bonsai-2-27B/models/Huihui-Qwen3.8-27B-abliterated-Ternary-Bonsai-PQ2_0.gguf \
  --out /opt/ninfer-4090/huihui-qwen3.8-27b-abliterated.ninfer
```

The conversion also writes:

```text
/opt/ninfer-4090/huihui-qwen3.8-27b-abliterated.ninfer.conversion.json
```

The tool refuses to overwrite an existing output artifact or report. The
source NInfer artifact is never modified.

## Verification

Inspect the converted artifact:

```bash
/opt/minicoda3/bin/python3 -m tools.artifact.inspect \
  /opt/ninfer-4090/huihui-qwen3.8-27b-abliterated.ninfer --json
```

Run a minimal text-generation check:

```bash
CUDA_VISIBLE_DEVICES=0 build/apps/ninfer \
  /opt/ninfer-4090/huihui-qwen3.8-27b-abliterated.ninfer \
  --prompt "Reply with exactly OK." \
  --max-context 512 --max-new 16 --kv-dtype int8 --greedy --no-thinking
```

Test the retained DFlash2 path:

```bash
CUDA_VISIBLE_DEVICES=0 build/apps/ninfer \
  /opt/ninfer-4090/huihui-qwen3.8-27b-abliterated.ninfer \
  --prompt "Reply with exactly DFLASH2_OK." \
  --max-context 512 --max-new 16 --kv-dtype int8 --greedy --no-thinking \
  --spec dflash2 --draft-tokens 3
```

## Limitations

To match the existing NInfer artifact layout, `Q2_K`/`Q3_K` tensors are
currently requantized to T2; they are not stored in native llama.cpp
Q2_K/Q3_K format. Full conversion requires CUDA and sufficient VRAM.
`--validate-only --device cpu` is a validation path only, not the production
conversion path.
