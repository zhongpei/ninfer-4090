# Huihui Ternary Bonsai replacement

This converter keeps an existing NInfer v3 artifact intact and replaces only
its text tower with the supplied PrismML `PQ2_0` GGUF. The existing vision
tower, MTP, DFlash2 adapter, tokenizer resources, proposal configuration, and
all unrelated tensor objects are retained. The proposal head and shared
Hadamard-sign auxiliaries are regenerated because they are coupled to the text
output head. The Huihui file's rotated Q2_K/Q3_K fallback matrices are decoded
with llama.cpp-compatible semantics and materialised into the existing NInfer
T2 slots.

Validate the local inputs first:

```bash
python3 tools/convert/huihui_bonsai/convert.py --validate-only
```

Convert to a new artifact (the source artifact is never overwritten):

```bash
python3 tools/convert/huihui_bonsai/convert.py \
  --base-artifact /opt/ninfer-4090/Ternary-Bonsai-2-27B-ninfer-v3.ninfer \
  --gguf /opt/llama.cpp-Ternary-Bonsai-2-27B/models/Huihui-Qwen3.8-27B-abliterated-Ternary-Bonsai-PQ2_0.gguf \
  --out /opt/ninfer-4090/out/huihui-qwen3.8-27b-abliterated.ninfer
```

The conversion uses CUDA by default. Use `--device cpu` only for the
structure/binding validation path; it is not intended as the production
conversion path for this 27B artifact.
