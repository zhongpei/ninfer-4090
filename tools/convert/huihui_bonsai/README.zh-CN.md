# Huihui Ternary Bonsai 替换工具

这个转换工具保留现有 NInfer v3 工件，只使用指定的 PrismML `PQ2_0` GGUF
替换文本塔。视觉塔、MTP、DFlash2 适配器、tokenizer 资源、proposal 配置及
其他无关 tensor 对象都会保留。由于文本输出头发生变化，proposal head 和共享
Hadamard-sign auxiliary 会重新生成。Huihui 文件中的旋转 `Q2_K`/`Q3_K` fallback
矩阵会按照 llama.cpp 兼容规则解码，再写入现有 NInfer T2 对象。

先校验本地输入：

```bash
python3 tools/convert/huihui_bonsai/convert.py --validate-only
```

转换到新工件（不会覆盖源工件）：

```bash
python3 tools/convert/huihui_bonsai/convert.py \
  --base-artifact /opt/ninfer-4090/Ternary-Bonsai-2-27B-ninfer-v3.ninfer \
  --gguf /opt/llama.cpp-Ternary-Bonsai-2-27B/models/Huihui-Qwen3.8-27B-abliterated-Ternary-Bonsai-PQ2_0.gguf \
  --out /opt/ninfer-4090/huihui-qwen3.8-27b-abliterated.ninfer
```

转换默认使用 CUDA。`--device cpu` 只用于结构和 binding 校验，不是这个
27B 工件的生产转换路径。

## 文档

- [英文入口](README.md)
- [英文完整转换说明](CONVERSION.md)
- [中文完整转换说明](CONVERSION.zh-CN.md)
