# Huihui GGUF → NInfer 转换说明

## 作用

`convert.py` 用 Huihui 的 PrismML GGUF 替换现有 NInfer 工件中的文本部分，
不会重新生成或覆盖整个模型。输入的旧 NInfer 工件保留以下内容：

- Vision 视觉塔和视觉资源
- MTP、DFlash2 权重与配置
- tokenizer、chat template、generation config
- 所有未选中的 tensor、目录绑定和执行 uses

会重新生成：

- `text/` 文本塔
- `proposal/` proposal head（如果旧工件包含）
- 文本使用的 Hadamard sign auxiliary

Huihui GGUF 中的 `PQ2_0` 直接导入；其中少量旋转矩阵使用 llama.cpp 的
`Q2_K`/`Q3_K`，工具会先按 llama.cpp 规则解码，再写入现有 NInfer 的
`t2_g128_fp16` 对象。

## 输入

默认路径是当前机器上的：

```text
旧工件：/opt/ninfer-4090/Ternary-Bonsai-2-27B-ninfer-v3.ninfer
GGUF：  /opt/llama.cpp-Ternary-Bonsai-2-27B/models/Huihui-Qwen3.8-27B-abliterated-Ternary-Bonsai-PQ2_0.gguf
```

也可以通过 `--base-artifact` 和 `--gguf` 指定其他路径。旧工件和 GGUF
必须对应同一套 Qwen3.8-27B Dense 几何结构。

## 转换步骤

先做不写文件的结构校验：

```bash
/opt/minicoda3/bin/python3 tools/convert/huihui_bonsai/convert.py \
  --validate-only --device cpu
```

确认 GPU 空闲后执行完整转换。下面示例使用 GPU0，输出到仓库根目录，
与原 NInfer 模型放在一起：

```bash
CUDA_VISIBLE_DEVICES=0 /opt/minicoda3/bin/python3 \
  tools/convert/huihui_bonsai/convert.py \
  --base-artifact /opt/ninfer-4090/Ternary-Bonsai-2-27B-ninfer-v3.ninfer \
  --gguf /opt/llama.cpp-Ternary-Bonsai-2-27B/models/Huihui-Qwen3.8-27B-abliterated-Ternary-Bonsai-PQ2_0.gguf \
  --out /opt/ninfer-4090/huihui-qwen3.8-27b-abliterated.ninfer
```

转换会额外写出：

```text
/opt/ninfer-4090/huihui-qwen3.8-27b-abliterated.ninfer.conversion.json
```

如果输出文件或报告已经存在，工具会拒绝覆盖；原始 NInfer 工件不会被修改。

## 转换后检查

检查工件目录：

```bash
/opt/minicoda3/bin/python3 -m tools.artifact.inspect \
  /opt/ninfer-4090/huihui-qwen3.8-27b-abliterated.ninfer --json
```

运行最小文本推理：

```bash
CUDA_VISIBLE_DEVICES=0 build/apps/ninfer \
  /opt/ninfer-4090/huihui-qwen3.8-27b-abliterated.ninfer \
  --prompt "Reply with exactly OK." \
  --max-context 512 --max-new 16 --kv-dtype int8 --greedy --no-thinking
```

测试保留的 DFlash2 路径：

```bash
CUDA_VISIBLE_DEVICES=0 build/apps/ninfer \
  /opt/ninfer-4090/huihui-qwen3.8-27b-abliterated.ninfer \
  --prompt "Reply with exactly DFLASH2_OK." \
  --max-context 512 --max-new 16 --kv-dtype int8 --greedy --no-thinking \
  --spec dflash2 --draft-tokens 3
```

## 限制

`Q2_K`/`Q3_K` 目前为了匹配原 NInfer 工件布局，会重新量化到 T2；它们不会
以 llama.cpp 原生 Q2_K/Q3_K 格式写入 NInfer。完整转换需要 CUDA 和足够的
显存；`--validate-only --device cpu` 只用于校验，不是生产转换路径。

## 文档

- [英文入口](README.md)
- [英文完整转换说明](CONVERSION.md)
- [中文入口](README.zh-CN.md)
