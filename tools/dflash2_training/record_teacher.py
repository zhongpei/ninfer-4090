#!/usr/bin/env python3
"""Record DFlash2 teacher data from a Hugging Face target model.

Each output sample stores:
  ids      int32 [N]
  fused    bf16  [N, len(taps)*H]  residual streams entering the requested target layers
  label    int32 [N]               target greedy next-token argmax
  top_ids  int32 [N,K]
  top_lp   fp16  [N,K]             target top-K log-probabilities

For standard Transformers causal models, output_hidden_states returns embeddings as element 0 and
the residual after each decoder layer thereafter. hidden_states[L] is therefore the residual
entering layer L, which matches the DFlash2 target_layer_ids convention.
"""
from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_taps(text: str) -> list[int]:
    taps = [int(x) for x in text.split(",") if x.strip()]
    if not taps or taps != sorted(set(taps)):
        raise ValueError("--taps must be a strictly increasing comma-separated list")
    return taps


@torch.inference_mode()
def record(model, ids: torch.Tensor, taps: list[int], topk: int) -> dict:
    out = model(input_ids=ids[None], use_cache=False, output_hidden_states=True,
                return_dict=True)
    hidden = out.hidden_states
    if hidden is None or max(taps) >= len(hidden):
        raise RuntimeError(
            f"teacher returned {0 if hidden is None else len(hidden)} hidden-state entries; "
            f"cannot collect taps {taps}")
    fused = torch.cat([hidden[layer][0] for layer in taps], dim=-1)
    logits = out.logits[0].float()
    logp = torch.log_softmax(logits, dim=-1)
    vals, idx = torch.topk(logp, min(topk, logp.shape[-1]), dim=-1)
    return {
        "ids": ids.to(torch.int32).cpu(),
        "fused": fused.to(torch.bfloat16).cpu(),
        "label": idx[:, 0].to(torch.int32).cpu(),
        "top_ids": idx.to(torch.int32).cpu(),
        "top_lp": vals.to(torch.float16).cpu(),
    }


def read_records(paths: list[str]) -> list[dict]:
    records: list[dict] = []
    for filename in paths:
        with open(filename, encoding="utf-8") as f:
            for line_no, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                value = json.loads(line)
                if isinstance(value, str):
                    value = {"text": value}
                if not isinstance(value, dict) or not isinstance(value.get("text"), str):
                    raise ValueError(f"{filename}:{line_no}: expected JSON object with text")
                records.append(value)
    return records


def render_prompt(tokenizer, record: dict, chat: bool, thinking: bool) -> list[int]:
    text = record["text"]
    if chat:
        text = tokenizer.apply_chat_template(
            [{"role": record.get("role", "user"), "content": text}],
            tokenize=False, add_generation_prompt=True, enable_thinking=thinking)
    return tokenizer(text, add_special_tokens=False).input_ids


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True, help="HF target model directory/repository")
    ap.add_argument("--input", action="append", required=True,
                    help="JSONL with text/topic/kind/split fields; repeatable")
    ap.add_argument("--out", required=True)
    ap.add_argument("--taps", default="5,19,33,47,61")
    ap.add_argument("--topk", type=int, default=64)
    ap.add_argument("--max-len", type=int, default=2048)
    ap.add_argument("--generate-new", type=int, default=0,
                    help="for kind=gen records, append this many greedy target tokens before recording")
    ap.add_argument("--device-map", default="auto",
                    help="Transformers device_map; auto can use both GPUs and host RAM")
    ap.add_argument("--dtype", default="bfloat16",
                    choices=("bfloat16", "float16", "float32"))
    ap.add_argument("--trust-remote-code", action="store_true")
    ap.add_argument("--chat", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--thinking", action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument("--holdout-fraction", type=float, default=0.1,
                    help="when split is absent, fraction assigned to heldout by seeded shuffle")
    ap.add_argument("--seed", type=int, default=20260930)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    if not 0.0 <= args.holdout_fraction < 1.0:
        raise SystemExit("--holdout-fraction must be in [0,1)")
    if args.topk <= 0:
        raise SystemExit("--topk must be positive")

    taps = parse_taps(args.taps)
    dtype = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[args.dtype]

    tokenizer = AutoTokenizer.from_pretrained(
        args.model, trust_remote_code=args.trust_remote_code)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=dtype, device_map=args.device_map,
        trust_remote_code=args.trust_remote_code, low_cpu_mem_usage=True)
    model.eval()
    input_device = model.get_input_embeddings().weight.device

    records = read_records(args.input)
    if args.limit:
        records = records[:args.limit]
    rng = random.Random(args.seed)
    inferred = [i for i, r in enumerate(records) if "split" not in r]
    rng.shuffle(inferred)
    held_n = int(round(len(inferred) * args.holdout_fraction))
    held = set(inferred[:held_n])

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest: list[dict] = []

    eos = tokenizer.eos_token_id
    for index, item in enumerate(records):
        name = str(item.get("name", f"seq-{index:06d}"))
        kind = str(item.get("kind", "corpus"))
        topic = str(item.get("topic", "prose"))
        split = str(item.get("split", "heldout" if index in held else "train"))

        prompt_ids = render_prompt(tokenizer, item, args.chat, args.thinking)
        ids = torch.tensor(prompt_ids[:args.max_len], dtype=torch.long, device=input_device)
        gen_start = 0
        if kind == "gen" and args.generate_new > 0:
            gen_start = int(ids.numel())
            budget = min(args.generate_new, max(0, args.max_len - gen_start))
            if budget:
                generated = model.generate(
                    input_ids=ids[None], do_sample=False, max_new_tokens=budget,
                    eos_token_id=eos, pad_token_id=eos)
                ids = generated[0]
        if ids.numel() < 2:
            continue
        ids = ids[:args.max_len]

        blob = record(model, ids, taps, args.topk)
        blob.update({
            "name": name,
            "kind": kind,
            "topic": topic,
            "split": split,
            "gen_start": gen_start,
            "target_layer_ids": taps,
        })
        torch.save(blob, out_dir / f"{name}.pt")
        manifest.append({
            "name": name,
            "kind": kind,
            "topic": topic,
            "split": split,
            "n": int(ids.numel()),
            "gen_start": gen_start,
        })
        print(f"[{index+1}/{len(records)}] {name} {kind} {split} {ids.numel()} tokens",
              flush=True)

    with open(out_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump({
            "version": 1,
            "model": args.model,
            "target_layer_ids": taps,
            "topk": args.topk,
            "sequences": manifest,
        }, f, indent=2)
    print(f"wrote {len(manifest)} sequences to {out_dir}", flush=True)


if __name__ == "__main__":
    main()
