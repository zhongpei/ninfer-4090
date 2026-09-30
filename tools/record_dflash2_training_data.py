#!/usr/bin/env python3
"""Record DFlash2 distillation data from a target Hugging Face Qwen checkpoint.

Each recorded position stores exactly what tools/train_dflash2.py needs:

    ids[i]       token at target position i
    fused[i]     residual streams ENTERING the configured target tap layers, concatenated
    label[i]     target argmax after position i
    top_ids[i]   target top-k next-token ids
    top_lp[i]    their log probabilities

The target is run only while recording. Training later needs the target embedding/head plus these
files, not the full target model.

This is the Hugging Face recorder. It intentionally does not claim bit-identical taps against a
quantized .ninfer target: use the same HF snapshot when comparing/training, or add a native NInfer
tap recorder when the quantized target distribution itself is the teacher.

Design adapted from 0xBakeer/TandemLLM/tools/train_data.py (AGPL-3.0-only).
"""

from __future__ import annotations

import argparse
import fnmatch
import hashlib
import json
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path

import torch

from tools.dflash2_training.module import load_config

DEFAULT_GLOBS = (
    "*.py", "*.go", "*.ts", "*.tsx", "*.js", "*.rs", "*.c", "*.h", "*.cpp", "*.cu",
    "*.md", "*.txt", "*.rst", "*.toml", "*.yaml", "*.yml", "*.json", "*.sh",
)
SKIP_DIRS = {".git", "__pycache__", "node_modules", ".venv", "venv", "dist", "build"}

PLAIN_PROMPTS = (
    ("systems", "Explain why memory bandwidth can limit autoregressive LLM decoding even when "
                "there is spare compute. Separate weight traffic, KV traffic, and kernel launch cost."),
    ("systems", "Describe how a paged KV cache avoids reserving maximum context for every request. "
                "Cover page tables, fragmentation, growth, and prefix sharing."),
    ("code", "Write a small C++ example that implements a bounded ring buffer with explicit error "
             "handling. Then explain the invariants."),
    ("code", "Explain the failure modes of speculative decoding when draft acceptance falls. "
             "Distinguish correctness from throughput."),
    ("reasoning", "A benchmark became 20 percent faster after one code change. Describe the minimum "
                  "experiment needed to show the change caused the improvement."),
    ("science", "Explain the difference between capacity, bandwidth, latency, and throughput using "
                "a concrete computer-system example."),
    ("everyday", "Give a practical six-step plan for diagnosing a home network that is fast near "
                 "the router but slow in one room."),
    ("business", "Write a concise engineering decision memo comparing a faster risky implementation "
                 "with a slower implementation that is easier to verify."),
)

PASSAGE_PROMPTS = (
    ("code", "Explain what the following code/text does, identify failure modes, then preserve its "
             "identifiers in a proposed rewrite:\n\n{passage}"),
    ("summary", "Summarize the following passage, then list the details that must not be lost in a "
                "rewrite:\n\n{passage}"),
    ("editing", "Rewrite the following passage more concisely while preserving names, numbers, code "
                "identifiers and ordering:\n\n{passage}"),
)


@dataclass
class CorpusDocument:
    path: str
    ids: list[int]


class Corpus:
    def __init__(self, roots: list[str], tokenizer, globs: tuple[str, ...], max_file_bytes: int):
        self.docs: list[CorpusDocument] = []
        for raw_root in roots:
            root = os.path.expanduser(raw_root)
            for base, dirs, names in os.walk(root):
                dirs[:] = [d for d in dirs if d not in SKIP_DIRS and not d.startswith(".")]
                for name in sorted(names):
                    if not any(fnmatch.fnmatch(name, pattern) for pattern in globs):
                        continue
                    path = os.path.join(base, name)
                    try:
                        size = os.path.getsize(path)
                        if size <= 0 or size > max_file_bytes:
                            continue
                        text = Path(path).read_text(encoding="utf-8")
                    except (OSError, UnicodeDecodeError):
                        continue
                    ids = tokenizer(text, add_special_tokens=False).input_ids
                    if len(ids) >= 32:
                        self.docs.append(CorpusDocument(path, list(map(int, ids))))
        if roots and not self.docs:
            raise RuntimeError("no usable UTF-8 corpus documents found")

    def window(self, n: int, rng: random.Random) -> list[int]:
        if not self.docs:
            raise RuntimeError("a corpus window was requested without --corpus-dir")
        for _ in range(128):
            doc = self.docs[rng.randrange(len(self.docs))]
            if len(doc.ids) < n:
                continue
            start = rng.randrange(0, len(doc.ids) - n + 1)
            return doc.ids[start:start + n]
        raise RuntimeError(f"no corpus document contains a {n}-token window")

    def text_window(self, n: int, rng: random.Random, tokenizer) -> str:
        return tokenizer.decode(self.window(n, rng), skip_special_tokens=True)


def _dtype(name: str):
    return {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }[name]


def _target_input_device(model) -> torch.device:
    try:
        return model.get_input_embeddings().weight.device
    except Exception:
        return next(model.parameters()).device


def _load_target(args):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_path = os.path.expanduser(args.model)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=args.trust_remote_code)
    kwargs = {
        "torch_dtype": _dtype(args.dtype),
        "trust_remote_code": args.trust_remote_code,
        "low_cpu_mem_usage": True,
    }
    if args.device_map:
        kwargs["device_map"] = args.device_map
    model = AutoModelForCausalLM.from_pretrained(model_path, **kwargs)
    if not args.device_map:
        model.to(args.device)
    model.eval()
    return model, tokenizer


def _model_identity(model_path: str) -> dict:
    path = Path(os.path.expanduser(model_path))
    result = {"path": str(path.resolve()) if path.exists() else model_path}
    config = path / "config.json"
    if config.exists():
        result["config_sha256"] = hashlib.sha256(config.read_bytes()).hexdigest()
    return result


def _tokenizer_identity(tokenizer, model_path: str) -> dict:
    path = Path(os.path.expanduser(model_path))
    tj = path / "tokenizer.json"
    out = {"name_or_path": getattr(tokenizer, "name_or_path", model_path)}
    if tj.exists():
        out["tokenizer_sha256"] = hashlib.sha256(tj.read_bytes()).hexdigest()
    return out


@torch.inference_mode()
def record_sequence(model, ids: list[int], taps: list[int], topk: int, chunk: int,
                    input_device: torch.device) -> dict[str, torch.Tensor]:
    """Teacher-force one sequence in chunks while preserving the target cache."""
    if not ids:
        raise ValueError("cannot record an empty sequence")
    if min(taps) < 0:
        raise ValueError("target tap ids must be nonnegative")

    fused_parts: list[torch.Tensor] = []
    label_parts: list[torch.Tensor] = []
    top_ids_parts: list[torch.Tensor] = []
    top_lp_parts: list[torch.Tensor] = []
    past = None
    at = 0
    while at < len(ids):
        piece = ids[at:at + chunk]
        tokens = torch.tensor(piece, dtype=torch.long, device=input_device).unsqueeze(0)
        out = model(
            input_ids=tokens,
            past_key_values=past,
            use_cache=True,
            output_hidden_states=True,
            return_dict=True,
        )
        past = out.past_key_values
        hidden = out.hidden_states
        if hidden is None:
            raise RuntimeError("target did not return hidden states")
        # HF hidden_states[0] is embedding output and hidden_states[L] is the residual ENTERING
        # layer L (equivalently, output of layer L-1), matching NInfer/Tandem target_layer_ids.
        if max(taps) >= len(hidden):
            raise RuntimeError(
                f"tap {max(taps)} is outside target hidden-state boundaries {len(hidden)}")
        fused = torch.cat([hidden[layer][0] for layer in taps], dim=-1)
        logits = out.logits[0].float()
        lp = torch.log_softmax(logits, dim=-1)
        vals, indices = torch.topk(lp, min(topk, lp.shape[-1]), dim=-1)

        fused_parts.append(fused.to(dtype=torch.bfloat16, device="cpu"))
        label_parts.append(indices[:, 0].to(dtype=torch.int32, device="cpu"))
        top_ids_parts.append(indices.to(dtype=torch.int32, device="cpu"))
        top_lp_parts.append(vals.to(dtype=torch.float16, device="cpu"))
        at += len(piece)

    return {
        "ids": torch.tensor(ids, dtype=torch.int32),
        "fused": torch.cat(fused_parts, dim=0).contiguous(),
        "label": torch.cat(label_parts, dim=0).contiguous(),
        "top_ids": torch.cat(top_ids_parts, dim=0).contiguous(),
        "top_lp": torch.cat(top_lp_parts, dim=0).contiguous(),
    }


@torch.inference_mode()
def greedy_generate(model, tokenizer, prompt_ids: list[int], max_new: int,
                    input_device: torch.device) -> list[int]:
    tokens = torch.tensor(prompt_ids, dtype=torch.long, device=input_device).unsqueeze(0)
    eos = getattr(model.generation_config, "eos_token_id", None)
    pad = getattr(model.generation_config, "pad_token_id", None)
    if pad is None:
        pad = tokenizer.pad_token_id
    if pad is None:
        if isinstance(eos, list):
            pad = eos[0] if eos else None
        else:
            pad = eos
    generated = model.generate(
        tokens,
        max_new_tokens=max_new,
        do_sample=False,
        use_cache=True,
        eos_token_id=eos,
        pad_token_id=pad,
    )
    return [int(x) for x in generated[0].tolist()]


def _chat_prompt(tokenizer, text: str) -> list[int]:
    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": text}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    return list(map(int, tokenizer(rendered, add_special_tokens=False).input_ids))


def _load_prompt_jsonl(path: str) -> list[tuple[str, str]]:
    out = []
    if not path:
        return out
    with open(os.path.expanduser(path), encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            raw = json.loads(line)
            if isinstance(raw, str):
                out.append(("custom", raw))
                continue
            text = raw.get("text") or raw.get("prompt")
            if not isinstance(text, str) or not text:
                raise ValueError(f"{path}:{line_no}: expected non-empty text/prompt")
            out.append((str(raw.get("topic", "custom")), text))
    return out


def _atomic_json(path: str, value: dict) -> None:
    temp = path + ".tmp"
    with open(temp, "w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2)
        handle.write("\n")
    os.replace(temp, path)


def _existing_manifest(path: str) -> dict:
    if not os.path.exists(path):
        return {"version": 1, "sequences": []}
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="target Hugging Face snapshot/directory")
    ap.add_argument("--drafter", required=True,
                    help="DFlash2 checkpoint; supplies target_layer_ids and block geometry")
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--device-map", default="",
                    help="Transformers device_map, e.g. auto for dual-GPU/CPU offload")
    ap.add_argument("--dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    ap.add_argument("--trust-remote-code", action="store_true")
    ap.add_argument("--chunk", type=int, default=256)
    ap.add_argument("--topk", type=int, default=64)
    ap.add_argument("--gen", type=int, default=64)
    ap.add_argument("--gen-new", type=int, default=256)
    ap.add_argument("--corpus-seqs", type=int, default=128)
    ap.add_argument("--corpus-len", type=int, default=512)
    ap.add_argument("--passage-tokens", type=int, default=160)
    ap.add_argument("--corpus-dir", action="append", default=[])
    ap.add_argument("--corpus-globs", default=",".join(DEFAULT_GLOBS))
    ap.add_argument("--max-file-bytes", type=int, default=1_000_000)
    ap.add_argument("--prompt-jsonl", default="")
    ap.add_argument("--holdout", type=int, default=16,
                    help="number of distinct self-generated sequences reserved for eval")
    ap.add_argument("--seed", type=int, default=20260930)
    ap.add_argument("--prefix", default="train")
    ap.add_argument("--append", action="store_true")
    ap.add_argument("--resume", action="store_true",
                    help="skip sequence names that already have .pt files")
    ap.add_argument("--budget-min", type=float, default=0.0)
    args = ap.parse_args()

    if args.chunk <= 0 or args.topk <= 0 or args.gen < 0 or args.corpus_seqs < 0:
        raise SystemExit("chunk/topk must be positive and sequence counts nonnegative")
    os.makedirs(args.out, exist_ok=True)

    draft_cfg, draft_snapshot = load_config(args.drafter)
    taps = list(draft_cfg.target_layer_ids)
    model, tokenizer = _load_target(args)
    input_device = _target_input_device(model)
    rng = random.Random(args.seed)
    globs = tuple(x.strip() for x in args.corpus_globs.split(",") if x.strip())
    corpus = Corpus(args.corpus_dir, tokenizer, globs, args.max_file_bytes)
    custom_prompts = _load_prompt_jsonl(args.prompt_jsonl)

    manifest_path = os.path.join(args.out, "manifest.json")
    old = _existing_manifest(manifest_path) if args.append else {"version": 1, "sequences": []}
    existing = {m["name"]: m for m in old.get("sequences", [])}
    recorded: list[dict] = list(existing.values())
    content_hashes = {m.get("sha256") for m in recorded if m.get("sha256")}
    started = time.perf_counter()

    def over_budget() -> bool:
        return bool(args.budget_min and (time.perf_counter() - started) / 60.0 >= args.budget_min)

    def save(name: str, meta: dict, blob: dict[str, torch.Tensor]) -> bool:
        path = os.path.join(args.out, name + ".pt")
        if args.resume and name in existing and os.path.exists(path):
            return False
        digest = hashlib.sha256(blob["ids"].numpy().tobytes()).hexdigest()
        if digest in content_hashes:
            print(f"[skip duplicate] {name} {digest[:12]}", flush=True)
            return False
        torch.save(blob, path)
        content_hashes.add(digest)
        row = {**meta, "name": name, "n": int(blob["ids"].numel()), "sha256": digest}
        existing[name] = row
        recorded.append(row)
        return True

    # Self-generated traffic is the most important distribution: every hidden state after the
    # prompt was produced by the target itself.
    prompts: list[tuple[str, str]] = list(custom_prompts)
    plain_i = 0
    passage_i = 0
    while len(prompts) < args.gen:
        if corpus.docs and passage_i <= plain_i:
            topic, template = PASSAGE_PROMPTS[passage_i % len(PASSAGE_PROMPTS)]
            passage = corpus.text_window(args.passage_tokens, rng, tokenizer).strip()
            prompts.append((topic, template.format(passage=passage)))
            passage_i += 1
        else:
            topic, text = PLAIN_PROMPTS[plain_i % len(PLAIN_PROMPTS)]
            cycle = plain_i // len(PLAIN_PROMPTS)
            # The suffix prevents deterministic greedy duplicates when --gen exceeds the template
            # count while keeping the prompt domain close to ordinary chat.
            if cycle:
                text += f"\n\nUse a different concrete example for variant {cycle}."
            prompts.append((topic, text))
            plain_i += 1
    prompts = prompts[:args.gen]

    generated_rows: list[dict] = []
    for i, (topic, text) in enumerate(prompts):
        if over_budget():
            print("[budget] stopping self-generated recording", flush=True)
            break
        name = f"{args.prefix}-gen-{i:04d}"
        path = os.path.join(args.out, name + ".pt")
        if args.resume and name in existing and os.path.exists(path):
            generated_rows.append(existing[name])
            continue
        prompt_ids = _chat_prompt(tokenizer, text)
        t0 = time.perf_counter()
        full = greedy_generate(model, tokenizer, prompt_ids, args.gen_new, input_device)
        gen_seconds = time.perf_counter() - t0
        rec = record_sequence(model, full, taps, args.topk, args.chunk, input_device)
        meta = {
            "kind": "gen",
            "topic": topic,
            "gen_start": len(prompt_ids),
            "split": "train",
            "gen_tok_s": round(max(0, len(full) - len(prompt_ids)) / max(gen_seconds, 1e-9), 3),
        }
        if save(name, meta, rec):
            generated_rows.append(existing[name])
            print(f"{name} {topic:12s} {len(full):5d} tok "
                  f"gen {meta['gen_tok_s']:6.2f} tok/s", flush=True)

    # Teacher-forced public/user corpus broadens conditioning cheaply. Labels remain target argmax;
    # the corpus supplies context, not supervision.
    if args.corpus_seqs and not corpus.docs:
        raise SystemExit("--corpus-seqs is nonzero but no --corpus-dir supplied")
    for i in range(args.corpus_seqs):
        if over_budget():
            print("[budget] stopping corpus recording", flush=True)
            break
        name = f"{args.prefix}-corpus-{i:04d}"
        path = os.path.join(args.out, name + ".pt")
        if args.resume and name in existing and os.path.exists(path):
            continue
        ids = corpus.window(args.corpus_len, rng)
        rec = record_sequence(model, ids, taps, args.topk, args.chunk, input_device)
        meta = {"kind": "corpus", "topic": "corpus", "gen_start": 0, "split": "train"}
        if save(name, meta, rec) and (i % 10 == 0):
            print(f"{name} {len(ids)} tok", flush=True)

    # Holdout only distinct self-generated sequences. Corpus is never used by the exact acceptance
    # replay because it is not a target-generated continuation.
    generated = [m for m in recorded if m.get("kind") == "gen"]
    generated.sort(key=lambda m: m["name"])
    hold = set(m["name"] for m in generated[-min(args.holdout, len(generated)):])
    for m in recorded:
        m["split"] = "heldout" if m["name"] in hold else "train"

    manifest = {
        "version": 1,
        "format": "ninfer-dflash2-distillation-v1",
        "model": _model_identity(args.model),
        "tokenizer": _tokenizer_identity(tokenizer, args.model),
        "drafter": {
            "path": os.path.abspath(os.path.expanduser(draft_snapshot)),
            "target_layer_ids": taps,
            "block_size": draft_cfg.block_size,
        },
        "topk": args.topk,
        "seed": args.seed,
        "sharded": False,
        "sequences": sorted(recorded, key=lambda m: m["name"]),
    }
    _atomic_json(manifest_path, manifest)
    print(f"wrote {len(manifest['sequences'])} sequences to {args.out}; "
          f"held out {len(hold)} generated sequences", flush=True)


if __name__ == "__main__":
    main()
