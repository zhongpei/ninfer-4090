#!/usr/bin/env python3
"""Offline DFlash2 fine-tuning against recorded target taps.

The target model is not resident during training. record_teacher.py has already materialised the
five target residual streams plus target argmax/top-K rows. The only target tensors loaded here are
embed_tokens and lm_head, because DFlash2 uses them to embed the mask block and score draft hidden.

Objective per live draft row:
    CE(draft_logits, target_argmax) +
    kl_weight * KL(target_topK || draft_logits)

The held-out gate replays the serving loop and reports accepted tokens/block, which is the metric
that matters to NInfer speculation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from tools.dflash2_training.model import (
    DFlash2Module,
    export_checkpoint,
    load_config,
    load_weight_tensors,
    target_embedding_and_head,
)


class Sample:
    def __init__(self, meta: dict, blob: dict, keep_on: str, device: str):
        self.name = meta["name"]
        self.kind = meta.get("kind", "corpus")
        self.topic = meta.get("topic", "prose")
        self.split = meta.get("split", "train")
        self.gen_start = int(meta.get("gen_start", 0))
        where = device if keep_on == "cuda" else "cpu"
        self.ids = blob["ids"].to(where).long()
        self.fused = blob["fused"].to(where)
        self.label = blob["label"].to(where).long()
        self.top_ids = blob["top_ids"].to(where).long()
        self.top_lp = blob["top_lp"].to(where)

    def __len__(self):
        return int(self.ids.numel())


def load_data(path: str, keep_on: str, device: str, limit: int = 0) -> list[Sample]:
    with open(os.path.join(path, "manifest.json"), encoding="utf-8") as f:
        manifest = json.load(f)
    metas = manifest["sequences"]
    if limit:
        metas = metas[:limit]
    out = []
    for meta in metas:
        filename = os.path.join(path, meta["name"] + ".pt")
        if not os.path.exists(filename):
            continue
        out.append(Sample(meta, torch.load(filename, map_location="cpu"), keep_on, device))
    return out


def fingerprint(samples: list[Sample]) -> str:
    h = hashlib.sha256()
    for s in samples:
        h.update(s.name.encode())
        h.update(s.split.encode())
        h.update(str(len(s)).encode())
    return h.hexdigest()[:16]


def block_masks(anchor_pos: torch.Tensor, ctx_len: int, block: int, window: int | None,
                device: str):
    b = anchor_pos.numel()
    t = b * block
    qblk = torch.arange(t, device=device) // block
    qpos = anchor_pos[qblk] + (torch.arange(t, device=device) % block)
    ctx_pos = torch.arange(ctx_len, device=device)
    kpos = torch.cat([ctx_pos, qpos])
    kblk = torch.cat([torch.full((ctx_len,), -1, device=device, dtype=torch.long), qblk])
    is_ctx = torch.arange(ctx_len + t, device=device) < ctx_len
    same_block = kblk[None, :] == qblk[:, None]
    before_anchor = kpos[None, :] < anchor_pos[qblk][:, None]
    full = torch.where(is_ctx[None, :], before_anchor, same_block)
    if window is None:
        return full, full
    in_window = (qpos[:, None] - kpos[None, :]) < window
    slide = torch.where(is_ctx[None, :], before_anchor & in_window, same_block)
    return full, slide


def run_blocks(module: DFlash2Module, embed: torch.Tensor, sample: Sample,
               anchors: torch.Tensor, device: str, block: int):
    cfg = module.cfg
    b = anchors.numel()
    ctx_len = int(anchors.max().item())
    fused = sample.fused[:ctx_len].to(device, non_blocking=True)
    ctx_hidden = module.project_context(fused.to(module.dtype))
    ctx_pos = torch.arange(ctx_len, device=device)
    ctx_kv = module.context_kv(ctx_hidden, ctx_pos)

    ids = sample.ids.to(device, non_blocking=True)
    block_ids = torch.full((b, block), cfg.mask_token_id, dtype=torch.long, device=device)
    block_ids[:, 0] = ids[anchors]
    noise = F.embedding(block_ids.reshape(-1), embed).to(module.dtype)
    positions = (anchors[:, None] + torch.arange(block, device=device)[None, :]).reshape(-1)
    masks = block_masks(anchors, ctx_len, block, cfg.sliding_window, device)
    hidden = module.forward_block(
        noise, positions, ctx_kv, ctx_pos, block_size=block, masks=masks)
    return hidden.view(b, block, -1)[:, 1:, :]


def loss_of(pred: torch.Tensor, head: torch.Tensor, sample: Sample,
            anchors: torch.Tensor, kl_weight: float, device: str):
    b, length, hidden = pred.shape
    logits = F.linear(pred.reshape(-1, hidden), head).float()
    slots = anchors[:, None] + torch.arange(1, length + 1, device=device)[None, :]
    idx = (slots - 1).reshape(-1)
    labels = sample.label.to(device, non_blocking=True)[idx]
    ce = F.cross_entropy(logits, labels)
    hits = (logits.argmax(-1) == labels).float().sum()
    if kl_weight <= 0:
        return ce, hits, ce.detach(), torch.zeros((), device=device)

    top_ids = sample.top_ids.to(device, non_blocking=True)[idx]
    top_lp = sample.top_lp.to(device, non_blocking=True)[idx].float()
    p = torch.softmax(top_lp, dim=-1)
    q = torch.log_softmax(logits, dim=-1).gather(-1, top_ids)
    kl = (p * (torch.log(p.clamp_min(1e-9)) - q)).sum(-1).mean()
    return ce + kl_weight * kl, hits, ce.detach(), kl.detach()


@torch.inference_mode()
def acceptance(module: DFlash2Module, embed: torch.Tensor, head: torch.Tensor,
               samples: list[Sample], device: str, block: int, max_blocks: int):
    cfg = module.cfg
    buckets: dict[str, list[float]] = {}
    for sample in samples:
        n = len(sample)
        fused = sample.fused.to(device, non_blocking=True)
        ids = sample.ids.to(device, non_blocking=True)
        labels = sample.label.to(device, non_blocking=True)
        ctx_hidden = module.project_context(fused.to(module.dtype))
        ctx_pos_all = torch.arange(n, device=device)
        kv_all = module.context_kv(ctx_hidden, ctx_pos_all)

        p = max(1, sample.gen_start - 1)
        values: list[float] = []
        while p <= n - block and len(values) < max_blocks:
            lo = 0 if cfg.sliding_window is None else max(0, p - cfg.sliding_window + 1)
            ctx_kv = [(k[:, lo:p], v[:, lo:p]) for k, v in kv_all]
            ctx_pos = ctx_pos_all[lo:p]
            block_ids = torch.full((block,), cfg.mask_token_id, dtype=torch.long, device=device)
            block_ids[0] = ids[p]
            noise = F.embedding(block_ids, embed).to(module.dtype)
            positions = torch.arange(p, p + block, device=device)
            pred = module.forward_block(noise, positions, ctx_kv, ctx_pos, block_size=block)[1:]
            logits = F.linear(pred.to(head.dtype), head).float()

            if cfg.selector_rank:
                cand, unary = module.unary_candidates(logits)
                scores = module.lattice(pred, cand, unary, int(ids[p]))
                draft = module.walk(cand, scores)
            else:
                draft = logits.argmax(-1)

            want = labels[p:p + block - 1]
            mismatch = draft != want
            accepted = int(mismatch.float().argmax()) if mismatch.any() else block - 1
            values.append(float(accepted + 1))
            p += accepted + 1
        klass = "code" if sample.topic == "code" else (
            "de" if sample.topic in ("de", "multilingual") else "prose")
        buckets.setdefault(klass, []).extend(values)

    result = {name: sum(values) / len(values) for name, values in buckets.items() if values}
    all_values = [v for values in buckets.values() for v in values]
    result["ALL"] = sum(all_values) / max(1, len(all_values))
    result["blocks"] = float(len(all_values))
    return result


def select_parameters(module: DFlash2Module, mode: str):
    cfg = module.cfg
    if mode == "all":
        names = [name for name, _ in module.named_checkpoint_parameters()
                 if not name.startswith("candidate_selector.")]
    elif mode == "fc":
        names = ["fc.weight", "hidden_norm.weight", "norm.weight"]
    elif mode.startswith("last"):
        count = int(mode[4:])
        first = max(0, cfg.num_hidden_layers - count)
        names = ["fc.weight", "hidden_norm.weight", "norm.weight"]
        names += [
            name for name, _ in module.named_checkpoint_parameters()
            if name.startswith("layers.") and int(name.split(".")[1]) >= first
        ]
    else:
        raise ValueError("--train must be fc, lastN, or all")

    selected = set(names)
    params = []
    for name, parameter in module.named_checkpoint_parameters():
        parameter.requires_grad_(name in selected)
        if parameter.requires_grad:
            params.append(parameter)
    return names, params


def make_optimizer(params, args):
    if args.optimizer == "adamw8bit":
        try:
            import bitsandbytes as bnb
        except ImportError as exc:
            raise SystemExit("adamw8bit requires bitsandbytes") from exc
        return bnb.optim.PagedAdamW8bit(
            params, lr=args.lr, weight_decay=args.weight_decay,
            betas=(0.9, 0.95), eps=1e-8)
    return torch.optim.AdamW(
        params, lr=args.lr, weight_decay=args.weight_decay,
        betas=(0.9, 0.95), eps=1e-8)


def save_resume(path: str, module: DFlash2Module, optimizer, step: int, best: float,
                rng: random.Random, data_fp: str, args):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    trainable = {
        name: p.detach().cpu()
        for name, p in module.named_checkpoint_parameters() if p.requires_grad
    }
    torch.save({
        "version": 1,
        "weights": trainable,
        "optimizer": optimizer.state_dict(),
        "step": step,
        "best": best,
        "rng": rng.getstate(),
        "torch_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "data_fingerprint": data_fp,
        "block": args.block,
        "train": args.train,
    }, path)


def load_resume(path: str, module: DFlash2Module, optimizer, rng: random.Random, data_fp: str):
    blob = torch.load(path, map_location="cpu")
    if blob.get("data_fingerprint") != data_fp:
        raise RuntimeError("resume data fingerprint does not match current training set")
    weights = blob["weights"]
    for name, parameter in module.named_checkpoint_parameters():
        if name in weights:
            parameter.data.copy_(weights[name].to(parameter.device, dtype=parameter.dtype))
    optimizer.load_state_dict(blob["optimizer"])
    rng.setstate(blob["rng"])
    torch.set_rng_state(blob["torch_rng"])
    if blob.get("cuda_rng") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(blob["cuda_rng"])
    return int(blob["step"]), float(blob["best"])


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--drafter", required=True, help="released/base DFlash2 HF checkpoint")
    ap.add_argument("--target", required=True, help="target HF checkpoint containing embed/head")
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--block", type=int, default=16,
                    help="training block width; 16 trains 15 proposed tokens")
    ap.add_argument("--train", default="last1",
                    help="fc, lastN, or all; last1 is the 24GB-friendly starting point")
    ap.add_argument("--optimizer", default="adamw8bit", choices=("adamw8bit", "adamw"))
    ap.add_argument("--steps", type=int, default=10000)
    ap.add_argument("--blocks", type=int, default=4,
                    help="packed anchors from one sequence per optimizer step")
    ap.add_argument("--lr", type=float, default=1.5e-4)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--warmup", type=int, default=250)
    ap.add_argument("--kl", type=float, default=0.1)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--gen-weight", type=float, default=0.5)
    ap.add_argument("--eval-every", type=int, default=250)
    ap.add_argument("--eval-blocks", type=int, default=64)
    ap.add_argument("--keep-on", choices=("cpu", "cuda"), default="cpu")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--resume", default="")
    ap.add_argument("--resume-every", type=int, default=500)
    ap.add_argument("--eval-only", action="store_true")
    args = ap.parse_args()

    if args.block < 2 or args.block > 16:
        raise SystemExit("--block must be in [2,16] for current NInfer DFlash2 K<=15")
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)

    cfg, snapshot = load_config(args.drafter)
    tensors = load_weight_tensors(snapshot, dtype=torch.bfloat16, device=args.device)
    module = DFlash2Module(cfg, tensors).to(args.device)
    embed, head = target_embedding_and_head(args.target, args.device)

    data = load_data(args.data, args.keep_on, args.device, args.limit)
    train = [s for s in data if s.split == "train"]
    held = [s for s in data if s.split == "heldout"]
    if not train or not held:
        raise SystemExit("manifest must contain both train and heldout sequences")
    train_gen = [s for s in train if s.kind == "gen"]
    train_corpus = [s for s in train if s.kind != "gen"]
    data_fp = fingerprint(data)

    names, params = select_parameters(module, args.train)
    print(f"train={args.train} tensors={len(names)} params="
          f"{sum(p.numel() for p in params)/1e9:.3f}B block={args.block}", flush=True)

    before = acceptance(module, embed, head, held, args.device, args.block, args.eval_blocks)
    print("accept/block before " +
          " ".join(f"{k}={v:.3f}" for k, v in sorted(before.items())), flush=True)
    if args.eval_only:
        out_dir = Path(args.out)
        out_dir.mkdir(parents=True, exist_ok=True)
        with open(out_dir / "training_metrics.json", "w", encoding="utf-8") as f:
            json.dump({
                "version": 1,
                "mode": "eval-only",
                "block": args.block,
                "train": args.train,
                "optimizer": args.optimizer,
                "data_fingerprint": data_fp,
                "before": before,
                "final": before,
                "best": float(before["ALL"]),
                "drafter": args.drafter,
                "target": args.target,
            }, f, indent=2)
        return

    optimizer = make_optimizer(params, args)
    start_step = 0
    best = float(before["ALL"])
    if args.resume and os.path.exists(args.resume):
        start_step, best = load_resume(args.resume, module, optimizer, rng, data_fp)
        print(f"resume step={start_step} best={best:.3f}", flush=True)

    hist: list[tuple[float, float, float, float]] = []
    started = time.perf_counter()
    for step in range(start_step + 1, args.steps + 1):
        phase = min(1.0, step / max(1, args.steps))
        warm = min(1.0, step / max(1, args.warmup))
        lr_scale = warm * (0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * phase)))
        for group in optimizer.param_groups:
            group["lr"] = args.lr * lr_scale

        pool = train_gen if (train_gen and train_corpus and rng.random() < args.gen_weight)             else (train_corpus or train_gen)
        sample = pool[rng.randrange(len(pool))]
        lo = max(1, sample.gen_start - 1) if sample.kind == "gen" else 1
        hi = len(sample) - args.block
        if hi <= lo:
            continue
        candidates = range(lo, hi + 1)
        count = min(args.blocks, hi - lo + 1)
        anchors = torch.tensor(sorted(rng.sample(list(candidates), count)),
                               dtype=torch.long, device=args.device)

        with torch.autocast("cuda", dtype=torch.bfloat16,
                            enabled=str(args.device).startswith("cuda")):
            pred = run_blocks(module, embed, sample, anchors, args.device, args.block)
            loss, hits, ce, kl = loss_of(
                pred, head, sample, anchors, args.kl, args.device)
        loss.backward()
        grad = torch.nn.utils.clip_grad_norm_(params, args.clip)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        rows = anchors.numel() * (args.block - 1)
        row_hit = float(hits) / max(1, rows)
        hist.append((float(loss.detach()), row_hit, float(ce), float(kl)))
        if step == 1 or step % 25 == 0:
            window = hist[-25:]
            print(
                f"step={step} loss={sum(x[0] for x in window)/len(window):.4f} "
                f"ce={sum(x[2] for x in window)/len(window):.4f} "
                f"kl={sum(x[3] for x in window)/len(window):.4f} "
                f"row_hit={sum(x[1] for x in window)/len(window):.3f} "
                f"grad={float(grad):.2f} lr={optimizer.param_groups[0]['lr']:.2e} "
                f"elapsed={(time.perf_counter()-started)/60:.1f}m",
                flush=True)

        if args.eval_every and step % args.eval_every == 0:
            score = acceptance(
                module, embed, head, held, args.device, args.block, args.eval_blocks)
            print(f"eval@{step} " +
                  " ".join(f"{k}={v:.3f}" for k, v in sorted(score.items())), flush=True)
            if score["ALL"] > best:
                best = float(score["ALL"])
                export_checkpoint(module, snapshot, args.out, args.block)
                print(f"saved best={best:.3f} -> {args.out}", flush=True)

        if args.resume and args.resume_every and step % args.resume_every == 0:
            save_resume(args.resume, module, optimizer, step, best, rng, data_fp, args)

    final = acceptance(module, embed, head, held, args.device, args.block, args.eval_blocks)
    if final["ALL"] >= best:
        export_checkpoint(module, snapshot, args.out, args.block)
        best = float(final["ALL"])
    print("accept/block final " +
          " ".join(f"{k}={v:.3f}" for k, v in sorted(final.items())), flush=True)
    print(f"best={best:.3f} checkpoint={args.out}", flush=True)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "training_metrics.json", "w", encoding="utf-8") as f:
        json.dump({
            "version": 1,
            "mode": "train",
            "block": args.block,
            "train": args.train,
            "optimizer": args.optimizer,
            "steps": args.steps,
            "data_fingerprint": data_fp,
            "before": before,
            "final": final,
            "best": best,
            "drafter": args.drafter,
            "target": args.target,
        }, f, indent=2)
    if args.resume:
        save_resume(args.resume, module, optimizer, args.steps, best, rng, data_fp, args)


if __name__ == "__main__":
    main()
