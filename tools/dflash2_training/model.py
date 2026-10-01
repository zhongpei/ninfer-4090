"""Standalone DFlash2 reference module used by NInfer training tools.

Adapted from 0xBakeer/TandemLLM engine/drafters/dflash2.py under AGPL-3.0-only.
It intentionally contains no Tandem engine dependency.  The exported checkpoint keeps the original
z-lab/HuggingFace tensor names so NInfer's existing tools.convert DFlash2 companion loader can
consume it unchanged.
"""
from __future__ import annotations

import glob
import json
import math
import os
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors import safe_open


class DFlash2Config:
    def __init__(self, raw: dict):
        d = raw["dflash_config"]
        self.raw = raw
        self.hidden_size = int(raw["hidden_size"])
        self.intermediate_size = int(raw["intermediate_size"])
        self.num_hidden_layers = int(raw["num_hidden_layers"])
        self.num_attention_heads = int(raw["num_attention_heads"])
        self.num_key_value_heads = int(raw["num_key_value_heads"])
        self.head_dim = int(raw.get("head_dim", self.hidden_size // self.num_attention_heads))
        self.vocab_size = int(raw["vocab_size"])
        self.rms_norm_eps = float(raw["rms_norm_eps"])
        self.rope_theta = float(raw.get("rope_parameters", {}).get("rope_theta", 1e7))
        self.layer_types = list(raw.get("layer_types", ["full_attention"] * self.num_hidden_layers))
        self.sliding_window = int(raw.get("sliding_window") or 0) or None
        self.is_causal = bool(raw.get("is_causal", False))
        bs = d.get("block_size", raw.get("block_size"))
        if bs is None:
            raise KeyError("DFlash2 config has no block_size")
        self.block_size = int(bs)
        self.conv_kernel_size = int(d.get("conv_kernel_size", 0))
        self.conv_group_size = int(d.get("conv_group_size", 0))
        self.mask_token_id = int(d["mask_token_id"])
        self.selector_rank = int(d.get("selector_rank", 0))
        self.selector_top_k = int(d.get("selector_top_k", 0))
        self.target_layer_ids = [int(x) for x in d["target_layer_ids"]]
        self.output_multiplier = float(d.get("output_multiplier", 1.0))
        cap = float(d.get("final_logit_softcapping") or 0.0)
        self.final_logit_softcapping = cap if cap > 0 else None

    @property
    def num_groups(self) -> int:
        return self.hidden_size // self.conv_group_size

    @property
    def kv_dim(self) -> int:
        return self.num_key_value_heads * self.head_dim

    @property
    def q_dim(self) -> int:
        return self.num_attention_heads * self.head_dim

    def is_sliding(self, layer: int) -> bool:
        return self.layer_types[layer] == "sliding_attention"

    def expected_tensors(self) -> dict[str, tuple[int, ...]]:
        h, hd = self.hidden_size, self.head_dim
        out = {
            "fc.weight": (h, len(self.target_layer_ids) * h),
            "hidden_norm.weight": (h,),
            "norm.weight": (h,),
        }
        if self.selector_rank:
            out["candidate_selector.hidden_projection.weight"] = (self.selector_rank, h)
            out["candidate_selector.predecessor_codebook"] = (self.vocab_size, self.selector_rank)
            out["candidate_selector.successor_codebook"] = (self.vocab_size, self.selector_rank)
        for i in range(self.num_hidden_layers):
            p = f"layers.{i}"
            out[f"{p}.input_layernorm.weight"] = (h,)
            out[f"{p}.post_attention_layernorm.weight"] = (h,)
            out[f"{p}.self_attn.q_proj.weight"] = (self.q_dim, h)
            out[f"{p}.self_attn.k_proj.weight"] = (self.kv_dim, h)
            out[f"{p}.self_attn.v_proj.weight"] = (self.kv_dim, h)
            out[f"{p}.self_attn.o_proj.weight"] = (h, self.q_dim)
            out[f"{p}.self_attn.q_norm.weight"] = (hd,)
            out[f"{p}.self_attn.k_norm.weight"] = (hd,)
            out[f"{p}.mlp.gate_proj.weight"] = (self.intermediate_size, h)
            out[f"{p}.mlp.up_proj.weight"] = (self.intermediate_size, h)
            out[f"{p}.mlp.down_proj.weight"] = (h, self.intermediate_size)
            if self.conv_kernel_size:
                rows = 2 * self.conv_kernel_size * self.num_groups
                for name in ("attention_conv", "mlp_conv"):
                    out[f"{p}.{name}.base_kernel"] = (2, self.conv_kernel_size, h)
                    out[f"{p}.{name}.kernel_projection.weight"] = (rows, h)
        return out


def resolve_checkpoint(path: str | os.PathLike) -> str:
    path = os.path.expanduser(os.fspath(path))
    if os.path.isfile(os.path.join(path, "config.json")):
        return path
    hits = sorted(glob.glob(os.path.join(path, "*", "config.json")))
    if not hits:
        raise FileNotFoundError(f"no config.json under {path}")
    return os.path.dirname(hits[0])


def load_config(path: str | os.PathLike) -> tuple[DFlash2Config, str]:
    snap = resolve_checkpoint(path)
    with open(os.path.join(snap, "config.json"), encoding="utf-8") as f:
        raw = json.load(f)
    return DFlash2Config(raw), snap


def load_weight_tensors(snapshot: str, dtype=torch.bfloat16, device="cpu") -> dict[str, torch.Tensor]:
    out: dict[str, torch.Tensor] = {}
    for path in sorted(glob.glob(os.path.join(snapshot, "*.safetensors"))):
        with safe_open(path, framework="pt", device="cpu") as f:
            for name in f.keys():
                t = f.get_tensor(name)
                if t.is_floating_point() and t.dtype != dtype:
                    t = t.to(dtype)
                out[name] = t.to(device)
    if not out:
        raise FileNotFoundError(f"no safetensors under {snapshot}")
    return out


def target_embedding_and_head(snapshot: str, device="cuda") -> tuple[torch.Tensor, torch.Tensor]:
    """Load only the target embedding and lm_head instead of making the 27B target resident."""
    want: dict[str, torch.Tensor | None] = {"embed_tokens.weight": None, "lm_head.weight": None}
    for path in sorted(glob.glob(os.path.join(snapshot, "*.safetensors"))):
        with safe_open(path, framework="pt", device="cpu") as f:
            keys = set(f.keys())
            for role in tuple(want):
                if want[role] is not None:
                    continue
                for candidate in (
                    role,
                    f"model.{role}",
                    f"model.language_model.{role}",
                ):
                    if candidate in keys:
                        want[role] = f.get_tensor(candidate).to(torch.bfloat16).to(device)
                        break
    if want["embed_tokens.weight"] is None:
        raise RuntimeError(f"no embed_tokens.weight under {snapshot}")
    if want["lm_head.weight"] is None:
        want["lm_head.weight"] = want["embed_tokens.weight"]
    return want["embed_tokens.weight"], want["lm_head.weight"]


def _rms(x: torch.Tensor, w: torch.Tensor, eps: float) -> torch.Tensor:
    dtype = x.dtype
    y = x.float()
    y = y * torch.rsqrt(y.pow(2).mean(-1, keepdim=True) + eps)
    return y.to(dtype) * w


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    a, b = x.chunk(2, dim=-1)
    return torch.cat([-b, a], dim=-1)


def _rope_tables(positions: torch.Tensor, dim: int, theta: float, dtype: torch.dtype):
    inv = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float32,
                                        device=positions.device) / dim))
    f = positions.float()[:, None] * inv[None, :]
    emb = torch.cat([f, f], dim=-1)
    return emb.cos().to(dtype), emb.sin().to(dtype)


def _apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    return x * cos[None, None] + _rotate_half(x) * sin[None, None]


def _grouped_conv(h: torch.Tensor, delta: torch.Tensor, base: torch.Tensor,
                  num_groups: int, group_size: int, taps: int,
                  block_pos: torch.Tensor) -> torch.Tensor:
    blocks = h.unflatten(-1, (num_groups, group_size))
    coef = base.view(1, taps, num_groups, group_size) + delta.unsqueeze(-1)
    out = coef[:, 0] * blocks
    for tap in range(1, taps):
        shifted = F.pad(blocks[:-tap], (0, 0, 0, 0, tap, 0))
        out = out + coef[:, tap] * shifted * (block_pos >= tap).view(-1, 1, 1)
    return out.flatten(-2)


class DFlash2Module(nn.Module):
    """Trainable standalone DFlash2. State-dict keys are kept identical to the source checkpoint."""

    def __init__(self, cfg: DFlash2Config, tensors: dict[str, torch.Tensor]):
        super().__init__()
        self.cfg = cfg
        expected = cfg.expected_tensors()
        missing = sorted(set(expected) - set(tensors))
        if missing:
            raise KeyError(f"DFlash2 checkpoint missing tensors: {missing[:8]}")
        self._names: dict[str, str] = {}
        for i, (name, t) in enumerate(tensors.items()):
            internal = f"p_{i}"
            self.register_parameter(internal, nn.Parameter(t, requires_grad=False))
            self._names[name] = internal

    def weight(self, name: str) -> torch.Tensor:
        return getattr(self, self._names[name])

    def named_checkpoint_parameters(self):
        for external, internal in self._names.items():
            yield external, getattr(self, internal)

    @property
    def device(self):
        return self.weight("fc.weight").device

    @property
    def dtype(self):
        return self.weight("fc.weight").dtype

    def rope(self, positions: torch.Tensor, dim: int, dtype: torch.dtype):
        return _rope_tables(positions, dim, self.cfg.rope_theta, dtype)

    def project_context(self, target_hidden: torch.Tensor) -> torch.Tensor:
        expected = len(self.cfg.target_layer_ids) * self.cfg.hidden_size
        if target_hidden.ndim != 2 or target_hidden.shape[-1] != expected:
            raise ValueError(f"target_hidden must be [N,{expected}]")
        return _rms(F.linear(target_hidden, self.weight("fc.weight")),
                    self.weight("hidden_norm.weight"), self.cfg.rms_norm_eps)

    def context_kv(self, ctx_hidden: torch.Tensor, positions: torch.Tensor):
        cfg = self.cfg
        n, hd, nkv = ctx_hidden.shape[0], cfg.head_dim, cfg.num_key_value_heads
        cos, sin = self.rope(positions, hd, ctx_hidden.dtype)
        out = []
        for i in range(cfg.num_hidden_layers):
            p = f"layers.{i}.self_attn"
            k = F.linear(ctx_hidden, self.weight(f"{p}.k_proj.weight")).view(n, nkv, hd)
            k = _rms(k, self.weight(f"{p}.k_norm.weight"), cfg.rms_norm_eps)
            v = F.linear(ctx_hidden, self.weight(f"{p}.v_proj.weight")).view(n, nkv, hd)
            k = _apply_rope(k.transpose(0, 1)[None], cos, sin)[0]
            out.append((k, v.transpose(0, 1)))
        return out

    def _conv(self, prefix: str, h: torch.Tensor, block_pos: torch.Tensor, side: int):
        cfg = self.cfg
        kp = self.weight(prefix + ".kernel_projection.weight")
        base = self.weight(prefix + ".base_kernel")[side]
        coef = F.linear(h, kp).reshape(h.shape[0], 2, cfg.conv_kernel_size, cfg.num_groups)
        return _grouped_conv(h, coef[:, side], base, cfg.num_groups, cfg.conv_group_size,
                             cfg.conv_kernel_size, block_pos), coef[:, 1 - side]

    def _masks(self, positions: torch.Tensor, ctx_positions: torch.Tensor | None, t: int):
        cfg = self.cfg
        if ctx_positions is None or ctx_positions.numel() == 0 or cfg.sliding_window is None:
            return None, None
        allpos = torch.cat([ctx_positions, positions])
        delta = positions[:, None] - allpos[None, :]
        return None, delta < cfg.sliding_window

    def forward_block(self, noise_emb: torch.Tensor, positions: torch.Tensor,
                      ctx_kv, ctx_positions: torch.Tensor | None,
                      block_size: int | None = None,
                      masks: tuple[torch.Tensor | None, torch.Tensor | None] | None = None):
        cfg = self.cfg
        bs = block_size or cfg.block_size
        t = noise_emb.shape[0]
        hd, nh, nkv = cfg.head_dim, cfg.num_attention_heads, cfg.num_key_value_heads
        rep = nh // nkv
        block_pos = torch.arange(t, device=noise_emb.device) % bs
        cos, sin = self.rope(positions, hd, noise_emb.dtype)
        if masks is None:
            masks = self._masks(positions, ctx_positions, t)

        h = noise_emb
        for i in range(cfg.num_hidden_layers):
            p = f"layers.{i}"
            res = h
            x = _rms(h, self.weight(f"{p}.input_layernorm.weight"), cfg.rms_norm_eps)
            attn_delta = None
            if cfg.conv_kernel_size:
                raw = F.linear(x, self.weight(f"{p}.attention_conv.kernel_projection.weight"))
                raw = raw.reshape(t, 2, cfg.conv_kernel_size, cfg.num_groups)
                x = _grouped_conv(x, raw[:, 0], self.weight(f"{p}.attention_conv.base_kernel")[0],
                                  cfg.num_groups, cfg.conv_group_size, cfg.conv_kernel_size,
                                  block_pos)
                attn_delta = raw[:, 1]

            q = F.linear(x, self.weight(f"{p}.self_attn.q_proj.weight")).view(t, nh, hd)
            q = _rms(q, self.weight(f"{p}.self_attn.q_norm.weight"), cfg.rms_norm_eps)
            k = F.linear(x, self.weight(f"{p}.self_attn.k_proj.weight")).view(t, nkv, hd)
            k = _rms(k, self.weight(f"{p}.self_attn.k_norm.weight"), cfg.rms_norm_eps)
            v = F.linear(x, self.weight(f"{p}.self_attn.v_proj.weight")).view(t, nkv, hd)
            q = _apply_rope(q.transpose(0, 1)[None], cos, sin)
            k = _apply_rope(k.transpose(0, 1)[None], cos, sin)
            v = v.transpose(0, 1)[None]
            if ctx_kv is not None:
                ck, cv = ctx_kv[i]
                k = torch.cat([ck[None], k], dim=2)
                v = torch.cat([cv[None], v], dim=2)
            o = F.scaled_dot_product_attention(
                q, k.repeat_interleave(rep, dim=1), v.repeat_interleave(rep, dim=1),
                attn_mask=masks[1] if cfg.is_sliding(i) else masks[0])
            o = o.transpose(1, 2).reshape(t, -1)
            o = F.linear(o, self.weight(f"{p}.self_attn.o_proj.weight"))
            if cfg.conv_kernel_size:
                o = _grouped_conv(o, attn_delta, self.weight(f"{p}.attention_conv.base_kernel")[1],
                                  cfg.num_groups, cfg.conv_group_size, cfg.conv_kernel_size,
                                  block_pos)
            h = res + o

            res = h
            x = _rms(h, self.weight(f"{p}.post_attention_layernorm.weight"), cfg.rms_norm_eps)
            mlp_delta = None
            if cfg.conv_kernel_size:
                raw = F.linear(x, self.weight(f"{p}.mlp_conv.kernel_projection.weight"))
                raw = raw.reshape(t, 2, cfg.conv_kernel_size, cfg.num_groups)
                x = _grouped_conv(x, raw[:, 0], self.weight(f"{p}.mlp_conv.base_kernel")[0],
                                  cfg.num_groups, cfg.conv_group_size, cfg.conv_kernel_size,
                                  block_pos)
                mlp_delta = raw[:, 1]
            g = F.linear(x, self.weight(f"{p}.mlp.gate_proj.weight"))
            u = F.linear(x, self.weight(f"{p}.mlp.up_proj.weight"))
            x = F.linear(F.silu(g) * u, self.weight(f"{p}.mlp.down_proj.weight"))
            if cfg.conv_kernel_size:
                x = _grouped_conv(x, mlp_delta, self.weight(f"{p}.mlp_conv.base_kernel")[1],
                                  cfg.num_groups, cfg.conv_group_size, cfg.conv_kernel_size,
                                  block_pos)
            h = res + x
        return _rms(h, self.weight("norm.weight"), cfg.rms_norm_eps)

    def unary_candidates(self, logits: torch.Tensor):
        vals, ids = torch.topk(logits.float(), self.cfg.selector_top_k, dim=-1)
        if self.cfg.output_multiplier != 1.0:
            vals = vals * self.cfg.output_multiplier
        if self.cfg.final_logit_softcapping is not None:
            cap = self.cfg.final_logit_softcapping
            vals = torch.tanh(vals / cap) * cap
        return ids.long(), vals

    def lattice(self, pred_hidden: torch.Tensor, candidate_ids: torch.Tensor,
                unary: torch.Tensor, anchor_id: int):
        k = self.cfg.selector_top_k
        hidden = F.linear(pred_hidden, self.weight("candidate_selector.hidden_projection.weight"))
        keys = self.weight("candidate_selector.successor_codebook")[candidate_ids]
        anchor = torch.full((1, k), int(anchor_id), dtype=torch.long, device=candidate_ids.device)
        pred_ids = torch.cat([anchor, candidate_ids[:-1]], dim=0)
        preds = self.weight("candidate_selector.predecessor_codebook")[pred_ids]
        pair = preds.float() * hidden.float()[:, None, :]
        return unary[:, None, :] + torch.einsum("lpr,lcr->lpc", pair, keys.float())

    @staticmethod
    def walk(candidate_ids: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
        idx = int(scores[0, 0].argmax())
        path = [idx]
        local = scores[1:].argmax(dim=-1)
        for row in local:
            idx = int(row[idx])
            path.append(idx)
        sel = torch.tensor(path, dtype=torch.long, device=candidate_ids.device)
        return candidate_ids.gather(-1, sel[:, None])[:, 0]


def make_module(checkpoint: str, device="cuda", train_dtype=torch.bfloat16):
    cfg, snap = load_config(checkpoint)
    raw = load_weight_tensors(snap, dtype=train_dtype, device=device)
    module = DFlash2Module(cfg, raw)
    return cfg, snap, module


def export_checkpoint(module: DFlash2Module, source_snapshot: str, out_dir: str,
                      block_size: int | None = None):
    from safetensors.torch import save_file
    import shutil

    os.makedirs(out_dir, exist_ok=True)
    for name in ("config.json", "tokenizer_config.json", "generation_config.json"):
        src = os.path.join(source_snapshot, name)
        if os.path.exists(src):
            shutil.copy(src, os.path.join(out_dir, name))
    config_path = os.path.join(out_dir, "config.json")
    if block_size is not None and os.path.exists(config_path):
        with open(config_path, encoding="utf-8") as f:
            raw = json.load(f)
        raw["dflash_config"]["block_size"] = int(block_size)
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(raw, f, indent=2)
    tensors = {
        name: p.detach().to(torch.bfloat16).contiguous().cpu()
        for name, p in module.named_checkpoint_parameters()
    }
    save_file(tensors, os.path.join(out_dir, "model.safetensors"))
