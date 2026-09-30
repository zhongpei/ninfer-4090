"""Training-only DFlash2 reference math adapted from 0xBakeer/TandemLLM.

This module intentionally contains no NInfer runtime code and no Tandem server/settings dependency.
It implements the checkpoint's mathematical contract closely enough for offline recording,
fine-tuning and acceptance evaluation. Runtime conversion remains owned by tools/convert.

Source provenance: TandemLLM engine/drafters/dflash2.py, AGPL-3.0-only.
"""

from __future__ import annotations

import glob
import json
import os

import torch
import torch.nn.functional as F
from safetensors import safe_open


def resolve_checkpoint(path: str) -> str:
    """Accept a DFlash2 snapshot directory or a Hugging Face snapshots directory."""
    if not path:
        raise ValueError("a DFlash2 checkpoint path is required")
    path = os.path.expanduser(path)
    if os.path.isfile(os.path.join(path, "config.json")):
        return path
    hits = sorted(glob.glob(os.path.join(path, "*", "config.json")))
    if not hits:
        raise FileNotFoundError(f"no config.json under {path}")
    return os.path.dirname(hits[0])


class DFlash2Config:
    """The draft module's own geometry. Nothing here is read from the target."""

    def __init__(self, raw: dict):
        d = raw["dflash_config"]
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
        # `is_causal: false` -> AttentionType.ENCODER_ONLY in the serving stack.
        self.is_causal = bool(raw.get("is_causal", False))
        # z-lab's DFlash2 carries `block_size` inside `dflash_config`; the SpecForge-derived
        # checkpoints (DSpark) carry it at the top level, where their base class reads it.
        bs = d.get("block_size", raw.get("block_size"))
        if bs is None:
            raise KeyError("block_size is in neither dflash_config nor the top level")
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

    # --- derived ---
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
        """Every tensor this drafter reads, with the shape it must have."""
        h, hd = self.hidden_size, self.head_dim
        out: dict[str, tuple[int, ...]] = {
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
                taps, g = self.conv_kernel_size, self.num_groups
                for conv in ("attention_conv", "mlp_conv"):
                    out[f"{p}.{conv}.base_kernel"] = (2, taps, h)
                    out[f"{p}.{conv}.kernel_projection.weight"] = (2 * taps * g, h)
        return out



def load_config(path: str) -> tuple[DFlash2Config, str]:
    snapshot = resolve_checkpoint(path)
    with open(os.path.join(snapshot, "config.json"), encoding="utf-8") as f:
        return DFlash2Config(json.load(f)), snapshot


def load_weights(snapshot: str, device: str = "cuda",
                 dtype: torch.dtype = torch.bfloat16) -> dict[str, torch.Tensor]:
    """Load all safetensors in a standard DFlash2 checkpoint."""
    snapshot = resolve_checkpoint(snapshot)
    out: dict[str, torch.Tensor] = {}
    for path in sorted(glob.glob(os.path.join(snapshot, "*.safetensors"))):
        with safe_open(path, framework="pt", device="cpu") as f:
            for name in f.keys():
                tensor = f.get_tensor(name)
                if tensor.is_floating_point() and tensor.dtype != dtype:
                    tensor = tensor.to(dtype)
                out[name] = tensor.to(device) if device != "cpu" else tensor
    if not out:
        raise FileNotFoundError(f"no safetensors under {snapshot}")
    return out


def _lin(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    return F.linear(x, weight)


def _rms(x: torch.Tensor, w: torch.Tensor, eps: float) -> torch.Tensor:
    """Qwen3 RMS norm: `normalize(x) * w`. NOT the target's `(1 + w)` convention."""
    d = x.dtype
    y = x.float()
    y = y * torch.rsqrt(y.pow(2).mean(-1, keepdim=True) + eps)
    return y.to(d) * w


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    a, b = x.chunk(2, dim=-1)
    return torch.cat([-b, a], dim=-1)


def _rope_tables(positions: torch.Tensor, dim: int, theta: float,
                 dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
    """Neox-style full rotary over all `dim` of the head; the draft has no partial factor.

    Distinct from the target's rotary in this engine, which is partial at 0.25 over head_dim 256.
    `get_rope(head_dim, rotary_dim=head_dim, base=1e7, is_neox_style=True)` in the serving stack.
    """
    inv = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float32,
                                        device=positions.device) / dim))
    f = positions.float()[:, None] * inv[None, :]
    emb = torch.cat([f, f], dim=-1)
    return emb.cos().to(dtype), emb.sin().to(dtype)


def _apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """`x` is [B, heads, T, head_dim]; `cos`/`sin` are [T, head_dim]."""
    return x * cos[None, None] + _rotate_half(x) * sin[None, None]


def _grouped_conv(h: torch.Tensor, delta: torch.Tensor, base: torch.Tensor,
                  num_groups: int, group_size: int, taps: int,
                  block_pos: torch.Tensor) -> torch.Tensor:
    """One dynamic depthwise K-tap convolution along the token axis, inside a block.

    Port of `_grouped_conv` in the serving model, operation for operation:

        blocks = hidden_states.unflatten(-1, (num_groups, group_size))
        coefficients = base.view(1, taps, num_groups, group_size) + delta.unsqueeze(-1)
        out = coefficients[:, 0] * blocks
        position = arange(T) & (block_size - 1)
        for tap in range(1, taps):
            shifted = F.pad(blocks[:-tap], (0, 0, 0, 0, tap, 0))
            out = out + coefficients[:, tap] * shifted * (position >= tap).view(-1, 1, 1)
        return out.flatten(-2)

    Three details that are the whole thing:

      * grouping -- `unflatten(-1, (num_groups, group_size))` splits the 5120 channels into 320
        contiguous runs of 16, so channel c is in group c // 16. `base_kernel` is per *channel*
        [2, taps, 5120]; the projected `delta` is per *group* [T, taps, 320] and is broadcast
        across the 16 channels of its group. `kernel_projection` emits 2 * taps * num_groups =
        1280 features laid out [side][tap][group], which is the reshape order used here.
      * direction -- `F.pad(blocks[:-tap], (..., tap, 0))` pads the *front* of the token axis, so
        `shifted[t] = blocks[t - tap]`. The convolution looks backwards: it is causal.
      * blocking -- `position` is the index *within* the block, and the `position >= tap` mask kills
        every tap that would reach across a block boundary. A block never sees the one before it.

    The coefficient used at tap t belongs to the *current* token: the kernel is dynamic per output
    row, not a shared filter.
    """
    blocks = h.unflatten(-1, (num_groups, group_size))
    coef = base.view(1, taps, num_groups, group_size) + delta.unsqueeze(-1)
    out = coef[:, 0] * blocks
    for tap in range(1, taps):
        shifted = F.pad(blocks[:-tap], (0, 0, 0, 0, tap, 0))
        out = out + coef[:, tap] * shifted * (block_pos >= tap).view(-1, 1, 1)
    return out.flatten(-2)


class _Conv:
    """`prepare` convolves a sublayer's input and hands back the kernel `finish` applies to its
    output -- both halves from one projection of the input, which is why `kernel_projection` emits
    two sides."""

    __slots__ = ("base", "kp", "taps", "groups", "gsize")

    def __init__(self, base: torch.Tensor, kp: torch.Tensor, taps: int, groups: int, gsize: int):
        self.base, self.kp = base, kp
        self.taps, self.groups, self.gsize = taps, groups, gsize

    def prepare(self, h: torch.Tensor, block_pos: torch.Tensor):
        coef = F.linear(h, self.kp).reshape(h.shape[0], 2, self.taps, self.groups)
        return (_grouped_conv(h, coef[:, 0], self.base[0], self.groups, self.gsize,
                              self.taps, block_pos),
                coef[:, 1])

    def finish(self, h: torch.Tensor, coef: torch.Tensor, block_pos: torch.Tensor):
        return _grouped_conv(h, coef, self.base[1], self.groups, self.gsize, self.taps, block_pos)


# ----------------------------------------------------------------------------- the module

class DFlash2Module:
    """The draft module itself. Knows nothing about the engine, so the probe can drive it."""

    def __init__(self, cfg: DFlash2Config, w: dict[str, torch.Tensor]):
        self.cfg = cfg
        self.w = w
        self.convs: list[tuple[_Conv, _Conv] | None] = []
        for i in range(cfg.num_hidden_layers):
            if not cfg.conv_kernel_size:
                self.convs.append(None)
                continue
            p = f"layers.{i}"
            self.convs.append((
                _Conv(w[f"{p}.attention_conv.base_kernel"],
                      w[f"{p}.attention_conv.kernel_projection.weight"],
                      cfg.conv_kernel_size, cfg.num_groups, cfg.conv_group_size),
                _Conv(w[f"{p}.mlp_conv.base_kernel"],
                      w[f"{p}.mlp_conv.kernel_projection.weight"],
                      cfg.conv_kernel_size, cfg.num_groups, cfg.conv_group_size),
            ))

    def rope(self, positions: torch.Tensor, dim: int,
             dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
        """The rotary this checkpoint was trained with. Plain NTK-free RoPE here; a subclass whose
        config asks for a scaled rotary overrides it, and getting this wrong is silent -- the
        drafter still returns tokens, they are just the wrong ones."""
        return _rope_tables(positions, dim, self.cfg.rope_theta, dtype)

    @property
    def device(self) -> torch.device:
        return self.w["fc.weight"].device

    @property
    def dtype(self) -> torch.dtype:
        return self.w["fc.weight"].dtype

    # ---- context ---------------------------------------------------------------
    def project_context(self, target_hidden: torch.Tensor) -> torch.Tensor:
        """`hidden_norm(fc(concat of the five captured hidden states))`, [N, 5*H] -> [N, H]."""
        cfg = self.cfg
        expected = len(cfg.target_layer_ids) * cfg.hidden_size
        if target_hidden.ndim != 2 or target_hidden.shape[-1] != expected:
            raise ValueError(f"target_hidden must be [N, {expected}], got "
                             f"{tuple(target_hidden.shape)}")
        return _rms(F.linear(target_hidden, self.w["fc.weight"]),
                    self.w["hidden_norm.weight"], cfg.rms_norm_eps)

    def context_kv(self, ctx_hidden: torch.Tensor,
                   positions: torch.Tensor) -> list[tuple[torch.Tensor, torch.Tensor]]:
        """Per layer, the K and V the context contributes: `k_proj`/`v_proj` of the *same*
        projected context hidden, K through `k_norm` and then RoPE at its absolute position.

        Returns [n_layers] of (k, v), each [n_kv_heads, N, head_dim].
        """
        cfg = self.cfg
        n, hd, nkv = ctx_hidden.shape[0], cfg.head_dim, cfg.num_key_value_heads
        cos, sin = self.rope(positions, hd, ctx_hidden.dtype)
        out = []
        for i in range(cfg.num_hidden_layers):
            p = f"layers.{i}.self_attn"
            k = _lin(ctx_hidden, self.w[f"{p}.k_proj.weight"]).view(n, nkv, hd)
            k = _rms(k, self.w[f"{p}.k_norm.weight"], cfg.rms_norm_eps)
            v = _lin(ctx_hidden, self.w[f"{p}.v_proj.weight"]).view(n, nkv, hd)
            k = _apply_rope(k.transpose(0, 1)[None], cos, sin)[0]
            out.append((k, v.transpose(0, 1)))
        return out

    # ---- the block pass --------------------------------------------------------
    def forward_block(self, noise_emb: torch.Tensor, positions: torch.Tensor,
                      ctx_kv: list[tuple[torch.Tensor, torch.Tensor]] | None,
                      ctx_positions: torch.Tensor | None,
                      block_size: int | None = None, return_kv: bool = False,
                      masks: tuple[torch.Tensor | None, torch.Tensor | None] | None = None):
        """One pass of the five layers over `noise_emb` [T, H]. Returns `norm(h)`, [T, H].

        `ctx_kv[i]` is (k, v) with k already normed and RoPE'd -- the draft KV cache. `positions`
        are the block's absolute positions; `ctx_positions` are the context's, used only to build
        the sliding-window mask.

        With `return_kv`, also returns the block's own per-layer (k, v) -- normed, RoPE'd, shaped
        [n_kv_heads, T, head_dim] -- so a chained second block can attend to this one.

        `masks` overrides the pair `_masks` would build, as (full-attention, sliding). Serving never
        passes it: one block at a time needs nothing the default does not do. Training does, because
        it packs many blocks of the same sequence into one pass and those blocks must not see each
        other -- the pass is non-causal inside a block by design, and that is exactly what would
        leak between two blocks sharing a tensor.
        """
        cfg = self.cfg
        bs = block_size or cfg.block_size
        t = noise_emb.shape[0]
        hd, nh, nkv = cfg.head_dim, cfg.num_attention_heads, cfg.num_key_value_heads
        rep = nh // nkv
        dev = noise_emb.device
        block_pos = torch.arange(t, device=dev) % bs
        cos, sin = self.rope(positions, hd, noise_emb.dtype)
        if masks is None:
            masks = self._masks(positions, ctx_positions, t)

        h = noise_emb
        block_kv: list[tuple[torch.Tensor, torch.Tensor]] = []
        for i in range(cfg.num_hidden_layers):
            p = f"layers.{i}"
            conv = self.convs[i]
            res = h
            x = _rms(h, self.w[f"{p}.input_layernorm.weight"], cfg.rms_norm_eps)
            akernel = None
            if conv is not None:
                x, akernel = conv[0].prepare(x, block_pos)

            q = _lin(x, self.w[f"{p}.self_attn.q_proj.weight"]).view(t, nh, hd)
            q = _rms(q, self.w[f"{p}.self_attn.q_norm.weight"], cfg.rms_norm_eps)
            k = _lin(x, self.w[f"{p}.self_attn.k_proj.weight"]).view(t, nkv, hd)
            k = _rms(k, self.w[f"{p}.self_attn.k_norm.weight"], cfg.rms_norm_eps)
            v = _lin(x, self.w[f"{p}.self_attn.v_proj.weight"]).view(t, nkv, hd)
            q = _apply_rope(q.transpose(0, 1)[None], cos, sin)
            k = _apply_rope(k.transpose(0, 1)[None], cos, sin)
            v = v.transpose(0, 1)[None]
            if return_kv:
                block_kv.append((k[0], v[0]))
            if ctx_kv is not None:
                ck, cv = ctx_kv[i]
                k = torch.cat([ck[None], k], dim=2)
                v = torch.cat([cv[None], v], dim=2)
            o = F.scaled_dot_product_attention(q, k.repeat_interleave(rep, dim=1),
                                               v.repeat_interleave(rep, dim=1),
                                               attn_mask=masks[1] if cfg.is_sliding(i) else masks[0])
            o = o.transpose(1, 2).reshape(t, -1)
            o = _lin(o, self.w[f"{p}.self_attn.o_proj.weight"])
            if conv is not None:
                o = conv[0].finish(o, akernel, block_pos)
            h = res + o

            res = h
            x = _rms(h, self.w[f"{p}.post_attention_layernorm.weight"], cfg.rms_norm_eps)
            mkernel = None
            if conv is not None:
                x, mkernel = conv[1].prepare(x, block_pos)
            g = _lin(x, self.w[f"{p}.mlp.gate_proj.weight"])
            u = _lin(x, self.w[f"{p}.mlp.up_proj.weight"])
            x = _lin(F.silu(g) * u, self.w[f"{p}.mlp.down_proj.weight"])
            if conv is not None:
                x = conv[1].finish(x, mkernel, block_pos)
            h = res + x

        out = _rms(h, self.w["norm.weight"], cfg.rms_norm_eps)
        return (out, block_kv) if return_kv else out

    def _masks(self, positions: torch.Tensor, ctx_positions: torch.Tensor | None,
               t: int) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """(full-attention mask, sliding mask). `None` means "attend to everything".

        Non-causal: nothing here masks a later position out, because `is_causal` is false and every
        row of the block is supposed to see every other row. The only thing masked is the context
        beyond the 2048-wide window of the five `sliding_attention` layers.
        """
        cfg = self.cfg
        if ctx_positions is None or ctx_positions.numel() == 0:
            return None, None
        if cfg.sliding_window is None:
            return None, None
        allpos = torch.cat([ctx_positions, positions])
        delta = positions[:, None] - allpos[None, :]
        # A key at distance >= window behind the query is out of the window; the block's own rows
        # are at distance <= block_size and always inside it.
        return None, delta < cfg.sliding_window

    # ---- candidate selection ---------------------------------------------------
    def unary_candidates(self, logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Top-k over the head's output. The head read has already happened by this point --
        `logits` is [P, V] and reading it whole is, in the reference's own words, "the selector's
        largest single cost"."""
        k = self.cfg.selector_top_k
        vals, ids = torch.topk(logits.float(), k, dim=-1)
        if self.cfg.output_multiplier != 1.0:
            vals = vals * self.cfg.output_multiplier
        if self.cfg.final_logit_softcapping is not None:
            cap = self.cfg.final_logit_softcapping
            vals = torch.tanh(vals / cap) * cap
        return ids.long(), vals

    def lattice(self, pred_hidden: torch.Tensor, candidate_ids: torch.Tensor,
                unary: torch.Tensor, anchor_id: int) -> torch.Tensor:
        """Score[e, p, c] = unary[e, c] + <A[pred[e, p]] * project(h[e]), B[c]>

        `pred[0, :]` is the verified anchor repeated across the k predecessor slots; `pred[e, :]`
        for e > 0 is the previous slot's candidate list. A is `predecessor_codebook`, B is
        `successor_codebook`, both [vocab, 256], both *gathered* by id -- 32 rows per slot, never
        read whole.
        """
        w, k = self.w, self.cfg.selector_top_k
        hidden = F.linear(pred_hidden, w["candidate_selector.hidden_projection.weight"])
        keys = w["candidate_selector.successor_codebook"][candidate_ids]            # [L, k, r]
        if torch.is_tensor(anchor_id):
            # a device scalar (graph): no read back to the host
            anchor = anchor_id.reshape(1, 1).expand(1, k)
        else:
            anchor = torch.full((1, k), int(anchor_id), dtype=torch.long,
                                device=candidate_ids.device)
        pred_ids = torch.cat([anchor, candidate_ids[:-1]], dim=0)                   # [L, k]
        preds = w["candidate_selector.predecessor_codebook"][pred_ids]              # [L, k, r]
        pair = (preds.float() * hidden.float()[:, None, :])
        return unary[:, None, :] + torch.einsum("lpr,lcr->lpc", pair, keys.float())

    @staticmethod
    def walk(candidate_ids: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
        """The greedy path through the lattice -- `sample_path` with `greedy_mask` all true.

        Slot 0 takes `scores[0, 0].argmax()` (every predecessor row of slot 0 is the same anchor),
        and each later slot takes the argmax of the row selected by the previous slot's index. A
        chain walk, not a full Viterbi: the reference does exactly this.
        """
        idx = int(scores[0, 0].argmax())
        path = [idx]
        local = scores[1:].argmax(dim=-1)            # [L-1, k]
        for e in range(local.shape[0]):
            idx = int(local[e, idx])
            path.append(idx)
        sel = torch.tensor(path, dtype=torch.long, device=candidate_ids.device)
        return candidate_ids.gather(-1, sel[:, None])[:, 0]

    @staticmethod
    def walk_host(candidate_ids: torch.Tensor, scores: torch.Tensor) -> tuple[list[int], list]:
        """`walk`, with every argmax taken on the device and brought over in one copy together
        with the candidate table. Returns (token ids, candidates [L][k]) as Python lists."""
        L, k = candidate_ids.shape
        return DFlash2Module._walk_blob(DFlash2Module._blob(candidate_ids, scores).tolist(), L, k)

    @staticmethod
    def walk_host_logp(candidate_ids: torch.Tensor, scores: torch.Tensor,
                       logp: torch.Tensor) -> tuple[list[int], list, list]:
        """`walk_host` plus the lattice's log-probabilities, in ONE synchronisation: both
        are copied into pinned memory behind the draft and the host waits once, where it used to
        read the walk and then launch the log-softmax and read again. The same numbers."""
        L, k = candidate_ids.shape
        blob = DFlash2Module._blob(candidate_ids, scores)
        if not blob.is_cuda:
            return DFlash2Module._walk_blob(blob.tolist(), L, k) + (logp.tolist(),)
        bh = torch.empty(blob.shape, dtype=blob.dtype, pin_memory=True)
        lh = torch.empty(logp.shape, dtype=logp.dtype, pin_memory=True)
        bh.copy_(blob, non_blocking=True)
        lh.copy_(logp, non_blocking=True)
        torch.cuda.current_stream().synchronize()
        return DFlash2Module._walk_blob(bh.tolist(), L, k) + (lh.tolist(),)

    @staticmethod
    def _blob(candidate_ids: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
        return torch.cat([scores[0, 0].argmax().view(1), scores[1:].argmax(dim=-1).reshape(-1),
                          candidate_ids.reshape(-1).long()])

    @staticmethod
    def _walk_blob(blob: list[int], L: int, k: int) -> tuple[list[int], list]:
        local = blob[1:1 + (L - 1) * k]
        cand = [blob[1 + (L - 1) * k + l * k: 1 + (L - 1) * k + (l + 1) * k] for l in range(L)]
        idx = blob[0]
        toks = [cand[0][idx]]
        for e in range(L - 1):
            idx = local[e * k + idx]
            toks.append(cand[e + 1][idx])
        return toks, cand

    @staticmethod
    def viterbi(candidate_ids: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
        """The *maximising* path through the same lattice.

        `walk` above is what the reference does and it is not the argmax of the selector's own
        objective. That objective is a first-order chain,

            score(t_0..t_{L-1}) = sum_l [ unary(l, t_l) + pair(t_{l-1}, t_l) ]

        and `scores[l, p, c]` already holds `unary(l, c) + pair(candidate p of slot l-1, c)`, so
        the maximiser is Viterbi over L slots and k candidates: L x k x k = 7 x 16 x 16 = 1,792
        additions over numbers that are already in registers. The pairwise term was computed for
        every (p, c) pair whether or not the greedy walk looked at it, so this costs no extra
        memory traffic at all -- it reads a tensor the greedy walk also builds and then throws
        away 15/16 of.

        It cannot change what the engine outputs. A different draft is still verified token by
        token against the target's own argmax; a better draft is only accepted further.
        """
        dp = scores[0, 0].clone()                     # [k]  slot 0: every predecessor is the anchor
        back: list[torch.Tensor] = []
        for l in range(1, scores.shape[0]):
            total = dp[:, None] + scores[l]           # [k_pred, k_cand]
            best, arg = total.max(dim=0)              # over the predecessor
            dp = best
            back.append(arg)
        idx = int(dp.argmax())
        path = [idx]
        for arg in reversed(back):
            idx = int(arg[idx])
            path.insert(0, idx)
        sel = torch.tensor(path, dtype=torch.long, device=candidate_ids.device)
        return candidate_ids.gather(-1, sel[:, None])[:, 0]


