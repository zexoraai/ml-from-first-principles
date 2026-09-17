"""Decoder-only GPT.

Primary sources
---------------
* Radford et al., "Language Models are Unsupervised Multitask Learners" (GPT-2, 2019) — the
  architecture, the pre-norm placement, the scaled residual initialisation.
* Radford et al., "Improving Language Understanding by Generative Pre-Training" (GPT-1, 2018).
* Karpathy, `nanoGPT` — the reference this implementation is checked against conceptually.
* Karpathy, `llm.c` — consulted for the same architecture expressed without a framework.

WHAT IS REUSED FROM PROJECT 1, AND WHY
--------------------------------------
`MultiHeadAttention` and `LayerNorm` are imported from `labs.p1_transformer`, not reimplemented.
Causal self-attention in GPT is *exactly* the decoder self-attention of the original Transformer —
same equation, same masking. Rewriting it would create two implementations that could drift, and
would make P1's correctness suite stop covering the code that actually runs here. Connecting
existing capability is the point.

WHAT DIFFERS FROM PROJECT 1, AND WHY EACH DIFFERENCE EXISTS
-----------------------------------------------------------
1. **No encoder, no cross-attention.** One stack, one stream. Every position attends only to
   itself and its past.
2. **Pre-norm** (`x + Sublayer(LayerNorm(x))`) instead of the paper-faithful post-norm of P1.
   This is GPT-2's choice and it is not cosmetic: the unbroken identity path is what lets the
   stack grow deep without the warmup gymnastics post-norm needs. P1 defaults to post-norm because
   it implements the 2017 paper; P2 defaults to pre-norm because it implements GPT-2. Having both
   in one repository, each cited, is the point of doing them separately.
3. **A final LayerNorm before the output head.** Required by pre-norm: the residual stream is never
   normalised on its way up, so without this the logits see an activation whose scale grows with
   depth.
4. **GELU instead of ReLU.** GPT-2's choice. `tanh` approximation available for parity with older
   reference code.
5. **Learned positional embeddings** instead of sinusoids. GPT-2's choice, and it caps context at
   `block_size` by construction — a real limitation, not an implementation shortcut.
6. **Scaled residual initialisation.** See `_init_weights`.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Literal

import torch
import torch.nn as nn

# Reused from Project 1 -- see the module docstring.
from labs.p1_transformer.attention import MultiHeadAttention
from labs.p1_transformer.layers import LayerNorm
from labs.p1_transformer.masks import causal_keep_mask

__all__ = ["GPTConfig", "GPTBlock", "GPT"]


@dataclass
class GPTConfig:
    vocab_size: int
    block_size: int = 192           # maximum context length; learned position embeddings cap this
    n_layer: int = 6
    n_head: int = 6
    d_model: int = 192
    d_ff: int | None = None         # defaults to 4 * d_model, GPT-2's ratio
    dropout: float = 0.1
    attention_dropout: float = 0.1
    bias: bool = True               # GPT-2 uses biases in Linear layers; nanoGPT makes it optional
    activation: Literal["gelu", "gelu_tanh", "relu"] = "gelu"
    tie_embeddings: bool = True     # GPT-2 ties wte with the output head
    layer_norm_eps: float = 1e-5

    def __post_init__(self) -> None:
        if self.d_ff is None:
            self.d_ff = 4 * self.d_model
        if self.d_model % self.n_head:
            raise ValueError(f"d_model={self.d_model} not divisible by n_head={self.n_head}")

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def gpt2_small(cls) -> "GPTConfig":
        """The ~124M-parameter configuration, for cost estimation only.

        Provided so the target has concrete numbers attached rather than being hand-waved. We do not
        train this; see records/GAPS.md G-001 and env/feasibility.md for the measured estimate.
        """
        return cls(vocab_size=50257, block_size=1024, n_layer=12, n_head=12, d_model=768)


class MLP(nn.Module):
    """Position-wise feed-forward, `d_model -> 4*d_model -> d_model`.

    Identical in shape to P1's `PositionwiseFeedForward` but with GELU, so it lives here rather than
    adding an activation flag to P1 and muddying which paper that file implements.
    """

    def __init__(self, cfg: GPTConfig) -> None:
        super().__init__()
        self.fc = nn.Linear(cfg.d_model, cfg.d_ff, bias=cfg.bias)
        self.proj = nn.Linear(cfg.d_ff, cfg.d_model, bias=cfg.bias)
        self.dropout = nn.Dropout(cfg.dropout)
        self.activation_name = cfg.activation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.fc(x)
        if self.activation_name == "gelu":
            h = nn.functional.gelu(h)
        elif self.activation_name == "gelu_tanh":
            h = nn.functional.gelu(h, approximate="tanh")
        else:
            h = torch.relu(h)
        return self.dropout(self.proj(h))


class GPTBlock(nn.Module):
    """One pre-norm transformer block.

        x = x + Attn(LN(x))
        x = x + MLP(LN(x))

    Written out rather than reusing P1's `SublayerConnection` because the residual is added here
    explicitly, which is what makes the identity path visible at the point a reader looks for it.
    """

    def __init__(self, cfg: GPTConfig) -> None:
        super().__init__()
        self.ln_1 = LayerNorm(cfg.d_model, eps=cfg.layer_norm_eps)
        self.attn = MultiHeadAttention(
            cfg.d_model, cfg.n_head, dropout=cfg.attention_dropout, bias=cfg.bias
        )
        self.ln_2 = LayerNorm(cfg.d_model, eps=cfg.layer_norm_eps)
        self.mlp = MLP(cfg)
        self.resid_dropout = nn.Dropout(cfg.dropout)
        self.last_attn: torch.Tensor | None = None

    def forward(
        self, x: torch.Tensor, keep_mask: torch.Tensor, *, store_weights: bool = False
    ) -> torch.Tensor:
        normed = self.ln_1(x)
        attn_out, w = self.attn(normed, normed, normed, keep_mask=keep_mask)
        if store_weights:
            self.last_attn = w.detach()
        x = x + self.resid_dropout(attn_out)
        return x + self.mlp(self.ln_2(x))

    def forward_cached(
        self,
        x_new: torch.Tensor,
        past: tuple[torch.Tensor, torch.Tensor] | None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """Same block, incremental. Returns `(output, (k, v))` for the next step."""
        normed = self.ln_1(x_new)
        past_k, past_v = past if past is not None else (None, None)
        attn_out, _, k, v = self.attn.forward_cached(normed, past_k=past_k, past_v=past_v)
        x = x_new + self.resid_dropout(attn_out)
        return x + self.mlp(self.ln_2(x)), (k, v)


class GPT(nn.Module):
    """Decoder-only language model.

    Shape trace for input `(B, T)`:
        token ids                  (B, T)
        wte(ids)                   (B, T, d_model)
        + wpe(positions)           (B, T, d_model)
        each block                 (B, T, d_model)
        final LayerNorm            (B, T, d_model)
        lm_head                    (B, T, vocab_size)
    """

    def __init__(self, cfg: GPTConfig) -> None:
        super().__init__()
        self.cfg = cfg

        self.wte = nn.Embedding(cfg.vocab_size, cfg.d_model)     # token embeddings
        self.wpe = nn.Embedding(cfg.block_size, cfg.d_model)     # learned position embeddings
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([GPTBlock(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = LayerNorm(cfg.d_model, eps=cfg.layer_norm_eps)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

        if cfg.tie_embeddings:
            self.lm_head.weight = self.wte.weight

        # The causal mask depends only on length, so build it once at max length and slice. A buffer
        # rather than a plain attribute so it follows `.to(device)`; non-persistent because it is
        # exactly reconstructible and would otherwise waste checkpoint bytes.
        self.register_buffer(
            "causal", causal_keep_mask(cfg.block_size), persistent=False
        )

        self.apply(self._init_weights)
        self._scale_residual_projections()

    # -----------------------------------------------------------------------------------------
    def _init_weights(self, module: nn.Module) -> None:
        """GPT-2's initialisation: normal(0, 0.02) for weights, zeros for biases."""
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def _scale_residual_projections(self) -> None:
        """Scale the *output* projection of each residual branch by 1/sqrt(2 * n_layer).

        The reason, from the GPT-2 paper: each block adds its output into the residual stream, so
        after N blocks the stream is a sum of 2N contributions (attention and MLP per block). If each
        contribution has unit-ish variance, the accumulated variance grows linearly in depth and the
        activations entering the final LayerNorm are far larger than at layer 0. Scaling each branch's
        output projection by 1/sqrt(2N) keeps the summed variance roughly constant with depth.

        Applies to `attn.w_o` and `mlp.proj` — the two projections that write into the residual
        stream — and not to the inner projections, which do not.

        `tests/test_p2_model.py::test_residual_scaling_keeps_activation_scale_flat_with_depth`
        measures the effect rather than restating it.
        """
        std = 0.02 / math.sqrt(2 * self.cfg.n_layer)
        for block in self.blocks:
            nn.init.normal_(block.attn.w_o.weight, mean=0.0, std=std)
            nn.init.normal_(block.mlp.proj.weight, mean=0.0, std=std)

    # -----------------------------------------------------------------------------------------
    def num_parameters(self, *, non_embedding: bool = False) -> int:
        """Count parameters once each, so tied weights are not double-counted.

        `non_embedding=True` subtracts the position embeddings, which is the convention used when
        people quote "124M" for GPT-2 small — worth knowing, because it is a common source of
        apparent disagreement between two correct counts.
        """
        seen: set[int] = set()
        total = 0
        for p in self.parameters():
            if id(p) in seen:
                continue
            seen.add(id(p))
            total += p.numel()
        if non_embedding:
            total -= self.wpe.weight.numel()
        return total

    def forward(
        self,
        idx: torch.Tensor,
        targets: torch.Tensor | None = None,
        *,
        store_weights: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Args: idx (B, T) token ids; targets (B, T) or None.

        Returns `(logits, loss)`. `loss` is None when targets are not supplied.

        The loss here is plain cross-entropy with **no label smoothing** — GPT-2 does not use it, and
        P1's page explains why a smoothed loss is not comparable to an unsmoothed one. Keeping them
        different across the two projects is deliberate: each matches its own paper.
        """
        b, t = idx.shape
        if t > self.cfg.block_size:
            raise ValueError(
                f"sequence length {t} exceeds block_size {self.cfg.block_size}. Learned position "
                f"embeddings have no row beyond that, so this is a hard architectural limit, not a "
                f"tunable buffer size."
            )

        pos = torch.arange(t, device=idx.device)
        x = self.drop(self.wte(idx) + self.wpe(pos))

        keep = self.causal[:, :, :t, :t]
        for block in self.blocks:
            x = block(x, keep, store_weights=store_weights)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = nn.functional.cross_entropy(
                logits.reshape(-1, logits.size(-1)), targets.reshape(-1), ignore_index=-1
            )
        return logits, loss

    def forward_cached(
        self,
        idx_new: torch.Tensor,
        past: list[tuple[torch.Tensor, torch.Tensor]] | None = None,
    ) -> tuple[torch.Tensor, list[tuple[torch.Tensor, torch.Tensor]]]:
        """Incremental forward using a KV cache. Returns `(logits_for_new_positions, present)`.

        Args:
            idx_new: (B, n_new) the token ids not yet in the cache.
            past: per-layer `(k, v)` from the previous call, or None for the first call.

        The **absolute positions** of the new tokens are `n_past .. n_past + n_new - 1`, read from the
        cache length. Using `0 .. n_new-1` instead — the obvious mistake — would give every generated
        token the position embedding of position 0, so the model would think it was always at the
        start of the sequence. That produces output which is fluent for a few tokens and then
        degenerates, which is a genuinely hard bug to attribute.

        `tests/test_p2_model.py::test_cached_generation_matches_uncached_exactly` proves this path
        agrees with the plain forward pass bit-for-bit, which is the only thing that makes the
        optimisation safe to ship.
        """
        b, n_new = idx_new.shape
        n_past = 0 if past is None else past[0][0].size(2)
        if n_past + n_new > self.cfg.block_size:
            raise ValueError(
                f"cache length {n_past} + {n_new} new tokens exceeds block_size "
                f"{self.cfg.block_size}"
            )

        pos = torch.arange(n_past, n_past + n_new, device=idx_new.device)
        x = self.drop(self.wte(idx_new) + self.wpe(pos))

        present: list[tuple[torch.Tensor, torch.Tensor]] = []
        for i, block in enumerate(self.blocks):
            x, kv = block.forward_cached(x, None if past is None else past[i])
            present.append(kv)

        return self.lm_head(self.ln_f(x)), present

    def collect_attention(self) -> list[torch.Tensor]:
        """Per-layer attention weights from the most recent `store_weights=True` pass."""
        return [b.last_attn for b in self.blocks if b.last_attn is not None]

    # -----------------------------------------------------------------------------------------
    def estimate_flops_per_token(self) -> dict[str, float]:
        """Analytic forward+backward FLOP estimate per token, for honest compute planning.

        Uses the standard `6 * N` approximation for parameter FLOPs (2 for the forward multiply-add,
        doubled again for the backward pass) plus the attention term, which is quadratic in context
        and is the part the `6N` rule of thumb omits.

        Returned as a dict rather than a single number so the attention share is visible — at
        `block_size` 1024 it stops being negligible, and that is exactly the regime Project 5 is
        about.
        """
        cfg = self.cfg
        n = self.num_parameters(non_embedding=True)
        params_flops = 6.0 * n
        # Attention scores + the weighted value sum: 2 matmuls of (T x d) x (d x T) per layer per
        # head-group, doubled for backward.
        attn_flops = 6.0 * 2.0 * cfg.n_layer * cfg.block_size * cfg.d_model
        return {
            "params_flops_per_token": params_flops,
            "attention_flops_per_token": attn_flops,
            "total_flops_per_token": params_flops + attn_flops,
            "attention_share": attn_flops / (params_flops + attn_flops),
        }
