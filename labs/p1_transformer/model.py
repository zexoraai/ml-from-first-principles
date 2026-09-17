"""The full encoder-decoder Transformer, assembled from the hand-written mechanisms.

Paper: Vaswani et al., arXiv:1706.03762v7. Figure 1 is the diagram this file implements.

READING ORDER
-------------
`EncoderLayer` -> `DecoderLayer` -> `Encoder` -> `Decoder` -> `Transformer`. Each is a thin
composition of the parts in `attention.py`, `layers.py`, `positional.py` and `masks.py`; almost no
new arithmetic appears here. That is the point of the architecture and it is worth noticing: the
paper's contribution is a *wiring* claim, not a pile of new operations.

THE THREE PLACES ATTENTION APPEARS (section 3.2.3), and how they differ
----------------------------------------------------------------------
    encoder self-attention   q = k = v = encoder states     mask: source padding
    decoder self-attention   q = k = v = decoder states     mask: target padding AND causal
    cross-attention          q = decoder states             mask: source padding
                             k = v = final encoder output
One `MultiHeadAttention` class serves all three. Only the arguments change.

WHY CROSS-ATTENTION USES THE *FINAL* ENCODER OUTPUT
--------------------------------------------------
Every decoder layer attends to the same tensor: the output of the last encoder layer. Decoder
layer 0 does not read encoder layer 0. The encoder runs to completion first, producing one fixed
memory that all decoder layers query. This is why the encoder can be computed once and reused for
every decoding step, which is what makes incremental generation affordable.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Literal

import torch
import torch.nn as nn

from .attention import MultiHeadAttention
from .layers import LayerNorm, PositionwiseFeedForward, SublayerConnection
from .masks import causal_keep_mask, combine_keep_masks, padding_key_mask
from .positional import SinusoidalPositionalEncoding

__all__ = ["TransformerConfig", "EncoderLayer", "DecoderLayer", "Encoder", "Decoder", "Transformer"]


@dataclass
class TransformerConfig:
    """Every architectural choice in one inspectable object.

    Defaults are the *tiny* configuration this project actually trains (see `env/feasibility.md`),
    not the paper's base model. The paper's base is available as `TransformerConfig.paper_base()`
    so the two are comparable without either pretending to be the other.
    """

    vocab_size: int
    d_model: int = 128
    num_heads: int = 4
    d_ff: int = 512
    num_encoder_layers: int = 2
    num_decoder_layers: int = 2
    dropout: float = 0.1
    attention_dropout: float = 0.0
    max_len: int = 64
    norm_style: Literal["post", "pre"] = "post"    # paper text is post-norm; see D-006
    tie_embeddings: bool = True                    # section 3.4
    scale_embeddings: bool = True                  # section 3.4: multiply by sqrt(d_model)
    pad_id: int = 0
    bos_id: int = 1
    eos_id: int = 2
    layer_norm_eps: float = 1e-5

    @classmethod
    def paper_base(cls, vocab_size: int = 37000) -> "TransformerConfig":
        """Table 3, base row. Provided for comparison; we do not train this."""
        return cls(
            vocab_size=vocab_size, d_model=512, num_heads=8, d_ff=2048,
            num_encoder_layers=6, num_decoder_layers=6, dropout=0.1, max_len=512,
        )

    def to_dict(self) -> dict:
        return asdict(self)

    def __post_init__(self) -> None:
        if self.d_model % self.num_heads:
            raise ValueError(f"d_model={self.d_model} not divisible by num_heads={self.num_heads}")
        if self.tie_embeddings and self.d_model <= 0:
            raise ValueError("d_model must be positive")


class EncoderLayer(nn.Module):
    """One encoder layer: self-attention, then position-wise FFN, each in a residual sub-layer.

    Section 3.1: "Each layer has two sub-layers. The first is a multi-head self-attention
    mechanism, and the second is a simple, position-wise fully connected feed-forward network."
    """

    def __init__(self, cfg: TransformerConfig) -> None:
        super().__init__()
        self.self_attn = MultiHeadAttention(
            cfg.d_model, cfg.num_heads, dropout=cfg.attention_dropout
        )
        self.ffn = PositionwiseFeedForward(cfg.d_model, cfg.d_ff, dropout=cfg.dropout)
        self.sub_attn = SublayerConnection(
            cfg.d_model, dropout=cfg.dropout, norm_style=cfg.norm_style, eps=cfg.layer_norm_eps
        )
        self.sub_ffn = SublayerConnection(
            cfg.d_model, dropout=cfg.dropout, norm_style=cfg.norm_style, eps=cfg.layer_norm_eps
        )
        self.last_self_attn: torch.Tensor | None = None

    def forward(
        self, x: torch.Tensor, *, keep_mask: torch.Tensor | None = None, store_weights: bool = False
    ) -> torch.Tensor:
        """(batch, src_len, d_model) -> (batch, src_len, d_model)."""

        def attn_sublayer(t: torch.Tensor) -> torch.Tensor:
            out, w = self.self_attn(t, t, t, keep_mask=keep_mask)
            if store_weights:
                self.last_self_attn = w.detach()
            return out

        x = self.sub_attn(x, attn_sublayer)
        return self.sub_ffn(x, self.ffn)


class DecoderLayer(nn.Module):
    """One decoder layer: masked self-attention, cross-attention, then FFN.

    Section 3.1: "the decoder inserts a third sub-layer, which performs multi-head attention over
    the output of the encoder stack."

    The ordering is load-bearing. Self-attention first lets the decoder consolidate what it has
    already produced; cross-attention then queries the source with that consolidated state. Swap
    them and the query into the encoder would be built from the raw shifted embedding instead.
    """

    def __init__(self, cfg: TransformerConfig) -> None:
        super().__init__()
        self.self_attn = MultiHeadAttention(cfg.d_model, cfg.num_heads, dropout=cfg.attention_dropout)
        self.cross_attn = MultiHeadAttention(cfg.d_model, cfg.num_heads, dropout=cfg.attention_dropout)
        self.ffn = PositionwiseFeedForward(cfg.d_model, cfg.d_ff, dropout=cfg.dropout)
        self.sub_self = SublayerConnection(cfg.d_model, dropout=cfg.dropout,
                                          norm_style=cfg.norm_style, eps=cfg.layer_norm_eps)
        self.sub_cross = SublayerConnection(cfg.d_model, dropout=cfg.dropout,
                                           norm_style=cfg.norm_style, eps=cfg.layer_norm_eps)
        self.sub_ffn = SublayerConnection(cfg.d_model, dropout=cfg.dropout,
                                         norm_style=cfg.norm_style, eps=cfg.layer_norm_eps)
        self.last_self_attn: torch.Tensor | None = None
        self.last_cross_attn: torch.Tensor | None = None

    def forward(
        self,
        x: torch.Tensor,
        memory: torch.Tensor,
        *,
        self_keep_mask: torch.Tensor | None = None,
        cross_keep_mask: torch.Tensor | None = None,
        store_weights: bool = False,
    ) -> torch.Tensor:
        """Args: x (batch, tgt_len, d_model), memory (batch, src_len, d_model). Returns x's shape."""

        def self_sublayer(t: torch.Tensor) -> torch.Tensor:
            out, w = self.self_attn(t, t, t, keep_mask=self_keep_mask)
            if store_weights:
                self.last_self_attn = w.detach()
            return out

        def cross_sublayer(t: torch.Tensor) -> torch.Tensor:
            # Queries come from the decoder; keys and values from the encoder memory.
            out, w = self.cross_attn(t, memory, memory, keep_mask=cross_keep_mask)
            if store_weights:
                self.last_cross_attn = w.detach()
            return out

        x = self.sub_self(x, self_sublayer)
        x = self.sub_cross(x, cross_sublayer)
        return self.sub_ffn(x, self.ffn)


class Encoder(nn.Module):
    """Embedding + positional encoding + a stack of `EncoderLayer`s."""

    def __init__(self, cfg: TransformerConfig, embedding: nn.Embedding) -> None:
        super().__init__()
        self.cfg = cfg
        self.embedding = embedding
        self.pos = SinusoidalPositionalEncoding(cfg.d_model, max_len=cfg.max_len, dropout=cfg.dropout)
        self.layers = nn.ModuleList([EncoderLayer(cfg) for _ in range(cfg.num_encoder_layers)])
        # A final normalization is required for pre-norm and must NOT be present for post-norm.
        # Post-norm already normalises the output of every sub-layer, so the stack output is
        # already normalised; adding another would be a silent deviation from the paper. Pre-norm
        # leaves the residual stream un-normalised all the way to the top, so without this the
        # logits would be fed an activation whose scale grows with depth.
        self.final_norm = (
            LayerNorm(cfg.d_model, eps=cfg.layer_norm_eps) if cfg.norm_style == "pre" else None
        )

    def forward(
        self, src: torch.Tensor, *, keep_mask: torch.Tensor | None = None, store_weights: bool = False
    ) -> torch.Tensor:
        """(batch, src_len) token ids -> (batch, src_len, d_model)."""
        x = self.embedding(src)
        if self.cfg.scale_embeddings:
            # Section 3.4. With tied weights the embedding matrix is also the output projection,
            # so it is initialised at a scale suited to producing logits. Multiplying by
            # sqrt(d_model) here lifts the embedding to a magnitude comparable to the positional
            # encoding, which is bounded in [-1, 1]. Skip it and the positional signal swamps the
            # token identity; the model then struggles to tell tokens apart at all.
            x = x * math.sqrt(self.cfg.d_model)
        x = self.pos(x)
        for layer in self.layers:
            x = layer(x, keep_mask=keep_mask, store_weights=store_weights)
        return self.final_norm(x) if self.final_norm is not None else x


class Decoder(nn.Module):
    """Embedding + positional encoding + a stack of `DecoderLayer`s."""

    def __init__(self, cfg: TransformerConfig, embedding: nn.Embedding) -> None:
        super().__init__()
        self.cfg = cfg
        self.embedding = embedding
        self.pos = SinusoidalPositionalEncoding(cfg.d_model, max_len=cfg.max_len, dropout=cfg.dropout)
        self.layers = nn.ModuleList([DecoderLayer(cfg) for _ in range(cfg.num_decoder_layers)])
        self.final_norm = (
            LayerNorm(cfg.d_model, eps=cfg.layer_norm_eps) if cfg.norm_style == "pre" else None
        )

    def forward(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        *,
        self_keep_mask: torch.Tensor | None = None,
        cross_keep_mask: torch.Tensor | None = None,
        store_weights: bool = False,
    ) -> torch.Tensor:
        """(batch, tgt_len) token ids + memory -> (batch, tgt_len, d_model)."""
        x = self.embedding(tgt)
        if self.cfg.scale_embeddings:
            x = x * math.sqrt(self.cfg.d_model)
        x = self.pos(x)
        for layer in self.layers:
            x = layer(x, memory, self_keep_mask=self_keep_mask,
                      cross_keep_mask=cross_keep_mask, store_weights=store_weights)
        return self.final_norm(x) if self.final_norm is not None else x


class Transformer(nn.Module):
    """The complete model of Figure 1.

    Args:
        cfg: see `TransformerConfig`.

    Vocabulary
    ----------
    A single shared vocabulary for source and target. The paper does this too (section 5.1: "a
    shared source-target vocabulary of about 37000 tokens"), and it is what makes the section 3.4
    three-way weight tying possible: input embedding, output embedding, and pre-softmax projection
    are all the same matrix.

    Why tying is more than a parameter saving
    -----------------------------------------
    It forces one representation to serve two jobs: "which token is this" on the way in, and "how
    much do I want to emit this token" on the way out. The logit for token *t* becomes the dot
    product of the final hidden state with *t*'s own embedding, so the model is scoring
    "how much does my current state look like token t". At our vocabulary size the parameter
    saving is negligible; the inductive bias is the actual reason to do it.
    """

    def __init__(self, cfg: TransformerConfig) -> None:
        super().__init__()
        self.cfg = cfg

        self.embedding = nn.Embedding(cfg.vocab_size, cfg.d_model, padding_idx=cfg.pad_id)
        nn.init.normal_(self.embedding.weight, mean=0.0, std=cfg.d_model ** -0.5)
        with torch.no_grad():
            self.embedding.weight[cfg.pad_id].zero_()

        if cfg.tie_embeddings:
            src_emb = tgt_emb = self.embedding
        else:
            src_emb = self.embedding
            tgt_emb = nn.Embedding(cfg.vocab_size, cfg.d_model, padding_idx=cfg.pad_id)
            nn.init.normal_(tgt_emb.weight, mean=0.0, std=cfg.d_model ** -0.5)
            self.tgt_embedding = tgt_emb

        self.encoder = Encoder(cfg, src_emb)
        self.decoder = Decoder(cfg, tgt_emb)

        self.generator = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        if cfg.tie_embeddings:
            # Same Parameter object, not a copy: one tensor, one gradient, always consistent.
            self.generator.weight = self.embedding.weight
        else:
            nn.init.normal_(self.generator.weight, mean=0.0, std=cfg.d_model ** -0.5)

    # ---------------------------------------------------------------------------------------
    # masks
    # ---------------------------------------------------------------------------------------
    def source_keep_mask(self, src: torch.Tensor) -> torch.Tensor:
        """(batch, 1, 1, src_len) — used by encoder self-attention and by cross-attention."""
        return padding_key_mask(src, self.cfg.pad_id)

    def target_keep_mask(self, tgt: torch.Tensor) -> torch.Tensor:
        """(batch, 1, tgt_len, tgt_len) — padding AND causality, combined.

        Both constraints must hold simultaneously, which is exactly a logical AND. Applying only
        one is the single most consequential bug available in this file: drop causality and the
        model trains by copying the answer from its own input, reaching near-zero loss while being
        completely useless at generation time, because at inference the future is not there to copy.
        """
        return combine_keep_masks(
            padding_key_mask(tgt, self.cfg.pad_id),
            causal_keep_mask(tgt.size(1), device=tgt.device),
        )

    # ---------------------------------------------------------------------------------------
    # forward
    # ---------------------------------------------------------------------------------------
    def encode(self, src: torch.Tensor, *, store_weights: bool = False) -> torch.Tensor:
        return self.encoder(src, keep_mask=self.source_keep_mask(src), store_weights=store_weights)

    def decode(
        self, tgt: torch.Tensor, memory: torch.Tensor, src: torch.Tensor, *, store_weights: bool = False
    ) -> torch.Tensor:
        return self.decoder(
            tgt, memory,
            self_keep_mask=self.target_keep_mask(tgt),
            cross_keep_mask=self.source_keep_mask(src),
            store_weights=store_weights,
        )

    def forward(
        self, src: torch.Tensor, tgt_in: torch.Tensor, *, store_weights: bool = False
    ) -> torch.Tensor:
        """Args: src (batch, src_len), tgt_in (batch, tgt_len). Returns (batch, tgt_len, vocab).

        `tgt_in` is the target sequence **shifted right**: `[BOS, y_0, ..., y_{n-2}]`. The label at
        position i is `y_i`, so position i predicts the next token from everything strictly before
        it. The shift plus the causal mask are what make this equivalent to `tgt_len` separate
        next-token problems solved in one parallel pass — the property that let this architecture
        replace recurrence.
        """
        memory = self.encode(src, store_weights=store_weights)
        hidden = self.decode(tgt_in, memory, src, store_weights=store_weights)
        return self.generator(hidden)

    # ---------------------------------------------------------------------------------------
    # introspection
    # ---------------------------------------------------------------------------------------
    def num_parameters(self, trainable_only: bool = True) -> int:
        """Counts each Parameter object once, so tied weights are not double-counted."""
        seen: set[int] = set()
        total = 0
        for p in self.parameters():
            if id(p) in seen or (trainable_only and not p.requires_grad):
                continue
            seen.add(id(p))
            total += p.numel()
        return total

    def collect_attention(self) -> dict[str, list[torch.Tensor]]:
        """Return the attention weights stored by the most recent `store_weights=True` pass.

        Shapes: encoder self (batch, heads, src_len, src_len); decoder self (batch, heads, tgt_len,
        tgt_len); cross (batch, heads, tgt_len, src_len). Used by the demo's head viewer, which is
        the reason `store_weights` exists at all.
        """
        return {
            "encoder_self": [l.last_self_attn for l in self.encoder.layers if l.last_self_attn is not None],
            "decoder_self": [l.last_self_attn for l in self.decoder.layers if l.last_self_attn is not None],
            "cross": [l.last_cross_attn for l in self.decoder.layers if l.last_cross_attn is not None],
        }
