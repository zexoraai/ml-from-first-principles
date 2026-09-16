"""Scaled dot-product attention and multi-head attention, written by hand.

Paper: Vaswani et al., "Attention Is All You Need", arXiv:1706.03762v7, sections 3.2.1-3.2.2.

WHAT IS DELIBERATELY NOT USED HERE
----------------------------------
`torch.nn.MultiheadAttention`, `torch.nn.functional.scaled_dot_product_attention`, and
`torch.nn.Transformer*` are absent from this file by design -- they are the mechanisms under
demonstration. They do appear in `tests/` as *independent oracles*: using a known-good
implementation to check ours is verification, not substitution, and the distinction is the
whole point of a correctness suite.

`nn.Linear` and `nn.Dropout` *are* used. Neither is a mechanism this project claims to teach:
`nn.Linear` is `x @ W.T + b`, and dropout is elementwise Bernoulli scaling. What matters --
the projection layout, the head split, the score scaling, the masking, the softmax axis, and
the concat-then-project -- is all explicit below.

--------------------------------------------------------------------------------------------
EQUATION 1  ->  `scaled_dot_product_attention`
--------------------------------------------------------------------------------------------
    Attention(Q, K, V) = softmax( Q K^T / sqrt(d_k) ) V

--------------------------------------------------------------------------------------------
EQUATION (section 3.2.2)  ->  `MultiHeadAttention.forward`
--------------------------------------------------------------------------------------------
    MultiHead(Q, K, V) = Concat(head_1, ..., head_h) W^O
    head_i             = Attention(Q W_i^Q, K W_i^K, V W_i^V)
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from .masks import fully_masked_rows

__all__ = ["scaled_dot_product_attention", "MultiHeadAttention", "split_heads", "merge_heads"]


def scaled_dot_product_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    keep_mask: torch.Tensor | None = None,
    dropout: nn.Dropout | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Equation 1 of the paper: ``softmax(Q K^T / sqrt(d_k)) V``.

    Args:
        query: (..., q_len, d_k)
        key:   (..., k_len, d_k)
        value: (..., k_len, d_v)
        keep_mask: optional bool tensor broadcastable to (..., q_len, k_len).
            True = keep, False = forbid. See `masks.py` for why the name says "keep".
        dropout: optional dropout module applied to the attention *weights* after softmax.

    Returns:
        (output, weights) where
            output  : (..., q_len, d_v)
            weights : (..., q_len, k_len), each row summing to 1 over permitted keys.

    The three lines that matter, and why each is the way it is
    ---------------------------------------------------------
    1. ``scores = query @ key.transpose(-2, -1) / sqrt(d_k)``

       The transpose turns (k_len, d_k) into (d_k, k_len) so the matmul contracts over the
       *feature* axis, producing one score per (query position, key position) pair. Every
       query is compared against every key in a single matmul; that parallelism is the whole
       reason this architecture replaced recurrence.

       The ``sqrt(d_k)`` divisor is not cosmetic. The paper's footnote 4 gives the argument:
       if the components of q and k are independent with mean 0 and variance 1, then
       ``q . k = sum_{i=1}^{d_k} q_i k_i`` has mean 0 and variance ``d_k``. So the raw scores
       grow like ``sqrt(d_k)`` in scale. Feed those into softmax and you get a near-one-hot
       distribution whose gradient is nearly zero -- the model cannot learn its way out
       because the very saturation that kills the gradient is what must change. Dividing by
       ``sqrt(d_k)`` restores unit variance and keeps softmax in its responsive region.
       `tests/test_p1_attention.py::test_scaling_keeps_score_variance_near_one` measures this.

    2. ``scores.masked_fill(~keep_mask, -inf)``

       ``exp(-inf) = 0`` exactly, so forbidden keys receive exactly zero probability and the
       remaining probabilities still sum to 1 without any renormalisation step. Using a large
       finite constant such as -1e9 instead is the more common trick and it *almost* works,
       but it leaks a tiny amount of probability mass to forbidden positions and, in fp16,
       -1e9 is not even representable. We use -inf and handle its one pathology explicitly:

    3. Fully-masked rows.

       If a query row has no permitted key, softmax is being asked to normalise over the empty
       set. That is undefined, and in floating point it surfaces as NaN across the row --
       which then contaminates the gradient of every parameter that fed it. We detect the
       condition and write exact zeros instead, so the failure stays local and inspectable
       rather than poisoning the run. Correct usage never triggers it (masks constrain keys,
       not queries -- see `masks.padding_key_mask`), so if `n_fully_masked` is ever nonzero,
       that is a bug signal, not a routine event.
    """
    d_k = query.size(-1)
    if key.size(-1) != d_k:
        raise ValueError(
            f"query/key feature dims must match to form Q K^T: got d_k(query)={d_k}, d_k(key)={key.size(-1)}"
        )
    if key.size(-2) != value.size(-2):
        raise ValueError(
            "key and value must have the same number of positions (each key indexes one value): "
            f"k_len(key)={key.size(-2)}, k_len(value)={value.size(-2)}"
        )

    # (..., q_len, d_k) @ (..., d_k, k_len) -> (..., q_len, k_len)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)

    empty_rows: torch.Tensor | None = None
    if keep_mask is not None:
        if keep_mask.dtype != torch.bool:
            raise TypeError(
                f"keep_mask must be bool, got {keep_mask.dtype}. An additive float mask is a "
                "different convention and silently inverting it is a real bug we refuse to risk."
            )
        empty_rows = fully_masked_rows(keep_mask)            # (..., q_len)
        scores = scores.masked_fill(~keep_mask, float("-inf"))

    weights = torch.softmax(scores, dim=-1)                  # normalise over KEYS, the last axis

    if empty_rows is not None and bool(empty_rows.any()):
        # Undefined softmax -> exact zeros, kept local instead of NaN-poisoning the graph.
        weights = torch.where(empty_rows.unsqueeze(-1).expand_as(weights),
                              torch.zeros_like(weights), weights)

    if dropout is not None:
        weights = dropout(weights)

    # (..., q_len, k_len) @ (..., k_len, d_v) -> (..., q_len, d_v)
    output = torch.matmul(weights, value)
    return output, weights


def split_heads(x: torch.Tensor, num_heads: int) -> torch.Tensor:
    """(batch, seq, num_heads * head_dim) -> (batch, num_heads, seq, head_dim).

    Why this reshape is legitimate
    ------------------------------
    The paper describes h separate projection matrices ``W_i^Q in R^{d_model x d_k}``. We
    instead apply ONE matrix of shape ``(d_model, h * d_k)`` and then slice the output into h
    contiguous blocks. These are the same computation: concatenating the h matrices
    column-wise and multiplying once produces exactly the concatenation of the h individual
    products. One big matmul is far faster than h small ones, and no expressivity is lost --
    the parameters are in bijection.

    The head axis is moved to position 1 so that the batched matmul inside
    `scaled_dot_product_attention` treats ``(batch, num_heads)`` as independent batch
    dimensions and the last two axes as the matrix to multiply. Each head then attends in its
    own subspace with no cross-talk, which is precisely the paper's intent: "jointly attend to
    information from different representation subspaces".
    """
    batch, seq, total = x.shape
    if total % num_heads != 0:
        raise ValueError(f"feature dim {total} is not divisible by num_heads {num_heads}")
    head_dim = total // num_heads
    x = x.view(batch, seq, num_heads, head_dim)   # split the feature axis
    return x.transpose(1, 2)                      # (batch, num_heads, seq, head_dim)


def merge_heads(x: torch.Tensor) -> torch.Tensor:
    """(batch, num_heads, seq, head_dim) -> (batch, seq, num_heads * head_dim).

    This is the ``Concat(head_1, ..., head_h)`` of section 3.2.2, and it is the exact inverse
    of `split_heads`.

    The `.contiguous()` is required, not defensive
    ---------------------------------------------
    `transpose` returns a view with permuted strides; the underlying memory order is unchanged.
    `view` demands that the requested shape be reachable by reinterpreting a contiguous buffer,
    which a transposed tensor's is not, so `view` raises. `.contiguous()` materialises the
    permuted order in memory (a real copy) and then `view` is free. Using `.reshape()` would
    hide the copy; we keep it visible because "where does the copy happen" is exactly the kind
    of question the tiled-attention project (P5) is about.
    """
    batch, num_heads, seq, head_dim = x.shape
    return x.transpose(1, 2).contiguous().view(batch, seq, num_heads * head_dim)


class MultiHeadAttention(nn.Module):
    """Multi-head attention, section 3.2.2.

        MultiHead(Q, K, V) = Concat(head_1, ..., head_h) W^O
        head_i             = Attention(Q W_i^Q, K W_i^K, V W_i^V)

    Args:
        d_model: model width. Paper base: 512.
        num_heads: h. Paper base: 8.
        dropout: dropout on attention weights. The paper's section 5.4 names residual and
            embedding dropout; attention dropout appears in the authors' own code and in
            Table 3's parsing setup ("both attention and residual"). Default 0.0 so it is
            opt-in and never silently on during correctness checks.
        bias: whether the four projections carry bias terms. Default **False**, because the
            paper writes them as plain parameter matrices ``W_i^Q`` etc. Note
            `nn.MultiheadAttention` defaults to True -- set `bias=True` when running the
            parity test against it.

    Serves all three uses in section 3.2.3 with no code change, only different arguments:
        encoder self-attention   : q = k = v = encoder states,  keep_mask = padding
        decoder self-attention   : q = k = v = decoder states,  keep_mask = padding AND causal
        encoder-decoder attention: q = decoder states, k = v = encoder output,
                                   keep_mask = source padding
    That one module covers all three is the paper's actual structural claim, and it is why the
    architecture is as small as it is.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        *,
        dropout: float = 0.0,
        bias: bool = False,
    ) -> None:
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError(
                f"d_model={d_model} must be divisible by num_heads={num_heads}; the paper sets "
                f"d_k = d_v = d_model / h so that total attention cost matches single-head "
                f"attention at full width (section 3.2.2)."
            )
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads          # d_k = d_v = d_model / h

        # One matmul per role, sliced into heads afterwards. See `split_heads` for why this is
        # equivalent to the paper's per-head matrices.
        self.w_q = nn.Linear(d_model, d_model, bias=bias)
        self.w_k = nn.Linear(d_model, d_model, bias=bias)
        self.w_v = nn.Linear(d_model, d_model, bias=bias)
        self.w_o = nn.Linear(d_model, d_model, bias=bias)   # W^O in R^{h*d_v x d_model}

        self.attn_dropout = nn.Dropout(dropout)
        self.last_attention_weights: torch.Tensor | None = None

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Xavier-uniform initialisation.

        The paper does not state an initialisation scheme. Xavier/Glorot uniform is what the
        authors' `tensor2tensor` release and the widely-read "Annotated Transformer" use, so we
        follow that and label it an implementation choice rather than a paper detail. Recorded
        here so the oral defence answer is "the paper is silent; this came from the reference
        code" and not a bluff.
        """
        for proj in (self.w_q, self.w_k, self.w_v, self.w_o):
            nn.init.xavier_uniform_(proj.weight)
            if proj.bias is not None:
                nn.init.zeros_(proj.bias)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        *,
        keep_mask: torch.Tensor | None = None,
        store_weights: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run multi-head attention.

        Args:
            query: (batch, q_len, d_model)
            key:   (batch, k_len, d_model)
            value: (batch, k_len, d_model)
            keep_mask: bool, broadcastable to (batch, num_heads, q_len, k_len).
            store_weights: keep the per-head weights on the module for the demo/visualiser.
                Off by default -- holding a (batch, heads, q_len, k_len) tensor alive across
                training steps is a memory leak with no upside.

        Returns:
            (output, weights): output (batch, q_len, d_model), weights
            (batch, num_heads, q_len, k_len).

        Shape trace -- the boundaries worth memorising
        ---------------------------------------------
            query                    (batch, q_len, d_model)
            w_q(query)               (batch, q_len, d_model)        = (batch, q_len, h*d_k)
            split_heads              (batch, h, q_len, d_k)
            scores = q @ k^T/sqrt    (batch, h, q_len, k_len)
            weights = softmax(-1)    (batch, h, q_len, k_len)
            weights @ v              (batch, h, q_len, d_v)
            merge_heads              (batch, q_len, h*d_v)          = (batch, q_len, d_model)
            w_o                      (batch, q_len, d_model)
        """
        if query.dim() != 3 or key.dim() != 3 or value.dim() != 3:
            raise ValueError(
                "expected rank-3 (batch, seq, d_model) inputs; got "
                f"query={tuple(query.shape)}, key={tuple(key.shape)}, value={tuple(value.shape)}"
            )
        for name, t in (("query", query), ("key", key), ("value", value)):
            if t.size(-1) != self.d_model:
                raise ValueError(f"{name} last dim {t.size(-1)} != d_model {self.d_model}")

        q = split_heads(self.w_q(query), self.num_heads)   # (batch, h, q_len, d_k)
        k = split_heads(self.w_k(key), self.num_heads)     # (batch, h, k_len, d_k)
        v = split_heads(self.w_v(value), self.num_heads)   # (batch, h, k_len, d_v)

        context, weights = scaled_dot_product_attention(
            q, k, v, keep_mask=keep_mask, dropout=self.attn_dropout if self.training else None
        )

        output = self.w_o(merge_heads(context))           # (batch, q_len, d_model)

        self.last_attention_weights = weights.detach() if store_weights else None
        return output, weights
