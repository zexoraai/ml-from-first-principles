"""Attention masks for the encoder-decoder Transformer.

Paper: Vaswani et al., "Attention Is All You Need", arXiv:1706.03762v7.
Relevant text: section 3.2.3 (three uses of attention) and section 3.1 (the decoder
"prevent[s] positions from attending to subsequent positions").

--------------------------------------------------------------------------------------------
THE ONE CONVENTION YOU MUST HOLD IN YOUR HEAD
--------------------------------------------------------------------------------------------
Every mask in this project is a **keep mask**:

    True  = this (query, key) pair is ALLOWED. Keep it.
    False = this (query, key) pair is FORBIDDEN. Remove it.

We name the arguments `keep_mask` rather than `mask` on purpose. "Mask" is ambiguous in the
wild and the ambiguity is a genuine, common, silent bug:

  * `torch.nn.MultiheadAttention(attn_mask=...)`      -> True means BLOCK
  * `torch.nn.functional.scaled_dot_product_attention` -> True means KEEP

PyTorch itself is inconsistent between two of its own APIs. Getting this backwards does not
crash; it trains a model that can see the future, or one that can see nothing, and you find
out days later. We follow the `scaled_dot_product_attention` convention (True = keep) and we
put the word "keep" in every identifier so a reader cannot guess wrong.

--------------------------------------------------------------------------------------------
SHAPES
--------------------------------------------------------------------------------------------
Masks are returned in a rank-4 shape that broadcasts against the attention score tensor

    scores : (batch, heads, q_len, k_len)

so a mask has shape (batch_or_1, heads_or_1, q_len_or_1, k_len). Broadcasting is what lets a
per-sequence padding mask of shape (batch, 1, 1, k_len) apply to every head and every query
position at once, and a causal mask of shape (1, 1, q_len, k_len) apply to every example.
"""

from __future__ import annotations

import torch


def padding_key_mask(token_ids: torch.Tensor, pad_id: int) -> torch.Tensor:
    """Build a keep mask that forbids attending TO padding positions.

    Args:
        token_ids: integer tensor of shape (batch, seq_len) containing token ids.
        pad_id: the id used for padding.

    Returns:
        Bool tensor of shape (batch, 1, 1, seq_len). Entry [b, 0, 0, j] is True iff
        key position j of example b is a real token.

    Why keys and not queries
    ------------------------
    We mask which positions may be *read from*, never which positions may *ask*. If we also
    blocked padding query positions, every key would be forbidden for those rows, softmax
    would be taken over an empty set, and the result would be undefined (see
    `scaled_dot_product_attention` for how that is detected). Padding query positions do
    produce garbage outputs -- that is fine and expected, because the loss masks those
    target positions out. Garbage that is never read is harmless; NaN is not, because NaN
    propagates into the gradient of every shared parameter and destroys the whole model.

    Shape walk-through
    ------------------
        token_ids            (batch, seq_len)
        != pad_id            (batch, seq_len)          bool
        [:, None, None, :]   (batch, 1, 1, seq_len)    ready to broadcast over heads, queries
    """
    if token_ids.dim() != 2:
        raise ValueError(f"expected token_ids of shape (batch, seq_len), got {tuple(token_ids.shape)}")
    keep = token_ids != pad_id                      # (batch, seq_len)
    return keep[:, None, None, :]                   # (batch, 1, 1, seq_len)


def causal_keep_mask(
    q_len: int,
    k_len: int | None = None,
    *,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Build the lower-triangular keep mask that forbids attending to future positions.

    Args:
        q_len: number of query positions.
        k_len: number of key positions. Defaults to `q_len` (the self-attention case).
        device: device for the returned tensor.

    Returns:
        Bool tensor of shape (1, 1, q_len, k_len) where entry [0, 0, i, j] is True iff j <= i.

    The invariant, stated precisely
    -------------------------------
    Output position i may depend on input positions 0..i and on nothing beyond. Combined with
    the fact that decoder inputs are the targets shifted right by one, this is what makes
    next-token training equivalent to `q_len` separate prediction problems solved in parallel
    (paper, section 3.1: "the predictions for position i can depend only on the known outputs
    at positions less than i").

    Why `j <= i` and not `j < i`
    ----------------------------
    Position i is allowed to attend to itself. It must be: the decoder input at position i is
    token i-1 of the target (because of the shift), so "attending to itself" means attending
    to the most recent *already generated* token, not to the answer.
    """
    if k_len is None:
        k_len = q_len
    keep = torch.ones(q_len, k_len, dtype=torch.bool, device=device).tril(diagonal=0)
    return keep[None, None, :, :]                   # (1, 1, q_len, k_len)


def combine_keep_masks(*masks: torch.Tensor | None) -> torch.Tensor | None:
    """Logical AND of several keep masks, ignoring `None`.

    A pair is kept only if EVERY mask keeps it. This is how the decoder's self-attention gets
    both constraints at once: "do not look at the future" AND "do not look at padding".

    Returns None when every argument is None, which downstream code reads as "no masking".
    """
    out: torch.Tensor | None = None
    for m in masks:
        if m is None:
            continue
        if m.dtype != torch.bool:
            raise TypeError(f"keep masks must be bool, got {m.dtype}. Refusing to guess the convention.")
        out = m if out is None else (out & m)
    return out


def fully_masked_rows(keep_mask: torch.Tensor) -> torch.Tensor:
    """Diagnostic: locate query rows that are allowed to attend to nothing at all.

    Args:
        keep_mask: bool tensor broadcastable to (batch, heads, q_len, k_len).

    Returns:
        Bool tensor of shape equal to keep_mask's leading dims + (q_len,), True where the row
        has no permitted key. Softmax over such a row is mathematically undefined; this
        function exists so the condition is *found* rather than silently turned into NaN or,
        worse, into a uniform distribution over forbidden positions.
    """
    return ~keep_mask.any(dim=-1)
