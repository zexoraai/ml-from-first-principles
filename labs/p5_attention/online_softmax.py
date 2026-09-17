"""Online (streaming) softmax — the identity that makes FlashAttention possible.

Primary sources
---------------
* Milakov & Gimelshein, "Online normalizer calculation for softmax", arXiv:1805.02867 — the streaming
  normaliser itself.
* Dao, Fu, Ermon, Rudra & Ré, "FlashAttention: Fast and Memory-Efficient Exact Attention with
  IO-Awareness", arXiv:2205.14135 — applying it to attention so the S×S score matrix is never
  materialised.
* Dao, "FlashAttention-2", arXiv:2307.08691 — the work-partitioning refinement, and the source of the
  "defer the division" trick used here.

THE PROBLEM
-----------
Safe softmax needs the row maximum before it can exponentiate anything:

    softmax(x)_i = exp(x_i - m) / Σ_j exp(x_j - m),     m = max_j x_j

The subtraction of `m` is not cosmetic. `exp(x)` in float32 overflows to `inf` around x ≈ 88, and
attention scores routinely exceed that before scaling. But needing `m` up front appears to force two
passes over the row — and in attention the row is a length-`T` slice of the score matrix, so two passes
means either storing all `T` scores or recomputing them. Storing them is the O(T²) memory that
FlashAttention exists to avoid.

THE IDENTITY
------------
Suppose we have processed a prefix and hold a running maximum `m` and a running sum
`l = Σ exp(x_j - m)` over that prefix. A new block arrives with maximum `m_b`. Let `m' = max(m, m_b)`.
Then, because

    exp(x_j - m') = exp(x_j - m) · exp(m - m')

every term in the old sum can be corrected by a single scalar factor:

    l' = l · exp(m - m') + Σ_{j in block} exp(x_j - m')

So the normaliser can be maintained exactly in one pass, with O(1) state per row. The correction factor
`exp(m - m')` is always ≤ 1, which is the whole reason this is numerically safe: it can underflow
harmlessly to 0 but it can never overflow.

WHY THIS IS EXACT AND NOT AN APPROXIMATION
------------------------------------------
This is a rearrangement of the same arithmetic, not a truncation, a sampling scheme, or a low-rank
approximation. The only difference from a two-pass softmax is floating-point rounding — the operations
are performed in a different order, and float addition is not associative. `tests/test_p5_*` pin the
agreement to a published tolerance rather than asserting exactness, and that tolerance is the honest
statement of the difference.

This matters because "memory-efficient attention" is a category that contains both exact methods
(FlashAttention) and approximate ones (Performer, Linformer, Nyströmformer). Conflating them is a
common and serious error: the approximations change what the model computes, and this does not.
"""

from __future__ import annotations

import torch

__all__ = ["online_softmax", "softmax_state_update", "OnlineSoftmaxState"]


class OnlineSoftmaxState:
    """Running `(m, l)` per row, plus an optional weighted accumulator.

    `m` is the running maximum and `l` the running sum of `exp(x - m)`. Together they are enough to
    normalise at the very end, which is why the score matrix never has to be kept.

    The accumulator exists because attention does not want `softmax(s)` itself — it wants
    `softmax(s) @ V`. Carrying the unnormalised `Σ exp(s_j - m) v_j` alongside `l` and dividing once at
    the end is FlashAttention-2's "defer the division": it removes a rescale of the full output tile
    from the inner loop, where it would otherwise run on every block.
    """

    def __init__(self, rows: int, dim: int | None = None, *, device=None, dtype=torch.float32) -> None:
        # -inf, not 0, and this is load-bearing. The first real block must always win the max
        # comparison; seeding with 0 would silently clamp all-negative rows to 0 and corrupt them.
        self.m = torch.full((rows,), float("-inf"), device=device, dtype=dtype)
        self.l = torch.zeros((rows,), device=device, dtype=dtype)
        self.acc = torch.zeros((rows, dim), device=device, dtype=dtype) if dim else None

    def update(self, scores: torch.Tensor, values: torch.Tensor | None = None) -> None:
        """Fold one block of scores (rows, block) and optional values (block, dim) into the state."""
        block_max = scores.amax(dim=-1)
        m_new = torch.maximum(self.m, block_max)

        # A row whose state is still -inf and whose block is entirely masked leaves m_new at -inf,
        # making (m - m_new) = (-inf) - (-inf) = nan. Clamping the *exponent argument* to 0 in that
        # case keeps the correction finite; l stays 0, which is the correct representation of "no
        # unmasked keys seen yet".
        finite = torch.isfinite(m_new)
        correction = torch.where(finite, torch.exp(self.m - m_new), torch.zeros_like(m_new))
        correction = torch.nan_to_num(correction, nan=0.0)

        probs = torch.where(
            finite.unsqueeze(-1),
            torch.exp(scores - m_new.unsqueeze(-1)),
            torch.zeros_like(scores),
        )
        probs = torch.nan_to_num(probs, nan=0.0)

        self.l = self.l * correction + probs.sum(dim=-1)
        if self.acc is not None and values is not None:
            self.acc = self.acc * correction.unsqueeze(-1) + probs @ values
        self.m = m_new

    def normalise(self) -> torch.Tensor:
        """Divide the accumulator by `l`, once, at the end.

        Rows with `l == 0` saw no unmasked key at all. Their output is defined as zero rather than
        `nan`: a fully-masked query has no attention distribution, and propagating `nan` would poison
        every downstream tensor and every gradient. This case is reachable in real models — a padded
        position in a right-padded batch under a causal mask can have zero valid keys — so it is
        handled explicitly rather than left to chance.
        """
        if self.acc is None:
            raise RuntimeError("no accumulator: construct with dim= to use normalise()")
        safe_l = torch.where(self.l > 0, self.l, torch.ones_like(self.l))
        out = self.acc / safe_l.unsqueeze(-1)
        return torch.where((self.l > 0).unsqueeze(-1), out, torch.zeros_like(out))


def softmax_state_update(
    m: torch.Tensor, l: torch.Tensor, block_scores: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """One functional step of the streaming normaliser. Returns `(m_new, l_new)`.

    Kept separate from the class so the identity can be tested on its own, without any attention
    machinery in the way.
    """
    m_new = torch.maximum(m, block_scores.amax(dim=-1))
    finite = torch.isfinite(m_new)
    correction = torch.where(finite, torch.exp(m - m_new), torch.zeros_like(m_new))
    probs = torch.where(finite.unsqueeze(-1),
                        torch.exp(block_scores - m_new.unsqueeze(-1)),
                        torch.zeros_like(block_scores))
    l_new = l * torch.nan_to_num(correction, nan=0.0) + torch.nan_to_num(probs, nan=0.0).sum(dim=-1)
    return m_new, l_new


def online_softmax(x: torch.Tensor, *, block_size: int = 32) -> torch.Tensor:
    """Softmax over the last dimension, computed in one streaming pass over blocks.

    Exists to demonstrate and test the identity in isolation. It is **not** faster than
    `torch.softmax` — it does the same arithmetic in a Python loop instead of one fused kernel, so it
    is considerably slower. The payoff appears only when the input is *generated* block-by-block and
    never stored, which is exactly the situation in attention and is what `tiled.py` exploits.
    """
    *lead, n = x.shape
    flat = x.reshape(-1, n)
    rows = flat.shape[0]

    m = torch.full((rows,), float("-inf"), device=x.device, dtype=x.dtype)
    l = torch.zeros((rows,), device=x.device, dtype=x.dtype)
    for start in range(0, n, block_size):
        m, l = softmax_state_update(m, l, flat[:, start : start + block_size])

    # Second pass here only because this helper returns the full probability matrix, which the caller
    # asked for. Attention never needs it -- it needs softmax(s) @ V, which the accumulator in
    # OnlineSoftmaxState produces in a single pass.
    safe_l = torch.where(l > 0, l, torch.ones_like(l))
    out = torch.exp(flat - m.unsqueeze(-1)) / safe_l.unsqueeze(-1)
    out = torch.nan_to_num(out, nan=0.0)
    return out.reshape(*lead, n)
