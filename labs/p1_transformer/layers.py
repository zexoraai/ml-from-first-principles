"""Layer normalization, the position-wise feed-forward network, and the residual wiring.

Paper: Vaswani et al., arXiv:1706.03762v7, sections 3.1, 3.3, 5.4.
LayerNorm itself: Ba, Kiros & Hinton, arXiv:1607.06450 (reference [1] of the Transformer paper).

`torch.nn.LayerNorm` is deliberately not used -- normalization is one of the mechanisms this
project claims to implement. It appears in `tests/` as an oracle only.
"""

from __future__ import annotations

from typing import Callable, Literal

import torch
import torch.nn as nn

__all__ = ["LayerNorm", "PositionwiseFeedForward", "SublayerConnection"]


class LayerNorm(nn.Module):
    """Layer normalization over the last dimension.

        y = (x - mean(x)) / sqrt(var(x) + eps) * gamma + beta

    where mean and var are taken over the **feature** axis of each token independently.

    What it normalizes, and why that choice
    ---------------------------------------
    For an input of shape (batch, seq_len, d_model) the statistics are computed per (batch,
    position) pair, over the d_model features. Every token vector is rescaled using only its
    own d_model numbers. Nothing is shared across the batch and nothing is shared across
    positions.

    That independence is the property that matters, and it is why LayerNorm and not BatchNorm:
      * Sequence lengths vary and batches contain padding. Batch statistics would be polluted
        by pad positions, so the normalisation of a real token would depend on how much padding
        happened to sit next to it in the batch.
      * At inference we often run a single sequence. Batch statistics of one example are
        degenerate, so BatchNorm needs a separate train/eval code path with running averages.
        LayerNorm behaves identically in both modes -- there is no train/eval divergence to get
        wrong.
      * Autoregressive decoding produces one position at a time. A statistic pooled over
        positions would leak information across the causal boundary.

    Two details that must match `nn.LayerNorm` for parity, and are easy to get wrong
    -------------------------------------------------------------------------------
    1. **Biased variance.** We divide the sum of squared deviations by `d_model`, not by
       `d_model - 1` (`unbiased=False`). The Bessel correction estimates a population variance
       from a sample; here we are not estimating anything, we are rescaling a fixed vector. Using
       `unbiased=True` produces a subtly different scale that grows as d_model shrinks, and it
       silently breaks numerical parity against the reference.
    2. **eps inside the square root**, added to the variance, not to the standard deviation.
       `sqrt(var + eps)` and `sqrt(var) + eps` differ, and the former is what every reference
       implementation uses.

    Args:
        d_model: size of the normalized (last) dimension.
        eps: numerical floor. 1e-5 matches `torch.nn.LayerNorm`'s default, chosen so the parity
            test is a test of our arithmetic rather than of a constant mismatch.
        elementwise_affine: whether to learn `gamma` and `beta`.
    """

    def __init__(self, d_model: int, *, eps: float = 1e-5, elementwise_affine: bool = True) -> None:
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if elementwise_affine:
            # gamma initialised to ones and beta to zeros so the layer starts as the identity
            # (up to normalisation) and cannot destroy the signal before training begins.
            self.gamma = nn.Parameter(torch.ones(d_model))
            self.beta = nn.Parameter(torch.zeros(d_model))
        else:
            self.register_parameter("gamma", None)
            self.register_parameter("beta", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Args: x of shape (..., d_model). Returns the same shape."""
        if x.size(-1) != self.d_model:
            raise ValueError(f"last dim {x.size(-1)} != d_model {self.d_model}")
        mean = x.mean(dim=-1, keepdim=True)                       # (..., 1)
        var = x.var(dim=-1, keepdim=True, unbiased=False)         # (..., 1) biased -- see docstring
        x_hat = (x - mean) / torch.sqrt(var + self.eps)
        if self.elementwise_affine:
            x_hat = x_hat * self.gamma + self.beta
        return x_hat


class PositionwiseFeedForward(nn.Module):
    """Section 3.3.

        FFN(x) = max(0, x W_1 + b_1) W_2 + b_2

    Args:
        d_model: input/output width (512 in the paper's base model).
        d_ff: inner width (2048 in the base model, i.e. 4x d_model).
        dropout: applied to the hidden activation.
        activation: "relu" for the paper as written. "gelu" is offered because every later
            model switched to it; it is flagged as a deviation, never a default.

    "Position-wise" is the load-bearing word
    ----------------------------------------
    The same two matrices are applied to every position independently. There is no mixing across
    the sequence axis here at all -- position 7's output depends only on position 7's input. All
    cross-position communication in a Transformer happens in the attention sub-layer, and only
    there. That division of labour is worth stating precisely, because it is what makes the
    architecture easy to reason about: attention moves information between positions, the FFN
    transforms information within a position.

    The paper notes this is equivalent to two convolutions with kernel size 1. Same thing said
    two ways: a width-1 convolution touches one position at a time.

    Why the 4x expansion
    --------------------
    The paper states d_ff = 2048 without deriving it. The widely-held reading is that the
    expand-then-project shape gives the layer room to compute nonlinear features in a higher
    dimensional space before compressing back, and that most of a Transformer's parameters (and,
    at inference, most of its FLOPs at short sequence lengths) live here: 2 * d_model * d_ff per
    layer versus 4 * d_model^2 for attention, i.e. 4:1 at the base configuration. Treat "4x is
    optimal" as convention, not as a result -- the paper reports no ablation over d_ff alone.
    """

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        *,
        dropout: float = 0.1,
        activation: Literal["relu", "gelu"] = "relu",
    ) -> None:
        super().__init__()
        # Biases are present here because the paper writes b_1 and b_2 explicitly in section 3.3,
        # in contrast to the attention projections which it writes as bare matrices W_i^Q etc.
        self.w_1 = nn.Linear(d_model, d_ff, bias=True)
        self.w_2 = nn.Linear(d_ff, d_model, bias=True)
        self.dropout = nn.Dropout(dropout)
        self.activation_name = activation
        self.activation: Callable[[torch.Tensor], torch.Tensor] = (
            torch.relu if activation == "relu" else nn.functional.gelu
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Xavier-uniform, matching the reference implementations. The paper is silent."""
        for proj in (self.w_1, self.w_2):
            nn.init.xavier_uniform_(proj.weight)
            nn.init.zeros_(proj.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Shape trace: (batch, seq, d_model) -> (batch, seq, d_ff) -> (batch, seq, d_model)."""
        return self.w_2(self.dropout(self.activation(self.w_1(x))))


class SublayerConnection(nn.Module):
    """Residual connection + dropout + layer normalization around one sub-layer.

    Two placements, and they are not interchangeable
    ------------------------------------------------
    `norm_style="post"` -- what the **paper text** specifies (section 3.1: "the output of each
    sub-layer is LayerNorm(x + Sublayer(x))"), with dropout placed per section 5.4 ("we apply
    dropout to the output of each sub-layer, before it is added to the sub-layer input and
    normalized"):

        y = LayerNorm( x + Dropout(Sublayer(x)) )

    `norm_style="pre"` -- what the authors' own `tensor2tensor` code actually shipped, and what
    essentially every model since GPT-2 uses:

        y = x + Dropout(Sublayer(LayerNorm(x)))

    Why the difference matters more than it looks
    ---------------------------------------------
    In the pre-norm form there is an unbroken identity path from the input of the stack to the
    output: the residual branch is added without passing through any normalisation. The gradient
    of the loss with respect to an early layer therefore contains a term that is not scaled by
    any LayerNorm Jacobian, so deep stacks train from initialisation without heroics.

    In the post-norm form every residual sum is immediately normalised, so the identity path is
    rescaled at every one of the N layers. That is why the original recipe *needs* the warmup
    schedule of section 5.3: early in training the normalisation statistics and the parameters
    are mismatched, gradients through deep post-norm stacks are badly scaled, and a large initial
    learning rate diverges. Warmup is not a decorative detail bolted onto post-norm -- it is
    what makes post-norm trainable.

    We default to `post` because decision D-006 commits to implementing what the paper says, and
    we cite section 3.1 when we say so. `pre` is available so the comparison is a runnable
    experiment on the project page rather than an anecdote.
    """

    def __init__(
        self,
        d_model: int,
        *,
        dropout: float = 0.1,
        norm_style: Literal["post", "pre"] = "post",
        eps: float = 1e-5,
    ) -> None:
        super().__init__()
        if norm_style not in ("post", "pre"):
            raise ValueError(f"norm_style must be 'post' or 'pre', got {norm_style!r}")
        self.norm = LayerNorm(d_model, eps=eps)
        self.dropout = nn.Dropout(dropout)
        self.norm_style = norm_style

    def forward(self, x: torch.Tensor, sublayer: Callable[[torch.Tensor], torch.Tensor]) -> torch.Tensor:
        """Apply `sublayer` to `x` with the configured residual/normalization wiring.

        Args:
            x: (batch, seq, d_model)
            sublayer: callable taking a tensor of x's shape and returning the same shape. The
                caller closes over any extra arguments (masks, encoder memory), which keeps this
                class agnostic about *which* sub-layer it is wrapping -- the same object wraps
                self-attention, cross-attention, and the FFN.

        Returns: (batch, seq, d_model)
        """
        if self.norm_style == "post":
            return self.norm(x + self.dropout(sublayer(x)))
        return x + self.dropout(sublayer(self.norm(x)))
