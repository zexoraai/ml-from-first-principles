"""Sinusoidal positional encoding.

Paper: Vaswani et al., arXiv:1706.03762v7, section 3.5.

    PE(pos, 2i)     = sin( pos / 10000^(2i / d_model) )
    PE(pos, 2i + 1) = cos( pos / 10000^(2i / d_model) )

--------------------------------------------------------------------------------------------
WHY POSITIONAL ENCODING EXISTS AT ALL
--------------------------------------------------------------------------------------------
Strip the positional information away and self-attention is *permutation-equivariant*: shuffle
the input tokens and the output tokens come back shuffled the same way, with identical values.
Nothing inside `softmax(QK^T/sqrt(d_k))V` depends on the index of a position -- only on the
content of the vectors. So "the cat sat" and "sat cat the" would be indistinguishable.
Recurrence got order for free by consuming tokens in sequence; attention threw recurrence away
and has to buy order back explicitly. That is the entire job of this file, and
`tests/test_p1_positional.py::test_attention_is_permutation_equivariant_without_pe` demonstrates
the failure mode directly rather than asserting it.

--------------------------------------------------------------------------------------------
WHY SINUSOIDS, SPECIFICALLY
--------------------------------------------------------------------------------------------
The paper's stated reason (section 3.5) is that for any fixed offset k, PE(pos+k) is a *linear*
function of PE(pos). Concretely, each (sin, cos) pair at frequency w behaves as a point on a
circle, and advancing the position by k rotates it by the fixed angle w*k:

    [ sin(w(pos+k)) ]   [ cos(wk)   sin(wk) ] [ sin(w pos) ]
    [ cos(w(pos+k)) ] = [ -sin(wk)  cos(wk) ] [ cos(w pos) ]

The rotation matrix depends on k but **not** on pos. So a single learned linear map -- exactly
what W^Q and W^K are -- can implement "look 3 positions back" uniformly across the sequence.
That is the mechanism by which relative position becomes learnable from absolute encodings, and
`test_relative_offset_is_a_fixed_linear_map` verifies the identity numerically.

The wavelengths form a geometric progression from 2*pi (i = 0) to 10000 * 2*pi (i = d_model/2),
so the encoding carries both fast-changing dimensions that separate neighbours and slow-changing
dimensions that distinguish the start of a long sequence from its end.

The paper also reports (Table 3, row E) that *learned* positional embeddings performed nearly
identically. Sinusoids were chosen for a different reason: they are defined for positions never
seen in training, so the model may extrapolate to longer sequences. Whether it actually does is
an empirical question and not something this file should be read as claiming.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

__all__ = ["sinusoidal_positional_encoding", "SinusoidalPositionalEncoding"]


def sinusoidal_positional_encoding(
    max_len: int,
    d_model: int,
    *,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Build the (max_len, d_model) table of section 3.5.

    Args:
        max_len: number of positions to tabulate.
        d_model: encoding width; must match the embedding width so the two can be summed.

    Returns:
        Tensor of shape (max_len, d_model). Row `pos` is the encoding of position `pos`.

    Numerical note on how the exponent is computed
    ---------------------------------------------
    The paper writes the divisor as ``10000^(2i / d_model)``. Computing that directly means a
    large base raised to a fractional power, then a division -- fine in fp64, avoidable
    precision loss in fp32. We use the algebraically identical form

        1 / 10000^(2i/d_model) = exp( -(2i/d_model) * ln(10000) )

    which is one exponential of a small negative number. Same values, better conditioned, and
    it is the form used by the reference implementations. `test_matches_paper_formula_directly`
    checks this rewrite against a literal transcription of the paper's expression in fp64, so
    the optimisation is verified rather than trusted.
    """
    if d_model <= 0:
        raise ValueError(f"d_model must be positive, got {d_model}")
    if max_len <= 0:
        raise ValueError(f"max_len must be positive, got {max_len}")

    pe = torch.zeros(max_len, d_model, device=device, dtype=dtype)
    position = torch.arange(max_len, device=device, dtype=dtype).unsqueeze(1)   # (max_len, 1)

    # 2i for i = 0, 1, ... -> 0, 2, 4, ...  (one value per sin/cos PAIR)
    two_i = torch.arange(0, d_model, 2, device=device, dtype=dtype)             # (ceil(d_model/2),)
    inv_freq = torch.exp(-(two_i / d_model) * math.log(10000.0))                # (ceil(d_model/2),)

    angles = position * inv_freq                     # (max_len, ceil(d_model/2)) by broadcasting
    pe[:, 0::2] = torch.sin(angles)
    # For odd d_model the final sine column has no cosine partner; slice the angles to match.
    pe[:, 1::2] = torch.cos(angles[:, : pe[:, 1::2].size(1)])
    return pe


class SinusoidalPositionalEncoding(nn.Module):
    """Adds the section 3.5 encoding to an embedded sequence, then applies dropout.

    Args:
        d_model: width.
        max_len: largest position tabulated at construction time.
        dropout: paper section 5.4 -- "we apply dropout to the sums of the embeddings and the
            positional encodings in both the encoder and decoder stacks". So the dropout
            belongs *here*, after the addition, not before it.

    Registered as a **buffer, not a parameter**
    ------------------------------------------
    The table is a fixed function of position with nothing to learn. Registering it as a buffer
    means it moves with `.to(device)` and appears in `state_dict()` (so a checkpoint is
    self-describing) while `parameters()` and the optimiser never see it. Making it a
    `nn.Parameter` would silently hand the optimiser d_model * max_len values to wreck; making
    it a plain attribute would leave it on the wrong device the first time the model moved.

    `persistent=False` keeps it out of the saved state_dict, because it is exactly reconstructible
    from (max_len, d_model) and there is no reason to spend checkpoint bytes on it.
    """

    def __init__(self, d_model: int, *, max_len: int = 5000, dropout: float = 0.0) -> None:
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len
        self.dropout = nn.Dropout(dropout)
        self.register_buffer(
            "pe", sinusoidal_positional_encoding(max_len, d_model), persistent=False
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Args: x of shape (batch, seq_len, d_model). Returns the same shape.

        Shape trace
        -----------
            x                 (batch, seq_len, d_model)
            self.pe           (max_len, d_model)
            self.pe[:seq_len] (seq_len, d_model)      -> broadcasts over batch
            x + pe            (batch, seq_len, d_model)
        """
        seq_len = x.size(1)
        if seq_len > self.max_len:
            raise ValueError(
                f"sequence length {seq_len} exceeds tabulated max_len {self.max_len}. Sinusoids "
                f"are defined for any position, so this is a table-size limit, not a model limit: "
                f"construct with max_len >= {seq_len}."
            )
        if x.size(-1) != self.d_model:
            raise ValueError(f"input width {x.size(-1)} != d_model {self.d_model}")
        return self.dropout(x + self.pe[:seq_len].unsqueeze(0))
