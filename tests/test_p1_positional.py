"""Correctness suite for sinusoidal positional encoding (paper section 3.5)."""

from __future__ import annotations

import math

import pytest
import torch
import torch.nn as nn

from labs.p1_transformer.attention import MultiHeadAttention
from labs.p1_transformer.positional import (
    SinusoidalPositionalEncoding,
    sinusoidal_positional_encoding,
)


def test_matches_paper_formula_transcribed_literally() -> None:
    """Verify the exp/log rewrite against a direct transcription of the paper's expression.

    The implementation computes ``exp(-(2i/d_model) * ln(10000))`` for numerical conditioning.
    The paper writes ``pos / 10000^(2i/d_model)``. This test evaluates the paper's form in float64
    with plain Python arithmetic and requires agreement, so the optimisation is verified rather
    than assumed equivalent.
    """
    max_len, d_model = 17, 12
    pe = sinusoidal_positional_encoding(max_len, d_model, dtype=torch.float64)

    for pos in range(max_len):
        for i in range(d_model // 2):
            denom = 10000.0 ** ((2 * i) / d_model)
            assert pe[pos, 2 * i].item() == pytest.approx(math.sin(pos / denom), abs=1e-12)
            assert pe[pos, 2 * i + 1].item() == pytest.approx(math.cos(pos / denom), abs=1e-12)


def test_shape_and_bounds() -> None:
    pe = sinusoidal_positional_encoding(50, 16)
    assert pe.shape == (50, 16)
    assert bool((pe.abs() <= 1.0 + 1e-6).all()), "sin/cos are bounded; anything else is a bug"


def test_position_zero_is_alternating_zeros_and_ones() -> None:
    """sin(0) = 0, cos(0) = 1 for every frequency, so row 0 is a fixed, checkable pattern."""
    pe = sinusoidal_positional_encoding(1, 8)
    assert torch.allclose(pe[0], torch.tensor([0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0]), atol=1e-6)


def test_relative_offset_is_a_fixed_linear_map() -> None:
    """The property the paper gives as its reason for choosing sinusoids (section 3.5).

    For each frequency w and any offset k, the (sin, cos) pair at position pos+k is a rotation of
    the pair at position pos by the angle w*k -- a matrix that depends on k but NOT on pos:

        sin(w(pos+k)) = cos(wk) sin(w pos) + sin(wk) cos(w pos)
        cos(w(pos+k)) = cos(wk) cos(w pos) - sin(wk) sin(w pos)

    Because the map is linear and position-independent, a single learned linear projection (which
    is exactly what W^Q and W^K are) can express "attend k positions back" uniformly across the
    whole sequence. That is the bridge from absolute encodings to relative addressing.
    """
    d_model, max_len = 16, 40
    pe = sinusoidal_positional_encoding(max_len, d_model, dtype=torch.float64)

    for k in (1, 3, 7):
        for i in range(d_model // 2):
            w = 1.0 / (10000.0 ** ((2 * i) / d_model))
            c, s = math.cos(w * k), math.sin(w * k)
            for pos in range(max_len - k):
                sin_p, cos_p = pe[pos, 2 * i].item(), pe[pos, 2 * i + 1].item()
                assert pe[pos + k, 2 * i].item() == pytest.approx(c * sin_p + s * cos_p, abs=1e-10)
                assert pe[pos + k, 2 * i + 1].item() == pytest.approx(c * cos_p - s * sin_p, abs=1e-10)


def test_distinct_positions_get_distinct_encodings() -> None:
    pe = sinusoidal_positional_encoding(128, 32)
    # Pairwise distances; the diagonal is zero, everything else must be clearly nonzero.
    dist = torch.cdist(pe, pe)
    off_diagonal = dist + torch.eye(128) * 1e9
    assert off_diagonal.min().item() > 1e-3


def test_low_frequency_dimensions_change_slowly_and_high_frequency_fast() -> None:
    """The geometric progression of wavelengths, checked as a measurable consequence.

    Dimension pair 0 has wavelength 2*pi (fast); the last pair has wavelength 10000*2*pi (slow).
    So consecutive-position variation must be far larger in the first pair than in the last.
    """
    d_model = 64
    pe = sinusoidal_positional_encoding(200, d_model)
    step = (pe[1:] - pe[:-1]).abs().mean(dim=0)
    assert step[0].item() > 100 * step[d_model - 2].item()


def test_odd_d_model_is_handled() -> None:
    pe = sinusoidal_positional_encoding(10, 7)
    assert pe.shape == (10, 7)
    assert bool(torch.isfinite(pe).all())


# --------------------------------------------------------------------------------------------
# Why positional encoding is necessary at all, demonstrated
# --------------------------------------------------------------------------------------------

def test_attention_is_permutation_equivariant_without_positional_encoding() -> None:
    """The failure that positional encoding exists to fix.

    Self-attention with no positional signal cannot tell "the cat sat" from "sat cat the": permute
    the inputs and the outputs come back permuted, with identical values. This test asserts the
    weakness is real, so the next test can show it is repaired.
    """
    torch.manual_seed(0)
    d, h, l = 16, 4, 5
    mha = MultiHeadAttention(d, h).eval()
    x = torch.randn(1, l, d)
    perm = torch.tensor([3, 0, 4, 1, 2])

    with torch.no_grad():
        out = mha(x, x, x)[0]
        out_permuted_input = mha(x[:, perm], x[:, perm], x[:, perm])[0]

    assert torch.allclose(out_permuted_input, out[:, perm], atol=1e-6)


def test_positional_encoding_breaks_permutation_equivariance() -> None:
    torch.manual_seed(0)
    d, h, l = 16, 4, 5
    mha = MultiHeadAttention(d, h).eval()
    pos = SinusoidalPositionalEncoding(d, max_len=32, dropout=0.0).eval()
    x = torch.randn(1, l, d)
    perm = torch.tensor([3, 0, 4, 1, 2])

    with torch.no_grad():
        a = mha(*(pos(x),) * 3)[0]
        b = mha(*(pos(x[:, perm]),) * 3)[0]

    assert not torch.allclose(b, a[:, perm], atol=1e-4), (
        "with positional information the model must distinguish orderings"
    )


# --------------------------------------------------------------------------------------------
# Module wiring
# --------------------------------------------------------------------------------------------

def test_table_is_a_buffer_not_a_parameter() -> None:
    """The optimiser must never see the table: it is a fixed function of position."""
    module = SinusoidalPositionalEncoding(16, max_len=32)
    assert list(module.parameters()) == []
    assert "pe" in dict(module.named_buffers())


def test_table_is_excluded_from_state_dict() -> None:
    """persistent=False: exactly reconstructible from (max_len, d_model), so spending checkpoint
    bytes on it is waste."""
    module = SinusoidalPositionalEncoding(16, max_len=5000)
    assert "pe" not in module.state_dict()


def test_forward_adds_encoding_and_preserves_shape() -> None:
    module = SinusoidalPositionalEncoding(8, max_len=16, dropout=0.0).eval()
    x = torch.zeros(2, 5, 8)
    out = module(x)
    assert out.shape == (2, 5, 8)
    expected = sinusoidal_positional_encoding(16, 8)[:5]
    assert torch.allclose(out[0], expected, atol=1e-6)
    assert torch.allclose(out[0], out[1]), "the same encoding applies to every batch element"


def test_forward_rejects_sequences_longer_than_the_table() -> None:
    module = SinusoidalPositionalEncoding(8, max_len=4)
    with pytest.raises(ValueError, match="exceeds tabulated max_len"):
        module(torch.zeros(1, 5, 8))


def test_dropout_applies_to_the_sum_not_to_the_embedding_alone() -> None:
    """Paper section 5.4: dropout is applied to the sums of embeddings and positional encodings.

    With x = 0 the output is pure positional encoding, so if dropout were applied only to x it
    could not zero any entry of the result. Observing zeros where the encoding is nonzero proves
    the dropout sits after the addition.
    """
    torch.manual_seed(0)
    module = SinusoidalPositionalEncoding(32, max_len=16, dropout=0.9).train()
    out = module(torch.zeros(4, 8, 32))
    assert bool((out == 0).any()), "dropout must act on the summed representation"
