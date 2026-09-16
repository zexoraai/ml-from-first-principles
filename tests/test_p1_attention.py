"""Correctness suite for scaled dot-product attention and multi-head attention.

Three kinds of check, in increasing order of how much they actually prove:

1. **Parity against independent oracles.** `torch.nn.functional.scaled_dot_product_attention`
   and `torch.nn.MultiheadAttention` are used *here* to verify our implementation. That is the
   opposite of using them to implement it, and the distinction is the point: an oracle you did
   not write is the strongest cheap evidence available.
2. **Parity against a naive transcription.** Explicit Python loops over batch/head/query/key,
   in float64, computing the paper's formula one dot product at a time. This catches errors that
   an oracle sharing our conceptual mistakes would not.
3. **Behavioural / interventional invariants.** Not "is there a zero in the weight matrix" but
   "if I change a masked token, does any permitted output move". Checking the weight matrix
   tests the mask; checking the output tests the *masking*. Only the second one is the property
   we actually care about, and only the second one would catch a mask applied at the wrong axis.
"""

from __future__ import annotations

import math

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from labs.p1_transformer.attention import (
    MultiHeadAttention,
    merge_heads,
    scaled_dot_product_attention,
    split_heads,
)
from labs.p1_transformer.masks import causal_keep_mask, padding_key_mask

TOL = 1e-5


# --------------------------------------------------------------------------------------------
# 1. Parity against PyTorch's own kernel
# --------------------------------------------------------------------------------------------

@pytest.mark.parametrize("keep_kind", ["none", "causal"])
def test_matches_torch_sdpa_oracle(keep_kind: str) -> None:
    b, h, lq, lk, dk, dv = 2, 3, 5, 5, 8, 8
    q = torch.randn(b, h, lq, dk)
    k = torch.randn(b, h, lk, dk)
    v = torch.randn(b, h, lk, dv)
    keep = None if keep_kind == "none" else causal_keep_mask(lq, lk)

    ours, _ = scaled_dot_product_attention(q, k, v, keep_mask=keep)
    # F.scaled_dot_product_attention uses the SAME convention we chose: bool True = keep.
    theirs = F.scaled_dot_product_attention(q, k, v, attn_mask=keep)

    assert torch.allclose(ours, theirs, atol=TOL), (ours - theirs).abs().max()


def test_matches_torch_sdpa_with_padding_mask() -> None:
    b, h, lk, dk = 2, 2, 6, 8
    ids = torch.tensor([[3, 4, 5, 0, 0, 0], [7, 8, 9, 1, 2, 0]])
    keep = padding_key_mask(ids, pad_id=0)                 # (b,1,1,lk)
    q = torch.randn(b, h, lk, dk)
    k = torch.randn(b, h, lk, dk)
    v = torch.randn(b, h, lk, dk)

    ours, _ = scaled_dot_product_attention(q, k, v, keep_mask=keep)
    theirs = F.scaled_dot_product_attention(q, k, v, attn_mask=keep.expand(b, h, lk, lk))
    assert torch.allclose(ours, theirs, atol=TOL)


# --------------------------------------------------------------------------------------------
# 2. Parity against a naive float64 transcription of Equation 1
# --------------------------------------------------------------------------------------------

def _naive_attention_f64(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, keep: torch.Tensor | None
) -> torch.Tensor:
    """Equation 1, one dot product at a time, in float64. Intentionally slow and literal."""
    q, k, v = q.double(), k.double(), v.double()
    b, h, lq, dk = q.shape
    lk, dv = k.shape[2], v.shape[-1]
    out = torch.zeros(b, h, lq, dv, dtype=torch.float64)

    keep_full = None
    if keep is not None:
        keep_full = keep.expand(b, h, lq, lk)

    scale = 1.0 / math.sqrt(dk)
    for bi in range(b):
        for hi in range(h):
            for i in range(lq):
                raw: list[float] = []
                for j in range(lk):
                    if keep_full is not None and not bool(keep_full[bi, hi, i, j]):
                        raw.append(-math.inf)
                    else:
                        raw.append(float(torch.dot(q[bi, hi, i], k[bi, hi, j])) * scale)
                m = max(raw)
                if m == -math.inf:              # fully masked row -> defined as zeros
                    continue
                exps = [0.0 if s == -math.inf else math.exp(s - m) for s in raw]
                z = sum(exps)
                for j, e in enumerate(exps):
                    out[bi, hi, i] += (e / z) * v[bi, hi, j]
    return out


@pytest.mark.parametrize("keep_kind", ["none", "causal", "padding"])
def test_matches_naive_float64_transcription(keep_kind: str) -> None:
    """Both sides run in float64 so the tolerance tests arithmetic, not float width.

    Comparing an fp32 forward pass against an fp64 reference forces a tolerance around 1e-6,
    which is loose enough to hide a genuine sign or indexing error. Promoting our own inputs to
    float64 removes precision from the equation entirely and lets the tolerance drop to 1e-12,
    where only real logic errors survive. fp32 behaviour is covered separately by the
    `F.scaled_dot_product_attention` parity tests above.
    """
    b, h, l, dk = 2, 2, 4, 6
    q, k, v = (torch.randn(b, h, l, dk, dtype=torch.float64) for _ in range(3))
    if keep_kind == "none":
        keep = None
    elif keep_kind == "causal":
        keep = causal_keep_mask(l)
    else:
        keep = padding_key_mask(torch.tensor([[1, 1, 0, 0], [1, 1, 1, 0]]), pad_id=0)

    ours, _ = scaled_dot_product_attention(q, k, v, keep_mask=keep)
    ref = _naive_attention_f64(q, k, v, keep)
    assert torch.allclose(ours, ref, rtol=0.0, atol=1e-12), (ours - ref).abs().max()


# --------------------------------------------------------------------------------------------
# 3. Properties of the weight matrix
# --------------------------------------------------------------------------------------------

def test_weights_form_a_probability_distribution_over_keys() -> None:
    q, k, v = (torch.randn(2, 2, 5, 8) for _ in range(3))
    _, w = scaled_dot_product_attention(q, k, v)
    assert w.shape == (2, 2, 5, 5)
    assert bool((w >= 0).all())
    assert torch.allclose(w.sum(dim=-1), torch.ones(2, 2, 5), atol=TOL), "must normalise over KEYS"


def test_softmax_is_over_the_key_axis_not_the_query_axis() -> None:
    """Guards the single most common transposition bug.

    If softmax were applied over queries (dim=-2) the columns would sum to 1 instead of the rows.
    With a non-square score matrix the two are not even the same shape, which is why the test
    uses q_len != k_len.
    """
    q = torch.randn(1, 1, 3, 8)
    k = torch.randn(1, 1, 7, 8)
    v = torch.randn(1, 1, 7, 8)
    _, w = scaled_dot_product_attention(q, k, v)
    assert w.shape == (1, 1, 3, 7)
    assert torch.allclose(w.sum(dim=-1), torch.ones(1, 1, 3), atol=TOL)
    assert not torch.allclose(w.sum(dim=-2), torch.ones(1, 1, 7), atol=1e-3)


def test_masked_positions_receive_exactly_zero_weight() -> None:
    q, k, v = (torch.randn(1, 1, 6, 8) for _ in range(3))
    keep = causal_keep_mask(6)
    _, w = scaled_dot_product_attention(q, k, v, keep_mask=keep)
    forbidden = w[0, 0][~keep[0, 0]]
    # exp(-inf) == 0 exactly, so this is an equality assertion, not an approximate one.
    assert bool((forbidden == 0).all()), forbidden.abs().max()


def test_scaling_keeps_score_variance_near_one() -> None:
    """Measures the paper's footnote-4 argument instead of restating it.

    With q, k componentwise ~ N(0,1) and dimension d_k, the raw dot product has variance d_k.
    Dividing by sqrt(d_k) brings it back to 1. Both halves are checked so the test fails if the
    scale factor is removed *or* doubled.
    """
    torch.manual_seed(0)
    d_k = 64
    q = torch.randn(1, 1, 512, d_k)
    k = torch.randn(1, 1, 512, d_k)

    raw = (q @ k.transpose(-2, -1))
    scaled = raw / math.sqrt(d_k)

    assert raw.var().item() == pytest.approx(d_k, rel=0.15)
    assert scaled.var().item() == pytest.approx(1.0, rel=0.15)


def test_large_dk_without_scaling_saturates_softmax() -> None:
    """The consequence the scaling exists to prevent, demonstrated rather than asserted.

    Unscaled scores at d_k = 256 push softmax towards one-hot: the maximum probability per row
    approaches 1 and the entropy collapses, which is exactly the low-gradient regime the paper
    warns about.
    """
    torch.manual_seed(0)
    d_k = 256
    q = torch.randn(1, 1, 32, d_k)
    k = torch.randn(1, 1, 32, d_k)
    raw = q @ k.transpose(-2, -1)

    unscaled_max = torch.softmax(raw, dim=-1).max(dim=-1).values.mean().item()
    scaled_max = torch.softmax(raw / math.sqrt(d_k), dim=-1).max(dim=-1).values.mean().item()

    assert unscaled_max > 0.9, unscaled_max
    assert scaled_max < 0.5, scaled_max


# --------------------------------------------------------------------------------------------
# 4. Interventional proof that masking actually blocks information flow
# --------------------------------------------------------------------------------------------

def test_causal_mask_blocks_information_from_the_future() -> None:
    """The real causality test: perturb a future token, assert earlier outputs are bit-identical.

    A test that only inspects the weight matrix would pass even if the mask were transposed or
    applied to the wrong axis. This one would not.
    """
    torch.manual_seed(0)
    b, l, d, h = 2, 6, 16, 4
    mha = MultiHeadAttention(d, h).eval()
    x = torch.randn(b, l, d)
    keep = causal_keep_mask(l)

    with torch.no_grad():
        out_before, _ = mha(x, x, x, keep_mask=keep)
        x_perturbed = x.clone()
        x_perturbed[:, 4:] += 10.0            # rewrite the future
        out_after, _ = mha(x_perturbed, x_perturbed, x_perturbed, keep_mask=keep)

    assert torch.equal(out_before[:, :4], out_after[:, :4]), (
        (out_before[:, :4] - out_after[:, :4]).abs().max()
    )
    # Sanity: the perturbation must actually matter somewhere, or the test proves nothing.
    assert not torch.allclose(out_before[:, 4:], out_after[:, 4:])


def test_padding_mask_blocks_information_from_padding_content() -> None:
    """Changing what sits in a padded slot must not move any real position's output."""
    torch.manual_seed(0)
    b, l, d, h = 2, 6, 16, 4
    mha = MultiHeadAttention(d, h).eval()
    ids = torch.tensor([[3, 4, 5, 0, 0, 0], [7, 8, 9, 2, 0, 0]])
    keep = padding_key_mask(ids, pad_id=0)
    real = ids != 0

    x = torch.randn(b, l, d)
    with torch.no_grad():
        out_before, _ = mha(x, x, x, keep_mask=keep)
        x_perturbed = x.clone()
        x_perturbed[~real] += 50.0            # garbage in the pad slots
        out_after, _ = mha(x_perturbed, x_perturbed, x_perturbed, keep_mask=keep)

    assert torch.equal(out_before[real], out_after[real])


def test_fully_masked_row_yields_zeros_not_nan() -> None:
    q, k, v = (torch.randn(1, 1, 2, 4) for _ in range(3))
    keep = torch.tensor([[[[True, True], [False, False]]]])
    out, w = scaled_dot_product_attention(q, k, v, keep_mask=keep)

    assert not bool(torch.isnan(out).any()), "NaN here would poison every upstream gradient"
    assert not bool(torch.isnan(w).any())
    assert torch.equal(w[0, 0, 1], torch.zeros(2))
    assert torch.equal(out[0, 0, 1], torch.zeros(4))


def test_float_mask_is_rejected_rather_than_reinterpreted() -> None:
    q, k, v = (torch.randn(1, 1, 3, 4) for _ in range(3))
    with pytest.raises(TypeError, match="bool"):
        scaled_dot_product_attention(q, k, v, keep_mask=torch.zeros(1, 1, 3, 3))


# --------------------------------------------------------------------------------------------
# 5. Head split / merge
# --------------------------------------------------------------------------------------------

def test_split_then_merge_is_the_identity() -> None:
    x = torch.randn(2, 5, 12)
    assert torch.equal(merge_heads(split_heads(x, 4)), x)


def test_split_heads_slices_the_feature_axis_contiguously() -> None:
    """Head i must receive features [i*d_h : (i+1)*d_h], matching the block structure implied by
    concatenating the paper's per-head matrices column-wise."""
    x = torch.arange(2 * 3 * 12, dtype=torch.float32).reshape(2, 3, 12)
    heads = split_heads(x, 4)
    assert heads.shape == (2, 4, 3, 3)
    for i in range(4):
        assert torch.equal(heads[:, i], x[:, :, i * 3:(i + 1) * 3])


def test_split_heads_rejects_indivisible_width() -> None:
    with pytest.raises(ValueError, match="divisible"):
        split_heads(torch.randn(1, 2, 10), 4)


# --------------------------------------------------------------------------------------------
# 6. Multi-head attention: shapes, parity with nn.MultiheadAttention, gradients
# --------------------------------------------------------------------------------------------

def test_mha_shape_contract() -> None:
    b, lq, lk, d, h = 3, 5, 7, 32, 8
    mha = MultiHeadAttention(d, h)
    out, w = mha(torch.randn(b, lq, d), torch.randn(b, lk, d), torch.randn(b, lk, d))
    assert out.shape == (b, lq, d)
    assert w.shape == (b, h, lq, lk)
    assert mha.head_dim == d // h == 4


def test_mha_rejects_indivisible_d_model() -> None:
    with pytest.raises(ValueError, match="divisible"):
        MultiHeadAttention(30, 8)


def test_mha_matches_nn_multiheadattention_after_weight_transfer() -> None:
    """Strongest single test in this file.

    Copy our four projection matrices into `nn.MultiheadAttention`'s packed layout and require
    identical outputs. This simultaneously pins down: the projection orientation, the order of the
    Q/K/V blocks, the head split, the score scaling, the softmax axis, the context contraction,
    the concat order, and the output projection. Any one of those being wrong breaks it.

    `nn.MultiheadAttention` uses the OPPOSITE mask convention (True = block), so `~keep` is
    passed -- an inversion this test would immediately catch if we had it backwards.
    """
    torch.manual_seed(0)
    b, l, d, h = 2, 6, 16, 4

    ours = MultiHeadAttention(d, h, bias=True).eval()
    theirs = nn.MultiheadAttention(d, h, batch_first=True, bias=True, dropout=0.0).eval()

    with torch.no_grad():
        # in_proj_weight is the row-wise concatenation [W_q ; W_k ; W_v].
        theirs.in_proj_weight.copy_(
            torch.cat([ours.w_q.weight, ours.w_k.weight, ours.w_v.weight], dim=0)
        )
        theirs.in_proj_bias.copy_(
            torch.cat([ours.w_q.bias, ours.w_k.bias, ours.w_v.bias], dim=0)
        )
        theirs.out_proj.weight.copy_(ours.w_o.weight)
        theirs.out_proj.bias.copy_(ours.w_o.bias)

    x = torch.randn(b, l, d)
    keep = causal_keep_mask(l)

    with torch.no_grad():
        out_ours, w_ours = ours(x, x, x, keep_mask=keep)
        out_theirs, w_theirs = theirs(
            x, x, x, attn_mask=~keep[0, 0], need_weights=True, average_attn_weights=False
        )

    assert torch.allclose(out_ours, out_theirs, atol=TOL), (out_ours - out_theirs).abs().max()
    assert torch.allclose(w_ours, w_theirs, atol=TOL), (w_ours - w_theirs).abs().max()


def test_mha_gradients_reach_every_parameter_and_are_finite() -> None:
    torch.manual_seed(0)
    mha = MultiHeadAttention(16, 4, bias=True)
    x = torch.randn(2, 5, 16, requires_grad=True)
    out, _ = mha(x, x, x, keep_mask=causal_keep_mask(5))
    out.pow(2).mean().backward()

    for name, p in mha.named_parameters():
        assert p.grad is not None, f"{name} received no gradient -- it is disconnected"
        assert torch.isfinite(p.grad).all(), f"{name} gradient has NaN/Inf"
        assert p.grad.abs().sum().item() > 0, f"{name} gradient is identically zero"
    assert x.grad is not None and torch.isfinite(x.grad).all()


def test_masked_out_input_positions_receive_zero_gradient() -> None:
    """A padded key contributes nothing forward, so it must receive nothing backward.

    Nonzero gradient at a masked position means the mask leaked -- for instance because a large
    finite constant was used in place of -inf.
    """
    torch.manual_seed(0)
    b, l, d, h = 1, 5, 16, 4
    mha = MultiHeadAttention(d, h).eval()
    ids = torch.tensor([[4, 5, 6, 0, 0]])
    keep = padding_key_mask(ids, pad_id=0)

    x = torch.randn(b, l, d, requires_grad=True)
    out, _ = mha(x, x, x, keep_mask=keep)
    # Only take the loss over real query positions; pad-position outputs are meaningless.
    out[:, :3].pow(2).mean().backward()

    assert x.grad is not None
    pad_grad = x.grad[0, 3:]
    assert torch.equal(pad_grad, torch.zeros_like(pad_grad)), pad_grad.abs().max()


def test_cross_attention_shape_handles_unequal_lengths() -> None:
    """Encoder-decoder attention, section 3.2.3: queries from the decoder, keys/values from the
    encoder, with different sequence lengths and only a source padding mask."""
    b, l_tgt, l_src, d, h = 2, 4, 9, 16, 4
    mha = MultiHeadAttention(d, h)
    src_ids = torch.randint(1, 10, (b, l_src))
    src_ids[:, 7:] = 0
    keep = padding_key_mask(src_ids, pad_id=0)

    out, w = mha(torch.randn(b, l_tgt, d), torch.randn(b, l_src, d), torch.randn(b, l_src, d),
                 keep_mask=keep)
    assert out.shape == (b, l_tgt, d)
    assert w.shape == (b, h, l_tgt, l_src)
    assert bool((w[..., 7:] == 0).all())


def test_dropout_is_inactive_in_eval_mode() -> None:
    """Two eval-mode calls must agree exactly; two train-mode calls with dropout must not.

    Guards against the classic bug of leaving dropout live at inference, which makes a demo
    non-deterministic and makes eval metrics noisy for no reason.
    """
    torch.manual_seed(0)
    mha = MultiHeadAttention(16, 4, dropout=0.5)
    x = torch.randn(2, 5, 16)

    mha.eval()
    with torch.no_grad():
        assert torch.equal(mha(x, x, x)[0], mha(x, x, x)[0])

    mha.train()
    torch.manual_seed(1)
    a = mha(x, x, x)[0]
    b = mha(x, x, x)[0]
    assert not torch.allclose(a, b)


def test_store_weights_is_off_by_default() -> None:
    mha = MultiHeadAttention(16, 4).eval()
    x = torch.randn(1, 4, 16)
    mha(x, x, x)
    assert mha.last_attention_weights is None, "holding weights by default would leak memory"
    mha(x, x, x, store_weights=True)
    assert mha.last_attention_weights is not None
    assert mha.last_attention_weights.shape == (1, 4, 4, 4)
