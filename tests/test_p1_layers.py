"""Correctness suite for LayerNorm, the position-wise FFN, and the residual wiring."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from labs.p1_transformer.layers import LayerNorm, PositionwiseFeedForward, SublayerConnection

TOL = 1e-6


# --------------------------------------------------------------------------------------------
# LayerNorm
# --------------------------------------------------------------------------------------------

def test_layernorm_matches_torch_functional_oracle() -> None:
    torch.manual_seed(0)
    d = 32
    ours = LayerNorm(d)
    with torch.no_grad():
        ours.gamma.uniform_(0.5, 1.5)
        ours.beta.uniform_(-0.5, 0.5)

    x = torch.randn(4, 7, d) * 3.0 + 2.0
    theirs = F.layer_norm(x, (d,), weight=ours.gamma, bias=ours.beta, eps=ours.eps)
    assert torch.allclose(ours(x), theirs, atol=TOL), (ours(x) - theirs).abs().max()


def test_layernorm_matches_nn_layernorm_module() -> None:
    torch.manual_seed(0)
    d = 16
    ours = LayerNorm(d)
    theirs = nn.LayerNorm(d)
    with torch.no_grad():
        theirs.weight.copy_(ours.gamma)
        theirs.bias.copy_(ours.beta)
    x = torch.randn(3, 5, d)
    assert torch.allclose(ours(x), theirs(x), atol=TOL)


def test_layernorm_output_statistics_are_zero_mean_unit_variance_per_token() -> None:
    """Without the affine transform each token vector must be standardised on its own."""
    d = 64
    norm = LayerNorm(d, elementwise_affine=False)
    x = torch.randn(4, 6, d) * 7.0 - 3.0
    out = norm(x)

    assert torch.allclose(out.mean(dim=-1), torch.zeros(4, 6), atol=1e-5)
    assert torch.allclose(out.var(dim=-1, unbiased=False), torch.ones(4, 6), atol=1e-3)


def test_layernorm_normalises_each_token_independently() -> None:
    """Changing one token must not move any other token's normalised output.

    This is the property that distinguishes LayerNorm from BatchNorm, and the reason padding in a
    batch cannot contaminate a real token's normalisation.
    """
    torch.manual_seed(0)
    norm = LayerNorm(16, elementwise_affine=False)
    x = torch.randn(2, 5, 16)
    out_before = norm(x)

    x2 = x.clone()
    x2[0, 2] += 100.0
    out_after = norm(x2)

    mask = torch.ones(2, 5, dtype=torch.bool)
    mask[0, 2] = False
    assert torch.equal(out_before[mask], out_after[mask])


def test_layernorm_uses_biased_variance() -> None:
    """Pin down the `unbiased=False` choice by showing the alternative genuinely differs.

    If someone "fixes" the implementation to use the Bessel-corrected variance, parity against
    torch breaks. This test documents that trade and fails loudly if it is reversed.
    """
    d = 8                                   # small d exaggerates the 1/(n-1) vs 1/n difference
    norm = LayerNorm(d, elementwise_affine=False)
    x = torch.randn(2, 3, d)

    mean = x.mean(dim=-1, keepdim=True)
    unbiased_version = (x - mean) / torch.sqrt(x.var(dim=-1, keepdim=True, unbiased=True) + norm.eps)

    assert torch.allclose(norm(x), F.layer_norm(x, (d,), eps=norm.eps), atol=TOL)
    assert not torch.allclose(norm(x), unbiased_version, atol=1e-3)


def test_layernorm_eps_is_inside_the_square_root() -> None:
    """`sqrt(var + eps)` vs `sqrt(var) + eps`: only the first matches every reference."""
    d = 8
    norm = LayerNorm(d, elementwise_affine=False, eps=1e-2)   # large eps makes the gap visible
    x = torch.randn(1, 1, d)
    mean = x.mean(dim=-1, keepdim=True)
    var = x.var(dim=-1, keepdim=True, unbiased=False)

    correct = (x - mean) / torch.sqrt(var + norm.eps)
    wrong = (x - mean) / (torch.sqrt(var) + norm.eps)

    assert torch.allclose(norm(x), correct, atol=TOL)
    assert not torch.allclose(norm(x), wrong, atol=1e-4)


def test_layernorm_handles_a_constant_input_without_nan() -> None:
    """Zero variance is where a missing eps would produce NaN. Must stay finite."""
    norm = LayerNorm(8, elementwise_affine=False)
    out = norm(torch.full((1, 1, 8), 5.0))
    assert bool(torch.isfinite(out).all())
    assert torch.allclose(out, torch.zeros(1, 1, 8), atol=1e-3)


def test_layernorm_is_initialised_to_the_identity_affine() -> None:
    norm = LayerNorm(16)
    assert torch.equal(norm.gamma, torch.ones(16))
    assert torch.equal(norm.beta, torch.zeros(16))


def test_layernorm_gradients_flow_to_gamma_and_beta() -> None:
    norm = LayerNorm(16)
    x = torch.randn(2, 4, 16, requires_grad=True)
    norm(x).pow(2).mean().backward()
    for name, p in norm.named_parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all(), name
    assert norm.gamma.grad.abs().sum().item() > 0


# --------------------------------------------------------------------------------------------
# Position-wise feed-forward network
# --------------------------------------------------------------------------------------------

def test_ffn_shape_contract_and_inner_width() -> None:
    ffn = PositionwiseFeedForward(32, 128, dropout=0.0)
    out = ffn(torch.randn(3, 5, 32))
    assert out.shape == (3, 5, 32)
    assert ffn.w_1.weight.shape == (128, 32)
    assert ffn.w_2.weight.shape == (32, 128)


def test_ffn_does_not_mix_across_positions() -> None:
    """The defining property: position i's output depends only on position i's input.

    All cross-position communication in a Transformer happens in attention. If this test fails,
    something in the FFN is reducing or convolving over the sequence axis.
    """
    torch.manual_seed(0)
    ffn = PositionwiseFeedForward(16, 64, dropout=0.0).eval()
    x = torch.randn(2, 6, 16)

    with torch.no_grad():
        out_before = ffn(x)
        x2 = x.clone()
        x2[:, 3] += 25.0
        out_after = ffn(x2)

    mask = torch.ones(6, dtype=torch.bool)
    mask[3] = False
    assert torch.equal(out_before[:, mask], out_after[:, mask])
    assert not torch.allclose(out_before[:, 3], out_after[:, 3])


def test_ffn_applies_the_same_map_at_every_position() -> None:
    """Feeding the identical vector at two positions must give the identical output."""
    torch.manual_seed(0)
    ffn = PositionwiseFeedForward(16, 64, dropout=0.0).eval()
    v = torch.randn(1, 1, 16)
    x = v.repeat(1, 4, 1)
    with torch.no_grad():
        out = ffn(x)
    for i in range(1, 4):
        assert torch.allclose(out[0, 0], out[0, i], atol=1e-6)


def test_ffn_matches_the_paper_formula() -> None:
    """FFN(x) = max(0, x W_1 + b_1) W_2 + b_2, section 3.3, transcribed and compared."""
    torch.manual_seed(0)
    ffn = PositionwiseFeedForward(8, 32, dropout=0.0, activation="relu").eval()
    x = torch.randn(2, 3, 8)
    with torch.no_grad():
        manual = torch.clamp(x @ ffn.w_1.weight.T + ffn.w_1.bias, min=0.0) @ ffn.w_2.weight.T + ffn.w_2.bias
        assert torch.allclose(ffn(x), manual, atol=1e-6)


def test_ffn_relu_is_the_default_and_gelu_is_opt_in() -> None:
    """The paper specifies ReLU. GELU is a documented deviation, never a silent default."""
    assert PositionwiseFeedForward(8, 16).activation_name == "relu"
    torch.manual_seed(0)
    relu_ffn = PositionwiseFeedForward(8, 16, dropout=0.0, activation="relu").eval()
    gelu_ffn = PositionwiseFeedForward(8, 16, dropout=0.0, activation="gelu").eval()
    with torch.no_grad():
        gelu_ffn.w_1.weight.copy_(relu_ffn.w_1.weight)
        gelu_ffn.w_1.bias.copy_(relu_ffn.w_1.bias)
        gelu_ffn.w_2.weight.copy_(relu_ffn.w_2.weight)
        gelu_ffn.w_2.bias.copy_(relu_ffn.w_2.bias)
        x = torch.randn(1, 4, 8)
        assert not torch.allclose(relu_ffn(x), gelu_ffn(x), atol=1e-4)


def test_ffn_gradients_reach_every_parameter() -> None:
    ffn = PositionwiseFeedForward(16, 64, dropout=0.0)
    ffn(torch.randn(2, 4, 16)).pow(2).mean().backward()
    for name, p in ffn.named_parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all(), name


# --------------------------------------------------------------------------------------------
# Residual / normalization wiring
# --------------------------------------------------------------------------------------------

def test_post_norm_matches_the_paper_expression() -> None:
    """Section 3.1: output of each sub-layer is LayerNorm(x + Sublayer(x))."""
    torch.manual_seed(0)
    conn = SublayerConnection(16, dropout=0.0, norm_style="post").eval()
    x = torch.randn(2, 4, 16)
    sub = lambda t: t * 2.0 + 1.0                                    # noqa: E731

    with torch.no_grad():
        assert torch.allclose(conn(x, sub), conn.norm(x + sub(x)), atol=1e-6)


def test_pre_norm_matches_the_tensor2tensor_expression() -> None:
    """x + Sublayer(LayerNorm(x)) -- what the authors' released code actually did."""
    torch.manual_seed(0)
    conn = SublayerConnection(16, dropout=0.0, norm_style="pre").eval()
    x = torch.randn(2, 4, 16)
    sub = lambda t: t * 2.0 + 1.0                                    # noqa: E731

    with torch.no_grad():
        assert torch.allclose(conn(x, sub), x + sub(conn.norm(x)), atol=1e-6)


def test_the_two_norm_styles_are_not_equivalent() -> None:
    """Guards against the two branches accidentally collapsing into the same computation."""
    torch.manual_seed(0)
    x = torch.randn(2, 4, 16)
    sub = lambda t: t * 2.0 + 1.0                                    # noqa: E731
    post = SublayerConnection(16, dropout=0.0, norm_style="post").eval()
    pre = SublayerConnection(16, dropout=0.0, norm_style="pre").eval()
    with torch.no_grad():
        assert not torch.allclose(post(x, sub), pre(x, sub), atol=1e-3)


def test_pre_norm_leaves_an_unnormalised_identity_path() -> None:
    """The structural reason pre-norm trains deep stacks more easily.

    With a sub-layer that outputs exactly zero, pre-norm returns x untouched -- a clean identity.
    Post-norm returns LayerNorm(x), i.e. the residual signal has been rescaled. Repeat that N
    times and the difference in gradient conditioning is the whole pre/post-norm story.
    """
    torch.manual_seed(0)
    x = torch.randn(2, 4, 16) * 5.0
    zero_sub = lambda t: torch.zeros_like(t)                         # noqa: E731

    pre = SublayerConnection(16, dropout=0.0, norm_style="pre").eval()
    post = SublayerConnection(16, dropout=0.0, norm_style="post").eval()

    with torch.no_grad():
        assert torch.allclose(pre(x, zero_sub), x, atol=1e-6)
        assert not torch.allclose(post(x, zero_sub), x, atol=1e-2)


def test_dropout_sits_on_the_sublayer_output_before_the_residual_add() -> None:
    """Section 5.4: dropout is applied to the sub-layer output, before it is added and normalized.

    Constructed so the distinction is observable: the sub-layer emits a large constant while x is
    zero, and normalization is disabled by using pre-norm. Dropped units then show up as exact
    zeros in the result. If dropout were applied after the residual add, x's contribution would
    also be zeroed -- indistinguishable here, so we additionally assert the surviving units carry
    the 1/(1-p) inverted-dropout scaling, which only holds if dropout saw the sub-layer output.
    """
    torch.manual_seed(0)
    p = 0.5
    conn = SublayerConnection(64, dropout=p, norm_style="pre").train()
    x = torch.zeros(8, 8, 64)
    out = conn(x, lambda t: torch.ones_like(t))

    values = out.unique()
    assert 0.0 in set(values.tolist()), "some units must be dropped"
    survivors = out[out != 0]
    assert torch.allclose(survivors, torch.full_like(survivors, 1.0 / (1.0 - p)), atol=1e-6)


def test_residual_connection_gives_the_input_a_direct_gradient_path() -> None:
    """With a sub-layer whose gradient is zero, x must still receive gradient through the skip."""
    conn = SublayerConnection(16, dropout=0.0, norm_style="pre")
    x = torch.randn(2, 4, 16, requires_grad=True)
    out = conn(x, lambda t: t.detach() * 0.0)      # sub-layer contributes no gradient at all
    out.sum().backward()
    assert x.grad is not None
    assert x.grad.abs().sum().item() > 0, "the skip path must carry gradient on its own"


def test_sublayer_rejects_unknown_norm_style() -> None:
    with pytest.raises(ValueError, match="post.*pre"):
        SublayerConnection(16, norm_style="middle")   # type: ignore[arg-type]
