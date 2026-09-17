"""Correctness suite for online softmax and tiled attention.

The project's central claim is **exactness** — that tiling changes the memory profile and not the
result. So the bulk of these tests are equivalence tests against a naive reference, swept over tile
sizes and shapes, including the awkward cases (sequence lengths that do not divide the tile size, a
single query against a long cache, fully-masked rows).

Tolerances are absolute and published rather than `==`, because float32 addition is not associative and
tiling deliberately reassociates the sums. Asserting bit-equality would be asserting something false.
"""

from __future__ import annotations

import math

import pytest
import torch

from labs.p5_attention import (
    OnlineSoftmaxState,
    TileStats,
    attention_memory_bytes,
    online_softmax,
    reference_attention,
    softmax_state_update,
    tiled_attention,
)

# float32 attention over up to a few hundred keys reassociates to roughly this level. Stated here once,
# referenced by every equivalence test, and reported on the project page.
ATOL = 1e-5


# =============================================================================================
# online softmax
# =============================================================================================

def test_online_softmax_matches_torch_softmax() -> None:
    torch.manual_seed(0)
    x = torch.randn(7, 96)
    assert torch.allclose(online_softmax(x, block_size=16), torch.softmax(x, dim=-1), atol=1e-6)


@pytest.mark.parametrize("block_size", [1, 2, 3, 7, 16, 31, 96, 200])
def test_online_softmax_is_independent_of_block_size(block_size: int) -> None:
    """Every blocking must give the same answer, including blocks larger than the row."""
    torch.manual_seed(0)
    x = torch.randn(5, 96)
    assert torch.allclose(online_softmax(x, block_size=block_size), torch.softmax(x, dim=-1),
                          atol=1e-6)


def test_online_softmax_survives_values_that_would_overflow_naive_exp() -> None:
    """exp(x) overflows float32 near x ≈ 88. This is why the running maximum is subtracted.

    The naive `exp(x) / exp(x).sum()` is shown failing on the same input, so the test documents the
    problem rather than only the fix.
    """
    x = torch.tensor([[500.0, 501.0, 499.0]])
    out = online_softmax(x, block_size=2)
    assert torch.isfinite(out).all()
    assert out.sum().item() == pytest.approx(1.0, abs=1e-6)
    assert torch.allclose(out, torch.softmax(x, dim=-1), atol=1e-6)

    naive = torch.exp(x)
    assert not torch.isfinite(naive).all(), "the input no longer overflows; pick larger values"


def test_online_softmax_handles_large_negative_values() -> None:
    x = torch.tensor([[-500.0, -501.0, -499.0]])
    out = online_softmax(x, block_size=2)
    assert torch.isfinite(out).all()
    assert out.sum().item() == pytest.approx(1.0, abs=1e-6)


def test_online_softmax_of_a_constant_row_is_uniform() -> None:
    out = online_softmax(torch.full((1, 8), 3.0), block_size=3)
    assert torch.allclose(out, torch.full((1, 8), 0.125), atol=1e-7)


def test_state_update_matches_a_two_pass_computation() -> None:
    """The identity itself, isolated from any attention machinery."""
    torch.manual_seed(0)
    x = torch.randn(4, 50)
    m = torch.full((4,), float("-inf"))
    l = torch.zeros(4)
    for start in range(0, 50, 8):
        m, l = softmax_state_update(m, l, x[:, start : start + 8])

    assert torch.allclose(m, x.amax(dim=-1), atol=0)
    assert torch.allclose(l, torch.exp(x - x.amax(dim=-1, keepdim=True)).sum(-1), atol=1e-5)


def test_state_accumulator_computes_softmax_times_v_in_one_pass() -> None:
    """The accumulator is what lets attention avoid a second pass entirely."""
    torch.manual_seed(0)
    scores = torch.randn(3, 40)
    values = torch.randn(40, 6)

    state = OnlineSoftmaxState(rows=3, dim=6)
    for start in range(0, 40, 9):
        state.update(scores[:, start : start + 9], values[start : start + 9])

    assert torch.allclose(state.normalise(), torch.softmax(scores, dim=-1) @ values, atol=1e-5)


def test_fully_masked_row_yields_zeros_not_nan() -> None:
    """A query with no valid keys has no attention distribution. Zero, never nan.

    Reachable in real models: a padded position under a causal mask can have zero valid keys. A nan
    here would propagate through every downstream tensor and every gradient.
    """
    state = OnlineSoftmaxState(rows=2, dim=4)
    state.update(torch.full((2, 5), float("-inf")), torch.randn(5, 4))
    out = state.normalise()
    assert torch.isfinite(out).all(), out
    assert torch.equal(out, torch.zeros(2, 4))


# =============================================================================================
# tiled attention — equivalence
# =============================================================================================

def qkv(b=2, h=3, t=64, d=16, *, t_k=None, seed=0):
    torch.manual_seed(seed)
    return (torch.randn(b, h, t, d),
            torch.randn(b, h, t_k or t, d),
            torch.randn(b, h, t_k or t, d))


@pytest.mark.parametrize("q_block,k_block", [(1, 1), (1, 64), (8, 8), (16, 32), (64, 64), (128, 128)])
def test_tiled_matches_reference_non_causal(q_block: int, k_block: int) -> None:
    q, k, v = qkv()
    got = tiled_attention(q, k, v, q_block=q_block, k_block=k_block)
    assert torch.allclose(got, reference_attention(q, k, v), atol=ATOL)


@pytest.mark.parametrize("q_block,k_block", [(1, 1), (8, 8), (16, 32), (64, 64), (128, 128)])
def test_tiled_matches_reference_causal(q_block: int, k_block: int) -> None:
    q, k, v = qkv()
    got = tiled_attention(q, k, v, causal=True, q_block=q_block, k_block=k_block)
    assert torch.allclose(got, reference_attention(q, k, v, causal=True), atol=ATOL)


@pytest.mark.parametrize("t", [1, 2, 3, 7, 17, 31, 33, 64, 65, 100])
def test_tiled_matches_reference_for_lengths_that_do_not_divide_the_tile(t: int) -> None:
    """Ragged final tiles are where off-by-one errors live."""
    q, k, v = qkv(b=1, h=2, t=t, d=8)
    for causal in (False, True):
        got = tiled_attention(q, k, v, causal=causal, q_block=16, k_block=16)
        want = reference_attention(q, k, v, causal=causal)
        assert torch.allclose(got, want, atol=ATOL), f"t={t} causal={causal}"


def test_tiled_matches_reference_for_a_single_query_against_a_long_cache() -> None:
    """The incremental-decoding shape: one new query, many cached keys.

    This is the case that catches top-left versus bottom-right causal alignment. With `t_q = 1` and
    `t_k = 128`, the single query is at absolute position 127 and must attend to **all** keys. A
    top-left-aligned mask would let it attend to only the first one.
    """
    q, k, v = qkv(b=1, h=2, t=1, d=8, t_k=128)
    got = tiled_attention(q, k, v, causal=True, q_block=8, k_block=16)
    want = reference_attention(q, k, v, causal=True)
    assert torch.allclose(got, want, atol=ATOL)
    # And it really did use every key: compare against unmasked attention, which must be identical.
    assert torch.allclose(got, reference_attention(q, k, v, causal=False), atol=ATOL)


def test_causal_and_non_causal_differ() -> None:
    """Guards against a masking argument that is silently ignored."""
    q, k, v = qkv(t=32)
    a = tiled_attention(q, k, v, causal=True, q_block=8, k_block=8)
    b = tiled_attention(q, k, v, causal=False, q_block=8, k_block=8)
    assert not torch.allclose(a, b, atol=1e-3)


def test_first_causal_row_attends_only_to_the_first_key() -> None:
    """An analytic check: with a bottom-right mask and t_q == t_k, row 0 sees exactly key 0, so the
    output must equal `v[0]` exactly, whatever the scores are."""
    q, k, v = qkv(b=1, h=1, t=16, d=4)
    out = tiled_attention(q, k, v, causal=True, q_block=4, k_block=4)
    assert torch.allclose(out[0, 0, 0], v[0, 0, 0], atol=1e-6)


def test_tiled_is_exact_under_scores_that_would_overflow() -> None:
    """Large-magnitude q and k push raw scores past the float32 exp limit."""
    torch.manual_seed(0)
    q = torch.randn(1, 1, 32, 8) * 30
    k = torch.randn(1, 1, 32, 8) * 30
    v = torch.randn(1, 1, 32, 8)
    got = tiled_attention(q, k, v, causal=True, q_block=8, k_block=8)
    assert torch.isfinite(got).all()
    assert torch.allclose(got, reference_attention(q, k, v, causal=True), atol=1e-4)


def test_custom_scale_is_honoured_by_both_paths() -> None:
    q, k, v = qkv(t=32)
    got = tiled_attention(q, k, v, scale=0.05, q_block=8, k_block=8)
    assert torch.allclose(got, reference_attention(q, k, v, scale=0.05), atol=ATOL)
    assert not torch.allclose(got, reference_attention(q, k, v), atol=1e-3)


def test_attention_output_rows_are_convex_combinations_of_value_rows() -> None:
    """Softmax weights are non-negative and sum to one, so every output row must lie inside the
    bounding box of the value rows. A cheap, assumption-free invariant."""
    q, k, v = qkv(b=1, h=1, t=48, d=8)
    out = tiled_attention(q, k, v, q_block=16, k_block=16)
    assert bool((out <= v.amax(dim=-2, keepdim=True) + 1e-5).all())
    assert bool((out >= v.amin(dim=-2, keepdim=True) - 1e-5).all())


def test_gradients_flow_and_match_the_reference() -> None:
    """The tiled forward must be differentiable and agree on gradients, not just on outputs.

    Autograd differentiates the tiled loop as written, which is not FlashAttention's hand-written
    recomputing backward — that lives in the Triton kernel. What this checks is that the tiled forward
    is a correct, differentiable function, which is the prerequisite for using it at all.
    """
    q, k, v = qkv(b=1, h=2, t=24, d=8)
    q1, k1, v1 = (t.clone().requires_grad_(True) for t in (q, k, v))
    q2, k2, v2 = (t.clone().requires_grad_(True) for t in (q, k, v))

    tiled_attention(q1, k1, v1, causal=True, q_block=8, k_block=8).square().sum().backward()
    reference_attention(q2, k2, v2, causal=True).square().sum().backward()

    for name, a, b in (("q", q1, q2), ("k", k1, k2), ("v", v1, v2)):
        assert a.grad is not None, f"no gradient reached {name}"
        assert torch.allclose(a.grad, b.grad, atol=1e-4), f"{name} gradient mismatch"


# =============================================================================================
# tile bookkeeping and the memory claim
# =============================================================================================

def test_causal_skips_fully_masked_key_tiles() -> None:
    """The skip is what makes causal attention cost ~half of full attention rather than the same."""
    q, k, v = qkv(b=1, h=1, t=64, d=8)
    stats = TileStats()
    tiled_attention(q, k, v, causal=True, q_block=16, k_block=16, stats=stats)

    total = stats.n_query_tiles * stats.n_key_tiles
    assert stats.n_tiles_skipped > 0, "no tiles skipped; the causal skip is not firing"
    assert stats.n_tiles_computed + stats.n_tiles_skipped == total
    # 4x4 tiles, lower triangle inclusive = 10 computed, 6 skipped.
    assert (stats.n_tiles_computed, stats.n_tiles_skipped) == (10, 6)


def test_non_causal_skips_nothing() -> None:
    q, k, v = qkv(b=1, h=1, t=64, d=8)
    stats = TileStats()
    tiled_attention(q, k, v, causal=False, q_block=16, k_block=16, stats=stats)
    assert stats.n_tiles_skipped == 0
    assert stats.n_tiles_computed == stats.n_query_tiles * stats.n_key_tiles


def test_skipping_does_not_change_the_result() -> None:
    """Correctness must not depend on the optimisation firing."""
    q, k, v = qkv(b=1, h=2, t=48, d=8)
    small = tiled_attention(q, k, v, causal=True, q_block=8, k_block=8)
    one_tile = tiled_attention(q, k, v, causal=True, q_block=48, k_block=48)   # nothing skippable
    assert torch.allclose(small, one_tile, atol=ATOL)


def test_memory_accounting_reduction_is_quadratic_in_t_over_block() -> None:
    """The claim is a ratio, so it is computed and checked rather than asserted."""
    m = attention_memory_bytes(batch=1, heads=12, t_q=4096, t_k=4096, head_dim=64,
                               q_block=64, k_block=64)
    assert m["reduction_factor"] == pytest.approx((4096 / 64) ** 2)
    assert m["tiled_total_bytes"] < m["reference_total_bytes"]


def test_memory_accounting_matches_a_hand_computed_figure() -> None:
    """12 heads, T=4096, float32: 12 · 4096² · 4 bytes ≈ 805 MB. Checked by hand."""
    m = attention_memory_bytes(batch=1, heads=12, t_q=4096, t_k=4096, head_dim=64,
                               q_block=64, k_block=64)
    assert m["reference_score_matrix_bytes"] == 12 * 4096 * 4096 * 4
    assert m["reference_score_matrix_bytes"] / 2**20 == pytest.approx(768.0, abs=0.1)


def test_memory_accounting_reports_qkv_separately_so_the_ratio_is_not_inflated() -> None:
    """Omitting Q/K/V from both sides would overstate the saving. They are listed explicitly."""
    m = attention_memory_bytes(batch=2, heads=4, t_q=512, t_k=512, head_dim=32,
                               q_block=64, k_block=64)
    assert m["qkv_bytes"] > 0
    assert m["reference_total_bytes"] == (m["reference_score_matrix_bytes"] + m["qkv_bytes"]
                                         + m["output_bytes"])
    assert m["tiled_total_bytes"] == (m["tiled_peak_tile_bytes"] + m["qkv_bytes"]
                                      + m["output_bytes"])


def test_peak_tile_elements_is_independent_of_sequence_length() -> None:
    """The headline memory property: tile cost does not grow with T."""
    peaks = []
    for t in (64, 256, 1024):
        stats = TileStats()
        q, k, v = qkv(b=1, h=1, t=t, d=8)
        tiled_attention(q, k, v, causal=True, q_block=32, k_block=32, stats=stats)
        peaks.append(stats.peak_tile_elements)
    assert len(set(peaks)) == 1, f"tile memory grew with T: {peaks}"


# =============================================================================================
# input validation
# =============================================================================================

def test_mismatched_head_dims_are_rejected() -> None:
    with pytest.raises(ValueError, match="head dims differ"):
        tiled_attention(torch.randn(1, 1, 8, 16), torch.randn(1, 1, 8, 8), torch.randn(1, 1, 8, 8))


def test_mismatched_key_value_lengths_are_rejected() -> None:
    with pytest.raises(ValueError, match="positions but v has"):
        tiled_attention(torch.randn(1, 1, 8, 8), torch.randn(1, 1, 8, 8), torch.randn(1, 1, 7, 8))


def test_different_value_head_dim_is_allowed() -> None:
    """d_v need not equal d_k. Attention only requires q·k to be defined."""
    torch.manual_seed(0)
    q = torch.randn(1, 2, 16, 8)
    k = torch.randn(1, 2, 16, 8)
    v = torch.randn(1, 2, 16, 5)
    got = tiled_attention(q, k, v, q_block=4, k_block=4)
    assert got.shape == (1, 2, 16, 5)
    assert torch.allclose(got, reference_attention(q, k, v), atol=ATOL)


def test_reference_and_tiled_agree_with_p1_scaled_dot_product_attention() -> None:
    """Cross-check against Project 1's independently written attention.

    Two implementations written months apart agreeing is stronger evidence than either agreeing with
    itself. P1's version takes a boolean keep-mask, so the causal mask is built explicitly.
    """
    from labs.p1_transformer.attention import scaled_dot_product_attention
    from labs.p1_transformer.masks import causal_keep_mask

    q, k, v = qkv(b=1, h=2, t=32, d=8)
    keep = causal_keep_mask(32, device=q.device)
    p1_out, _ = scaled_dot_product_attention(q, k, v, keep_mask=keep)

    assert torch.allclose(tiled_attention(q, k, v, causal=True, q_block=8, k_block=8),
                          p1_out, atol=ATOL)
    assert torch.allclose(reference_attention(q, k, v, causal=True), p1_out, atol=ATOL)


def test_scale_default_is_one_over_sqrt_head_dim() -> None:
    """The paper's 1/√d_k. Verified by comparing against an explicit value."""
    q, k, v = qkv(b=1, h=1, t=16, d=64)
    assert torch.allclose(tiled_attention(q, k, v, q_block=8, k_block=8),
                          tiled_attention(q, k, v, scale=1.0 / math.sqrt(64),
                                          q_block=8, k_block=8), atol=0)
