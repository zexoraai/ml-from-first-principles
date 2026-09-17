"""Correctness suite for the KV cache.

A cache is an optimisation, so the only thing that makes it safe to ship is a proof that it changes
nothing. These tests are that proof. Without them the cache is a plausible-looking speedup that might
be quietly generating different text — and the failure mode is subtle: output stays fluent for a few
tokens and then degrades, which is easy to misattribute to the model rather than the cache.
"""

from __future__ import annotations

import pytest
import torch

from labs.p1_transformer.attention import MultiHeadAttention
from labs.p1_transformer.masks import causal_keep_mask
from labs.p2_gpt import GPT, GPTConfig, generate

TINY = dict(vocab_size=48, block_size=24, n_layer=3, n_head=4, d_model=32,
            dropout=0.0, attention_dropout=0.0)


def make(**over) -> GPT:
    return GPT(GPTConfig(**{**TINY, **over})).eval()


# --------------------------------------------------------------------------------------------
# the attention module in isolation
# --------------------------------------------------------------------------------------------

def test_cached_attention_matches_full_causal_attention() -> None:
    """Feeding a sequence one token at a time must equal one masked pass over the whole thing."""
    torch.manual_seed(0)
    d, h, t = 32, 4, 9
    mha = MultiHeadAttention(d, h).eval()
    x = torch.randn(1, t, d)

    with torch.no_grad():
        full, _ = mha(x, x, x, keep_mask=causal_keep_mask(t))

        incremental = []
        past_k = past_v = None
        for i in range(t):
            out, _, past_k, past_v = mha.forward_cached(
                x[:, i:i + 1], past_k=past_k, past_v=past_v
            )
            incremental.append(out)
        stepwise = torch.cat(incremental, dim=1)

    assert torch.allclose(full, stepwise, atol=1e-5), (full - stepwise).abs().max().item()


def test_cached_attention_handles_a_multi_token_prefill() -> None:
    """Pre-filling a prompt in one call must match feeding it token by token.

    This is the path that needs the *rectangular* causal mask: query i of the new block sits at
    absolute position n_past + i. A square lower-triangular mask here would forbid the new tokens
    from seeing most of the cached prefix.
    """
    torch.manual_seed(0)
    d, h, t = 32, 4, 10
    mha = MultiHeadAttention(d, h).eval()
    x = torch.randn(1, t, d)

    with torch.no_grad():
        # prefill 6, then 4 more in one call
        out_a, _, k, v = mha.forward_cached(x[:, :6])
        out_b, _, k, v = mha.forward_cached(x[:, 6:], past_k=k, past_v=v)
        chunked = torch.cat([out_a, out_b], dim=1)

        full, _ = mha(x, x, x, keep_mask=causal_keep_mask(t))

    assert torch.allclose(full, chunked, atol=1e-5), (full - chunked).abs().max().item()


def test_cache_grows_by_exactly_the_number_of_new_tokens() -> None:
    torch.manual_seed(0)
    mha = MultiHeadAttention(32, 4).eval()
    with torch.no_grad():
        _, _, k, v = mha.forward_cached(torch.randn(2, 5, 32))
        assert k.shape == (2, 4, 5, 8) and v.shape == (2, 4, 5, 8)
        _, _, k, v = mha.forward_cached(torch.randn(2, 1, 32), past_k=k, past_v=v)
        assert k.shape == (2, 4, 6, 8)


def test_single_new_token_needs_no_mask_and_sees_all_history() -> None:
    """With n_new == 1 every cached key is strictly in the past, so masking would be wrong.

    Probed behaviourally: the new token's output must depend on the cached prefix. If a square causal
    mask were applied it would see only itself, and changing the prefix would not move its output.
    """
    torch.manual_seed(0)
    mha = MultiHeadAttention(32, 4).eval()
    new = torch.randn(1, 1, 32)

    with torch.no_grad():
        _, _, k1, v1 = mha.forward_cached(torch.randn(1, 6, 32))
        out1, _, _, _ = mha.forward_cached(new, past_k=k1, past_v=v1)
        _, _, k2, v2 = mha.forward_cached(torch.randn(1, 6, 32))
        out2, _, _, _ = mha.forward_cached(new, past_k=k2, past_v=v2)

    assert not torch.allclose(out1, out2, atol=1e-4), (
        "the new token's output must depend on the cached history"
    )


def test_mismatched_past_arguments_are_rejected() -> None:
    mha = MultiHeadAttention(32, 4).eval()
    with pytest.raises(ValueError, match="both be given or both be None"):
        mha.forward_cached(torch.randn(1, 1, 32), past_k=torch.randn(1, 4, 3, 8), past_v=None)


# --------------------------------------------------------------------------------------------
# the whole model
# --------------------------------------------------------------------------------------------

def test_cached_forward_matches_plain_forward() -> None:
    """Logits from the cached path must match the plain forward pass over the same tokens."""
    torch.manual_seed(0)
    m = make()
    idx = torch.randint(0, 48, (2, 12))

    with torch.no_grad():
        plain, _ = m(idx)
        cached, _ = m.forward_cached(idx, None)

    assert torch.allclose(plain, cached, atol=1e-5), (plain - cached).abs().max().item()


def test_cached_forward_token_by_token_matches_plain_forward() -> None:
    torch.manual_seed(0)
    m = make()
    idx = torch.randint(0, 48, (1, 14))

    with torch.no_grad():
        plain, _ = m(idx)
        past = None
        outs = []
        for i in range(idx.size(1)):
            logits, past = m.forward_cached(idx[:, i:i + 1], past)
            outs.append(logits)
        stepwise = torch.cat(outs, dim=1)

    assert torch.allclose(plain, stepwise, atol=1e-5), (plain - stepwise).abs().max().item()


def test_cached_generation_matches_uncached_exactly() -> None:
    """THE test that makes the cache shippable.

    Same seed, same sampling settings, cache on and off. The token sequences must be identical — not
    similar. Any divergence means the two paths compute different distributions, and the fast path
    would be silently generating different text from the model that was evaluated.
    """
    torch.manual_seed(0)
    m = make()
    prompt = torch.randint(0, 48, (1, 5))

    for temperature, top_k in ((1.0, None), (0.8, 10), (0.0, None), (1.2, 5)):
        cached = generate(m, prompt, 14, temperature=temperature, top_k=top_k,
                          generator=torch.Generator().manual_seed(42), use_cache=True)
        naive = generate(m, prompt, 14, temperature=temperature, top_k=top_k,
                         generator=torch.Generator().manual_seed(42), use_cache=False)
        assert torch.equal(cached, naive), (
            f"cache changed the output at temperature={temperature}, top_k={top_k}\n"
            f"cached {cached.tolist()}\nnaive  {naive.tolist()}"
        )


def test_positions_come_from_the_cache_length_not_from_zero() -> None:
    """Guards the highest-consequence cache bug.

    If the incremental path used positions 0..n_new-1 instead of n_past..n_past+n_new-1, every
    generated token would receive position embedding 0. Output stays fluent briefly then degenerates
    — hard to attribute. Detected here by requiring the cached logits for a *later* position to differ
    from what the model produces for that same token at position 0.
    """
    torch.manual_seed(0)
    m = make()
    token = torch.tensor([[7]])

    with torch.no_grad():
        at_position_zero, _ = m.forward_cached(token, None)
        _, past = m.forward_cached(torch.randint(0, 48, (1, 8)), None)
        at_position_eight, _ = m.forward_cached(token, past)

    assert not torch.allclose(at_position_zero, at_position_eight, atol=1e-4), (
        "the same token at different positions must produce different logits"
    )


def test_cache_rejects_overflowing_the_context_window() -> None:
    m = make()
    with pytest.raises(ValueError, match="exceeds block_size"):
        m.forward_cached(torch.zeros(1, TINY["block_size"] + 1, dtype=torch.long), None)


def test_generation_past_the_context_window_rebuilds_the_cache() -> None:
    """A live audience will generate past the window. It must degrade gracefully, not raise.

    Beyond `block_size` the model genuinely cannot represent older positions, so the cache is dropped
    and re-filled from the most recent window. The test asserts it completes and stays consistent with
    the naive path, which does the same cropping.
    """
    torch.manual_seed(0)
    m = make()
    prompt = torch.randint(0, 48, (1, TINY["block_size"] - 2))

    cached = generate(m, prompt, 12, temperature=0.0, use_cache=True)
    naive = generate(m, prompt, 12, temperature=0.0, use_cache=False)

    assert cached.shape == (1, prompt.size(1) + 12)
    assert torch.equal(cached, naive), "cropping behaviour must agree between the two paths"


def test_cache_is_faster_than_recomputing() -> None:
    """Measured, not assumed — the cache exists for a reason and the reason should be checkable.

    Deliberately loose (only requires *any* speedup): this runs on a contended shared machine where
    timing varies by up to 14x (GAPS G-008), so a tight bound would be flaky for reasons unrelated to
    correctness. The real measurement lives in the project's evidence record.
    """
    import time

    torch.manual_seed(0)
    torch.set_num_threads(2)
    m = make(block_size=64, n_layer=4, d_model=64)
    prompt = torch.randint(0, 48, (1, 8))

    def run(use_cache: bool) -> float:
        generate(m, prompt, 4, temperature=0.0, use_cache=use_cache)   # warm up
        start = time.perf_counter()
        generate(m, prompt, 40, temperature=0.0, use_cache=use_cache)
        return time.perf_counter() - start

    assert run(True) < run(False), "the KV cache should not be slower than recomputing everything"
