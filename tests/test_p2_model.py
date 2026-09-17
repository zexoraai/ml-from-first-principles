"""Correctness suite for the GPT model and its sampling controls."""

from __future__ import annotations

import math

import pytest
import torch

from labs.p2_gpt import GPT, GPTConfig, apply_sampling_filters, generate, generate_with_trace

TINY = dict(vocab_size=64, block_size=32, n_layer=3, n_head=4, d_model=32, dropout=0.0,
            attention_dropout=0.0)


def make(**over) -> GPT:
    return GPT(GPTConfig(**{**TINY, **over}))


# --------------------------------------------------------------------------------------------
# shapes, parameters, limits
# --------------------------------------------------------------------------------------------

def test_forward_shapes_and_loss() -> None:
    m = make().eval()
    idx = torch.randint(0, 64, (2, 16))
    logits, loss = m(idx)
    assert logits.shape == (2, 16, 64)
    assert loss is None
    _, loss = m(idx, idx)
    assert loss.shape == ()
    assert torch.isfinite(loss)


def test_parameter_count_matches_hand_derivation() -> None:
    """Derived from the architecture, independent of the implementation."""
    cfg = GPTConfig(**TINY)
    d, ff, v, n, blk = cfg.d_model, cfg.d_ff, cfg.vocab_size, cfg.n_layer, cfg.block_size

    per_block = (
        4 * d * d + 4 * d          # attention: 4 projections + 4 biases (bias=True)
        + 2 * d                    # ln_1
        + d * ff + ff              # mlp.fc
        + ff * d + d               # mlp.proj
        + 2 * d                    # ln_2
    )
    expected = v * d + blk * d + n * per_block + 2 * d   # wte + wpe + blocks + ln_f
    # lm_head is tied to wte, so it adds nothing.
    assert make().num_parameters() == expected


def test_tying_shares_one_parameter_object() -> None:
    m = make(tie_embeddings=True)
    assert m.lm_head.weight is m.wte.weight
    untied = make(tie_embeddings=False)
    assert untied.lm_head.weight is not untied.wte.weight
    assert untied.num_parameters() - m.num_parameters() == TINY["vocab_size"] * TINY["d_model"]


def test_non_embedding_count_excludes_positions() -> None:
    m = make()
    assert m.num_parameters() - m.num_parameters(non_embedding=True) == m.wpe.weight.numel()


def test_sequence_longer_than_block_size_is_rejected_with_a_clear_reason() -> None:
    """Learned position embeddings have no row past block_size -- a hard architectural limit."""
    m = make().eval()
    with pytest.raises(ValueError, match="block_size"):
        m(torch.randint(0, 64, (1, TINY["block_size"] + 1)))


def test_causal_mask_is_a_non_persistent_buffer() -> None:
    m = make()
    assert "causal" in dict(m.named_buffers())
    assert "causal" not in m.state_dict()
    assert not any(id(p) == id(m.causal) for p in m.parameters())


# --------------------------------------------------------------------------------------------
# causality, end to end
# --------------------------------------------------------------------------------------------

def test_logits_at_position_i_ignore_all_later_tokens() -> None:
    """The defining property of a causal LM, held through the whole stack.

    Every position's prediction must depend only on itself and its past. If this leaks, training loss
    collapses (the model reads the answer) while generation is garbage -- and the loss curve looks
    healthy the whole time.
    """
    torch.manual_seed(0)
    m = make().eval()
    idx = torch.randint(0, 64, (2, 20))
    with torch.no_grad():
        before, _ = m(idx)
        perturbed = idx.clone()
        perturbed[:, 12:] = torch.randint(0, 64, (2, 8))
        after, _ = m(perturbed)

    assert torch.equal(before[:, :12], after[:, :12]), (
        (before[:, :12] - after[:, :12]).abs().max().item()
    )
    assert not torch.allclose(before[:, 12:], after[:, 12:]), "perturbation must matter somewhere"


def test_prefix_extension_does_not_change_earlier_logits() -> None:
    """Appending tokens must leave earlier predictions untouched -- what makes caching sound."""
    torch.manual_seed(0)
    m = make().eval()
    short = torch.randint(0, 64, (1, 10))
    longer = torch.cat([short, torch.randint(0, 64, (1, 5))], dim=1)
    with torch.no_grad():
        a, _ = m(short)
        b, _ = m(longer)
    assert torch.allclose(a, b[:, :10], atol=1e-6)


# --------------------------------------------------------------------------------------------
# initialisation
# --------------------------------------------------------------------------------------------

def test_initial_loss_is_near_uniform_entropy() -> None:
    """At init the model should be near-uniform, so loss ~ log(V).

    Far below means the init is accidentally informative and would flatter every later number; far
    above means it is actively broken.
    """
    torch.manual_seed(0)
    m = make().eval()
    idx = torch.randint(0, 64, (8, 24))
    with torch.no_grad():
        _, loss = m(idx, idx)
    assert loss.item() == pytest.approx(math.log(64), rel=0.15)


def test_residual_projections_are_scaled_by_depth() -> None:
    """GPT-2's 1/sqrt(2*n_layer) scaling, checked as a measured standard deviation."""
    n_layer = 8
    m = make(n_layer=n_layer, d_model=64)
    expected = 0.02 / math.sqrt(2 * n_layer)
    for block in m.blocks:
        assert block.attn.w_o.weight.std().item() == pytest.approx(expected, rel=0.25)
        assert block.mlp.proj.weight.std().item() == pytest.approx(expected, rel=0.25)
    # The *inner* projections keep the unscaled 0.02: they do not write into the residual stream.
    assert m.blocks[0].mlp.fc.weight.std().item() == pytest.approx(0.02, rel=0.25)


def test_residual_scaling_keeps_activation_scale_flat_with_depth() -> None:
    """Measures what the scaling is for, rather than restating the formula.

    Without it, the residual stream accumulates 2N contributions and its magnitude grows with depth.
    With it, a deep model's pre-final-norm activation scale stays close to a shallow one's.
    """
    torch.manual_seed(0)
    idx = torch.randint(0, 64, (4, 24))

    def stream_scale(model: GPT) -> float:
        with torch.no_grad():
            x = model.drop(model.wte(idx) + model.wpe(torch.arange(idx.size(1))))
            keep = model.causal[:, :, :idx.size(1), :idx.size(1)]
            for block in model.blocks:
                x = block(x, keep)
        return float(x.std())

    shallow = stream_scale(make(n_layer=2, d_model=64).eval())
    deep = stream_scale(make(n_layer=12, d_model=64).eval())
    assert deep < shallow * 3.0, f"scale grew from {shallow:.3f} to {deep:.3f} with depth"


# --------------------------------------------------------------------------------------------
# gradients and overfitting
# --------------------------------------------------------------------------------------------

def test_every_parameter_gets_a_finite_gradient() -> None:
    torch.manual_seed(0)
    m = make()
    idx = torch.randint(0, 64, (4, 16))
    _, loss = m(idx, idx)
    loss.backward()
    for name, p in m.named_parameters():
        assert p.grad is not None, f"{name}: no gradient"
        assert torch.isfinite(p.grad).all(), f"{name}: non-finite gradient"


def test_overfits_a_single_batch() -> None:
    """The joint proof that forward, loss, backward, optimizer and masking are wired correctly.

    A causal LM can always memorise one fixed batch. Failure here is a wiring bug, not a tuning
    problem, and no amount of training on real data will fix it.
    """
    torch.manual_seed(0)
    torch.set_num_threads(2)
    m = make(n_layer=2, d_model=64, dropout=0.0)
    idx = torch.randint(0, 64, (4, 24))
    opt = torch.optim.AdamW(m.parameters(), lr=3e-3, betas=(0.9, 0.95))

    m.train()
    first = last = None
    for step in range(320):
        opt.zero_grad(set_to_none=True)
        _, loss = m(idx, idx)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        opt.step()
        if step == 0:
            first = loss.item()
        last = loss.item()

    assert last < 0.15, f"failed to overfit one batch: {first:.3f} -> {last:.3f}"


# --------------------------------------------------------------------------------------------
# sampling controls
# --------------------------------------------------------------------------------------------

def test_temperature_zero_is_exactly_argmax() -> None:
    """The T -> 0 limit, handled exactly rather than with a tiny epsilon that could divide by zero."""
    logits = torch.tensor([[1.0, 5.0, 2.0, 4.0]])
    probs = apply_sampling_filters(logits, temperature=0.0)
    assert probs.argmax().item() == 1
    assert probs.max().item() == 1.0
    assert probs.sum().item() == pytest.approx(1.0)


def test_temperature_one_is_the_unmodified_distribution() -> None:
    logits = torch.randn(1, 32)
    assert torch.allclose(apply_sampling_filters(logits, temperature=1.0),
                          logits.softmax(dim=-1), atol=1e-6)


def test_higher_temperature_flattens_and_lower_sharpens() -> None:
    """Entropy is the right summary: it must increase monotonically with temperature."""
    torch.manual_seed(0)
    logits = torch.randn(1, 64) * 3.0

    def entropy(t: float) -> float:
        p = apply_sampling_filters(logits, temperature=t)
        return float(-(p * p.clamp_min(1e-12).log()).sum())

    assert entropy(0.5) < entropy(1.0) < entropy(2.0)


def test_temperature_never_changes_the_ranking() -> None:
    """Temperature rescales; it does not reorder. A common misconception worth pinning."""
    torch.manual_seed(0)
    logits = torch.randn(1, 40)
    order_a = apply_sampling_filters(logits, temperature=0.3).argsort(descending=True)
    order_b = apply_sampling_filters(logits, temperature=3.0).argsort(descending=True)
    assert torch.equal(order_a, order_b)


def test_top_k_keeps_exactly_k_candidates() -> None:
    torch.manual_seed(0)
    probs = apply_sampling_filters(torch.randn(1, 100), top_k=7)
    assert int((probs > 0).sum()) == 7
    assert probs.sum().item() == pytest.approx(1.0, abs=1e-6)


def test_top_k_larger_than_vocab_is_clamped() -> None:
    probs = apply_sampling_filters(torch.randn(1, 10), top_k=999)
    assert int((probs > 0).sum()) == 10


def test_top_p_nucleus_is_never_empty_even_when_one_token_dominates() -> None:
    """A single token above p must still be kept, or sampling would have nothing to choose from."""
    logits = torch.tensor([[20.0, 0.0, 0.0, 0.0]])
    probs = apply_sampling_filters(logits, top_p=0.5)
    assert int((probs > 0).sum()) >= 1
    assert probs.argmax().item() == 0


def test_top_p_adapts_to_confidence_where_top_k_does_not() -> None:
    """The reason nucleus sampling exists, demonstrated on two distributions.

    On a peaked distribution the nucleus is small; on a flat one it is large. Top-k is the same width
    in both cases, which is exactly its weakness.
    """
    peaked = torch.tensor([[10.0] + [0.0] * 49])
    flat = torch.zeros(1, 50)

    n_peaked = int((apply_sampling_filters(peaked, top_p=0.9) > 0).sum())
    n_flat = int((apply_sampling_filters(flat, top_p=0.9) > 0).sum())
    assert n_peaked < n_flat

    assert int((apply_sampling_filters(peaked, top_k=10) > 0).sum()) == 10
    assert int((apply_sampling_filters(flat, top_k=10) > 0).sum()) == 10


def test_filters_produce_a_valid_distribution() -> None:
    torch.manual_seed(0)
    for kwargs in ({}, {"temperature": 0.7}, {"top_k": 5}, {"top_p": 0.9},
                   {"temperature": 1.3, "top_k": 20, "top_p": 0.95}):
        probs = apply_sampling_filters(torch.randn(3, 50), **kwargs)
        assert torch.isfinite(probs).all()
        assert bool((probs >= 0).all())
        assert torch.allclose(probs.sum(dim=-1), torch.ones(3), atol=1e-6)


def test_negative_temperature_is_rejected() -> None:
    with pytest.raises(ValueError, match="temperature"):
        apply_sampling_filters(torch.randn(1, 10), temperature=-1.0)


# --------------------------------------------------------------------------------------------
# generation
# --------------------------------------------------------------------------------------------

def test_generate_refuses_train_mode() -> None:
    m = make().train()
    with pytest.raises(RuntimeError, match="eval mode"):
        generate(m, torch.zeros(1, 4, dtype=torch.long), 5)


def test_generate_appends_the_requested_number_of_tokens() -> None:
    m = make().eval()
    idx = torch.zeros(2, 5, dtype=torch.long)
    assert generate(m, idx, 7, top_k=3).shape == (2, 12)


def test_generate_is_reproducible_with_a_seeded_generator() -> None:
    """The demo exposes a seed, so this must actually hold."""
    m = make().eval()
    idx = torch.zeros(1, 4, dtype=torch.long)
    a = generate(m, idx, 12, temperature=1.0, generator=torch.Generator().manual_seed(7))
    b = generate(m, idx, 12, temperature=1.0, generator=torch.Generator().manual_seed(7))
    assert torch.equal(a, b)
    c = generate(m, idx, 12, temperature=1.0, generator=torch.Generator().manual_seed(8))
    assert not torch.equal(a, c)


def test_greedy_generation_is_deterministic_without_a_seed() -> None:
    """At temperature 0 there is nothing to sample, so no generator is needed."""
    m = make().eval()
    idx = torch.zeros(1, 4, dtype=torch.long)
    assert torch.equal(generate(m, idx, 8, temperature=0.0),
                       generate(m, idx, 8, temperature=0.0))


def test_generation_beyond_block_size_crops_instead_of_crashing() -> None:
    """A live audience will hold Enter. Running past the context window must degrade, not raise."""
    m = make().eval()
    idx = torch.zeros(1, TINY["block_size"], dtype=torch.long)
    out = generate(m, idx, 20, top_k=5)
    assert out.shape == (1, TINY["block_size"] + 20)


def test_trace_records_raw_and_filtered_distributions() -> None:
    """Showing only the filtered distribution would hide what the sampling controls are doing."""
    m = make().eval()
    idx = torch.zeros(1, 4, dtype=torch.long)
    out, trace = generate_with_trace(
        m, idx, 6, temperature=0.8, top_k=5, generator=torch.Generator().manual_seed(0)
    )
    assert out.shape == (1, 10)
    assert len(trace) == 6
    for step in trace:
        assert step["n_candidates"] <= 5, "top_k=5 must leave at most 5 candidates"
        assert len(step["raw_top"]) == 8
        assert step["entropy_nats"] > 0
        # the raw distribution is untruncated, so it has mass on more tokens than the filtered one
        assert sum(p for _, p in step["raw_top"]) <= 1.0 + 1e-6


def test_trace_rejects_batches_larger_than_one() -> None:
    m = make().eval()
    with pytest.raises(ValueError, match="batch size 1"):
        generate_with_trace(m, torch.zeros(3, 4, dtype=torch.long), 2)


# --------------------------------------------------------------------------------------------
# compute accounting
# --------------------------------------------------------------------------------------------

def test_flop_estimate_exposes_the_attention_share() -> None:
    """The 6N rule of thumb omits attention; at long context that omission stops being safe."""
    short = GPT(GPTConfig(vocab_size=64, block_size=32, n_layer=4, n_head=4, d_model=64))
    long = GPT(GPTConfig(vocab_size=64, block_size=1024, n_layer=4, n_head=4, d_model=64))
    assert long.estimate_flops_per_token()["attention_share"] > \
           short.estimate_flops_per_token()["attention_share"]


def test_gpt2_small_config_has_the_documented_geometry() -> None:
    """We never train this; it exists so the 124M target carries real numbers."""
    cfg = GPTConfig.gpt2_small()
    assert (cfg.n_layer, cfg.n_head, cfg.d_model, cfg.block_size) == (12, 12, 768, 1024)
    assert cfg.d_ff == 4 * 768
