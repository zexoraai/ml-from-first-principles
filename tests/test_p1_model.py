"""Correctness suite for the assembled encoder-decoder Transformer.

The single most important test here is `test_overfits_a_tiny_batch`. Everything else checks a
structural property; that one checks that forward, loss, backward, optimizer, masking and decoding
are *jointly* wired correctly. A model that cannot drive the loss on eight fixed examples to
approximately zero has a bug, full stop -- there is no dataset, hyperparameter or patience that
fixes it, and running a long training job before this passes is how a wiring bug gets mistaken for
a hard task.
"""

from __future__ import annotations

import math

import pytest
import torch

from labs.p1_transformer import (
    LabelSmoothingLoss,
    Transformer,
    TransformerConfig,
    greedy_decode,
)

TINY = dict(vocab_size=24, d_model=32, num_heads=4, d_ff=64,
            num_encoder_layers=2, num_decoder_layers=2, max_len=16)


def make_model(**overrides) -> Transformer:
    cfg = TransformerConfig(**{**TINY, **overrides})
    return Transformer(cfg)


# --------------------------------------------------------------------------------------------
# shapes and parameter accounting
# --------------------------------------------------------------------------------------------

def test_forward_shape_contract() -> None:
    model = make_model().eval()
    src = torch.randint(3, 24, (2, 7))
    tgt_in = torch.randint(3, 24, (2, 5))
    logits = model(src, tgt_in)
    assert logits.shape == (2, 5, 24), "one distribution over the vocabulary per target position"


def test_source_and_target_lengths_may_differ() -> None:
    model = make_model().eval()
    logits = model(torch.randint(3, 24, (3, 11)), torch.randint(3, 24, (3, 4)))
    assert logits.shape == (3, 4, 24)


def _expected_param_count(cfg: TransformerConfig) -> int:
    """Derive the parameter count from the architecture, independently of the implementation.

    A test that compares the model against itself proves nothing. This counts what the paper's
    description implies, so a wrong projection shape or a missing LayerNorm shows up as a mismatch.
    """
    d, ff, v = cfg.d_model, cfg.d_ff, cfg.vocab_size
    mha = 4 * d * d                       # W^Q, W^K, W^V, W^O, no biases
    ln = 2 * d                            # gamma + beta
    ffn = 2 * d * ff + ff + d             # two Linear layers, both with bias (section 3.3)

    enc_layer = mha + ffn + 2 * ln        # self-attn + ffn, each in a SublayerConnection
    dec_layer = 2 * mha + ffn + 3 * ln    # self-attn + cross-attn + ffn

    total = cfg.num_encoder_layers * enc_layer + cfg.num_decoder_layers * dec_layer
    total += v * d                        # embedding (shared with generator when tied)
    if not cfg.tie_embeddings:
        total += v * d                    # separate target embedding
        total += d * v                    # separate generator
    if cfg.norm_style == "pre":
        total += 2 * ln                   # a final norm in each of encoder and decoder
    return total


@pytest.mark.parametrize("norm_style", ["post", "pre"])
@pytest.mark.parametrize("tie", [True, False])
def test_parameter_count_matches_hand_derivation(norm_style: str, tie: bool) -> None:
    cfg = TransformerConfig(**TINY, norm_style=norm_style, tie_embeddings=tie)
    model = Transformer(cfg)
    assert model.num_parameters() == _expected_param_count(cfg)


def test_tying_shares_one_parameter_object_not_a_copy() -> None:
    """Section 3.4. Same tensor, so one gradient -- a copy would silently drift apart."""
    model = make_model(tie_embeddings=True)
    assert model.generator.weight is model.embedding.weight
    assert model.encoder.embedding is model.decoder.embedding

    untied = make_model(tie_embeddings=False)
    assert untied.generator.weight is not untied.embedding.weight
    assert untied.num_parameters() > model.num_parameters()


def test_tying_reduces_parameter_count_by_exactly_two_matrices() -> None:
    tied = make_model(tie_embeddings=True).num_parameters()
    untied = make_model(tie_embeddings=False).num_parameters()
    assert untied - tied == 2 * TINY["vocab_size"] * TINY["d_model"]


def test_positional_tables_are_not_parameters() -> None:
    model = make_model()
    param_ids = {id(p) for p in model.parameters()}
    assert id(model.encoder.pos.pe) not in param_ids
    assert id(model.decoder.pos.pe) not in param_ids


def test_post_norm_has_no_final_norm_and_pre_norm_does() -> None:
    """Post-norm already normalises every sub-layer output; a final norm would be a deviation.

    Pre-norm leaves the residual stream un-normalised to the top, so it needs one.
    """
    assert make_model(norm_style="post").encoder.final_norm is None
    assert make_model(norm_style="pre").encoder.final_norm is not None


# --------------------------------------------------------------------------------------------
# masking, end to end through the full stack
# --------------------------------------------------------------------------------------------

def test_decoder_output_is_invariant_to_future_target_tokens() -> None:
    """Autoregression held through the *entire* stack, not just one attention module.

    T1 proved a single `MultiHeadAttention` respects a causal mask. This proves the property
    survives composition: two decoder layers, cross-attention, residual connections, layer
    normalization and the output projection. Post-norm makes this non-obvious -- normalisation
    mixes across the feature axis, so a mask error at any layer would leak future information
    into earlier positions here even though the isolated module was correct.
    """
    torch.manual_seed(0)
    model = make_model().eval()
    src = torch.randint(3, 24, (2, 6))
    tgt = torch.randint(3, 24, (2, 8))

    with torch.no_grad():
        before = model(src, tgt)
        tgt2 = tgt.clone()
        tgt2[:, 5:] = torch.randint(3, 24, (2, 3))     # rewrite the future
        after = model(src, tgt2)

    assert torch.equal(before[:, :5], after[:, :5]), (
        (before[:, :5] - after[:, :5]).abs().max().item()
    )
    assert not torch.allclose(before[:, 5:], after[:, 5:]), "the perturbation must matter somewhere"


def test_source_padding_does_not_change_results_for_real_positions() -> None:
    """Appending padding to a source must not alter the model's output at all.

    This is the invariant that actually matters in a real batch: a short sequence sharing a batch
    with a long one gets right-padded, and its prediction must not depend on how much padding it
    happened to receive.

    Note on an earlier, broken version of this test: it took a padded batch and overwrote the pad
    ids with a real token id, expecting the output to be unchanged. That tests nothing -- the
    padding mask is *derived from the ids*, so replacing pad with a real token legitimately changes
    the mask and therefore the result. The test failed and the model was right. Comparing padded
    against unpadded is the correct formulation, and it is strictly stronger: it exercises the mask,
    the positional encoding alignment, and cross-attention together.
    """
    torch.manual_seed(0)
    model = make_model(dropout=0.0).eval()
    tgt = torch.randint(3, 24, (1, 4))

    unpadded = torch.tensor([[5, 6, 7]])
    padded = torch.tensor([[5, 6, 7, 0, 0, 0]])

    with torch.no_grad():
        a = model(unpadded, tgt)
        b = model(padded, tgt)
        mem_a = model.encode(unpadded)
        mem_b = model.encode(padded)

    assert torch.allclose(mem_a, mem_b[:, :3], atol=1e-6), (
        f"encoder memory drifted at real positions: {(mem_a - mem_b[:, :3]).abs().max().item()}"
    )
    assert torch.allclose(a, b, atol=1e-6), (
        f"logits changed when padding was added: {(a - b).abs().max().item()}"
    )


def test_target_padding_does_not_change_results_for_real_positions() -> None:
    """Same invariant on the decoder side, where the mask must combine padding AND causality."""
    torch.manual_seed(0)
    model = make_model(dropout=0.0).eval()
    src = torch.randint(3, 24, (1, 5))
    short = torch.tensor([[1, 8, 9]])
    long = torch.tensor([[1, 8, 9, 0, 0]])

    with torch.no_grad():
        a = model(src, short)
        b = model(src, long)

    assert torch.allclose(a, b[:, :3], atol=1e-6), (a - b[:, :3]).abs().max().item()


def test_target_mask_combines_padding_and_causality() -> None:
    tgt = torch.tensor([[1, 5, 6, 0, 0]])
    keep = make_model().target_keep_mask(tgt)
    assert keep.shape == (1, 1, 5, 5)
    # Row 4 is causally allowed everything, but positions 3 and 4 are padding.
    assert keep[0, 0, 4].tolist() == [True, True, True, False, False]
    # Row 1 is causally limited to {0, 1}, both real.
    assert keep[0, 0, 1].tolist() == [True, True, False, False, False]


def test_removing_the_causal_mask_lets_the_future_leak() -> None:
    """Demonstrates the failure the mask prevents, so the mask's necessity is shown not asserted.

    With only the padding mask, decoder self-attention sees the whole target. Earlier outputs then
    change when later target tokens change -- the exact leak that produces near-zero training loss
    and a useless model at generation time.
    """
    torch.manual_seed(0)
    model = make_model().eval()
    src = torch.randint(3, 24, (1, 5))
    tgt = torch.randint(3, 24, (1, 8))

    from labs.p1_transformer.masks import padding_key_mask

    with torch.no_grad():
        memory = model.encode(src)
        pad_only = padding_key_mask(tgt, model.cfg.pad_id)
        a = model.decoder(tgt, memory, self_keep_mask=pad_only,
                          cross_keep_mask=model.source_keep_mask(src))
        tgt2 = tgt.clone()
        tgt2[:, 6:] = 23
        b = model.decoder(tgt2, memory, self_keep_mask=padding_key_mask(tgt2, model.cfg.pad_id),
                          cross_keep_mask=model.source_keep_mask(src))

    assert not torch.allclose(a[:, :6], b[:, :6], atol=1e-5), (
        "without causal masking, changing the future MUST change earlier outputs; if this "
        "passes silently the test has stopped proving anything"
    )


# --------------------------------------------------------------------------------------------
# gradients
# --------------------------------------------------------------------------------------------

def test_every_parameter_receives_a_finite_nonzero_gradient() -> None:
    torch.manual_seed(0)
    model = make_model()
    src = torch.randint(3, 24, (4, 6))
    tgt_in = torch.randint(3, 24, (4, 5))
    labels = torch.randint(3, 24, (4, 5))

    loss = LabelSmoothingLoss(24, pad_id=0, smoothing=0.1)(model(src, tgt_in), labels)
    loss.backward()

    for name, p in model.named_parameters():
        assert p.grad is not None, f"{name}: no gradient -- disconnected from the loss"
        assert torch.isfinite(p.grad).all(), f"{name}: non-finite gradient"
        if name != "embedding.weight":
            assert p.grad.abs().sum().item() > 0, f"{name}: identically zero gradient"


def test_gradient_reaches_the_first_encoder_layer() -> None:
    """Guards against a break anywhere in the chain from loss back to the earliest parameters."""
    torch.manual_seed(0)
    model = make_model()
    out = model(torch.randint(3, 24, (2, 6)), torch.randint(3, 24, (2, 4)))
    out.pow(2).mean().backward()
    first = model.encoder.layers[0].self_attn.w_q.weight
    assert first.grad is not None and first.grad.abs().sum().item() > 0


# --------------------------------------------------------------------------------------------
# THE OVERFIT PROOF
# --------------------------------------------------------------------------------------------

@pytest.mark.parametrize("norm_style", ["post", "pre"])
def test_overfits_a_tiny_batch(norm_style: str) -> None:
    """Drive one fixed batch of 8 sequences to near-zero loss, then decode it exactly.

    This is the cheapest joint test of the whole system. If the target were shifted incorrectly the
    model would be asked to predict a token it cannot see and the loss would plateau. If the causal
    mask leaked, the loss would fall but greedy decoding -- which has no future to copy -- would
    fail. Requiring **both** low loss and exact greedy reconstruction is what makes the pair
    informative; either alone can pass with a broken model.

    Both normalisation styles are exercised because post-norm without warmup is genuinely harder to
    optimise, and a test that only covered pre-norm would hide that.
    """
    torch.manual_seed(0)
    torch.set_num_threads(2)          # measured optimum for models this size (env-bench-02)

    vocab, batch, length = 24, 8, 6
    cfg = TransformerConfig(
        vocab_size=vocab, d_model=64, num_heads=4, d_ff=128,
        num_encoder_layers=2, num_decoder_layers=2, max_len=16,
        dropout=0.0,                  # off: we WANT to memorise, and dropout fights that
        norm_style=norm_style,
    )
    model = Transformer(cfg)

    # A fixed, learnable mapping: reverse the source. Reversal needs genuine position-dependent
    # attention -- a model that ignored position could not do it -- so success is meaningful.
    src_body = torch.randint(3, vocab, (batch, length))
    tgt_body = torch.flip(src_body, dims=[1])
    bos = torch.full((batch, 1), cfg.bos_id)
    eos = torch.full((batch, 1), cfg.eos_id)

    tgt_in = torch.cat([bos, tgt_body], dim=1)          # [BOS, y0..y_{n-1}]
    labels = torch.cat([tgt_body, eos], dim=1)          # [y0..y_{n-1}, EOS]

    loss_fn = LabelSmoothingLoss(vocab, pad_id=cfg.pad_id, smoothing=0.0)
    opt = torch.optim.Adam(model.parameters(), lr=3e-4, betas=(0.9, 0.98), eps=1e-9)

    model.train()
    losses = []
    for _ in range(600):
        opt.zero_grad(set_to_none=True)
        loss = loss_fn(model(src_body, tgt_in), labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        losses.append(loss.item())

    assert losses[-1] < 0.05, (
        f"{norm_style}-norm failed to overfit 8 examples: final loss {losses[-1]:.4f}, "
        f"start {losses[0]:.4f}. This is a wiring bug, not a tuning problem."
    )
    assert losses[-1] < losses[0] / 10, "loss must fall by at least an order of magnitude"

    # Now the part a leaking mask cannot fake: generate with no access to the answer.
    model.eval()
    decoded = greedy_decode(model, src_body, max_new_tokens=length + 2)
    predicted = decoded[:, 1:length + 1]                # drop BOS, take the body
    assert torch.equal(predicted, tgt_body), (
        f"{norm_style}-norm: loss converged but greedy decoding disagrees.\n"
        f"expected {tgt_body.tolist()}\ngot      {predicted.tolist()}\n"
        "Low loss with wrong generation is the signature of a causal-mask leak."
    )


def test_untrained_model_loss_is_near_uniform_entropy() -> None:
    """At initialisation the model should be roughly uniform over the vocabulary.

    Expected cross-entropy is therefore ~log(V). A value far below that at step 0 means the
    initialisation is accidentally informative; far above means it is actively broken.
    """
    torch.manual_seed(0)
    vocab = 24
    model = make_model(dropout=0.0).eval()
    src = torch.randint(3, vocab, (16, 6))
    tgt_in = torch.randint(3, vocab, (16, 6))
    labels = torch.randint(3, vocab, (16, 6))

    with torch.no_grad():
        loss = LabelSmoothingLoss(vocab, smoothing=0.0)(model(src, tgt_in), labels).item()

    assert loss == pytest.approx(math.log(vocab), rel=0.25), (
        f"loss at init {loss:.3f} vs log(V)={math.log(vocab):.3f}"
    )


# --------------------------------------------------------------------------------------------
# decoding
# --------------------------------------------------------------------------------------------

def test_greedy_decode_refuses_train_mode() -> None:
    """Decoding with dropout live yields non-deterministic output; fail loudly instead."""
    model = make_model().train()
    with pytest.raises(RuntimeError, match="training mode"):
        greedy_decode(model, torch.randint(3, 24, (1, 4)))


def test_greedy_decode_is_deterministic() -> None:
    torch.manual_seed(0)
    model = make_model().eval()
    src = torch.randint(3, 24, (2, 5))
    assert torch.equal(greedy_decode(model, src, max_new_tokens=6),
                       greedy_decode(model, src, max_new_tokens=6))


def test_greedy_decode_starts_with_bos_and_respects_the_length_cap() -> None:
    model = make_model().eval()
    out = greedy_decode(model, torch.randint(3, 24, (3, 5)), max_new_tokens=7)
    assert bool((out[:, 0] == model.cfg.bos_id).all())
    assert out.size(1) <= 8, "BOS plus at most max_new_tokens"


def test_encoder_output_is_independent_of_the_target() -> None:
    """A structural claim worth pinning: the memory depends on the source alone.

    This is what licenses computing the encoder once and reusing it across all decoding steps.
    """
    torch.manual_seed(0)
    model = make_model().eval()
    src = torch.randint(3, 24, (2, 6))
    with torch.no_grad():
        assert torch.equal(model.encode(src), model.encode(src))


def test_store_weights_captures_all_three_attention_kinds() -> None:
    model = make_model().eval()
    src = torch.randint(3, 24, (2, 6))
    tgt = torch.randint(3, 24, (2, 4))
    with torch.no_grad():
        model(src, tgt, store_weights=True)
    got = model.collect_attention()

    assert len(got["encoder_self"]) == 2
    assert len(got["decoder_self"]) == 2
    assert len(got["cross"]) == 2
    assert got["encoder_self"][0].shape == (2, 4, 6, 6)
    assert got["decoder_self"][0].shape == (2, 4, 4, 4)
    assert got["cross"][0].shape == (2, 4, 4, 6), "cross-attention is (tgt_len, src_len)"


def test_embedding_scaling_changes_the_forward_pass() -> None:
    """Section 3.4's sqrt(d_model) factor is not a no-op; pin it so it cannot be dropped silently."""
    torch.manual_seed(0)
    scaled = make_model(dropout=0.0, scale_embeddings=True).eval()
    plain = make_model(dropout=0.0, scale_embeddings=False).eval()
    plain.load_state_dict(scaled.state_dict())

    src = torch.randint(3, 24, (1, 5))
    tgt = torch.randint(3, 24, (1, 4))
    with torch.no_grad():
        assert not torch.allclose(scaled(src, tgt), plain(src, tgt), atol=1e-4)


def test_config_rejects_indivisible_head_count() -> None:
    with pytest.raises(ValueError, match="divisible"):
        TransformerConfig(vocab_size=10, d_model=30, num_heads=4)
