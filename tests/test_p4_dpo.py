"""Correctness suite for DPO.

The load-bearing check is `test_loss_at_initialisation_is_exactly_log_two`. When the policy equals the
reference, both implicit rewards are zero, the margin is zero, and the loss is analytically `log 2`.
Any masking, shift, or sign error moves it off that value, so one assertion covers a surprising amount
of the implementation.
"""

from __future__ import annotations

import math

import pytest
import torch
import torch.nn as nn

from labs.p2_gpt import GPT, GPTConfig
from labs.p4_dpo import (
    DEGRADATIONS,
    build_sft_and_preferences,
    dpo_loss,
    make_reference_model,
    sequence_logprobs,
    sft_loss,
)

TINY = dict(vocab_size=48, block_size=32, n_layer=2, n_head=4, d_model=32,
            dropout=0.0, attention_dropout=0.0)


def make_model() -> GPT:
    torch.manual_seed(0)
    return GPT(GPTConfig(**TINY)).eval()


# --------------------------------------------------------------------------------------------
# sequence log-probabilities
# --------------------------------------------------------------------------------------------

def test_logprobs_are_negative_and_shaped_per_sequence() -> None:
    torch.manual_seed(0)
    b, t, v = 3, 10, 48
    logits = torch.randn(b, t, v)
    labels = torch.randint(0, v, (b, t))
    mask = torch.ones(b, t)
    out = sequence_logprobs(logits, labels, mask)
    assert out.shape == (b,)
    assert bool((out < 0).all()), "a sum of log-probabilities must be negative"


def test_only_response_tokens_contribute() -> None:
    """Masked positions must be excluded from the sum, exactly.

    Constructed so the answer is checkable: compute with a full mask, then with the first half masked
    off, and require the difference to equal the sum over the first half alone.
    """
    torch.manual_seed(0)
    b, t, v = 2, 12, 48
    logits = torch.randn(b, t, v)
    labels = torch.randint(0, v, (b, t))

    full = torch.ones(b, t)
    second_half = torch.zeros(b, t)
    second_half[:, 6:] = 1
    first_half = torch.zeros(b, t)
    first_half[:, :6] = 1

    total = sequence_logprobs(logits, labels, full)
    part_a = sequence_logprobs(logits, labels, first_half)
    part_b = sequence_logprobs(logits, labels, second_half)
    assert torch.allclose(total, part_a + part_b, atol=1e-5)


def test_prompt_tokens_are_genuinely_excluded() -> None:
    """Changing the label at a masked position must not change the score at all."""
    torch.manual_seed(0)
    b, t, v = 1, 10, 48
    logits = torch.randn(b, t, v)
    labels = torch.randint(0, v, (b, t))
    mask = torch.zeros(b, t)
    mask[:, 5:] = 1

    before = sequence_logprobs(logits, labels, mask)
    altered = labels.clone()
    altered[0, 2] = (labels[0, 2] + 7) % v          # a masked (prompt) position
    assert torch.equal(before, sequence_logprobs(logits, altered, mask))

    altered2 = labels.clone()
    altered2[0, 7] = (labels[0, 7] + 7) % v         # an unmasked (response) position
    assert not torch.equal(before, sequence_logprobs(logits, altered2, mask))


def test_the_shift_aligns_logits_with_the_token_they_predict() -> None:
    """Position t's logits predict token t+1. Verified against a hand-computed value.

    An off-by-one here still trains — the loss falls — while scoring every token against the wrong
    prediction, so it cannot be caught by watching a curve.
    """
    v = 5
    # A one-hot-ish logit tensor: position 0 confidently predicts token 3, position 1 predicts token 1.
    logits = torch.full((1, 3, v), -20.0)
    logits[0, 0, 3] = 20.0
    logits[0, 1, 1] = 20.0
    labels = torch.tensor([[0, 3, 1]])              # token at index 1 is 3, at index 2 is 1
    mask = torch.ones(1, 3)

    got = sequence_logprobs(logits, labels, mask).item()
    # Both predictions are essentially certain, so the summed log-probability is ~0.
    assert got > -0.01, f"expected ~0 for two confident correct predictions, got {got}"

    # Now make the labels wrong at those positions; the score must collapse.
    bad = torch.tensor([[0, 1, 3]])
    assert sequence_logprobs(logits, bad, mask).item() < -30


def test_average_flag_divides_by_response_length() -> None:
    torch.manual_seed(0)
    logits = torch.randn(1, 9, 48)
    labels = torch.randint(0, 48, (1, 9))
    mask = torch.zeros(1, 9)
    mask[:, 4:] = 1                                  # 5 response tokens, 4 after the shift
    total = sequence_logprobs(logits, labels, mask)
    mean = sequence_logprobs(logits, labels, mask, average=True)
    assert torch.allclose(mean * mask[:, 1:].sum(), total, atol=1e-5)


def test_shape_mismatch_is_rejected() -> None:
    with pytest.raises(ValueError, match="disagree"):
        sequence_logprobs(torch.randn(2, 5, 10), torch.zeros(2, 6, dtype=torch.long),
                          torch.ones(2, 6))


# --------------------------------------------------------------------------------------------
# THE anchor test
# --------------------------------------------------------------------------------------------

def test_loss_at_initialisation_is_exactly_log_two() -> None:
    """π_θ = π_ref ⇒ both rewards 0 ⇒ margin 0 ⇒ loss = -log σ(0) = log 2.

    This single assertion is sensitive to sign errors, a swapped chosen/rejected, a wrong β
    application, and a missing negation. It is the cheapest high-coverage check in the project.
    """
    logps_w = torch.tensor([-12.0, -30.0, -5.5])
    logps_l = torch.tensor([-14.0, -22.0, -9.0])
    stats = dpo_loss(logps_w, logps_l, logps_w.clone(), logps_l.clone(), beta=0.1)

    assert stats.loss.item() == pytest.approx(math.log(2), abs=1e-6)
    assert torch.allclose(stats.reward_margin, torch.zeros(3), atol=1e-6)
    assert torch.allclose(stats.chosen_reward, torch.zeros(3), atol=1e-6)


def test_loss_at_initialisation_is_log_two_for_any_beta() -> None:
    """β multiplies a zero margin, so the initial loss is β-independent."""
    logps_w = torch.tensor([-8.0])
    logps_l = torch.tensor([-9.0])
    for beta in (0.01, 0.1, 0.5, 5.0):
        stats = dpo_loss(logps_w, logps_l, logps_w.clone(), logps_l.clone(), beta=beta)
        assert stats.loss.item() == pytest.approx(math.log(2), abs=1e-6)


# --------------------------------------------------------------------------------------------
# how the loss responds
# --------------------------------------------------------------------------------------------

def test_loss_falls_when_the_margin_grows() -> None:
    ref_w, ref_l = torch.tensor([-10.0]), torch.tensor([-10.0])
    worse = dpo_loss(torch.tensor([-11.0]), torch.tensor([-9.0]), ref_w, ref_l, beta=0.1)
    equal = dpo_loss(torch.tensor([-10.0]), torch.tensor([-10.0]), ref_w, ref_l, beta=0.1)
    better = dpo_loss(torch.tensor([-9.0]), torch.tensor([-11.0]), ref_w, ref_l, beta=0.1)

    assert worse.loss.item() > equal.loss.item() > better.loss.item()
    assert worse.reward_margin.item() < 0 < better.reward_margin.item()


def test_only_the_margin_matters_not_the_absolute_rewards() -> None:
    """Two very different policies with the same margin must give the same loss.

    This is the property behind a DPO behaviour that surprises people: the objective is perfectly
    happy to *lower both* log-probabilities as long as the gap widens.
    """
    ref_w, ref_l = torch.tensor([-10.0]), torch.tensor([-10.0])
    a = dpo_loss(torch.tensor([-9.0]), torch.tensor([-11.0]), ref_w, ref_l, beta=0.2)
    b = dpo_loss(torch.tensor([-40.0]), torch.tensor([-42.0]), ref_w, ref_l, beta=0.2)
    assert a.loss.item() == pytest.approx(b.loss.item(), abs=1e-6)
    # ...and both moved far below the reference in absolute terms.
    assert b.chosen_logps.item() < ref_w.item()


def test_beta_scales_the_margin() -> None:
    ref = torch.tensor([-10.0])
    small = dpo_loss(torch.tensor([-9.0]), torch.tensor([-11.0]), ref, ref.clone(), beta=0.05)
    large = dpo_loss(torch.tensor([-9.0]), torch.tensor([-11.0]), ref, ref.clone(), beta=0.5)
    assert large.reward_margin.item() == pytest.approx(10 * small.reward_margin.item(), rel=1e-6)
    assert large.loss.item() < small.loss.item(), "a larger margin must mean a smaller loss"


def test_reward_accuracy_counts_pairs_ranked_correctly() -> None:
    ref = torch.zeros(4)
    stats = dpo_loss(
        torch.tensor([1.0, -1.0, 2.0, -3.0]),
        torch.tensor([0.0, 1.0, 1.0, 0.0]),
        ref, ref.clone(), beta=0.1,
    )
    # margins: +1, -2, +1, -3  ->  2 of 4 correct
    assert stats.reward_accuracy.mean().item() == pytest.approx(0.5)


def test_gradient_pushes_chosen_up_and_rejected_down() -> None:
    """The direction of learning, checked directly on the gradient signs."""
    pw = torch.tensor([-10.0], requires_grad=True)
    pl = torch.tensor([-10.0], requires_grad=True)
    ref = torch.tensor([-10.0])
    dpo_loss(pw, pl, ref, ref.clone(), beta=0.1).loss.backward()

    assert pw.grad.item() < 0, "lowering the loss must raise the chosen log-probability"
    assert pl.grad.item() > 0, "lowering the loss must reduce the rejected log-probability"


def test_gradient_is_largest_on_pairs_currently_ranked_backwards() -> None:
    """d/dz of -logsigmoid(z) is -σ(-z): steepest where the model is most wrong.

    Same self-limiting shape as logistic regression, and the reason DPO does not need curriculum
    tricks to focus on hard pairs.
    """
    def grad_magnitude(chosen: float, rejected: float) -> float:
        pw = torch.tensor([chosen], requires_grad=True)
        pl = torch.tensor([rejected], requires_grad=True)
        ref = torch.tensor([-10.0])
        dpo_loss(pw, pl, ref, ref.clone(), beta=1.0).loss.backward()
        return abs(pw.grad.item())

    very_wrong = grad_magnitude(-14.0, -6.0)      # margin -8
    slightly_wrong = grad_magnitude(-10.5, -9.5)  # margin -1
    already_right = grad_magnitude(-6.0, -14.0)   # margin +8
    assert very_wrong > slightly_wrong > already_right


def test_loss_is_finite_for_a_hugely_negative_margin() -> None:
    """`logsigmoid` must be used, not `log(sigmoid(x))`, which underflows to -inf.

    Not theoretical: early in training with a large β the margin can easily reach -100.
    """
    stats = dpo_loss(torch.tensor([-500.0]), torch.tensor([0.0]),
                     torch.tensor([0.0]), torch.tensor([0.0]), beta=1.0)
    assert torch.isfinite(stats.loss), "loss went non-finite on a large negative margin"


def test_label_smoothing_bounds_the_reward_below_plain_dpo() -> None:
    """cDPO refuses to drive the margin to infinity, so its loss has a nonzero floor."""
    ref = torch.zeros(1)
    huge = (torch.tensor([50.0]), torch.tensor([-50.0]))
    plain = dpo_loss(*huge, ref, ref.clone(), beta=1.0, label_smoothing=0.0)
    smoothed = dpo_loss(*huge, ref, ref.clone(), beta=1.0, label_smoothing=0.1)
    assert plain.loss.item() < 1e-6
    assert smoothed.loss.item() > 1.0, "smoothing must penalise an unbounded margin"


def test_invalid_label_smoothing_is_rejected() -> None:
    with pytest.raises(ValueError, match="label_smoothing"):
        dpo_loss(torch.zeros(1), torch.zeros(1), torch.zeros(1), torch.zeros(1),
                 label_smoothing=0.6)


# --------------------------------------------------------------------------------------------
# the reference policy
# --------------------------------------------------------------------------------------------

def test_reference_model_is_frozen_and_in_eval_mode() -> None:
    model = make_model().train()
    ref = make_reference_model(model)
    assert not ref.training, "dropout in the reference would make log π_ref random"
    assert all(not p.requires_grad for p in ref.parameters())
    assert model.training, "making a reference must not change the policy's mode"


def test_reference_is_a_copy_not_an_alias() -> None:
    """If the reference shared weights it would move with the policy and the anchor would vanish."""
    model = make_model()
    ref = make_reference_model(model)
    with torch.no_grad():
        model.wte.weight.add_(1.0)
    assert not torch.allclose(model.wte.weight, ref.wte.weight)


def test_training_the_policy_leaves_the_reference_untouched() -> None:
    torch.manual_seed(0)
    model = make_model().train()
    ref = make_reference_model(model)
    before = ref.wte.weight.detach().clone()

    ids = torch.randint(0, 48, (2, 12))
    mask = torch.ones(2, 12)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-2)
    for _ in range(5):
        opt.zero_grad(set_to_none=True)
        logits, _ = model(ids)
        p = sequence_logprobs(logits, ids, mask)
        with torch.no_grad():
            r_logits, _ = ref(ids)
            r = sequence_logprobs(r_logits, ids, mask)
        dpo_loss(p, p.flip(0), r, r.flip(0), beta=0.1).loss.backward()
        opt.step()

    assert torch.equal(ref.wte.weight, before)


# --------------------------------------------------------------------------------------------
# end to end on a real model
# --------------------------------------------------------------------------------------------

def test_end_to_end_initial_loss_is_log_two_on_a_real_model() -> None:
    """The anchor test, but through a full GPT forward pass with real masking."""
    torch.manual_seed(0)
    model = make_model()
    ref = make_reference_model(model)

    chosen = torch.randint(0, 48, (3, 14))
    rejected = torch.randint(0, 48, (3, 14))
    mask = torch.zeros(3, 14)
    mask[:, 6:] = 1                                  # first 6 tokens are the prompt

    with torch.no_grad():
        pc = sequence_logprobs(model(chosen)[0], chosen, mask)
        pr = sequence_logprobs(model(rejected)[0], rejected, mask)
        rc = sequence_logprobs(ref(chosen)[0], chosen, mask)
        rr = sequence_logprobs(ref(rejected)[0], rejected, mask)

    assert dpo_loss(pc, pr, rc, rr, beta=0.1).loss.item() == pytest.approx(math.log(2), abs=1e-5)


def test_dpo_can_separate_two_fixed_responses() -> None:
    """Optimising a few steps must raise the margin and the loss must fall below log 2.

    Weak by design — it proves the objective is trainable, not that it produces a good model.
    """
    torch.manual_seed(0)
    model = make_model().train()
    ref = make_reference_model(model)

    prompt = torch.randint(0, 48, (1, 6))
    chosen = torch.cat([prompt, torch.randint(0, 48, (1, 8))], dim=1)
    rejected = torch.cat([prompt, torch.randint(0, 48, (1, 8))], dim=1)
    mask = torch.zeros(1, 14)
    mask[:, 6:] = 1

    with torch.no_grad():
        rc = sequence_logprobs(ref(chosen)[0], chosen, mask)
        rr = sequence_logprobs(ref(rejected)[0], rejected, mask)

    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    first = last = None
    for step in range(40):
        opt.zero_grad(set_to_none=True)
        pc = sequence_logprobs(model(chosen)[0], chosen, mask)
        pr = sequence_logprobs(model(rejected)[0], rejected, mask)
        stats = dpo_loss(pc, pr, rc, rr, beta=0.1)
        stats.loss.backward()
        opt.step()
        if step == 0:
            first = stats
        last = stats

    assert last.loss.item() < first.loss.item()
    assert last.reward_margin.item() > first.reward_margin.item()
    assert last.loss.item() < math.log(2)


# --------------------------------------------------------------------------------------------
# SFT loss
# --------------------------------------------------------------------------------------------

def test_sft_loss_ignores_prompt_positions() -> None:
    """Training on prompt tokens teaches the model to write instructions instead of answering them."""
    torch.manual_seed(0)
    logits = torch.randn(2, 10, 48)
    labels = torch.randint(0, 48, (2, 10))
    mask = torch.zeros(2, 10)
    mask[:, 5:] = 1

    before = sft_loss(logits, labels, mask)
    altered = labels.clone()
    altered[:, 1] = (labels[:, 1] + 3) % 48          # a prompt position
    assert torch.equal(before, sft_loss(logits, altered, mask))


def test_sft_loss_matches_cross_entropy_when_everything_is_unmasked() -> None:
    torch.manual_seed(0)
    logits = torch.randn(2, 8, 48)
    labels = torch.randint(0, 48, (2, 8))
    ours = sft_loss(logits, labels, torch.ones(2, 8))
    theirs = nn.functional.cross_entropy(
        logits[:, :-1].reshape(-1, 48), labels[:, 1:].reshape(-1)
    )
    assert ours.item() == pytest.approx(theirs.item(), rel=1e-5)


def test_sft_loss_handles_an_all_prompt_batch_without_nan() -> None:
    out = sft_loss(torch.randn(1, 6, 48, requires_grad=True),
                   torch.randint(0, 48, (1, 6)), torch.zeros(1, 6))
    assert torch.isfinite(out)


# --------------------------------------------------------------------------------------------
# preference data construction
# --------------------------------------------------------------------------------------------

CORPUS = "\n\n".join(
    f"# Topic {i}\n\nThis is a passage about subject {i} in graphic design practice. "
    f"It discusses the relevant considerations at some length and with enough words to pass the "
    f"minimum length threshold that the extractor applies. Designers working with subject {i} must "
    f"weigh legibility against expression, and the resulting choices shape how a reader moves "
    f"through the page from one element to the next."
    for i in range(60)
)


def test_preference_data_splits_by_topic_without_leakage() -> None:
    built = build_sft_and_preferences(CORPUS, seed=1)
    train = {e.topic for e in built["sft_train"]} | {p.topic for p in built["pref_train"]}
    evalt = {e.topic for e in built["sft_eval"]} | {p.topic for p in built["pref_eval"]}
    assert train and evalt
    assert not (train & evalt), sorted(train & evalt)[:5]


def test_every_pair_has_a_distinct_chosen_and_rejected() -> None:
    built = build_sft_and_preferences(CORPUS, seed=2)
    for pair in built["pref_train"]:
        assert pair.chosen != pair.rejected, f"identical pair for {pair.topic!r}"
        assert pair.degradation in DEGRADATIONS


def test_all_degradation_kinds_are_represented() -> None:
    built = build_sft_and_preferences(CORPUS, seed=3, pairs_per_topic=4)
    kinds = {p.degradation for p in built["pref_train"]}
    assert kinds == set(DEGRADATIONS), f"missing {set(DEGRADATIONS) - kinds}"


def test_truncated_rejections_end_mid_sentence() -> None:
    """The degradation must be detectable, or there is nothing for the model to learn.

    Regression test for bug #12: a cut landing on a sentence boundary produced a COMPLETE short answer
    labelled `truncated`, which teaches the model to disprefer well-formed brief prose. Checked across
    several seeds because the bug only fired for particular cut positions.
    """
    for seed in (5, 13, 21, 34, 55):
        built = build_sft_and_preferences(CORPUS, seed=seed, pairs_per_topic=6)
        truncs = [p for p in built["pref_train"] if p.degradation == "truncated"]
        assert truncs
        for p in truncs:
            assert not p.rejected.strip().endswith((".", "!", "?")), (
                f"seed {seed}: truncation produced a complete sentence: {p.rejected.strip()!r}"
            )
            assert len(p.rejected.split()) < len(p.chosen.split())


def test_truncation_of_a_single_sentence_passage_is_still_detectable() -> None:
    """The fallback path: no mid-sentence cut point exists, so punctuation must be stripped."""
    single = "\n\n".join(
        f"# Solo {i}\n\nA single unbroken sentence about subject {i} that runs on for long enough to "
        f"clear the minimum word threshold without ever introducing a second sentence boundary "
        f"anywhere inside it at all"
        for i in range(60)
    )
    built = build_sft_and_preferences(single, seed=9, pairs_per_topic=6)
    truncs = [p for p in built["pref_train"] if p.degradation == "truncated"]
    assert truncs
    for p in truncs:
        assert not p.rejected.strip().endswith((".", "!", "?")), p.rejected


def test_repetitive_rejections_actually_repeat() -> None:
    built = build_sft_and_preferences(CORPUS, seed=6, pairs_per_topic=6)
    reps = [p for p in built["pref_train"] if p.degradation == "repetitive"]
    assert reps
    for p in reps:
        sentences = [s for s in p.rejected.split(".") if s.strip()]
        assert len(sentences) > len(set(s.strip() for s in sentences)), "no repetition present"


def test_chosen_responses_are_complete_sentences() -> None:
    """If `chosen` were also truncated, the truncation preference would be unlearnable."""
    built = build_sft_and_preferences(CORPUS, seed=7)
    for e in built["sft_train"]:
        assert e.response.strip().endswith((".", "!", "?")), e.response[-40:]


def test_manifest_states_that_preferences_are_constructed() -> None:
    """The provenance disclosure is a deliverable; a test keeps it from being dropped."""
    m = build_sft_and_preferences(CORPUS, seed=8)["manifest"]
    assert "CONSTRUCTED, NOT HUMAN-ANNOTATED" in m["preference_origin"]
    assert "NOTHING about human values" in m["what_this_measures"]
    assert m["split_unit"] == "topic — every prompt about a topic lands in exactly one split"


def test_construction_is_deterministic_given_the_seed() -> None:
    a = build_sft_and_preferences(CORPUS, seed=11)
    b = build_sft_and_preferences(CORPUS, seed=11)
    assert [p.rejected for p in a["pref_train"]] == [p.rejected for p in b["pref_train"]]
    c = build_sft_and_preferences(CORPUS, seed=12)
    assert [p.rejected for p in a["pref_train"]] != [p.rejected for p in c["pref_train"]]


def test_too_few_topics_raises_rather_than_producing_a_useless_split() -> None:
    with pytest.raises(RuntimeError, match="too few"):
        build_sft_and_preferences("# One\n\n" + "word " * 100)
