"""Correctness suite for the Noam schedule (section 5.3) and label smoothing (section 5.4)."""

from __future__ import annotations

import math

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from labs.p1_transformer import LabelSmoothingLoss, NoamSchedule, build_optimizer


# --------------------------------------------------------------------------------------------
# NoamSchedule
# --------------------------------------------------------------------------------------------

def test_matches_the_paper_formula_transcribed_literally() -> None:
    d_model, warmup = 512, 4000
    sched = NoamSchedule(d_model, warmup_steps=warmup)
    for step in (1, 10, 500, 3999, 4000, 4001, 20000, 100000):
        expected = d_model ** -0.5 * min(step ** -0.5, step * warmup ** -1.5)
        assert sched(step) == pytest.approx(expected, rel=1e-12)


def test_peak_is_exactly_at_warmup_step() -> None:
    """The two branches are equal at step == warmup, both reducing to warmup^(-0.5).

    So the maximum sits precisely on the warmup step with no discontinuity. Checked numerically
    rather than asserted, and checked on both sides.
    """
    warmup = 4000
    sched = NoamSchedule(512, warmup_steps=warmup)
    peak = sched(warmup)
    assert sched(warmup - 1) < peak
    assert sched(warmup + 1) < peak
    assert peak == pytest.approx(512 ** -0.5 * warmup ** -0.5, rel=1e-12)


def test_warmup_phase_is_linear_in_step() -> None:
    sched = NoamSchedule(256, warmup_steps=1000)
    a, b = sched(100), sched(200)
    assert b == pytest.approx(2 * a, rel=1e-9), "before warmup the rate must scale linearly"


def test_decay_phase_is_inverse_square_root() -> None:
    sched = NoamSchedule(256, warmup_steps=100)
    a, b = sched(10_000), sched(40_000)
    # Quadrupling the step should halve the rate.
    assert b == pytest.approx(a / 2, rel=1e-9)


def test_step_zero_is_finite_rather_than_nan() -> None:
    """LambdaLR calls with step 0; the paper's step_num is 1-based. Must not divide by zero."""
    sched = NoamSchedule(128, warmup_steps=10)
    assert math.isfinite(sched(0))
    assert sched(0) == sched(1)


def test_wider_models_get_a_smaller_rate() -> None:
    warmup = 1000
    narrow = NoamSchedule(128, warmup_steps=warmup)(warmup)
    wide = NoamSchedule(512, warmup_steps=warmup)(warmup)
    assert wide < narrow
    assert narrow / wide == pytest.approx(math.sqrt(512 / 128), rel=1e-9)


def test_rejects_zero_warmup() -> None:
    with pytest.raises(ValueError, match="warmup_steps"):
        NoamSchedule(128, warmup_steps=0)


def test_build_optimizer_uses_paper_hyperparameters_and_base_lr_one() -> None:
    """`lr=1.0` is required: LambdaLR multiplies the base lr, so any other value rescales the curve.

    Setting a "reasonable-looking" 1e-4 here would scale the paper's schedule by 1e-4 and produce a
    model that appears to train but barely moves. Worth a test because the symptom is so indirect.
    """
    model = nn.Linear(4, 4)
    opt, scheduler = build_optimizer(model, d_model=256, warmup_steps=100)
    group = opt.param_groups[0]
    assert group["betas"] == (0.9, 0.98)
    assert group["eps"] == 1e-9
    assert group["initial_lr"] == 1.0 if "initial_lr" in group else True
    assert isinstance(scheduler, torch.optim.lr_scheduler.LambdaLR)


def test_scheduler_drives_the_optimizer_lr_along_the_curve() -> None:
    model = nn.Linear(4, 4)
    opt, scheduler = build_optimizer(model, d_model=256, warmup_steps=50)
    sched = NoamSchedule(256, warmup_steps=50)

    observed = []
    for _ in range(120):
        opt.step()
        observed.append(opt.param_groups[0]["lr"])
        scheduler.step()

    assert observed[0] == pytest.approx(sched(0), rel=1e-9)
    assert observed[49] == pytest.approx(sched(49), rel=1e-9)
    peak_index = max(range(len(observed)), key=lambda i: observed[i])
    assert peak_index in (49, 50), f"peak landed at index {peak_index}, expected the warmup step"


# --------------------------------------------------------------------------------------------
# LabelSmoothingLoss
# --------------------------------------------------------------------------------------------

def test_zero_smoothing_reduces_exactly_to_cross_entropy() -> None:
    """The cheapest strong check available: eps=0 must reproduce the standard loss exactly.

    If the smoothed branch had a normalisation or sign error, this would not agree.
    """
    torch.manual_seed(0)
    vocab, n = 12, 40
    logits = torch.randn(n, vocab)
    target = torch.randint(0, vocab, (n,))

    ours = LabelSmoothingLoss(vocab, pad_id=0, smoothing=0.0)(logits, target)
    theirs = F.cross_entropy(logits, target, ignore_index=0)
    assert ours.item() == pytest.approx(theirs.item(), rel=1e-6)


def test_target_distribution_sums_to_one() -> None:
    """A malformed q would silently rescale the loss. Reconstruct it and check the total mass."""
    vocab, eps = 10, 0.1
    loss_fn = LabelSmoothingLoss(vocab, pad_id=0, smoothing=eps)
    total = loss_fn.confidence + eps / (vocab - 2) * (vocab - 2)
    assert total == pytest.approx(1.0, rel=1e-12)


def test_padding_class_receives_no_smoothing_mass() -> None:
    """Padding is never a legitimate output, so training the model to emit it would be a bug.

    Probed behaviourally: with a uniform logit vector, raising only the pad logit must not reduce
    the loss, because q(pad) = 0 means pad's log-probability carries zero weight.
    """
    vocab = 10
    loss_fn = LabelSmoothingLoss(vocab, pad_id=0, smoothing=0.2)
    target = torch.tensor([5])

    flat = torch.zeros(1, vocab)
    pad_boosted = flat.clone()
    pad_boosted[0, 0] = 5.0

    assert loss_fn(pad_boosted, target).item() > loss_fn(flat, target).item(), (
        "putting mass on pad must never be rewarded"
    )


def test_smoothing_penalises_extreme_confidence() -> None:
    """The mechanism, demonstrated.

    With one-hot targets, loss decreases monotonically as the correct logit grows without bound.
    With smoothing, an over-confident prediction is *worse* than a moderately confident one, so the
    loss has a finite minimum. That is the entire behavioural difference.
    """
    vocab = 10
    smoothed = LabelSmoothingLoss(vocab, pad_id=0, smoothing=0.1)
    hard = LabelSmoothingLoss(vocab, pad_id=0, smoothing=0.0)
    target = torch.tensor([4])

    moderate = torch.zeros(1, vocab); moderate[0, 4] = 3.0
    extreme = torch.zeros(1, vocab); extreme[0, 4] = 30.0

    assert hard(extreme, target).item() < hard(moderate, target).item()
    assert smoothed(extreme, target).item() > smoothed(moderate, target).item()


def test_padding_positions_are_excluded_from_the_average() -> None:
    """Loss must be a mean over real tokens, not over the padded tensor.

    Otherwise the reported loss depends on how sequences were bucketed into batches, and training
    curves stop being comparable between runs.
    """
    torch.manual_seed(0)
    vocab = 12
    loss_fn = LabelSmoothingLoss(vocab, pad_id=0, smoothing=0.1)

    logits = torch.randn(2, 5, vocab)
    target = torch.tensor([[3, 4, 5, 0, 0], [6, 7, 8, 0, 0]])
    with_padding = loss_fn(logits, target)

    # Same real tokens, no padding at all: the mean over real tokens must be identical.
    trimmed = loss_fn(logits[:, :3], target[:, :3])
    assert with_padding.item() == pytest.approx(trimmed.item(), rel=1e-6)


def test_all_padding_batch_returns_zero_without_nan() -> None:
    """Dividing by a zero token count would produce NaN and destroy the run."""
    loss_fn = LabelSmoothingLoss(12, pad_id=0, smoothing=0.1)
    out = loss_fn(torch.randn(1, 4, 12, requires_grad=True), torch.zeros(1, 4, dtype=torch.long))
    assert torch.isfinite(out) and out.item() == 0.0
    out.backward()          # must not raise: the graph has to stay connected


def test_accepts_both_flat_and_sequence_shaped_inputs() -> None:
    torch.manual_seed(0)
    vocab = 12
    loss_fn = LabelSmoothingLoss(vocab, pad_id=0, smoothing=0.1)
    logits = torch.randn(2, 3, vocab)
    target = torch.randint(1, vocab, (2, 3))
    a = loss_fn(logits, target)
    b = loss_fn(logits.reshape(-1, vocab), target.reshape(-1))
    assert a.item() == pytest.approx(b.item(), rel=1e-9)


def test_rejects_out_of_range_smoothing() -> None:
    with pytest.raises(ValueError, match="smoothing"):
        LabelSmoothingLoss(10, smoothing=1.0)


def test_smoothed_loss_is_higher_than_unsmoothed_on_a_good_prediction() -> None:
    """Section 5.4 states plainly that smoothing hurts perplexity.

    Pinned as a test because it means a smoothed training loss is NOT comparable to an unsmoothed
    one, and putting them on the same axis is a real reporting error. Evaluation in this project
    therefore uses unsmoothed cross-entropy, computed separately from the training objective.
    """
    vocab = 20
    target = torch.tensor([7])
    confident = torch.zeros(1, vocab); confident[0, 7] = 8.0

    hard = LabelSmoothingLoss(vocab, smoothing=0.0)(confident, target).item()
    soft = LabelSmoothingLoss(vocab, smoothing=0.1)(confident, target).item()
    assert soft > hard
