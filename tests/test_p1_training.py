"""Checkpoint/resume correctness and the training-loop contract.

A resumed run must continue the *same* experiment. If it does not, the loss curve has a hidden
discontinuity at every resume point and the run is really several different runs stitched together
— which invalidates any claim built on the curve.
"""

from __future__ import annotations

import random

import numpy as np
import pytest
import torch

from labs.p1_transformer import (
    LabelSmoothingLoss,
    Transformer,
    TransformerConfig,
    build_optimizer,
)
from labs.p1_transformer.data import CharTokenizer, DateDataset, DateTaskConfig, collate

PAIRS = [
    ("March 3, 2019", "2019-03-03"),
    ("3 Mar 1987", "1987-03-03"),
    ("Sunday, March 3, 2019", "2019-03-03"),
    ("22 September 2001", "2001-09-22"),
]


def make_setup(seed: int = 0):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    tok = CharTokenizer()
    cfg = TransformerConfig(
        vocab_size=len(tok), d_model=32, num_heads=4, d_ff=64,
        num_encoder_layers=1, num_decoder_layers=1, max_len=40, dropout=0.1,
        pad_id=tok.pad_id, bos_id=tok.bos_id, eos_id=tok.eos_id,
    )
    model = Transformer(cfg)
    optimizer, scheduler = build_optimizer(model, d_model=cfg.d_model, warmup_steps=10)
    loss_fn = LabelSmoothingLoss(len(tok), pad_id=tok.pad_id, smoothing=0.1)
    ds = DateDataset(PAIRS, tok, DateTaskConfig())
    batch = collate([ds[i] for i in range(len(PAIRS))], tok.pad_id)
    return tok, model, optimizer, scheduler, loss_fn, batch


def train_steps(model, optimizer, scheduler, loss_fn, batch, n: int) -> list[float]:
    model.train()
    out = []
    for _ in range(n):
        optimizer.zero_grad(set_to_none=True)
        loss = loss_fn(model(batch["src"], batch["tgt_in"]), batch["labels"])
        loss.backward()
        optimizer.step()
        scheduler.step()
        out.append(loss.item())
    return out


def test_resume_reproduces_the_next_step_loss_exactly(tmp_path) -> None:
    """Save mid-run, reload into a fresh object, and require the next losses to match bit-for-bit.

    This is only possible if optimizer moments, scheduler position AND all three RNG states were
    saved. Dropout masks come from the torch RNG, so omitting it produces a small but real
    divergence — the kind that looks like noise and quietly makes a run unreproducible.
    """
    tok, model, optimizer, scheduler, loss_fn, batch = make_setup(0)
    train_steps(model, optimizer, scheduler, loss_fn, batch, 12)

    ckpt = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "rng": {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state()},
    }
    path = tmp_path / "ckpt.pt"
    torch.save(ckpt, path)

    continued = train_steps(model, optimizer, scheduler, loss_fn, batch, 5)

    # Fresh objects, restored state.
    _, model2, optimizer2, scheduler2, loss_fn2, batch2 = make_setup(999)   # deliberately different seed
    loaded = torch.load(path, map_location="cpu", weights_only=False)
    model2.load_state_dict(loaded["model"])
    optimizer2.load_state_dict(loaded["optimizer"])
    scheduler2.load_state_dict(loaded["scheduler"])
    random.setstate(loaded["rng"]["python"])
    np.random.set_state(loaded["rng"]["numpy"])
    torch.set_rng_state(loaded["rng"]["torch"])

    resumed = train_steps(model2, optimizer2, scheduler2, loss_fn2, batch2, 5)

    for i, (a, b) in enumerate(zip(continued, resumed)):
        assert a == pytest.approx(b, rel=0, abs=1e-12), (
            f"step {i} after resume diverged: {a!r} vs {b!r}. Something in the training state "
            f"was not checkpointed."
        )


def test_omitting_rng_state_causes_divergence() -> None:
    """Shows the guard above is load-bearing, not vacuous.

    Same weights and optimizer state, different RNG: dropout masks differ, so the losses differ. If
    this test ever fails, dropout has been turned off somewhere and the resume test has stopped
    proving anything about RNG handling.
    """
    tok, model, optimizer, scheduler, loss_fn, batch = make_setup(0)
    train_steps(model, optimizer, scheduler, loss_fn, batch, 6)
    state = {"model": model.state_dict(), "optimizer": optimizer.state_dict(),
             "scheduler": scheduler.state_dict()}

    torch.manual_seed(1234)
    a = train_steps(model, optimizer, scheduler, loss_fn, batch, 3)

    _, model2, optimizer2, scheduler2, loss_fn2, batch2 = make_setup(0)
    model2.load_state_dict(state["model"])
    optimizer2.load_state_dict(state["optimizer"])
    scheduler2.load_state_dict(state["scheduler"])
    torch.manual_seed(4321)
    b = train_steps(model2, optimizer2, scheduler2, loss_fn2, batch2, 3)

    assert a != b, "with dropout active, different RNG streams must give different losses"


def test_scheduler_state_survives_a_round_trip(tmp_path) -> None:
    """Restoring a scheduler must restore its *position*, and continuing must follow the same curve.

    A subtlety worth knowing, and the reason an earlier version of this test failed:
    `LRScheduler.load_state_dict` restores `last_epoch` and `_last_lr`, but it does **not** write the
    learning rate back into `optimizer.param_groups`. Load only the scheduler and the optimizer is
    still sitting on whatever lr the scheduler's constructor set (the step-1 value), which for the
    Noam curve during warmup is far smaller than the true value -- 0.0056 versus 0.0291 at step 37
    in this configuration.

    Real resume code is unaffected because `optimizer.state_dict()` carries `param_groups`, lr
    included, so loading the optimizer restores it. This test therefore asserts what PyTorch
    actually guarantees (`get_last_lr`), and then verifies the property that matters: the lr sequence
    after a resume matches the sequence an uninterrupted run would have produced.
    `scripts/train_p1.py` additionally syncs the lr explicitly so correctness does not depend on the
    order in which the two state dicts happen to be loaded.
    """
    _, model, optimizer, scheduler, _, _ = make_setup(0)
    for _ in range(37):
        optimizer.step()
        scheduler.step()
    lr_before = optimizer.param_groups[0]["lr"]

    path = tmp_path / "s.pt"
    torch.save({"optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict()}, path)

    # The uninterrupted continuation, for comparison.
    expected_next = []
    for _ in range(4):
        optimizer.step()
        scheduler.step()
        expected_next.append(optimizer.param_groups[0]["lr"])

    _, model2, optimizer2, scheduler2, _, _ = make_setup(0)
    blob = torch.load(path, weights_only=False)
    scheduler2.load_state_dict(blob["scheduler"])

    # What PyTorch does promise: the scheduler remembers where it was.
    assert scheduler2.get_last_lr()[0] == pytest.approx(lr_before, rel=1e-12)

    # What it does not promise: that the optimizer was updated. Loading the optimizer fixes it.
    optimizer2.load_state_dict(blob["optimizer"])
    assert optimizer2.param_groups[0]["lr"] == pytest.approx(lr_before, rel=1e-12)

    got_next = []
    for _ in range(4):
        optimizer2.step()
        scheduler2.step()
        got_next.append(optimizer2.param_groups[0]["lr"])

    assert got_next == pytest.approx(expected_next, rel=1e-12), (
        "the learning-rate curve after resume diverged from the uninterrupted run"
    )


def test_loading_only_the_scheduler_leaves_the_optimizer_lr_stale() -> None:
    """Pins the surprising behaviour above so it cannot silently change under us.

    If a future PyTorch starts syncing param_groups on load, this test fails and we learn about it
    deliberately rather than by noticing an odd loss curve months later.
    """
    _, _, optimizer, scheduler, _, _ = make_setup(0)
    for _ in range(37):
        optimizer.step()
        scheduler.step()
    state = scheduler.state_dict()
    lr_at_37 = optimizer.param_groups[0]["lr"]

    _, _, optimizer2, scheduler2, _, _ = make_setup(0)
    fresh_lr = optimizer2.param_groups[0]["lr"]
    scheduler2.load_state_dict(state)

    assert optimizer2.param_groups[0]["lr"] == pytest.approx(fresh_lr), (
        "load_state_dict is not expected to touch optimizer.param_groups"
    )
    assert optimizer2.param_groups[0]["lr"] != pytest.approx(lr_at_37), (
        "if these are equal the distinction this test documents has disappeared"
    )


def test_loss_decreases_over_a_short_run() -> None:
    """A weak but useful smoke test: the loop must make progress on four memorisable examples."""
    tok, model, optimizer, scheduler, loss_fn, batch = make_setup(0)
    losses = train_steps(model, optimizer, scheduler, loss_fn, batch, 150)
    assert losses[-1] < losses[0], f"loss went from {losses[0]:.4f} to {losses[-1]:.4f}"
    assert all(torch.isfinite(torch.tensor(l)) for l in losses), "non-finite loss during training"


def test_state_dict_round_trip_preserves_outputs_exactly() -> None:
    """Saving and loading must not perturb the model at all."""
    tok, model, _, _, _, batch = make_setup(0)
    model.eval()
    with torch.no_grad():
        before = model(batch["src"], batch["tgt_in"])

    cfg = model.cfg
    clone = Transformer(cfg)
    clone.load_state_dict(model.state_dict())
    clone.eval()
    with torch.no_grad():
        after = clone(batch["src"], batch["tgt_in"])

    assert torch.equal(before, after)


def test_tied_weights_survive_a_state_dict_round_trip() -> None:
    """Tying must be re-established on load, not silently broken into two independent tensors."""
    tok, model, _, _, _, _ = make_setup(0)
    clone = Transformer(model.cfg)
    clone.load_state_dict(model.state_dict())
    assert clone.generator.weight is clone.embedding.weight
    assert torch.equal(clone.generator.weight, model.generator.weight)
