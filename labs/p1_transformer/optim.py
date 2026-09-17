"""The paper's learning-rate schedule and label-smoothed loss, both written by hand.

Paper: Vaswani et al., arXiv:1706.03762v7, sections 5.3 and 5.4.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["NoamSchedule", "LabelSmoothingLoss", "build_optimizer"]


class NoamSchedule:
    """Equation 3 of section 5.3.

        lrate = d_model^(-0.5) * min(step^(-0.5), step * warmup^(-1.5))

    Callable, returning a multiplier suitable for `torch.optim.lr_scheduler.LambdaLR`. Because
    LambdaLR multiplies the optimizer's base `lr`, the optimizer must be constructed with
    ``lr=1.0`` for this to reproduce the paper's absolute values — `build_optimizer` does that and
    the reason is worth stating, because setting a "sensible" base lr like 1e-4 alongside this
    schedule silently scales the whole curve by 1e-4 and is a common way to get a model that
    never learns.

    Shape of the curve, and why it is this shape
    -------------------------------------------
    Two regimes meeting at ``step == warmup``:

      * ``step < warmup``  -> ``step * warmup^(-1.5)`` dominates: linear ramp from ~0.
      * ``step > warmup``  -> ``step^(-0.5)`` dominates: inverse-square-root decay.

    At exactly ``step == warmup`` the two expressions are equal (both reduce to
    ``warmup^(-0.5)``), so the peak sits precisely at the warmup step with no discontinuity.
    `tests/test_p1_optim.py::test_peak_is_exactly_at_warmup_step` checks that algebra numerically.

    The warmup is not decorative. The paper's architecture is **post-norm** (section 3.1), where
    every residual sum is immediately re-normalised, so the identity path is rescaled at each of
    the N layers. Early in training the normalisation statistics and the parameters are mismatched,
    gradients through a deep post-norm stack are badly scaled, and a large initial learning rate
    diverges. Warmup is what makes the original recipe trainable at all — which is also why
    pre-norm models can often drop it.

    The ``d_model^(-0.5)`` factor scales the whole curve down as the model widens, because wider
    layers produce larger-magnitude updates for the same nominal learning rate.
    """

    def __init__(self, d_model: int, warmup_steps: int = 4000, scale: float = 1.0) -> None:
        if warmup_steps < 1:
            raise ValueError(f"warmup_steps must be >= 1, got {warmup_steps}")
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        self.scale = scale      # our deviation knob, see note below

    def __call__(self, step: int) -> float:
        # LambdaLR calls with step=0 first. The formula divides by sqrt(step), so step 0 is
        # undefined; the paper's step_num is 1-based. Clamping is the standard fix and it makes
        # the first update use the same lr as step 1 rather than NaN.
        step = max(1, int(step))
        return (
            self.scale
            * self.d_model ** -0.5
            * min(step ** -0.5, step * self.warmup_steps ** -1.5)
        )

    def peak_lr(self) -> float:
        return self(self.warmup_steps)


def build_optimizer(
    model: nn.Module, *, d_model: int, warmup_steps: int = 4000, scale: float = 1.0
) -> tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR]:
    """Adam with the paper's hyperparameters (section 5.3) plus the Noam schedule.

    ``betas=(0.9, 0.98)``, ``eps=1e-9``. Both differ from PyTorch's defaults of ``(0.9, 0.999)``
    and ``1e-8``, and the difference is deliberate on the paper's part: a lower beta2 shortens the
    second-moment memory, which suits a learning rate that is itself changing quickly.

    ``lr=1.0`` is required, not a placeholder — see `NoamSchedule`.
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=1.0, betas=(0.9, 0.98), eps=1e-9)
    schedule = NoamSchedule(d_model, warmup_steps=warmup_steps, scale=scale)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=schedule)
    return optimizer, scheduler


class LabelSmoothingLoss(nn.Module):
    """Label smoothing, section 5.4, with ``eps_ls = 0.1`` in the paper.

    Instead of training against a one-hot target, we train against

        q(k) = 1 - eps                    if k is the correct token
        q(k) = eps / (V - 2)              for every other real token
        q(k) = 0                          if k is the padding token

    and minimise the cross-entropy ``-sum_k q(k) log p(k)``.

    Why exclude padding from the smoothing mass
    -------------------------------------------
    Padding is a bookkeeping symbol, never a legitimate output. Handing it a share of the smoothing
    mass would explicitly train the model to sometimes emit it. Excluding it leaves ``V - 2``
    recipients: the whole vocabulary minus the correct token and minus pad. This matches the
    reference implementations.

    What smoothing actually buys, and what it costs
    ----------------------------------------------
    A one-hot target is only minimised as the correct logit runs to ``+inf``, so the model is
    pushed towards ever more extreme confidence, which is both unattainable and badly calibrated.
    Capping the target at ``1 - eps`` gives the loss a finite minimum at a finite logit gap.

    The paper is unusually blunt about the trade (section 5.4): this **hurts perplexity** — the
    model is deliberately made less certain, and perplexity measures certainty — while improving
    accuracy and BLEU. So a smoothed training loss is *not* comparable to an unsmoothed one, and
    reporting them side by side as if they were is a real error. This is why evaluation here uses
    unsmoothed cross-entropy and exact-match accuracy, both computed separately from the training
    objective.

    Args:
        vocab_size: V.
        pad_id: index excluded from smoothing and ignored in the loss.
        smoothing: eps_ls. ``0.0`` reduces exactly to cross-entropy with ``ignore_index=pad_id``,
            which `tests/test_p1_optim.py` asserts — a cheap way to prove the smoothed path has no
            sign or normalisation error hiding in it.
    """

    def __init__(self, vocab_size: int, *, pad_id: int = 0, smoothing: float = 0.1) -> None:
        super().__init__()
        if not 0.0 <= smoothing < 1.0:
            raise ValueError(f"smoothing must be in [0, 1), got {smoothing}")
        if vocab_size < 3:
            raise ValueError("need at least 3 tokens (correct, pad, and one other) to smooth")
        self.vocab_size = vocab_size
        self.pad_id = pad_id
        self.smoothing = smoothing
        self.confidence = 1.0 - smoothing

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Args: logits (batch, seq, vocab) or (N, vocab); target (batch, seq) or (N,).

        Returns: scalar mean loss over **non-padding** target positions. Averaging over non-pad
        tokens rather than over all tokens matters: otherwise a batch that happens to contain more
        padding reports a smaller loss for no modelling reason, and the curve becomes a function of
        how sequences were bucketed.
        """
        if logits.dim() == 3:
            logits = logits.reshape(-1, logits.size(-1))
            target = target.reshape(-1)
        if logits.size(0) != target.size(0):
            raise ValueError(f"logits/target batch mismatch: {logits.size(0)} vs {target.size(0)}")

        log_probs = F.log_softmax(logits, dim=-1)          # (N, V)
        non_pad = target != self.pad_id                    # (N,)
        n_tokens = int(non_pad.sum())
        if n_tokens == 0:
            return logits.sum() * 0.0                      # keeps the graph connected

        if self.smoothing == 0.0:
            return F.nll_loss(log_probs, target, ignore_index=self.pad_id, reduction="mean")

        # Build q. Spread eps over the V-2 tokens that are neither correct nor pad.
        smooth_value = self.smoothing / (self.vocab_size - 2)
        q = torch.full_like(log_probs, smooth_value)
        q[:, self.pad_id] = 0.0
        q.scatter_(1, target.unsqueeze(1), self.confidence)

        loss_per_token = -(q * log_probs).sum(dim=-1)       # (N,)
        return (loss_per_token * non_pad).sum() / n_tokens
