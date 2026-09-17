"""Direct Preference Optimization, implemented from the paper.

Primary source
--------------
Rafailov, Sharma, Mitchell, Ermon, Manning & Finn, "Direct Preference Optimization: Your Language
Model is Secretly a Reward Model", arXiv:2305.18290.

THE OBJECTIVE
-------------
Given a prompt `x`, a preferred response `y_w` and a dispreferred one `y_l`, and a frozen reference
policy `π_ref` (in practice the SFT checkpoint the run started from):

    L_DPO = -E log σ( β · [ log π_θ(y_w|x) − log π_ref(y_w|x)
                          − log π_θ(y_l|x) + log π_ref(y_l|x) ] )

WHAT MAKES THIS INTERESTING, STATED PRECISELY
---------------------------------------------
RLHF trains an explicit reward model on preferences and then optimises the policy against it with
PPO: two models, a sampling loop, and a KL penalty to keep the policy near the reference. DPO's
result is that for the standard RLHF objective the optimal policy has a closed form in terms of the
reward, which can be inverted — the reward is *implicitly* defined by the policy ratio

    r(x, y) = β · log( π_θ(y|x) / π_ref(y|x) )

so the preference likelihood can be written directly in terms of the policy. No reward model, no
sampling during training, no PPO. It becomes a supervised classification problem on pairs.

The KL constraint has not vanished — it is *inside* the objective. `β` is the same `β` that weighted
the KL penalty in the RLHF formulation, and the reference log-probabilities are what keep the policy
anchored. That is why `π_ref` is not optional and why forgetting to freeze it silently changes the
objective into something with no fixed point.

FOUR THINGS THAT MUST BE RIGHT, AND ARE EASY TO GET WRONG
---------------------------------------------------------
1. **Response-only masking.** `log π(y|x)` sums over the tokens of `y`, never over `x`. Include the
   prompt and every pair sharing a prompt gets the same large constant added to both sides — which
   cancels in the *difference*, but not in the per-response numbers you log, and not at all once
   pairs have different prompt lengths. `sequence_logprobs` masks explicitly.
2. **Sum, not mean.** The paper's `log π(y|x)` is a sum over response tokens. Using a mean makes the
   objective length-invariant, which sounds desirable and is a different algorithm. The consequence
   of the sum is a known length bias — longer responses have more negative log-probability — and it
   is documented rather than silently "fixed".
3. **The reference must be frozen and in eval mode.** `requires_grad_(False)` and `.eval()`. Dropout
   left active in the reference makes `log π_ref` stochastic, so the same pair yields a different
   loss each epoch and the anchor drifts randomly.
4. **`logsigmoid`, not `log(sigmoid(·))`.** For large negative arguments `sigmoid` underflows to 0
   and the log is `-inf`. `F.logsigmoid` is computed stably. Early in training the margin can be
   large, so this is a live concern rather than a theoretical one.

WHAT THIS PROJECT DOES NOT CLAIM
--------------------------------
That a better preference score means a better, safer, or more truthful model. It means the policy has
moved toward the preferences it was shown. Those are different statements and the project page says
so next to every number.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    "sequence_logprobs",
    "dpo_loss",
    "DPOStats",
    "make_reference_model",
    "sft_loss",
]


def sequence_logprobs(
    logits: torch.Tensor,
    labels: torch.Tensor,
    response_mask: torch.Tensor,
    *,
    average: bool = False,
) -> torch.Tensor:
    """Sum of token log-probabilities over the response only.

    Args:
        logits: (B, T, V) raw model outputs for the full sequence.
        labels: (B, T) token ids of the full sequence (prompt + response).
        response_mask: (B, T) 1 where the position is a **response** token that should count, 0 for
            prompt and padding.
        average: divide by the response length. Not the paper's objective; provided only so the
            length-bias experiment on the project page can be run rather than asserted.

    Returns:
        (B,) log π(y|x) per sequence.

    The shift, which is the whole subtlety
    --------------------------------------
    A causal model's output at position `t` predicts the token at position `t+1`. So the
    log-probability of `labels[t]` comes from `logits[t-1]`. We therefore drop the last logit and the
    first label:

        logits[:, :-1]  aligns with  labels[:, 1:]

    Getting this off by one produces a model that appears to train — the loss goes down — while
    scoring every token against the wrong prediction. The mask is shifted identically, so a response
    token still lines up with the logit that predicted it.
    """
    if logits.shape[:2] != labels.shape:
        raise ValueError(f"logits {tuple(logits.shape)[:2]} and labels {tuple(labels.shape)} disagree")
    if response_mask.shape != labels.shape:
        raise ValueError("response_mask must have the same shape as labels")

    logits = logits[:, :-1, :]
    target = labels[:, 1:]
    mask = response_mask[:, 1:].to(logits.dtype)

    log_probs = F.log_softmax(logits.float(), dim=-1)
    token_logps = log_probs.gather(dim=-1, index=target.unsqueeze(-1)).squeeze(-1)   # (B, T-1)
    token_logps = token_logps * mask

    total = token_logps.sum(dim=-1)
    if average:
        return total / mask.sum(dim=-1).clamp_min(1.0)
    return total


@dataclass
class DPOStats:
    """Diagnostics that make a DPO run interpretable rather than just a falling number."""
    loss: torch.Tensor
    chosen_reward: torch.Tensor        # β·(logπ − logπ_ref) for the preferred response, mean
    rejected_reward: torch.Tensor
    reward_margin: torch.Tensor        # chosen − rejected; the quantity the loss actually maximises
    reward_accuracy: torch.Tensor      # fraction of pairs with margin > 0
    chosen_logps: torch.Tensor
    rejected_logps: torch.Tensor

    def to_dict(self) -> dict[str, float]:
        # `loss` is deliberately NOT detached on the dataclass -- it is the backward target. So detach
        # here, at the logging boundary, rather than calling float() on a grad-tracking tensor.
        return {k: getattr(self, k).detach().mean().item() for k in
                ("loss", "chosen_reward", "rejected_reward", "reward_margin", "reward_accuracy",
                 "chosen_logps", "rejected_logps")}


def dpo_loss(
    policy_chosen_logps: torch.Tensor,
    policy_rejected_logps: torch.Tensor,
    ref_chosen_logps: torch.Tensor,
    ref_rejected_logps: torch.Tensor,
    *,
    beta: float = 0.1,
    label_smoothing: float = 0.0,
) -> DPOStats:
    """The DPO objective, plus the diagnostics worth logging.

    Args:
        *_logps: (B,) sequence log-probabilities, all four for the same batch of pairs.
        beta: controls how far the policy may move from the reference. See the note below.
        label_smoothing: the cDPO variant from the paper's appendix, which assumes preference labels
            are noisy with probability `label_smoothing`. `0.0` is plain DPO.

    HOW THE LOSS RESPONDS TO ITS INPUTS — the thing to be able to explain
    --------------------------------------------------------------------
    Write the implicit rewards as
        r_w = β·(log π_θ(y_w) − log π_ref(y_w))
        r_l = β·(log π_θ(y_l) − log π_ref(y_l))
    then `loss = -logsigmoid(r_w − r_l)`, so **only the margin matters**, not either reward alone.

    * At initialisation `π_θ = π_ref`, so both rewards are 0, the margin is 0, and the loss is
      exactly `-log σ(0) = log 2 ≈ 0.6931`. That is a free correctness check and it is tested.
    * The gradient of `-logsigmoid(z)` is `-σ(-z)`, which is largest when `z` is very negative — i.e.
      the model learns fastest from pairs it currently gets *backwards*. Pairs it already ranks
      correctly contribute little. This is the same self-limiting shape as logistic regression.
    * Raising the chosen log-probability and lowering the rejected one both increase the margin.
      Nothing in the objective requires the chosen response's absolute probability to rise: DPO
      routinely *decreases both* while increasing the gap. That is not a bug, it is what the objective
      asks for, and it is why `chosen_logps` is logged separately from `reward_margin`.

    WHAT β DOES
    -----------
    `β` scales the margin before the sigmoid, so it sets how large a policy shift counts as
    "confident". Small `β` (0.01–0.05) permits large deviation from the reference and risks drifting
    into degenerate text; large `β` (0.5+) keeps the policy tightly anchored and barely moves it. It
    is the KL weight from the RLHF formulation, surviving into a supervised objective. Because it
    multiplies the margin, it also scales the gradient — so `β` and the learning rate are not
    independent, and the project page sweeps `β` rather than picking one and asserting it.
    """
    if not 0.0 <= label_smoothing < 0.5:
        raise ValueError(f"label_smoothing must be in [0, 0.5), got {label_smoothing}")

    chosen_reward = beta * (policy_chosen_logps - ref_chosen_logps)
    rejected_reward = beta * (policy_rejected_logps - ref_rejected_logps)
    margin = chosen_reward - rejected_reward

    if label_smoothing > 0.0:
        # cDPO: with probability `label_smoothing` the annotator was wrong, so the target is a
        # mixture. Both directions appear, which stops the model driving the margin to infinity on
        # pairs that may be mislabelled.
        losses = (
            -F.logsigmoid(margin) * (1.0 - label_smoothing)
            - F.logsigmoid(-margin) * label_smoothing
        )
    else:
        # logsigmoid, never log(sigmoid(x)): for margin << 0 the latter underflows to log(0) = -inf.
        losses = -F.logsigmoid(margin)

    return DPOStats(
        loss=losses.mean(),
        chosen_reward=chosen_reward.detach(),
        rejected_reward=rejected_reward.detach(),
        reward_margin=margin.detach(),
        reward_accuracy=(margin.detach() > 0).float(),
        chosen_logps=policy_chosen_logps.detach(),
        rejected_logps=policy_rejected_logps.detach(),
    )


def make_reference_model(model: nn.Module) -> nn.Module:
    """A frozen, eval-mode deep copy to serve as `π_ref`.

    Both parts matter and both have been got wrong in published code:

    * **Frozen.** If the reference receives gradients it moves with the policy, the ratio
      `π_θ/π_ref` stays near 1 by construction, and the objective loses its anchor entirely — the
      loss falls while nothing is learned.
    * **eval mode.** Dropout left active makes `log π_ref` a random variable, so the same pair scores
      differently every epoch and the anchor jitters. The loss curve gets noisy for a reason that
      looks like data noise.

    A deep copy costs a second set of weights in memory. At our scale (2.9 M parameters, ~12 MB) that
    is irrelevant; at 7 B it is the dominant memory cost of DPO and the reason implementations cache
    reference log-probabilities in a pre-pass instead. `precompute_reference_logps` exists for that
    and is what the training script uses.
    """
    reference = copy.deepcopy(model)
    reference.requires_grad_(False)
    reference.eval()
    return reference


def sft_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    response_mask: torch.Tensor,
) -> torch.Tensor:
    """Supervised fine-tuning loss: cross-entropy on **response tokens only**.

    This is the baseline DPO is compared against, and it is also how `π_ref` is produced — both arms
    of the comparison start from the same SFT checkpoint, so any difference is attributable to the
    preference optimisation rather than to different starting points.

    Masking the prompt is not an optimisation. Training the model to predict the prompt teaches it to
    generate instructions rather than answer them, which at small scale shows up as a model that
    happily writes its own questions.
    """
    logits = logits[:, :-1, :]
    target = labels[:, 1:]
    mask = response_mask[:, 1:]

    loss_per_token = F.cross_entropy(
        logits.reshape(-1, logits.size(-1)).float(),
        target.reshape(-1),
        reduction="none",
    ).view(target.shape)

    counted = mask.sum().clamp_min(1)
    return (loss_per_token * mask).sum() / counted


@torch.no_grad()
def precompute_reference_logps(
    reference: nn.Module,
    batches,
    *,
    device: torch.device | str = "cpu",
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Score every pair under the reference once, up front.

    `log π_ref` never changes during training — the reference is frozen — so recomputing it every
    step is pure waste: two extra forward passes per step, forever. Precomputing turns DPO's cost
    from four forward passes per step into two.

    At 2.9 M parameters this is a convenience. At 7 B it is the difference between DPO fitting in
    memory and not, because the reference model can then be discarded entirely after the pre-pass.
    """
    reference.eval()
    out = []
    for batch in batches:
        chosen_logits, _ = reference(batch["chosen_ids"].to(device))
        rejected_logits, _ = reference(batch["rejected_ids"].to(device))
        out.append((
            sequence_logprobs(chosen_logits, batch["chosen_ids"].to(device),
                              batch["chosen_mask"].to(device)),
            sequence_logprobs(rejected_logits, batch["rejected_ids"].to(device),
                              batch["rejected_mask"].to(device)),
        ))
    return out
