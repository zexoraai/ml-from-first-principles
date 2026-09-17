"""LoRA — Low-Rank Adaptation, implemented from the paper.

Primary source
--------------
Hu, Shen, Wallis, Allen-Zhu, Li, Wang, Wang & Chen, "LoRA: Low-Rank Adaptation of Large Language
Models", arXiv:2106.09685.

THE IDEA, PRECISELY
-------------------
A pretrained weight matrix `W0 ∈ R^{d_out × d_in}` is frozen. The update is constrained to be
low-rank:

    W = W0 + ΔW,     ΔW = (α / r) · B A,     A ∈ R^{r × d_in},  B ∈ R^{d_out × r}

with `r ≪ min(d_in, d_out)`. Only `A` and `B` are trained, so the trainable parameter count drops
from `d_in · d_out` to `r · (d_in + d_out)`.

At `d_in = d_out = 192` and `r = 8` that is 36,864 → 3,072 parameters: a 12× reduction. The saving
grows with width, which is why the technique matters at scale and is merely convenient at ours.

WHY IT WORKS AT ALL (the paper's hypothesis, stated as a hypothesis)
-------------------------------------------------------------------
The paper's argument is that the *change* required to adapt a pretrained model to a downstream task
has low intrinsic rank, even though the weights themselves do not. That is an empirical claim which
the paper supports and this project does not independently verify — we verify the *mechanism*, and
measure whether it works on our task.

THE INITIALISATION IS NOT ARBITRARY
-----------------------------------
`A ~ N(0, σ²)` and `B = 0`. Therefore `BA = 0` at step zero, so the adapted model **starts exactly
equal to the base model**. This matters for two reasons:

1. Fine-tuning begins from the pretrained function rather than from a randomly perturbed one, so
   there is no initial loss spike destroying pretrained knowledge.
2. It gives a free correctness check: an untrained adapter must be a no-op. If it is not, something
   is wired wrong. `tests/test_p3_lora.py::test_untrained_adapter_is_exactly_the_identity` is that
   check.

Initialising *both* to zero would leave the gradient of both at zero forever (the product rule gives
`∂(BA)/∂A ∝ B = 0` and `∂(BA)/∂B ∝ A = 0`), so nothing would ever learn. Initialising both randomly
would perturb the pretrained model before training starts. Exactly one of them must be zero.

WHY α/r AND NOT JUST α
----------------------
Section 4.1: the update is scaled by `α/r`. Dividing by the rank means that when you change `r`, the
*magnitude* of the update stays roughly comparable, so a learning rate tuned at `r = 8` remains
sensible at `r = 64`. Without it, higher rank silently means a larger effective step size, and rank
and learning rate become entangled hyperparameters.

MERGING
-------
Because `ΔW` is just a matrix, it can be folded into `W0` once training finishes:
`W ← W0 + (α/r)·BA`. The merged model has **identical architecture and identical inference cost** to
the original — no extra matmuls, no latency penalty. That is LoRA's main practical advantage over
adapter layers that insert new modules, and it is why `merge()`/`unmerge()` are part of the
implementation rather than a footnote.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

__all__ = ["LoRALinear", "apply_lora", "merge_all", "unmerge_all", "lora_state_dict",
           "count_parameters", "mark_only_lora_trainable"]


class LoRALinear(nn.Module):
    """Wraps a frozen `nn.Linear` with a trainable low-rank update.

    Args:
        base: the pretrained layer. Its weight and bias are frozen in place.
        r: rank of the update. `r = 0` disables LoRA entirely (useful as a control arm).
        alpha: scaling numerator; the applied scale is `alpha / r`.
        dropout: applied to the *input* of the low-rank branch, as in the paper's implementation.

    The base layer is held as a submodule rather than copied, so `state_dict()` still contains the
    original weights under a predictable name and a checkpoint stays inspectable.
    """

    def __init__(
        self,
        base: nn.Linear,
        *,
        r: int = 8,
        alpha: float = 16.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if r < 0:
            raise ValueError(f"rank must be >= 0, got {r}")
        self.base = base
        self.r = r
        self.alpha = float(alpha)
        self.scaling = (self.alpha / r) if r > 0 else 0.0
        self.merged = False

        # Freeze the pretrained weights. This is the whole point: gradients must not reach them, so
        # the optimiser never allocates moment buffers for them either -- which is where most of
        # LoRA's memory saving actually comes from, not from the parameter count itself.
        self.base.weight.requires_grad_(False)
        if self.base.bias is not None:
            self.base.bias.requires_grad_(False)

        if r > 0:
            self.lora_A = nn.Parameter(torch.empty(r, base.in_features))
            self.lora_B = nn.Parameter(torch.zeros(base.out_features, r))
            self.lora_dropout = nn.Dropout(dropout)
            self.reset_lora_parameters()
        else:
            self.register_parameter("lora_A", None)
            self.register_parameter("lora_B", None)
            self.lora_dropout = nn.Identity()

    def reset_lora_parameters(self) -> None:
        """Kaiming-uniform `A`, zeros `B` — so `BA = 0` and the adapter starts as the identity.

        The paper says "random Gaussian" for A; the authors' released code uses Kaiming uniform, and
        we follow the code. Either satisfies the requirement that exactly one factor is nonzero.
        """
        if self.r > 0:
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)

    @property
    def in_features(self) -> int:
        return self.base.in_features

    @property
    def out_features(self) -> int:
        return self.base.out_features

    def delta_weight(self) -> torch.Tensor:
        """`(α/r) · B A`, shaped like `base.weight` — i.e. `(out_features, in_features)`."""
        if self.r == 0:
            return torch.zeros_like(self.base.weight)
        return (self.lora_B @ self.lora_A) * self.scaling

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """`base(x) + (α/r) · (dropout(x) A^T) B^T`, or just `base(x)` once merged.

        Note the low-rank branch is computed as two small matmuls rather than by materialising `ΔW`.
        For `d = 192, r = 8` that is `x·A^T` (192→8) then `·B^T` (8→192): 3,072 multiply-accumulates
        per token instead of the 36,864 a full `d × d` matrix would cost. Materialising `ΔW` every
        forward pass would throw away the entire computational advantage — it is only worth doing
        once, at merge time.
        """
        if self.merged or self.r == 0:
            return self.base(x)
        base_out = self.base(x)
        low_rank = self.lora_dropout(x) @ self.lora_A.T @ self.lora_B.T
        return base_out + low_rank * self.scaling

    # -----------------------------------------------------------------------------------------
    @torch.no_grad()
    def merge(self) -> None:
        """Fold `ΔW` into the base weight. Inference then costs exactly what the base model cost."""
        if self.merged or self.r == 0:
            return
        self.base.weight.add_(self.delta_weight())
        self.merged = True

    @torch.no_grad()
    def unmerge(self) -> None:
        """Subtract `ΔW` back out.

        Exact reversal is not guaranteed in floating point: `(w + d) - d != w` in general. The
        round-trip error is bounded by a few float32 epsilons relative to `|w|`, which is far below
        anything that affects behaviour — but the test asserts a tolerance rather than equality,
        because asserting equality would be false.
        """
        if not self.merged or self.r == 0:
            return
        self.base.weight.sub_(self.delta_weight())
        self.merged = False

    def extra_repr(self) -> str:
        return (f"in={self.in_features}, out={self.out_features}, r={self.r}, "
                f"alpha={self.alpha}, scaling={self.scaling:.4f}, merged={self.merged}")


# ---------------------------------------------------------------------------------------------
# applying LoRA to an existing model
# ---------------------------------------------------------------------------------------------

def apply_lora(
    model: nn.Module,
    *,
    target_suffixes: tuple[str, ...] = ("w_q", "w_v"),
    r: int = 8,
    alpha: float = 16.0,
    dropout: float = 0.0,
) -> list[str]:
    """Replace matching `nn.Linear` submodules with `LoRALinear`, in place.

    Args:
        target_suffixes: attribute names to adapt. Default `("w_q", "w_v")` follows the paper's
            main configuration: **query and value projections only**.

    Why query and value, and not key or the MLP
    -------------------------------------------
    Section 7.1 of the paper ablates this and reports that, for a fixed parameter budget, adapting
    `W_q` and `W_v` beats spending the same budget on a single matrix at higher rank. We follow that
    finding but do not independently verify it — the choice is configurable precisely so it can be
    tested rather than assumed, and `--targets` on the training script exposes it.

    Returns the list of adapted module paths, so a caller can record exactly what was touched instead
    of assuming the pattern matched what it intended. A silent zero-match is a real failure mode: you
    get an untrainable model that looks fine.
    """
    adapted: list[str] = []
    for name, module in list(model.named_modules()):
        for attr, child in list(module.named_children()):
            if attr in target_suffixes and isinstance(child, nn.Linear):
                setattr(module, attr, LoRALinear(child, r=r, alpha=alpha, dropout=dropout))
                adapted.append(f"{name}.{attr}" if name else attr)
    if not adapted:
        raise ValueError(
            f"no modules matched {target_suffixes!r}. Available Linear attribute names include: "
            f"{sorted({a for _, m in model.named_modules() for a, c in m.named_children() if isinstance(c, nn.Linear)})}"
        )
    return adapted


def mark_only_lora_trainable(model: nn.Module) -> None:
    """Freeze everything except `lora_A` / `lora_B`.

    `apply_lora` already freezes the wrapped base layers, but the rest of the model — embeddings,
    layer norms, the output head — is still trainable. Leaving them so would make the comparison
    against full fine-tuning meaningless, because the "LoRA" arm would be training most of the model.
    """
    for name, param in model.named_parameters():
        param.requires_grad_("lora_A" in name or "lora_B" in name)


def merge_all(model: nn.Module) -> int:
    n = 0
    for module in model.modules():
        if isinstance(module, LoRALinear):
            module.merge()
            n += 1
    return n


def unmerge_all(model: nn.Module) -> int:
    n = 0
    for module in model.modules():
        if isinstance(module, LoRALinear):
            module.unmerge()
            n += 1
    return n


def lora_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    """Only the adapter tensors — what you would actually ship.

    This is the deployment story: a few hundred kilobytes per task instead of a full model copy, all
    sharing one frozen base. `tests/test_p3_lora.py` checks that loading these into a fresh base
    reproduces the adapted model exactly.
    """
    return {k: v.detach().clone() for k, v in model.state_dict().items()
            if "lora_A" in k or "lora_B" in k}


def count_parameters(model: nn.Module) -> dict[str, int]:
    """Trainable / frozen / total, counting each Parameter object once.

    Tied weights must not be double-counted, and `named_parameters()` already de-duplicates by
    identity — but an explicit id set makes that guarantee local and obvious rather than inherited.
    """
    seen: set[int] = set()
    trainable = frozen = 0
    lora = 0
    for name, p in model.named_parameters():
        if id(p) in seen:
            continue
        seen.add(id(p))
        if p.requires_grad:
            trainable += p.numel()
            if "lora_A" in name or "lora_B" in name:
                lora += p.numel()
        else:
            frozen += p.numel()
    return {
        "trainable": trainable,
        "frozen": frozen,
        "total": trainable + frozen,
        "lora": lora,
        "trainable_fraction": trainable / max(trainable + frozen, 1),
    }
