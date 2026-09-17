"""Project 3 — LoRA, implemented from the paper.

Primary source
--------------
Hu, Shen, Wallis, Allen-Zhu, Li, Wang, Wang & Chen, "LoRA: Low-Rank Adaptation of Large Language
Models", arXiv:2106.09685.

Fidelity claim
--------------
Tier **E** for the mechanism: the low-rank update, the `α/r` scaling, the frozen base, and
merge/unmerge equivalence are all implemented and verified. Tier **R** for the fine-tuning
comparison.

The comparison the brief requires
---------------------------------
Base model (no adaptation) versus LoRA versus **full fine-tuning of the same model**. The base model
is Project 2's trained GPT, which is small enough that full fine-tuning genuinely runs on this CPU —
that is the sizing criterion, because a comparison against a full fine-tune that could not be
executed would be a fiction.

What is measured: trainable parameter count, peak memory, wall clock, and task quality, all under
documented conditions.
"""

from __future__ import annotations

from .lora import (
    LoRALinear,
    apply_lora,
    count_parameters,
    lora_state_dict,
    mark_only_lora_trainable,
    merge_all,
    unmerge_all,
)

__all__ = [
    "LoRALinear",
    "apply_lora",
    "mark_only_lora_trainable",
    "merge_all",
    "unmerge_all",
    "lora_state_dict",
    "count_parameters",
]

PAPER = {
    "title": "LoRA: Low-Rank Adaptation of Large Language Models",
    "arxiv": "2106.09685",
    "url": "https://arxiv.org/abs/2106.09685",
    "update": "W = W0 + (alpha/r) * B @ A,  A in R^{r x d_in}, B in R^{d_out x r}",
    "init": "A ~ Kaiming uniform (paper says random Gaussian; the released code uses Kaiming), B = 0",
    "default_targets": "W_q and W_v only, following the section 7.1 ablation",
}
