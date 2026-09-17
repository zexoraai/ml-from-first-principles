"""Project 4 — Direct Preference Optimization versus supervised fine-tuning.

Primary source
--------------
Rafailov, Sharma, Mitchell, Ermon, Manning & Finn, "Direct Preference Optimization: Your Language
Model is Secretly a Reward Model", arXiv:2305.18290.

Fidelity claim
--------------
Tier **E** for the mechanism: the DPO objective, reference-policy handling, sequence log-probabilities
under response-only masking, and the β sweep are implemented and verified. Tier **R** for the
comparison.

The comparison
--------------
Both arms start from **the same SFT checkpoint**, which is also the frozen reference policy. Any
difference between them is therefore attributable to the preference optimisation and not to different
initialisation — that is the only way the comparison means anything.

Preference data provenance
--------------------------
**Constructed, not human-annotated.** See `data.py` for the exact rule and its four degradation types.
The signal is real and verifiable (on-topic, complete, non-repetitive prose versus documented
degradations) which makes it a valid test of the mechanism. It measures nothing about human values,
helpfulness or truthfulness, and every reported number is qualified accordingly.
"""

from __future__ import annotations

from .batching import (
    PairBatcher,
    SFTBatcher,
    encode_pair_batch,
    encode_sft_batch,
)
from .data import (
    DEGRADATIONS,
    InstructionExample,
    PreferencePair,
    build_sft_and_preferences,
)
from .dpo import (
    DPOStats,
    dpo_loss,
    make_reference_model,
    precompute_reference_logps,
    sequence_logprobs,
    sft_loss,
)

__all__ = [
    "sequence_logprobs",
    "dpo_loss",
    "DPOStats",
    "make_reference_model",
    "precompute_reference_logps",
    "sft_loss",
    "build_sft_and_preferences",
    "InstructionExample",
    "PreferencePair",
    "DEGRADATIONS",
    "encode_sft_batch",
    "encode_pair_batch",
    "SFTBatcher",
    "PairBatcher",
]

PAPER = {
    "title": "Direct Preference Optimization: Your Language Model is Secretly a Reward Model",
    "arxiv": "2305.18290",
    "url": "https://arxiv.org/abs/2305.18290",
    "objective": "-log sigmoid( beta * [ (logp_w - logp_ref_w) - (logp_l - logp_ref_l) ] )",
    "implicit_reward": "r(x,y) = beta * log( pi_theta(y|x) / pi_ref(y|x) )",
    "loss_at_initialisation": "exactly log(2) ~= 0.6931, because pi_theta == pi_ref gives margin 0",
}
