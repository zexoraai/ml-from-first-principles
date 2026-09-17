"""Project 2 — decoder-only GPT, implemented from the paper.

Primary sources
---------------
* Radford, Wu, Child, Luan, Amodei & Sutskever, "Language Models are Unsupervised Multitask
  Learners" (GPT-2, 2019) — architecture, pre-norm placement, scaled residual init, learned
  positional embeddings, byte-level BPE.
* Radford, Narasimhan, Salimans & Sutskever, "Improving Language Understanding by Generative
  Pre-Training" (GPT-1, 2018) — the decoder-only formulation.
* Sennrich, Haddow & Birch, arXiv:1508.07909 — BPE.
* Karpathy, `nanoGPT` and `llm.c` — reference implementations consulted for the same architecture.

Fidelity claim
--------------
Tier **E** for the mechanisms (each verified against oracles and invariants) and tier **R** for the
trained model: a small GPT trained to convergence on TinyShakespeare. The ~124M GPT-2-small
configuration is available as `GPTConfig.gpt2_small()` for cost estimation and is **not trained** —
see `records/GAPS.md` G-001 and `env/feasibility.md`.

No claim is made about GPT-2's published zero-shot benchmark results. Those were not attempted.

Reuse from Project 1
--------------------
`MultiHeadAttention`, `LayerNorm` and `causal_keep_mask` are imported from `labs.p1_transformer`
rather than reimplemented — GPT's causal self-attention is exactly the original Transformer's
decoder self-attention. This keeps P1's correctness suite covering the code that runs here.

Deliberate differences from P1, each matching GPT-2 rather than the 2017 paper: pre-norm instead of
post-norm, a final LayerNorm before the head, GELU instead of ReLU, learned positional embeddings
instead of sinusoids, no label smoothing, and scaled residual initialisation.
"""

from __future__ import annotations

from .data import BatchSampler, download_corpus, prepare
from .model import GPT, GPTBlock, GPTConfig
from .sample import (
    apply_sampling_filters,
    estimate_loss,
    generate,
    generate_with_trace,
)
from .tokenizer import ByteBPETokenizer

__all__ = [
    "GPTConfig",
    "GPT",
    "GPTBlock",
    "ByteBPETokenizer",
    "prepare",
    "download_corpus",
    "BatchSampler",
    "generate",
    "generate_with_trace",
    "apply_sampling_filters",
    "estimate_loss",
]

PAPER = {
    "title": "Language Models are Unsupervised Multitask Learners",
    "authors": "Radford, Wu, Child, Luan, Amodei, Sutskever",
    "year": 2019,
    "url": "https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf",
    "gpt2_small": {"n_layer": 12, "n_head": 12, "d_model": 768,
                   "block_size": 1024, "vocab_size": 50257, "params": "~124M"},
    "our_config": "see runs/<run_id>/meta.json for the exact configuration trained",
}
