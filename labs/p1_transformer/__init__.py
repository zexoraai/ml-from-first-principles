"""Project 1 -- the original encoder-decoder Transformer, implemented from the paper.

Primary source
--------------
Vaswani, Shazeer, Parmar, Uszkoreit, Jones, Gomez, Kaiser & Polosukhin,
"Attention Is All You Need", arXiv:1706.03762, **version v7** (2 Aug 2023 revision of the
June 2017 paper). Section numbers quoted throughout this package refer to v7.

Fidelity claim
--------------
Tier **E** (educational implementation of a mechanism) for everything in this milestone: each
mechanism is written from the paper's equations and checked for correctness against independent
oracles, gradient flow, and structural invariants. Nothing here is a reproduction of the paper's
WMT14 results, and no such claim is made anywhere. See `records/SCOPE.md` section 0.

Which version of the method
---------------------------
The paper text and the authors' own `tensor2tensor` release disagree about where layer
normalization goes. We implement the **paper text** -- post-norm, `LayerNorm(x + Sublayer(x))`,
section 3.1 -- as the default, with pre-norm available behind a flag. Decision D-006.

Equation-to-code map
--------------------
    Eq. 1, section 3.2.1   softmax(QK^T / sqrt(d_k)) V   -> `attention.scaled_dot_product_attention`
    section 3.2.2          MultiHead / head_i            -> `attention.MultiHeadAttention`
    section 3.2.2          Concat(head_1..head_h)        -> `attention.merge_heads`
    section 3.3            FFN(x) = max(0, xW1+b1)W2+b2  -> `layers.PositionwiseFeedForward`
    section 3.1 + 5.4      LayerNorm(x + Dropout(S(x)))  -> `layers.SublayerConnection`
    ref [1] / section 3.1  layer normalization           -> `layers.LayerNorm`
    section 3.5            PE(pos, 2i), PE(pos, 2i+1)    -> `positional.sinusoidal_positional_encoding`
    section 3.1 / 3.2.3    causal + padding masking      -> `masks`

Mechanisms are hand-written. `torch.nn.Transformer`, `torch.nn.MultiheadAttention`,
`torch.nn.functional.scaled_dot_product_attention` and `torch.nn.LayerNorm` are absent from this
package and their absence is enforced by `tests/test_p1_no_builtin_shortcuts.py`.
"""

from __future__ import annotations

from .attention import (
    MultiHeadAttention,
    merge_heads,
    scaled_dot_product_attention,
    split_heads,
)
from .layers import LayerNorm, PositionwiseFeedForward, SublayerConnection
from .masks import (
    causal_keep_mask,
    combine_keep_masks,
    fully_masked_rows,
    padding_key_mask,
)
from .positional import SinusoidalPositionalEncoding, sinusoidal_positional_encoding

__all__ = [
    # attention
    "scaled_dot_product_attention",
    "MultiHeadAttention",
    "split_heads",
    "merge_heads",
    # layers
    "LayerNorm",
    "PositionwiseFeedForward",
    "SublayerConnection",
    # masks
    "padding_key_mask",
    "causal_keep_mask",
    "combine_keep_masks",
    "fully_masked_rows",
    # positional
    "sinusoidal_positional_encoding",
    "SinusoidalPositionalEncoding",
]

PAPER = {
    "title": "Attention Is All You Need",
    "arxiv": "1706.03762",
    "version": "v7",
    "url": "https://arxiv.org/abs/1706.03762",
    "base_config": {  # Table 3, base row / sections 3.2.2, 3.3, 5.4
        "N_layers": 6,
        "d_model": 512,
        "d_ff": 2048,
        "num_heads": 8,
        "d_k": 64,
        "d_v": 64,
        "P_drop": 0.1,
        "label_smoothing": 0.1,
        "train_steps": 100_000,
    },
    "optimizer": {  # section 5.3
        "name": "Adam",
        "beta1": 0.9,
        "beta2": 0.98,
        "eps": 1e-9,
        "schedule": "d_model**-0.5 * min(step**-0.5, step * warmup**-1.5)",
        "warmup_steps": 4000,
    },
    "reported_hardware": "8x NVIDIA P100, 0.4 s/step, 12 h for the base model (section 5.2)",
}
