"""Project 5 — memory-efficient exact attention: online softmax, tiling, and a Triton kernel.

Primary sources
---------------
* Dao, Fu, Ermon, Rudra & Ré, "FlashAttention: Fast and Memory-Efficient Exact Attention with
  IO-Awareness", arXiv:2205.14135
* Dao, "FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning",
  arXiv:2307.08691
* Milakov & Gimelshein, "Online normalizer calculation for softmax", arXiv:1805.02867

FIDELITY, SPLIT BY COMPONENT — this project's claims differ per part, so they are stated per part
-------------------------------------------------------------------------------------------------
| Component | Tier | Basis |
|---|---|---|
| Online softmax identity | **E** | implemented, verified exact against `torch.softmax` |
| Tiled attention (PyTorch) | **E** | implemented, verified against a naive reference, memory scaling measured |
| Triton kernel | **authored, NOT EXECUTED** | no CUDA device on this machine (G-001, D-004) |
| Speedup over standard attention | **NOT CLAIMED** | requires a GPU; no timing is invented |

THE ONE THING TO BE CLEAR ABOUT
-------------------------------
The PyTorch tiled implementation here is **slower than standard attention on CPU**, and that is the
expected, correct outcome rather than a defect. FlashAttention's speed comes from trading extra
arithmetic for less traffic between GPU HBM and SRAM. On a CPU there is no such hierarchy to exploit,
and a Python loop over tiles cannot beat a single well-tuned multithreaded BLAS call.

What *is* demonstrated locally, and is real:

1. **Exactness** — the tiled result matches the reference to float32 reassociation error, for every
   tile size, with and without causal masking, including sequence lengths that do not divide evenly.
   Exactness is what separates FlashAttention from approximate methods like Linformer or Performer, and
   conflating those categories is a serious error.
2. **Memory scaling** — score-matrix storage falls from `O(T²)` to `O(block²)`, with exact byte counts.
3. **Causal work saving** — fully-masked key tiles are skipped, counted, and asserted.
"""

from __future__ import annotations

from .online_softmax import (
    OnlineSoftmaxState,
    online_softmax,
    softmax_state_update,
)
from .tiled import (
    TileStats,
    attention_memory_bytes,
    reference_attention,
    tiled_attention,
)
from .triton_kernel import (
    HAS_TRITON,
    KERNEL_STATUS,
    flash_attention_triton,
    theoretical_analysis,
    verify_against_reference,
)

__all__ = [
    "online_softmax",
    "softmax_state_update",
    "OnlineSoftmaxState",
    "reference_attention",
    "tiled_attention",
    "attention_memory_bytes",
    "TileStats",
    "flash_attention_triton",
    "verify_against_reference",
    "theoretical_analysis",
    "KERNEL_STATUS",
    "HAS_TRITON",
]

PAPERS = {
    "flash_attention": {
        "title": "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness",
        "arxiv": "2205.14135",
        "url": "https://arxiv.org/abs/2205.14135",
        "key_claim_we_reproduce": "exactness and O(block^2) intermediate memory",
        "key_claim_we_do_NOT_reproduce": "wall-clock speedup, which requires a GPU memory hierarchy",
    },
    "flash_attention_2": {
        "title": "FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning",
        "arxiv": "2307.08691",
        "url": "https://arxiv.org/abs/2307.08691",
        "borrowed": "deferring the division by the softmax normaliser until after the inner loop",
    },
    "online_softmax": {
        "title": "Online normalizer calculation for softmax",
        "arxiv": "1805.02867",
        "url": "https://arxiv.org/abs/1805.02867",
        "borrowed": "the single-pass running-maximum/running-sum identity",
    },
}
