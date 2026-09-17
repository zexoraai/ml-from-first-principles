"""Tiled attention with online softmax — FlashAttention's algorithm, in PyTorch.

Primary source
--------------
Dao, Fu, Ermon, Rudra & Ré, "FlashAttention: Fast and Memory-Efficient Exact Attention with
IO-Awareness", arXiv:2205.14135. Algorithm 1.

WHAT FLASHATTENTION ACTUALLY CHANGES
------------------------------------
Standard attention computes, for each head:

    S = Q Kᵀ / √d        (T, T)   ← materialised
    P = softmax(S)       (T, T)   ← materialised
    O = P V              (T, d)

The two intermediate `(T, T)` tensors are the problem. At `T = 4096` with 12 heads in float32, `S`
alone is 12 · 4096² · 4 bytes ≈ **805 MB**, and the backward pass needs `P` as well. Attention's memory
becomes quadratic in sequence length while its *output* is linear.

FlashAttention never forms them. It walks blocks of keys, maintains the softmax normaliser online, and
accumulates the output directly. Memory for the intermediates drops to one tile — `O(block² )` — which
is independent of `T`.

THE CRITICAL POINT ABOUT WHY IT IS FASTER ON A GPU
--------------------------------------------------
Not because it does less arithmetic. It does slightly *more*: the rescaling in the inner loop is extra
work, and the backward pass recomputes `S` rather than storing it.

It is faster because attention at these shapes is **memory-bandwidth bound, not compute bound**. The
standard implementation writes an `(T, T)` matrix to GPU high-bandwidth memory and reads it back at
least twice. FlashAttention keeps the tile in SRAM, which is roughly an order of magnitude faster to
access, and pays for it with redundant FLOPs. Trading arithmetic for data movement is the entire idea,
and it is why the paper's title says "IO-Awareness" rather than "fewer operations".

**This is also precisely why the implementation in this file is SLOWER than standard attention on our
CPU, and that is not a failure.** A Python loop over tiles calling into BLAS per tile has none of the
properties that make the trade pay off: there is no SRAM/HBM hierarchy to exploit, the loop overhead is
interpreted, and one large `torch.matmul` already hits well-tuned multithreaded BLAS. What this
implementation demonstrates locally is **exactness and the memory scaling**, both of which are real and
measurable here. The speed claim belongs to the Triton kernel, which needs a GPU this machine does not
have (G-001), and which therefore ships **authored but not executed** with no invented timings (D-004).

Anyone who reports a CPU speedup for a Python-level FlashAttention has measured something other than
what they think they measured.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

__all__ = ["reference_attention", "tiled_attention", "attention_memory_bytes", "TileStats"]


@dataclass
class TileStats:
    """Bookkeeping so the memory claim is a measurement, not an assertion."""
    n_query_tiles: int = 0
    n_key_tiles: int = 0
    n_tiles_computed: int = 0
    n_tiles_skipped: int = 0
    peak_tile_elements: int = 0
    notes: list[str] = field(default_factory=list)


def reference_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    causal: bool = False,
    scale: float | None = None,
) -> torch.Tensor:
    """Standard attention, materialising the full score matrix. The correctness baseline.

    Args:
        q, k, v: `(B, H, T, d)`.

    Deliberately the naive formulation. Its purpose is to be obviously right, so that any disagreement
    with `tiled_attention` is attributable to the tiled implementation.
    """
    scale = scale if scale is not None else q.shape[-1] ** -0.5
    scores = (q @ k.transpose(-2, -1)) * scale
    if causal:
        t_q, t_k = q.shape[-2], k.shape[-2]
        # Aligned to the BOTTOM-RIGHT, so that a short query block attends to all keys up to its own
        # absolute position. This is what makes the function correct for incremental decoding, where
        # t_q = 1 and t_k is the full cache. Top-left alignment silently breaks that case.
        offset = t_k - t_q
        mask = torch.ones(t_q, t_k, dtype=torch.bool, device=q.device).tril(diagonal=offset)
        scores = scores.masked_fill(~mask, float("-inf"))
    probs = torch.softmax(scores, dim=-1)
    return probs @ v


def tiled_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    causal: bool = False,
    scale: float | None = None,
    q_block: int = 64,
    k_block: int = 64,
    stats: TileStats | None = None,
) -> torch.Tensor:
    """Attention computed in tiles with an online softmax. Never forms the full `(T, T)` matrix.

    Args:
        q, k, v: `(B, H, T, d)`.
        q_block, k_block: tile sizes. On a GPU these are chosen so a tile fits in SRAM; here they only
            trade Python loop overhead against tile memory, and every value must give the same answer.
        stats: optional `TileStats` to fill in, so the memory and skip behaviour can be asserted by
            tests rather than described in prose.

    Returns `(B, H, T, d)`, equal to `reference_attention` up to float32 reassociation.
    """
    if q.shape[:2] != k.shape[:2] or k.shape[:-2] != v.shape[:-2]:
        raise ValueError(f"batch/head dims disagree: q{tuple(q.shape)} k{tuple(k.shape)} v{tuple(v.shape)}")
    if k.shape[-2] != v.shape[-2]:
        raise ValueError(f"k has {k.shape[-2]} positions but v has {v.shape[-2]}")
    if q.shape[-1] != k.shape[-1]:
        raise ValueError(f"q and k head dims differ: {q.shape[-1]} vs {k.shape[-1]}")

    b, h, t_q, d = q.shape
    t_k = k.shape[-2]
    d_v = v.shape[-1]
    scale = scale if scale is not None else d**-0.5
    offset = t_k - t_q                          # bottom-right causal alignment, as in the reference

    out = torch.zeros((b, h, t_q, d_v), device=q.device, dtype=q.dtype)
    if stats is not None:
        stats.n_query_tiles = (t_q + q_block - 1) // q_block
        stats.n_key_tiles = (t_k + k_block - 1) // k_block
        stats.peak_tile_elements = b * h * q_block * k_block

    for qs in range(0, t_q, q_block):
        qe = min(qs + q_block, t_q)
        q_tile = q[:, :, qs:qe, :]                                   # (B, H, bq, d)
        rows = qe - qs

        # Online softmax state for this query tile. O(bq) per (batch, head) -- independent of T, which
        # is the whole point.
        m = torch.full((b, h, rows), float("-inf"), device=q.device, dtype=q.dtype)
        l = torch.zeros((b, h, rows), device=q.device, dtype=q.dtype)
        acc = torch.zeros((b, h, rows, d_v), device=q.device, dtype=q.dtype)

        for ks in range(0, t_k, k_block):
            ke = min(ks + k_block, t_k)

            # Causal skip: if the FIRST query in this tile (absolute position qs + offset) already
            # precedes the first key in this block, no query in the tile can attend to any key in it.
            # Skipping is not an optimisation detail -- it is what makes causal attention cost about
            # half of full attention instead of the same. Without it the algorithm is still correct but
            # does twice the necessary work.
            if causal and ks > qs + offset + rows - 1:
                if stats is not None:
                    stats.n_tiles_skipped += 1
                continue

            k_tile = k[:, :, ks:ke, :]
            v_tile = v[:, :, ks:ke, :]
            scores = (q_tile @ k_tile.transpose(-2, -1)) * scale     # (B, H, bq, bk)

            if causal:
                q_pos = torch.arange(qs, qe, device=q.device).unsqueeze(-1) + offset
                k_pos = torch.arange(ks, ke, device=q.device).unsqueeze(0)
                scores = scores.masked_fill(k_pos > q_pos, float("-inf"))

            if stats is not None:
                stats.n_tiles_computed += 1

            block_max = scores.amax(dim=-1)                          # (B, H, bq)
            m_new = torch.maximum(m, block_max)
            finite = torch.isfinite(m_new)

            # exp(m - m_new) is always <= 1, so it can underflow to 0 but never overflow. When a row
            # has seen nothing and this block is fully masked, m_new stays -inf and (-inf) - (-inf)
            # is nan; the `where` on `finite` keeps that row at l=0, which normalise() reads as
            # "no valid keys" and turns into a zero output rather than nan.
            correction = torch.where(finite, torch.exp(m - m_new), torch.zeros_like(m_new))
            probs = torch.where(finite.unsqueeze(-1),
                                torch.exp(scores - m_new.unsqueeze(-1)),
                                torch.zeros_like(scores))

            l = l * correction + probs.sum(dim=-1)
            # Rescale the accumulator by the same scalar per row, then add this block's contribution.
            # The division by `l` is deferred to after the loop (FlashAttention-2), so the inner loop
            # touches the (bq, d) accumulator once instead of twice.
            acc = acc * correction.unsqueeze(-1) + probs @ v_tile
            m = m_new

        safe_l = torch.where(l > 0, l, torch.ones_like(l))
        tile_out = acc / safe_l.unsqueeze(-1)
        out[:, :, qs:qe, :] = torch.where((l > 0).unsqueeze(-1), tile_out,
                                          torch.zeros_like(tile_out))

    if stats is not None and causal and stats.n_tiles_skipped:
        stats.notes.append(
            f"skipped {stats.n_tiles_skipped} of "
            f"{stats.n_query_tiles * stats.n_key_tiles} key tiles as fully masked"
        )
    return out


def attention_memory_bytes(
    *, batch: int, heads: int, t_q: int, t_k: int, head_dim: int, q_block: int, k_block: int,
    bytes_per_element: int = 4,
) -> dict:
    """Exact byte counts for the intermediates each method holds. Analytic, not measured.

    Counted here is only the *score/probability* storage, because that is the term that differs. `Q`,
    `K`, `V` and the output are identical in both methods and are reported separately so the ratio
    cannot be inflated by omitting them.

    The reference figure counts `S` once. A backward pass through the naive formulation needs the
    probabilities too, so the practical peak is roughly double — noted rather than folded in, to keep
    the number one that can be checked by hand.
    """
    ref_scores = batch * heads * t_q * t_k * bytes_per_element
    tile_scores = batch * heads * q_block * k_block * bytes_per_element
    qkv = (batch * heads * t_q * head_dim + 2 * batch * heads * t_k * head_dim) * bytes_per_element
    out = batch * heads * t_q * head_dim * bytes_per_element
    return {
        "reference_score_matrix_bytes": ref_scores,
        "tiled_peak_tile_bytes": tile_scores,
        "reduction_factor": ref_scores / tile_scores if tile_scores else None,
        "qkv_bytes": qkv,
        "output_bytes": out,
        "reference_total_bytes": ref_scores + qkv + out,
        "tiled_total_bytes": tile_scores + qkv + out,
        "note": (
            "Score-matrix storage only for the two *_score_* figures; Q/K/V and the output are "
            "identical in both methods and are listed separately so the ratio is not inflated by "
            "omitting them. A backward pass through the naive formulation also needs the probability "
            "matrix, so its practical peak is about double reference_score_matrix_bytes. Analytic "
            "byte counts, not measured allocator peaks — on CPU the allocator is shared with the "
            "container and a measured peak would be dominated by unrelated load (G-008)."
        ),
    }
