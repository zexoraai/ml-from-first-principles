"""FlashAttention forward pass as a Triton kernel.

╔══════════════════════════════════════════════════════════════════════════════════════════════════╗
║  THIS KERNEL HAS NEVER BEEN EXECUTED.                                                            ║
║                                                                                                  ║
║  The development machine has no CUDA device (GAPS G-001: AMD Ryzen 5 PRO 5650U with integrated    ║
║  Radeon graphics). Triton requires one. Per decision D-004 this code is published as **authored,  ║
║  not executed**: it has not been compiled, not been run, not been verified against the reference, ║
║  and produces no timing whatsoever.                                                              ║
║                                                                                                  ║
║  There are NO benchmark numbers for this kernel anywhere in this repository, and there will be    ║
║  none until it runs on real hardware. Any speedup figure attached to it would be fabricated.      ║
║                                                                                                  ║
║  `verify_against_reference()` at the bottom is the acceptance test, written in advance. It is the ║
║  first thing to run when a GPU becomes available. Expect it to fail initially — unrun kernels     ║
║  essentially always have at least one bug, and claiming otherwise would be the same dishonesty in ║
║  a different costume.                                                                            ║
╚══════════════════════════════════════════════════════════════════════════════════════════════════╝

Primary sources
---------------
* Dao et al., "FlashAttention", arXiv:2205.14135 — Algorithm 1, which this implements.
* Dao, "FlashAttention-2", arXiv:2307.08691 — parallelising over query blocks rather than batch·head
  alone, and deferring the normaliser division.
* Tillet, Kung & Cox, "Triton: an intermediate language and compiler for tiled neural network
  computations", MAPL 2019.

WHY THIS NEEDS TO BE A KERNEL AT ALL
------------------------------------
`tiled.py` implements the identical algorithm in PyTorch and is *correct*. It is also slower than
standard attention, because every tile boundary is a separate kernel launch and a round trip through
GPU global memory — exactly the traffic FlashAttention exists to eliminate. The algorithm's benefit is
only realised when the whole inner loop stays resident in on-chip SRAM, and expressing that requires
writing at the tile level. That is what Triton is for.

THE MEMORY HIERARCHY, WHICH IS THE ENTIRE POINT
-----------------------------------------------
On an A100: HBM is ~40–80 GB at ~1.5–2.0 TB/s; SRAM is ~20 MB at ~19 TB/s — an order of magnitude
faster. Standard attention writes an (T, T) score matrix to HBM and reads it back. FlashAttention keeps
tiles in SRAM and never writes the score matrix at all, paying redundant arithmetic for the privilege.
Because attention at these shapes is bandwidth-bound rather than compute-bound, that trade wins.

This is also the reason the block sizes below are *tunable constants*: they must be chosen so that
`Q_tile + K_tile + V_tile + accumulator` fits in a streaming multiprocessor's shared memory. Too large
and the kernel fails to launch or spills; too small and launch overhead dominates. The correct values
are hardware-specific and must be found by autotuning **on the target device** — which is another thing
that cannot be done here.
"""

from __future__ import annotations

import math

import torch

__all__ = ["HAS_TRITON", "flash_attention_triton", "verify_against_reference", "KERNEL_STATUS"]

KERNEL_STATUS = {
    "authored": True,
    "compiled": False,
    "executed": False,
    "verified_against_reference": False,
    "benchmarked": False,
    "reason": "no CUDA device on the development machine (GAPS G-001); Triton requires one",
    "decision": "D-004 — Triton authored locally, executed externally",
    "acceptance_test": "labs.p5_attention.triton_kernel.verify_against_reference",
    "expectation_when_first_run": (
        "failure is more likely than success. An unrun kernel of this length almost always has at "
        "least one indexing or masking bug. The acceptance test exists to find them, not to rubber-"
        "stamp the code."
    ),
}

try:
    import triton
    import triton.language as tl

    HAS_TRITON = True
except ImportError:                                     # pragma: no cover - expected on this machine
    HAS_TRITON = False

    # Stubs so the module imports and can be inspected, documented and unit-tested for its guard
    # behaviour on a machine with no Triton. Importing this module must never raise.
    class _TritonStub:
        def jit(self, fn):
            return fn

        def __getattr__(self, name):
            raise RuntimeError(
                f"triton.{name} is unavailable: Triton is not installed and this machine has no CUDA "
                f"device. See KERNEL_STATUS."
            )

    triton = _TritonStub()                              # type: ignore[assignment]
    tl = _TritonStub()                                  # type: ignore[assignment]


if HAS_TRITON:                                          # pragma: no cover - never taken here

    @triton.jit
    def _flash_attention_fwd_kernel(
        Q, K, V, Out,
        stride_qb, stride_qh, stride_qm, stride_qd,
        stride_kb, stride_kh, stride_kn, stride_kd,
        stride_vb, stride_vh, stride_vn, stride_vd,
        stride_ob, stride_oh, stride_om, stride_od,
        n_heads, seq_q, seq_k,
        scale,
        BLOCK_M: tl.constexpr,       # query positions per program
        BLOCK_N: tl.constexpr,       # key positions per inner-loop step
        HEAD_DIM: tl.constexpr,
        IS_CAUSAL: tl.constexpr,
    ):
        """One program computes one (BLOCK_M x HEAD_DIM) tile of the output.

        Parallelisation follows FlashAttention-2: the grid is (query blocks, batch·heads), so query
        blocks run concurrently. FlashAttention-1 parallelised over batch·heads only, which starves a
        large GPU when batch·heads is small — a long-context, small-batch workload, which is precisely
        the case people reach for FlashAttention to serve.
        """
        pid_m = tl.program_id(0)
        pid_bh = tl.program_id(1)
        batch = pid_bh // n_heads
        head = pid_bh % n_heads

        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)      # query positions
        offs_d = tl.arange(0, HEAD_DIM)

        q_base = Q + batch * stride_qb + head * stride_qh
        k_base = K + batch * stride_kb + head * stride_kh
        v_base = V + batch * stride_vb + head * stride_vh

        # ---- load the query tile once; it stays resident for the whole inner loop ----------------
        q_ptrs = q_base + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qd
        q_mask = offs_m[:, None] < seq_q
        q_tile = tl.load(q_ptrs, mask=q_mask, other=0.0)

        # ---- online softmax state, kept in registers ---------------------------------------------
        # float32 regardless of the input dtype. Accumulating a softmax normaliser in fp16 loses
        # precision fast, and the whole selling point of this kernel is that it is EXACT.
        m_i = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
        l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
        acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)

        # Bottom-right causal alignment, matching reference_attention in tiled.py, so the kernel is
        # correct for incremental decoding where seq_q = 1 and seq_k is the full cache.
        offset = seq_k - seq_q

        # With causal masking, keys beyond the last query in this tile can never contribute, so the
        # loop is bounded rather than running to seq_k. This is where the ~2x saving comes from.
        if IS_CAUSAL:
            hi = tl.minimum(seq_k, pid_m * BLOCK_M + BLOCK_M + offset)
        else:
            hi = seq_k

        for start_n in range(0, hi, BLOCK_N):
            offs_n = start_n + tl.arange(0, BLOCK_N)

            k_ptrs = k_base + offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kd
            v_ptrs = v_base + offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vd
            kv_mask = offs_n[:, None] < seq_k
            k_tile = tl.load(k_ptrs, mask=kv_mask, other=0.0)
            v_tile = tl.load(v_ptrs, mask=kv_mask, other=0.0)

            scores = tl.dot(q_tile, tl.trans(k_tile)) * scale

            # Out-of-range keys must be -inf, not 0. Zero is a perfectly plausible score and would be
            # given real probability mass; -inf makes exp() give exactly 0.
            scores = tl.where(offs_n[None, :] < seq_k, scores, float("-inf"))
            if IS_CAUSAL:
                scores = tl.where(offs_n[None, :] <= (offs_m[:, None] + offset),
                                  scores, float("-inf"))

            # ---- the online softmax update, identical to tiled.py -------------------------------
            m_new = tl.maximum(m_i, tl.max(scores, 1))
            # exp(m_i - m_new) <= 1 always: it may underflow to 0, it can never overflow.
            correction = tl.exp(m_i - m_new)
            p = tl.exp(scores - m_new[:, None])

            l_i = l_i * correction + tl.sum(p, 1)
            acc = acc * correction[:, None] + tl.dot(p.to(v_tile.dtype), v_tile)
            m_i = m_new

        # ---- deferred division (FlashAttention-2) ------------------------------------------------
        # Dividing once here rather than inside the loop removes a full rescale of `acc` per key block.
        # l_i == 0 means the row saw no unmasked key; emit zeros rather than nan.
        l_safe = tl.where(l_i > 0, l_i, 1.0)
        acc = acc / l_safe[:, None]
        acc = tl.where((l_i > 0)[:, None], acc, 0.0)

        o_ptrs = (Out + batch * stride_ob + head * stride_oh
                  + offs_m[:, None] * stride_om + offs_d[None, :] * stride_od)
        tl.store(o_ptrs, acc.to(Out.dtype.element_ty), mask=offs_m[:, None] < seq_q)


def flash_attention_triton(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    causal: bool = False,
    scale: float | None = None,
    block_m: int = 64,
    block_n: int = 64,
) -> torch.Tensor:
    """Launch the Triton kernel. Raises a clear error when it cannot run.

    The guard is deliberately loud. A silent fallback to `tiled_attention` would be worse than an
    exception: benchmark numbers would be attributed to a kernel that never ran, which is the exact
    dishonesty this project is structured to avoid.
    """
    if not HAS_TRITON:
        raise RuntimeError(
            "Triton is not installed. This kernel has never been executed — see KERNEL_STATUS. "
            "Use labs.p5_attention.tiled_attention for the verified PyTorch implementation of the "
            "same algorithm."
        )
    if not q.is_cuda:
        raise RuntimeError(
            f"Triton kernels require CUDA tensors; got device {q.device}. This machine has no CUDA "
            f"device (GAPS G-001), which is why this kernel ships unexecuted."
        )
    if q.shape[-1] not in (16, 32, 64, 128):
        raise ValueError(
            f"head_dim must be a power of two in [16, 128] for tl.dot; got {q.shape[-1]}. This is a "
            f"Triton tiling constraint, not a mathematical one."
        )

    b, h, t_q, d = q.shape
    t_k = k.shape[-2]
    scale = scale if scale is not None else d**-0.5
    out = torch.empty_like(q)

    grid = (triton.cdiv(t_q, block_m), b * h)
    _flash_attention_fwd_kernel[grid](
        q, k, v, out,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        h, t_q, t_k,
        scale,
        BLOCK_M=block_m, BLOCK_N=block_n, HEAD_DIM=d, IS_CAUSAL=causal,
    )
    return out


def verify_against_reference(*, device: str = "cuda", atol: float = 1e-2,
                             verbose: bool = True) -> dict:
    """The acceptance test, written before the kernel has ever run.

    Run this FIRST on any GPU-equipped machine, before any benchmark. Its job is to find the bugs an
    unrun kernel is expected to contain: mask alignment, ragged tails, the single-query decode shape,
    and the fully-masked row.

    `atol` defaults to 1e-2 rather than the 1e-5 used for the PyTorch path because `tl.dot` may use
    TF32 or fp16 accumulation depending on device and dtype. A tolerance that loose is not a
    correctness claim on its own — it must be tightened once the actual numerics on the target device
    are known, and the tightened value published.

    Returns a dict of per-case results. **Does not raise on mismatch**: the point is to report every
    failing case at once rather than stopping at the first.
    """
    from .tiled import reference_attention

    if not torch.cuda.is_available():
        return {"ran": False, "reason": "no CUDA device", "status": KERNEL_STATUS}

    cases = [
        # (name, b, h, t_q, t_k, d, causal) — chosen to cover the failure modes, not to look tidy.
        ("square_noncausal", 2, 4, 128, 128, 64, False),
        ("square_causal", 2, 4, 128, 128, 64, True),
        ("ragged_causal", 1, 2, 100, 100, 64, True),       # t not divisible by block_m
        ("single_query_decode", 1, 2, 1, 256, 64, True),   # catches top-left vs bottom-right masking
        ("short_seq", 1, 1, 7, 7, 16, True),               # sequence shorter than one tile
        ("wide_head_dim", 1, 2, 64, 64, 128, True),
        ("cross_attention", 1, 2, 64, 192, 64, False),     # t_q != t_k, non-causal
    ]

    results = {}
    for name, b, h, t_q, t_k, d, causal in cases:
        torch.manual_seed(0)
        q = torch.randn(b, h, t_q, d, device=device, dtype=torch.float16)
        k = torch.randn(b, h, t_k, d, device=device, dtype=torch.float16)
        v = torch.randn(b, h, t_k, d, device=device, dtype=torch.float16)
        try:
            got = flash_attention_triton(q, k, v, causal=causal)
            want = reference_attention(q.float(), k.float(), v.float(), causal=causal)
            diff = (got.float() - want).abs().max().item()
            results[name] = {"ok": diff <= atol, "max_abs_diff": diff,
                             "shape": [b, h, t_q, t_k, d], "causal": causal}
        except Exception as exc:                                       # noqa: BLE001
            results[name] = {"ok": False, "error": f"{type(exc).__name__}: {exc}",
                             "shape": [b, h, t_q, t_k, d], "causal": causal}
        if verbose:
            r = results[name]
            mark = "PASS" if r.get("ok") else "FAIL"
            detail = r.get("error", f"max|diff| {r.get('max_abs_diff', float('nan')):.2e}")
            print(f"  [{mark}] {name:<22} {detail}")

    n_ok = sum(1 for r in results.values() if r.get("ok"))
    summary = {"ran": True, "device": torch.cuda.get_device_name(0), "atol": atol,
               "n_cases": len(cases), "n_passed": n_ok, "cases": results,
               "all_passed": n_ok == len(cases)}
    if verbose:
        print(f"\n{n_ok}/{len(cases)} cases passed on {summary['device']}")
        if not summary["all_passed"]:
            print("Kernel is NOT verified. Do not benchmark it and do not publish any timing.")
    return summary


def theoretical_analysis(*, batch: int, heads: int, seq: int, head_dim: int,
                         block_m: int = 64, block_n: int = 64,
                         bytes_per_element: int = 2) -> dict:
    """Analytic FLOP and HBM-traffic comparison. Arithmetic only — no measurement, no device needed.

    Included so P5 has a quantitative story even without a GPU, and clearly labelled as analysis. The
    ratio it computes is the *reason* FlashAttention wins; it is not evidence that this kernel does.
    """
    # Attention FLOPs: QK^T is 2·T²·d, PV is another 2·T²·d. Softmax is O(T²) and negligible beside them.
    flops = 4 * batch * heads * seq * seq * head_dim

    qkv_traffic = 3 * batch * heads * seq * head_dim * bytes_per_element
    out_traffic = batch * heads * seq * head_dim * bytes_per_element
    # Standard attention additionally writes S, reads it for softmax, writes P, reads P for PV.
    score_traffic = 4 * batch * heads * seq * seq * bytes_per_element

    standard_bytes = qkv_traffic + out_traffic + score_traffic
    # FlashAttention re-reads K and V once per query block, and never touches the score matrix.
    n_q_blocks = math.ceil(seq / block_m)
    flash_bytes = (batch * heads * seq * head_dim * bytes_per_element          # Q, once
                   + 2 * batch * heads * seq * head_dim * bytes_per_element * n_q_blocks
                   + out_traffic)

    return {
        "flops": flops,
        "standard_hbm_bytes": standard_bytes,
        "flash_hbm_bytes": flash_bytes,
        "traffic_reduction_factor": standard_bytes / flash_bytes,
        "standard_arithmetic_intensity_flops_per_byte": flops / standard_bytes,
        "flash_arithmetic_intensity_flops_per_byte": flops / flash_bytes,
        "n_query_blocks": n_q_blocks,
        "note": (
            "Analytic model, not a measurement. FLOPs are essentially identical between the two "
            "methods — FlashAttention does slightly MORE arithmetic. The difference is HBM traffic, "
            "which is why the comparison is framed as arithmetic intensity. FlashAttention's traffic "
            "GROWS with the number of query blocks because K and V are re-read per block: the win is "
            "not unconditional, and at very small sequence lengths with many query blocks standard "
            "attention moves less data."
        ),
        "why_the_ratio_may_look_suspiciously_flat": (
            "The dominant terms are 4·B·H·T²·e for standard attention (write S, read S, write P, read "
            "P) against 2·B·H·T²·d·e/block_m for FlashAttention (re-read K and V once per query "
            "block). Their ratio is 2·block_m/d — independent of T. So whenever block_m == d the model "
            "reports almost exactly 2.00x at every sequence length, which is arithmetic, not a bug. "
            "Vary block_m or head_dim and the ratio moves. This also shows the model's limit: it "
            "predicts a T-independent constant, whereas measured FlashAttention speedups on real GPUs "
            "do grow with T. The gap is everything this first-order account omits — L2 cache reuse, "
            "occupancy, the backward pass's recomputation, and the fact that a large S matrix stops "
            "fitting in any cache at all. Treat the 2x as a floor on the traffic argument, not as a "
            "predicted speedup."
        ),
        "ratio_identity": "2 * block_m / head_dim (for the T² terms)",
    }
