"""Tests for the Triton module on a machine that cannot run Triton.

The kernel itself cannot be tested here — that is the whole point of D-004. What *can* and must be
tested is everything around it:

* the module **imports cleanly** with no Triton and no CUDA, so the rest of the project is not held
  hostage by an unavailable dependency;
* the launch path **fails loudly** rather than silently falling back, because a silent fallback would
  let benchmark numbers be attributed to a kernel that never ran;
* the honesty metadata is present and says "not executed", so it cannot be quietly dropped later;
* the analytic model is arithmetically sound.
"""

from __future__ import annotations

import pytest
import torch

from labs.p5_attention.triton_kernel import (
    HAS_TRITON,
    KERNEL_STATUS,
    flash_attention_triton,
    theoretical_analysis,
    verify_against_reference,
)


# =============================================================================================
# import safety and honest failure
# =============================================================================================

def test_module_imports_without_triton_or_cuda() -> None:
    """Importing must never raise, so the module can be inspected and documented anywhere."""
    assert isinstance(HAS_TRITON, bool)


def test_launch_raises_clearly_instead_of_falling_back_silently() -> None:
    """A silent fallback would let timings be credited to a kernel that never executed."""
    q = torch.randn(1, 1, 16, 16)
    with pytest.raises(RuntimeError) as exc:
        flash_attention_triton(q, q.clone(), q.clone())
    message = str(exc.value)
    assert "never been executed" in message or "require CUDA" in message
    # It must also point the user at the implementation that does work.
    if not HAS_TRITON:
        assert "tiled_attention" in message


def test_verify_against_reference_reports_that_it_could_not_run() -> None:
    """The acceptance test must return a truthful "did not run" rather than a vacuous pass."""
    result = verify_against_reference(verbose=False)
    if not torch.cuda.is_available():
        assert result["ran"] is False
        assert result["reason"] == "no CUDA device"
        assert result.get("all_passed") is not True, "must not claim success without running"


# =============================================================================================
# the honesty metadata is a deliverable, so a test keeps it from being dropped
# =============================================================================================

def test_kernel_status_declares_it_unexecuted_and_unverified() -> None:
    assert KERNEL_STATUS["authored"] is True
    assert KERNEL_STATUS["executed"] is False
    assert KERNEL_STATUS["compiled"] is False
    assert KERNEL_STATUS["verified_against_reference"] is False
    assert KERNEL_STATUS["benchmarked"] is False


def test_kernel_status_cites_the_gap_and_the_decision() -> None:
    assert "G-001" in KERNEL_STATUS["reason"]
    assert "D-004" in KERNEL_STATUS["decision"]


def test_kernel_status_expects_failure_on_first_run() -> None:
    """Stating this in advance is what stops a first-run pass being treated as vindication."""
    assert "failure is more likely than success" in KERNEL_STATUS["expectation_when_first_run"]


def test_status_flags_cannot_be_true_while_the_kernel_is_unexecuted() -> None:
    """Guards against a later edit flipping `benchmarked` without actually benchmarking."""
    if not KERNEL_STATUS["executed"]:
        assert not KERNEL_STATUS["benchmarked"]
        assert not KERNEL_STATUS["verified_against_reference"]


# =============================================================================================
# the analytic model
# =============================================================================================

def test_flop_count_matches_the_hand_derivation() -> None:
    """QKᵀ costs 2·T²·d and PV another 2·T²·d, so 4·B·H·T²·d in total."""
    a = theoretical_analysis(batch=1, heads=1, seq=1024, head_dim=64)
    assert a["flops"] == 4 * 1024 * 1024 * 64


def test_flash_moves_less_memory_than_standard_at_long_sequence() -> None:
    a = theoretical_analysis(batch=1, heads=12, seq=4096, head_dim=64)
    assert a["flash_hbm_bytes"] < a["standard_hbm_bytes"]
    assert a["traffic_reduction_factor"] > 1.0


def test_flash_arithmetic_intensity_is_higher() -> None:
    """The mechanism of the speedup: more FLOPs per byte moved, on a bandwidth-bound problem."""
    a = theoretical_analysis(batch=1, heads=12, seq=4096, head_dim=64)
    assert (a["flash_arithmetic_intensity_flops_per_byte"]
            > a["standard_arithmetic_intensity_flops_per_byte"])


def test_the_traffic_advantage_grows_with_sequence_length() -> None:
    """Standard attention's traffic is O(T²) while FlashAttention's is O(T²/block) — so the ratio
    must widen as T grows. If it ever narrowed, the model would be wrong."""
    factors = [theoretical_analysis(batch=1, heads=8, seq=t, head_dim=64)["traffic_reduction_factor"]
               for t in (512, 1024, 2048, 4096, 8192)]
    assert factors == sorted(factors), factors


def test_the_advantage_is_not_unconditional_at_short_sequence() -> None:
    """An honest model must be able to show the method losing.

    At very short sequences the score matrix is small and FlashAttention's re-reading of K and V per
    query block can move MORE data than standard attention. Asserting this keeps the analysis from
    being a one-sided advertisement.
    """
    short = theoretical_analysis(batch=1, heads=1, seq=64, head_dim=128, block_m=8)
    assert short["traffic_reduction_factor"] < 1.0, (
        "expected FlashAttention to lose at tiny sequence length with many query blocks; "
        f"got {short['traffic_reduction_factor']:.3f}"
    )


def test_analysis_states_that_flops_are_not_the_saving() -> None:
    """The single most common misreading of the paper, so the note is asserted."""
    note = theoretical_analysis(batch=1, heads=1, seq=512, head_dim=64)["note"]
    assert "MORE arithmetic" in note
    assert "not unconditional" in note


def test_the_flat_ratio_is_explained_by_block_m_over_head_dim() -> None:
    """The T² terms give a ratio of 2·block_m/head_dim, independent of T.

    This is why the model reports almost exactly 2.00x at every sequence length when
    `block_m == head_dim`, which is the default configuration and looks suspicious on a results table.
    Asserting the identity here means the explanation on the page is checked, not just claimed.
    """
    for block_m, head_dim, expected in ((64, 64, 2.0), (128, 64, 4.0), (32, 64, 1.0),
                                        (64, 128, 1.0)):
        a = theoretical_analysis(batch=1, heads=12, seq=16384, head_dim=head_dim, block_m=block_m)
        # At large T the T² terms dominate, so the measured ratio approaches the identity.
        assert a["traffic_reduction_factor"] == pytest.approx(expected, rel=0.02), (
            f"block_m={block_m} head_dim={head_dim}: expected ~{expected}, "
            f"got {a['traffic_reduction_factor']:.3f}"
        )


def test_ratio_identity_is_documented_in_the_output() -> None:
    a = theoretical_analysis(batch=1, heads=1, seq=1024, head_dim=64)
    assert a["ratio_identity"] == "2 * block_m / head_dim (for the T² terms)"
    assert "not as a predicted speedup" in a["why_the_ratio_may_look_suspiciously_flat"]
