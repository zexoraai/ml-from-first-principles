"""Correctness suite for LoRA.

The three properties that make LoRA LoRA, each tested directly:

1. **An untrained adapter is exactly the identity.** `B = 0` so `BA = 0`. This is not a nicety — it
   is what lets fine-tuning start from the pretrained function instead of a perturbed one, and it is
   a free end-to-end check that the wiring is right.
2. **The base weights never move.** If they do, it is not LoRA, and the memory and parameter claims
   are false.
3. **Merging is equivalent.** `W0 + (α/r)BA` applied once must equal the two-matmul path applied
   every forward. Without this, "merged deployment costs nothing extra" is an unverified claim.
"""

from __future__ import annotations

import math

import pytest
import torch
import torch.nn as nn

from labs.p3_lora.lora import (
    LoRALinear,
    apply_lora,
    count_parameters,
    lora_state_dict,
    mark_only_lora_trainable,
    merge_all,
    unmerge_all,
)


def tiny_model(d: int = 32) -> nn.Module:
    """A stand-in with the same attribute names P2's attention uses, so `apply_lora` targets it."""
    class Attn(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.w_q = nn.Linear(d, d, bias=False)
            self.w_k = nn.Linear(d, d, bias=False)
            self.w_v = nn.Linear(d, d, bias=False)
            self.w_o = nn.Linear(d, d, bias=False)

        def forward(self, x):
            return self.w_o(self.w_q(x) + self.w_k(x) + self.w_v(x))

    class Block(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.attn = Attn()
            self.mlp = nn.Linear(d, d)

        def forward(self, x):
            return self.mlp(self.attn(x))

    class Model(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.blocks = nn.ModuleList([Block() for _ in range(2)])

        def forward(self, x):
            for b in self.blocks:
                x = b(x)
            return x

    torch.manual_seed(0)
    return Model()


# --------------------------------------------------------------------------------------------
# 1. the untrained adapter is a no-op
# --------------------------------------------------------------------------------------------

def test_untrained_adapter_is_exactly_the_identity() -> None:
    """B = 0 => BA = 0 => the adapted layer equals the base layer, bit for bit.

    Exact equality is asserted, not approximate: adding a tensor of exact zeros changes nothing at
    all in floating point.
    """
    torch.manual_seed(0)
    base = nn.Linear(32, 48)
    x = torch.randn(4, 32)
    with torch.no_grad():
        expected = base(x)
        wrapped = LoRALinear(base, r=8, alpha=16)
        assert torch.equal(wrapped(x), expected)


def test_b_is_zero_and_a_is_not_at_initialisation() -> None:
    """Exactly one factor must be zero.

    Both zero and neither factor ever receives gradient (the product rule gives ∂(BA)/∂A ∝ B and
    ∂(BA)/∂B ∝ A), so nothing learns. Both nonzero and the pretrained model is perturbed before
    training starts.
    """
    layer = LoRALinear(nn.Linear(16, 16), r=4)
    assert torch.equal(layer.lora_B, torch.zeros_like(layer.lora_B))
    assert layer.lora_A.abs().sum() > 0


def test_zero_rank_disables_lora_entirely() -> None:
    """r = 0 is the control arm: the model must be untouched and have no adapter parameters."""
    torch.manual_seed(0)
    base = nn.Linear(16, 16)
    x = torch.randn(2, 16)
    layer = LoRALinear(base, r=0)
    with torch.no_grad():
        assert torch.equal(layer(x), base(x))
    assert layer.lora_A is None and layer.lora_B is None
    assert layer.scaling == 0.0
    assert torch.equal(layer.delta_weight(), torch.zeros_like(base.weight))


# --------------------------------------------------------------------------------------------
# 2. the base is frozen
# --------------------------------------------------------------------------------------------

def test_base_weights_are_frozen_and_receive_no_gradient() -> None:
    layer = LoRALinear(nn.Linear(16, 16, bias=True), r=4)
    assert layer.base.weight.requires_grad is False
    assert layer.base.bias.requires_grad is False
    assert layer.lora_A.requires_grad and layer.lora_B.requires_grad

    layer(torch.randn(3, 16)).pow(2).mean().backward()
    assert layer.base.weight.grad is None, "a frozen weight must not accumulate gradient"
    assert layer.lora_A.grad is not None
    assert layer.lora_B.grad is not None


def test_training_the_adapter_leaves_the_base_weights_untouched() -> None:
    """The claim that matters. If the base moves, the parameter and memory story is false."""
    torch.manual_seed(0)
    layer = LoRALinear(nn.Linear(24, 24), r=6, alpha=12)
    before = layer.base.weight.detach().clone()
    opt = torch.optim.AdamW([p for p in layer.parameters() if p.requires_grad], lr=1e-2)

    x = torch.randn(8, 24)
    target = torch.randn(8, 24)
    for _ in range(40):
        opt.zero_grad(set_to_none=True)
        nn.functional.mse_loss(layer(x), target).backward()
        opt.step()

    assert torch.equal(layer.base.weight, before)
    assert layer.lora_B.abs().sum() > 0, "B must have moved away from zero, or nothing trained"


def test_adapter_actually_changes_the_output_after_training() -> None:
    """Guards against a no-op adapter passing every other test by doing nothing."""
    torch.manual_seed(0)
    layer = LoRALinear(nn.Linear(24, 24), r=6)
    x = torch.randn(4, 24)
    with torch.no_grad():
        before = layer(x).clone()

    opt = torch.optim.AdamW([p for p in layer.parameters() if p.requires_grad], lr=5e-2)
    target = torch.randn(4, 24)
    for _ in range(60):
        opt.zero_grad(set_to_none=True)
        nn.functional.mse_loss(layer(x), target).backward()
        opt.step()

    with torch.no_grad():
        assert not torch.allclose(layer(x), before, atol=1e-4)


# --------------------------------------------------------------------------------------------
# 3. merge equivalence
# --------------------------------------------------------------------------------------------

def test_merged_output_equals_unmerged_output() -> None:
    """`W0 + (α/r)BA` once must equal the two-matmul path every forward.

    This is what licenses "merged deployment costs exactly what the base cost". Tolerance rather than
    equality because the two paths sum in a different order.
    """
    torch.manual_seed(0)
    layer = LoRALinear(nn.Linear(32, 32), r=8, alpha=16).eval()
    with torch.no_grad():
        layer.lora_B.normal_(0, 0.05)          # make the adapter non-trivial
        x = torch.randn(6, 32)
        unmerged = layer(x).clone()
        layer.merge()
        merged = layer(x)

    assert layer.merged
    assert torch.allclose(unmerged, merged, atol=1e-5), (unmerged - merged).abs().max().item()


def test_merge_then_unmerge_restores_the_base_weight() -> None:
    """Round-trip is asserted to a tolerance, not to equality — (w+d)-d != w in floating point."""
    torch.manual_seed(0)
    layer = LoRALinear(nn.Linear(32, 32), r=8, alpha=16)
    with torch.no_grad():
        layer.lora_B.normal_(0, 0.05)
    original = layer.base.weight.detach().clone()

    layer.merge()
    assert not torch.allclose(layer.base.weight, original, atol=1e-6), "merge must change the weight"
    layer.unmerge()

    assert not layer.merged
    assert torch.allclose(layer.base.weight, original, atol=1e-6), (
        (layer.base.weight - original).abs().max().item()
    )


def test_merge_is_idempotent_and_unmerge_on_unmerged_is_a_noop() -> None:
    """A live demo will let someone press the button twice."""
    torch.manual_seed(0)
    layer = LoRALinear(nn.Linear(16, 16), r=4)
    with torch.no_grad():
        layer.lora_B.normal_(0, 0.05)

    layer.merge()
    after_one = layer.base.weight.detach().clone()
    layer.merge()
    assert torch.equal(layer.base.weight, after_one), "merging twice must not apply ΔW twice"

    layer.unmerge()
    once = layer.base.weight.detach().clone()
    layer.unmerge()
    assert torch.equal(layer.base.weight, once)


def test_delta_weight_has_the_shape_of_the_base_weight() -> None:
    layer = LoRALinear(nn.Linear(20, 36), r=5)
    assert layer.delta_weight().shape == layer.base.weight.shape == (36, 20)


@pytest.mark.parametrize("r", [1, 2, 4, 8])
def test_delta_weight_rank_is_exactly_r(r: int) -> None:
    """The defining constraint: `BA` with inner dimension r has rank at most r.

    Numerical rank needs a **relative** threshold, not an absolute one. The first version of this
    test used `> 1e-5` and failed at r=4, reporting rank 5 — because the true singular values were
    `[202.6, 163.7, 127.8, 93.8, 1.5e-5, 8.7e-6, ...]`. The fifth value is float32 round-off from the
    SVD of a rank-4 matrix whose leading value is ~200, and it happened to land just above a fixed
    1e-5 cut. The matrix was exactly right; the test was wrong.

    The standard definition of numerical rank is the count of singular values exceeding
    `σ_max · max(m, n) · ε`, which scales with the magnitude of the matrix instead of assuming it.
    With that, the answer is r for every rank, and the test also catches the opposite failure —
    a ΔW whose rank is *less* than r, which would mean the adapter is not using its capacity.
    """
    torch.manual_seed(0)
    d_in, d_out = 32, 40
    layer = LoRALinear(nn.Linear(d_in, d_out), r=r)
    with torch.no_grad():
        layer.lora_A.normal_()
        layer.lora_B.normal_()

    delta = layer.delta_weight()
    singular = torch.linalg.svdvals(delta)
    tol = singular[0].item() * max(d_in, d_out) * torch.finfo(delta.dtype).eps
    numerical_rank = int((singular > tol).sum())

    assert numerical_rank == r, (
        f"expected rank {r}, got {numerical_rank}; tol={tol:.3e}, "
        f"leading singular values {[round(v, 4) for v in singular[:r + 2].tolist()]}"
    )


# --------------------------------------------------------------------------------------------
# scaling
# --------------------------------------------------------------------------------------------

def test_scaling_is_alpha_over_r() -> None:
    assert LoRALinear(nn.Linear(8, 8), r=8, alpha=16).scaling == pytest.approx(2.0)
    assert LoRALinear(nn.Linear(8, 8), r=16, alpha=16).scaling == pytest.approx(1.0)
    assert LoRALinear(nn.Linear(8, 8), r=4, alpha=8).scaling == pytest.approx(2.0)


def test_alpha_over_r_keeps_update_magnitude_comparable_across_ranks() -> None:
    """The reason for dividing by r (§4.1).

    With identical per-element statistics in A and B, the raw product `BA` grows with r because it
    sums r rank-one terms. Dividing by r compensates, so a learning rate tuned at one rank stays
    sensible at another. Without it, rank and learning rate become entangled.
    """
    torch.manual_seed(0)
    d = 64
    norms = {}
    for r in (2, 8, 32):
        layer = LoRALinear(nn.Linear(d, d), r=r, alpha=float(r))   # alpha = r => scaling = 1
        with torch.no_grad():
            layer.lora_A.normal_(0, 0.02)
            layer.lora_B.normal_(0, 0.02)
        raw = (layer.lora_B @ layer.lora_A).norm().item()
        scaled = layer.delta_weight().norm().item()
        norms[r] = (raw, scaled)

    # raw magnitude grows with rank...
    assert norms[32][0] > norms[2][0] * 2
    # ...and with alpha/r held at 1 the scaled version tracks it, which is the point: the KNOB is
    # alpha. Setting alpha to a constant instead makes the scaled norms comparable:
    const_alpha = {}
    for r in (2, 8, 32):
        layer = LoRALinear(nn.Linear(d, d), r=r, alpha=16.0)
        with torch.no_grad():
            layer.lora_A.normal_(0, 0.02)
            layer.lora_B.normal_(0, 0.02)
        const_alpha[r] = layer.delta_weight().norm().item()
    ratio = const_alpha[32] / const_alpha[2]
    assert 0.2 < ratio < 5.0, f"update magnitude varied {ratio:.1f}x across a 16x rank change"


# --------------------------------------------------------------------------------------------
# applying to a model
# --------------------------------------------------------------------------------------------

def test_apply_lora_targets_only_the_named_projections() -> None:
    model = tiny_model()
    adapted = apply_lora(model, target_suffixes=("w_q", "w_v"), r=4)
    assert sorted(adapted) == ["blocks.0.attn.w_q", "blocks.0.attn.w_v",
                               "blocks.1.attn.w_q", "blocks.1.attn.w_v"]
    assert isinstance(model.blocks[0].attn.w_q, LoRALinear)
    assert isinstance(model.blocks[0].attn.w_k, nn.Linear)
    assert not isinstance(model.blocks[0].attn.w_k, LoRALinear)


def test_apply_lora_raises_when_nothing_matches() -> None:
    """A silent zero-match yields an untrainable model that looks fine — the worst kind of failure."""
    with pytest.raises(ValueError, match="no modules matched"):
        apply_lora(tiny_model(), target_suffixes=("does_not_exist",))


def test_adapted_model_output_is_unchanged_before_training() -> None:
    """End-to-end version of the identity property, through a whole model."""
    torch.manual_seed(0)
    model = tiny_model().eval()
    x = torch.randn(3, 32)
    with torch.no_grad():
        before = model(x).clone()
    apply_lora(model, r=8, alpha=16)
    model.eval()
    with torch.no_grad():
        assert torch.equal(model(x), before)


def test_mark_only_lora_trainable_freezes_everything_else() -> None:
    """Without this the 'LoRA' arm would be training embeddings, norms and the head as well —
    which would make the comparison against full fine-tuning meaningless."""
    model = tiny_model()
    apply_lora(model, r=4)
    mark_only_lora_trainable(model)
    for name, p in model.named_parameters():
        expected = "lora_A" in name or "lora_B" in name
        assert p.requires_grad is expected, f"{name}: requires_grad={p.requires_grad}"


def test_parameter_counts_show_the_expected_reduction() -> None:
    """The headline claim, computed rather than quoted."""
    d, r = 32, 4
    model = tiny_model(d)
    full = sum(p.numel() for p in model.parameters())

    apply_lora(model, target_suffixes=("w_q", "w_v"), r=r, alpha=8)
    mark_only_lora_trainable(model)
    counts = count_parameters(model)

    # 4 adapted layers (2 blocks x {w_q, w_v}), each contributing r*(d_in + d_out)
    assert counts["lora"] == 4 * r * (d + d)
    assert counts["trainable"] == counts["lora"]
    assert counts["total"] == full + counts["lora"]
    assert counts["trainable_fraction"] < 0.1, counts


def test_merge_all_and_unmerge_all_cover_every_adapter() -> None:
    model = tiny_model()
    apply_lora(model, r=4)
    assert merge_all(model) == 4
    assert all(m.merged for m in model.modules() if isinstance(m, LoRALinear))
    assert unmerge_all(model) == 4
    assert not any(m.merged for m in model.modules() if isinstance(m, LoRALinear))


def test_merged_model_matches_unmerged_end_to_end() -> None:
    torch.manual_seed(0)
    model = tiny_model()
    apply_lora(model, r=8, alpha=16)
    for m in model.modules():
        if isinstance(m, LoRALinear):
            with torch.no_grad():
                m.lora_B.normal_(0, 0.05)
    model.eval()

    x = torch.randn(4, 32)
    with torch.no_grad():
        unmerged = model(x).clone()
        merge_all(model)
        merged = model(x)

    assert torch.allclose(unmerged, merged, atol=1e-5), (unmerged - merged).abs().max().item()


# --------------------------------------------------------------------------------------------
# the deployment story
# --------------------------------------------------------------------------------------------

def test_adapter_state_dict_is_tiny_and_sufficient() -> None:
    """Ship a few hundred KB per task, not a whole model copy — and prove it round-trips."""
    torch.manual_seed(0)
    trained = tiny_model()
    apply_lora(trained, r=4, alpha=8)
    for m in trained.modules():
        if isinstance(m, LoRALinear):
            with torch.no_grad():
                m.lora_B.normal_(0, 0.05)
    trained.eval()

    adapter = lora_state_dict(trained)
    assert adapter, "no adapter tensors found"
    assert all("lora_" in k for k in adapter)
    # Adapter is far smaller than the full state dict.
    full_elems = sum(v.numel() for v in trained.state_dict().values())
    adapter_elems = sum(v.numel() for v in adapter.values())
    assert adapter_elems < full_elems * 0.1

    # A fresh model with the SAME base weights plus this adapter must reproduce the output exactly.
    torch.manual_seed(0)
    fresh = tiny_model()
    apply_lora(fresh, r=4, alpha=8)
    missing, unexpected = fresh.load_state_dict(adapter, strict=False)
    assert not unexpected, unexpected
    fresh.eval()

    x = torch.randn(3, 32)
    with torch.no_grad():
        assert torch.allclose(trained(x), fresh(x), atol=1e-6)


def test_two_adapters_can_share_one_frozen_base() -> None:
    """The multi-task deployment claim: swap adapters, keep one copy of the base."""
    torch.manual_seed(0)
    model = tiny_model()
    apply_lora(model, r=4, alpha=8)
    model.eval()
    x = torch.randn(2, 32)

    adapters = []
    for scale in (0.05, 0.2):
        for m in model.modules():
            if isinstance(m, LoRALinear):
                with torch.no_grad():
                    m.lora_B.normal_(0, scale)
        adapters.append((lora_state_dict(model), model(x).detach().clone()))

    for state, expected in adapters:
        model.load_state_dict(state, strict=False)
        with torch.no_grad():
            assert torch.allclose(model(x), expected, atol=1e-6)


def test_rejects_negative_rank() -> None:
    with pytest.raises(ValueError, match="rank"):
        LoRALinear(nn.Linear(8, 8), r=-1)
