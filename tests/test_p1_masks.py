"""Structural checks on the mask builders.

These are cheap invariant tests. The *behavioural* proof that masking works -- that a masked
position genuinely cannot influence an output -- lives in test_p1_attention.py, because that is
a property of attention, not of a boolean matrix.
"""

from __future__ import annotations

import pytest
import torch

from labs.p1_transformer.masks import (
    causal_keep_mask,
    combine_keep_masks,
    fully_masked_rows,
    padding_key_mask,
)


def test_padding_key_mask_shape_and_values() -> None:
    ids = torch.tensor([[5, 7, 0, 0], [1, 2, 3, 0]])
    keep = padding_key_mask(ids, pad_id=0)

    assert keep.shape == (2, 1, 1, 4), "must be rank-4 to broadcast over (heads, queries)"
    assert keep.dtype == torch.bool
    assert keep[0, 0, 0].tolist() == [True, True, False, False]
    assert keep[1, 0, 0].tolist() == [True, True, True, False]


def test_padding_key_mask_rejects_wrong_rank() -> None:
    with pytest.raises(ValueError, match="batch, seq_len"):
        padding_key_mask(torch.zeros(3, dtype=torch.long), pad_id=0)


def test_causal_keep_mask_is_lower_triangular_including_diagonal() -> None:
    keep = causal_keep_mask(4)
    assert keep.shape == (1, 1, 4, 4)
    expected = [
        [True, False, False, False],
        [True, True, False, False],
        [True, True, True, False],
        [True, True, True, True],
    ]
    assert keep[0, 0].tolist() == expected


def test_causal_keep_mask_diagonal_is_permitted() -> None:
    """Position i must be allowed to attend to itself.

    Because decoder inputs are the targets shifted right, "itself" is the previous target token,
    not the answer. Excluding the diagonal would make position 0 attend to nothing at all.
    """
    keep = causal_keep_mask(5)[0, 0]
    assert bool(keep.diagonal().all())


def test_causal_keep_mask_rectangular() -> None:
    keep = causal_keep_mask(2, 5)
    assert keep.shape == (1, 1, 2, 5)
    assert keep[0, 0, 0].tolist() == [True, False, False, False, False]
    assert keep[0, 0, 1].tolist() == [True, True, False, False, False]


def test_combine_keep_masks_is_logical_and() -> None:
    pad = padding_key_mask(torch.tensor([[1, 1, 0, 0]]), pad_id=0)   # (1,1,1,4)
    causal = causal_keep_mask(4)                                     # (1,1,4,4)
    both = combine_keep_masks(pad, causal)

    assert both is not None
    assert both.shape == (1, 1, 4, 4)
    # Row 3 may attend to everything causally, but positions 2 and 3 are padding.
    assert both[0, 0, 3].tolist() == [True, True, False, False]
    # Row 1 is limited by causality to {0,1}, both of which are real tokens.
    assert both[0, 0, 1].tolist() == [True, True, False, False]


def test_combine_keep_masks_ignores_none_and_returns_none_when_empty() -> None:
    causal = causal_keep_mask(3)
    assert torch.equal(combine_keep_masks(None, causal, None), causal)
    assert combine_keep_masks(None, None) is None


def test_combine_keep_masks_refuses_float_masks() -> None:
    """A float mask is the additive -inf convention. Silently ANDing it would be a real bug."""
    with pytest.raises(TypeError, match="bool"):
        combine_keep_masks(torch.zeros(1, 1, 3, 3))


def test_fully_masked_rows_detects_empty_rows() -> None:
    keep = torch.tensor([[[[True, False], [False, False]]]])
    empty = fully_masked_rows(keep)
    assert empty.shape == (1, 1, 2)
    assert empty[0, 0].tolist() == [False, True]


def test_padding_mask_alone_never_creates_empty_rows() -> None:
    """The reason `padding_key_mask` masks keys and not queries.

    Any sequence with at least one real token leaves every query row with something to attend to,
    so softmax is never asked to normalise over the empty set.
    """
    ids = torch.tensor([[9, 0, 0, 0], [1, 2, 0, 0]])
    keep = padding_key_mask(ids, pad_id=0).expand(2, 1, 4, 4)
    assert not bool(fully_masked_rows(keep).any())
