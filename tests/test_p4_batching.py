"""Tests for tokenising and padding preference data.

The two checks that matter most are `test_padding_does_not_change_the_score` (the whole justification
for right-padding without an attention mask) and `test_response_mask_covers_exactly_the_response`.
"""

from __future__ import annotations

import pytest
import torch

from labs.p2_gpt import GPT, GPTConfig
from labs.p2_gpt.tokenizer import ByteBPETokenizer
from labs.p4_dpo import (
    PairBatcher,
    SFTBatcher,
    encode_pair_batch,
    encode_sft_batch,
    sequence_logprobs,
)
from labs.p4_dpo.data import InstructionExample, PreferencePair


@pytest.fixture(scope="module")
def tok() -> ByteBPETokenizer:
    text = ("Typography is the art of arranging type. Kerning adjusts the space between two "
            "letters. A grid organises the page into columns and rows. Colour theory explains "
            "how hues interact. ") * 12
    return ByteBPETokenizer().train(text, vocab_size=400)


def pairs() -> list[PreferencePair]:
    return [
        PreferencePair(prompt="### Instruction:\nWhat is kerning?\n\n### Response:\n",
                       chosen="Kerning adjusts the space between two letters.\n",
                       rejected="A grid organises the page.\n",
                       topic="kerning", degradation="off_topic"),
        PreferencePair(prompt="### Instruction:\nWhat is a grid?\n\n### Response:\n",
                       chosen="A grid organises the page into columns and rows and gives the "
                              "designer a repeatable structure to work against.\n",
                       rejected="A grid organises\n",
                       topic="grid", degradation="truncated"),
    ]


def examples() -> list[InstructionExample]:
    return [
        InstructionExample(prompt="### Instruction:\nWhat is kerning?\n\n### Response:\n",
                           response="Kerning adjusts the space between two letters.\n",
                           topic="kerning"),
        InstructionExample(prompt="### Instruction:\nWhat is a grid?\n\n### Response:\n",
                           response="A grid organises the page into columns and rows.\n",
                           topic="grid"),
    ]


# --------------------------------------------------------------------------------------------
# masks
# --------------------------------------------------------------------------------------------

def test_response_mask_covers_exactly_the_response(tok) -> None:
    """Decoding the masked span must reproduce the response text and nothing else."""
    batch = encode_sft_batch(examples(), tok, block_size=192, pad_id=tok.eot_id)
    for row, ex in enumerate(examples()):
        picked = batch["ids"][row][batch["mask"][row].bool()].tolist()
        assert tok.decode(picked) == ex.response


def test_prompt_positions_are_not_in_the_mask(tok) -> None:
    batch = encode_sft_batch(examples(), tok, block_size=192, pad_id=tok.eot_id)
    n_prompt = len(tok.encode(examples()[0].prompt))
    assert batch["mask"][0, :n_prompt].sum().item() == 0.0
    assert batch["mask"][0, n_prompt].item() == 1.0


def test_padding_positions_are_not_in_the_mask(tok) -> None:
    batch = encode_sft_batch(examples(), tok, block_size=192, pad_id=tok.eot_id)
    lengths = [len(tok.encode(e.prompt)) + len(tok.encode(e.response)) for e in examples()]
    for row, length in enumerate(lengths):
        assert batch["mask"][row, length:].sum().item() == 0.0, "pad positions leaked into the mask"


def test_batch_is_right_padded_not_left_padded(tok) -> None:
    """The correctness of using a causal-only model on padded batches depends on this."""
    batch = encode_sft_batch(examples(), tok, block_size=192, pad_id=tok.eot_id)
    row_len = len(tok.encode(examples()[0].prompt)) + len(tok.encode(examples()[0].response))
    if row_len < batch["ids"].shape[1]:
        assert bool((batch["ids"][0, row_len:] == tok.eot_id).all()), "padding is not on the right"
        assert batch["ids"][0, 0] != tok.eot_id or row_len == 0


# --------------------------------------------------------------------------------------------
# THE padding-safety test
# --------------------------------------------------------------------------------------------

def test_padding_does_not_change_the_score(tok) -> None:
    """A sequence scored alone and scored inside a padded batch must give the same log-probability.

    This is the empirical form of the module's argument: with causal attention and right-padding, no
    real token can attend to a pad, so the extra columns cannot influence the result. If this ever
    fails, either the padding side changed or the model stopped being purely causal — and in both
    cases every DPO number computed on a batch would be quietly wrong.
    """
    torch.manual_seed(0)
    model = GPT(GPTConfig(vocab_size=tok.vocab_size, block_size=192, n_layer=2, n_head=4,
                          d_model=32, dropout=0.0, attention_dropout=0.0)).eval()

    batch = encode_sft_batch(examples(), tok, block_size=192, pad_id=tok.eot_id)
    with torch.no_grad():
        batched = sequence_logprobs(model(batch["ids"])[0], batch["ids"], batch["mask"])

    for row, ex in enumerate(examples()):
        solo = encode_sft_batch([ex], tok, block_size=192, pad_id=tok.eot_id)
        with torch.no_grad():
            alone = sequence_logprobs(model(solo["ids"])[0], solo["ids"], solo["mask"])
        assert alone.item() == pytest.approx(batched[row].item(), abs=1e-4), (
            f"row {row}: padding changed the score by "
            f"{abs(alone.item() - batched[row].item()):.2e}"
        )


# --------------------------------------------------------------------------------------------
# pair encoding
# --------------------------------------------------------------------------------------------

def test_pair_encoding_shares_the_prompt_between_chosen_and_rejected(tok) -> None:
    batch = encode_pair_batch(pairs(), tok, block_size=192, pad_id=tok.eot_id)
    n_prompt = len(tok.encode(pairs()[0].prompt))
    assert torch.equal(batch["chosen_ids"][0, :n_prompt], batch["rejected_ids"][0, :n_prompt])


def test_chosen_and_rejected_may_have_different_widths(tok) -> None:
    """They are summed per sequence before comparison, so alignment is unnecessary."""
    batch = encode_pair_batch(pairs(), tok, block_size=192, pad_id=tok.eot_id)
    assert batch["chosen_ids"].shape[0] == batch["rejected_ids"].shape[0]
    assert batch["chosen_mask"].sum() > batch["rejected_mask"].sum()


def test_pair_batch_carries_degradation_labels_for_per_type_reporting(tok) -> None:
    batch = encode_pair_batch(pairs(), tok, block_size=192, pad_id=tok.eot_id)
    assert batch["degradations"] == ["off_topic", "truncated"]
    assert batch["topics"] == ["kerning", "grid"]


# --------------------------------------------------------------------------------------------
# truncation policy
# --------------------------------------------------------------------------------------------

def test_truncation_removes_response_tokens_and_keeps_the_prompt_whole(tok) -> None:
    """Cutting the prompt would change the question, invalidating the pair."""
    long_response = "Kerning adjusts the space between letters. " * 30
    ex = InstructionExample(prompt="### Instruction:\nWhat is kerning?\n\n### Response:\n",
                            response=long_response, topic="kerning")
    batch = encode_sft_batch([ex], tok, block_size=64, pad_id=tok.eot_id)
    n_prompt = len(tok.encode(ex.prompt))
    assert batch["ids"].shape[1] == 64
    assert batch["n_truncated"] == 1
    assert tok.decode(batch["ids"][0, :n_prompt].tolist()) == ex.prompt


def test_a_prompt_longer_than_the_block_is_a_hard_error(tok) -> None:
    ex = InstructionExample(prompt="word " * 200, response="x\n", topic="t")
    with pytest.raises(ValueError, match="no room for a response"):
        encode_sft_batch([ex], tok, block_size=32, pad_id=tok.eot_id)


def test_nothing_ever_exceeds_the_block_size(tok) -> None:
    """block_size=48 leaves room for the ~40-token prompt but forces the response to be cut."""
    batch = encode_pair_batch(pairs(), tok, block_size=48, pad_id=tok.eot_id)
    assert batch["chosen_ids"].shape[1] <= 48
    assert batch["rejected_ids"].shape[1] <= 48


# --------------------------------------------------------------------------------------------
# batchers
# --------------------------------------------------------------------------------------------

def test_batcher_visits_every_example_once_per_epoch(tok) -> None:
    """Sampling with replacement would leave some pairs unseen; on a few hundred pairs that matters."""
    items = [InstructionExample(prompt="### Instruction:\nQ\n\n### Response:\n",
                               response=f"answer {i}\n", topic=f"t{i}") for i in range(10)]
    b = SFTBatcher(items, tok, block_size=192, batch_size=2, pad_id=tok.eot_id, seed=0)
    seen: list[int] = []
    for _ in range(5):
        seen.extend(b.next_indices())
    assert sorted(seen) == list(range(10))
    assert b.epochs == 1


def test_batcher_is_reproducible_and_restorable(tok) -> None:
    items = [InstructionExample(prompt="### Instruction:\nQ\n\n### Response:\n",
                               response=f"a {i}\n", topic=f"t{i}") for i in range(12)]
    kw = dict(block_size=192, batch_size=3, pad_id=tok.eot_id, seed=7)
    a = SFTBatcher(items, tok, **kw)
    b = SFTBatcher(items, tok, **kw)
    assert [a.next_indices() for _ in range(3)] == [b.next_indices() for _ in range(3)]

    state = a.state_dict()
    expected = a.next_indices()
    c = SFTBatcher(items, tok, **kw)
    c.load_state_dict(state)
    assert c.next_indices() == expected


def test_all_batches_covers_every_pair_exactly_once(tok) -> None:
    many = pairs() * 5
    b = PairBatcher(many, tok, block_size=192, batch_size=3, pad_id=tok.eot_id)
    batches = b.all_batches()
    assert sum(x["chosen_ids"].shape[0] for x in batches) == len(many)
    assert [d for x in batches for d in x["degradations"]] == [p.degradation for p in many]
