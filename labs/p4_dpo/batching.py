"""Tokenising and padding instruction and preference data into batches.

WHY RIGHT-PADDING NEEDS NO ATTENTION MASK HERE, AND LEFT-PADDING WOULD
---------------------------------------------------------------------
The Project 2 GPT applies causal masking only — it takes no padding mask. That is safe with
**right**-padding and unsafe with left-padding, and the reason is worth stating precisely because it
is a common source of silent corruption.

Causal masking lets position `t` attend to positions `0..t` and no further. With padding on the right,
every pad sits at a position *after* every real token, so no real token can attend to a pad. The pad
positions do attend to real tokens and produce garbage outputs, but those positions are excluded by
`response_mask`, so the garbage never reaches the loss.

Left-padding inverts this: pads occupy positions *before* the real tokens, every real token attends to
them, and the representations are corrupted with no error raised anywhere. Loss curves look fine and
the model is quietly worse. Left-padding therefore requires an explicit padding mask, which this model
does not accept — so this module right-pads, and asserts it.

(Left-padding is the correct choice for *batched generation*, where every sequence must end at the same
position. That is a different code path with a different model requirement, and this project does not
batch its generation.)

WHY THE PAD TOKEN IS `eot` AND WHY IT DOES NOT MATTER
-----------------------------------------------------
Pad positions are masked out of every loss, so the pad *value* is arithmetically irrelevant. Using
`eot` rather than `0` still matters for debugging: a decoded batch reads as text ending in
`<|endoftext|>` instead of text ending in a wall of whatever byte 0 renders as.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .data import InstructionExample, PreferencePair

__all__ = ["EncodedPair", "encode_sft_batch", "encode_pair_batch", "PairBatcher", "SFTBatcher"]


@dataclass
class EncodedPair:
    n_prompt_tokens: int
    n_chosen_tokens: int
    n_rejected_tokens: int
    truncated: bool


def _encode_one(tokenizer, prompt: str, response: str, block_size: int) -> tuple[list[int], int, bool]:
    """Return `(ids, n_prompt, was_truncated)` for one prompt/response pair.

    Truncation drops tokens from the **end of the response**, never from the prompt. Cutting the
    prompt would change the question the model is being asked, which silently invalidates the pair —
    the `chosen` text would no longer be an answer to the recorded instruction.
    """
    p_ids = tokenizer.encode(prompt)
    r_ids = tokenizer.encode(response)
    if len(p_ids) >= block_size:
        raise ValueError(
            f"prompt alone is {len(p_ids)} tokens, at or beyond block_size {block_size}; "
            f"there is no room for a response. Shorten the prompt template."
        )
    budget = block_size - len(p_ids)
    truncated = len(r_ids) > budget
    return p_ids + r_ids[:budget], len(p_ids), truncated


def _pad_stack(sequences: list[list[int]], prompt_lens: list[int], pad_id: int
               ) -> tuple[torch.Tensor, torch.Tensor]:
    """Right-pad to the longest sequence and build the response mask.

    The mask is 1 only on response tokens: not on the prompt, not on padding. Both exclusions are
    load-bearing — see the module docstring for padding and `dpo.sequence_logprobs` for the prompt.
    """
    width = max(len(s) for s in sequences)
    ids = torch.full((len(sequences), width), pad_id, dtype=torch.long)
    mask = torch.zeros((len(sequences), width), dtype=torch.float32)
    for row, (seq, n_prompt) in enumerate(zip(sequences, prompt_lens)):
        ids[row, : len(seq)] = torch.tensor(seq, dtype=torch.long)
        mask[row, n_prompt : len(seq)] = 1.0
    return ids, mask


def encode_sft_batch(examples: list[InstructionExample], tokenizer, block_size: int,
                     pad_id: int) -> dict:
    seqs, plens, n_trunc = [], [], 0
    for ex in examples:
        ids, n_prompt, trunc = _encode_one(tokenizer, ex.prompt, ex.response, block_size)
        seqs.append(ids)
        plens.append(n_prompt)
        n_trunc += int(trunc)
    ids, mask = _pad_stack(seqs, plens, pad_id)
    return {"ids": ids, "mask": mask, "n_truncated": n_trunc}


def encode_pair_batch(pairs: list[PreferencePair], tokenizer, block_size: int, pad_id: int) -> dict:
    """Encode preference pairs.

    Chosen and rejected are padded **independently**, so they may have different widths. That is
    correct: their log-probabilities are summed per sequence before ever being compared, so the two
    tensors never need to align. Forcing them to a common width would only add padding that is
    masked out anyway.
    """
    c_seqs, c_plens, r_seqs, r_plens, n_trunc = [], [], [], [], 0
    for pair in pairs:
        c_ids, c_np, c_t = _encode_one(tokenizer, pair.prompt, pair.chosen, block_size)
        r_ids, r_np, r_t = _encode_one(tokenizer, pair.prompt, pair.rejected, block_size)
        c_seqs.append(c_ids); c_plens.append(c_np)
        r_seqs.append(r_ids); r_plens.append(r_np)
        n_trunc += int(c_t or r_t)
    chosen_ids, chosen_mask = _pad_stack(c_seqs, c_plens, pad_id)
    rejected_ids, rejected_mask = _pad_stack(r_seqs, r_plens, pad_id)
    return {
        "chosen_ids": chosen_ids, "chosen_mask": chosen_mask,
        "rejected_ids": rejected_ids, "rejected_mask": rejected_mask,
        "n_truncated": n_trunc,
        "degradations": [p.degradation for p in pairs],
        "topics": [p.topic for p in pairs],
    }


class _EpochShuffler:
    """Shuffled epochs over a fixed list, with a restorable RNG.

    Sampling with replacement would be simpler, but then "one epoch" has no meaning and some examples
    are never seen while others are seen repeatedly. On a dataset of a few hundred pairs that
    difference is large enough to change the result.
    """

    def __init__(self, n: int, batch_size: int, seed: int) -> None:
        self.n, self.batch_size = n, batch_size
        self.rng = torch.Generator().manual_seed(seed)
        self.order: list[int] = []
        self.epochs = 0

    def next_indices(self) -> list[int]:
        if len(self.order) < self.batch_size:
            perm = torch.randperm(self.n, generator=self.rng).tolist()
            self.order.extend(perm)
            self.epochs += 1
        take, self.order = self.order[: self.batch_size], self.order[self.batch_size :]
        return take

    def state_dict(self) -> dict:
        return {"rng": self.rng.get_state(), "order": list(self.order), "epochs": self.epochs}

    def load_state_dict(self, state: dict) -> None:
        self.rng.set_state(state["rng"])
        self.order = list(state["order"])
        self.epochs = state["epochs"]


class SFTBatcher(_EpochShuffler):
    def __init__(self, examples: list[InstructionExample], tokenizer, *, block_size: int,
                 batch_size: int, pad_id: int, seed: int = 0) -> None:
        super().__init__(len(examples), batch_size, seed)
        self.examples, self.tokenizer = examples, tokenizer
        self.block_size, self.pad_id = block_size, pad_id

    def next_batch(self) -> dict:
        picks = [self.examples[i] for i in self.next_indices()]
        return encode_sft_batch(picks, self.tokenizer, self.block_size, self.pad_id)


class PairBatcher(_EpochShuffler):
    def __init__(self, pairs: list[PreferencePair], tokenizer, *, block_size: int,
                 batch_size: int, pad_id: int, seed: int = 0) -> None:
        super().__init__(len(pairs), batch_size, seed)
        self.pairs, self.tokenizer = pairs, tokenizer
        self.block_size, self.pad_id = block_size, pad_id

    def next_batch(self) -> dict:
        picks = [self.pairs[i] for i in self.next_indices()]
        return encode_pair_batch(picks, self.tokenizer, self.block_size, self.pad_id)

    def all_batches(self) -> list[dict]:
        """Every pair exactly once, in order — for evaluation, where sampling would add noise."""
        return [
            encode_pair_batch(self.pairs[i : i + self.batch_size], self.tokenizer,
                              self.block_size, self.pad_id)
            for i in range(0, len(self.pairs), self.batch_size)
        ]
