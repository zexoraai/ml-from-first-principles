"""Byte-level BPE, trained from scratch.

Primary sources
---------------
* Sennrich, Haddow & Birch, "Neural Machine Translation of Rare Words with Subword Units",
  arXiv:1508.07909 — BPE for text.
* Radford et al., GPT-2 (2019) — the *byte-level* variant, which is what this implements.

WHY BYTE-LEVEL, AND WHY IT MATTERS
----------------------------------
Classic BPE operates on Unicode characters and needs an `<unk>` token plus a normalisation step for
anything outside its alphabet. Byte-level BPE starts from the 256 possible byte values, so **every
possible input string is representable with no unknown token at all**. That property is the reason
GPT-2 uses it: the tokenizer can never fail on an input, which removes an entire class of
production bug.

The trade is that non-ASCII text costs more tokens (a character outside ASCII is 2-4 bytes, hence
2-4 initial symbols). We accept that; our corpus is English.

WHAT BPE ACTUALLY DOES
----------------------
Start with the sequence as individual bytes. Repeatedly find the most frequent adjacent pair and
merge it into a new symbol. Do that `V - 256` times and you have a vocabulary of `V` symbols where
common sequences ("the", " and") became single tokens while rare ones stayed fragmented. It is
frequency-driven compression, learned from the corpus, with no linguistic knowledge.

The merge list is **ordered** and that order is the model. Encoding replays the merges in the same
order they were learned; apply them in a different order and you get different tokens for the same
string, which would silently invalidate every trained checkpoint.

WHAT THIS IS NOT
----------------
This is not `tiktoken`, and it is not GPT-2's released vocabulary. It is the same algorithm trained
on our own corpus, so token ids are ours and are not interchangeable with any public model. It also
omits GPT-2's regex pre-tokenisation pattern (see `PRETOKEN_PATTERN` for what we do instead and
why).
"""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

__all__ = ["ByteBPETokenizer", "PRETOKEN_PATTERN"]

# Pre-tokenisation splits text into chunks *before* BPE, and merges are never allowed to cross a
# chunk boundary. Without it, BPE happily learns a single token for ".\nThe" -- merging punctuation,
# a newline and a word -- which wastes vocabulary on artefacts of the corpus's formatting.
#
# GPT-2 uses a hand-tuned regex. Ours is a simplified version that keeps the property that matters:
# a leading space stays attached to its word (so " the" and "the" are different tokens, which is
# what lets the model learn word boundaries without a separate marker), and runs of letters,
# digits, whitespace and punctuation are separated.
PRETOKEN_PATTERN = re.compile(
    r"""'(?:[sdmt]|ll|ve|re)| ?[A-Za-z]+| ?[0-9]+| ?[^\sA-Za-z0-9]+|\s+(?!\S)|\s+"""
)


class ByteBPETokenizer:
    """Trainable byte-level BPE.

    Vocabulary layout:
        0 .. 255          the raw byte values
        256 .. V-1-S      learned merges, in the order they were learned
        last S ids        special tokens

    Special tokens live at the END so that adding one does not renumber the byte ids or the merges,
    which would invalidate every existing checkpoint.
    """

    def __init__(self, merges: list[tuple[int, int]] | None = None,
                 specials: list[str] | None = None) -> None:
        self.merges: list[tuple[int, int]] = merges or []
        self.specials: list[str] = specials or ["<|endoftext|>"]
        self._rebuild()

    # -----------------------------------------------------------------------------------------
    def _rebuild(self) -> None:
        """Recompute the derived lookup tables after merges or specials change."""
        # rank[pair] = the step at which that pair was merged. Lower rank = merged earlier.
        self.rank: dict[tuple[int, int], int] = {p: i for i, p in enumerate(self.merges)}
        # new token id produced by each merge
        self.pair_to_id: dict[tuple[int, int], int] = {
            p: 256 + i for i, p in enumerate(self.merges)
        }
        n_learned = 256 + len(self.merges)
        self.special_to_id: dict[str, int] = {
            tok: n_learned + i for i, tok in enumerate(self.specials)
        }
        self.id_to_special: dict[int, str] = {v: k for k, v in self.special_to_id.items()}

        # Byte expansion for decoding: token id -> the bytes it stands for.
        self.token_bytes: list[bytes] = [bytes([b]) for b in range(256)]
        for a, b in self.merges:
            self.token_bytes.append(self.token_bytes[a] + self.token_bytes[b])

    @property
    def vocab_size(self) -> int:
        return 256 + len(self.merges) + len(self.specials)

    @property
    def eot_id(self) -> int:
        return self.special_to_id["<|endoftext|>"]

    def __len__(self) -> int:
        return self.vocab_size

    # -----------------------------------------------------------------------------------------
    # training
    # -----------------------------------------------------------------------------------------
    def train(
        self, text: str, vocab_size: int, *, min_frequency: int = 2, verbose: bool = False
    ) -> "ByteBPETokenizer":
        """Learn merges from `text` until the vocabulary reaches `vocab_size`.

        Args:
            vocab_size: **target** total, including the 256 bytes and the specials.
            min_frequency: stop merging once the best remaining pair is rarer than this. A pair
                occurring once contributes a vocabulary slot used exactly once, which is worse than
                leaving the sequence fragmented.

        **The resulting vocabulary may be SMALLER than requested**, and callers must read
        `tokenizer.vocab_size` rather than assuming their argument. Two things cause it: the corpus
        can run out of adjacent pairs entirely (every chunk already a single token), or the remaining
        pairs can fall below `min_frequency`. Both are properties of the corpus, not errors.

        This matters beyond tidiness: the model's `vocab_size` is taken from the tokenizer, so a
        caller that hard-codes its request would build an embedding table with rows that can never be
        indexed — wasted parameters and a silent mismatch with any other run.
        `tests/test_p2_tokenizer.py::test_small_corpus_yields_a_smaller_vocabulary_than_requested`
        pins the behaviour.

        Complexity: the straightforward O(merges x corpus) implementation, rescanning each merge. For
        our ~1 MB corpus and ~1k merges that is seconds. A multi-gigabyte corpus would need
        incremental pair counts; written the simple way on purpose, because the fast version obscures
        the algorithm.
        """
        n_merges = vocab_size - 256 - len(self.specials)
        if n_merges < 0:
            raise ValueError(
                f"vocab_size={vocab_size} is below the floor of 256 bytes + "
                f"{len(self.specials)} specials"
            )

        # Pre-tokenise, then represent each distinct chunk once with a count. Merges never cross
        # chunk boundaries, and identical chunks share work.
        chunk_counts = Counter(
            tuple(m.group().encode("utf-8")) for m in PRETOKEN_PATTERN.finditer(text)
        )
        words: list[list[int]] = [list(chunk) for chunk in chunk_counts]
        counts: list[int] = list(chunk_counts.values())

        self.merges = []
        for step in range(n_merges):
            pair_freq: Counter[tuple[int, int]] = Counter()
            for word, freq in zip(words, counts):
                for i in range(len(word) - 1):
                    pair_freq[(word[i], word[i + 1])] += freq
            if not pair_freq:
                if verbose:
                    print(f"  stopped early at {step} merges: no adjacent pairs remain")
                break

            best = max(pair_freq, key=lambda p: (pair_freq[p], -p[0], -p[1]))  # deterministic ties
            if pair_freq[best] < min_frequency:
                if verbose:
                    print(f"  stopped at {step} merges: most frequent pair occurs only "
                          f"{pair_freq[best]} times (min_frequency={min_frequency})")
                break
            new_id = 256 + len(self.merges)
            self.merges.append(best)
            words = [_merge_word(w, best, new_id) for w in words]

            if verbose and (step + 1) % 200 == 0:
                self._rebuild()
                print(f"  merge {step + 1:>5}/{n_merges}  {best} -> {new_id}  "
                      f"({pair_freq[best]:,} occurrences)  "
                      f"{self.token_bytes[new_id][:16]!r}")

        self._rebuild()
        return self

    # -----------------------------------------------------------------------------------------
    # encoding / decoding
    # -----------------------------------------------------------------------------------------
    def _encode_chunk(self, raw: bytes) -> list[int]:
        """Apply the learned merges to one pre-token, in the order they were learned."""
        ids = list(raw)
        while len(ids) >= 2:
            # Find the adjacent pair with the LOWEST rank, i.e. the one learned earliest.
            # Greedily merging the most frequent pair instead would produce a different
            # tokenisation than training did, which silently breaks a trained model.
            best_rank = None
            best_i = -1
            for i in range(len(ids) - 1):
                r = self.rank.get((ids[i], ids[i + 1]))
                if r is not None and (best_rank is None or r < best_rank):
                    best_rank, best_i = r, i
            if best_i < 0:
                break
            pair = (ids[best_i], ids[best_i + 1])
            ids[best_i:best_i + 2] = [self.pair_to_id[pair]]
        return ids

    def encode(self, text: str, *, allow_specials: bool = True) -> list[int]:
        """Text -> token ids. Never fails: every byte sequence is representable."""
        if allow_specials and self.specials:
            # Split on special-token literals so they map to their reserved id rather than being
            # BPE-encoded into pieces.
            pattern = "(" + "|".join(re.escape(s) for s in self.specials) + ")"
            parts = re.split(pattern, text)
        else:
            parts = [text]

        out: list[int] = []
        for part in parts:
            if not part:
                continue
            if allow_specials and part in self.special_to_id:
                out.append(self.special_to_id[part])
                continue
            for m in PRETOKEN_PATTERN.finditer(part):
                out.extend(self._encode_chunk(m.group().encode("utf-8")))
        return out

    def decode(self, ids: list[int], *, errors: str = "replace") -> str:
        """Token ids -> text.

        `errors="replace"` because a *partial* token sequence can end mid-UTF-8-character. That is
        normal during streaming generation, not a bug, and it must not raise.
        """
        buf = bytearray()
        for i in ids:
            i = int(i)
            if i in self.id_to_special:
                buf.extend(self.id_to_special[i].encode("utf-8"))
            elif 0 <= i < len(self.token_bytes):
                buf.extend(self.token_bytes[i])
            else:
                raise ValueError(f"token id {i} out of range for vocab size {self.vocab_size}")
        return buf.decode("utf-8", errors=errors)

    # -----------------------------------------------------------------------------------------
    # persistence
    # -----------------------------------------------------------------------------------------
    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps({
            "kind": "byte-level-bpe",
            "merges": [list(p) for p in self.merges],
            "specials": self.specials,
            "vocab_size": self.vocab_size,
        }), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "ByteBPETokenizer":
        blob = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(merges=[tuple(p) for p in blob["merges"]], specials=blob["specials"])

    def to_web_json(self) -> str:
        """Serialise for the browser, including the decoded byte strings.

        The browser needs `token_bytes` to render text. Recomputing the merge expansion in JS is
        possible but duplicates logic for no benefit, and a divergence there would corrupt every
        displayed string. Shipping the expansion makes the two implementations share one source of
        truth.
        """
        return json.dumps({
            "kind": "byte-level-bpe",
            "vocab_size": self.vocab_size,
            "merges": [list(p) for p in self.merges],
            "specials": self.specials,
            "special_ids": self.special_to_id,
            # latin-1 round-trips bytes 0-255 to code points 0-255 exactly, so this is a lossless
            # byte carrier through JSON. JS reverses it with charCodeAt.
            "token_bytes_latin1": [b.decode("latin-1") for b in self.token_bytes],
        })


def _merge_word(word: list[int], pair: tuple[int, int], new_id: int) -> list[int]:
    """Replace every non-overlapping occurrence of `pair` in `word` with `new_id`."""
    if len(word) < 2:
        return word
    out: list[int] = []
    i = 0
    a, b = pair
    while i < len(word):
        if i < len(word) - 1 and word[i] == a and word[i + 1] == b:
            out.append(new_id)
            i += 2
        else:
            out.append(word[i])
            i += 1
    return out
