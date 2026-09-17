"""Correctness suite for the byte-level BPE tokenizer.

The load-bearing property is **lossless round-tripping on arbitrary input**. A tokenizer that
silently mangles a rare character produces a model that cannot reproduce it, and the failure surfaces
much later as an inexplicable generation artefact.
"""

from __future__ import annotations

import json

import pytest

from labs.p2_gpt.tokenizer import PRETOKEN_PATTERN, ByteBPETokenizer

CORPUS = (
    "First Citizen:\nBefore we proceed any further, hear me speak.\n"
    "All:\nSpeak, speak.\n"
    "First Citizen:\nYou are all resolved rather to die than to famish?\n"
    "All:\nResolved. resolved.\n"
    "First Citizen:\nFirst, you know Caius Marcius is chief enemy to the people.\n"
) * 12


@pytest.fixture(scope="module")
def tok() -> ByteBPETokenizer:
    return ByteBPETokenizer().train(CORPUS, vocab_size=400)


# --------------------------------------------------------------------------------------------
# the property that matters most
# --------------------------------------------------------------------------------------------

@pytest.mark.parametrize("text", [
    "First Citizen:",
    "Speak, speak.",
    "",
    " ",
    "\n\n\t  ",
    "aaaaaaaaaaaaaaaaaaaa",
    "!@#$%^&*()_+-=[]{}|;':\",./<>?",
    "1234567890",
    "MiXeD CaSe WoRdS",
    "unseen vocabulary: zyzzyva quixotic",
    "café naïve résumé",              # multi-byte UTF-8
    "日本語のテキスト",                    # 3-byte UTF-8
    "emoji 🎭 and 🗡️",                  # 4-byte UTF-8 with a variation selector
    "\x00\x01\x02 control bytes",
])
def test_round_trip_is_lossless_for_arbitrary_text(tok: ByteBPETokenizer, text: str) -> None:
    """Byte-level BPE has no <unk>: every possible string must survive encode->decode exactly.

    The non-ASCII cases matter even though the corpus is English. They were never seen in training,
    so they exercise the fallback to raw byte tokens -- which is precisely the property that makes
    byte-level BPE unable to fail.
    """
    assert tok.decode(tok.encode(text)) == text


def test_round_trip_holds_for_the_whole_training_corpus(tok: ByteBPETokenizer) -> None:
    assert tok.decode(tok.encode(CORPUS)) == CORPUS


def test_every_token_id_is_within_the_vocabulary(tok: ByteBPETokenizer) -> None:
    ids = tok.encode(CORPUS)
    assert ids, "encoding produced nothing"
    assert all(0 <= i < tok.vocab_size for i in ids)


# --------------------------------------------------------------------------------------------
# vocabulary structure
# --------------------------------------------------------------------------------------------

def test_vocabulary_never_exceeds_the_request(tok: ByteBPETokenizer) -> None:
    assert tok.vocab_size <= 400
    assert tok.vocab_size == 256 + len(tok.merges) + len(tok.specials)


def test_small_corpus_yields_a_smaller_vocabulary_than_requested() -> None:
    """`vocab_size` is a target, not a guarantee, and callers must read the real value.

    A tiny corpus runs out of pairs worth merging: once every chunk is a single token, or the best
    remaining pair occurs less than `min_frequency` times, training stops. Both are properties of
    the corpus, not errors.

    Why this is worth a test rather than a docstring: the model's embedding table is sized from
    `tokenizer.vocab_size`. A caller that assumed its request was honoured would allocate rows that
    can never be indexed — wasted parameters, and a silent mismatch with any run that used the real
    size.
    """
    tiny = ByteBPETokenizer().train("abab " * 30, vocab_size=2000)
    assert tiny.vocab_size < 2000
    assert tiny.decode(tiny.encode("abab abab")) == "abab abab"


def test_min_frequency_prevents_single_use_merges() -> None:
    """A merge used once spends a vocabulary slot to save nothing."""
    text = "the the the the the " + "".join(f"z{i}q " for i in range(40))
    lenient = ByteBPETokenizer().train(text, vocab_size=600, min_frequency=1)
    strict = ByteBPETokenizer().train(text, vocab_size=600, min_frequency=3)
    assert len(strict.merges) < len(lenient.merges)


def test_first_256_ids_are_the_raw_bytes(tok: ByteBPETokenizer) -> None:
    """The byte floor is what guarantees no input is unrepresentable."""
    for b in range(256):
        assert tok.token_bytes[b] == bytes([b])


def test_specials_sit_at_the_end_of_the_vocabulary(tok: ByteBPETokenizer) -> None:
    """So that adding a special token cannot renumber bytes or merges and invalidate checkpoints."""
    assert tok.eot_id == tok.vocab_size - 1
    assert tok.eot_id >= 256 + len(tok.merges)


def test_special_token_encodes_to_its_reserved_id_not_to_pieces(tok: ByteBPETokenizer) -> None:
    ids = tok.encode("hello<|endoftext|>world")
    assert tok.eot_id in ids
    assert tok.decode(ids) == "hello<|endoftext|>world"


def test_merges_reduce_token_count_versus_raw_bytes(tok: ByteBPETokenizer) -> None:
    """The entire point of BPE. If this fails, no merges are being applied."""
    raw_bytes = len(CORPUS.encode("utf-8"))
    encoded = len(tok.encode(CORPUS))
    assert encoded < raw_bytes * 0.7, f"{encoded} tokens for {raw_bytes} bytes is barely compressed"


def test_more_merges_compress_better() -> None:
    small = ByteBPETokenizer().train(CORPUS, vocab_size=300, min_frequency=1)
    large = ByteBPETokenizer().train(CORPUS, vocab_size=600, min_frequency=1)
    assert len(large.merges) > len(small.merges)
    assert len(large.encode(CORPUS)) < len(small.encode(CORPUS))


def test_frequent_sequences_become_single_tokens(tok: ByteBPETokenizer) -> None:
    """"First Citizen" appears repeatedly, so BPE should have collapsed large parts of it."""
    ids = tok.encode("First Citizen:")
    assert len(ids) < len("First Citizen:"), "no compression on the most frequent phrase in the corpus"


# --------------------------------------------------------------------------------------------
# determinism and the merge order
# --------------------------------------------------------------------------------------------

def test_training_is_deterministic() -> None:
    """Ties are broken deterministically, so two trainings on the same text agree exactly.

    Without this, two runs produce different vocabularies and a checkpoint silently mismatches its
    tokenizer -- a failure that looks like a broken model rather than a broken tokenizer.
    """
    a = ByteBPETokenizer().train(CORPUS, vocab_size=350)
    b = ByteBPETokenizer().train(CORPUS, vocab_size=350)
    assert a.merges == b.merges


def test_encoding_is_deterministic(tok: ByteBPETokenizer) -> None:
    assert tok.encode(CORPUS) == tok.encode(CORPUS)


def test_encoding_applies_merges_in_learned_order_not_greedily_by_length() -> None:
    """Encoding must replay merges in rank order.

    Constructed case: merges are (a,b)->256 learned first, then (256,c)->257. Encoding "abc" must
    produce [257], because the first merge fires before the second becomes applicable. A tokenizer
    that picked the *longest* match, or that iterated merges in an arbitrary order, would produce
    something else and disagree with what training saw.
    """
    t = ByteBPETokenizer(merges=[(ord("a"), ord("b")), (256, ord("c"))], specials=[])
    assert t.encode("abc") == [257]
    assert t.decode([257]) == "abc"


def test_no_merge_crosses_a_pretoken_boundary() -> None:
    """Pre-tokenisation stops BPE learning tokens like ".\\nThe" that span punctuation and words."""
    text = ("word.\nword.\n" * 200)
    t = ByteBPETokenizer().train(text, vocab_size=300)
    for token in t.token_bytes[256:]:
        decoded = token.decode("utf-8", errors="replace")
        assert "\n" not in decoded or decoded.strip() == "", (
            f"merge {decoded!r} spans a newline boundary"
        )


def test_leading_space_stays_attached_to_its_word() -> None:
    """" the" and "the" must be distinguishable, which is how the model learns word boundaries."""
    chunks = [m.group() for m in PRETOKEN_PATTERN.finditer("the cat sat")]
    assert chunks == ["the", " cat", " sat"]


def test_pretokeniser_covers_the_input_exactly() -> None:
    """Concatenating the chunks must reproduce the input: no dropped or duplicated characters."""
    for text in ["First Citizen:\nSpeak, speak.\n", "a  b\t\tc\n\n", "1234 abc!?", "  leading"]:
        assert "".join(m.group() for m in PRETOKEN_PATTERN.finditer(text)) == text


# --------------------------------------------------------------------------------------------
# persistence and the browser handoff
# --------------------------------------------------------------------------------------------

def test_save_load_round_trip_preserves_encoding(tmp_path, tok: ByteBPETokenizer) -> None:
    path = tmp_path / "tok.json"
    tok.save(path)
    reloaded = ByteBPETokenizer.load(path)
    assert reloaded.merges == tok.merges
    assert reloaded.vocab_size == tok.vocab_size
    assert reloaded.encode(CORPUS) == tok.encode(CORPUS)


def test_web_json_carries_bytes_losslessly(tok: ByteBPETokenizer) -> None:
    """The browser reads `token_bytes_latin1`; latin-1 round-trips bytes 0-255 exactly.

    If this encoding were wrong, the demo would render mojibake for any token containing a byte above
    0x7F -- and it would do so only for rare tokens, which is the worst kind of bug to find late.
    """
    blob = json.loads(tok.to_web_json())
    assert blob["vocab_size"] == tok.vocab_size
    assert len(blob["token_bytes_latin1"]) == len(tok.token_bytes)
    for expected, carried in zip(tok.token_bytes, blob["token_bytes_latin1"]):
        assert carried.encode("latin-1") == expected


def test_web_json_is_valid_json_and_json_safe(tok: ByteBPETokenizer) -> None:
    """Must survive a JSON round trip, including tokens containing quotes and backslashes."""
    reparsed = json.loads(json.dumps(json.loads(tok.to_web_json())))
    assert reparsed["kind"] == "byte-level-bpe"


# --------------------------------------------------------------------------------------------
# guardrails
# --------------------------------------------------------------------------------------------

def test_rejects_vocab_below_the_byte_floor() -> None:
    with pytest.raises(ValueError, match="floor of 256"):
        ByteBPETokenizer().train(CORPUS, vocab_size=100)


def test_decode_rejects_an_out_of_range_id(tok: ByteBPETokenizer) -> None:
    with pytest.raises(ValueError, match="out of range"):
        tok.decode([tok.vocab_size + 5])


def test_partial_multibyte_sequence_decodes_without_raising(tok: ByteBPETokenizer) -> None:
    """Streaming generation can stop mid-character; that must not be an exception.

    A generation UI appends one token at a time, so a multi-byte character is momentarily incomplete
    on almost every emoji or accented word. Raising here would crash the demo mid-sentence.
    """
    ids = tok.encode("café")
    for cut in range(1, len(ids) + 1):
        assert isinstance(tok.decode(ids[:cut]), str)
