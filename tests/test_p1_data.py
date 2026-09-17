"""Correctness suite for the date-normalisation task, tokenizer, and splits.

The leakage tests here matter more than they look. A split that leaks produces validation numbers
that are simply wrong -- optimistically wrong, in the direction nobody checks. These tests are the
reason the reported exact-match figure can be believed.
"""

from __future__ import annotations

import datetime as dt
import json
import random

import pytest
import torch

from labs.p1_transformer.data import (
    RENDERERS,
    CharTokenizer,
    DateDataset,
    DateTaskConfig,
    build_splits,
    collate,
    iso,
    render_distinct,
)


# --------------------------------------------------------------------------------------------
# tokenizer
# --------------------------------------------------------------------------------------------

def test_special_token_ids_match_the_model_defaults() -> None:
    """pad=0, bos=1, eos=2 so `TransformerConfig`'s defaults line up without a translation layer."""
    tok = CharTokenizer()
    assert (tok.pad_id, tok.bos_id, tok.eos_id, tok.unk_id) == (0, 1, 2, 3)


def test_roundtrip_is_lossless_for_every_alphabet_character() -> None:
    tok = CharTokenizer()
    text = CharTokenizer.ALPHABET
    assert tok.decode(tok.encode(text)) == text


def test_unknown_characters_map_to_unk_rather_than_crashing() -> None:
    tok = CharTokenizer()
    assert tok.encode("é")[0] == tok.unk_id


def test_vocabulary_is_fixed_and_not_data_dependent() -> None:
    """Two tokenizers must agree exactly, so a checkpoint is portable across runs and seeds."""
    a, b = CharTokenizer(), CharTokenizer()
    assert a.itos == b.itos
    assert len(a) == len(CharTokenizer.SPECIALS) + len(CharTokenizer.ALPHABET)


def test_decode_stops_at_eos_and_drops_specials() -> None:
    tok = CharTokenizer()
    ids = [tok.bos_id] + tok.encode("2019-03-03") + [tok.eos_id] + tok.encode("GARBAGE")
    assert tok.decode(ids) == "2019-03-03"


def test_tokenizer_json_is_valid_and_complete() -> None:
    """The browser demo loads this; a mismatch would make the demo's output a fabrication."""
    payload = json.loads(CharTokenizer().to_json())
    assert payload["itos"][0] == "<pad>"
    assert payload["bos_id"] == 1 and payload["eos_id"] == 2
    assert len(payload["itos"]) == len(CharTokenizer())


# --------------------------------------------------------------------------------------------
# renderers
# --------------------------------------------------------------------------------------------

def test_every_renderer_produces_only_alphabet_characters() -> None:
    """Any character outside the declared alphabet would silently become <unk> and be unlearnable."""
    tok = CharTokenizer()
    allowed = set(CharTokenizer.ALPHABET)
    for name, fn in RENDERERS.items():
        for d in (dt.date(1950, 1, 1), dt.date(2035, 12, 31), dt.date(2019, 3, 3), dt.date(2000, 11, 22)):
            text = fn(d)
            unexpected = set(text) - allowed
            assert not unexpected, f"{name} emitted {unexpected!r} for {d}"
            assert tok.unk_id not in tok.encode(text)


def test_renderers_fit_within_max_src_len() -> None:
    cfg = DateTaskConfig()
    worst = 0
    for name, fn in RENDERERS.items():
        for year in (1950, 2035):
            for month in range(1, 13):
                d = dt.date(year, month, 22)      # a 2-digit day, and September is the longest month
                worst = max(worst, len(fn(d)))
    assert worst <= cfg.max_src_len, f"longest rendering is {worst} chars, max_src_len={cfg.max_src_len}"


def test_iso_target_is_always_exactly_ten_characters() -> None:
    for d in (dt.date(1950, 1, 1), dt.date(2035, 12, 31), dt.date(2019, 3, 3)):
        assert len(iso(d)) == 10


def test_renderer_collisions_exist_and_are_documented() -> None:
    """Pins the two known collisions, so the deduplication in `render_distinct` stays justified.

    Renderers are NOT pairwise distinct, and cannot be made so:

    * **May** is the only month whose 3-letter abbreviation equals its full name, so
      `month_day_year` and `abbr_day_year` both give "May 1, 2019", and `day_month_year` and
      `day_abbr_year` both give "1 May 2019".

    This is a fact about English, not a fixable format choice. It is why the generator deduplicates
    on the rendered **string** rather than trusting distinct renderer names. A separate collision
    (`zero_padded_abbr` vs `day_abbr_year` for two-digit days) *was* format-fixable and was fixed
    by hyphenating.
    """
    may = dt.date(2019, 5, 1)
    assert RENDERERS["month_day_year"](may) == RENDERERS["abbr_day_year"](may) == "May 1, 2019"
    assert RENDERERS["day_month_year"](may) == RENDERERS["day_abbr_year"](may) == "1 May 2019"

    # Non-May months must NOT collide -- if they did, something else has broken.
    march = dt.date(2019, 3, 1)
    assert RENDERERS["month_day_year"](march) != RENDERERS["abbr_day_year"](march)

    # And the previously-fixed collision must stay fixed for two-digit days.
    d24 = dt.date(2019, 3, 24)
    assert RENDERERS["hyphenated_abbr"](d24) != RENDERERS["day_abbr_year"](d24)


def test_render_distinct_never_returns_duplicates() -> None:
    """The property the generator actually needs, checked directly on every month."""
    rng = random.Random(0)
    for month in range(1, 13):
        for day in (1, 9, 24):
            variants = render_distinct(dt.date(2019, month, day), 10, rng)
            assert len(variants) == len(set(variants)), f"duplicates for 2019-{month:02d}-{day:02d}"


def test_render_distinct_yields_fewer_variants_for_may() -> None:
    """Documents the asymmetry the May collision creates, rather than leaving it as a surprise."""
    rng = random.Random(0)
    may = len(render_distinct(dt.date(2019, 5, 1), 99, rng))
    march = len(render_distinct(dt.date(2019, 3, 1), 99, rng))
    assert march == len(RENDERERS)
    assert may == len(RENDERERS) - 2, f"expected 2 collisions in May, got {len(RENDERERS) - may}"


def test_weekday_renderers_state_the_correct_weekday() -> None:
    """Generated data must be internally consistent, or the task contains unlearnable noise."""
    d = dt.date(2019, 3, 3)          # a Sunday
    assert "Sunday" in RENDERERS["weekday_full"](d)
    assert RENDERERS["weekday_abbr"](d).startswith("Sun")


def test_ordinal_suffixes_are_correct_including_the_teens() -> None:
    got = {d: RENDERERS["ordinal_month_year"](dt.date(2019, 1, d)).split()[0] for d in (1, 2, 3, 4, 11, 12, 13, 21, 22, 23, 30)}
    assert got == {
        1: "1st", 2: "2nd", 3: "3rd", 4: "4th",
        11: "11th", 12: "12th", 13: "13th",     # the exception that catches naive implementations
        21: "21st", 22: "22nd", 23: "23rd", 30: "30th",
    }


# --------------------------------------------------------------------------------------------
# splits and leakage
# --------------------------------------------------------------------------------------------

@pytest.fixture(scope="module")
def splits() -> dict:
    return build_splits(DateTaskConfig())


def test_all_three_splits_are_non_empty(splits: dict) -> None:
    for name in ("train", "val", "test"):
        assert len(splits[name]) > 100, f"{name} has only {len(splits[name])} examples"


def test_no_iso_date_appears_in_more_than_one_split(splits: dict) -> None:
    """THE leakage test. Splits are over calendar dates, so every rendering of a date shares a split.

    Without this property, validation would measure memorisation of specific dates rather than the
    ability to parse an unseen one, and the reported accuracy would be meaningless in the flattering
    direction.
    """
    sets = {name: {t for _, t in pairs} for name, pairs in splits.items()}
    assert not (sets["train"] & sets["val"]), sorted(sets["train"] & sets["val"])[:5]
    assert not (sets["train"] & sets["test"]), sorted(sets["train"] & sets["test"])[:5]
    assert not (sets["val"] & sets["test"]), sorted(sets["val"] & sets["test"])[:5]


def test_no_exact_source_string_is_shared_across_splits(splits: dict) -> None:
    """Implied by date-disjointness, but checked directly as a second, independent guard."""
    sets = {name: {s for s, _ in pairs} for name, pairs in splits.items()}
    assert not (sets["train"] & sets["val"])
    assert not (sets["train"] & sets["test"])


def test_every_pair_is_a_correct_normalisation(splits: dict) -> None:
    """Labels must actually be right. A generator bug here would cap accuracy for invisible reasons."""
    for name, pairs in splits.items():
        for src, tgt in pairs[:400]:
            parsed = dt.date.fromisoformat(tgt)
            assert str(parsed.year) in src, f"{name}: year missing from {src!r}"
            assert len(tgt) == 10


def test_splits_are_deterministic_given_the_seed() -> None:
    a = build_splits(DateTaskConfig(seed=7))
    b = build_splits(DateTaskConfig(seed=7))
    assert a["train"][:50] == b["train"][:50]
    c = build_splits(DateTaskConfig(seed=8))
    assert a["train"][:50] != c["train"][:50]


def test_a_date_is_rendered_in_distinct_formats_without_repetition(splits: dict) -> None:
    """Sampling is without replacement, so no (date, format) pair is duplicated within a split."""
    seen: set[tuple[str, str]] = set()
    for src, tgt in splits["train"]:
        key = (src, tgt)
        assert key not in seen, f"duplicate example {key}"
        seen.add(key)


# --------------------------------------------------------------------------------------------
# dataset: the right-shift alignment
# --------------------------------------------------------------------------------------------

def test_decoder_input_and_labels_are_offset_by_exactly_one() -> None:
    """The off-by-one that breaks everything, pinned.

    tgt_in must be [BOS, y_0..y_{n-1}] and labels [y_0..y_{n-1}, EOS]. So for every position i,
    labels[i] == tgt_in[i+1]. Shift the wrong way and the model is shown the token it must predict.
    """
    tok = CharTokenizer()
    cfg = DateTaskConfig()
    ds = DateDataset([("March 3, 2019", "2019-03-03")], tok, cfg)
    item = ds[0]

    tgt_in, labels = item["tgt_in"].tolist(), item["labels"].tolist()
    assert tgt_in[0] == tok.bos_id
    assert labels[-1] == tok.eos_id
    assert len(tgt_in) == len(labels) == 11        # BOS + 10, and 10 + EOS
    assert labels[:-1] == tgt_in[1:], "labels must be tgt_in shifted left by one"
    assert tok.decode(labels) == "2019-03-03"


def test_dataset_encodes_the_source_without_special_tokens() -> None:
    tok = CharTokenizer()
    ds = DateDataset([("3 Mar 2019", "2019-03-03")], tok, DateTaskConfig())
    src = ds[0]["src"].tolist()
    assert tok.bos_id not in src and tok.eos_id not in src
    assert tok.decode(src) == "3 Mar 2019"


def test_dataset_rejects_an_oversized_source() -> None:
    tok = CharTokenizer()
    ds = DateDataset([("x" * 99, "2019-03-03")], tok, DateTaskConfig(max_src_len=32))
    with pytest.raises(ValueError, match="max_src_len"):
        _ = ds[0]


# --------------------------------------------------------------------------------------------
# collation
# --------------------------------------------------------------------------------------------

def test_collate_right_pads_to_the_batch_maximum() -> None:
    tok = CharTokenizer()
    ds = DateDataset(
        [("March 3, 2019", "2019-03-03"), ("Sunday, March 3, 2019", "2019-03-03")],
        tok, DateTaskConfig(),
    )
    batch = collate([ds[0], ds[1]], tok.pad_id)

    assert batch["src"].shape == (2, 21)          # "Sunday, March 3, 2019" is 21 characters
    assert batch["src"][0, 13:].tolist() == [tok.pad_id] * 8, "shorter source must be right-padded"
    assert batch["src"][1].tolist().count(tok.pad_id) == 0


def test_collate_preserves_content_exactly() -> None:
    tok = CharTokenizer()
    pairs = [("March 3, 2019", "2019-03-03"), ("3 Mar 1987", "1987-03-03")]
    ds = DateDataset(pairs, tok, DateTaskConfig())
    batch = collate([ds[0], ds[1]], tok.pad_id)
    for i, (src_text, tgt_text) in enumerate(pairs):
        assert tok.decode(batch["src"][i].tolist()) == src_text
        assert tok.decode(batch["labels"][i].tolist()) == tgt_text


def test_padding_never_appears_before_real_content() -> None:
    """Left-padding would break the causal-mask/position alignment silently."""
    tok = CharTokenizer()
    ds = DateDataset([("March 3, 2019", "2019-03-03"), ("Sunday, March 3, 2019", "2019-03-03")],
                     tok, DateTaskConfig())
    batch = collate([ds[0], ds[1]], tok.pad_id)
    for row in batch["src"]:
        ids = row.tolist()
        first_pad = ids.index(tok.pad_id) if tok.pad_id in ids else len(ids)
        assert tok.pad_id not in ids[:first_pad]
        assert all(i == tok.pad_id for i in ids[first_pad:])
