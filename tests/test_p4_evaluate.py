"""Tests for the generation-side degradation detectors.

These detectors are the metric the project reports, so they need to be right and their limits need to
be pinned down by tests rather than only described in prose. Each detector has a test for the
behaviour it claims *and* a test for a case where it is known to be misleading — those second tests
exist so the limitation cannot be quietly forgotten when writing the project page.
"""

from __future__ import annotations

import pytest
import torch

from labs.p2_gpt import GPT, GPTConfig
from labs.p4_dpo import make_reference_model
from labs.p4_dpo.evaluate import (
    ends_mid_sentence,
    preference_metrics,
    repetition_score,
    score_generation,
    topic_term_overlap,
)


# --------------------------------------------------------------------------------------------
# repetition
# --------------------------------------------------------------------------------------------

def test_repetition_score_is_zero_for_varied_prose() -> None:
    text = ("Typography arranges type on a page. Kerning adjusts pairs of letters. A grid divides "
            "the surface into columns. Colour choices carry meaning for the reader.")
    assert repetition_score(text) == pytest.approx(0.0)


def test_repetition_score_approaches_one_for_a_pure_loop() -> None:
    text = " ".join(["the quick brown fox jumps"] * 12)
    assert repetition_score(text) > 0.85


def test_repetition_score_rises_monotonically_with_repeats() -> None:
    base = "designers choose a typeface for its voice and its legibility on the page"
    scores = [repetition_score(" ".join([base] * k)) for k in (1, 2, 4, 8)]
    assert scores == sorted(scores)
    assert scores[0] == pytest.approx(0.0)


def test_repetition_score_is_zero_for_text_shorter_than_one_window() -> None:
    """Returns 0.0 rather than raising: too short to judge, and a crash here would be worse."""
    assert repetition_score("two words") == 0.0
    assert repetition_score("") == 0.0


# --------------------------------------------------------------------------------------------
# truncation
# --------------------------------------------------------------------------------------------

def test_ends_mid_sentence_detects_a_cut_clause() -> None:
    assert ends_mid_sentence("Kerning adjusts the space between two")
    assert not ends_mid_sentence("Kerning adjusts the space between two letters.")


def test_ends_mid_sentence_accepts_other_sentence_final_marks() -> None:
    for ending in ('Is it legible?', 'It is legible!', 'He called it "legible."',
                   "she called it 'legible.'"):
        assert not ends_mid_sentence(ending), ending


def test_ends_mid_sentence_is_false_for_empty_text() -> None:
    """Empty output is a different failure and must not be reported as truncation."""
    assert not ends_mid_sentence("")
    assert not ends_mid_sentence("   \n ")


# --------------------------------------------------------------------------------------------
# topic overlap
# --------------------------------------------------------------------------------------------

def test_topic_overlap_is_high_for_the_same_subject() -> None:
    ref = "Kerning adjusts the optical space between individual letter pairs in a word."
    same = "Kerning changes the optical space between letter pairs."
    other = "Photolithography transfers circuit patterns onto silicon wafers using ultraviolet light."
    assert topic_term_overlap(same, ref) > topic_term_overlap(other, ref)


def test_topic_overlap_is_zero_for_disjoint_vocabulary() -> None:
    assert topic_term_overlap("aaaa bbbb cccc", "dddd eeee ffff") == 0.0


def test_topic_overlap_is_zero_when_either_side_has_no_content_words() -> None:
    assert topic_term_overlap("", "kerning letters") == 0.0
    assert topic_term_overlap("a of to in", "kerning letters") == 0.0


def test_topic_overlap_rewards_parroting_which_is_why_it_only_detects_drift() -> None:
    """A documented weakness, asserted so the project page cannot overstate the metric.

    Copying the reference verbatim scores a perfect 1.0. That makes the measure useful for spotting
    off-topic answers and useless as a quality score, which is exactly how it is used.
    """
    ref = "A grid divides the page into columns and rows for consistent placement."
    assert topic_term_overlap(ref, ref) == pytest.approx(1.0)


def test_score_generation_reports_all_detectors() -> None:
    out = score_generation("Kerning adjusts space between letters", "Kerning adjusts letter space.")
    assert set(out) == {"repetition_4gram", "ends_mid_sentence", "topic_overlap", "n_words"}
    assert out["ends_mid_sentence"] == 1.0
    assert out["n_words"] == 5.0


# --------------------------------------------------------------------------------------------
# scoring-side metrics
# --------------------------------------------------------------------------------------------

def make_batch(n: int = 4, width: int = 12, vocab: int = 40) -> dict:
    torch.manual_seed(0)
    mask = torch.zeros(n, width)
    mask[:, 5:] = 1
    return {
        "chosen_ids": torch.randint(0, vocab, (n, width)),
        "rejected_ids": torch.randint(0, vocab, (n, width)),
        "chosen_mask": mask.clone(),
        "rejected_mask": mask.clone(),
        "degradations": ["off_topic", "truncated", "repetitive", "generic"][:n],
        "topics": [f"t{i}" for i in range(n)],
    }


def test_preference_metrics_are_chance_level_at_initialisation() -> None:
    """A model equal to its reference has zero margin and zero log-ratio, by construction."""
    torch.manual_seed(0)
    model = GPT(GPTConfig(vocab_size=40, block_size=32, n_layer=2, n_head=4, d_model=32,
                          dropout=0.0, attention_dropout=0.0)).eval()
    ref = make_reference_model(model)

    m = preference_metrics(model, ref, [make_batch()], beta=0.1)
    assert m["mean_implicit_reward_margin"] == pytest.approx(0.0, abs=1e-5)
    assert m["mean_logratio_vs_reference"] == pytest.approx(0.0, abs=1e-5)
    assert 0.0 <= m["preference_accuracy"] <= 1.0
    assert m["n_pairs"] == 4


def test_preference_metrics_break_out_accuracy_by_degradation_type() -> None:
    """Per-type accuracy is what shows *which* failure the model learned to reject."""
    torch.manual_seed(0)
    model = GPT(GPTConfig(vocab_size=40, block_size=32, n_layer=2, n_head=4, d_model=32,
                          dropout=0.0, attention_dropout=0.0)).eval()
    ref = make_reference_model(model)
    m = preference_metrics(model, ref, [make_batch()], beta=0.1)

    assert set(m["accuracy_by_degradation"]) == {"off_topic", "truncated", "repetitive", "generic"}
    assert sum(v["n"] for v in m["accuracy_by_degradation"].values()) == 4


def test_preference_metrics_report_raw_and_length_normalised_accuracy() -> None:
    """Both are required: their difference is the length bias the summed objective introduces."""
    torch.manual_seed(0)
    model = GPT(GPTConfig(vocab_size=40, block_size=32, n_layer=2, n_head=4, d_model=32,
                          dropout=0.0, attention_dropout=0.0)).eval()
    ref = make_reference_model(model)
    m = preference_metrics(model, ref, [make_batch()], beta=0.1)
    assert "preference_accuracy" in m
    assert "preference_accuracy_length_normalised" in m
