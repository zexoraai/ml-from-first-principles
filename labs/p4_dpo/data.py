"""Preference data for Project 4, built on top of the shared instruction builder.

The instruction half — extracting `{topic: passage}` from the corpus headings and splitting by topic —
lives in `labs/common/instructions.py`, because Project 3 needs the same fine-tuning task (D-005:
promote on the second caller). This module adds the part that is specific to preference optimisation:
the degradations, the pairs, and the provenance manifest.

PROVENANCE — read this before believing any number from this project
--------------------------------------------------------------------
There is **no human preference data here.** Collecting it properly needs annotators, an interface,
inter-annotator agreement, and a budget. Claiming to have it would be the single most dishonest thing
this portfolio could do, so instead the preferences are **constructed by a documented, deterministic
rule**, and the rule is stated on the project page.

The rule: for a prompt derived from a design topic,

    chosen   = the corpus passage that sits under that topic heading
    rejected = one of four documented degradations

| Degradation | Models | Why a 2.9 M-parameter model can learn it |
|---|---|---|
| `off_topic` | answering a different question | different vocabulary distribution |
| `truncated` | stopping mid-thought | ends without sentence-final punctuation |
| `repetitive` | degenerate looping | n-gram repetition, the classic sampling failure |
| `generic` | vacuous filler that answers nothing | template phrasing, no domain terms |

WHY THIS IS DEFENSIBLE, AND WHERE IT IS NOT
-------------------------------------------
Defensible: the preference signal is **real and verifiable** — on-topic, complete, non-repetitive prose
genuinely is preferable to its degradations, the labels are correct by construction rather than by
opinion, and a small model can actually learn the distinction. That makes it a valid test of whether
*the DPO mechanism works*, which is what tier E claims.

Not defensible, and not claimed: it measures nothing about human values, helpfulness, harmlessness or
truthfulness. A model that has learned "prefer on-topic complete prose" has learned exactly that. The
project page says so in the same breath as reporting the result.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from labs.common.instructions import (
    PROMPT_TEMPLATES,
    InstructionExample,
    build_instruction_split,
    split_sentences,
)

__all__ = [
    "InstructionExample",
    "PreferencePair",
    "build_sft_and_preferences",
    "DEGRADATIONS",
    "PROMPT_TEMPLATES",
]

DEGRADATIONS = ("off_topic", "truncated", "repetitive", "generic")

GENERIC_FILLERS = (
    "This is an important topic in design. There are many things to consider. "
    "Designers should think carefully about it. It depends on the situation and the goals of the "
    "project. Many books have been written about it.",
    "It is a matter of taste and experience. Some people prefer one approach and others prefer "
    "another. The best choice depends on the context. There is no single correct answer.",
    "This concept is widely used. It has a long history and continues to be relevant today. "
    "Understanding it is useful for anyone working in the field.",
)


@dataclass
class PreferencePair:
    prompt: str
    chosen: str
    rejected: str
    topic: str
    degradation: str


def _degrade(kind: str, chosen: str, other: str, rng: random.Random) -> str:
    """Produce a rejected response of the named kind. Deterministic given `rng`."""
    if kind == "off_topic":
        return other

    if kind == "truncated":
        words = chosen.split()
        lo = max(4, int(len(words) * 0.25))
        hi = max(lo, min(int(len(words) * 0.55), len(words) - 1))
        cut = min(rng.randint(lo, hi) if hi > lo else lo, len(words) - 1)

        # A cut that happens to land on a sentence boundary yields a COMPLETE short answer, not a
        # truncated one -- so the `truncated` label would be false, and the pair would be teaching the
        # model to disprefer perfectly well-formed brief prose. Walk the cut forward until it lands
        # mid-sentence. (Bug #12, found by test_truncated_rejections_end_mid_sentence after the word
        # budget was reduced to fit block_size=192. At the original 130-word budget the first sentence
        # boundary always fell past the 25-55% window, so the bug was latent rather than absent -- a
        # reminder that changing a data parameter can activate a dormant defect elsewhere.)
        while cut < len(words) - 1 and words[cut - 1].endswith((".", "!", "?")):
            cut += 1
        out = " ".join(words[:cut])
        if out.endswith((".", "!", "?")):
            # No mid-sentence point existed inside the budget (a single-sentence passage). Drop the
            # terminal punctuation so the degradation stays detectable.
            out = out.rstrip(".!?")
        return out

    if kind == "repetitive":
        sents = split_sentences(chosen)
        if not sents:
            return chosen
        # Degenerate looping, which is exactly what unbounded sampling produces.
        return " ".join([sents[0]] * 4)

    if kind == "generic":
        return rng.choice(GENERIC_FILLERS)

    raise ValueError(f"unknown degradation {kind!r}")


def build_sft_and_preferences(
    corpus_text: str,
    *,
    seed: int = 4242,
    eval_topic_fraction: float = 0.15,
    pairs_per_topic: int = 2,
) -> dict:
    """Build SFT examples and preference pairs, split by **topic**.

    Returns `sft_train`, `sft_eval`, `pref_train`, `pref_eval`, `passages`, and a `manifest` recording
    exactly how everything was produced.

    Both arms of Project 4's comparison use the same SFT set, so the DPO arm's only advantage is the
    preference optimisation itself. Evaluation topics appear in no training example of either kind, and
    that is asserted below rather than assumed.
    """
    split = build_instruction_split(
        corpus_text, seed=seed, eval_topic_fraction=eval_topic_fraction
    )
    passages = split["passages"]
    rng = split["rng"]        # continue the same stream, so one seed reproduces the whole build

    def make_pairs(topic_list: list[str]) -> list[PreferencePair]:
        out = []
        for topic in topic_list:
            for _ in range(pairs_per_topic):
                kind = rng.choice(DEGRADATIONS)
                # For off_topic, draw a passage from a DIFFERENT topic in the SAME split, so the
                # rejected text is real prose. The distinction the model must learn is relevance, not
                # fluency -- pairing against noise would make the task trivially easy and the result
                # meaningless.
                other = topic
                while other == topic and len(topic_list) > 1:
                    other = rng.choice(topic_list)
                out.append(PreferencePair(
                    prompt=rng.choice(PROMPT_TEMPLATES).format(topic=topic),
                    chosen=passages[topic] + "\n",
                    rejected=_degrade(kind, passages[topic], passages[other], rng) + "\n",
                    topic=topic,
                    degradation=kind,
                ))
        return out

    pref_train = make_pairs(split["train_topics"])
    pref_eval = make_pairs(split["eval_topics"])
    sft_train, sft_eval = split["sft_train"], split["sft_eval"]

    # Leakage guard, asserted rather than assumed.
    train_set = {e.topic for e in sft_train} | {p.topic for p in pref_train}
    eval_set = {e.topic for e in sft_eval} | {p.topic for p in pref_eval}
    if train_set & eval_set:
        raise AssertionError(f"topic leakage between train and eval: {sorted(train_set & eval_set)[:5]}")

    manifest = {
        "source": "the Project 2 design corpus TRAIN split (see data/p2/attribution.json for licences)",
        "preference_origin": (
            "CONSTRUCTED, NOT HUMAN-ANNOTATED. chosen = the corpus passage under a topic heading; "
            "rejected = a documented degradation of it."
        ),
        "degradations": {
            "off_topic": "a real passage about a different topic — tests relevance, not fluency",
            "truncated": "cut to 25-55% of its words, forced to end mid-sentence",
            "repetitive": "first sentence repeated four times — models degenerate looping",
            "generic": "vacuous filler containing no domain terms",
        },
        "split_unit": "topic — every prompt about a topic lands in exactly one split",
        "leakage_check": "build_sft_and_preferences raises if the train and eval topic sets intersect",
        "n_topics": len(passages),
        "n_train_topics": len(split["train_topics"]),
        "n_eval_topics": len(split["eval_topics"]),
        "n_sft_train": len(sft_train),
        "n_sft_eval": len(sft_eval),
        "n_pref_train": len(pref_train),
        "n_pref_eval": len(pref_eval),
        "pairs_per_topic": pairs_per_topic,
        "seed": seed,
        "prompt_templates": list(PROMPT_TEMPLATES),
        "what_this_measures": (
            "whether the DPO mechanism moves a policy toward a verifiable, rule-defined preference. "
            "It measures NOTHING about human values, helpfulness, harmlessness or truthfulness."
        ),
        "degradation_counts_train": {
            k: sum(1 for p in pref_train if p.degradation == k) for k in DEGRADATIONS
        },
    }

    return {
        "sft_train": sft_train, "sft_eval": sft_eval,
        "pref_train": pref_train, "pref_eval": pref_eval,
        "manifest": manifest, "passages": passages,
    }
