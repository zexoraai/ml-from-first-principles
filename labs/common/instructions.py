"""Building instruction-following data out of a corpus's heading structure.

Promoted here from `labs/p4_dpo/data.py` under decision **D-005** — promote to `labs/common/` on the
second caller. The callers are:

* **Project 3 (LoRA)** — needs a fine-tuning task to compare LoRA against a full fine-tune on.
* **Project 4 (DPO)** — needs the same task for its SFT stage, plus preference pairs built on top.

Sharing the builder is not only about avoiding duplication. It means P3's full-fine-tune arm and P4's
SFT arm are *the same computation on the same data*, so the two projects' numbers can be read next to
each other instead of being two unrelated experiments that happen to use the same base model.

HOW THE TASK IS DERIVED, WITH NO LABELLING
------------------------------------------
`corpus.py` writes the Wikipedia portion of the design corpus with `# Title` and `## Section` markers.
That structure is the supervision: the heading *is* a topic and the prose beneath it *is* an answer
about that topic. Wrapping the pair in an instruction template turns a plain text corpus into an
instruction dataset without a single human annotation.

What that buys and what it does not: the responses are real, licensed, on-topic prose, so the task is
genuine language modelling conditioned on an instruction. But they are encyclopaedia passages, not
answers written to be helpful to a person asking a question. A model that fits this data has learned to
continue reference prose on cue — which is the claim made, and nothing more.
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass

__all__ = [
    "InstructionExample",
    "PROMPT_TEMPLATES",
    "split_sentences",
    "extract_topic_passages",
    "build_instruction_split",
]

# Few and plain on purpose. With a 2.9 M-parameter model, template variety consumes capacity that is
# better spent on the answer. The `###` markers give an unambiguous, learnable boundary between
# instruction and response -- the role a chat template plays at scale.
PROMPT_TEMPLATES = (
    "### Instruction:\nExplain {topic} in graphic design.\n\n### Response:\n",
    "### Instruction:\nWhat is {topic}?\n\n### Response:\n",
    "### Instruction:\nDescribe the role of {topic} in design.\n\n### Response:\n",
    "### Instruction:\nWrite about {topic}.\n\n### Response:\n",
)


@dataclass
class InstructionExample:
    prompt: str
    response: str
    topic: str

    @property
    def text(self) -> str:
        return self.prompt + self.response


def split_sentences(text: str) -> list[str]:
    """Split into sentences well enough for this purpose, not well enough for linguistics.

    A real splitter would handle abbreviations, quotations and decimals. Here the only consequence of a
    bad split is a slightly odd truncation point, so the simple version is correct enough — and the
    limitation is stated rather than hidden behind a dependency.
    """
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p.strip() for p in parts if len(p.strip()) > 20]


def extract_topic_passages(corpus_text: str, *, min_words: int = 28,
                           max_words: int = 58) -> dict[str, str]:
    """Pull `{topic: passage}` out of a corpus that uses Markdown headings.

    Why the word budget is small
    ---------------------------
    `max_words=58` is not an aesthetic choice. The Project 2 model has `block_size=192` with
    **learned** position embeddings, so 192 tokens is a hard architectural ceiling — there is no
    embedding row beyond it. At the corpus's measured 2.62 chars/token, 58 words of English is roughly
    350 characters and about 135 tokens, leaving room for the instruction wrapper. Extracting the
    130-word passages that read better as prose would produce sequences the model physically cannot
    represent, and the failure would surface as a shape error deep in training rather than here.

    Passages are always trimmed to a **sentence boundary**, so every response is a complete thought.
    Project 4 depends on this: its `truncated` degradation is only detectable if the undegraded text is
    known to be complete.
    """
    passages: dict[str, str] = {}
    current_topic: str | None = None
    buffer: list[str] = []

    def flush() -> None:
        if not current_topic or not buffer:
            return
        words = " ".join(buffer).strip().split()
        if len(words) < min_words:
            return
        sents = split_sentences(" ".join(words[:max_words]))
        if not sents:
            return
        text = " ".join(sents)
        if not text.endswith((".", "!", "?")):
            text = " ".join(sents[:-1]) if len(sents) > 1 else text + "."
        if len(text.split()) >= min_words and current_topic not in passages:
            passages[current_topic] = text

    for line in corpus_text.split("\n"):
        stripped = line.strip()
        heading = re.fullmatch(r"#{1,2}\s+(.+)", stripped)
        if heading:
            flush()
            title = heading.group(1).strip()
            # Section headings are often generic ("History", "Overview") and make poor topics alone;
            # only accept ones that read like a subject.
            current_topic = title if 3 <= len(title) <= 48 else None
            buffer = []
            continue
        if stripped and current_topic:
            buffer.append(stripped)
    flush()
    return passages


def build_instruction_split(
    corpus_text: str,
    *,
    seed: int = 4242,
    eval_topic_fraction: float = 0.15,
    min_topics: int = 40,
) -> dict:
    """Extract passages and split them into train/eval instruction sets **by topic**.

    Returns `{"sft_train", "sft_eval", "train_topics", "eval_topics", "passages", "rng"}`.

    Splitting by topic, not by example
    ----------------------------------
    Every prompt about a topic lands in exactly one split. Splitting by *example* would put
    "Explain kerning" in train and "What is kerning?" in eval, where the correct answer is the same
    passage the model already memorised — measuring recall and reporting it as generalisation. This is
    the same failure Project 1 avoided with its date split, avoided the same way.

    The returned `rng` is the live generator, so a caller (Project 4) can continue drawing from the
    same stream and keep the whole construction reproducible from one seed.
    """
    rng = random.Random(seed)
    passages = extract_topic_passages(corpus_text)
    if len(passages) < min_topics:
        raise RuntimeError(
            f"only {len(passages)} topic passages extracted — too few to split into train/eval. "
            f"The corpus heading structure has probably changed; fix the extractor rather than "
            f"lowering the threshold."
        )

    topics = sorted(passages)
    rng.shuffle(topics)
    n_eval = max(8, int(len(topics) * eval_topic_fraction))
    eval_topics = sorted(topics[:n_eval])
    train_topics = [t for t in topics if t not in set(eval_topics)]

    def make(topic_list: list[str]) -> list[InstructionExample]:
        return [
            InstructionExample(prompt=rng.choice(PROMPT_TEMPLATES).format(topic=t),
                               response=passages[t] + "\n", topic=t)
            for t in topic_list
        ]

    return {
        "sft_train": make(train_topics),
        "sft_eval": make(eval_topics),
        "train_topics": train_topics,
        "eval_topics": eval_topics,
        "passages": passages,
        "rng": rng,
    }
