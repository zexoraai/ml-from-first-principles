"""Evaluating a preference-optimised model.

FOUR MEASUREMENTS, AND WHAT EACH ONE IS ACTUALLY EVIDENCE OF
------------------------------------------------------------
1. **Preference accuracy** — does the model score `chosen` above `rejected`? Cheap, low variance, and
   directly the quantity DPO optimises. Its weakness is that it is a *scoring* test: it asks the model
   to rank two texts it was handed, not to produce anything. A model can rank well and still generate
   badly, so this alone is not evidence of improved generation.

2. **Implicit reward margin and KL from the reference** — how far the policy moved, and in which
   direction. This is the diagnostic that makes the β sweep interpretable: it shows that β genuinely
   trades preference gain against divergence, rather than asserting it.

3. **Generation-side degradation rates** — sample from the model and count, with deterministic
   detectors, how often the output exhibits the failures the preferences were built to suppress
   (repetition, truncation, off-topic drift). This is the measurement that speaks to behaviour rather
   than ranking, and it is the one that can embarrass the method. It is reported either way.

4. **Held-out perplexity on the pretraining corpus** — the alignment-tax check. Preference
   optimisation can improve preference metrics while degrading general language modelling. Reporting
   the base model's perplexity alongside each arm's makes that cost visible instead of unmeasured.

WHY THE DETECTORS ARE RULE-BASED AND NOT A JUDGE MODEL
-----------------------------------------------------
An LLM judge would give more human-like scores and would also be unauditable, non-reproducible without
an API, and impossible to defend when asked "what exactly did it measure?". These detectors are the
same rules that *constructed* the degradations, applied in reverse. That symmetry is the point: the
preference signal and the metric are the same definition, so a change in the metric is a change in the
thing that was optimised. It also caps what may be claimed — these detect surface failures, not quality.
"""

from __future__ import annotations

import math
import re

import torch

from .dpo import sequence_logprobs

__all__ = [
    "repetition_score",
    "ends_mid_sentence",
    "topic_term_overlap",
    "score_generation",
    "preference_metrics",
]


def repetition_score(text: str, *, n: int = 4) -> float:
    """Fraction of `n`-grams that are duplicates. 0.0 = no repetition, → 1.0 = degenerate looping.

    Measured over word `n`-grams with `n=4`, which is long enough that natural English rarely repeats
    a window by chance and short enough to catch a looping model within a couple of sentences.
    Sequences shorter than one window score 0.0 rather than raising: a model that emitted three words
    has other problems, and this metric has nothing to say about them.
    """
    words = text.split()
    if len(words) < n:
        return 0.0
    grams = [" ".join(words[i : i + n]) for i in range(len(words) - n + 1)]
    return 1.0 - len(set(grams)) / len(grams)


def ends_mid_sentence(text: str) -> bool:
    """True if the text stops without sentence-final punctuation.

    A generation that hits its token limit mid-clause is the `truncated` failure. Note the confound,
    which is stated on the project page: a fixed generation budget truncates *any* model that has not
    learned to stop, so this rate is only meaningful when compared across arms at the same budget.
    """
    stripped = text.strip()
    return bool(stripped) and not stripped.endswith((".", "!", "?", '"', "'"))


_WORD = re.compile(r"[a-z]{4,}")


def topic_term_overlap(response: str, reference: str) -> float:
    """Jaccard overlap of content words between a generated response and the reference passage.

    A crude relevance proxy, and crude in a specific way worth naming: it rewards vocabulary reuse, so
    a model that parrots the reference scores 1.0. It is therefore only used to detect *off-topic
    drift* — a near-zero score means the model answered a different question — and never as a quality
    score. Words shorter than four characters are dropped so the measure is not dominated by function
    words shared by all English text.
    """
    a = set(_WORD.findall(response.lower()))
    b = set(_WORD.findall(reference.lower()))
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def score_generation(response: str, reference: str) -> dict[str, float]:
    """All three generation-side detectors for one sample."""
    return {
        "repetition_4gram": repetition_score(response),
        "ends_mid_sentence": float(ends_mid_sentence(response)),
        "topic_overlap": topic_term_overlap(response, reference),
        "n_words": float(len(response.split())),
    }


@torch.no_grad()
def preference_metrics(model, reference, batches, *, beta: float) -> dict:
    """Scoring-side metrics over pre-encoded evaluation batches.

    Reports accuracy twice, raw and length-normalised, because DPO's objective uses the **sum** of
    token log-probabilities and therefore carries a length bias: `rejected` responses of the
    `truncated` kind are shorter, so they have fewer negative terms and can score higher for reasons
    that have nothing to do with preference. Reporting only the raw number would let that bias
    masquerade as skill; reporting only the normalised one would hide what the objective actually did.
    The gap between the two *is* the length bias, measured.
    """
    model.eval()
    n = 0
    raw_correct = norm_correct = 0
    margin_sum = kl_sum = 0.0
    per_type: dict[str, list[int]] = {}

    for batch in batches:
        # Four forward passes, not six: the policy's logits are reused for both the summed and the
        # length-normalised score. Recomputing them would double the dominant cost of evaluation to
        # produce bit-identical tensors.
        pc_logits = model(batch["chosen_ids"])[0]
        pr_logits = model(batch["rejected_ids"])[0]

        pc = sequence_logprobs(pc_logits, batch["chosen_ids"], batch["chosen_mask"])
        pr = sequence_logprobs(pr_logits, batch["rejected_ids"], batch["rejected_mask"])
        nc = sequence_logprobs(pc_logits, batch["chosen_ids"], batch["chosen_mask"], average=True)
        nr = sequence_logprobs(pr_logits, batch["rejected_ids"], batch["rejected_mask"],
                               average=True)

        rc = sequence_logprobs(reference(batch["chosen_ids"])[0], batch["chosen_ids"],
                               batch["chosen_mask"])
        rr = sequence_logprobs(reference(batch["rejected_ids"])[0], batch["rejected_ids"],
                               batch["rejected_mask"])

        raw_hits = (pc > pr)
        raw_correct += int(raw_hits.sum())
        norm_correct += int((nc > nr).sum())
        margin_sum += float((beta * ((pc - rc) - (pr - rr))).sum())
        # A one-sided estimate of how far the policy moved on this data: E[log pi - log pi_ref] over
        # the chosen responses. Not a symmetric KL and not called one -- it is the sequence-level
        # log-ratio, which is exactly the quantity beta penalises.
        kl_sum += float((pc - rc).sum())
        n += int(pc.numel())

        for kind, hit in zip(batch["degradations"], raw_hits.tolist()):
            per_type.setdefault(kind, []).append(int(hit))

    return {
        "n_pairs": n,
        "preference_accuracy": raw_correct / max(n, 1),
        "preference_accuracy_length_normalised": norm_correct / max(n, 1),
        "mean_implicit_reward_margin": margin_sum / max(n, 1),
        "mean_logratio_vs_reference": kl_sum / max(n, 1),
        "accuracy_by_degradation": {
            k: {"n": len(v), "accuracy": sum(v) / len(v)} for k, v in sorted(per_type.items())
        },
    }


@torch.no_grad()
def corpus_perplexity(model, get_batch, *, iters: int = 100) -> float:
    """Held-out perplexity on the pretraining distribution — the alignment-tax measurement."""
    model.eval()
    total = 0.0
    for _ in range(iters):
        x, y = get_batch("val")
        total += float(model(x, y)[1])
    return math.exp(min(total / max(iters, 1), 20))
