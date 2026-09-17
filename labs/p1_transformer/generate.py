"""Autoregressive decoding.

The paper uses beam search with beam 4 and length penalty 0.6 (section 6.1). We implement **greedy**
decoding, and say so rather than implying beam search. Greedy is the right default here for two
reasons: the target task has a single correct output string, so there is little for a beam to
explore; and greedy makes the demo's per-step probability inspection interpretable, because the
displayed distribution is exactly the one the chosen token came from.
"""

from __future__ import annotations

import torch

from .model import Transformer

__all__ = ["greedy_decode", "decode_step_trace"]


@torch.no_grad()
def greedy_decode(
    model: Transformer,
    src: torch.Tensor,
    *,
    max_new_tokens: int = 32,
    bos_id: int | None = None,
    eos_id: int | None = None,
) -> torch.Tensor:
    """Decode greedily: at each step emit the argmax token, then feed it back in.

    Args:
        model: a `Transformer` in eval mode. (Not forced here -- an accidental train-mode decode
            would apply dropout and produce non-deterministic output, so `assert` on it instead of
            silently fixing it, because silently fixing it hides the caller's bug.)
        src: (batch, src_len) source token ids.
        max_new_tokens: hard cap on generated length. Required, not optional: a model that never
            emits EOS would otherwise loop forever, and an untrained model never emits EOS.

    Returns:
        (batch, generated_len) token ids **including** the leading BOS.

    Cost, stated honestly
    ---------------------
    This re-runs the whole decoder over the full prefix at every step, so producing n tokens costs
    O(n^2) decoder work. A KV cache would make it O(n) by reusing the keys and values of tokens
    that have not changed. It is not implemented here, and no claim of efficient generation is
    made: at our sequence lengths (<= 32) the quadratic term is irrelevant, and adding a cache
    would introduce a second, subtly different code path for the demo to disagree with.

    The encoder, however, *is* computed once and reused. That is not an optimisation but a
    structural fact: every decoder layer attends to the same final encoder output.
    """
    if model.training:
        raise RuntimeError(
            "greedy_decode called on a model in training mode: dropout would be active and the "
            "output non-deterministic. Call model.eval() first."
        )

    cfg = model.cfg
    bos_id = cfg.bos_id if bos_id is None else bos_id
    eos_id = cfg.eos_id if eos_id is None else eos_id
    batch = src.size(0)
    device = src.device

    memory = model.encode(src)                                        # computed exactly once
    ys = torch.full((batch, 1), bos_id, dtype=torch.long, device=device)
    finished = torch.zeros(batch, dtype=torch.bool, device=device)

    for _ in range(max_new_tokens):
        hidden = model.decoder(
            ys, memory,
            self_keep_mask=model.target_keep_mask(ys),
            cross_keep_mask=model.source_keep_mask(src),
        )
        # Only the LAST position's logits matter: that is the distribution over the next token.
        logits = model.generator(hidden[:, -1])                       # (batch, vocab)
        next_token = logits.argmax(dim=-1)                            # (batch,)

        # Once a sequence has emitted EOS, keep appending pad so the tensor stays rectangular
        # without letting a finished sequence carry on generating real tokens.
        next_token = torch.where(finished, torch.full_like(next_token, cfg.pad_id), next_token)
        ys = torch.cat([ys, next_token.unsqueeze(1)], dim=1)
        finished = finished | (next_token == eos_id)
        if bool(finished.all()):
            break

    return ys


@torch.no_grad()
def decode_step_trace(
    model: Transformer,
    src: torch.Tensor,
    *,
    max_new_tokens: int = 32,
    top_k: int = 5,
) -> list[dict]:
    """Greedy decode a **single** sequence, recording the top-k distribution at every step.

    This exists for the demo requirement "token probabilities at selected steps" and for failure
    diagnosis: a wrong output is far easier to understand when you can see whether the model was
    confidently wrong or nearly right.

    Returns a list with one dict per step: `step`, `chosen` (token id), `top_k` (list of
    (token_id, probability)), and `entropy` of the full distribution in nats.

    Entropy is included because it distinguishes two very different failures that look identical in
    the output: a confident wrong prediction (low entropy, the model has learned something wrong)
    versus an undecided one (high entropy, the model has learned nothing here yet).
    """
    if model.training:
        raise RuntimeError("decode_step_trace requires eval mode; call model.eval() first.")
    if src.size(0) != 1:
        raise ValueError(f"trace expects a batch of exactly 1, got {src.size(0)}")

    cfg = model.cfg
    memory = model.encode(src)
    ys = torch.full((1, 1), cfg.bos_id, dtype=torch.long, device=src.device)
    trace: list[dict] = []

    for step in range(max_new_tokens):
        hidden = model.decoder(
            ys, memory,
            self_keep_mask=model.target_keep_mask(ys),
            cross_keep_mask=model.source_keep_mask(src),
        )
        logits = model.generator(hidden[:, -1])[0]                    # (vocab,)
        probs = torch.softmax(logits, dim=-1)
        k = min(top_k, probs.numel())
        top_p, top_i = probs.topk(k)
        chosen = int(logits.argmax())

        # Entropy over the full distribution, with a floor to keep log finite at exact zeros.
        entropy = float(-(probs * torch.log(probs.clamp_min(1e-12))).sum())

        trace.append({
            "step": step,
            "chosen": chosen,
            "top_k": [(int(i), float(p)) for i, p in zip(top_i, top_p)],
            "entropy_nats": entropy,
        })

        ys = torch.cat([ys, torch.tensor([[chosen]], device=src.device)], dim=1)
        if chosen == cfg.eos_id:
            break

    return trace
