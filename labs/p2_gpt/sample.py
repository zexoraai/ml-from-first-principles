"""Sampling from a trained GPT: temperature, top-k, top-p, and per-step inspection.

Each control changes the distribution in a different way, and conflating them is a common source of
confused reasoning about "creativity". Stated precisely:

**Temperature** rescales the logits before softmax: `p ∝ exp(z / T)`.
  * `T -> 0` approaches argmax (fully deterministic).
  * `T = 1` is the model's own distribution, unmodified.
  * `T > 1` flattens it, raising the relative probability of unlikely tokens.
  It changes *how sharply* the model's own ranking is followed. It never changes the ranking itself,
  and it never removes any token from consideration.

**Top-k** keeps only the `k` highest-probability tokens and renormalises.
  This *truncates* the distribution. It removes the long tail entirely, which is what prevents the
  rare-but-catastrophic token that derails a generation. Note the interaction: `T = 2` with `k = 5`
  is not "more random" in the way `T = 2` alone is, because the tail it would have sampled is gone.

**Top-p / nucleus** keeps the smallest set of tokens whose cumulative probability reaches `p`.
  Adaptive where top-k is fixed: when the model is confident, the nucleus is one or two tokens; when
  it is uncertain, it widens. That is usually the behaviour people actually want from top-k.

Order of application here: temperature, then top-k, then top-p. Applying truncation before
temperature would let temperature reintroduce nothing, since the removed mass is already gone — so
the order matters and is not arbitrary.
"""

from __future__ import annotations

import torch

from .model import GPT

__all__ = ["generate", "generate_with_trace", "apply_sampling_filters"]


def apply_sampling_filters(
    logits: torch.Tensor,
    *,
    temperature: float = 1.0,
    top_k: int | None = None,
    top_p: float | None = None,
) -> torch.Tensor:
    """Transform raw logits into the distribution we will sample from.

    Args:
        logits: (B, vocab) raw scores for the next token.
    Returns:
        (B, vocab) probabilities that sum to 1 along the last axis.
    """
    if temperature < 0:
        raise ValueError("temperature must be >= 0")

    if temperature == 0.0:
        # The limit of T -> 0 is a point mass on the argmax. Computing it as a limit would divide by
        # zero, so handle it exactly instead of with a tiny epsilon.
        probs = torch.zeros_like(logits)
        probs.scatter_(1, logits.argmax(dim=-1, keepdim=True), 1.0)
        return probs

    logits = logits / temperature

    if top_k is not None and top_k > 0:
        k = min(top_k, logits.size(-1))
        top = logits.topk(k, dim=-1)
        # Select by INDEX, not by comparing against the k-th value.
        #
        # The obvious implementation is `masked_fill(logits < top.values[..., -1:], -inf)`, and it is
        # wrong whenever logits tie at the threshold: every tied token satisfies `>= threshold` and
        # survives, so `top_k=10` on a distribution like [10, 0, 0, ..., 0] returns 50 candidates.
        # That is not hypothetical — a freshly initialised model is nearly uniform, so ties at the
        # cut are the *normal* case early in training, and the bug silently disables top-k exactly
        # when it matters most. Scattering the k selected values into a -inf tensor keeps exactly k.
        # tests/test_p2_model.py::test_top_p_adapts_to_confidence_where_top_k_does_not caught this.
        filtered = torch.full_like(logits, float("-inf"))
        logits = filtered.scatter_(-1, top.indices, top.values)

    if top_p is not None and 0.0 < top_p < 1.0:
        sorted_logits, sorted_idx = logits.sort(dim=-1, descending=True)
        cumulative = sorted_logits.softmax(dim=-1).cumsum(dim=-1)
        # Remove tokens once the cumulative mass has already reached top_p. The shift keeps the
        # first token that crosses the threshold, so the nucleus is never empty even when a single
        # token already exceeds p.
        remove = cumulative > top_p
        remove[..., 1:] = remove[..., :-1].clone()
        remove[..., 0] = False
        sorted_logits = sorted_logits.masked_fill(remove, float("-inf"))
        logits = torch.full_like(logits, float("-inf")).scatter_(1, sorted_idx, sorted_logits)

    return logits.softmax(dim=-1)


@torch.no_grad()
def generate(
    model: GPT,
    idx: torch.Tensor,
    max_new_tokens: int,
    *,
    temperature: float = 1.0,
    top_k: int | None = None,
    top_p: float | None = None,
    stop_ids: set[int] | None = None,
    generator: torch.Generator | None = None,
    use_cache: bool = True,
) -> torch.Tensor:
    """Autoregressively extend `idx` by up to `max_new_tokens`.

    Args:
        idx: (B, T) prompt token ids.
        generator: a seeded `torch.Generator` for reproducible sampling. Without one the output is
            genuinely random and two calls differ — correct for a sampler, and the reason the demo
            exposes a seed.
        use_cache: use the KV cache (default). `False` selects the naive path that re-encodes the
            whole context each step. Kept because it is the reference the cache is verified against,
            and because it is the only way to measure what the cache actually buys.

    Cost: with the cache, O(N) forward passes over one new token each, so total work is linear in
    output length plus the inherent linear-in-context attention read. Without it, O(N²).

    Context overflow: once the sequence reaches `block_size`, the cache is dropped and rebuilt from
    the most recent `block_size` tokens. The model has no position embedding beyond `block_size`, so
    older context is genuinely gone — the interface says so rather than pretending to unbounded
    memory.
    """
    if model.training:
        raise RuntimeError("generate() requires eval mode: dropout would corrupt the samples")

    if not use_cache:
        for _ in range(max_new_tokens):
            context = idx[:, -model.cfg.block_size:]
            logits, _ = model(context)
            probs = apply_sampling_filters(
                logits[:, -1, :], temperature=temperature, top_k=top_k, top_p=top_p
            )
            next_id = torch.multinomial(probs, num_samples=1, generator=generator)
            idx = torch.cat([idx, next_id], dim=1)
            if stop_ids and int(next_id[0]) in stop_ids and idx.size(0) == 1:
                break
        return idx

    # Cached path: pre-fill the prompt in one pass, then feed one token at a time.
    past = None
    feed = idx[:, -model.cfg.block_size:]
    for _ in range(max_new_tokens):
        logits, past = model.forward_cached(feed, past)
        probs = apply_sampling_filters(
            logits[:, -1, :], temperature=temperature, top_k=top_k, top_p=top_p
        )
        next_id = torch.multinomial(probs, num_samples=1, generator=generator)
        idx = torch.cat([idx, next_id], dim=1)

        if stop_ids and int(next_id[0]) in stop_ids and idx.size(0) == 1:
            break

        if past[0][0].size(2) >= model.cfg.block_size:
            # Context is full. Drop the cache and re-prefill from the most recent window: the
            # position embeddings have no row past block_size, so the oldest tokens cannot be
            # represented at all. This is the architectural limit surfacing, not a policy choice.
            past = None
            feed = idx[:, -model.cfg.block_size:]
        else:
            feed = next_id
    return idx


@torch.no_grad()
def generate_with_trace(
    model: GPT,
    idx: torch.Tensor,
    max_new_tokens: int,
    *,
    temperature: float = 1.0,
    top_k: int | None = None,
    top_p: float | None = None,
    trace_top_k: int = 8,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, list[dict]]:
    """Same as `generate` for a batch of 1, recording the distribution at every step.

    Records both the **raw model distribution** and the **filtered** one actually sampled from.
    Showing only the filtered version would hide what the sampling controls are doing, which is the
    single most useful thing to see when explaining temperature and top-k to someone.
    """
    if model.training:
        raise RuntimeError("generate_with_trace() requires eval mode")
    if idx.size(0) != 1:
        raise ValueError(f"trace expects batch size 1, got {idx.size(0)}")

    trace: list[dict] = []
    for step in range(max_new_tokens):
        context = idx[:, -model.cfg.block_size:]
        logits, _ = model(context)
        raw_logits = logits[:, -1, :]

        raw_probs = raw_logits.softmax(dim=-1)
        filtered = apply_sampling_filters(
            raw_logits, temperature=temperature, top_k=top_k, top_p=top_p
        )
        next_id = torch.multinomial(filtered, num_samples=1, generator=generator)

        k = min(trace_top_k, raw_probs.size(-1))
        raw_top = raw_probs[0].topk(k)
        filt_top = filtered[0].topk(k)
        # Entropy of the raw distribution, in nats. Distinguishes "the model is confident" from
        # "the sampler happened to pick something unlikely".
        entropy = float(-(raw_probs[0] * raw_probs[0].clamp_min(1e-12).log()).sum())
        # How many tokens survived filtering -- makes top-k/top-p concrete.
        n_survivors = int((filtered[0] > 0).sum())

        trace.append({
            "step": step,
            "chosen": int(next_id[0]),
            "entropy_nats": entropy,
            "n_candidates": n_survivors,
            "raw_top": [(int(i), float(p)) for i, p in zip(raw_top.indices, raw_top.values)],
            "filtered_top": [(int(i), float(p)) for i, p in zip(filt_top.indices, filt_top.values)],
        })
        idx = torch.cat([idx, next_id], dim=1)

    return idx, trace


@torch.no_grad()
def estimate_loss(
    model: GPT, get_batch, *, eval_iters: int = 50, splits: tuple[str, ...] = ("train", "val")
) -> dict[str, float]:
    """Average the loss over `eval_iters` random batches per split.

    Averaging matters: a single batch's loss on a small validation set is noisy enough that
    consecutive evaluations can differ more than a real improvement would, which makes early-stopping
    decisions arbitrary. `nanoGPT` does the same for the same reason.
    """
    was_training = model.training
    model.eval()
    out: dict[str, float] = {}
    for split in splits:
        losses = torch.zeros(eval_iters)
        for i in range(eval_iters):
            x, y = get_batch(split)
            _, loss = model(x, y)
            losses[i] = loss.item()
        out[split] = float(losses.mean())
    if was_training:
        model.train()
    return out
