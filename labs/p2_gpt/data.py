"""Corpus loading, tokenisation and batching for Project 2.

DATASET AND LICENCE
-------------------
**TinyShakespeare** — a ~1.1 MB concatenation of Shakespeare's plays, assembled by Andrej Karpathy
for `char-rnn` and used by `nanoGPT`. Shakespeare's works are **public domain**; the compilation is
distributed under the MIT licence of the `char-rnn` repository. Nothing here is scraped and nothing
requires a licence review.

Why this corpus for a complete project rather than something larger:
* It fits the disk budget (GAPS G-002: 31.4 GB free, and 20 GB of that is other people's).
* It is small enough to train a real model to convergence on a 6-core CPU in about an hour, which
  means the training curve on the project page is a *finished* curve, not a truncated one.
* Its output is instantly recognisable, so a live audience can judge quality without a metric.
* It is the standard reference corpus for exactly this model size, so the validation loss is
  comparable to numbers other people publish for small GPTs — a genuinely useful sanity anchor,
  even though we make no claim of matching any specific published figure.

The download is fetched once and cached. If the network is unavailable, a clear error is raised
rather than silently substituting different data.

THE SPLIT
---------
A single contiguous 90/10 split by character position, not a shuffle. This is deliberate and it is
the opposite of what P1 does.

For a language model the unit of evaluation is *continuation*, and neighbouring windows overlap
almost entirely. Shuffling windows and then splitting would put window `[i, i+T]` in train and
`[i+1, i+T+1]` in validation — a near-duplicate — so validation loss would measure memorisation of
sequences it had effectively already seen. A contiguous cut means the held-out text is genuinely
unseen continuous prose. The only leakage is at the single boundary, which is one window out of
tens of thousands.
"""

from __future__ import annotations

import hashlib
import json
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np
import torch

from .corpus import CORPORA, load_corpus
from .tokenizer import ByteBPETokenizer

__all__ = ["TINY_SHAKESPEARE_URL", "download_corpus", "prepare", "BatchSampler", "CORPORA"]

TINY_SHAKESPEARE_URL = (
    "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
)


def download_corpus(dest: Path, url: str = TINY_SHAKESPEARE_URL) -> str:
    """Fetch the corpus once and cache it on disk."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        return dest.read_text(encoding="utf-8")
    try:
        with urllib.request.urlopen(url, timeout=60) as response:  # noqa: S310
            text = response.read().decode("utf-8")
    except (urllib.error.URLError, TimeoutError) as exc:
        raise RuntimeError(
            f"could not download the corpus from {url}: {exc}. "
            f"Place the file manually at {dest} and re-run. No substitute data will be used."
        ) from exc
    dest.write_text(text, encoding="utf-8")
    return text


def prepare(
    data_dir: str | Path = "data/p2",
    *,
    corpus: str = "design",
    vocab_size: int = 1024,
    val_fraction: float = 0.1,
    verbose: bool = True,
) -> dict:
    """Acquire the corpus, train the tokenizer, encode, and write train/val id arrays.

    Args:
        corpus: a key from `labs.p2_gpt.corpus.CORPORA`. Default `"design"`.

    The tokenizer is trained on the **training portion only**. Fitting it on the whole corpus would
    leak: the merge list would be chosen partly from held-out text, so validation tokens would be
    represented more efficiently than genuinely unseen text. The effect is small, it is real, and it
    is free to avoid.

    The split is contiguous **per source** (see `CorpusSpec.split`). A full attribution record is
    written to `attribution.json`, which is the licence artefact for the CC BY-SA portion.
    """
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    if verbose:
        print(f"corpus: {corpus} — {CORPORA.get(corpus, 'unknown')}")
    spec = load_corpus(corpus, data_dir, verbose=verbose)
    if len(spec.text) < 10_000:
        raise RuntimeError(
            f"corpus {corpus!r} yielded only {len(spec.text)} characters — too small to train on. "
            f"Check the network and the source list rather than proceeding."
        )

    train_text, val_text = spec.split(val_fraction)

    # Attribution is a deliverable, not a comment. Written every time so it cannot drift from the
    # corpus actually used.
    (data_dir / "attribution.json").write_text(
        json.dumps(spec.manifest(), indent=2), encoding="utf-8"
    )

    # The cache key includes a hash of the ACTUAL training text, not just the corpus name.
    #
    # This is a real bug that already happened: an earlier run cached `tokenizer_design_1024.json`
    # from a corpus that did not yet include the open textbook. Adding the textbook changed the
    # training text but not the filename, so the next run silently reused merges learned from
    # different data — while the manifest went on claiming the tokenizer was trained on this split.
    # Nothing crashes; byte-level BPE never fails on unseen input, so the only symptom is slightly
    # worse compression and a provenance claim that is false. Hashing the text makes the cache
    # self-invalidating.
    corpus_hash = hashlib.sha256(train_text.encode("utf-8")).hexdigest()[:12]
    tok_path = data_dir / f"tokenizer_{corpus}_{vocab_size}_{corpus_hash}.json"
    if tok_path.exists():
        tok = ByteBPETokenizer.load(tok_path)
        if verbose:
            print(f"  loaded tokenizer from {tok_path.name} ({tok.vocab_size} tokens)")
    else:
        if verbose:
            print(f"  training byte-level BPE to vocab {vocab_size} on the TRAIN split only…")
        tok = ByteBPETokenizer().train(train_text, vocab_size, verbose=verbose)
        tok.save(tok_path)

    if tok.vocab_size > 65535:
        raise ValueError("vocab exceeds uint16; widen the dtype before increasing vocab_size")

    train_ids = np.array(tok.encode(train_text), dtype=np.uint16)
    val_ids = np.array(tok.encode(val_text), dtype=np.uint16)
    train_ids.tofile(data_dir / f"train_{corpus}.bin")
    val_ids.tofile(data_dir / f"val_{corpus}.bin")

    stats = {
        "corpus": corpus,
        "corpus_description": spec.description,
        "n_sources": len(spec.sources),
        "segments": [{"label": lbl, "chars": len(t)} for lbl, t in spec.segments],
        "corpus_chars": len(spec.text),
        "train_chars": len(train_text),
        "val_chars": len(val_text),
        "train_tokens": int(train_ids.size),
        "val_tokens": int(val_ids.size),
        "vocab_size": tok.vocab_size,
        "n_merges": len(tok.merges),
        "compression_chars_per_token": len(train_text) / max(train_ids.size, 1),
        "tokenizer_path": str(tok_path),
        "train_text_sha256_12": corpus_hash,
        "attribution_path": str(data_dir / "attribution.json"),
        "licences": sorted({s["licence"] for s in spec.sources}),
        "split": ("contiguous per-source split so every register appears in both train and val; "
                  "tokenizer trained on the train portion only"),
    }
    if verbose:
        print(f"  {stats['n_sources']} sources | {stats['corpus_chars']:,} chars")
        print(f"  train {stats['train_tokens']:,} tokens | val {stats['val_tokens']:,} tokens")
        print(f"  compression {stats['compression_chars_per_token']:.2f} chars/token")
        for lic in stats["licences"]:
            print(f"  licence: {lic}")
    # `train_text` / `val_text` are returned alongside the token arrays because Project 4 builds
    # instruction and preference data out of the corpus's heading structure, which the flat uint16
    # token array has thrown away. Returning the text costs nothing (it is already in memory) and
    # keeps Project 4 reading the *same* split as Project 2 trained on -- re-splitting it separately
    # would risk the two disagreeing and leaking val text into Project 4's training pairs.
    return {"tokenizer": tok, "stats": stats, "spec": spec,
            "train_ids": train_ids, "val_ids": val_ids,
            "train_text": train_text, "val_text": val_text}


class BatchSampler:
    """Draws random fixed-length windows from a flat token array.

    Why random offsets rather than sequential chunks: sequential chunking would make every example
    in a batch come from adjacent text, so the gradient of one step is dominated by one scene. Random
    offsets decorrelate the batch. It also means "one epoch" is not well defined here — we count
    tokens seen, not epochs, and the project page reports the token budget for that reason.

    `x` is the window and `y` is the same window shifted by one. Position `i` of `x` predicts
    position `i` of `y`, so a single window of length T supplies T next-token training signals.
    """

    def __init__(
        self,
        train_ids: np.ndarray,
        val_ids: np.ndarray,
        *,
        block_size: int,
        batch_size: int,
        seed: int = 1337,
    ) -> None:
        if len(train_ids) <= block_size + 1:
            raise ValueError("training data shorter than one window")
        self.data = {"train": train_ids, "val": val_ids}
        self.block_size = block_size
        self.batch_size = batch_size
        self.rng = np.random.default_rng(seed)

    def get_batch(self, split: str) -> tuple[torch.Tensor, torch.Tensor]:
        ids = self.data[split]
        high = len(ids) - self.block_size - 1
        if high <= 0:
            raise ValueError(f"split {split!r} is shorter than block_size + 1")
        offsets = self.rng.integers(0, high, size=self.batch_size)
        # astype(int64) because torch.long is required for embedding indices; the on-disk uint16
        # keeps the file small (2 bytes/token) but is not a valid index dtype.
        x = np.stack([ids[o:o + self.block_size] for o in offsets]).astype(np.int64)
        y = np.stack([ids[o + 1:o + 1 + self.block_size] for o in offsets]).astype(np.int64)
        return torch.from_numpy(x), torch.from_numpy(y)

    def state_dict(self) -> dict:
        """The sampler's RNG is part of the training state; a resume must restore it."""
        return {"bit_generator": self.rng.bit_generator.state}

    def load_state_dict(self, state: dict) -> None:
        self.rng.bit_generator.state = state["bit_generator"]
