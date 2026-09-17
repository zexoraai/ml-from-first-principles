"""The sequence-to-sequence task: normalising human-written dates to ISO-8601.

    "March 3, 2019"            ->  "2019-03-03"
    "Sunday, 3 Mar 2019"       ->  "2019-03-03"
    "3rd September 1987"       ->  "1987-09-03"

WHY THIS TASK
-------------
It has to be small enough to train on a 6-core CPU in under an hour, and still require the
mechanisms we claim to have implemented. Date normalisation does:

* **It needs attention, not memorisation.** The output digits come from specific, *variably
  positioned* parts of the input. The year may be at the start or the end; the month may be a word
  of 3 to 9 letters. To emit position 0 of the output the model must locate the year wherever it
  happens to be. Cross-attention is the mechanism for that, and the attention maps are legible
  enough to inspect on the project page.
* **It needs ordering.** The output is a reordering of input fields, so a bag-of-characters model
  cannot do it. That exercises positional encoding for real.
* **Evaluation is unambiguous.** Exactly one correct output string per input, so exact-match
  accuracy needs no judgement, no reference set and no metric hand-waving.
* **Licence-free.** The data is produced by the generator in this file. There is no third-party
  corpus, no licence to review, and no download.

It is a *deliberately easy* task. That is the point: at tier E/R the goal is to demonstrate the
mechanisms work, not to claim a hard benchmark. The project page says so.

THE SPLIT, AND THE LEAKAGE ARGUMENT
-----------------------------------
Splits are drawn over **calendar dates, not over rendered strings**. Every rendering of
2019-03-03 lands in exactly one split.

The naive alternative -- generate all the strings, then shuffle and split -- leaks badly. The model
would see "March 3, 2019" in training and be evaluated on "3 Mar 2019", having already learned that
particular date's output by heart. Validation accuracy would then measure memorisation of dates
rather than the ability to parse an unseen date, and it would look excellent. Splitting on the
underlying date makes the evaluation answer the question we actually care about: does it generalise
to a date it has never been shown, in any format.
"""

from __future__ import annotations

import datetime as dt
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Literal

import torch

__all__ = ["CharTokenizer", "DateTaskConfig", "build_splits", "DateDataset", "collate",
           "RENDERERS", "render_distinct", "iso", "write_split_manifest"]

MONTHS_FULL = ["January", "February", "March", "April", "May", "June",
               "July", "August", "September", "October", "November", "December"]
MONTHS_ABBR = [m[:3] for m in MONTHS_FULL]
WEEKDAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]


def _ordinal(n: int) -> str:
    if 10 <= n % 100 <= 20:
        suffix = "th"
    else:
        suffix = {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


# Every renderer is unambiguous on purpose. Purely numeric forms such as "03/04/2019" are excluded:
# they are ambiguous between day-first and month-first conventions, and including them would make
# some examples unanswerable, which would put a floor on achievable accuracy for reasons that have
# nothing to do with the model.
RENDERERS: dict[str, Callable[[dt.date], str]] = {
    "month_day_year":      lambda d: f"{MONTHS_FULL[d.month - 1]} {d.day}, {d.year}",
    "day_month_year":      lambda d: f"{d.day} {MONTHS_FULL[d.month - 1]} {d.year}",
    "abbr_day_year":       lambda d: f"{MONTHS_ABBR[d.month - 1]} {d.day}, {d.year}",
    "day_abbr_year":       lambda d: f"{d.day} {MONTHS_ABBR[d.month - 1]} {d.year}",
    "ordinal_month_year":  lambda d: f"{_ordinal(d.day)} {MONTHS_FULL[d.month - 1]} {d.year}",
    "month_ordinal_year":  lambda d: f"{MONTHS_FULL[d.month - 1]} {_ordinal(d.day)}, {d.year}",
    "year_month_day_words": lambda d: f"{d.year} {MONTHS_FULL[d.month - 1]} {d.day}",
    "weekday_full":        lambda d: f"{WEEKDAYS[d.weekday()]}, {MONTHS_FULL[d.month - 1]} {d.day}, {d.year}",
    "weekday_abbr":        lambda d: f"{WEEKDAYS[d.weekday()][:3]}, {d.day} {MONTHS_ABBR[d.month - 1]} {d.year}",
    # Hyphen-separated, and zero-padded. The hyphens are load-bearing: an earlier version of this
    # renderer was `f"{d.day:02d} {abbr} {d.year}"`, which is byte-identical to `day_abbr_year` for
    # every two-digit day -- "24 Sep 1984" from both. Two "different" formats collapsing into one
    # string meant ~70% of dates could be drawn twice with the same rendering, quietly duplicating
    # examples and overstating the effective dataset size.
    # `tests/test_p1_data.py::test_a_date_is_rendered_in_distinct_formats_without_repetition`
    # caught it; `test_all_renderers_are_pairwise_distinct` now prevents it recurring.
    "hyphenated_abbr":     lambda d: f"{d.day:02d}-{MONTHS_ABBR[d.month - 1]}-{d.year}",
}


def iso(d: dt.date) -> str:
    return d.isoformat()          # always exactly 10 characters: YYYY-MM-DD


def render_distinct(d: dt.date, k: int, rng: random.Random) -> list[str]:
    """Return up to `k` **distinct** renderings of one date.

    Why deduplication happens here rather than being avoided by design
    -----------------------------------------------------------------
    Sampling `k` distinct *renderer names* is not enough, because different renderers can produce
    byte-identical strings. Two such collisions were found by
    `tests/test_p1_data.py`, and the second one is subtle enough to be worth recording:

    1. `zero_padded_abbr` (since replaced) emitted `"24 Sep 1984"`, identical to `day_abbr_year`
       for every day >= 10 -- a two-digit day makes zero-padding a no-op.
    2. **May** is the only month whose three-letter abbreviation equals its full name. So for May
       dates, `month_day_year` and `abbr_day_year` both give `"May 1, 2019"`, and
       `day_month_year` and `day_abbr_year` both give `"1 May 2019"`.

    The first was fixable by changing a format. The second is not: it is a fact about English month
    names, and any future renderer pair could collide again for reasons just as local. So the
    generator renders candidates and deduplicates on the resulting **string**, which is the property
    actually required -- no example appears twice -- rather than the proxy property of distinct
    renderer names.

    Consequence, stated because it is a real asymmetry in the data: May dates yield 8 distinct
    renderings where other months yield 10. With `renderings_per_train_date = 2` that is invisible
    (both are far above 2), but it would matter if that value were raised near the limit.
    """
    names = list(RENDERERS)
    rng.shuffle(names)
    seen: set[str] = set()
    picked: list[str] = []
    for name in names:
        text = RENDERERS[name](d)
        if text in seen:
            continue
        seen.add(text)
        picked.append(text)
        if len(picked) == k:
            break
    return picked


class CharTokenizer:
    """Character-level tokenizer over a **fixed, declared** alphabet.

    The alphabet is hard-coded rather than inferred from the sampled data. Inferring it would make
    the vocabulary -- and therefore every tensor shape, the parameter count, and any checkpoint --
    depend on which examples happened to be drawn. A checkpoint would then be silently incompatible
    with a differently seeded run, which is a genuinely painful class of bug.

    Character level is the right granularity here and not a shortcut: the output is built from
    digits and hyphens, the input from letters, digits, spaces and punctuation. Sub-word tokens
    would add a vocabulary-construction step with nothing to teach for this task, and would hide
    the digit-level copying that makes the attention maps legible.

    Special tokens occupy indices 0-3 so `pad_id=0` matches the model default:
        0 <pad>   1 <bos>   2 <eos>   3 <unk>
    """

    PAD, BOS, EOS, UNK = "<pad>", "<bos>", "<eos>", "<unk>"
    SPECIALS = [PAD, BOS, EOS, UNK]
    ALPHABET = " ,.-0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"

    def __init__(self) -> None:
        self.itos: list[str] = list(self.SPECIALS) + list(self.ALPHABET)
        self.stoi: dict[str, int] = {t: i for i, t in enumerate(self.itos)}
        self.pad_id = self.stoi[self.PAD]
        self.bos_id = self.stoi[self.BOS]
        self.eos_id = self.stoi[self.EOS]
        self.unk_id = self.stoi[self.UNK]

    def __len__(self) -> int:
        return len(self.itos)

    def encode(self, text: str) -> list[int]:
        return [self.stoi.get(c, self.unk_id) for c in text]

    def decode(self, ids: Iterable[int], *, strip_specials: bool = True) -> str:
        out = []
        for i in ids:
            i = int(i)
            tok = self.itos[i] if 0 <= i < len(self.itos) else self.UNK
            if strip_specials and tok in self.SPECIALS:
                if tok == self.EOS:
                    break
                continue
            out.append(tok)
        return "".join(out)

    def to_json(self) -> str:
        """Serialised for the browser demo, so JS and PyTorch share one vocabulary by construction."""
        return json.dumps({
            "itos": self.itos, "pad_id": self.pad_id, "bos_id": self.bos_id,
            "eos_id": self.eos_id, "unk_id": self.unk_id,
        })


@dataclass
class DateTaskConfig:
    """Data-generation parameters. Written into every run directory so a split is reconstructible."""
    start_year: int = 1950
    end_year: int = 2035
    renderings_per_train_date: int = 2
    renderings_per_eval_date: int = 1
    train_frac: float = 0.70
    val_frac: float = 0.15          # test gets the remainder
    max_src_len: int = 32           # longest renderer output is "Wednesday, September 30, 2035" = 29
    max_tgt_len: int = 12           # 10 ISO characters + BOS + EOS
    seed: int = 1706


def _all_dates(cfg: DateTaskConfig) -> list[dt.date]:
    start = dt.date(cfg.start_year, 1, 1)
    end = dt.date(cfg.end_year, 12, 31)
    return [start + dt.timedelta(days=i) for i in range((end - start).days + 1)]


def build_splits(cfg: DateTaskConfig) -> dict[str, list[tuple[str, str]]]:
    """Produce train/val/test as lists of (source_text, target_text) pairs.

    Dates are partitioned first, then rendered. See the module docstring for why.
    """
    rng = random.Random(cfg.seed)
    dates = _all_dates(cfg)
    rng.shuffle(dates)

    n = len(dates)
    n_train = int(n * cfg.train_frac)
    n_val = int(n * cfg.val_frac)
    partitions = {
        "train": dates[:n_train],
        "val": dates[n_train:n_train + n_val],
        "test": dates[n_train + n_val:],
    }

    out: dict[str, list[tuple[str, str]]] = {}
    for split, split_dates in partitions.items():
        k = cfg.renderings_per_train_date if split == "train" else cfg.renderings_per_eval_date
        pairs: list[tuple[str, str]] = []
        for d in split_dates:
            for text in render_distinct(d, k, rng):
                pairs.append((text, iso(d)))
        rng.shuffle(pairs)
        out[split] = pairs

    # Leakage guard, asserted rather than assumed. Cheap, and it would have caught the naive split.
    seen: dict[str, str] = {}
    for split, pairs in out.items():
        for _, target in pairs:
            if target in seen and seen[target] != split:
                raise AssertionError(
                    f"date {target} appears in both {seen[target]} and {split}: splits leak"
                )
            seen[target] = split
    return out


class DateDataset(torch.utils.data.Dataset):
    """Tokenised (src, tgt_in, labels) triples.

    The right-shift lives here, once, rather than in the training loop:

        target text     y = "2019-03-03"
        tgt_in          [BOS, y_0, ..., y_9]      what the decoder consumes
        labels          [y_0, ..., y_9, EOS]      what it must predict

    Position i of `tgt_in` is token i-1 of the target, so predicting `labels[i]` from
    `tgt_in[:i+1]` is a genuine next-token problem. Getting this offset wrong by one is the classic
    way to build a model that either sees the answer (loss collapses, generation fails) or is asked
    to predict something it cannot know (loss plateaus). `tests/test_p1_data.py` pins the alignment.
    """

    def __init__(self, pairs: list[tuple[str, str]], tokenizer: CharTokenizer, cfg: DateTaskConfig) -> None:
        self.pairs = pairs
        self.tok = tokenizer
        self.cfg = cfg

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        src_text, tgt_text = self.pairs[idx]
        src = self.tok.encode(src_text)
        tgt = self.tok.encode(tgt_text)
        if len(src) > self.cfg.max_src_len:
            raise ValueError(f"source {src_text!r} exceeds max_src_len={self.cfg.max_src_len}")

        return {
            "src": torch.tensor(src, dtype=torch.long),
            "tgt_in": torch.tensor([self.tok.bos_id] + tgt, dtype=torch.long),
            "labels": torch.tensor(tgt + [self.tok.eos_id], dtype=torch.long),
            "index": torch.tensor(idx, dtype=torch.long),
        }


def collate(batch: list[dict[str, torch.Tensor]], pad_id: int = 0) -> dict[str, torch.Tensor]:
    """Right-pad each field to the longest member of the batch.

    Padding to the batch maximum rather than to a global maximum keeps the tensors as small as the
    batch allows, which matters on a CPU. The padding is then removed from consideration in two
    independent places -- the attention masks and the loss -- and both are necessary: masks stop
    padding influencing representations, the loss ignore stops it influencing gradients.
    """
    def pad_stack(key: str) -> torch.Tensor:
        seqs = [b[key] for b in batch]
        width = max(s.size(0) for s in seqs)
        out = torch.full((len(seqs), width), pad_id, dtype=torch.long)
        for i, s in enumerate(seqs):
            out[i, : s.size(0)] = s
        return out

    return {
        "src": pad_stack("src"),
        "tgt_in": pad_stack("tgt_in"),
        "labels": pad_stack("labels"),
        "index": torch.stack([b["index"] for b in batch]),
    }


def write_split_manifest(splits: dict[str, list[tuple[str, str]]], cfg: DateTaskConfig, path: Path) -> dict:
    """Record exactly what the splits were, so a reported metric is tied to a reconstructible set."""
    manifest = {
        "task": "date normalisation to ISO-8601",
        "generator": "labs/p1_transformer/data.py",
        "licence": "generated by this repository; no third-party data",
        "config": cfg.__dict__,
        "renderers": sorted(RENDERERS),
        "split_sizes": {k: len(v) for k, v in splits.items()},
        "unique_dates": {k: len({t for _, t in v}) for k, v in splits.items()},
        "split_unit": "calendar date -- all renderings of a date share one split",
        "leakage_check": "build_splits asserts no ISO target appears in two splits",
        "examples": {k: v[:3] for k, v in splits.items()},
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    return manifest
