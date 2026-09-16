# NEXT ACTION

Single source of truth for "what happens next". Read this first when resuming. Update it **before**
context compaction.

---

## Awaiting user input

**Visual confirmation of the interface** before it is expanded further (brief §8). Live now:

- https://zexoraai.github.io/ml-from-first-principles/
- https://zexoraai.github.io/ml-from-first-principles/environment.html

Both verified by fetch. The design language established here (dark technical palette, fidelity-tier
pills, status pills, evidence tables, callout boxes for rejected/accepted decisions) is what all
eight project pages will inherit, so it is cheaper to change now than after eight pages exist.

---

## Active on approval

**M1.2 — P1 T2: encoder–decoder assembly + single-batch overfit proof.**

Exact behaviour being completed: a full `Transformer` module assembled from the T1 mechanisms
performs a forward pass with correct masking, and a training loop drives the loss on **one fixed
batch** to near zero, then greedy-decodes that batch exactly. This is the cheapest possible proof
that forward, loss, backward and masking are all wired correctly — before spending an hour on real
training and mistaking a wiring bug for a hard task.

Definition of done:
- [ ] `model.py`: `EncoderLayer`, `DecoderLayer`, `Encoder`, `Decoder`, `Transformer`
- [ ] Embedding scaling by `sqrt(d_model)` and the three-way weight tying of §3.4, behind a flag
- [ ] `generate.py`: greedy decode with an incremental causal mask
- [ ] `optim.py`: the §5.3 warmup schedule `d_model**-0.5 * min(step**-0.5, step * warmup**-1.5)`
      and §5.4 label smoothing, both hand-written
- [ ] Test: parameter count matches a hand-derived formula (catches silent shape errors)
- [ ] Test: overfit 8 examples to loss < 0.01 and exact greedy match
- [ ] Test: decoder output at position `i` is invariant to target tokens `> i` (autoregression held
      end-to-end through the full stack, not just one attention module)
- [ ] Test: pre-norm vs post-norm both train; record the difference as a real observation
- [ ] Thread count pinned to 2 for training (measured optimum, EXPERIMENTS env-bench-02)

## Queue after that

1. M1.3 — synthetic date-normalisation seq2seq, train/val/test splits, checkpoint + resume, first
   full EXPERIMENTS.md entry with a real run_id and metrics
2. M1.5 — P1 project page: 14-step composition walkthrough, three explanation depths, attention-head
   viewer, mask toggles, tensor inspector, in-browser demo with published parity tolerance
3. M1.6 — P1 mastery pack + first back-explanation quiz
4. P2 (GPT) — reuses P1's attention, adds tokenizer/checkpointing/sampling

## Open questions (defaults assumed until answered)

| # | Question | Default being assumed |
|---|---|---|
| Q2 | Free Colab session available to execute the P5 Triton kernel? | **No** — kernel ships labelled "authored, not executed", no timings |
| Q3 | Any budget for rented GPU hours? | **No** — 124 M GPT stays a documented estimate |
| Q4 | May we have one quiet benchmarking window (container stack stopped)? | **No** — all timings stay labelled lower bounds |

Q1 (repo name) — **answered: `ml-from-first-principles`, public. Done.**

## Do-not-forget (environment quirks that cost time to rediscover)

- All Python runs via `.\run.cmd <cmd>` — a Linux container. The host cannot import torch (D-008).
- `run.cmd` must be invoked as `.\run.cmd` in PowerShell; bare `run.cmd` is not resolved.
- The driving shell **truncates long command strings**. Keep commands short; write output to a file
  and read the file. Exit code is reported as -1 even on success, so verify via output, not codes.
- `npm.cmd`, never `npm`. No `.ps1` scripts — execution policy is Restricted.
- PowerShell writes UTF-16LE when redirecting; files read back with spaces between characters are
  fine, not corrupt.
- Locale uses comma decimal separators: `31,4GB` is 31.4 GB.
- Thread count is per workload: **2 for small-model training**, 6 for large matmuls.
