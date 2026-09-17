# NEXT ACTION

Single source of truth for "what happens next". Read this first when resuming. Update it **before**
context compaction.

---

## Project 1 is COMPLETE

Live and verified:

- https://zexoraai.github.io/ml-from-first-principles/ — portfolio home
- https://zexoraai.github.io/ml-from-first-principles/environment.html — measured environment
- https://zexoraai.github.io/ml-from-first-principles/projects/p1-transformer.html — **Project 1**

Delivered: 161 passing tests · trained model at **99.94% test exact-match** (run
`p1-date-post-6k`) · live in-browser inference with parity verified to 5.3e-7 · attention head
viewer · 14-step teaching document · mastery pack with separate answer notes · full evidence
register entry.

**Not delivered, deliberately:** any tier-P claim, any beam search, any KV cache, any validated
interpretability claim, more than one seed. All recorded in GAPS.

---

## Active

**Nothing.** Awaiting direction on whether to start Project 2 (decoder-only GPT).

P2 is the natural next step and reuses P1's attention module directly. Before starting it, three
things are worth resolving:

1. **Q2/Q3 (Colab / GPU budget)** still unanswered. They do not block P2's small model but they
   determine whether the ~124 M target is attempted or documented.
2. **The P1 mastery pack is written and unused.** `records/PROGRESS.md`'s mastery ledger is entirely
   "not yet". Implementation is complete; understanding is unverified. Per the brief, these are
   tracked separately and a finished project is not an understood one. Running the day-0 questions
   (A1, A3, A6 + B1 + C1) before moving on would keep the two in step.
3. **The late-training instability in `p1-date-post-6k` is unexplained.** Val exact-match hit 1.0000
   at step 3500–5000 then fell to 0.7441 by step 6000, after a gradient-norm spike of 22.65. Best
   checkpoint selection caught it. A pre-norm run would be the obvious comparison and is already set
   up as the mastery pack's prediction experiment.

## Queue

1. P1 mastery: run the day-0 back-explanation set, score it, record in the mastery ledger
2. P2 (GPT): decoder-only, next-token training, resumable checkpointing, temperature/top-k sampling
3. P3 (LoRA) — depends on P2's checkpoint
4. P4 (DPO) — depends on P3's SFT arm

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
