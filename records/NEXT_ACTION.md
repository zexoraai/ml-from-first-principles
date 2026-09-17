# NEXT ACTION

Single source of truth for "what happens next". Read this first when resuming. Update it **before**
context compaction.

---

## Project 1 is COMPLETE and DEPLOYED

- https://zexoraai.github.io/ml-from-first-principles/ — portfolio home
- https://zexoraai.github.io/ml-from-first-principles/environment.html — measured environment
- https://zexoraai.github.io/ml-from-first-principles/projects/p1-transformer.html — **Project 1**

161 passing tests · **99.94% test exact-match** (run `p1-date-post-6k`, 4710/4713, 3 real failures
published) · live in-browser inference, JS↔PyTorch parity 5.26e-7 / 3.58e-6 · attention head viewer ·
14-step teaching document · mastery pack with separate answer notes · full evidence register entry.

**Not delivered, deliberately:** any tier-P claim, beam search, KV cache in P1, any validated
interpretability claim, more than one seed. All recorded in GAPS.

---

## Active right now

**P2 training is running.** Terminal `term_1789647667939_am7lg3ef89v`, log `p2d.txt`.

```
run p2-design-3m | 2,903,040 params | vocab 1024 | block 192
2,117,570 train / 221,297 val tokens | 2.62 chars/token | 11,059,200 token budget (5.2 epochs)
2400 steps | ~1300-1420 tok/s | ETA ~2.2 h from step 250
```

Progress: step 350, train 4.29, val 4.36, ppl 78.2 (random baseline is ln 1024 = 6.93).
**Do not stop this terminal.** Poll `p2d.txt` with `read_file` at varying offsets.

---

## Ready and waiting on P2's checkpoint

Everything below is written and unit-tested; all of it needs `runs/p2-design-3m/checkpoint_best.pt`.

| Project | Code | Tests | Training script | Page |
|---|---|---|---|---|
| P3 LoRA | `labs/p3_lora/lora.py` | 26 pass | **not written** | not written |
| P4 DPO | `labs/p4_dpo/{dpo,data,batching,evaluate}.py` | **64 pass** | `scripts/train_p4.py` | not written |

### P4 design, so it does not have to be rediscovered

Arms, all starting from the **same SFT checkpoint** which is also the frozen `π_ref`:

- `base` — the pretrained P2 model, untouched
- `sft` — base + N supervised steps on instruction data (**this is π_ref**)
- `sft_continued` — sft + N *more* supervised steps ← **the control that makes the result mean
  something.** Without it, any DPO gain is confounded with simply having taken more gradient steps.
- `dpo_beta{0.02, 0.1, 0.5}` — sft + N DPO steps

Metrics per arm: preference accuracy (raw **and** length-normalised — the gap *is* the length bias of
the summed objective), implicit reward margin, log-ratio from reference, generation-side degradation
rates from deterministic detectors, and held-out pretraining perplexity as an **alignment-tax** check.

Preferences are **CONSTRUCTED, NOT HUMAN-ANNOTATED** — four documented degradations (`off_topic`,
`truncated`, `repetitive`, `generic`), split by topic with leakage asserted in code. This is stated in
`data.py`, in the run manifest, in `result.json`'s limitations, and must be stated on the page.

---

## Queue

1. **P2 finishes** → `export_p2_web.py` → `node scripts/verify_js_parity.mjs` → commit → **verify the
   deployed URL by fetch**, not by assuming
2. `scripts/train_p4.py` real run against the P2 checkpoint (smoke-tested first against
   `runs/p4base-smoke`, which is throwaway and must be deleted before committing)
3. `scripts/train_p3.py` — base vs LoRA vs **full fine-tune**, three-way, measuring trainable
   parameters, memory, wall-clock, and quality. The full fine-tune arm is the one that makes the LoRA
   claim falsifiable.
4. P3 and P4 project pages
5. P5 tiled attention — reference + tiled online-softmax are measurable locally; the Triton kernel
   ships **authored, not executed** (D-004)
6. P6 quantization, P7 CLIP, P8 DDPM
7. `records/EXPERIMENTS.md` entry for the P2 run; keep this file current

## Open questions (defaults assumed until answered)

| # | Question | Default being assumed |
|---|---|---|
| Q2 | Free Colab session available to execute the P5 Triton kernel? | **No** — kernel ships labelled "authored, not executed", no timings |
| Q3 | Any budget for rented GPU hours? | **No** — 124 M GPT stays a documented estimate |
| Q4 | May we have one quiet benchmarking window (container stack stopped)? | **No** — all timings stay labelled lower bounds |

Q1 (repo name) — **answered: `ml-from-first-principles`, public. Done.**

Cloud is **phase 2**, after all local work (D-009). Everything is being built portable now:
device-agnostic, portable checkpoints, scale in CLI flags. **Cloud does not upgrade fidelity tiers.**

## Do-not-forget (environment quirks that cost time to rediscover)

- All Python runs via `.\run.cmd <cmd>` — a Linux container. The host cannot import torch (D-008).
- `run.cmd` must be invoked as `.\run.cmd` in PowerShell; bare `run.cmd` is not resolved.
- The driving shell **truncates long command strings**. Keep commands short; write output to a file
  and read the file. Exit code is reported as -1 even on success, so verify via output, not codes.
- `Start-Sleep` in the driving shell is a **no-op** — it returns instantly. There is no way to block;
  poll output files with `read_file` at *varying offsets* to force a fresh read.
- **Stale background terminals wedge Docker.** Stop finished ones; never stop a training terminal.
- Docker takes tens of seconds to start a container while training holds the CPU. An empty output
  file usually means "not started yet", not "failed".
- `npm.cmd`, never `npm`. No `.ps1` scripts — execution policy is Restricted.
- PowerShell writes UTF-16LE when redirecting; files read back with spaces between characters are
  fine, not corrupt.
- Locale uses comma decimal separators: `31,4GB` is 31.4 GB.
- Thread count is per workload: **2 for small-model training**, 6 for large matmuls.
