# PROGRESS

Milestone ledger. One row per milestone. `Status` ∈ {PLANNED, IN PROGRESS, DONE, BLOCKED}.
A milestone is DONE only when its stated behaviour is demonstrable *and* its checks have been run.

**Legend for tier:** E = educational mechanism, R = reduced-scale reproduction, P = published-result
reproduction (see SCOPE.md §0).

---

## Session 1 — 2026-09-15

| ID | Milestone | Behaviour completed | Tier | Status |
|---|---|---|---|---|
| M0.1 | Environment probe | Hardware/software inventory measured, not assumed | — | DONE |
| M0.2 | Scope + decision records | SCOPE.md, DECISIONS.md written; D-001..D-006 accepted | — | DONE |
| M0.3 | Repo skeleton + pinned environment | torch importable; layout per D-005. **Scope changed mid-milestone**: Smart App Control blocked torch on the host, so the environment became a pinned Linux container (D-008) rather than a Windows venv | — | DONE |
| M0.4 | Measured feasibility table | Benchmark run 3x; a 300x harness bug found and fixed; `env/feasibility.md` written with ESTIMATE labels; load capture added | — | DONE |
| M1.1 | **P1 T1: Transformer mechanisms** | Every mechanism from arXiv:1706.03762v7 hand-written and correctness-checked in isolation. **79 tests passing.** Commit `06736c1` | E | DONE |
| M1.4 | P1 T4: repo public + Pages live | **Verified**: `https://zexoraai.github.io/ml-from-first-principles/` and `/environment.html` both fetched and returning content. Commit `5363086` | — | DONE |
| M1.2 | P1 T2: model assembly + overfit proof | Full encoder–decoder forward; overfit a single batch to ~0 loss; greedy decode | E | NEXT |
| M1.3 | P1 T3: seq2seq training + held-out eval | Recorded run, checkpoint/resume, val metrics with seeds | R | PLANNED |
| M1.5 | P1 T5: project page + demo | Live in-browser inference with published parity tolerance vs PyTorch | — | PLANNED |
| M1.6 | P1 mastery pack | Defence questions, exercises, planted bugs, prediction experiment | — | PLANNED |

**Note on ordering:** M1.4 was pulled ahead of M1.2/M1.3 so a verified public URL exists early and
the interface can be reviewed before more of it is built. The site currently ships the home page
and the environment/method case study — both complete content, not placeholders. No demo is linked
because no model has been trained yet, and the site says so explicitly.

### Deployed URLs (each one fetched and confirmed, not assumed)

| URL | Verified | Content |
|---|---|---|
| https://zexoraai.github.io/ml-from-first-principles/ | 2026-09-15 | portfolio home, eight project cards, fidelity tier table |
| https://zexoraai.github.io/ml-from-first-principles/environment.html | 2026-09-15 | measured hardware, three findings, per-project compute estimates |
| https://github.com/zexoraai/ml-from-first-principles | 2026-09-15 | source + all records |

## Mastery ledger (tracked separately from implementation — see brief §6)

Implementation progress does **not** imply understanding. This table only moves when the user has
explained a mechanism back and the reasoning was evaluated.

| Project | Mechanism | Explained back? | Reasoning quality | Gaps found | Retaught | Next review |
|---|---|---|---|---|---|---|
| P1 | scaled dot-product attention | not yet | — | — | — | — |
| P1 | multi-head split/concat | not yet | — | — | — | — |
| P1 | sinusoidal positional encoding | not yet | — | — | — | — |
| P1 | LayerNorm + residual placement | not yet | — | — | — | — |
| P1 | padding vs causal masking | not yet | — | — | — | — |
| P1 | label smoothing + warmup schedule | not yet | — | — | — | — |
