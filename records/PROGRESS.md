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
| M0.3 | Repo skeleton + pinned CPU venv | `.venv` importable torch; layout per D-005 | — | IN PROGRESS |
| M0.4 | Measured feasibility table | Real benchmark on this CPU → `evidence/env/` + `docs/` feasibility page | — | PLANNED |
| M1.1 | **P1 T1: Transformer mechanisms** | Every mechanism from arXiv:1706.03762 hand-written and correctness-checked in isolation | E | PLANNED |
| M1.2 | P1 T2: model assembly + overfit proof | Full encoder–decoder forward; overfit a single batch to ~0 loss; greedy decode | E | PLANNED |
| M1.3 | P1 T3: seq2seq training + held-out eval | Recorded run, checkpoint/resume, val metrics with seeds | R | PLANNED |
| M1.4 | P1 T4: repo public + Pages live | Verified public URL responds | — | PLANNED |
| M1.5 | P1 T5: project page + demo | Live in-browser inference with parity test vs PyTorch | — | PLANNED |
| M1.6 | P1 mastery pack | Defence questions, exercises, planted bugs, prediction experiment | — | PLANNED |

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
