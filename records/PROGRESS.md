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
| M1.2 | P1 T2: model assembly + overfit proof | `model.py`, `optim.py`, `generate.py`. Single-batch overfit to loss < 0.05 **and** exact greedy reconstruction, for both norm styles. Parameter count checked against a hand derivation | E | DONE |
| M1.3 | P1 T3: seq2seq training + held-out eval | Date-normalisation task, date-disjoint splits with an asserted leakage guard, resumable training, three defined metrics | R | DONE |
| M1.5 | P1 T5: project page + demo | Live in-browser inference, attention head viewer, per-step probabilities, published parity tolerance vs PyTorch | — | DONE |
| M1.6 | P1 mastery pack | 10 defence questions + separate answer notes, 5 shape exercises, 3 numerical, 3 planted bugs, 2 blind implementations, 1 prediction experiment, spaced schedule | — | DONE |

### Test suite growth

| Milestone | Tests | Added |
|---|---|---|
| M1.1 (mechanisms) | 79 | attention, positional, layers, masks, no-shortcuts |
| M1.2–M1.3 (model, training, data) | **161** | model assembly, overfit proof, optim, data/splits, checkpoint resume |

### Bugs found by the suite (not by inspection)

| # | Bug | Found by | Kind |
|---|---|---|---|
| 1 | Benchmark harness wrong by 300x (single-call timing) | internal contradiction: 27 M-param step faster than 0.53 M | measurement |
| 2 | `zero_padded_abbr` byte-identical to `day_abbr_year` for all 2-digit days | `test_a_date_is_rendered_in_distinct_formats_without_repetition` | data |
| 3 | "May" abbreviation equals its full name → 2 more renderer collisions | `test_all_renderers_are_pairwise_distinct` (new) | data |
| 4 | Padding test asserted a false invariant (mask is derived from ids) | the test failed; the model was right | test |
| 5 | `LRScheduler.load_state_dict` does not restore `optimizer.param_groups` lr | `test_scheduler_state_survives_a_round_trip` | resume |
| 6 | "best checkpoint" froze at first eval while exact-match was 0.0 | smoke run inspection | training |

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

**Status: the pack is written and unused.** `mastery/p1_transformer.md` (questions, exercises,
planted bugs, prediction experiment, spaced schedule) and `mastery/p1_transformer_answers.md`
(separate answer notes) both exist. **Nothing below has been assessed.** Project 1's implementation is
complete; the user's understanding of it is entirely unverified, and this table is the only place that
distinction is recorded.

| Project | Mechanism | Explained back? | Reasoning quality | Gaps found | Retaught | Next review |
|---|---|---|---|---|---|---|
| P1 | scaled dot-product attention (A1) | not yet | — | — | — | day 0 |
| P1 | three uses of attention / Q,K,V sources (A2) | not yet | — | — | — | day 1 |
| P1 | sinusoidal positional encoding (A3) | not yet | — | — | — | day 0 |
| P1 | post- vs pre-norm and warmup (A4) | not yet | — | — | — | day 3 |
| P1 | tensor shapes end to end (A5) | not yet | — | — | — | day 1 |
| P1 | padding masks constrain keys not queries (A6) | not yet | — | — | — | day 0 |
| P1 | label smoothing trade-off (A7) | not yet | — | — | — | day 3 |
| P1 | diagnosing low loss + bad generation (A8) | not yet | — | — | — | day 7 |
| P1 | weight tying (A9) | not yet | — | — | — | day 7 |
| P1 | epistemic limits of attention maps (A10) | not yet | — | — | — | day 14 |

**Bar for "can defend this project":** score 3 on A1, A4, A6, A8, A10 and ≥2 elsewhere. Those five
separate having built it from understanding it — the scaling argument, the paper-versus-code
discrepancy, the masking asymmetry, the diagnostic reasoning, and the limits of interpretability
pictures.
