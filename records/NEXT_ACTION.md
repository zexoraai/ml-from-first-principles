# NEXT ACTION

Single source of truth for "what happens next". Read this first when resuming. Keep it to one
active item plus the queue. Update it **before** context compaction.

---

## Active

**M1.1 — Project 1, milestone T1: hand-written Transformer mechanisms.**

Exact behaviour being completed: every mechanism named in arXiv:1706.03762v7 exists as an
independently testable function/module in `labs/p1_transformer/`, and each one passes a correctness
check that would fail if the mechanism were wrong.

Definition of done for M1.1:
- [ ] `attention.py`: `scaled_dot_product_attention` (Eq. 1) returning `(output, weights)`
- [ ] `attention.py`: `MultiHeadAttention` with explicit split/concat, own `W^Q,W^K,W^V,W^O`
- [ ] `positional.py`: `sinusoidal_positional_encoding` (§3.5)
- [ ] `layers.py`: `LayerNorm` (hand-written, not `nn.LayerNorm`), `PositionwiseFeedForward` (§3.3),
      `SublayerConnection` implementing `LayerNorm(x + Dropout(Sublayer(x)))` (§3.1 + §5.4)
- [ ] `masks.py`: `padding_mask`, `causal_mask`, `combine_masks`
- [ ] Correctness suite passing: numeric parity vs an independent reference, mask invariants
      (a masked position cannot influence any output), gradient flow (no None/NaN grads),
      PE identity checks, LayerNorm statistics, shape contracts at every boundary
- [ ] Zero use of `nn.Transformer`, `nn.MultiheadAttention`, `F.scaled_dot_product_attention`,
      `nn.LayerNorm` in the demonstrated path — enforced by a test that greps the source

## Queue (in order)

1. M0.4 — CPU benchmark + feasibility table (needs torch install to finish)
2. M1.2 — model assembly, greedy decode, single-batch overfit proof
3. M1.3 — seq2seq task + training + held-out eval, first EXPERIMENTS.md entry
4. M1.4 — repo public, Pages live, URL verified (**confirm repo name with user first — G-007**)
5. M1.5 — P1 page + in-browser demo with published parity tolerance (**stop for visual review here**)
6. M1.6 — P1 mastery pack + first back-explanation quiz

## Open questions for the user (do not guess)

- **Q1 (blocks M1.4):** repo name. Proposal: `ml-from-first-principles`. Public under `zexoraai`.
- **Q2 (blocks P5 timings):** is a free Google Colab session available to execute the Triton
  benchmark? If not, P5 ships as "kernel authored, not executed" indefinitely.
- **Q3 (blocks D-007c):** any budget at all for rented GPU hours, or is free-tier-only the hard rule?

## Do-not-forget

- Commands in this environment: call `.venv\Scripts\python.exe` by path (no activation, D-001);
  use `npm.cmd` not `npm`; keep shell command strings short — the harness truncates long ones.
- The shell tool reports exit code -1 even on success; verify by inspecting output files, not codes.
- Locale renders decimals with commas in PowerShell output; don't misread `31,4GB` as 314 GB.
