# GAPS — known limitations, open risks, and things we are not claiming

Honest register of what is missing. This file is the antidote to portfolio inflation: anything a
visitor might reasonably assume, but which is not true, belongs here and on the relevant page.

---

## G-001 — No CUDA device on the development machine
**Severity:** high · **Affects:** P2 (scale), P5 (Triton), P6 (low-bit compute), P7/P8 (training time)

`nvidia-smi` is absent; the GPU is an integrated AMD Radeon (Vega) under Windows. Consequences:
- Triton kernels cannot be compiled or benchmarked locally (see DECISIONS D-004).
- All training is CPU training, so every experiment is scale-limited by wall clock, not by ideas.
- Any claim about GPU kernel performance must come from an external runtime and be labelled with
  that runtime's hardware.

**Exact remaining requirement to close this gap:** access to one NVIDIA GPU with CUDA ≥ 12 and
Triton support (a free Colab T4 session suffices for P5 correctness + timing; a rented A10/A100
hour would be needed for the P2 124 M target).

---

## G-002 — 31.4 GB free disk
**Severity:** medium · **Affects:** all dataset choices

Rules out WMT14, OpenWebText, LAION, ImageNet. Enforced budget: deps ≤ 3 GB, datasets ≤ 2 GB
total, checkpoints ≤ 1 GB total. Every dataset decision must state its on-disk footprint.

---

## G-003 — Tier P (published-result reproduction) is out of reach for all eight projects
**Severity:** by design · **Affects:** every claim

Stated up front so no page has to weasel. See SCOPE.md §0.

---

## G-004 — Single-seed results where compute forbids repeats
**Severity:** medium · **Affects:** any run > 10 min

Mitigation: cheap experiments get ≥ 3 seeds; expensive ones are explicitly labelled
`SINGLE RUN, no variance estimate` and no significance is claimed from them.

---

## G-005 — Attention visualisations are not causal explanations
**Severity:** conceptual, must be stated on P1's page

Attention weights show *where a weighted average drew from*, at one layer, in one head, for one
input. They do not establish that the information was used, that the head is doing the job the
picture suggests, or that changing the input would change the output as implied. Residual streams
mix information across layers; a head with high weight on a token can still contribute nothing to
the logit that matters. Establishing use requires intervention (ablation, patching), not
inspection. P1's page will say this next to the visualiser, not in a footnote.

---

## G-006 — Browser demo parity risk
**Severity:** medium · **Affects:** every demo (D-003)

A hand-written JS forward pass or an ONNX export can silently diverge from the trained PyTorch
model. Unmitigated, this turns "genuine model output" into a fabrication. **Mitigation is
mandatory, not optional:** each demo ships a parity fixture (fixed inputs → PyTorch outputs) and
the page publishes the measured max absolute deviation and the tolerance.

---

## G-008 — The development machine is not an isolated benchmark host
**Severity:** high for any timing claim · **Affects:** every measured number in this portfolio

At the time of the first benchmarks, `docker ps` showed **37 unrelated containers running**, all up
seven hours: OpenSearch 2.19, ClamAV 1.4, rspamd, ten-ish Postgres/pgvector instances, several
Redis, MinIO, MongoDB, NATS, Temporal (server + UI), LiteLLM, Mailpit, and multiple application
services. They belong to the user's other projects and are **not ours to stop**.

Measured impact — the same workload, minutes apart:

| Measurement | Quieter moment | Contended moment | Ratio |
|---|---|---|---|
| 2048² fp32 matmul, 6 threads | 95.8 ms (179.4 GFLOP/s) | 266.6 ms (64.4 GFLOP/s) | 2.8x |
| 256² fp32 matmul, 4 threads | 1.36 ms | 19.17 ms | 14x |
| 0.53 M-param training step, 2 threads | 100.5 ms | 197.7 ms | 2.0x |

Consequences, applied consistently from here on:
1. **Every timing is a lower bound** on this hardware's capability, and is labelled as such.
2. Every benchmark record carries a load snapshot (`labs/common/sysload.py`) so an outlier is
   annotated rather than mysterious.
3. Extrapolated training durations are reported as a **range** spanning quiet and contended
   throughput, never as a single number.
4. Thread-count conclusions are only reported where they hold across *both* contention regimes.
   The one that does: the tiny configuration is fastest at **2 threads**, and 6 threads is
   actively harmful for it. That reproduced in every run.

**Exact remaining requirement to close this gap:** one benchmarking window with the unrelated
container stack stopped (`docker stop $(docker ps -q)` — the user's call, not ours, since those
services hold state and have been up for hours). Until then, treat all absolute timings as
"achievable while the dev stack is running".

---

## G-007 — Deployment credential scope
**Severity:** low · **Affects:** M1.4

`gh` is authenticated as `zexoraai` with `repo`/`workflow`. Creating a **public** repository under
that account is a visible, hard-to-undo-quietly action. It is authorised by the brief's explicit
request for a public portfolio URL, but the repo name is confirmed with the user before creation.
