# DECISIONS — architecture decision record

Append-only. Each entry: context → decision → consequences → status. Never silently rewrite a
decision; supersede it with a new numbered entry and mark the old one `SUPERSEDED by D-0xx`.

---

## D-001 — Python environment: venv at `.venv`, CPU-only PyTorch, pinned
**Date:** 2026-09-15 · **Status:** ACCEPTED

**Context.** System Python 3.12.10 has no torch/numpy. Only 31.4 GB disk free. PowerShell
execution policy is Restricted, so `.venv\Scripts\Activate.ps1` cannot run.

**Decision.** Create `.venv` with `python -m venv`. Never activate it — invoke
`.venv\Scripts\python.exe` by path in every command and script. Install the **CPU** wheel index
(`--index-url https://download.pytorch.org/whl/cpu`) so we do not pull ~2.5 GB of unusable CUDA
libraries onto a 31 GB disk. Pin exact versions in `requirements.txt`.

**Consequences.** Commands are more verbose. `torch.cuda.is_available()` is `False` everywhere;
all device handling must be written device-agnostic from day one so the same code runs unchanged
if a GPU is ever attached. Saves an estimated ~2 GB of disk versus the default CUDA wheel.

---

## D-002 — Application & deployment stack: static site on GitHub Pages, in-browser inference
**Date:** 2026-09-15 · **Status:** ACCEPTED

**Context.** The brief requires a public portfolio URL plus an interactive working demo per
project, usable without signing in, with bounded inference cost and training kept away from public
requests. Available credential: `gh` CLI authenticated as `zexoraai` with `repo` + `workflow`
scopes. No cloud account, no payment authorisation, no server budget.

**Rejected options.**
- *Hugging Face Spaces (Gradio/Streamlit).* Natural fit for ML demos, free CPU tier. Rejected for
  now: no HF token in this environment, and a server-side demo needs concurrency/timeout
  policing we would have to build and cannot load-test. Kept as a documented optional path for
  P8 (DDPM), the one project whose demo is genuinely too heavy for a browser at useful quality.
- *Vercel / Netlify / Render.* No account, and each adds a paid-tier cliff.
- *Self-hosted inference API.* Laptop is not a server; violates "training separate from public
  demo requests" in the worst way and would expose the user's home network.

**Decision.** **GitHub Pages serving a static site from `docs/` on the default branch**, with all
demo inference running **client-side in the visitor's browser**.

Consequences that make this the right call rather than a compromise:
- Inference cost is structurally bounded — it runs on the visitor's machine, so there is no
  shared resource to exhaust and no concurrency limit to enforce. Input/output caps and step
  limits are still enforced in the UI for honesty about latency.
- Training is *physically* separated from serving: the site contains no training code path.
- Zero cost, zero credentials in the deployed artefact, no sign-in.
- Weights must ship as static files → forces the models to stay small, which is already forced
  by the hardware. Constraint and design agree.

**Consequences (costs).** Model size caps: hard limit 25 MB per demo weight file, target < 5 MB.
Anything larger must be a *labelled* cached/prerecorded artefact instead of live inference.
GitHub Pages has no server-side compute, so no demo may require it.

**Frontend technology.** Vanilla HTML + CSS + ES modules. No React/Vite/Next.
Rationale: nine mostly-static pages with a handful of interactive widgets do not need a framework;
`node_modules` would cost 200–400 MB against a 31 GB disk; a build step adds a failure mode
between "works locally" and "works deployed"; and hand-written DOM code is auditable by a
visitor who clicks *view source*, which suits a portfolio whose whole premise is transparency.

---

## D-003 — Two in-browser inference engines, chosen per project by instrumentation need
**Date:** 2026-09-15 · **Status:** ACCEPTED

**Context.** The demos must show *genuine* model outputs and also expose internals (attention
per head, tensor shapes at boundaries, per-step token probabilities, quantization error). A
black-box runtime gives outputs but not internals.

**Decision.**
- **Engine A — hand-written JS numeric core** (`docs/assets/js/nn/`): matmul, softmax, layernorm,
  gelu/relu, attention. Used where deep instrumentation *is* the demo: P1 (attention heads, mask
  views, shape inspector), P3 (adapter merge equivalence), P5 (tile-by-tile walkthrough).
  Weights ship as a `.bin` + `.json` manifest.
- **Engine B — ONNX Runtime Web (WASM)** for models where a hand-written forward pass is not
  worth the maintenance risk: P2 (GPT sampling), P6, P7, P8.

**Non-negotiable guard.** Either engine must pass a **parity test against PyTorch** on fixed
inputs with a stated tolerance, and the tolerance must be published on the project page. A
browser demo that silently disagrees with the trained model is a fabricated output. Parity
fixtures live in `evidence/<project>/parity/`.

**Consequences.** Engine A duplicates a small amount of math in two languages. Accepted: the
duplication is itself a correctness check, and it is the only way to make the internals demos
truthful rather than illustrative.

---

## D-004 — Project 5: Triton kernel authored locally, executed on external free GPU
**Date:** 2026-09-15 · **Status:** ACCEPTED (execution pending)

**Context.** `nvidia-smi` absent; GPU is integrated AMD Radeon; OS is Windows. Triton emits GPU
code (NVIDIA PTX / AMD ROCm) and has no supported path on this machine. The brief demands a
"functioning Triton implementation" *and* forbids presenting simulations as running kernels.

**Decision.** Split the deliverable by what can be honestly verified where:
1. `reference_attention` — transparent PyTorch, runs and is measured here.
2. `tiled_attention` with online softmax (Milakov & Gimelshein recurrence) — pure PyTorch with
   explicit tile loops, runs and is measured here, numerically compared to (1).
3. `triton_attention` — real Triton kernel source, shipped with `bench_triton.py` (warm-up,
   `torch.cuda.synchronize()`, repeated trials, tolerance-checked parity vs (1)) **plus** a
   self-contained Colab notebook that runs it on a free T4.

Until (3) has actually been executed, the project page states **"Triton kernel: authored, not yet
executed on GPU — no timings available"** and shows no numbers for it. When executed, the page
records the GPU model, driver, Triton version, and that the measurement came from an external
runtime, not from this laptop.

**Consequences.** The kernel's *correctness* claim is deferred until execution. The tiled-softmax
*algorithm* claim is fully verifiable here today. This is the honest partition and it is stated on
the page rather than hidden.

---

## D-005 — Repository layout: one repo, `labs/` python package, `docs/` site
**Date:** 2026-09-15 · **Status:** ACCEPTED

**Decision.** Single public repo. `labs/pN_<name>/` per project (importable Python package),
`labs/common/` for genuinely shared utilities *only after* a second project needs them (no
speculative abstraction), `tests/` mirroring `labs/`, `docs/` = deployed Pages site,
`evidence/` = run artefacts, `records/` = these persistent records, `mastery/` = mastery packs,
`scripts/` = entry points.

**Consequence.** `labs/common/` starts nearly empty by design. Utilities are promoted into it when
a second caller appears, not in anticipation of one.

---

## D-006 — Project 1 normalisation placement: post-norm (paper), pre-norm behind a flag
**Date:** 2026-09-15 · **Status:** ACCEPTED

**Context.** arXiv:1706.03762v7 §3.1 states the sub-layer output is `LayerNorm(x + Sublayer(x))`
— post-norm. The authors' own `tensor2tensor` implementation applies normalisation *before* the
sub-layer and adds the residual after, i.e. pre-norm. Papers and code disagree; most modern
implementations follow the code. The brief says "original."

**Decision.** Default `norm_style="post"`, matching the paper text, which is what we cite.
`norm_style="pre"` available as a flag so the difference is a demonstrable experiment rather than
a footnote. The project page names the discrepancy explicitly and cites §3.1.

**Consequence.** Post-norm at 6 layers without warmup is known to be unstable; we therefore keep
the paper's warmup schedule (§5.3) rather than a constant LR, and our depth is small enough that
this is not a blocker. The pre/post comparison becomes a legitimate small experiment for the page.

---

## D-008 — All Python/PyTorch execution moves into a Linux container (SAC blocker)
**Date:** 2026-09-15 · **Status:** ACCEPTED · **Supersedes the execution half of D-001**

**Context — a hard blocker discovered by running the code, not by reading docs.** With
`torch==2.14.0+cpu` correctly installed into `.venv` on the Windows host, `import torch` fails:

```
OSError: [WinError 4551] An Application Control policy has blocked this file.
Error loading "...\.venv\Lib\site-packages\torch\lib\shm.dll" or one of its dependencies.
```

Diagnosis (measured, `HKLM\SYSTEM\CurrentControlSet\Control\CI\Policy`):
`VerifiedAndReputablePolicyState = 1`, `SAC_EnforcementReason = 3` → **Windows Smart App Control
is ON and enforcing.** SAC refuses to load binaries it does not judge verified-and-reputable.
`numpy 2.5.3` imports fine (its wheels are signed/reputable); PyTorch's own DLLs are unsigned, so
they are blocked. This is a host policy, not a PyTorch bug, and no amount of reinstalling fixes it.

**Rejected option: turn Smart App Control off.**
It would work, and it is the first thing most guides suggest. Rejected on two grounds.
(a) It is a machine-wide security downgrade affecting every application on the user's laptop,
made to satisfy a hobby project's dependency. (b) It is effectively one-way: Microsoft's own
guidance is that re-enabling SAC generally requires resetting or reinstalling Windows, with only
recent Windows builds able to re-enable it without a clean install
([Smart App Control FAQ](https://support.microsoft.com/topic/285ea03d-fa88-4d56-882e-6698afdb7003)).
Trading an irreversible security change for a `pip install` is a bad trade and not ours to make
unilaterally. *(Sources rephrased for compliance with licensing restrictions.)*

**Rejected option: pure-numpy implementations.** numpy loads fine, so this would work. Rejected:
the brief specifies PyTorch for model implementations, and hand-rolling autograd across eight
projects would replace the subject matter with an unrelated engineering problem.

**Decision.** Run every Python/PyTorch process inside a **Linux container** on the already-present
Docker Desktop 4.72.0 (engine 29.4.2, `linux/amd64`, WSL2 backend). SAC governs Windows PE
binaries; Linux ELF binaries inside the WSL2 VM are outside its scope, so torch loads normally
**without weakening the host's security posture at all**. Artefacts: `Dockerfile`,
`requirements.txt` (pinned), `run.cmd` (short wrapper — the driving shell truncates long commands,
and `.ps1` is blocked by execution policy).

**Consequences.**
- *Positive, and not a small one:* the environment is now pinned and reproducible by construction.
  A `Dockerfile` plus a `pip freeze` captured per run is a far stronger reproducibility record
  than "it worked on my laptop", which the evidence schema in EXPERIMENTS.md demands anyway.
- Every command gains a `run.cmd` prefix. Paths inside the container are `/work/...`.
- Disk cost ≈ 1.5–2 GB for the image, inside the WSL2 VHDX on C:. Budget in GAPS G-002 adjusted.
- The Windows `.venv` torch install (~1.2 GB) is dead weight and is removed to reclaim disk. The
  venv itself is kept only for numpy-only host-side scripting.
- Benchmarks now measure *containerised Linux* CPU performance, not native Windows. That is the
  honest label and it goes on every timing we publish. Thread count is pinned in the image
  (`OMP_NUM_THREADS=6`, matching 6 physical cores) so timings are comparable run to run.
- A latent upside for P5: a Linux userspace makes an experimental `triton` CPU backend at least
  *possible* to attempt, which was a non-starter on Windows. Still gated on D-004.

---

## D-009 — Cloud training is a planned second phase; everything must be built portable now
**Date:** 2026-09-16 · **Status:** ACCEPTED (user-directed)

**Context.** The user has stated the plan: finish all eight projects locally, then move to cloud GPUs
for further training. That is a sequencing decision, not a change of scope, and it is authorised in
principle — but no paid resource is provisioned until an actual cost estimate has been approved
against a specific run.

**What this changes about work done *before* the move.** The expensive failure mode is finishing
eight CPU-shaped projects and then rewriting all of them to run on a GPU. Avoiding that costs almost
nothing if it is done as we go, so it is done as we go:

1. **Device-agnostic by construction.** No `.cpu()` in model or training code; a single `--device`
   argument resolved once and threaded through. Tensors are created on the parameter's device rather
   than defaulted. `torch.cuda.is_available()` is checked in exactly one place. Already largely true
   because the environment forced it (`cuda_available = False` everywhere), which turns out to have
   been useful discipline rather than a limitation.
2. **Checkpoints are portable.** Saved from CPU tensors and loaded with `map_location`, so a
   CPU-trained checkpoint resumes on a GPU and vice versa. Already the case; now it is a requirement
   rather than an accident, and the resume test is what protects it.
3. **Scale lives in CLI flags and config files, never in constants.** Every run directory already
   contains the exact `argv` and a config dict. Moving to a larger model must be a different
   command, not a different codebase. `GPTConfig.gpt2_small()` exists precisely so the 124 M target
   is a config change.
4. **Mixed precision is a hook, not a rewrite.** Training loops keep the loss scaling and autocast
   insertion points obvious and unused on CPU, so enabling bf16/fp16 on a GPU is a flag rather than
   surgery.
5. **Dependencies are split by target.** `requirements.txt` pins the **CPU** wheel index because a
   CUDA build cannot run here and would cost ~2 GB against a 31 GB disk. A sibling
   `requirements-cuda.txt` is added when the move happens, pinning the same library versions against
   the CUDA index so results stay comparable.

**What the move unlocks, and what it must not be allowed to obscure.**

| Currently blocked | Unblocked by cloud | Must still be labelled |
|---|---|---|
| P5's Triton kernel cannot be compiled or benchmarked (D-004, G-001) | Executes and gets real timings | The GPU model, driver and Triton version; timings never attributed to this laptop |
| P2's ~124 M target is a documented estimate only (D-007c) | Becomes attemptable | Any result is a *new* run with its own record; the small-model results are not retroactively upgraded |
| P6 low-bit **compute** speedup is unlikely on this CPU | Measurable on a GPU | Simulated vs packed-storage vs genuine low-bit compute stay three distinct claims |
| P7/P8 are wall-clock-bound at 3–6 h each | Hours instead | Reduced-scale results already published are not relabelled as reproductions |

**The rule that does not change.** Moving to more compute changes what we can *attempt*; it does not
change the fidelity tiers in SCOPE.md §0. A tier-R result trained on a rented A100 is still tier R
unless it matches a published result at published scale. Cloud access is not a licence to upgrade
existing claims.

**Consequence for cost estimates.** Every project's page and `env/feasibility.md` already carries a
*measured* local throughput figure. Those become the basis for honest GPU estimates by ratio rather
than by guesswork, so the user gets a real number before any card is charged. `estimate_flops_per_token`
on the GPT exists for exactly this.

---

## D-007 — Deferred decisions (do not guess these; measure first)
**Status:** OPEN

| # | Decision | Blocked on | Owner note |
|---|---|---|---|
| D-007a | P6 named PTQ method: GPTQ vs AWQ | P2 model existing; measured CPU cost of GPTQ's layer-wise Hessian inverse vs AWQ's activation-scale search at our model size | Choose the one we can run *faithfully*, not the more famous one |
| D-007b | P7 dataset + licence | Licence review of Flickr8k/30k vs building captions over a permissive image set | Must be resolved **before** any download; record licence text location |
| D-007c | P2 124 M-param target: attempt or document-only | Measured tokens/sec from the P2 benchmark → honest wall-clock estimate | Do not commit to paid compute without explicit approval |
| D-007d | P8 demo engine: ORT Web vs prerecorded grids | Measured browser sampling latency for T=200 reverse steps | If > ~10 s, ship labelled prerecorded runs + a shorter live option |
