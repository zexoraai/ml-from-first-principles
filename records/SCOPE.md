# SCOPE — Eight-Project ML Research Portfolio

**Owner:** titus (GitHub: `zexoraai`)
**Role of this file:** authoritative statement of what we are building, at what fidelity,
and what each project is *not*. Read this before resuming work. Update it when scope changes.

---

## 0. Fidelity vocabulary (used everywhere, no exceptions)

We label every artefact with exactly one of these three tiers. This vocabulary exists to stop
the single most common portfolio lie: implying a paper was reproduced when a mechanism was merely
implemented.

| Tier | Name | Means | Allowed claim |
|---|---|---|---|
| **E** | Educational implementation of a mechanism | The mechanism is implemented from the paper's equations and verified for *correctness* (numerical parity against a reference, gradient checks, shape/mask invariants). Trained on a toy task or not trained at all. | "I implemented and verified mechanism X." |
| **R** | Reduced-scale reproduction | The full method runs end-to-end on real data at a scale far below the paper. Trends may be compared to the paper qualitatively. Absolute numbers are **not** comparable. | "At 1/1000 scale the method behaves qualitatively as reported: <trend>." |
| **P** | Reproduction of published results | Same dataset, same scale, metric within published error bars. | "I reproduced the published result." |

**On this hardware, tier P is unreachable for every one of the eight projects.** No project will
claim it. Where a project could reach P with rented compute, the record states the exact missing
resource and the measured basis for the estimate.

---

## 1. Environment as measured (2026-09-15)

Measured, not assumed. Raw probe transcript: `evidence/env/probe-2026-09-15/`.

| Property | Value | Consequence |
|---|---|---|
| Machine | HP EliteBook 845 G8 Notebook PC | Mobile thermal envelope; sustained clocks < burst clocks |
| CPU | AMD Ryzen 5 PRO 5650U, 6 cores / 12 threads, Zen 3 | All training is CPU training |
| RAM | 33,622,650,880 B = 31.3 GiB | Generous; not the binding constraint |
| GPU | AMD Radeon(TM) Graphics (integrated Vega), driver 31.0.21924.61 | **No CUDA. `nvidia-smi` absent.** |
| Disk free | **31.4 GB on C:** | *Binding constraint.* Datasets must be MB-scale, not GB-scale |
| OS / shell | Windows, PowerShell | Execution policy blocks `*.ps1` → cannot use venv `Activate.ps1`, must call venv python by absolute path; `npm` must be invoked as `npm.cmd` |
| Python | 3.12.10 (`AppData\Local\Programs\Python\Python312`) | |
| Preinstalled pkgs | attrs, boto3, botocore, hypothesis, jmespath, pglast, python-dateutil, s3transfer, six, sortedcontainers, urllib3 | **No torch, no numpy** — clean slate |
| Node | v24.13.0, npm 11.6.2 | Available for site build if needed |
| Git | 2.53.0.windows.1 | |
| `gh` CLI | Authenticated as **`zexoraai`**, scopes `gist`, `read:org`, `repo`, `workflow` | **Public URL is achievable** via GitHub Pages + Actions |

### The two constraints that reshape the brief

**C1 — No CUDA device.** Project 5 requires "a functioning Triton implementation." Triton compiles
to GPU targets (NVIDIA PTX / AMD ROCm). There is no such device here and no ROCm on Windows.
Therefore Project 5 splits: the tiled/online-softmax algorithm is implemented and *measured* in
PyTorch on this CPU, and the Triton kernel is *authored and shipped with a parity+benchmark
harness* that is executed on a free external GPU runtime. Until that harness has been run, the
project page will state "Triton kernel: authored, not yet executed" — it will not display invented
timings. See `records/DECISIONS.md` D-004.

**C2 — 31.4 GB free disk.** Rules out WMT14 (~4.5 M pairs), OpenWebText (~40 GB), LAION, ImageNet.
Every dataset must be small and permissively licensed. Budget: torch CPU + deps ≈ 1.5–3 GB,
all datasets combined ≤ 2 GB, all checkpoints combined ≤ 1 GB.

---

## 2. The eight projects

Order is fixed and dependency-driven: 1 → 2 gives us a base model that 3 → 4 fine-tune and align;
5 optimises the attention primitive from 1–2; 6 compresses the model from 2; 7 and 8 are
independent architectures that reuse the shared training/eval/evidence infrastructure.

### P1 — Encoder–decoder Transformer
- **Primary source:** Vaswani et al., *Attention Is All You Need*, arXiv:1706.03762**v7** (2 Aug 2023 revision of the Jun 2017 paper).
- **Version we implement:** the **paper's** post-norm formulation, `LayerNorm(x + Sublayer(x))` (§3.1) — *not* the pre-norm variant that the authors' own `tensor2tensor` release actually shipped. Both are provided behind a flag so the difference is demonstrable; post-norm is the default because the brief says "original."
- **Mechanisms hand-written (no `torch.nn.Transformer`, no `torch.nn.MultiheadAttention`, no `F.scaled_dot_product_attention`):** scaled dot-product attention (Eq. 1), multi-head projection/split/concat (§3.2.2), sinusoidal positional encoding (§3.5), LayerNorm, residual + dropout placement (§5.4), position-wise FFN (§3.3), padding mask, causal mask.
- **Target tier:** **E** for the mechanisms (parity + gradient + mask + overfit checks), **R** for the seq2seq task.
- **Not claimed:** WMT14 BLEU. Not attempted. Not attemptable here.

### P2 — Decoder-only GPT
- **Primary sources:** Radford et al. GPT-2 (*Language Models are Unsupervised Multitask Learners*, 2019); Karpathy `nanoGPT`; Karpathy `llm.c`.
- **Target tier:** **R** for a small model trained to a documented validation loss on a small permissive corpus. The ~124 M-parameter GPT-2-small configuration is a **separate, gated target**: it gets a *measured* compute estimate derived from a benchmark on this machine, and is only attempted if a funded/free GPU path is explicitly approved.
- **Not claimed:** GPT-2 zero-shot benchmark numbers.

### P3 — LoRA
- **Primary source:** Hu et al., *LoRA: Low-Rank Adaptation of Large Language Models*, arXiv:2106.09685.
- Hand-written: `B A` low-rank update, `α/r` scaling, frozen base, merge/unmerge.
- **Comparison arm required by brief:** base vs LoRA vs **full** fine-tune of *the same* model. Chosen base model must be small enough that full fine-tuning genuinely runs on this CPU, otherwise the comparison is a fiction. That is the sizing criterion.
- **Target tier:** **E** for the adapter mechanism + merge-equivalence; **R** for the fine-tuning comparison.

### P4 — DPO vs SFT
- **Primary source:** Rafailov et al., *Direct Preference Optimization*, arXiv:2305.18290.
- Hand-written: the DPO loss, reference-policy log-prob handling, sequence log-probs under response-only masking.
- Both arms start from the **same SFT checkpoint**. Preference data provenance documented; eval prompts disjoint from training.
- **Target tier:** **E** + **R**.
- **Not claimed:** that a preference-score improvement is a safety or capability improvement.

### P5 — Tiled attention → Triton
- **Primary sources:** Dao et al. *FlashAttention* arXiv:2205.14135; Dao *FlashAttention-2* arXiv:2307.08691; Milakov & Gimelshein *Online normalizer calculation for softmax* arXiv:1805.02867; official Triton tutorial `06-fused-attention`.
- Three artefacts: (a) transparent reference attention, (b) tiled attention with online softmax in PyTorch — measurable here, (c) Triton kernel — authored here, executed externally (see C1).
- **Target tier:** **E** for all three. Forward/backward status stated explicitly per artefact. No "training speedup" claim unless backward is implemented *and* measured.

### P6 — Quantization
- **Primary sources:** baseline = symmetric/asymmetric min-max affine INT8; named PTQ method = **GPTQ** (Frantar et al., arXiv:2210.17323) *or* **AWQ** (Lin et al., arXiv:2306.00978) — selection deferred until P2's model exists and the CPU cost of the Hessian/activation pass is measured. Decision to be recorded in DECISIONS.md.
- Must distinguish *simulated* (fake-quant in fp32) from *packed low-bit storage* from *low-bit compute*. On this CPU, genuine low-bit *compute* speedup is unlikely; that will be reported as measured, not wished for.
- **Target tier:** **E** + **R**.
- **Not claimed:** that rounding pretrained weights to {-1,0,1} reproduces BitNet. BitNet-style ternary *training* is a clearly separated optional extension.

### P7 — Miniature CLIP
- **Primary source:** Radford et al., *Learning Transferable Visual Models From Natural Language Supervision*, arXiv:2103.00020.
- Hand-written: dual encoders, L2-normalised embeddings, learnable temperature (logit scale, clamped as in the paper's implementation), symmetric InfoNCE.
- Data: small, explicitly licensed image–text pairs with documented splits (candidate: Flickr8k / Flickr30k subject to licence review, or a synthetic-caption dataset built from a permissive image set — decision recorded before any download).
- **Target tier:** **E** + **R**. Encoders trained **from scratch** at miniature scale; the page explains why from-scratch recall at this scale is nowhere near CLIP's, and what adapting pretrained encoders would change.
- **Metric:** Recall@{1,5,10} on a held-out split, both directions.

### P8 — Unconditional DDPM
- **Primary source:** Ho et al., *Denoising Diffusion Probabilistic Models*, arXiv:2006.11239.
- Hand-written: forward `q(x_t|x_0)` closed form, timestep embedding, U-Net denoiser, simplified `L_simple` objective, ancestral reverse sampler.
- Data: small permissive image set at low resolution (28–32 px).
- **Target tier:** **E** + **R**. FID will **not** be reported unless sample count and reference statistics make it meaningful; if not, we report what is honest (loss curves, fixed-seed sample grids, checkpoint-to-checkpoint comparison) and say why FID is omitted.

---

## 3. Deliverable per project (all eight)

1. Working implementation, mechanisms hand-written where the brief demands it.
2. Correctness suite appropriate to the risk (parity, gradient, invariant, overfit).
3. At least one completed, recorded experiment with the full evidence schema (`records/EXPERIMENTS.md`).
4. A public project page: research question, interactive demo, architecture walkthrough, equations wired to code, results, limitations + failure examples, source links, evidence bundle, case study.
5. A 14-step composition teaching document + three explanation depths (30 s / 5 min / technical).
6. A mastery pack (10 defence Qs + answers, 5 shape exercises, 3 numerical, 3 planted bugs, 2 blind implementations, 1 prediction experiment, spaced schedule).

A project is **done** only when its deployed URL has been fetched and verified.

---

## 4. Explicit non-goals

- No paid infrastructure without prior approval.
- No fabricated metrics, no invented URLs, no animation presented as a kernel or a trained model.
- No `torch.nn` shortcut for any mechanism the brief says to implement by hand.
- No speculative abstraction layers. Each project's code exists to make its own experiments run.
- No claim of authorship on the user's behalf that he cannot defend in an oral exam.
