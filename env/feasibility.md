# Feasibility — measured, then extrapolated

**Rule of this document.** Anything under a *Measured* heading was executed on this machine and is
traceable to a run_id in `records/EXPERIMENTS.md`. Anything marked **ESTIMATE** is arithmetic on
top of those measurements, with its assumptions written out. No number here comes from a
datasheet, a vendor TFLOPS figure, or a paper's reported hardware.

---

## 1. Measured capability

Source: run `env-bench-02` (`evidence/env/cpu_benchmark.json`) and the diagnostic
`evidence/env/thread-diagnosis-transcript.txt`. Both were taken with the user's unrelated
container stack running (GAPS G-008), so **every figure is a lower bound** on the hardware.

### Compute

| Workload | Quietest observed | Most contended observed | Best thread count |
|---|---|---|---|
| 2048² fp32 matmul | **179.4 GFLOP/s** (95.8 ms) | 64.4 GFLOP/s (266.6 ms) | 6 |
| 512² fp32 matmul | 109.4 GFLOP/s | 12.9 GFLOP/s | 4–6, unstable |
| 128² fp32 matmul | 55.5 GFLOP/s (0.076 ms) | 2.8 GFLOP/s | 1–2 |

The spread between columns is contention, not throttling. Treat ~180 GFLOP/s as the fp32 ceiling
and ~65–80 GFLOP/s as what is actually available while your dev stack is up.

### Training step, hand-written Transformer encoder

| Config | Layers / d_model / batch / seq | Params | ms/step (quiet → contended) | tok/s (quiet → contended) | Peak RSS |
|---|---|---|---|---|---|
| **tiny** | 2 / 128 / 32 / 32 | 0.53 M | **100.5 → 278.9** | **10,194 → 3,672** | 391 MiB |
| small | 4 / 256 / 32 / 64 | 4.20 M | — → 1,445 | — → 1,417 | 844 MiB |
| paper-base width | 6 / 512 / 8 / 64 | 27.29 M | — → 1,667 | — → 307 | 1,084 MiB |

**Thread count is workload-dependent and this reproduced in every run:** the tiny config is
fastest at **2 threads** and roughly 3–6x slower at 6; large matmuls scale to 6. Project 1's
training loop therefore pins 2 threads. A single global `OMP_NUM_THREADS` for the whole repo would
be wrong.

### Resources

| Resource | Measured | Note |
|---|---|---|
| Host RAM | 31.3 GiB | not the operative limit |
| **RAM visible inside the container** | **15.6 GiB** | WSL2 VM allocation — half the host |
| **RAM actually available** | **~9.5 GiB** | remainder held by 37 unrelated containers |
| Disk free (C:) | 31.4 GB | binding constraint (G-002) |
| CUDA | absent | integrated AMD Radeon only (G-001) |

The container-visible 15.6 GiB, not the host's 31.3 GiB, is the number that governs batch sizes.

---

## 2. Per-project feasibility

Compute column reads *"smallest meaningful experiment"* → *"larger target"*. All durations are
**ESTIMATE**, computed as `steps × ms_per_step` using the measured contended throughput above,
which is the pessimistic end. Quiet-machine runtimes would be ~2.5x faster.

| # | Project | Training config | Dataset & licence | Compute (ESTIMATE) | Demo inference | Missing resource | Tier |
|---|---|---|---|---|---|---|---|
| **1** | Transformer | 2 layers, d_model 128, h 4, d_ff 512, batch 32, seq ≤ 32, 2 threads | **Synthetic date-normalisation** (`"March 3, 2019"` → `2019-03-03`) + copy/reverse sanity tasks. Generator is our own code, no third-party licence. Stretch: Tatoeba En–De subset (CC-BY 2.0 FR) | 8k steps ≈ **37 min**; 20k steps ≈ **93 min** | < 1 MB weights, hand-written JS forward pass, < 50 ms/token in browser | none — fully runnable | E + R |
| **2** | GPT | 4–6 layers, d_model 256, seq 128–256, batch 16 | TinyShakespeare (public domain) → then a small permissive corpus. ≤ 200 MB on disk | 20k steps ≈ **2–5 h** | ONNX Runtime Web, ≤ 25 MB weights, capped tokens + timeout | none for the small model | R |
| **2b** | GPT @ ~124 M params | GPT-2-small geometry | same | **Not attemptable here.** At 307 tok/s measured for a 27 M model, and 124 M being ~4.5x more compute, a Chinchilla-ish 2.5 B-token budget is ~10⁴ h. Needs ~10–20 A100-hours | — | **1 rented GPU, or a free T4 for a scaled-down variant** | documented estimate only |
| **3** | LoRA | rank 4–16 adapters on the P2 small model; base frozen | reuse P2 corpus + a small instruction set with stated licence | 3 arms (base / LoRA / full FT) × ~15 min ≈ **1 h** | side-by-side checkpoint comparison, merge-equivalence check | none — full FT genuinely runs at this size, so the comparison is real | E + R |
| **4** | DPO vs SFT | same SFT checkpoint both arms; β sweep {0.05, 0.1, 0.5} | small preference set, provenance documented, eval prompts disjoint | SFT ~30 min + 3 DPO arms ≈ **1.5 h** | response pairs + chosen/rejected log-prob inspector | none | E + R |
| **5** | Tiled attention | n/a (kernel work) | random tensors | reference + tiled PyTorch benchmarks ≈ **20 min** | tile-by-tile walkthrough (labelled visualisation, not a kernel) | **NVIDIA GPU for the Triton half** (D-004) | E; Triton unexecuted until GPU |
| **6** | Quantization | PTQ on the P2 model; calibration set disjoint from eval | reuse P2 corpus; calibration split documented | INT8 baseline ~10 min; GPTQ/AWQ ~30–60 min | fp32 vs quantized outputs, weight-distribution viewer | none for simulated + packed storage; **low-bit *compute* speedup unlikely on this CPU** and will be reported as measured | E + R |
| **7** | Mini CLIP | ViT-tiny-ish + small text encoder, from scratch, batch 64–128 | **licence review pending (D-007b)** — candidates: Flickr8k, or captions over a permissive image set. ≤ 1.5 GB | 10–20 epochs ≈ **3–6 h** | ONNX Web, precomputed image embeddings + live text encoding | none, but dataset decision must precede download | E + R |
| **8** | DDPM | U-Net ~1–3 M params, 28–32 px, T=200–400 | MNIST/Fashion-MNIST (permissive) or CIFAR-10 subset | 15–30 min/epoch → **3–8 h** for a usable sampler | 200 reverse steps in-browser may exceed 10 s → likely labelled prerecorded grids + a short live option (D-007d) | none; FID omitted unless sample count makes it meaningful | E + R |

**Total for the seven locally-runnable projects: roughly 12–25 hours of CPU time**, dominated by
7 and 8. That fits, and it fits without paid infrastructure.

---

## 3. The smallest meaningful experiment, per project

Each is a single behaviour that would *fail loudly* if the implementation were wrong, chosen to run
in minutes rather than hours:

1. **P1** — overfit one batch of 8 date pairs to near-zero loss, then greedy-decode them exactly. Proves forward + loss + backward + masking are all correctly wired.
2. **P2** — same overfit test, then confirm a resumed checkpoint reproduces the pre-save loss bit-for-bit.
3. **P3** — merge the adapter into the base weights and assert `merged(x) == base_plus_adapter(x)` within tolerance. Proves the low-rank algebra, no training required.
4. **P4** — assert the DPO loss reduces to its known closed form when policy = reference (gradient should be zero at initialisation for symmetric pairs).
5. **P5** — tiled attention with online softmax must match reference attention to fp32 tolerance at several tile sizes, including tiles that do not divide the sequence length.
6. **P6** — round-trip quantize/dequantize must be exact for values on the quantization grid, and error must be bounded by half a step elsewhere.
7. **P7** — with a batch of N pairs, the contrastive loss at initialisation must be ≈ `log N`, and the logit matrix diagonal must be the matched pairs.
8. **P8** — the closed-form `q(x_t|x_0)` must match iterated single-step noising in distribution, and `x_T` must be ≈ standard normal.

---

## 4. What this table does not claim

- No tier-P reproduction anywhere (SCOPE §0). No BLEU on WMT14, no GPT-2 benchmark numbers, no CLIP zero-shot ImageNet, no published FID.
- All durations are single-configuration extrapolations from a contended machine, not measured end-to-end training runs. They will be replaced by measured wall-clock in `records/EXPERIMENTS.md` as each run completes.
- The P2b row is arithmetic, not a plan. It exists so the 124 M target has an honest cost attached instead of being quietly dropped.
