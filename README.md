# ML from first principles

Eight foundational ML papers, implemented from the equations, verified against independent
oracles, and documented with reproducible evidence.

**Portfolio site:** https://zexoraai.github.io/ml-from-first-principles/

---

## What this repository is, stated honestly

Every artefact here is labelled with one of three fidelity tiers. The vocabulary exists to prevent
the most common portfolio overclaim — implying a paper was reproduced when a mechanism was merely
implemented.

| Tier | Meaning | The claim it licenses |
|---|---|---|
| **E** | Educational implementation of a mechanism, verified for correctness | "I implemented and verified mechanism X" |
| **R** | Reduced-scale reproduction: the full method runs end to end on real data, far below paper scale | "At a fraction of the scale, the method behaves qualitatively as reported" |
| **P** | Reproduction of published results: same data, same scale, within published error bars | "I reproduced the published result" |

**No project in this repository claims tier P.** The development machine is a 6-core mobile CPU with
no CUDA device; tier P is unreachable here for all eight papers, and where it would be reachable
with rented compute, the exact missing resource and a *measured* cost estimate are recorded instead
of hand-waving. See [`records/SCOPE.md`](records/SCOPE.md).

## Projects

| # | Project | Primary source | Status |
|---|---|---|---|
| 1 | Encoder–decoder Transformer | [arXiv:1706.03762v7](https://arxiv.org/abs/1706.03762) | mechanisms implemented + verified (79 tests) |
| 2 | Decoder-only GPT | GPT-2 / nanoGPT / llm.c | not started |
| 3 | LoRA | [arXiv:2106.09685](https://arxiv.org/abs/2106.09685) | not started |
| 4 | DPO vs SFT | [arXiv:2305.18290](https://arxiv.org/abs/2305.18290) | not started |
| 5 | Tiled attention → Triton | [arXiv:2205.14135](https://arxiv.org/abs/2205.14135), [arXiv:1805.02867](https://arxiv.org/abs/1805.02867) | not started |
| 6 | Post-training quantization | GPTQ / AWQ (method TBD, D-007a) | not started |
| 7 | Miniature CLIP | [arXiv:2103.00020](https://arxiv.org/abs/2103.00020) | not started |
| 8 | Unconditional DDPM | [arXiv:2006.11239](https://arxiv.org/abs/2006.11239) | not started |

Project 1 implements the **paper's** post-norm formulation, `LayerNorm(x + Sublayer(x))` (§3.1),
not the pre-norm variant the authors' own `tensor2tensor` release shipped. Both are available
behind a flag so the discrepancy is a runnable experiment rather than a footnote.

## Running it

All Python executes in a pinned Linux container. This is not a preference — the development host
has Windows Smart App Control enforcing, which blocks PyTorch's unsigned DLLs from loading. The
container sidesteps that **without weakening the host's security configuration**. Full reasoning
and evidence: [`records/DECISIONS.md`](records/DECISIONS.md) D-008.

```bash
docker build -t ml-labs:cpu .          # once, ~2 GB

.\run.cmd python -m pytest tests -q    # correctness suite
.\run.cmd python scripts/bench_cpu.py  # hardware benchmark
```

`run.cmd` is a thin wrapper that bind-mounts this repo at `/work` inside the container. It is a
`.cmd` file rather than PowerShell because the host's execution policy blocks `.ps1`.

## Layout

```
labs/            implementations, one package per project
  common/        utilities, promoted only once a second caller exists
tests/           correctness suites mirroring labs/
scripts/         entry points: benchmarks, training, diagnostics
docs/            the deployed GitHub Pages site
evidence/        run artefacts, raw transcripts, benchmark JSON
records/         SCOPE, DECISIONS, PROGRESS, EXPERIMENTS, GAPS, NEXT_ACTION
env/             measured feasibility analysis
mastery/         self-assessment packs per project
```

## Reading the records

These are maintained as working documents, not marketing:

- [`records/SCOPE.md`](records/SCOPE.md) — what each project is and explicitly is not
- [`records/DECISIONS.md`](records/DECISIONS.md) — append-only ADR, including rejected options
- [`records/EXPERIMENTS.md`](records/EXPERIMENTS.md) — every reported number, with its full provenance. Includes `env-bench-01`, a benchmark that was **wrong by 300x**, retained on purpose
- [`records/GAPS.md`](records/GAPS.md) — known limitations and things deliberately not claimed
- [`env/feasibility.md`](env/feasibility.md) — measured capability, then labelled extrapolations

## Environment as measured

| | |
|---|---|
| CPU | AMD Ryzen 5 PRO 5650U, 6C/12T, mobile |
| RAM | 31.3 GiB host / **15.6 GiB visible in container** |
| GPU | integrated AMD Radeon — **no CUDA** |
| Disk free | 31.4 GB |
| Peak fp32 matmul | 179 GFLOP/s quiet, ~65–80 GFLOP/s under load |

Timings in this repository are **lower bounds**: the machine runs an unrelated 37-container
development stack, which varies throughput by up to 14x. Every benchmark record carries a load
snapshot so outliers are annotated rather than mysterious.

## Licence

Code: MIT. Datasets retain their own licences, documented per project before download.
