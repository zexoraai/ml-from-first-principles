# EXPERIMENTS — evidence register

Every experiment we *report anywhere* must appear here first, with all fields filled. A missing
field is written as `UNKNOWN`, never guessed. If a metric is not in this register, it must not
appear on a project page.

## Mandatory schema

```yaml
run_id:            # <project>-<short-name>-<YYYYMMDD>-<nn>
source_commit:     # git rev-parse HEAD at launch (dirty tree => "DIRTY:<sha>", must be fixed before reporting)
config:            # path to the exact config file written into the run dir
seeds:             # {python, numpy, torch} — all three
dataset:           # name, version/hash, licence
split:             # how train/val/test were produced; leakage argument
deps:              # requirements.txt hash / pip freeze path
hardware:          # cpu, threads, ram, device, os
duration:          # wall clock
budget:            # tokens or examples seen; steps; epochs
metrics:           # definition of each metric, in words, before its value
results:           # raw numbers, per-seed if repeated
checkpoint:        # path + sha256
repro_cmd:         # exact command line that reproduces it
n_runs:            # if 1 => "SINGLE RUN, no variance estimate"
limitations:       # what this run does not show
```

## Rules

1. **No number without a run_id.** Project pages cite run_ids.
2. **Repeat where feasible.** Cheap runs (< 10 min) get ≥ 3 seeds and we report mean ± range.
   Expensive runs get 1 seed and are labelled `SINGLE RUN, no variance estimate`.
3. **Metric definitions precede metric values.** "Accuracy" is not a definition; "fraction of
   held-out examples whose full decoded target sequence matches exactly, teacher forcing off" is.
4. **Calibration/training data never touches evaluation data.** The leakage argument is written
   out for each split, not assumed.
5. **Dirty trees may not be reported.** Commit first.

---

## Register

---

### `env-bench-01` — CPU benchmark, **INVALID METHOD, RETAINED DELIBERATELY**

```yaml
run_id:          env-bench-01
status:          INVALID -- superseded by env-bench-02. Do not cite these numbers.
source_commit:   UNKNOWN (pre-git-init)
config:          scripts/bench_cpu.py, first version (single call per timing sample)
seeds:           torch.manual_seed(0) per config; matmul inputs unseeded
dataset:         n/a -- synthetic tensors
split:           n/a
deps:            torch==2.14.0+cpu, numpy==2.5.3 (see requirements.txt)
hardware:        AMD Ryzen 5 PRO 5650U, 6C/12T; containerised Linux (WSL2); OMP_NUM_THREADS=6
duration:        ~4 min
budget:          n/a
metrics:         GFLOP/s = 2n^3 / median_seconds; ms/step = median wall clock of
                 (zero_grad, forward, loss, backward, opt.step)
results:
  matmul_128:    31.68 ms   ->   0.1 GFLOP/s
  matmul_2048:   252.18 ms  ->  68.1 GFLOP/s
  step_0.53M:    4979.7 ms  ->     206 tok/s
  step_4.20M:    3547.7 ms  ->     577 tok/s
  step_27.29M:   1147.2 ms  ->     446 tok/s
checkpoint:      n/a
repro_cmd:       n/a -- the faulty harness was replaced, not preserved as runnable code
n_runs:          SINGLE RUN, no variance estimate
limitations:     Everything. See below.
```

**Why it is wrong, and why it is still here.**

The internal contradiction is the giveaway: the **27.29 M-parameter step (1147 ms) came out faster
than the 0.53 M-parameter step (4980 ms)**. A model doing strictly more arithmetic cannot finish
sooner. Something outside the computation was being measured.

Cause: each timing sample wrapped exactly one call. That pays a large fixed entry cost per sample —
consistent with PyTorch's intra-op thread pool parking between calls and being rewoken, which is
expensive under a hypervisor. Warm-up and a median over repeats did not help, because every sample
paid the cost. The error reached **300x** on small operations.

This entry is retained because deleting it would erase the most transferable lesson in the whole
environment-setup phase, and because reading `68 GFLOP/s` as "this laptop is slow" would have
propagated a ~50x error into the compute estimate of all eight projects. Diagnosis:
`scripts/diag_threads.py`, `evidence/env/thread-diagnosis-transcript.txt`.

---

### `env-bench-02` — CPU benchmark, corrected method

```yaml
run_id:          env-bench-02
status:          VALID, but lower-bound only (machine contended -- see GAPS G-008)
source_commit:   UNKNOWN (pre-git-init; re-run after first commit to attach a SHA)
config:          scripts/bench_cpu.py, using labs.common.timing.timed
                 (adaptive inner repeats, ~0.3 s per timed block, median of 5, min/max retained)
seeds:           torch.manual_seed(0) per training config
dataset:         n/a -- synthetic tensors and random token ids
split:           n/a
deps:            torch==2.14.0+cpu, numpy==2.5.3, python 3.12.14; image built from ./Dockerfile
hardware:        AMD Ryzen 5 PRO 5650U 6C/12T; Linux 6.6.114.1 WSL2 container;
                 cgroup cpu.max = "max 100000" (no quota); container-visible RAM 15.6 GiB;
                 cuda_available = False
duration:        ~5 min
budget:          n/a
metrics:         GFLOP/s = 2n^3 / median_per_call_seconds
                 tok/s   = (batch * seq_len) / median_seconds_per_optimizer_step
                 peak RSS = ru_maxrss of the benchmark process, MiB
results:
  peak_matmul:            81.5 GFLOP/s at 2048^2, 6 threads (this run)
  peak_matmul_quietest:   179.4 GFLOP/s at 2048^2, 6 threads (diagnostic run, quieter machine)
  step_0.53M_best:        278.85 ms @ 2 threads ->  3,672 tok/s   (100.45 ms / 10,194 tok/s quiet)
  step_4.20M_best:       1445.18 ms @ 4 threads ->  1,417 tok/s
  step_27.29M_best:      1667.24 ms @ 6 threads ->    307 tok/s
  peak_rss:               391 / 844 / 1084 MiB respectively
  load_at_start:          loadavg 3.89, load/cpu 0.324, mem_available 10,106 MiB
  load_at_end:            loadavg 10.16, load/cpu 0.846, mem_available 9,468 MiB -> CONTENDED
checkpoint:      n/a
repro_cmd:       .\run.cmd python scripts/bench_cpu.py
n_runs:          3 independent executions of the corrected harness (2 full + 1 diagnostic sweep).
                 Spread across runs is ~2.8x on matmul and ~2.8x on step time, entirely attributable
                 to contention. Reported as a range, never as a point estimate.
limitations:
  - Machine shared with 37 unrelated containers throughout. All timings are LOWER BOUNDS.
  - Containerised Linux under WSL2, not native Windows. Not comparable to native benchmarks.
  - `timed()` measures steady-state throughput of repeated calls. It is the right model for a
    training loop and the WRONG model for cold-start single-shot inference latency.
  - Training-step configs use random token ids, so data-loading and tokenisation cost is excluded.
    Real training will be slower by the dataloader's contribution.
  - No fp16/bf16 measurement; CPU fp32 only.
```

**Robust conclusion across all three runs** (the only thread finding stable enough to act on):
small models train fastest at **2 threads** and are 3–6x slower at 6; large matmuls scale to 6.
Consequence: thread count is set per workload, not once globally.

---
