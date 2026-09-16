"""Diagnose why small PyTorch operations are implausibly slow in this container.

Observed anomaly (evidence/env/cpu_benchmark.json, first run):
  * 128x128 fp32 matmul  -> 31.7 ms  (0.1 GFLOP/s)  -- should be tens of microseconds
  * 2048x2048 matmul     -> 252 ms   (68.1 GFLOP/s) -- plausible for 6 Zen 3 mobile cores
  * 0.53 M-param training step -> 4980 ms/step
  * 27.29 M-param training step -> 1147 ms/step  <-- bigger model, FASTER step

The last line cannot be explained by compute. A larger model doing strictly more arithmetic in
less wall-clock time means the small configurations are paying a large fixed cost per operation
that the large one amortises. Two candidate causes, and this script separates them:

  H1  Thread-barrier overhead. PyTorch parallelises each op across `torch.get_num_threads()`
      threads. If the container's CPU quota is smaller than the thread count, every parallel
      region ends in a barrier where runnable threads wait on descheduled ones. Cost is per-op
      and roughly constant, so it dominates small ops and vanishes on large ones. Signature:
      **fewer threads is dramatically faster** for small ops.

  H2  Timer/measurement artefact. A per-call overhead outside the kernel (allocator, dispatch,
      or a coarse clock) inflates short measurements. Signature: **amortising many calls inside
      one timed region removes the floor**, and thread count barely matters.

Both are measured here. The fix follows the evidence rather than the guess.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import torch


def read_first_line(path: str) -> str:
    try:
        return Path(path).read_text().strip().splitlines()[0]
    except Exception as exc:  # noqa: BLE001
        return f"unavailable ({exc.__class__.__name__})"


def cpu_facts() -> dict:
    facts = {
        "os.cpu_count": os.cpu_count(),
        "len(os.sched_getaffinity)": len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
        "torch.get_num_threads": torch.get_num_threads(),
        "torch.get_num_interop_threads": torch.get_num_interop_threads(),
        "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS", "unset"),
        "cgroup_v2_cpu.max": read_first_line("/sys/fs/cgroup/cpu.max"),
        "cgroup_v1_quota_us": read_first_line("/sys/fs/cgroup/cpu/cpu.cfs_quota_us"),
        "cgroup_v1_period_us": read_first_line("/sys/fs/cgroup/cpu/cpu.cfs_period_us"),
    }
    try:
        facts["nproc"] = subprocess.check_output(["nproc"]).decode().strip()
    except Exception:  # noqa: BLE001
        facts["nproc"] = "unavailable"
    return facts


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from labs.common.timing import timed  # noqa: E402
# The adaptive timer was originally written inline here, then promoted to labs/common/timing.py
# once bench_cpu.py needed it too (decision D-005: promote on the second caller, not before).


def main() -> None:
    facts = cpu_facts()
    print("=" * 78)
    print("CPU / CGROUP FACTS")
    print("=" * 78)
    for k, v in facts.items():
        print(f"  {k:32} {v}")

    thread_options = [1, 2, 4, 6]
    sizes = [128, 512, 2048]

    print()
    print("=" * 78)
    print("MATMUL: per-call time vs thread count, with amortised timing")
    print("=" * 78)
    header = "  size   " + "".join(f"{t:>4}thr(ms) " for t in thread_options)
    print(header)

    matmul_results: dict[str, dict[int, dict]] = {}
    for n in sizes:
        a = torch.randn(n, n)
        b = torch.randn(n, n)
        row = f"  {n:<6} "
        matmul_results[str(n)] = {}
        for t in thread_options:
            torch.set_num_threads(t)
            r = timed(lambda: a @ b, warmup=3)
            matmul_results[str(n)][t] = {**r, "gflops": 2.0 * n ** 3 / r["per_call_s"] / 1e9}
            row += f"{r['per_call_s']*1e3:11.3f} "
        print(row)

    print()
    print("  GFLOP/s for the same measurements")
    print("  size   " + "".join(f"{t:>4}thr(GF) " for t in thread_options))
    for n in sizes:
        row = f"  {n:<6} "
        for t in thread_options:
            row += f"{matmul_results[str(n)][t]['gflops']:11.1f} "
        print(row)

    print()
    print("=" * 78)
    print("TINY TRAINING STEP vs thread count")
    print("=" * 78)
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from scripts.bench_cpu import EncoderStack  # noqa: PLC0415

    step_results: dict[int, dict] = {}
    for t in thread_options:
        torch.set_num_threads(t)
        torch.manual_seed(0)
        model = EncoderStack(2, 128, 4, 512, 512, 32)
        opt = torch.optim.Adam(model.parameters(), lr=1e-4)
        ids = torch.randint(0, 512, (32, 32))
        tgt = torch.randint(0, 512, (32, 32))
        loss_fn = torch.nn.CrossEntropyLoss()

        def step() -> None:
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(model(ids).reshape(-1, 512), tgt.reshape(-1))
            loss.backward()
            opt.step()

        r = timed(step, warmup=2, target_seconds=1.0)
        step_results[t] = {**r, "tokens_per_s": 32 * 32 / r["per_call_s"]}
        print(f"  threads={t}  {r['per_call_s']*1e3:9.2f} ms/step   "
              f"{32*32/r['per_call_s']:9.0f} tok/s   (inner={r['inner']})")

    out = Path("evidence/env/thread_diagnosis.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(
        {"facts": facts, "matmul": matmul_results, "train_step": step_results}, indent=2), encoding="utf-8")

    print()
    print("=" * 78)
    print("VERDICT")
    print("=" * 78)
    small_1 = matmul_results["128"][1]["per_call_s"]
    small_6 = matmul_results["128"][6]["per_call_s"]
    print(f"  128x128 matmul: 1 thread {small_1*1e6:9.1f} us   6 threads {small_6*1e6:9.1f} us"
          f"   ratio {small_6/small_1:6.2f}x")
    if small_1 * 1e6 < 500 and small_6 / small_1 > 2:
        print("  -> H1 CONFIRMED: thread-barrier overhead dominates small ops.")
    elif small_1 * 1e6 < 500:
        print("  -> H2 CONFIRMED: the earlier floor was a measurement artefact; amortised timing fixes it.")
    else:
        print("  -> Neither signature matched cleanly; inspect the JSON before concluding.")
    best = min(step_results.items(), key=lambda kv: kv[1]["per_call_s"])
    print(f"  fastest training step at threads={best[0]}: {best[1]['per_call_s']*1e3:.1f} ms")
    print(f"  written to {out}")


if __name__ == "__main__":
    main()
