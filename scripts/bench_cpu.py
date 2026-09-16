"""Measure this machine, then use the measurement to estimate training cost.

The brief requires a real benchmark *before* any compute estimate. Estimating from published
TFLOPS numbers is how people end up claiming a 12-hour run that would actually take three weeks.

WHAT IS MEASURED
----------------
1. Dense fp32 matmul throughput across sizes and thread counts -- the ceiling everything else
   lives under.
2. A full training step (forward + loss + backward + optimizer) of the hand-written Transformer
   encoder stack, swept over thread counts, at several model widths.
3. Peak resident memory for those steps.

METHOD, AND A BUG THIS SCRIPT USED TO HAVE
------------------------------------------
Timing uses `labs.common.timing.timed`, which adapts an inner repeat count so each timed region
lasts ~0.3 s. The first version of this script timed a single call per sample and reported numbers
wrong by up to 300x -- see the docstring of `labs/common/timing.py` and the paired records
`env-bench-01` (wrong) and `env-bench-02` (corrected) in records/EXPERIMENTS.md. Both are kept.
Deleting the wrong run would hide the most useful lesson in the whole exercise.

Every reported figure is a measurement. The extrapolations that use them live in
`env/feasibility.md`, labelled ESTIMATE with their assumptions written out.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import resource
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from labs.common.sysload import describe_load, load_snapshot  # noqa: E402
from labs.common.timing import timed  # noqa: E402
from labs.p1_transformer import (  # noqa: E402
    LayerNorm,
    MultiHeadAttention,
    PositionwiseFeedForward,
    SinusoidalPositionalEncoding,
    SublayerConnection,
)

THREAD_OPTIONS = (1, 2, 4, 6)


def peak_rss_mb() -> float:
    """Peak resident set size in MiB. `ru_maxrss` is reported in KiB on Linux."""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


class EncoderStack(torch.nn.Module):
    """A minimal encoder built only from this project's hand-written parts.

    Used purely as a realistic timing workload: it exercises the same matmuls, softmaxes and
    normalisations a real training step would, so the seconds/step figure transfers to Project 1's
    actual training loop. It is not the deliverable model.
    """

    def __init__(self, n_layers: int, d_model: int, n_heads: int, d_ff: int, vocab: int, max_len: int):
        super().__init__()
        self.embed = torch.nn.Embedding(vocab, d_model)
        self.pos = SinusoidalPositionalEncoding(d_model, max_len=max_len, dropout=0.1)
        self.attn = torch.nn.ModuleList(
            [MultiHeadAttention(d_model, n_heads, dropout=0.1) for _ in range(n_layers)]
        )
        self.ffn = torch.nn.ModuleList(
            [PositionwiseFeedForward(d_model, d_ff, dropout=0.1) for _ in range(n_layers)]
        )
        self.conn = torch.nn.ModuleList(
            [SublayerConnection(d_model, dropout=0.1) for _ in range(2 * n_layers)]
        )
        self.final_norm = LayerNorm(d_model)
        self.head = torch.nn.Linear(d_model, vocab, bias=False)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        x = self.pos(self.embed(ids))
        for i, (attn, ffn) in enumerate(zip(self.attn, self.ffn)):
            x = self.conn[2 * i](x, lambda t: attn(t, t, t)[0])
            x = self.conn[2 * i + 1](x, ffn)
        return self.head(self.final_norm(x))


def bench_matmul(sizes: list[int], threads: tuple[int, ...]) -> dict:
    """Square matmul throughput. An (n,n)x(n,n) product is 2*n^3 FLOPs."""
    results: dict[str, dict[str, dict]] = {}
    print("  size   " + "".join(f"{t:>5}thr ms " for t in threads)
          + "|" + "".join(f"{t:>5}thr GF " for t in threads))
    for n in sizes:
        a = torch.randn(n, n)
        b = torch.randn(n, n)
        results[str(n)] = {}
        ms_cells, gf_cells = "", ""
        for t in threads:
            torch.set_num_threads(t)
            r = timed(lambda: a @ b, warmup=3)
            gflops = 2.0 * n ** 3 / r["per_call_s"] / 1e9
            results[str(n)][str(t)] = {**r, "gflops": gflops}
            ms_cells += f"{r['per_call_s'] * 1e3:10.3f} "
            gf_cells += f"{gflops:10.1f} "
        print(f"  {n:<6} {ms_cells}|{gf_cells}")
    return results


def bench_train_step(cfg: dict, threads: tuple[int, ...]) -> dict:
    """One training-step configuration, swept over thread count."""
    per_thread: dict[str, dict] = {}
    n_params = 0
    for t in threads:
        torch.set_num_threads(t)
        torch.manual_seed(0)
        model = EncoderStack(
            cfg["n_layers"], cfg["d_model"], cfg["n_heads"], cfg["d_ff"], cfg["vocab"], cfg["seq_len"]
        )
        opt = torch.optim.Adam(model.parameters(), lr=1e-4, betas=(0.9, 0.98), eps=1e-9)
        ids = torch.randint(0, cfg["vocab"], (cfg["batch"], cfg["seq_len"]))
        targets = torch.randint(0, cfg["vocab"], (cfg["batch"], cfg["seq_len"]))
        loss_fn = torch.nn.CrossEntropyLoss()
        n_params = sum(p.numel() for p in model.parameters())

        def step() -> None:
            opt.zero_grad(set_to_none=True)
            logits = model(ids)
            loss = loss_fn(logits.reshape(-1, cfg["vocab"]), targets.reshape(-1))
            loss.backward()
            opt.step()

        r = timed(step, warmup=2, target_seconds=1.0, samples=3)
        tokens = cfg["batch"] * cfg["seq_len"]
        per_thread[str(t)] = {
            **r,
            "tokens_per_s": tokens / r["per_call_s"],
            "peak_rss_mb": peak_rss_mb(),
        }
        print(f"    threads={t}  {r['per_call_s'] * 1e3:9.2f} ms/step  "
              f"{tokens / r['per_call_s']:9.0f} tok/s  peak_rss={peak_rss_mb():6.0f} MiB")

    best_t = min(per_thread, key=lambda k: per_thread[k]["per_call_s"])
    return {
        **cfg,
        "n_params": n_params,
        "tokens_per_step": cfg["batch"] * cfg["seq_len"],
        "by_threads": per_thread,
        "best_threads": int(best_t),
        "best_ms_per_step": per_thread[best_t]["per_call_s"] * 1e3,
        "best_tokens_per_s": per_thread[best_t]["tokens_per_s"],
    }


def git_commit() -> str:
    try:
        sha = subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
        dirty = subprocess.call(["git", "diff", "--quiet"], stderr=subprocess.DEVNULL) != 0
        return ("DIRTY:" if dirty else "") + sha.decode().strip()
    except Exception:  # noqa: BLE001
        return "UNKNOWN (not a git repository yet)"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="evidence/env")
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()

    env = {
        "run_id": "env-bench-02",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit(),
        "torch": torch.__version__,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
        "host_cpu": "AMD Ryzen 5 PRO 5650U (6 cores / 12 threads) -- read from the Windows host",
        "os_cpu_count": os.cpu_count(),
        "sched_affinity": len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
        "cgroup_cpu_max": Path("/sys/fs/cgroup/cpu.max").read_text().strip()
        if Path("/sys/fs/cgroup/cpu.max").exists() else "unavailable",
        "cuda_available": torch.cuda.is_available(),
        "timing_method": "labs.common.timing.timed -- adaptive inner repeats, ~0.3 s per timed "
                         "block, median of samples, min/max reported to expose throttling",
        "note": "measured inside the Linux container defined by ./Dockerfile (decision D-008). "
                "This is containerised Linux CPU performance under WSL2, not native Windows.",
        "supersedes": "env-bench-01, which used single-call timing and was wrong by up to 300x",
    }

    print("=" * 88)
    print("ENVIRONMENT")
    print("=" * 88)
    for k, v in env.items():
        print(f"  {k:18} {v}")

    load_start = load_snapshot()
    print()
    print("=" * 88)
    print("SYSTEM CONTENTION AT START")
    print("=" * 88)
    print(f"  {describe_load(load_start)}")
    if load_start["contended"]:
        print("  WARNING: the machine is busy with unrelated work. Every timing below is a LOWER")
        print("  BOUND on this hardware's capability, not a measurement of it. See GAPS G-008.")

    print()
    print("=" * 88)
    print("1. DENSE MATMUL THROUGHPUT (fp32), amortised timing, thread sweep")
    print("=" * 88)
    sizes = [128, 512, 1024] if args.quick else [128, 256, 512, 1024, 2048]
    matmul = bench_matmul(sizes, THREAD_OPTIONS)
    peak_gflops = max(
        cell["gflops"] for size in matmul.values() for cell in size.values()
    )

    print()
    print("=" * 88)
    print("2. FULL TRAINING STEP, hand-written Transformer encoder, thread sweep")
    print("=" * 88)
    if args.quick:
        cfgs = [dict(label="tiny", n_layers=2, d_model=128, n_heads=4, d_ff=512,
                     vocab=512, batch=16, seq_len=32)]
    else:
        cfgs = [
            dict(label="tiny (P1 target)", n_layers=2, d_model=128, n_heads=4, d_ff=512,
                 vocab=512, batch=32, seq_len=32),
            dict(label="small", n_layers=4, d_model=256, n_heads=8, d_ff=1024,
                 vocab=2048, batch=32, seq_len=64),
            dict(label="paper-base width", n_layers=6, d_model=512, n_heads=8, d_ff=2048,
                 vocab=8192, batch=8, seq_len=64),
        ]
    steps = []
    for cfg in cfgs:
        print(f"  {cfg['label']}  (n_layers={cfg['n_layers']}, d_model={cfg['d_model']}, "
              f"batch={cfg['batch']}, seq_len={cfg['seq_len']})")
        steps.append(bench_train_step(cfg, THREAD_OPTIONS))

    load_end = load_snapshot()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "env": env,
        "load_at_start": load_start,
        "load_at_end": load_end,
        "matmul_fp32": matmul,
        "train_steps": steps,
        "peak_matmul_gflops": peak_gflops,
    }
    out_path = out_dir / "cpu_benchmark.json"
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print()
    print("=" * 88)
    print("SUMMARY (measurements only -- extrapolations live in env/feasibility.md)")
    print("=" * 88)
    print(f"  peak fp32 matmul            {peak_gflops:.1f} GFLOP/s")
    for s in steps:
        print(f"  {s['label']:<20} {s['n_params'] / 1e6:6.2f}M params  "
              f"{s['best_ms_per_step']:8.2f} ms/step  {s['best_tokens_per_s']:9.0f} tok/s  "
              f"(best at {s['best_threads']} threads)")
    print()
    print(f"  load at start: {describe_load(load_start)}")
    print(f"  load at end:   {describe_load(load_end)}")
    print(f"\n  written to {out_path}")


if __name__ == "__main__":
    main()
