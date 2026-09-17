"""Train Project 2's GPT on TinyShakespeare.

    .\\run.cmd python scripts/train_p2.py --run-name p2-shake-v1 --steps 6000
    .\\run.cmd python scripts/train_p2.py --resume runs/p2-shake-v1/checkpoint_last.pt

Produces a run directory with everything records/EXPERIMENTS.md requires: config, seeds, data
statistics, dependency snapshot, hardware, per-step history, checkpoints with sha256, and the exact
reproduction command.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import random
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from labs.common.sysload import describe_load, load_snapshot  # noqa: E402
from labs.p2_gpt import (  # noqa: E402
    GPT,
    BatchSampler,
    GPTConfig,
    estimate_loss,
    generate,
    prepare,
)


# ---------------------------------------------------------------------------------------------
# learning-rate schedule
# ---------------------------------------------------------------------------------------------

def lr_at(step: int, *, base_lr: float, warmup: int, total: int, min_lr: float) -> float:
    """Linear warmup then cosine decay to `min_lr` — GPT-2's schedule, as used by nanoGPT.

    Different from Project 1's Noam schedule on purpose, and the difference is instructive.

    Noam (P1) decays as `step^-0.5` forever, with no notion of a training end. It suits post-norm,
    where warmup is *required* for stability, and it needs no knowledge of the total step count.

    Cosine (here) is defined relative to a known horizon: it reaches `min_lr` exactly at `total`.
    That produces a smooth, complete anneal, which reliably buys a lower final loss — but it means
    stopping early leaves the model at a mid-decay learning rate, and extending training past `total`
    is meaningless. The schedule and the step budget are one decision, not two.
    """
    if step < warmup:
        # Start at step 1's value rather than 0, so the very first update is not a no-op.
        return base_lr * (step + 1) / max(warmup, 1)
    if step >= total:
        return min_lr
    progress = (step - warmup) / max(total - warmup, 1)
    return min_lr + 0.5 * (base_lr - min_lr) * (1.0 + math.cos(math.pi * progress))


def configure_optimizer(model: GPT, *, weight_decay: float, lr: float, betas: tuple[float, float]):
    """AdamW with weight decay on matmul weights only.

    Parameters that participate in matrix multiplications (Linear and Embedding weights) are
    regularised. Biases and LayerNorm gains/shifts are **not**.

    The reason is that weight decay is a prior towards zero on the *scale of a linear map*. A
    LayerNorm gain of zero deletes its channel entirely, and a bias of zero is not a smaller model,
    just a shifted one — so decaying them is regularising a quantity where "smaller" does not mean
    "simpler". This split is what nanoGPT does and it is worth knowing that a naive
    `AdamW(model.parameters(), weight_decay=0.1)` silently decays all of them.
    """
    decay, no_decay = [], []
    seen: set[int] = set()
    for module in model.modules():
        for name, param in module.named_parameters(recurse=False):
            if id(param) in seen:
                continue
            seen.add(id(param))
            if param.dim() >= 2:
                decay.append(param)
            else:
                no_decay.append(param)

    groups = [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    return torch.optim.AdamW(groups, lr=lr, betas=betas), len(decay), len(no_decay)


# ---------------------------------------------------------------------------------------------
# plumbing
# ---------------------------------------------------------------------------------------------

def git_commit() -> str:
    try:
        sha = subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL).decode().strip()
        dirty = subprocess.call(["git", "diff", "--quiet"], stderr=subprocess.DEVNULL) != 0
        return ("DIRTY:" if dirty else "") + sha
    except Exception:  # noqa: BLE001
        return "UNKNOWN"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def pip_freeze() -> list[str]:
    try:
        return subprocess.check_output(
            [sys.executable, "-m", "pip", "freeze"], stderr=subprocess.DEVNULL
        ).decode().strip().splitlines()
    except Exception:  # noqa: BLE001
        return ["UNKNOWN"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-name", default=None)
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--batch-size", type=int, default=24)
    ap.add_argument("--grad-accum", type=int, default=1)
    ap.add_argument("--block-size", type=int, default=192)
    ap.add_argument("--n-layer", type=int, default=6)
    ap.add_argument("--n-head", type=int, default=6)
    ap.add_argument("--d-model", type=int, default=192)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--vocab-size", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--min-lr", type=float, default=1e-4)
    ap.add_argument("--warmup", type=int, default=200)
    ap.add_argument("--weight-decay", type=float, default=0.1)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--eval-every", type=int, default=250)
    ap.add_argument("--eval-iters", type=int, default=40)
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--sample-every", type=int, default=1000)
    ap.add_argument("--resume", default=None)
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    run_name = args.run_name or f"p2-shake-{datetime.now(timezone.utc):%Y%m%d-%H%M%S}"
    run_dir = Path("runs") / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    # ---- data --------------------------------------------------------------------------------
    bundle = prepare(vocab_size=args.vocab_size, verbose=True)
    tok, stats = bundle["tokenizer"], bundle["stats"]
    sampler = BatchSampler(
        bundle["train_ids"], bundle["val_ids"],
        block_size=args.block_size, batch_size=args.batch_size, seed=args.seed,
    )

    # ---- model -------------------------------------------------------------------------------
    cfg = GPTConfig(
        vocab_size=tok.vocab_size, block_size=args.block_size, n_layer=args.n_layer,
        n_head=args.n_head, d_model=args.d_model, dropout=args.dropout,
    )
    model = GPT(cfg)
    optimizer, n_decay, n_nodecay = configure_optimizer(
        model, weight_decay=args.weight_decay, lr=args.lr, betas=(0.9, 0.95)
    )

    start_step = 0
    history: list[dict] = []
    best_val = float("inf")
    if args.resume:
        ck = torch.load(args.resume, map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model"])
        optimizer.load_state_dict(ck["optimizer"])
        sampler.load_state_dict(ck["sampler"])
        random.setstate(ck["rng"]["python"])
        np.random.set_state(ck["rng"]["numpy"])
        torch.set_rng_state(ck["rng"]["torch"])
        start_step = ck["step"]
        history = ck.get("history", [])
        best_val = ck.get("best_val", float("inf"))
        print(f"resumed from {args.resume} at step {start_step}")

    tokens_per_step = args.batch_size * args.block_size * args.grad_accum
    flops = model.estimate_flops_per_token()

    meta = {
        "run_id": run_name,
        "project": "p2_gpt",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "source_commit": git_commit(),
        "repro_cmd": ".\\run.cmd python " + " ".join(sys.argv),
        "seeds": {"python": args.seed, "numpy": args.seed, "torch": args.seed,
                  "batch_sampler": args.seed},
        "args": vars(args),
        "model_config": cfg.to_dict(),
        "n_parameters": model.num_parameters(),
        "n_parameters_non_embedding": model.num_parameters(non_embedding=True),
        "flop_estimate_per_token": flops,
        "optimizer": {"name": "AdamW", "betas": [0.9, 0.95], "weight_decay": args.weight_decay,
                      "decayed_tensors": n_decay, "undecayed_tensors": n_nodecay,
                      "schedule": "linear warmup then cosine decay to min_lr"},
        "tokens_per_step": tokens_per_step,
        "planned_token_budget": tokens_per_step * args.steps,
        "data": stats,
        "hardware": {"platform": platform.platform(), "host_cpu": "AMD Ryzen 5 PRO 5650U 6C/12T",
                     "torch_threads": torch.get_num_threads(),
                     "cuda_available": torch.cuda.is_available(),
                     "container": "built from ./Dockerfile (decision D-008)"},
        "deps": pip_freeze(),
        "load_at_start": load_snapshot(),
    }
    (run_dir / "meta.json").write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")
    (run_dir / "tokenizer.json").write_text(tok.to_web_json(), encoding="utf-8")

    print("=" * 88)
    print(f"run {run_name} | {model.num_parameters():,} params "
          f"({model.num_parameters(non_embedding=True):,} non-embedding) | vocab {tok.vocab_size}")
    print(f"block {args.block_size} | batch {args.batch_size} x accum {args.grad_accum} "
          f"= {tokens_per_step:,} tokens/step | budget {tokens_per_step * args.steps:,} tokens")
    print(f"data: {stats['train_tokens']:,} train / {stats['val_tokens']:,} val tokens "
          f"({stats['compression_chars_per_token']:.2f} chars/token)")
    print(f"attention share of FLOPs at this context: {flops['attention_share'] * 100:.1f}%")
    print(f"threads={torch.get_num_threads()} | {describe_load(meta['load_at_start'])}")
    print("=" * 88)

    def save(tag: str, step: int) -> Path:
        path = run_dir / f"checkpoint_{tag}.pt"
        torch.save({
            "step": step, "model": model.state_dict(), "optimizer": optimizer.state_dict(),
            "sampler": sampler.state_dict(),
            "rng": {"python": random.getstate(), "numpy": np.random.get_state(),
                    "torch": torch.get_rng_state()},
            "config": cfg.to_dict(), "history": history, "best_val": best_val,
            "run_id": run_name, "vocab_size": tok.vocab_size,
        }, path)
        return path

    # ---- loop --------------------------------------------------------------------------------
    step = start_step
    t_start = time.perf_counter()
    tokens_seen = step * tokens_per_step
    model.train()

    while step < args.steps:
        lr = lr_at(step, base_lr=args.lr, warmup=args.warmup, total=args.steps, min_lr=args.min_lr)
        for group in optimizer.param_groups:
            group["lr"] = lr

        optimizer.zero_grad(set_to_none=True)
        total_loss = 0.0
        for _ in range(args.grad_accum):
            x, y = sampler.get_batch("train")
            _, loss = model(x, y)
            # Divide before backward so accumulated gradients average rather than sum -- otherwise
            # the effective learning rate scales with grad_accum and the setting stops being neutral.
            (loss / args.grad_accum).backward()
            total_loss += loss.item() / args.grad_accum

        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()
        step += 1
        tokens_seen += tokens_per_step

        if step % args.log_every == 0:
            elapsed = time.perf_counter() - t_start
            done = tokens_seen - start_step * tokens_per_step
            rec = {"step": step, "train_loss": total_loss, "lr": lr,
                   "grad_norm": float(grad_norm), "tokens_seen": tokens_seen,
                   "elapsed_s": elapsed, "tokens_per_s": done / max(elapsed, 1e-9)}
            history.append(rec)
            print(f"step {step:>6} loss {total_loss:.4f} lr {lr:.2e} "
                  f"gnorm {rec['grad_norm']:6.2f} {rec['tokens_per_s']:7.0f} tok/s")

        if step % args.eval_every == 0 or step == args.steps:
            losses = estimate_loss(model, sampler.get_batch, eval_iters=args.eval_iters)
            rec = {"step": step, "eval_train_loss": losses["train"], "eval_val_loss": losses["val"],
                   "val_perplexity": math.exp(min(losses["val"], 20))}
            history.append(rec)
            flag = ""
            if losses["val"] < best_val:
                best_val = losses["val"]
                save("best", step)
                flag = "  <- best"
            save("last", step)
            (run_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
            print(f"  [eval] step {step} train {losses['train']:.4f} val {losses['val']:.4f} "
                  f"ppl {rec['val_perplexity']:.2f}{flag}")

        if args.sample_every and step % args.sample_every == 0:
            model.eval()
            prompt = torch.tensor([tok.encode("\n")], dtype=torch.long)
            out = generate(model, prompt, 120, temperature=0.8, top_k=40,
                           generator=torch.Generator().manual_seed(0))
            print("  sample: " + tok.decode(out[0].tolist()).replace("\n", " | ")[:220])
            model.train()

    duration = time.perf_counter() - t_start

    # ---- final ------------------------------------------------------------------------------
    best_path = run_dir / "checkpoint_best.pt"
    if best_path.exists():
        model.load_state_dict(torch.load(best_path, map_location="cpu", weights_only=False)["model"])
    final = estimate_loss(model, sampler.get_batch, eval_iters=200)

    model.eval()
    samples = []
    for temp, k in ((0.5, 40), (0.8, 40), (1.0, 200), (1.2, 0)):
        prompt = torch.tensor([tok.encode("\n")], dtype=torch.long)
        out = generate(model, prompt, 240, temperature=temp, top_k=k or None,
                       generator=torch.Generator().manual_seed(1234))
        samples.append({"temperature": temp, "top_k": k or None,
                        "text": tok.decode(out[0].tolist())})

    result = {
        **meta,
        "duration_s": duration,
        "steps_completed": step,
        "tokens_seen": tokens_seen,
        "final_metrics": {
            "train_loss": final["train"], "val_loss": final["val"],
            "val_perplexity": math.exp(min(final["val"], 20)),
            "eval_iters": 200,
        },
        "best_val_loss": best_val,
        "history": history,
        "samples": samples,
        "load_at_end": load_snapshot(),
        "checkpoints": {
            name: {"path": str(run_dir / f"checkpoint_{name}.pt"),
                   "sha256": sha256_file(run_dir / f"checkpoint_{name}.pt")}
            for name in ("best", "last") if (run_dir / f"checkpoint_{name}.pt").exists()
        },
        "metric_definitions": {
            "val_loss": "mean cross-entropy per token on held-out contiguous text, nats, "
                        "averaged over 200 random windows. No label smoothing (GPT-2 uses none).",
            "val_perplexity": "exp(val_loss). Per-BPE-token, so NOT comparable to per-character or "
                              "per-word perplexities, nor to models with a different vocabulary.",
        },
        "limitations": [
            "Tier R. No GPT-2 published benchmark is reproduced or claimed.",
            "Perplexity is per-BPE-token on our own 1024-token vocabulary trained on this corpus; "
            "it is not comparable across tokenizers.",
            "Single seed unless a companion run says otherwise.",
            "TinyShakespeare is ~1.1 MB; this model is far below the scale where LM ability emerges.",
            "Throughput is a lower bound: the machine shares an unrelated container stack (G-008).",
            "No KV cache, so generation is quadratic in length. No efficiency claim is made.",
        ],
    }
    (run_dir / "result.json").write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")

    print()
    print("=" * 88)
    print(f"final (200 eval iters): train {final['train']:.4f}  val {final['val']:.4f}  "
          f"ppl {math.exp(min(final['val'], 20)):.2f}")
    print(f"duration {duration / 60:.1f} min | {tokens_seen:,} tokens | "
          f"{(tokens_seen - start_step * tokens_per_step) / duration:.0f} tok/s")
    print("=" * 88)
    print("sample @ T=0.8, top_k=40:")
    print(samples[1]["text"][:600])
    print(f"\nwritten to {run_dir}")


if __name__ == "__main__":
    main()
