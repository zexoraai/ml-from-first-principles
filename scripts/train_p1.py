"""Train Project 1's Transformer on the date-normalisation task.

Produces a run directory under `runs/` containing everything `records/EXPERIMENTS.md` requires:
config, seeds, split manifest, dependency snapshot, hardware, per-step metrics, checkpoints with
sha256, and the exact command to reproduce it.

    .\\run.cmd python scripts/train_p1.py --steps 4000 --run-name p1-date-v1
    .\\run.cmd python scripts/train_p1.py --resume runs/p1-date-v1/checkpoint_last.pt

CHECKPOINTING IS A CORRECTNESS FEATURE, NOT A CONVENIENCE
--------------------------------------------------------
A resumed run must continue the *same* trajectory, which means saving more than the weights:
optimizer moments, scheduler step, dataloader epoch, and the RNG states of python/numpy/torch.
Omit the RNG state and a resumed run silently sees a different shuffle and different dropout masks;
the loss curve then has a visible discontinuity at the resume point and the run is no longer one
experiment. `tests/test_p1_training.py` asserts that save-then-load reproduces the next step's loss
bit-for-bit.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import random
import subprocess
import sys
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from labs.common.sysload import describe_load, load_snapshot  # noqa: E402
from labs.p1_transformer import (  # noqa: E402
    LabelSmoothingLoss,
    Transformer,
    TransformerConfig,
    build_optimizer,
    greedy_decode,
)
from labs.p1_transformer.data import (  # noqa: E402
    CharTokenizer,
    DateDataset,
    DateTaskConfig,
    build_splits,
    collate,
    write_split_manifest,
)


# ---------------------------------------------------------------------------------------------
# reproducibility plumbing
# ---------------------------------------------------------------------------------------------

def set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def rng_state() -> dict:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }


def restore_rng(state: dict) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])


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
        out = subprocess.check_output([sys.executable, "-m", "pip", "freeze"], stderr=subprocess.DEVNULL)
        return out.decode().strip().splitlines()
    except Exception:  # noqa: BLE001
        return ["UNKNOWN"]


# ---------------------------------------------------------------------------------------------
# evaluation
# ---------------------------------------------------------------------------------------------

@torch.no_grad()
def evaluate(
    model: Transformer,
    loader: torch.utils.data.DataLoader,
    tok: CharTokenizer,
    *,
    max_batches: int | None = None,
    decode_limit: int = 12,
) -> dict:
    """Compute the metrics we report. Definitions first, values second.

    `val_loss`        Mean **unsmoothed** cross-entropy per non-padding target token, in nats.
                      Unsmoothed on purpose: the training objective uses label smoothing, which the
                      paper notes hurts perplexity by design (section 5.4). Reporting a smoothed
                      training loss next to an unsmoothed validation loss on the same axis would be
                      an apples-to-oranges comparison, so evaluation defines its own metric.

    `token_accuracy`  Fraction of non-padding target positions whose argmax matches the label, under
                      **teacher forcing** -- the model sees the correct prefix. This is the
                      optimistic metric and is reported as such.

    `exact_match`     Fraction of examples where **greedy decoding from BOS alone**, with no access
                      to the target, reproduces the full ISO string exactly. This is the honest
                      metric and the headline number. It is strictly harder than token accuracy
                      because a single early mistake derails the whole sequence, and because it
                      removes teacher forcing entirely.

    Reporting both is deliberate: a wide gap between them is the signature of a model that is fine
    at next-token prediction but compounds its own errors during free generation.
    """
    model.eval()
    loss_fn = LabelSmoothingLoss(len(tok), pad_id=tok.pad_id, smoothing=0.0)

    total_loss, total_tokens = 0.0, 0
    correct_tokens = 0
    exact, seen = 0, 0

    for i, batch in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        logits = model(batch["src"], batch["tgt_in"])
        labels = batch["labels"]
        non_pad = labels != tok.pad_id
        n = int(non_pad.sum())

        total_loss += loss_fn(logits, labels).item() * n
        total_tokens += n
        correct_tokens += int(((logits.argmax(-1) == labels) & non_pad).sum())

        decoded = greedy_decode(model, batch["src"], max_new_tokens=decode_limit)
        for row, label_row in zip(decoded, labels):
            got = tok.decode(row.tolist())
            want = tok.decode(label_row.tolist())
            exact += int(got == want)
            seen += 1

    return {
        "val_loss_nats_per_token": total_loss / max(total_tokens, 1),
        "token_accuracy_teacher_forced": correct_tokens / max(total_tokens, 1),
        "exact_match_free_running": exact / max(seen, 1),
        "n_examples": seen,
        "n_target_tokens": total_tokens,
    }


@torch.no_grad()
def sample_predictions(model: Transformer, dataset: DateDataset, tok: CharTokenizer, n: int = 12) -> list[dict]:
    """Concrete predictions, including failures, for the project page's failure-example section."""
    model.eval()
    out = []
    for idx in range(min(n, len(dataset))):
        src_text, tgt_text = dataset.pairs[idx]
        src = torch.tensor([tok.encode(src_text)], dtype=torch.long)
        got = tok.decode(greedy_decode(model, src, max_new_tokens=12)[0].tolist())
        out.append({"source": src_text, "target": tgt_text, "prediction": got, "correct": got == tgt_text})
    return out


# ---------------------------------------------------------------------------------------------
# training
# ---------------------------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-name", default=None)
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--d-model", type=int, default=128)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--d-ff", type=int, default=512)
    ap.add_argument("--enc-layers", type=int, default=2)
    ap.add_argument("--dec-layers", type=int, default=2)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--label-smoothing", type=float, default=0.1)
    ap.add_argument("--warmup", type=int, default=400)
    ap.add_argument("--lr-scale", type=float, default=1.0)
    ap.add_argument("--norm-style", choices=["post", "pre"], default="post")
    ap.add_argument("--seed", type=int, default=1706)
    ap.add_argument("--threads", type=int, default=2, help="2 measured fastest for this size")
    ap.add_argument("--eval-every", type=int, default=250)
    ap.add_argument("--eval-batches", type=int, default=8)
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--resume", default=None)
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    set_seeds(args.seed)

    run_name = args.run_name or f"p1-date-{datetime.now(timezone.utc):%Y%m%d-%H%M%S}"
    run_dir = Path("runs") / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    # ---- data -------------------------------------------------------------------------------
    task_cfg = DateTaskConfig(seed=args.seed)
    tok = CharTokenizer()
    splits = build_splits(task_cfg)
    manifest = write_split_manifest(splits, task_cfg, run_dir / "split_manifest.json")

    datasets = {k: DateDataset(v, tok, task_cfg) for k, v in splits.items()}
    loaders = {
        k: torch.utils.data.DataLoader(
            ds, batch_size=args.batch_size, shuffle=(k == "train"),
            collate_fn=lambda b: collate(b, tok.pad_id), num_workers=0, drop_last=(k == "train"),
        )
        for k, ds in datasets.items()
    }

    # ---- model ------------------------------------------------------------------------------
    cfg = TransformerConfig(
        vocab_size=len(tok), d_model=args.d_model, num_heads=args.heads, d_ff=args.d_ff,
        num_encoder_layers=args.enc_layers, num_decoder_layers=args.dec_layers,
        dropout=args.dropout, max_len=max(task_cfg.max_src_len, task_cfg.max_tgt_len) + 4,
        norm_style=args.norm_style, pad_id=tok.pad_id, bos_id=tok.bos_id, eos_id=tok.eos_id,
    )
    model = Transformer(cfg)
    optimizer, scheduler = build_optimizer(
        model, d_model=cfg.d_model, warmup_steps=args.warmup, scale=args.lr_scale
    )
    loss_fn = LabelSmoothingLoss(len(tok), pad_id=tok.pad_id, smoothing=args.label_smoothing)

    start_step = 0
    history: list[dict] = []
    if args.resume:
        ckpt = torch.load(args.resume, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        # `LRScheduler.load_state_dict` restores the scheduler's position but does NOT write the
        # learning rate back into optimizer.param_groups. Loading the optimizer state above happens
        # to restore it, because param_groups (lr included) are part of that state dict -- but that
        # makes correctness depend on the order of these two calls. Syncing explicitly removes the
        # dependency, so reordering them later cannot silently resume at the step-1 warmup rate.
        # See tests/test_p1_training.py::test_scheduler_state_survives_a_round_trip.
        for group, lr in zip(optimizer.param_groups, scheduler.get_last_lr()):
            group["lr"] = lr
        restore_rng(ckpt["rng"])
        start_step = ckpt["step"]
        history = ckpt.get("history", [])
        print(f"resumed from {args.resume} at step {start_step}, lr={optimizer.param_groups[0]['lr']:.3e}")

    meta = {
        "run_id": run_name,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "source_commit": git_commit(),
        "argv": sys.argv,
        "repro_cmd": ".\\run.cmd python " + " ".join(sys.argv[0:1] + sys.argv[1:]),
        "seeds": {"python": args.seed, "numpy": args.seed, "torch": args.seed},
        "model_config": cfg.to_dict(),
        "task_config": asdict(task_cfg),
        "args": vars(args),
        "n_parameters": model.num_parameters(),
        "vocab_size": len(tok),
        "hardware": {
            "platform": platform.platform(),
            "host_cpu": "AMD Ryzen 5 PRO 5650U, 6C/12T",
            "torch_threads": torch.get_num_threads(),
            "cuda_available": torch.cuda.is_available(),
            "container": "built from ./Dockerfile (decision D-008)",
        },
        "deps": pip_freeze(),
        "split_sizes": manifest["split_sizes"],
        "unique_dates": manifest["unique_dates"],
        "load_at_start": load_snapshot(),
    }
    (run_dir / "meta.json").write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")
    (run_dir / "tokenizer.json").write_text(tok.to_json(), encoding="utf-8")

    print("=" * 84)
    print(f"run {run_name}  |  {model.num_parameters():,} params  |  vocab {len(tok)}")
    print(f"train {len(datasets['train']):,} ex   val {len(datasets['val']):,}   test {len(datasets['test']):,}")
    print(f"unique dates: {manifest['unique_dates']}")
    print(f"threads={torch.get_num_threads()}  {describe_load(meta['load_at_start'])}")
    print("=" * 84)

    def save(tag: str, step: int) -> Path:
        path = run_dir / f"checkpoint_{tag}.pt"
        torch.save({
            "step": step, "model": model.state_dict(), "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(), "rng": rng_state(),
            "config": cfg.to_dict(), "history": history, "run_id": run_name,
        }, path)
        return path

    # ---- loop -------------------------------------------------------------------------------
    train_iter = iter(loaders["train"])
    model.train()
    step = start_step
    t_start = time.perf_counter()
    tokens_seen = 0

    # Checkpoint selection: highest exact-match, with LOWER validation loss as the tie-break.
    # The tie-break is not cosmetic. Exact-match is 0.0 for the first few hundred steps (a single
    # wrong character fails the whole sequence), so a rule of "strictly greater exact-match" would
    # freeze `best` at the very first evaluation and then never move, and the final report would
    # load a near-untrained model. Comparing (exact, -loss) lexicographically keeps improving even
    # while exact-match is pinned at zero.
    best_score = (-1.0, float("-inf"))
    best_exact = -1.0

    while step < args.steps:
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(loaders["train"])
            batch = next(train_iter)

        model.train()
        optimizer.zero_grad(set_to_none=True)
        logits = model(batch["src"], batch["tgt_in"])
        loss = loss_fn(logits, batch["labels"])
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        step += 1
        tokens_seen += int((batch["labels"] != tok.pad_id).sum())

        if step % args.log_every == 0:
            elapsed = time.perf_counter() - t_start
            rec = {
                "step": step, "train_loss_smoothed": loss.item(),
                "lr": optimizer.param_groups[0]["lr"], "grad_norm": float(grad_norm),
                "tokens_seen": tokens_seen, "elapsed_s": elapsed,
                "tokens_per_s": tokens_seen / max(elapsed, 1e-9),
            }
            history.append(rec)
            print(f"step {step:>6}  loss {loss.item():.4f}  lr {rec['lr']:.2e}  "
                  f"gnorm {rec['grad_norm']:.2f}  {rec['tokens_per_s']:.0f} tok/s")

        if step % args.eval_every == 0 or step == args.steps:
            metrics = evaluate(model, loaders["val"], tok, max_batches=args.eval_batches)
            metrics["step"] = step
            metrics["split"] = "val"
            history.append(metrics)
            print(f"  [val] step {step}  loss {metrics['val_loss_nats_per_token']:.4f}  "
                  f"tok_acc {metrics['token_accuracy_teacher_forced']:.4f}  "
                  f"exact {metrics['exact_match_free_running']:.4f}  "
                  f"(n={metrics['n_examples']})")
            score = (metrics["exact_match_free_running"], -metrics["val_loss_nats_per_token"])
            if score > best_score:
                best_score = score
                best_exact = metrics["exact_match_free_running"]
                save("best", step)
                print(f"        new best (exact={score[0]:.4f}, loss={-score[1]:.4f}) -> checkpoint_best.pt")
            save("last", step)
            (run_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")

    duration = time.perf_counter() - t_start

    # ---- final evaluation on the held-out TEST split ----------------------------------------
    best_path = run_dir / "checkpoint_best.pt"
    if best_path.exists():
        model.load_state_dict(torch.load(best_path, map_location="cpu", weights_only=False)["model"])

    print("\nfinal evaluation on held-out splits (full, not subsampled)")
    final = {}
    for split in ("val", "test"):
        final[split] = evaluate(model, loaders[split], tok, max_batches=None)
        print(f"  {split:>5}: loss {final[split]['val_loss_nats_per_token']:.4f}  "
              f"tok_acc {final[split]['token_accuracy_teacher_forced']:.4f}  "
              f"exact {final[split]['exact_match_free_running']:.4f}  "
              f"(n={final[split]['n_examples']})")

    predictions = sample_predictions(model, datasets["test"], tok, n=24)
    failures = [p for p in predictions if not p["correct"]]

    result = {
        **meta,
        "duration_s": duration,
        "steps_completed": step,
        "tokens_seen": tokens_seen,
        "final_metrics": final,
        "best_val_exact_match": best_exact,
        "history": history,
        "sample_predictions": predictions,
        "n_failures_in_sample": len(failures),
        "load_at_end": load_snapshot(),
        "checkpoints": {
            name: {"path": str(run_dir / f"checkpoint_{name}.pt"),
                   "sha256": sha256_file(run_dir / f"checkpoint_{name}.pt")}
            for name in ("best", "last") if (run_dir / f"checkpoint_{name}.pt").exists()
        },
        "metric_definitions": {
            "val_loss_nats_per_token": "mean unsmoothed cross-entropy per non-pad target token, nats",
            "token_accuracy_teacher_forced": "fraction of non-pad target positions argmax-correct, correct prefix supplied",
            "exact_match_free_running": "fraction of examples whose full greedy decode from BOS equals the target string exactly; no teacher forcing",
        },
        "limitations": [
            "Synthetic task generated by labs/p1_transformer/data.py; not a natural-language benchmark.",
            "Tier R (reduced-scale reproduction) at most. No comparison to the paper's WMT14 results is made or implied.",
            "Single seed unless a companion run_id says otherwise.",
            "Measured on a machine shared with unrelated containers; throughput is a lower bound (GAPS G-008).",
        ],
    }
    (run_dir / "result.json").write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")

    print(f"\nduration {duration / 60:.1f} min | tokens {tokens_seen:,} | "
          f"{tokens_seen / duration:.0f} tok/s")
    print(f"failures in a 24-example sample: {len(failures)}")
    for f in failures[:5]:
        print(f"    {f['source']!r} -> got {f['prediction']!r}, want {f['target']!r}")
    print(f"\nwritten to {run_dir}")


if __name__ == "__main__":
    main()
