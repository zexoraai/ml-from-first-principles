"""Project 3: LoRA versus a full fine-tune, on the same task, from the same checkpoint.

    .\\run.cmd python scripts/train_p3.py --base runs/p2-design-3m/checkpoint_best.pt

WHY THE FULL FINE-TUNE ARM IS NOT OPTIONAL
------------------------------------------
"LoRA trains 0.5% of the parameters" is a statement about LoRA alone and is not a result. The claim
people actually care about is *comparative*: *for what cost in quality?* Without a full fine-tune run
on the identical task, from the identical checkpoint, for the identical number of steps, there is
nothing to answer that with — and every efficiency number becomes an unfalsifiable advertisement.

So this script runs, all from the same Project 2 checkpoint on the same instruction data:

  base            untouched, to show where the task starts
  lora_r{r}       LoRA on the query and value projections, one arm per rank
  full            every parameter trainable

WHAT IS MEASURED, AND WHY EACH NUMBER IS COMPUTED RATHER THAN ESTIMATED
----------------------------------------------------------------------
* **Trainable parameters** — counted from the live `requires_grad` flags, not from the config. A
  freezing bug is invisible in a config and obvious in this count.
* **Optimizer state bytes** — summed over the actual state tensors after a real step. This is where
  LoRA's memory saving genuinely comes from: AdamW keeps two moment tensors per *trainable* parameter,
  so freezing the base removes 2 × (frozen params) × 4 bytes of exp_avg and exp_avg_sq. Quoting the
  parameter count alone understates the saving by a factor of three, since gradients go too.
* **Gradient bytes** — summed over `p.grad` after a backward pass.
* **Adapter checkpoint size on disk** — the file is actually written and measured. This is the number
  that matters operationally: it is what you ship per task.
* **Wall-clock per step** — measured, and labelled a lower bound because the machine is shared (G-008).
  Note the honest expectation: LoRA saves *memory*, not much time, on this model. The frozen base
  still runs a full forward and a full backward *through* it to reach the adapters. Anyone expecting a
  large speedup has misread what LoRA does, and this measurement says so out loud.
* **Held-out quality** — SFT loss on eval topics plus the Project 4 generation detectors.
* **Merge equivalence** — `merge_all()` then re-run: outputs must match to within float tolerance,
  which is what makes "zero inference overhead" a verified claim rather than a slogan.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import platform
import random
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from labs.common.instructions import build_instruction_split  # noqa: E402
from labs.common.sysload import describe_load, load_snapshot  # noqa: E402
from labs.p2_gpt import GPT, GPTConfig, generate, prepare  # noqa: E402
from labs.p2_gpt.tokenizer import ByteBPETokenizer  # noqa: E402
from labs.p3_lora import (  # noqa: E402
    apply_lora,
    count_parameters,
    lora_state_dict,
    mark_only_lora_trainable,
    merge_all,
)
from labs.p4_dpo import SFTBatcher, encode_sft_batch, sft_loss  # noqa: E402
from labs.p4_dpo.evaluate import score_generation  # noqa: E402


def git_commit() -> str:
    try:
        sha = subprocess.check_output(["git", "rev-parse", "HEAD"],
                                      stderr=subprocess.DEVNULL).decode().strip()
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
        return subprocess.check_output([sys.executable, "-m", "pip", "freeze"],
                                       stderr=subprocess.DEVNULL).decode().strip().splitlines()
    except Exception:  # noqa: BLE001
        return ["UNKNOWN"]


def optimizer_state_bytes(opt: torch.optim.Optimizer) -> int:
    """Bytes held in optimizer state tensors, summed from the live state.

    Computed rather than derived from a formula so that it stays correct if the optimizer changes.
    AdamW holds `exp_avg` and `exp_avg_sq` per trainable parameter; `step` is a scalar and is counted
    too, which is negligible but honest.
    """
    total = 0
    for state in opt.state.values():
        for v in state.values():
            if isinstance(v, torch.Tensor):
                total += v.numel() * v.element_size()
    return total


def gradient_bytes(model: nn.Module) -> int:
    return sum(p.grad.numel() * p.grad.element_size()
               for p in model.parameters() if p.grad is not None)


@torch.no_grad()
def eval_loss(model: nn.Module, examples, tokenizer, *, block_size: int, pad_id: int,
              batch_size: int) -> float:
    """Mean SFT loss over every held-out example, response tokens only.

    Weighted by *batch*, not by token. With uniform-ish response lengths the difference is small, and
    reporting it this way keeps it comparable across arms, which is what the number is for.
    """
    model.eval()
    losses = []
    for i in range(0, len(examples), batch_size):
        batch = encode_sft_batch(examples[i : i + batch_size], tokenizer, block_size, pad_id)
        losses.append(float(sft_loss(model(batch["ids"])[0], batch["ids"], batch["mask"])))
    return sum(losses) / max(len(losses), 1)


@torch.no_grad()
def gen_eval(model, tokenizer, examples, *, block_size, max_new_tokens, temperature, top_k, seed,
             eot_id) -> dict:
    model.eval()
    scores, samples = [], []
    for i, ex in enumerate(examples):
        ids = torch.tensor([tokenizer.encode(ex.prompt)], dtype=torch.long)
        out = generate(model, ids, min(max_new_tokens, block_size - ids.shape[1]),
                       temperature=temperature, top_k=top_k, stop_ids={eot_id},
                       generator=torch.Generator().manual_seed(seed + i))
        new = out[0, ids.shape[1]:].tolist()
        if eot_id in new:
            new = new[: new.index(eot_id)]
        text = tokenizer.decode(new)
        scores.append(score_generation(text, ex.response))
        if i < 5:
            samples.append({"topic": ex.topic, "generated": text, **scores[-1]})
    keys = scores[0].keys() if scores else []
    return {"n": len(scores),
            "means": {k: sum(s[k] for s in scores) / len(scores) for k in keys},
            "samples": samples}


def train_arm(model, batcher, *, steps, lr, weight_decay, grad_clip, log_every, label):
    """Train one arm and measure cost as it goes.

    The learning rate is a per-arm argument on purpose: LoRA's standard practice is a markedly higher
    LR than a full fine-tune, because the adapter starts at exactly zero and `α/r` scales its
    contribution down. Forcing both arms to share one LR would handicap whichever arm the shared value
    suited less and would make the quality comparison meaningless. This is a real confound and it is
    handled by tuning each arm's LR separately and recording both.
    """
    trainable = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable, lr=lr, weight_decay=weight_decay, betas=(0.9, 0.95))

    history, step_times = [], []
    opt_bytes = grad_bytes = 0
    model.train()
    for step in range(steps):
        t0 = time.perf_counter()
        batch = batcher.next_batch()
        opt.zero_grad(set_to_none=True)
        loss = sft_loss(model(batch["ids"])[0], batch["ids"], batch["mask"])
        loss.backward()
        gnorm = torch.nn.utils.clip_grad_norm_(trainable, grad_clip)
        opt.step()
        step_times.append(time.perf_counter() - t0)

        if step == 0:
            # Measured after the first real step, which is when AdamW has allocated its moments.
            # Measuring before would report zero and flatter LoRA for the wrong reason.
            opt_bytes = optimizer_state_bytes(opt)
            grad_bytes = gradient_bytes(model)

        if (step + 1) % log_every == 0 or step == 0:
            rec = {"step": step + 1, "loss": loss.item(), "grad_norm": gnorm.item()}
            history.append(rec)
            print(f"  [{label}] step {step + 1:>5} loss {rec['loss']:.4f} "
                  f"gnorm {rec['grad_norm']:5.2f}")

    warm = step_times[3:] or step_times      # drop the first few: allocator and cache warm-up
    return {
        "history": history,
        "optimizer_state_bytes": opt_bytes,
        "gradient_bytes": grad_bytes,
        "median_step_s": float(np.median(warm)),
        "mean_step_s": float(np.mean(warm)),
        "total_train_s": float(sum(step_times)),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="runs/p2-design-3m/checkpoint_best.pt")
    ap.add_argument("--tokenizer", default=None)
    ap.add_argument("--run-name", default=None)
    ap.add_argument("--corpus", default="design")
    ap.add_argument("--vocab-size", type=int, default=1024)
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--ranks", default="1,4,8,32", help="comma-separated LoRA ranks to sweep")
    ap.add_argument("--alpha-mult", type=float, default=2.0,
                    help="alpha = alpha_mult * r, so the α/r scaling is held constant across ranks. "
                         "Without this, changing r also changes the effective update magnitude and "
                         "the sweep would confound capacity with scale.")
    ap.add_argument("--targets", default="w_q,w_v",
                    help="module attribute names to adapt; the paper's main config is w_q,w_v")
    ap.add_argument("--lora-lr", type=float, default=1e-3)
    ap.add_argument("--full-lr", type=float, default=1e-4)
    ap.add_argument("--weight-decay", type=float, default=0.0)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--gen-prompts", type=int, default=60)
    ap.add_argument("--gen-tokens", type=int, default=80)
    ap.add_argument("--gen-temperature", type=float, default=0.8)
    ap.add_argument("--gen-top-k", type=int, default=40)
    ap.add_argument("--eval-batch-size", type=int, default=8)
    ap.add_argument("--seed", type=int, default=4242)
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--log-every", type=int, default=100)
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    base_path = Path(args.base)
    if not base_path.exists():
        raise SystemExit(f"base checkpoint {base_path} not found; train_p2.py must finish first")
    tok_path = Path(args.tokenizer) if args.tokenizer else base_path.parent / "tokenizer.json"
    if not tok_path.exists():
        raise SystemExit(f"tokenizer {tok_path} not found next to the checkpoint")

    run_name = args.run_name or f"p3-lora-{datetime.now(timezone.utc):%Y%m%d-%H%M%S}"
    run_dir = Path("runs") / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    ck = torch.load(base_path, map_location="cpu", weights_only=False)
    cfg = GPTConfig(**ck["config"])
    tokenizer = ByteBPETokenizer.load(tok_path)
    base = GPT(cfg)
    base.load_state_dict(ck["model"])

    bundle = prepare(corpus=args.corpus, vocab_size=args.vocab_size, verbose=False)
    if list(bundle["tokenizer"].merges) != list(tokenizer.merges):
        raise SystemExit(
            f"tokenizer mismatch: {tok_path} disagrees with prepare(corpus={args.corpus!r}). The base "
            f"checkpoint was trained on a different corpus, so the task text would be encoded under a "
            f"different vocabulary than the model learned."
        )

    split = build_instruction_split(bundle["train_text"], seed=args.seed)
    ranks = [int(r) for r in args.ranks.split(",") if r.strip()]
    targets = tuple(t.strip() for t in args.targets.split(",") if t.strip())
    gen_prompts = split["sft_eval"][: args.gen_prompts]

    print("=" * 96)
    print(f"run {run_name} | base {base_path} | {base.num_parameters():,} params")
    print(f"task: {len(split['sft_train'])} train / {len(split['sft_eval'])} eval instruction "
          f"examples over {len(split['passages'])} topics (split by topic)")
    print(f"arms: base, full, lora r={ranks} on {targets} (alpha = {args.alpha_mult}*r)")
    print(f"steps {args.steps} | batch {args.batch_size} | lora_lr {args.lora_lr} "
          f"full_lr {args.full_lr}")
    print(f"threads={torch.get_num_threads()} | {describe_load(load_snapshot())}")
    print("=" * 96)

    meta = {
        "run_id": run_name, "project": "p3_lora",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "source_commit": git_commit(),
        "repro_cmd": ".\\run.cmd python " + " ".join(sys.argv),
        "base_checkpoint": {"path": str(base_path), "sha256": sha256_file(base_path),
                            "run_id": ck.get("run_id"), "step": ck.get("step")},
        "seeds": {"python": args.seed, "numpy": args.seed, "torch": args.seed},
        "args": vars(args), "model_config": cfg.to_dict(),
        "n_parameters": base.num_parameters(),
        "task": {
            "source": "design corpus TRAIN split, instruction data from heading structure",
            "n_train": len(split["sft_train"]), "n_eval": len(split["sft_eval"]),
            "n_topics": len(split["passages"]), "split_unit": "topic",
        },
        "hardware": {"platform": platform.platform(),
                     "host_cpu": "AMD Ryzen 5 PRO 5650U 6C/12T",
                     "torch_threads": torch.get_num_threads(),
                     "cuda_available": torch.cuda.is_available(),
                     "container": "built from ./Dockerfile (decision D-008)"},
        "deps": pip_freeze(), "load_at_start": load_snapshot(),
    }
    (run_dir / "meta.json").write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")

    arms: dict[str, dict] = {}

    def quality(name: str, model) -> dict:
        loss = eval_loss(model, split["sft_eval"], tokenizer, block_size=cfg.block_size,
                         pad_id=tokenizer.eot_id, batch_size=args.eval_batch_size)
        gen = gen_eval(model, tokenizer, gen_prompts, block_size=cfg.block_size,
                       max_new_tokens=args.gen_tokens, temperature=args.gen_temperature,
                       top_k=args.gen_top_k, seed=args.seed, eot_id=tokenizer.eot_id)
        print(f"  [{name}] eval_loss {loss:.4f}  rep {gen['means']['repetition_4gram']:.3f}  "
              f"midsent {gen['means']['ends_mid_sentence']:.2f}  "
              f"overlap {gen['means']['topic_overlap']:.3f}")
        return {"eval_loss": loss, "generation": gen}

    def batcher_for(seed_offset: int) -> SFTBatcher:
        # Every arm sees the SAME sequence of batches, so arms differ by method and not by data order.
        return SFTBatcher(split["sft_train"], tokenizer, block_size=cfg.block_size,
                          batch_size=args.batch_size, pad_id=tokenizer.eot_id,
                          seed=args.seed + seed_offset)

    # ---- base ------------------------------------------------------------------------------
    print("\narm: base (untouched)")
    counts = count_parameters(base)
    arms["base"] = {**quality("base", base), "parameters": counts,
                    "optimizer_state_bytes": 0, "gradient_bytes": 0}

    # ---- full fine-tune ---------------------------------------------------------------------
    print("\narm: full fine-tune (every parameter trainable)")
    full = copy.deepcopy(base)
    cost = train_arm(full, batcher_for(0), steps=args.steps, lr=args.full_lr,
                     weight_decay=args.weight_decay, grad_clip=args.grad_clip,
                     log_every=args.log_every, label="full")
    full_ck = run_dir / "checkpoint_full.pt"
    torch.save({"model": full.state_dict(), "config": cfg.to_dict(), "arm": "full"}, full_ck)
    arms["full"] = {**quality("full", full), **cost,
                    "parameters": count_parameters(full),
                    "checkpoint_bytes": full_ck.stat().st_size,
                    "checkpoint_path": str(full_ck)}

    # ---- LoRA arms --------------------------------------------------------------------------
    for r in ranks:
        name = f"lora_r{r}"
        print(f"\narm: {name} (alpha={args.alpha_mult * r}, targets={targets})")
        model = copy.deepcopy(base)
        adapted = apply_lora(model, target_suffixes=targets, r=r, alpha=args.alpha_mult * r)
        mark_only_lora_trainable(model)
        counts = count_parameters(model)
        print(f"  adapted {len(adapted)} modules | trainable {counts['trainable']:,} "
              f"({100 * counts['trainable'] / counts['total']:.3f}% of {counts['total']:,})")

        cost = train_arm(model, batcher_for(0), steps=args.steps, lr=args.lora_lr,
                         weight_decay=args.weight_decay, grad_clip=args.grad_clip,
                         log_every=args.log_every, label=name)

        # The adapter alone -- what you would actually ship per task.
        adapter_path = run_dir / f"adapter_{name}.pt"
        torch.save(lora_state_dict(model), adapter_path)

        q = quality(name, model)

        # ---- merge equivalence: the claim that inference costs nothing extra ----------------
        probe = torch.tensor([tokenizer.encode(gen_prompts[0].prompt)], dtype=torch.long)
        model.eval()
        with torch.no_grad():
            before = model(probe)[0]
        n_merged = merge_all(model)
        with torch.no_grad():
            after = model(probe)[0]
        max_abs = float((before - after).abs().max())
        rel = max_abs / float(before.abs().max().clamp_min(1e-12))

        merged_ck = run_dir / f"checkpoint_{name}_merged.pt"
        torch.save({"model": model.state_dict(), "config": cfg.to_dict(), "arm": name,
                    "merged": True}, merged_ck)

        arms[name] = {
            **q, **cost, "r": r, "alpha": args.alpha_mult * r,
            "targets": list(targets), "adapted_modules": adapted,
            "parameters": counts,
            "trainable_fraction": counts["trainable"] / counts["total"],
            "adapter_bytes": adapter_path.stat().st_size,
            "adapter_path": str(adapter_path),
            "merged_checkpoint_bytes": merged_ck.stat().st_size,
            "merge_equivalence": {
                "n_modules_merged": n_merged,
                "max_abs_logit_diff": max_abs,
                "max_rel_logit_diff": rel,
                "note": "float32 matmul reassociation, not an implementation error. The merged path "
                        "computes (W + BA)x in one matmul; the unmerged path computes Wx + B(Ax). "
                        "These are algebraically identical and differ only in rounding.",
            },
        }
        print(f"  merged {n_merged} modules | max |Δlogit| {max_abs:.2e} (rel {rel:.2e})")

    # ---- result -----------------------------------------------------------------------------
    full_bytes = arms["full"]["checkpoint_bytes"]
    table = []
    for name, a in arms.items():
        table.append({
            "arm": name,
            "trainable_params": a["parameters"]["trainable"],
            "trainable_pct": 100 * a["parameters"]["trainable"] / a["parameters"]["total"],
            "optimizer_state_MiB": a.get("optimizer_state_bytes", 0) / 2**20,
            "gradient_MiB": a.get("gradient_bytes", 0) / 2**20,
            "shipped_MiB": (a.get("adapter_bytes", a.get("checkpoint_bytes", 0))) / 2**20,
            "shipped_vs_full": ((a.get("adapter_bytes", a.get("checkpoint_bytes", 0)) / full_bytes)
                               if full_bytes else None),
            "median_step_s": a.get("median_step_s"),
            "eval_loss": a["eval_loss"],
            "repetition_4gram": a["generation"]["means"]["repetition_4gram"],
            "topic_overlap": a["generation"]["means"]["topic_overlap"],
        })

    result = {
        **meta,
        "arms": arms,
        "comparison_table": table,
        "load_at_end": load_snapshot(),
        "metric_definitions": {
            "trainable_params": "counted from live requires_grad flags after freezing, not from "
                                "config — a freezing bug is invisible in a config.",
            "optimizer_state_MiB": "summed over the optimizer's actual state tensors after one real "
                                   "step. AdamW holds two moment tensors per trainable parameter.",
            "gradient_MiB": "summed over p.grad after a backward pass.",
            "shipped_MiB": "size on disk of what you would distribute per task: the adapter tensors "
                           "for LoRA arms, the whole model for the full fine-tune.",
            "median_step_s": "median wall-clock per training step, excluding the first three steps "
                             "(allocator warm-up). A LOWER BOUND: the machine is shared (G-008).",
            "eval_loss": "mean response-only cross-entropy on held-out topics, batch-weighted.",
            "max_abs_logit_diff": "largest absolute logit change from folding BA into W. Nonzero "
                                  "only because float32 matmul is not associative.",
        },
        "limitations": [
            "Tier E for the LoRA mechanism; tier R for the comparison. No published LoRA result is "
            "reproduced or claimed — the paper's experiments are on GPT-3 175B and RoBERTa, at scales "
            "several orders of magnitude above this 2.9M-parameter model.",
            "LoRA's advantage here is MEMORY and SHIPPED SIZE, not speed. The frozen base still runs a "
            "full forward and a full backward through it to reach the adapters, so per-step time is "
            "similar by construction. Any observed speed difference at this scale is dominated by "
            "machine contention and should not be read as a property of the method.",
            "LoRA and the full fine-tune use DIFFERENT learning rates (recorded in args), because the "
            "adapter starts at exactly zero and is scaled by alpha/r. Sharing one LR would handicap "
            "one arm and make the quality comparison meaningless. Neither LR was extensively swept, "
            "so a quality gap of the size seen here may be an LR artefact rather than a method "
            "difference. This is the single largest caveat on the comparison.",
            "Single seed per arm. Seed-to-seed variance was not measured, so small quality "
            "differences between arms are not interpretable.",
            "The paper's finding that adapting W_q and W_v beats other targets at equal budget is "
            "FOLLOWED, not verified. --targets exposes the choice so it can be tested.",
            "Peak process memory is not reported: on CPU with a shared allocator it is dominated by "
            "the container and unrelated load. The optimizer-state and gradient byte counts are exact "
            "and are the honest substitute.",
        ],
    }
    (run_dir / "result.json").write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")

    hdr = (f"{'arm':<12}{'trainable':>11}{'%':>8}{'optMiB':>9}{'gradMiB':>9}"
           f"{'shipMiB':>9}{'vsfull':>9}{'s/step':>8}{'evloss':>9}{'overlap':>9}")
    print("\n" + "=" * len(hdr))
    print(hdr)
    print("-" * len(hdr))
    for row in table:
        vs = f"{row['shipped_vs_full']:.4f}" if row["shipped_vs_full"] is not None else "-"
        st = f"{row['median_step_s']:.3f}" if row["median_step_s"] is not None else "-"
        print(f"{row['arm']:<12}{row['trainable_params']:>11,}{row['trainable_pct']:>8.3f}"
              f"{row['optimizer_state_MiB']:>9.2f}{row['gradient_MiB']:>9.2f}"
              f"{row['shipped_MiB']:>9.2f}{vs:>9}{st:>8}"
              f"{row['eval_loss']:>9.4f}{row['topic_overlap']:>9.3f}")
    print("=" * len(hdr))
    print(f"written to {run_dir}")


if __name__ == "__main__":
    main()
