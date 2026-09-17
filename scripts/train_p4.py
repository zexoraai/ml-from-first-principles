"""Project 4: SFT then DPO on the Project 2 GPT, with a β sweep and a compute-matched control.

    .\\run.cmd python scripts/train_p4.py --base runs/p2-design-3m/checkpoint_best.pt

WHAT THIS SCRIPT IS DESIGNED TO BE ABLE TO CONCLUDE
--------------------------------------------------
"DPO beat SFT" is worthless without saying what was held fixed. This script runs:

  base            the pretrained Project 2 model, untouched
  sft             base + N_sft supervised steps on instruction data. **This is also π_ref.**
  sft_continued   sft + N_dpo *more* supervised steps  ← the control that matters
  dpo_b{β}        sft + N_dpo DPO steps, one arm per β

`sft_continued` exists because without it, any DPO gain is confounded with simply having taken more
gradient steps on more data. Every DPO arm and the control start from the identical checkpoint, see the
same number of optimiser steps, and are evaluated on topics that appear in no training example of any
arm. That is the whole design: the only thing varying between `sft_continued` and `dpo_b0.1` is the
objective.

Reported per arm: preference accuracy (raw and length-normalised), implicit reward margin, log-ratio
from the reference, generation-side degradation rates from deterministic detectors, and held-out
perplexity on the pretraining corpus as an alignment-tax check.
"""

from __future__ import annotations

import argparse
import copy
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
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from labs.common.sysload import describe_load, load_snapshot  # noqa: E402
from labs.p2_gpt import GPT, BatchSampler, GPTConfig, generate, prepare  # noqa: E402
from labs.p2_gpt.tokenizer import ByteBPETokenizer  # noqa: E402
from labs.p4_dpo import (  # noqa: E402
    PairBatcher,
    SFTBatcher,
    build_sft_and_preferences,
    dpo_loss,
    make_reference_model,
    sequence_logprobs,
    sft_loss,
)
from labs.p4_dpo.evaluate import corpus_perplexity, preference_metrics, score_generation  # noqa: E402


# ---------------------------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------------------------

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


def set_dropout(model: nn.Module, p: float) -> None:
    """Force every dropout probability in the model.

    Used to disable dropout for the DPO stage, which is a deliberate departure from "just keep the
    policy in train mode".

    The reference is frozen in eval mode, so `log π_ref` is deterministic. If the policy keeps dropout
    active then `log π_θ` is stochastic, and the objective depends on the *difference* of the two — so
    the noise enters the margin asymmetrically and is not averaged away by the reference term. The
    result is a preference signal competing with sampling noise on a dataset of a few hundred pairs.
    Dropout stays on for SFT, where the loss is a plain expectation and dropout is doing its usual job.
    """
    for module in model.modules():
        if isinstance(module, nn.Dropout):
            module.p = p


def lr_at(step: int, *, base_lr: float, warmup: int, total: int, min_lr: float) -> float:
    if step < warmup:
        return base_lr * (step + 1) / max(warmup, 1)
    if step >= total:
        return min_lr
    progress = (step - warmup) / max(total - warmup, 1)
    return min_lr + 0.5 * (base_lr - min_lr) * (1.0 + math.cos(math.pi * progress))


def make_optimizer(model: nn.Module, lr: float, weight_decay: float):
    decay, no_decay, seen = [], [], set()
    for module in model.modules():
        for _, p in module.named_parameters(recurse=False):
            if id(p) in seen:
                continue
            seen.add(id(p))
            (decay if p.dim() >= 2 else no_decay).append(p)
    return torch.optim.AdamW(
        [{"params": decay, "weight_decay": weight_decay},
         {"params": no_decay, "weight_decay": 0.0}], lr=lr, betas=(0.9, 0.95),
    )


# ---------------------------------------------------------------------------------------------
# stages
# ---------------------------------------------------------------------------------------------

def run_sft(model, batcher, *, steps, lr, warmup, min_lr, weight_decay, grad_clip, log_every,
            label) -> list[dict]:
    """Supervised fine-tuning on instruction data, loss on response tokens only."""
    opt = make_optimizer(model, lr, weight_decay)
    history = []
    model.train()
    for step in range(steps):
        cur_lr = lr_at(step, base_lr=lr, warmup=warmup, total=steps, min_lr=min_lr)
        for g in opt.param_groups:
            g["lr"] = cur_lr
        batch = batcher.next_batch()
        opt.zero_grad(set_to_none=True)
        logits, _ = model(batch["ids"])
        loss = sft_loss(logits, batch["ids"], batch["mask"])
        loss.backward()
        gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        opt.step()
        if (step + 1) % log_every == 0 or step == 0:
            rec = {"step": step + 1, "loss": loss.item(), "lr": cur_lr,
                   "grad_norm": gnorm.item()}
            history.append(rec)
            print(f"  [{label}] step {step + 1:>5} loss {rec['loss']:.4f} "
                  f"lr {cur_lr:.2e} gnorm {rec['grad_norm']:5.2f}")
    return history


def run_dpo(model, reference, batcher, *, steps, beta, lr, warmup, min_lr, weight_decay, grad_clip,
            label_smoothing, log_every, label) -> list[dict]:
    """DPO. The reference is frozen; the policy has dropout disabled (see `set_dropout`)."""
    opt = make_optimizer(model, lr, weight_decay)
    history = []
    set_dropout(model, 0.0)
    model.train()
    for step in range(steps):
        cur_lr = lr_at(step, base_lr=lr, warmup=warmup, total=steps, min_lr=min_lr)
        for g in opt.param_groups:
            g["lr"] = cur_lr
        batch = batcher.next_batch()

        opt.zero_grad(set_to_none=True)
        pc = sequence_logprobs(model(batch["chosen_ids"])[0], batch["chosen_ids"],
                               batch["chosen_mask"])
        pr = sequence_logprobs(model(batch["rejected_ids"])[0], batch["rejected_ids"],
                               batch["rejected_mask"])
        with torch.no_grad():
            rc = sequence_logprobs(reference(batch["chosen_ids"])[0], batch["chosen_ids"],
                                   batch["chosen_mask"])
            rr = sequence_logprobs(reference(batch["rejected_ids"])[0], batch["rejected_ids"],
                                   batch["rejected_mask"])

        stats = dpo_loss(pc, pr, rc, rr, beta=beta, label_smoothing=label_smoothing)
        stats.loss.backward()
        gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        opt.step()

        if (step + 1) % log_every == 0 or step == 0:
            rec = {"step": step + 1, "lr": cur_lr, "grad_norm": gnorm.item(), **stats.to_dict()}
            history.append(rec)
            print(f"  [{label}] step {step + 1:>5} loss {rec['loss']:.4f} "
                  f"margin {rec['reward_margin']:+.4f} acc {rec['reward_accuracy']:.2f} "
                  f"logp_w {rec['chosen_logps']:8.2f} logp_l {rec['rejected_logps']:8.2f}")
    return history


@torch.no_grad()
def generation_eval(model, tokenizer, examples, *, block_size, max_new_tokens, temperature, top_k,
                    seed, eot_id) -> dict:
    """Sample one response per held-out prompt and score it with the deterministic detectors.

    Generation stops at the first end-of-text token so that `ends_mid_sentence` measures the model's
    own stopping behaviour rather than the token budget. The budget confound is still present for
    models that never emit `eot`, and the raw `n_words` is reported so that case is visible.
    """
    model.eval()
    scores, samples = [], []
    for i, ex in enumerate(examples):
        ids = torch.tensor([tokenizer.encode(ex.prompt)], dtype=torch.long)
        room = block_size - ids.shape[1]
        out = generate(model, ids, min(max_new_tokens, room), temperature=temperature,
                       top_k=top_k, stop_ids={eot_id},
                       generator=torch.Generator().manual_seed(seed + i))
        new = out[0, ids.shape[1]:].tolist()
        if eot_id in new:
            new = new[: new.index(eot_id)]
        text = tokenizer.decode(new)
        scores.append(score_generation(text, ex.response))
        if i < 6:
            samples.append({"topic": ex.topic, "prompt": ex.prompt, "generated": text,
                            "reference": ex.response, **scores[-1]})
    keys = scores[0].keys() if scores else []
    return {
        "n": len(scores),
        "means": {k: sum(s[k] for s in scores) / len(scores) for k in keys},
        "samples": samples,
    }


# ---------------------------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="runs/p2-design-3m/checkpoint_best.pt",
                    help="pretrained Project 2 checkpoint")
    ap.add_argument("--tokenizer", default=None,
                    help="defaults to <base dir>/tokenizer.json")
    ap.add_argument("--run-name", default=None)
    ap.add_argument("--corpus", default="design")
    ap.add_argument("--vocab-size", type=int, default=1024)
    ap.add_argument("--sft-steps", type=int, default=600)
    ap.add_argument("--dpo-steps", type=int, default=400)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--pref-batch-size", type=int, default=4)
    ap.add_argument("--sft-lr", type=float, default=3e-4)
    ap.add_argument("--dpo-lr", type=float, default=5e-5)
    ap.add_argument("--min-lr-frac", type=float, default=0.1)
    ap.add_argument("--warmup-frac", type=float, default=0.1)
    ap.add_argument("--weight-decay", type=float, default=0.0,
                    help="0 for fine-tuning: the pretrained weights are the prior, not zero")
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--betas", default="0.02,0.1,0.5",
                    help="comma-separated DPO beta values to sweep")
    ap.add_argument("--label-smoothing", type=float, default=0.0)
    ap.add_argument("--pairs-per-topic", type=int, default=3)
    ap.add_argument("--gen-prompts", type=int, default=80,
                    help="held-out prompts to generate from per arm. Generation is sequential and "
                         "is the dominant cost of evaluation; 80 keeps five arms tractable on CPU. "
                         "The count is recorded in result.json so the sampling error is visible.")
    ap.add_argument("--gen-tokens", type=int, default=100)
    ap.add_argument("--gen-temperature", type=float, default=0.8)
    ap.add_argument("--gen-top-k", type=int, default=40)
    ap.add_argument("--ppl-iters", type=int, default=60)
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
        raise SystemExit(
            f"base checkpoint {base_path} not found. Project 4 fine-tunes the Project 2 model, so "
            f"train_p2.py must finish first."
        )
    tok_path = Path(args.tokenizer) if args.tokenizer else base_path.parent / "tokenizer.json"
    if not tok_path.exists():
        raise SystemExit(f"tokenizer {tok_path} not found next to the checkpoint")

    run_name = args.run_name or f"p4-dpo-{datetime.now(timezone.utc):%Y%m%d-%H%M%S}"
    run_dir = Path("runs") / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    # ---- base model and tokenizer ------------------------------------------------------------
    ck = torch.load(base_path, map_location="cpu", weights_only=False)
    cfg = GPTConfig(**ck["config"])
    tokenizer = ByteBPETokenizer.load(tok_path)
    if tokenizer.vocab_size != cfg.vocab_size:
        raise SystemExit(
            f"tokenizer vocab {tokenizer.vocab_size} != model vocab {cfg.vocab_size}. The checkpoint "
            f"and tokenizer are from different runs; a mismatch here silently produces garbage."
        )

    base = GPT(cfg)
    base.load_state_dict(ck["model"])

    # ---- data --------------------------------------------------------------------------------
    bundle = prepare(corpus=args.corpus, vocab_size=args.vocab_size, verbose=False)

    # Matching vocab *sizes* is not enough. `pretraining_val_perplexity` is computed on
    # `bundle["val_ids"]`, which were encoded by the corpus tokenizer, while every generation and
    # every preference score uses the tokenizer loaded next to the checkpoint. If those two disagree
    # on even one merge, the same id means different text in different metrics and every number in
    # result.json is quietly wrong while nothing raises. Compare the merge lists directly.
    if list(bundle["tokenizer"].merges) != list(tokenizer.merges):
        raise SystemExit(
            f"tokenizer mismatch: {tok_path} does not have the same merges as the tokenizer "
            f"prepare(corpus={args.corpus!r}) just built. The base checkpoint was trained on a "
            f"different corpus or vocab size, so perplexity and generation would be measured under "
            f"two different encodings. Retrain the base on --corpus {args.corpus} or point --base at "
            f"a checkpoint that was."
        )
    # Deliberately the TRAIN split, not the whole corpus: the val text is what
    # `pretraining_val_perplexity` is measured on, so building instruction data from it would leak
    # the alignment-tax evaluation into fine-tuning.
    corpus_text = bundle["train_text"]

    built = build_sft_and_preferences(
        corpus_text, seed=args.seed, pairs_per_topic=args.pairs_per_topic
    )
    manifest = built["manifest"]

    sft_batcher = SFTBatcher(built["sft_train"], tokenizer, block_size=cfg.block_size,
                             batch_size=args.batch_size, pad_id=tokenizer.eot_id, seed=args.seed)
    pref_batcher = PairBatcher(built["pref_train"], tokenizer, block_size=cfg.block_size,
                               batch_size=args.pref_batch_size, pad_id=tokenizer.eot_id,
                               seed=args.seed + 1)
    eval_pref_batches = PairBatcher(built["pref_eval"], tokenizer, block_size=cfg.block_size,
                                    batch_size=args.pref_batch_size,
                                    pad_id=tokenizer.eot_id).all_batches()

    val_sampler = BatchSampler(bundle["train_ids"], bundle["val_ids"], block_size=cfg.block_size,
                               batch_size=8, seed=args.seed)

    betas = [float(b) for b in args.betas.split(",") if b.strip()]

    print("=" * 92)
    print(f"run {run_name} | base {base_path} | {base.num_parameters():,} params")
    print(f"topics {manifest['n_topics']} -> {manifest['n_train_topics']} train / "
          f"{manifest['n_eval_topics']} eval (split by topic, leakage asserted)")
    print(f"sft {manifest['n_sft_train']} examples | pref {manifest['n_pref_train']} pairs "
          f"({manifest['degradation_counts_train']})")
    print(f"stages: sft {args.sft_steps} steps, then {args.dpo_steps} steps for each of "
          f"sft_continued + dpo betas {betas}")
    print(f"threads={torch.get_num_threads()} | {describe_load(load_snapshot())}")
    print("=" * 92)

    meta = {
        "run_id": run_name,
        "project": "p4_dpo",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "source_commit": git_commit(),
        "repro_cmd": ".\\run.cmd python " + " ".join(sys.argv),
        "base_checkpoint": {"path": str(base_path), "sha256": sha256_file(base_path),
                            "run_id": ck.get("run_id"), "step": ck.get("step")},
        "seeds": {"python": args.seed, "numpy": args.seed, "torch": args.seed},
        "args": vars(args),
        "model_config": cfg.to_dict(),
        "n_parameters": base.num_parameters(),
        "preference_data": manifest,
        "hardware": {"platform": platform.platform(),
                     "host_cpu": "AMD Ryzen 5 PRO 5650U 6C/12T",
                     "torch_threads": torch.get_num_threads(),
                     "cuda_available": torch.cuda.is_available(),
                     "container": "built from ./Dockerfile (decision D-008)"},
        "deps": pip_freeze(),
        "load_at_start": load_snapshot(),
    }
    (run_dir / "meta.json").write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")

    t0 = time.perf_counter()

    # ---- stage 1: SFT ------------------------------------------------------------------------
    print("\nstage 1 — supervised fine-tuning (produces the reference policy)")
    sft_model = copy.deepcopy(base)
    sft_history = run_sft(
        sft_model, sft_batcher, steps=args.sft_steps, lr=args.sft_lr,
        warmup=int(args.sft_steps * args.warmup_frac), min_lr=args.sft_lr * args.min_lr_frac,
        weight_decay=args.weight_decay, grad_clip=args.grad_clip, log_every=args.log_every,
        label="sft",
    )
    torch.save({"model": sft_model.state_dict(), "config": cfg.to_dict(), "arm": "sft",
                "run_id": run_name}, run_dir / "checkpoint_sft.pt")

    # π_ref, fixed for every arm below. Made once, from the SFT weights, so all arms share it.
    reference = make_reference_model(sft_model)

    # ---- arms -------------------------------------------------------------------------------
    arms: dict[str, dict] = {}

    # The same held-out prompts, in the same order, for every arm. Drawing a fresh subset per arm
    # would make the arms differ by which prompts they were asked as well as by their objective.
    gen_prompts = built["sft_eval"][: args.gen_prompts]

    def evaluate(name: str, model) -> dict:
        pref = preference_metrics(model, reference, eval_pref_batches, beta=0.1)
        gen = generation_eval(model, tokenizer, gen_prompts, block_size=cfg.block_size,
                              max_new_tokens=args.gen_tokens, temperature=args.gen_temperature,
                              top_k=args.gen_top_k, seed=args.seed, eot_id=tokenizer.eot_id)
        ppl = corpus_perplexity(model, val_sampler.get_batch, iters=args.ppl_iters)
        out = {"preference": pref, "generation": gen, "pretraining_val_perplexity": ppl}
        print(f"  [{name}] pref_acc {pref['preference_accuracy']:.3f} "
              f"(len-norm {pref['preference_accuracy_length_normalised']:.3f})  "
              f"margin {pref['mean_implicit_reward_margin']:+.4f}  "
              f"logratio {pref['mean_logratio_vs_reference']:+.2f}  "
              f"rep {gen['means']['repetition_4gram']:.3f}  "
              f"midsent {gen['means']['ends_mid_sentence']:.2f}  "
              f"overlap {gen['means']['topic_overlap']:.3f}  ppl {ppl:.2f}")
        return out

    print("\nstage 2 — evaluation of the fixed points")
    arms["base"] = evaluate("base", base)
    arms["sft"] = {**evaluate("sft", sft_model), "history": sft_history}

    # compute-matched control: more SFT, not DPO
    print("\nstage 3 — control arm: sft_continued (same extra steps, supervised objective)")
    ctrl = copy.deepcopy(sft_model)
    ctrl_batcher = SFTBatcher(built["sft_train"], tokenizer, block_size=cfg.block_size,
                              batch_size=args.batch_size, pad_id=tokenizer.eot_id,
                              seed=args.seed + 7)
    ctrl_history = run_sft(
        ctrl, ctrl_batcher, steps=args.dpo_steps, lr=args.dpo_lr,
        warmup=int(args.dpo_steps * args.warmup_frac), min_lr=args.dpo_lr * args.min_lr_frac,
        weight_decay=args.weight_decay, grad_clip=args.grad_clip, log_every=args.log_every,
        label="sft_cont",
    )
    arms["sft_continued"] = {**evaluate("sft_continued", ctrl), "history": ctrl_history}
    torch.save({"model": ctrl.state_dict(), "config": cfg.to_dict(), "arm": "sft_continued",
                "run_id": run_name}, run_dir / "checkpoint_sft_continued.pt")

    # ---- stage 4: DPO, one arm per beta -----------------------------------------------------
    for beta in betas:
        print(f"\nstage 4 — DPO, beta={beta}")
        policy = copy.deepcopy(sft_model)
        b_batcher = PairBatcher(built["pref_train"], tokenizer, block_size=cfg.block_size,
                                batch_size=args.pref_batch_size, pad_id=tokenizer.eot_id,
                                seed=args.seed + 1)
        hist = run_dpo(
            policy, reference, b_batcher, steps=args.dpo_steps, beta=beta, lr=args.dpo_lr,
            warmup=int(args.dpo_steps * args.warmup_frac), min_lr=args.dpo_lr * args.min_lr_frac,
            weight_decay=args.weight_decay, grad_clip=args.grad_clip,
            label_smoothing=args.label_smoothing, log_every=args.log_every,
            label=f"dpo b={beta}",
        )
        name = f"dpo_beta{beta}"
        arms[name] = {**evaluate(name, policy), "history": hist, "beta": beta}
        torch.save({"model": policy.state_dict(), "config": cfg.to_dict(), "arm": name,
                    "beta": beta, "run_id": run_name}, run_dir / f"checkpoint_{name}.pt")

    duration = time.perf_counter() - t0

    # ---- result -----------------------------------------------------------------------------
    table = [
        {
            "arm": name,
            "preference_accuracy": a["preference"]["preference_accuracy"],
            "preference_accuracy_length_normalised":
                a["preference"]["preference_accuracy_length_normalised"],
            "mean_implicit_reward_margin": a["preference"]["mean_implicit_reward_margin"],
            "mean_logratio_vs_reference": a["preference"]["mean_logratio_vs_reference"],
            "repetition_4gram": a["generation"]["means"]["repetition_4gram"],
            "ends_mid_sentence": a["generation"]["means"]["ends_mid_sentence"],
            "topic_overlap": a["generation"]["means"]["topic_overlap"],
            "pretraining_val_perplexity": a["pretraining_val_perplexity"],
        }
        for name, a in arms.items()
    ]

    result = {
        **meta,
        "duration_s": duration,
        "arms": arms,
        "comparison_table": table,
        "load_at_end": load_snapshot(),
        "checkpoints": {
            p.stem.replace("checkpoint_", ""): {"path": str(p), "sha256": sha256_file(p)}
            for p in sorted(run_dir.glob("checkpoint_*.pt"))
        },
        "metric_definitions": {
            "preference_accuracy": "fraction of held-out pairs where the policy assigns a higher "
                                   "SUMMED response log-probability to chosen than to rejected.",
            "preference_accuracy_length_normalised": "the same, using the MEAN per-token "
                                                     "log-probability. The gap between the two is "
                                                     "the length bias of the summed objective.",
            "mean_implicit_reward_margin": "beta * [(logp_w - logp_ref_w) - (logp_l - logp_ref_l)], "
                                           "evaluated at beta=0.1 for every arm so the arms are "
                                           "comparable regardless of their training beta.",
            "mean_logratio_vs_reference": "E[log pi - log pi_ref] on chosen responses. A one-sided "
                                          "sequence-level log-ratio, NOT a symmetric KL.",
            "repetition_4gram": "fraction of duplicate word 4-grams in generated text. 0 = none.",
            "ends_mid_sentence": "fraction of generations not ending in sentence-final punctuation.",
            "topic_overlap": "Jaccard overlap of 4+-character content words with the reference "
                             "passage. Detects off-topic drift ONLY; rewards parroting.",
            "pretraining_val_perplexity": "exp(cross-entropy) on the held-out pretraining corpus. "
                                          "The alignment-tax check.",
        },
        "limitations": [
            "Tier E for the DPO mechanism; tier R for the comparison. No published DPO result is "
            "reproduced or claimed.",
            "PREFERENCES ARE CONSTRUCTED, NOT HUMAN-ANNOTATED. See preference_data.manifest. The "
            "result measures whether DPO moves a policy toward a verifiable, rule-defined "
            "preference. It measures nothing about human values, helpfulness or truthfulness.",
            "The base model has 2.9M parameters and was pretrained on 6.2M characters. It is far "
            "below the scale at which instruction following emerges, so absolute generation quality "
            "is poor in every arm. The comparison between arms is the result, not the samples.",
            "Single seed per arm. Differences smaller than seed-to-seed variance are not "
            "interpretable, and seed variance was not measured.",
            "ends_mid_sentence is confounded by the fixed generation budget for any model that never "
            "emits an end-of-text token; n_words is reported so that case is visible.",
            "topic_overlap rewards verbatim copying and is used only to detect off-topic drift.",
            "The detectors are the same rules that constructed the degradations. That makes the "
            "metric exactly aligned with the training signal, which is the point, but it also means "
            "they cannot detect failures the degradations did not model.",
        ],
    }
    (run_dir / "result.json").write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")

    print("\n" + "=" * 92)
    hdr = f"{'arm':<18}{'pref_acc':>9}{'len-norm':>10}{'margin':>10}{'logratio':>10}{'rep':>7}{'midsent':>9}{'overlap':>9}{'ppl':>9}"
    print(hdr)
    print("-" * len(hdr))
    for row in table:
        print(f"{row['arm']:<18}{row['preference_accuracy']:>9.3f}"
              f"{row['preference_accuracy_length_normalised']:>10.3f}"
              f"{row['mean_implicit_reward_margin']:>+10.4f}"
              f"{row['mean_logratio_vs_reference']:>+10.2f}"
              f"{row['repetition_4gram']:>7.3f}{row['ends_mid_sentence']:>9.2f}"
              f"{row['topic_overlap']:>9.3f}{row['pretraining_val_perplexity']:>9.2f}")
    print("=" * 92)
    print(f"duration {duration / 60:.1f} min | written to {run_dir}")


if __name__ == "__main__":
    main()
