"""Export a trained Project 2 GPT for in-browser inference, with a parity fixture.

    .\\run.cmd python scripts/export_p2_web.py \\
        --checkpoint runs/p2-shake-3m/checkpoint_best.pt \\
        --result runs/p2-shake-3m/result.json

Writes into `docs/assets/models/p2/`: weights.bin, manifest.json, tokenizer.json, parity.json,
results.json.

WHAT THE PARITY FIXTURE CAN AND CANNOT CHECK
--------------------------------------------
Sampling is stochastic and JavaScript's PRNG cannot reproduce `torch.multinomial`, so **sampled text
is not compared** — that would be a meaningless test that either always fails or is rigged.

What is compared is everything deterministic:
  * the raw logits for fixed prompts (isolates every weight, every layer, the KV cache, the
    positional offsets, GELU, and the tied output projection),
  * the argmax at the first step,
  * a full greedy continuation, which compounds any drift over many steps and is where a subtle
    cache or position bug shows up.

If the logits match and greedy continuations match token-for-token, the two implementations compute
the same distribution; only the coin flips differ. The project page states this boundary explicitly.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from labs.common.webexport import check_demo_budget, export_weights  # noqa: E402
from labs.p2_gpt import GPT, GPTConfig, ByteBPETokenizer  # noqa: E402

PARITY_PROMPTS = [
    "\n",
    "ROMEO:",
    "First Citizen:\nBefore we proceed",
    "\nKING RICHARD III:\nWhat",
    "To be, or not to be",
]


@torch.no_grad()
def build_parity(model: GPT, tok: ByteBPETokenizer, prompts: list[str], n_greedy: int = 24) -> dict:
    model.eval()
    cases = []
    for text in prompts:
        ids = tok.encode(text) or tok.encode("\n")
        ids = ids[-(model.cfg.block_size - n_greedy - 1):]
        idx = torch.tensor([ids], dtype=torch.long)

        logits, _ = model(idx)
        last = logits[0, -1]

        # Greedy continuation via the cached path -- the same path the browser uses.
        past = None
        feed = idx
        greedy: list[int] = []
        for _ in range(n_greedy):
            lg, past = model.forward_cached(feed, past)
            nxt = int(lg[0, -1].argmax())
            greedy.append(nxt)
            feed = torch.tensor([[nxt]], dtype=torch.long)

        cases.append({
            "prompt": text,
            "prompt_ids": ids,
            "logits_head": [round(float(v), 6) for v in last[:16]],
            "argmax": int(last.argmax()),
            "greedy_ids": greedy,
            "greedy_text": tok.decode(greedy),
        })

    return {
        "tolerance": {
            "logits_abs": 1e-3,
            "note": "float32 accumulation order differs between PyTorch's BLAS and a naive JS "
                    "triple loop, and GELU is evaluated via an erf approximation in JS, so exact "
                    "equality is not expected. 1e-3 is far tighter than any logic error while "
                    "absorbing both effects.",
            "not_compared": "sampled text. JavaScript's PRNG cannot reproduce torch.multinomial, so "
                            "only deterministic quantities (logits, argmax, greedy continuations) "
                            "are checked.",
        },
        "cases": cases,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--result", default=None)
    ap.add_argument("--out", default="docs/assets/models/p2")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    ck = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg = GPTConfig(**ck["config"])
    model = GPT(cfg)
    model.load_state_dict(ck["model"])
    model.eval()

    run_dir = Path(args.checkpoint).parent
    tok_path = run_dir.parent.parent / "data" / "p2" / f"tokenizer_{cfg.vocab_size}.json"
    if not tok_path.exists():
        tok_path = Path("data/p2") / f"tokenizer_{cfg.vocab_size}.json"
    tok = ByteBPETokenizer.load(tok_path)
    if tok.vocab_size != cfg.vocab_size:
        raise SystemExit(
            f"tokenizer vocab {tok.vocab_size} != checkpoint vocab {cfg.vocab_size}. "
            f"These must match exactly or every token id is wrong."
        )

    weights = export_weights(model, out_dir)
    check_demo_budget(weights)

    parity = build_parity(model, tok, PARITY_PROMPTS)

    manifest = {
        "project": "p2_gpt",
        "paper": "Radford et al., GPT-2 (2019)",
        "source_run": ck.get("run_id", "UNKNOWN"),
        "checkpoint_step": ck.get("step"),
        "config": cfg.to_dict(),
        "n_parameters": model.num_parameters(),
        "n_parameters_non_embedding": model.num_parameters(non_embedding=True),
        "weights": weights,
        "engine": "hand-written JS forward pass with a KV cache (decision D-003), verified against "
                  "parity.json by scripts/verify_js_parity.mjs",
        "flop_estimate_per_token": model.estimate_flops_per_token(),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (out_dir / "tokenizer.json").write_text(tok.to_web_json(), encoding="utf-8")
    (out_dir / "parity.json").write_text(json.dumps(parity, indent=2), encoding="utf-8")

    if args.result:
        full = json.loads(Path(args.result).read_text(encoding="utf-8"))
        keep = ("run_id", "source_commit", "timestamp_utc", "n_parameters",
                "n_parameters_non_embedding", "steps_completed", "tokens_seen", "duration_s",
                "seeds", "data", "final_metrics", "metric_definitions", "best_val_loss",
                "limitations", "model_config", "repro_cmd", "checkpoints", "samples",
                "optimizer", "tokens_per_step", "flop_estimate_per_token")
        trimmed = {k: full[k] for k in keep if k in full}
        trimmed["history"] = [
            {k: v for k, v in h.items()
             if k in ("step", "train_loss", "lr", "eval_train_loss", "eval_val_loss",
                      "val_perplexity")}
            for h in full["history"]
        ]
        (out_dir / "results.json").write_text(json.dumps(trimmed, indent=2), encoding="utf-8")
        m = trimmed["final_metrics"]
        print(f"published results.json  val loss {m['val_loss']:.4f}  ppl {m['val_perplexity']:.2f}")

    print(f"exported {model.num_parameters():,} parameters "
          f"({weights['bytes'] / 1e6:.2f} MB, {weights['n_distinct_tensors']} distinct tensors "
          f"under {weights['n_names']} names) to {out_dir}")
    print(f"weights sha256 {weights['sha256'][:16]}...")
    print(f"parity cases: {len(parity['cases'])}")
    for c in parity["cases"]:
        preview = c["greedy_text"].replace("\n", " | ")[:70]
        print(f"  {c['prompt']!r:38} -> {preview!r}")


if __name__ == "__main__":
    main()
