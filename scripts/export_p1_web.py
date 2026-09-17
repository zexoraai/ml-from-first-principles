"""Export a trained Project 1 checkpoint for in-browser inference, plus a parity fixture.

Writes into `docs/assets/models/p1/`:

    weights.bin      all tensors concatenated, float32 little-endian
    manifest.json    tensor name -> {offset, shape}, plus the model config
    tokenizer.json   the exact vocabulary used in training
    parity.json      fixed inputs with PyTorch's outputs, to prove the JS engine agrees

WHY A PARITY FIXTURE IS NOT OPTIONAL
------------------------------------
The demo re-implements the forward pass in JavaScript (decision D-003) so the page can expose
attention heads, tensor shapes and per-step probabilities. A re-implementation that quietly
disagrees with the trained model turns "genuine model output" into a fabrication, which is exactly
what this portfolio promises not to do. So the export ships reference outputs, the browser checks
itself against them on load, and the measured deviation is displayed on the page. If the check
fails the demo says so rather than showing plausible-looking numbers.

Float32 little-endian is chosen because that is what `DataView.getFloat32(offset, true)` and
`Float32Array` read natively on every platform browsers run on, so no conversion is needed and no
precision is lost relative to the PyTorch weights.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from labs.p1_transformer import Transformer, TransformerConfig, decode_step_trace  # noqa: E402
from labs.p1_transformer.data import CharTokenizer  # noqa: E402


def export_weights(model: Transformer, out_dir: Path) -> dict:
    """Concatenate every tensor into one buffer and record where each one starts.

    Tensors are written in sorted-name order so the file is byte-reproducible from a checkpoint.
    Tied weights appear under both names but are stored **once**, with the second name pointing at
    the same offset -- so the browser sees the tying too rather than loading a stale copy.
    """
    state = model.state_dict()
    buffers: list[bytes] = []
    manifest_tensors: dict[str, dict] = {}
    offset = 0
    seen: dict[int, int] = {}   # id(tensor storage) -> offset, so tied weights are stored once

    for name in sorted(state):
        tensor = state[name].detach().cpu().contiguous().to(torch.float32)
        key = tensor.data_ptr()
        if key in seen:
            manifest_tensors[name] = {
                "offset": seen[key], "shape": list(tensor.shape), "shared_with_earlier_name": True
            }
            continue
        raw = tensor.numpy().astype("<f4").tobytes()
        buffers.append(raw)
        manifest_tensors[name] = {"offset": offset, "shape": list(tensor.shape)}
        seen[key] = offset
        offset += tensor.numel()

    blob = b"".join(buffers)
    (out_dir / "weights.bin").write_bytes(blob)
    return {
        "tensors": manifest_tensors,
        "total_floats": offset,
        "bytes": len(blob),
        "dtype": "float32",
        "byte_order": "little-endian",
        "sha256": hashlib.sha256(blob).hexdigest(),
    }


@torch.no_grad()
def build_parity_fixture(model: Transformer, tok: CharTokenizer, sources: list[str]) -> dict:
    """Reference outputs the JS engine must reproduce.

    Includes three levels so a failure localises itself instead of just saying "wrong":
      * `encoder_output_head` -- first few values of the encoder memory: isolates embedding,
        positional encoding, encoder attention and normalisation.
      * `first_step_logits` -- the decoder's logits at step 0 from BOS alone: adds cross-attention
        and the output projection.
      * `greedy_trace` -- the full decode with per-step top-k: catches any drift that only appears
        after several autoregressive steps.
    """
    model.eval()
    cases = []
    for text in sources:
        src = torch.tensor([tok.encode(text)], dtype=torch.long)
        memory = model.encode(src)
        bos = torch.tensor([[tok.bos_id]], dtype=torch.long)
        hidden = model.decoder(
            bos, memory,
            self_keep_mask=model.target_keep_mask(bos),
            cross_keep_mask=model.source_keep_mask(src),
        )
        logits = model.generator(hidden[:, -1])[0]
        trace = decode_step_trace(model, src, max_new_tokens=12, top_k=3)
        decoded = "".join(
            tok.itos[s["chosen"]] for s in trace if s["chosen"] not in (tok.bos_id, tok.eos_id, tok.pad_id)
        )
        cases.append({
            "source": text,
            "src_ids": tok.encode(text),
            "encoder_output_head": [round(float(v), 6) for v in memory[0, 0, :8]],
            "encoder_output_sum": round(float(memory.sum()), 4),
            "first_step_logits_head": [round(float(v), 6) for v in logits[:8]],
            "first_step_argmax": int(logits.argmax()),
            "greedy_output": decoded,
            "greedy_chosen_ids": [s["chosen"] for s in trace],
            "step_entropies": [round(s["entropy_nats"], 5) for s in trace],
        })
    return {
        "tolerance": {
            # Measured deviations on the first verified export (scripts/verify_js_parity.mjs,
            # Node 24 on this CPU) were 5.26e-7 for the encoder and 3.58e-6 for the logits. The
            # bounds below sit roughly 30-200x above that: tight enough that any real logic error
            # fails immediately, loose enough to absorb float32 accumulation-order differences
            # between PyTorch's BLAS and a naive JS matmul across browsers and CPU SIMD widths.
            # The first version of this file used 2e-3 / 5e-3, which passed but was ~1000x looser
            # than the truth and therefore not a meaningful check.
            "encoder_output_abs": 1e-4,
            "logits_abs": 1e-4,
            "note": "exact equality is not expected: PyTorch's BLAS and a naive JS triple loop "
                    "accumulate in different orders. Observed deviation on the reference export was "
                    "5.3e-7 (encoder) and 3.6e-6 (logits), so these bounds carry a 30-200x margin.",
            "measured_on_reference_export": {
                "encoder_max_abs": 5.259e-7,
                "logits_max_abs": 3.580e-6,
                "harness": "scripts/verify_js_parity.mjs, Node 24, AMD Ryzen 5 PRO 5650U",
                "all_cases_string_identical": True,
            },
        },
        "cases": cases,
    }


@torch.no_grad()
def scan_test_failures(model: Transformer, tok: CharTokenizer, limit: int = 40) -> dict:
    """Find the model's *actual* failures across the whole test split.

    Why this is a separate pass rather than a sample: at 99.9% exact-match, a 24-example sample
    contains zero failures, and "here are 24 correct outputs" is not a failure analysis. The brief
    asks for failure examples, so we go and find the real ones instead of reporting that we could
    not find any in a sample too small to contain them.

    Returns the failures found, plus the counts needed to state the rate honestly.
    """
    from labs.p1_transformer.data import (  # noqa: PLC0415
        DateDataset, DateTaskConfig, build_splits, collate,
    )
    from labs.p1_transformer import greedy_decode  # noqa: PLC0415

    cfg = DateTaskConfig()
    splits = build_splits(cfg)
    ds = DateDataset(splits["test"], tok, cfg)
    loader = torch.utils.data.DataLoader(
        ds, batch_size=64, shuffle=False, collate_fn=lambda b: collate(b, tok.pad_id)
    )

    model.eval()
    failures: list[dict] = []
    total = 0
    correct = 0
    for batch in loader:
        decoded = greedy_decode(model, batch["src"], max_new_tokens=12)
        for row, label_row, idx in zip(decoded, batch["labels"], batch["index"]):
            got = tok.decode(row.tolist())
            want = tok.decode(label_row.tolist())
            total += 1
            if got == want:
                correct += 1
            elif len(failures) < limit:
                src_text, _ = ds.pairs[int(idx)]
                # Character-level diff, so the page can say *how* it was wrong, not just that it was.
                diff = [i for i in range(max(len(got), len(want)))
                        if got[i:i + 1] != want[i:i + 1]]
                failures.append({
                    "source": src_text, "target": want, "prediction": got,
                    "correct": False,
                    "wrong_positions": diff,
                    "n_wrong_chars": len(diff),
                })

    return {
        "n_test": total,
        "n_correct": correct,
        "n_failures": total - correct,
        "exact_match": correct / max(total, 1),
        "failures": failures,
        "note": "every failure below is a real held-out test example, found by scanning the full "
                "split rather than sampling",
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out", default="docs/assets/models/p1")
    ap.add_argument("--result", default=None,
                    help="path to the run's result.json; a trimmed copy is published so the "
                         "project page reads its numbers from evidence instead of hard-coding them")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg = TransformerConfig(**ckpt["config"])
    model = Transformer(cfg)
    model.load_state_dict(ckpt["model"])
    model.eval()

    tok = CharTokenizer()
    if len(tok) != cfg.vocab_size:
        raise SystemExit(f"tokenizer size {len(tok)} != checkpoint vocab_size {cfg.vocab_size}")

    weights_meta = export_weights(model, out_dir)

    demo_sources = [
        "March 3, 2019",
        "3 Mar 1987",
        "Sunday, September 22, 2001",
        "22nd December 1999",
        "2035 July 4",
    ]
    parity = build_parity_fixture(model, tok, demo_sources)

    manifest = {
        "project": "p1_transformer",
        "paper": "arXiv:1706.03762v7",
        "source_run": ckpt.get("run_id", "UNKNOWN"),
        "checkpoint_step": ckpt.get("step"),
        "config": cfg.to_dict(),
        "n_parameters": model.num_parameters(),
        "weights": weights_meta,
        "engine": "hand-written JS forward pass (decision D-003), verified against parity.json",
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (out_dir / "tokenizer.json").write_text(tok.to_json(), encoding="utf-8")
    (out_dir / "parity.json").write_text(json.dumps(parity, indent=2), encoding="utf-8")

    # Publish a trimmed results file for the project page. Only fields the page renders are copied;
    # the full record (including the pip freeze and every logged step) stays in the run directory and
    # in records/EXPERIMENTS.md. The page reads THIS file rather than having numbers typed into it,
    # so a figure on screen cannot drift from the run that produced it.
    if args.result:
        full = json.loads(Path(args.result).read_text(encoding="utf-8"))
        trimmed = {
            "run_id": full["run_id"],
            "source_commit": full["source_commit"],
            "timestamp_utc": full["timestamp_utc"],
            "n_parameters": full["n_parameters"],
            "steps_completed": full["steps_completed"],
            "tokens_seen": full["tokens_seen"],
            "duration_s": full["duration_s"],
            "seeds": full["seeds"],
            "split_sizes": full["split_sizes"],
            "unique_dates": full["unique_dates"],
            "final_metrics": full["final_metrics"],
            "metric_definitions": full["metric_definitions"],
            "sample_predictions": full["sample_predictions"],
            "limitations": full["limitations"],
            "model_config": full["model_config"],
            "repro_cmd": full["repro_cmd"],
            "checkpoints": full.get("checkpoints", {}),
            "failure_scan": scan_test_failures(model, tok),
            # keep only the loss/metric series the curve needs, not the whole history
            "history": [
                {k: v for k, v in h.items()
                 if k in ("step", "train_loss_smoothed", "lr",
                          "val_loss_nats_per_token", "token_accuracy_teacher_forced",
                          "exact_match_free_running")}
                for h in full["history"]
            ],
        }
        (out_dir / "results.json").write_text(json.dumps(trimmed, indent=2), encoding="utf-8")
        m = trimmed["final_metrics"]
        print(f"published results.json  test exact-match "
              f"{m['test']['exact_match_free_running'] * 100:.2f}%")

    print(f"exported {model.num_parameters():,} parameters "
          f"({weights_meta['bytes'] / 1e6:.2f} MB) to {out_dir}")
    print(f"weights sha256 {weights_meta['sha256'][:16]}...")
    print(f"parity cases: {len(parity['cases'])}")
    for c in parity["cases"]:
        print(f"  {c['source']!r} -> {c['greedy_output']!r}")

    if weights_meta["bytes"] > 25_000_000:
        raise SystemExit("weights exceed the 25 MB demo budget (decision D-002)")


if __name__ == "__main__":
    main()
