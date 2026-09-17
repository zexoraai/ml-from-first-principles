"""Export PyTorch weights to a flat float32 buffer the browser can read.

Promoted here from `scripts/export_p1_web.py` once Project 2 needed the same thing (decision D-005:
promote on the second caller, not in anticipation of one). Two copies of a weight serialiser is how
two demos end up loading subtly different tensors.

FORMAT
------
    weights.bin   every tensor concatenated, float32, little-endian, no header
    manifest      name -> {offset (in floats), shape}

Little-endian float32 because that is what `Float32Array` reads natively on every platform a browser
runs on: no byte-swapping, no precision loss relative to the checkpoint, and the browser can map the
buffer directly.

Tied weights are stored **once**. A tied tensor appears under all of its names, with the later names
pointing at the same offset and flagged `shared_with_earlier_name`. This matters for more than file
size: if the export wrote two copies, a future change that updated one name and not the other would
produce a browser model whose embedding and output projection had silently drifted apart, which is
exactly the class of bug tying is supposed to make impossible.

Tensors are written in sorted-name order so the file is byte-reproducible from a given checkpoint,
which is what makes the published sha256 meaningful.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import torch
import torch.nn as nn

__all__ = ["export_weights", "MAX_DEMO_BYTES"]

# Decision D-002: a demo weight file above this is not shipped as live inference. A visitor on a
# phone should not be made to download more than this before anything happens on screen.
MAX_DEMO_BYTES = 25_000_000


def export_weights(model: nn.Module, out_dir: Path, *, filename: str = "weights.bin") -> dict:
    """Write every tensor of `model.state_dict()` into one buffer and describe where each one is.

    Returns the manifest fragment: tensor table, total float count, byte size, dtype, byte order and
    the sha256 of the buffer.
    """
    state = model.state_dict()
    buffers: list[bytes] = []
    tensors: dict[str, dict] = {}
    offset = 0
    seen: dict[int, str] = {}          # storage pointer -> the name that owns those bytes

    for name in sorted(state):
        tensor = state[name].detach().cpu().contiguous().to(torch.float32)
        ptr = tensor.data_ptr()
        if ptr in seen:
            owner = seen[ptr]
            tensors[name] = {
                "offset": tensors[owner]["offset"],
                "shape": list(tensor.shape),
                "shared_with_earlier_name": owner,
            }
            continue
        buffers.append(tensor.numpy().astype("<f4").tobytes())
        tensors[name] = {"offset": offset, "shape": list(tensor.shape)}
        seen[ptr] = name
        offset += tensor.numel()

    blob = b"".join(buffers)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / filename).write_bytes(blob)

    return {
        "file": filename,
        "tensors": tensors,
        "total_floats": offset,
        "bytes": len(blob),
        "dtype": "float32",
        "byte_order": "little-endian",
        "sha256": hashlib.sha256(blob).hexdigest(),
        "n_distinct_tensors": len(seen),
        "n_names": len(tensors),
    }


def check_demo_budget(meta: dict) -> None:
    """Refuse to ship an oversized demo rather than discovering it as a slow page."""
    if meta["bytes"] > MAX_DEMO_BYTES:
        raise SystemExit(
            f"weights are {meta['bytes'] / 1e6:.1f} MB, over the {MAX_DEMO_BYTES / 1e6:.0f} MB "
            f"demo budget (decision D-002). Either shrink the model or ship labelled cached "
            f"outputs instead of live inference."
        )
