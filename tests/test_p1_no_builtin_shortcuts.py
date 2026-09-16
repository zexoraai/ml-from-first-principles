"""Enforce that Project 1 implements its mechanisms rather than calling PyTorch's versions.

The brief for Project 1 says: "Implement multi-head attention, positional encoding, layer
normalization, residual connections, feed-forward blocks, padding masks, and causal masks
directly. Do not use `torch.nn.Transformer` or built-in equivalents for the mechanisms being
demonstrated."

A promise in a README decays. This test makes the promise executable: if anyone (including a
future me, mid-refactor, reaching for a quick fix) swaps a hand-written mechanism for the library
version, the suite goes red and the portfolio claim stops being false.

Implemented with the `ast` module rather than a text search on purpose. This package's docstrings
*discuss* `torch.nn.MultiheadAttention` at length -- explaining what we did not use and why is
part of the teaching material. A grep would flag those sentences; parsing the syntax tree looks
only at code, so prose is free to name the forbidden APIs.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

PACKAGE_DIR = Path(__file__).resolve().parents[1] / "labs" / "p1_transformer"

# Attribute leaves that would mean a demonstrated mechanism was outsourced.
FORBIDDEN_LEAVES = {
    "Transformer",
    "TransformerEncoder",
    "TransformerDecoder",
    "TransformerEncoderLayer",
    "TransformerDecoderLayer",
    "MultiheadAttention",
    "LayerNorm",
    "scaled_dot_product_attention",
}

# Roots that indicate the leaf came from PyTorch rather than from this package. Our own
# `scaled_dot_product_attention` is called as a bare name or via a relative import, never through
# one of these, which is what keeps the check free of false positives.
TORCH_ROOTS = {"torch", "nn", "F", "functional"}


def _dotted(node: ast.AST) -> tuple[str, ...] | None:
    """Reduce an attribute chain such as `torch.nn.functional.foo` to ('torch','nn','functional','foo')."""
    parts: list[str] = []
    current: ast.AST = node
    while isinstance(current, ast.Attribute):
        parts.append(current.attr)
        current = current.value
    if not isinstance(current, ast.Name):
        return None
    parts.append(current.id)
    return tuple(reversed(parts))


def _source_files() -> list[Path]:
    files = sorted(PACKAGE_DIR.glob("*.py"))
    assert files, f"no source files found under {PACKAGE_DIR} -- this test would vacuously pass"
    return files


@pytest.mark.parametrize("path", _source_files(), ids=lambda p: p.name)
def test_no_builtin_transformer_mechanisms_are_called(path: Path) -> None:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    violations: list[str] = []

    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute):
            dotted = _dotted(node)
            if dotted and dotted[0] in TORCH_ROOTS and dotted[-1] in FORBIDDEN_LEAVES:
                violations.append(f"line {node.lineno}: {'.'.join(dotted)}")

        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if module.startswith("torch"):
                for alias in node.names:
                    if alias.name in FORBIDDEN_LEAVES:
                        violations.append(f"line {node.lineno}: from {module} import {alias.name}")

        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.split(".")[-1] in FORBIDDEN_LEAVES:
                    violations.append(f"line {node.lineno}: import {alias.name}")

    assert not violations, (
        f"{path.name} outsources a mechanism Project 1 claims to implement by hand:\n  "
        + "\n  ".join(violations)
    )


def test_the_detector_actually_detects() -> None:
    """A guard test for the guard test.

    A checker that never fires is indistinguishable from a checker that is broken. This feeds it a
    known violation and requires a hit, so the parametrised tests above mean something.
    """
    snippet = "import torch.nn as nn\nlayer = nn.MultiheadAttention(8, 2)\nn = nn.LayerNorm(8)\n"
    tree = ast.parse(snippet)
    hits = [
        ".".join(d)
        for node in ast.walk(tree)
        if isinstance(node, ast.Attribute)
        and (d := _dotted(node))
        and d[0] in TORCH_ROOTS
        and d[-1] in FORBIDDEN_LEAVES
    ]
    assert sorted(hits) == ["nn.LayerNorm", "nn.MultiheadAttention"]


def test_prose_mentioning_forbidden_apis_is_not_flagged() -> None:
    """Docstrings must stay free to explain what we deliberately avoided."""
    snippet = '"""We do not use torch.nn.MultiheadAttention or nn.LayerNorm here."""\nx = 1\n'
    tree = ast.parse(snippet)
    hits = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Attribute)
        and (d := _dotted(node))
        and d[0] in TORCH_ROOTS
        and d[-1] in FORBIDDEN_LEAVES
    ]
    assert hits == []
