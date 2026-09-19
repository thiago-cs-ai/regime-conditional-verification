"""JSONL serialization and checksum helpers for corpus reconstruction.

Files serialize each row with ``json.dumps(row, ensure_ascii=False)`` followed by a newline. Frozen
file SHA-256 values depend on this representation.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any


class RebuildError(RuntimeError):
    """Raised when a rebuilder cannot write or verify an artifact."""


def sha256_file(path: Path | str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_jsonl(path: Path | str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(
    path: Path | str, rows: Iterable[Mapping[str, Any]], *, force: bool = False
) -> dict[str, Any]:
    """Write a nonempty JSONL file atomically and return its row count and SHA-256.

    Existing output requires ``force=True``.
    """
    path = Path(path)
    if path.exists() and not force:
        raise RebuildError(f"output file already exists; pass force=True to overwrite: {path}")
    rows = list(rows)
    if not rows:
        raise RebuildError(f"cannot write an empty JSONL file: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)
    return {"n": len(rows), "sha256": sha256_file(path)}


def enable_hf_offline() -> None:
    """Configure Hugging Face offline mode before importing ``datasets``."""
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    os.environ["HF_HUB_OFFLINE"] = "1"


def check_rebuild_outputs(
    out_dir: Path, relative_paths: Iterable[str], *, force: bool = False
) -> None:
    """Reject unexpected JSONL files or existing outputs before a multi-file rebuild writes."""
    expected = {out_dir / name for name in relative_paths}
    unexpected = sorted(set(out_dir.rglob("*.jsonl")) - expected)
    if unexpected:
        raise RebuildError(f"Unexpected corpus file: {unexpected[0]}. Use a fresh output directory.")
    existing = sorted(p for p in expected if p.exists())
    if existing and not force:
        raise RebuildError(f"output file already exists; pass force=True to overwrite: {existing[0]}")
