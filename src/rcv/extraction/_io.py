"""Chunked JSONL and NumPy I/O for extraction artifacts."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path

import numpy as np


def read_jsonl(path, max_rows=None):
    rows = []
    with open(path, encoding="utf-8") as fh:
        for i, line in enumerate(fh):
            if max_rows is not None and i >= max_rows:
                break
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def validate_item_ids(rows) -> None:
    seen = set()
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError(f"Row {index} must be a JSON object.")
        iid = row.get("item_id")
        if isinstance(iid, bool) or not isinstance(iid, (str, int)) or str(iid) == "":
            raise ValueError(f"Row {index} requires a nonempty string or integer item_id.")
        if str(iid) in seen:
            raise ValueError(f"Duplicate item_id {iid!r} at row {index}.")
        seen.add(str(iid))


def validate_labeled_rows(rows) -> None:
    """Require unique item IDs, text fields, and binary target labels."""
    validate_item_ids(rows)
    for index, row in enumerate(rows):
        for field in ("prompt", "response"):
            if not isinstance(row.get(field), str):
                raise ValueError(f"Row {index} requires a string {field}.")
        if row.get("ground_truth") not in (0, 1):
            raise ValueError(f"Row {index} requires ground_truth 0 or 1.")


def binary_vector(values, n: int, name: str):
    arr = np.asarray(values)
    if arr.shape != (n,) or not np.isin(arr, (0, 1)).all():
        raise AssertionError(f"{name} must be a binary array of shape ({n},); got {arr.shape}.")
    return arr.astype(np.int64)


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    os.replace(tmp, path)


def sha_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def flush_chunk(emb_path: Path, out_jsonl: Path, rows, vecs, idx: int) -> tuple[Path, Path, tuple[int, ...]]:
    """Write a JSONL/vector chunk and return its paths and array shape."""
    cj = out_jsonl.with_suffix(f".chunk{idx:04d}.jsonl")
    cn = emb_path.with_suffix(f".chunk{idx:04d}.npy")
    arr = np.stack(vecs, axis=0).astype(np.float32)
    if arr.ndim != 2 or len(rows) != arr.shape[0]:
        raise ValueError(f"Chunk {idx}: {len(rows)} rows do not match embedding shape {arr.shape}.")
    write_jsonl(cj, rows)
    cn.parent.mkdir(parents=True, exist_ok=True)
    np.save(cn, arr)
    return cj, cn, arr.shape


def _chunk_paths(path: Path, suffix: str) -> dict[int, Path]:
    prefix = path.with_suffix(".chunk").name
    chunks = {}
    for candidate in path.parent.glob("*"):
        name = candidate.name
        if not (name.startswith(prefix) and name.endswith(suffix)):
            continue
        index = name[len(prefix):-len(suffix)]
        if not index.isdecimal() or int(index) in chunks:
            raise ValueError(f"Invalid or duplicate chunk index: {candidate}")
        chunks[int(index)] = candidate
    return chunks


def check_extraction_start(emb_path: Path, out_jsonl: Path, n_rows: int) -> None:
    if n_rows == 0:
        raise ValueError("Extraction input is empty.")
    chunks = list(_chunk_paths(out_jsonl, ".jsonl").values()) + list(_chunk_paths(emb_path, ".npy").values())
    if chunks:
        raise ValueError(f"Existing extraction chunks: {chunks[0]}. Use new output paths.")


def stitch_chunks(emb_path: Path, out_jsonl: Path, *, expected_rows: int):
    """Validate chunk pairs and the input row count before replacing outputs."""
    cjs = _chunk_paths(out_jsonl, ".jsonl")
    cns = _chunk_paths(emb_path, ".npy")
    if not cjs or cjs.keys() != cns.keys():
        raise ValueError("JSONL and NumPy chunks must have matching, nonempty index sets.")
    arrays = []
    nrow = 0
    for idx in sorted(cjs):
        with open(cjs[idx], encoding="utf-8") as source:
            count = sum(1 for _ in source)
        arr = np.load(cns[idx], allow_pickle=False)
        if arr.ndim != 2 or arr.dtype != np.float32 or arr.shape[0] != count:
            raise ValueError(f"Chunk {idx}: {count} rows require a matching 2-D float32 array; got {arr.shape}, {arr.dtype}.")
        arrays.append(arr)
        nrow += count
    if nrow != expected_rows:
        raise ValueError(f"Chunk rows ({nrow}) do not match input rows ({expected_rows}).")
    full = np.concatenate(arrays, axis=0)

    with (
        tempfile.TemporaryDirectory(dir=out_jsonl.parent) as jd,
        tempfile.TemporaryDirectory(dir=emb_path.parent) as nd,
    ):
        tmp_jsonl, tmp_npy = Path(jd) / "items.jsonl", Path(nd) / "Z.npy"
        with open(tmp_jsonl, "wb") as output:
            for idx in sorted(cjs):
                with open(cjs[idx], "rb") as source:
                    shutil.copyfileobj(source, output)
        np.save(tmp_npy, full)
        os.replace(tmp_jsonl, out_jsonl)
        os.replace(tmp_npy, emb_path)
    for path in [*cjs.values(), *cns.values()]:
        path.unlink()
    return nrow, full.shape
