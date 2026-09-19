"""Build reproducible RCV frames from corpus rows and classifier representations."""

import hashlib
import os
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.lib import format as npy_format

BEAVER_COST_THRESHOLD = 3.0

PINNED_ARCHIVE_TIMESTAMP = (1980, 1, 1, 0, 0, 0)


@dataclass(frozen=True)
class FrameRow:
    """One corpus row used to build a frame."""
    item_id: str
    prompt: str
    verdict: int
    oracle: int
    seed: str | None = None
    human: int | None = None
    clf_score: float | None = None
    is_stream: bool = False


def beaver_verdict_from_costs(costs):
    """Classify Beaver costs at the fixed threshold of 3.0."""
    costs = np.asarray(costs, dtype=np.float64)
    if not np.isfinite(costs).all():
        raise ValueError(f"Beaver costs contain {int((~np.isfinite(costs)).sum())} non-finite "
                         "values.")
    return (costs >= BEAVER_COST_THRESHOLD).astype(np.int64)


def build_beaver_verdicts(costs, stored_verdict):
    """Rebuild Beaver verdicts from costs and validate the stored column."""
    stored = np.asarray(stored_verdict, dtype=np.float64)
    if not np.isin(stored, (0.0, 1.0)).all():
        raise ValueError("Stored Beaver verdicts must contain only 0 and 1.")
    costs = np.asarray(costs, dtype=np.float64)
    constructed = beaver_verdict_from_costs(costs)
    stored = np.asarray(stored_verdict, dtype=np.int64)
    if len(stored) != len(constructed):
        raise ValueError(f"Beaver costs and stored verdicts have different lengths: "
                         f"{len(constructed)} and {len(stored)}.")
    disagreeing = np.flatnonzero(constructed != stored)
    if len(disagreeing):
        first = int(disagreeing[0])
        raise ValueError(f"Stored Beaver verdicts disagree with cost >= 3.0 at "
                         f"{len(disagreeing)} of {len(stored)} rows; first row {first}: "
                         f"cost={float(costs[first])}, constructed={int(constructed[first])}, "
                         f"stored={int(stored[first])}.")
    return constructed


def normalised_key(text):
    """Normalize whitespace and case for grouping keys."""
    collapsed = " ".join(text.split()).casefold()
    if not collapsed:
        raise ValueError("Grouping key is empty.")
    return collapsed


def family_codes(prompt, seed):
    """Assign connected-family codes from normalized prompts and optional seeds."""
    if len(prompt) != len(seed):
        raise ValueError(f"Prompts and seeds have different lengths: {len(prompt)} and "
                         f"{len(seed)}.")

    parent = list(range(len(prompt)))

    def root(row):
        while parent[row] != row:
            row = parent[row]
        return row

    def union(row, other):
        first, second = root(row), root(other)
        if first != second:
            parent[max(first, second)] = min(first, second)

    first_row_holding = {}
    for row in range(len(prompt)):
        keys = [("prompt", normalised_key(prompt[row]))]
        if seed[row] is not None:
            keys.append(("seed", normalised_key(seed[row])))
        for key in keys:
            union(row, first_row_holding.setdefault(key, row))

    code_of_root = {}
    codes = np.empty(len(prompt), dtype=np.int64)
    for row in range(len(prompt)):
        codes[row] = code_of_root.setdefault(root(row), len(code_of_root))
    return codes


def build_frame(rows, representation_item_id, representation):
    """Join corpus rows to representations by item ID and return a sorted frame."""
    rows = list(rows)
    representation_item_id = np.asarray(representation_item_id)
    representation = np.asarray(representation)
    if representation.ndim != 2:
        raise ValueError(f"Representation must be two-dimensional; got shape "
                         f"{representation.shape}.")
    if len(representation_item_id) != len(representation):
        raise ValueError(f"Representation IDs and rows have different lengths: "
                         f"{len(representation_item_id)} and {len(representation)}.")

    ordered = sorted(rows, key=lambda row: row.item_id)
    item_id = np.array([row.item_id for row in ordered])

    _refuse_duplicates(representation_item_id, "representation item ids")
    _refuse_duplicates(item_id, "corpus rows")

    representation_row_of = {represented: row for row, represented
                            in enumerate(representation_item_id.tolist())}
    missing = [row.item_id for row in ordered if row.item_id not in representation_row_of]
    if missing:
        raise ValueError(f"{len(missing)} corpus rows are missing representations; first "
                         f"item ID {missing[0]!r}.")

    taken = [representation_row_of[row.item_id] for row in ordered]
    joined = np.ascontiguousarray(representation[taken])
    if not np.isfinite(joined).all():
        raise ValueError(f"Joined representation contains {int((~np.isfinite(joined)).sum())} "
                         "non-finite values.")

    frame = {
        "item_id": item_id,
        "representation": joined,
        "verdict": _binary_column([row.verdict for row in ordered], "verdict"),
        "oracle": _binary_column([row.oracle for row in ordered], "oracle"),
        "family": family_codes([row.prompt for row in ordered], [row.seed for row in ordered]),
        "is_stream": np.array([row.is_stream for row in ordered], dtype=np.bool_),
    }
    human = _human_column(ordered)
    if human is not None:
        frame["human"] = human
    clf_score = _clf_score_column(ordered)
    if clf_score is not None:
        frame["clf_score"] = clf_score
    return frame


def save_frame(frame, path):
    """Write a deterministic frame archive without overwriting an existing file."""
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite existing frame: {path}.")
    writing = path.with_name(path.name + ".writing")
    try:
        with zipfile.ZipFile(writing, "w", compression=zipfile.ZIP_STORED, allowZip64=True) as archive:
            for name in sorted(frame):
                member = zipfile.ZipInfo(f"{name}.npy", date_time=PINNED_ARCHIVE_TIMESTAMP)
                with archive.open(member, "w", force_zip64=True) as stream:
                    npy_format.write_array(stream, np.asanyarray(frame[name]), allow_pickle=False)
    except BaseException:
        writing.unlink(missing_ok=True)
        raise
    os.replace(writing, path)
    return digest_of(path)


def digest_of(path):
    with open(path, "rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def input_manifest(paths):
    """Return each input path's SHA-256 digest and byte count."""
    return {str(path): {"sha256": digest_of(path), "bytes": Path(path).stat().st_size}
            for path in paths}


def _refuse_duplicates(item_id, what):
    values, counts = np.unique(item_id, return_counts=True)
    repeated = values[counts > 1]
    if len(repeated):
        raise ValueError(f"{what} contain {len(repeated)} duplicate item IDs; first "
                         f"{str(repeated[0])!r}.")


def _binary_column(values, name):
    values = np.array(values, dtype=np.int64)
    outside = set(np.unique(values).tolist()) - {0, 1}
    if outside:
        raise ValueError(f"{name} must contain only 0 and 1; found {sorted(outside)}.")
    return values


def _clf_score_column(rows):
    present = [row.clf_score is not None for row in rows]
    if not any(present):
        return None
    if not all(present):
        raise ValueError(f"clf_score must be present for all rows or none; found "
                         f"{sum(present)} of {len(rows)}.")
    return np.array([row.clf_score for row in rows], dtype=np.float64)


def _human_column(rows):
    present = [row.human is not None for row in rows]
    if not any(present):
        return None
    if not all(present):
        raise ValueError(f"human must be present for all rows or none; found "
                         f"{sum(present)} of {len(rows)}.")
    return _binary_column([row.human for row in rows], "human")
