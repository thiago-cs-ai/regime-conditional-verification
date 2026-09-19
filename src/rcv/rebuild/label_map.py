"""Export, validate, and apply label maps keyed by item ID.

Validation checks allowed fields, value formats, duplicate IDs, and free-text heuristics.
PKU label maps retain CC-BY-NC-4.0 terms; WildGuardMix label maps retain ODC-BY terms.
"""

from __future__ import annotations

import argparse
import json
import re
import time
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from ._jsonl import RebuildError, read_jsonl, write_jsonl

ALLOWED_COLUMNS: tuple[str, ...] = (
    "item_id",
    "new_label",  # WildGuardMix rubric label: 1=unsafe, 0=safe.
    "old_label",  # WildGuardMix human label: 1=unsafe, 0=safe.
    "rubric_ystar",  # PKU rubric label: 1=unsafe, 0=safe.
    "native_label",  # PKU human label: 1=unsafe, 0=safe.
    "native_is_safe",  # PKU human label in the upstream encoding: True=safe.
    "rule_fired",
    "group_key",
    "harm_categories",
    "severity",
)

# Reject these diagnostic fields explicitly when validating an exported map.
HARD_DROPPED_COLUMNS: tuple[str, ...] = ("raw", "reason", "reply", "rule_fired_raw")

PRIMARY_LABEL = {"native": "rubric_ystar", "wgmix": "new_label"}
_BINARY_LABEL_COLUMNS = ("new_label", "old_label", "rubric_ystar", "native_label")

_ITEM_ID_RE = re.compile(r"(it_[0-9a-f]{12}|PKU_test_[0-9]+_r[01])")
_GROUP_KEY_RE = re.compile(r"[0-9a-f]{16}")
_ENUM_RE = re.compile(r"[a-z0-9._\-]{1,40}")
_CATEGORY_RE = re.compile(r"[A-Za-z0-9\- ]{1,48}")

_FREE_TEXT_MAX_CHARS = 64
_FREE_TEXT_MAX_WORDS = 4


class LabelMapGateError(RebuildError):
    """A label map failed validation or did not cover the requested items."""


def looks_like_free_text(value: str) -> bool:
    """Apply a length, word-count, and newline heuristic to identify possible prose."""
    return (
        "\n" in value
        or len(value) > _FREE_TEXT_MAX_CHARS
        or len(value.split()) > _FREE_TEXT_MAX_WORDS
    )


def _check_binary(value: Any, col: str, row_i: int, errors: list[str]) -> None:
    if isinstance(value, bool) or value not in (0, 1):
        errors.append(f"row {row_i}: {col}={value!r} is not a 0/1 label (booleans excluded)")


def _validate_column(col: str, value: Any, row_i: int, errors: list[str]) -> None:
    if col == "item_id":
        if not (isinstance(value, str) and _ITEM_ID_RE.fullmatch(value)):
            errors.append(f"row {row_i}: item_id={value!r} has an unexpected shape")
    elif col in _BINARY_LABEL_COLUMNS:
        _check_binary(value, col, row_i, errors)
    elif col == "native_is_safe":
        if not isinstance(value, bool):
            errors.append(f"row {row_i}: native_is_safe={value!r} is not a bool")
    elif col == "rule_fired":
        if not (isinstance(value, str) and _ENUM_RE.fullmatch(value)):
            errors.append(f"row {row_i}: rule_fired={value!r} is not an enum-shaped tag")
    elif col == "group_key":
        if not (isinstance(value, str) and _GROUP_KEY_RE.fullmatch(value)):
            errors.append(f"row {row_i}: group_key={value!r} is not a 16-hex key")
    elif col == "harm_categories":
        if not isinstance(value, list):
            errors.append(f"row {row_i}: harm_categories={value!r} is not a list")
            return
        for cat in value:
            if not (
                isinstance(cat, str)
                and _CATEGORY_RE.fullmatch(cat)
                and not looks_like_free_text(cat)
            ):
                errors.append(f"row {row_i}: harm_categories entry {cat!r} fails format checks")
    elif col == "severity":
        if isinstance(value, bool) or not isinstance(value, int) or not (0 <= value <= 3):
            errors.append(f"row {row_i}: severity={value!r} is not an int in [0, 3]")


def gate_rows(rows: Sequence[Mapping[str, Any]], *, primary_label: str) -> dict[str, Any]:
    """Require a nonempty map with allowed fields, valid labels, and unique item IDs.

    ``primary_label`` must name a 0/1 label column. Return a report or raise
    ``LabelMapGateError``. Text checks use format and length heuristics.
    """
    if primary_label not in _BINARY_LABEL_COLUMNS:
        raise LabelMapGateError(f"primary_label {primary_label!r} must name a 0/1 label column")
    if not rows:
        raise LabelMapGateError("Label map is empty.")
    errors: list[str] = []
    seen_ids: set[str] = set()
    columns: set[str] = set()
    for i, row in enumerate(rows):
        for col, value in row.items():
            columns.add(col)
            if col in HARD_DROPPED_COLUMNS:
                errors.append(f"row {i}: excluded diagnostic column {col!r} present in export")
                continue
            if col not in ALLOWED_COLUMNS:
                errors.append(f"row {i}: column {col!r} is not allowlisted")
                continue
            _validate_column(col, value, i, errors)
            if isinstance(value, str) and looks_like_free_text(value):
                errors.append(f"row {i}: {col} value looks like free text")
        if primary_label not in row or row[primary_label] is None:
            errors.append(f"row {i}: primary label {primary_label!r} missing/null")
        iid = row.get("item_id")
        if isinstance(iid, str):
            if iid in seen_ids:
                errors.append(f"row {i}: duplicate item_id {iid}")
            seen_ids.add(iid)
        else:
            errors.append(f"row {i}: item_id must be a string")
        if len(errors) > 50:
            errors.append("... (further violations truncated)")
            break
    if errors:
        raise LabelMapGateError(
            "Label-map validation failed:\n  - "
            + "\n  - ".join(errors)
        )
    return {
        "gate": "column-allowlist+free-text-scan",
        "gate_pass": True,
        "n_rows": len(rows),
        "columns": sorted(columns),
        "primary_label": primary_label,
    }


def export_label_map(
    raw_rows: Iterable[Mapping[str, Any]], *, primary_label: str
) -> list[dict[str, Any]]:
    """Return validated rows containing only allowed fields.

    Skip rows with missing or null primary labels. For each item ID, keep the last
    remaining row; fields from earlier rows are not merged.
    """
    by_id: dict[str, dict[str, Any]] = {}
    for row in raw_rows:
        iid = row.get("item_id")
        if not isinstance(iid, str):
            raise LabelMapGateError(f"source row requires a string item_id: keys={sorted(row)}")
        if row.get(primary_label) is None:
            continue
        exported = {col: row[col] for col in ALLOWED_COLUMNS if col in row}
        by_id[iid] = exported
    out = list(by_id.values())
    gate_rows(out, primary_label=primary_label)
    return out


def apply_label_map(
    item_rows: Sequence[Mapping[str, Any]],
    label_rows: Sequence[Mapping[str, Any]],
    *,
    primary_label: str,
) -> list[dict[str, Any]]:
    """Validate label rows and merge their fields into copies of item rows by ``item_id``.

    Require coverage for every item. Label-map fields overwrite existing item fields.
    """
    gate_rows(label_rows, primary_label=primary_label)
    labels_by_id = {r["item_id"]: r for r in label_rows}
    missing: list[str] = []
    out: list[dict[str, Any]] = []
    for row in item_rows:
        iid = row["item_id"]
        lab = labels_by_id.get(iid)
        if lab is None:
            missing.append(iid)
            continue
        merged = dict(row)
        merged.update({k: v for k, v in lab.items() if k != "item_id"})
        out.append(merged)
    if missing:
        raise LabelMapGateError(
            f"label map does not cover {len(missing)} item(s); first: {missing[:5]}"
        )
    return out


def _write_report(path: str | None, report: dict[str, Any]) -> None:
    if path is None:
        return
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(report, indent=2) + "\n")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Export, validate, or apply a label map.")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_exp = sub.add_parser("export", help="filter labeled JSONL rows to allowed fields and validate")
    p_exp.add_argument("--src", required=True, nargs="+", help="source jsonl file(s)")
    p_exp.add_argument("--out", required=True, help="exported label-map jsonl")
    p_exp.add_argument("--corpus", required=True, choices=sorted(PRIMARY_LABEL))
    p_exp.add_argument("--report", default=None, help="optional gate-report JSON")
    p_exp.add_argument("--force", action="store_true")

    p_gate = sub.add_parser("gate", help="validate an exported/downloaded label map")
    p_gate.add_argument("--path", required=True, help="label-map jsonl to gate")
    p_gate.add_argument("--corpus", required=True, choices=sorted(PRIMARY_LABEL))
    p_gate.add_argument("--report", default=None, help="optional gate-report JSON")

    p_apply = sub.add_parser("apply", help="join a gated label map onto rebuilt corpus rows")
    p_apply.add_argument("--labels", required=True, help="exported label-map jsonl")
    p_apply.add_argument("--items", required=True, nargs="+", help="rebuilt corpus jsonl file(s)")
    p_apply.add_argument("--out", required=True, help="output JSONL containing corpus text and labels")
    p_apply.add_argument("--corpus", required=True, choices=sorted(PRIMARY_LABEL))
    p_apply.add_argument("--force", action="store_true")

    args = parser.parse_args(argv)
    primary = PRIMARY_LABEL[args.corpus]

    if args.cmd == "export":
        raw: list[dict[str, Any]] = []
        for src in args.src:
            raw.extend(read_jsonl(src))
        exported = export_label_map(raw, primary_label=primary)
        written = write_jsonl(args.out, exported, force=args.force)
        report = gate_rows(exported, primary_label=primary)
        report.update(
            {
                "component": "rebuilders",
                "artifact": str(args.out),
                "corpus": args.corpus,
                "export_sha256": written["sha256"],
                "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            }
        )
        _write_report(args.report, report)
        print(
            f"[rcv.rebuild.label_map] exported {written['n']} rows -> {args.out} "
            f"(sha256 {written['sha256'][:12]}…); validation passed"
        )
        return 0

    if args.cmd == "gate":
        rows = read_jsonl(args.path)
        report = gate_rows(rows, primary_label=primary)
        from ._jsonl import sha256_file

        report.update(
            {
                "component": "rebuilders",
                "artifact": str(args.path),
                "corpus": args.corpus,
                "map_sha256": sha256_file(args.path),
                "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            }
        )
        _write_report(args.report, report)
        print(f"[rcv.rebuild.label_map] validation passed for {report['n_rows']} rows ({args.path})")
        return 0

    items: list[dict[str, Any]] = []
    for f in args.items:
        items.extend(read_jsonl(f))
    labels = read_jsonl(args.labels)
    merged = apply_label_map(items, labels, primary_label=primary)
    written = write_jsonl(args.out, merged, force=args.force)
    print(f"[rcv.rebuild.label_map] applied labels to {written['n']} rows -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
