"""Reconstruct PKU-SafeRLHF extraction input and human-label metadata from its test split.

The reference frame contains both responses to 8,211 prompts: 16,422 rows. Sidecar
``y_star_native`` labels encode the dataset's human judgments as 1=unsafe, 0=safe.

Dataset license: CC-BY-NC-4.0 (Ji et al., 2024, PKU-Alignment). Dataset content is not
redistributed here; use of the rebuilt frame remains subject to CC-BY-NC-4.0.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from ._jsonl import RebuildError, check_rebuild_outputs, enable_hf_offline, write_jsonl

# Upstream revision used to build the reference frame.
PINNED_PKU_REVISION = "9421ffafec3fa40a1f1a7d567b4d525079477ecb"

EXPECTED_N_PROMPTS = 8211
EXPECTED_N_ROWS = 16422

# SHA-256 values of the reference input and sidecar files.
EXPECTED_SHA256 = {
    "input": "e72794476bcc0c915daab17c06758b4f1217c4ae824215d36de51bea85724f31",
    "sidecar": "a1cee99e3ee0a74948b0299642715c600541e3fd87a05f5767b0bfa567734d20",
}


def group_key(prompt: str) -> str:
    """Return a prompt-derived key for grouping its two response rows."""
    return hashlib.md5(prompt.encode("utf-8")).hexdigest()[:16]


def build_frame(
    ds_rows: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Return ``(input_rows, sidecar_rows, summary)`` in upstream prompt and response order.

    Both responses are retained, including blanks; None becomes an empty string.
    Row order determines item IDs, and field order affects serialized checksums.
    """
    n_prompts = len(ds_rows)
    input_rows: list[dict[str, Any]] = []
    sidecar_rows: list[dict[str, Any]] = []
    n_unsafe = 0
    n_empty_resp = 0

    for i in range(n_prompts):
        r = ds_rows[i]
        prompt = r["prompt"]
        gk = group_key(prompt)
        for ridx in (0, 1):
            response = r[f"response_{ridx}"]
            is_safe = bool(r[f"is_response_{ridx}_safe"])
            y_star = int(not is_safe)
            n_unsafe += y_star
            if response is None or len(response.strip()) == 0:
                n_empty_resp += 1
                response = response or ""
            item_id = f"PKU_test_{i}_r{ridx}"
            input_rows.append({"item_id": item_id, "prompt": prompt, "response": response})
            harm = r[f"response_{ridx}_harm_category"]
            active_cats = sorted([k for k, v in harm.items() if v]) if isinstance(harm, dict) else []
            sidecar_rows.append(
                {
                    "item_id": item_id,
                    "y_star_native": y_star,
                    "is_safe": is_safe,
                    "harm_categories": active_cats,
                    "severity": int(r[f"response_{ridx}_severity_level"]),
                    "response_idx": ridx,
                    "group_key": gk,
                    "prompt_source": r.get("prompt_source"),
                    "response_source": r.get(f"response_{ridx}_source"),
                    "response_sha256": r.get(f"response_{ridx}_sha256"),
                }
            )

    summary = {
        "dataset": "PKU-Alignment/PKU-SafeRLHF",
        "split": "test",
        "n_prompts": n_prompts,
        "n_rows_both_responses": len(input_rows),
        "n_native_unsafe": n_unsafe,
        "native_unsafe_rate": round(n_unsafe / len(input_rows), 6) if input_rows else None,
        "n_empty_responses": n_empty_resp,
        "polarity": "1=UNSAFE,0=SAFE  (Y*_native = int(not is_response_n_safe))",
        "group_key": "md5(prompt)[:16]  (both responses of a prompt share a group)",
    }
    return input_rows, sidecar_rows, summary


REQUIRED_COLUMNS = (
    "prompt",
    "response_0",
    "response_1",
    "is_response_0_safe",
    "is_response_1_safe",
    "response_0_harm_category",
    "response_1_harm_category",
    "response_0_severity_level",
    "response_1_severity_level",
)


def _validate_test_split(ds: Any, revision: str) -> Any:
    """Check required columns and the reference prompt count."""
    missing = [c for c in REQUIRED_COLUMNS if c not in ds.column_names]
    if missing or len(ds) != EXPECTED_N_PROMPTS:
        raise RebuildError(
            f"PKU-SafeRLHF test split failed validation for requested revision {revision[:12]}: "
            f"{len(ds)} prompts (expected {EXPECTED_N_PROMPTS}); missing columns: {missing}."
        )
    return ds


def _pinned_processed_cache(revision: str) -> Path:
    """Locate the revision's Arrow file in the expected datasets cache layout."""
    import os

    cache_root = Path(
        os.environ.get("HF_DATASETS_CACHE", Path.home() / ".cache" / "huggingface" / "datasets")
    )
    return (
        cache_root
        / "PKU-Alignment___pku-safe_rlhf"
        / "default"
        / "0.0.0"
        / revision
        / "pku-safe_rlhf-test.arrow"
    )


def load_pku_test_split(revision: str = PINNED_PKU_REVISION) -> Any:
    """Request the PKU test split and validate its schema and row count.

    If validation fails, try the processed cache for ``revision``. Requires the ``rebuild``
    extra; for cached data only, call ``enable_hf_offline`` before importing ``datasets``.
    """
    try:
        from datasets import Dataset, load_dataset
    except ImportError as exc:  # pragma: no cover - environment guard
        raise RebuildError(
            "Missing dependency 'datasets'. "
            "Install with: pip install 'regime-conditional-verification[rebuild]'"
        ) from exc
    try:
        ds = load_dataset("PKU-Alignment/PKU-SafeRLHF", split="test", revision=revision)
        return _validate_test_split(ds, revision)
    except RebuildError:
        arrow = _pinned_processed_cache(revision)
        if arrow.is_file():
            return _validate_test_split(Dataset.from_file(str(arrow)), revision)
        raise


def rebuild(
    out_dir: Path | str,
    *,
    revision: str = PINNED_PKU_REVISION,
    force: bool = False,
) -> dict[str, Any]:
    """Write extraction input and label metadata and return their reference comparison.

    Outputs are ``native_pku_test.jsonl`` and ``native_pku_test.sidecar.jsonl``.
    Existing files require ``force=True``. Check ``all_match`` in the returned report:
    checksum or row-count mismatches set it to false and leave the written files in place.
    """
    out_dir = Path(out_dir)
    check_rebuild_outputs(
        out_dir, ["native_pku_test.jsonl", "native_pku_test.sidecar.jsonl"], force=force
    )
    ds = load_pku_test_split(revision=revision)
    input_rows, sidecar_rows, summary = build_frame(ds)

    written = {
        "input": write_jsonl(out_dir / "native_pku_test.jsonl", input_rows, force=force),
        "sidecar": write_jsonl(
            out_dir / "native_pku_test.sidecar.jsonl", sidecar_rows, force=force
        ),
    }

    files: dict[str, dict[str, Any]] = {}
    n_matched = 0
    for key, expected in EXPECTED_SHA256.items():
        got = written[key]["sha256"]
        match = got == expected
        n_matched += int(match)
        files[key] = {
            "n": written[key]["n"],
            "sha256": got,
            "expected_sha256": expected,
            "match": match,
        }

    counts_ok = (
        summary["n_prompts"] == EXPECTED_N_PROMPTS
        and summary["n_rows_both_responses"] == EXPECTED_N_ROWS
    )
    return {
        "component": "rebuilders",
        "corpus": "PKU-Alignment/PKU-SafeRLHF (test split)",
        "revision": revision,
        "license": "CC-BY-NC-4.0 (attribution: Ji et al., PKU-Alignment Team, 2024) — "
        "not redistributed; derived bytes never committed to this repo",
        "files": files,
        "n_files_expected": len(EXPECTED_SHA256),
        "n_matched": n_matched,
        "all_match": n_matched == len(EXPECTED_SHA256) and counts_ok,
        "summary": summary,
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Reconstruct PKU-SafeRLHF test files and compare them with the reference frame."
    )
    parser.add_argument("--out-dir", required=True, help="directory for the rebuilt frame files")
    parser.add_argument("--report", required=True, help="path for the verification report JSON")
    parser.add_argument(
        "--offline", action="store_true", help="use only the local Hugging Face cache"
    )
    parser.add_argument("--force", action="store_true", help="overwrite existing rebuilt files")
    parser.add_argument("--revision", default=PINNED_PKU_REVISION)
    args = parser.parse_args(argv)

    if args.offline:
        enable_hf_offline()

    report = rebuild(args.out_dir, revision=args.revision, force=args.force)
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n")

    ok = bool(report["all_match"])
    print(
        f"[rcv.rebuild.pku] {report['n_matched']}/{report['n_files_expected']} file checksums matched; "
        f"n_rows={report['summary']['n_rows_both_responses']} (expected {EXPECTED_N_ROWS}); "
        f"report -> {report_path}"
    )
    if not ok:
        print(
            "[rcv.rebuild.pku] Verification failed: rebuilt files or row counts differ from "
            "the reference frame. See the report for details.",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
