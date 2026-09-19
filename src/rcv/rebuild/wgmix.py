"""Reconstruct WildGuardMix evaluation, pool, and drift files and compare their checksums.

The reference carve has 20,199 items: 1,709 evaluation rows, 9,999 pool rows, and
8,491 rows across 11 drift families. Reproducing its bytes requires the same upstream
revision, row order, sampling procedure, and JSONL serialization.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import time
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ._jsonl import RebuildError, check_rebuild_outputs, enable_hf_offline, write_jsonl

PINNED_WGMIX_REVISION = "d29c47f41c8b51348b5c8e8c81c039b3132b66d1"

SEED = 42
POOL_TARGET = 10000
DRIFT_MIN_ADV = 300
NON_FAMILY_SUBCATS = {"benign", "others"}  # Exclude categories without a specific attack topic.

HARMFUL = "harmful"
UNHARMFUL = "unharmful"
NONE_LABEL = "None"

EXPECTED_UNIVERSE_N = 20199

# SHA-256 values of the reference corpus files.
DATA_MANIFEST_SHA256: dict[str, str] = {
    "eval": "ac9feb91c55ee8d626c839bce60798bea6d2c8ddf88d4c802a577b0c0e58e5df",
    "pool": "9100b1612fa6da53f9d0ba8ab240b5859a1556bd5817d090bb573fb3fb76546f",
    "drift/social_stereotypes_and_unfair_discrimination": (
        "837c683b617cfa50d824ce2e752da51ac7ff7ce4787def2521d4bc1fcc0ac1bb"
    ),
    "drift/private_information_individual": (
        "b9fc3414b460d85b990d501af568edd1f5225e69c3f04940804f8cfaf73c254e"
    ),
    "drift/disseminating_false_or_misleading_information_encouraging_disinformation_campaigns": (
        "8d72f6facbd476ec6335620905a7623e6273bc954316dcfecbb3e504c1cd69eb"
    ),
    "drift/sensitive_information_organization_government": (
        "d171fd419ac082ac9a9af3362517b756577257a6a592635c7e35ede70377da14"
    ),
    "drift/fraud_assisting_illegal_activities": (
        "95e31f96655cb3400fe842f777cb015d18dd8ff289982381660c227e37c7db6b"
    ),
    "drift/toxic_language_hate_speech": (
        "ffcaa0e69d4634cf081c3425e2c0b151d3fe2c65b3a7d7f2084bad17d256d2b3"
    ),
    "drift/violence_and_physical_harm": (
        "f122da31ebcc368573d24e59146716b306e3de599d262f4859ccc36826421916"
    ),
    "drift/defamation_encouraging_unethical_or_unsafe_actions": (
        "16de05d3d71aa7f5c2cf7a03a14f3bf9253b6384a02908c987792bd332a24b2a"
    ),
    "drift/causing_material_harm_by_disseminating_misinformation": (
        "bee8bef1f1dce6e7d647c99c1be53d5038ea47be847bc67785835e972e791d61"
    ),
    "drift/mental_health_over-reliance_crisis": (
        "8b0b224b9dec690a378384680da4a72fa7d16e9cd085ceae76191068a2d1ea77"
    ),
    "drift/cyberattack": "ce50bce5cae637c4be011ceae9ee48014dce8dc31388813e1e9f54c42bf3e559",
}


class CarveError(RebuildError):
    """The corpus carve failed an input or integrity check."""


def make_item_id(idx: int, prompt: str, response: str) -> str:
    """Derive an item ID from the zero-based output row index and its text."""
    return "it_" + hashlib.sha1(f"{idx}|{prompt}|{response}".encode()).hexdigest()[:12]


def harm_polarity(label: str) -> int | None:
    """Map harmful to 1, unharmful to 0, and unrecognized labels to None."""
    if label == HARMFUL:
        return 1
    if label == UNHARMFUL:
        return 0
    return None


def response_present(row: Mapping[str, Any]) -> bool:
    """Require a nonblank response and a harm label other than None or ``'None'``."""
    resp = row["response"]
    if resp is None or str(resp).strip() == "":
        return False
    return str(row["response_harm_label"]) != NONE_LABEL


def to_record(
    split_name: str, wg_source: str, orig_index: int, row_idx: int, row: Mapping[str, Any]
) -> dict[str, Any]:
    """Convert an upstream row to a corpus record, preserving field order for checksums."""
    gt = harm_polarity(str(row["response_harm_label"]))
    if gt is None:
        raise CarveError(
            f"{wg_source}[{orig_index}]: unsupported response_harm_label "
            f"{row['response_harm_label']!r}"
        )
    prompt = row["prompt"]
    response = row["response"]
    return {
        "split": split_name,
        "row_id": row_idx,  # line index within this file
        "item_id": make_item_id(row_idx, prompt, response),
        "wg_source": wg_source,
        "wg_orig_index": orig_index,  # index in the original upstream split
        "prompt": prompt,
        "response": response,
        "beaver_pred": -1,  # No classifier prediction yet.
        "ground_truth": gt,  # Upstream response-harm label, not the deployer's rubric label.
        "ai2_prompt_harm": harm_polarity(str(row["prompt_harm_label"])),
        "ai2_response_harm": gt,
        "ai2_refusal": str(row["response_refusal_label"]),
        "adversarial": bool(row["adversarial"]),
        "subcategory": str(row["subcategory"]),
    }


def reassign_row_ids(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Copy rows in input order, deriving row and item IDs from their new positions."""
    out = []
    for i, row in enumerate(rows):
        row = dict(row)
        row["row_id"] = i
        row["item_id"] = make_item_id(i, row["prompt"], row["response"])
        out.append(row)
    return out


@dataclass
class CarveResult:
    """Evaluation and pool rows, drift families by subcategory, and carve statistics."""

    eval_rows: list[dict[str, Any]]
    pool_rows: list[dict[str, Any]]
    drift: dict[str, list[dict[str, Any]]]
    stats: dict[str, Any] = field(default_factory=dict)

    @property
    def universe_n(self) -> int:
        return (
            len(self.eval_rows) + len(self.pool_rows) + sum(len(r) for r in self.drift.values())
        )

    def universe_ids(self) -> set[str]:
        ids = {r["item_id"] for r in self.eval_rows} | {r["item_id"] for r in self.pool_rows}
        for rows in self.drift.values():
            ids |= {r["item_id"] for r in rows}
        return ids


def carve(
    test_rows: Sequence[Mapping[str, Any]],
    train_rows: Sequence[Mapping[str, Any]],
    *,
    seed: int = SEED,
    pool_target: int = POOL_TARGET,
    drift_min_adv: int = DRIFT_MIN_ADV,
) -> CarveResult:
    """Build a deterministic carve from upstream test and train rows in their original order.

    After filtering with ``response_present``, evaluation uses the test rows. Drift families
    contain all adversarial train rows in subcategories with at least ``drift_min_adv`` such
    rows, excluding ``benign`` and ``others``. Pool rows are sampled from the remaining train
    rows, stratified by adversarial status and subcategory; ``pool_target`` is approximate.

    Reject shared prompt-response pairs between evaluation and the selected train rows,
    shared source indices between pool and drift, and evaluation sets without both labels.
    """
    test_present = [(i, test_rows[i]) for i in range(len(test_rows)) if response_present(test_rows[i])]
    train_present = [
        (i, train_rows[i]) for i in range(len(train_rows)) if response_present(train_rows[i])
    ]

    eval_rows = [
        to_record("eval", "wildguardtest", oi, k, r) for k, (oi, r) in enumerate(test_present)
    ]

    adv_counts: Counter[str] = Counter()
    for _oi, r in train_present:
        if bool(r["adversarial"]) and str(r["subcategory"]) not in NON_FAMILY_SUBCATS:
            adv_counts[str(r["subcategory"])] += 1
    candidates = sorted(
        [sc for sc, n in adv_counts.items() if n >= drift_min_adv], key=lambda sc: -adv_counts[sc]
    )
    if not candidates:
        raise CarveError(
            f"No drift candidate subcategory has at least {drift_min_adv} eligible adversarial rows."
        )
    candidate_set = set(candidates)

    drift_by_sc: dict[str, list[tuple[int, Mapping[str, Any]]]] = defaultdict(list)
    for oi, r in train_present:
        sc = str(r["subcategory"])
        if sc in candidate_set and bool(r["adversarial"]):
            drift_by_sc[sc].append((oi, r))
    drift_records: dict[str, list[dict[str, Any]]] = {}
    drift_held_out_orig: set[int] = set()
    for sc in candidates:
        items = sorted(drift_by_sc[sc], key=lambda t: t[0])
        rows = reassign_row_ids(
            [to_record(f"drift_{sc}", "wildguardtrain", oi, 0, r) for oi, r in items]
        )
        drift_records[sc] = rows
        drift_held_out_orig.update(oi for oi, _ in items)

    pool_eligible = [(oi, r) for oi, r in train_present if oi not in drift_held_out_orig]
    if not pool_eligible:
        raise CarveError("No pool-eligible rows remain after drift selection.")

    strata: dict[tuple[bool, str], list[tuple[int, Mapping[str, Any]]]] = defaultdict(list)
    for oi, r in pool_eligible:
        strata[(bool(r["adversarial"]), str(r["subcategory"]))].append((oi, r))
    total_elig = len(pool_eligible)
    frac = pool_target / total_elig
    rng = random.Random(seed)
    pool_items: list[tuple[int, Mapping[str, Any]]] = []
    # Stratum order fixes the sequence of draws from the shared RNG.
    for key in sorted(strata.keys()):
        bucket = sorted(strata[key], key=lambda t: t[0])
        # Independent rounding yields 9,999 rows for the reference target of 10,000.
        take = round(len(bucket) * frac)
        take = min(take, len(bucket))
        if take <= 0:
            continue
        idxs = sorted(rng.sample(range(len(bucket)), take))
        pool_items.extend(bucket[j] for j in idxs)
    pool_items.sort(key=lambda t: t[0])
    pool_rows = reassign_row_ids(
        [to_record("pool", "wildguardtrain", oi, 0, r) for oi, r in pool_items]
    )

    def pr_set(rows: Sequence[Mapping[str, Any]]) -> set[tuple[str, str]]:
        return {(row["prompt"], row["response"]) for row in rows}

    eval_pr = pr_set(eval_rows)
    train_pr = pr_set(pool_rows)
    for sc in candidates:
        train_pr |= pr_set(drift_records[sc])
    overlap = eval_pr & train_pr
    if overlap:
        raise CarveError(
            f"Evaluation shares {len(overlap)} prompt-response pairs with pool or drift rows."
        )
    pool_orig = {row["wg_orig_index"] for row in pool_rows}
    if pool_orig & drift_held_out_orig:
        raise CarveError(
            f"Pool/drift overlap: {len(pool_orig & drift_held_out_orig)} shared source indices."
        )
    for name, rows in [("eval", eval_rows), ("pool", pool_rows)]:
        for row in rows:
            for f in ("prompt", "response", "beaver_pred", "ground_truth"):
                if f not in row:
                    raise CarveError(f"{name}: record missing required field {f}")
    eval_gt = Counter(row["ground_truth"] for row in eval_rows)
    if len(eval_gt) < 2:
        raise CarveError("Evaluation rows must include both harm labels (0 and 1).")

    stats = {
        "seed": seed,
        "counts": {
            "test_total": len(test_rows),
            "test_response_present": len(test_present),
            "train_total": len(train_rows),
            "train_response_present": len(train_present),
            "eval_n": len(eval_rows),
            "pool_n": len(pool_rows),
            "pool_eligible": len(pool_eligible),
            "drift_held_out_adversarial": len(drift_held_out_orig),
            "n_candidates": len(candidates),
        },
        "eval_ground_truth": {str(k): v for k, v in sorted(eval_gt.items())},
        "drift_candidate_sizes": {sc: len(drift_records[sc]) for sc in candidates},
    }
    return CarveResult(eval_rows=eval_rows, pool_rows=pool_rows, drift=drift_records, stats=stats)


def load_wildguardmix(revision: str = PINNED_WGMIX_REVISION) -> tuple[Any, Any]:
    """Return ``(test_split, train_split)`` from the requested WildGuardMix revision.

    Requires the ``rebuild`` extra. For cached data only, call ``enable_hf_offline`` before
    importing ``datasets``.
    """
    try:
        from datasets import load_dataset
    except ImportError as exc:  # pragma: no cover - environment guard
        raise RebuildError(
            "Missing dependency 'datasets'. "
            "Install with: pip install 'regime-conditional-verification[rebuild]'"
        ) from exc
    test_ds = load_dataset("allenai/wildguardmix", "wildguardtest", revision=revision)["test"]
    train_ds = load_dataset("allenai/wildguardmix", "wildguardtrain", revision=revision)["train"]
    return test_ds, train_ds


def verify_against_manifest(written: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Compare supplied file checksums with the reference manifest and return a report.

    ``all_match`` is false for mismatched checksums, missing files, or unexpected files.
    """
    files: dict[str, dict[str, Any]] = {}
    n_matched = 0
    missing = sorted(set(DATA_MANIFEST_SHA256) - set(written))
    unexpected = sorted(set(written) - set(DATA_MANIFEST_SHA256))
    for key, expected in DATA_MANIFEST_SHA256.items():
        if key not in written:
            continue
        got = written[key]["sha256"]
        match = got == expected
        n_matched += int(match)
        files[key] = {
            "n": written[key]["n"],
            "sha256": got,
            "expected_sha256": expected,
            "match": match,
        }
    report = {
        "files": files,
        "n_files_expected": len(DATA_MANIFEST_SHA256),
        "n_matched": n_matched,
        "all_match": n_matched == len(DATA_MANIFEST_SHA256) and not missing and not unexpected,
        "missing_files": missing,
        "unexpected_files": unexpected,
    }
    return report


def rebuild(
    out_dir: Path | str,
    *,
    revision: str = PINNED_WGMIX_REVISION,
    force: bool = False,
) -> dict[str, Any]:
    """Write the reconstructed corpus and return its comparison with the reference carve.

    Existing files require ``force=True``. Check ``all_match`` in the returned report:
    checksum or row-count mismatches set it to false and leave the written files in place.
    """
    out_dir = Path(out_dir)
    test_ds, train_ds = load_wildguardmix(revision=revision)
    result = carve(test_ds, train_ds)

    check_rebuild_outputs(
        out_dir,
        ["eval.jsonl", "pool.jsonl", *[f"drift_candidates/{sc}.jsonl" for sc in result.drift]],
        force=force,
    )

    written: dict[str, dict[str, Any]] = {}
    written["eval"] = write_jsonl(out_dir / "eval.jsonl", result.eval_rows, force=force)
    written["pool"] = write_jsonl(out_dir / "pool.jsonl", result.pool_rows, force=force)
    for sc, rows in result.drift.items():
        written[f"drift/{sc}"] = write_jsonl(
            out_dir / "drift_candidates" / f"{sc}.jsonl", rows, force=force
        )

    report = verify_against_manifest(written)
    report.update(
        {
            "component": "rebuilders",
            "corpus": "allenai/wildguardmix",
            "revision": revision,
            "license": "ODC-BY 1.0 (attribution: Han et al., WildGuard, AI2, 2024) — not redistributed",
            "universe_n": result.universe_n,
            "universe_n_expected": EXPECTED_UNIVERSE_N,
            "carve_stats": result.stats,
            "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
    )
    if result.universe_n != EXPECTED_UNIVERSE_N:
        report["all_match"] = False
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Reconstruct WildGuardMix corpus files and compare them with the reference carve."
    )
    parser.add_argument("--out-dir", required=True, help="directory for the rebuilt jsonl files")
    parser.add_argument("--report", required=True, help="path for the verification report JSON")
    parser.add_argument(
        "--offline", action="store_true", help="use only the local Hugging Face cache"
    )
    parser.add_argument("--force", action="store_true", help="overwrite existing rebuilt files")
    parser.add_argument("--revision", default=PINNED_WGMIX_REVISION)
    args = parser.parse_args(argv)

    if args.offline:
        enable_hf_offline()

    report = rebuild(args.out_dir, revision=args.revision, force=args.force)
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n")

    ok = bool(report["all_match"])
    print(
        f"[rcv.rebuild.wgmix] {report['n_matched']}/{report['n_files_expected']} file checksums matched; "
        f"universe_n={report['universe_n']} (expected {EXPECTED_UNIVERSE_N}); "
        f"report -> {report_path}"
    )
    if not ok:
        print(
            "[rcv.rebuild.wgmix] Verification failed: rebuilt files or row count differ from "
            "the reference carve. See the report for details.",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
