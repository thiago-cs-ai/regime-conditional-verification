"""Recompute arXiv Table 1 from the shipped per-seed records.

The verifier requires the six published cells and all ten seed identities, checks each saved
aggregate, and compares the rounded result with the paper. It does not rerun classifier inference
or probe training.
"""

import json
import os
import sys
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

import numpy as np

ROOT = Path(os.environ.get("RCV_BUNDLE_ROOT") or Path(__file__).resolve().parents[1])
STEERING_RELATIVE = Path("results/final_steering_20260728/steering")
REFERENCE_RELATIVE = Path("results/table1_reference.json")

AXIS = "oracle"
FIELDS = ("raw_adherence", "corrected_adherence", "caught_share")
EXPECTED_CELLS = (
    "pku_beaver",
    "pku_lg3",
    "pku_wg",
    "wgmix_big_beaver",
    "wgmix_big_lg3",
    "wgmix_big_wg",
)
EXPECTED_SEEDS = (42, 123, 456, 789, 1024, 7, 2026, 31415, 271828, 8675309)
ABSOLUTE_TOLERANCE = 1e-12


class TableVerificationError(ValueError):
    """The shipped records cannot substantiate the published table."""


def round_half_up(value, decimals=3):
    return Decimal(repr(float(value))).quantize(
        Decimal(1).scaleb(-decimals), rounding=ROUND_HALF_UP
    )


def read_json(path):
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise TableVerificationError(f"cannot read {path}: {error}") from error


def seed_records(cell_dir):
    expected_names = {f"seed_{seed}.json" for seed in EXPECTED_SEEDS}
    actual_names = {path.name for path in cell_dir.glob("seed_*.json")}
    if actual_names != expected_names:
        raise TableVerificationError(
            f"{cell_dir.name} seed files: expected {sorted(expected_names)}, "
            f"found {sorted(actual_names)}"
        )

    records = []
    for seed in EXPECTED_SEEDS:
        path = cell_dir / f"seed_{seed}.json"
        record = read_json(path)
        if record.get("seed") != seed:
            raise TableVerificationError(
                f"{path.name} records seed {record.get('seed')}; its filename records seed {seed}"
            )
        try:
            for field in FIELDS:
                float(record["readouts"][AXIS][field])
        except (KeyError, TypeError, ValueError) as error:
            raise TableVerificationError(f"{path} lacks a numeric readouts.{AXIS}.{field}") from error
        records.append(record)
    return records


def cell_summary(records):
    values = {
        field: np.asarray([record["readouts"][AXIS][field] for record in records], dtype=float)
        for field in FIELDS
    }
    summary = {"n": len(records)}
    for field, samples in values.items():
        summary[field] = (float(samples.mean()), float(samples.std(ddof=1)))
    return summary


def require_close(actual, expected, description):
    if not np.isclose(actual, expected, rtol=0.0, atol=ABSOLUTE_TOLERANCE):
        raise TableVerificationError(f"{description}: computed {actual}, recorded {expected}")


def check_saved_aggregate(cell_dir, summary):
    aggregate = read_json(cell_dir / "aggregate.json")
    if aggregate.get("n") != summary["n"]:
        raise TableVerificationError(
            f"{cell_dir.name} aggregate n: computed {summary['n']}, recorded {aggregate.get('n')}"
        )
    for field in FIELDS:
        try:
            metric = aggregate["metrics"][field]
            recorded = (float(metric["mean"]), float(metric["sd"]))
        except (KeyError, TypeError, ValueError) as error:
            raise TableVerificationError(
                f"{cell_dir / 'aggregate.json'} lacks metrics.{field}.mean or sd"
            ) from error
        for name, actual, expected in zip(("mean", "sd"), summary[field], recorded, strict=True):
            require_close(actual, expected, f"{cell_dir.name} aggregate {field} {name}")


def rounded_pair(pair):
    return [str(round_half_up(value)) for value in pair]


def verify_table1(root=ROOT):
    root = Path(root)
    steering = root / STEERING_RELATIVE
    if not steering.is_dir():
        raise TableVerificationError(f"steering records not found at {steering}")

    actual_cells = {path.name for path in steering.iterdir() if path.is_dir()}
    if actual_cells != set(EXPECTED_CELLS):
        raise TableVerificationError(
            f"cell directories: expected {sorted(EXPECTED_CELLS)}, found {sorted(actual_cells)}"
        )

    reference = read_json(root / REFERENCE_RELATIVE)
    if set(reference.get("cells", {})) != set(EXPECTED_CELLS):
        raise TableVerificationError("the Table 1 reference does not name the six published cells")

    summaries = {}
    for cell in EXPECTED_CELLS:
        cell_dir = steering / cell
        summary = cell_summary(seed_records(cell_dir))
        check_saved_aggregate(cell_dir, summary)
        for field in FIELDS:
            computed = rounded_pair(summary[field])
            published = reference["cells"][cell].get(field)
            if computed != published:
                raise TableVerificationError(
                    f"arXiv Table 1 {cell} {field}: computed {computed}, published {published}"
                )
        summaries[cell] = summary
    return summaries


def format_pair(pair):
    return f"{round_half_up(pair[0])} +/- {round_half_up(pair[1])}"


def main():
    try:
        summaries = verify_table1(ROOT)
    except TableVerificationError as error:
        raise SystemExit(str(error)) from error

    print("Table 1 — deployed configuration, oracle axis")
    print(f"source: {STEERING_RELATIVE}/<cell>/seed_*.json")
    print(f"fields: readouts.{AXIS}." + "{" + ", ".join(FIELDS) + "}")
    print("reduction: mean +/- sd (ddof=1) over seeds, rounded half-up to 3 decimals")
    print()

    header = (
        f"{'cell':<18}{'n':>3}   {'raw adherence':<18}->  "
        f"{'corrected adherence':<22}{'caught share'}"
    )
    print(header)
    print("-" * len(header))
    for cell in EXPECTED_CELLS:
        summary = summaries[cell]
        print(
            f"{cell:<18}{summary['n']:>3}   "
            f"{format_pair(summary['raw_adherence']):<18}->  "
            f"{format_pair(summary['corrected_adherence']):<22}"
            f"{format_pair(summary['caught_share'])}"
        )
    print()
    print("Cell names read corpus_classifier. Raw adherence is the share of evaluation items")
    print("on which the classifier's verdict matches the label axis; corrected adherence is")
    print("the same share after the flip. Caught share is, among unsafe items the classifier")
    print("passed, the share the correction recovers. Polarity throughout: 1 = unsafe.")
    print("Verified against Table 1 of arXiv:2608.14089v2.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
