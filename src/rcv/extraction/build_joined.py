"""Build row-aligned fine-tuned classifier artifacts against a base reference.

Checks item/set order, target labels, agreement, and finite float32 embeddings
with one row per item. Writes embeddings, verdict/score/label arrays, item IDs,
and a report with output SHA-256 digests. Runs on CPU; pass ``--base-items`` and
``--base-arrays`` in the public checkout.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from ._io import binary_vector, read_jsonl, sha_file, validate_item_ids


def join_ft_surface(rows, Z, base_items, base_arrays, verdict_key="lg3_pred",
                    score_key="lg3_score", agree_key="lg3_agreement"):
    """Return ``(arrays, report)`` after checking extraction rows against the base reference.

    Inputs are ordered extraction rows, float32 embeddings ``Z`` of shape ``(N, D)``,
    base item records, and base arrays containing ``ystar``. Verdicts and target
    labels use 0 for safe and 1 for unsafe; ``z_error`` flags are expected to be 0/1.
    Outputs are int64 verdict/ystar/agreement and finite float64 scores.
    Embedding width is not compared to the base.
    """
    n = len(rows)
    validate_item_ids(rows)
    validate_item_ids(base_items)
    base_n = len(base_items)
    if n != base_n:
        raise AssertionError(f"Extraction row count {n} differs from base item count {base_n}")
    if Z.ndim != 2 or Z.shape[0] != n:
        raise AssertionError(f"Embedding shape {Z.shape} must be ({n}, D)")
    if Z.dtype != np.float32:
        raise AssertionError(f"Embedding dtype is {Z.dtype}; expected float32")
    if not np.isfinite(Z).all():
        raise AssertionError("Embeddings contain non-finite values")
    if any(r.get("z_error", 0) != 0 for r in rows):
        raise AssertionError("All z_error flags must be 0")
    order_ok = all(
        rows[i]["item_id"] == base_items[i]["item_id"]
        and rows[i].get("set_id") == base_items[i].get("set_id")
        for i in range(n)
    )
    if not order_ok:
        raise AssertionError("Extraction item_id/set_id sequence differs from base items")

    verdict = binary_vector([r[verdict_key] for r in rows], n, verdict_key)
    score = np.array([float(r[score_key]) for r in rows], dtype=np.float64)
    if score.shape != (n,) or not np.isfinite(score).all():
        raise AssertionError(f"{score_key} must contain {n} finite scalar scores")
    ystar = binary_vector([r["ground_truth"] for r in rows], n, "ground_truth")
    base_ystar = binary_vector(base_arrays["ystar"], n, "base ystar")
    if "verdict" in base_arrays:
        binary_vector(base_arrays["verdict"], n, "base verdict")
    if not np.array_equal(ystar, base_ystar):
        raise AssertionError("Extraction target labels differ from base ystar")
    agreement = (verdict == ystar).astype(np.int64)
    ext_agree = binary_vector([r[agree_key] for r in rows], n, agree_key)
    if not np.array_equal(agreement, ext_agree):
        raise AssertionError("Recomputed agreement differs from extractor agreement")

    arrays = {"verdict": verdict, "clf_score": score, "ystar": ystar, "agreement": agreement}
    report = {
        "n": n, "z_dim": int(Z.shape[1]), "z_error_rows": 0,
        "order_matches_base": True, "ystar_matches_base": True,
        "ft_verdict_unsafe_rate": round(float(verdict.mean()), 4),
        "ft_adherence": round(float(agreement.mean()), 4),
    }
    return arrays, report


def _dry_run(a) -> int:
    base_items, base_arrays = a.base_items, a.base_arrays
    print("[build-joined][dry-run] plan:")
    print(f"  pulled dir   = {a.pulled}  (exists={Path(a.pulled).exists() if a.pulled else False})")
    print(f"  base items   = {base_items}  (exists={Path(base_items).exists() if base_items else False})")
    print(f"  base arrays  = {base_arrays}  (exists={Path(base_arrays).exists() if base_arrays else False})")
    print(f"  out dir      = {a.out_dir}")
    print(f"  verdict/score/agree keys = {a.verdict_key}/{a.score_key}/{a.agree_key}")
    print("  checks       = shapes, item/set order, binary labels/verdicts, finite float32 Z and scores, zero z_error flags, agreement")
    if base_items is None:
        print("  Pass --base-items and --base-arrays to select the base reference.")
    return 0


def run(a) -> int:
    if a.dry_run:
        return _dry_run(a)
    for name in ("pulled", "base_items", "base_arrays"):
        if not getattr(a, name):
            raise SystemExit(f"[build-joined] Supply --{name.replace('_', '-')}.")
    base_items_path, base_arrays_path = Path(a.base_items), Path(a.base_arrays)
    if not base_items_path or not base_items_path.exists():
        raise SystemExit(f"[build-joined] Base items file not found: {base_items_path}; supply --base-items")
    pulled = Path(a.pulled)
    rows = read_jsonl(pulled / a.items_name)
    Z = np.load(pulled / a.z_name)
    base_items = read_jsonl(base_items_path)
    base_arrays = np.load(base_arrays_path)

    arrays, report = join_ft_surface(rows, Z, base_items, base_arrays,
                                     a.verdict_key, a.score_key, a.agree_key)
    report["base_adherence"] = round(
        float((np.asarray(base_arrays["verdict"]).astype(int) == arrays["ystar"]).mean()), 4
    ) if "verdict" in base_arrays else None
    out_dir = Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / a.z_name, np.ascontiguousarray(Z, dtype=np.float32))
    np.savez(out_dir / (a.prefix + ".arrays.npz"), **arrays)
    with open(out_dir / (a.prefix + ".items.jsonl"), "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps({"set_id": r.get("set_id"), "item_id": r["item_id"]}) + "\n")
    report["shas"] = {
        a.z_name: sha_file(out_dir / a.z_name),
        a.prefix + ".arrays.npz": sha_file(out_dir / (a.prefix + ".arrays.npz")),
        a.prefix + ".items.jsonl": sha_file(out_dir / (a.prefix + ".items.jsonl")),
    }
    report["out_dir"] = str(out_dir)
    (out_dir / f"JOIN_REPORT_{a.prefix}.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    print("[build-joined] Wrote joined artifacts to", out_dir)
    return 0


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--pulled", help="extraction directory containing --items-name and --z-name")
    ap.add_argument("--out-dir", required=True, help="joined output dir")
    ap.add_argument("--prefix", default="lg3ft", help="output file prefix (e.g. lg3ft)")
    ap.add_argument("--items-name", default="lg3ft.items.jsonl")
    ap.add_argument("--z-name", default="lg3ft.Z.npy")
    ap.add_argument("--base-items", default=None, help="base items JSONL; supply explicitly in the public checkout")
    ap.add_argument("--base-arrays", default=None, help="base arrays NPZ; supply explicitly in the public checkout")
    ap.add_argument("--verdict-key", default="lg3_pred")
    ap.add_argument("--score-key", default="lg3_score")
    ap.add_argument("--agree-key", default="lg3_agreement")
    ap.add_argument("--dry-run", action="store_true", help="print the plan and path existence without reading input files")
    return ap


def main(argv=None) -> int:
    return run(build_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
