"""Compare fine-tuned and base classifier verdicts on aligned items.

Reports flips, agreement with target labels, unsafe recall, and false-positive
rate overall and for eval/pool/drift:* sets. Optional exclusion groups use supplied
training and selection item IDs. Runs on CPU; pass ``--base-items`` and
``--base-arrays`` in the public checkout.
"""

from __future__ import annotations

import argparse
import json
from collections import OrderedDict
from pathlib import Path

import numpy as np

from ._io import binary_vector, read_jsonl, validate_item_ids


def recall_fpr(pred, ystar):
    """Return unsafe recall and false-positive rate; an absent target class gives NaN.

    Inputs are aligned binary arrays of shape ``(N,)``; 1 means unsafe.
    """
    pos = ystar == 1
    neg = ystar == 0
    tpr = float((pred[pos] == 1).mean()) if pos.any() else float("nan")
    fpr = float((pred[neg] == 1).mean()) if neg.any() else float("nan")
    return tpr, fpr


def flip_block(mask, base_pred, ft_pred, ystar):
    """Summarize aligned binary verdicts under a Boolean row mask.

    Oracle movement means gaining or losing agreement with ``ystar``. Metric
    deltas are fine-tuned minus base. An empty mask returns only ``{"n": 0}``.
    """
    m = mask
    cnt = int(m.sum())
    if cnt == 0:
        return {"n": 0}
    bp, fp, ys = base_pred[m], ft_pred[m], ystar[m]
    flips = int((bp != fp).sum())
    s2u = int(((bp == 0) & (fp == 1)).sum())
    u2s = int(((bp == 1) & (fp == 0)).sum())
    toward = int(((bp != ys) & (fp == ys)).sum())
    away = int(((bp == ys) & (fp != ys)).sum())
    btpr, bfpr = recall_fpr(bp, ys)
    ftpr, ffpr = recall_fpr(fp, ys)
    return {
        "n": cnt, "flips": flips, "flip_rate": round(flips / cnt, 4),
        "safe_to_unsafe": s2u, "unsafe_to_safe": u2s,
        "flips_toward_oracle": toward, "flips_away_from_oracle": away,
        "base_recall_unsafe": round(btpr, 4), "ft_recall_unsafe": round(ftpr, 4),
        "d_recall": round(ftpr - btpr, 4),
        "base_fpr": round(bfpr, 4), "ft_fpr": round(ffpr, 4), "d_fpr": round(ffpr - bfpr, 4),
        "base_adherence": round(float((bp == ys).mean()), 4),
        "ft_adherence": round(float((fp == ys).mean()), 4),
    }


def compute_profile(items, base_arrays, ft_arrays, exclusion=None):
    """Profile verdict arrays in item order; array-to-item identity is assumed.

    ``by_set`` includes eval, pool, and drift:* sets; other sets contribute only
    to ``overall`` and any exclusion groups. Exclusion lists use ``ft_train_ids``
    and ``ft_test_ids``; ``held_out_remainder`` means absent from both supplied lists.
    """
    validate_item_ids(items)
    set_ids = np.array([it.get("set_id") for it in items])
    item_ids = np.array([it["item_id"] for it in items])
    n = len(items)
    ystar = binary_vector(base_arrays["ystar"], n, "base ystar")
    bp = binary_vector(base_arrays["verdict"], n, "base verdict")
    fp = binary_vector(ft_arrays["verdict"], n, "fine-tuned verdict")
    ft_ystar = binary_vector(ft_arrays["ystar"], n, "fine-tuned ystar")
    if not np.array_equal(ystar, ft_ystar):
        raise AssertionError("Fine-tuned target labels differ from base")

    overall = flip_block(np.ones(n, dtype=bool), bp, fp, ystar)
    by_set = OrderedDict()
    drift = sorted({s for s in set_ids if isinstance(s, str) and s.startswith("drift:")})
    for sid in ["eval", "pool"] + drift:
        by_set[sid] = flip_block(set_ids == sid, bp, fp, ystar)

    profile = {"overall": overall, "by_set": by_set,
               "note": "positive class = UNSAFE (1). flips_toward_oracle = base wrong & repaired right."}
    if exclusion:
        train_ids = set(exclusion.get("ft_train_ids", []))
        test_ids = set(exclusion.get("ft_test_ids", []))
        is_train = np.array([iid in train_ids for iid in item_ids])
        is_test = np.array([iid in test_ids for iid in item_ids])
        held_out = ~(is_train | is_test)
        profile["ft_exclusion_view"] = {
            "ft_train_burned": flip_block(is_train, bp, fp, ystar),
            "ft_test_selection_touched": flip_block(is_test, bp, fp, ystar),
            "held_out_remainder": flip_block(held_out, bp, fp, ystar),
        }
    return profile


def _dry_run(a) -> int:
    print("[diff-profile][dry-run] plan:")
    print(f"  base items  = {a.base_items}")
    print(f"  base arrays = {a.base_arrays}")
    print(f"  ft arrays   = {a.ft_arrays}")
    print(f"  exclusion   = {a.exclusion or '(none)'}")
    print(f"  out         = {a.out}")
    print("  positive class = UNSAFE (1); flips and target-label agreement overall and for eval/pool/drift:* sets")
    return 0


def run(a) -> int:
    if a.dry_run:
        return _dry_run(a)
    for name in ("base_items", "base_arrays", "ft_arrays"):
        if not getattr(a, name):
            raise SystemExit(f"[diff-profile] Supply --{name.replace('_', '-')}.")
    base_items_path, base_arrays_path = Path(a.base_items), Path(a.base_arrays)
    items = read_jsonl(base_items_path)
    base = np.load(base_arrays_path)
    ft = np.load(a.ft_arrays)
    exclusion = json.loads(Path(a.exclusion).read_text()) if a.exclusion else None
    profile = compute_profile(items, base, ft, exclusion)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "DIFF_PROFILE.json").write_text(json.dumps(profile, indent=2))
    print(json.dumps(profile["overall"], indent=2))
    print("[diff-profile] Wrote DIFF_PROFILE.json to", out)
    return 0


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--ft-arrays", help="fine-tuned verdict and ystar arrays in base item order")
    ap.add_argument("--out", required=True, help="output dir for DIFF_PROFILE.json")
    ap.add_argument("--base-items", default=None, help="base items JSONL; supply explicitly in the public checkout")
    ap.add_argument("--base-arrays", default=None, help="base verdict and ystar NPZ; supply explicitly in the public checkout")
    ap.add_argument("--exclusion", default=None, help="optional JSON with ft_train_ids and ft_test_ids lists")
    ap.add_argument("--dry-run", action="store_true", help="print the plan without reading input files or checking paths")
    return ap


def main(argv=None) -> int:
    return run(build_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
