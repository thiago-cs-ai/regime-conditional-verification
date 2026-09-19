"""Prepare policy-labeled inputs and run RCV on aligned LG3 extraction artifacts."""

import argparse
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np
import yaml

from rcv.frame_building import (
    FrameRow,
    build_frame,
    digest_of,
    family_codes,
    input_manifest,
    save_frame,
)
from rcv.rebuild._jsonl import read_jsonl, write_jsonl
from rcv.rebuild._policy_prompt import CUSTOM_PROMPT_VERSION, input_sha256
from rcv.runner import run_protocol
from rcv.study import validate_config


def positive_integer(value, field):
    if type(value) is not int or value < 1:
        raise ValueError(f"{field} must be a positive integer.")
    return value


def binary(value, field, item_id):
    if type(value) is not int or value not in (0, 1):
        raise ValueError(f"{item_id}: {field} must be integer 0 or 1.")
    return value


def validate_items(items):
    """Validate source identities and return connected prompt/group families."""
    if not items:
        raise ValueError("Input items are empty.")
    seen = set()
    for row in items:
        if not isinstance(row, dict):
            raise ValueError("Each input row must be a JSON object.")
        iid = row.get("item_id")
        if not isinstance(iid, str) or not iid.strip():
            raise ValueError("item_id must be a nonempty string.")
        if iid in seen:
            raise ValueError(f"Duplicate item_id: {iid!r}.")
        seen.add(iid)
        if not isinstance(row.get("prompt"), str) or not row["prompt"].strip():
            raise ValueError(f"{iid}: prompt must be a nonblank string.")
        if not isinstance(row.get("response"), str):
            raise ValueError(f"{iid}: response must be a string.")
        if "group_id" in row and (
            not isinstance(row["group_id"], str) or not row["group_id"].strip()
        ):
            raise ValueError(f"{iid}: group_id must be a nonblank string.")
    return family_codes([r["prompt"] for r in items], [r.get("group_id") for r in items])


def join_labels(items, label_paths, label_field):
    """Join the last non-null target per ID, validating custom judge identities."""
    by_id = {}
    for path in label_paths:
        for row in read_jsonl(path):
            if not isinstance(row, dict) or not isinstance(row.get("item_id"), str):
                raise ValueError(f"{path}: label rows require a string item_id.")
            value = row.get(label_field)
            if value is not None:
                binary(value, label_field, row["item_id"])
                by_id[row["item_id"]] = row
    missing = [r["item_id"] for r in items if r["item_id"] not in by_id]
    if missing:
        raise ValueError(f"Missing labels for {len(missing)} items; first: {missing[0]!r}.")
    selected = [by_id[r["item_id"]] for r in items]
    custom = any(r.get("prompt_version") == CUSTOM_PROMPT_VERSION
                 or "policy_sha256" in r or "input_sha256" in r for r in selected)
    identity = None
    out = []
    for item, label in zip(items, selected, strict=True):
        if custom:
            current = tuple(label.get(k) for k in ("model", "prompt_version", "policy_sha256"))
            if (not all(isinstance(v, str) and v for v in current)
                    or current[1] != CUSTOM_PROMPT_VERSION):
                raise ValueError(f"{item['item_id']}: missing or unsupported custom-policy identity.")
            if identity is not None and current != identity:
                raise ValueError(f"{item['item_id']}: inconsistent custom-policy identity.")
            identity = current
            if label.get("input_sha256") != input_sha256(item):
                raise ValueError(f"{item['item_id']}: input_sha256 does not match the source text.")
        out.append({**item, "ground_truth": label[label_field]})
    return out


def prepare(baseline, stream, labels, label_field, out, *, exclude_shared_groups=False,
            max_baseline_items=None, max_stream_items=None):
    for field, value in (("max_baseline_items", max_baseline_items),
                         ("max_stream_items", max_stream_items)):
        if value is not None:
            positive_integer(value, field)
    baseline_rows = read_jsonl(baseline)
    stream = Path(stream)
    paths = sorted(stream.glob("*.jsonl")) if stream.is_dir() else [stream]
    stream_rows = [row for path in paths for row in read_jsonl(path)]
    if not baseline_rows or not stream_rows:
        raise ValueError("Baseline and stream must both contain items.")
    items = baseline_rows + stream_rows
    families = validate_items(items)
    n_base = len(baseline_rows)
    stop = n_base + min(len(stream_rows), max_stream_items or len(stream_rows))
    stream_indices = list(range(n_base, stop))
    shared = set(families[:n_base]) & set(families[stream_indices])
    if shared and not exclude_shared_groups:
        raise ValueError(f"Baseline and stream share {len(shared)} families; "
                         "use separate groups or --exclude-shared-groups.")
    candidates = [i for i in range(n_base) if families[i] not in shared]
    sizes = Counter(families[candidates])
    remaining = max_baseline_items if max_baseline_items is not None else len(candidates)
    selected_groups = set()
    for family, size in sizes.items():
        if size <= remaining:
            selected_groups.add(family)
            remaining -= size
    baseline_indices = [i for i in candidates if families[i] in selected_groups]
    if not baseline_indices:
        raise ValueError("No baseline items remain after group selection.")
    projected = []
    for i in baseline_indices + stream_indices:
        row = items[i]
        projected.append({**{k: row[k] for k in ("item_id", "prompt", "response", "group_id", "set_id")
                             if k in row}, "is_stream": i >= n_base})
    joined = join_labels(projected, labels, label_field)
    write_jsonl(out, joined)
    print(f"Prepared baseline={len(baseline_indices)}, stream={len(stream_indices)}; "
          f"baseline excluded for overlap={n_base - len(candidates)}, "
          f"size={len(candidates) - len(baseline_indices)}; "
          f"stream excluded for size={len(stream_rows) - len(stream_indices)}. Output: {out}")
    return joined


def extraction_frame(inputs, extracted, embeddings):
    """Validate LG3 artifacts and build a frame in the original input order."""
    items, outputs = read_jsonl(inputs), read_jsonl(extracted)
    families = validate_items(items)
    if len(items) != len(outputs):
        raise ValueError("Extraction row count does not match inputs.")
    for item, row in zip(items, outputs, strict=True):
        iid = item["item_id"]
        if type(item.get("is_stream")) is not bool:
            raise ValueError(f"{iid}: is_stream must be boolean.")
        if not isinstance(row, dict) or any(
            row.get(k) != item.get(k)
            for k in ("item_id", "prompt", "response", "ground_truth", "is_stream", "group_id")
        ):
            raise ValueError(f"{iid}: extraction source fields or item order differ from inputs.")
        if type(row.get("is_stream")) is not bool:
            raise ValueError(f"{iid}: extracted is_stream must be boolean.")
        binary(item.get("ground_truth"), "ground_truth", iid)
        for field in ("ground_truth", "lg3_pred", "lg3_agreement"):
            binary(row.get(field), field, iid)
        if row["lg3_agreement"] != int(row["lg3_pred"] == row["ground_truth"]):
            raise ValueError(f"{iid}: lg3_agreement does not match verdict and target.")
        score = row.get("lg3_score")
        if type(score) not in (int, float) or not math.isfinite(score):
            raise ValueError(f"{iid}: lg3_score must be finite.")
        if row.get("z_error"):
            raise ValueError(f"{iid}: extraction contains a z_error placeholder.")
    roles = np.array([r["is_stream"] for r in items], dtype=bool)
    if not roles.any() or roles.all():
        raise ValueError("Baseline and stream must both contain items.")
    if set(families[~roles]) & set(families[roles]):
        raise ValueError("Baseline and stream share families.")
    z = np.load(embeddings, allow_pickle=False)
    if (z.ndim != 2 or z.shape[0] != len(items) or z.shape[1] == 0
            or z.dtype != np.float32 or not np.isfinite(z).all()):
        raise ValueError("Embeddings must be finite float32 (N, D) matching the input rows.")
    stats_path = Path(extracted).with_suffix(".stats.json")
    stats = json.loads(stats_path.read_text())
    expected = {"n": len(items), "n_rows": len(items), "n_err": 0,
                "shape": list(z.shape), "D": z.shape[1], "z_npy_sha256": digest_of(embeddings)}
    for field, value in expected.items():
        if stats.get(field) != value:
            raise ValueError(f"{stats_path}: {field} does not match extraction artifacts.")
    rows = [FrameRow(item_id=r["item_id"], prompt=r["prompt"], verdict=r["lg3_pred"],
                     oracle=r["ground_truth"], seed=r.get("group_id"),
                     clf_score=r["lg3_score"], is_stream=r["is_stream"]) for r in outputs]
    frame = build_frame(rows, [r["item_id"] for r in outputs], z)
    position = {iid: i for i, iid in enumerate(frame["item_id"])}
    order = [position[r["item_id"]] for r in items]
    # Restore chronology after build_frame's canonical ID sort, across every column.
    return {key: values[order] for key, values in frame.items()}


def resolve_config(template, frame_path, stream_length):
    config = yaml.safe_load(Path(template).read_text())
    if config["frame"]["path"] is not None:
        raise ValueError("Template frame.path must be null; run creates the frame.")
    config["frame"]["path"] = str(Path(frame_path).resolve())
    config["frame"].pop("sha256", None)
    calibration = config["monitor"]["calibration"]
    horizon = calibration["stream_length"]
    calibration["stream_length"] = (stream_length if horizon is None
                                     else positive_integer(horizon, "monitor.calibration.stream_length"))
    validate_config(config)
    return config


def summary_lines(record):
    readout = record["readout"]
    events = Counter(event["kind"] for event in record["loop_events"])
    initial = readout["oracle_labels_spent"]
    total = record["total_oracle_labels_spent"]
    raw, corrected = readout["raw_adherence"], readout["corrected_adherence"]
    endings = [kind for kind in ("escalation_demanded", "fresh_data_exhausted",
                                "reference_window_exhausted", "gate_uncomposable") if events[kind]]
    return [
        f"Seed {record['seed']}: adherence {raw:.1%} → {corrected:.1%} "
        f"({100 * (corrected - raw):+.1f} percentage points)",
        f"Caught share: {readout['caught_share']:.1%} of "
        f"{readout['caught_share_population']} unsafe items passed by LG3",
        f"Alarms: {events['alarm']}; accepted repairs: {events['repair_accepted']}; "
        f"reference refreshes: {len(record['monitor']['recalibrations'])}",
        f"Terminal events: {', '.join(endings) if endings else 'none'}",
        f"Labels: initial={initial}, maintenance={total - initial}, total={total}",
    ]


def run(inputs, extracted, embeddings, config, out):
    out = Path(out)
    if out.exists():
        raise FileExistsError(f"Run directory already exists: {out}.")
    frame = extraction_frame(inputs, extracted, embeddings)
    resolved = resolve_config(config, out / "frame.npz", int(frame["is_stream"].sum()))
    manifest = input_manifest([inputs, extracted, embeddings,
                               Path(extracted).with_suffix(".stats.json"), config, Path(__file__)])
    out.mkdir(parents=True)
    resolved["frame"]["sha256"] = save_frame(frame, out / "frame.npz")
    (out / "config.yaml").write_text(yaml.safe_dump(resolved, sort_keys=False))
    (out / "input_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    aggregate = run_protocol(resolved, out / "results")
    for seed in resolved["seeds"]:
        path = out / "results" / f"seed_{seed}.json"
        print("\n".join(summary_lines(json.loads(path.read_text()))))
        print(f"Readouts, loop events, and monitor recalibrations: {path}")
    print(f"Aggregate: {out / 'results' / 'aggregate.json'}")
    return aggregate


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prep = commands.add_parser("prepare", help="select groups and join policy labels by item ID")
    prep.add_argument("--baseline", required=True, type=Path)
    prep.add_argument("--stream", required=True, type=Path, help="ordered JSONL or directory of JSONL segments")
    prep.add_argument("--labels", required=True, nargs="+", type=Path)
    prep.add_argument("--label-field", required=True)
    prep.add_argument("--out", required=True, type=Path)
    prep.add_argument("--exclude-shared-groups", action="store_true")
    prep.add_argument("--max-baseline-items", type=int)
    prep.add_argument("--max-stream-items", type=int)
    study = commands.add_parser("run", help="validate extraction artifacts and run the RCV study")
    for name in ("inputs", "extracted", "embeddings", "config", "out"):
        study.add_argument(f"--{name}", required=True, type=Path)
    args = vars(parser.parse_args(argv))
    command = args.pop("command")
    try:
        (prepare if command == "prepare" else run)(**args)
    except (OSError, ValueError, RuntimeError) as exc:
        parser.exit(1, f"{command}: {exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
