"""Run studies across seeds and write reproducible result records."""

import hashlib
import json
import platform
from dataclasses import asdict
from pathlib import Path

import numpy as np
import scipy
import sklearn
import yaml

from rcv.study import run_study

AGGREGATE_NAME = "aggregate.json"
SEED_RESULT_PATTERN = "seed_{seed}.json"
MINIMUM_DRAWS_FOR_A_SPREAD = 2


class ResultExistsError(FileExistsError):
    """A result destination already exists."""


def run_protocol(config, output_dir):
    """Run configured seeds and write append-only records and an aggregate."""
    if not isinstance(config, dict):
        config = yaml.safe_load(Path(config).read_text())
    seeds = seeds_of(config)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    refuse_existing_results(output_dir, seeds)

    provenance = provenance_of(config)
    records, refusals = [], {}
    for seed in seeds:
        try:
            result = run_study(config, seed)
        except (ValueError, RuntimeError, NotImplementedError) as refusal:
            refusals[str(seed)] = f"{type(refusal).__name__}: {refusal}"
            continue
        record = record_of(result, provenance)
        write_once(output_dir / SEED_RESULT_PATTERN.format(seed=seed), record)
        records.append(record)

    if refusals:
        write_once(output_dir / "REFUSALS.json", refusals)
        raise RuntimeError(f"{len(refusals)} of {len(seeds)} seeds refused: "
                           f"{sorted(refusals)}. No aggregate was written; see REFUSALS.json.")

    aggregate = aggregate_of(records, config, seeds, provenance)
    write_once(output_dir / AGGREGATE_NAME, aggregate)
    return aggregate


def seeds_of(config):
    seeds = list(config["seeds"])
    if not seeds:
        raise ValueError("Configuration must name at least one seed.")
    if len(set(seeds)) != len(seeds):
        raise ValueError(f"Configured seeds must be unique; got {seeds}.")
    return seeds


def refuse_existing_results(output_dir, seeds):
    """Refuse an output directory containing a planned result."""
    planned = [output_dir / SEED_RESULT_PATTERN.format(seed=seed) for seed in seeds]
    planned.append(output_dir / AGGREGATE_NAME)
    for path in planned:
        if path.exists():
            raise ResultExistsError(f"Result destination already exists: {path}.")


def write_once(path, payload):
    """Write a canonical JSON payload without overwriting an existing file."""
    if path.exists():
        raise ResultExistsError(f"Result destination already exists: {path}.")
    path.write_text(as_written(payload))


def as_written(payload):
    """Return canonical, newline-terminated JSON."""
    return json.dumps(payload, sort_keys=True, indent=2, separators=(",", ": ")) + "\n"


def provenance_of(config):
    """Record configuration, frame, code, and environment identifiers."""
    return {
        "config": config,
        "config_sha256": hashlib.sha256(as_written(config).encode("utf-8")).hexdigest(),
        "frame_sha256": sha256_of(config["frame"]["path"]),
        "code_sha256": code_digest(),
        "environment": environment(),
    }


def sha256_of(path):
    with open(path, "rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def code_digest():
    """Digest top-level RCV source files in a deterministic order."""
    package_root = Path(__file__).parent
    digest = hashlib.sha256()
    for source in sorted(package_root.glob("*.py")):
        digest.update(source.name.encode("utf-8"))
        digest.update(source.read_bytes())
    return digest.hexdigest()


def environment():
    """Return the numerical-library versions used by a run."""
    return {"python": platform.python_version(), "numpy": np.__version__,
            "scipy": scipy.__version__, "scikit-learn": sklearn.__version__}


def record_of(result, provenance):
    """Serialize one study result and its provenance."""
    record = {
        "seed": result.seed,
        "readout": asdict(result.readout),
        "readouts": {axis: asdict(axis_readout)
                     for axis, axis_readout in (result.readouts or
                                                {result.readout.label_set:
                                                 result.readout}).items()},
        "monitor": monitor_record_of(result.monitor),
        "loop_events": [asdict(event) for event in result.loop.events],
        "change_log": [change_log_record_of(entry) for entry in result.change_log],
        "total_oracle_labels_spent": result.total_oracle_labels_spent,
        "provenance": provenance,
    }
    if result.probe_selection is not None:
        record["probe_selection"] = result.probe_selection
    return record


WIRE_KEYED_CALIBRATION_FIELDS = (
    "reference", "threshold", "realized_alarm_rate_replayed",
    "realized_alarm_rate_at_reference", "detectable_shift", "break_even",
    "reference_posterior_quantile", "resolved_detectable_shift", "streams_in_the_tail")


ALARM_SENTENCE = ", at item "


def alarm_of(alarm_event):
    """Normalize current and legacy alarm records to one shape."""
    if alarm_event.get("monitored_position") is not None:
        return {"regime": int(alarm_event["regime"]),
                "boundary": alarm_event.get("boundary"),
                "monitored_position": int(alarm_event["monitored_position"]),
                "stream_position": alarm_event.get("stream_position")}
    regime, _, position = alarm_event["detail"].partition(ALARM_SENTENCE)
    return {"regime": int(regime.removeprefix("regime ")),
            "boundary": None,
            "monitored_position": int(position),
            "stream_position": None}


def as_wire_rows(mapping):
    """Convert a regime-boundary map to keyed rows."""
    return [{"regime": int(regime), "boundary": float(boundary), "value": mapping[regime][boundary]}
            for regime in sorted(mapping) for boundary in sorted(mapping[regime])]


def wire_map_of(rows_):
    """Convert keyed rows to a regime-boundary map."""
    restored = {}
    for row in rows_:
        restored.setdefault(int(row["regime"]), {})[float(row["boundary"])] = row["value"]
    return restored


def calibration_record_of(calibration):
    written = asdict(calibration)
    for field in WIRE_KEYED_CALIBRATION_FIELDS:
        if written.get(field) is not None:
            written[field] = as_wire_rows(written[field])
    return written


def monitor_record_of(monitor_state):
    """Serialize monitor state and calibration history."""
    if monitor_state.deployed is None:
        return {"streamless": True, "deployed": None, "recalibrations": [], "final": None}
    return {
        "streamless": False,
        "deployed": calibration_record_of(monitor_state.deployed),
        "recalibrations": [recalibration_record_of(entry)
                           for entry in monitor_state.recalibrations],
        "final": calibration_record_of(monitor_state.final),
    }


def recalibration_record_of(entry):
    """Serialize one post-repair reference recalibration."""
    return {
        "trigger": entry.trigger,
        "monitored_position": entry.monitored_position,
        "stream_position": entry.stream_position,
        "replay_seed": entry.replay_seed,
        "reference_audit_overlap_fraction": entry.reference_audit_overlap_fraction,
        "reference_window_length": entry.reference_window_length,
        "reference_window_monitored_span": entry.reference_window_monitored_span,
        "reference_window_skipped_positions": entry.reference_window_skipped_positions,
        "calibration": calibration_record_of(entry.calibration),
    }


def change_log_record_of(entry):
    """Serialize a change entry without duplicate calibration data."""
    written = {name: value for name, value in asdict(entry).items() if name != "calibration"}
    if written["probe_selection"] is None:
        del written["probe_selection"]
    return written


def aggregate_of(records, config, seeds, provenance):
    return {
        "study": config["study"],
        "seeds": seeds,
        "n": len(seeds),
        **naming_of(records),
        "metrics": {name: summarised(values)
                    for name, values in metric_series_of(records).items()},
        "caught_share_population": population_of(records),
        "provenance": provenance,
    }


def population_of(records):
    """Return caught-share denominators by seed and range."""
    per_seed = {str(record["seed"]): record["readout"]["caught_share_population"]
                for record in records}
    return {"per_seed": per_seed, "min": min(per_seed.values()), "max": max(per_seed.values())}


def metric_series_of(records):
    """Extract aggregate metric series from written seed records."""
    return {
        "raw_adherence": [record["readout"]["raw_adherence"] for record in records],
        "corrected_adherence": [record["readout"]["corrected_adherence"] for record in records],
        "caught_share": [record["readout"]["caught_share"] for record in records],
        "total_oracle_labels_spent": [record["total_oracle_labels_spent"] for record in records],
    }


def summarised(values):
    """Return mean, sample standard deviation, count, and observed range."""
    n = len(values)
    return {
        "mean": float(np.mean(values, dtype=np.float64)),
        "sd": (float(np.std(values, ddof=1, dtype=np.float64))
               if n >= MINIMUM_DRAWS_FOR_A_SPREAD else None),
        "n": n,
        "observed_range": [min(values), max(values)],
    }


def naming_of(records):
    """Return shared frame/label identifiers and per-seed label spending."""
    frame_names = {record["readout"]["frame_name"] for record in records}
    label_sets = {record["readout"]["label_set"] for record in records}
    if len(frame_names) != 1:
        raise ValueError(f"Cannot aggregate records with different frame names: "
                         f"{sorted(frame_names)}.")
    if len(label_sets) != 1:
        raise ValueError(f"Cannot aggregate records with different label sets: "
                         f"{sorted(label_sets)}.")
    spent = {str(record["seed"]): record["readout"]["oracle_labels_spent"] for record in records}
    return {
        "frame_name": frame_names.pop(),
        "label_set": label_sets.pop(),
        "oracle_labels_spent": {"per_seed": spent,
                                "mean": float(np.mean(list(spent.values()),
                                                      dtype=np.float64))},
    }
