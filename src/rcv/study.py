"""Run one RCV study from a validated configuration."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml
from sklearn.metrics import roc_auc_score

from rcv.belief_bank import (
    BeliefBankMonitor,
    assert_boundaries,
    assert_budget_allocation,
    assert_streams_per_centre,
)
from rcv.cusum import assert_shift_specification
from rcv.estimator import Estimator
from rcv.flip import FlipRule
from rcv.frames import (
    agreement_of,
    concatenated,
    fit_on_slices,
    load_frame,
    rows,
    sha256_of,
    slice_of,
)
from rcv.guards import assert_all_four_cases
from rcv.loop import (
    DEPLOYMENT_MODE,
    EPISODE_MODE,
    EXHAUST_FRESH_DATA,
    RETRY_POLICIES,
    SCHEDULE_COLUMN,
    assert_loop_mode,
    audit_ladder_of,
    watch_traffic,
)
from rcv.probes import assert_probe_spec
from rcv.splitting import draw_group_blocked_split

EVALUATION_MATERIAL = "evaluation"
STREAM_PREFIX_MATERIAL = "stream_prefix"
CALIBRATION_MATERIALS = (EVALUATION_MATERIAL, STREAM_PREFIX_MATERIAL)

CONFIG_SCHEMA = {
    "study": None,
    "seeds": None,
    "frame": {"required": {"path", "name", "fitting_labels"},
              "optional": {"sha256", "scoring_labels"}},
    "split": {"required": {"evaluation_fraction", "calibration_fraction"},
              "optional": {"strata"}},
    "estimator": {"required": {"probe", "calibration", "route_probe", "route_calibration"},
                  "optional": {"feature_source"}},
    "flip": {"required": {"threshold"}, "optional": set()},
    "monitor": {"required": {"quiet_horizon_confidence", "detectable_shift", "calibration",
                             "event_bank"},
                "optional": set()},
    "loop": {"required": {"audit_sampling_window", "audit_budget", "acceptance",
                          "max_enlarging_retries", "cross_fit_folds"},
             "optional": {"observe_only", "post_repair_reference_window", "mode",
                          "post_alarm"}},
}

MONITOR_CALIBRATION_SCHEMA = {
    "required": {"stream_length", "n_streams", "streams_per_centre"},
    "optional": {"material", "prefix_length"},
}
PREFIX_LENGTH_BY_MATERIAL = {STREAM_PREFIX_MATERIAL: {"prefix_length"},
                             EVALUATION_MATERIAL: set()}
EVENT_BANK_SCHEMA = {"required": {"boundaries", "budget_allocation"},
                     "optional": {"counterfactual_without_flip_threshold"}}

RETIRED_MONITOR_KEYS = {
    "false_alarm_budget": "renamed quiet_horizon_confidence, WITH INVERTED MEANING: the new key "
                          "states q and the calibration level is 1 - q",
}
RETIRED_CALIBRATION_KEYS = {
    "reference_uncertainty": "retired; the posterior-drawn replay is the only calibration "
                             "semantics",
}

POST_ALARM_SCHEMA = {"required": {"composition", "gate_labels"}, "optional": set()}
STATIONARY_COMPOSITION = "stationary"
CENSUS_BASELINE_COMPONENT = "base"
POST_ALARM_SEGMENT_SALT = 20260730

ACCEPTANCE_FORMS = {"baseline_relative": {"form", "recall_slack", "over_block_inflation"},
                    None: {"recall_floor", "false_positive_tolerance"}}


def assert_section_keys(section, keys, required, optional):
    """Require a configuration section's keys to match its schema."""
    unknown = keys - required - optional
    missing = required - keys
    if unknown or missing:
        raise ValueError(f"{section}: unknown keys {sorted(unknown)}; missing keys "
                         f"{sorted(missing)}; allowed keys {sorted(required | optional)}.")


def validate_config(config):
    """Validate the complete study configuration before execution."""
    unknown = set(config) - set(CONFIG_SCHEMA)
    missing = set(CONFIG_SCHEMA) - set(config)
    if unknown or missing:
        raise ValueError(f"Configuration: unknown sections {sorted(unknown)}; missing sections "
                         f"{sorted(missing)}.")
    for key, migration in RETIRED_MONITOR_KEYS.items():
        if key in config["monitor"]:
            raise ValueError(f"monitor.{key} is a retired key: {migration}")
    for section, shape in CONFIG_SCHEMA.items():
        if shape is None:
            continue
        assert_section_keys(section, set(config[section]), shape["required"], shape["optional"])
    q = config["monitor"]["quiet_horizon_confidence"]
    if not 0.0 < q < 1.0:
        raise ValueError(f"monitor.quiet_horizon_confidence must be in (0, 1); got {q!r}.")
    validate_monitor_calibration(config["monitor"]["calibration"])
    acceptance = config["loop"]["acceptance"]
    form = acceptance.get("form")
    if form not in ACCEPTANCE_FORMS:
        raise ValueError(f"Unknown loop.acceptance form {form!r}; use 'baseline_relative' or "
                         "omit form.")
    assert_section_keys("loop.acceptance", set(acceptance), set(), ACCEPTANCE_FORMS[form])
    assert_retry_policy(config["loop"])
    for name, minimum in (("cross_fit_folds", 2), ("audit_sampling_window", 1),
                          ("audit_budget", 1)):
        if config["loop"][name] < minimum:
            raise ValueError(f"loop.{name} must be at least {minimum}; got "
                             f"{config['loop'][name]!r}.")
    assert_loop_mode(config["loop"].get("mode", DEPLOYMENT_MODE))
    validate_post_alarm(config["loop"])
    named_window = config["loop"].get("post_repair_reference_window")
    if named_window is not None and named_window < 1:
        raise ValueError(f"loop.post_repair_reference_window must be at least 1; got "
                         f"{named_window!r}.")
    assert_shift_specification(config["monitor"]["detectable_shift"])
    validate_event_bank(config["monitor"].get("event_bank"), config["flip"]["threshold"])
    assert_probe_spec(config["estimator"]["probe"])


def assert_retry_policy(loop_config):
    """Validate the bounded or fresh-data retry policy."""
    policy = loop_config["max_enlarging_retries"]
    if isinstance(policy, str):
        if policy not in RETRY_POLICIES:
            raise ValueError(f"Unknown loop.max_enlarging_retries policy {policy!r}; use a "
                             f"nonnegative integer or {EXHAUST_FRESH_DATA!r}.")
        if loop_config.get("mode", DEPLOYMENT_MODE) != EPISODE_MODE:
            raise ValueError(f"loop.max_enlarging_retries={policy!r} requires mode "
                             f"{EPISODE_MODE!r}; got "
                             f"{loop_config.get('mode', DEPLOYMENT_MODE)!r}.")
        if loop_config.get("post_alarm") is None:
            raise ValueError(f"loop.max_enlarging_retries={policy!r} requires "
                             "loop.post_alarm.")
        return
    if policy < 0:
        raise ValueError(f"loop.max_enlarging_retries must be nonnegative; got {policy!r}.")


def validate_post_alarm(loop_config):
    """Validate an episode-mode stationary post-alarm segment."""
    post_alarm_config = loop_config.get("post_alarm")
    if post_alarm_config is None:
        return
    if "length" in post_alarm_config:
        ladder = audit_ladder_of(loop_config["audit_budget"],
                                 loop_config["max_enlarging_retries"])
        raise ValueError(f"loop.post_alarm must not set length; the retry ladder determines "
                         f"audit length ({ladder}).")
    assert_section_keys("loop.post_alarm", set(post_alarm_config),
                        POST_ALARM_SCHEMA["required"], POST_ALARM_SCHEMA["optional"])
    composition = post_alarm_config["composition"]
    if composition != STATIONARY_COMPOSITION:
        raise ValueError(f"Unsupported loop.post_alarm.composition {composition!r}; expected "
                         f"{STATIONARY_COMPOSITION!r}.")
    if loop_config.get("mode", DEPLOYMENT_MODE) != EPISODE_MODE:
        raise ValueError(f"loop.post_alarm requires mode {EPISODE_MODE!r}; got "
                         f"{loop_config.get('mode', DEPLOYMENT_MODE)!r}.")
    if post_alarm_config["gate_labels"] < 1:
        raise ValueError(f"loop.post_alarm.gate_labels must be at least 1; got "
                         f"{post_alarm_config['gate_labels']!r}.")


def stationary_audit_traffic_of(loop_config, frame, traffic, seed):
    """Build the post-alarm audit-and-gate block factory."""
    from rcv import chaining

    if SCHEDULE_COLUMN not in traffic:
        raise ValueError(f"Stationary post-alarm composition requires traffic column "
                         f"{SCHEDULE_COLUMN!r}.")
    base_pool, attack_pool = chaining.post_alarm_pools(frame)
    audit_length = (chaining.ALL_FRESH
                    if loop_config["max_enlarging_retries"] == EXHAUST_FRESH_DATA
                    else audit_ladder_of(loop_config["audit_budget"],
                                         loop_config["max_enlarging_retries"]))
    gate_labels = loop_config["post_alarm"]["gate_labels"]

    def audit_traffic(alarmed_at, consumed_items, _rng):
        rate = float(traffic[SCHEDULE_COLUMN][alarmed_at])
        return chaining.audit_and_gate_blocks(
            {CENSUS_BASELINE_COMPONENT: (1.0, base_pool)}, attack_pool, rate, audit_length,
            gate_labels, np.random.default_rng([seed, alarmed_at, POST_ALARM_SEGMENT_SALT]),
            consumed_items=consumed_items)

    return audit_traffic


def validate_event_bank(bank_config, flip_threshold):
    """Validate monitor event-bank settings."""
    if bank_config is None:
        raise ValueError("monitor.event_bank is required.")
    assert_section_keys("monitor.event_bank", set(bank_config),
                        EVENT_BANK_SCHEMA["required"], EVENT_BANK_SCHEMA["optional"])
    boundaries = assert_boundaries(bank_config["boundaries"])
    assert_budget_allocation(bank_config["budget_allocation"])
    if flip_threshold not in boundaries:
        if bank_config.get("counterfactual_without_flip_threshold") is True:
            return
        raise ValueError(f"monitor.event_bank boundaries {boundaries} omit flip threshold "
                         f"{flip_threshold}; add it or set "
                         "counterfactual_without_flip_threshold: true.")


def validate_monitor_calibration(calibration_config):
    """Validate the source and dimensions of monitor calibration data."""
    material = calibration_config.get("material", EVALUATION_MATERIAL)
    if material not in CALIBRATION_MATERIALS:
        raise ValueError(f"Unknown monitor.calibration.material {material!r}.")
    for key, migration in RETIRED_CALIBRATION_KEYS.items():
        if key in calibration_config:
            raise ValueError(f"monitor.calibration.{key} is a retired key: {migration}")
    assert_section_keys("monitor.calibration", set(calibration_config),
                        MONITOR_CALIBRATION_SCHEMA["required"] | PREFIX_LENGTH_BY_MATERIAL[material],
                        MONITOR_CALIBRATION_SCHEMA["optional"] - {"prefix_length"})
    for name in ("stream_length", "n_streams"):
        if calibration_config[name] < 1:
            raise ValueError(f"monitor.calibration.{name} must be at least 1; got "
                             f"{calibration_config[name]!r}.")
    per_centre = assert_streams_per_centre(calibration_config["streams_per_centre"])
    if calibration_config["n_streams"] % per_centre:
        raise ValueError(f"monitor.calibration.n_streams "
                         f"({calibration_config['n_streams']}) must be divisible by "
                         f"streams_per_centre ({per_centre}).")


def post_repair_reference_window_of(loop_config, calibration_config):
    """Resolve the fresh reference-window length after repair."""
    named = loop_config.get("post_repair_reference_window")
    if named is not None:
        return named
    if loop_config.get("mode", DEPLOYMENT_MODE) == EPISODE_MODE:
        return None
    if calibration_config.get("material", EVALUATION_MATERIAL) == STREAM_PREFIX_MATERIAL:
        return calibration_config["prefix_length"]
    raise ValueError("loop.post_repair_reference_window is required when "
                     "monitor.calibration.material is 'evaluation'.")


def assert_acceptance_criteria(acceptance):
    """Validate absolute or baseline-relative acceptance criteria."""
    if acceptance.get("form") == "baseline_relative":
        for name in ("recall_slack", "over_block_inflation"):
            if not 0.0 <= acceptance.get(name, -1) <= 1.0:
                raise ValueError(f"baseline_relative acceptance.{name} must be in [0, 1]; got "
                                 f"{acceptance.get(name, -1)!r}.")
        return acceptance
    for name in ("recall_floor", "false_positive_tolerance"):
        if not 0.0 <= acceptance[name] <= 1.0:
            raise ValueError(f"acceptance.{name} must be in [0, 1]; got "
                             f"{acceptance[name]!r}.")
    return acceptance


@dataclass
class Readout:
    """Metrics for one scoring label set."""
    raw_adherence: float
    corrected_adherence: float
    caught_share: float
    caught_share_population: int
    auroc: float
    ece: float
    frame_name: str
    label_set: str
    oracle_labels_spent: int


def caught_share_of(verdict, corrected, oracle):
    """Return the share of passed unsafe items corrected by RCV."""
    passed_unsafe = (verdict == 0) & (oracle == 1)
    if not passed_unsafe.any():
        raise ValueError("Caught share is undefined: no unsafe item was passed by the "
                         "classifier.")
    return float(np.mean(corrected[passed_unsafe] == 1))


def calibration_error_of(belief, agreement, bins=15):
    """Compute equal-width expected calibration error."""
    edges = np.linspace(0.0, 1.0, bins + 1)
    which = np.clip(np.digitize(belief, edges[1:-1], right=False), 0, bins - 1)
    ece = 0.0
    for occupied in range(bins):
        rows_in_bin = which == occupied
        if not rows_in_bin.any():
            continue
        ece += rows_in_bin.mean() * abs(belief[rows_in_bin].mean() - agreement[rows_in_bin].mean())
    return float(ece)


def auroc_of(agreement, probability):
    """Return AUROC for agreement probabilities, or None for one class."""
    if len(np.unique(agreement)) < 2:
        return None
    return float(roc_auc_score(agreement, probability))


@dataclass
class MonitorState:
    """Deployed, recalibrated, and final monitor states."""
    deployed: object
    recalibrations: list
    final: object


@dataclass
class LoopState:
    events: list


@dataclass
class StudyResult:
    readout: Readout
    monitor: MonitorState
    loop: LoopState
    change_log: list
    total_oracle_labels_spent: int
    seed: int
    readouts: dict = None
    probe_selection: list = None


def loaded_frame(frame_config, estimator_config):
    """Load the declared frame and select the estimator feature column."""
    declared_digest = frame_config.get("sha256")
    if declared_digest is not None:
        actual_digest = sha256_of(frame_config["path"])
        if actual_digest != declared_digest:
            raise ValueError(f"Frame SHA-256 mismatch: configuration declares "
                             f"{declared_digest[:12]}…; {frame_config['path']} hashes to "
                             f"{actual_digest[:12]}….")
    frame = load_frame(frame_config["path"])
    feature_source = estimator_config.get("feature_source", "representation")
    if feature_source == "representation":
        return frame
    if feature_source not in frame:
        raise ValueError(f"feature_source {feature_source!r} is not a frame column.")
    return dict(frame, representation=np.asarray(frame[feature_source]).reshape(
        len(frame["verdict"]), -1))


def estimator_builder(estimator_config, seed):
    """Return a factory for fresh estimators seeded for this draw."""
    specification = {key: value for key, value in estimator_config.items()
                     if key != "feature_source"}
    return lambda: Estimator(**specification, random_state=seed)


def fitting_slices_of(serving, split_config, labels, seed):
    """Split serving data into whole-family training, calibration, and evaluation slices."""
    strata = None
    strata_axes = split_config.get("strata")
    if strata_axes is not None:
        if strata_axes != ["verdict", "agreement"]:
            raise ValueError(f"Unsupported split.strata {strata_axes!r}; expected "
                             "['verdict', 'agreement'].")
        strata = serving["verdict"] * 2 + agreement_of(serving, labels)
    split = draw_group_blocked_split(serving["family"], split_config["evaluation_fraction"],
                                     split_config["calibration_fraction"], seed, strata=strata)
    return (rows(serving, split["training"]), rows(serving, split["calibration"]),
            rows(serving, split["evaluation"]))


def readouts_of(evaluation, corrected, frame_config, scoring_axes, labels, labels_spent,
                probability):
    """Compute a readout for each requested scoring label set."""
    readouts = {}
    for axis in scoring_axes:
        if axis not in evaluation:
            raise ValueError(f"Scoring label set {axis!r} is not a frame column.")
        agreement = (evaluation["verdict"] == evaluation[axis]).astype(np.int64)
        readouts[axis] = Readout(
            raw_adherence=float(np.mean(evaluation["verdict"] == evaluation[axis])),
            corrected_adherence=float(np.mean(corrected == evaluation[axis])),
            caught_share=caught_share_of(evaluation["verdict"], corrected, evaluation[axis]),
            caught_share_population=int(((evaluation["verdict"] == 0)
                                         & (evaluation[axis] == 1)).sum()),
            auroc=auroc_of(agreement, probability),
            ece=calibration_error_of(probability, agreement),
            frame_name=frame_config["name"],
            label_set=axis,
            oracle_labels_spent=labels_spent,
        )
    if labels not in readouts:
        raise ValueError(f"Fitting label set {labels!r} must appear in scoring_labels.")
    return readouts


def calibrated_monitor(monitor_config, estimator, flip_rule, evaluation, evaluation_probability,
                       traffic, seed, fitted=None):
    """Calibrate the declared monitor from evaluation data or a stream prefix."""
    monitor = declared_monitor(monitor_config)
    calibration_config = monitor_config["calibration"]
    material = calibration_config.get("material", EVALUATION_MATERIAL)
    if material == STREAM_PREFIX_MATERIAL:
        prefix_length = calibration_config["prefix_length"]
        if prefix_length >= len(traffic["verdict"]):
            raise ValueError(f"prefix_length must be smaller than stream length; got "
                             f"{prefix_length} for {len(traffic['verdict'])} rows.")
        reference_slice = rows(traffic, slice(0, prefix_length))
        prefix_drift_checked = assert_prefix_is_drift_free(reference_slice, prefix_length)
        traffic = rows(traffic, slice(prefix_length, len(traffic["verdict"])))
        reference_probability = estimator.probability_of_agreement(
            reference_slice["representation"], reference_slice["verdict"])
        monitored_offset = prefix_length
    elif material == EVALUATION_MATERIAL:
        reference_slice = evaluation
        reference_probability = evaluation_probability
        prefix_drift_checked = None
        monitored_offset = 0
    else:
        raise ValueError(f"Unknown monitor.calibration.material {material!r}.")

    monitor.calibrate(reference_probability, reference_slice["verdict"],
                      calibration_config["stream_length"], calibration_config["n_streams"], seed,
                      prefix_drift_checked=prefix_drift_checked,
                      monitored_length=len(traffic["verdict"]),
                      fitted_family_overlap=(None if fitted is None
                                             else fitted_family_overlap(fitted, traffic)))
    return monitor, traffic, reference_slice, monitored_offset


def declared_monitor(monitor_config):
    """Construct the monitor configured for this study."""
    bank_config = monitor_config["event_bank"]
    return BeliefBankMonitor(calibration_level_of(monitor_config["quiet_horizon_confidence"]),
                             monitor_config["detectable_shift"],
                             boundaries=bank_config["boundaries"],
                             streams_per_centre=monitor_config["calibration"][
                                 "streams_per_centre"],
                             budget_allocation=bank_config["budget_allocation"])


def calibration_level_of(quiet_horizon_confidence):
    """Convert quiet-horizon confidence q to calibration level 1 - q."""
    return round(1.0 - quiet_horizon_confidence, 12)


def fitted_family_overlap(fitted, traffic):
    """Report fitted-family overlap across traffic and pool sides."""
    seen = np.isin(traffic["family"], fitted["family"])
    overlap = {"all": float(np.mean(seen)), "base": None, "drift": None}
    if "is_drift_item" in traffic:
        drift = np.asarray(traffic["is_drift_item"]).astype(bool)
        overlap["base"] = float(np.mean(seen[~drift])) if (~drift).any() else None
        overlap["drift"] = float(np.mean(seen[drift])) if drift.any() else None
    return overlap


def assert_prefix_is_drift_free(prefix, prefix_length):
    """Require a recorded stream prefix to be drift-free."""
    scheduled_rate = prefix.get("stream_lambda")
    drew_from_the_drift_pool = prefix.get("is_drift_item")
    if scheduled_rate is None and drew_from_the_drift_pool is None:
        return False
    contaminated = np.zeros(prefix_length, dtype=bool)
    if scheduled_rate is not None:
        contaminated |= np.asarray(scheduled_rate) > 0.0
    if drew_from_the_drift_pool is not None:
        contaminated |= np.asarray(drew_from_the_drift_pool).astype(bool)
    if contaminated.any():
        maximum = ("unrecorded" if scheduled_rate is None
                   else f"{float(np.max(np.asarray(scheduled_rate))):.6f}")
        raise ValueError(f"Calibration prefix has {int(contaminated.sum())} contaminated "
                         f"positions (maximum stream_lambda {maximum}); prefix reference "
                         "requires no contamination.")
    return True


def run_study(config, seed=None):
    """Run one validated study configuration for one seed."""
    if not isinstance(config, dict):
        config = yaml.safe_load(Path(config).read_text())
    validate_config(config)
    if seed is None:
        if len(config["seeds"]) > 1:
            raise ValueError(f"run_study needs an explicit seed when config.seeds has "
                             f"{len(config['seeds'])} values.")
        seed = config["seeds"][0]
    labels = config["frame"]["fitting_labels"]
    assert_acceptance_criteria(config["loop"]["acceptance"])

    frame = loaded_frame(config["frame"], config["estimator"])
    serving = rows(frame, ~frame["is_stream"])
    traffic = rows(frame, frame["is_stream"])

    training, calibration_slice, evaluation = fitting_slices_of(serving, config["split"], labels,
                                                                seed)
    assert_all_four_cases(slice_of(evaluation, labels), "evaluation")

    build_estimator = estimator_builder(config["estimator"], seed)
    estimator = fit_on_slices(build_estimator(), training, calibration_slice, labels)
    probe_selection = estimator.probe_selection()
    flip_rule = FlipRule(config["flip"]["threshold"])

    probability = estimator.probability_of_agreement(evaluation["representation"],
                                                     evaluation["verdict"])
    corrected = flip_rule.corrected_verdict(evaluation["verdict"], probability)

    labels_spent = (len(training["verdict"]) + len(calibration_slice["verdict"])
                    + len(evaluation["verdict"]))
    readouts = readouts_of(evaluation, corrected, config["frame"],
                           config["frame"].get("scoring_labels", [labels]), labels, labels_spent,
                           probability)
    readout = readouts[labels]

    if len(traffic["verdict"]) == 0:
        return StudyResult(seed=seed, readout=readout, readouts=readouts,
                           monitor=MonitorState(deployed=None, recalibrations=[], final=None),
                           loop=LoopState(events=[]), change_log=[],
                           total_oracle_labels_spent=readout.oracle_labels_spent,
                           probe_selection=probe_selection)

    monitor, traffic, _deployment_reference, monitored_offset = calibrated_monitor(
        config["monitor"], estimator, flip_rule, evaluation, probability, traffic, seed,
        fitted=concatenated(training, calibration_slice))
    deployed = monitor.calibration

    unsafe = evaluation[labels] == 1
    baseline = {"recall": float(np.mean(corrected[unsafe] == 1)),
                "over_block": float(np.mean(corrected[~unsafe] == 1))}

    loop_config = config["loop"]
    estimator, events, change_log, _carry_out = watch_traffic(
        build_estimator, estimator, training, calibration_slice, traffic, monitor, flip_rule,
        loop_config["audit_sampling_window"], loop_config["audit_budget"],
        loop_config["acceptance"], loop_config["max_enlarging_retries"],
        loop_config["cross_fit_folds"], seed, labels,
        observe_only=loop_config.get("observe_only", False), baseline=baseline,
        post_repair_reference_window=post_repair_reference_window_of(
            loop_config, config["monitor"]["calibration"]),
        monitored_offset=monitored_offset,
        mode=loop_config.get("mode", DEPLOYMENT_MODE),
        audit_traffic=(None if loop_config.get("post_alarm") is None
                       else stationary_audit_traffic_of(loop_config, frame, traffic, seed)))

    return StudyResult(seed=seed, readout=readout, readouts=readouts,
                       monitor=MonitorState(
                           deployed=deployed,
                           recalibrations=[entry for entry in change_log
                                           if entry.calibration is not None],
                           final=monitor.calibration),
                       loop=LoopState(events=events), change_log=change_log,
                       total_oracle_labels_spent=readout.oracle_labels_spent + sum(
                           entry.oracle_labels_spent for entry in change_log),
                       probe_selection=probe_selection)
