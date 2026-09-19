"""Monitor traffic and gate estimator repairs after drift alarms."""

from dataclasses import dataclass

import numpy as np

from rcv.frames import concatenated, fit_on_slices, rows
from rcv.splitting import cross_fit_fold_of

FITTED_ITEM_COLUMN = "item_id"

FITTING_SIDE = "fitting"
CALIBRATING_SIDE = "calibrating"

DEPLOYMENT_MODE = "deployment"
EPISODE_MODE = "episode"
LOOP_MODES = (DEPLOYMENT_MODE, EPISODE_MODE)

EXHAUST_FRESH_DATA = "exhaust-fresh-data"
RETRY_POLICIES = (EXHAUST_FRESH_DATA,)

SCHEDULE_COLUMN = "stream_lambda"

POST_ALARM_PROVENANCE_KEYS = ("rate", "audit_length", "gate_length", "eligible_attack",
                              "eligible_base", "audit_reused_rows", "gate_reused_rows",
                              "audit_sha256", "gate_sha256")


@dataclass
class LoopEvent:
    """Record loop events with positions in monitored traffic and the full stream."""
    kind: str
    detail: str
    regime: int = None
    monitored_position: int = None
    stream_position: int = None
    boundary: float = None
    gate_recall: float = None
    gate_over_block: float = None
    gate_passed: bool = None
    audit_labels: int = None


REFERENCE_WINDOW_EXHAUSTED = "reference_window_exhausted"
FRESH_DATA_EXHAUSTED = "fresh_data_exhausted"


@dataclass
class AuditCarry:
    """Retain accepted audit rows and item assignments between episodes."""
    fitting: dict = None
    calibrating: dict = None
    side_of_item: dict = None


@dataclass
class ChangeLogEntry:
    what: str
    trigger: str
    oracle_labels_spent: int
    monitored_position: int = None
    stream_position: int = None
    calibration: object = None
    replay_seed: object = None
    reference_audit_overlap_fraction: float = None
    reference_window_length: int = None
    reference_window_monitored_span: list = None
    reference_window_skipped_positions: int = None
    reference_window_eligible_positions: int = None
    probe_selection: list = None
    gate_recall: float = None
    gate_over_block: float = None
    post_alarm_rate: float = None
    post_alarm_length: int = None
    post_alarm_reused_rows: int = None
    post_alarm_gate_length: int = None
    gate_labels: int = None
    old_probe_gate_recall: float = None
    old_probe_gate_over_block: float = None


def draw_audit_indices(post_alarm_start, traffic_length, sampling_window, budget, rng):
    """Sample post-alarm positions uniformly without replacement; return indices in stream order."""
    window_end = min(post_alarm_start + sampling_window, traffic_length)
    candidate_indices = np.arange(post_alarm_start, window_end)
    draw_size = min(budget, len(candidate_indices))
    return np.sort(rng.choice(candidate_indices, size=draw_size, replace=False))


def enlarged_audit(audit, traffic, consumed_up_to, sampling_window, budget, rng):
    """Append the next window's sample; return the audit and window end.

    Return (None, consumed_up_to) if traffic is exhausted.
    """
    further = draw_audit_indices(consumed_up_to, len(traffic["verdict"]), sampling_window, budget,
                                 rng)
    if len(further) == 0:
        return None, consumed_up_to
    return (concatenated(audit, rows(traffic, further)),
            min(consumed_up_to + sampling_window, len(traffic["verdict"])))


def assert_loop_mode(mode):
    if mode not in LOOP_MODES:
        raise ValueError(f"Unknown loop mode {mode!r}; expected "
                         f"{DEPLOYMENT_MODE!r} or {EPISODE_MODE!r}.")
    return mode


def assign_unseen_items(item_of_row, drawn_order, side_of_item):
    """Assign unseen items in place, halving in draw order; fitting gets the odd item."""
    unseen, seen_here = [], set()
    for row in drawn_order:
        item = item_of_row[row]
        if item not in side_of_item and item not in seen_here:
            seen_here.add(item)
            unseen.append(item)
    cut = len(unseen) - len(unseen) // 2
    for rank, item in enumerate(unseen):
        side_of_item[item] = FITTING_SIDE if rank < cut else CALIBRATING_SIDE


def split_audit_batch(audit, rng, side_of_item):
    """Return fitting and calibration frames, extending the shared item assignments.

    Stable assignments keep repeated items on the same side across audit batches.
    """
    if FITTED_ITEM_COLUMN not in audit:
        raise ValueError(
            f"Audit batch is missing required column {FITTED_ITEM_COLUMN!r}.")
    item_of_row = audit[FITTED_ITEM_COLUMN].tolist()
    drawn_order = rng.permutation(len(audit["verdict"]))
    assign_unseen_items(item_of_row, drawn_order, side_of_item)
    on_the_fitting_side = np.array([side_of_item[item_of_row[row]] == FITTING_SIDE
                                    for row in drawn_order], dtype=bool)
    return (rows(audit, drawn_order[on_the_fitting_side]),
            rows(audit, drawn_order[~on_the_fitting_side]))


def gate_uncomposable_error():
    """Load the exception lazily so ordinary monitoring does not import chain composition."""
    from rcv.chaining import GateUncomposable

    return GateUncomposable


def post_alarm_provenance_of(reported):
    """Require complete provenance so missing measurements cannot appear as zero reuse."""
    missing = [name for name in POST_ALARM_PROVENANCE_KEYS if name not in reported]
    if missing:
        raise ValueError(
            f"Post-alarm provenance is missing fields: {missing}.")
    return {"post_alarm_rate": float(reported["rate"]),
            "post_alarm_length": int(reported["audit_length"]),
            "post_alarm_gate_length": int(reported["gate_length"]),
            "post_alarm_reused_rows": int(reported["audit_reused_rows"]
                                          + reported["gate_reused_rows"])}


def episode_label_bill(audit, gate_block):
    """Count labeled audit and gate positions, including repeated items."""
    return len(audit["verdict"]) + (0 if gate_block is None else len(gate_block["verdict"]))


def audit_ladder_of(audit_budget, max_enlarging_retries):
    return audit_budget * (1 + max_enlarging_retries)


def audit_sizes_of(audit_budget, max_enlarging_retries, composed_length):
    """Return cumulative audit sizes, including any partial final budget when exhausting data."""
    if max_enlarging_retries == EXHAUST_FRESH_DATA:
        whole_budgets = list(range(audit_budget, composed_length, audit_budget))
        return whole_budgets + [composed_length]
    return [min((attempt + 1) * audit_budget, composed_length)
            for attempt in range(max_enlarging_retries + 1)]


def assert_segment_feeds_the_attempts(segment, audit_budget, max_enlarging_retries):
    length = len(segment["verdict"])
    if max_enlarging_retries == EXHAUST_FRESH_DATA:
        if length < 1:
            raise ValueError(
                f"Post-alarm audit segment is empty under {EXHAUST_FRESH_DATA!r}.")
        return segment
    ladder = audit_ladder_of(audit_budget, max_enlarging_retries)
    if length < ladder:
        raise ValueError(
            f"Post-alarm audit segment has {length} positions; {ladder} required "
            f"({audit_budget} per attempt × {1 + max_enlarging_retries} attempts).")
    return segment


def assert_segment_carries_the_fitting_columns(segment, *fitting_slices):
    """Validate before concatenation, which follows the first frame's columns."""
    required = set().union(*(set(fitting_slice) for fitting_slice in fitting_slices))
    missing = sorted(required - set(segment))
    if missing:
        raise ValueError(
            f"Post-alarm block is missing fitting columns: {missing}.")
    return segment


def standing_slice(deployment_slice, accumulated):
    return (deployment_slice if accumulated is None
            else concatenated(deployment_slice, accumulated))


def carried_audits(carry):
    if carry.fitting is None:
        return carry.calibrating
    if carry.calibrating is None:
        return carry.fitting
    return concatenated(carry.fitting, carry.calibrating)


def reference_audit_overlap(reference_slice, audit):
    """Return the fraction of reference rows with audit item IDs, or None if IDs are missing."""
    if FITTED_ITEM_COLUMN not in reference_slice or FITTED_ITEM_COLUMN not in audit:
        return None
    return float(np.mean(np.isin(reference_slice[FITTED_ITEM_COLUMN],
                                 audit[FITTED_ITEM_COLUMN])))


def fitted_item_ids(audit):
    if FITTED_ITEM_COLUMN not in audit:
        raise ValueError(
            f"Audit window is missing required column {FITTED_ITEM_COLUMN!r}.")
    return np.asarray(audit[FITTED_ITEM_COLUMN])


def eligible_reference_positions(traffic, opens_at, fitted_items):
    """Return positions at or after opens_at, excluding fitted item IDs."""
    if FITTED_ITEM_COLUMN not in traffic:
        raise ValueError(
            f"Traffic is missing required column {FITTED_ITEM_COLUMN!r}.")
    candidates = np.arange(opens_at, len(traffic["verdict"]))
    return candidates[~np.isin(traffic[FITTED_ITEM_COLUMN][candidates], fitted_items)]


def fresh_reference_window(traffic, positions, fitted_items):
    """Build a reference window free of fitted item IDs.

    Check identity because the stream samples with replacement.
    """
    window = rows(traffic, positions)
    if FITTED_ITEM_COLUMN not in window:
        raise ValueError(
            f"Reference window is missing required column {FITTED_ITEM_COLUMN!r}.")
    contained = np.isin(window[FITTED_ITEM_COLUMN], fitted_items)
    if contained.any():
        raise ValueError(
            f"Reference window of {len(positions)} positions includes "
            f"{int(contained.sum())} positions with fitted item IDs.")
    return window


def update_meets_relative_criteria(update_recall, update_over_block, baseline_recall,
                                   baseline_over_block, recall_slack, over_block_inflation):
    return bool(update_recall >= baseline_recall - recall_slack
                and update_over_block <= baseline_over_block + over_block_inflation)


def watch_traffic(build_estimator, estimator, training, calibration_slice, traffic, monitor,
                  flip_rule, audit_sampling_window, audit_budget, acceptance,
                  max_enlarging_retries, cross_fit_folds, seed, labels, observe_only=False,
                  baseline=None, post_repair_reference_window=None, monitored_offset=0,
                  mode=DEPLOYMENT_MODE, carry=None, audit_traffic=None):
    """Monitor traffic, audit alarms, and return the estimator, events, change log, and carry.

    Pass frames of aligned NumPy arrays and a factory for fresh estimators.
    The labels argument names the target column.

    Deployment mode resumes after an accepted repair and a fresh reference
    window; episode mode ends at its first outcome. Observe-only stops at
    the first alarm without purchasing labels.

    Default audits sample successive post-alarm windows and use cross-fitting.
    In episode mode, the optional audit callback returns an audit segment,
    a held-out gate block, and provenance. It receives the alarm index,
    previously assigned item IDs, and RNG. Retries are bounded by count or,
    for composed audits, exhaustion of available data.

    Updates the monitor and any existing item-assignment map in place.
    """
    assert_loop_mode(mode)
    if max_enlarging_retries == EXHAUST_FRESH_DATA and audit_traffic is None:
        raise ValueError(
            f"Retry policy {EXHAUST_FRESH_DATA!r} requires an audit_traffic callback.")
    if audit_traffic is not None and mode != EPISODE_MODE:
        raise ValueError(
            f"Invalid mode {mode!r} for composed audits; expected {EPISODE_MODE!r}.")
    events, change_log = [], []
    rng = np.random.default_rng(seed)
    traffic_length = len(traffic["verdict"])

    carry = AuditCarry() if carry is None else carry
    accumulated_fitting, accumulated_calibrating = carry.fitting, carry.calibrating
    accumulated_audits = carried_audits(carry)
    side_of_item = {} if carry.side_of_item is None else carry.side_of_item
    standing_fitting = standing_slice(training, accumulated_fitting)
    standing_calibrating = standing_slice(calibration_slice, accumulated_calibrating)

    observed_up_to = 0
    while observed_up_to < traffic_length:
        chunk = rows(traffic, slice(observed_up_to, observed_up_to + audit_sampling_window))
        chunk_end = observed_up_to + len(chunk["verdict"])
        probability = estimator.probability_of_agreement(chunk["representation"],
                                                         chunk["verdict"])
        alarm = monitor.observe(probability, chunk["verdict"])
        if alarm is None:
            observed_up_to = chunk_end
            continue
        regime, boundary = alarm.regime, alarm.boundary
        wire = f"regime {regime} at boundary {boundary}"
        alarmed_at = observed_up_to + alarm.item_index

        events.append(LoopEvent("alarm",
                                f"regime {regime}, boundary {boundary}, monitored position {alarmed_at}",
                                regime=regime, boundary=boundary,
                                monitored_position=alarmed_at,
                                stream_position=monitored_offset + alarmed_at))
        if observe_only:
            events.append(LoopEvent("watch_ended", "observe-only: first alarm recorded"))
            break
        segment = gate_block = episode_provenance = None
        if audit_traffic is None:
            drawn = draw_audit_indices(alarmed_at + 1, traffic_length, audit_sampling_window,
                                       audit_budget, rng)
            if len(drawn) == 0:
                events.append(LoopEvent("watch_ended",
                                        "no post-alarm audit positions sampled"))
                break
            audit = rows(traffic, drawn)
            consumed_up_to = min(alarmed_at + 1 + audit_sampling_window, traffic_length)
            audit_material = "sampled uniformly within post-alarm windows"
            reading_kind, reading_labels = "out-of-fold", "audit labels"
        else:
            try:
                segment, gate_block, reported = audit_traffic(alarmed_at,
                                                              frozenset(side_of_item), rng)
            except gate_uncomposable_error() as refusal:
                events.append(LoopEvent("gate_uncomposable", str(refusal),
                                        monitored_position=alarmed_at,
                                        stream_position=monitored_offset + alarmed_at))
                change_log.append(ChangeLogEntry(
                    what="gate_uncomposable", trigger=f"alarm in {wire}",
                    oracle_labels_spent=0, monitored_position=alarmed_at,
                    stream_position=monitored_offset + alarmed_at))
                break
            for block in (segment, gate_block):
                assert_segment_carries_the_fitting_columns(block, standing_fitting,
                                                           standing_calibrating)
            assert_segment_feeds_the_attempts(segment, audit_budget, max_enlarging_retries)
            attempt_sizes = audit_sizes_of(audit_budget, max_enlarging_retries,
                                           len(segment["verdict"]))
            audit = rows(segment, slice(0, attempt_sizes[0]))
            audit_material = "taken in order from composed segment"
            reading_kind, reading_labels = "held-out", "gate labels"
            old_probe = held_out_gate_reading(estimator, gate_block, flip_rule, acceptance,
                                              labels, baseline)
            episode_provenance = {
                **post_alarm_provenance_of(reported),
                "gate_labels": int(len(gate_block["verdict"])),
                "old_probe_gate_recall": old_probe["recall"],
                "old_probe_gate_over_block": old_probe["over_block"]}
            events.append(LoopEvent(
                "gate_block_composed",
                f"held-out gate labels: {len(gate_block['verdict'])}; current estimator recall "
                f"{old_probe['recall']:.4f}, over-block {old_probe['over_block']:.4f}",
                gate_recall=old_probe["recall"], gate_over_block=old_probe["over_block"],
                monitored_position=alarmed_at,
                stream_position=monitored_offset + alarmed_at))
        events.append(LoopEvent("audit", f"audit labels: {len(audit['verdict'])}; {audit_material}"))

        accepted = False
        for attempt in range(len(attempt_sizes) if segment is not None
                             else max_enlarging_retries + 1):
            if attempt > 0:
                if segment is None:
                    enlarged, consumed_up_to = enlarged_audit(audit, traffic, consumed_up_to,
                                                              audit_sampling_window, audit_budget,
                                                              rng)
                    if enlarged is None:
                        break
                    audit = enlarged
                else:
                    audit = rows(segment, slice(0, attempt_sizes[attempt]))
                events.append(LoopEvent("audit", f"audit labels: {len(audit['verdict'])} "
                                                 f"after enlargement; {audit_material}"))
            if gate_block is None:
                candidate = None
                reading = gate_reading(
                    build_estimator, standing_fitting, standing_calibrating, audit, flip_rule,
                    acceptance, cross_fit_folds, rng, labels, baseline)
            else:
                fitting_half, calibrating_half = split_audit_batch(audit, rng, side_of_item)
                candidate = fit_on_slices(build_estimator(),
                                          concatenated(standing_fitting, fitting_half),
                                          concatenated(standing_calibrating, calibrating_half),
                                          labels)
                reading = held_out_gate_reading(candidate, gate_block, flip_rule, acceptance,
                                                labels, baseline)
            accepted = reading["passed"]
            events.append(LoopEvent(
                "acceptance_evaluated",
                f"attempt {attempt + 1}: {reading_kind} recall {reading['recall']:.4f}, "
                f"over-block {reading['over_block']:.4f} on {reading['n_labels']} "
                f"{reading_labels} — {'passed' if accepted else 'failed'}",
                gate_recall=reading["recall"], gate_over_block=reading["over_block"],
                gate_passed=accepted, audit_labels=int(len(audit["verdict"]))))
            if accepted:
                break
        if not accepted:
            if max_enlarging_retries == EXHAUST_FRESH_DATA:
                terminal = FRESH_DATA_EXHAUSTED
                ending = (f"No candidate passed after {len(audit['verdict'])} audit labels; "
                          f"fresh audit data exhausted after alarm in {wire}. Watch ended.")
            else:
                terminal = "escalation_demanded"
                ending = (f"No attempt passed after alarm in {wire}; "
                          f"watch ended without fine-tuning.")
            events.append(LoopEvent(terminal, ending))
            change_log.append(ChangeLogEntry(what=terminal,
                                             trigger=f"alarm in {wire}",
                                             oracle_labels_spent=episode_label_bill(audit,
                                                                                    gate_block),
                                             monitored_position=alarmed_at,
                                             stream_position=monitored_offset + alarmed_at,
                                             **(episode_provenance or {})))
            break

        fresh_fitting, fresh_calibrating = ((fitting_half, calibrating_half) if candidate
                                            else split_audit_batch(audit, rng, side_of_item))
        accumulated_fitting = (fresh_fitting if accumulated_fitting is None
                               else concatenated(accumulated_fitting, fresh_fitting))
        accumulated_calibrating = (fresh_calibrating if accumulated_calibrating is None
                                   else concatenated(accumulated_calibrating, fresh_calibrating))
        accumulated_audits = (audit if accumulated_audits is None
                              else concatenated(accumulated_audits, audit))
        standing_fitting = standing_slice(training, accumulated_fitting)
        standing_calibrating = standing_slice(calibration_slice, accumulated_calibrating)

        estimator = candidate or fit_on_slices(build_estimator(), standing_fitting,
                                               standing_calibrating, labels)
        events.append(LoopEvent("repair_accepted", f"after alarm in {wire}"))

        if mode == EPISODE_MODE:
            change_log.append(accepted_repair_entry(
                wire, episode_label_bill(audit, gate_block),
                alarmed_at if segment is not None else consumed_up_to,
                monitored_offset, estimator.probe_selection(), reading,
                **(episode_provenance or {})))
            events.append(LoopEvent("watch_ended", "episode resolved: repaired"))
            break

        if post_repair_reference_window is None:
            raise ValueError(
                "Resuming monitoring requires post_repair_reference_window.")
        fitted = fitted_item_ids(accumulated_audits)
        eligible = eligible_reference_positions(traffic, consumed_up_to, fitted)
        if len(eligible) < post_repair_reference_window:
            change_log.append(accepted_repair_entry(
                wire, episode_label_bill(audit, gate_block), consumed_up_to, monitored_offset,
                estimator.probe_selection(), reading,
                reference_window_length=post_repair_reference_window))
            events.append(LoopEvent(
                REFERENCE_WINDOW_EXHAUSTED,
                f"Insufficient reference data: {len(eligible)} eligible positions, "
                f"{post_repair_reference_window} required after repair in {wire}. Watch ended."))
            change_log.append(ChangeLogEntry(
                what=REFERENCE_WINDOW_EXHAUSTED, trigger=f"alarm in {wire}",
                oracle_labels_spent=0,
                monitored_position=consumed_up_to,
                stream_position=monitored_offset + consumed_up_to,
                reference_window_length=post_repair_reference_window,
                reference_window_eligible_positions=int(len(eligible))))
            break
        window_positions = eligible[:post_repair_reference_window]
        window = fresh_reference_window(traffic, window_positions, fitted)

        standing = monitor.calibration
        reference_probability = estimator.probability_of_agreement(
            window["representation"], window["verdict"])
        era = len(change_log) + 1
        replay_seed = [seed, era]
        monitor.calibrate(reference_probability, window["verdict"],
                          standing.replay_stream_length, standing.n_streams, replay_seed,
                          prefix_drift_checked=standing.prefix_drift_checked,
                          monitored_length=standing.monitored_length,
                          fitted_family_overlap=standing.fitted_family_overlap)
        opened_at, closed_at = int(window_positions[0]), int(window_positions[-1])
        events.append(LoopEvent("reference_rederived",
                                f"monitor recalibrated on {post_repair_reference_window} "
                                f"reference positions, monitored span [{opened_at}, {closed_at}]"))

        change_log.append(accepted_repair_entry(
            wire, episode_label_bill(audit, gate_block), consumed_up_to, monitored_offset,
            estimator.probe_selection(), reading,
            calibration=monitor.calibration, replay_seed=replay_seed,
            reference_audit_overlap_fraction=reference_audit_overlap(window, accumulated_audits),
            reference_window_length=post_repair_reference_window,
            reference_window_monitored_span=[opened_at, closed_at],
            reference_window_skipped_positions=(closed_at - opened_at + 1
                                                - post_repair_reference_window)))
        # Keep the reference window out of the new monitoring period.
        observed_up_to = closed_at + 1

    return estimator, events, change_log, AuditCarry(fitting=accumulated_fitting,
                                                     calibrating=accumulated_calibrating,
                                                     side_of_item=side_of_item)


def accepted_repair_entry(wire, oracle_labels_spent, takes_effect_at, monitored_offset,
                          probe_selection, reading, **era):
    return ChangeLogEntry(
        what="repair", trigger=f"alarm in {wire}",
        oracle_labels_spent=oracle_labels_spent,
        monitored_position=takes_effect_at,
        stream_position=monitored_offset + takes_effect_at,
        probe_selection=probe_selection,
        gate_recall=reading["recall"], gate_over_block=reading["over_block"], **era)


def meets_acceptance(recall, over_block, acceptance, baseline):
    """Use absolute limits or additive tolerances around the pre-drift baseline."""
    if acceptance.get("form") == "baseline_relative":
        return update_meets_relative_criteria(
            recall, over_block, baseline["recall"], baseline["over_block"],
            acceptance["recall_slack"], acceptance["over_block_inflation"])
    return bool(recall >= acceptance["recall_floor"]
                and over_block <= acceptance["false_positive_tolerance"])


def acceptance_reading(corrected, block, labels, acceptance, baseline=None):
    """Return recall, over-blocking, pass/fail, and label count.

    Require both safe (0) and unsafe (1) target labels.
    """
    unsafe = block[labels] == 1
    if not unsafe.any() or unsafe.all():
        raise RuntimeError("Acceptance evaluation requires both safe (0) and unsafe (1) labels.")
    recall = float(np.mean(corrected[unsafe] == 1))
    over_block = float(np.mean(corrected[~unsafe] == 1))
    return {"recall": recall, "over_block": over_block,
            "passed": meets_acceptance(recall, over_block, acceptance, baseline),
            "n_labels": int(len(block["verdict"]))}


def held_out_gate_reading(candidate, gate_block, flip_rule, acceptance, labels, baseline=None):
    """Evaluate a fitted candidate.

    Supply gate items excluded from its fitting and calibration data.
    """
    probability = candidate.probability_of_agreement(gate_block["representation"],
                                                     gate_block["verdict"])
    corrected = flip_rule.corrected_verdict(gate_block["verdict"], probability)
    return acceptance_reading(corrected, gate_block, labels, acceptance, baseline)


def gate_reading(build_estimator, training, calibration_slice, audit, flip_rule, acceptance,
                 cross_fit_folds, rng, labels, baseline=None):
    """Evaluate acceptance using audit predictions cross-fitted by family."""
    corrected = _corrected_out_of_fold(build_estimator, training, calibration_slice, audit,
                                       flip_rule, cross_fit_folds, rng, labels)
    return acceptance_reading(corrected, audit, labels, acceptance, baseline)


def _update_meets_criteria_cross_fit(build_estimator, training, calibration_slice, audit,
                                     flip_rule, acceptance, cross_fit_folds, rng, labels,
                                     baseline=None):
    return gate_reading(build_estimator, training, calibration_slice, audit, flip_rule,
                        acceptance, cross_fit_folds, rng, labels, baseline)["passed"]


def _corrected_out_of_fold(build_estimator, training, calibration_slice, audit, flip_rule,
                           cross_fit_folds, rng, labels):
    """Cross-fit by audit family, augmenting standing training with other folds
    and reusing standing calibration.
    """
    fold_of = cross_fit_fold_of(audit["family"], cross_fit_folds, rng)
    corrected = np.empty(len(audit["verdict"]), dtype=np.int64)
    for fold in range(cross_fit_folds):
        held_out = fold_of == fold
        if not held_out.any():
            continue
        candidate = fit_on_slices(build_estimator(),
                                  concatenated(training, rows(audit, ~held_out)),
                                  calibration_slice, labels)
        probability = candidate.probability_of_agreement(audit["representation"][held_out],
                                                         audit["verdict"][held_out])
        corrected[held_out] = flip_rule.corrected_verdict(audit["verdict"][held_out], probability)
    return corrected
