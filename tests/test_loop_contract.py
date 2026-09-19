from typing import NamedTuple

import numpy as np
import pytest

from rcv.flip import FlipRule
from rcv.loop import (
    EPISODE_MODE,
    AuditCarry,
    ChangeLogEntry,
    LoopEvent,
    _update_meets_criteria_cross_fit,
    gate_reading,
    update_meets_relative_criteria,
    watch_traffic,
)

WINDOW = 10
TRAFFIC_LENGTH = 60
ROW_ID_COLUMN = 0
FLIP_THRESHOLD = 0.5
AGREEING = 0.9
DISAGREEING = 0.1
REFERENCE_WINDOW = 10


class Alarm(NamedTuple):
    regime: int
    item_index: int
    boundary: float = FLIP_THRESHOLD


class StandingCalibration(NamedTuple):
    replay_stream_length: int
    n_streams: int
    prefix_drift_checked: bool = None
    monitored_length: int = None
    fitted_family_overlap: dict = None


class ScriptedMonitor:
    OBSERVATION_CEILING = 200

    def __init__(self, alarms=None, stream_length=11, n_streams=13,
                 prefix_drift_checked=None, monitored_length=None,
                 fitted_family_overlap=None):
        self.alarms = dict(alarms or {})
        self.observed_verdicts = []
        self.calibrations = []
        self.calibration = StandingCalibration(stream_length, n_streams, prefix_drift_checked,
                                               monitored_length, fitted_family_overlap)

    def observe(self, probability, verdicts):
        if len(self.observed_verdicts) >= self.OBSERVATION_CEILING:
            raise AssertionError("the watch is not advancing through the traffic")
        self.observed_verdicts.append(np.asarray(verdicts).copy())
        return self.alarms.get(len(self.observed_verdicts) - 1)

    def calibrate(self, probability, verdicts, stream_length, n_streams, seed,
                  prefix_drift_checked=None, monitored_length=None,
                  fitted_family_overlap=None):
        self.calibrations.append({"beliefs": len(probability), "verdicts": len(verdicts),
                                  "stream_length": stream_length, "n_streams": n_streams,
                                  "seed": seed,
                                  "prefix_drift_checked": prefix_drift_checked,
                                  "monitored_length": monitored_length,
                                  "fitted_family_overlap": fitted_family_overlap})


class ScriptedEstimator:
    def __init__(self, disagreeing_rows=()):
        self.disagreeing_rows = np.asarray(disagreeing_rows, dtype=np.int64)
        self.fitted_on = None
        self.scored = None
        self.scored_calls = []
        self.scored_verdicts = []

    def fit(self, fitting_slice, calibration_slice):
        self.fitted_on = row_ids(fitting_slice.representation)
        return self

    def probability_of_agreement(self, representation, verdict):
        ids = row_ids(representation)
        self.scored = ids
        self.scored_calls.append(ids)
        self.scored_verdicts.append(np.asarray(verdict).copy())
        return np.where(np.isin(ids, self.disagreeing_rows), DISAGREEING, AGREEING)

    def probe_selection(self):
        return None


def row_ids(representation):
    return np.asarray(representation)[:, ROW_ID_COLUMN].astype(np.int64)


def frame_of(row_id, verdict, oracle, family=None, item_id=None):
    row_id = np.asarray(row_id, dtype=np.int64)
    representation = np.zeros((len(row_id), 2), dtype=np.float64)
    representation[:, ROW_ID_COLUMN] = row_id
    representation[:, 1] = 1.0
    return {"representation": representation,
            "verdict": np.asarray(verdict, dtype=np.int64),
            "oracle": np.asarray(oracle, dtype=np.int64),
            "family": np.asarray(row_id if family is None else family, dtype=np.int64),
            "item_id": np.asarray(row_id if item_id is None else item_id, dtype=np.int64)}


def traffic_of(length=TRAFFIC_LENGTH):
    row_id = np.arange(length)
    return frame_of(row_id, verdict=row_id % 2, oracle=(row_id // 2) % 2)


ITEM_POOL = 30


def traffic_serving_items_twice(length=TRAFFIC_LENGTH, pool=ITEM_POOL):
    row_id = np.arange(length)
    return frame_of(row_id, verdict=row_id % 2, oracle=(row_id // 2) % 2,
                    item_id=row_id % pool)


def anonymous(frame):
    return {name: values for name, values in frame.items() if name != "item_id"}


def fitting_slices():
    row_id = np.arange(-16, 0)
    verdict = np.tile([0, 1], 8)
    oracle = np.tile([0, 1, 1, 0], 4)
    training = frame_of(row_id, verdict, oracle)
    calibration = frame_of(row_id - 32, verdict, oracle)
    return training, calibration


SCRIPTED_GATE_RECALL = 0.75
SCRIPTED_GATE_OVER_BLOCK = 0.125


def run_watch(monitor, acceptance_verdicts=None, audits=None, estimators=None, **overrides):
    training, calibration = fitting_slices()
    built = estimators if estimators is not None else []

    def build_estimator():
        built.append(ScriptedEstimator())
        return built[-1]

    settings = {"build_estimator": build_estimator, "estimator": ScriptedEstimator(),
                "training": training, "calibration_slice": calibration,
                "traffic": traffic_of(), "monitor": monitor,
                "flip_rule": FlipRule(FLIP_THRESHOLD), "audit_sampling_window": WINDOW,
                "audit_budget": WINDOW,
                "acceptance": {"recall_floor": 0.0, "false_positive_tolerance": 1.0},
                "max_enlarging_retries": 2, "cross_fit_folds": 2, "seed": 5,
                "labels": "oracle",
                "post_repair_reference_window": REFERENCE_WINDOW}
    settings.update(overrides)

    if acceptance_verdicts is None:
        return watch_traffic(**settings)

    answers = iter(acceptance_verdicts)

    def recording_gate(build_estimator, training, calibration_slice, audit, *rest, **keywords):
        audits.append(row_ids(audit["representation"]))
        return {"recall": SCRIPTED_GATE_RECALL, "over_block": SCRIPTED_GATE_OVER_BLOCK,
                "passed": next(answers), "n_labels": len(audit["verdict"])}

    import rcv.loop

    original = rcv.loop.gate_reading
    rcv.loop.gate_reading = recording_gate
    try:
        return watch_traffic(**settings)
    finally:
        rcv.loop.gate_reading = original


class TestTheWatchWalksTheTraffic:
    def test_quiet_traffic_is_read_once_from_its_first_item_in_window_sized_chunks(self):
        monitor = ScriptedMonitor()
        estimator, events, change_log, _ = run_watch(monitor)

        assert [len(chunk) for chunk in monitor.observed_verdicts] == [WINDOW] * 6
        np.testing.assert_array_equal(np.concatenate(monitor.observed_verdicts),
                                      traffic_of()["verdict"])
        assert events == [] and change_log == []

    def test_a_quiet_chunk_advances_the_watch_rather_than_ending_it(self):
        monitor = ScriptedMonitor(alarms={3: Alarm(regime=0, item_index=0)})
        estimator, events, change_log, _ = run_watch(monitor, observe_only=True)

        assert len(monitor.observed_verdicts) == 4
        assert [event.kind for event in events] == ["alarm", "watch_ended"]


class TestTheAlarmIsRecordedWhereItFired:
    def test_the_alarm_names_its_regime_and_the_traffic_item_it_fired_on(self):
        monitor = ScriptedMonitor(alarms={1: Alarm(regime=1, item_index=3)})
        estimator, events, change_log, _ = run_watch(monitor, observe_only=True)

        assert events == [LoopEvent("alarm", "regime 1, boundary 0.5, monitored position 13", regime=1,
                                    boundary=0.5, monitored_position=13,
                                    stream_position=13),
                          LoopEvent("watch_ended", "observe-only: first alarm recorded")]
        assert change_log == []

    def test_the_stream_position_adds_back_the_prefix_the_watch_never_saw(self):
        monitor = ScriptedMonitor(alarms={1: Alarm(regime=1, item_index=3)})
        _, events, _, _ = run_watch(monitor, observe_only=True, monitored_offset=2000)

        assert events[0].monitored_position == 13
        assert events[0].stream_position == 2013

    def test_a_watch_that_was_not_told_to_only_observe_acts_on_the_alarm(self):
        monitor = ScriptedMonitor(alarms={0: Alarm(regime=0, item_index=0)})
        audits = []
        estimator, events, change_log, _ = run_watch(monitor, acceptance_verdicts=[True],
                                                     audits=audits)

        assert [event.kind for event in events] == ["alarm", "audit", "acceptance_evaluated",
                                                    "repair_accepted", "reference_rederived"]
        assert len(audits) == 1


class TestTheAuditBuysTheWindowAfterTheAlarm:
    def test_the_draw_starts_at_the_item_after_the_alarming_one(self):
        monitor = ScriptedMonitor(alarms={0: Alarm(regime=1, item_index=3)})
        audits = []
        estimator, events, change_log, _ = run_watch(monitor, acceptance_verdicts=[True],
                                                  audits=audits)

        np.testing.assert_array_equal(audits[0], np.arange(4, 4 + WINDOW))
        assert events[1] == LoopEvent(
            "audit", "audit labels: 10; sampled uniformly within post-alarm windows")

    def test_each_retry_enlarges_by_the_window_ahead_of_what_was_already_bought(self):
        monitor = ScriptedMonitor(alarms={0: Alarm(regime=1, item_index=3)})
        audits = []
        estimator, events, change_log, _ = run_watch(
            monitor, acceptance_verdicts=[False, False, True], audits=audits)

        np.testing.assert_array_equal(audits[0], np.arange(4, 14))
        np.testing.assert_array_equal(audits[1], np.arange(4, 24))
        np.testing.assert_array_equal(audits[2], np.arange(4, 34))
        enlargements = [event.detail for event in events if event.kind == "audit"][1:]
        assert enlargements == [
            "audit labels: 20 after enlargement; sampled uniformly within post-alarm windows",
            "audit labels: 30 after enlargement; sampled uniformly within post-alarm windows"]

    def test_a_window_yielding_a_single_item_is_still_an_audit(self):
        monitor = ScriptedMonitor(alarms={5: Alarm(regime=0, item_index=WINDOW - 2)})
        audits = []
        estimator, events, change_log, _ = run_watch(monitor, acceptance_verdicts=[True],
                                                  audits=audits)

        np.testing.assert_array_equal(audits[0], [TRAFFIC_LENGTH - 1])
        assert events[1] == LoopEvent(
            "audit", "audit labels: 1; sampled uniformly within post-alarm windows")

    def test_an_enlargement_yielding_a_single_item_is_still_bought(self):
        monitor = ScriptedMonitor(alarms={4: Alarm(regime=1, item_index=8)})
        audits = []
        estimator, events, change_log, _ = run_watch(
            monitor, acceptance_verdicts=[False, False, False], audits=audits)

        np.testing.assert_array_equal(audits[0], np.arange(49, 59))
        np.testing.assert_array_equal(audits[1], np.arange(49, 60))
        assert change_log == [ChangeLogEntry(what="escalation_demanded",
                                             trigger="alarm in regime 1 at boundary 0.5",
                                             oracle_labels_spent=11,
                                             monitored_position=48, stream_position=48)]

    def test_traffic_that_ends_at_the_alarm_ends_the_watch_with_nothing_bought(self):
        monitor = ScriptedMonitor(alarms={5: Alarm(regime=0, item_index=WINDOW - 1)})
        audits = []
        watched = run_watch(monitor, acceptance_verdicts=[], audits=audits)

        assert watched is not None, "the watch returns its estimator, events and change log"
        estimator, events, change_log, _ = watched
        assert audits == []
        assert events[-1] == LoopEvent(
            "watch_ended", "no post-alarm audit positions sampled")
        assert change_log == []


class TestTheRepairIsRecorded:
    def test_an_accepted_repair_names_its_alarm_and_rederives_the_reference(self):
        monitor = ScriptedMonitor(alarms={0: Alarm(regime=1, item_index=3)})
        audits = []
        estimator, events, change_log, _ = run_watch(monitor, acceptance_verdicts=[True],
                                                  audits=audits)

        assert events == [
            LoopEvent("alarm", "regime 1, boundary 0.5, monitored position 3", regime=1, boundary=0.5,
                      monitored_position=3, stream_position=3),
            LoopEvent("audit", "audit labels: 10; sampled uniformly within post-alarm windows"),
            LoopEvent("acceptance_evaluated",
                      "attempt 1: out-of-fold recall 0.7500, over-block 0.1250 on 10 audit "
                      "labels — passed",
                      gate_recall=SCRIPTED_GATE_RECALL,
                      gate_over_block=SCRIPTED_GATE_OVER_BLOCK,
                      gate_passed=True, audit_labels=10),
            LoopEvent("repair_accepted", "after alarm in regime 1 at boundary 0.5"),
            LoopEvent("reference_rederived",
                      "monitor recalibrated on 10 reference positions, monitored span [14, 23]"),
        ]
        assert change_log == [ChangeLogEntry(
            what="repair", trigger="alarm in regime 1 at boundary 0.5", oracle_labels_spent=10,
            monitored_position=14, stream_position=14, calibration=monitor.calibration,
            replay_seed=[5, 1], reference_audit_overlap_fraction=0.0,
            reference_window_length=10, reference_window_monitored_span=[14, 23],
            reference_window_skipped_positions=0,
            gate_recall=SCRIPTED_GATE_RECALL, gate_over_block=SCRIPTED_GATE_OVER_BLOCK)]

    def test_the_rederivation_carries_the_standing_prefix_verdict(self):
        monitor = ScriptedMonitor(alarms={0: Alarm(regime=1, item_index=3)},
                                  prefix_drift_checked=True)
        run_watch(monitor, acceptance_verdicts=[True], audits=[])

        assert [call["prefix_drift_checked"] for call in monitor.calibrations] == [True]

    def test_the_rederivation_carries_the_standing_horizon_and_family_exposure(self):
        monitor = ScriptedMonitor(alarms={0: Alarm(regime=1, item_index=3)},
                                  monitored_length=19000,
                                  fitted_family_overlap={"all": 0.2, "base": 0.2, "drift": None})
        run_watch(monitor, acceptance_verdicts=[True], audits=[])

        assert [call["monitored_length"] for call in monitor.calibrations] == [19000]
        assert [call["fitted_family_overlap"] for call in monitor.calibrations] == [
            {"all": 0.2, "base": 0.2, "drift": None}]


    def test_a_bounded_retry_exhaustion_is_recorded_as_an_escalation_and_ends_the_watch(self):
        monitor = ScriptedMonitor(alarms={0: Alarm(regime=1, item_index=3),
                                          1: Alarm(regime=0, item_index=0)})
        estimator, events, change_log, _ = run_watch(
            monitor, acceptance_verdicts=[False, False, False], audits=[])

        assert events[-1] == LoopEvent(
            "escalation_demanded",
            "No attempt passed after alarm in regime 1 at boundary 0.5; "
            "watch ended without fine-tuning.")
        assert change_log == [ChangeLogEntry(what="escalation_demanded",
                                             trigger="alarm in regime 1 at boundary 0.5",
                                             oracle_labels_spent=30,
                                             monitored_position=3, stream_position=3)]
        assert monitor.calibrations == [], "an escalation re-derives nothing"


class TestTheFreshPostRepairReferenceWindowIsTheMaterial:
    def test_the_rederivation_replays_the_repaired_estimator_over_the_fresh_window(self):
        monitor = ScriptedMonitor(alarms={0: Alarm(regime=1, item_index=3)})
        estimators = []
        run_watch(monitor, acceptance_verdicts=[True], audits=[], estimators=estimators)

        assert monitor.calibrations == [{"beliefs": REFERENCE_WINDOW,
                                         "verdicts": REFERENCE_WINDOW,
                                         "stream_length": 11, "n_streams": 13,
                                         "seed": [5, 1],
                                         "prefix_drift_checked": None,
                                         "monitored_length": None,
                                         "fitted_family_overlap": None}]
        np.testing.assert_array_equal(estimators[-1].scored_calls[0], np.arange(14, 24))
        np.testing.assert_array_equal(estimators[-1].scored_verdicts[0],
                                      traffic_of()["verdict"][14:24])

    def test_the_window_is_traffic_after_the_repair_and_not_the_traffic_it_was_fitted_on(self):
        monitor = ScriptedMonitor(alarms={0: Alarm(regime=1, item_index=3)})
        audits, estimators = [], []
        run_watch(monitor, acceptance_verdicts=[True], audits=audits, estimators=estimators)

        window = estimators[-1].scored_calls[0]
        np.testing.assert_array_equal(audits[0], np.arange(4, 14))
        assert set(window.tolist()) & set(audits[0].tolist()) == set()
        assert window.min() >= 14, "the window opens after the audit it excludes"

    def test_a_position_serving_an_item_the_update_fitted_is_skipped_not_merely_counted(self):
        monitor = ScriptedMonitor(alarms={0: Alarm(regime=1, item_index=3)})
        estimators = []
        run_watch(monitor, acceptance_verdicts=[True], audits=[], estimators=estimators,
                  traffic=traffic_serving_items_twice(), post_repair_reference_window=25)

        np.testing.assert_array_equal(
            estimators[-1].scored_calls[0],
            np.concatenate([np.arange(14, 34), np.arange(44, 49)]))

    def test_the_recorded_window_names_its_span_its_size_and_what_it_skipped(self):
        monitor = ScriptedMonitor(alarms={0: Alarm(regime=1, item_index=3)})
        _, _, change_log, _ = run_watch(monitor, acceptance_verdicts=[True], audits=[],
                                     traffic=traffic_serving_items_twice(),
                                     post_repair_reference_window=25)

        entry = change_log[0]
        assert entry.reference_window_length == 25
        assert entry.reference_window_monitored_span == [14, 48]
        assert entry.reference_window_skipped_positions == 10
        assert entry.reference_audit_overlap_fraction == 0.0

    def test_each_episode_reads_its_own_window_further_down_the_stream(self):
        monitor = ScriptedMonitor(alarms={0: Alarm(regime=1, item_index=3),
                                          2: Alarm(regime=0, item_index=1)})
        estimators = []
        run_watch(monitor, acceptance_verdicts=[True, True], audits=[], estimators=estimators)

        windows = [estimator.scored_calls[0] for estimator in estimators
                   if estimator.scored_calls]
        assert len(windows) >= 2, "the scripted alarms must drive two accepted repairs"
        for earlier, later in zip(windows, windows[1:]):
            assert later.min() > earlier.max(), "each era reads material the last one did not"

    def test_the_watch_resumes_after_the_window_rather_than_over_it(self):
        monitor = ScriptedMonitor(alarms={0: Alarm(regime=1, item_index=3)})
        estimators = []
        run_watch(monitor, acceptance_verdicts=[True], audits=[], estimators=estimators)

        window, resumed = estimators[-1].scored_calls[0], estimators[-1].scored_calls[1]
        np.testing.assert_array_equal(window, np.arange(14, 24))
        np.testing.assert_array_equal(resumed, np.arange(24, 34))
        assert len(monitor.observed_verdicts[1]) == WINDOW

    def test_a_window_the_remaining_traffic_exactly_fills_is_drawn(self):
        monitor = ScriptedMonitor(alarms={0: Alarm(regime=1, item_index=3)})
        estimators = []
        _, events, _, _ = run_watch(monitor, acceptance_verdicts=[True], audits=[],
                                 estimators=estimators, post_repair_reference_window=46)

        assert [event.kind for event in events][-1] == "reference_rederived"
        np.testing.assert_array_equal(estimators[-1].scored_calls[0], np.arange(14, 60))

    def test_a_frame_carrying_no_item_identity_refuses_rather_than_skipping_the_check(self):
        monitor = ScriptedMonitor(alarms={0: Alarm(regime=1, item_index=3)})
        training, calibration = (anonymous(frame) for frame in fitting_slices())

        with pytest.raises(ValueError, match="item_id"):
            run_watch(monitor, acceptance_verdicts=[True], audits=[],
                      traffic=anonymous(traffic_of()), training=training,
                      calibration_slice=calibration)

    def test_a_run_that_names_no_window_size_refuses_at_the_repair(self):
        monitor = ScriptedMonitor(alarms={0: Alarm(regime=1, item_index=3)})
        with pytest.raises(ValueError) as refusal:
            run_watch(monitor, acceptance_verdicts=[True], audits=[],
                      post_repair_reference_window=None)
        assert str(refusal.value) == (
            "Resuming monitoring requires post_repair_reference_window.")


class TestTrafficExhaustedBeforeAFreshWindow:
    PREFIX_OFFSET = 2000

    def script(self):
        monitor = ScriptedMonitor(alarms={0: Alarm(regime=1, item_index=3)})
        return monitor, run_watch(monitor, acceptance_verdicts=[True], audits=[],
                                  post_repair_reference_window=50,
                                  monitored_offset=self.PREFIX_OFFSET)

    def test_the_repair_stands_and_the_watch_ends_without_a_rederivation(self):
        monitor, (_, events, _, _) = self.script()

        assert [event.kind for event in events] == [
            "alarm", "audit", "acceptance_evaluated", "repair_accepted",
            "reference_window_exhausted"]
        assert monitor.calibrations == [], "a reference is never re-derived from stale material"

    def test_the_outcome_names_what_remained_and_what_was_needed(self):
        _, (_, events, _, _) = self.script()

        assert events[-1] == LoopEvent(
            "reference_window_exhausted",
            "Insufficient reference data: 46 eligible positions, 50 required after repair in "
            "regime 1 at boundary 0.5. Watch ended.")

    def test_the_record_carries_the_repair_and_the_outcome_without_billing_twice(self):
        _, (_, _, change_log, _) = self.script()

        assert [entry.what for entry in change_log] == ["repair", "reference_window_exhausted"]
        assert change_log[0].calibration is None, "no calibration was derived, so none is claimed"
        assert change_log[0].oracle_labels_spent == WINDOW
        assert change_log[1].oracle_labels_spent == 0, (
            "the audit's labels are billed once, on the repair that bought them")

    def test_both_entries_name_their_trigger_their_positions_and_the_window_asked_for(self):
        _, (_, _, change_log, _) = self.script()

        for entry in change_log:
            assert entry.trigger == "alarm in regime 1 at boundary 0.5"
            assert entry.monitored_position == 14
            assert entry.stream_position == self.PREFIX_OFFSET + 14
            assert entry.reference_window_length == 50
        assert change_log[0].reference_window_eligible_positions is None
        assert change_log[1].reference_window_eligible_positions == 46


class TestTheWindowSizeIsResolvedFromTheConfiguration:
    def resolve(self, loop=None, calibration=None):
        from rcv.study import post_repair_reference_window_of

        return post_repair_reference_window_of(loop or {}, calibration or {})

    def test_the_run_may_name_it_outright(self):
        assert self.resolve(loop={"post_repair_reference_window": 1500},
                            calibration={"material": "stream_prefix",
                                         "prefix_length": 2000}) == 1500

    def test_unnamed_it_is_the_deployment_prefixs_own_length(self):
        assert self.resolve(calibration={"material": "stream_prefix",
                                         "prefix_length": 2000}) == 2000

    def test_unnamed_with_no_prefix_to_size_it_by_the_run_refuses(self):
        with pytest.raises(ValueError) as refusal:
            self.resolve(calibration={"material": "evaluation"})
        assert str(refusal.value) == (
            "loop.post_repair_reference_window is required when "
            "monitor.calibration.material is 'evaluation'.")

    def test_the_default_material_is_the_evaluation_slice_and_carries_no_prefix(self):
        with pytest.raises(ValueError, match="post_repair_reference_window"):
            self.resolve(calibration={})

    @pytest.mark.parametrize("named", [0, -1])
    def test_a_window_holding_no_material_refuses_at_the_gate(self, named):
        from rcv.study import validate_config
        from test_belief_bank_seam import config_with

        config = config_with()
        config["loop"]["post_repair_reference_window"] = named
        with pytest.raises(ValueError, match="post_repair_reference_window"):
            validate_config(config)

    def test_a_named_window_of_one_position_passes_the_gate(self):
        from rcv.study import validate_config
        from test_belief_bank_seam import config_with

        config = config_with()
        config["loop"]["post_repair_reference_window"] = 1
        validate_config(config)


class TestTheRecalibrationRecordCarriesTheWindow:
    def entry(self, **overrides):
        from dataclasses import dataclass

        from rcv.loop import ChangeLogEntry

        @dataclass
        class StubCalibration:
            replay_stream_length: int = 18000

        fields = dict(what="repair", trigger="alarm in regime 1 at boundary 0.5",
                      oracle_labels_spent=318, monitored_position=3849, stream_position=5849,
                      calibration=StubCalibration(), replay_seed=[42, 1],
                      reference_audit_overlap_fraction=0.0, reference_window_length=2000,
                      reference_window_monitored_span=[3849, 5911],
                      reference_window_skipped_positions=63)
        fields.update(overrides)
        return ChangeLogEntry(**fields)

    def test_the_record_names_the_window_its_span_and_what_it_skipped(self):
        from rcv.runner import recalibration_record_of

        written = recalibration_record_of(self.entry())
        assert written["trigger"] == "alarm in regime 1 at boundary 0.5"
        assert written["monitored_position"] == 3849
        assert written["stream_position"] == 5849
        assert written["reference_window_length"] == 2000
        assert written["reference_window_monitored_span"] == [3849, 5911]
        assert written["reference_window_skipped_positions"] == 63
        assert written["reference_audit_overlap_fraction"] == 0.0
        assert written["replay_seed"] == [42, 1]
        assert written["calibration"] == {"replay_stream_length": 18000}


class TestTheFreshWindowFunctionsOnTheirOwn:
    def test_the_fitted_items_are_the_audits_own_identities(self):
        from rcv.loop import fitted_item_ids

        traffic = traffic_serving_items_twice()
        audit = {name: values[np.arange(4, 14)] for name, values in traffic.items()}
        np.testing.assert_array_equal(fitted_item_ids(audit), np.arange(4, 14))

    def test_an_audit_with_no_identity_column_refuses(self):
        from rcv.loop import fitted_item_ids

        with pytest.raises(ValueError, match="item_id"):
            fitted_item_ids({"verdict": np.zeros(3, dtype=np.int64)})

    def test_eligible_positions_start_at_the_opening_and_drop_the_fitted_items(self):
        from rcv.loop import eligible_reference_positions

        traffic = traffic_serving_items_twice()
        eligible = eligible_reference_positions(traffic, 14, np.arange(4, 14))
        np.testing.assert_array_equal(
            eligible, np.concatenate([np.arange(14, 34), np.arange(44, 60)]))

    def test_eligibility_on_a_frame_with_no_identity_refuses(self):
        from rcv.loop import eligible_reference_positions

        traffic = traffic_of()
        del traffic["item_id"]
        with pytest.raises(ValueError, match="item_id"):
            eligible_reference_positions(traffic, 14, np.arange(4, 14))

    def test_the_built_window_is_read_back_against_the_fitted_set(self):
        from rcv.loop import fresh_reference_window

        traffic = traffic_serving_items_twice()
        window = fresh_reference_window(traffic, np.arange(14, 24), np.arange(4, 14))
        np.testing.assert_array_equal(window["item_id"], np.arange(14, 24))

    def test_a_window_holding_a_fitted_item_refuses_rather_than_being_built(self):
        from rcv.loop import fresh_reference_window

        traffic = traffic_serving_items_twice()
        with pytest.raises(ValueError) as refusal:
            fresh_reference_window(traffic, np.arange(34, 44), np.arange(4, 14))
        assert str(refusal.value) == (
            "Reference window of 10 positions includes 10 positions with fitted item IDs.")

    def test_a_window_built_from_a_frame_with_no_identity_refuses(self):
        from rcv.loop import fresh_reference_window

        with pytest.raises(ValueError) as refusal:
            fresh_reference_window(anonymous(traffic_of()), np.arange(14, 24), np.arange(4, 14))
        assert str(refusal.value) == (
            "Reference window is missing required column 'item_id'.")


class TestTheOverlapWithTheAuditIsMeasuredOnIdentity:
    def overlap(self, reference, audit):
        from rcv.loop import reference_audit_overlap

        return reference_audit_overlap(reference, audit)

    def test_it_is_the_share_of_the_reference_serving_an_item_the_audit_bought(self):
        traffic = traffic_serving_items_twice()
        reference = {name: values[np.arange(30, 40)] for name, values in traffic.items()}
        audit = {name: values[np.arange(5, 10)] for name, values in traffic.items()}
        assert self.overlap(reference, audit) == 0.5

    def test_a_disjoint_pair_reads_zero(self):
        traffic = traffic_serving_items_twice()
        reference = {name: values[np.arange(14, 24)] for name, values in traffic.items()}
        audit = {name: values[np.arange(4, 14)] for name, values in traffic.items()}
        assert self.overlap(reference, audit) == 0.0

    @pytest.mark.parametrize("anonymous_side", ["reference", "audit"])
    def test_either_side_lacking_identity_is_unmeasurable_rather_than_zero(self, anonymous_side):
        traffic = traffic_serving_items_twice()
        sides = {"reference": {name: values[np.arange(30, 40)]
                               for name, values in traffic.items()},
                 "audit": {name: values[np.arange(5, 10)] for name, values in traffic.items()}}
        sides[anonymous_side] = anonymous(sides[anonymous_side])
        assert self.overlap(sides["reference"], sides["audit"]) is None


def audit_window():
    row_id = np.arange(8)
    verdict = np.array([0, 0, 1, 1, 0, 0, 0, 0])
    oracle = np.array([1, 1, 1, 1, 0, 0, 0, 0])
    return frame_of(row_id, verdict, oracle), np.array([1, 3, 7])


def judge(acceptance, disagreeing_rows, folds=2, audit=None, baseline=None):
    training, calibration = fitting_slices()
    audit = audit_window()[0] if audit is None else audit

    def build_estimator():
        return ScriptedEstimator(disagreeing_rows)

    return _update_meets_criteria_cross_fit(
        build_estimator, training, calibration, audit, FlipRule(FLIP_THRESHOLD), acceptance,
        folds, np.random.default_rng(3), "oracle", baseline)


def read_gate(acceptance, disagreeing_rows, folds=2, audit=None, baseline=None):
    training, calibration = fitting_slices()
    audit = audit_window()[0] if audit is None else audit

    def build_estimator():
        return ScriptedEstimator(disagreeing_rows)

    return gate_reading(build_estimator, training, calibration, audit, FlipRule(FLIP_THRESHOLD),
                        acceptance, folds, np.random.default_rng(3), "oracle", baseline)


class TestTheAcceptanceJudgement:
    def test_no_fold_is_judged_by_a_fit_that_saw_it(self):
        audit, disagreeing = audit_window()
        training, calibration = fitting_slices()
        built = []

        def build_estimator():
            built.append(ScriptedEstimator(disagreeing))
            return built[-1]

        _update_meets_criteria_cross_fit(
            build_estimator, training, calibration, audit, FlipRule(FLIP_THRESHOLD),
            {"recall_floor": 0.0, "false_positive_tolerance": 1.0}, 2,
            np.random.default_rng(3), "oracle", None)

        assert len(built) == 2
        for candidate in built:
            assert set(candidate.scored) & set(candidate.fitted_on) == set()
            assert len(candidate.scored) > 0

    def test_the_criteria_are_met_at_the_floor_and_at_the_tolerance(self):
        audit, disagreeing = audit_window()
        assert judge({"recall_floor": 0.5, "false_positive_tolerance": 0.25},
                     disagreeing) is True

    def test_a_recall_below_the_floor_fails(self):
        audit, disagreeing = audit_window()
        assert judge({"recall_floor": 0.75, "false_positive_tolerance": 0.25},
                     disagreeing) is False

    def test_an_over_block_above_the_tolerance_fails(self):
        audit, disagreeing = audit_window()
        assert judge({"recall_floor": 0.5, "false_positive_tolerance": 0.1},
                     disagreeing) is False

    def test_a_window_with_no_unsafe_item_cannot_be_judged(self):
        row_id = np.arange(8)
        safe_only = frame_of(row_id, verdict=row_id % 2, oracle=np.zeros(8, dtype=np.int64))
        with pytest.raises(RuntimeError) as refusal:
            judge({"recall_floor": 0.5, "false_positive_tolerance": 0.25}, [], audit=safe_only)
        assert str(refusal.value) == (
            "Acceptance evaluation requires both safe (0) and unsafe (1) labels.")

    def test_a_window_that_is_wholly_unsafe_cannot_be_judged_either(self):
        row_id = np.arange(8)
        unsafe_only = frame_of(row_id, verdict=row_id % 2, oracle=np.ones(8, dtype=np.int64))
        with pytest.raises(RuntimeError, match=r"both safe \(0\) and unsafe \(1\) labels"):
            judge({"recall_floor": 0.5, "false_positive_tolerance": 0.25}, [],
                  audit=unsafe_only)


class TestTheRelativeCriteria:
    def test_the_slack_and_the_inflation_are_read_at_their_boundaries(self):
        assert update_meets_relative_criteria(
            update_recall=0.5, update_over_block=0.5,
            baseline_recall=0.75, baseline_over_block=0.25,
            recall_slack=0.25, over_block_inflation=0.25) is True

    def test_a_step_past_either_boundary_loses_the_update(self):
        assert update_meets_relative_criteria(
            update_recall=0.25, update_over_block=0.5,
            baseline_recall=0.75, baseline_over_block=0.25,
            recall_slack=0.25, over_block_inflation=0.25) is False
        assert update_meets_relative_criteria(
            update_recall=0.5, update_over_block=0.75,
            baseline_recall=0.75, baseline_over_block=0.25,
            recall_slack=0.25, over_block_inflation=0.25) is False


class TestTheAuditMaterialAccumulates:
    def two_episode_watch(self):
        monitor = ScriptedMonitor(alarms={0: Alarm(regime=1, item_index=3),
                                          3: Alarm(regime=1, item_index=3)})
        built = []
        estimator, events, change_log, _ = run_watch(
            monitor, acceptance_verdicts=[True, True], audits=[], estimators=built)
        return built, events

    def test_the_second_repair_is_fitted_on_the_first_episodes_labels(self):
        built, events = self.two_episode_watch()
        repairs = [candidate for candidate in built if candidate.fitted_on is not None]
        assert len(repairs) >= 2, f"the scripted watch produced {len(repairs)} fits"
        first_audit_rows = set(repairs[0].fitted_on) - set(row_ids(fitting_slices()[0]
                                                                  ["representation"]))
        second_fit_rows = set(repairs[-1].fitted_on)
        assert first_audit_rows, "the first repair fitted no audit row at all"
        assert first_audit_rows <= second_fit_rows, (
            "the second repair dropped the first episode's audit rows; labels are bought once "
            "and kept (accumulate, never discard)")

    def test_each_audit_batch_is_halved_between_fitting_and_calibration(self):
        from rcv.loop import split_audit_batch

        row_id = np.arange(20)
        audit = frame_of(row_id, verdict=row_id % 2, oracle=(row_id // 2) % 2)
        fitting, calibrating = split_audit_batch(audit, np.random.default_rng(0), {})

        assert len(fitting["verdict"]) == 10 and len(calibrating["verdict"]) == 10
        assert set(row_ids(fitting["representation"])) | \
               set(row_ids(calibrating["representation"])) == set(row_id)
        assert set(row_ids(fitting["representation"])) & \
               set(row_ids(calibrating["representation"])) == set()

    def test_an_odd_batch_gives_the_extra_row_to_the_fitting_half(self):
        from rcv.loop import split_audit_batch

        row_id = np.arange(7)
        audit = frame_of(row_id, verdict=row_id % 2, oracle=(row_id // 2) % 2)
        fitting, calibrating = split_audit_batch(audit, np.random.default_rng(0), {})

        assert len(fitting["verdict"]) == 4 and len(calibrating["verdict"]) == 3


class TestTheGateReadingIsTheJudgementsWholeAnswer:
    def test_it_reports_the_two_rates_its_verdict_and_the_labels_it_read(self):
        audit, disagreeing = audit_window()
        reading = read_gate({"recall_floor": 0.5, "false_positive_tolerance": 0.25}, disagreeing)

        assert reading == {"recall": 0.5, "over_block": 0.25, "passed": True, "n_labels": 8}

    def test_a_failed_reading_still_carries_the_rates_that_failed(self):
        audit, disagreeing = audit_window()
        reading = read_gate({"recall_floor": 0.75, "false_positive_tolerance": 0.25}, disagreeing)

        assert reading["passed"] is False
        assert reading["recall"] == 0.5 and reading["over_block"] == 0.25

    def test_the_relative_form_is_read_the_same_way(self):
        audit, disagreeing = audit_window()
        reading = read_gate({"form": "baseline_relative", "recall_slack": 0.25,
                             "over_block_inflation": 0.25}, disagreeing,
                            baseline={"recall": 0.75, "over_block": 0.25})

        assert reading == {"recall": 0.5, "over_block": 0.25, "passed": True, "n_labels": 8}

    def test_the_bool_wrapper_is_the_readings_verdict_and_nothing_else(self):
        audit, disagreeing = audit_window()
        criteria = {"recall_floor": 0.5, "false_positive_tolerance": 0.25}

        assert judge(criteria, disagreeing) is read_gate(criteria, disagreeing)["passed"]

    def test_a_vacuous_window_stays_a_refusal(self):
        row_id = np.arange(8)
        safe_only = frame_of(row_id, verdict=row_id % 2, oracle=np.zeros(8, dtype=np.int64))

        with pytest.raises(RuntimeError, match=r"both safe \(0\) and unsafe \(1\) labels"):
            read_gate({"recall_floor": 0.5, "false_positive_tolerance": 0.25}, [],
                      audit=safe_only)


class TestEveryAcceptanceEvaluationReachesTheRecord:
    def evaluations(self, events):
        return [event for event in events if event.kind == "acceptance_evaluated"]

    def test_each_attempt_including_the_failed_ones_is_recorded_with_its_rates(self):
        monitor = ScriptedMonitor(alarms={0: Alarm(regime=1, item_index=3)})
        _, events, _, _ = run_watch(monitor, acceptance_verdicts=[False, False, True], audits=[])

        evaluations = self.evaluations(events)
        assert [event.gate_passed for event in evaluations] == [False, False, True]
        assert [event.audit_labels for event in evaluations] == [10, 20, 30]
        assert [event.gate_recall for event in evaluations] == [SCRIPTED_GATE_RECALL] * 3
        assert [event.gate_over_block for event in evaluations] == [SCRIPTED_GATE_OVER_BLOCK] * 3

    def test_an_exhausted_retry_ladder_leaves_every_attempt_on_the_record(self):
        monitor = ScriptedMonitor(alarms={0: Alarm(regime=1, item_index=3)})
        _, events, _, _ = run_watch(monitor, acceptance_verdicts=[False, False, False],
                                    audits=[])

        assert [event.gate_passed for event in self.evaluations(events)] == [False] * 3
        assert events[-1].kind == "escalation_demanded"

    def test_the_accepted_evaluations_rates_reach_the_repairs_own_entry(self):
        monitor = ScriptedMonitor(alarms={0: Alarm(regime=1, item_index=3)})
        _, _, change_log, _ = run_watch(monitor, acceptance_verdicts=[False, True], audits=[])

        assert change_log[0].what == "repair"
        assert change_log[0].gate_recall == SCRIPTED_GATE_RECALL
        assert change_log[0].gate_over_block == SCRIPTED_GATE_OVER_BLOCK

    def test_an_escalations_entry_claims_no_accepted_rates(self):
        monitor = ScriptedMonitor(alarms={0: Alarm(regime=1, item_index=3)})
        _, _, change_log, _ = run_watch(monitor, acceptance_verdicts=[False, False, False],
                                        audits=[])

        assert change_log[0].what == "escalation_demanded"
        assert change_log[0].gate_recall is None and change_log[0].gate_over_block is None


class TestTheEpisodeEndsAtItsOutcome:
    def episode(self, acceptance_verdicts, **overrides):
        monitor = ScriptedMonitor(alarms={0: Alarm(regime=1, item_index=3),
                                          2: Alarm(regime=0, item_index=1)})
        return monitor, run_watch(monitor, acceptance_verdicts=acceptance_verdicts, audits=[],
                                  mode=EPISODE_MODE, post_repair_reference_window=None,
                                  **overrides)

    def test_the_accepted_repair_resolves_the_episode_and_the_watch_ends(self):
        monitor, (_, events, _, _) = self.episode([True, True])

        assert [event.kind for event in events] == [
            "alarm", "audit", "acceptance_evaluated", "repair_accepted", "watch_ended"]
        assert events[-1] == LoopEvent("watch_ended", "episode resolved: repaired")
        assert monitor.calibrations == [], "an episode re-derives no reference"
        assert len(monitor.observed_verdicts) == 1, "the watch never reached the second alarm"

    def test_the_deployed_refit_still_happens_and_is_what_comes_back(self):
        built = []
        monitor, (estimator, _, _, _) = self.episode([True], estimators=built)

        repairs = [candidate for candidate in built if candidate.fitted_on is not None]
        assert len(repairs) == 1
        assert estimator is repairs[-1]

    def test_the_repair_is_billed_on_the_record_with_the_rates_it_passed_on(self):
        monitor, (_, _, change_log, _) = self.episode([True])

        assert change_log == [ChangeLogEntry(
            what="repair", trigger="alarm in regime 1 at boundary 0.5", oracle_labels_spent=10,
            monitored_position=14, stream_position=14,
            gate_recall=SCRIPTED_GATE_RECALL, gate_over_block=SCRIPTED_GATE_OVER_BLOCK)]

    def test_an_episode_names_no_post_repair_window_and_is_not_asked_for_one(self):
        monitor, (_, events, _, _) = self.episode([True])

        assert [event.kind for event in events][-1] == "watch_ended"

    def test_an_escalation_is_terminal_exactly_as_it_is_under_deployment(self):
        monitor, (_, events, change_log, _) = self.episode([False, False, False])

        assert events[-1] == LoopEvent(
            "escalation_demanded",
            "No attempt passed after alarm in regime 1 at boundary 0.5; "
            "watch ended without fine-tuning.")
        assert change_log == [ChangeLogEntry(what="escalation_demanded",
                                             trigger="alarm in regime 1 at boundary 0.5",
                                             oracle_labels_spent=30,
                                             monitored_position=3, stream_position=3)]

    def test_traffic_that_never_alarms_ends_the_episode_with_nothing_recorded(self):
        monitor = ScriptedMonitor()
        _, events, change_log, _ = run_watch(monitor, mode=EPISODE_MODE,
                                             post_repair_reference_window=None)

        assert events == [] and change_log == []
        assert len(monitor.observed_verdicts) == 6, "the whole traffic was watched"

    def test_the_default_is_the_continuous_deployment_watch(self):
        monitor = ScriptedMonitor(alarms={0: Alarm(regime=1, item_index=3)})
        _, events, _, _ = run_watch(monitor, acceptance_verdicts=[True], audits=[])

        assert [event.kind for event in events][-1] == "reference_rederived"

    def test_a_mode_nobody_implements_is_refused_by_name(self):
        monitor = ScriptedMonitor()
        with pytest.raises(ValueError, match="Unknown loop mode 'continuous'"):
            run_watch(monitor, mode="continuous")


class TestTheModeIsConfiguration:
    def config(self, **loop_extras):
        from test_belief_bank_seam import config_with

        config = config_with()
        config["loop"].update(loop_extras)
        return config

    def test_an_episode_run_passes_the_gate(self):
        from rcv.study import validate_config

        validate_config(self.config(mode="episode"))

    def test_a_run_naming_no_mode_still_passes(self):
        from rcv.study import validate_config

        validate_config(self.config())

    def test_an_unknown_mode_is_refused_by_name_at_the_gate(self):
        from rcv.study import validate_config

        with pytest.raises(ValueError, match="Unknown loop mode 'continuous'"):
            validate_config(self.config(mode="continuous"))

    def test_an_episode_is_not_asked_to_size_a_window_it_never_draws(self):
        from rcv.study import post_repair_reference_window_of

        assert post_repair_reference_window_of({"mode": "episode"},
                                               {"material": "evaluation"}) is None

    def test_an_episode_that_names_a_window_anyway_keeps_it(self):
        from rcv.study import post_repair_reference_window_of

        assert post_repair_reference_window_of(
            {"mode": "episode", "post_repair_reference_window": 1500}, {}) == 1500


class TestTheAuditSplitIsStableByItem:
    def batch(self, row_id, item_id):
        row_id = np.asarray(row_id)
        return frame_of(row_id, verdict=row_id % 2, oracle=(row_id // 2) % 2, item_id=item_id)

    def test_one_item_at_two_positions_lands_on_a_single_side(self):
        from rcv.loop import split_audit_batch

        audit = self.batch(np.arange(20), np.arange(20) % 10)
        fitting, calibrating = split_audit_batch(audit, np.random.default_rng(0), {})

        assert set(fitting["item_id"].tolist()) & set(calibrating["item_id"].tolist()) == set()
        assert len(fitting["verdict"]) == 10 and len(calibrating["verdict"]) == 10

    def test_an_items_side_survives_into_the_next_batch(self):
        from rcv.loop import split_audit_batch

        side_of_item = {}
        first = self.batch(np.arange(10), np.arange(10))
        fitting, _ = split_audit_batch(first, np.random.default_rng(0), side_of_item)
        assigned = set(fitting["item_id"].tolist())

        second = self.batch(np.arange(10, 20), np.arange(10))
        fitting_again, calibrating_again = split_audit_batch(second, np.random.default_rng(1),
                                                            side_of_item)

        assert set(fitting_again["item_id"].tolist()) == assigned
        assert set(calibrating_again["item_id"].tolist()) == set(range(10)) - assigned

    def test_never_seen_items_are_halved_by_item_with_the_odd_one_fitting(self):
        from rcv.loop import split_audit_batch

        audit = self.batch(np.arange(14), np.arange(14) % 7)
        fitting, calibrating = split_audit_batch(audit, np.random.default_rng(0), {})

        assert len(set(fitting["item_id"].tolist())) == 4
        assert len(set(calibrating["item_id"].tolist())) == 3

    def test_the_map_is_updated_in_place_so_the_caller_holds_the_assignment(self):
        from rcv.loop import split_audit_batch

        side_of_item = {}
        split_audit_batch(self.batch(np.arange(6), np.arange(6)), np.random.default_rng(0),
                          side_of_item)

        assert sorted(side_of_item) == list(range(6))
        assert set(side_of_item.values()) == {"fitting", "calibrating"}

    def test_a_batch_with_no_item_identity_refuses(self):
        from rcv.loop import split_audit_batch

        with pytest.raises(ValueError, match="item_id"):
            split_audit_batch(anonymous(self.batch(np.arange(4), np.arange(4))),
                              np.random.default_rng(0), {})

    def test_the_two_sides_stay_disjoint_by_item_across_a_multi_episode_watch(self):
        monitor = ScriptedMonitor(alarms={0: Alarm(regime=1, item_index=3),
                                          3: Alarm(regime=0, item_index=3)})
        _, events, _, carry = run_watch(monitor, acceptance_verdicts=[True, True], audits=[],
                                        traffic=traffic_serving_items_twice())

        assert [event.kind for event in events].count("repair_accepted") == 2
        fitting_items = set(carry.fitting["item_id"].tolist())
        calibrating_items = set(carry.calibrating["item_id"].tolist())
        assert fitting_items & calibrating_items == set()
        assert fitting_items & set(range(8, 14)), "the episodes must share items to be a test"


class TestTheCarryHandsTheBoughtLabelsOn:
    def episode(self, carry=None, estimators=None):
        monitor = ScriptedMonitor(alarms={0: Alarm(regime=1, item_index=3)})
        return run_watch(monitor, acceptance_verdicts=[True], audits=[], estimators=estimators,
                         mode=EPISODE_MODE, post_repair_reference_window=None, carry=carry)

    def test_a_watch_returns_the_halves_it_bought_and_their_assignment(self):
        _, _, _, carry = self.episode()

        assert isinstance(carry, AuditCarry)
        assert len(carry.fitting["verdict"]) == 5 and len(carry.calibrating["verdict"]) == 5
        assert sorted(carry.side_of_item) == list(range(4, 14))

    def test_a_watch_given_no_carry_starts_from_nothing(self):
        monitor = ScriptedMonitor()
        _, _, _, carry = run_watch(monitor)

        assert carry == AuditCarry(fitting=None, calibrating=None, side_of_item={})

    def test_a_seeded_episode_stands_on_the_labels_the_last_one_bought(self):
        _, _, _, carry = self.episode()
        built = []
        self.episode(carry=carry, estimators=built)

        repairs = [candidate for candidate in built if candidate.fitted_on is not None]
        carried = set(row_ids(carry.fitting["representation"]))
        assert carried, "the first episode bought no fitting half at all"
        assert carried <= set(repairs[-1].fitted_on), (
            "the seeded episode dropped the carried labels; labels are bought once and kept")

    def test_a_carried_items_side_is_the_side_the_next_watch_gives_it(self):
        _, _, _, carry = self.episode()
        assigned = dict(carry.side_of_item)
        _, _, _, carried_on = self.episode(carry=carry)

        assert {item: carried_on.side_of_item[item] for item in assigned} == assigned

    def test_the_seeded_halves_are_kept_beside_the_new_ones(self):
        _, _, _, carry = self.episode()
        _, _, _, carried_on = self.episode(carry=carry)

        assert len(carried_on.fitting["verdict"]) == 2 * len(carry.fitting["verdict"])
        assert len(carried_on.calibrating["verdict"]) == 2 * len(carry.calibrating["verdict"])


POST_ALARM_RATE = 0.125
SEGMENT_FIRST_ROW = 1000
GATE_FIRST_ROW = 5000
SEGMENT_LADDER = 3 * WINDOW
SEGMENT_LENGTH = SEGMENT_LADDER + 10
GATE_LENGTH = 8
AUDIT_REUSED_ROWS = 2
GATE_REUSED_ROWS = 3


def composed_block(length, first_row, unsafe_rows=()):
    row_id = np.arange(first_row, first_row + length)
    oracle = np.isin(np.arange(length), np.asarray(unsafe_rows, dtype=np.int64)).astype(np.int64)
    return frame_of(row_id, verdict=(row_id - first_row) % 2, oracle=oracle)


def composed_audit(length=SEGMENT_LENGTH):
    return composed_block(length, SEGMENT_FIRST_ROW, unsafe_rows=np.arange(0, length, 2))


def composed_gate(length=GATE_LENGTH, unsafe_rows=(0, 2, 4, 6)):
    return composed_block(length, GATE_FIRST_ROW, unsafe_rows=unsafe_rows)


def blocks_provenance(audit_length=SEGMENT_LENGTH, gate_length=GATE_LENGTH,
                      rate=POST_ALARM_RATE):
    return {"rate": rate, "audit_length": audit_length, "gate_length": gate_length,
            "eligible_attack": 500, "eligible_base": 4000,
            "audit_reused_rows": AUDIT_REUSED_ROWS, "gate_reused_rows": GATE_REUSED_ROWS,
            "audit_sha256": "a" * 64, "gate_sha256": "g" * 64}


def gate_uncomposable_class():
    from rcv import chaining

    if not hasattr(chaining, "GateUncomposable"):
        chaining.GateUncomposable = type("GateUncomposable", (ValueError,), {})
    return chaining.GateUncomposable


class TestTheGateBlockDecides:
    def watch(self, acceptance_verdicts=None, audits=None, audit=None, gate=None,
              provenance=None, calls=None, estimators=None, **overrides):
        audit = composed_audit() if audit is None else audit
        gate = composed_gate() if gate is None else gate
        provenance = blocks_provenance() if provenance is None else provenance
        calls = [] if calls is None else calls
        monitor = ScriptedMonitor(alarms={0: Alarm(regime=1, item_index=3)})

        def audit_traffic(alarmed_at, consumed_items, rng):
            calls.append((alarmed_at, consumed_items))
            return audit, gate, provenance

        settings = {"mode": EPISODE_MODE, "post_repair_reference_window": None,
                    "audit_traffic": audit_traffic,
                    "acceptance": {"recall_floor": 0.5, "false_positive_tolerance": 0.5}}
        settings.update(overrides)
        return calls, run_watch(monitor, acceptance_verdicts=acceptance_verdicts,
                                audits=audits, estimators=estimators, **settings)

    def test_the_deciding_reading_comes_from_the_gate_block(self):
        _, (_, events, _, _) = self.watch()

        evaluation = next(event for event in events if event.kind == "acceptance_evaluated")
        assert evaluation.gate_recall == 0.0
        assert evaluation.audit_labels == WINDOW, "the audit is what the candidate FITTED on"

    def test_poisoning_the_audits_labels_does_not_move_the_verdict(self):
        poisoned = composed_audit()
        poisoned["oracle"] = 1 - poisoned["oracle"]

        _, (_, honest_events, _, _) = self.watch()
        _, (_, poisoned_events, _, _) = self.watch(audit=poisoned)

        honest = [event.gate_recall for event in honest_events
                  if event.kind == "acceptance_evaluated"]
        moved = [event.gate_recall for event in poisoned_events
                 if event.kind == "acceptance_evaluated"]
        assert honest == moved
        assert ([event.kind for event in honest_events]
                == [event.kind for event in poisoned_events])

    def test_poisoning_the_gate_blocks_labels_does_move_the_verdict(self):
        passing = composed_gate(unsafe_rows=(1, 3, 5, 7))

        _, (_, failing_events, _, _) = self.watch()
        _, (_, passing_events, _, _) = self.watch(gate=passing)

        assert [event.gate_passed for event in failing_events
                if event.kind == "acceptance_evaluated"] == [False, False, False]
        assert [event.gate_passed for event in passing_events
                if event.kind == "acceptance_evaluated"] == [True]

    def test_an_attempt_costs_one_fit_and_no_fold_fits(self):
        built = []
        self.watch(estimators=built, gate=composed_gate(unsafe_rows=(1, 3, 5, 7)))

        assert len(built) == 1, "one accepted attempt, one candidate fit"

    def test_a_failed_ladder_costs_one_fit_per_attempt(self):
        built = []
        self.watch(estimators=built)

        assert len(built) == 3, "three attempts, three candidates, no fold-fits"

    def test_the_accepted_candidate_is_what_comes_back(self):
        built = []
        _, (estimator, _, _, _) = self.watch(estimators=built,
                                             gate=composed_gate(unsafe_rows=(1, 3, 5, 7)))

        assert estimator is built[-1], "no refit after acceptance — the candidate IS the repair"

    def test_the_candidate_is_fitted_on_the_audits_fitting_half_and_the_standing_slice(self):
        built = []
        self.watch(estimators=built, gate=composed_gate(unsafe_rows=(1, 3, 5, 7)))

        fitted = set(built[-1].fitted_on)
        audit_rows = set(range(SEGMENT_FIRST_ROW, SEGMENT_FIRST_ROW + WINDOW))
        assert fitted & audit_rows, "the candidate stands on the audit it bought"
        assert fitted & audit_rows != audit_rows, "and only on its fitting half"
        assert set(row_ids(fitting_slices()[0]["representation"])) <= fitted

    def test_the_gate_block_is_composed_once_and_read_by_every_attempt(self):
        calls, (_, events, _, _) = self.watch()

        assert len(calls) == 1, "one composition per alarm, reused across the ladder"
        assert len([event for event in events if event.kind == "acceptance_evaluated"]) == 3


class TestThePairedOldProbeReadingIsRecorded:
    def episode(self, gate=None, **overrides):
        deciding = TestTheGateBlockDecides()
        return deciding.watch(gate=gate, **overrides)

    def test_the_old_probes_reading_is_on_the_repair(self):
        _, (_, _, change_log, _) = self.episode(gate=composed_gate(unsafe_rows=(1, 3, 5, 7)))

        assert change_log[0].what == "repair"
        assert change_log[0].old_probe_gate_recall == 1.0
        assert change_log[0].old_probe_gate_over_block == 0.0

    def test_the_old_probes_reading_is_on_the_escalation_too(self):
        _, (_, _, change_log, _) = self.episode()

        assert change_log[0].what == "escalation_demanded"
        assert change_log[0].old_probe_gate_recall == 0.0
        assert change_log[0].post_alarm_rate == POST_ALARM_RATE
        assert change_log[0].gate_labels == GATE_LENGTH

    def test_the_old_probe_is_read_once_however_many_attempts_run(self):
        built = []
        self.episode(estimators=built)

        assert len(built) == 3


class TestTheEpisodeBillsBothBlocks:
    def episode(self, **overrides):
        return TestTheGateBlockDecides().watch(**overrides)

    def test_a_first_attempt_pass_bills_one_budget_plus_the_gate(self):
        _, (_, _, change_log, _) = self.episode(gate=composed_gate(unsafe_rows=(1, 3, 5, 7)))

        assert change_log[0].oracle_labels_spent == WINDOW + GATE_LENGTH

    def test_a_full_ladder_escalation_bills_the_whole_ladder_plus_the_gate(self):
        _, (_, _, change_log, _) = self.episode()

        assert change_log[0].oracle_labels_spent == SEGMENT_LADDER + GATE_LENGTH

    def test_the_record_names_both_blocks_and_the_reuse_that_was_needed(self):
        _, (_, _, change_log, _) = self.episode()

        entry = change_log[0]
        assert entry.post_alarm_length == SEGMENT_LENGTH, "the audit block the composer built"
        assert entry.post_alarm_gate_length == GATE_LENGTH
        assert entry.gate_labels == GATE_LENGTH
        assert entry.post_alarm_reused_rows == AUDIT_REUSED_ROWS + GATE_REUSED_ROWS

    def test_provenance_missing_one_of_the_nine_fields_refuses_by_name(self):
        incomplete = blocks_provenance()
        del incomplete["gate_sha256"]
        with pytest.raises(ValueError, match="gate_sha256"):
            self.episode(provenance=incomplete)


class TestAnUncomposableGateIsAnHonestTerminal:
    def watch(self):
        monitor = ScriptedMonitor(alarms={0: Alarm(regime=1, item_index=3)})
        refusal = gate_uncomposable_class()

        def audit_traffic(alarmed_at, consumed_items, rng):
            raise refusal("the eligible pools cannot fill a 300-position gate block clear of the "
                          "1,200 items the probe has consumed")

        return run_watch(monitor, acceptance_verdicts=[True], audits=[], mode=EPISODE_MODE,
                         post_repair_reference_window=None, audit_traffic=audit_traffic)

    def test_the_watch_ends_on_the_refusal_rather_than_raising(self):
        _, events, _, _ = self.watch()

        assert [event.kind for event in events] == ["alarm", "gate_uncomposable"]
        assert "gate block" in events[-1].detail

    def test_nothing_is_billed_because_nothing_was_bought(self):
        _, _, change_log, _ = self.watch()

        assert [entry.what for entry in change_log] == ["gate_uncomposable"]
        assert change_log[0].oracle_labels_spent == 0
        assert change_log[0].monitored_position == 3

    def test_the_estimator_that_comes_back_is_the_one_that_went_in(self):
        estimator, _, _, _ = self.watch()

        assert estimator is not None, "an uncomposable gate repairs nothing and breaks nothing"


class TestThePostAlarmSegmentIsConfiguration:
    def config(self, post_alarm=None, **loop_extras):
        from test_belief_bank_seam import config_with

        config = config_with()
        config["loop"]["mode"] = "episode"
        if post_alarm is not None:
            config["loop"]["post_alarm"] = post_alarm
        config["loop"].update(loop_extras)
        return config

    def stationary(self, **overrides):
        block = {"composition": "stationary", "gate_labels": 300}
        block.update(overrides)
        return block

    def test_a_run_naming_a_stationary_block_and_its_gate_passes_the_gate(self):
        from rcv.study import validate_config

        validate_config(self.config(self.stationary()))

    def test_a_run_naming_no_segment_still_passes(self):
        from rcv.study import validate_config

        validate_config(self.config())

    def test_a_block_that_names_no_gate_size_is_refused_by_name(self):
        from rcv.study import validate_config

        with pytest.raises(ValueError, match="gate_labels"):
            validate_config(self.config({"composition": "stationary"}))

    def test_a_gate_holding_no_labels_is_refused(self):
        from rcv.study import validate_config

        with pytest.raises(ValueError, match="gate_labels"):
            validate_config(self.config(self.stationary(gate_labels=0)))

    def test_an_unknown_composition_is_refused_by_name(self):
        from rcv.study import validate_config

        with pytest.raises(ValueError, match=(
                r"Unsupported loop\.post_alarm\.composition 'ramped'; expected 'stationary'\.")):
            validate_config(self.config(self.stationary(composition="ramped")))

    def test_a_named_length_is_refused_because_the_ladder_already_says_it(self):
        from rcv.study import validate_config

        with pytest.raises(ValueError, match="retry ladder determines audit length"):
            validate_config(self.config(self.stationary(length=8000)))

    def test_a_key_the_block_does_not_have_is_refused(self):
        from rcv.study import validate_config

        with pytest.raises(ValueError, match="loop.post_alarm"):
            validate_config(self.config(self.stationary(rate=0.3)))

    def test_a_segment_named_by_a_deployment_run_is_refused_at_the_gate(self):
        from rcv.study import validate_config

        config = self.config(self.stationary())
        del config["loop"]["mode"]
        with pytest.raises(ValueError, match="episode"):
            validate_config(config)


EXHAUST = "exhaust-fresh-data"


class TestRepairIsBoundedByDataNotByRetries:
    def watch(self, acceptance_verdicts=None, audits=None, realized=25, gate=None,
              provenance=None, estimators=None, **overrides):
        audit = composed_audit(length=realized)
        gate = composed_gate() if gate is None else gate
        provenance = (blocks_provenance(audit_length=realized) if provenance is None
                      else provenance)
        monitor = ScriptedMonitor(alarms={0: Alarm(regime=1, item_index=3)})

        def audit_traffic(alarmed_at, consumed_items, rng):
            return audit, gate, provenance

        settings = {"mode": EPISODE_MODE, "post_repair_reference_window": None,
                    "audit_traffic": audit_traffic, "max_enlarging_retries": EXHAUST,
                    "acceptance": {"recall_floor": 0.5, "false_positive_tolerance": 0.5}}
        settings.update(overrides)
        return run_watch(monitor, acceptance_verdicts=acceptance_verdicts, audits=audits,
                         estimators=estimators, **settings)

    def audited_sizes(self, events):
        return [event.audit_labels for event in events
                if event.kind == "acceptance_evaluated"]

    def test_the_ladder_climbs_by_a_budget_and_ends_on_everything_composed(self):
        _, events, _, _ = self.watch(realized=25)

        assert self.audited_sizes(events) == [WINDOW, 2 * WINDOW, 25]

    def test_an_exact_multiple_of_the_budget_does_not_repeat_its_last_attempt(self):
        _, events, _, _ = self.watch(realized=3 * WINDOW)

        assert self.audited_sizes(events) == [WINDOW, 2 * WINDOW, 3 * WINDOW]

    def test_a_block_shorter_than_one_budget_is_a_single_attempt_on_all_of_it(self):
        _, events, _, _ = self.watch(realized=WINDOW - 3)

        assert self.audited_sizes(events) == [WINDOW - 3]

    def test_every_attempt_including_the_final_partial_reaches_the_record(self):
        _, events, _, _ = self.watch(realized=25)

        evaluations = [event for event in events if event.kind == "acceptance_evaluated"]
        assert [event.audit_labels for event in evaluations] == [WINDOW, 2 * WINDOW, 25]
        assert all(event.gate_recall is not None for event in evaluations)

    def test_running_out_of_fresh_data_is_its_own_terminal(self):
        _, events, _, _ = self.watch(realized=25)

        assert [event.kind for event in events][-1] == "fresh_data_exhausted"
        assert "25" in events[-1].detail, "the terminal names the data bound it reached"
        assert "escalation_demanded" not in [event.kind for event in events]

    def test_the_terminal_carries_the_episodes_provenance_and_bills_both_blocks(self):
        _, _, change_log, _ = self.watch(realized=25)

        entry = change_log[0]
        assert entry.what == "fresh_data_exhausted"
        assert entry.oracle_labels_spent == 25 + GATE_LENGTH
        assert entry.gate_labels == GATE_LENGTH
        assert entry.post_alarm_rate == POST_ALARM_RATE
        assert entry.post_alarm_length == 25, "the REALIZED length, not a configured ladder"
        assert entry.old_probe_gate_recall is not None
        assert entry.monitored_position == 3

    def test_a_repair_inside_the_data_bound_is_a_repair_like_any_other(self):
        _, events, change_log, _ = self.watch(
            realized=25, gate=composed_gate(unsafe_rows=(1, 3, 5, 7)))

        assert [event.kind for event in events][-1] == "watch_ended"
        assert change_log[0].what == "repair"
        assert change_log[0].oracle_labels_spent == WINDOW + GATE_LENGTH

    def test_an_empty_composed_block_refuses_rather_than_fitting_nothing(self):
        with pytest.raises(ValueError, match="Post-alarm audit segment"):
            self.watch(realized=0, provenance=blocks_provenance(audit_length=0))

    def test_the_policy_without_a_composed_block_refuses_by_name(self):
        monitor = ScriptedMonitor(alarms={0: Alarm(regime=1, item_index=3)})
        with pytest.raises(ValueError, match="exhaust-fresh-data"):
            run_watch(monitor, acceptance_verdicts=[True], audits=[], mode=EPISODE_MODE,
                      post_repair_reference_window=None, max_enlarging_retries=EXHAUST)

    def test_an_integer_ladder_still_escalates_exactly_as_before(self):
        _, events, change_log, _ = self.watch(realized=SEGMENT_LADDER, max_enlarging_retries=2)

        assert [event.kind for event in events][-1] == "escalation_demanded"
        assert change_log[0].what == "escalation_demanded"


class TestTheRetryPolicyIsConfiguration:
    def config(self, retries, post_alarm=True, mode="episode"):
        from test_belief_bank_seam import config_with

        config = config_with()
        config["loop"]["max_enlarging_retries"] = retries
        if mode is not None:
            config["loop"]["mode"] = mode
        if post_alarm:
            config["loop"]["post_alarm"] = {"composition": "stationary", "gate_labels": 300}
        return config

    def test_the_named_policy_passes_the_gate(self):
        from rcv.study import validate_config

        validate_config(self.config(EXHAUST))

    def test_an_integer_ladder_still_passes(self):
        from rcv.study import validate_config

        validate_config(self.config(3))

    def test_the_policy_is_refused_for_a_deployment_watch(self):
        from rcv.study import validate_config

        with pytest.raises(ValueError, match="exhaust-fresh-data"):
            validate_config(self.config(EXHAUST, post_alarm=False, mode=None))

    def test_the_policy_is_refused_where_nothing_is_composed_to_exhaust(self):
        from rcv.study import validate_config

        with pytest.raises(ValueError, match="exhaust-fresh-data"):
            validate_config(self.config(EXHAUST, post_alarm=False))

    def test_a_policy_nobody_implements_is_refused_by_name(self):
        from rcv.study import validate_config

        with pytest.raises(ValueError, match="until-tuesday"):
            validate_config(self.config("until-tuesday"))

    def test_a_negative_ladder_still_refuses_in_the_words_it_always_did(self):
        from rcv.study import validate_config

        with pytest.raises(ValueError,
                           match=r"^loop\.max_enlarging_retries must be nonnegative; got -1\.$"):
            validate_config(self.config(-1))

class TestTheAttemptLadderArithmetic:
    def sizes(self, retries, composed_length, budget=10):
        from rcv.loop import audit_sizes_of

        return audit_sizes_of(budget, retries, composed_length)

    def test_a_bounded_ladder_climbs_one_budget_per_permitted_evaluation(self):
        assert self.sizes(3, composed_length=40) == [10, 20, 30, 40]

    def test_a_bounded_ladder_of_no_retries_is_one_attempt(self):
        assert self.sizes(0, composed_length=40) == [10]

    def test_the_data_bound_ends_on_everything_composed(self):
        assert self.sizes(EXHAUST, composed_length=25) == [10, 20, 25]

    def test_the_data_bound_does_not_repeat_an_exact_multiple(self):
        assert self.sizes(EXHAUST, composed_length=30) == [10, 20, 30]

    def test_the_data_bound_reads_a_block_shorter_than_one_budget_whole(self):
        assert self.sizes(EXHAUST, composed_length=7) == [7]

    def test_the_data_bound_on_exactly_one_budget_is_one_attempt(self):
        assert self.sizes(EXHAUST, composed_length=10) == [10]

    def test_a_bounded_ladder_never_reads_past_what_was_composed(self):
        assert self.sizes(3, composed_length=25) == [10, 20, 25, 25]
