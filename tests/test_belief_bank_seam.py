import re
from dataclasses import replace

import numpy as np
import pytest

from rcv.belief_bank import JOINT, BankAlarm, BeliefBankCalibration
from rcv.flip import FlipRule
from rcv.loop import ChangeLogEntry, LoopEvent, watch_traffic
from rcv.study import calibrated_monitor, validate_config

FLIP_THRESHOLD = 0.5
SUB_FLIP_BANK = [0.5, 0.9]


def config_with(monitor_extras=None, flip_threshold=FLIP_THRESHOLD):
    monitor = {"quiet_horizon_confidence": 0.95,
               "detectable_shift": {"indifference_zone_quantile": 0.99},
               "event_bank": {"boundaries": [FLIP_THRESHOLD], "budget_allocation": "joint"},
               "calibration": {"stream_length": 100, "n_streams": 400, "streams_per_centre": 1}}
    monitor.update(monitor_extras or {})
    return {
        "study": "belief-bank-seam",
        "seeds": [0],
        "frame": {"path": "frames/none.npz", "name": "none", "fitting_labels": "oracle"},
        "split": {"evaluation_fraction": 0.2, "calibration_fraction": 0.4375},
        "estimator": {"probe": {"family": "linear"}, "calibration": {"method": "platt"},
                      "route_probe": True, "route_calibration": True},
        "flip": {"threshold": flip_threshold},
        "monitor": monitor,
        "loop": {"audit_sampling_window": 120, "audit_budget": 60,
                 "acceptance": {"recall_floor": 0.5, "false_positive_tolerance": 0.25},
                 "max_enlarging_retries": 2, "cross_fit_folds": 4},
    }


def bank_config(boundaries=SUB_FLIP_BANK, budget_allocation=JOINT,
                flip_threshold=FLIP_THRESHOLD, **extras):
    return config_with({"event_bank": {"boundaries": boundaries,
                                       "budget_allocation": budget_allocation, **extras}},
                       flip_threshold=flip_threshold)


class TestTheBankIsDeclaredByTheConfiguration:
    def test_the_adopted_bank_validates(self):
        validate_config(bank_config())

    def test_a_configuration_naming_no_bank_still_validates(self):
        validate_config(config_with())

    def test_an_unknown_key_under_the_bank_refuses_naming_the_known_ones(self):
        with pytest.raises(ValueError, match=(
                r"monitor\.event_bank: unknown keys \['flavour'\]; missing keys \[\]; allowed keys "
                r"\['boundaries', 'budget_allocation', "
                r"'counterfactual_without_flip_threshold'\]\.")):
            validate_config(bank_config(flavour="banana"))

    @pytest.mark.parametrize("missing", ["boundaries", "budget_allocation"])
    def test_a_bank_missing_either_of_its_two_keys_refuses(self, missing):
        config = bank_config()
        del config["monitor"]["event_bank"][missing]
        with pytest.raises(ValueError,
                           match=rf"monitor\.event_bank: .*missing keys \['{missing}'\]"):
            validate_config(config)

    def test_boundaries_that_do_not_ascend_refuse_at_the_gate(self):
        with pytest.raises(ValueError, match="strictly ascending"):
            validate_config(bank_config(boundaries=[0.9, 0.5]))

    def test_a_boundary_outside_the_unit_interval_refuses_at_the_gate(self):
        with pytest.raises(ValueError, match="outside"):
            validate_config(bank_config(boundaries=[0.5, 1.0]))

    def test_an_unknown_budget_allocation_refuses_at_the_gate(self):
        with pytest.raises(ValueError, match="budget allocation"):
            validate_config(bank_config(budget_allocation="even"))

    def test_a_bank_that_omits_the_flip_threshold_refuses(self):
        with pytest.raises(ValueError, match=re.escape(
                "monitor.event_bank boundaries (0.7, 0.9) omit flip threshold 0.5; add it or "
                "set counterfactual_without_flip_threshold: true.")):
            validate_config(bank_config(boundaries=[0.7, 0.9]))

    def test_a_bank_carrying_a_different_flip_threshold_validates_against_that_one(self):
        validate_config(bank_config(boundaries=[0.3, 0.5], flip_threshold=0.3))


def frame_of(probability, verdict):
    return {"representation": np.asarray(probability, dtype=np.float64).reshape(-1, 1),
            "verdict": np.asarray(verdict, dtype=np.int64),
            "family": np.arange(len(verdict)),
            "item_id": np.arange(len(verdict))}


class BeliefEchoingEstimator:
    def __init__(self):
        self.scored = []

    def fit(self, fitting_slice, calibration_slice):
        return self

    def probability_of_agreement(self, representation, verdict):
        belief = np.asarray(representation)[:, 0].astype(np.float64)
        self.scored.append(belief.copy())
        return belief

    def probe_selection(self):
        return None


def reference_material(rows=600):
    draw = np.random.default_rng(4)
    verdict = np.arange(rows) % 2
    belief = draw.choice([0.2, 0.6, 0.95], size=rows, p=[0.15, 0.25, 0.60])
    return frame_of(belief, verdict)


class TestTheMonitorIsCalibratedOnTheBeliefRatherThanTheFlip:
    MONITOR_CONFIG = {"quiet_horizon_confidence": 0.8,
                      "detectable_shift": {"indifference_zone_quantile": 0.99},
                      "event_bank": {"boundaries": SUB_FLIP_BANK,
                                     "budget_allocation": "joint"},
                      "calibration": {"stream_length": 200, "n_streams": 400, "streams_per_centre": 1}}

    def _calibrated(self, monitor_config=None):
        evaluation = reference_material()
        estimator = BeliefEchoingEstimator()
        probability = estimator.probability_of_agreement(evaluation["representation"],
                                                          evaluation["verdict"])
        monitor, traffic, reference_slice, offset = calibrated_monitor(
            monitor_config or self.MONITOR_CONFIG, estimator, FlipRule(FLIP_THRESHOLD),
            evaluation, probability, reference_material(200), seed=1)
        return monitor, estimator

    def test_the_reference_is_each_wires_event_rate_on_the_belief(self):
        monitor, _ = self._calibrated()
        reference = monitor.calibration.reference
        assert set(reference) == {0, 1}
        for regime in (0, 1):
            assert set(reference[regime]) == {0.5, 0.9}
            assert reference[regime][0.5] < reference[regime][0.9]

    def test_the_calibration_records_the_bank_the_configuration_declared(self):
        monitor, _ = self._calibrated()
        assert monitor.calibration.boundaries == (0.5, 0.9)
        assert monitor.calibration.budget_allocation == JOINT

    def test_a_configuration_naming_no_bank_refuses_rather_than_deploying_a_default(self):
        without_a_bank = {name: value for name, value in self.MONITOR_CONFIG.items()
                          if name != "event_bank"}
        with pytest.raises(KeyError):
            self._calibrated(without_a_bank)

    def test_the_monitor_was_handed_the_belief_and_not_a_column_of_booleans(self):
        monitor, estimator = self._calibrated()
        assert monitor.calibration.reference[0][0.5] != monitor.calibration.reference[0][0.9], (
            "two wires reading the same column would mean the seam handed over a flip indicator")


WINDOW = 10
TRAFFIC_LENGTH = 60
REFERENCE_WINDOW = 10


class StandingCalibration:
    def __init__(self):
        self.replay_stream_length = 11
        self.n_streams = 13
        self.reference_uncertainty = "point"
        self.prefix_drift_checked = None
        self.monitored_length = None
        self.fitted_family_overlap = None


class ScriptedBank:
    def __init__(self, alarms=None):
        self.alarms = dict(alarms or {})
        self.observed = []
        self.calibrations = []
        self.calibration = StandingCalibration()

    def observe(self, probability, verdict):
        self.observed.append(np.asarray(probability).copy())
        return self.alarms.get(len(self.observed) - 1)

    def calibrate(self, probability, verdict, stream_length, n_streams, seed, **keywords):
        self.calibrations.append(np.asarray(probability).copy())


def loop_traffic():
    row_id = np.arange(TRAFFIC_LENGTH)
    frame = frame_of(np.choose(row_id % 3, [0.2, 0.6, 0.95]), row_id % 2)
    frame["oracle"] = (row_id // 2) % 2
    return frame


def fitting_slice():
    row_id = np.arange(16)
    frame = frame_of(np.tile([0.2, 0.6, 0.95, 0.4], 4), np.tile([0, 1], 8))
    frame["oracle"] = np.tile([0, 1, 1, 0], 4)
    frame["family"] = row_id
    return frame


def run_watch(monitor, **overrides):
    training = fitting_slice()
    settings = {"build_estimator": BeliefEchoingEstimator, "estimator": BeliefEchoingEstimator(),
                "training": training, "calibration_slice": training, "traffic": loop_traffic(),
                "monitor": monitor, "flip_rule": FlipRule(FLIP_THRESHOLD),
                "audit_sampling_window": WINDOW, "audit_budget": WINDOW,
                "acceptance": {"recall_floor": 0.0, "false_positive_tolerance": 1.0},
                "max_enlarging_retries": 2, "cross_fit_folds": 2, "seed": 5,
                "labels": "oracle", "post_repair_reference_window": REFERENCE_WINDOW,
                "observe_only": True}
    settings.update(overrides)
    return watch_traffic(**settings)


class TestTheWatchReadsTheBelief:
    def test_the_monitor_is_shown_the_estimators_belief_item_by_item(self):
        monitor = ScriptedBank()
        run_watch(monitor)
        shown = np.concatenate(monitor.observed)
        np.testing.assert_array_equal(shown, loop_traffic()["representation"][:, 0])

    def test_what_the_monitor_reads_is_not_the_flip_the_rule_would_have_derived(self):
        monitor = ScriptedBank()
        run_watch(monitor)
        shown = np.concatenate(monitor.observed)
        assert len(np.unique(shown)) > 2, ("a column of two values is a flip indicator, not the "
                                           "belief the bank monitors")

    def test_the_rederivation_replays_the_belief_over_the_fresh_post_repair_window(self):
        monitor = ScriptedBank(alarms={0: BankAlarm(regime=1, item_index=3, boundary=0.9)})
        run_watch(monitor, observe_only=False)
        assert len(monitor.calibrations) == 1
        np.testing.assert_array_equal(monitor.calibrations[0],
                                      loop_traffic()["representation"][14:24, 0])


class TestTheAlarmNamesItsWireInTheRecord:
    def test_the_loop_event_carries_the_boundary_beside_the_regime_and_the_positions(self):
        monitor = ScriptedBank(alarms={1: BankAlarm(regime=1, item_index=3, boundary=0.9)})
        _, events, _, _ = run_watch(monitor)
        assert events[0] == LoopEvent("alarm", "regime 1, boundary 0.9, monitored position 13", regime=1,
                                      boundary=0.9, monitored_position=13, stream_position=13)

    def test_the_change_log_entrys_trigger_names_the_wire(self):
        monitor = ScriptedBank(alarms={0: BankAlarm(regime=0, item_index=2, boundary=0.5)})
        _, _, change_log, _ = run_watch(monitor, observe_only=False)
        assert change_log == [ChangeLogEntry(
            what="repair", trigger="alarm in regime 0 at boundary 0.5", oracle_labels_spent=10,
            monitored_position=13, stream_position=13, calibration=monitor.calibration,
            replay_seed=[5, 1], reference_audit_overlap_fraction=0.0,
            reference_window_length=10, reference_window_monitored_span=[13, 22],
            reference_window_skipped_positions=0,
            gate_recall=0.6, gate_over_block=0.4)]

    def test_a_sub_flip_alarm_obliges_the_same_audit_window_as_a_flip_alarm(self):
        at_the_flip = ScriptedBank(alarms={0: BankAlarm(regime=0, item_index=2, boundary=0.5)})
        below_it = ScriptedBank(alarms={0: BankAlarm(regime=0, item_index=2, boundary=0.9)})
        _, flip_events, flip_log, _ = run_watch(at_the_flip, observe_only=False)
        _, sub_events, sub_log, _ = run_watch(below_it, observe_only=False)

        assert [event.kind for event in sub_events] == [event.kind for event in flip_events]
        assert ([entry.oracle_labels_spent for entry in sub_log]
                == [entry.oracle_labels_spent for entry in flip_log])


class TestTheRecordCarriesEveryWiresKey:
    def _calibration(self):
        per_wire = {0: {0.5: 1.0, 0.9: 2.0}, 1: {0.5: 3.0, 0.9: 4.0}}
        return BeliefBankCalibration(
            reference=per_wire, threshold=per_wire,
            realized_alarm_rate_replayed=per_wire,
            realized_alarm_rate_combined_replayed=0.05,
            replay_stream_length=100, n_streams=400, seed=0,
            detectable_shift=per_wire, break_even=per_wire,
            boundaries=(0.5, 0.9), budget_allocation=JOINT,
            streams_in_the_tail={0: {0.5: 10, 0.9: 10}, 1: {0.5: 10, 0.9: 10}})

    def test_every_wire_keyed_field_is_written_as_rows_naming_regime_and_boundary(self):
        from rcv.runner import calibration_record_of

        written = calibration_record_of(self._calibration())
        assert written["threshold"] == [
            {"regime": 0, "boundary": 0.5, "value": 1.0},
            {"regime": 0, "boundary": 0.9, "value": 2.0},
            {"regime": 1, "boundary": 0.5, "value": 3.0},
            {"regime": 1, "boundary": 0.9, "value": 4.0}]

    def test_the_rows_read_back_with_the_keys_the_code_speaks_in(self):
        from rcv.runner import calibration_record_of, wire_map_of

        written = calibration_record_of(self._calibration())
        assert wire_map_of(written["reference"]) == {0: {0.5: 1.0, 0.9: 2.0},
                                                     1: {0.5: 3.0, 0.9: 4.0}}

    def test_a_field_the_legacy_modes_leave_empty_is_written_as_nothing(self):
        from rcv.runner import calibration_record_of

        written = calibration_record_of(replace(self._calibration(),
                                                resolved_detectable_shift=None))
        assert written["resolved_detectable_shift"] is None

    def test_the_record_survives_the_runners_own_serialisation(self):
        import json

        from rcv.runner import as_written, calibration_record_of

        written = json.loads(as_written(calibration_record_of(self._calibration())))
        assert written["boundaries"] == [0.5, 0.9]
        assert written["budget_allocation"] == JOINT


class TestTheGateReadsEveryBranchOfTheConfiguration:
    def test_an_unknown_section_refuses(self):
        config = config_with()
        config["flavour"] = "banana"
        with pytest.raises(ValueError,
                           match=r"Configuration: unknown sections \['flavour'\]"):
            validate_config(config)

    def test_a_missing_section_refuses(self):
        config = config_with()
        del config["monitor"]
        with pytest.raises(ValueError, match=r"missing sections \['monitor'\]"):
            validate_config(config)

    def test_an_unknown_key_inside_a_section_refuses(self):
        config = config_with()
        config["loop"]["flavour"] = "banana"
        with pytest.raises(ValueError, match=r"loop: unknown keys \['flavour'\]"):
            validate_config(config)

    def test_an_unknown_monitor_calibration_key_refuses(self):
        config = config_with()
        config["monitor"]["calibration"]["flavour"] = "banana"
        with pytest.raises(ValueError, match=r"monitor\.calibration: unknown keys \['flavour'\]"):
            validate_config(config)

    def test_a_stream_prefix_reference_without_a_prefix_length_refuses(self):
        config = config_with()
        config["monitor"]["calibration"]["material"] = "stream_prefix"
        with pytest.raises(ValueError,
                           match=r"monitor\.calibration: .*missing keys \['prefix_length'\]"):
            validate_config(config)

    def test_a_prefix_length_under_an_evaluation_reference_refuses_as_a_key_that_does_nothing(
            self):
        config = config_with()
        config["monitor"]["calibration"]["prefix_length"] = 100
        with pytest.raises(ValueError,
                           match=r"monitor\.calibration: unknown keys \['prefix_length'\]"):
            validate_config(config)

    def test_an_unknown_material_refuses(self):
        config = config_with()
        config["monitor"]["calibration"]["material"] = "stream_prefixx"
        with pytest.raises(ValueError, match="Unknown monitor.calibration.material"):
            validate_config(config)

    def test_a_calibration_still_naming_the_retired_treatment_refuses_by_name(self):
        config = config_with()
        config["monitor"]["calibration"]["reference_uncertainty"] = "jeffreys"
        with pytest.raises(ValueError, match="reference_uncertainty is a retired key"):
            validate_config(config)

    def test_the_retired_budget_key_refuses_naming_its_successor_and_the_inversion(self):
        config = config_with()
        config["monitor"]["false_alarm_budget"] = 0.05
        with pytest.raises(ValueError, match="renamed quiet_horizon_confidence, WITH INVERTED"):
            validate_config(config)

    @pytest.mark.parametrize("name", ["stream_length", "n_streams"])
    def test_a_replay_dimension_below_one_refuses(self, name):
        config = config_with()
        config["monitor"]["calibration"][name] = 0
        with pytest.raises(ValueError,
                           match=rf"monitor\.calibration\.{name} must be at least 1; got 0\."):
            validate_config(config)

    @pytest.mark.parametrize("name,below", [("cross_fit_folds", 1), ("audit_sampling_window", 0),
                                            ("audit_budget", 0), ("max_enlarging_retries", -1)])
    def test_a_loop_value_below_its_floor_refuses(self, name, below):
        config = config_with()
        config["loop"][name] = below
        expected = (r"loop\.max_enlarging_retries must be nonnegative; got -1\."
                    if name == "max_enlarging_retries"
                    else rf"loop\.{name} must be at least")
        with pytest.raises(ValueError, match=expected):
            validate_config(config)

    def test_an_unknown_acceptance_form_refuses(self):
        config = config_with()
        config["loop"]["acceptance"] = {"form": "relative"}
        with pytest.raises(ValueError, match="Unknown loop.acceptance form"):
            validate_config(config)

    def test_the_relative_acceptance_form_keeps_its_own_key_set(self):
        config = config_with()
        config["loop"]["acceptance"] = {"form": "baseline_relative", "recall_slack": 0.02,
                                        "over_block_inflation": 0.05}
        validate_config(config)

    def test_an_unknown_key_under_the_acceptance_criteria_refuses(self):
        config = config_with()
        config["loop"]["acceptance"]["recall_flooor"] = 0.5
        with pytest.raises(ValueError, match=r"loop\.acceptance: unknown keys"):
            validate_config(config)

    def test_a_malformed_detectable_shift_refuses_at_the_gate(self):
        config = config_with()
        config["monitor"]["detectable_shift"] = {"factor": 0.5}
        with pytest.raises(ValueError, match="Unknown detectable_shift keys"):
            validate_config(config)

    def test_a_retired_shift_form_refuses_at_the_gate_naming_its_successor(self):
        for retired, successor in ((0.1, "must be a mapping"),
                                   ({"mode": "relative", "factor": 0.5}, "no longer accepts"),
                                   ({"baseline_credibility": 0.99}, "indifference_zone_quantile")):
            config = config_with()
            config["monitor"]["detectable_shift"] = retired
            with pytest.raises(ValueError, match=successor):
                validate_config(config)


def stream_frame(prefix_length=100, rows=400, contaminated=()):
    draw = np.random.default_rng(7)
    frame = frame_of(draw.choice([0.2, 0.6, 0.95], size=rows, p=[0.15, 0.25, 0.60]),
                     np.arange(rows) % 2)
    schedule = np.zeros(rows)
    schedule[prefix_length:] = np.linspace(0.0, 0.25, rows - prefix_length)
    drift = np.zeros(rows, dtype=bool)
    for position, rate in contaminated:
        schedule[position], drift[position] = rate, True
    frame["stream_lambda"] = schedule
    frame["is_drift_item"] = drift
    return frame


class TestTheReferenceCanBeDrawnFromTheStreamsOwnHead:
    PREFIX = 100

    def _config(self, **calibration):
        return {"quiet_horizon_confidence": 0.8,
                "detectable_shift": {"indifference_zone_quantile": 0.99},
                "event_bank": {"boundaries": SUB_FLIP_BANK, "budget_allocation": "joint"},
                "calibration": {"material": "stream_prefix", "prefix_length": self.PREFIX,
                                "stream_length": 200, "n_streams": 400, "streams_per_centre": 1, **calibration}}

    def _calibrated(self, traffic, monitor_config=None, fitted=None):
        evaluation = reference_material()
        estimator = BeliefEchoingEstimator()
        probability = estimator.probability_of_agreement(evaluation["representation"],
                                                         evaluation["verdict"])
        return calibrated_monitor(monitor_config or self._config(), estimator,
                                  FlipRule(FLIP_THRESHOLD), evaluation, probability, traffic,
                                  seed=1, fitted=fitted)

    def test_the_prefix_is_the_reference_and_the_watch_starts_where_it_ends(self):
        traffic = stream_frame(self.PREFIX)
        monitor, watched, reference_slice, offset = self._calibrated(traffic)
        assert len(reference_slice["verdict"]) == self.PREFIX
        assert len(watched["verdict"]) == len(traffic["verdict"]) - self.PREFIX
        assert offset == self.PREFIX
        np.testing.assert_array_equal(watched["verdict"], traffic["verdict"][self.PREFIX:])

    def test_the_record_says_the_prefix_was_checked_and_the_horizon_it_watches(self):
        monitor, watched, _, _ = self._calibrated(stream_frame(self.PREFIX))
        assert monitor.calibration.prefix_drift_checked is True
        assert monitor.calibration.monitored_length == len(watched["verdict"])

    def test_a_frame_without_the_stream_columns_records_an_unverifiable_prefix(self):
        traffic = stream_frame(self.PREFIX)
        del traffic["stream_lambda"]
        del traffic["is_drift_item"]
        monitor, _, _, _ = self._calibrated(traffic)
        assert monitor.calibration.prefix_drift_checked is False

    def test_a_contaminated_prefix_refuses_naming_the_count_and_the_maximum_rate(self):
        traffic = stream_frame(self.PREFIX, contaminated=[(11, 0.0625), (37, 0.2375)])
        with pytest.raises(ValueError, match=(
                r"^Calibration prefix has 2 contaminated positions "
                r"\(maximum stream_lambda 0\.237500\); prefix reference requires no "
                r"contamination\.$")):
            self._calibrated(traffic)

    def test_a_prefix_that_consumes_the_whole_stream_refuses(self):
        with pytest.raises(ValueError, match="must be smaller than stream length"):
            self._calibrated(stream_frame(self.PREFIX),
                             self._config(prefix_length=10_000))

    def test_an_evaluation_reference_leaves_the_prefix_verdict_unrecorded_and_the_offset_at_zero(
            self):
        without_a_prefix = {name: value for name, value in self._config().items()
                            if name != "calibration"}
        without_a_prefix["calibration"] = {"stream_length": 200, "n_streams": 400, "streams_per_centre": 1,
                                           "reference_uncertainty": "point"}
        monitor, watched, reference_slice, offset = self._calibrated(stream_frame(self.PREFIX),
                                                                     without_a_prefix)
        assert monitor.calibration.prefix_drift_checked is None
        assert offset == 0
        assert len(watched["verdict"]) == len(stream_frame(self.PREFIX)["verdict"]), (
            "nothing was sliced off the front, so the whole stream is watched")
        assert reference_slice is not watched

    def test_the_fitted_family_exposure_is_measured_where_the_caller_names_the_fit(self):
        traffic = stream_frame(self.PREFIX)
        fitted = {"family": traffic["family"][: self.PREFIX + 50]}
        monitor, _, _, _ = self._calibrated(traffic, fitted=fitted)
        overlap = monitor.calibration.fitted_family_overlap
        assert 0.0 < overlap["all"] < 1.0
        assert overlap["drift"] is None
        assert overlap["base"] == overlap["all"]

        unnamed, _, _, _ = self._calibrated(traffic)
        assert unnamed.calibration.fitted_family_overlap is None


class TestTheOneReaderOfTheAlarmRecordKnowsBothEras:
    def test_a_record_from_after_the_bank_reads_its_wire_and_both_positions(self):
        from rcv.runner import alarm_of

        assert alarm_of({"kind": "alarm", "detail": "regime 1, boundary 0.9, at item 40",
                         "regime": 1, "boundary": 0.9, "monitored_position": 40,
                         "stream_position": 2040}) == {
            "regime": 1, "boundary": 0.9, "monitored_position": 40, "stream_position": 2040}

    def test_a_record_from_before_the_bank_reads_no_boundary_rather_than_a_made_up_one(self):
        from rcv.runner import alarm_of

        assert alarm_of({"kind": "alarm", "detail": "regime 0, at item 7"}) == {
            "regime": 0, "boundary": None, "monitored_position": 7, "stream_position": None}

    def test_the_structured_fields_win_over_the_sentence_when_both_are_there(self):
        from rcv.runner import alarm_of

        read = alarm_of({"kind": "alarm", "detail": "regime 0, at item 7", "regime": 1,
                         "boundary": 0.5, "monitored_position": 40, "stream_position": 40})
        assert read["regime"] == 1 and read["monitored_position"] == 40

    def test_a_record_carrying_no_stream_position_reads_it_as_absent(self):
        from rcv.runner import alarm_of

        assert alarm_of({"kind": "alarm", "detail": "regime 1, boundary 0.5, at item 3",
                         "regime": 1, "boundary": 0.5,
                         "monitored_position": 3})["stream_position"] is None


class TestTheFloorsAreTheValuesTheyName:
    @pytest.mark.parametrize("name,at_the_floor", [("cross_fit_folds", 2),
                                                   ("audit_sampling_window", 1),
                                                   ("audit_budget", 1),
                                                   ("max_enlarging_retries", 0)])
    def test_a_loop_value_sitting_on_its_floor_is_admitted(self, name, at_the_floor):
        config = config_with()
        config["loop"][name] = at_the_floor
        validate_config(config)

    @pytest.mark.parametrize("name", ["stream_length", "n_streams"])
    def test_a_replay_dimension_of_one_is_admitted(self, name):
        config = config_with()
        config["monitor"]["calibration"][name] = 1
        validate_config(config)


class TestThePrefixIsSlicedAtTheValueItNames:
    def test_a_prefix_as_long_as_the_stream_refuses_because_nothing_would_be_monitored(self):
        traffic = stream_frame(100, rows=400)
        config = TestTheReferenceCanBeDrawnFromTheStreamsOwnHead._config(
            TestTheReferenceCanBeDrawnFromTheStreamsOwnHead(), prefix_length=400)
        evaluation = reference_material()
        estimator = BeliefEchoingEstimator()
        probability = estimator.probability_of_agreement(evaluation["representation"],
                                                         evaluation["verdict"])
        with pytest.raises(ValueError, match=re.escape(
                "prefix_length must be smaller than stream length; got 400 for 400 rows.")):
            calibrated_monitor(config, estimator, FlipRule(FLIP_THRESHOLD), evaluation,
                               probability, traffic, seed=1)

    def test_the_same_seed_gives_the_same_deployed_thresholds(self):
        traffic = stream_frame(100, rows=400)
        config = TestTheReferenceCanBeDrawnFromTheStreamsOwnHead._config(
            TestTheReferenceCanBeDrawnFromTheStreamsOwnHead())
        evaluation = reference_material()
        estimator = BeliefEchoingEstimator()
        probability = estimator.probability_of_agreement(evaluation["representation"],
                                                         evaluation["verdict"])
        first, *_ = calibrated_monitor(config, estimator, FlipRule(FLIP_THRESHOLD), evaluation,
                                       probability, traffic, seed=1)
        again, *_ = calibrated_monitor(config, estimator, FlipRule(FLIP_THRESHOLD), evaluation,
                                       probability, traffic, seed=1)
        moved, *_ = calibrated_monitor(config, estimator, FlipRule(FLIP_THRESHOLD), evaluation,
                                       probability, traffic, seed=2)
        assert first.calibration.threshold == again.calibration.threshold
        assert first.calibration.threshold != moved.calibration.threshold
        assert first.calibration.seed == 1
