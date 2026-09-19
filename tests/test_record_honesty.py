import json

import numpy as np

from fixture_frame import build_fixture_frame
from test_prefix_drift_and_config_strictness import _prefix_config, _stream_frame
from test_v1_skeleton import write_frame, write_v1_config

REFERENCE_MATERIAL_ROWS = 600
REPLAY_STREAMS = 400
REPLAY_LENGTH = 300


def _material(seed=0, rate_by_regime=(0.05, 0.20)):
    draw = np.random.default_rng(seed)
    verdict = (draw.random(REFERENCE_MATERIAL_ROWS) < 0.5).astype(np.int64)
    rate = np.where(verdict == 1, rate_by_regime[1], rate_by_regime[0])
    return draw.random(REFERENCE_MATERIAL_ROWS) < rate, verdict


FLIP_THRESHOLD = 0.5


def _beliefs(flips):
    return np.where(flips, 0.1, 0.9)


def _calibrated(seed=0):
    from rcv.belief_bank import BeliefBankMonitor

    monitor = BeliefBankMonitor(0.05, {"indifference_zone_quantile": 0.99},
                                boundaries=(FLIP_THRESHOLD,), budget_allocation="joint")
    flips, verdict = _material(seed)
    return monitor, monitor.calibrate(_beliefs(flips), verdict, stream_length=REPLAY_LENGTH,
                                      n_streams=REPLAY_STREAMS, seed=seed)


class TestTheRealizedAlarmRateNamesItsQuantity:
    def test_the_old_ambiguous_name_is_gone(self):
        from rcv.belief_bank import BeliefBankCalibration

        assert not hasattr(BeliefBankCalibration, "realized_alarm_rate")
        assert "realized_alarm_rate" not in BeliefBankCalibration.__dataclass_fields__
        assert "realized_alarm_rate_combined" not in BeliefBankCalibration.__dataclass_fields__

    def test_both_are_present_and_they_are_different_quantities(self):
        _, calibration = _calibrated()

        assert set(calibration.realized_alarm_rate_at_reference) == {0, 1}
        assert (calibration.realized_alarm_rate_combined_at_reference
                < calibration.realized_alarm_rate_combined_replayed)

    def test_the_at_reference_reading_reproduces_a_hand_computation_on_point_centred_replays(self):
        from rcv.belief_bank import events_of, wire_peak_accumulation
        from rcv.cusum import at_reference_replay_seed
        from rcv.guards import REGIMES

        seed = 3
        _, calibration = _calibrated(seed=seed)
        flips, verdict = _material(seed)
        event = events_of(_beliefs(flips), (FLIP_THRESHOLD,))
        replay = np.random.default_rng(at_reference_replay_seed(seed))
        alarmed = []
        for _ in range(REPLAY_STREAMS):
            drawn = replay.integers(0, event.shape[1], size=REPLAY_LENGTH)
            drawn_verdict = verdict[drawn]
            peaks = {regime: wire_peak_accumulation(
                event[0][drawn], drawn_verdict == regime,
                calibration.reference[regime][FLIP_THRESHOLD],
                calibration.detectable_shift[regime][FLIP_THRESHOLD], regime)
                for regime in REGIMES}
            alarmed.append({regime: peaks[regime] >= calibration.threshold[regime][FLIP_THRESHOLD]
                            for regime in REGIMES})
        by_hand = {regime: {FLIP_THRESHOLD: float(np.mean([reached[regime]
                                                           for reached in alarmed]))}
                   for regime in REGIMES}

        assert calibration.realized_alarm_rate_at_reference == by_hand
        assert calibration.realized_alarm_rate_combined_at_reference == float(np.mean(
            [any(reached.values()) for reached in alarmed]))


class TestTheDiagnosticBatchDrawsFromNoCalibrationsStream:
    ERAS = (1, 2, 3, 17)

    @staticmethod
    def _first_draws(seed):
        return np.random.default_rng(seed).integers(0, 2**32, 8).tolist()

    def test_the_deployed_diagnostic_shares_no_stream_with_any_calibration(self):
        from rcv.cusum import at_reference_replay_seed

        seed = 42
        diagnostic = self._first_draws(at_reference_replay_seed(seed))
        assert diagnostic != self._first_draws(seed), "shares the deployed calibration's stream"
        for era in self.ERAS:
            assert diagnostic != self._first_draws([seed, era]), f"shares era {era}'s stream"

    def test_each_eras_diagnostic_shares_no_stream_with_any_calibration(self):
        from rcv.cusum import at_reference_replay_seed

        seed = 42
        for era in self.ERAS:
            diagnostic = self._first_draws(at_reference_replay_seed([seed, era]))
            assert diagnostic != self._first_draws(seed)
            for other in self.ERAS:
                assert diagnostic != self._first_draws([seed, other]), (era, other)

    def test_the_trailing_zero_equivalence_this_guards_against_is_real(self):
        assert self._first_draws([42, 0]) == self._first_draws(42)
        assert self._first_draws([42, 1, 0]) == self._first_draws([42, 1])


class TestTheCalibrationSeparatesTheReplayLengthFromTheMonitoredHorizon:
    def test_the_old_ambiguous_name_is_gone(self):
        from rcv.belief_bank import BeliefBankCalibration

        assert "stream_length" not in BeliefBankCalibration.__dataclass_fields__

    def test_a_study_records_the_replay_length_and_the_length_it_watched(self, tmp_path):
        from rcv.study import run_study

        prefix_length = 100
        frame = _stream_frame(prefix_length)
        stream_rows = int(frame["is_stream"].sum())
        result = run_study(_prefix_config(tmp_path, frame, prefix_length))

        assert result.monitor.deployed.replay_stream_length == 120
        assert result.monitor.deployed.monitored_length == stream_rows - prefix_length


class TestTheAlarmPositionIsData:
    def test_the_alarm_event_carries_its_regime_and_both_positions(self, tmp_path):
        from rcv.study import run_study

        prefix_length = 100
        result = run_study(_prefix_config(tmp_path, _stream_frame(prefix_length), prefix_length))
        alarm = next(event for event in result.loop.events if event.kind == "alarm")

        assert alarm.regime in (0, 1)
        assert alarm.monitored_position >= 0
        assert alarm.stream_position == alarm.monitored_position + prefix_length
        assert str(alarm.monitored_position) in alarm.detail

    def test_an_evaluation_reference_leaves_the_two_positions_equal(self, tmp_path):
        from rcv.study import run_study

        config_path = write_v1_config(tmp_path, write_frame(tmp_path, build_fixture_frame()))
        result = run_study(config_path)
        alarm = next(event for event in result.loop.events if event.kind == "alarm")

        assert alarm.stream_position == alarm.monitored_position

    def test_the_named_reader_handles_a_record_from_either_era(self):
        from rcv.runner import alarm_of

        after = {"kind": "alarm", "detail": "regime 1, boundary 0.9, at item 40", "regime": 1,
                 "boundary": 0.9, "monitored_position": 40, "stream_position": 2040}
        before = {"kind": "alarm", "detail": "regime 1, at item 40"}
        assert alarm_of(after) == {"regime": 1, "boundary": 0.9, "monitored_position": 40,
                                   "stream_position": 2040}
        assert alarm_of(before) == {"regime": 1, "boundary": None, "monitored_position": 40,
                                    "stream_position": None}


class TestWireKeyedFieldsSurviveTheRecord:
    WIRE_KEYED = ("reference", "threshold", "realized_alarm_rate_replayed",
                  "realized_alarm_rate_at_reference", "detectable_shift", "break_even")

    def test_every_wire_keyed_field_reads_back_with_the_keys_the_code_speaks_in(self, tmp_path):
        from rcv.runner import as_written, record_of, wire_map_of
        from rcv.study import run_study

        result = run_study(write_v1_config(tmp_path, write_frame(tmp_path,
                                                                 build_fixture_frame())))
        written = json.loads(as_written(record_of(result, provenance={})))
        deployed = written["monitor"]["deployed"]
        boundaries = deployed["boundaries"]

        for field in self.WIRE_KEYED:
            assert [row["regime"] for row in deployed[field]] == [0] * len(boundaries) + [
                1] * len(boundaries), field
            restored = wire_map_of(deployed[field])
            assert set(restored) == {0, 1}, field
            assert set(restored[0]) == set(boundaries), field

