import numpy as np
from scipy.stats import norm

from rcv.belief_bank import (
    BeliefBankMonitor,
    bin_counts_of,
    events_of,
    wire_peak_accumulation,
)
from rcv.guards import REGIMES

CALIBRATION_LEVEL = 0.05
INDIFFERENCE_ZONE_QUANTILE = {"indifference_zone_quantile": 0.99}
FLIP_BOUNDARY = 0.5
FLIP_BANK = (FLIP_BOUNDARY,)

BELOW_THE_BOUNDARY = 0.1
ABOVE_THE_BOUNDARY = 0.9

REFERENCE_RATE = 0.12
DRIFTED_RATE = 0.14
ELEVATED_RATE = 0.20
MEMORYLESS_WINDOW = 200
MATERIAL_ROWS_PER_REGIME = 20000


def drift_free_material(rate=REFERENCE_RATE, rows_per_regime=MATERIAL_ROWS_PER_REGIME):
    belief, verdict = [], []
    for regime in REGIMES:
        column = np.full(rows_per_regime, ABOVE_THE_BOUNDARY)
        column[: round(rate * rows_per_regime)] = BELOW_THE_BOUNDARY
        belief.append(column)
        verdict.append(np.full(rows_per_regime, regime))
    return np.concatenate(belief), np.concatenate(verdict)


def calibrated_monitor(stream_length, n_streams, seed, calibration_level=CALIBRATION_LEVEL,
                       detectable_shift=INDIFFERENCE_ZONE_QUANTILE,
                       reference_rate=REFERENCE_RATE,
                       rows_per_regime=MATERIAL_ROWS_PER_REGIME):
    monitor = BeliefBankMonitor(calibration_level, detectable_shift, boundaries=FLIP_BANK)
    monitor.calibrate(*drift_free_material(reference_rate, rows_per_regime),
                      stream_length=stream_length, n_streams=n_streams, seed=seed)
    return monitor


def build_stream(length, seed, rate_by_regime):
    draw = np.random.default_rng(seed)
    verdict = draw.integers(0, 2, size=length)
    rate = np.where(verdict == 1, rate_by_regime[1], rate_by_regime[0])
    fired = draw.random(length) < rate
    return np.where(fired, BELOW_THE_BOUNDARY, ABOVE_THE_BOUNDARY), verdict


def fired_of(belief, boundary=FLIP_BOUNDARY):
    return np.asarray(belief) < boundary


def window_rates(fired, window=MEMORYLESS_WINDOW):
    usable = (len(fired) // window) * window
    return fired[:usable].reshape(-1, window).mean(axis=1)


def memoryless_critical_rate():
    standard_error = np.sqrt(REFERENCE_RATE * (1 - REFERENCE_RATE) / MEMORYLESS_WINDOW)
    return REFERENCE_RATE + norm.isf(CALIBRATION_LEVEL / len(REGIMES)) * standard_error


def binomial_tolerance(rate, n_streams):
    return 3.0 * np.sqrt(rate * (1 - rate) / n_streams)


class TestP29ExcessAccumulatedAcrossItems:
    def test_an_excess_no_single_window_can_see_alarms_across_a_long_stream(self):
        monitor = calibrated_monitor(stream_length=20000, n_streams=600, seed=7)
        belief, verdict = build_stream(20000, seed=2103,
                                       rate_by_regime={0: REFERENCE_RATE, 1: DRIFTED_RATE})
        quiet_belief, quiet_verdict = build_stream(
            20000, seed=2103, rate_by_regime={0: REFERENCE_RATE, 1: REFERENCE_RATE})

        quiet_windows = window_rates(fired_of(quiet_belief)[quiet_verdict == 1])
        drifted_windows = window_rates(fired_of(belief)[verdict == 1])
        assert DRIFTED_RATE - REFERENCE_RATE < quiet_windows.std(ddof=1)
        assert np.mean(drifted_windows > memoryless_critical_rate()) < 0.5

        alarm = monitor.observe(belief, verdict)
        assert alarm is not None
        assert alarm.regime == 1

        monitor.reset()
        assert monitor.observe(quiet_belief, quiet_verdict) is None


class TestP31TheThresholdIsReadFromDriftFreeBehaviour:
    def test_drift_free_traffic_alarms_no_more_often_than_the_calibration_level(self):
        monitor = calibrated_monitor(stream_length=2000, n_streams=2000, seed=13)

        fresh_streams = 800
        draw = np.random.default_rng(4242)
        alarms = 0
        for _ in range(fresh_streams):
            monitor.reset()
            verdict = draw.integers(0, 2, size=2000)
            fired = draw.random(2000) < REFERENCE_RATE
            alarms += monitor.observe(
                np.where(fired, BELOW_THE_BOUNDARY, ABOVE_THE_BOUNDARY), verdict) is not None
        assert alarms / fresh_streams <= CALIBRATION_LEVEL + binomial_tolerance(CALIBRATION_LEVEL,
                                                                               fresh_streams)

    def test_the_sequential_alarm_agrees_with_the_accumulation_the_replay_measures(self):
        monitor = calibrated_monitor(stream_length=2000, n_streams=1000, seed=13)
        calibration = monitor.calibration
        for seed in (11, 12, 13, 14):
            monitor.reset()
            belief, verdict = build_stream(2000, seed=seed,
                                           rate_by_regime={0: REFERENCE_RATE, 1: ELEVATED_RATE})
            event = events_of(belief, FLIP_BANK)[0]
            reached = any(
                wire_peak_accumulation(event, verdict == regime,
                                       calibration.reference[regime][FLIP_BOUNDARY],
                                       calibration.detectable_shift[regime][FLIP_BOUNDARY],
                                       regime) >= calibration.threshold[regime][FLIP_BOUNDARY]
                for regime in REGIMES)
            assert (monitor.observe(belief, verdict) is not None) == reached, seed


class DictatedRatesReplay:
    def __init__(self, rate_by_regime, uniform):
        self._rates, self._uniform = list(rate_by_regime), uniform

    def integers(self, low, high, size):
        return np.arange(size) % high

    def beta(self, *_parameters):
        return self._rates.pop(0)

    def random(self, size):
        return np.full(size, self._uniform)


class RecordingReplay:
    def __init__(self, generator, calls):
        self._draw = generator
        self.calls = calls

    def integers(self, *args, **kwargs):
        self.calls.append("integers")
        return self._draw.integers(*args, **kwargs)

    def beta(self, *args, **kwargs):
        self.calls.append("beta")
        return self._draw.beta(*args, **kwargs)

    def random(self, *args, **kwargs):
        self.calls.append("random")
        return self._draw.random(*args, **kwargs)


DRAW_SEQUENCE_RATE = 0.30
DRAW_SEQUENCE_ROWS = 200
DRAW_SEQUENCE_LENGTH = 200
DRAW_SEQUENCE_STREAMS = 400
DRAW_SEQUENCE_LEVEL = 0.10


class TestTheReplayDrawsWhatTheCalibrationNames:
    def test_each_regimes_events_are_bernoulli_at_that_regimes_own_drawn_rate(self):
        belief, verdict = drift_free_material(rate=DRAW_SEQUENCE_RATE, rows_per_regime=4)
        monitor = BeliefBankMonitor(DRAW_SEQUENCE_LEVEL, INDIFFERENCE_ZONE_QUANTILE,
                                    boundaries=FLIP_BANK)
        posterior = monitor._dirichlet_posterior(
            {regime: bin_counts_of(belief, verdict, FLIP_BANK, regime) for regime in REGIMES})
        reference = {regime: {FLIP_BOUNDARY: DRAW_SEQUENCE_RATE} for regime in REGIMES}
        shift = {regime: {FLIP_BOUNDARY: 0.10} for regime in REGIMES}
        shared = (events_of(belief, FLIP_BANK), verdict, reference, shift, len(verdict), 1)

        fires_in_regime_zero = monitor._replayed_peaks(
            *shared, DictatedRatesReplay([0.9, 0.1], 0.5), posterior)
        fires_in_regime_one = monitor._replayed_peaks(
            *shared, DictatedRatesReplay([0.1, 0.9], 0.5), posterior)
        assert fires_in_regime_zero[0][FLIP_BOUNDARY][0] > 0.0
        assert fires_in_regime_zero[1][FLIP_BOUNDARY][0] == 0.0
        assert fires_in_regime_one[0][FLIP_BOUNDARY][0] == 0.0
        assert fires_in_regime_one[1][FLIP_BOUNDARY][0] > 0.0

    def test_the_calibration_spends_one_draw_set_per_replayed_stream_in_each_of_its_batches(
            self, monkeypatch):
        real_default_rng = np.random.default_rng
        recorded = []
        monkeypatch.setattr(
            np.random, "default_rng",
            lambda seed, _real=real_default_rng, _log=recorded: RecordingReplay(_real(seed),
                                                                                _log))
        monitor = BeliefBankMonitor(DRAW_SEQUENCE_LEVEL, INDIFFERENCE_ZONE_QUANTILE,
                                    boundaries=FLIP_BANK)
        calibration = monitor.calibrate(
            *drift_free_material(rate=DRAW_SEQUENCE_RATE, rows_per_regime=DRAW_SEQUENCE_ROWS),
            stream_length=DRAW_SEQUENCE_LENGTH, n_streams=DRAW_SEQUENCE_STREAMS, seed=5)

        assert calibration.detectable_shift == calibration.resolved_detectable_shift, (
            "the proviso must stay inert here, or the reachability search is more than one batch")
        per_stream = ["integers", "beta", "beta", "random"]
        assert recorded == (per_stream * (3 * DRAW_SEQUENCE_STREAMS)
                            + ["integers"] * DRAW_SEQUENCE_STREAMS)
