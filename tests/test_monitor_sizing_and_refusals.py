import re

import numpy as np
import pytest
from scipy import stats

from rcv.belief_bank import BeliefBankCalibration, BeliefBankMonitor
from rcv.cusum import (
    LARGEST_SHIFT_FRACTION,
    break_even_rate,
    log_likelihood_ratio_increments,
    noise_boundary_shift,
    positions_to_alarm,
)
from rcv.guards import REGIMES

CALIBRATION_LEVEL = 0.05
WIDE_LEVEL = 0.5
CALIBRATION_STREAMS = 400
FLIP_BOUNDARY = 0.5
FLIP_BANK = (FLIP_BOUNDARY,)
SIGNED_QUANTILE = 0.977
NOISE_BOUNDARY = {"indifference_zone_quantile": SIGNED_QUANTILE}

BELOW_THE_BOUNDARY = 0.1
ABOVE_THE_BOUNDARY = 0.9

SUBSTRATE_PREFIX = (1644, 34, 356, 65)


def material_of(n0, k0, n1, k1):
    verdict = np.concatenate([np.zeros(n0, dtype=int), np.ones(n1, dtype=int)])
    belief = np.full(n0 + n1, ABOVE_THE_BOUNDARY)
    belief[:k0] = BELOW_THE_BOUNDARY
    belief[n0:n0 + k1] = BELOW_THE_BOUNDARY
    return belief, verdict


def jeffreys_posterior(items, flipped):
    return flipped + 0.5, items - flipped + 0.5


class TestTheSizingArithmeticPredictsWhereTheAlarmFires:
    REFERENCE = 0.10
    SHIFT = 0.06
    SHARE = 0.5
    THRESHOLD = 6.0

    def monitor_at(self, threshold):
        monitor = BeliefBankMonitor(CALIBRATION_LEVEL, NOISE_BOUNDARY, boundaries=FLIP_BANK)
        monitor.calibration = BeliefBankCalibration(
            reference={regime: {FLIP_BOUNDARY: self.REFERENCE} for regime in REGIMES},
            threshold={regime: {FLIP_BOUNDARY: threshold} for regime in REGIMES},
            realized_alarm_rate_replayed={regime: {FLIP_BOUNDARY: 0.0} for regime in REGIMES},
            realized_alarm_rate_combined_replayed=0.0, replay_stream_length=1,
            n_streams=1, seed=0,
            detectable_shift={regime: {FLIP_BOUNDARY: self.SHIFT} for regime in REGIMES},
            boundaries=FLIP_BANK)
        return monitor

    def design_rate_stream(self, length, seed):
        draw = np.random.default_rng(seed)
        verdict = (draw.random(length) < self.SHARE).astype(np.int64)
        rate = np.where(verdict == 1, self.REFERENCE + self.SHIFT, self.REFERENCE)
        fired = draw.random(length) < rate
        return np.where(fired, BELOW_THE_BOUNDARY, ABOVE_THE_BOUNDARY), verdict

    def test_the_alarm_lands_where_the_sizing_says_it_will(self):
        predicted = positions_to_alarm(self.THRESHOLD, self.REFERENCE, self.SHIFT, self.SHARE,
                                       regime=1)
        observed = []
        for seed in range(40):
            monitor = self.monitor_at(self.THRESHOLD)
            belief, verdict = self.design_rate_stream(int(20 * predicted), seed)
            alarm = monitor.observe(belief, verdict)
            assert alarm is not None and alarm.regime == 1, seed
            observed.append(alarm.item_index)
        assert float(np.mean(observed)) == pytest.approx(predicted, rel=0.25)

    def test_a_regime_carrying_less_of_the_traffic_needs_proportionally_more_stream(self):
        whole = positions_to_alarm(self.THRESHOLD, self.REFERENCE, self.SHIFT, 1.0, regime=1)
        half = positions_to_alarm(self.THRESHOLD, self.REFERENCE, self.SHIFT, 0.5, regime=1)
        assert half == pytest.approx(2.0 * whole)

    def test_a_higher_threshold_needs_proportionally_more_stream(self):
        single = positions_to_alarm(1.0, self.REFERENCE, self.SHIFT, self.SHARE, regime=1)
        triple = positions_to_alarm(3.0, self.REFERENCE, self.SHIFT, self.SHARE, regime=1)
        assert triple == pytest.approx(3.0 * single)


class TestEveryRefusalNamesTheRegimeItIsAbout:
    ZERO_REFERENCE_REFUSAL = "Reference flip rate for regime 1 must be in (0, 1); got 0.0."

    def test_the_break_even_rate_refuses_naming_its_regime(self):
        with pytest.raises(ValueError, match=re.escape(self.ZERO_REFERENCE_REFUSAL)):
            break_even_rate(0.0, 0.05, regime=1)

    def test_the_sizing_refuses_naming_its_regime(self):
        with pytest.raises(ValueError, match=re.escape(self.ZERO_REFERENCE_REFUSAL)):
            positions_to_alarm(1.0, 0.0, 0.05, 0.5, regime=1)

    def test_the_noise_boundary_refuses_naming_its_regime(self):
        with pytest.raises(ValueError, match=re.escape(self.ZERO_REFERENCE_REFUSAL)):
            noise_boundary_shift(0.0, jeffreys_posterior(500, 0), SIGNED_QUANTILE, regime=1)


class TestTheIncrementOneItemContributes:
    @pytest.mark.parametrize("reference_rate", [0.0, 1.0])
    def test_a_reference_at_certainty_has_no_defined_increment(self, reference_rate):
        with pytest.raises(ValueError, match=(
                rf"^Reference flip rate for regime 1 must be in \(0, 1\); got "
                rf"{reference_rate}\.$")):
            log_likelihood_ratio_increments(reference_rate, 0.02, regime=1)

    def test_a_shift_that_lands_exactly_on_certainty_is_refused(self):
        with pytest.raises(ValueError, match=(
                r"^Shifted flip rate for regime 0 must be less than 1; got 1\.0 "
                r"\(reference 0\.5, shift 0\.5\)\.$")):
            log_likelihood_ratio_increments(0.5, 0.5, regime=0)


class TestTheNoiseBoundaryRefusesWhatItCannotResolve:
    def test_a_quantile_that_lands_on_the_reference_itself_names_no_shift(self):
        posterior = jeffreys_posterior(1644, 34)
        target = float(stats.beta(*posterior).ppf(SIGNED_QUANTILE))
        with pytest.raises(ValueError, match=re.escape(
                f"Posterior quantile for regime 0 must exceed the reference rate; got {target} "
                f"at quantile {SIGNED_QUANTILE}, reference {target}.")):
            noise_boundary_shift(target, posterior, SIGNED_QUANTILE, regime=0)

    def test_a_quantile_no_admissible_shift_reaches_refuses_in_full(self):
        reference, quantile = 0.9, 0.999
        posterior = jeffreys_posterior(10, 9)
        target = float(stats.beta(*posterior).ppf(quantile))
        assert target > break_even_rate(reference, (1.0 - reference) * LARGEST_SHIFT_FRACTION,
                                        regime=1), "the premise: no admissible shift reaches it"
        with pytest.raises(ValueError, match=re.escape(
                f"Posterior quantile {target} for regime 1 is unreachable from reference "
                f"{reference} within the allowed shift range.")):
            noise_boundary_shift(reference, posterior, quantile, regime=1)

    def test_a_calibration_carries_the_regime_into_the_boundarys_refusal(self):
        monitor = BeliefBankMonitor(CALIBRATION_LEVEL, {"indifference_zone_quantile": 0.999},
                                    boundaries=FLIP_BANK)
        with pytest.raises(ValueError, match=r"^Posterior quantile .* for regime 1 is unreachable "):
            monitor.calibrate(*material_of(1644, 34, 10, 9), stream_length=2000,
                              n_streams=CALIBRATION_STREAMS, seed=0)


class TestAHorizonNoShiftCanAlarmWithin:
    HORIZON = 5

    def test_the_refusal_reads_in_full(self):
        monitor = BeliefBankMonitor(WIDE_LEVEL, NOISE_BOUNDARY, boundaries=FLIP_BANK)
        with pytest.raises(ValueError, match=re.escape(
                f"Regime 1, boundary {FLIP_BOUNDARY}: estimated positions to alarm exceed "
                f"the {self.HORIZON}-position horizon at the shift ceiling.")):
            monitor.calibrate(*material_of(*SUBSTRATE_PREFIX), stream_length=self.HORIZON,
                              n_streams=CALIBRATION_STREAMS, seed=0)

    def test_a_horizon_long_enough_deploys_a_raised_shift_instead_of_refusing(self):
        monitor = BeliefBankMonitor(WIDE_LEVEL, NOISE_BOUNDARY, boundaries=FLIP_BANK)
        calibration = monitor.calibrate(*material_of(*SUBSTRATE_PREFIX), stream_length=200,
                                        n_streams=CALIBRATION_STREAMS, seed=0)
        assert (calibration.detectable_shift[1][FLIP_BOUNDARY]
                > calibration.resolved_detectable_shift[1][FLIP_BOUNDARY])
