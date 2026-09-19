import re

import numpy as np
import pytest
from scipy import stats

from rcv.belief_bank import BeliefBankMonitor, bin_counts_of, events_of
from rcv.cusum import (
    break_even_rate,
    log_likelihood_ratio_increments,
    noise_boundary_shift,
    positions_to_alarm,
)
from rcv.guards import REGIMES
from rcv.study import validate_config

CALIBRATION_LEVEL = 0.05
SIGNED_QUANTILE = 0.977
NOISE_BOUNDARY = {"indifference_zone_quantile": SIGNED_QUANTILE}
FLIP_BOUNDARY = 0.5
FLIP_BANK = (FLIP_BOUNDARY,)

BELOW_THE_BOUNDARY = 0.1
ABOVE_THE_BOUNDARY = 0.9

SUBSTRATE_PREFIX_FLIP_COUNTS = {
    0: ((1644, 34), (356, 65)),
    1: ((1644, 35), (356, 55)),
    2: ((1644, 30), (356, 69)),
    42: ((1644, 33), (356, 49)),
    123: ((1644, 34), (356, 51)),
}
VERIFIED_RESOLVED_SHIFT = {
    0: {0: 0.01753, 1: 0.09118},
    1: {0: 0.01772, 1: 0.08696},
    2: {0: 0.01673, 1: 0.09266},
    42: {0: 0.01734, 1: 0.08403},
    123: {0: 0.01753, 1: 0.08505},
}
VERIFIED_RESOLVED_MEAN = {0: 0.01737, 1: 0.08798}
VERIFIED_RESOLVED_SD = {0: 0.00038, 1: 0.00379}
SUBSTRATE_HORIZON = 19000
SUBSTRATE_SHARE = {0: 1644 / 2000, 1: 356 / 2000}
CALIBRATION_STREAMS = 400


def material_of(n0, k0, n1, k1):
    verdict = np.concatenate([np.zeros(n0, dtype=int), np.ones(n1, dtype=int)])
    belief = np.full(n0 + n1, ABOVE_THE_BOUNDARY)
    belief[:k0] = BELOW_THE_BOUNDARY
    belief[n0:n0 + k1] = BELOW_THE_BOUNDARY
    return belief, verdict


def substrate_material(seed):
    (n0, k0), (n1, k1) = SUBSTRATE_PREFIX_FLIP_COUNTS[seed]
    return material_of(n0, k0, n1, k1)


def jeffreys_posterior(items, flipped):
    return flipped + 0.5, items - flipped + 0.5


def monitor_for(quantile=SIGNED_QUANTILE, calibration_level=CALIBRATION_LEVEL):
    return BeliefBankMonitor(calibration_level, {"indifference_zone_quantile": quantile},
                             boundaries=FLIP_BANK)


def calibrated_on_the_substrate(seed, quantile=SIGNED_QUANTILE,
                                stream_length=SUBSTRATE_HORIZON,
                                calibration_level=CALIBRATION_LEVEL):
    return monitor_for(quantile, calibration_level).calibrate(
        *substrate_material(seed), stream_length=stream_length,
        n_streams=CALIBRATION_STREAMS, seed=seed)


class TestTheIndifferenceZoneQuantileIsConfiguration:
    def test_the_noise_boundary_specification_is_accepted(self):
        assert monitor_for().detectable_shift == NOISE_BOUNDARY

    def test_a_configuration_naming_the_noise_boundary_validates(self):
        validate_config(config_with_shift(NOISE_BOUNDARY))

    def test_a_configuration_naming_a_quantile_outside_the_band_refuses(self):
        with pytest.raises(ValueError, match="indifference_zone_quantile"):
            validate_config(config_with_shift({"indifference_zone_quantile": 0.5}))

    @pytest.mark.parametrize("quantile", [0.5, 0.0, -0.1, 1.0, 1.5])
    def test_a_quantile_outside_the_open_unit_upper_half_refuses(self, quantile):
        with pytest.raises(ValueError, match="indifference_zone_quantile"):
            monitor_for(quantile=quantile)

    def test_a_noise_boundary_naming_no_quantile_refuses_and_reads_in_full(self):
        with pytest.raises(ValueError, match=re.escape(
                "detectable_shift is missing required key 'indifference_zone_quantile'.")):
            BeliefBankMonitor(CALIBRATION_LEVEL, {}, boundaries=FLIP_BANK)

    def test_a_noise_boundary_carrying_a_foreign_key_refuses_naming_it(self):
        with pytest.raises(ValueError, match="factor"):
            BeliefBankMonitor(CALIBRATION_LEVEL,
                              {"indifference_zone_quantile": SIGNED_QUANTILE, "factor": 0.5},
                              boundaries=FLIP_BANK)


def config_with_shift(shift):
    return {
        "study": "noise-boundary-configuration",
        "seeds": [0],
        "frame": {"path": "frames/none.npz", "name": "none", "fitting_labels": "oracle"},
        "split": {"evaluation_fraction": 0.2, "calibration_fraction": 0.4375},
        "estimator": {"probe": {"family": "linear"}, "calibration": {"method": "platt"},
                      "route_probe": True, "route_calibration": True},
        "flip": {"threshold": FLIP_BOUNDARY},
        "monitor": {"quiet_horizon_confidence": 1.0 - CALIBRATION_LEVEL,
                    "detectable_shift": shift,
                    "event_bank": {"boundaries": list(FLIP_BANK), "budget_allocation": "joint"},
                    "calibration": {"material": "stream_prefix", "prefix_length": 2000,
                                    "stream_length": 19000, "n_streams": 400, "streams_per_centre": 1}},
        "loop": {"audit_sampling_window": 500, "audit_budget": 318,
                 "acceptance": {"recall_floor": 0.5, "false_positive_tolerance": 0.25},
                 "max_enlarging_retries": 2, "cross_fit_folds": 4, "observe_only": True},
    }


class TestTheBreakEvenRateIsWhereTheAccumulationTurns:
    @pytest.mark.parametrize("reference,shift", [(0.02, 0.01), (0.16, 0.08), (0.30, 0.05)])
    def test_the_expected_increment_at_the_break_even_rate_is_zero(self, reference, shift):
        rate = break_even_rate(reference, shift, regime=0)
        flip, no_flip = log_likelihood_ratio_increments(reference, shift, regime=0)
        assert rate * flip + (1 - rate) * no_flip == pytest.approx(0.0, abs=1e-15)

    @pytest.mark.parametrize("reference,shift", [(0.02, 0.01), (0.16, 0.08)])
    def test_the_break_even_sits_between_the_reference_and_the_shifted_rate(self, reference,
                                                                            shift):
        assert reference < break_even_rate(reference, shift, regime=0) < reference + shift

    def test_the_break_even_rises_with_the_shift(self):
        rates = [break_even_rate(0.02, shift, regime=0)
                 for shift in (0.001, 0.005, 0.01, 0.05, 0.2)]
        assert rates == sorted(rates)
        assert len(set(rates)) == len(rates)


class TestTheResolvedShiftPutsTheBreakEvenOnTheQuantile:
    @pytest.mark.parametrize("seed", sorted(SUBSTRATE_PREFIX_FLIP_COUNTS))
    @pytest.mark.parametrize("regime", REGIMES)
    def test_the_resolved_shifts_break_even_is_the_exact_beta_quantile(self, seed, regime):
        (n0, k0), (n1, k1) = SUBSTRATE_PREFIX_FLIP_COUNTS[seed]
        items, flipped = ((n0, k0), (n1, k1))[regime]
        posterior = jeffreys_posterior(items, flipped)
        shift = noise_boundary_shift(flipped / items, posterior, SIGNED_QUANTILE, regime)
        target = float(stats.beta(*posterior).ppf(SIGNED_QUANTILE))
        assert break_even_rate(flipped / items, shift, regime) == pytest.approx(target,
                                                                                abs=1e-12)

    def test_the_posterior_is_the_jeffreys_one_the_reference_count_names(self):
        assert jeffreys_posterior(1644, 34) == (34.5, 1610.5)
        assert jeffreys_posterior(356, 65) == (65.5, 291.5)

    def test_a_higher_quantile_asks_a_larger_shift(self):
        posterior = jeffreys_posterior(1644, 34)
        shifts = [noise_boundary_shift(34 / 1644, posterior, quantile, regime=0)
                  for quantile in (0.90, 0.925, 0.95, 0.975, 0.977, 0.99)]
        assert shifts == sorted(shifts)

    @pytest.mark.parametrize("seed", sorted(SUBSTRATE_PREFIX_FLIP_COUNTS))
    def test_the_resolved_shift_reproduces_the_verified_figures(self, seed):
        (n0, k0), (n1, k1) = SUBSTRATE_PREFIX_FLIP_COUNTS[seed]
        for regime, (items, flipped) in enumerate(((n0, k0), (n1, k1))):
            shift = noise_boundary_shift(flipped / items, jeffreys_posterior(items, flipped),
                                         SIGNED_QUANTILE, regime)
            assert shift == pytest.approx(VERIFIED_RESOLVED_SHIFT[seed][regime], abs=5e-6)

    def test_the_resolved_shift_reproduces_the_verified_spread_across_the_five_seeds(self):
        for regime in REGIMES:
            drawn = []
            for seed in sorted(SUBSTRATE_PREFIX_FLIP_COUNTS):
                items, flipped = SUBSTRATE_PREFIX_FLIP_COUNTS[seed][regime]
                drawn.append(noise_boundary_shift(flipped / items,
                                                  jeffreys_posterior(items, flipped),
                                                  SIGNED_QUANTILE, regime))
            assert float(np.mean(drawn)) == pytest.approx(VERIFIED_RESOLVED_MEAN[regime],
                                                          abs=5e-6)
            assert float(np.std(drawn, ddof=1)) == pytest.approx(VERIFIED_RESOLVED_SD[regime],
                                                                 abs=5e-6)

    def test_the_two_regimes_resolve_to_shifts_no_single_factor_expresses(self):
        calibration = calibrated_on_the_substrate(seed=0)
        as_a_factor = {regime: (calibration.detectable_shift[regime][FLIP_BOUNDARY]
                                / calibration.reference[regime][FLIP_BOUNDARY])
                       for regime in REGIMES}
        assert as_a_factor[0] == pytest.approx(0.848, abs=0.02)
        assert as_a_factor[1] == pytest.approx(0.499, abs=0.02)
        assert abs(as_a_factor[0] - as_a_factor[1]) > 0.1

    def test_a_reference_the_material_never_showed_refuses(self):
        with pytest.raises(ValueError, match="Reference flip rate"):
            monitor_for().calibrate(*material_of(500, 0, 500, 0), stream_length=1000,
                                    n_streams=CALIBRATION_STREAMS, seed=0)


class TestTheCalibrationRecordsWhatWasResolved:
    def test_the_calibration_carries_the_resolved_shift_the_deployed_shift_and_the_quantile(self):
        calibration = calibrated_on_the_substrate(seed=0)
        assert calibration.indifference_zone_quantile == SIGNED_QUANTILE
        for regime in REGIMES:
            assert calibration.resolved_detectable_shift[regime][FLIP_BOUNDARY] > 0.0
            assert calibration.detectable_shift[regime][FLIP_BOUNDARY] > 0.0
            assert (calibration.break_even[regime][FLIP_BOUNDARY]
                    > calibration.reference[regime][FLIP_BOUNDARY])
            assert (calibration.reference_posterior_quantile[regime][FLIP_BOUNDARY]
                    > calibration.reference[regime][FLIP_BOUNDARY])

    def test_the_deployed_break_even_sits_on_the_quantile_where_reachability_does_not_bind(self):
        calibration = calibrated_on_the_substrate(seed=0)
        for regime in REGIMES:
            assert (calibration.detectable_shift[regime][FLIP_BOUNDARY]
                    == calibration.resolved_detectable_shift[regime][FLIP_BOUNDARY])
            assert calibration.break_even[regime][FLIP_BOUNDARY] == pytest.approx(
                calibration.reference_posterior_quantile[regime][FLIP_BOUNDARY], abs=1e-12)

    def test_the_recorded_break_even_is_the_one_the_deployed_shift_implies(self):
        calibration = calibrated_on_the_substrate(seed=0)
        for regime in REGIMES:
            assert calibration.break_even[regime][FLIP_BOUNDARY] == break_even_rate(
                calibration.reference[regime][FLIP_BOUNDARY],
                calibration.detectable_shift[regime][FLIP_BOUNDARY], regime)

    def test_the_recorded_quantile_is_the_exact_beta_quantile_of_the_reference_count(self):
        calibration = calibrated_on_the_substrate(seed=42)
        (n0, k0), (n1, k1) = SUBSTRATE_PREFIX_FLIP_COUNTS[42]
        for regime, (items, flipped) in enumerate(((n0, k0), (n1, k1))):
            assert calibration.reference_posterior_quantile[regime][FLIP_BOUNDARY] == (
                pytest.approx(
                    float(stats.beta(*jeffreys_posterior(items, flipped)).ppf(SIGNED_QUANTILE)),
                    abs=1e-15))

    def test_the_monitor_watches_traffic_against_the_resolved_shift(self):
        monitor = monitor_for()
        calibration = monitor.calibrate(*substrate_material(0), stream_length=2000,
                                        n_streams=CALIBRATION_STREAMS, seed=0)
        on_event, _ = log_likelihood_ratio_increments(
            calibration.reference[0][FLIP_BOUNDARY],
            calibration.detectable_shift[0][FLIP_BOUNDARY], 0)
        monitor.observe(np.array([BELOW_THE_BOUNDARY]), np.array([0]))
        assert monitor.accumulation[0][FLIP_BOUNDARY] == on_event


class TestTheShiftIsRaisedUntilTheThresholdIsReachable:
    SHORT_HORIZON = 400
    SHORT_HORIZON_LEVEL = 0.2

    def calibrated(self):
        return calibrated_on_the_substrate(seed=0, stream_length=self.SHORT_HORIZON,
                                           calibration_level=self.SHORT_HORIZON_LEVEL)

    def test_the_resolved_shift_is_unreachable_on_a_horizon_this_short(self):
        (items, flipped) = SUBSTRATE_PREFIX_FLIP_COUNTS[0][0]
        resolved = noise_boundary_shift(flipped / items, jeffreys_posterior(items, flipped),
                                        SIGNED_QUANTILE, regime=0)
        calibration = self.calibrated()
        assert positions_to_alarm(calibration.threshold[0][FLIP_BOUNDARY],
                                  calibration.reference[0][FLIP_BOUNDARY], resolved,
                                  SUBSTRATE_SHARE[0], regime=0) > self.SHORT_HORIZON

    def test_the_deployed_shift_is_raised_above_the_resolved_one(self):
        calibration = self.calibrated()
        raised = [regime for regime in REGIMES
                  if calibration.detectable_shift[regime][FLIP_BOUNDARY]
                  > calibration.resolved_detectable_shift[regime][FLIP_BOUNDARY]]
        assert raised, "no regime was raised, so this horizon does not exercise the proviso"
        for regime in raised:
            assert (calibration.break_even[regime][FLIP_BOUNDARY]
                    > calibration.reference_posterior_quantile[regime][FLIP_BOUNDARY])

    def test_the_deployed_shift_is_reachable_within_the_calibrated_horizon(self):
        calibration = self.calibrated()
        for regime in REGIMES:
            assert positions_to_alarm(calibration.threshold[regime][FLIP_BOUNDARY],
                                      calibration.reference[regime][FLIP_BOUNDARY],
                                      calibration.detectable_shift[regime][FLIP_BOUNDARY],
                                      SUBSTRATE_SHARE[regime],
                                      regime) <= self.SHORT_HORIZON

    def test_the_raise_stops_at_the_smallest_reachable_shift(self):
        calibration = self.calibrated()
        belief, verdict = substrate_material(0)
        monitor = monitor_for(calibration_level=self.SHORT_HORIZON_LEVEL)
        posterior = monitor._dirichlet_posterior(
            {regime: bin_counts_of(belief, verdict, FLIP_BANK, regime) for regime in REGIMES})
        for regime in REGIMES:
            if (calibration.detectable_shift[regime][FLIP_BOUNDARY]
                    == calibration.resolved_detectable_shift[regime][FLIP_BOUNDARY]):
                continue
            below = {other: dict(calibration.detectable_shift[other]) for other in REGIMES}
            below[regime][FLIP_BOUNDARY] *= 0.9
            if (below[regime][FLIP_BOUNDARY]
                    < calibration.resolved_detectable_shift[regime][FLIP_BOUNDARY]):
                continue
            threshold = monitor._thresholds_at(
                events_of(belief, FLIP_BANK), verdict, calibration.reference, below,
                self.SHORT_HORIZON, CALIBRATION_STREAMS, 0, posterior)
            assert positions_to_alarm(threshold[regime][FLIP_BOUNDARY],
                                      calibration.reference[regime][FLIP_BOUNDARY],
                                      below[regime][FLIP_BOUNDARY], SUBSTRATE_SHARE[regime],
                                      regime) > self.SHORT_HORIZON

    def test_the_signed_quantile_needs_no_raise_on_the_built_substrate(self):
        for seed in sorted(SUBSTRATE_PREFIX_FLIP_COUNTS):
            calibration = calibrated_on_the_substrate(seed=seed)
            assert calibration.detectable_shift == calibration.resolved_detectable_shift


class TestTheBoundaryComposesWithWhatWasAlreadySigned:
    def test_the_calibration_is_identical_on_a_second_run(self):
        for stream_length, level in (
                (SUBSTRATE_HORIZON, CALIBRATION_LEVEL),
                (TestTheShiftIsRaisedUntilTheThresholdIsReachable.SHORT_HORIZON,
                 TestTheShiftIsRaisedUntilTheThresholdIsReachable.SHORT_HORIZON_LEVEL)):
            first = calibrated_on_the_substrate(seed=0, stream_length=stream_length,
                                                calibration_level=level)
            second = calibrated_on_the_substrate(seed=0, stream_length=stream_length,
                                                 calibration_level=level)
            assert first == second
