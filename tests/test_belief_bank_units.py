import re

import numpy as np
import pytest
from scipy import stats

from rcv.belief_bank import (
    JOINT,
    BankAlarm,
    BeliefBankCalibration,
    BeliefBankMonitor,
    as_belief,
    assert_boundaries,
    assert_budget_allocation,
    bin_counts_of,
    events_of,
    wire_peak_accumulation,
)
from rcv.cusum import break_even_rate, log_likelihood_ratio_increments
from rcv.guards import REGIMES

BUDGET = 0.05
GENEROUS_BUDGET = 0.5
REALISTIC_BUDGET = 0.10
REPLAY_STREAMS = 600
WATCHED_LENGTH = 1500
FLIP_BOUNDARY = 0.5
SUB_FLIP_BANK = (0.5, 0.9)
SHIFT = {"indifference_zone_quantile": 0.977}
WIDER_SHIFT = {"indifference_zone_quantile": 0.999}
NOISE_BOUNDARY = {"indifference_zone_quantile": 0.977}

BELOW_THE_FLIP = 0.25
BETWEEN_THE_BOUNDARIES = 0.70
ABOVE_THE_BANK = 0.95

SUBSTRATE_PREFIX = ((1644, 34), (356, 65))


def belief_material(bins_by_regime, boundaries=SUB_FLIP_BANK):
    edges = (0.0, *boundaries, 1.0)
    inside_each_bin = [0.5 * (edges[bin_index] + edges[bin_index + 1])
                       for bin_index in range(len(edges) - 1)]
    belief, verdict = [], []
    for regime in REGIMES:
        for count, representative in zip(bins_by_regime[regime], inside_each_bin):
            belief.append(np.full(count, representative))
            verdict.append(np.full(count, regime))
    return np.concatenate(belief), np.concatenate(verdict)


def substrate_material():
    bins = {}
    for regime, (items, flipped) in enumerate(SUBSTRATE_PREFIX):
        above = items - flipped
        bins[regime] = (flipped, above // 2, above - above // 2)
    return belief_material(bins)


def flips_of(belief, boundary=FLIP_BOUNDARY):
    return np.asarray(belief) < boundary


def calibrated_bank(belief, verdict, boundaries=SUB_FLIP_BANK, budget=REALISTIC_BUDGET,
                    detectable_shift=SHIFT, budget_allocation=JOINT,
                    stream_length=WATCHED_LENGTH, n_streams=REPLAY_STREAMS, seed=3):
    monitor = BeliefBankMonitor(budget, detectable_shift, boundaries=boundaries,
                                budget_allocation=budget_allocation)
    monitor.calibrate(belief, verdict, stream_length, n_streams, seed)
    return monitor


def stream_of(length, seed, rate_by_regime, boundaries=SUB_FLIP_BANK):
    draw = np.random.default_rng(seed)
    verdict = draw.integers(0, 2, size=length)
    uniform = draw.random(length)
    rate = np.stack([np.where(verdict == 1, rate_by_regime[1][position],
                              rate_by_regime[0][position])
                     for position in range(len(boundaries))])
    below = uniform[None, :] < rate
    belief = np.full(length, ABOVE_THE_BANK)
    for position in reversed(range(len(boundaries))):
        inside = np.full(length, boundaries[position]) - 0.01
        belief = np.where(below[position], inside, belief)
    return belief, verdict


class TestTheMonitoredEventIsABeliefThresholdCrossing:
    def test_event_k_is_the_belief_falling_below_boundary_k(self):
        belief = np.array([0.1, 0.5, 0.6, 0.95, 1.0])
        fired = events_of(belief, SUB_FLIP_BANK)
        np.testing.assert_array_equal(fired[0], [True, False, False, False, False])
        np.testing.assert_array_equal(fired[1], [True, True, True, False, False])

    def test_a_crossing_at_the_lower_boundary_is_a_crossing_at_every_higher_one(self):
        belief = np.linspace(0.0, 1.0, 51)
        fired = events_of(belief, (0.2, 0.5, 0.9))
        assert (fired[0] <= fired[1]).all() and (fired[1] <= fired[2]).all()

    def test_the_event_at_the_flip_boundary_is_the_flip_the_deployed_monitor_reads(self):
        belief, _ = substrate_material()
        np.testing.assert_array_equal(events_of(belief, (FLIP_BOUNDARY,))[0],
                                      flips_of(belief))

    def test_the_bins_are_the_multinomial_counts_the_boundaries_cut(self):
        belief, verdict = belief_material({0: (3, 5, 7), 1: (2, 4, 6)})
        np.testing.assert_array_equal(bin_counts_of(belief, verdict, SUB_FLIP_BANK, 0), [3, 5, 7])
        np.testing.assert_array_equal(bin_counts_of(belief, verdict, SUB_FLIP_BANK, 1), [2, 4, 6])

    def test_a_belief_exactly_on_a_boundary_sits_above_it(self):
        belief = np.array([0.5, 0.9])
        verdict = np.array([0, 0])
        np.testing.assert_array_equal(bin_counts_of(belief, verdict, SUB_FLIP_BANK, 0), [0, 1, 1])


class TestTheBankIsDeclaredAscending:
    def test_boundaries_are_returned_as_floats_in_the_order_given(self):
        assert assert_boundaries([0.5, 0.9]) == (0.5, 0.9)

    def test_a_bank_with_no_boundary_refuses(self):
        with pytest.raises(ValueError, match=re.escape(
                "At least one bank boundary is required.")):
            assert_boundaries(())

    @pytest.mark.parametrize("boundaries", [(0.0, 0.5), (0.5, 1.0), (-0.1,), (1.5,)])
    def test_a_boundary_outside_the_unit_interval_refuses(self, boundaries):
        with pytest.raises(ValueError, match="outside"):
            assert_boundaries(boundaries)

    @pytest.mark.parametrize("boundaries", [(0.9, 0.5), (0.5, 0.5)])
    def test_boundaries_that_do_not_strictly_ascend_refuse(self, boundaries):
        with pytest.raises(ValueError, match=re.escape(
                "are not strictly ascending.")):
            assert_boundaries(boundaries)

    def test_an_unknown_budget_allocation_refuses_naming_the_known_ones(self):
        with pytest.raises(ValueError, match=r"budget allocation 'even'.*\('joint',\)"):
            assert_budget_allocation("even")

    def test_the_wires_are_every_regime_by_every_boundary(self):
        monitor = BeliefBankMonitor(BUDGET, SHIFT, boundaries=SUB_FLIP_BANK)
        assert monitor.wires == [(0, 0.5), (0, 0.9), (1, 0.5), (1, 0.9)]


class TestTheReferenceIsEachWiresOwnEventRate:
    def test_the_reference_is_the_running_sum_of_the_bins_below_the_boundary(self):
        belief, verdict = belief_material({0: (10, 30, 60), 1: (20, 20, 60)})
        calibration = calibrated_bank(belief, verdict).calibration
        assert calibration.reference[0] == {0.5: 0.10, 0.9: 0.40}
        assert calibration.reference[1] == {0.5: 0.20, 0.9: 0.40}

    def test_the_flip_boundarys_reference_is_the_flip_rate_the_deployed_monitor_reads(self):
        belief, verdict = substrate_material()
        calibration = calibrated_bank(belief, verdict).calibration
        for regime in REGIMES:
            in_regime = verdict == regime
            assert calibration.reference[regime][FLIP_BOUNDARY] == pytest.approx(
                float(np.mean(flips_of(belief)[in_regime])))

    def test_a_regime_absent_from_the_material_refuses(self):
        monitor = BeliefBankMonitor(GENEROUS_BUDGET, SHIFT, boundaries=SUB_FLIP_BANK)
        with pytest.raises(ValueError, match=r"^Reference data has no rows in regime 1\.$"):
            monitor.calibrate(np.array([0.1, 0.6, 0.95]), np.zeros(3, dtype=int), 100, 200, 0)

    def test_a_wire_the_material_never_showed_refuses_at_the_log_likelihood_ratio(self):
        belief, verdict = belief_material({0: (0, 50, 50), 1: (5, 45, 50)})
        monitor = BeliefBankMonitor(GENEROUS_BUDGET, SHIFT, boundaries=SUB_FLIP_BANK)
        with pytest.raises(ValueError, match="Reference flip rate"):
            monitor.calibrate(belief, verdict, 100, 200, 0)


class TestP29EachWireAccumulatesInExcessOfItsOwnReference:
    def test_an_excess_sustained_across_items_alarms(self):
        belief, verdict = belief_material({0: (100, 300, 600), 1: (100, 300, 600)})
        monitor = calibrated_bank(belief, verdict, detectable_shift=WIDER_SHIFT,
                                  seed=7)
        drifted, drifted_verdict = stream_of(WATCHED_LENGTH, 11, {0: (0.10, 0.40), 1: (0.30, 0.60)})
        alarm = monitor.observe(drifted, drifted_verdict)
        assert alarm is not None and alarm.regime == 1

    def test_a_burst_returned_to_the_reference_is_undone(self):
        belief, verdict = belief_material({0: (100, 300, 600), 1: (100, 300, 600)})
        monitor = calibrated_bank(belief, verdict, detectable_shift=WIDER_SHIFT,
                                  seed=7)
        quiet, quiet_verdict = stream_of(WATCHED_LENGTH, 5,
                                         {0: (0.10, 0.40), 1: (0.10, 0.40)})
        burst, burst_verdict = stream_of(60, 6, {0: (0.35, 0.65), 1: (0.35, 0.65)})

        assert monitor.observe(burst, burst_verdict) is None
        assert max(value for events in monitor.accumulation.values()
                   for value in events.values()) > 0.0
        assert monitor.observe(quiet, quiet_verdict) is None

        never_saw_the_burst = calibrated_bank(belief, verdict, detectable_shift=WIDER_SHIFT,
                                              seed=7)
        assert never_saw_the_burst.observe(quiet, quiet_verdict) is None
        assert monitor.accumulation == never_saw_the_burst.accumulation

    def test_only_the_items_in_a_regime_move_that_regimes_wire(self):
        event = np.array([True, True, False, False])
        in_regime = np.array([True, True, False, False])
        peak = wire_peak_accumulation(event, in_regime, 0.10, 0.10, regime=0)
        assert peak == pytest.approx(2 * np.log(0.20 / 0.10))
        assert wire_peak_accumulation(event, ~in_regime, 0.10, 0.10, regime=0) == 0.0


class TestP30TheMonitoredFailureIsARateAboveTheReference:
    def test_belief_moving_away_from_the_boundary_never_alarms(self):
        belief, verdict = belief_material({0: (200, 300, 500), 1: (200, 300, 500)})
        monitor = calibrated_bank(belief, verdict, detectable_shift=WIDER_SHIFT,
                                  seed=9)
        calm, calm_verdict = stream_of(WATCHED_LENGTH, 13, {0: (0.02, 0.05), 1: (0.02, 0.05)})
        assert monitor.observe(calm, calm_verdict) is None

    def test_no_event_at_all_leaves_every_wire_on_the_floor(self):
        belief, verdict = belief_material({0: (200, 300, 500), 1: (200, 300, 500)})
        monitor = calibrated_bank(belief, verdict, seed=9)
        confident = np.full(400, ABOVE_THE_BANK)
        assert monitor.observe(confident, np.arange(400) % 2) is None
        assert monitor.accumulation == {regime: {boundary: 0.0 for boundary in SUB_FLIP_BANK}
                                        for regime in REGIMES}


class TestTheAlarmNamesTheWire:
    def _bank_on_a_confident_reference(self, seed=17):
        belief, verdict = belief_material({0: (60, 140, 800), 1: (60, 140, 800)})
        return calibrated_bank(belief, verdict, detectable_shift=WIDER_SHIFT,
                               seed=seed)

    def test_the_alarm_carries_the_regime_the_position_and_the_boundary(self):
        monitor = self._bank_on_a_confident_reference()
        drifted, drifted_verdict = stream_of(WATCHED_LENGTH, 21, {0: (0.30, 0.60), 1: (0.30, 0.60)})
        alarm = monitor.observe(drifted, drifted_verdict)
        assert isinstance(alarm, BankAlarm)
        assert alarm.regime in REGIMES
        assert alarm.boundary in SUB_FLIP_BANK
        assert 0 <= alarm.item_index < len(drifted_verdict)

    def test_the_first_two_fields_unpack_as_the_deployed_alarms_do(self):
        alarm = BankAlarm(regime=1, item_index=7, boundary=0.9)
        assert tuple(alarm)[:2] == (1, 7)

    def test_a_sub_flip_wire_alarms_where_the_flip_wire_does_not(self):
        belief, verdict = belief_material({0: (60, 140, 800), 1: (60, 140, 800)})
        crowding, crowding_verdict = stream_of(WATCHED_LENGTH, 23, {0: (0.03, 0.55), 1: (0.03, 0.55)})

        bank = calibrated_bank(belief, verdict, detectable_shift=WIDER_SHIFT,
                               seed=19)
        flip_only = calibrated_bank(belief, verdict, boundaries=(FLIP_BOUNDARY,),
                                    detectable_shift=WIDER_SHIFT, seed=19)

        bank_alarm = bank.observe(crowding, crowding_verdict)
        assert bank_alarm is not None and bank_alarm.boundary == 0.9
        assert flip_only.observe(crowding, crowding_verdict) is None

    def test_the_alarm_fires_on_the_item_that_reaches_the_wires_threshold(self):
        monitor = self._bank_on_a_confident_reference()
        drifted, drifted_verdict = stream_of(WATCHED_LENGTH, 21, {0: (0.30, 0.60), 1: (0.30, 0.60)})
        alarm = monitor.observe(drifted, drifted_verdict)
        threshold = monitor.calibration.threshold[alarm.regime][alarm.boundary]
        assert monitor.accumulation[alarm.regime][alarm.boundary] >= threshold

        monitor.reset()
        assert monitor.observe(drifted[:alarm.item_index],
                               drifted_verdict[:alarm.item_index]) is None


class TestP31TheThresholdIsReadFromTheReplayedStreams:
    def test_the_combined_realized_rate_stays_within_the_budget(self):
        belief, verdict = belief_material({0: (100, 300, 600), 1: (100, 300, 600)})
        calibration = calibrated_bank(belief, verdict, budget=BUDGET, n_streams=1000,
                                      stream_length=400, seed=5).calibration
        assert calibration.realized_alarm_rate_combined_replayed <= BUDGET + 0.03

    def test_the_realized_rate_is_measured_on_streams_the_threshold_was_not_read_from(self):
        belief, verdict = belief_material({0: (100, 300, 600), 1: (100, 300, 600)})
        calibration = calibrated_bank(belief, verdict, budget=BUDGET, n_streams=1000,
                                      stream_length=400, seed=5).calibration
        assert calibration.realized_alarm_rate_combined_replayed > 0.0
        assert (calibration.realized_alarm_rate_combined_replayed
                != calibration.calibration_level)

    def test_a_larger_budget_buys_a_lower_threshold_on_every_wire(self):
        belief, verdict = belief_material({0: (100, 300, 600), 1: (100, 300, 600)})
        strict = calibrated_bank(belief, verdict, budget=0.10, n_streams=1000, seed=5).calibration
        loose = calibrated_bank(belief, verdict, budget=0.30, n_streams=1000, seed=5).calibration
        for regime, boundary in [(r, b) for r in REGIMES for b in SUB_FLIP_BANK]:
            assert strict.threshold[regime][boundary] > loose.threshold[regime][boundary]

    def test_replayed_streams_that_reach_no_accumulation_refuse_a_threshold(self):
        belief, verdict = belief_material({0: (2, 100, 400), 1: (2, 100, 400)})
        monitor = BeliefBankMonitor(GENEROUS_BUDGET, SHIFT, boundaries=SUB_FLIP_BANK)
        with pytest.raises(ValueError, match="violating-centre fraction"):
            monitor.calibrate(belief, verdict, 1, 200, 5)

    def test_a_budget_leaving_no_stream_in_the_tail_refuses_under_joint(self):
        belief, verdict = belief_material({0: (100, 300, 600), 1: (100, 300, 600)})
        monitor = BeliefBankMonitor(0.001, SHIFT, boundaries=SUB_FLIP_BANK)
        with pytest.raises(ValueError, match="leaves no stream in the tail"):
            monitor.calibrate(belief, verdict, 100, 200, 5)

    def test_a_joint_level_too_thin_to_read_a_threshold_from_refuses(self):
        belief, verdict = belief_material({0: (100, 300, 600), 1: (100, 300, 600)})
        monitor = BeliefBankMonitor(0.015, SHIFT, boundaries=SUB_FLIP_BANK)
        with pytest.raises(ValueError, match=re.escape(
                "is below the required minimum 10 per wire.")):
            monitor.calibrate(belief, verdict, 100, 500, 5)


class TestTheBudgetReachesTheWires:
    def _material(self):
        return belief_material({0: (100, 300, 600), 1: (100, 300, 600)})

    def test_joint_puts_every_wire_at_one_common_quantile_level(self):
        calibration = calibrated_bank(*self._material(), budget=BUDGET, n_streams=1000,
                                      stream_length=400, seed=5).calibration
        counts = {calibration.streams_in_the_tail[regime][boundary]
                  for regime in REGIMES for boundary in SUB_FLIP_BANK}
        assert len(counts) == 1
        assert calibration.wire_quantile_level == counts.pop() / 1000

    def test_the_joint_level_is_the_largest_that_holds_the_union_inside_the_budget(self):
        calibration = calibrated_bank(*self._material(), budget=BUDGET, n_streams=1000,
                                      stream_length=400, seed=5).calibration
        assert calibration.calibration_level <= BUDGET


class TestTheReferencesEstimationErrorIsTreatedPerWire:
    def test_the_posterior_is_the_dirichlet_the_bin_counts_name(self):
        belief, verdict = belief_material({0: (10, 30, 60), 1: (20, 20, 60)})
        monitor = BeliefBankMonitor(GENEROUS_BUDGET, SHIFT, boundaries=SUB_FLIP_BANK)
        counts = {regime: bin_counts_of(belief, verdict, SUB_FLIP_BANK, regime)
                  for regime in REGIMES}
        posterior = monitor._dirichlet_posterior(counts)
        np.testing.assert_array_equal(posterior[0], [10.5, 30.5, 60.5])
        np.testing.assert_array_equal(posterior[1], [20.5, 20.5, 60.5])

    def test_one_boundarys_dirichlet_is_the_jeffreys_beta_the_deployed_monitor_names(self):
        belief, verdict = belief_material({0: (34, 1610), 1: (65, 291)}, boundaries=(0.5,))
        monitor = BeliefBankMonitor(GENEROUS_BUDGET, SHIFT, boundaries=(0.5,))
        counts = {regime: bin_counts_of(belief, verdict, (0.5,), regime) for regime in REGIMES}
        posterior = monitor._dirichlet_posterior(counts)
        assert monitor._event_posterior(posterior, 0, 0) == (34.5, 1610.5)
        assert monitor._event_posterior(posterior, 1, 0) == (65.5, 291.5)

    def test_an_events_marginal_posterior_splits_the_dirichlet_at_its_own_boundary(self):
        belief, verdict = belief_material({0: (10, 30, 60), 1: (20, 20, 60)})
        monitor = BeliefBankMonitor(GENEROUS_BUDGET, SHIFT, boundaries=SUB_FLIP_BANK)
        counts = {regime: bin_counts_of(belief, verdict, SUB_FLIP_BANK, regime)
                  for regime in REGIMES}
        posterior = monitor._dirichlet_posterior(counts)
        assert monitor._event_posterior(posterior, 0, 1) == (41.0, 60.5)



    def test_the_at_reference_reading_is_recorded_beside_the_replayed_one(self):
        belief, verdict = belief_material({0: (40, 120, 240), 1: (40, 120, 240)})
        calibration = calibrated_bank(belief, verdict, n_streams=400, seed=5).calibration
        assert calibration.realized_alarm_rate_combined_at_reference is not None
        assert (calibration.realized_alarm_rate_combined_at_reference
                != calibration.realized_alarm_rate_combined_replayed)




class TestTheNoiseBoundaryResolvesPerWire:
    def _material(self):
        return belief_material({0: (60, 200, 1384), 1: (40, 100, 216)})

    def test_each_wires_break_even_sits_on_its_own_posterior_quantile(self):
        calibration = calibrated_bank(*self._material(), detectable_shift=NOISE_BOUNDARY,
                                      stream_length=4000, seed=5).calibration
        for regime in REGIMES:
            for boundary in SUB_FLIP_BANK:
                assert calibration.break_even[regime][boundary] == pytest.approx(
                    calibration.reference_posterior_quantile[regime][boundary], abs=1e-12)

    def test_the_quantile_is_the_exact_beta_quantile_of_the_wires_marginal(self):
        belief, verdict = self._material()
        calibration = calibrated_bank(belief, verdict, detectable_shift=NOISE_BOUNDARY,
                                      stream_length=4000, seed=5).calibration
        counts = bin_counts_of(belief, verdict, SUB_FLIP_BANK, 0)
        marginal = (counts[0] + 0.5, counts[1] + counts[2] + 1.0)
        assert calibration.reference_posterior_quantile[0][0.5] == pytest.approx(
            float(stats.beta(*marginal).ppf(0.977)), abs=1e-15)




    def test_the_recorded_break_even_is_the_one_the_deployed_shift_implies(self):
        calibration = calibrated_bank(*self._material(), detectable_shift=WIDER_SHIFT,
                                      seed=5).calibration
        for regime in REGIMES:
            for boundary in SUB_FLIP_BANK:
                assert calibration.break_even[regime][boundary] == break_even_rate(
                    calibration.reference[regime][boundary],
                    calibration.detectable_shift[regime][boundary], regime)


class TestTheShiftIsRaisedUntilEveryWiresThresholdIsReachable:
    def _material(self):
        return belief_material({0: (60, 200, 1384), 1: (40, 100, 216)})

    def test_a_short_horizon_raises_the_deployed_shift_above_the_resolved_one(self):
        calibration = calibrated_bank(*self._material(), detectable_shift=NOISE_BOUNDARY,
                                      budget=GENEROUS_BUDGET, stream_length=200, n_streams=200,
                                      seed=5).calibration
        raised = [(regime, boundary) for regime in REGIMES for boundary in SUB_FLIP_BANK
                  if calibration.detectable_shift[regime][boundary]
                  > calibration.resolved_detectable_shift[regime][boundary]]
        assert raised, "this horizon does not exercise the proviso"
        for regime, boundary in raised:
            assert (calibration.break_even[regime][boundary]
                    > calibration.reference_posterior_quantile[regime][boundary])

    def test_a_horizon_no_shift_can_alarm_within_refuses_naming_the_wire(self):
        monitor = BeliefBankMonitor(GENEROUS_BUDGET, NOISE_BOUNDARY, boundaries=SUB_FLIP_BANK)
        with pytest.raises(ValueError, match=(
                r"^Regime \d, boundary 0\.\d+: estimated positions to alarm exceed "
                r"the 5-position horizon at the shift ceiling\.$")):
            monitor.calibrate(*self._material(), stream_length=5, n_streams=200, seed=5)

    def test_a_long_enough_horizon_leaves_the_proviso_inert(self):
        calibration = calibrated_bank(*self._material(), detectable_shift=NOISE_BOUNDARY,
                                      budget=GENEROUS_BUDGET, stream_length=8000, n_streams=200,
                                      seed=5).calibration
        assert calibration.detectable_shift == calibration.resolved_detectable_shift


class TestTheCalibrationSaysWhatRan:
    def test_the_record_carries_the_bank_and_how_the_budget_reached_it(self):
        calibration = calibrated_bank(*belief_material({0: (100, 300, 600), 1: (100, 300, 600)}),
                                      seed=5).calibration
        assert calibration.boundaries == SUB_FLIP_BANK
        assert calibration.budget_allocation == JOINT
        assert calibration.replay_stream_length == WATCHED_LENGTH
        assert calibration.n_streams == REPLAY_STREAMS
        assert calibration.seed == 5

    def test_the_caller_supplied_context_is_recorded_and_never_read_into_the_arithmetic(self):
        belief, verdict = belief_material({0: (100, 300, 600), 1: (100, 300, 600)})
        monitor = BeliefBankMonitor(GENEROUS_BUDGET, SHIFT, boundaries=SUB_FLIP_BANK)
        annotated = monitor.calibrate(belief, verdict, 300, 200, 5, prefix_drift_checked=True,
                                      monitored_length=19000,
                                      fitted_family_overlap={"all": 0.2})
        bare = BeliefBankMonitor(GENEROUS_BUDGET, SHIFT,
                                 boundaries=SUB_FLIP_BANK).calibrate(belief, verdict, 300, 200, 5)
        assert annotated.prefix_drift_checked is True
        assert annotated.monitored_length == 19000
        assert annotated.fitted_family_overlap == {"all": 0.2}
        assert annotated.threshold == bare.threshold

    def test_the_calibration_is_frozen(self):
        calibration = calibrated_bank(*belief_material({0: (100, 300, 600), 1: (100, 300, 600)}),
                                      seed=5).calibration
        with pytest.raises(Exception):
            calibration.threshold = {}

    def test_a_record_constructs_from_its_required_fields_alone(self):
        assert BeliefBankCalibration(
            reference={}, threshold={}, realized_alarm_rate_replayed={},
            realized_alarm_rate_combined_replayed=0.0, replay_stream_length=1,
            n_streams=1, seed=0).quiet_horizon_confidence is None


class TestTheMonitorsStateAcrossCalls:
    def _monitor(self):
        belief, verdict = belief_material({0: (60, 140, 800), 1: (60, 140, 800)})
        return calibrated_bank(belief, verdict, detectable_shift=WIDER_SHIFT,
                               seed=19)

    def test_the_accumulation_persists_across_calls(self):
        monitor = self._monitor()
        drifted, drifted_verdict = stream_of(WATCHED_LENGTH, 21, {0: (0.30, 0.60), 1: (0.30, 0.60)})
        whole = monitor.observe(drifted, drifted_verdict)

        monitor.reset()
        halfway = len(drifted) // 2
        piecewise = monitor.observe(drifted[:halfway], drifted_verdict[:halfway])
        if piecewise is None:
            piecewise = monitor.observe(drifted[halfway:], drifted_verdict[halfway:])
            piecewise = piecewise._replace(item_index=piecewise.item_index + halfway)
        assert piecewise == whole

    def test_items_after_the_alarm_are_not_consumed(self):
        monitor = self._monitor()
        drifted, drifted_verdict = stream_of(WATCHED_LENGTH, 21, {0: (0.30, 0.60), 1: (0.30, 0.60)})
        alarm = monitor.observe(drifted, drifted_verdict)
        assert monitor.items_observed == alarm.item_index + 1

    def test_reset_returns_every_wire_to_the_floor_and_keeps_the_calibration(self):
        monitor = self._monitor()
        drifted, drifted_verdict = stream_of(WATCHED_LENGTH, 21, {0: (0.30, 0.60), 1: (0.30, 0.60)})
        calibration = monitor.calibration
        monitor.observe(drifted, drifted_verdict)
        monitor.reset()
        assert monitor.accumulation == {regime: {boundary: 0.0 for boundary in SUB_FLIP_BANK}
                                        for regime in REGIMES}
        assert monitor.items_observed == 0
        assert monitor.calibration == calibration

    def test_the_same_seed_gives_the_same_calibration(self):
        belief, verdict = belief_material({0: (60, 140, 800), 1: (60, 140, 800)})
        first = calibrated_bank(belief, verdict, seed=19).calibration
        again = calibrated_bank(belief, verdict, seed=19).calibration
        assert first == again

    def test_a_different_seed_moves_the_thresholds(self):
        belief, verdict = belief_material({0: (60, 140, 800), 1: (60, 140, 800)})
        first = calibrated_bank(belief, verdict, seed=19).calibration
        moved = calibrated_bank(belief, verdict, seed=20).calibration
        assert first.threshold != moved.threshold


class TestWhatTheMonitorReadsIsConditionedOnce:
    def test_a_belief_column_is_returned_as_float64(self):
        conditioned = as_belief([0.25, 0.75])
        assert conditioned.dtype == np.float64

    def test_a_missing_belief_is_refused_as_the_unknown_it_is(self):
        with pytest.raises(ValueError, match=re.escape(
                "Probability of agreement contains NaN.")):
            as_belief(np.array([0.5, np.nan]))

    @pytest.mark.parametrize("outside", [-0.01, 1.01])
    def test_a_value_outside_the_unit_interval_is_refused_as_not_a_probability(self, outside):
        with pytest.raises(ValueError, match=re.escape(
                "Probability of agreement outside [0, 1].")):
            as_belief(np.array([0.5, outside]))

    def test_a_belief_column_is_one_value_per_item(self):
        with pytest.raises(ValueError, match="one value per item"):
            as_belief(np.zeros((3, 2)))

    def test_a_verdict_outside_the_regimes_is_refused(self):
        monitor = calibrated_bank(*belief_material({0: (60, 140, 800), 1: (60, 140, 800)}),
                                  seed=19)
        with pytest.raises(ValueError, match=(
                r"^monitor routing: verdict contains values outside regimes \(0, 1\)\.")):
            monitor.observe(np.array([0.1, 0.2]), np.array([0, 2]))

    def test_mismatched_traffic_lengths_name_the_traffic(self):
        monitor = calibrated_bank(*belief_material({0: (60, 140, 800), 1: (60, 140, 800)}),
                                  seed=19)
        with pytest.raises(ValueError, match=(
                r"^the traffic carries 2 beliefs against 3 verdicts; lengths must match\.$")):
            monitor.observe(np.array([0.1, 0.2]), np.array([0, 1, 1]))

    def test_mismatched_material_lengths_name_the_material(self):
        monitor = BeliefBankMonitor(GENEROUS_BUDGET, SHIFT, boundaries=SUB_FLIP_BANK)
        with pytest.raises(ValueError, match=r"^the drift-free material carries 2 beliefs"):
            monitor.calibrate(np.array([0.1, 0.2]), np.array([0, 1, 1]), 100, 200, 5)


class TestRefusalsBeforeAnyTrafficIsJudged:
    def test_observing_before_calibration_is_refused(self):
        monitor = BeliefBankMonitor(BUDGET, SHIFT, boundaries=SUB_FLIP_BANK)
        with pytest.raises(ValueError, match=(
                r"^Monitor is not calibrated; call calibrate before observe\.$")):
            monitor.observe(np.array([0.1]), np.array([0]))

    @pytest.mark.parametrize("level", [0.0, 1.0, 1.5])
    def test_a_calibration_level_outside_the_unit_interval_is_refused(self, level):
        with pytest.raises(ValueError, match="calibration level"):
            BeliefBankMonitor(level, SHIFT, boundaries=SUB_FLIP_BANK)

    def test_a_malformed_shift_specification_is_refused_at_construction(self):
        with pytest.raises(ValueError, match="Unknown detectable_shift keys"):
            BeliefBankMonitor(BUDGET, {"factor": 0.5}, boundaries=SUB_FLIP_BANK)

    def test_a_retired_shift_form_refuses_naming_its_successor(self):
        for retired, successor in ((0.02, "must be a mapping"),
                                   ({"mode": "relative", "factor": 0.5}, "no longer accepts"),
                                   ({"baseline_credibility": 0.99}, "indifference_zone_quantile")):
            with pytest.raises(ValueError, match=successor):
                BeliefBankMonitor(BUDGET, retired, boundaries=SUB_FLIP_BANK)

    def test_a_stream_with_no_items_has_no_behaviour_to_read_a_threshold_from(self):
        monitor = BeliefBankMonitor(GENEROUS_BUDGET, SHIFT, boundaries=SUB_FLIP_BANK)
        with pytest.raises(ValueError, match=(
                r"^Replay stream_length must be at least 1; got 0\.$")):
            monitor.calibrate(*belief_material({0: (10, 30, 60), 1: (10, 30, 60)}), 0, 200, 5)

    def test_a_re_derived_reference_the_shift_carries_past_certainty_refuses_at_observe(self):
        from dataclasses import replace

        monitor = calibrated_bank(*belief_material({0: (60, 140, 800), 1: (60, 140, 800)}),
                                  seed=19)
        broken = dict(monitor.calibration.reference)
        broken[1] = {**broken[1], 0.5: 1.0}
        monitor.calibration = replace(monitor.calibration, reference=broken)
        with pytest.raises(ValueError, match=r"^Reference flip rate for regime 1 .* got 1\.0\.$"):
            monitor.observe(np.array([0.1, 0.2]), np.array([0, 1]))


def raw_bytes_of(value):
    if isinstance(value, dict):
        return {key: raw_bytes_of(inner) for key, inner in sorted(value.items())}
    if isinstance(value, (int, np.integer)) and not isinstance(value, bool):
        return int(value)
    return np.float64(value).tobytes().hex()


class TestTheWiresIncrementIsTheDeployedOne:
    def test_the_increment_one_item_contributes_is_the_deployed_one(self):
        assert wire_peak_accumulation(np.array([True]), np.array([True]), 0.10, 0.05,
                                      regime=0) == pytest.approx(
            log_likelihood_ratio_increments(0.10, 0.05, 0)[0])


class TestTheBoundaryCasesTheGateFound:
    def test_a_narrow_belief_column_is_widened_before_it_is_read(self):
        assert as_belief(np.array([0.25, 0.75], dtype=np.float32)).dtype == np.float64
        assert as_belief(np.array([0, 1], dtype=np.int64)).dtype == np.float64

    def test_the_one_value_per_item_refusal_reads_in_full(self):
        with pytest.raises(ValueError, match=(
                r"^Probability of agreement must be one-dimensional: one value per item\.$")):
            as_belief(np.zeros((3, 2)))

    def test_the_bins_carry_one_count_per_bin_even_where_a_bin_is_empty(self):
        belief, verdict = belief_material({0: (5, 5, 0), 1: (5, 5, 5)})
        counts = bin_counts_of(belief, verdict, SUB_FLIP_BANK, 0)
        assert len(counts) == len(SUB_FLIP_BANK) + 1
        np.testing.assert_array_equal(counts, [5, 5, 0])

    def test_a_wires_accumulation_refuses_naming_the_regime_it_is_about(self):
        with pytest.raises(ValueError, match=r"^Reference flip rate for regime 1 .* got 0\.0\.$"):
            wire_peak_accumulation(np.array([True]), np.array([True]), 0.0, 0.05, regime=1)


    def test_a_belief_sitting_exactly_on_a_boundary_does_not_fire_that_wire(self):
        belief, verdict = belief_material({0: (60, 140, 800), 1: (60, 140, 800)})
        monitor = calibrated_bank(belief, verdict, detectable_shift=WIDER_SHIFT, seed=19)
        on_the_boundary = np.full(800, max(SUB_FLIP_BANK))
        assert monitor.observe(on_the_boundary, np.arange(800) % 2) is None
        assert monitor.accumulation == {regime: {boundary: 0.0 for boundary in SUB_FLIP_BANK}
                                        for regime in REGIMES}

    def test_an_accumulation_landing_exactly_on_a_wires_threshold_alarms(self):
        from dataclasses import replace

        belief, verdict = belief_material({0: (60, 140, 800), 1: (60, 140, 800)})
        monitor = calibrated_bank(belief, verdict, detectable_shift=WIDER_SHIFT, seed=19)
        reference = monitor.calibration.reference
        shift = monitor.calibration.detectable_shift
        on_event, _ = log_likelihood_ratio_increments(reference[0][0.5], shift[0][0.5], 0)
        monitor.calibration = replace(
            monitor.calibration,
            threshold={regime: {boundary: (on_event if (regime, boundary) == (0, 0.5) else 1e9)
                                for boundary in SUB_FLIP_BANK} for regime in REGIMES})
        alarm = monitor.observe(np.array([0.25]), np.array([0]))
        assert alarm == BankAlarm(regime=0, item_index=0, boundary=0.5)

    def test_a_replayed_peak_equal_to_the_threshold_counts_as_an_alarm(self):
        peaks = {0: {0.5: np.array([0.0, 2.0]), 0.9: np.array([0.0, 3.0])},
                 1: {0.5: np.array([0.0, 1.0]), 0.9: np.array([0.0, 4.0])}}
        threshold = {0: {0.5: 2.0, 0.9: 9.0}, 1: {0.5: 9.0, 0.9: 4.0}}
        monitor = BeliefBankMonitor(GENEROUS_BUDGET, SHIFT, boundaries=SUB_FLIP_BANK)
        realized, combined = monitor._alarm_rate_of(peaks, threshold)
        assert realized[0][0.5] == 0.5 and realized[1][0.9] == 0.5
        assert realized[0][0.9] == 0.0 and realized[1][0.5] == 0.0
        assert combined == 0.5
        assert monitor._union_rate_at(peaks, {regime: {boundary: 1 for boundary in SUB_FLIP_BANK}
                                              for regime in REGIMES}) == 0.5

    def test_a_uniform_draw_landing_on_the_drawn_rate_is_not_an_event(self):
        class DictatedReplay:
            def __init__(self, rate, uniform):
                self._rate, self._uniform = rate, uniform

            def integers(self, low, high, size):
                return np.arange(size) % high

            def beta(self, *_parameters):
                return self._rate

            def random(self, size):
                return np.full(size, self._uniform)

        belief, verdict = belief_material({0: (10, 10, 20), 1: (10, 10, 20)})
        monitor = BeliefBankMonitor(GENEROUS_BUDGET, SHIFT, boundaries=(0.5,))
        counts = {regime: bin_counts_of(belief, verdict, (0.5,), regime) for regime in REGIMES}
        reference = {regime: {0.5: 0.25} for regime in REGIMES}
        shift = {regime: {0.5: 0.10} for regime in REGIMES}
        shared = (events_of(belief, (0.5,)), verdict, reference, shift, 8, 1)

        on_the_rate = monitor._replayed_peaks(*shared, DictatedReplay(0.3, 0.3),
                                              monitor._dirichlet_posterior(counts))
        below_the_rate = monitor._replayed_peaks(*shared, DictatedReplay(0.3, 0.29),
                                                 monitor._dirichlet_posterior(counts))
        assert all(peak == 0.0 for wire in on_the_rate.values() for peak in wire[0.5])
        assert any(peak > 0.0 for wire in below_the_rate.values() for peak in wire[0.5])

    def test_a_budget_leaving_exactly_one_stream_in_the_tail_refuses_at_the_joint_minimum(self):
        belief, verdict = belief_material({0: (100, 300, 600), 1: (100, 300, 600)})
        monitor = BeliefBankMonitor(0.002, SHIFT, boundaries=SUB_FLIP_BANK)
        with pytest.raises(ValueError, match=re.escape(
                "At tail count 1, violating-centre fraction")):
            monitor.calibrate(belief, verdict, 100, 500, 5)

    def test_the_refusal_at_one_stream_per_wire_reads_in_full(self):
        belief, verdict = belief_material({0: (100, 300, 600), 1: (100, 300, 600)})
        monitor = BeliefBankMonitor(0.002, SHIFT, boundaries=SUB_FLIP_BANK)
        with pytest.raises(ValueError, match=re.escape(
                "calibration level 0.002 (2 boundaries, 500 streams).")):
            monitor.calibrate(belief, verdict, 100, 500, 5)

    def test_the_joint_level_is_the_LARGEST_that_holds_the_union_inside_the_budget(self):
        belief, verdict = belief_material({0: (100, 300, 600), 1: (100, 300, 600)})
        monitor = calibrated_bank(belief, verdict, budget=BUDGET, n_streams=1000,
                                  stream_length=400, seed=5)
        calibration = monitor.calibration
        resolved = calibration.streams_in_the_tail[0][0.5]
        assert calibration.calibration_level <= BUDGET

        counts = {regime: bin_counts_of(belief, verdict, SUB_FLIP_BANK, regime)
                  for regime in REGIMES}
        peaks = monitor._replayed_peaks(
            events_of(belief, SUB_FLIP_BANK), verdict, calibration.reference,
            calibration.detectable_shift, 400, 1000, np.random.default_rng(5), "point",
            monitor._dirichlet_posterior(counts))
        one_more = {regime: {boundary: resolved + 1 for boundary in SUB_FLIP_BANK}
                    for regime in REGIMES}
        assert monitor._union_rate_at(peaks, one_more) > BUDGET

    def test_the_noise_boundarys_search_is_reproducible_where_the_proviso_binds(self):
        belief, verdict = belief_material({0: (60, 200, 1384), 1: (40, 100, 216)})
        shared = dict(detectable_shift=NOISE_BOUNDARY, budget=GENEROUS_BUDGET,
                      stream_length=200, n_streams=200, seed=5)
        first = calibrated_bank(belief, verdict, **shared).calibration
        again = calibrated_bank(belief, verdict, **shared).calibration
        assert first.detectable_shift != first.resolved_detectable_shift, (
            "this horizon must exercise the proviso for the test to mean anything")
        assert first == again
