import numpy as np
import pytest

from rcv.belief_bank import violating_centre_fraction


def peaks_of(by_wire):
    return {regime: {boundary: np.asarray(values, dtype=np.float64)
                     for boundary, values in boundaries.items()}
            for regime, boundaries in by_wire.items()}


class TestTheFractionIsOverCentresRatherThanStreams:
    def test_one_alarming_stream_makes_its_whole_centre_violate(self):
        peaks = peaks_of({0: {0.5: [9.0, 1.0, 1.0, 1.0, 1.0, 1.0]},
                          1: {0.5: [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]}})
        threshold = {0: {0.5: 5.0}, 1: {0.5: 5.0}}

        assert violating_centre_fraction(peaks, threshold, 3, 1) == 0.5

    def test_a_centre_needing_two_alarms_is_not_violated_by_one(self):
        peaks = peaks_of({0: {0.5: [9.0, 1.0, 1.0, 1.0, 1.0, 1.0]},
                          1: {0.5: [1.0] * 6}})
        threshold = {0: {0.5: 5.0}, 1: {0.5: 5.0}}

        assert violating_centre_fraction(peaks, threshold, 3, 2) == 0.0

    def test_two_alarms_in_one_centre_violate_it_at_the_stricter_minimum(self):
        peaks = peaks_of({0: {0.5: [9.0, 9.0, 1.0, 1.0, 1.0, 1.0]},
                          1: {0.5: [1.0] * 6}})
        threshold = {0: {0.5: 5.0}, 1: {0.5: 5.0}}

        assert violating_centre_fraction(peaks, threshold, 3, 2) == 0.5


class TestTheReductionToThePooledRate:
    @pytest.mark.parametrize("seed", [0, 1, 2])
    def test_one_stream_per_centre_is_the_pooled_union_alarm_rate(self, seed):
        draw = np.random.default_rng(seed)
        peaks = peaks_of({regime: {boundary: draw.random(40) * 10
                                   for boundary in (0.5, 0.9)}
                          for regime in (0, 1)})
        threshold = {regime: {boundary: 5.0 for boundary in (0.5, 0.9)} for regime in (0, 1)}

        pooled = float(np.mean(np.logical_or.reduce(
            [peaks[regime][boundary] >= threshold[regime][boundary]
             for regime in (0, 1) for boundary in (0.5, 0.9)])))

        assert violating_centre_fraction(peaks, threshold, 1, 1) == pooled


class TestAStreamCrossingSeveralWiresCountsOnce:
    def test_the_union_is_taken_before_the_centre_is_counted(self):
        peaks = peaks_of({0: {0.5: [9.0, 1.0], 0.9: [9.0, 1.0]},
                          1: {0.5: [1.0, 1.0], 0.9: [1.0, 1.0]}})
        threshold = {0: {0.5: 5.0, 0.9: 5.0}, 1: {0.5: 5.0, 0.9: 5.0}}

        assert violating_centre_fraction(peaks, threshold, 2, 2) == 0.0
        assert violating_centre_fraction(peaks, threshold, 2, 1) == 1.0


class TestTheEdgesReadInFull:
    def test_a_threshold_no_peak_reaches_leaves_every_centre_clean(self):
        peaks = peaks_of({0: {0.5: [1.0] * 6}, 1: {0.5: [1.0] * 6}})
        assert violating_centre_fraction(peaks, {0: {0.5: 99.0}, 1: {0.5: 99.0}}, 3, 1) == 0.0

    def test_a_threshold_every_peak_reaches_violates_every_centre(self):
        peaks = peaks_of({0: {0.5: [1.0] * 6}, 1: {0.5: [1.0] * 6}})
        assert violating_centre_fraction(peaks, {0: {0.5: 0.5}, 1: {0.5: 0.5}}, 3, 1) == 1.0

    def test_streams_that_do_not_divide_into_whole_centres_refuse(self):
        peaks = peaks_of({0: {0.5: [1.0] * 7}, 1: {0.5: [1.0] * 7}})
        with pytest.raises(ValueError, match="whole centres"):
            violating_centre_fraction(peaks, {0: {0.5: 5.0}, 1: {0.5: 5.0}}, 3, 1)

    def test_a_minimum_above_the_centre_size_refuses(self):
        peaks = peaks_of({0: {0.5: [1.0] * 6}, 1: {0.5: [1.0] * 6}})
        with pytest.raises(ValueError, match="minimum_alarms 4 exceeds centre_size 3"):
            violating_centre_fraction(peaks, {0: {0.5: 5.0}, 1: {0.5: 5.0}}, 3, 4)
