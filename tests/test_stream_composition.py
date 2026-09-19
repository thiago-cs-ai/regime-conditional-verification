import re

import numpy as np
import pytest

from rcv.stream_composition import ComposedStream, compose_stream, linear_contamination_ramp


def pools(n_base=40, n_drift=15):
    return (np.array([f"base{index:03d}" for index in range(n_base)]),
            np.array([f"drift{index:03d}" for index in range(n_drift)]))


class TestTheLinearRamp:
    def test_the_prefix_before_the_onset_carries_no_contamination(self):
        schedule = linear_contamination_ramp(length=100, onset=20, maximum_rate=0.5)
        assert not schedule[:20].any()

    def test_the_ramp_starts_at_zero_and_reaches_the_maximum_at_the_final_position(self):
        schedule = linear_contamination_ramp(length=100, onset=20, maximum_rate=0.5)
        assert schedule[20] == 0.0
        assert schedule[-1] == pytest.approx(0.5)

    def test_the_rise_is_linear(self):
        schedule = linear_contamination_ramp(length=100, onset=20, maximum_rate=0.5)
        increments = np.diff(schedule[20:])
        assert increments.min() == pytest.approx(increments.max())
        assert increments.min() > 0.0

    def test_the_schedule_sums_to_half_the_maximum_over_the_monitored_positions(self):
        schedule = linear_contamination_ramp(length=20000, onset=1000, maximum_rate=0.3)
        assert schedule.sum() == pytest.approx(0.3 * (20000 - 1000) / 2)

    def test_a_zero_maximum_rate_is_the_null_schedule(self):
        schedule = linear_contamination_ramp(length=100, onset=20, maximum_rate=0.0)
        assert not schedule.any()
        assert len(schedule) == 100

    def test_the_schedule_is_float64(self):
        assert linear_contamination_ramp(length=100, onset=20,
                                         maximum_rate=0.5).dtype == np.float64

    def test_a_maximum_rate_outside_the_unit_interval_refuses(self):
        with pytest.raises(ValueError, match="maximum_rate"):
            linear_contamination_ramp(length=100, onset=20, maximum_rate=1.0)
        with pytest.raises(ValueError, match="maximum_rate"):
            linear_contamination_ramp(length=100, onset=20, maximum_rate=-0.1)

    def test_an_onset_leaving_no_ramp_refuses(self):
        with pytest.raises(ValueError, match="onset"):
            linear_contamination_ramp(length=100, onset=99, maximum_rate=0.5)
        with pytest.raises(ValueError, match="onset"):
            linear_contamination_ramp(length=100, onset=0, maximum_rate=0.5)

    def test_the_smallest_prefix_the_refusal_admits_is_one_position(self):
        schedule = linear_contamination_ramp(length=100, onset=1, maximum_rate=0.5)
        assert schedule[0] == 0.0
        assert schedule[1] == 0.0
        assert schedule[-1] == pytest.approx(0.5)

    def test_the_shortest_ramp_the_refusal_admits_is_two_positions(self):
        schedule = linear_contamination_ramp(length=100, onset=98, maximum_rate=0.5)
        assert schedule[97] == 0.0
        assert schedule[98] == 0.0
        assert schedule[99] == pytest.approx(0.5)


class TestTheComposedStream:
    def test_the_stream_is_as_long_as_its_schedule(self):
        base, drift = pools()
        schedule = linear_contamination_ramp(length=500, onset=100, maximum_rate=0.4)
        stream = compose_stream(schedule, base, drift, seed=7)
        assert len(stream.item_id) == 500
        assert len(stream.is_drift_item) == 500
        assert len(stream.stream_lambda) == 500

    def test_the_stream_carries_the_schedule_that_drew_it(self):
        base, drift = pools()
        schedule = linear_contamination_ramp(length=500, onset=100, maximum_rate=0.4)
        stream = compose_stream(schedule, base, drift, seed=7)
        np.testing.assert_array_equal(stream.stream_lambda, schedule)

    def test_a_contaminated_position_takes_a_drift_item_and_a_clean_one_a_base_item(self):
        base, drift = pools()
        schedule = linear_contamination_ramp(length=500, onset=100, maximum_rate=0.4)
        stream = compose_stream(schedule, base, drift, seed=7)
        assert set(stream.item_id[stream.is_drift_item]) <= set(drift.tolist())
        assert set(stream.item_id[~stream.is_drift_item]) <= set(base.tolist())

    def test_no_position_is_contaminated_where_the_schedule_is_zero(self):
        base, drift = pools()
        schedule = linear_contamination_ramp(length=500, onset=100, maximum_rate=0.4)
        stream = compose_stream(schedule, base, drift, seed=7)
        assert not stream.is_drift_item[schedule == 0.0].any()

    def test_the_flag_is_boolean_and_the_schedule_float64(self):
        base, drift = pools()
        stream = compose_stream(linear_contamination_ramp(500, 100, 0.4), base, drift, seed=7)
        assert stream.is_drift_item.dtype == np.bool_
        assert stream.stream_lambda.dtype == np.float64

    def test_the_draw_is_with_replacement(self):
        base, drift = pools(n_base=10, n_drift=5)
        stream = compose_stream(linear_contamination_ramp(500, 100, 0.4), base, drift, seed=7)
        assert len(np.unique(stream.item_id)) < 500

    def test_the_realized_contamination_follows_the_schedule(self):
        base, drift = pools()
        schedule = linear_contamination_ramp(length=20000, onset=1000, maximum_rate=0.3)
        stream = compose_stream(schedule, base, drift, seed=7)
        expected = schedule.sum()
        deviation = np.sqrt((schedule * (1 - schedule)).sum())
        assert abs(int(stream.is_drift_item.sum()) - expected) < 5 * deviation

    def test_every_id_in_each_pool_reaches_the_stream(self):
        base, drift = pools()
        stream = compose_stream(linear_contamination_ramp(2000, 200, 0.30), base, drift, seed=7)
        assert set(np.unique(stream.item_id[~stream.is_drift_item])) == set(base.tolist())
        assert set(np.unique(stream.item_id[stream.is_drift_item])) == set(drift.tolist())

    def test_a_scheduled_rate_the_coin_lands_exactly_on_is_not_contamination(self):
        base, drift = pools()
        coin = np.random.default_rng(11).random(300)
        stream = compose_stream(coin, base, drift, seed=11)
        assert not stream.is_drift_item.any()

    def test_a_schedule_given_as_whole_numbers_is_still_a_float64_column(self):
        base, drift = pools()
        assert compose_stream([0, 0, 0, 0, 0], base, drift,
                              seed=1).stream_lambda.dtype == np.float64

    def test_the_contamination_rises_across_the_ramp(self):
        base, drift = pools()
        schedule = linear_contamination_ramp(length=20000, onset=1000, maximum_rate=0.3)
        contaminated = compose_stream(schedule, base, drift, seed=7).is_drift_item[1000:]
        first_half, second_half = np.array_split(contaminated, 2)
        assert second_half.mean() > 2 * first_half.mean()


class TestTheDrawIsSeeded:
    def test_the_same_seed_composes_the_same_stream(self):
        base, drift = pools()
        schedule = linear_contamination_ramp(length=500, onset=100, maximum_rate=0.4)
        first = compose_stream(schedule, base, drift, seed=7)
        second = compose_stream(schedule, base, drift, seed=7)
        np.testing.assert_array_equal(first.item_id, second.item_id)
        np.testing.assert_array_equal(first.is_drift_item, second.is_drift_item)

    def test_a_re_composition_reproduces_every_field_of_the_stream(self):
        base, drift = pools()
        schedule = linear_contamination_ramp(length=500, onset=100, maximum_rate=0.4)
        composed = compose_stream(schedule, base, drift, seed=7)
        recomposed = compose_stream(schedule, base, drift, seed=7)
        for field in composed._fields:
            np.testing.assert_array_equal(getattr(recomposed, field), getattr(composed, field),
                                          err_msg=field)

    def test_the_pools_order_is_the_declaration_and_a_reordered_pool_is_a_different_stream(self):
        base, drift = pools()
        schedule = linear_contamination_ramp(length=500, onset=100, maximum_rate=0.4)
        composed = compose_stream(schedule, base, drift, seed=7)
        reordered = compose_stream(schedule, base[::-1], drift, seed=7)
        assert not np.array_equal(composed.item_id, reordered.item_id)
        np.testing.assert_array_equal(composed.is_drift_item, reordered.is_drift_item)

    def test_a_different_seed_composes_a_different_stream(self):
        base, drift = pools()
        schedule = linear_contamination_ramp(length=500, onset=100, maximum_rate=0.4)
        first = compose_stream(schedule, base, drift, seed=7)
        second = compose_stream(schedule, base, drift, seed=8)
        assert not np.array_equal(first.item_id, second.item_id)

    def test_the_draw_leaves_the_global_random_state_alone(self):
        base, drift = pools()
        np.random.seed(0)
        before = np.random.random()
        np.random.seed(0)
        compose_stream(linear_contamination_ramp(500, 100, 0.4), base, drift, seed=7)
        assert np.random.random() == before


class TestVariantsAtOneSeedNest:
    def schedules_and_streams(self, seed=7):
        base, drift = pools()
        gentle = linear_contamination_ramp(length=2000, onset=200, maximum_rate=0.15)
        steep = linear_contamination_ramp(length=2000, onset=200, maximum_rate=0.30)
        null = linear_contamination_ramp(length=2000, onset=200, maximum_rate=0.0)
        return [compose_stream(schedule, base, drift, seed=seed)
                for schedule in (null, gentle, steep)]

    def test_the_null_contaminates_nothing(self):
        null, _, _ = self.schedules_and_streams()
        assert not null.is_drift_item.any()

    def test_the_gentler_ramp_contaminates_a_subset_of_the_steeper_one(self):
        _, gentle, steep = self.schedules_and_streams()
        assert gentle.is_drift_item.sum() > 0
        assert not (gentle.is_drift_item & ~steep.is_drift_item).any()

    def test_the_three_variants_nest_as_a_chain(self):
        null, gentle, steep = self.schedules_and_streams()
        for lower, upper in ((null, gentle), (gentle, steep), (null, steep)):
            assert not (lower.is_drift_item & ~upper.is_drift_item).any()

    def test_positions_neither_ramp_contaminates_carry_the_same_base_item(self):
        null, gentle, steep = self.schedules_and_streams()
        both_clean = ~gentle.is_drift_item & ~steep.is_drift_item
        np.testing.assert_array_equal(gentle.item_id[both_clean], steep.item_id[both_clean])
        np.testing.assert_array_equal(null.item_id[both_clean], steep.item_id[both_clean])

    def test_positions_both_ramps_contaminate_carry_the_same_drift_item(self):
        _, gentle, steep = self.schedules_and_streams()
        both_contaminated = gentle.is_drift_item & steep.is_drift_item
        assert both_contaminated.any()
        np.testing.assert_array_equal(gentle.item_id[both_contaminated],
                                      steep.item_id[both_contaminated])

    def test_the_null_is_the_base_stream_the_ramps_substitute_into(self):
        null, _, steep = self.schedules_and_streams()
        clean = ~steep.is_drift_item
        np.testing.assert_array_equal(null.item_id[clean], steep.item_id[clean])

    def test_nesting_is_pointwise_dominance_and_not_a_property_of_ramps(self):
        base, drift = pools()
        drawn = np.random.default_rng(0)
        lower_schedule = drawn.random(500) * 0.4
        upper_schedule = lower_schedule + drawn.random(500) * 0.5
        lower = compose_stream(lower_schedule, base, drift, seed=3)
        upper = compose_stream(upper_schedule, base, drift, seed=3)

        assert lower.is_drift_item.any() and (upper.is_drift_item & ~lower.is_drift_item).any()
        assert not (lower.is_drift_item & ~upper.is_drift_item).any()
        for shared in (lower.is_drift_item & upper.is_drift_item,
                       ~lower.is_drift_item & ~upper.is_drift_item):
            np.testing.assert_array_equal(lower.item_id[shared], upper.item_id[shared])

    def test_a_variant_at_another_seed_is_not_a_matched_control(self):
        base, drift = pools()
        null = linear_contamination_ramp(length=2000, onset=200, maximum_rate=0.0)
        steep = linear_contamination_ramp(length=2000, onset=200, maximum_rate=0.30)
        elsewhere = compose_stream(null, base, drift, seed=8)
        here = compose_stream(steep, base, drift, seed=7)
        clean = ~here.is_drift_item
        assert not np.array_equal(elsewhere.item_id[clean], here.item_id[clean])


class TestTheComposerRefusesAnUncomposableRequest:
    def test_a_pool_shared_between_base_and_drift_refuses(self):
        base, drift = pools()
        overlapping = np.concatenate([drift, base[:1]])
        with pytest.raises(ValueError, match="Base and drift pools share"):
            compose_stream(linear_contamination_ramp(500, 100, 0.4), base, overlapping, seed=7)

    def test_an_empty_pool_refuses(self):
        base, drift = pools()
        with pytest.raises(ValueError, match="at least one item"):
            compose_stream(linear_contamination_ramp(500, 100, 0.4), base, drift[:0], seed=7)
        with pytest.raises(ValueError, match="at least one item"):
            compose_stream(linear_contamination_ramp(500, 100, 0.4), base[:0], drift, seed=7)

    def test_a_repeated_id_within_one_pool_refuses(self):
        base, drift = pools()
        with pytest.raises(ValueError, match="duplicate item IDs"):
            compose_stream(linear_contamination_ramp(500, 100, 0.4),
                           np.concatenate([base, base[:1]]), drift, seed=7)

    def test_a_schedule_outside_the_unit_interval_refuses(self):
        base, drift = pools()
        with pytest.raises(ValueError, match="Schedule rates"):
            compose_stream(np.full(500, 1.5), base, drift, seed=7)

    def test_a_scheduled_rate_of_exactly_one_refuses(self):
        base, drift = pools()
        with pytest.raises(ValueError, match="Schedule rates"):
            compose_stream(np.full(500, 1.0), base, drift, seed=7)

    def test_the_schedule_refusal_reads_in_full(self):
        base, drift = pools()
        with pytest.raises(ValueError, match=re.escape("Schedule rates must be in [0, 1).")):
            compose_stream(np.full(500, 1.5), base, drift, seed=7)

    @pytest.mark.parametrize("which,label", [("base", "Base pool"),
                                             ("drift", "Drift pool")])
    def test_each_pool_refusal_names_the_pool_it_read(self, which, label):
        base, drift = pools()
        empty = {"base": (base[:0], drift), "drift": (base, drift[:0])}[which]
        repeated = {"base": (np.concatenate([base, base[:1]]), drift),
                    "drift": (base, np.concatenate([drift, drift[:1]]))}[which]
        with pytest.raises(ValueError, match=re.escape(
                f"{label} must contain at least one item.")):
            compose_stream(linear_contamination_ramp(500, 100, 0.4), *empty, seed=7)
        with pytest.raises(ValueError, match=re.escape(
                f"{label} contains duplicate item IDs; sampling would weight them unevenly.")):
            compose_stream(linear_contamination_ramp(500, 100, 0.4), *repeated, seed=7)

    def test_an_empty_schedule_refuses(self):
        base, drift = pools()
        with pytest.raises(ValueError, match="non-empty"):
            compose_stream(np.zeros(0), base, drift, seed=7)


class TestTheComposedStreamIsTheFramesColumns:
    def test_the_stream_names_the_two_frame_columns_it_supplies(self):
        assert ComposedStream._fields == ("item_id", "is_drift_item", "stream_lambda")
