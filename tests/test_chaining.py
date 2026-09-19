from types import SimpleNamespace

import numpy as np
import pytest

from rcv.chaining import (
    ALL_FRESH,
    ATTACK_COMPONENT,
    BASE_COMPONENT,
    EXHAUST_FRESH_DATA,
    CycleRecipe,
    GateUncomposable,
    absorbed,
    assembled_cycle_frame,
    assert_bounded_ladder,
    assert_columns_identical,
    assert_is_a_mixture,
    assert_pool_shares_are_within_noise,
    assert_pools_are_disjoint,
    assert_recursion_reproduces,
    assert_run_in_composes_the_alarm_time_distribution,
    assert_run_in_draws_no_attack,
    assert_schedule_is_the_declared_ramp,
    assert_serving_carves_match,
    attack_pool_of,
    audit_and_gate_blocks,
    base_pool_of,
    baseline_with_pools,
    chain_permutation,
    columns_digest,
    composed_cycle,
    cycle_generator,
    deduplicated_pool,
    initial_baseline,
    ladder_length,
    post_alarm_pools,
    rate_at,
    recipe_arrays,
    recipe_of_arrays,
    reference_after,
    stationary_segment,
    stream_of_recipe,
    unique_draw,
)
from rcv.stream_composition import linear_contamination_ramp

DIMENSION = 3


def item_block(prefix, size, repeats=1):
    identity = np.array([f"{prefix}{index:03d}" for index in range(size)])
    order = np.tile(np.arange(size), repeats)[::-1]
    return {
        "item_id": identity[order],
        "representation": (np.arange(size * DIMENSION, dtype=np.float32)
                           .reshape(size, DIMENSION)[order]),
        "verdict": (np.arange(size) % 2)[order],
        "oracle": (np.arange(size) % 3 == 0).astype(np.int64)[order],
        "human": (np.arange(size) % 4 == 0).astype(np.int64)[order],
        "family": np.arange(size)[order],
    }


def pool(prefix, size):
    return deduplicated_pool(item_block(prefix, size))


def pools_of(sizes):
    return {name: pool(f"{name}_", size) for name, size in sizes.items()}


def framed(serving_size, stream_block, schedule):
    serving = item_block("serve_", serving_size)
    serving["is_drift_item"] = np.zeros(serving_size, dtype=bool)
    stream_length = len(stream_block["item_id"])
    frame = {name: np.concatenate([serving[name], stream_block[name]]) for name in serving}
    frame["is_stream"] = np.concatenate([np.zeros(serving_size, dtype=bool),
                                         np.ones(stream_length, dtype=bool)])
    frame["stream_lambda"] = np.concatenate([np.zeros(serving_size), schedule])
    return frame


class TestTheMixingLaw:
    def test_the_first_baseline_is_all_base(self):
        assert initial_baseline() == {BASE_COMPONENT: 1.0}

    def test_authors_worked_example(self):
        after_the_first_alarm = absorbed(initial_baseline(), "a1", 0.10)
        assert after_the_first_alarm == {"base": 0.90, "a1": 0.10}

        after_the_second_alarm = absorbed(after_the_first_alarm, "a2", 0.20)
        assert after_the_second_alarm == pytest.approx({"base": 0.72, "a1": 0.08, "a2": 0.20})

    def test_absorbing_at_a_zero_rate_leaves_the_baseline_untouched(self):
        assert absorbed({"base": 0.9, "a1": 0.1}, "a2", 0.0) == {"base": 0.9, "a1": 0.1,
                                                                 "a2": 0.0}

    def test_absorbing_at_a_rate_of_one_leaves_the_attack_alone_holding_the_traffic(self):
        assert absorbed({"base": 0.9, "a1": 0.1}, "a2", 1.0) == {"base": 0.0, "a1": 0.0,
                                                                 "a2": 1.0}

    def test_the_shares_stay_a_distribution_over_a_whole_roster(self):
        baseline = initial_baseline()
        for index, rate in enumerate([0.02, 0.11, 0.3, 0.07, 0.19, 0.25, 0.04, 0.13, 0.28]):
            baseline = absorbed(baseline, f"a{index}", rate)
        assert_is_a_mixture(baseline)
        assert sum(baseline.values()) == pytest.approx(1.0)

    def test_the_categorical_keeps_the_order_the_campaigns_arrived_in(self):
        baseline = absorbed(absorbed(initial_baseline(), "a1", 0.1), "a2", 0.2)
        assert list(baseline) == ["base", "a1", "a2"]

    @pytest.mark.parametrize("rate", (-0.01, 1.01, float("nan")))
    def test_a_rate_outside_the_unit_interval_refuses(self, rate):
        with pytest.raises(ValueError, match="rate"):
            absorbed(initial_baseline(), "a1", rate)

    def test_absorbing_a_family_the_baseline_already_carries_refuses(self):
        with pytest.raises(ValueError, match="already"):
            absorbed({"base": 0.9, "a1": 0.1}, "a1", 0.2)

    def test_a_baseline_whose_shares_do_not_sum_to_one_refuses(self):
        with pytest.raises(ValueError, match="must sum to 1"):
            assert_is_a_mixture({"base": 0.5, "a1": 0.2})

    def test_an_empty_baseline_refuses(self):
        with pytest.raises(ValueError, match="must contain at least one component"):
            assert_is_a_mixture({})


class TestTheRecordedRecursionIsRederivable:
    @staticmethod
    def _recorded():
        first = initial_baseline()
        second = absorbed(first, "a1", 0.10)
        third = absorbed(second, "a2", 0.20)
        return [{"baseline_in": first, "absorption": ["a1", 0.10]},
                {"baseline_in": second, "absorption": ["a2", 0.20]},
                {"baseline_in": third, "absorption": None}]

    def test_a_faithful_record_passes(self):
        assert_recursion_reproduces(self._recorded())

    def test_a_recorded_baseline_that_does_not_follow_from_the_rates_refuses(self):
        recorded = self._recorded()
        recorded[2]["baseline_in"] = {"base": 0.72, "a1": 0.08, "a2": 0.21}
        with pytest.raises(ValueError, match="cycle 3"):
            assert_recursion_reproduces(recorded)

    def test_a_reordered_categorical_refuses(self):
        recorded = self._recorded()
        recorded[2]["baseline_in"] = {"a1": 0.08, "base": 0.72, "a2": 0.20}
        with pytest.raises(ValueError, match="cycle 3"):
            assert_recursion_reproduces(recorded)


class TestThePools:
    def test_a_pool_is_deduplicated_by_item_and_keeps_every_carried_column(self):
        drawn = deduplicated_pool(item_block("x_", 5, repeats=4))
        assert len(drawn["item_id"]) == 5
        assert set(drawn) == {"item_id", "representation", "verdict", "oracle", "human", "family"}
        assert drawn["representation"].shape == (5, DIMENSION)

    def test_the_pools_order_is_the_items_own_and_not_the_order_the_stream_served_them(self):
        served_one_way = deduplicated_pool(item_block("x_", 6, repeats=3))
        block = item_block("x_", 6, repeats=3)
        shuffled = {name: values[np.random.default_rng(0).permutation(len(values))]
                    for name, values in block.items()}
        served_another_way = deduplicated_pool(shuffled)
        assert_columns_identical(served_one_way, served_another_way, "the pool")

    def test_each_pool_row_carries_its_own_items_columns(self):
        drawn = deduplicated_pool(item_block("x_", 5, repeats=2))
        for position, identity in enumerate(drawn["item_id"]):
            index = int(identity.removeprefix("x_"))
            assert drawn["verdict"][position] == index % 2
            assert drawn["representation"][position][0] == np.float32(index * DIMENSION)

    def test_an_empty_pool_refuses(self):
        with pytest.raises(ValueError, match="empty"):
            deduplicated_pool(item_block("x_", 0))

    def test_a_block_missing_a_carried_column_refuses(self):
        block = item_block("x_", 4)
        del block["human"]
        with pytest.raises(ValueError, match="human"):
            deduplicated_pool(block)

    def test_the_base_pool_is_read_from_the_all_base_null_frame(self):
        stream = item_block("b_", 8, repeats=3)
        stream["is_drift_item"] = np.zeros(len(stream["item_id"]), dtype=bool)
        frame = framed(4, stream, np.zeros(len(stream["item_id"])))
        assert len(base_pool_of(frame)["item_id"]) == 8

    def test_a_base_pool_read_from_a_contaminated_stream_refuses(self):
        stream = item_block("b_", 8, repeats=3)
        stream["is_drift_item"] = np.zeros(len(stream["item_id"]), dtype=bool)
        stream["is_drift_item"][2] = True
        frame = framed(4, stream, np.zeros(len(stream["item_id"])))
        with pytest.raises(ValueError, match="Base stream contains"):
            base_pool_of(frame)

    def test_the_attack_pool_is_the_drift_side_of_a_family_frame(self):
        stream = item_block("d_", 10, repeats=2)
        drift = np.zeros(len(stream["item_id"]), dtype=bool)
        drift[:6] = True
        stream["is_drift_item"] = drift
        frame = framed(4, stream, np.linspace(0.0, 0.3, len(stream["item_id"])))
        assert len(attack_pool_of(frame)["item_id"]) == len(np.unique(stream["item_id"][drift]))

    def test_a_family_frame_carrying_no_drift_row_refuses(self):
        stream = item_block("d_", 10, repeats=2)
        stream["is_drift_item"] = np.zeros(len(stream["item_id"]), dtype=bool)
        frame = framed(4, stream, np.zeros(len(stream["item_id"])))
        with pytest.raises(ValueError, match="no drift"):
            attack_pool_of(frame)

    def test_pools_sharing_an_item_refuse(self):
        shared = pools_of({"base": 5, "a1": 4})
        shared["a1"] = dict(shared["a1"], item_id=np.array(shared["base"]["item_id"][:4]))
        with pytest.raises(ValueError, match="both"):
            assert_pools_are_disjoint(shared)

    def test_disjoint_pools_pass(self):
        assert_pools_are_disjoint(pools_of({"base": 5, "a1": 4, "a2": 3}))


class TestTheSharedServingCarve:
    def test_two_frames_carrying_the_same_carve_pass(self):
        carve = item_block("serve_", 20)
        assert_serving_carves_match(carve, dict(carve), "the null frame", "a family frame")

    def test_a_carve_whose_items_differ_refuses(self):
        carve = item_block("serve_", 20)
        other = dict(carve, item_id=np.array(carve["item_id"])[::-1])
        with pytest.raises(ValueError, match="item_id"):
            assert_serving_carves_match(carve, other, "the null frame", "a family frame")

    def test_a_carve_whose_labels_differ_refuses(self):
        carve = item_block("serve_", 20)
        other = dict(carve, oracle=1 - np.asarray(carve["oracle"]))
        with pytest.raises(ValueError, match="oracle"):
            assert_serving_carves_match(carve, other, "the null frame", "a family frame")


class TestTheSchedule:
    def test_the_frames_ramp_is_the_declared_one(self):
        schedule = linear_contamination_ramp(2100, 200, 0.30)
        assert_schedule_is_the_declared_ramp(schedule, run_in=200, monitored=1900, cap=0.30)

    def test_a_schedule_that_departs_from_the_closed_form_refuses(self):
        schedule = linear_contamination_ramp(2100, 200, 0.30)
        schedule[1500] += 1e-9
        with pytest.raises(ValueError, match="1500"):
            assert_schedule_is_the_declared_ramp(schedule, run_in=200, monitored=1900, cap=0.30)

    def test_a_schedule_of_the_wrong_length_refuses(self):
        schedule = linear_contamination_ramp(2100, 200, 0.30)
        with pytest.raises(ValueError, match="positions"):
            assert_schedule_is_the_declared_ramp(schedule, run_in=200, monitored=1800, cap=0.30)

    def test_the_locked_rate_is_read_at_the_stream_position(self):
        schedule = linear_contamination_ramp(2100, 200, 0.30)
        assert rate_at(schedule, 200) == 0.0
        assert rate_at(schedule, 2099) == pytest.approx(0.30)
        assert rate_at(schedule, 1200) != rate_at(schedule, 1000)

    def test_a_position_outside_the_schedule_refuses(self):
        with pytest.raises(ValueError, match="position"):
            rate_at(linear_contamination_ramp(2100, 200, 0.30), 2100)


class TestTheCycleDraw:
    @staticmethod
    def _material(cap=0.30, run_in=200, monitored=1800):
        return (pools_of({"base": 40, "a1": 20, "a2": 15}),
                linear_contamination_ramp(run_in + monitored, run_in, cap))

    def test_the_same_seeds_draw_the_identical_recipe(self):
        drawn, schedule = self._material()
        baseline = absorbed(initial_baseline(), "a1", 0.10)
        first, _ = composed_cycle(baseline, "a2", drawn, schedule, cycle_generator(42, 3))
        second, _ = composed_cycle(baseline, "a2", drawn, schedule, cycle_generator(42, 3))
        np.testing.assert_array_equal(first.pool, second.pool)
        np.testing.assert_array_equal(first.item_index, second.item_index)

    @pytest.mark.parametrize("other", ((42, 4), (43, 3)))
    def test_another_chain_seed_or_cycle_index_draws_another_recipe(self, other):
        drawn, schedule = self._material()
        baseline = absorbed(initial_baseline(), "a1", 0.10)
        first, _ = composed_cycle(baseline, "a2", drawn, schedule, cycle_generator(42, 3))
        second, _ = composed_cycle(baseline, "a2", drawn, schedule, cycle_generator(*other))
        assert not np.array_equal(first.item_index, second.item_index)

    def test_the_run_in_draws_no_current_attack(self):
        drawn, schedule = self._material()
        recipe, _ = composed_cycle(initial_baseline(), "a1", drawn, schedule,
                                   cycle_generator(42, 1))
        assert_run_in_draws_no_attack(recipe, run_in=200)
        assert not (recipe.pool[:200] == "a1").any()

    def test_a_run_in_carrying_the_attack_refuses(self):
        drawn, schedule = self._material()
        recipe, _ = composed_cycle(initial_baseline(), "a1", drawn, schedule,
                                   cycle_generator(42, 1))
        contaminated = recipe.pool.copy()
        contaminated[7] = "a1"
        with pytest.raises(ValueError, match="Run-in samples attack"):
            assert_run_in_draws_no_attack(recipe._replace(pool=contaminated), run_in=200)

    def test_the_drawn_shares_match_the_design(self):
        drawn, schedule = self._material(monitored=18000)
        baseline = absorbed(initial_baseline(), "a1", 0.10)
        recipe, _ = composed_cycle(baseline, "a2", drawn, schedule, cycle_generator(42, 2))
        assert_pool_shares_are_within_noise(recipe, baseline, sigmas=5.0)

    def test_a_recipe_whose_shares_were_forced_refuses(self):
        drawn, schedule = self._material(monitored=18000)
        baseline = absorbed(initial_baseline(), "a1", 0.10)
        recipe, _ = composed_cycle(baseline, "a2", drawn, schedule, cycle_generator(42, 2))
        forced = recipe.pool.copy()
        forced[recipe.pool == "base"] = "a1"
        with pytest.raises(ValueError, match="a1"):
            assert_pool_shares_are_within_noise(recipe._replace(pool=forced), baseline,
                                                sigmas=5.0)

    def test_an_unknown_attack_family_refuses(self):
        drawn, schedule = self._material()
        with pytest.raises(ValueError, match="No pool named"):
            composed_cycle(initial_baseline(), "nosuchfamily", drawn, schedule,
                           cycle_generator(42, 1))

    def test_a_baseline_component_with_no_pool_refuses(self):
        drawn, schedule = self._material()
        baseline = absorbed(initial_baseline(), "ghost", 0.10)
        with pytest.raises(ValueError, match="No pool named"):
            composed_cycle(baseline, "a1", drawn, schedule, cycle_generator(42, 1))

    def test_an_empty_pool_refuses(self):
        drawn, schedule = self._material()
        drawn["a1"] = {name: values[:0] for name, values in drawn["a1"].items()}
        with pytest.raises(ValueError, match="empty"):
            composed_cycle(initial_baseline(), "a1", drawn, schedule, cycle_generator(42, 1))

    def test_an_attack_the_baseline_has_already_absorbed_refuses(self):
        drawn, schedule = self._material()
        baseline = absorbed(initial_baseline(), "a1", 0.10)
        with pytest.raises(ValueError, match="already"):
            composed_cycle(baseline, "a1", drawn, schedule, cycle_generator(42, 2))


class TestTheAssembledStream:
    @staticmethod
    def _composed():
        drawn = pools_of({"base": 40, "a1": 20, "a2": 15})
        schedule = linear_contamination_ramp(2000, 200, 0.30)
        baseline = absorbed(initial_baseline(), "a1", 0.10)
        recipe, stream = composed_cycle(baseline, "a2", drawn, schedule, cycle_generator(42, 2))
        return drawn, recipe, stream

    def test_every_position_serves_the_item_its_recipe_names(self):
        drawn, recipe, stream = self._composed()
        for position in (0, 199, 200, 1000, 1999):
            source = drawn[str(recipe.pool[position])]
            assert stream["item_id"][position] == source["item_id"][recipe.item_index[position]]

    def test_only_the_current_attack_is_marked_as_drift(self):
        _, recipe, stream = self._composed()
        np.testing.assert_array_equal(stream["is_drift_item"], recipe.pool == "a2")
        assert not stream["is_drift_item"][recipe.pool == "a1"].any()

    def test_the_provenance_column_records_where_each_position_drew_from(self):
        _, recipe, stream = self._composed()
        np.testing.assert_array_equal(stream["pool"], recipe.pool)
        assert set(np.unique(stream["pool"])) == {"base", "a1", "a2"}

    def test_the_stream_carries_the_source_frames_columns(self):
        _, _, stream = self._composed()
        assert set(stream) == {"item_id", "representation", "verdict", "oracle", "human",
                               "family", "is_drift_item", "is_stream", "stream_lambda", "pool"}
        assert stream["is_stream"].all()

    def test_the_cycle_frame_is_the_shared_carve_then_the_composed_stream(self):
        _, _, stream = self._composed()
        serving = item_block("serve_", 30)
        serving["is_stream"] = np.zeros(30, dtype=bool)
        serving["is_drift_item"] = np.zeros(30, dtype=bool)
        serving["stream_lambda"] = np.zeros(30)
        frame = assembled_cycle_frame(serving, stream)
        assert len(frame["item_id"]) == 30 + len(stream["item_id"])
        assert set(frame) == set(stream)
        assert not frame["is_stream"][:30].any() and frame["is_stream"][30:].all()
        assert set(np.unique(frame["pool"][:30])) == {"serving"}


class TestTheRecipeIsTheArtifact:
    @staticmethod
    def _composed():
        drawn = pools_of({"base": 40, "a1": 20, "a2": 15})
        schedule = linear_contamination_ramp(2000, 200, 0.30)
        baseline = absorbed(initial_baseline(), "a1", 0.10)
        recipe, stream = composed_cycle(baseline, "a2", drawn, schedule, cycle_generator(42, 2))
        return drawn, recipe, stream

    def test_the_recipe_round_trips_through_its_arrays(self):
        _, recipe, stream = self._composed()
        restored, served = recipe_of_arrays(recipe_arrays(recipe, stream["item_id"]))
        assert restored.attack_family == recipe.attack_family
        np.testing.assert_array_equal(restored.pool, recipe.pool)
        np.testing.assert_array_equal(restored.item_index, recipe.item_index)
        np.testing.assert_array_equal(restored.stream_lambda, recipe.stream_lambda)
        np.testing.assert_array_equal(served, stream["item_id"])

    def test_the_recipe_survives_the_npz_it_is_stored_in(self, tmp_path):
        drawn, recipe, stream = self._composed()
        path = tmp_path / "recipe_1.npz"
        np.savez(path, **recipe_arrays(recipe, stream["item_id"]))
        with np.load(path, allow_pickle=False) as stored:
            restored, served = recipe_of_arrays({name: stored[name] for name in stored.files})
        assert_columns_identical(stream_of_recipe(restored, drawn), stream, "the recomposed cycle")
        np.testing.assert_array_equal(served, stream["item_id"])

    def test_a_tampered_recipe_refuses_on_its_digest(self):
        _, recipe, stream = self._composed()
        arrays = recipe_arrays(recipe, stream["item_id"])
        arrays["item_index"] = np.asarray(arrays["item_index"]).copy()
        arrays["item_index"][17] += 1
        with pytest.raises(ValueError, match="SHA-256 mismatch"):
            recipe_of_arrays(arrays)

    def test_a_recipe_pointing_past_the_end_of_a_pool_refuses(self):
        drawn, recipe, _ = self._composed()
        overrun = recipe.item_index.copy()
        overrun[3] = 10_000
        with pytest.raises(ValueError, match="pool"):
            stream_of_recipe(recipe._replace(item_index=overrun), drawn)

    def test_a_recipe_naming_a_pool_that_was_not_built_refuses(self):
        drawn, recipe, _ = self._composed()
        with pytest.raises(ValueError, match="Recipe draws from"):
            stream_of_recipe(recipe, {"base": drawn["base"]})


class TestByteIdentity:
    def test_identical_blocks_pass(self):
        assert_columns_identical(item_block("x_", 5), item_block("x_", 5), "a block")

    def test_a_column_that_moved_refuses_by_name(self):
        first, second = item_block("x_", 5), item_block("x_", 5)
        second["oracle"] = 1 - np.asarray(second["oracle"])
        with pytest.raises(ValueError, match="oracle"):
            assert_columns_identical(first, second, "a block")

    def test_a_column_that_changed_dtype_refuses(self):
        first, second = item_block("x_", 5), item_block("x_", 5)
        second["representation"] = np.asarray(second["representation"], dtype=np.float64)
        with pytest.raises(ValueError, match="representation"):
            assert_columns_identical(first, second, "a block")

    def test_a_missing_column_refuses(self):
        first, second = item_block("x_", 5), item_block("x_", 5)
        del second["human"]
        with pytest.raises(ValueError, match="human"):
            assert_columns_identical(first, second, "a block")


class TestThePermutation:
    def test_the_permutation_is_a_function_of_the_chain_seed(self):
        roster = ["cyber", "fraud", "toxic", "violence"]
        assert chain_permutation(roster, 42) == chain_permutation(roster, 42)
        assert chain_permutation(roster, 42) != chain_permutation(roster, 1024)

    def test_the_permutation_serves_every_family_once(self):
        roster = ["cyber", "fraud", "toxic", "violence"]
        assert sorted(chain_permutation(roster, 42)) == sorted(roster)

    def test_a_roster_repeating_a_family_refuses(self):
        with pytest.raises(ValueError, match="repeats"):
            chain_permutation(["cyber", "cyber"], 42)

    def test_the_permutation_is_plain_strings_a_record_can_carry(self):
        assert all(type(family) is str for family in chain_permutation(["a", "b", "c"], 42))


class TestTheRecipeType:
    def test_the_recipe_names_its_four_fields(self):
        assert CycleRecipe._fields == ("pool", "item_index", "stream_lambda", "attack_family")




def ramp_frame(base_size=30, attack_size=12, repeats=8, serving_size=20):
    base, attack = item_block("b_", base_size, repeats), item_block("d_", attack_size, repeats)
    stream = {name: np.concatenate([values, attack[name]]) for name, values in base.items()}
    length = len(stream["item_id"])
    drift = np.zeros(length, dtype=bool)
    drift[base_size * repeats:] = True
    stream["is_drift_item"] = drift
    return framed(serving_size, stream, np.linspace(0.0, 0.3, length))


class TestThePostAlarmPools:
    def test_the_stream_splits_into_a_base_pool_and_an_attack_pool(self):
        base, attack = post_alarm_pools(ramp_frame())
        assert set(np.asarray(base["item_id"])) == {f"b_{index:03d}" for index in range(30)}
        assert set(np.asarray(attack["item_id"])) == {f"d_{index:03d}" for index in range(12)}

    def test_both_pools_are_deduplicated_by_item(self):
        base, attack = post_alarm_pools(ramp_frame())
        assert len(base["item_id"]) == len(set(np.asarray(base["item_id"])))
        assert len(attack["item_id"]) == len(set(np.asarray(attack["item_id"])))

    def test_the_serving_block_is_in_neither_pool(self):
        base, attack = post_alarm_pools(ramp_frame())
        served = set(np.asarray(base["item_id"])) | set(np.asarray(attack["item_id"]))
        assert not any(identity.startswith("serve_") for identity in served)

    def test_a_stream_carrying_no_attack_refuses(self):
        frame = ramp_frame()
        frame["is_drift_item"] = np.zeros(len(frame["item_id"]), dtype=bool)
        with pytest.raises(ValueError, match="attack"):
            post_alarm_pools(frame)

    def test_a_stream_that_is_all_attack_refuses(self):
        frame = ramp_frame()
        frame["is_drift_item"] = np.asarray(frame["is_stream"]).copy()
        with pytest.raises(ValueError, match="base"):
            post_alarm_pools(frame)


def segment_frame(*arguments, **keywords):
    frame, _provenance = stationary_segment(*arguments, **keywords)
    return frame


class TestTheStationarySegment:
    @staticmethod
    def _material():
        drawn = pools_of({"base": 4000, "a1": 2000})
        return baseline_with_pools({"base": 1.0}, drawn), drawn

    def test_the_segment_is_the_length_it_was_asked_for(self):
        baseline, drawn = self._material()
        assert len(segment_frame(baseline, drawn["a1"], 0.2, 500,
                                 np.random.default_rng(7))["item_id"]) == 500

    def test_the_attack_share_is_within_binomial_noise_of_the_rate(self):
        baseline, drawn = self._material()
        rate, length = 0.2, 4000
        segment = segment_frame(baseline, drawn["a1"], rate, length, np.random.default_rng(7))
        drew_the_attack = int(np.asarray(segment["is_drift_item"]).sum())
        sigma = np.sqrt(length * rate * (1 - rate))
        assert abs(drew_the_attack - rate * length) < 5 * sigma

    def test_the_same_seed_rate_and_exclusion_compose_the_identical_segment(self):
        baseline, drawn = self._material()
        excluded = frozenset(np.asarray(drawn["base"]["item_id"])[:50])
        first = segment_frame(baseline, drawn["a1"], 0.2, 800, np.random.default_rng(11),
                              exclude_items=excluded)
        second = segment_frame(baseline, drawn["a1"], 0.2, 800, np.random.default_rng(11),
                               exclude_items=excluded)
        assert_columns_identical(first, second, "the segment")

    def test_another_exclusion_composes_another_segment(self):
        baseline, drawn = self._material()
        first = segment_frame(baseline, drawn["a1"], 0.2, 800, np.random.default_rng(11))
        second = segment_frame(baseline, drawn["a1"], 0.2, 800, np.random.default_rng(11),
                               exclude_items=frozenset(np.asarray(drawn["base"]["item_id"])[:50]))
        assert not np.array_equal(np.asarray(first["item_id"]), np.asarray(second["item_id"]))

    def test_another_seed_composes_another_segment(self):
        baseline, drawn = self._material()
        first = segment_frame(baseline, drawn["a1"], 0.2, 800, np.random.default_rng(11))
        second = segment_frame(baseline, drawn["a1"], 0.2, 800, np.random.default_rng(12))
        assert not np.array_equal(np.asarray(first["item_id"]), np.asarray(second["item_id"]))

    def test_the_pool_assignment_nests_across_rates(self):
        baseline, drawn = self._material()
        hot = segment_frame(baseline, drawn["a1"], 0.30, 600, np.random.default_rng(3))
        mild = segment_frame(baseline, drawn["a1"], 0.10, 600, np.random.default_rng(3))
        clean_when_hot = ~np.asarray(hot["is_drift_item"])
        assert not np.asarray(mild["is_drift_item"])[clean_when_hot].any()
        np.testing.assert_array_equal(np.asarray(mild["pool"])[clean_when_hot],
                                      np.asarray(hot["pool"])[clean_when_hot])

    def test_no_item_is_served_twice_while_the_pools_allow(self):
        baseline, drawn = self._material()
        served = np.asarray(segment_frame(baseline, drawn["a1"], 0.2, 1200,
                                          np.random.default_rng(4))["item_id"])
        assert len(set(served)) == len(served)

    def test_no_consumed_item_appears_at_all(self):
        baseline, drawn = self._material()
        consumed = frozenset(np.asarray(drawn["base"]["item_id"])[:3000]) | frozenset(
            np.asarray(drawn["a1"]["item_id"])[:1500])
        served = np.asarray(segment_frame(baseline, drawn["a1"], 0.2, 1200,
                                          np.random.default_rng(4),
                                          exclude_items=consumed)["item_id"])
        assert not (set(served) & consumed)

    def test_the_provenance_names_exactly_what_the_addendum_pins(self):
        baseline, drawn = self._material()
        _segment, provenance = stationary_segment(baseline, drawn["a1"], 0.2, 1200,
                                                  np.random.default_rng(4))
        assert set(provenance) == {"rate", "length", "eligible_attack", "eligible_base",
                                   "reused_attack_rows", "reused_base_rows", "sha256"}
        assert (provenance["rate"], provenance["length"]) == (0.2, 1200)
        assert (provenance["eligible_attack"], provenance["eligible_base"]) == (2000, 4000)
        assert provenance["reused_attack_rows"] == provenance["reused_base_rows"] == 0

    def test_the_exclusion_shows_up_in_the_eligible_counts(self):
        baseline, drawn = self._material()
        _segment, provenance = stationary_segment(
            baseline, drawn["a1"], 0.2, 1200, np.random.default_rng(4),
            exclude_items=frozenset(np.asarray(drawn["base"]["item_id"])[:1000]))
        assert provenance["eligible_base"] == 3000
        assert provenance["eligible_attack"] == 2000

    def test_a_pool_that_runs_dry_is_reused_and_the_reuse_is_counted(self):
        drawn = pools_of({"base": 4000, "a1": 40})
        baseline = baseline_with_pools({"base": 1.0}, drawn)
        segment, provenance = stationary_segment(baseline, drawn["a1"], 0.3, 1200,
                                                 np.random.default_rng(4))
        attack_rows = int(np.asarray(segment["is_drift_item"]).sum())
        assert provenance["reused_attack_rows"] == attack_rows - 40
        assert provenance["reused_base_rows"] == 0

    def test_a_pool_the_exclusion_empties_falls_back_with_every_row_counted(self):
        drawn = pools_of({"base": 4000, "a1": 40})
        baseline = baseline_with_pools({"base": 1.0}, drawn)
        segment, provenance = stationary_segment(
            baseline, drawn["a1"], 0.3, 1200, np.random.default_rng(4),
            exclude_items=frozenset(np.asarray(drawn["a1"]["item_id"])))
        assert provenance["eligible_attack"] == 0
        assert provenance["reused_attack_rows"] == int(np.asarray(segment["is_drift_item"]).sum())

    def test_a_later_cycles_segment_holds_nothing_the_earlier_one_audited(self):
        drawn = pools_of({"base": 4000, "a1": 2000, "a2": 1500})
        first = segment_frame(baseline_with_pools({"base": 1.0}, drawn), drawn["a1"], 0.15,
                              ladder_length(300, 3), np.random.default_rng([1, 1]))
        audited = frozenset(np.asarray(first["item_id"]))
        second = segment_frame(baseline_with_pools(absorbed(initial_baseline(), "a1", 0.15), drawn),
                               drawn["a2"], 0.2, ladder_length(300, 3),
                               np.random.default_rng([1, 2]), exclude_items=audited)
        assert not (set(np.asarray(second["item_id"])) & audited)

    def test_only_the_attack_rows_are_marked_as_drift(self):
        baseline, drawn = self._material()
        segment = segment_frame(baseline, drawn["a1"], 0.25, 500, np.random.default_rng(5))
        attack_ids = set(np.asarray(drawn["a1"]["item_id"]))
        for identity, drift in zip(np.asarray(segment["item_id"]),
                                   np.asarray(segment["is_drift_item"])):
            assert bool(drift) == (identity in attack_ids)

    def test_the_provenance_column_names_the_pool_each_position_drew_from(self):
        baseline, drawn = self._material()
        segment = segment_frame(baseline, drawn["a1"], 0.25, 500, np.random.default_rng(5))
        assert set(np.unique(segment["pool"])) == {"base", ATTACK_COMPONENT}
        np.testing.assert_array_equal(np.asarray(segment["pool"]) == ATTACK_COMPONENT,
                                      np.asarray(segment["is_drift_item"]))

    def test_the_segment_carries_every_column_a_fitting_slice_concatenates(self):
        baseline, drawn = self._material()
        segment = segment_frame(baseline, drawn["a1"], 0.2, 100, np.random.default_rng(5))
        assert set(segment) == {"item_id", "representation", "verdict", "oracle", "human",
                                "family", "is_drift_item", "is_stream", "stream_lambda", "pool"}

    def test_the_scheduled_rate_column_records_the_rate_it_was_composed_at(self):
        baseline, drawn = self._material()
        segment = segment_frame(baseline, drawn["a1"], 0.2, 100, np.random.default_rng(5))
        assert (np.asarray(segment["stream_lambda"]) == 0.2).all()

    def test_a_multi_component_baseline_appears_in_its_designed_shares(self):
        drawn = pools_of({"base": 40000, "a1": 20000, "a2": 15000})
        baseline = baseline_with_pools(absorbed(initial_baseline(), "a1", 0.25), drawn)
        segment = segment_frame(baseline, drawn["a2"], 0.2, 40000, np.random.default_rng(9))
        pool = np.asarray(segment["pool"])
        assert int((pool == ATTACK_COMPONENT).sum()) == pytest.approx(0.2 * 40000, rel=0.05)
        assert int((pool == "a1").sum()) == pytest.approx(0.8 * 0.25 * 40000, rel=0.05)
        assert int((pool == "base").sum()) == pytest.approx(0.8 * 0.75 * 40000, rel=0.05)

    @pytest.mark.parametrize("rate", (-0.01, 1.0, 1.5))
    def test_a_rate_outside_the_unit_interval_refuses(self, rate):
        baseline, drawn = self._material()
        with pytest.raises(ValueError, match="rate"):
            stationary_segment(baseline, drawn["a1"], rate, 100, np.random.default_rng(5))

    @pytest.mark.parametrize("length", (0, -5))
    def test_a_length_that_composes_nothing_refuses(self, length):
        baseline, drawn = self._material()
        with pytest.raises(ValueError, match="length"):
            stationary_segment(baseline, drawn["a1"], 0.2, length, np.random.default_rng(5))

    def test_an_empty_attack_pool_refuses(self):
        baseline, drawn = self._material()
        empty = {name: values[:0] for name, values in drawn["a1"].items()}
        with pytest.raises(ValueError, match="empty"):
            stationary_segment(baseline, empty, 0.2, 100, np.random.default_rng(5))

    def test_a_baseline_that_is_not_a_mixture_refuses(self):
        drawn = pools_of({"base": 40, "a1": 20})
        with pytest.raises(ValueError, match="must sum to 1"):
            stationary_segment({"base": (0.5, drawn["base"])}, drawn["a1"], 0.2, 100,
                               np.random.default_rng(5))

    def test_a_baseline_carrying_the_attacks_own_name_refuses(self):
        drawn = pools_of({"base": 40, "a1": 20})
        with pytest.raises(ValueError, match=ATTACK_COMPONENT):
            stationary_segment({ATTACK_COMPONENT: (1.0, drawn["base"])}, drawn["a1"], 0.2, 100,
                               np.random.default_rng(5))


class TestTheLadderSizesTheSegment:
    def test_the_length_is_the_budget_times_the_attempts(self):
        assert ladder_length(300, 3) == 1200
        assert ladder_length(300, 0) == 300

    @pytest.mark.parametrize("knobs", ((0, 3), (300, -1)))
    def test_knobs_that_buy_nothing_refuse(self, knobs):
        with pytest.raises(ValueError, match="audit_budget must be at least 1"):
            ladder_length(*knobs)


class TestTheUniqueDraw:
    def test_a_pool_that_covers_the_need_repeats_nothing(self):
        drawn, repeats = unique_draw(50, 20, np.random.default_rng(1))
        assert repeats == 0
        assert len(set(drawn.tolist())) == 20

    def test_a_pool_shorter_than_the_need_reports_the_shortfall(self):
        drawn, repeats = unique_draw(10, 25, np.random.default_rng(1))
        assert repeats == 15
        assert len(drawn) == 25
        assert set(drawn.tolist()) == set(range(10))

    def test_the_draw_depends_on_the_pool_and_the_seed_and_not_on_the_need(self):
        long, _ = unique_draw(50, 40, np.random.default_rng(2))
        short, _ = unique_draw(50, 10, np.random.default_rng(2))
        np.testing.assert_array_equal(long[:10], short)


class TestTheBaselineCarriesItsPools:
    def test_every_component_is_paired_with_its_pool(self):
        drawn = pools_of({"base": 40, "a1": 20})
        paired = baseline_with_pools({"base": 0.9, "a1": 0.1}, drawn)
        assert paired["base"][0] == 0.9
        assert paired["a1"][1] is drawn["a1"]

    def test_a_component_with_no_pool_refuses(self):
        with pytest.raises(ValueError, match="No pools for baseline components"):
            baseline_with_pools({"base": 0.9, "ghost": 0.1}, pools_of({"base": 40}))


class TestTheSegmentsDigest:
    def test_the_same_block_digests_the_same(self):
        assert columns_digest(item_block("x_", 5)) == columns_digest(item_block("x_", 5))

    def test_a_moved_value_moves_the_digest(self):
        block = item_block("x_", 5)
        moved = dict(block, oracle=1 - np.asarray(block["oracle"]))
        assert columns_digest(block) != columns_digest(moved)

    def test_a_recomposed_segment_digests_to_what_the_first_composition_did(self):
        drawn = pools_of({"base": 4000, "a1": 2000})
        baseline = baseline_with_pools({"base": 1.0}, drawn)
        first, first_provenance = stationary_segment(baseline, drawn["a1"], 0.17, 900,
                                                     np.random.default_rng([1, 2]))
        again, again_provenance = stationary_segment(baseline, drawn["a1"], 0.17, 900,
                                                     np.random.default_rng([1, 2]))
        assert columns_digest(first) == columns_digest(again) == first_provenance["sha256"]
        assert first_provenance == again_provenance




def block_ids(block):
    return set(np.asarray(block["item_id"]).tolist())


class TestTheAuditAndGateBlocks:
    @staticmethod
    def _material(base=4000, absorbed_size=2000, attack=1500):
        drawn = pools_of({"base": base, "a1": absorbed_size, "a2": attack})
        return baseline_with_pools(absorbed(initial_baseline(), "a1", 0.2), drawn), drawn

    def test_both_blocks_are_the_length_they_were_asked_for(self):
        baseline, drawn = self._material()
        audit, gate, _ = audit_and_gate_blocks(baseline, drawn["a2"], 0.15, 1200, 300,
                                               np.random.default_rng(1))
        assert len(audit["item_id"]) == 1200
        assert len(gate["item_id"]) == 300

    def test_the_two_blocks_never_share_an_item(self):
        baseline, drawn = self._material()
        audit, gate, _ = audit_and_gate_blocks(baseline, drawn["a2"], 0.15, 1200, 300,
                                               np.random.default_rng(1))
        assert not (block_ids(audit) & block_ids(gate))

    def test_no_consumed_item_reaches_the_gate_block(self):
        baseline, drawn = self._material()
        consumed = frozenset(np.asarray(drawn["base"]["item_id"])[:2500]) | frozenset(
            np.asarray(drawn["a2"]["item_id"])[:900])
        _audit, gate, _ = audit_and_gate_blocks(baseline, drawn["a2"], 0.15, 1200, 300,
                                                np.random.default_rng(1),
                                                consumed_items=consumed)
        assert not (block_ids(gate) & consumed)

    def test_the_gate_has_first_claim_on_the_fresh_items(self):
        drawn = pools_of({"base": 4000, "a2": 1500})
        baseline = baseline_with_pools({"base": 1.0}, drawn)
        consumed = frozenset(np.asarray(drawn["base"]["item_id"])[:3670]) | frozenset(
            np.asarray(drawn["a2"]["item_id"])[:1440])
        audit, gate, provenance = audit_and_gate_blocks(baseline, drawn["a2"], 0.15, 1200, 300,
                                                        np.random.default_rng(1),
                                                        consumed_items=consumed)
        assert not (block_ids(gate) & consumed)
        assert provenance["gate_reused_rows"] == 0
        assert provenance["audit_reused_rows"] > 0
        assert not (block_ids(audit) & block_ids(gate))

    def test_reuse_is_counted_per_block(self):
        drawn = pools_of({"base": 4000, "a2": 120})
        baseline = baseline_with_pools({"base": 1.0}, drawn)
        audit, gate, provenance = audit_and_gate_blocks(baseline, drawn["a2"], 0.30, 1200, 300,
                                                        np.random.default_rng(1))
        assert provenance["audit_reused_rows"] > 0
        assert provenance["gate_reused_rows"] == 0
        assert len(block_ids(audit)) < len(audit["item_id"])

    def test_a_gate_that_cannot_be_composed_refuses_by_name(self):
        drawn = pools_of({"base": 4000, "a2": 60})
        baseline = baseline_with_pools({"base": 1.0}, drawn)
        consumed = frozenset(np.asarray(drawn["a2"]["item_id"]))
        with pytest.raises(GateUncomposable, match="a2|attack"):
            audit_and_gate_blocks(baseline, drawn["a2"], 0.30, 1200, 300,
                                  np.random.default_rng(1), consumed_items=consumed)

    def test_an_uncomposable_gate_is_a_refusal_a_caller_can_catch_as_a_value_error(self):
        assert issubclass(GateUncomposable, ValueError)

    def test_the_gate_repeats_never_consumed_items_rather_than_reaching_for_a_consumed_one(self):
        drawn = pools_of({"base": 4000, "a2": 1500})
        baseline = baseline_with_pools({"base": 1.0}, drawn)
        consumed = frozenset(np.asarray(drawn["a2"]["item_id"])[:1450])
        _audit, gate, provenance = audit_and_gate_blocks(baseline, drawn["a2"], 0.30, 1200, 300,
                                                         np.random.default_rng(1),
                                                         consumed_items=consumed)
        assert provenance["gate_reused_rows"] > 0
        assert len(block_ids(gate)) < len(gate["item_id"])
        assert not (block_ids(gate) & consumed)

    def test_a_pool_the_gate_exhausts_outright_leaves_the_audit_uncomposable_and_says_so(self):
        drawn = pools_of({"base": 4000, "a2": 20})
        baseline = baseline_with_pools({"base": 1.0}, drawn)
        with pytest.raises(ValueError, match="audit block"):
            audit_and_gate_blocks(baseline, drawn["a2"], 0.30, 1200, 300,
                                  np.random.default_rng(1))

    def test_the_provenance_names_exactly_what_the_addendum_pins(self):
        baseline, drawn = self._material()
        _audit, _gate, provenance = audit_and_gate_blocks(baseline, drawn["a2"], 0.15, 1200, 300,
                                                          np.random.default_rng(1))
        assert set(provenance) == {"rate", "audit_length", "gate_length", "eligible_attack",
                                   "eligible_base", "audit_reused_rows", "gate_reused_rows",
                                   "audit_sha256", "gate_sha256"}
        assert (provenance["rate"], provenance["audit_length"],
                provenance["gate_length"]) == (0.15, 1200, 300)
        assert (provenance["eligible_attack"], provenance["eligible_base"]) == (1500, 6000)

    def test_the_eligible_counts_fall_as_the_probe_consumes(self):
        baseline, drawn = self._material()
        _audit, _gate, provenance = audit_and_gate_blocks(
            baseline, drawn["a2"], 0.15, 1200, 300, np.random.default_rng(1),
            consumed_items=frozenset(np.asarray(drawn["base"]["item_id"])[:1000]))
        assert provenance["eligible_base"] == 5000
        assert provenance["eligible_attack"] == 1500

    def test_the_digests_are_of_the_blocks_that_were_returned(self):
        baseline, drawn = self._material()
        audit, gate, provenance = audit_and_gate_blocks(baseline, drawn["a2"], 0.15, 1200, 300,
                                                        np.random.default_rng(1))
        assert provenance["audit_sha256"] == columns_digest(audit)
        assert provenance["gate_sha256"] == columns_digest(gate)

    def test_the_pair_is_deterministic_in_seed_rate_lengths_and_the_consumed_set(self):
        baseline, drawn = self._material()
        consumed = frozenset(np.asarray(drawn["base"]["item_id"])[:400])
        first = audit_and_gate_blocks(baseline, drawn["a2"], 0.15, 1200, 300,
                                      np.random.default_rng([7, 2]), consumed_items=consumed)
        second = audit_and_gate_blocks(baseline, drawn["a2"], 0.15, 1200, 300,
                                       np.random.default_rng([7, 2]), consumed_items=consumed)
        assert_columns_identical(first[0], second[0], "the audit block")
        assert_columns_identical(first[1], second[1], "the gate block")
        assert first[2] == second[2]

    def test_another_consumed_set_composes_another_pair(self):
        baseline, drawn = self._material()
        first = audit_and_gate_blocks(baseline, drawn["a2"], 0.15, 1200, 300,
                                      np.random.default_rng([7, 2]))
        second = audit_and_gate_blocks(baseline, drawn["a2"], 0.15, 1200, 300,
                                       np.random.default_rng([7, 2]),
                                       consumed_items=frozenset(
                                           np.asarray(drawn["base"]["item_id"])[:400]))
        assert first[2]["gate_sha256"] != second[2]["gate_sha256"]

    @pytest.mark.parametrize("block, length", ((0, 1200), (1, 300)))
    def test_both_blocks_carry_the_attack_at_the_alarms_rate(self, block, length):
        baseline, drawn = self._material()
        rate = 0.15
        composed = audit_and_gate_blocks(baseline, drawn["a2"], rate, 1200, 300,
                                         np.random.default_rng(3))[block]
        drew_the_attack = int(np.asarray(composed["is_drift_item"]).sum())
        sigma = np.sqrt(length * rate * (1 - rate))
        assert abs(drew_the_attack - rate * length) < 5 * sigma

    def test_both_blocks_carry_the_columns_a_fitting_slice_concatenates(self):
        baseline, drawn = self._material()
        for composed in audit_and_gate_blocks(baseline, drawn["a2"], 0.15, 1200, 300,
                                              np.random.default_rng(3))[:2]:
            assert set(composed) == {"item_id", "representation", "verdict", "oracle", "human",
                                     "family", "is_drift_item", "is_stream", "stream_lambda",
                                     "pool"}

    def test_a_later_cycles_gate_shares_nothing_with_the_earlier_cycles_blocks(self):
        drawn = pools_of({"base": 6000, "a1": 2000, "a2": 1500})
        first_audit, first_gate, _ = audit_and_gate_blocks(
            baseline_with_pools({"base": 1.0}, drawn), drawn["a1"], 0.15,
            ladder_length(300, 3), 300, np.random.default_rng([1, 1]))
        consumed = frozenset(block_ids(first_audit) | block_ids(first_gate))
        _second_audit, second_gate, _ = audit_and_gate_blocks(
            baseline_with_pools(absorbed(initial_baseline(), "a1", 0.15), drawn), drawn["a2"], 0.2,
            ladder_length(300, 3), 300, np.random.default_rng([1, 2]), consumed_items=consumed)
        assert not (block_ids(second_gate) & consumed)


class TestTheReferenceTravelsWithTheWorld:
    @staticmethod
    def _entry(what, recall=None, over_block=None):
        return SimpleNamespace(what=what, gate_recall=recall, gate_over_block=over_block)

    def test_a_cycle_that_repaired_hands_on_its_own_gate_reading(self):
        standing = {"recall": 0.83, "over_block": 0.014}
        travelled = reference_after([self._entry("repair", 0.86, 0.021)], standing)
        assert travelled == {"recall": 0.86, "over_block": 0.021}

    def test_a_cycle_that_did_not_repair_leaves_the_standard_where_it_was(self):
        standing = {"recall": 0.83, "over_block": 0.014}
        assert reference_after([self._entry("escalation_demanded")], standing) is standing
        assert reference_after([], standing) is standing

    def test_a_repair_recorded_without_its_gate_reading_refuses(self):
        with pytest.raises(ValueError, match="gate"):
            reference_after([self._entry("repair")], {"recall": 0.83, "over_block": 0.014})

    def test_the_last_repair_of_an_episode_is_the_one_that_travels(self):
        travelled = reference_after([self._entry("repair", 0.80, 0.03),
                                     self._entry("repair", 0.88, 0.02)],
                                    {"recall": 0.83, "over_block": 0.014})
        assert travelled == {"recall": 0.88, "over_block": 0.02}


class TestTheRunInIsTheAlarmTimeDistribution:
    @staticmethod
    def _cycle(baseline, attack, rate, run_in=200, monitored=1800, seed=5):
        drawn = pools_of({"base": 4000, "a1": 2000, "a2": 1500})
        schedule = linear_contamination_ramp(run_in + monitored, run_in, 0.30)
        recipe, _stream = composed_cycle(baseline, attack, drawn, schedule,
                                         np.random.default_rng(seed))
        return recipe

    def test_a_run_in_composed_at_the_absorbed_mixture_passes(self):
        first = initial_baseline()
        second = absorbed(first, "a1", 0.2)
        recipe = self._cycle(second, "a2", 0.2)
        assert_run_in_composes_the_alarm_time_distribution(
            recipe, 200, second, {"baseline_in": first, "family": "a1", "r_locked": 0.2}, 5.0)

    def test_the_first_cycles_run_in_is_the_untouched_baseline(self):
        recipe = self._cycle(initial_baseline(), "a1", 0.2)
        assert_run_in_composes_the_alarm_time_distribution(
            recipe, 200, initial_baseline(), None, 5.0)

    def test_a_run_in_that_is_not_the_frozen_distribution_refuses(self):
        first = initial_baseline()
        second = absorbed(first, "a1", 0.2)
        recipe = self._cycle(second, "a2", 0.2)
        with pytest.raises(ValueError, match="prior absorbed distribution"):
            assert_run_in_composes_the_alarm_time_distribution(
                recipe, 200, second,
                {"baseline_in": first, "family": "a1", "r_locked": 0.25}, 5.0)

    def test_a_run_in_whose_draw_departs_from_its_own_design_refuses(self):
        first = initial_baseline()
        second = absorbed(first, "a1", 0.2)
        recipe = self._cycle(second, "a2", 0.2)
        forced = recipe.pool.copy()
        forced[:200] = "base"
        with pytest.raises(ValueError, match="a1"):
            assert_run_in_composes_the_alarm_time_distribution(
                recipe._replace(pool=forced), 200, second,
                {"baseline_in": first, "family": "a1", "r_locked": 0.2}, 5.0)




class TestComposingUntilDry:
    @staticmethod
    def _material(base=900, attack=120):
        drawn = pools_of({"base": base, "a2": attack})
        return baseline_with_pools({"base": 1.0}, drawn), drawn

    def test_the_realised_length_is_recorded_in_place_of_the_sentinel(self):
        baseline, drawn = self._material()
        _audit, _gate, provenance = audit_and_gate_blocks(baseline, drawn["a2"], 0.3, ALL_FRESH,
                                                          300, np.random.default_rng(1))
        assert isinstance(provenance["audit_length"], int)
        assert provenance["audit_length"] > 0

    def test_the_audit_is_the_length_the_provenance_names(self):
        baseline, drawn = self._material()
        audit, _gate, provenance = audit_and_gate_blocks(baseline, drawn["a2"], 0.3, ALL_FRESH,
                                                         300, np.random.default_rng(1))
        assert len(audit["item_id"]) == provenance["audit_length"]

    def test_nothing_is_reused_because_freshness_is_the_bound(self):
        baseline, drawn = self._material()
        audit, _gate, provenance = audit_and_gate_blocks(baseline, drawn["a2"], 0.3, ALL_FRESH,
                                                         300, np.random.default_rng(1))
        assert provenance["audit_reused_rows"] == 0
        assert len(block_ids(audit)) == len(audit["item_id"])

    def test_the_composition_stops_where_the_first_pool_it_needed_ran_dry(self):
        baseline, drawn = self._material()
        audit, gate, provenance = audit_and_gate_blocks(baseline, drawn["a2"], 0.3, ALL_FRESH,
                                                        300, np.random.default_rng(1))
        reserved = block_ids(gate)
        served = np.asarray(audit["pool"])
        spent, available = {}, {}
        for name, pool in (("base", drawn["base"]), (ATTACK_COMPONENT, drawn["a2"])):
            spent[name] = int((served == name).sum())
            available[name] = len(set(np.asarray(pool["item_id"])) - reserved)
        assert all(spent[name] <= available[name] for name in spent)
        assert any(spent[name] == available[name] for name in spent), "no pool ran dry"

    def test_the_binding_pool_is_identifiable_from_the_provenance(self):
        baseline, drawn = self._material()
        audit, gate, provenance = audit_and_gate_blocks(baseline, drawn["a2"], 0.3, ALL_FRESH,
                                                        300, np.random.default_rng(1))
        attack_served = int(np.asarray(audit["is_drift_item"]).sum())
        gate_attack = int(np.asarray(gate["is_drift_item"]).sum())
        assert attack_served + gate_attack == provenance["eligible_attack"]

    def test_a_wider_base_pool_moves_the_bound_to_the_base_side(self):
        drawn = pools_of({"base": 400, "a2": 2000})
        baseline = baseline_with_pools({"base": 1.0}, drawn)
        audit, gate, _provenance = audit_and_gate_blocks(baseline, drawn["a2"], 0.1, ALL_FRESH,
                                                         300, np.random.default_rng(1))
        base_served = int((np.asarray(audit["pool"]) == "base").sum())
        reserved_base = len(block_ids(gate) & set(np.asarray(drawn["base"]["item_id"]).tolist()))
        assert base_served == 400 - reserved_base

    def test_the_same_seed_rate_and_consumed_set_compose_the_identical_audit(self):
        baseline, drawn = self._material()
        consumed = frozenset(np.asarray(drawn["base"]["item_id"])[:100])
        first = audit_and_gate_blocks(baseline, drawn["a2"], 0.3, ALL_FRESH, 300,
                                      np.random.default_rng([4, 4]), consumed_items=consumed)
        second = audit_and_gate_blocks(baseline, drawn["a2"], 0.3, ALL_FRESH, 300,
                                       np.random.default_rng([4, 4]), consumed_items=consumed)
        assert_columns_identical(first[0], second[0], "the audit block")
        assert first[2] == second[2]

    def test_the_ceiling_falls_as_the_probe_consumes(self):
        baseline, drawn = self._material(base=900, attack=400)
        fresh = audit_and_gate_blocks(baseline, drawn["a2"], 0.3, ALL_FRESH, 300,
                                      np.random.default_rng(2))[2]
        later = audit_and_gate_blocks(
            baseline, drawn["a2"], 0.3, ALL_FRESH, 300, np.random.default_rng(2),
            consumed_items=frozenset(np.asarray(drawn["a2"]["item_id"])[:60]))[2]
        assert later["audit_length"] < fresh["audit_length"]
        assert later["eligible_attack"] == fresh["eligible_attack"] - 60

    def test_no_consumed_item_reaches_the_audit_either_under_this_policy(self):
        baseline, drawn = self._material()
        consumed = frozenset(np.asarray(drawn["base"]["item_id"])[:400])
        audit, gate, _ = audit_and_gate_blocks(baseline, drawn["a2"], 0.3, ALL_FRESH, 300,
                                               np.random.default_rng(2),
                                               consumed_items=consumed)
        assert not (block_ids(audit) & consumed)
        assert not (block_ids(gate) & consumed)
        assert not (block_ids(audit) & block_ids(gate))

    def test_the_gate_block_is_the_one_the_bounded_policy_would_have_composed(self):
        baseline, drawn = self._material(base=4000, attack=1500)
        bounded = audit_and_gate_blocks(baseline, drawn["a2"], 0.15, 1200, 300,
                                        np.random.default_rng([9, 1]))
        until_dry = audit_and_gate_blocks(baseline, drawn["a2"], 0.15, ALL_FRESH, 300,
                                          np.random.default_rng([9, 1]))
        assert_columns_identical(bounded[1], until_dry[1], "the gate block")
        assert bounded[2]["gate_sha256"] == until_dry[2]["gate_sha256"]

    def test_the_provenance_still_names_exactly_the_nine_pinned_keys(self):
        baseline, drawn = self._material()
        provenance = audit_and_gate_blocks(baseline, drawn["a2"], 0.3, ALL_FRESH, 300,
                                           np.random.default_rng(1))[2]
        assert set(provenance) == {"rate", "audit_length", "gate_length", "eligible_attack",
                                   "eligible_base", "audit_reused_rows", "gate_reused_rows",
                                   "audit_sha256", "gate_sha256"}

    def test_a_consumed_world_refuses_at_the_gate_before_the_audit_policy_is_reached(self):
        baseline, drawn = self._material()
        consumed = frozenset(np.asarray(drawn["base"]["item_id"]).tolist()
                             + np.asarray(drawn["a2"]["item_id"]).tolist())
        with pytest.raises(GateUncomposable):
            audit_and_gate_blocks(baseline, drawn["a2"], 0.3, ALL_FRESH, 300,
                                  np.random.default_rng(1), consumed_items=consumed)

    def test_an_audit_with_no_fresh_item_for_its_first_position_refuses(self):
        baseline, drawn = self._material(base=900, attack=120)
        with pytest.raises(ValueError, match="First audit position needs pool"):
            audit_and_gate_blocks(
                baseline, drawn["a2"], 0.3, ALL_FRESH, 300, np.random.default_rng(2),
                consumed_items=frozenset(np.asarray(drawn["a2"]["item_id"])[:60]))

    def test_the_sentinel_is_not_an_integer_a_caller_could_pass_by_accident(self):
        assert not isinstance(ALL_FRESH, int)


class TestTheChainKeepsTheBoundedLadder:
    def test_an_integer_ladder_passes(self):
        assert assert_bounded_ladder(3) == 3

    def test_the_data_bounded_policy_is_refused_with_its_reason(self):
        with pytest.raises(ValueError, match="exhaust-fresh-data"):
            assert_bounded_ladder(EXHAUST_FRESH_DATA)

    def test_the_refusal_says_why_a_chain_cannot_afford_it(self):
        with pytest.raises(ValueError, match="unsupported for chained campaigns"):
            assert_bounded_ladder(EXHAUST_FRESH_DATA)

    def test_a_negative_ladder_refuses(self):
        with pytest.raises(ValueError, match="retries"):
            assert_bounded_ladder(-1)
