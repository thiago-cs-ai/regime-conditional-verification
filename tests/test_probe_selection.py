import inspect
import json

import numpy as np
import pytest
import yaml

from fixture_frame import build_fixture_frame, split_for_fitting
from rcv.estimator import Estimator
from rcv.frames import Slice, rows, slice_of
from rcv.probes import (
    PROBE_FAMILIES,
    SELECTABLE_PARAMETERS,
    LinearProbe,
    MLPProbe,
    SelectableParameter,
    build_probe,
    unchanged,
)
from test_v1_skeleton import write_frame, write_v1_config

DEPLOYED_GRID = [1e-4, 1e-3, 1e-2, 1e-1]
MLP_GRID = {"alpha": [1.0e-5, 1.0], "hidden_layer_sizes": [[4], [16]]}



REPRESENTATION_WIDTH = 16
DENSE_WEAK_SIGNAL = 0.25
SPARSE_STRONG_SIGNAL = 0.9
FITTING_ROWS = (24, 240)
FOLD_ROWS = (200, 200)


def _dense_weak_regime(rng, n, verdict_value):
    agreement = np.repeat([0, 1], n // 2)
    representation = rng.normal(size=(n, REPRESENTATION_WIDTH))
    representation += DENSE_WEAK_SIGNAL * (2 * agreement - 1)[:, None]
    return representation, agreement, np.full(n, verdict_value)


def _sparse_strong_regime(rng, n, verdict_value):
    agreement = np.repeat([0, 1], n // 2)
    representation = rng.normal(size=(n, REPRESENTATION_WIDTH))
    representation[:, 0] += SPARSE_STRONG_SIGNAL * (2 * agreement - 1)
    return representation, agreement, np.full(n, verdict_value)


def _two_regime_slice(rng, rows_per_regime):
    dense = _dense_weak_regime(rng, rows_per_regime[0], 0)
    sparse = _sparse_strong_regime(rng, rows_per_regime[1], 1)
    return Slice(np.vstack([dense[0], sparse[0]]),
                 np.concatenate([dense[2], sparse[2]]),
                 np.concatenate([dense[1], sparse[1]]))


@pytest.fixture(scope="module")
def two_regime_task():
    rng = np.random.default_rng(5)
    fitting = _two_regime_slice(rng, FITTING_ROWS)
    calibration = _two_regime_slice(rng, FOLD_ROWS)
    return fitting, calibration


def _fitted(spec, task, route_probe=True, random_state=0):
    fitting, calibration = task
    estimator = Estimator(probe=spec, calibration={"method": "platt"},
                          route_probe=route_probe, route_calibration=route_probe,
                          random_state=random_state)
    return estimator.fit(fitting_slice=fitting, calibration_slice=calibration)


def _selected(estimator):
    return {row["group"]: row["value"] for row in estimator.probe_selection()}


def _selected_pairs(estimator):
    pairs = {}
    for row in estimator.probe_selection():
        pairs.setdefault(row["group"], {})[row["parameter"]] = row["value"]
    return pairs




class TestTheConfigurationSurface:
    def test_naming_both_the_value_and_its_grid_is_refused_rather_than_resolved(self):
        with pytest.raises(ValueError, match=r"names 'C' directly and in its 'select' block"):
            build_probe({"family": "linear", "C": 1.0, "select": {"C": DEPLOYED_GRID}})

    def test_the_refusal_reaches_the_configuration_gate(self):
        from rcv.study import validate_config

        config = _config_with_probe({"family": "linear", "C": 1.0,
                                     "select": {"C": DEPLOYED_GRID}})
        with pytest.raises(ValueError, match=r"names 'C' directly and in its 'select' block"):
            validate_config(config)

    def test_an_empty_grid_is_refused(self):
        with pytest.raises(ValueError, match=r"grid for 'C' is empty"):
            build_probe({"family": "linear", "select": {"C": []}})

    def test_a_single_candidate_grid_is_accepted_and_is_that_value(self, two_regime_task):
        estimator = _fitted({"family": "linear", "select": {"C": [0.01]}}, two_regime_task)

        assert _selected(estimator) == {0: 0.01, 1: 0.01}

    def test_a_single_candidate_grid_fits_the_probe_the_fixed_value_would_have(self,
                                                                              two_regime_task):
        fitting, calibration = two_regime_task
        selected = _fitted({"family": "linear", "select": {"C": [0.01]}}, two_regime_task)
        fixed = _fitted({"family": "linear", "C": 0.01}, two_regime_task)

        np.testing.assert_array_equal(
            selected.probe_scores(calibration.representation, calibration.verdict),
            fixed.probe_scores(calibration.representation, calibration.verdict))

    def test_a_repeated_candidate_is_refused(self):
        with pytest.raises(ValueError, match=r"repeats"):
            build_probe({"family": "linear", "select": {"C": [0.01, 0.001, 0.01]}})

    def test_a_grid_that_is_not_a_list_of_candidates_is_refused(self):
        with pytest.raises(ValueError, match=r"grid for 'C' must be a list"):
            build_probe({"family": "linear", "select": {"C": 0.01}})

    def test_a_select_block_that_is_not_a_mapping_is_refused(self):
        with pytest.raises(ValueError, match=r"'select' block must be a mapping"):
            build_probe({"family": "linear", "select": [0.01, 0.1]})

    def test_selecting_over_a_second_parameter_the_entry_does_not_authorise_is_refused(self):
        with pytest.raises(ValueError, match=r"names unsupported parameter 'max_iter'"):
            build_probe({"family": "linear",
                         "select": {"C": DEPLOYED_GRID, "max_iter": [100, 1000]}})

    def test_selecting_over_a_parameter_whose_weakest_direction_is_unnamed_is_refused(self):
        with pytest.raises(ValueError,
                           match=r"Selectable parameters for 'linear': 'C'"):
            build_probe({"family": "linear", "select": {"max_iter": [100, 1000]}})

    def test_an_unknown_key_beside_a_grid_still_refuses(self):
        with pytest.raises(TypeError):
            build_probe({"family": "linear", "banana": 1, "select": {"C": DEPLOYED_GRID}})


def _config_with_probe(probe_spec):
    from test_prefix_drift_and_config_strictness import _floor_config

    config = _floor_config()
    config["estimator"]["probe"] = probe_spec
    return config




class TestSelectionIsPerRegime:
    def test_two_regimes_may_choose_differently(self, two_regime_task):
        estimator = _fitted({"family": "linear", "select": {"C": DEPLOYED_GRID}},
                            two_regime_task)
        selected = _selected(estimator)

        assert selected[0] != selected[1]
        assert set(selected) == {0, 1}

    def test_each_group_fits_the_probe_its_own_selection_names(self, two_regime_task):
        fitting, calibration = two_regime_task
        estimator = _fitted({"family": "linear", "select": {"C": DEPLOYED_GRID}},
                            two_regime_task)
        selected = _selected(estimator)

        for regime, value in selected.items():
            group = fitting.verdict == regime
            reference = LinearProbe(random_state=0, C=value).fit(
                fitting.representation[group], fitting.agreement[group])
            held_out = calibration.verdict == regime
            np.testing.assert_array_equal(
                estimator._probe_by_group[regime].logit_scores(
                    calibration.representation[held_out]),
                reference.logit_scores(calibration.representation[held_out]))

    def test_the_seed_the_run_names_reaches_every_candidate_the_selection_fits(self,
                                                                              two_regime_task):
        estimator = _fitted({"family": "linear", "select": {"C": DEPLOYED_GRID}},
                            two_regime_task, random_state=7)

        for probe in estimator._probe_by_group.values():
            assert probe._probe._pipeline.named_steps["logistic"].random_state == 7

    def test_a_pooled_probe_selects_once_over_every_row(self, two_regime_task):
        estimator = _fitted({"family": "linear", "select": {"C": DEPLOYED_GRID}},
                            two_regime_task, route_probe=False)

        assert list(_selected(estimator)) == ["pooled"]

    def test_the_selection_groups_are_the_probes_and_not_the_calibrations(self, two_regime_task):
        fitting, calibration = two_regime_task
        estimator = Estimator(probe={"family": "linear", "select": {"C": DEPLOYED_GRID}},
                              calibration={"method": "platt"},
                              route_probe=False, route_calibration=True, random_state=0)
        estimator.fit(fitting_slice=fitting, calibration_slice=calibration)

        assert list(_selected(estimator)) == ["pooled"]




class TestTheFoldIsTheCalibrationSlice:
    def test_the_estimator_is_handed_no_evaluation_slice(self):
        assert list(inspect.signature(Estimator.fit).parameters) == [
            "self", "fitting_slice", "calibration_slice"]

    def test_holding_the_fitting_slice_and_moving_the_fold_moves_the_selection(self):
        rng = np.random.default_rng(5)
        fitting = _two_regime_slice(rng, FITTING_ROWS)
        honest_fold = _two_regime_slice(rng, FOLD_ROWS)
        misleading = _two_regime_slice(rng, FOLD_ROWS)
        regime_one = misleading.verdict == 1
        shuffled = misleading.representation.copy()
        shuffled[regime_one] = rng.normal(size=shuffled[regime_one].shape)

        honest = _fitted({"family": "linear", "select": {"C": DEPLOYED_GRID}},
                         (fitting, honest_fold))
        moved = _fitted({"family": "linear", "select": {"C": DEPLOYED_GRID}},
                        (fitting, Slice(shuffled, misleading.verdict, misleading.agreement)))

        assert _selected(honest) != _selected(moved)

    def test_a_study_selection_is_reproduced_from_the_fitting_and_calibration_slices_alone(
            self, tmp_path):
        from rcv.study import fitting_slices_of, run_study

        config = _study_config(tmp_path, {"family": "linear", "select": {"C": DEPLOYED_GRID}})
        result = run_study(yaml.safe_load(config.read_text()), 0)

        frame = build_fixture_frame()
        serving = rows(frame, ~frame["is_stream"])
        split = yaml.safe_load(config.read_text())["split"]
        training, calibration, _evaluation = fitting_slices_of(serving, split, "oracle", 0)
        estimator = Estimator(probe={"family": "linear", "select": {"C": DEPLOYED_GRID}},
                              calibration={"method": "platt"}, route_probe=True,
                              route_calibration=True, random_state=0)
        estimator.fit(fitting_slice=slice_of(training, "oracle"),
                      calibration_slice=slice_of(calibration, "oracle"))

        assert result.probe_selection == estimator.probe_selection()




class TestTheTieBreak:
    def test_an_exact_tie_selects_the_largest_candidate(self, one_dimensional_task):
        estimator = _fitted({"family": "linear", "select": {"C": DEPLOYED_GRID}},
                            one_dimensional_task)

        assert set(_selected(estimator).values()) == {max(DEPLOYED_GRID)}

    def test_the_tie_test_is_exact_equality_and_a_discriminating_fold_is_not_overridden(
            self, two_regime_task):
        estimator = _fitted({"family": "linear", "select": {"C": DEPLOYED_GRID}},
                            two_regime_task)
        probe = estimator._probe_by_group[0]

        best = max(probe.held_out_auroc.values())
        winners = [candidate for candidate, auroc in probe.held_out_auroc.items()
                   if auroc == best]
        assert winners == [(probe.selection[0]["value"],)]
        assert probe.selection[0]["value"] != max(DEPLOYED_GRID)

    def test_the_grid_order_the_run_writes_does_not_decide_a_tie(self, one_dimensional_task):
        ascending = _fitted({"family": "linear", "select": {"C": DEPLOYED_GRID}},
                            one_dimensional_task)
        descending = _fitted({"family": "linear",
                              "select": {"C": list(reversed(DEPLOYED_GRID))}},
                             one_dimensional_task)

        assert _selected(ascending) == _selected(descending)


@pytest.fixture(scope="module")
def one_dimensional_task():
    rng = np.random.default_rng(11)

    def draw(n):
        verdict = np.repeat([0, 1], n // 2)
        agreement = rng.integers(0, 2, size=n)
        score = rng.normal(size=(n, 1)) + (2 * agreement - 1)[:, None]
        return Slice(score, verdict, agreement)

    return draw(600), draw(600)




class TestTheOneDimensionalArm:
    def test_every_candidate_scores_the_fold_identically(self, one_dimensional_task):
        estimator = _fitted({"family": "linear", "select": {"C": DEPLOYED_GRID}},
                            one_dimensional_task)

        for probe in estimator._probe_by_group.values():
            assert len(set(probe.held_out_auroc.values())) == 1

    def test_a_score_source_study_selects_the_largest_candidate_and_the_calibration_stands(
            self, tmp_path):
        from rcv.study import run_study

        config = yaml.safe_load(
            _study_config(tmp_path, {"family": "linear",
                                     "select": {"C": DEPLOYED_GRID}}).read_text())
        config["estimator"]["feature_source"] = "clf_score"

        result = run_study(config, 0)

        assert {row["value"] for row in result.probe_selection} == {max(DEPLOYED_GRID)}

    def test_the_selected_probe_ranks_the_arm_exactly_as_the_deployed_one_does(
            self, one_dimensional_task):
        fitting, calibration = one_dimensional_task
        selected = _fitted({"family": "linear", "select": {"C": DEPLOYED_GRID}},
                           one_dimensional_task)
        deployed = _fitted({"family": "linear", "C": 1.0}, one_dimensional_task)

        for regime in (0, 1):
            group = calibration.verdict == regime
            selected_scores = selected._probe_by_group[regime].logit_scores(
                calibration.representation[group])
            deployed_scores = deployed._probe_by_group[regime].logit_scores(
                calibration.representation[group])
            np.testing.assert_array_equal(np.argsort(selected_scores, kind="stable"),
                                          np.argsort(deployed_scores, kind="stable"))




def _study_config(tmp_path, probe_spec, study="probe-selection"):
    frame_path = write_frame(tmp_path, build_fixture_frame())
    path = write_v1_config(tmp_path, frame_path)
    config = yaml.safe_load(path.read_text())
    config["study"] = study
    config["estimator"]["probe"] = probe_spec
    config["split"] = {"evaluation_fraction": 0.2, "calibration_fraction": 0.4375}
    path.write_text(yaml.safe_dump(config))
    return path


class TestTheRecordNamesWhatItFitted:
    def test_the_selected_value_per_group_reaches_the_written_record(self, tmp_path):
        from rcv.runner import run_protocol

        config = yaml.safe_load(
            _study_config(tmp_path, {"family": "linear",
                                     "select": {"C": DEPLOYED_GRID}}).read_text())
        run_protocol(config, tmp_path / "out")
        record = json.loads((tmp_path / "out" / "seed_0.json").read_text())

        assert record["probe_selection"] == [
            {"group": 0, "parameter": "C", "value": pytest.approx(record["probe_selection"][0]
                                                                  ["value"])},
            {"group": 1, "parameter": "C", "value": pytest.approx(record["probe_selection"][1]
                                                                  ["value"])}]
        assert {row["value"] for row in record["probe_selection"]} <= set(DEPLOYED_GRID)

    def test_the_record_survives_a_round_trip_through_json(self, tmp_path):
        from rcv.runner import run_protocol

        config = yaml.safe_load(
            _study_config(tmp_path, {"family": "linear",
                                     "select": {"C": DEPLOYED_GRID}}).read_text())
        run_protocol(config, tmp_path / "out")
        record = json.loads((tmp_path / "out" / "seed_0.json").read_text())

        assert [row["group"] for row in record["probe_selection"]] == [0, 1]
        assert {row["parameter"] for row in record["probe_selection"]} == {"C"}

    def test_a_configuration_naming_no_grid_writes_no_selection_field(self, tmp_path):
        from rcv.runner import run_protocol

        config = yaml.safe_load(
            _study_config(tmp_path, {"family": "linear", "C": 1.0}).read_text())
        run_protocol(config, tmp_path / "out")
        record = json.loads((tmp_path / "out" / "seed_0.json").read_text())

        assert "probe_selection" not in record
        assert all("probe_selection" not in entry for entry in record["change_log"])

    def test_a_repair_names_the_value_its_own_refit_selected(self, tmp_path):
        from rcv.study import run_study

        config = yaml.safe_load(
            _study_config(tmp_path, {"family": "linear",
                                     "select": {"C": DEPLOYED_GRID}}).read_text())
        result = run_study(config, 0)
        repairs = [entry for entry in result.change_log if entry.what == "repair"]

        assert repairs, "the fixture study runs one full repair cycle"
        for entry in repairs:
            assert [row["parameter"] for row in entry.probe_selection] == ["C", "C"]




class TestASpecificationNamingNoGridIsUnmoved:
    def test_build_probe_returns_the_bare_family(self):
        assert type(build_probe({"family": "linear", "C": 1.0})) is LinearProbe

    def test_no_shipped_family_asks_for_a_fold(self):
        assert {family.needs_validation for family in PROBE_FAMILIES.values()} == {False}

    def test_every_shipped_family_declares_that_it_selected_nothing(self):
        for family in PROBE_FAMILIES.values():
            assert family.selection is None

    def test_a_family_that_asks_for_no_fold_is_fitted_with_the_fitting_slice_alone(self):
        assert list(inspect.signature(LinearProbe.fit).parameters) == [
            "self", "representation", "agreement"]

    def test_the_deployed_estimator_reproduces_its_pre_selection_values(self):
        frame = build_fixture_frame()
        serving = rows(frame, ~frame["is_stream"])
        fitting, calibration = split_for_fitting(serving)
        estimator = Estimator(probe={"family": "linear", "C": 1.0},
                              calibration={"method": "platt"}, route_probe=True,
                              route_calibration=True, random_state=0)
        estimator.fit(slice_of(fitting, "oracle"), slice_of(calibration, "oracle"))

        probability = estimator.probability_of_agreement(calibration["representation"],
                                                         calibration["verdict"])
        scores = estimator.probe_scores(calibration["representation"], calibration["verdict"])

        assert len(probability) == 220
        np.testing.assert_allclose(
            probability[:8],
            [0.9983797579294162, 0.9999753714924664, 0.9999831229887056,
             0.9998984289816444, 0.9998923954792089, 0.9581553314534567,
             2.1008843888952444e-05, 0.9980906682660545],
            rtol=1e-14,
            atol=0.0,
        )
        np.testing.assert_allclose(probability.sum(), 191.00079460282578, rtol=1e-14, atol=0.0)
        np.testing.assert_allclose(scores.sum(), 1261.3259535003945, rtol=1e-14, atol=0.0)

    def test_the_estimator_reports_no_selection_where_none_was_made(self):
        frame = build_fixture_frame()
        serving = rows(frame, ~frame["is_stream"])
        fitting, calibration = split_for_fitting(serving)
        estimator = Estimator(probe={"family": "linear", "C": 1.0},
                              calibration={"method": "platt"}, route_probe=True,
                              route_calibration=True, random_state=0)
        estimator.fit(slice_of(fitting, "oracle"), slice_of(calibration, "oracle"))

        assert estimator.probe_selection() is None




MLP_MAX_ITER = 80
MLP_FITTING_ROWS = (160, 240)


@pytest.fixture(scope="module")
def two_regime_mlp_task():
    rng = np.random.default_rng(5)
    return (_two_regime_slice(rng, MLP_FITTING_ROWS), _two_regime_slice(rng, FOLD_ROWS))


@pytest.fixture(scope="module")
def mlp_selection(two_regime_mlp_task):
    return _fitted({"family": "mlp", "max_iter": MLP_MAX_ITER, "select": MLP_GRID},
                   two_regime_mlp_task)


class TestTheJointGridIsOneGrid:
    def test_every_pair_in_the_cross_product_is_a_candidate(self, mlp_selection):
        for probe in mlp_selection._probe_by_group.values():
            assert set(probe.held_out_auroc) == {
                (alpha, tuple(hidden)) for alpha in MLP_GRID["alpha"]
                for hidden in MLP_GRID["hidden_layer_sizes"]}

    def test_each_group_selects_its_own_pair_from_the_joint_grid(self, mlp_selection):
        pairs = _selected_pairs(mlp_selection)

        assert set(pairs) == {0, 1}
        for pair in pairs.values():
            assert list(pair) == ["alpha", "hidden_layer_sizes"]
            assert pair["alpha"] in MLP_GRID["alpha"]
            assert pair["hidden_layer_sizes"] in MLP_GRID["hidden_layer_sizes"]

    def test_each_group_fits_the_probe_its_own_pair_names(self, two_regime_mlp_task,
                                                          mlp_selection):
        fitting, calibration = two_regime_mlp_task

        for regime, pair in _selected_pairs(mlp_selection).items():
            group = fitting.verdict == regime
            reference = MLPProbe(random_state=0, max_iter=MLP_MAX_ITER, **pair).fit(
                fitting.representation[group], fitting.agreement[group])
            held_out = calibration.verdict == regime
            np.testing.assert_array_equal(
                mlp_selection._probe_by_group[regime].logit_scores(
                    calibration.representation[held_out]),
                reference.logit_scores(calibration.representation[held_out]))

    def test_the_two_regimes_may_choose_different_pairs(self, mlp_selection):
        pairs = _selected_pairs(mlp_selection)

        assert pairs[0] != pairs[1]

    def test_a_pooled_probe_selects_one_pair_over_every_row(self, two_regime_mlp_task):
        estimator = _fitted({"family": "mlp", "max_iter": MLP_MAX_ITER, "select": MLP_GRID},
                            two_regime_mlp_task, route_probe=False)

        assert list(_selected_pairs(estimator)) == ["pooled"]

    def test_a_single_entry_joint_grid_fits_the_probe_the_fixed_pair_would_have(
            self, two_regime_mlp_task):
        _fitting, calibration = two_regime_mlp_task
        pinned = {"alpha": [1.0], "hidden_layer_sizes": [[4]]}
        selected = _fitted({"family": "mlp", "max_iter": MLP_MAX_ITER, "select": pinned},
                           two_regime_mlp_task)
        fixed = _fitted({"family": "mlp", "max_iter": MLP_MAX_ITER, "alpha": 1.0,
                         "hidden_layer_sizes": [4]}, two_regime_mlp_task)

        np.testing.assert_array_equal(
            selected.probe_scores(calibration.representation, calibration.verdict),
            fixed.probe_scores(calibration.representation, calibration.verdict))

    def test_the_seed_the_run_names_reaches_every_candidate_the_joint_selection_fits(
            self, two_regime_mlp_task):
        estimator = _fitted({"family": "mlp", "max_iter": MLP_MAX_ITER, "select": MLP_GRID},
                            two_regime_mlp_task, random_state=7)

        for probe in estimator._probe_by_group.values():
            assert probe._probe._pipeline.named_steps["mlp"].random_state == 7




class _ScriptedProbe:
    scores_are_continuous = True
    needs_validation = False
    selection = None
    ranking_by_candidate = {}

    def __init__(self, random_state=None, penalty=None, width=()):
        self._candidate = (penalty, tuple(width))

    def fit(self, representation, agreement):
        return self

    def logit_scores(self, representation):
        return np.asarray(self.ranking_by_candidate[self._candidate], dtype=np.float64)


SCRIPTED_FOLD_AGREEMENT = np.array([0, 0, 1, 1])
RANKS_THE_FOLD_PERFECTLY = [0.0, 1.0, 2.0, 3.0]
RANKS_THE_FOLD_BACKWARDS = [3.0, 2.0, 1.0, 0.0]
SCRIPTED_GRID = {"penalty": [1e-5, 1e-3], "width": [[16], [64]]}


def _scripted_selection(monkeypatch, ranking, grid=None):
    monkeypatch.setitem(PROBE_FAMILIES, "scripted", _ScriptedProbe)
    monkeypatch.setitem(SELECTABLE_PARAMETERS, "scripted",
                        {"penalty": SelectableParameter(min, unchanged, unchanged),
                         "width": SelectableParameter(max, tuple, list)})
    monkeypatch.setattr(_ScriptedProbe, "ranking_by_candidate", ranking)
    probe = build_probe({"family": "scripted", "select": grid or SCRIPTED_GRID})
    material = np.zeros((4, 1))
    probe.fit(material, SCRIPTED_FOLD_AGREEMENT, material, SCRIPTED_FOLD_AGREEMENT)
    return {row["parameter"]: row["value"] for row in probe.selection}


def _script(winners):
    return {(penalty, tuple(width)): (RANKS_THE_FOLD_PERFECTLY
                                      if (penalty, tuple(width)) in winners
                                      else RANKS_THE_FOLD_BACKWARDS)
            for penalty in SCRIPTED_GRID["penalty"]
            for width in SCRIPTED_GRID["width"]}


class TestTheLexicographicTieBreak:
    def test_a_tie_on_the_penalty_axis_alone_takes_the_smallest_penalty(self, monkeypatch):
        selected = _scripted_selection(
            monkeypatch, _script({(1e-5, (16,)), (1e-3, (16,))}))

        assert selected == {"penalty": 1e-5, "width": [16]}

    def test_a_tie_on_the_width_axis_alone_takes_the_largest_width(self, monkeypatch):
        selected = _scripted_selection(
            monkeypatch, _script({(1e-3, (16,)), (1e-3, (64,))}))

        assert selected == {"penalty": 1e-3, "width": [64]}

    def test_a_tie_on_both_axes_at_once_is_decided_by_the_penalty_first(self, monkeypatch):
        selected = _scripted_selection(
            monkeypatch, _script({(1e-5, (16,)), (1e-5, (64,)),
                                  (1e-3, (16,)), (1e-3, (64,))}))

        assert selected == {"penalty": 1e-5, "width": [64]}

    def test_a_strict_winner_is_never_overridden_by_the_tie_break(self, monkeypatch):
        selected = _scripted_selection(monkeypatch, _script({(1e-3, (16,))}))

        assert selected == {"penalty": 1e-3, "width": [16]}

    def test_the_grid_order_the_run_writes_does_not_decide_a_tie(self, monkeypatch):
        script = _script({(1e-5, (16,)), (1e-3, (64,))})
        ascending = _scripted_selection(monkeypatch, script)
        descending = _scripted_selection(monkeypatch, script,
                                         grid={"penalty": [1e-3, 1e-5],
                                               "width": [[64], [16]]})

        assert ascending == descending == {"penalty": 1e-5, "width": [16]}

    def test_a_selecting_probe_reports_the_continuity_of_the_family_it_wraps(self):
        for spec in ({"family": "linear", "select": {"C": DEPLOYED_GRID}},
                     {"family": "mlp", "select": MLP_GRID}):
            wrapped = PROBE_FAMILIES[spec["family"]].scores_are_continuous
            assert build_probe(spec).scores_are_continuous is wrapped is True

    def test_the_direction_is_attached_to_the_family_and_parameter_not_to_the_position(self):
        assert SELECTABLE_PARAMETERS["linear"]["C"].weakest_regularisation is max
        assert SELECTABLE_PARAMETERS["mlp"]["alpha"].weakest_regularisation is min
        assert SELECTABLE_PARAMETERS["mlp"]["hidden_layer_sizes"].weakest_regularisation is max
        assert list(SELECTABLE_PARAMETERS["mlp"]) == ["alpha", "hidden_layer_sizes"]




class TestWhatNoSignedEntryAuthorisesIsRefused:
    def test_selection_on_a_tree_parameter_is_refused(self):
        with pytest.raises(ValueError,
                           match=r"No selectable parameters are registered for 'gradient_boosted'"):
            build_probe({"family": "gradient_boosted",
                         "select": {"l2_regularization": [0.0, 0.1, 1.0]}})

    def test_the_tree_refusal_reaches_the_configuration_gate(self):
        from rcv.study import validate_config

        config = _config_with_probe({"family": "gradient_boosted",
                                     "select": {"l2_regularization": [0.0, 1.0]}})
        with pytest.raises(ValueError, match=r"names unsupported parameter 'l2_regularization'"):
            validate_config(config)

    def test_selecting_over_the_penalty_alone_is_refused(self):
        with pytest.raises(ValueError, match=r"must name exactly"):
            build_probe({"family": "mlp", "select": {"alpha": [1e-5, 1.0]}})

    def test_selecting_over_the_width_alone_is_refused(self):
        with pytest.raises(ValueError, match=r"must name exactly"):
            build_probe({"family": "mlp", "select": {"hidden_layer_sizes": [[16], [64]]}})

    def test_the_partial_pair_refusal_reaches_the_configuration_gate(self):
        from rcv.study import validate_config

        config = _config_with_probe({"family": "mlp", "select": {"alpha": [1e-5, 1.0]}})
        with pytest.raises(ValueError, match=r"must name exactly"):
            validate_config(config)

    def test_a_third_parameter_beside_the_authorised_pair_is_refused(self):
        with pytest.raises(ValueError,
                           match=r"Selectable parameters for 'mlp': 'alpha', 'hidden_layer_sizes'"):
            build_probe({"family": "mlp",
                         "select": {**MLP_GRID, "max_iter": [100, 500]}})

    def test_naming_one_half_of_the_pair_directly_and_in_the_grid_is_refused(self):
        with pytest.raises(ValueError, match=r"names 'alpha' directly and in its 'select' block"):
            build_probe({"family": "mlp", "alpha": 1e-4, "select": MLP_GRID})

    def test_naming_the_other_half_directly_and_in_the_grid_is_refused(self):
        with pytest.raises(ValueError,
                           match=r"names 'hidden_layer_sizes' directly and in its 'select' block"):
            build_probe({"family": "mlp", "hidden_layer_sizes": [16], "select": MLP_GRID})

    def test_an_empty_grid_on_one_axis_of_the_pair_is_refused(self):
        with pytest.raises(ValueError, match=r"grid for 'alpha' is empty"):
            build_probe({"family": "mlp",
                         "select": {"alpha": [], "hidden_layer_sizes": [[16], [64]]}})

    def test_a_repeated_width_is_refused(self):
        with pytest.raises(ValueError, match=r"repeats"):
            build_probe({"family": "mlp",
                         "select": {"alpha": [1e-5, 1.0],
                                    "hidden_layer_sizes": [[16], [64], [16]]}})

    def test_a_width_grid_that_is_not_a_list_of_candidates_is_refused(self):
        with pytest.raises(ValueError, match=r"grid for 'hidden_layer_sizes' must be a list"):
            build_probe({"family": "mlp",
                         "select": {"alpha": [1e-5, 1.0], "hidden_layer_sizes": 16}})

    def test_a_width_candidate_that_is_not_a_width_is_named_rather_than_left_to_a_typeerror(self):
        with pytest.raises(
                ValueError,
                match=r"grid for 'mlp'\.hidden_layer_sizes has an invalid candidate"):
            build_probe({"family": "mlp",
                         "select": {"alpha": [1e-5, 1.0], "hidden_layer_sizes": [16, 64]}})

    def test_an_empty_select_block_is_refused(self):
        with pytest.raises(ValueError, match=r"must name exactly"):
            build_probe({"family": "mlp", "select": {}})




@pytest.fixture(scope="module")
def mlp_study_record(tmp_path_factory):
    from rcv.runner import run_protocol

    workspace = tmp_path_factory.mktemp("mlp-selection")
    config = yaml.safe_load(
        _study_config(workspace, {"family": "mlp", "max_iter": MLP_MAX_ITER,
                                  "select": MLP_GRID}).read_text())
    run_protocol(config, workspace / "out")
    return json.loads((workspace / "out" / "seed_0.json").read_text())


class TestTheRecordNamesBothHalvesOfThePair:
    def test_the_selected_pair_per_group_reaches_the_written_record(self, mlp_study_record):
        assert [(row["group"], row["parameter"])
                for row in mlp_study_record["probe_selection"]] == [
            (0, "alpha"), (0, "hidden_layer_sizes"),
            (1, "alpha"), (1, "hidden_layer_sizes")]

    def test_every_recorded_value_is_one_the_grid_offered(self, mlp_study_record):
        for row in mlp_study_record["probe_selection"]:
            assert row["value"] in MLP_GRID[row["parameter"]]

    def test_the_width_is_written_as_the_list_it_came_in_as(self, mlp_study_record):
        widths = [row["value"] for row in mlp_study_record["probe_selection"]
                  if row["parameter"] == "hidden_layer_sizes"]

        assert widths and all(isinstance(width, list) for width in widths)

    def test_a_repair_names_the_pair_its_own_refit_selected(self, mlp_study_record):
        repairs = [entry for entry in mlp_study_record["change_log"]
                   if entry["what"] == "repair"]

        assert repairs, "the fixture study runs one full repair cycle"
        for entry in repairs:
            assert [row["parameter"] for row in entry["probe_selection"]] == [
                "alpha", "hidden_layer_sizes", "alpha", "hidden_layer_sizes"]
