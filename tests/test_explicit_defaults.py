import warnings

import numpy as np
import pytest

from rcv.calibrations import PlattCalibration, build_calibration
from rcv.estimator import Estimator
from rcv.frames import Slice
from rcv.probes import GradientBoostedProbe, LinearProbe, MLPProbe, build_probe
from rcv.splitting import draw_group_blocked_split


def test_linear_probe_default_configuration():
    pipeline = LinearProbe()._pipeline
    expected = {
        "C": 1.0, "max_iter": 1000, "random_state": None, "l1_ratio": 0.0,
        "dual": False, "tol": 1e-4, "fit_intercept": True, "intercept_scaling": 1,
        "class_weight": None, "solver": "lbfgs",
    }
    params = pipeline.named_steps["logistic"].get_params()
    assert {name: params[name] for name in expected} == expected
    scaler = pipeline.named_steps["scale_invariance"]
    assert scaler.with_mean is True and scaler.with_std is True


def test_mlp_probe_default_configuration():
    pipeline = MLPProbe()._pipeline
    expected = {
        "hidden_layer_sizes": (16,), "max_iter": 500, "random_state": None,
        "activation": "relu", "solver": "adam", "alpha": 1e-4, "batch_size": "auto",
        "learning_rate": "constant", "learning_rate_init": 1e-3, "power_t": 0.5,
        "shuffle": True, "tol": 1e-4, "momentum": 0.9, "nesterovs_momentum": True,
        "early_stopping": False, "validation_fraction": 0.1, "beta_1": 0.9,
        "beta_2": 0.999, "epsilon": 1e-8, "n_iter_no_change": 10, "max_fun": 15000,
    }
    params = pipeline.named_steps["mlp"].get_params()
    assert {name: params[name] for name in expected} == expected
    scaler = pipeline.named_steps["scale_invariance"]
    assert scaler.with_mean is True and scaler.with_std is True


def test_gradient_boosted_probe_default_configuration():
    expected = {
        "max_iter": 100, "random_state": None, "loss": "log_loss", "learning_rate": 0.1,
        "max_leaf_nodes": 31, "max_depth": None, "min_samples_leaf": 20,
        "l2_regularization": 0.0, "max_features": 1.0, "max_bins": 255,
        "categorical_features": "from_dtype", "monotonic_cst": None,
        "interaction_cst": None, "early_stopping": False, "scoring": "loss",
        "validation_fraction": 0.1, "n_iter_no_change": 10, "tol": 1e-7,
        "class_weight": None,
    }
    params = GradientBoostedProbe()._model.get_params()
    assert {name: params[name] for name in expected} == expected


def test_platt_calibration_default_configuration():
    expected = {
        "C": 1.0, "max_iter": 100, "l1_ratio": 0.0, "dual": False, "tol": 1e-4,
        "fit_intercept": True, "intercept_scaling": 1, "class_weight": None,
        "solver": "lbfgs", "random_state": None,
    }
    params = PlattCalibration()._logistic.get_params()
    assert {name: params[name] for name in expected} == expected


def material(dimension=4, per_class=70, seed=11):
    rng = np.random.default_rng(seed)
    representation = np.vstack([rng.normal(-0.8, 1.0, size=(per_class, dimension)),
                                rng.normal(+0.8, 1.0, size=(per_class, dimension))])
    return representation, np.repeat([0, 1], per_class)


def four_case_slice(rows=160, dimension=4, seed=3):
    rng = np.random.default_rng(seed)
    verdict = np.tile([0, 1], rows // 2)
    agreement = np.tile([0, 0, 1, 1], rows // 4)
    representation = rng.normal(size=(rows, dimension)) + agreement[:, None]
    return Slice(representation, verdict, agreement)


class TestTheConstructorsRemainCompatible:
    @pytest.mark.parametrize("build", (lambda: LinearProbe(random_state=0),
                                       lambda: PlattCalibration()))
    def test_no_component_raises_a_deprecation_by_naming_a_retired_parameter(self, build):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            build()
        retired = [str(entry.message) for entry in caught
                   if issubclass(entry.category, (FutureWarning, DeprecationWarning))]
        assert not retired, f"a restatement named a retired parameter: {retired}"


class TestConfigurationReachesComponents:
    def test_the_linear_probes_new_parameters_arrive_at_the_model(self):
        model = build_probe({"family": "linear", "tol": 0.5, "fit_intercept": False,
                             "solver": "liblinear", "dual": False, "class_weight": "balanced",
                             "intercept_scaling": 3, "with_mean": False},
                            random_state=0)._pipeline
        logistic = model.named_steps["logistic"]
        assert (logistic.tol, logistic.fit_intercept, logistic.solver) == (0.5, False,
                                                                          "liblinear")
        assert (logistic.class_weight, logistic.intercept_scaling) == ("balanced", 3)
        assert model.named_steps["scale_invariance"].with_mean is False

    def test_the_mlp_probes_new_parameters_arrive_at_the_network(self):
        network = build_probe({"family": "mlp", "alpha": 0.75, "solver": "lbfgs",
                               "learning_rate_init": 0.02, "batch_size": 32, "shuffle": False,
                               "early_stopping": True, "n_iter_no_change": 4, "tol": 0.25,
                               "with_std": False},
                              random_state=0)._pipeline
        mlp = network.named_steps["mlp"]
        assert (mlp.alpha, mlp.solver, mlp.learning_rate_init) == (0.75, "lbfgs", 0.02)
        assert (mlp.batch_size, mlp.shuffle, mlp.early_stopping) == (32, False, True)
        assert (mlp.n_iter_no_change, mlp.tol) == (4, 0.25)
        assert network.named_steps["scale_invariance"].with_std is False

    def test_the_tree_probes_new_parameters_arrive_at_the_ensemble(self):
        model = build_probe({"family": "gradient_boosted", "learning_rate": 0.03,
                             "max_leaf_nodes": 7, "min_samples_leaf": 4,
                             "l2_regularization": 1.5, "max_bins": 64, "early_stopping": False,
                             "max_depth": 3, "tol": 1e-4, "class_weight": "balanced"},
                            random_state=0)._model
        assert (model.learning_rate, model.max_leaf_nodes, model.min_samples_leaf) == (0.03, 7,
                                                                                       4)
        assert (model.l2_regularization, model.max_bins, model.max_depth) == (1.5, 64, 3)
        assert (model.early_stopping, model.tol, model.class_weight) == (False, 1e-4,
                                                                        "balanced")

    def test_the_calibrations_new_parameters_arrive_at_its_model(self):
        logistic = build_calibration({"method": "platt", "C": 0.25, "max_iter": 40,
                                      "tol": 0.05, "fit_intercept": False,
                                      "solver": "newton-cg"})._logistic
        assert (logistic.C, logistic.max_iter, logistic.tol) == (0.25, 40, 0.05)
        assert (logistic.fit_intercept, logistic.solver) == (False, "newton-cg")

    def test_the_specs_travel_the_whole_way_from_the_estimator_to_the_components(self):
        estimator = Estimator(probe={"family": "linear", "tol": 0.03},
                              calibration={"method": "platt", "C": 0.4},
                              route_probe=True, route_calibration=True, random_state=0)
        estimator.fit(four_case_slice(seed=3), four_case_slice(seed=4))
        for regime in (0, 1):
            assert estimator._probe_by_group[regime]._pipeline.named_steps["logistic"].tol == 0.03
            assert estimator._calibration_by_group[regime]._logistic.C == 0.4

    def test_an_unknown_parameter_still_fails_loudly_rather_than_being_swallowed(self):
        with pytest.raises(TypeError):
            build_probe({"family": "linear", "no_such_knob": 1})
        with pytest.raises(TypeError):
            build_calibration({"method": "platt", "no_such_knob": 1})


class TestTheActivationIsAStructuralConstantAndSaysSo:
    def test_relu_is_accepted_because_the_exit_reproduces_it(self):
        representation, agreement = material()
        probe = MLPProbe(random_state=0, hidden_layer_sizes=[6], max_iter=200,
                         activation="relu").fit(representation, agreement)
        assert np.isfinite(probe.logit_scores(representation)).all()

    @pytest.mark.parametrize("activation", ("tanh", "logistic", "identity"))
    def test_any_other_activation_is_refused_rather_than_silently_mis_scored(self, activation):
        with pytest.raises(ValueError, match="logit extraction supports"):
            build_probe({"family": "mlp", "activation": activation}, random_state=0)


def test_the_pinned_generator_reproduces_the_seeded_evaluation_allocation():
    family = np.repeat(np.arange(40), 5)
    drawn = draw_group_blocked_split(family, 0.2, 0.4375, 42)
    np.testing.assert_array_equal(
        np.unique(family[drawn["evaluation"]]), [4, 5, 7, 16, 18, 32, 37, 39])
