import numpy as np
import pytest

from fixture_frame import build_fixture_frame, split_for_fitting
from rcv.frames import fit_on_slices


def fitted(probe_spec, frame, routed=True):
    from rcv.estimator import Estimator

    estimator = Estimator(probe=probe_spec, calibration={"method": "platt"},
                          route_probe=routed, route_calibration=routed, random_state=0)
    return fit_on_slices(estimator, *split_for_fitting(frame), labels="oracle")


@pytest.fixture(scope="module")
def frame():
    return build_fixture_frame()


class TestGradientBoostedFamily:
    def test_fits_scores_and_tolerates_its_own_ties(self, frame):
        estimator = fitted({"family": "gradient_boosted", "max_iter": 20}, frame)
        scores = estimator.probe_scores(frame["representation"], frame["verdict"])
        assert scores.dtype == np.float64
        assert np.isfinite(scores).all()
        assert np.unique(scores).size < len(scores)

    def test_the_family_declares_its_ties(self):
        from rcv.probes import PROBE_FAMILIES

        assert PROBE_FAMILIES["gradient_boosted"].scores_are_continuous is False
        assert PROBE_FAMILIES["linear"].scores_are_continuous is True
        assert PROBE_FAMILIES["mlp"].scores_are_continuous is True

    def test_early_stopping_is_ours_and_not_a_row_count(self):
        import inspect

        from rcv.probes import GradientBoostedProbe

        declared = inspect.signature(GradientBoostedProbe).parameters["early_stopping"].default
        assert declared is False, "early_stopping must be pinned, never left to a row count"

        probe = GradientBoostedProbe(random_state=0)
        assert probe._model.early_stopping is False

    def test_a_fitted_tree_probe_did_not_early_stop(self, frame):
        from rcv.probes import GradientBoostedProbe

        probe = GradientBoostedProbe(random_state=0, max_iter=10).fit(
            frame["representation"], (frame["verdict"] == frame["oracle"]).astype(int))
        assert probe._model.do_early_stopping_ is False
        assert probe._model.n_iter_ == 10


class TestMlpFamily:
    def test_the_logits_are_the_presigmoid_values(self, frame):
        from scipy.special import expit

        estimator = fitted({"family": "mlp", "hidden_layer_sizes": [8], "max_iter": 400}, frame)
        probe = estimator._probe_by_group[0]
        rows = frame["verdict"] == 0
        scores = probe.logit_scores(frame["representation"][rows])
        probabilities = probe._pipeline.predict_proba(
            frame["representation"][rows].astype(np.float64))[:, 1]
        np.testing.assert_allclose(expit(scores), probabilities, atol=1e-12)

    def test_a_saturating_fit_still_yields_finite_logits(self):
        from rcv.probes import MLPProbe

        rng = np.random.default_rng(0)
        representation = np.vstack([rng.normal(-1, 1, size=(60, 3)),
                                    rng.normal(+1, 1, size=(60, 3))])
        agreement = np.repeat([0, 1], 60)
        probe = MLPProbe(random_state=0, hidden_layer_sizes=[8], max_iter=400).fit(
            representation, agreement)
        mlp = probe._pipeline.named_steps["mlp"]
        mlp.coefs_ = [w * 20.0 for w in mlp.coefs_]
        mlp.intercepts_ = [b * 20.0 for b in mlp.intercepts_]
        probabilities = probe._pipeline.predict_proba(representation)[:, 1]
        assert (probabilities == 1.0).any(), "the premise: saturation genuinely occurs"
        scores = probe.logit_scores(representation)
        assert np.isfinite(scores).all()
        assert scores.dtype == np.float64

    def test_fits_and_yields_float64_finite_logits(self, frame):
        estimator = fitted({"family": "mlp", "hidden_layer_sizes": [8], "max_iter": 400}, frame)
        scores = estimator.probe_scores(frame["representation"], frame["verdict"])
        assert scores.dtype == np.float64
        assert np.isfinite(scores).all()
        assert scores.min() < 0.0 or scores.max() > 1.0

    def test_a_saturated_probability_refuses_at_the_seam_with_no_clip(self, frame):
        from rcv.guards import ScoreIntegrityError

        estimator = fitted({"family": "mlp", "hidden_layer_sizes": [8], "max_iter": 400}, frame)
        probe = estimator._probe_by_group[0]
        healthy = probe.logit_scores
        probe.logit_scores = lambda representation: np.where(
            np.arange(len(representation)) < 3, np.inf, healthy(representation))
        try:
            with pytest.raises(ScoreIntegrityError, match="non-finite"):
                estimator.probe_scores(frame["representation"], frame["verdict"])
        finally:
            probe.logit_scores = healthy


def separable_material(dimension=4, per_class=60, seed=0):
    rng = np.random.default_rng(seed)
    representation = np.vstack([rng.normal(-1, 1, size=(per_class, dimension)),
                                rng.normal(+1, 1, size=(per_class, dimension))])
    return representation, np.repeat([0, 1], per_class)


class TestMlpConfigurationReachesTheNetwork:
    def test_the_configured_shape_and_iteration_budget_reach_the_network(self):
        from rcv.probes import MLPProbe

        network = MLPProbe(random_state=0, hidden_layer_sizes=[6, 4],
                           max_iter=37)._pipeline.named_steps["mlp"]
        assert network.hidden_layer_sizes == (6, 4)
        assert network.max_iter == 37

    def test_the_seed_governs_the_fit(self):
        from rcv.probes import MLPProbe

        representation, agreement = separable_material()
        scores = [MLPProbe(random_state=state, hidden_layer_sizes=[6], max_iter=300)
                  .fit(representation, agreement).logit_scores(representation)
                  for state in (0, 0, 1)]
        np.testing.assert_array_equal(scores[0], scores[1])
        assert not np.array_equal(scores[0], scores[2]), "the seed reaches nothing"


class TestMlpLogitExit:
    def test_the_exit_walks_every_hidden_layer_to_the_last(self):
        from scipy.special import expit

        from rcv.probes import MLPProbe

        representation, agreement = separable_material()
        probe = MLPProbe(random_state=0, hidden_layer_sizes=[6, 4], max_iter=400).fit(
            representation, agreement)
        assert len(probe._pipeline.named_steps["mlp"].coefs_) == 3, "the premise: three layers"
        np.testing.assert_allclose(
            expit(probe.logit_scores(representation)),
            probe._pipeline.predict_proba(representation)[:, 1], atol=1e-12)


class TestMlpConditionsItsInput:
    def test_a_narrow_representation_is_widened_before_the_fit(self):
        from rcv.probes import MLPProbe

        representation, agreement = separable_material()
        narrow = representation.astype(np.float32)
        widened = narrow.astype(np.float64)
        settings = dict(random_state=0, hidden_layer_sizes=[6], max_iter=300)
        from_narrow = MLPProbe(**settings).fit(narrow, agreement).logit_scores(widened)
        from_widened = MLPProbe(**settings).fit(widened, agreement).logit_scores(widened)
        np.testing.assert_array_equal(from_narrow, from_widened)

    def test_a_narrow_representation_is_widened_before_it_is_scored(self):
        from rcv.probes import MLPProbe

        representation, agreement = separable_material()
        narrow = representation.astype(np.float32)
        probe = MLPProbe(random_state=0, hidden_layer_sizes=[6], max_iter=300).fit(
            representation, agreement)
        np.testing.assert_array_equal(probe.logit_scores(narrow),
                                      probe.logit_scores(narrow.astype(np.float64)))


class TestPinningScopedByFamily:
    def test_the_guard_skips_pinning_when_told_the_ties_are_honest(self):
        from rcv.guards import assert_scores_rankable

        tied = np.concatenate([np.full(5, 3.0), np.linspace(-2, 2, 20)])
        inputs = np.random.default_rng(0).normal(size=(25, 4))
        assert assert_scores_rankable(tied, "test", inputs=inputs,
                                      check_pinning=False) is tied

    def test_a_tree_estimator_never_refuses_its_bound_ties(self, frame):
        estimator = fitted({"family": "gradient_boosted", "max_iter": 5}, frame)
        scores = estimator.probe_scores(frame["representation"], frame["verdict"])
        assert np.isfinite(scores).all()


class TestFamilyPairs:
    def test_the_a2_pairs_run_on_the_same_seeds_with_a_per_family_contrast(self, tmp_path):
        from rcv.ablation import run_probe_family_pairs
        from test_ablation import streamless_config

        config = streamless_config(tmp_path)
        config["seeds"] = [0, 1]
        result = run_probe_family_pairs(
            config, tmp_path / "a2",
            family_specs=[{"family": "linear", "C": 1.0},
                          {"family": "gradient_boosted", "max_iter": 20}])
        assert set(result["cells"]) == {"linear_pooled", "linear_routed",
                                        "gradient_boosted_pooled", "gradient_boosted_routed"}
        contrast = result["family_contrasts"]["gradient_boosted"]
        assert sum(contrast["corrected_adherence_wins_ties_losses"]) == 2
        assert contrast["factor"] == "routing: probe and calibration are routed by verdict regime"


class TestProbeConfigurationReachesItsModel:
    def test_the_linear_probes_regularisation_and_budget_arrive(self):
        from rcv.probes import LinearProbe

        model = LinearProbe(random_state=3, C=0.25,
                            max_iter=77)._pipeline.named_steps["logistic"]
        assert (model.C, model.max_iter, model.random_state) == (0.25, 77, 3)

    def test_the_linear_probes_shipped_regularisation_is_one(self):
        from rcv.probes import LinearProbe

        assert LinearProbe()._pipeline.named_steps["logistic"].C == 1.0

    def test_the_tree_probes_budget_and_seed_arrive(self):
        from rcv.probes import GradientBoostedProbe

        model = GradientBoostedProbe(random_state=5, max_iter=13)._model
        assert (model.max_iter, model.random_state) == (13, 5)


class TestTheEstimatorTellsTheGuardWhichFamilyItIsScoring:
    def test_a_continuous_family_pinned_at_a_bound_is_refused_at_the_seam(self, frame):
        from rcv.guards import ScoreIntegrityError

        estimator = fitted({"family": "linear"}, frame)
        probe = estimator._probe_by_group[0]
        healthy = probe.logit_scores

        def pinned_at_the_top(representation):
            scores = healthy(representation).copy()
            scores[np.argsort(scores)[-5:]] = scores.max()
            return scores

        probe.logit_scores = pinned_at_the_top
        try:
            with pytest.raises(ScoreIntegrityError, match="pinned"):
                estimator.probe_scores(frame["representation"], frame["verdict"])
        finally:
            probe.logit_scores = healthy
