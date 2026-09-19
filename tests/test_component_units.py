import numpy as np
import pytest

from rcv.calibrations import PlattCalibration, build_calibration
from rcv.flip import FlipRule
from rcv.frames import agreement_of
from rcv.guards import ScoreIntegrityError, assert_scores_rankable
from rcv.probes import LinearProbe, build_probe
from rcv.splitting import draw_group_blocked_split


class TestFlipRule:
    def test_no_flip_exactly_at_the_threshold(self):
        rule = FlipRule(threshold=0.5)
        assert not rule.flips(np.array([0.5])).any()
        assert rule.flips(np.array([0.4999])).all()

    def test_a_flip_inverts_the_verdict_in_both_regimes(self):
        corrected = FlipRule(threshold=0.5).corrected_verdict(np.array([0, 1]),
                                                              np.array([0.1, 0.1]))
        np.testing.assert_array_equal(corrected, [1, 0])

    def test_a_trusted_verdict_passes_unchanged(self):
        corrected = FlipRule(threshold=0.5).corrected_verdict(np.array([0, 1]),
                                                              np.array([0.9, 0.9]))
        np.testing.assert_array_equal(corrected, [0, 1])


class TestFlipRuleThresholdDomain:
    @pytest.mark.parametrize("threshold", [0.0, 1.0])
    def test_a_threshold_at_either_end_of_the_unit_interval_is_refused(self, threshold):
        with pytest.raises(ValueError, match="Flip threshold"):
            FlipRule(threshold=threshold)


class TestPlattCalibrationContract:
    def test_a_flat_fit_is_refused_as_erasing_discrimination(self):
        constant_scores = np.full(40, 2.5)
        with pytest.raises(ValueError, match=(
                r"^Fitted calibration is flat; slope must be positive to preserve logit order "
                r"\(got 0\.0\)\.$")):
            PlattCalibration().fit(constant_scores, np.array([0, 1] * 20))

    def test_the_increasing_contract_rules_all_three_signs(self):
        from rcv.calibrations import assert_calibration_increasing

        assert_calibration_increasing(0.5)
        with pytest.raises(ValueError, match="slope must be positive"):
            assert_calibration_increasing(-0.5)
        with pytest.raises(ValueError, match="slope must be positive"):
            assert_calibration_increasing(0.0)

    def test_the_refusal_of_a_reversing_calibration_reads_in_full(self):
        scores = np.linspace(-5, 5, 40)
        with pytest.raises(ValueError, match=(
                r"^Fitted calibration is decreasing; slope must be positive to preserve logit "
                r"order \(got -[0-9.eE]+\)\.$")):
            PlattCalibration().fit(scores, (scores < 0).astype(int))


class TestCaughtShare:
    def test_a_population_with_no_passed_unsafe_item_is_refused_in_full(self):
        from rcv.study import caught_share_of

        with pytest.raises(ValueError, match=(
                r"^Caught share is undefined: no unsafe item was passed by the classifier\.$")):
            caught_share_of(np.array([1, 1]), np.array([1, 1]), np.array([1, 0]))


class TestAgreement:
    def test_agreement_is_integer_valued(self):
        frame = {"verdict": np.array([0, 1, 1]), "oracle": np.array([0, 0, 1])}
        agreement = agreement_of(frame, "oracle")
        assert agreement.dtype.kind == "i"
        np.testing.assert_array_equal(agreement, [1, 0, 1])


class TestScoresRankableBoundary:
    def test_fires_at_exactly_the_pinned_row_threshold(self):
        scores = np.concatenate([np.linspace(-5, 4, 47), np.full(3, 5.0)])
        with pytest.raises(ScoreIntegrityError, match="pinned"):
            assert_scores_rankable(scores, "test")

    def test_two_duplicated_rows_are_tolerated(self):
        scores = np.concatenate([np.linspace(-5, 4, 48), np.full(2, 5.0)])
        assert assert_scores_rankable(scores, "test") is scores


@pytest.fixture(scope="module")
def float32_task():
    rng = np.random.default_rng(0)
    representation = rng.normal(size=(200, 8)).astype(np.float32)
    agreement = (representation[:, 0] > 0).astype(int)
    return representation, agreement


class TestProbeBoundary:
    def test_float32_representation_still_yields_float64_rankable_scores(self, float32_task):
        representation, agreement = float32_task
        probe = LinearProbe(random_state=0).fit(representation, agreement)
        scores = probe.logit_scores(representation)
        assert scores.dtype == np.float64
        np.testing.assert_array_equal(scores,
                                      probe.logit_scores(representation.astype(np.float64)))

    def test_fit_casts_to_float64_before_learning(self, float32_task):
        representation, agreement = float32_task
        widened = representation.astype(np.float64)
        fitted_narrow = LinearProbe(random_state=0).fit(representation, agreement)
        fitted_wide = LinearProbe(random_state=0).fit(widened, agreement)
        np.testing.assert_array_equal(fitted_narrow.logit_scores(widened),
                                      fitted_wide.logit_scores(widened))

    def test_unknown_family_names_the_known_ones(self):
        with pytest.raises(ValueError, match="Unknown probe family"):
            build_probe({"family": "quantum"})

    def test_an_unknown_spec_key_fails_loudly(self):
        with pytest.raises(TypeError):
            build_probe({"family": "linear", "depth": 3})

    def test_unknown_calibration_method_names_the_known_ones(self):
        with pytest.raises(ValueError, match="Unknown calibration method"):
            build_calibration({"method": "isotonic"})


@pytest.fixture(scope="module")
def fixture_frame():
    from fixture_frame import build_fixture_frame

    return build_fixture_frame()


class TestEstimatorComposition:
    def fitted(self, fixture_frame, **flags):
        from fixture_frame import split_for_fitting
        from rcv.estimator import Estimator
        from rcv.frames import fit_on_slices

        estimator = Estimator(probe={"family": "linear"}, calibration={"method": "platt"},
                              route_probe=flags.get("route_probe", True),
                              route_calibration=flags.get("route_calibration", True),
                              random_state=flags.get("random_state", 0))
        return fit_on_slices(estimator, *split_for_fitting(fixture_frame), labels="oracle")

    def test_routing_the_probe_changes_what_is_fitted(self, fixture_frame):
        routed = self.fitted(fixture_frame)
        pooled_probe = self.fitted(fixture_frame, route_probe=False)
        assert not np.array_equal(
            routed.probe_scores(fixture_frame["representation"], fixture_frame["verdict"]),
            pooled_probe.probe_scores(fixture_frame["representation"], fixture_frame["verdict"]))

    def test_routing_the_calibration_changes_the_probabilities(self, fixture_frame):
        routed = self.fitted(fixture_frame)
        pooled_calibration = self.fitted(fixture_frame, route_calibration=False)
        assert not np.array_equal(
            routed.probability_of_agreement(fixture_frame["representation"],
                                            fixture_frame["verdict"]),
            pooled_calibration.probability_of_agreement(fixture_frame["representation"],
                                                        fixture_frame["verdict"]))

    def test_a_verdict_outside_the_regimes_fails_loudly(self, fixture_frame):
        estimator = self.fitted(fixture_frame)
        bad_verdict = fixture_frame["verdict"].copy()
        bad_verdict[0] = 2
        with pytest.raises(ValueError, match=(
                r"^estimator routing: verdict contains values outside regimes \(0, 1\)\.$")):
            estimator.probe_scores(fixture_frame["representation"], bad_verdict)

    def test_a_calibration_slice_missing_a_case_is_named_as_such(self, fixture_frame):
        from fixture_frame import split_for_fitting
        from rcv.estimator import Estimator
        from rcv.frames import fit_on_slices, rows
        from rcv.guards import MissingCaseError

        fit_slice, calibration_slice = split_for_fitting(fixture_frame)
        wrong_in_regime_zero = ((calibration_slice["verdict"] == 0)
                                & (calibration_slice["verdict"] != calibration_slice["oracle"]))
        degenerate_calibration = rows(calibration_slice, ~wrong_in_regime_zero)
        estimator = Estimator(probe={"family": "linear"}, calibration={"method": "platt"},
                              route_probe=True, route_calibration=True)
        with pytest.raises(MissingCaseError, match="calibration slice"):
            fit_on_slices(estimator, fit_slice, degenerate_calibration, labels="oracle")

    def test_the_seed_reaches_the_probe_component(self, fixture_frame):
        pipeline = self.fitted(fixture_frame, random_state=7)._probe_by_group[0]._pipeline
        assert "scale_invariance" in pipeline.named_steps
        assert pipeline.named_steps["logistic"].random_state == 7


@pytest.fixture(scope="module")
def family():
    return np.repeat(np.arange(40), 3)


class TestGroupBlockedSplit:
    def test_the_three_masks_partition_every_row(self, family):
        masks = draw_group_blocked_split(family, 0.3, 0.25, seed=1)
        one_split_each = sum(mask.astype(int) for mask in masks.values())
        assert (one_split_each == 1).all()

    def test_a_family_never_straddles_splits(self, family):
        masks = draw_group_blocked_split(family, 0.3, 0.25, seed=1)
        for mask in masks.values():
            for one_family in np.unique(family):
                rows_of_family = mask[family == one_family]
                assert rows_of_family.all() or not rows_of_family.any()

    def test_the_fractions_are_honoured_in_families(self, family):
        masks = draw_group_blocked_split(family, 0.3, 0.25, seed=1)
        assert len(np.unique(family[masks["evaluation"]])) == 12
        assert len(np.unique(family[masks["calibration"]])) == 7

    def test_the_seed_governs_the_assignment(self, family):
        first = draw_group_blocked_split(family, 0.3, 0.25, seed=1)
        again = draw_group_blocked_split(family, 0.3, 0.25, seed=1)
        moved = draw_group_blocked_split(family, 0.3, 0.25, seed=2)
        np.testing.assert_array_equal(first["evaluation"], again["evaluation"])
        assert not np.array_equal(first["evaluation"], moved["evaluation"])


class TestH1MonitorAndCalibrationRefusals:
    def test_a_calibration_that_would_reverse_the_probes_order_is_refused(self):
        from rcv.calibrations import PlattCalibration

        scores = np.linspace(-5, 5, 40)
        anti_correlated_agreement = (scores < 0).astype(int)
        with pytest.raises(ValueError, match="Fitted calibration is decreasing"):
            PlattCalibration().fit(scores, anti_correlated_agreement)

    def test_a_flip_threshold_outside_the_unit_interval_is_refused(self):
        from rcv.flip import FlipRule

        with pytest.raises(ValueError, match="threshold"):
            FlipRule(threshold=1.5)


class TestH1SplitterValidation:
    def test_fractions_outside_the_unit_interval_are_refused(self, family):
        with pytest.raises(ValueError, match="fraction"):
            draw_group_blocked_split(family, 1.5, 0.25, seed=1)
        with pytest.raises(ValueError, match="fraction"):
            draw_group_blocked_split(family, 0.3, -0.2, seed=1)

    def test_a_fraction_that_rounds_a_split_empty_is_refused(self, family):
        with pytest.raises(ValueError, match="split has no families"):
            draw_group_blocked_split(family, 0.001, 0.25, seed=1)

    @pytest.mark.parametrize("fraction", [0.0, 1.0])
    def test_a_fraction_at_either_end_of_the_unit_interval_is_refused(self, family, fraction):
        with pytest.raises(ValueError, match="must be in"):
            draw_group_blocked_split(family, fraction, 0.25, seed=1)
        with pytest.raises(ValueError, match="must be in"):
            draw_group_blocked_split(family, 0.3, fraction, seed=1)

    def test_the_refusal_names_the_fraction_that_is_out_of_range(self, family):
        with pytest.raises(
                ValueError,
                match=r"^evaluation_fraction must be in \(0, 1\); got 1\.5\.$"):
            draw_group_blocked_split(family, 1.5, 0.25, seed=1)
        with pytest.raises(
                ValueError,
                match=r"^calibration_fraction must be in \(0, 1\); got -0\.2\.$"):
            draw_group_blocked_split(family, 0.3, -0.2, seed=1)


class TestH1ScoringEdges:
    def test_scoring_a_single_regime_window_returns_without_error(self, fixture_frame):
        from fixture_frame import split_for_fitting
        from rcv.frames import fit_on_slices

        estimator = fit_on_slices(_fitted_spec(), *split_for_fitting(fixture_frame),
                                  labels="oracle")
        regime_zero_rows = fixture_frame["verdict"] == 0
        scores = estimator.probe_scores(fixture_frame["representation"][regime_zero_rows],
                                        fixture_frame["verdict"][regime_zero_rows])
        assert len(scores) == regime_zero_rows.sum()
        assert np.isfinite(scores).all()


class TestH1CrossFitFolds:
    def test_no_family_straddles_two_folds(self):
        from rcv.splitting import cross_fit_fold_of

        family = np.repeat(np.arange(10), 5)
        fold_of = cross_fit_fold_of(family, folds=4, rng=np.random.default_rng(3))
        for one_family in np.unique(family):
            assert len(np.unique(fold_of[family == one_family])) == 1

    def test_the_folds_cover_and_the_rng_governs(self):
        from rcv.splitting import cross_fit_fold_of

        family = np.repeat(np.arange(40), 3)
        fold_of = cross_fit_fold_of(family, folds=4, rng=np.random.default_rng(3))
        assert set(np.unique(fold_of)) == set(range(4))
        np.testing.assert_array_equal(
            fold_of, cross_fit_fold_of(family, folds=4, rng=np.random.default_rng(3)))
        assert not np.array_equal(
            fold_of, cross_fit_fold_of(family, folds=4, rng=np.random.default_rng(4)))


def _fitted_spec():
    from rcv.estimator import Estimator

    return Estimator(probe={"family": "linear"}, calibration={"method": "platt"},
                     route_probe=True, route_calibration=True, random_state=0)


class TestPinnedCohortDuplicates:
    def test_identical_inputs_tied_at_the_bound_are_honest(self):
        from rcv.guards import assert_scores_rankable

        inputs = np.vstack([np.ones((3, 4)), np.random.default_rng(0).normal(size=(47, 4))])
        scores = np.concatenate([np.full(3, 9.0), np.linspace(-5, 5, 47)])
        assert assert_scores_rankable(scores, "test", inputs=inputs) is scores

    def test_duplicates_at_one_bound_never_excuse_a_clip_at_the_other(self):
        from rcv.guards import ScoreIntegrityError, assert_scores_rankable

        rng = np.random.default_rng(0)
        inputs = np.vstack([np.ones((3, 4)), rng.normal(size=(44, 4)), rng.normal(size=(3, 4))])
        scores = np.concatenate([np.full(3, -9.0), np.linspace(-5, 5, 44), np.full(3, 9.0)])
        with pytest.raises(ScoreIntegrityError, match="pinned"):
            assert_scores_rankable(scores, "test", inputs=inputs)

    def test_distinct_inputs_pinned_at_the_bound_still_refuse(self):
        from rcv.guards import ScoreIntegrityError, assert_scores_rankable

        inputs = np.random.default_rng(0).normal(size=(50, 4))
        scores = np.concatenate([np.full(3, 9.0), np.linspace(-5, 5, 47)])
        with pytest.raises(ScoreIntegrityError, match="pinned"):
            assert_scores_rankable(scores, "test", inputs=inputs)
