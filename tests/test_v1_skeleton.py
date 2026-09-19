import numpy as np
import pytest
import yaml

from fixture_frame import FIXTURE_REFERENCE_WINDOW, build_fixture_frame, split_for_fitting
from rcv.frames import fit_on_slices


def linear_platt_estimator(route_probe=True, route_calibration=True, random_state=None):
    from rcv.estimator import Estimator

    return Estimator(probe={"family": "linear"}, calibration={"method": "platt"},
                     route_probe=route_probe, route_calibration=route_calibration,
                     random_state=random_state)


def write_frame(tmp_path, frame, name="fixture_frame.npz"):
    path = tmp_path / name
    np.savez(path, **frame)
    return path


def write_v1_config(tmp_path, frame_path):
    config = {
        "study": "v1-skeleton",
        "seeds": [0],
        "frame": {"path": str(frame_path), "name": "synthetic-fixture", "fitting_labels": "oracle"},
        "split": {"evaluation_fraction": 0.3, "calibration_fraction": 0.25},
        "estimator": {
            "probe": {"family": "linear"},
            "calibration": {"method": "platt"},
            "route_probe": True,
            "route_calibration": True,
        },
        "flip": {"threshold": 0.5},
        "monitor": {"quiet_horizon_confidence": 0.95,
                    "detectable_shift": {"indifference_zone_quantile": 0.99},
                    "event_bank": {"boundaries": [0.5], "budget_allocation": "joint"},
                    "calibration": {"stream_length": 120, "n_streams": 2000, "streams_per_centre": 1}},
        "loop": {
            "audit_sampling_window": 120,
            "audit_budget": 60,
            "acceptance": {"recall_floor": 0.5, "false_positive_tolerance": 0.25},
            "max_enlarging_retries": 2,
            "cross_fit_folds": 4,
            "post_repair_reference_window": FIXTURE_REFERENCE_WINDOW,
        },
    }
    path = tmp_path / "v1.yaml"
    path.write_text(yaml.safe_dump(config))
    return path


@pytest.fixture(scope="module")
def study_result(tmp_path_factory):
    from rcv.study import run_study

    tmp_path = tmp_path_factory.mktemp("v1")
    frame_path = write_frame(tmp_path, build_fixture_frame())
    return run_study(write_v1_config(tmp_path, frame_path))


class TestP00WalkingSkeleton:
    def test_correction_improves_adherence_on_the_evaluation_split(self, study_result):
        readout = study_result.readout
        assert readout.corrected_adherence > readout.raw_adherence

    def test_every_reported_number_names_frame_labels_and_labels_spent(self, study_result):
        readout = study_result.readout
        assert readout.frame_name == "synthetic-fixture"
        assert readout.label_set == "oracle"
        assert readout.oracle_labels_spent >= 0

    def test_the_monitor_reads_a_per_regime_event_rate_reference(self, study_result):
        deployed = study_result.monitor.deployed
        assert set(deployed.reference) == {0, 1}
        for wires in deployed.reference.values():
            assert set(wires) == set(deployed.boundaries)
            for rate in wires.values():
                assert 0.0 <= rate <= 1.0

    def test_the_drift_segment_drives_one_full_loop_cycle(self, study_result):
        kinds = [event.kind for event in study_result.loop.events]
        assert kinds.index("alarm") < kinds.index("audit")
        assert kinds.index("audit") < kinds.index("repair_accepted")
        assert "reference_rederived" in kinds

    def test_every_change_is_recorded_with_trigger_and_label_cost(self, study_result):
        assert study_result.change_log, "the loop changed RCV; every change requires a record"
        for entry in study_result.change_log:
            assert entry.trigger
            assert entry.oracle_labels_spent >= 0


class TestP09FourCases:
    def test_a_slice_missing_a_case_refuses_to_fit(self):
        from rcv.frames import rows
        from rcv.guards import MissingCaseError

        frame = build_fixture_frame()
        wrong_in_regime_zero = (frame["verdict"] == 0) & (frame["verdict"] != frame["oracle"])
        degenerate = rows(frame, ~wrong_in_regime_zero)
        with pytest.raises(MissingCaseError,
                           match="fitting slice lacks regime=0, agreement=0"):
            fit_on_slices(linear_platt_estimator(), *split_for_fitting(degenerate),
                          labels="oracle")


class TestP15OrderSurvives:
    def test_probe_scores_are_float64_logits_with_no_saturation_ties(self, fitted_estimator, frame):
        scores = fitted_estimator.probe_scores(frame["representation"], frame["verdict"])
        assert scores.dtype == np.float64
        assert scores.min() < 0.0 or scores.max() > 1.0, "logit-valued, not probability-bounded"
        assert np.unique(scores).size == scores.size, "distinct inputs collapsed to tied scores"


class TestP15Guard:
    def test_refuses_float32_scores(self):
        from rcv.guards import ScoreIntegrityError, assert_scores_rankable

        with pytest.raises(ScoreIntegrityError, match="score dtype must be float64"):
            assert_scores_rankable(np.linspace(-5, 5, 100, dtype=np.float32), "test")

    def test_refuses_nonfinite_scores(self):
        from rcv.guards import ScoreIntegrityError, assert_scores_rankable

        scores = np.linspace(-5, 5, 100)
        scores[7] = np.nan
        with pytest.raises(ScoreIntegrityError, match="non-finite"):
            assert_scores_rankable(scores, "test")

    def test_refuses_scores_pinned_to_a_bound(self):
        from rcv.guards import ScoreIntegrityError, assert_scores_rankable

        clipped = np.clip(np.linspace(-30, 30, 100), -16.118, 16.118)
        with pytest.raises(ScoreIntegrityError, match="pinned"):
            assert_scores_rankable(clipped, "test")

    def test_passes_healthy_logits_through_unchanged(self):
        from rcv.guards import assert_scores_rankable

        scores = np.linspace(-30, 30, 100)
        assert assert_scores_rankable(scores, "test") is scores

    def test_the_estimator_seam_inherits_the_guard(self, fitted_estimator, frame):
        from rcv.guards import ScoreIntegrityError

        probe = fitted_estimator._probe_by_group[0]
        healthy_scores = probe.logit_scores

        probe.logit_scores = lambda representation: healthy_scores(representation).astype(
            np.float32)
        try:
            with pytest.raises(ScoreIntegrityError, match=r"probe\[0\]"):
                fitted_estimator.probe_scores(frame["representation"], frame["verdict"])
        finally:
            probe.logit_scores = healthy_scores


class TestP16ScaleInvariance:
    def test_scaling_the_representation_leaves_decisions_identical(self, frame):
        def fit_and_score(scale):
            fit_slice, calibration_slice = split_for_fitting(frame)
            fit_slice = dict(fit_slice, representation=fit_slice["representation"] * scale)
            calibration_slice = dict(calibration_slice,
                                     representation=calibration_slice["representation"] * scale)
            estimator = fit_on_slices(linear_platt_estimator(random_state=0),
                                      fit_slice, calibration_slice, labels="oracle")
            return estimator.probability_of_agreement(frame["representation"] * scale,
                                                      frame["verdict"])

        np.testing.assert_allclose(fit_and_score(1000.0), fit_and_score(1.0), atol=1e-6)


class TestA1CompositionPoint:
    @pytest.mark.parametrize("route_probe", [False, True])
    @pytest.mark.parametrize("route_calibration", [False, True])
    def test_every_cell_of_the_factorial_fits_and_scores(self, frame, route_probe,
                                                         route_calibration):
        estimator = fit_on_slices(
            linear_platt_estimator(route_probe=route_probe,
                                   route_calibration=route_calibration),
            *split_for_fitting(frame), labels="oracle")
        probabilities = estimator.probability_of_agreement(frame["representation"],
                                                           frame["verdict"])
        assert probabilities.shape == frame["verdict"].shape
        assert np.all((probabilities >= 0.0) & (probabilities <= 1.0))


@pytest.fixture(scope="module")
def frame():
    return build_fixture_frame()


@pytest.fixture(scope="module")
def fitted_estimator(frame):
    return fit_on_slices(linear_platt_estimator(random_state=0), *split_for_fitting(frame),
                         labels="oracle")


class TestH1FrameAndConfigContracts:
    def test_an_integer_is_stream_column_is_refused(self, tmp_path):
        from rcv.study import run_study

        frame = build_fixture_frame()
        frame["is_stream"] = frame["is_stream"].astype(np.int8)
        config_path = write_v1_config(tmp_path, write_frame(tmp_path, frame))
        with pytest.raises(ValueError, match="is_stream"):
            run_study(config_path)

    def test_an_unknown_top_level_config_key_is_refused(self, tmp_path):
        from rcv.study import run_study

        frame_path = write_frame(tmp_path, build_fixture_frame())
        config_path = write_v1_config(tmp_path, frame_path)
        config = yaml.safe_load(config_path.read_text())
        config["standardise"] = False
        config_path.write_text(yaml.safe_dump(config))
        with pytest.raises(ValueError, match="standardise"):
            run_study(config_path)

    def test_a_fitting_label_set_absent_from_the_frame_is_refused(self, tmp_path):
        from rcv.study import run_study

        frame_path = write_frame(tmp_path, build_fixture_frame())
        config_path = write_v1_config(tmp_path, frame_path)
        config = yaml.safe_load(config_path.read_text())
        config["frame"]["fitting_labels"] = "human_gold"
        config_path.write_text(yaml.safe_dump(config))
        with pytest.raises(ValueError, match="human_gold"):
            run_study(config_path)

    def test_the_fitting_label_set_genuinely_reaches_the_fit(self, tmp_path):
        from rcv.study import run_study

        frame = build_fixture_frame()
        rng = np.random.default_rng(11)
        frame["second_axis"] = np.where(rng.random(len(frame["oracle"])) < 0.05,
                                        1 - frame["oracle"], frame["oracle"])
        frame_path = write_frame(tmp_path, frame)
        config_path = write_v1_config(tmp_path, frame_path)
        config = yaml.safe_load(config_path.read_text())
        config["frame"]["fitting_labels"] = "second_axis"
        config_path.write_text(yaml.safe_dump(config))
        swapped = run_study(config_path).readout

        assert swapped.label_set == "second_axis"
        original = run_study(write_v1_config(tmp_path, frame_path)).readout
        assert swapped.corrected_adherence != original.corrected_adherence


class TestH1HonestAccounting:
    def test_labels_spent_covers_every_label_the_readout_consumed(self, study_result):
        frame = build_fixture_frame()
        serving_rows = int((~frame["is_stream"]).sum())
        assert study_result.readout.oracle_labels_spent == serving_rows

    def test_the_study_total_reconciles_readout_and_loop(self, study_result):
        loop_labels = sum(entry.oracle_labels_spent for entry in study_result.change_log)
        assert (study_result.total_oracle_labels_spent
                == study_result.readout.oracle_labels_spent + loop_labels)


class TestH1EvaluationSliceChecked:
    def test_an_evaluation_slice_missing_a_case_is_refused(self, tmp_path):
        from rcv.splitting import draw_group_blocked_split
        from rcv.study import run_study

        frame = build_fixture_frame()
        serving = {name: values[~frame["is_stream"]] for name, values in frame.items()}
        masks = draw_group_blocked_split(serving["family"], 0.3, 0.25, seed=0)
        in_evaluation = np.isin(frame["family"], np.unique(serving["family"][masks["evaluation"]]))
        wrong_in_regime_zero = (frame["verdict"] == 0) & (frame["verdict"] != frame["oracle"])
        keep = ~(in_evaluation & wrong_in_regime_zero & ~frame["is_stream"])
        degenerate = {name: values[keep] for name, values in frame.items()}

        config_path = write_v1_config(tmp_path, write_frame(tmp_path, degenerate))
        from rcv.guards import MissingCaseError
        with pytest.raises(MissingCaseError, match=(
                r"^evaluation slice lacks regime=0, agreement=0\.$")):
            run_study(config_path)


class TestCaughtShare:
    def test_the_arithmetic_on_a_hand_computable_case(self):
        from rcv.study import caught_share_of

        verdict = np.array([0, 0, 0, 1, 1, 0])
        oracle = np.array([1, 1, 0, 0, 1, 1])
        corrected = np.array([1, 0, 0, 1, 1, 0])
        assert caught_share_of(verdict, corrected, oracle) == pytest.approx(1 / 3)

    def test_a_frame_with_no_misses_refuses_rather_than_divides_by_zero(self):
        from rcv.study import caught_share_of

        all_correct = np.array([0, 1])
        with pytest.raises(ValueError, match="Caught share is undefined"):
            caught_share_of(all_correct, all_correct, all_correct)

    def test_the_readout_carries_it(self, study_result):
        assert 0.0 <= study_result.readout.caught_share <= 1.0


class TestStrataWiring:
    def test_the_strata_key_reaches_the_splitter_as_four_classes(self, tmp_path, monkeypatch):
        import rcv.study as study_module
        from rcv.study import run_study

        captured = {}
        real_draw = study_module.draw_group_blocked_split

        def spy(family, evaluation_fraction, calibration_fraction, seed, strata=None):
            captured["strata"] = strata
            return real_draw(family, evaluation_fraction, calibration_fraction, seed,
                             strata=strata)

        monkeypatch.setattr(study_module, "draw_group_blocked_split", spy)
        frame_path = write_frame(tmp_path, build_fixture_frame())
        config_path = write_v1_config(tmp_path, frame_path)
        config = yaml.safe_load(config_path.read_text())
        config["split"]["strata"] = ["verdict", "agreement"]
        config_path.write_text(yaml.safe_dump(config))
        run_study(config_path)

        assert captured["strata"] is not None
        assert set(np.unique(captured["strata"])) <= {0, 1, 2, 3}

    def test_an_unsupported_strata_axis_is_refused(self, tmp_path):
        from rcv.study import run_study

        frame_path = write_frame(tmp_path, build_fixture_frame())
        config_path = write_v1_config(tmp_path, frame_path)
        config = yaml.safe_load(config_path.read_text())
        config["split"]["strata"] = ["verdict", "family"]
        config_path.write_text(yaml.safe_dump(config))
        with pytest.raises(ValueError, match="strata"):
            run_study(config_path)


class TestStreamlessStudy:
    def test_a_frame_without_traffic_needs_no_monitor_and_no_loop(self, tmp_path):
        from rcv.study import run_study

        frame = build_fixture_frame()
        streamless = {name: values[~frame["is_stream"]] for name, values in frame.items()}
        config_path = write_v1_config(tmp_path, write_frame(tmp_path, streamless))
        result = run_study(config_path)
        assert result.monitor.deployed is None
        assert result.loop.events == []
        assert result.change_log == []
        assert 0.0 <= result.readout.corrected_adherence <= 1.0


class TestRunStudyValidationBoundaries:
    def test_two_seeds_without_an_explicit_seed_refuse_in_full(self, tmp_path):
        from rcv.study import run_study

        frame_path = write_frame(tmp_path, build_fixture_frame())
        config_path = write_v1_config(tmp_path, frame_path)
        config = yaml.safe_load(config_path.read_text())
        config["seeds"] = [0, 1]
        config_path.write_text(yaml.safe_dump(config))
        with pytest.raises(ValueError, match=(
                r"^run_study needs an explicit seed when config\.seeds has 2 values\.$")):
            run_study(config_path)

    def test_acceptance_boundaries_zero_and_one_are_legal(self, tmp_path):
        from rcv.study import run_study

        frame = build_fixture_frame()
        streamless = {name: values[~frame["is_stream"]] for name, values in frame.items()}
        config_path = write_v1_config(tmp_path, write_frame(tmp_path, streamless))
        config = yaml.safe_load(config_path.read_text())
        config["loop"]["acceptance"] = {"recall_floor": 1.0, "false_positive_tolerance": 0.0}
        config_path.write_text(yaml.safe_dump(config))
        run_study(config_path)

    def test_an_acceptance_value_above_one_refuses_in_full(self, tmp_path):
        from rcv.study import run_study

        frame_path = write_frame(tmp_path, build_fixture_frame())
        config_path = write_v1_config(tmp_path, frame_path)
        config = yaml.safe_load(config_path.read_text())
        config["loop"]["acceptance"]["recall_floor"] = 1.5
        config_path.write_text(yaml.safe_dump(config))
        with pytest.raises(ValueError,
                           match=r"^acceptance\.recall_floor must be in \[0, 1\]; got 1\.5\.$"):
            run_study(config_path)


class TestH2ConfigContracts:
    def _config(self, tmp_path):
        frame_path = write_frame(tmp_path, build_fixture_frame())
        config_path = write_v1_config(tmp_path, frame_path)
        return config_path, yaml.safe_load(config_path.read_text())

    def test_a_nested_unknown_key_is_refused(self, tmp_path):
        from rcv.study import run_study

        config_path, config = self._config(tmp_path)
        config["split"]["stratum"] = ["verdict", "agreement"]
        config_path.write_text(yaml.safe_dump(config))
        with pytest.raises(ValueError, match="stratum"):
            run_study(config_path)

    def test_a_missing_section_is_refused(self, tmp_path):
        from rcv.study import run_study

        config_path, config = self._config(tmp_path)
        del config["monitor"]
        config_path.write_text(yaml.safe_dump(config))
        with pytest.raises(ValueError, match="monitor"):
            run_study(config_path)

    def test_degenerate_numeric_settings_are_refused(self, tmp_path):
        from rcv.study import run_study

        config_path, config = self._config(tmp_path)
        config["loop"]["cross_fit_folds"] = 0
        config_path.write_text(yaml.safe_dump(config))
        with pytest.raises(ValueError, match="cross_fit_folds"):
            run_study(config_path)

    def test_a_frame_digest_mismatch_is_refused(self, tmp_path):
        from rcv.study import run_study

        config_path, config = self._config(tmp_path)
        config["frame"]["sha256"] = "0" * 64
        config_path.write_text(yaml.safe_dump(config))
        with pytest.raises(ValueError, match=r"Frame SHA-256 mismatch: configuration declares "
                                             r"000000000000…; .* hashes to [0-9a-f]{12}…\.$"):
            run_study(config_path)

    def test_probe_hyperparameters_reach_the_component(self, tmp_path):
        from rcv.study import run_study

        config_path, config = self._config(tmp_path)
        config["estimator"]["probe"] = {"family": "linear", "C": 0.5}
        config_path.write_text(yaml.safe_dump(config))
        result = run_study(config_path)
        assert 0.0 <= result.readout.corrected_adherence <= 1.0


class TestB1ScoringAxes:
    def _swap_config(self, tmp_path):
        frame = build_fixture_frame()
        rng = np.random.default_rng(11)
        frame["second_axis"] = np.where(rng.random(len(frame["oracle"])) < 0.05,
                                        1 - frame["oracle"], frame["oracle"])
        frame_path = write_frame(tmp_path, frame)
        config_path = write_v1_config(tmp_path, frame_path)
        config = yaml.safe_load(config_path.read_text())
        config["frame"]["scoring_labels"] = ["oracle", "second_axis"]
        config_path.write_text(yaml.safe_dump(config))
        return config_path

    def test_every_named_axis_gets_a_readout_and_the_fit_is_shared(self, tmp_path):
        from rcv.study import run_study

        result = run_study(self._swap_config(tmp_path))
        assert set(result.readouts) == {"oracle", "second_axis"}
        assert result.readouts["oracle"] == result.readout
        assert result.readouts["second_axis"].label_set == "second_axis"
        assert (result.readouts["second_axis"].corrected_adherence
                != result.readout.corrected_adherence)

    def test_a_scoring_axis_absent_from_the_frame_is_refused(self, tmp_path):
        from rcv.study import run_study

        frame_path = write_frame(tmp_path, build_fixture_frame())
        config_path = write_v1_config(tmp_path, frame_path)
        config = yaml.safe_load(config_path.read_text())
        config["frame"]["scoring_labels"] = ["oracle", "human_gold"]
        config_path.write_text(yaml.safe_dump(config))
        with pytest.raises(ValueError, match="human_gold"):
            run_study(config_path)

    def test_r17_the_swap_axis_never_touches_the_fit(self, tmp_path):
        from rcv.study import run_study

        config_path = self._swap_config(tmp_path)
        with_swap = run_study(config_path)
        config = yaml.safe_load(config_path.read_text())
        del config["frame"]["scoring_labels"]
        config_path.write_text(yaml.safe_dump(config))
        without_swap = run_study(config_path)
        assert with_swap.readout == without_swap.readout


class TestA4FeatureSource:
    def test_the_classifier_score_can_feed_the_probe(self, tmp_path):
        from rcv.study import run_study

        frame_path = write_frame(tmp_path, build_fixture_frame())
        config_path = write_v1_config(tmp_path, frame_path)
        config = yaml.safe_load(config_path.read_text())
        config["estimator"]["feature_source"] = "clf_score"
        config_path.write_text(yaml.safe_dump(config))
        from_score = run_study(config_path)

        baseline = run_study(write_v1_config(tmp_path, frame_path))
        assert (from_score.readout.corrected_adherence
                != baseline.readout.corrected_adherence)

    def test_an_unknown_feature_source_is_refused(self, tmp_path):
        from rcv.study import run_study

        frame_path = write_frame(tmp_path, build_fixture_frame())
        config_path = write_v1_config(tmp_path, frame_path)
        config = yaml.safe_load(config_path.read_text())
        config["estimator"]["feature_source"] = "vibes"
        config_path.write_text(yaml.safe_dump(config))
        with pytest.raises(ValueError, match="vibes"):
            run_study(config_path)


class TestStreamPrefixCalibration:
    @staticmethod
    def _frame_with_drift_free_stream_head(head_rows=100):
        frame = build_fixture_frame()
        calm_positions = np.where(~frame["is_stream"])[0]
        frame["is_stream"] = frame["is_stream"].copy()
        frame["is_stream"][calm_positions[-head_rows:]] = True
        return frame

    def test_the_monitor_can_calibrate_on_the_streams_drift_free_prefix(self, tmp_path):
        from rcv.study import run_study

        frame_path = write_frame(tmp_path, self._frame_with_drift_free_stream_head())
        config_path = write_v1_config(tmp_path, frame_path)
        config = yaml.safe_load(config_path.read_text())
        config["monitor"]["calibration"]["material"] = "stream_prefix"
        config["monitor"]["calibration"]["prefix_length"] = 100
        config_path.write_text(yaml.safe_dump(config))
        result = run_study(config_path)
        assert result.monitor.deployed is not None
        alarms = [event for event in result.loop.events if event.kind == "alarm"]
        assert alarms, "the drift beyond the prefix must still alarm"
        assert result.change_log, "the repair cycle must complete"

    def test_a_prefix_longer_than_the_stream_is_refused(self, tmp_path):
        from rcv.study import run_study

        frame_path = write_frame(tmp_path, build_fixture_frame())
        config_path = write_v1_config(tmp_path, frame_path)
        config = yaml.safe_load(config_path.read_text())
        config["monitor"]["calibration"]["material"] = "stream_prefix"
        config["monitor"]["calibration"]["prefix_length"] = 10_000
        config_path.write_text(yaml.safe_dump(config))
        with pytest.raises(ValueError, match="prefix"):
            run_study(config_path)




def clf_score_config(tmp_path):
    config_path = write_v1_config(tmp_path, write_frame(tmp_path, build_fixture_frame()))
    config = yaml.safe_load(config_path.read_text())
    config["estimator"]["feature_source"] = "clf_score"
    config_path.write_text(yaml.safe_dump(config))
    return config_path


class TestBaselineRelativeAcceptanceBounds:
    @staticmethod
    def _relative(tmp_path, acceptance):
        config_path = write_v1_config(tmp_path, write_frame(tmp_path, build_fixture_frame()))
        config = yaml.safe_load(config_path.read_text())
        config["loop"]["acceptance"] = {"form": "baseline_relative", **acceptance}
        config_path.write_text(yaml.safe_dump(config))
        return config_path

    def test_both_ends_of_the_unit_interval_are_legal(self, tmp_path):
        from rcv.study import run_study

        run_study(self._relative(tmp_path, {"recall_slack": 0.0,
                                            "over_block_inflation": 1.0}))

    def test_a_knob_the_configuration_omits_refuses_in_full(self, tmp_path):
        from rcv.study import run_study

        with pytest.raises(ValueError, match=(
                r"^baseline_relative acceptance\.over_block_inflation must be in \[0, 1\]; "
                r"got -1\.$")):
            run_study(self._relative(tmp_path, {"recall_slack": 0.1}))

    def test_a_knob_above_one_refuses(self, tmp_path):
        from rcv.study import run_study

        with pytest.raises(ValueError, match=(
                r"baseline_relative acceptance\.recall_slack must be in \[0, 1\]; "
                r"got 1\.5\.$")):
            run_study(self._relative(tmp_path, {"recall_slack": 1.5,
                                                "over_block_inflation": 0.1}))


class TestTheBaselineTheLoopComparesAgainst:
    def test_the_corrected_system_blocks_unsafe_items_more_often_than_safe_ones(
            self, tmp_path, monkeypatch):
        import rcv.study as study_module
        from rcv.study import run_study

        seen = {}
        real_watch = study_module.watch_traffic

        def recording_watch(*arguments, **keywords):
            seen.update(keywords)
            return real_watch(*arguments, **keywords)

        monkeypatch.setattr(study_module, "watch_traffic", recording_watch)
        run_study(clf_score_config(tmp_path), 1)

        baseline = seen["baseline"]
        assert 0.5 < baseline["recall"] < 1.0
        assert 0.0 < baseline["over_block"] < 0.5


class TestTheReadoutsArithmetic:
    def test_raw_adherence_is_agreement_with_the_label_set_not_disagreement(self, study_result):
        assert study_result.readout.raw_adherence > 0.5

    def test_the_caught_shares_population_is_the_items_it_is_a_share_of(self, tmp_path):
        from rcv.study import run_study

        readout = run_study(clf_score_config(tmp_path), 1).readout
        numerator = readout.caught_share * readout.caught_share_population
        assert numerator == round(numerator)
        assert 0 < readout.caught_share_population < readout.oracle_labels_spent

    def test_a_result_names_the_seed_it_was_drawn_at_on_both_paths(self, tmp_path):
        from rcv.study import run_study

        frame = build_fixture_frame()
        streamless = {name: values[~frame["is_stream"]] for name, values in frame.items()}
        streamed_path = write_frame(tmp_path, frame, "streamed.npz")
        streamless_path = write_frame(tmp_path, streamless, "streamless.npz")
        with_stream = yaml.safe_load(
            write_v1_config(tmp_path, streamed_path).read_text())
        without = yaml.safe_load(
            write_v1_config(tmp_path, streamless_path).read_text())
        assert run_study(with_stream, 3).seed == 3
        assert run_study(without, 5).seed == 5

    def test_a_streamless_result_still_carries_its_scoring_axes(self, tmp_path):
        from rcv.study import run_study

        frame = build_fixture_frame()
        streamless = {name: values[~frame["is_stream"]] for name, values in frame.items()}
        result = run_study(write_v1_config(tmp_path, write_frame(tmp_path, streamless)))
        assert set(result.readouts) == {"oracle"}
        assert result.readouts["oracle"] == result.readout

    def test_a_fitting_axis_absent_from_the_scoring_axes_refuses_in_full(self, tmp_path):
        from rcv.study import run_study

        frame = build_fixture_frame()
        frame["human"] = frame["oracle"].copy()
        config_path = write_v1_config(tmp_path, write_frame(tmp_path, frame))
        config = yaml.safe_load(config_path.read_text())
        config["frame"]["scoring_labels"] = ["human"]
        config_path.write_text(yaml.safe_dump(config))
        with pytest.raises(ValueError, match=(
                r"^Fitting label set 'oracle' must appear in scoring_labels\.$")):
            run_study(config_path)


class TestTheSeedGovernsTheFit:
    def test_the_draws_seed_reaches_every_estimator_the_study_builds(self, tmp_path,
                                                                     monkeypatch):
        import rcv.study as study_module
        from rcv.study import run_study

        seen = []
        real_estimator = study_module.Estimator

        class RecordingEstimator(real_estimator):
            def __init__(self, *arguments, **keywords):
                seen.append(keywords.get("random_state", "absent"))
                super().__init__(*arguments, **keywords)

        monkeypatch.setattr(study_module, "Estimator", RecordingEstimator)
        run_study(write_v1_config(tmp_path, write_frame(tmp_path, build_fixture_frame())), 7)

        assert seen and set(seen) == {7}


class TestConfigFloorsAreTheValuesTheyName:
    @staticmethod
    def _config():
        return {
            "study": "floors", "seeds": [0],
            "frame": {"path": "f.npz", "name": "n", "fitting_labels": "oracle"},
            "split": {"evaluation_fraction": 0.3, "calibration_fraction": 0.25},
            "estimator": {"probe": {"family": "linear"}, "calibration": {"method": "platt"},
                          "route_probe": True, "route_calibration": True},
            "flip": {"threshold": 0.5},
            "monitor": {"quiet_horizon_confidence": 0.95,
                        "detectable_shift": {"indifference_zone_quantile": 0.99},
                        "event_bank": {"boundaries": [0.5], "budget_allocation": "joint"},
                        "calibration": {"stream_length": 1, "n_streams": 1, "streams_per_centre": 1}},
            "loop": {"audit_sampling_window": 1, "audit_budget": 1,
                     "acceptance": {"recall_floor": 0.5, "false_positive_tolerance": 0.25},
                     "max_enlarging_retries": 0, "cross_fit_folds": 2},
        }

    def test_a_configuration_sitting_on_every_floor_is_accepted(self):
        from rcv.study import validate_config

        validate_config(self._config())

    def test_a_loop_setting_below_its_floor_refuses_in_full(self):
        from rcv.study import validate_config

        for name, below in (("cross_fit_folds", 1), ("audit_sampling_window", 0),
                           ("audit_budget", 0), ("max_enlarging_retries", -1)):
            config = self._config()
            floor = config["loop"][name]
            config["loop"][name] = below
            expected = (r"^loop\.max_enlarging_retries must be nonnegative; got -1\.$"
                        if name == "max_enlarging_retries"
                        else rf"^loop\.{name} must be at least {floor}; got {below}\.$")
            with pytest.raises(ValueError,
                               match=expected):
                validate_config(config)

    def test_a_monitor_calibration_below_one_refuses_in_full(self):
        from rcv.study import validate_config

        for name in ("stream_length", "n_streams"):
            config = self._config()
            config["monitor"]["calibration"][name] = 0
            with pytest.raises(
                    ValueError,
                    match=rf"^monitor\.calibration\.{name} must be at least 1; got 0\.$"):
                validate_config(config)


class TestTheStreamPrefixIsSliced:
    def test_the_prefix_is_the_streams_head_and_the_watch_starts_where_it_ends(
            self, tmp_path, monkeypatch):
        import rcv.study as study_module
        from rcv.belief_bank import BeliefBankMonitor
        from rcv.study import run_study

        prefix_length = 100
        frame = TestStreamPrefixCalibration._frame_with_drift_free_stream_head(prefix_length)
        stream_rows = int(frame["is_stream"].sum())
        calibrated_on, watched = [], {}

        real_calibrate = BeliefBankMonitor.calibrate

        def recording_calibrate(self, probability, verdicts, stream_length, n_streams, seed,
                                **keywords):
            calibrated_on.append({"rows": len(probability), "seed": seed})
            return real_calibrate(self, probability, verdicts, stream_length, n_streams, seed,
                                  **keywords)

        real_watch = study_module.watch_traffic

        def recording_watch(*arguments, **keywords):
            watched["rows"] = len(arguments[4]["verdict"])
            return real_watch(*arguments, **keywords)

        monkeypatch.setattr(BeliefBankMonitor, "calibrate", recording_calibrate)
        monkeypatch.setattr(study_module, "watch_traffic", recording_watch)

        config_path = write_v1_config(tmp_path, write_frame(tmp_path, frame))
        config = yaml.safe_load(config_path.read_text())
        config["monitor"]["calibration"]["material"] = "stream_prefix"
        config["monitor"]["calibration"]["prefix_length"] = prefix_length
        config_path.write_text(yaml.safe_dump(config))
        run_study(config_path, 4)

        assert calibrated_on[0] == {"rows": prefix_length, "seed": 4}
        assert watched["rows"] == stream_rows - prefix_length

    def test_a_prefix_that_consumes_the_whole_stream_is_refused(self, tmp_path):
        from rcv.study import run_study

        frame = TestStreamPrefixCalibration._frame_with_drift_free_stream_head(100)
        stream_rows = int(frame["is_stream"].sum())
        config_path = write_v1_config(tmp_path, write_frame(tmp_path, frame))
        config = yaml.safe_load(config_path.read_text())
        config["monitor"]["calibration"]["material"] = "stream_prefix"
        config["monitor"]["calibration"]["prefix_length"] = stream_rows
        config_path.write_text(yaml.safe_dump(config))
        with pytest.raises(ValueError, match=(
                rf"^prefix_length must be smaller than stream length; got {stream_rows} for "
                rf"{stream_rows} rows\.$")):
            run_study(config_path)

    def test_an_unknown_calibration_material_refuses_in_full(self, tmp_path):
        from rcv.study import run_study

        config_path = write_v1_config(tmp_path, write_frame(tmp_path, build_fixture_frame()))
        config = yaml.safe_load(config_path.read_text())
        config["monitor"]["calibration"]["material"] = "hearsay"
        config_path.write_text(yaml.safe_dump(config))
        with pytest.raises(ValueError,
                           match=r"^Unknown monitor\.calibration\.material 'hearsay'\.$"):
            run_study(config_path)


class TestTheQuietHorizonConfidenceIsConfigured:
    def test_a_configuration_naming_it_validates(self):
        from rcv.study import validate_config

        config = TestConfigFloorsAreTheValuesTheyName._config()
        config["monitor"]["quiet_horizon_confidence"] = 0.9
        validate_config(config)

    @pytest.mark.parametrize("q", [0.0, 1.0, 1.5])
    def test_a_confidence_outside_the_unit_interval_refuses_in_full(self, q):
        from rcv.study import validate_config

        config = TestConfigFloorsAreTheValuesTheyName._config()
        config["monitor"]["quiet_horizon_confidence"] = q
        with pytest.raises(ValueError, match="quiet_horizon_confidence"):
            validate_config(config)

    def test_the_configured_confidence_reaches_the_monitor_as_its_complement(self, tmp_path,
                                                                            monkeypatch):
        from rcv.belief_bank import BeliefBankMonitor
        from rcv.study import run_study

        seen = []
        constructed = BeliefBankMonitor.__init__

        def recording(self, calibration_level, detectable_shift, **keywords):
            seen.append(calibration_level)
            constructed(self, calibration_level, detectable_shift, **keywords)

        monkeypatch.setattr(BeliefBankMonitor, "__init__", recording)
        config_path = write_v1_config(tmp_path, write_frame(tmp_path, build_fixture_frame()))
        config = yaml.safe_load(config_path.read_text())
        config["monitor"]["quiet_horizon_confidence"] = 0.9
        config_path.write_text(yaml.safe_dump(config))
        run_study(config_path, 0)
        assert seen == [0.1]


class TestTheDetectableShiftIsConfigured:
    @pytest.mark.parametrize("quantile", [0.75, 0.99])
    def test_the_one_form_validates(self, quantile):
        from rcv.study import validate_config

        config = TestConfigFloorsAreTheValuesTheyName._config()
        config["monitor"]["detectable_shift"] = {"indifference_zone_quantile": quantile}
        validate_config(config)

    @pytest.mark.parametrize("shift,expected", [
        (0.1, "must be a mapping"),
        ({"mode": "relative", "factor": 0.5}, "no longer accepts"),
        ({"baseline_credibility": 0.99}, "indifference_zone_quantile"),
        ({"indifference_zone_quantile": 0.99, "cap": 1}, "Unknown detectable_shift keys"),
        ({}, "missing required key"),
        ({"indifference_zone_quantile": 0.5}, "greater than 0.5 and less than 1.0"),
    ])
    def test_a_malformed_shift_refuses(self, shift, expected):
        from rcv.study import validate_config

        config = TestConfigFloorsAreTheValuesTheyName._config()
        config["monitor"]["detectable_shift"] = shift
        with pytest.raises(ValueError, match=expected):
            validate_config(config)

    def test_the_configured_shift_reaches_the_monitor(self, tmp_path, monkeypatch):
        from rcv.belief_bank import BeliefBankMonitor
        from rcv.study import run_study

        seen = []
        constructed = BeliefBankMonitor.__init__

        def recording(self, calibration_level, detectable_shift, **keywords):
            seen.append(detectable_shift)
            constructed(self, calibration_level, detectable_shift, **keywords)

        monkeypatch.setattr(BeliefBankMonitor, "__init__", recording)
        config_path = write_v1_config(tmp_path, write_frame(tmp_path, build_fixture_frame()))
        config = yaml.safe_load(config_path.read_text())
        config["monitor"]["detectable_shift"] = {"indifference_zone_quantile": 0.75}
        config["loop"]["observe_only"] = True
        config_path.write_text(yaml.safe_dump(config))
        run_study(config_path, 0)
        assert seen == [{"indifference_zone_quantile": 0.75}]
