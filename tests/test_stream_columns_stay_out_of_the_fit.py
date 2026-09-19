import numpy as np
import pytest
import yaml

from fixture_frame import FIXTURE_REFERENCE_WINDOW, build_fixture_frame


def frame_with_stream_columns(poisoned):
    frame = build_fixture_frame()
    rows = len(frame["verdict"])
    on_stream = frame["is_stream"]
    schedule = np.where(on_stream, np.linspace(0.0, 0.3, rows), 0.0)
    drifted = on_stream & (np.arange(rows) % 3 == 0)
    if poisoned:
        return dict(frame, is_drift_item=~drifted, stream_lambda=np.full(rows, np.nan))
    return dict(frame, is_drift_item=drifted, stream_lambda=schedule)


def written_study(tmp_path, name, frame):
    frame_path = tmp_path / f"{name}.npz"
    np.savez(frame_path, **frame)
    config = {
        "study": "p33-separation",
        "seeds": [0],
        "frame": {"path": str(frame_path), "name": "synthetic-fixture",
                  "fitting_labels": "oracle"},
        "split": {"evaluation_fraction": 0.3, "calibration_fraction": 0.25},
        "estimator": {"probe": {"family": "linear"}, "calibration": {"method": "platt"},
                      "route_probe": True, "route_calibration": True},
        "flip": {"threshold": 0.5},
        "monitor": {"quiet_horizon_confidence": 0.95,
                    "detectable_shift": {"indifference_zone_quantile": 0.99},
                    "event_bank": {"boundaries": [0.5], "budget_allocation": "joint"},
                    "calibration": {"stream_length": 120, "n_streams": 2000, "streams_per_centre": 1}},
        "loop": {"audit_sampling_window": 120, "audit_budget": 60,
                 "acceptance": {"recall_floor": 0.5, "false_positive_tolerance": 0.25},
                 "max_enlarging_retries": 2, "cross_fit_folds": 4,
                 "post_repair_reference_window": FIXTURE_REFERENCE_WINDOW},
    }
    config_path = tmp_path / f"{name}.yaml"
    config_path.write_text(yaml.safe_dump(config))
    return config_path


def without_the_composition_measurement(entry):
    from dataclasses import replace

    if entry.calibration is None:
        return entry
    return replace(entry, calibration=replace(entry.calibration, fitted_family_overlap=None))


def everything_the_study_returns(result):
    calibration = result.monitor.deployed
    return (result.readout,
            sorted(result.readouts.items()),
            result.monitor.deployed.reference,
            None if calibration is None else (calibration.threshold,
                                              calibration.realized_alarm_rate_replayed,
                                              calibration.realized_alarm_rate_combined_replayed,
                                              calibration.detectable_shift),
            result.loop.events,
            [without_the_composition_measurement(entry) for entry in result.change_log],
            result.total_oracle_labels_spent)


@pytest.fixture(scope="module")
def arms(tmp_path_factory):
    from rcv.study import run_study

    tmp_path = tmp_path_factory.mktemp("p33")
    return {name: run_study(written_study(tmp_path, name, frame))
            for name, frame in (("absent", build_fixture_frame()),
                                ("honest", frame_with_stream_columns(poisoned=False)),
                                ("poisoned", frame_with_stream_columns(poisoned=True)))}


class TestP33TheRecordedCompositionIsNeverRead:
    def test_the_poison_survives_the_round_trip_so_the_comparison_is_not_vacuous(self,
                                                                                tmp_path):
        from rcv.frames import load_frame

        path = tmp_path / "poisoned.npz"
        np.savez(path, **frame_with_stream_columns(poisoned=True))
        loaded = load_frame(path)
        assert np.isnan(loaded["stream_lambda"]).all()
        assert loaded["is_drift_item"].dtype == np.bool_
        honest = frame_with_stream_columns(poisoned=False)
        assert np.array_equal(loaded["is_drift_item"], ~honest["is_drift_item"])

    def test_poisoning_the_two_stream_columns_changes_nothing_the_study_reports(self, arms):
        assert (everything_the_study_returns(arms["poisoned"])
                == everything_the_study_returns(arms["honest"]))

    def test_admitting_the_two_columns_at_all_changes_nothing_the_study_reports(self, arms):
        assert (everything_the_study_returns(arms["honest"])
                == everything_the_study_returns(arms["absent"]))

    def test_the_fitted_probe_scores_every_row_identically_across_the_three_frames(self):
        from fixture_frame import split_for_fitting
        from rcv.estimator import Estimator
        from rcv.frames import fit_on_slices

        scored = {}
        for name, frame in (("absent", build_fixture_frame()),
                            ("honest", frame_with_stream_columns(poisoned=False)),
                            ("poisoned", frame_with_stream_columns(poisoned=True))):
            fitting, calibration = split_for_fitting(frame)
            estimator = fit_on_slices(
                Estimator(probe={"family": "linear"}, calibration={"method": "platt"},
                          route_probe=True, route_calibration=True, random_state=0),
                fitting, calibration, "oracle")
            scored[name] = estimator.probability_of_agreement(frame["representation"],
                                                              frame["verdict"])

        assert np.isfinite(scored["honest"]).all()
        np.testing.assert_array_equal(scored["poisoned"], scored["honest"])
        np.testing.assert_array_equal(scored["absent"], scored["honest"])

    def test_the_columns_move_exactly_one_recorded_field_and_it_is_the_one_they_exist_for(
            self, arms):
        honest = arms["honest"].monitor.deployed.fitted_family_overlap
        absent = arms["absent"].monitor.deployed.fitted_family_overlap

        assert honest != absent
        assert honest["base"] is not None and honest["drift"] is not None
        assert absent["base"] is None and absent["drift"] is None
        assert honest["all"] == absent["all"]

    def test_the_arms_are_a_real_study_rather_than_an_empty_one(self, arms):
        result = arms["honest"]
        assert result.monitor.deployed is not None
        assert result.readout.corrected_adherence > result.readout.raw_adherence
        assert result.loop.events
