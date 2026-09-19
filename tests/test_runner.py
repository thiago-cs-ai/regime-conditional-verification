import hashlib
import json
import statistics

import numpy as np
import pytest
import yaml

from fixture_frame import FIXTURE_REFERENCE_WINDOW, build_fixture_frame

HAND_COMPUTABLE_DRAWS = {
    0: (0.5, 0.80, 0.2, 100, 100),
    1: (0.6, 0.90, 0.4, 100, 110),
    2: (0.7, 1.00, 0.9, 100, 150),
}


def write_frame(tmp_path, frame, name="fixture_frame.npz"):
    path = tmp_path / name
    np.savez(path, **frame)
    return path


def write_stub_frame(tmp_path):
    return write_frame(tmp_path, {"verdict": np.arange(4)}, name="stub_frame.npz")


def protocol_config(frame_path, seeds):
    return {
        "study": "i3-protocol",
        "seeds": list(seeds),
        "frame": {"path": str(frame_path), "name": "synthetic-fixture",
                  "fitting_labels": "oracle"},
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


def study_result_for(seed, raw, corrected, caught, readout_labels, total_labels,
                     frame_name="synthetic-fixture", label_set="oracle"):
    from rcv.belief_bank import BeliefBankCalibration
    from rcv.loop import ChangeLogEntry, LoopEvent
    from rcv.study import LoopState, MonitorState, Readout, StudyResult

    calibration = BeliefBankCalibration(
        reference={0: {0.5: 0.10}, 1: {0.5: 0.20}}, threshold={0: {0.5: 3.0}, 1: {0.5: 4.0}},
        realized_alarm_rate_replayed={0: {0.5: 0.02}, 1: {0.5: 0.03}},
        realized_alarm_rate_combined_replayed=0.05,
        realized_alarm_rate_at_reference={0: {0.5: 0.01}, 1: {0.5: 0.02}},
        realized_alarm_rate_combined_at_reference=0.03,
        replay_stream_length=120, monitored_length=240, n_streams=400, seed=seed,
        boundaries=(0.5,))
    return StudyResult(
        seed=seed,
        readout=Readout(raw_adherence=raw, corrected_adherence=corrected, caught_share_population=21, caught_share=caught,
                        auroc=0.9, ece=0.01,
                        frame_name=frame_name, label_set=label_set,
                        oracle_labels_spent=readout_labels),
        monitor=MonitorState(deployed=calibration, recalibrations=[], final=calibration),
        loop=LoopState(events=[LoopEvent("alarm", f"regime 0, boundary 0.5, at item {seed}")]),
        change_log=[ChangeLogEntry(what="repair", trigger="alarm in regime 0 at boundary 0.5",
                                   oracle_labels_spent=total_labels - readout_labels)],
        total_oracle_labels_spent=total_labels)


def stub_the_study(monkeypatch, draws, frame_name_by_seed=None, label_set_by_seed=None):
    def fake_run_study(config, seed):
        return study_result_for(
            seed, *draws[seed],
            frame_name=(frame_name_by_seed or {}).get(seed, "synthetic-fixture"),
            label_set=(label_set_by_seed or {}).get(seed, "oracle"))

    monkeypatch.setattr("rcv.runner.run_study", fake_run_study)


def series_of(draws, position):
    return [draws[seed][position] for seed in sorted(draws)]


@pytest.fixture(scope="module")
def two_runs(tmp_path_factory):
    from rcv.runner import run_protocol

    tmp_path = tmp_path_factory.mktemp("determinism")
    config = protocol_config(write_frame(tmp_path, build_fixture_frame()), [0, 1, 2])
    first, second = tmp_path / "first", tmp_path / "second"
    run_protocol(config, first)
    run_protocol(config, second)
    return first, second


@pytest.fixture(scope="module")
def protocol_aggregate(tmp_path_factory):
    from rcv.runner import run_protocol

    tmp_path = tmp_path_factory.mktemp("protocol")
    config = protocol_config(write_frame(tmp_path, build_fixture_frame()), [0, 1, 2])
    return run_protocol(config, tmp_path / "results")


class TestR07Aggregation:
    @pytest.fixture
    def aggregate(self, tmp_path, monkeypatch):
        from rcv.runner import run_protocol

        stub_the_study(monkeypatch, HAND_COMPUTABLE_DRAWS)
        config = protocol_config(write_stub_frame(tmp_path), sorted(HAND_COMPUTABLE_DRAWS))
        return run_protocol(config, tmp_path / "results")

    @pytest.mark.parametrize("metric,position", [("raw_adherence", 0),
                                                 ("corrected_adherence", 1),
                                                 ("caught_share", 2),
                                                 ("total_oracle_labels_spent", 4)])
    def test_the_mean_is_the_mean_over_the_draws(self, aggregate, metric, position):
        drawn = series_of(HAND_COMPUTABLE_DRAWS, position)
        assert aggregate["metrics"][metric]["mean"] == pytest.approx(statistics.fmean(drawn))

    @pytest.mark.parametrize("metric,position", [("raw_adherence", 0),
                                                 ("corrected_adherence", 1),
                                                 ("caught_share", 2),
                                                 ("total_oracle_labels_spent", 4)])
    def test_the_spread_is_the_sample_standard_deviation_not_the_population_one(self, aggregate,
                                                                               metric, position):
        drawn = series_of(HAND_COMPUTABLE_DRAWS, position)
        assert aggregate["metrics"][metric]["sd"] == pytest.approx(statistics.stdev(drawn))
        assert aggregate["metrics"][metric]["sd"] != pytest.approx(statistics.pstdev(drawn))

    @pytest.mark.parametrize("metric", ["raw_adherence", "corrected_adherence", "caught_share",
                                        "total_oracle_labels_spent"])
    def test_the_count_of_draws_is_stated_beside_every_value(self, aggregate, metric):
        assert aggregate["metrics"][metric]["n"] == len(HAND_COMPUTABLE_DRAWS)

    @pytest.mark.parametrize("metric,position", [("raw_adherence", 0),
                                                 ("caught_share", 2),
                                                 ("total_oracle_labels_spent", 4)])
    def test_the_observed_range_is_bracketed_and_labelled_as_a_range(self, aggregate, metric,
                                                                     position):
        drawn = series_of(HAND_COMPUTABLE_DRAWS, position)
        assert aggregate["metrics"][metric]["observed_range"] == [min(drawn), max(drawn)]

    def test_a_single_draw_records_no_spread_rather_than_a_spread_of_zero(self, tmp_path,
                                                                         monkeypatch):
        from rcv.runner import run_protocol

        stub_the_study(monkeypatch, HAND_COMPUTABLE_DRAWS)
        config = protocol_config(write_stub_frame(tmp_path), [1])
        aggregate = run_protocol(config, tmp_path / "one_draw")

        for metric in aggregate["metrics"].values():
            assert metric["n"] == 1
            assert metric["sd"] is None

    def test_two_draws_are_already_enough_for_a_sample_standard_deviation(self, tmp_path,
                                                                          monkeypatch):
        from rcv.runner import run_protocol

        stub_the_study(monkeypatch, HAND_COMPUTABLE_DRAWS)
        aggregate = run_protocol(protocol_config(write_stub_frame(tmp_path), [0, 1]),
                                 tmp_path / "two_draws")

        expected = statistics.stdev([HAND_COMPUTABLE_DRAWS[0][0], HAND_COMPUTABLE_DRAWS[1][0]])
        assert aggregate["metrics"]["raw_adherence"]["sd"] == pytest.approx(expected)

    def test_a_repeated_seed_is_refused_rather_than_counted_twice(self, tmp_path, monkeypatch):
        from rcv.runner import run_protocol

        stub_the_study(monkeypatch, HAND_COMPUTABLE_DRAWS)
        config = protocol_config(write_stub_frame(tmp_path), [0, 1, 1])
        with pytest.raises(ValueError, match="Configured seeds must be unique"):
            run_protocol(config, tmp_path / "repeated")

    def test_an_empty_seeds_list_is_refused(self, tmp_path, monkeypatch):
        from rcv.runner import run_protocol

        stub_the_study(monkeypatch, HAND_COMPUTABLE_DRAWS)
        config = protocol_config(write_stub_frame(tmp_path), [])
        with pytest.raises(ValueError, match=r"^Configuration must name at least one seed\.$"):
            run_protocol(config, tmp_path / "empty")

    def test_the_aggregate_names_the_study_and_counts_its_draws(self, aggregate):
        assert aggregate["study"] == "i3-protocol"
        assert aggregate["n"] == len(HAND_COMPUTABLE_DRAWS)
        assert aggregate["seeds"] == sorted(HAND_COMPUTABLE_DRAWS)


class TestR09Naming:
    def test_the_aggregate_names_its_frame_label_set_and_labels_spent(self, tmp_path, monkeypatch):
        from rcv.runner import run_protocol

        stub_the_study(monkeypatch, HAND_COMPUTABLE_DRAWS)
        seeds = sorted(HAND_COMPUTABLE_DRAWS)
        aggregate = run_protocol(protocol_config(write_stub_frame(tmp_path), seeds),
                                 tmp_path / "results")

        assert aggregate["frame_name"] == "synthetic-fixture"
        assert aggregate["label_set"] == "oracle"
        spent = series_of(HAND_COMPUTABLE_DRAWS, 3)
        assert aggregate["oracle_labels_spent"]["per_seed"] == {
            str(seed): value for seed, value in zip(seeds, spent)}
        assert aggregate["oracle_labels_spent"]["mean"] == pytest.approx(statistics.fmean(spent))

    def test_draws_disagreeing_on_the_frame_cannot_be_aggregated_under_one_name(self, tmp_path,
                                                                                monkeypatch):
        from rcv.runner import run_protocol

        stub_the_study(monkeypatch, HAND_COMPUTABLE_DRAWS, frame_name_by_seed={2: "another-frame"})
        output_dir = tmp_path / "results"
        with pytest.raises(ValueError, match="frame"):
            run_protocol(protocol_config(write_stub_frame(tmp_path),
                                         sorted(HAND_COMPUTABLE_DRAWS)), output_dir)

        assert (output_dir / "seed_2.json").exists(), "each draw is still written as generated"
        assert not (output_dir / "aggregate.json").exists()

    def test_draws_scored_against_different_label_sets_cannot_be_aggregated(self, tmp_path,
                                                                            monkeypatch):
        from rcv.runner import run_protocol

        stub_the_study(monkeypatch, HAND_COMPUTABLE_DRAWS, label_set_by_seed={2: "human_gold"})
        with pytest.raises(ValueError, match="label set"):
            run_protocol(protocol_config(write_stub_frame(tmp_path),
                                         sorted(HAND_COMPUTABLE_DRAWS)), tmp_path / "results")


class TestR11WrittenOnceWithDigests:
    @pytest.fixture
    def written(self, tmp_path, monkeypatch):
        from rcv.runner import run_protocol

        stub_the_study(monkeypatch, HAND_COMPUTABLE_DRAWS)
        frame_path = write_stub_frame(tmp_path)
        config = protocol_config(frame_path, sorted(HAND_COMPUTABLE_DRAWS))
        output_dir = tmp_path / "results"
        run_protocol(config, output_dir)
        return config, frame_path, output_dir

    def test_one_file_per_draw_and_one_aggregate(self, written):
        _, _, output_dir = written
        assert sorted(path.name for path in output_dir.iterdir()) == [
            "aggregate.json", "seed_0.json", "seed_1.json", "seed_2.json"]

    @pytest.mark.parametrize("name", ["seed_0.json", "aggregate.json"])
    def test_every_file_echoes_the_configuration_whole(self, written, name):
        config, _, output_dir = written
        written_config = json.loads((output_dir / name).read_text())["provenance"]["config"]
        assert written_config == config

    @pytest.mark.parametrize("name", ["seed_0.json", "aggregate.json"])
    def test_every_file_carries_the_frame_digest(self, written, name):
        _, frame_path, output_dir = written
        expected = hashlib.sha256(frame_path.read_bytes()).hexdigest()
        assert json.loads((output_dir / name).read_text())["provenance"]["frame_sha256"] == expected

    @pytest.mark.parametrize("name", ["seed_0.json", "aggregate.json"])
    def test_every_file_carries_the_environment(self, written, name):
        import platform

        import scipy
        import sklearn

        _, _, output_dir = written
        environment = json.loads((output_dir / name).read_text())["provenance"]["environment"]
        assert environment == {"python": platform.python_version(), "numpy": np.__version__,
                               "scipy": scipy.__version__, "scikit-learn": sklearn.__version__}

    def test_the_seed_is_on_its_own_result_and_the_seed_list_on_the_aggregate(self, written):
        _, _, output_dir = written
        assert json.loads((output_dir / "seed_1.json").read_text())["seed"] == 1
        assert json.loads((output_dir / "aggregate.json").read_text())["seeds"] == [0, 1, 2]

    def test_a_draw_record_carries_every_named_part_of_what_the_study_produced(self, written):
        _, _, output_dir = written
        record = json.loads((output_dir / "seed_1.json").read_text())
        assert set(record) == {"seed", "readout", "readouts", "monitor", "loop_events",
                               "change_log", "total_oracle_labels_spent", "provenance"}
        assert record["monitor"]["streamless"] is False
        assert record["monitor"]["deployed"]["threshold"] == [
            {"regime": 0, "boundary": 0.5, "value": 3.0},
            {"regime": 1, "boundary": 0.5, "value": 4.0}]
        assert [event["kind"] for event in record["loop_events"]] == ["alarm"]
        assert [entry["what"] for entry in record["change_log"]] == ["repair"]

    def test_the_provenance_digests_the_configuration_as_it_is_written(self, written):
        from rcv.runner import as_written

        config, _, output_dir = written
        digest = json.loads((output_dir / "seed_0.json").read_text())["provenance"]["config_sha256"]
        assert digest == hashlib.sha256(as_written(config).encode("utf-8")).hexdigest()

    def test_each_draw_is_written_before_the_next_is_run(self, tmp_path, monkeypatch):
        from rcv.runner import run_protocol

        output_dir = tmp_path / "results"
        seen_when_each_draw_started = {}

        def fake_run_study(config, seed):
            seen_when_each_draw_started[seed] = sorted(
                path.name for path in output_dir.iterdir()) if output_dir.exists() else []
            return study_result_for(seed, *HAND_COMPUTABLE_DRAWS[seed])

        monkeypatch.setattr("rcv.runner.run_study", fake_run_study)
        run_protocol(protocol_config(write_stub_frame(tmp_path), sorted(HAND_COMPUTABLE_DRAWS)),
                     output_dir)

        assert seen_when_each_draw_started[0] == []
        assert seen_when_each_draw_started[1] == ["seed_0.json"]
        assert seen_when_each_draw_started[2] == ["seed_0.json", "seed_1.json"]

    def test_an_existing_draw_result_refuses_and_overwrites_nothing(self, tmp_path, monkeypatch):
        from rcv.runner import ResultExistsError, run_protocol

        stub_the_study(monkeypatch, HAND_COMPUTABLE_DRAWS)
        output_dir = tmp_path / "results"
        output_dir.mkdir()
        standing = "the result of an earlier run\n"
        (output_dir / "seed_1.json").write_text(standing)

        with pytest.raises(ResultExistsError, match="seed_1.json"):
            run_protocol(protocol_config(write_stub_frame(tmp_path),
                                         sorted(HAND_COMPUTABLE_DRAWS)), output_dir)

        assert (output_dir / "seed_1.json").read_text() == standing
        assert not (output_dir / "seed_0.json").exists()
        assert not (output_dir / "aggregate.json").exists()

    def test_an_existing_aggregate_refuses_and_overwrites_nothing(self, tmp_path, monkeypatch):
        from rcv.runner import ResultExistsError, run_protocol

        stub_the_study(monkeypatch, HAND_COMPUTABLE_DRAWS)
        output_dir = tmp_path / "results"
        output_dir.mkdir()
        standing = "the aggregate of an earlier run\n"
        (output_dir / "aggregate.json").write_text(standing)

        with pytest.raises(ResultExistsError, match="aggregate.json"):
            run_protocol(protocol_config(write_stub_frame(tmp_path),
                                         sorted(HAND_COMPUTABLE_DRAWS)), output_dir)

        assert (output_dir / "aggregate.json").read_text() == standing
        assert not (output_dir / "seed_0.json").exists()

    def test_the_write_itself_refuses_a_path_that_already_holds_a_result(self, tmp_path):
        from rcv.runner import ResultExistsError, write_once

        path = tmp_path / "already_there.json"
        standing = "a result written earlier\n"
        path.write_text(standing)

        with pytest.raises(ResultExistsError, match="Result destination already exists"):
            write_once(path, {"anything": 1})
        assert path.read_text() == standing


class TestOneYamlFullySpecifiesTheRun:
    def test_the_runner_reads_its_configuration_from_the_yaml_path_it_is_given(self, tmp_path,
                                                                               monkeypatch):
        from rcv.runner import run_protocol

        stub_the_study(monkeypatch, HAND_COMPUTABLE_DRAWS)
        config = protocol_config(write_stub_frame(tmp_path), sorted(HAND_COMPUTABLE_DRAWS))
        config_path = tmp_path / "run.yaml"
        config_path.write_text(yaml.safe_dump(config))

        aggregate = run_protocol(config_path, tmp_path / "results")
        assert aggregate["provenance"]["config"] == config
        assert aggregate["seeds"] == sorted(HAND_COMPUTABLE_DRAWS)

    def test_the_output_directory_is_created_with_the_parents_it_needs(self, tmp_path,
                                                                       monkeypatch):
        from rcv.runner import run_protocol

        stub_the_study(monkeypatch, HAND_COMPUTABLE_DRAWS)
        nested = tmp_path / "runs" / "2026-07" / "results"
        run_protocol(protocol_config(write_stub_frame(tmp_path), [0]), nested)
        assert (nested / "aggregate.json").exists()


class TestTheOneSerialisation:
    def test_the_written_form_is_key_sorted_two_space_indented_and_newline_terminated(self):
        from rcv.runner import as_written

        assert as_written({"b": 1, "a": {"d": 2, "c": 3}}) == (
            '{\n  "a": {\n    "c": 3,\n    "d": 2\n  },\n  "b": 1\n}\n')


class TestR10Determinism:
    @pytest.mark.parametrize("name", ["seed_0.json", "seed_1.json", "seed_2.json",
                                      "aggregate.json"])
    def test_two_runs_into_fresh_directories_are_byte_identical(self, two_runs, name):
        first, second = two_runs
        assert (first / name).read_bytes() == (second / name).read_bytes()

    def test_the_seed_governs_the_draw_so_two_seeds_differ(self, two_runs):
        first, _ = two_runs
        assert (first / "seed_0.json").read_bytes() != (first / "seed_1.json").read_bytes()


class TestTheProtocolOnTheFixtureFrame:
    def test_every_metric_carries_three_draws_and_a_spread(self, protocol_aggregate):
        for metric in protocol_aggregate["metrics"].values():
            assert metric["n"] == 3
            assert metric["sd"] is not None

    def test_the_draws_genuinely_differ_so_the_spread_is_not_a_formality(self, protocol_aggregate):
        assert protocol_aggregate["metrics"]["raw_adherence"]["sd"] > 0.0

    @pytest.mark.parametrize("metric", ["raw_adherence", "corrected_adherence", "caught_share"])
    def test_the_shares_lie_in_the_unit_interval(self, protocol_aggregate, metric):
        summary = protocol_aggregate["metrics"][metric]
        assert 0.0 <= summary["mean"] <= 1.0
        assert all(0.0 <= bound <= 1.0 for bound in summary["observed_range"])

    def test_the_labels_spent_are_counted_not_shared(self, protocol_aggregate):
        assert protocol_aggregate["metrics"]["total_oracle_labels_spent"]["mean"] > 1.0
        assert protocol_aggregate["oracle_labels_spent"]["mean"] > 1.0

    def test_the_aggregate_is_the_file_it_returns(self, tmp_path):
        from rcv.runner import run_protocol

        config = protocol_config(write_frame(tmp_path, build_fixture_frame()), [0])
        output_dir = tmp_path / "results"
        returned = run_protocol(config, output_dir)
        assert json.loads((output_dir / "aggregate.json").read_text()) == returned


class TestH2RunnerBlockers:
    @staticmethod
    def _streamless_config(tmp_path):
        from fixture_frame import build_fixture_frame

        frame = build_fixture_frame()
        streamless = {name: values[~frame["is_stream"]] for name, values in frame.items()}
        frame_path = tmp_path / "streamless.npz"
        np.savez(frame_path, **streamless)
        return {
            "study": "h2-streamless", "seeds": [0, 1, 2],
            "frame": {"path": str(frame_path), "name": "streamless-fixture",
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

    def test_h2_1_a_streamless_protocol_writes_records_end_to_end(self, tmp_path):
        from rcv.runner import run_protocol

        aggregate = run_protocol(self._streamless_config(tmp_path), tmp_path / "out")
        assert aggregate["n"] == 3
        assert (tmp_path / "out" / "seed_0.json").exists()
        import json
        record = json.loads((tmp_path / "out" / "seed_0.json").read_text())
        assert record["monitor"]["streamless"] is True

    def test_h2_3_the_caught_share_population_is_recorded(self, tmp_path):
        from rcv.runner import run_protocol

        aggregate = run_protocol(self._streamless_config(tmp_path), tmp_path / "out")
        assert aggregate["caught_share_population"]["min"] >= 1
        import json
        record = json.loads((tmp_path / "out" / "seed_0.json").read_text())
        assert record["readout"]["caught_share_population"] >= 1

    def test_h2_8_a_refusing_seed_yields_refusals_and_no_aggregate(self, tmp_path, monkeypatch):
        import rcv.runner as runner_module
        from rcv.runner import run_protocol

        real_run_study = runner_module.run_study

        def refusing_on_one(config, seed=None):
            if seed == 1:
                raise ValueError("simulated refusal for seed 1")
            return real_run_study(config, seed)

        monkeypatch.setattr(runner_module, "run_study", refusing_on_one)
        with pytest.raises(Exception, match="refus"):
            run_protocol(self._streamless_config(tmp_path), tmp_path / "out")
        import json
        refusals = json.loads((tmp_path / "out" / "REFUSALS.json").read_text())
        assert refusals["1"].startswith("ValueError")
        assert not (tmp_path / "out" / "aggregate.json").exists(), (
            "an aggregate over surviving seeds would be a seed-selected object")


class TestH2Provenance:
    def test_the_record_carries_a_code_digest(self, tmp_path):
        from fixture_frame import build_fixture_frame
        from rcv.runner import provenance_of

        frame = build_fixture_frame()
        frame_path = tmp_path / "f.npz"
        np.savez(frame_path, **frame)
        provenance = provenance_of({"frame": {"path": str(frame_path)}, "seeds": [0]})
        assert len(provenance["code_sha256"]) == 64
        assert provenance["code_sha256"] == provenance_of(
            {"frame": {"path": str(frame_path)}, "seeds": [0]})["code_sha256"]




class TestTheRecordsShape:
    def test_a_streamless_record_says_the_monitor_did_not_run(self, tmp_path):
        from rcv.runner import monitor_record_of
        from rcv.study import MonitorState

        assert monitor_record_of(MonitorState(deployed=None, recalibrations=[],
                                              final=None)) == {
            "streamless": True, "deployed": None, "recalibrations": [], "final": None}

    def test_the_caught_share_population_is_recorded_per_seed_with_its_bounds(self, tmp_path):
        from rcv.runner import population_of

        records = [{"seed": 7, "readout": {"caught_share_population": 21}},
                   {"seed": 11, "readout": {"caught_share_population": 703}}]
        assert population_of(records) == {"per_seed": {"7": 21, "11": 703},
                                          "min": 21, "max": 703}

    def test_the_code_digest_is_read_off_the_packages_own_sources(self, tmp_path):
        import hashlib
        from pathlib import Path

        import rcv
        from rcv.runner import code_digest

        package_root = Path(rcv.__file__).parent
        sources = sorted(package_root.glob("*.py"))
        assert len(sources) > 1, "the premise: the package has sources to digest"
        expected = hashlib.sha256()
        for source in sources:
            expected.update(source.name.encode("utf-8"))
            expected.update(source.read_bytes())
        assert code_digest() == expected.hexdigest()


class TestEverySeedIsAttempted:
    def test_a_refusing_seed_does_not_stop_the_draws_that_follow_it(self, tmp_path,
                                                                    monkeypatch):
        import rcv.runner as runner_module
        from rcv.runner import run_protocol

        real_run_study = runner_module.run_study

        def refusing_on_two(config, seed=None):
            if seed in (0, 1):
                raise ValueError(f"simulated refusal for seed {seed}")
            return real_run_study(config, seed)

        monkeypatch.setattr(runner_module, "run_study", refusing_on_two)
        with pytest.raises(RuntimeError, match=r"^2 of 3 seeds refused: \['0', '1'\]\. "
                                               r"No aggregate was written; see REFUSALS\.json\.$"):
            run_protocol(TestH2RunnerBlockers._streamless_config(tmp_path), tmp_path / "out")

        written = sorted(path.name for path in (tmp_path / "out").iterdir())
        assert "REFUSALS.json" in written
        refusals = json.loads((tmp_path / "out" / "REFUSALS.json").read_text())
        assert sorted(refusals) == ["0", "1"]
