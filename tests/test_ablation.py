import json

import numpy as np
import pytest
import yaml


def streamless_config(tmp_path):
    from fixture_frame import FIXTURE_REFERENCE_WINDOW, build_fixture_frame

    frame = build_fixture_frame()
    streamless = {name: values[~frame["is_stream"]] for name, values in frame.items()}
    frame_path = tmp_path / "frame.npz"
    np.savez(frame_path, **streamless)
    return {
        "study": "a1-fixture", "seeds": [0, 1, 2],
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


class TestRoutingFactorial:
    @pytest.fixture(scope="class")
    def factorial(self, tmp_path_factory):
        from rcv.ablation import run_routing_factorial

        tmp_path = tmp_path_factory.mktemp("a1")
        return (run_routing_factorial(streamless_config(tmp_path), tmp_path / "out"),
                tmp_path / "out")

    def test_all_four_cells_run_on_the_same_seeds(self, factorial):
        result, out = factorial
        assert set(result["cells"]) == {"neither", "calibration_only", "probe_only", "both"}
        seed_lists = {name: cell["seeds"] for name, cell in result["cells"].items()}
        assert len({tuple(seeds) for seeds in seed_lists.values()}) == 1

    def test_the_flags_reach_each_variant(self, factorial):
        result, out = factorial
        for name, flags in {"neither": (False, False), "calibration_only": (False, True),
                            "probe_only": (True, False), "both": (True, True)}.items():
            echoed = json.loads((out / name / "aggregate.json").read_text())
            estimator = echoed["provenance"]["config"]["estimator"]
            assert (estimator["route_probe"], estimator["route_calibration"]) == flags

    def test_the_headline_contrast_is_the_diagonal_with_per_seed_deltas(self, factorial):
        result, out = factorial
        contrast = result["routing_contrast"]
        assert contrast["factor"] == "routing: probe and calibration are routed by verdict regime"
        deltas = contrast["per_seed_corrected_adherence_delta"]
        assert len(deltas) == 3
        wins = contrast["corrected_adherence_wins_ties_losses"]
        assert sum(wins) == 3

    def test_the_factorial_record_is_written_once(self, factorial):
        result, out = factorial
        assert (out / "factorial.json").exists()
        from rcv.ablation import run_routing_factorial
        from rcv.runner import ResultExistsError
        with pytest.raises(ResultExistsError):
            run_routing_factorial(result["provenance"]["config"], out)


    def test_each_cell_is_named_for_its_composition_and_the_caller_keeps_its_config(self,
                                                                                    tmp_path):
        import copy

        from rcv.ablation import run_routing_factorial

        config = streamless_config(tmp_path)
        config["seeds"] = [0]
        config["estimator"]["route_probe"] = False
        config["estimator"]["route_calibration"] = False
        untouched = copy.deepcopy(config)
        out = tmp_path / "named"
        run_routing_factorial(config, out)

        assert config == untouched, "a variant wrote through to the caller's configuration"
        for name in ("neither", "calibration_only", "probe_only", "both"):
            written = json.loads((out / name / "aggregate.json").read_text())
            assert written["study"] == f"a1-fixture-a1-{name}"

    def test_a_configuration_named_by_path_is_read_from_the_file(self, tmp_path):
        from rcv.ablation import run_routing_factorial

        config = streamless_config(tmp_path)
        config["seeds"] = [0]
        config_path = tmp_path / "a1.yaml"
        config_path.write_text(yaml.safe_dump(config))
        result = run_routing_factorial(str(config_path), tmp_path / "from_file")

        assert set(result["cells"]) == {"neither", "calibration_only", "probe_only", "both"}
        assert result["provenance"]["config"] == config

    def test_the_factorial_record_carries_the_contrast_under_its_own_name(self, tmp_path):
        from rcv.ablation import run_routing_factorial

        config = streamless_config(tmp_path)
        config["seeds"] = [0, 1]
        out = tmp_path / "record"
        returned = run_routing_factorial(config, out)

        assert "factorial.json" in sorted(path.name for path in out.iterdir())
        written = json.loads((out / "factorial.json").read_text())
        assert written == returned
        assert written["provenance"]["config"] == config

    def test_the_diagonal_is_both_minus_neither_read_off_the_written_draws(self, tmp_path):
        from rcv.ablation import run_routing_factorial

        config = streamless_config(tmp_path)
        config["seeds"] = [0, 1]
        out = tmp_path / "diagonal"
        result = run_routing_factorial(config, out)

        pairs = [(readout_of(out / "both", seed)[metric],
                  readout_of(out / "neither", seed)[metric], metric, seed)
                 for metric in ("corrected_adherence", "caught_share") for seed in (0, 1)]
        assert any(both != neither for both, neither, _, _ in pairs), (
            "the premise: the two compositions differ somewhere")
        for both, neither, metric, seed in pairs:
            assert (result["routing_contrast"][f"per_seed_{metric}_delta"][str(seed)]
                    == both - neither)


def readout_of(cell_dir, seed):
    return json.loads((cell_dir / f"seed_{seed}.json").read_text())["readout"]


def identical_source_config(tmp_path):
    from fixture_frame import build_fixture_frame

    frame = build_fixture_frame()
    streamless = {name: values[~frame["is_stream"]] for name, values in frame.items()}
    streamless["clf_score"] = streamless["representation"].copy()
    frame_path = tmp_path / "twinned.npz"
    np.savez(frame_path, **streamless)
    config = streamless_config(tmp_path)
    config["frame"]["path"] = str(frame_path)
    return config


class TestFeatureSourcePairs:
    def test_the_a4_pair_contrasts_internal_state_against_the_score(self, tmp_path):
        from rcv.ablation import run_feature_source_pairs

        config = streamless_config(tmp_path)
        config["seeds"] = [0, 1]
        result = run_feature_source_pairs(config, tmp_path / "a4")
        assert set(result["cells"]) == {"representation", "clf_score"}
        contrast = result["source_contrast"]
        assert contrast["factor"] == "probe feature source: representation versus clf_score"
        assert sum(contrast["corrected_adherence_wins_ties_losses"]) == 2

    def test_a_configuration_named_by_path_is_read_from_the_file(self, tmp_path):
        from rcv.ablation import run_feature_source_pairs

        config = streamless_config(tmp_path)
        config["seeds"] = [0]
        config_path = tmp_path / "a4.yaml"
        config_path.write_text(yaml.safe_dump(config))
        result = run_feature_source_pairs(str(config_path), tmp_path / "from_file")

        assert set(result["cells"]) == {"representation", "clf_score"}
        assert result["provenance"]["config"] == config

    def test_each_cell_is_named_for_its_source_and_the_caller_keeps_its_config(self, tmp_path):
        import copy

        from rcv.ablation import run_feature_source_pairs

        config = streamless_config(tmp_path)
        config["seeds"] = [0]
        untouched = copy.deepcopy(config)
        out = tmp_path / "named"
        run_feature_source_pairs(config, out)

        assert config == untouched, "a variant wrote through to the caller's configuration"
        for source in ("representation", "clf_score"):
            written = json.loads((out / source / "aggregate.json").read_text())
            assert written["study"] == f"a1-fixture-a4-{source}"

    def test_the_pair_record_carries_its_provenance_under_its_own_name(self, tmp_path):
        from rcv.ablation import run_feature_source_pairs

        config = streamless_config(tmp_path)
        config["seeds"] = [0]
        out = tmp_path / "record"
        returned = run_feature_source_pairs(config, out)

        assert "feature_source_pairs.json" in sorted(path.name for path in out.iterdir())
        written = json.loads((out / "feature_source_pairs.json").read_text())
        assert written == returned
        assert written["provenance"] == {"config": config,
                                         "sources": ["representation", "clf_score"]}

    def test_the_delta_is_the_internal_state_minus_the_score_on_the_same_draw(self, tmp_path):
        from rcv.ablation import run_feature_source_pairs

        config = streamless_config(tmp_path)
        config["seeds"] = [0, 1]
        out = tmp_path / "orientation"
        result = run_feature_source_pairs(config, out)

        pairs = [(readout_of(out / "representation", seed)[metric],
                  readout_of(out / "clf_score", seed)[metric], metric, seed)
                 for metric in ("corrected_adherence", "caught_share") for seed in (0, 1)]
        assert any(internal != score for internal, score, _, _ in pairs), (
            "the premise: the two sources differ somewhere")
        for internal, score, metric, seed in pairs:
            assert (result["source_contrast"][f"per_seed_{metric}_delta"][str(seed)]
                    == internal - score)

    def test_two_sources_that_carry_the_same_column_are_counted_as_ties(self, tmp_path):
        from rcv.ablation import run_feature_source_pairs

        config = identical_source_config(tmp_path)
        config["seeds"] = [0, 1]
        result = run_feature_source_pairs(config, tmp_path / "twins")

        contrast = result["source_contrast"]
        for metric in ("corrected_adherence", "caught_share"):
            assert set(contrast[f"per_seed_{metric}_delta"].values()) == {0.0}
            assert contrast[f"{metric}_wins_ties_losses"] == [0, 2, 0]


LINEAR_AND_TREE = [{"family": "linear", "C": 1.0},
                   {"family": "gradient_boosted", "max_iter": 20}]


class TestProbeFamilyPairsRecord:
    @pytest.fixture(scope="class")
    def pairs(self, tmp_path_factory):
        from rcv.ablation import run_probe_family_pairs

        tmp_path = tmp_path_factory.mktemp("a2record")
        config = streamless_config(tmp_path)
        config["seeds"] = [0, 1]
        out = tmp_path / "a2"
        return run_probe_family_pairs(config, out, LINEAR_AND_TREE), out, config

    def test_a_configuration_named_by_path_is_read_from_the_file(self, tmp_path):
        from rcv.ablation import run_probe_family_pairs

        config = streamless_config(tmp_path)
        config["seeds"] = [0]
        config_path = tmp_path / "a2.yaml"
        config_path.write_text(yaml.safe_dump(config))
        result = run_probe_family_pairs(str(config_path), tmp_path / "from_file",
                                        [{"family": "linear", "C": 1.0}])

        assert set(result["cells"]) == {"linear_pooled", "linear_routed"}
        assert result["provenance"]["config"] == config

    def test_each_cell_is_named_and_composed_and_the_caller_keeps_its_config(self, tmp_path):
        import copy

        from rcv.ablation import run_probe_family_pairs

        config = streamless_config(tmp_path)
        config["seeds"] = [0]
        config["estimator"]["route_probe"] = False
        config["estimator"]["route_calibration"] = False
        untouched = copy.deepcopy(config)
        out = tmp_path / "named"
        run_probe_family_pairs(config, out, LINEAR_AND_TREE)

        assert config == untouched, "a variant wrote through to the caller's configuration"
        for spec in LINEAR_AND_TREE:
            for routed in (False, True):
                name = f"{spec['family']}_{'routed' if routed else 'pooled'}"
                written = json.loads((out / name / "aggregate.json").read_text())
                assert written["study"] == f"a1-fixture-a2-{name}"
                estimator = written["provenance"]["config"]["estimator"]
                assert estimator["probe"] == spec
                assert estimator["route_probe"] is routed
                assert estimator["route_calibration"] is routed

    def test_the_record_carries_its_provenance_under_its_own_name(self, pairs):
        result, out, config = pairs

        assert "family_pairs.json" in sorted(path.name for path in out.iterdir())
        written = json.loads((out / "family_pairs.json").read_text())
        assert written == result
        assert written["provenance"] == {"config": config, "family_specs": LINEAR_AND_TREE}

    def test_the_contrast_is_routed_minus_pooled_read_off_the_written_draws(self, pairs):
        result, out, config = pairs

        pairs_read = [(readout_of(out / f"{family}_routed", seed)[metric],
                       readout_of(out / f"{family}_pooled", seed)[metric], family, metric, seed)
                      for family in ("linear", "gradient_boosted")
                      for metric in ("corrected_adherence", "caught_share")
                      for seed in (0, 1)]
        assert any(routed != pooled for routed, pooled, *_ in pairs_read), (
            "the premise: routing moves something somewhere")
        for routed, pooled, family, metric, seed in pairs_read:
            assert (result["family_contrasts"][family][f"per_seed_{metric}_delta"][str(seed)]
                    == routed - pooled)

    def test_a_threshold_no_probability_reaches_leaves_every_draw_a_tie(self, tmp_path):
        from rcv.ablation import run_probe_family_pairs

        config = streamless_config(tmp_path)
        config["seeds"] = [0, 1]
        config["flip"]["threshold"] = 1e-9
        config["monitor"]["event_bank"]["boundaries"] = [1e-9]
        result = run_probe_family_pairs(config, tmp_path / "ties",
                                        [{"family": "linear", "C": 1.0}])

        contrast = result["family_contrasts"]["linear"]
        for metric in ("corrected_adherence", "caught_share"):
            assert set(contrast[f"per_seed_{metric}_delta"].values()) == {0.0}
            assert contrast[f"{metric}_wins_ties_losses"] == [0, 2, 0]
