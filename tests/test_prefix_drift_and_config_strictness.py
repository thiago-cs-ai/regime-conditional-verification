from pathlib import Path

import numpy as np
import pytest
import yaml

from fixture_frame import build_fixture_frame
from test_v1_skeleton import TestStreamPrefixCalibration, write_frame, write_v1_config

CONFIGS_DIRECTORY = Path(__file__).resolve().parent.parent / "configs"


def _stream_frame(prefix_length=100, contaminated_positions=(), schedule_maximum=0.25):
    frame = TestStreamPrefixCalibration._frame_with_drift_free_stream_head(prefix_length)
    stream_rows = int(frame["is_stream"].sum())
    schedule = np.zeros(len(frame["verdict"]), dtype=np.float64)
    drift = np.zeros(len(frame["verdict"]), dtype=bool)
    stream_positions = np.where(frame["is_stream"])[0]
    schedule[stream_positions[prefix_length:]] = np.linspace(
        0.0, schedule_maximum, stream_rows - prefix_length)
    for position, rate in contaminated_positions:
        schedule[stream_positions[position]] = rate
        drift[stream_positions[position]] = True
    frame["stream_lambda"] = schedule
    frame["is_drift_item"] = drift
    return frame


def _prefix_config(tmp_path, frame, prefix_length=100):
    config_path = write_v1_config(tmp_path, write_frame(tmp_path, frame))
    config = yaml.safe_load(config_path.read_text())
    config["monitor"]["calibration"]["material"] = "stream_prefix"
    config["monitor"]["calibration"]["prefix_length"] = prefix_length
    config_path.write_text(yaml.safe_dump(config))
    return config_path


def _deployed(result):
    return result.monitor.deployed


def _floor_config():
    from test_v1_skeleton import TestConfigFloorsAreTheValuesTheyName

    return TestConfigFloorsAreTheValuesTheyName._config()


class TestTheCalibrationPrefixIsCheckedDriftFree:
    def test_a_contaminated_prefix_refuses_naming_the_count_and_the_maximum_rate(self, tmp_path):
        from rcv.study import run_study

        frame = _stream_frame(contaminated_positions=[(11, 0.0625), (37, 0.2375)])
        with pytest.raises(ValueError, match=(
                r"^Calibration prefix has 2 contaminated positions \(maximum stream_lambda "
                r"0\.237500\); prefix reference requires no contamination\.$")):
            run_study(_prefix_config(tmp_path, frame))

    def test_a_clean_prefix_runs_and_records_that_the_check_ran(self, tmp_path):
        from rcv.study import run_study

        result = run_study(_prefix_config(tmp_path, _stream_frame()))
        assert _deployed(result).prefix_drift_checked is True

    def test_a_frame_without_the_stream_columns_records_that_the_check_could_not_run(
            self, tmp_path):
        from rcv.study import run_study

        frame = TestStreamPrefixCalibration._frame_with_drift_free_stream_head(100)
        result = run_study(_prefix_config(tmp_path, frame))
        assert _deployed(result).prefix_drift_checked is False

    def test_an_evaluation_reference_records_no_prefix_verdict_at_all(self, tmp_path):
        from rcv.study import run_study

        config_path = write_v1_config(tmp_path, write_frame(tmp_path, build_fixture_frame()))
        result = run_study(config_path)
        assert _deployed(result).prefix_drift_checked is None

    def test_a_contaminated_prefix_named_only_by_the_pool_column_still_refuses(self, tmp_path):
        from rcv.study import run_study

        frame = _stream_frame()
        stream_positions = np.where(frame["is_stream"])[0]
        frame["is_drift_item"][stream_positions[5]] = True
        del frame["stream_lambda"]
        with pytest.raises(ValueError, match=(
                r"^Calibration prefix has 1 contaminated positions \(maximum stream_lambda "
                r"unrecorded\); prefix reference requires no contamination\.$")):
            run_study(_prefix_config(tmp_path, frame))


class TestTheConfigurationIsStrictAtEveryNestingLevel:
    def test_a_misspelled_material_key_refuses_naming_it(self):
        from rcv.study import validate_config

        config = _floor_config()
        config["monitor"]["calibration"]["materal"] = "stream_prefix"
        with pytest.raises(ValueError, match=r"monitor\.calibration: unknown keys \['materal'\]"):
            validate_config(config)

    def test_an_unknown_subkey_refuses_naming_the_known_ones(self):
        from rcv.study import validate_config

        config = _floor_config()
        config["monitor"]["calibration"]["flavour"] = "banana"
        with pytest.raises(ValueError, match=(
                r"monitor\.calibration: unknown keys \['flavour'\]; missing keys \[\]; "
                r"allowed keys \['material', 'n_streams', 'stream_length', "
                r"'streams_per_centre'\]\.")):
            validate_config(config)

    def test_a_misspelled_material_value_refuses(self):
        from rcv.study import validate_config

        config = _floor_config()
        config["monitor"]["calibration"]["material"] = "stream_prefixx"
        with pytest.raises(ValueError,
                           match=r"^Unknown monitor\.calibration\.material 'stream_prefixx'\.$"):
            validate_config(config)

    def test_a_stream_prefix_reference_without_a_prefix_length_refuses_at_the_gate(self):
        from rcv.study import validate_config

        config = _floor_config()
        config["monitor"]["calibration"]["material"] = "stream_prefix"
        with pytest.raises(ValueError,
                           match=r"monitor\.calibration: .*missing keys \['prefix_length'\]"):
            validate_config(config)

    def test_a_prefix_length_under_an_evaluation_reference_refuses_as_a_key_that_does_nothing(
            self):
        from rcv.study import validate_config

        config = _floor_config()
        config["monitor"]["calibration"]["prefix_length"] = 100
        with pytest.raises(ValueError,
                           match=r"monitor\.calibration: unknown keys \['prefix_length'\]"):
            validate_config(config)

    @pytest.mark.parametrize("section,key", [
        (None, "flavour"),
        ("monitor", "flavour"),
        ("loop", "flavour"),
        ("split", "flavour"),
        ("flip", "flavour"),
    ])
    def test_an_unknown_key_at_every_nesting_level_refuses(self, section, key):
        from rcv.study import validate_config

        config = _floor_config()
        if section is None:
            config[key] = "banana"
        else:
            config[section][key] = "banana"
        with pytest.raises(ValueError, match=key):
            validate_config(config)

    def test_an_unknown_key_under_the_acceptance_criteria_refuses(self):
        from rcv.study import validate_config

        config = _floor_config()
        config["loop"]["acceptance"]["recall_flooor"] = 0.5
        with pytest.raises(ValueError,
                           match=r"loop\.acceptance: unknown keys \['recall_flooor'\]"):
            validate_config(config)

    def test_the_relative_acceptance_form_keeps_its_own_key_set(self):
        from rcv.study import validate_config

        config = _floor_config()
        config["loop"]["acceptance"] = {"form": "baseline_relative", "recall_slack": 0.02,
                                        "over_block_inflation": 0.05}
        validate_config(config)


class TestEveryShippedConfigurationValidates:
    @staticmethod
    def _shipped_configs():
        paths = sorted(CONFIGS_DIRECTORY.glob("*.yaml"))
        assert paths, "the shipped configurations are the point of this test"
        return paths

    def test_all_of_them(self):
        from rcv.study import validate_config

        refused = {}
        for path in self._shipped_configs():
            try:
                validate_config(yaml.safe_load(path.read_text()))
            except ValueError as refusal:
                refused[path.name] = str(refusal)
        assert refused == {}
