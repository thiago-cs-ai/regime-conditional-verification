import numpy as np
import pytest
import yaml

from fixture_frame import FIXTURE_REFERENCE_WINDOW, build_fixture_frame


def stream_study_config(frame_path, name):
    return {
        "study": name,
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


def written(tmp_path, name, frame, edit=None):
    frame_path = tmp_path / f"{name}.npz"
    np.savez(frame_path, **frame)
    config = stream_study_config(frame_path, name)
    if edit is not None:
        edit(config)
    config_path = tmp_path / f"{name}.yaml"
    config_path.write_text(yaml.safe_dump(config))
    return config_path


def frame_with_a_stream_head(head_rows):
    frame = build_fixture_frame()
    calm_positions = np.where(~frame["is_stream"])[0]
    frame["is_stream"] = frame["is_stream"].copy()
    frame["is_stream"][calm_positions[-head_rows:]] = True
    return frame


def recorded_calibration_material(monkeypatch):
    from rcv.belief_bank import BeliefBankMonitor

    calls = []
    real_calibrate = BeliefBankMonitor.calibrate

    def recording_calibrate(self, probability, verdicts, stream_length, n_streams,
                                seed, **keywords):
        calls.append({"rows": len(probability), "verdict": np.asarray(verdicts).copy()})
        return real_calibrate(self, probability, verdicts, stream_length, n_streams,
                                  seed, **keywords)

    monkeypatch.setattr(BeliefBankMonitor, "calibrate", recording_calibrate)
    return calls


def recorded_monitored_traffic(monkeypatch):
    import rcv.study as study_module

    watched = []
    real_watch = study_module.watch_traffic

    def recording_watch(*arguments, **keywords):
        watched.append(arguments[4])
        return real_watch(*arguments, **keywords)

    monkeypatch.setattr(study_module, "watch_traffic", recording_watch)
    return watched


class TestP34TheReferenceIsDrawnFromTheStreamItIsAboutToWatch:
    PREFIX = 100

    def test_the_calibration_material_is_the_streams_leading_rows_row_for_row(self, tmp_path,
                                                                              monkeypatch):
        from rcv.study import run_study

        frame = frame_with_a_stream_head(self.PREFIX)
        calls = recorded_calibration_material(monkeypatch)
        run_study(written(tmp_path, "p34", frame, edit=self._calibrate_on_the_prefix))

        head = frame["verdict"][frame["is_stream"]][:self.PREFIX]
        assert calls, "the run must calibrate a monitor"
        np.testing.assert_array_equal(calls[0]["verdict"], head)

    def test_a_same_sized_slice_of_the_serving_carve_is_not_the_same_rows(self, tmp_path,
                                                                          monkeypatch):
        frame = frame_with_a_stream_head(self.PREFIX)
        head = frame["verdict"][frame["is_stream"]][:self.PREFIX]
        serving_head = frame["verdict"][~frame["is_stream"]][:self.PREFIX]
        assert len(serving_head) == len(head)
        assert not np.array_equal(serving_head, head)

    @staticmethod
    def _calibrate_on_the_prefix(config):
        config["monitor"]["calibration"]["material"] = "stream_prefix"
        config["monitor"]["calibration"]["prefix_length"] = (
            TestP34TheReferenceIsDrawnFromTheStreamItIsAboutToWatch.PREFIX)


class TestP39LengtheningThePrefixCostsNoMonitoredHorizon:
    DRIFT_ROWS = 240

    @pytest.mark.parametrize("head_rows", [60, 120])
    def test_the_monitored_stretch_keeps_its_length_however_long_the_prefix(self, tmp_path,
                                                                            monkeypatch,
                                                                            head_rows):
        from rcv.study import run_study

        frame = frame_with_a_stream_head(head_rows)
        assert int(frame["is_stream"].sum()) == head_rows + self.DRIFT_ROWS, (
            "the prefix and the stream must have grown by the same amount")

        watched = recorded_monitored_traffic(monkeypatch)

        def calibrate_on_the_prefix(config):
            config["monitor"]["calibration"]["material"] = "stream_prefix"
            config["monitor"]["calibration"]["prefix_length"] = head_rows

        run_study(written(tmp_path, f"p39_{head_rows}", frame, edit=calibrate_on_the_prefix))

        assert len(watched) == 1
        assert len(watched[0]["verdict"]) == self.DRIFT_ROWS


class TestR06AmendmentOnlyTheScheduleMoves:
    @staticmethod
    def _frame_with_stream_block(reordered):
        frame = build_fixture_frame()
        positions = np.arange(len(frame["verdict"]))
        stream = positions[frame["is_stream"]]
        order = np.concatenate([positions[~frame["is_stream"]], reordered(stream)])
        return {name: values[order] for name, values in frame.items()}

    @pytest.fixture(scope="class")
    def arms(self, tmp_path_factory):
        from rcv.belief_bank import BeliefBankMonitor
        from rcv.study import run_study

        tmp_path = tmp_path_factory.mktemp("r06")
        blocks = {"as_served": lambda stream: stream,
                  "reversed": lambda stream: stream[::-1],
                  "halved": lambda stream: stream[: len(stream) // 2]}
        real_calibrate = BeliefBankMonitor.calibrate
        deployed = {}
        running = {}

        def recording_calibrate(monitor, *arguments, **keywords):
            calibration = real_calibrate(monitor, *arguments, **keywords)
            deployed.setdefault(running["arm"], calibration)
            return calibration

        BeliefBankMonitor.calibrate = recording_calibrate
        try:
            runs = {}
            for name, reordered in blocks.items():
                running["arm"] = name
                frame = self._frame_with_stream_block(reordered)
                runs[name] = (frame, run_study(written(tmp_path, name, frame)))
        finally:
            BeliefBankMonitor.calibrate = real_calibrate
        return runs, deployed

    def test_the_arms_really_do_serve_different_streams(self, arms):
        runs, _ = arms
        served = {name: frame["verdict"][frame["is_stream"]] for name, (frame, _) in runs.items()}
        assert len(served["halved"]) < len(served["as_served"])
        assert not np.array_equal(served["reversed"], served["as_served"])
        assert runs["as_served"][1].loop.events, "the held arm must be a real run"

    def test_the_serving_carve_is_untouched_by_what_the_stream_serves(self, arms):
        runs, _ = arms
        spent = {name: result.readout.oracle_labels_spent for name, (_, result) in runs.items()}
        assert len(set(spent.values())) == 1, spent

    def test_the_readouts_are_untouched_by_what_the_stream_serves(self, arms):
        runs, _ = arms
        readouts = [result.readout for _, result in runs.values()]
        assert readouts[1:] == readouts[:-1]

    def test_the_deployed_monitor_calibration_is_untouched_by_what_the_stream_serves(self, arms):
        from dataclasses import replace

        _, deployed = arms
        calibrations = list(deployed.values())
        assert len(calibrations) == 3
        held = [replace(calibration, monitored_length=None) for calibration in calibrations]
        assert held[1:] == held[:-1]

    def test_the_deployed_calibration_names_the_horizon_each_arm_served(self, arms):
        runs, deployed = arms
        served = {name: int(frame["is_stream"].sum()) for name, (frame, _) in runs.items()}
        assert {name: calibration.monitored_length
                for name, calibration in deployed.items()} == served
