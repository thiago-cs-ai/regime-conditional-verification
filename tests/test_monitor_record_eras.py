import json

import numpy as np
import yaml

from fixture_frame import build_fixture_frame
from test_prefix_drift_and_config_strictness import _prefix_config, _stream_frame
from test_v1_skeleton import write_frame, write_v1_config

PREFIX = 100
SERVED_ITEM_POOL = 140


def _with_item_ids(frame):
    stream = np.asarray(frame["is_stream"]).astype(bool)
    serving = int((~stream).sum())
    item_id = np.empty(len(frame["verdict"]), dtype=object)
    item_id[~stream] = [f"serving-{position}" for position in range(serving)]
    item_id[stream] = [f"item-{position % SERVED_ITEM_POOL}"
                       for position in range(int(stream.sum()))]
    return dict(frame, item_id=np.array(item_id.tolist()))


def _loop_active(tmp_path, observe_only=False, name="loop"):
    from rcv.study import run_study

    frame = _with_item_ids(_stream_frame(PREFIX))
    config_path = _prefix_config(tmp_path / name, frame, PREFIX)
    config = yaml.safe_load(config_path.read_text())
    config["loop"]["post_repair_reference_window"] = 20
    if observe_only:
        config["loop"]["observe_only"] = True
    config_path.write_text(yaml.safe_dump(config))
    return run_study(config_path)


class TestTheRecordCarriesEveryCalibrationEra:
    def test_a_loop_active_run_separates_what_was_deployed_from_what_stood_last(self, tmp_path):
        (tmp_path / "loop").mkdir()
        result = _loop_active(tmp_path)
        repairs = [entry for entry in result.change_log if entry.what == "repair"]

        assert repairs, "the fixture's drift must drive at least one accepted repair"
        assert len(result.monitor.recalibrations) == len(repairs)
        assert result.monitor.deployed != result.monitor.final
        assert result.monitor.final is result.monitor.recalibrations[-1].calibration

    def test_every_recalibration_names_its_trigger_and_where_it_took_effect(self, tmp_path):
        (tmp_path / "loop").mkdir()
        result = _loop_active(tmp_path)

        for entry in result.monitor.recalibrations:
            assert entry.trigger.startswith("alarm in regime ")
            assert entry.monitored_position > 0
            assert entry.stream_position == entry.monitored_position + PREFIX

    def test_the_deployed_calibration_is_the_one_a_run_that_never_repaired_would_have(
            self, tmp_path):
        (tmp_path / "loop").mkdir()
        (tmp_path / "watch").mkdir()
        with_repairs = _loop_active(tmp_path)
        watching_only = _loop_active(tmp_path, observe_only=True, name="watch")

        assert with_repairs.monitor.deployed == watching_only.monitor.deployed

    def test_an_observe_only_run_has_one_era_and_says_so(self, tmp_path):
        (tmp_path / "watch").mkdir()
        result = _loop_active(tmp_path, observe_only=True, name="watch")

        assert result.monitor.recalibrations == []
        assert result.monitor.deployed == result.monitor.final

    def test_a_streamless_run_carries_the_three_slots_empty(self, tmp_path):
        from rcv.runner import monitor_record_of
        from rcv.study import MonitorState

        assert monitor_record_of(MonitorState(deployed=None, recalibrations=[], final=None)) == {
            "streamless": True, "deployed": None, "recalibrations": [], "final": None}


class TestTheSerialisedRecordForcesTheReaderToChoose:
    def test_the_record_carries_all_three_and_no_ambiguous_slot(self, tmp_path):
        from rcv.runner import as_written, record_of

        (tmp_path / "loop").mkdir()
        written = json.loads(as_written(record_of(_loop_active(tmp_path), provenance={})))
        monitor = written["monitor"]

        assert set(monitor) == {"streamless", "deployed", "recalibrations", "final"}
        assert "calibration" not in monitor
        assert monitor["deployed"]["threshold"] != monitor["final"]["threshold"]
        assert len(monitor["recalibrations"]) >= 1
        assert set(monitor["recalibrations"][0]) == {
            "trigger", "monitored_position", "stream_position", "replay_seed",
            "reference_audit_overlap_fraction", "reference_window_length",
            "reference_window_monitored_span", "reference_window_skipped_positions",
            "calibration"}

    def test_the_change_log_keeps_its_own_shape_and_does_not_repeat_the_calibration(self,
                                                                                    tmp_path):
        from rcv.runner import as_written, record_of

        (tmp_path / "loop").mkdir()
        written = json.loads(as_written(record_of(_loop_active(tmp_path), provenance={})))

        for entry in written["change_log"]:
            assert "calibration" not in entry
            assert {"what", "trigger", "oracle_labels_spent"} <= set(entry)

    def test_a_streamless_record_says_the_monitor_did_not_run(self, tmp_path):
        from rcv.runner import as_written, record_of
        from rcv.study import run_study

        frame = build_fixture_frame()
        streamless = {name: values[~frame["is_stream"]] for name, values in frame.items()}
        written = json.loads(as_written(record_of(
            run_study(write_v1_config(tmp_path, write_frame(tmp_path, streamless))),
            provenance={})))

        assert written["monitor"] == {"streamless": True, "deployed": None,
                                      "recalibrations": [], "final": None}


class TestEachEraDrawsItsOwnReplay:
    def test_a_recalibration_records_the_seed_it_actually_replayed_under(self, tmp_path):
        (tmp_path / "loop").mkdir()
        result = _loop_active(tmp_path)

        seeds = [entry.replay_seed for entry in result.monitor.recalibrations]
        assert seeds == [entry.calibration.seed for entry in result.monitor.recalibrations]
        assert result.monitor.deployed.seed not in seeds
        assert len(set(map(str, seeds))) == len(seeds)


class TestTheOverlapWithTheAuditIsRecorded:
    def test_each_recalibration_records_a_reference_the_update_had_not_fitted(self, tmp_path):
        (tmp_path / "loop").mkdir()
        result = _loop_active(tmp_path)

        for entry in result.monitor.recalibrations:
            assert entry.reference_audit_overlap_fraction == 0.0


    def test_a_frame_without_item_identity_refuses_rather_than_recording_unmeasurable(
            self, tmp_path):
        import pytest

        from rcv.study import run_study

        (tmp_path / "loop").mkdir()
        anonymous = {name: values for name, values in _stream_frame(PREFIX).items()
                     if name != "item_id"}
        with pytest.raises(ValueError, match="item_id"):
            run_study(_prefix_config(tmp_path / "loop", anonymous, PREFIX))
