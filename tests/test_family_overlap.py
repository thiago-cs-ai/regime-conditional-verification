import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
for entry in (ROOT / "src", ROOT / "sandbox"):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

from test_prefix_drift_and_config_strictness import _prefix_config, _stream_frame


class TestTheStudyRecordsHowMuchOfItsStreamTheProbeHadSeen:
    def test_the_fraction_is_computed_per_side_on_a_hand_case(self):
        from rcv.study import fitted_family_overlap

        fitted = {"family": np.array([1, 1, 2])}
        traffic = {"family": np.array([1, 2, 3, 4, 1, 3, 3, 4]),
                   "is_drift_item": np.array([False] * 4 + [True] * 4)}

        assert fitted_family_overlap(fitted, traffic) == {
            "all": 0.375, "base": 0.5, "drift": 0.25}

    def test_a_stream_the_probe_never_saw_reads_zero_rather_than_absent(self):
        from rcv.study import fitted_family_overlap

        fitted = {"family": np.array([9, 9])}
        traffic = {"family": np.array([1, 2]), "is_drift_item": np.array([False, True])}

        assert fitted_family_overlap(fitted, traffic) == {"all": 0.0, "base": 0.0, "drift": 0.0}

    def test_a_frame_without_the_pool_column_reports_the_whole_stream_only(self):
        from rcv.study import fitted_family_overlap

        fitted = {"family": np.array([1])}
        traffic = {"family": np.array([1, 2, 3, 4])}

        assert fitted_family_overlap(fitted, traffic) == {
            "all": 0.25, "base": None, "drift": None}

    def test_the_calibration_carries_it(self, tmp_path):
        from rcv.study import run_study

        result = run_study(_prefix_config(tmp_path, _stream_frame(100), 100))
        overlap = result.monitor.deployed.fitted_family_overlap

        assert set(overlap) == {"all", "base", "drift"}
        assert 0.0 <= overlap["all"] <= 1.0

    def test_it_reaches_the_written_record(self, tmp_path):
        import json

        from rcv.runner import as_written, record_of
        from rcv.study import run_study

        result = run_study(_prefix_config(tmp_path, _stream_frame(100), 100))
        written = json.loads(as_written(record_of(result, provenance={})))

        assert "fitted_family_overlap" in written["monitor"]["deployed"]


class TestTheBuildReportsTheSameFractionAgainstTheServingCarve:
    def test_the_fractions_are_computed_per_side(self):
        import build_ramp_stream_frames as builder

        frame = {
            "family": np.array([1, 2, 1, 3, 4, 3, 5]),
            "is_stream": np.array([False, False, True, True, True, True, True]),
            "is_drift_item": np.array([False, False, False, False, False, True, True]),
        }
        assert builder.serving_family_overlap(frame) == {
            "all": 0.2, "base": 1 / 3, "drift": 0.0}

    def test_the_report_line_names_both_sides_and_the_reason(self):
        import build_ramp_stream_frames as builder

        line = builder.family_overlap_sentence({"all": 0.17, "base": 0.201, "drift": 0.0118})

        assert "17.00%" in line and "20.10%" in line and "1.18%" in line
        assert "ITEM level" in line and "FAMILY" in line
