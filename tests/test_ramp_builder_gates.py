import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
for entry in (ROOT / "src", ROOT / "sandbox"):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

import build_ramp_stream_frames as builder


class _Stream:
    def __init__(self, is_drift_item, item_id):
        self.is_drift_item = np.asarray(is_drift_item, dtype=bool)
        self.item_id = np.asarray(item_id)


def _nesting_pair():
    gentle = _Stream([False, True, False, False], ["a", "d1", "c", "e"])
    steep = _Stream([False, True, True, False], ["a", "d1", "d2", "e"])
    return {"null": _Stream([False] * 4, ["a", "b", "c", "e"]),
            "0.15": gentle, "0.30": steep}


class TestTheNestingGateNeverPrintsWhatItDidNotCheck:
    def test_a_whole_run_build_reports_the_chain_it_verified(self):
        assert builder.assert_variants_nest(_nesting_pair()) == "null ⊆ 0.15 ⊆ 0.30"

    def test_a_single_variant_build_reports_that_the_gate_did_not_run(self):
        streams = {"0.15": _nesting_pair()["0.15"]}

        assert builder.assert_variants_nest(streams) is None

    def test_the_report_says_so_rather_than_claiming_a_chain(self):
        skipped = builder.nesting_sentence(None)
        checked = builder.nesting_sentence("null ⊆ 0.15 ⊆ 0.30")

        assert "NOT CHECKED" in skipped
        assert "verified" not in skipped, "a gate that did not run is never printed as verified"
        assert "verified" in checked
        assert "null ⊆ 0.15 ⊆ 0.30" in checked

    def test_a_real_violation_still_refuses(self):
        streams = _nesting_pair()
        streams["0.30"] = _Stream([False, False, True, False], ["a", "b", "d2", "e"])
        with pytest.raises(ValueError, match="not a subset"):
            builder.assert_variants_nest(streams)


class TestAFailingDeterminismGateLeavesNoPublishedFrame:
    @staticmethod
    def _frame():
        return {"verdict": np.arange(4, dtype=np.int64)}

    def test_the_frame_lands_only_after_the_gate_passes(self, tmp_path):
        path = tmp_path / "ramp.npz"
        digest = builder.publish_after_gate(self._frame(), path, gate=lambda staged, sha: None)

        assert path.exists()
        assert digest == builder.digest_of(path)
        assert list(tmp_path.glob("*.staging")) == []

    def test_a_failing_gate_publishes_nothing(self, tmp_path):
        path = tmp_path / "ramp.npz"

        def refuse(staged_path, sha256):
            raise ValueError("a straight rebuild digests something else")

        with pytest.raises(ValueError, match="straight rebuild"):
            builder.publish_after_gate(self._frame(), path, gate=refuse)

        assert not path.exists(), "a failing gate left a published artefact in place"
        assert list(tmp_path.glob("*.staging")) == [], "and it left its staging file behind"

    def test_the_gate_is_handed_the_staged_path_not_the_destination(self, tmp_path):
        path = tmp_path / "ramp.npz"
        seen = {}

        def record(staged_path, sha256):
            seen["path"] = staged_path
            seen["existed"] = staged_path.exists()
            seen["destination_existed"] = path.exists()

        builder.publish_after_gate(self._frame(), path, gate=record)

        assert seen["existed"] is True
        assert seen["destination_existed"] is False
        assert seen["path"] != path
