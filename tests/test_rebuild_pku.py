from __future__ import annotations

import hashlib
from typing import Any

import pytest

from rcv.rebuild import pku as P
from rcv.rebuild._jsonl import RebuildError
from rcv.rebuild.pku import (
    EXPECTED_N_PROMPTS,
    EXPECTED_N_ROWS,
    EXPECTED_SHA256,
    REQUIRED_COLUMNS,
    _validate_test_split,
    build_frame,
    group_key,
)


def pku_row(
    prompt: str,
    r0: str = "synthetic reply zero",
    r1: str = "synthetic reply one",
    *,
    safe0: bool = True,
    safe1: bool = False,
    harm0: dict[str, bool] | None = None,
    harm1: dict[str, bool] | None = None,
    sev0: int = 0,
    sev1: int = 2,
) -> dict[str, Any]:
    return {
        "prompt": prompt,
        "response_0": r0,
        "response_1": r1,
        "is_response_0_safe": safe0,
        "is_response_1_safe": safe1,
        "response_0_harm_category": harm0 or {},
        "response_1_harm_category": harm1 or {"Synthetic Harm": True, "Other Harm": False},
        "response_0_severity_level": sev0,
        "response_1_severity_level": sev1,
        "prompt_source": "synthetic",
        "response_0_source": "synthetic",
        "response_1_source": "synthetic",
        "response_0_sha256": "0" * 64,
        "response_1_sha256": "1" * 64,
    }


def test_group_key_is_md5_prefix():
    assert group_key("alpha") == hashlib.md5(b"alpha").hexdigest()[:16]


def test_build_frame_shapes_and_ids():
    rows = [pku_row("prompt-a"), pku_row("prompt-b")]
    input_rows, sidecar_rows, summary = build_frame(rows)
    assert len(input_rows) == len(sidecar_rows) == 4
    assert [r["item_id"] for r in input_rows] == [
        "PKU_test_0_r0",
        "PKU_test_0_r1",
        "PKU_test_1_r0",
        "PKU_test_1_r1",
    ]
    assert list(input_rows[0].keys()) == ["item_id", "prompt", "response"]
    assert summary["n_prompts"] == 2 and summary["n_rows_both_responses"] == 4


def test_build_frame_polarity_and_groups():
    rows = [pku_row("prompt-a", safe0=True, safe1=False)]
    _, sidecar, summary = build_frame(rows)
    assert sidecar[0]["y_star_native"] == 0 and sidecar[0]["is_safe"] is True
    assert sidecar[1]["y_star_native"] == 1 and sidecar[1]["is_safe"] is False
    assert summary["n_native_unsafe"] == 1
    assert sidecar[0]["group_key"] == sidecar[1]["group_key"] == group_key("prompt-a")
    assert sidecar[0]["response_idx"] == 0 and sidecar[1]["response_idx"] == 1


def test_build_frame_harm_categories_are_sorted_active_keys():
    rows = [
        pku_row(
            "prompt-a",
            harm1={"Zeta Harm": True, "Alpha Harm": True, "Inactive Harm": False},
        )
    ]
    _, sidecar, _ = build_frame(rows)
    assert sidecar[1]["harm_categories"] == ["Alpha Harm", "Zeta Harm"]
    assert sidecar[0]["harm_categories"] == []


def test_build_frame_counts_and_coerces_empty_responses():
    rows = [pku_row("prompt-a", r0="", r1="  ")]
    input_rows, _, summary = build_frame(rows)
    assert summary["n_empty_responses"] == 2
    assert input_rows[0]["response"] == "" and input_rows[1]["response"] == "  "


def test_build_frame_severity_passthrough():
    rows = [pku_row("prompt-a", sev0=0, sev1=3)]
    _, sidecar, _ = build_frame(rows)
    assert sidecar[0]["severity"] == 0 and sidecar[1]["severity"] == 3




class _FakeSplit:
    def __init__(self, n: int, columns: list[str]):
        self._n = n
        self.column_names = columns

    def __len__(self) -> int:
        return self._n


def test_validate_test_split_accepts_expected_shape():
    ds = _FakeSplit(EXPECTED_N_PROMPTS, list(REQUIRED_COLUMNS))
    assert _validate_test_split(ds, "cafe" * 10) is ds


def test_validate_test_split_rejects_wrong_size_or_schema():
    with pytest.raises(RebuildError, match="33044 prompts"):
        _validate_test_split(_FakeSplit(33044, list(REQUIRED_COLUMNS)), "cafe" * 10)
    cols = [c for c in REQUIRED_COLUMNS if not c.endswith("harm_category")]
    with pytest.raises(RebuildError, match="missing columns"):
        _validate_test_split(_FakeSplit(EXPECTED_N_PROMPTS, cols), "cafe" * 10)


def test_frozen_expectations_are_consistent():
    assert EXPECTED_N_ROWS == 2 * EXPECTED_N_PROMPTS
    assert set(EXPECTED_SHA256) == {"input", "sidecar"}
    assert all(len(v) == 64 for v in EXPECTED_SHA256.values())


@pytest.mark.parametrize("force", [False, True])
def test_rebuild_rejects_stale_corpus_files_without_mutation(tmp_path, monkeypatch, force):
    monkeypatch.setattr(P, "load_pku_test_split", lambda **kw: [pku_row("p")])
    stale = tmp_path / "old_input.jsonl"
    stale.write_text("stale\n")
    with pytest.raises(RebuildError, match="Unexpected.*old_input"):
        P.rebuild(tmp_path, force=force)
    assert stale.read_text() == "stale\n"
    assert not (tmp_path / "native_pku_test.jsonl").exists()


def test_rebuild_preflights_sidecar_and_force_preserves_bytes(tmp_path, monkeypatch):
    monkeypatch.setattr(P, "load_pku_test_split", lambda **kw: [pku_row("p", r0="", r1="  ")])
    sidecar = tmp_path / "native_pku_test.sidecar.jsonl"
    sidecar.write_text("existing\n")
    with pytest.raises(RebuildError, match="already exists"):
        P.rebuild(tmp_path)
    assert not (tmp_path / "native_pku_test.jsonl").exists()
    assert sidecar.read_text() == "existing\n"
    P.rebuild(tmp_path, force=True)
    before = {p.name: p.read_bytes() for p in tmp_path.glob("*.jsonl")}
    P.rebuild(tmp_path, force=True)
    assert {p.name: p.read_bytes() for p in tmp_path.glob("*.jsonl")} == before
