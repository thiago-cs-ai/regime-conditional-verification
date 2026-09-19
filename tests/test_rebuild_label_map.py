from __future__ import annotations

from typing import Any

import pytest

from rcv.rebuild.label_map import (
    ALLOWED_COLUMNS,
    HARD_DROPPED_COLUMNS,
    LabelMapGateError,
    apply_label_map,
    export_label_map,
    gate_rows,
    looks_like_free_text,
)


def native_label_row(i: int = 0, **over: Any) -> dict[str, Any]:
    base = {
        "item_id": f"PKU_test_{i}_r0",
        "rubric_ystar": 1,
        "rule_fired": "2.2_commitment",
        "native_is_safe": False,
        "native_label": 1,
        "group_key": "0123456789abcdef",
        "harm_categories": ["Synthetic Harm", "Other Harm"],
        "severity": 2,
    }
    base.update(over)
    return base


def wgmix_label_row(i: int = 0, **over: Any) -> dict[str, Any]:
    base = {
        "item_id": f"it_{i:012x}",
        "new_label": 0,
        "old_label": 1,
        "rule_fired": "none",
    }
    base.update(over)
    return base


EXPECTED_HARD_DROPPED_COLUMNS = ("raw", "reason", "reply", "rule_fired_raw")


def test_free_text_word_boundary():
    assert not looks_like_free_text("one two three four")
    assert looks_like_free_text("one two three four five")


def test_free_text_length_and_newline():
    assert not looks_like_free_text("x" * 64)
    assert looks_like_free_text("x" * 65)
    assert looks_like_free_text("short\nbut multiline")




def test_gate_passes_valid_native_and_wgmix_rows():
    rep = gate_rows([native_label_row(i) for i in range(3)], primary_label="rubric_ystar")
    assert rep["gate_pass"] and rep["n_rows"] == 3
    rep = gate_rows([wgmix_label_row(i) for i in range(3)], primary_label="new_label")
    assert rep["gate_pass"] and rep["n_rows"] == 3


@pytest.mark.parametrize("primary", ["rubric_ystar", "new_label"])
def test_gate_rejects_empty_label_map(primary):
    with pytest.raises(LabelMapGateError, match="empty"):
        gate_rows([], primary_label=primary)


def test_gate_fails_on_hard_dropped_column():
    row = native_label_row(reason="a judge explanation sentence that must never ship")
    with pytest.raises(LabelMapGateError, match="excluded diagnostic column"):
        gate_rows([row], primary_label="rubric_ystar")


def test_hard_dropped_columns_are_complete():
    assert HARD_DROPPED_COLUMNS == EXPECTED_HARD_DROPPED_COLUMNS


@pytest.mark.parametrize("col", EXPECTED_HARD_DROPPED_COLUMNS)
def test_gate_names_every_hard_dropped_column(col):
    row = native_label_row(**{col: "anything"})
    with pytest.raises(LabelMapGateError, match="excluded diagnostic column"):
        gate_rows([row], primary_label="rubric_ystar")


def test_gate_fails_on_non_allowlisted_column():
    row = native_label_row(model="synthetic/model-name")
    with pytest.raises(LabelMapGateError, match="not allowlisted"):
        gate_rows([row], primary_label="rubric_ystar")


def test_gate_fails_on_free_text_shaped_value():
    row = wgmix_label_row(rule_fired="a long sentence pretending to be an enum tag")
    with pytest.raises(LabelMapGateError):
        gate_rows([row], primary_label="new_label")


def test_gate_fails_on_free_text_harm_category():
    row = native_label_row(
        harm_categories=["this category entry is really a full sentence of judge text"]
    )
    with pytest.raises(LabelMapGateError, match="harm_categories"):
        gate_rows([row], primary_label="rubric_ystar")


@pytest.mark.parametrize(
    ("col", "value", "match"),
    [
        ("item_id", "not-an-id", "unexpected shape"),
        ("rubric_ystar", 2, "0/1 label"),
        ("rubric_ystar", True, "0/1 label"),
        ("native_is_safe", 1, "not a bool"),
        ("group_key", "xyz", "16-hex"),
        ("severity", 7, r"\[0, 3\]"),
        ("severity", True, r"\[0, 3\]"),
    ],
)
def test_gate_per_column_validators(col, value, match):
    row = native_label_row(**{col: value})
    with pytest.raises(LabelMapGateError, match=match):
        gate_rows([row], primary_label="rubric_ystar")


def test_gate_fails_on_missing_or_null_primary_label():
    row = native_label_row()
    del row["rubric_ystar"]
    with pytest.raises(LabelMapGateError, match="primary label"):
        gate_rows([row], primary_label="rubric_ystar")


def test_gate_fails_on_duplicate_item_id():
    with pytest.raises(LabelMapGateError, match="duplicate"):
        gate_rows([native_label_row(1), native_label_row(1)], primary_label="rubric_ystar")


def test_gate_rejects_non_allowlisted_primary():
    with pytest.raises(LabelMapGateError, match="primary_label"):
        gate_rows([native_label_row()], primary_label="reason")


@pytest.mark.parametrize("primary_label", ["item_id", "severity"])
def test_gate_rejects_metadata_as_primary_label(primary_label):
    row = {"item_id": "PKU_test_0_r0", "severity": 1}
    with pytest.raises(LabelMapGateError, match="primary_label"):
        gate_rows([row], primary_label=primary_label)




def test_export_filters_to_allowlist_and_hard_drops_text():
    raw = [
        dict(
            native_label_row(0),
            reason="a long free-text judge explanation that must be dropped",
            rule_fired_raw="2.1_synthetic",
            model="synthetic/judge",
            prompt_version="v1",
            response_idx=0,
        )
    ]
    out = export_label_map(raw, primary_label="rubric_ystar")
    assert len(out) == 1
    assert set(out[0]) <= set(ALLOWED_COLUMNS)
    assert not set(out[0]) & set(HARD_DROPPED_COLUMNS)
    assert "model" not in out[0] and "response_idx" not in out[0]


def test_export_dedupes_last_non_null_label_wins():
    raw = [
        wgmix_label_row(1, new_label=None),
        wgmix_label_row(2, new_label=1),
        wgmix_label_row(1, new_label=0),
    ]
    out = export_label_map(raw, primary_label="new_label")
    by_id = {r["item_id"]: r for r in out}
    assert len(out) == 2
    assert by_id[f"it_{1:012x}"]["new_label"] == 0


def test_export_requires_item_id():
    with pytest.raises(LabelMapGateError, match="item_id"):
        export_label_map([{"new_label": 1}], primary_label="new_label")


def test_export_output_is_gated():
    raw = [native_label_row(0, severity=9)]
    with pytest.raises(LabelMapGateError):
        export_label_map(raw, primary_label="rubric_ystar")




def items(n: int) -> list[dict[str, Any]]:
    return [{"item_id": f"it_{i:012x}", "prompt": f"p{i}", "response": f"r{i}"} for i in range(n)]


def test_apply_joins_by_item_id():
    labels = [wgmix_label_row(i, new_label=i % 2) for i in range(3)]
    merged = apply_label_map(items(3), labels, primary_label="new_label")
    assert len(merged) == 3
    assert merged[1]["new_label"] == 1 and merged[1]["prompt"] == "p1"


def test_apply_fails_on_missing_coverage():
    labels = [wgmix_label_row(0)]
    with pytest.raises(LabelMapGateError, match="does not cover"):
        apply_label_map(items(2), labels, primary_label="new_label")


def test_apply_regates_the_label_rows():
    labels = [wgmix_label_row(0, reason="tampered free text smuggled into a downloaded map")]
    with pytest.raises(LabelMapGateError, match="excluded diagnostic column"):
        apply_label_map(items(1), labels, primary_label="new_label")
