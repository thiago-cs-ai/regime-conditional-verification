from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from rcv.rebuild import wgmix as W
from rcv.rebuild._jsonl import RebuildError, sha256_file, write_jsonl
from rcv.rebuild.wgmix import (
    DATA_MANIFEST_SHA256,
    CarveError,
    carve,
    harm_polarity,
    make_item_id,
    reassign_row_ids,
    response_present,
    to_record,
    verify_against_manifest,
)


def wg_row(
    prompt: str,
    response: str,
    *,
    response_harm: str = "unharmful",
    prompt_harm: str = "unharmful",
    refusal: str = "no",
    adversarial: bool = False,
    subcategory: str = "benign",
) -> dict[str, Any]:
    return {
        "prompt": prompt,
        "response": response,
        "response_harm_label": response_harm,
        "prompt_harm_label": prompt_harm,
        "response_refusal_label": refusal,
        "adversarial": adversarial,
        "subcategory": subcategory,
    }




def test_make_item_id_matches_recipe():
    expected = "it_" + hashlib.sha1(b"3|alpha|beta").hexdigest()[:12]
    assert make_item_id(3, "alpha", "beta") == expected


def test_make_item_id_depends_on_index():
    assert make_item_id(0, "alpha", "beta") != make_item_id(1, "alpha", "beta")


def test_harm_polarity_mapping():
    assert harm_polarity("harmful") == 1
    assert harm_polarity("unharmful") == 0
    assert harm_polarity("None") is None
    assert harm_polarity("something-else") is None


@pytest.mark.parametrize(
    ("response", "harm", "present"),
    [
        ("a synthetic reply", "unharmful", True),
        ("", "unharmful", False),
        ("   ", "unharmful", False),
        (None, "unharmful", False),
        ("a synthetic reply", "None", False),
        ("a synthetic reply", None, False),
    ],
)
def test_response_present(response, harm, present):
    row = wg_row("p", "x")
    row["response"] = response
    row["response_harm_label"] = harm
    assert response_present(row) is present




def test_to_record_field_order_is_frozen():
    rec = to_record("pool", "wildguardtrain", 7, 2, wg_row("p-alpha", "r-alpha"))
    assert list(rec.keys()) == [
        "split",
        "row_id",
        "item_id",
        "wg_source",
        "wg_orig_index",
        "prompt",
        "response",
        "beaver_pred",
        "ground_truth",
        "ai2_prompt_harm",
        "ai2_response_harm",
        "ai2_refusal",
        "adversarial",
        "subcategory",
    ]
    assert rec["beaver_pred"] == -1
    assert rec["item_id"] == make_item_id(2, "p-alpha", "r-alpha")


def test_to_record_polarity_and_types():
    rec = to_record("eval", "wildguardtest", 0, 0, wg_row("p", "r", response_harm="harmful"))
    assert rec["ground_truth"] == 1 and rec["ai2_response_harm"] == 1
    rec = to_record("eval", "wildguardtest", 0, 0, wg_row("p", "r", response_harm="unharmful"))
    assert rec["ground_truth"] == 0


def test_to_record_rejects_polarity_free_label():
    with pytest.raises(CarveError):
        to_record("eval", "wildguardtest", 0, 0, wg_row("p", "r", response_harm="None"))


def test_reassign_row_ids_mints_ids_from_line_index():
    rows = [to_record("pool", "wildguardtrain", oi, 0, wg_row(f"p{oi}", f"r{oi}")) for oi in (5, 9)]
    out = reassign_row_ids(rows)
    for i, row in enumerate(out):
        assert row["row_id"] == i
        assert row["item_id"] == make_item_id(i, row["prompt"], row["response"])




def synthetic_corpus() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    test_rows = [
        wg_row("eval-p0", "eval-r0", response_harm="harmful"),
        wg_row("eval-p1", "eval-r1", response_harm="unharmful"),
        wg_row("eval-p2", "eval-r2", response_harm="unharmful"),
        wg_row("eval-p3", "", response_harm="harmful"),
    ]
    train_rows = []
    for i in range(3):
        train_rows.append(
            wg_row(
                f"adv-p{i}",
                f"adv-r{i}",
                response_harm="harmful",
                adversarial=True,
                subcategory="synthetic_topic_a",
            )
        )
    for i in range(4):
        train_rows.append(
            wg_row(f"van-p{i}", f"van-r{i}", subcategory="synthetic_topic_a")
        )
    for i in range(6):
        train_rows.append(wg_row(f"ben-p{i}", f"ben-r{i}", subcategory="benign"))
    return test_rows, train_rows


def test_carve_structure_and_integrity():
    test_rows, train_rows = synthetic_corpus()
    res = carve(test_rows, train_rows, seed=42, pool_target=6, drift_min_adv=2)

    assert len(res.eval_rows) == 3
    assert set(res.drift) == {"synthetic_topic_a"}
    assert len(res.drift["synthetic_topic_a"]) == 3
    pool_orig = {r["wg_orig_index"] for r in res.pool_rows}
    drift_orig = {r["wg_orig_index"] for r in res.drift["synthetic_topic_a"]}
    assert not pool_orig & drift_orig
    assert res.universe_n == len(res.eval_rows) + len(res.pool_rows) + 3
    for rows in [res.eval_rows, res.pool_rows, res.drift["synthetic_topic_a"]]:
        for i, row in enumerate(rows):
            assert row["row_id"] == i
            assert row["item_id"] == make_item_id(i, row["prompt"], row["response"])


def test_carve_is_deterministic():
    test_rows, train_rows = synthetic_corpus()
    a = carve(test_rows, train_rows, seed=42, pool_target=6, drift_min_adv=2)
    b = carve(test_rows, train_rows, seed=42, pool_target=6, drift_min_adv=2)
    assert json.dumps(a.eval_rows) == json.dumps(b.eval_rows)
    assert json.dumps(a.pool_rows) == json.dumps(b.pool_rows)
    assert json.dumps(a.drift) == json.dumps(b.drift)


def test_carve_rejects_single_class_eval():
    test_rows = [wg_row("eval-p0", "eval-r0"), wg_row("eval-p1", "eval-r1")]
    _, train_rows = synthetic_corpus()
    with pytest.raises(CarveError, match="both harm labels"):
        carve(test_rows, train_rows, seed=42, pool_target=6, drift_min_adv=2)


def test_carve_rejects_eval_train_leakage():
    test_rows, train_rows = synthetic_corpus()
    train_rows.append(
        wg_row(
            "eval-p0",
            "eval-r0",
            response_harm="harmful",
            adversarial=True,
            subcategory="synthetic_topic_a",
        )
    )
    with pytest.raises(CarveError, match="prompt-response pairs with pool or drift rows"):
        carve(test_rows, train_rows, seed=42, pool_target=6, drift_min_adv=2)


def test_carve_requires_a_drift_candidate():
    test_rows, _ = synthetic_corpus()
    train_rows = [wg_row(f"ben-p{i}", f"ben-r{i}") for i in range(6)]
    with pytest.raises(CarveError, match="candidate"):
        carve(test_rows, train_rows, seed=42, pool_target=3, drift_min_adv=2)


def test_carve_rejects_no_pool_rows_after_drift_selection():
    test_rows, train_rows = synthetic_corpus()
    train_rows = [row for row in train_rows if row["adversarial"]]
    with pytest.raises(CarveError, match="No pool-eligible rows remain"):
        carve(test_rows, train_rows, pool_target=6, drift_min_adv=2)




def test_data_manifest_covers_13_files():
    assert len(DATA_MANIFEST_SHA256) == 13
    assert {"eval", "pool"} <= set(DATA_MANIFEST_SHA256)
    assert sum(k.startswith("drift/") for k in DATA_MANIFEST_SHA256) == 11


def test_verify_against_manifest_flags_mismatch_and_missing():
    written = {k: {"n": 1, "sha256": v} for k, v in DATA_MANIFEST_SHA256.items()}
    ok = verify_against_manifest(written)
    assert ok["all_match"] and ok["n_matched"] == 13

    bad = {k: dict(v) for k, v in written.items()}
    bad["eval"]["sha256"] = "0" * 64
    rep = verify_against_manifest(bad)
    assert not rep["all_match"] and rep["n_matched"] == 12
    assert rep["files"]["eval"]["match"] is False

    partial = {k: v for k, v in written.items() if k != "pool"}
    rep = verify_against_manifest(partial)
    assert not rep["all_match"] and "pool" in rep["missing_files"]


def test_write_jsonl_requires_force_to_overwrite_and_rejects_empty(tmp_path):
    target = tmp_path / "x.jsonl"
    info = write_jsonl(target, [{"a": 1}])
    assert info["n"] == 1 and info["sha256"] == sha256_file(target)
    with pytest.raises(RebuildError, match="output file already exists"):
        write_jsonl(target, [{"a": 2}])
    write_jsonl(target, [{"a": 2}], force=True)
    with pytest.raises(RebuildError, match="cannot write an empty JSONL file"):
        write_jsonl(tmp_path / "y.jsonl", [])


def test_write_jsonl_serialization_is_stable(tmp_path):
    rows = [{"item_id": "it_000000000000", "ground_truth": 1, "text": "café"}]
    a = write_jsonl(tmp_path / "a.jsonl", rows)
    b = write_jsonl(tmp_path / "b.jsonl", rows)
    assert a["sha256"] == b["sha256"]
    raw = (tmp_path / "a.jsonl").read_text(encoding="utf-8")
    assert raw == json.dumps(rows[0], ensure_ascii=False) + "\n"


@pytest.mark.parametrize("force", [False, True])
def test_rebuild_rejects_stale_corpus_files_before_writing(tmp_path, monkeypatch, force):
    test, train = synthetic_corpus()
    result = carve(test, train, pool_target=6, drift_min_adv=2)
    monkeypatch.setattr(W, "load_wildguardmix", lambda **kw: (test, train))
    monkeypatch.setattr(W, "carve", lambda *args: result)
    stale = tmp_path / "drift_candidates/old_family.jsonl"
    stale.parent.mkdir()
    stale.write_text('{"item_id":"stale"}\n')
    before = stale.read_bytes()
    with pytest.raises(RebuildError, match="Unexpected.*old_family"):
        W.rebuild(tmp_path, force=force)
    assert stale.read_bytes() == before
    assert not (tmp_path / "eval.jsonl").exists()


def test_rebuild_preflights_all_outputs_and_force_preserves_bytes(tmp_path, monkeypatch):
    test, train = synthetic_corpus()
    result = carve(test, train, pool_target=6, drift_min_adv=2)
    monkeypatch.setattr(W, "load_wildguardmix", lambda **kw: (test, train))
    monkeypatch.setattr(W, "carve", lambda *args: result)
    (tmp_path / "pool.jsonl").write_text("existing\n")
    with pytest.raises(RebuildError, match="already exists"):
        W.rebuild(tmp_path)
    assert not (tmp_path / "eval.jsonl").exists()
    assert (tmp_path / "pool.jsonl").read_text() == "existing\n"
    W.rebuild(tmp_path, force=True)
    before = {p.relative_to(tmp_path): p.read_bytes() for p in tmp_path.rglob("*.jsonl")}
    (tmp_path / "report.json").write_text("{}")
    W.rebuild(tmp_path, force=True)
    assert {p.relative_to(tmp_path): p.read_bytes() for p in tmp_path.rglob("*.jsonl")} == before
    assert before[Path("eval.jsonl")] == "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in result.eval_rows).encode()
