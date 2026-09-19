from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from rcv.rebuild.manifests import (
    ManifestVerificationError,
    canonical_payload_sha256,
    iter_id_list_fields,
    load_payload_verified,
    load_universe_ids,
    sha_ids,
    verify_carve_manifest,
)


def write_manifest(path: Path, payload: dict[str, Any], *, tamper: bool = False) -> Path:
    obj = {
        "payload": payload,
        "payload_sha256": canonical_payload_sha256(payload),
        "generated_utc": "2026-01-01T00:00:00Z",
    }
    if tamper:
        obj["payload"] = dict(payload, extra="edited-after-write")
    path.write_text(json.dumps(obj, indent=2))
    return path


def carve_payload(ids: list[str], **extra: Any) -> dict[str, Any]:
    payload = {
        "manifest_kind": "probe_slice",
        "seed": 42,
        "item_ids": ids,
        "item_ids_sha256": sha_ids(ids),
    }
    payload.update(extra)
    return payload




def test_sha_ids_is_newline_joined_sha256():
    ids = ["it_aaaaaaaaaaaa", "it_bbbbbbbbbbbb"]
    assert sha_ids(ids) == hashlib.sha256("\n".join(ids).encode()).hexdigest()


def test_sha_ids_is_order_sensitive():
    assert sha_ids(["a", "b"]) != sha_ids(["b", "a"])


def test_canonical_payload_sha_ignores_key_order():
    assert canonical_payload_sha256({"a": 1, "b": [1, 2]}) == canonical_payload_sha256(
        {"b": [1, 2], "a": 1}
    )


def test_canonical_payload_sha_sanitizes_non_finite():
    assert canonical_payload_sha256({"x": float("nan")}) == canonical_payload_sha256({"x": None})


def test_canonical_payload_sha_excludes_envelope(tmp_path):
    payload = carve_payload(["it_aaaaaaaaaaaa"])
    p1 = write_manifest(tmp_path / "m1.json", payload)
    obj = json.loads(p1.read_text())
    obj["generated_utc"] = "2030-12-31T23:59:59Z"
    p2 = tmp_path / "m2.json"
    p2.write_text(json.dumps(obj))
    assert load_payload_verified(p1) == load_payload_verified(p2)




def test_load_payload_verified_detects_tamper(tmp_path):
    p = write_manifest(tmp_path / "m.json", carve_payload(["it_aaaaaaaaaaaa"]), tamper=True)
    with pytest.raises(ManifestVerificationError, match="Payload SHA-256 mismatch"):
        load_payload_verified(p)


def test_load_payload_verified_rejects_unbound_json(tmp_path):
    p = tmp_path / "m.json"
    p.write_text(json.dumps({"ids": ["a"]}))
    with pytest.raises(ManifestVerificationError, match="missing 'payload'"):
        load_payload_verified(p)




def test_iter_id_list_fields_matches_both_twin_conventions():
    ids_a, ids_b = ["it_aaaaaaaaaaaa"], ["it_bbbbbbbbbbbb", "it_cccccccccccc"]
    payload = {
        "ft_train_ids": ids_a,
        "ft_train_sha256": sha_ids(ids_a),
        "item_ids": ids_b,
        "item_ids_sha256": sha_ids(ids_b),
        "unpaired_ids": ["it_dddddddddddd"],
        "n_max": 600,
    }
    found = {name: (ids, sha) for name, ids, sha in iter_id_list_fields(payload)}
    assert set(found) == {"ft_train_ids", "item_ids"}
    assert found["ft_train_ids"][0] == ids_a and found["item_ids"][0] == ids_b




def test_verify_carve_manifest_passes_and_reports(tmp_path):
    ids = ["it_aaaaaaaaaaaa", "it_bbbbbbbbbbbb"]
    p = write_manifest(tmp_path / "m.json", carve_payload(ids))
    rep = verify_carve_manifest(p, universe_ids=set(ids) | {"it_cccccccccccc"})
    assert rep["verified"] and rep["manifest_kind"] == "probe_slice" and rep["seed"] == 42
    lists = rep["id_lists"]["item_ids"]
    assert lists["n_ids"] == 2 and lists["sha_match"] and lists["n_outside_universe"] == 0


def test_verify_carve_manifest_fails_on_id_sha_mismatch(tmp_path):
    payload = carve_payload(["it_aaaaaaaaaaaa"])
    payload["item_ids_sha256"] = "0" * 64
    p = write_manifest(tmp_path / "m.json", payload)
    with pytest.raises(ManifestVerificationError, match="ID-list SHA-256 mismatch"):
        verify_carve_manifest(p)


def test_verify_carve_manifest_fails_on_universe_escape(tmp_path):
    ids = ["it_aaaaaaaaaaaa", "it_bbbbbbbbbbbb"]
    p = write_manifest(tmp_path / "m.json", carve_payload(ids))
    with pytest.raises(ManifestVerificationError, match="outside the supplied universe"):
        verify_carve_manifest(p, universe_ids={"it_aaaaaaaaaaaa"})


def test_verify_carve_manifest_skips_membership_without_universe(tmp_path):
    p = write_manifest(tmp_path / "m.json", carve_payload(["it_aaaaaaaaaaaa"]))
    rep = verify_carve_manifest(p, universe_ids=None)
    assert rep["verified"]
    assert rep["id_lists"]["item_ids"]["n_outside_universe"] is None


def test_verify_carve_manifest_reads_config_seed(tmp_path):
    payload = carve_payload(["it_aaaaaaaaaaaa"])
    del payload["seed"]
    payload["config"] = {"seed": 20260705}
    p = write_manifest(tmp_path / "m.json", payload)
    assert verify_carve_manifest(p)["seed"] == 20260705


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        ({"item_ids": ["item-a"]}, "item_ids.*checksum"),
        ({"item_ids": ["item-a"], "item_ids_sha256": 123}, "item_ids.*checksum"),
        ({"item_ids": "item-a"}, "item_ids.*list of string IDs"),
        ({"item_ids": [123], "item_ids_sha256": sha_ids(["123"])}, "item_ids.*list of string IDs"),
        ({"manifest_kind": "probe_slice"}, "no top-level ID lists"),
        (carve_payload(["item-a"], unchecked_ids=["item-b"]), "unchecked_ids.*checksum"),
    ],
    ids=["missing-checksum", "non-string-checksum", "non-list", "non-string-id", "no-lists", "partial-pairs"],
)
def test_verify_carve_manifest_rejects_incomplete_id_metadata(tmp_path, payload, message):
    path = write_manifest(tmp_path / "m.json", payload)
    with pytest.raises(ManifestVerificationError, match=message):
        verify_carve_manifest(path)


def test_verify_carve_manifest_preserves_zero_seed_over_config(tmp_path):
    payload = carve_payload(["item-a"], seed=0, config={"seed": 42})
    path = write_manifest(tmp_path / "m.json", payload)
    assert verify_carve_manifest(path)["seed"] == 0


def test_verify_carve_manifest_accepts_checksummed_empty_list(tmp_path):
    path = write_manifest(tmp_path / "m.json", carve_payload([]))
    report = verify_carve_manifest(path, universe_ids=set())
    assert report["verified"] and report["id_lists"]["item_ids"]["n_ids"] == 0




def test_load_universe_ids_reads_all_corpus_files(tmp_path):
    (tmp_path / "drift_candidates").mkdir()

    def dump(path: Path, ids: list[str]) -> None:
        path.write_text("".join(json.dumps({"item_id": i}) + "\n" for i in ids))

    dump(tmp_path / "eval.jsonl", ["it_aaaaaaaaaaaa"])
    dump(tmp_path / "pool.jsonl", ["it_bbbbbbbbbbbb"])
    dump(tmp_path / "drift_candidates" / "topic.jsonl", ["it_cccccccccccc"])
    assert load_universe_ids(tmp_path) == {
        "it_aaaaaaaaaaaa",
        "it_bbbbbbbbbbbb",
        "it_cccccccccccc",
    }


def test_load_universe_ids_fails_on_empty_dir(tmp_path):
    with pytest.raises(ManifestVerificationError, match="no rebuilt corpus files"):
        load_universe_ids(tmp_path)
