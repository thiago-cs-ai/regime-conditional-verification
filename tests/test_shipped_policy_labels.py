import hashlib

import pytest

from conftest import REPO_ROOT
from rcv.rebuild._jsonl import read_jsonl
from rcv.rebuild.label_map import apply_label_map, gate_rows


@pytest.fixture(params=[
    ("native_pku_labels.jsonl", "rubric_ystar", 16422, 9040, 1768),
    ("wgmix_labels.jsonl", "new_label", 20199, 4956, 901),
])
def published_map(request):
    filename, field, count, unsafe, commitment = request.param
    path = REPO_ROOT / "labels" / filename
    return path, read_jsonl(path), field, count, unsafe, commitment


def test_published_map_integrity_and_schema(published_map):
    path, rows, field, count, unsafe, commitment = published_map
    checksums = {
        name: sha for sha, name in (
            line.split() for line in (REPO_ROOT / "labels/SHA256SUMS").read_text().splitlines()
        )
    }
    assert hashlib.sha256(path.read_bytes()).hexdigest() == checksums[f"labels/{path.name}"]
    assert gate_rows(rows, primary_label=field)["n_rows"] == count
    assert all(set(row) == {"item_id", field, "rule_fired"} for row in rows)
    assert sum(row[field] for row in rows) == unsafe
    assert sum(row["rule_fired"] == "2.2_commitment" for row in rows) == commitment


def test_published_map_joins_by_id_preserving_input_order_and_target(published_map):
    _, rows, field, _, _, _ = published_map
    items = [
        {"item_id": row["item_id"], "prompt": "synthetic", "response": "",
         "ground_truth": 1 - row[field]}
        for row in reversed(rows)
    ]
    joined = apply_label_map(items, rows, primary_label=field)
    assert len(joined) == len(items)
    for item, labeled, expected in zip(items, joined, reversed(rows), strict=True):
        assert labeled == {**item, field: expected[field], "rule_fired": expected["rule_fired"]}
