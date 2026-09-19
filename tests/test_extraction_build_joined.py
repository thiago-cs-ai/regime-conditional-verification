from __future__ import annotations

import numpy as np
import pytest

from rcv.extraction import build_joined as BJ
from rcv.extraction import diff_profile as DP


def _rows(preds, ystar, set_ids=None):
    n = len(preds)
    set_ids = set_ids or ["pool"] * n
    rows = []
    for i in range(n):
        agree = int(int(preds[i]) == int(ystar[i]))
        rows.append({
            "item_id": f"it_{i}", "set_id": set_ids[i], "ground_truth": int(ystar[i]),
            "lg3_pred": int(preds[i]), "lg3_score": 0.5, "lg3_agreement": agree,
        })
    return rows


def _base_items(n, set_ids=None):
    set_ids = set_ids or ["pool"] * n
    return [{"item_id": f"it_{i}", "set_id": set_ids[i]} for i in range(n)]


def test_join_ft_surface_passes_on_aligned_input():
    ystar = [1, 0, 1, 0]
    preds = [1, 0, 0, 0]
    rows = _rows(preds, ystar)
    Z = np.ones((4, 3), dtype=np.float32)
    base = {"ystar": np.array(ystar)}
    arrays, report = BJ.join_ft_surface(rows, Z, _base_items(4), base)
    assert report["n"] == 4 and report["order_matches_base"] is True
    assert list(arrays["verdict"]) == preds
    assert list(arrays["agreement"]) == [1, 1, 0, 1]


def test_join_ft_surface_rejects_order_mismatch():
    ystar = [1, 0]
    rows = _rows([1, 0], ystar)
    rows[0]["item_id"] = "WRONG"
    with pytest.raises(AssertionError):
        BJ.join_ft_surface(rows, np.ones((2, 3), np.float32), _base_items(2), {"ystar": np.array(ystar)})


def test_join_ft_surface_rejects_ystar_mismatch():
    rows = _rows([1, 0], [1, 0])
    with pytest.raises(AssertionError):
        BJ.join_ft_surface(rows, np.ones((2, 3), np.float32), _base_items(2), {"ystar": np.array([0, 0])})


def test_join_ft_surface_rejects_z_error_rows():
    rows = _rows([1, 0], [1, 0])
    rows[1]["z_error"] = 1
    with pytest.raises(AssertionError):
        BJ.join_ft_surface(rows, np.ones((2, 3), np.float32), _base_items(2), {"ystar": np.array([1, 0])})


def test_join_ft_surface_rejects_wrong_dtype():
    rows = _rows([1, 0], [1, 0])
    with pytest.raises(AssertionError):
        BJ.join_ft_surface(rows, np.ones((2, 3), np.float64), _base_items(2), {"ystar": np.array([1, 0])})


def test_join_ft_surface_rejects_agreement_tamper():
    rows = _rows([1, 0], [1, 0])
    rows[0]["lg3_agreement"] = 0
    with pytest.raises(AssertionError):
        BJ.join_ft_surface(rows, np.ones((2, 3), np.float32), _base_items(2), {"ystar": np.array([1, 0])})


def test_recall_fpr_basic():
    pred = np.array([1, 1, 0, 0])
    ystar = np.array([1, 0, 1, 0])
    tpr, fpr = DP.recall_fpr(pred, ystar)
    assert tpr == 0.5 and fpr == 0.5


def test_flip_block_counts_directions_and_oracle_movement():
    base = np.array([0, 1, 1, 0])
    ft = np.array([1, 1, 0, 0])
    ystar = np.array([1, 1, 1, 0])
    b = DP.flip_block(np.ones(4, dtype=bool), base, ft, ystar)
    assert b["flips"] == 2 and b["safe_to_unsafe"] == 1 and b["unsafe_to_safe"] == 1
    assert b["flips_toward_oracle"] == 1
    assert b["flips_away_from_oracle"] == 1


def test_flip_block_empty_mask():
    assert DP.flip_block(np.zeros(3, dtype=bool), np.zeros(3, int), np.zeros(3, int), np.zeros(3, int)) == {"n": 0}


def test_compute_profile_by_set_and_exclusion():
    items = [{"item_id": f"it_{i}", "set_id": s}
             for i, s in enumerate(["eval", "pool", "drift:x", "drift:x"])]
    base = {"ystar": np.array([1, 0, 1, 0]), "verdict": np.array([0, 0, 1, 0])}
    ft = {"ystar": np.array([1, 0, 1, 0]), "verdict": np.array([1, 0, 0, 0])}
    excl = {"ft_train_ids": ["it_2"], "ft_test_ids": ["it_3"]}
    prof = DP.compute_profile(items, base, ft, excl)
    assert prof["overall"]["flips"] == 2
    assert "drift:x" in prof["by_set"]
    assert prof["ft_exclusion_view"]["ft_train_burned"]["n"] == 1
    assert prof["ft_exclusion_view"]["held_out_remainder"]["n"] == 2


def test_compute_profile_rejects_length_mismatch():
    items = [{"item_id": "a", "set_id": "pool"}]
    base = {"ystar": np.array([1]), "verdict": np.array([1])}
    ft = {"ystar": np.array([1, 0]), "verdict": np.array([1, 0])}
    with pytest.raises(AssertionError):
        DP.compute_profile(items, base, ft)


@pytest.mark.parametrize("failure", ["rank", "score", "fractional_label", "nonbinary_verdict", "canceling_flags", "base_verdict_shape"])
def test_join_rejects_malformed_arrays(failure):
    rows = _rows([1, 0], [1, 0])
    z = np.ones((2, 3), np.float32)
    base = {"ystar": np.array([1, 0]), "verdict": np.array([1, 0])}
    if failure == "rank":
        z = z[:, :, None]
    elif failure == "score":
        rows[0]["lg3_score"] = float("nan")
    elif failure == "fractional_label":
        rows[1]["ground_truth"] = 0.5
    elif failure == "nonbinary_verdict":
        rows[1]["lg3_pred"] = -1
        rows[1]["lg3_agreement"] = 0
    elif failure == "canceling_flags":
        rows[0]["z_error"], rows[1]["z_error"] = -1, 1
    else:
        base["verdict"] = base["verdict"][:, None]
    with pytest.raises(AssertionError):
        BJ.join_ft_surface(rows, z, _base_items(2), base)


@pytest.mark.parametrize("field", ["base_verdict", "base_ystar", "ft_verdict", "ft_ystar"])
@pytest.mark.parametrize("failure", ["column", "nonbinary"])
def test_profile_rejects_broadcasting_and_lossy_label_casts(field, failure):
    base = {"ystar": np.array([1, 0, 1]), "verdict": np.array([1, 0, 0])}
    ft = {"ystar": np.array([1, 0, 1]), "verdict": np.array([0, 1, 0])}
    side, key = field.split("_")
    target = base if side == "base" else ft
    target[key] = target[key][:, None] if failure == "column" else np.array([0.5, 0, 1])
    with pytest.raises(AssertionError):
        DP.compute_profile(_base_items(3), base, ft)


def test_bad_base_verdict_does_not_replace_joined_outputs(tmp_path):
    from rcv.extraction._io import write_jsonl

    pulled, output = tmp_path / "pulled", tmp_path / "joined"
    pulled.mkdir()
    output.mkdir()
    marker = output / "JOIN_REPORT_lg3ft.json"
    marker.write_bytes(b"previous report")
    write_jsonl(pulled / "lg3ft.items.jsonl", _rows([1, 0], [1, 0]))
    np.save(pulled / "lg3ft.Z.npy", np.ones((2, 3), np.float32))
    write_jsonl(tmp_path / "base.jsonl", _base_items(2))
    np.savez(tmp_path / "base.npz", ystar=np.array([1, 0]), verdict=np.array([[1], [0]]))
    with pytest.raises(AssertionError):
        BJ.main([
            "--pulled", str(pulled), "--out-dir", str(output),
            "--base-items", str(tmp_path / "base.jsonl"), "--base-arrays", str(tmp_path / "base.npz"),
        ])
    assert list(output.iterdir()) == [marker]
    assert marker.read_bytes() == b"previous report"
