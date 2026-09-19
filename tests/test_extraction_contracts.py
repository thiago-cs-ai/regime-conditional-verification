import json

import numpy as np
import pytest

from rcv.extraction import _io, beaver_extract, lg3_extract, wg_extract


def item(**updates):
    return {"item_id": "i0", "prompt": "p", "response": "", "ground_truth": 0, **updates}


@pytest.mark.parametrize("rows", [
    [item(item_id=None)], [item(prompt=None)], [item(response=[])],
    [item(ground_truth=0.5)], [item(ground_truth="0")], [item(), item()],
    [{"item_id": "i0", "prompt": "p", "response": ""}],
])
def test_invalid_labeled_rows_are_rejected(rows):
    with pytest.raises(ValueError):
        _io.validate_labeled_rows(rows)


def test_empty_response_is_valid():
    _io.validate_labeled_rows([item()])


@pytest.fixture
def extraction(tmp_path, monkeypatch):
    def invoke(module, rows, result=None):
        source, output, embeddings = (tmp_path / p for p in ("input.jsonl", "items.jsonl", "Z.npy"))
        _io.write_jsonl(source, rows)
        argv = ["--input", str(source), "--output", str(output), "--embeddings", str(embeddings)]
        if module is lg3_extract:
            monkeypatch.setattr(module, "enable_determinism", lambda seed: None)
            monkeypatch.setattr(module, "load_model", lambda *args: (None, None, 1, 2, 2))
            monkeypatch.setattr(module, "template_hash", lambda tok: "fixture")
            monkeypatch.setattr(module, "extract_row", lambda *args: result)
            argv += ["--expect-z-dim", "2", "--fb-abort", "1"]
        elif module is wg_extract:
            monkeypatch.setattr(module, "enable_determinism", lambda seed: None)
            monkeypatch.setattr(module, "WG_Z_DIM", 2)
            monkeypatch.setattr(module, "load_wildguard", lambda *args: (None, None, 2))
            monkeypatch.setattr(module, "process_one", lambda *args: result)
        else:
            monkeypatch.setattr(module, "BEAVER_Z_DIM", 2)
            monkeypatch.setattr(module.beaver_stub, "load_beaver", lambda *args: (None, None))
            monkeypatch.setattr(module.beaver_stub, "beaver_verdict", lambda *args: result[:2])
            monkeypatch.setattr(module.beaver_stub, "beaver_embedding", lambda *args: result[2])
        code = module.main(argv)
        return code, _io.read_jsonl(output), np.load(embeddings), json.loads(output.with_suffix(".stats.json").read_text())
    return invoke


@pytest.mark.parametrize("module", [lg3_extract, wg_extract, beaver_extract])
def test_missing_target_is_rejected_before_inference(extraction, module):
    row = item()
    del row["ground_truth"]
    with pytest.raises(ValueError, match="ground_truth"):
        extraction(module, [row])


@pytest.mark.parametrize("score,vector", [
    (np.nan, np.ones(2)), (np.inf, np.ones(2)), (1.2, np.ones(2)),
    (0.5, np.ones((2, 1))), (0.5, np.array([1.0, np.nan])),
])
def test_lg3_invalid_result_uses_existing_error_policy(extraction, score, vector):
    code, rows, z, stats = extraction(lg3_extract, [item()], (0, score, vector))
    assert code == 0
    assert stats["n_err"] == 1 and stats["fb_rate"] == 1.0
    assert rows[0]["z_error"] == 1 and rows[0]["lg3_score"] == 0.5
    np.testing.assert_array_equal(z, np.zeros((1, 2), dtype=np.float32))


@pytest.mark.parametrize("score,vector", [(np.nan, np.ones(2)), (0.5, np.ones((2, 1)))])
def test_wg_invalid_result_cannot_complete(extraction, score, vector):
    with pytest.raises(ValueError, match="score|embedding"):
        extraction(wg_extract, [item()], (vector, score, 1, "yes", False))


def test_wg_placeholder_and_parser_fallback_are_distinct(extraction):
    _, rows, _, stats = extraction(wg_extract, [item()], None)
    assert rows[0]["z_error"] == 1 and stats["n_degenerate"] == 1
    _, rows, _, stats = extraction(wg_extract, [item()], (np.ones(2), 0.75, 1, "yes", True))
    assert "z_error" not in rows[0]
    assert stats["n_degenerate"] == 0 and stats["n_parse_fallback"] == 1


def test_beaver_rejects_wrong_embedding_width(extraction):
    with pytest.raises(ValueError, match="shape"):
        extraction(beaver_extract, [item()], (np.array([4.0]), np.array([1]), np.ones((1, 3))))
