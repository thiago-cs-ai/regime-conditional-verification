import json

import numpy as np
import pytest

from rcv.extraction import _io, beaver_extract, lg3_extract, wg_extract


@pytest.mark.parametrize("module", [lg3_extract, wg_extract, beaver_extract])
@pytest.mark.parametrize("failure", ["empty", "stale"])
def test_invalid_extraction_start_does_not_load_model(tmp_path, monkeypatch, module, failure):
    source = tmp_path / "input.jsonl"
    source.write_text("" if failure == "empty" else json.dumps({
        "item_id": "new", "prompt": "p", "response": "", "ground_truth": 0,
    }) + "\n")
    output, embeddings = tmp_path / "items.jsonl", tmp_path / "Z.npy"
    output.write_bytes(b"previous output\n")
    embeddings.write_bytes(b"previous embeddings")
    if failure == "stale":
        _io.flush_chunk(embeddings, output, [{"item_id": "old"}], [np.ones(2)], 10)
    before = {p.name: p.read_bytes() for p in tmp_path.iterdir()}

    def unexpected_load(*args, **kwargs):
        pytest.fail("Model loading must follow input and chunk checks")

    if module is beaver_extract:
        monkeypatch.setattr(module.beaver_stub, "load_beaver", unexpected_load)
    else:
        monkeypatch.setattr(module, "enable_determinism", lambda seed: None)
        monkeypatch.setattr(module, "load_model" if module is lg3_extract else "load_wildguard", unexpected_load)
    args = module.build_parser().parse_args([
        "--input", str(source), "--output", str(output), "--embeddings", str(embeddings),
    ])

    with pytest.raises(ValueError, match="empty|chunks"):
        module.run(args)
    assert {p.name: p.read_bytes() for p in tmp_path.iterdir()} == before


@pytest.mark.parametrize("failure", ["indices", "pair_rows", "total_rows", "rank", "width"])
def test_stitch_rejects_invalid_chunks_without_changing_files(tmp_path, failure):
    output, embeddings = tmp_path / "items.jsonl", tmp_path / "Z.npy"
    output.write_bytes(b"previous output\n")
    embeddings.write_bytes(b"previous embeddings")
    _, first, _ = _io.flush_chunk(embeddings, output, [{"id": 0}], [np.ones(2)], 0)
    _, second, _ = _io.flush_chunk(embeddings, output, [{"id": 1}], [np.ones(2)], 1)
    if failure == "indices":
        second.rename(embeddings.with_suffix(".chunk0002.npy"))
    elif failure == "pair_rows":
        np.save(first, np.ones((2, 2), dtype=np.float32))
        np.save(second, np.ones((0, 2), dtype=np.float32))
    elif failure == "rank":
        np.save(first, np.ones(1, dtype=np.float32))
    elif failure == "width":
        np.save(second, np.ones((1, 3), dtype=np.float32))
    before = {p.name: p.read_bytes() for p in tmp_path.iterdir()}

    with pytest.raises(ValueError):
        _io.stitch_chunks(embeddings, output, expected_rows=3 if failure == "total_rows" else 2)
    assert {p.name: p.read_bytes() for p in tmp_path.iterdir()} == before


def test_stitch_orders_chunks_numerically_and_preserves_artifacts(tmp_path):
    output, embeddings = tmp_path / "items.jsonl", tmp_path / "Z.npy"
    rows = [{"id": "first"}, {"id": "second"}, {"id": "third"}]
    vectors = np.arange(6, dtype=np.float32).reshape(3, 2)
    cj1, cn1, _ = _io.flush_chunk(embeddings, output, rows[2:], vectors[2:], 10)
    cj0, cn0, _ = _io.flush_chunk(embeddings, output, rows[:2], vectors[:2], 2)
    cj0 = cj0.rename(output.with_suffix(".chunk2.jsonl"))
    cn0 = cn0.rename(embeddings.with_suffix(".chunk2.npy"))
    expected_jsonl = cj0.read_bytes() + cj1.read_bytes()
    reference = tmp_path / "expected.npy"
    np.save(reference, vectors)

    assert _io.stitch_chunks(embeddings, output, expected_rows=3) == (3, (3, 2))
    assert output.read_bytes() == expected_jsonl
    assert embeddings.read_bytes() == reference.read_bytes()
    assert not any(p.exists() for p in (cj0, cn0, cj1, cn1))
