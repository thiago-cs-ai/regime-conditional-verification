import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from fixture_frame import build_fixture_frame
from rcv.rebuild.gen_rubric_ystar import MODEL, record_metadata

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("policy_workflow", ROOT / "examples/bring_your_policy.py")
W = importlib.util.module_from_spec(spec)
spec.loader.exec_module(W)
TEMPLATE = ROOT / "examples/bring_your_policy.yaml"


def jsonl(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    return path


def item(iid, **extra):
    return {"item_id": iid, "prompt": iid, "response": "", **extra}


def test_prepare_joins_by_id_uses_last_success_and_preserves_order(tmp_path):
    baseline = jsonl(tmp_path / "base.jsonl", [item("z", ground_truth=0), item("a")])
    stream = tmp_path / "stream"
    stream.mkdir()
    jsonl(stream / "b.jsonl", [item("second")])
    jsonl(stream / "a.jsonl", [item("first")])
    labels = jsonl(tmp_path / "labels.jsonl", [
        {"item_id": "first", "label": 0}, {"item_id": "z", "label": 0},
        {"item_id": "z", "label": 1, "reason": "omit"}, {"item_id": "z", "label": None},
        {"item_id": "second", "label": 1}, {"item_id": "a", "label": 1},
        {"item_id": "extra", "label": 0},
    ])
    out = tmp_path / "prepared.jsonl"
    rows = W.prepare(baseline, stream, [labels], "label", out)
    assert [r["item_id"] for r in rows] == ["z", "a", "first", "second"]
    assert [r["ground_truth"] for r in rows] == [1, 1, 0, 1]
    assert [r["is_stream"] for r in rows] == [False, False, True, True]
    assert all(set(r) == {"item_id", "prompt", "response", "ground_truth", "is_stream"} for r in rows)
    before = out.read_bytes()
    with pytest.raises(RuntimeError, match="already exists"):
        W.prepare(baseline, stream, [labels], "label", out)
    assert out.read_bytes() == before


def test_overlap_exclusion_and_limits_preserve_connected_baseline_groups(tmp_path):
    base = [item("a", prompt="Same", group_id="g1"), item("b", group_id="g1"),
            item("c", prompt=" same ", group_id="g2"), item("d", group_id="g2"),
            item("e", group_id="pair"), item("f", group_id="pair"), item("g")]
    traffic = [item("s", prompt="b"), item("later", prompt="g")]
    baseline = jsonl(tmp_path / "base.jsonl", base)
    stream = jsonl(tmp_path / "stream.jsonl", traffic)
    labels = jsonl(tmp_path / "labels.jsonl", [{"item_id": r["item_id"], "y": 0}
                                              for r in base + traffic])
    out = tmp_path / "prepared.jsonl"
    with pytest.raises(ValueError, match="share"):
        W.prepare(baseline, stream, [labels], "y", out)
    assert not out.exists()
    rows = W.prepare(baseline, stream, [labels], "y", out, exclude_shared_groups=True,
                     max_baseline_items=1, max_stream_items=1)
    assert [r["item_id"] for r in rows] == ["g", "s"]


@pytest.mark.parametrize("bad", ["missing", "duplicate", "fraction", "string"])
def test_bad_preparation_inputs_leave_no_output(tmp_path, bad):
    base = [item("b")]
    stream = [item("b" if bad == "duplicate" else "s")]
    labels = [{"item_id": "b", "y": 0}, {"item_id": "s", "y": 1}]
    if bad == "missing":
        labels.pop()
    if bad in ("fraction", "string"):
        labels[0]["y"] = 0.5 if bad == "fraction" else "0"
    out = tmp_path / "prepared.jsonl"
    with pytest.raises(ValueError):
        W.prepare(jsonl(tmp_path / "base.jsonl", base), jsonl(tmp_path / "stream.jsonl", stream),
                  [jsonl(tmp_path / "labels.jsonl", labels)], "y", out)
    assert not out.exists()


@pytest.mark.parametrize("changed", [None, "text", "policy", "missing_metadata"])
def test_join_checks_custom_policy_and_text_identity(tmp_path, changed):
    rows = [item("a"), item("b")]
    labels = [{"item_id": r["item_id"], "rubric_ystar": 1,
               **record_metadata(r, MODEL, "policy")} for r in rows]
    if changed == "text":
        rows[1]["response"] = "changed"
    elif changed == "policy":
        labels[1]["policy_sha256"] = "other"
    elif changed == "missing_metadata":
        del labels[1]["policy_sha256"]
    path = jsonl(tmp_path / "labels.jsonl", labels)
    if changed is None:
        assert [r["ground_truth"] for r in W.join_labels(rows, [path], "rubric_ystar")] == [1, 1]
    else:
        with pytest.raises(ValueError, match="identity|input_sha256"):
            W.join_labels(rows, [path], "rubric_ystar")


def extraction_files(tmp_path, inputs, verdicts, z):
    prepared = jsonl(tmp_path / "inputs.jsonl", inputs)
    outputs = [{**r, "lg3_pred": int(v), "lg3_score": 0.7 if v else 0.3,
                "lg3_agreement": int(v == r["ground_truth"])}
               for r, v in zip(inputs, verdicts, strict=True)]
    extracted = jsonl(tmp_path / "items.jsonl", outputs)
    embeddings = tmp_path / "Z.npy"
    np.save(embeddings, z)
    stats = {"n": len(inputs), "n_rows": len(inputs), "n_err": 0, "D": z.shape[1],
             "shape": list(z.shape), "z_npy_sha256": W.digest_of(embeddings)}
    extracted.with_suffix(".stats.json").write_text(json.dumps(stats))
    return prepared, extracted, embeddings


@pytest.fixture
def small_extraction(tmp_path):
    inputs = [item("z", is_stream=False, ground_truth=1),
              item("a", is_stream=False, ground_truth=0),
              item("stream-z", is_stream=True, ground_truth=0),
              item("stream-a", is_stream=True, ground_truth=1)]
    return extraction_files(tmp_path, inputs, [0, 0, 1, 1],
                            np.arange(12, dtype=np.float32).reshape(4, 3))


def test_frame_restores_every_column_after_canonical_id_sort(small_extraction):
    frame = W.extraction_frame(*small_extraction)
    assert frame["item_id"].tolist() == ["z", "a", "stream-z", "stream-a"]
    assert frame["verdict"].tolist() == [0, 0, 1, 1]
    assert frame["oracle"].tolist() == [1, 0, 0, 1]
    assert frame["is_stream"].tolist() == [False, False, True, True]
    np.testing.assert_array_equal(frame["representation"], np.arange(12).reshape(4, 3))


@pytest.mark.parametrize(("bad", "match"), [
    ("placeholder", "z_error"), ("agreement", "lg3_agreement"), ("order", "order"),
    ("checksum", "z_npy_sha256"), ("shape", "shape"), ("nan", "finite"),
    ("role", "boolean"), ("target", "ground_truth"), ("shared", "share families"),
])
def test_extraction_errors_prevent_run_artifacts(tmp_path, small_extraction, bad, match):
    inputs, extracted, embeddings = small_extraction
    rows = W.read_jsonl(extracted)
    if bad == "placeholder":
        rows[0]["z_error"] = 1
    elif bad == "agreement":
        rows[0]["lg3_agreement"] = 1
    elif bad == "order":
        rows.reverse()
    elif bad in ("role", "target", "shared"):
        original = W.read_jsonl(inputs)
        if bad == "role":
            original[0]["is_stream"] = rows[0]["is_stream"] = 0
        elif bad == "target":
            original[0]["ground_truth"] = rows[0]["ground_truth"] = True
        else:
            original[2]["prompt"] = rows[2]["prompt"] = original[0]["prompt"]
        jsonl(inputs, original)
    elif bad in ("checksum", "shape"):
        path = extracted.with_suffix(".stats.json")
        stats = json.loads(path.read_text())
        stats["z_npy_sha256" if bad == "checksum" else "shape"] = "wrong"
        path.write_text(json.dumps(stats))
    elif bad == "nan":
        z = np.load(embeddings)
        z[0, 0] = np.nan
        np.save(embeddings, z)
    jsonl(extracted, rows)
    out = tmp_path / "run"
    with pytest.raises(ValueError, match=match):
        W.run(inputs, extracted, embeddings, TEMPLATE, out)
    assert not out.exists()


@pytest.mark.parametrize("horizon", [None, 342, 0, "bad", True])
def test_config_horizon_is_defaulted_or_preserved(tmp_path, horizon):
    config = yaml.safe_load(TEMPLATE.read_text())
    config["monitor"]["calibration"]["stream_length"] = horizon
    path = tmp_path / "template.yaml"
    path.write_text(yaml.safe_dump(config))
    if horizon in (0, "bad") or horizon is True:
        with pytest.raises(ValueError, match="stream_length"):
            W.resolve_config(path, tmp_path / "frame.npz", 500)
    else:
        resolved = W.resolve_config(path, tmp_path / "frame.npz", 500)
        assert resolved["monitor"]["calibration"]["stream_length"] == (500 if horizon is None else horizon)


def test_prepare_and_run_through_real_fitting_monitor_and_audit(tmp_path):
    source = build_fixture_frame()
    n = len(source["verdict"])
    inputs = [item(f"user-{n-i:05d}", group_id=f"family-{int(source['family'][i])}")
              for i in range(n)]
    base = jsonl(tmp_path / "base.jsonl", [r for r, traffic in zip(inputs, source["is_stream"], strict=True) if not traffic])
    stream = jsonl(tmp_path / "stream.jsonl", [r for r, traffic in zip(inputs, source["is_stream"], strict=True) if traffic])
    labels = jsonl(tmp_path / "labels.jsonl", [{"item_id": r["item_id"], "y": int(y)}
                                              for r, y in zip(inputs, source["oracle"], strict=True)][::-1])
    prepared = tmp_path / "prepared.jsonl"
    assert W.main(["prepare", "--baseline", str(base), "--stream", str(stream),
                   "--labels", str(labels), "--label-field", "y", "--out", str(prepared)]) == 0
    files = extraction_files(tmp_path, W.read_jsonl(prepared), source["verdict"],
                             source["representation"].astype(np.float32))
    out = tmp_path / "run"
    args = ["run", "--inputs", str(files[0]), "--extracted", str(files[1]),
            "--embeddings", str(files[2]), "--config", str(TEMPLATE), "--out", str(out)]
    assert W.main(args) == 0
    record = json.loads((out / "results/seed_42.json").read_text())
    kinds = [event["kind"] for event in record["loop_events"]]
    assert "alarm" in kinds and "acceptance_evaluated" in kinds
    assert record["monitor"]["deployed"] is not None
    assert record["readout"]["corrected_adherence"] > record["readout"]["raw_adherence"]
    assert record["total_oracle_labels_spent"] == (
        record["readout"]["oracle_labels_spent"]
        + sum(change["oracle_labels_spent"] for change in record["change_log"]))
    assert record["provenance"]["frame_sha256"] == W.digest_of(out / "frame.npz")
    with np.load(out / "frame.npz") as frame:
        assert frame["item_id"].tolist() == [r["item_id"] for r in inputs]
        np.testing.assert_array_equal(frame["representation"], source["representation"].astype(np.float32))
        for family in np.unique(source["family"]):
            assert len(set(frame["family"][source["family"] == family])) == 1
    before = (out / "results/seed_42.json").read_bytes()
    with pytest.raises(SystemExit) as exc:
        W.main(args)
    assert exc.value.code == 1
    assert (out / "results/seed_42.json").read_bytes() == before


@pytest.mark.parametrize("terminal", ["escalation_demanded", "reference_window_exhausted"])
def test_summary_uses_label_ledger_and_keeps_terminal_kinds_distinct(terminal):
    record = {"seed": 42, "readout": {"raw_adherence": 0.8, "corrected_adherence": 0.85,
                                       "caught_share": 0.5, "caught_share_population": 10,
                                       "oracle_labels_spent": 100},
              "total_oracle_labels_spent": 160, "monitor": {"recalibrations": []},
              "loop_events": [{"kind": k} for k in ["alarm", "audit", "audit",
                                                       "repair_accepted", terminal]]}
    lines = "\n".join(W.summary_lines(record))
    assert "initial=100, maintenance=60, total=160" in lines
    assert "accepted repairs: 1; reference refreshes: 0" in lines
    assert f"Terminal events: {terminal}" in lines
