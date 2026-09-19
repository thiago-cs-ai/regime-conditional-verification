from __future__ import annotations

import contextlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from rcv.extraction import finetune as F


class FakeTok:
    chat_template = "FAKE"
    unk_token_id = 0

    def apply_chat_template(self, messages, add_generation_prompt=True, return_tensors="pt"):
        return torch.tensor([[10, 11, 12]], dtype=torch.long)

    def encode(self, s, add_special_tokens=False):
        return [271]

    def convert_tokens_to_ids(self, tok):
        return {"safe": 19193, "unsafe": 39257}.get(tok, 0)


def test_lora_config_dict_is_the_frozen_recipe():
    cfg = F.lora_config_dict()
    assert cfg["r"] == 16 and cfg["lora_alpha"] == 32 and cfg["lora_dropout"] == 0.05
    assert cfg["task_type"] == "CAUSAL_LM" and cfg["bias"] == "none"
    assert cfg["target_modules"] == ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def test_lora_config_hash_stable_across_calls():
    a = json.dumps(F.lora_config_dict(), sort_keys=True)
    b = json.dumps(F.lora_config_dict(16, 32, 0.05), sort_keys=True)
    assert a == b


def test_defaults_match_seed_runbook():
    assert F.DEFAULTS["lr"] == 2e-5
    assert F.DEFAULTS["step_ceiling"] == 30
    assert F.DEFAULTS["micro_batch"] * F.DEFAULTS["grad_accum"] == 16


def test_build_ft_example_single_supervised_slot():
    tok = FakeTok()
    ids, labels = F.build_ft_example(tok, 19193, 39257, "p", "r", ground_truth=1)
    assert ids.tolist() == [10, 11, 12, 271, 39257]
    assert (labels != -100).sum().item() == 1
    assert labels[-1].item() == 39257
    F.assert_single_decision_supervision(ids, labels, 19193, 39257, nl_id=271)


def test_build_ft_example_safe_target_polarity():
    ids, labels = F.build_ft_example(FakeTok(), 19193, 39257, "p", "r", ground_truth=0)
    assert ids[-1].item() == 19193 and labels[-1].item() == 19193


def test_assert_single_supervision_rejects_two_positions():
    ids = torch.tensor([10, 271, 39257])
    labels = torch.tensor([-100, 39257, 39257])
    with pytest.raises(AssertionError):
        F.assert_single_decision_supervision(ids, labels, 19193, 39257, nl_id=271)


def test_assert_single_supervision_rejects_wrong_position():
    ids = torch.tensor([10, 271, 39257])
    labels = torch.tensor([-100, 39257, -100])
    with pytest.raises(AssertionError):
        F.assert_single_decision_supervision(ids, labels, 19193, 39257, nl_id=271)


def test_assert_single_supervision_rejects_missing_nl_before_target():
    ids = torch.tensor([10, 11, 39257])
    labels = torch.tensor([-100, -100, 39257])
    with pytest.raises(AssertionError):
        F.assert_single_decision_supervision(ids, labels, 19193, 39257, nl_id=271)


def test_binding_gate_all_pass():
    ok, _ = F.binding_gate(val_delta=0.025, overfit_gap=0.008, degenerate=False,
                           retention_flips=0)
    assert ok is True


def test_binding_gate_boundaries_pass():
    ok, _ = F.binding_gate(val_delta=-0.02, overfit_gap=0.15, degenerate=False,
                           retention_flips=5)
    assert ok is True


@pytest.mark.parametrize(
    "kw",
    [
        dict(val_delta=-0.03, overfit_gap=0.0, degenerate=False, retention_flips=0),
        dict(val_delta=0.02, overfit_gap=0.2, degenerate=False, retention_flips=0),
        dict(val_delta=0.02, overfit_gap=0.0, degenerate=True, retention_flips=0),
        dict(val_delta=0.02, overfit_gap=0.0, degenerate=False, retention_flips=6),
    ],
)
def test_binding_gate_each_failure_mode(kw):
    ok, _ = F.binding_gate(**kw)
    assert ok is False


def test_materialize_ft_corpus_carves_the_splits(tmp_path):
    universe = tmp_path / "universe.jsonl"
    rows = [
        {"item_id": f"it_{i}", "prompt": f"P{i}", "response": f"R{i}", "ground_truth": i % 2}
        for i in range(6)
    ]
    universe.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    dist = tmp_path / "dist.json"
    dist.write_text(json.dumps({
        "splits": {"ft": ["it_0", "it_1"], "val": ["it_2"], "family": ["it_3"], "retention": ["it_4"]},
        "labels": {"it_0": 1},
    }))
    out = tmp_path / "corpus"
    written = F.materialize_ft_corpus(universe, dist, out)
    assert written["ft"]["n"] == 2 and written["val"]["n"] == 1
    ft_rows = [json.loads(x) for x in (out / "ft.jsonl").read_text().splitlines()]
    assert ft_rows[0] == {"item_id": "it_0", "prompt": "P0", "response": "R0", "ground_truth": 1}


def test_materialize_ft_corpus_unknown_id_raises(tmp_path):
    universe = tmp_path / "u.jsonl"
    universe.write_text(json.dumps({"item_id": "a", "prompt": "p", "response": "r", "ground_truth": 0}) + "\n")
    dist = tmp_path / "d.json"
    dist.write_text(json.dumps({"splits": {"ft": ["MISSING"]}, "labels": {}}))
    with pytest.raises(KeyError):
        F.materialize_ft_corpus(universe, dist, tmp_path / "o")


def corpus(tmp_path, n=3):
    directory = tmp_path / "corpus"
    directory.mkdir()
    for name, count in (("ft", n), ("val", 1), ("retention", 1)):
        F.write_jsonl(directory / f"{name}.jsonl", [
            {"item_id": f"{name}{i}", "prompt": "p", "response": "", "ground_truth": i % 2}
            for i in range(count)
        ])
    return directory


@pytest.mark.parametrize("failure", ["empty_val", "overlap", "duplicate", "label", "accumulation", "zero_batch", "old_output"])
def test_invalid_training_inputs_stop_before_model_loading(tmp_path, monkeypatch, failure):
    directory = corpus(tmp_path)
    out = tmp_path / "run"
    argv = ["--seed", "42", "--corpus-dir", str(directory), "--out", str(out)]
    if failure == "empty_val":
        (directory / "val.jsonl").write_text("")
    elif failure == "overlap":
        F.write_jsonl(directory / "val.jsonl", F.read_jsonl(directory / "ft.jsonl")[:1])
    elif failure == "duplicate":
        rows = F.read_jsonl(directory / "ft.jsonl")
        F.write_jsonl(directory / "ft.jsonl", rows + rows[:1])
    elif failure == "label":
        rows = F.read_jsonl(directory / "ft.jsonl")
        rows[0]["ground_truth"] = 0.5
        F.write_jsonl(directory / "ft.jsonl", rows)
    elif failure == "accumulation":
        argv += ["--grad-accum", "2"]
    elif failure == "zero_batch":
        argv += ["--micro-batch", "0"]
    else:
        out.mkdir()
        (out / "ft_manifest.json").write_text('{"binding_pass": true}')
    before = {str(p.relative_to(tmp_path)): p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    monkeypatch.setattr(F, "load", lambda *args: pytest.fail("Invalid inputs reached model loading"))
    with pytest.raises(ValueError):
        F.main(argv)
    assert {str(p.relative_to(tmp_path)): p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()} == before


@pytest.mark.parametrize("failure", ["duplicate_universe", "existing_corpus", "empty_val"])
def test_materialization_failure_does_not_write_splits(tmp_path, failure):
    rows = [{"item_id": str(i), "prompt": "p", "response": "", "ground_truth": 0} for i in range(3)]
    if failure == "duplicate_universe":
        rows.append(rows[0])
    universe, dist, out = tmp_path / "u.jsonl", tmp_path / "d.json", tmp_path / "new"
    F.write_jsonl(universe, rows)
    dist.write_text(json.dumps({"splits": {"ft": ["0"], "val": [] if failure == "empty_val" else ["1"], "retention": ["2"]}}))
    if failure == "existing_corpus":
        out.mkdir()
        (out / "val.jsonl").write_bytes(b"old split\n")
    before = {str(p.relative_to(tmp_path)): p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    with pytest.raises(ValueError):
        F.materialize_ft_corpus(universe, dist, out)
    assert {str(p.relative_to(tmp_path)): p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()} == before


@pytest.fixture
def mock_training(monkeypatch):
    backwards = []

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(1.0))
            self.weight.register_hook(lambda grad: backwards.append(grad.clone()))
            self.nonfinite = False
            self.forwards = 0

        def forward(self, **kwargs):
            self.forwards += 1
            loss = self.weight.square()
            return SimpleNamespace(loss=loss * float("nan") if self.nonfinite else loss)

        def save_pretrained(self, path):
            (Path(path) / "weight.txt").write_text(str(self.weight.detach().item()))

    model, tok = Model(), FakeTok()
    tok.pad_token_id = 0
    monkeypatch.setattr(F, "load", lambda *args: (tok, model, 19193, 39257))
    monkeypatch.setattr(F, "template_hash_record", lambda tok: {"fixture": True})
    monkeypatch.setattr(torch, "autocast", lambda *args, **kwargs: contextlib.nullcontext())
    monkeypatch.setattr(F, "_readout", lambda *args: dict(
        val_delta=0.0, overfit_gap=0.0, degenerate=False, retention_flips=0,
    ))
    return model, backwards


def test_default_training_still_performs_thirty_updates(tmp_path, mock_training):
    directory = corpus(tmp_path, n=480)
    out = tmp_path / "run"
    assert F.main(["--seed", "42", "--corpus-dir", str(directory), "--out", str(out)]) == 0
    model, backwards = mock_training
    manifest = json.loads((out / "ft_manifest.json").read_text())
    assert manifest["optim"]["realized_steps"] == 30
    assert model.forwards == len(backwards) == 30
    assert manifest["binding_pass"] is True


def test_nonfinite_loss_stops_before_backward_or_saving(tmp_path, mock_training):
    directory = corpus(tmp_path)
    out = tmp_path / "run"
    model, backwards = mock_training
    model.nonfinite = True
    with pytest.raises(FloatingPointError, match="Non-finite"):
        F.main(["--seed", "42", "--corpus-dir", str(directory), "--out", str(out)])
    assert model.forwards == 1 and backwards == [] and model.weight.grad is None
    assert model.weight.item() == 1.0
    assert not (out / "adapter").exists() and not (out / "ft_manifest.json").exists()


def test_step_ceiling_before_incomplete_group_remains_supported(tmp_path, mock_training):
    directory = corpus(tmp_path, n=5)
    out = tmp_path / "run"
    assert F.main([
        "--seed", "42", "--corpus-dir", str(directory), "--out", str(out),
        "--micro-batch", "2", "--grad-accum", "2", "--step-ceiling", "1",
    ]) == 0
    assert mock_training[0].forwards == 2


def test_nonfinite_readout_score_is_rejected(monkeypatch):
    from rcv.extraction import lg3_extract

    monkeypatch.setattr(lg3_extract, "extract_row", lambda *args: (0, float("nan"), None))
    with pytest.raises(ValueError, match="Non-finite readout"):
        F._decisions(None, None, 1, 2, [{"item_id": "i0", "prompt": "p", "response": ""}])
