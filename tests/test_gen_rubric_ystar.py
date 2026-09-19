from __future__ import annotations

import json
from typing import Any

import pytest

from rcv.rebuild import gen_rubric_ystar as G
from rcv.rebuild.label_map import HARD_DROPPED_COLUMNS, export_label_map, gate_rows


def test_parse_verdict_unsafe():
    p = G.parse_verdict('{"verdict": "UNSAFE", "rule_fired": "2.2_commitment", "reason": "commits"}')
    assert p == {"verdict": "UNSAFE", "rule_fired": "2.2_commitment", "reason": "commits"}


def test_parse_verdict_safe_forces_rule_none():
    p = G.parse_verdict('{"verdict": "SAFE", "rule_fired": "2.1_content", "reason": "refuses"}')
    assert p is not None and p["verdict"] == "SAFE" and p["rule_fired"] == "none"


def test_parse_verdict_coerces_unknown_rule_to_none():
    p = G.parse_verdict('{"verdict": "UNSAFE", "rule_fired": "2.1_unknown", "reason": "x"}')
    assert p is not None and p["rule_fired"] == "none"


def test_parse_verdict_strips_code_fence():
    p = G.parse_verdict('```json\n{"verdict": "UNSAFE", "rule_fired": "none", "reason": "y"}\n```')
    assert p is not None and p["verdict"] == "UNSAFE"


def test_parse_verdict_returns_none_on_garbage():
    assert G.parse_verdict("not json at all") is None
    assert G.parse_verdict("") is None
    assert G.parse_verdict('{"verdict": "MAYBE"}') is None


def test_extract_raw_rule_preserves_granular_tag():
    assert G.extract_raw_rule('{"verdict": "UNSAFE", "rule_fired": "2.1_violence"}') == "2.1_violence"


def test_build_messages_shape():
    msgs = G.build_messages("PROMPT_X", "RESPONSE_Y")
    assert [m["role"] for m in msgs] == ["system", "user"]
    assert "PROMPT_X" in msgs[1]["content"] and "RESPONSE_Y" in msgs[1]["content"]
    assert "strict" in msgs[1]["content"].lower()


def test_load_api_key_missing_raises_without_path(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    with pytest.raises(G.YStarKeyError) as exc:
        G.load_api_key()
    msg = str(exc.value)
    assert "OPENROUTER_API_KEY" in msg
    assert "/" not in msg


def test_load_api_key_reads_env(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key-123")
    assert G.load_api_key() == "test-key-123"


def test_label_real_sends_no_temperature(monkeypatch):
    captured: dict[str, Any] = {}

    def fake_post(payload, api_key, timeout=90):
        captured.update(payload)
        return {"choices": [{"message": {"content": '{"verdict":"SAFE","rule_fired":"none","reason":"r"}'}}],
                "usage": {"prompt_tokens": 11, "completion_tokens": 5}}

    monkeypatch.setattr(G, "_post", fake_post)
    content, pin, pout = G.label_real("p", "r", "k", G.MODEL, 8192)
    assert "temperature" not in captured
    assert captured["model"] == G.MODEL and captured["max_tokens"] == 8192
    assert (pin, pout) == (11, 5)


def test_label_one_row_schema_and_polarity(monkeypatch):
    def fake_post(payload, api_key, timeout=90):
        return {"choices": [{"message": {"content": '{"verdict":"UNSAFE","rule_fired":"2.2_commitment","reason":"commits"}'}}],
                "usage": {"prompt_tokens": 100, "completion_tokens": 20}}

    monkeypatch.setattr(G, "_post", fake_post)
    row = G.label_one({"item_id": "PKU_test_0_r0", "prompt": "p", "response": "r"}, "k", G.MODEL, 8192)
    assert row["item_id"] == "PKU_test_0_r0"
    assert row["rubric_ystar"] == 1
    assert row["rule_fired"] == "2.2_commitment"
    assert set(row) == {"item_id", "rubric_ystar", "verdict", "rule_fired", "rule_fired_raw",
                        "reason", "prompt_tokens", "completion_tokens", "model", "prompt_version"}


def test_label_one_parse_error_gives_null_label(monkeypatch):
    monkeypatch.setattr(G, "_post", lambda p, k, timeout=90: {
        "choices": [{"message": {"content": "totally not json"}}], "usage": {}})
    row = G.label_one({"item_id": "PKU_test_1_r0", "prompt": "p", "response": "r"}, "k", G.MODEL, 8192)
    assert row["rubric_ystar"] is None and row["verdict"] == "PARSE_ERROR"


def test_emitted_row_passes_label_map_allowlist(monkeypatch):
    def fake_post(payload, api_key, timeout=90):
        return {"choices": [{"message": {"content": '{"verdict":"UNSAFE","rule_fired":"2.2_commitment","reason":"free text reason here"}'}}],
                "usage": {"prompt_tokens": 100, "completion_tokens": 20}}

    monkeypatch.setattr(G, "_post", fake_post)
    row = G.label_one({"item_id": "PKU_test_0_r0", "prompt": "p", "response": "r"}, "k", G.MODEL, 8192)
    assert "rule_fired_raw" in row and "rule_fired_raw" in HARD_DROPPED_COLUMNS
    exported = export_label_map([row], primary_label="rubric_ystar")
    assert len(exported) == 1
    ex = exported[0]
    assert set(ex) == {"item_id", "rubric_ystar", "rule_fired"}
    assert not (set(ex) & set(HARD_DROPPED_COLUMNS))
    report = gate_rows(exported, primary_label="rubric_ystar")
    assert report["gate_pass"] is True


def test_load_items_limit(tmp_path):
    p = tmp_path / "frame.jsonl"
    p.write_text("\n".join(json.dumps({"item_id": f"PKU_test_{i}_r0", "prompt": "p", "response": "r"})
                           for i in range(5)) + "\n")
    assert len(G.load_items(p, limit=1)) == 1
    assert len(G.load_items(p)) == 5


@pytest.mark.parametrize(
    ("second_reply", "expected_label", "expected_usage"),
    [
        ('{"verdict":"SAFE"}', 0, (300, 50)),
        ("still not JSON", None, (300, 50)),
        (RuntimeError("request failed"), None, (100, 20)),
    ],
)
def test_label_one_counts_usage_across_parse_retry(
    monkeypatch, second_reply, expected_label, expected_usage
):
    replies = iter([("not JSON", 100, 20), (second_reply, 200, 30)])

    def request(*args):
        raw, pin, pout = next(replies)
        if isinstance(raw, Exception):
            raise raw
        return raw, pin, pout

    monkeypatch.setattr(G, "label_real", request)
    row = G.label_one(
        {"item_id": "PKU_test_0_r0", "prompt": "p", "response": "r"},
        "fake-key", G.MODEL, 8192,
    )
    assert row["rubric_ystar"] == expected_label
    assert (row["prompt_tokens"], row["completion_tokens"]) == expected_usage
    if isinstance(second_reply, Exception):
        assert row["reason"] == "Labeling failed: request failed"


@pytest.mark.parametrize(
    ("failure", "successful_items", "expected_status"),
    [(None, 2, 0), ("parse", 1, 1), ("parse", 0, 1), ("request", 0, 1)],
)
def test_cli_exit_status_reflects_label_failures(
    tmp_path, monkeypatch, failure, successful_items, expected_status
):
    source = tmp_path / "input.jsonl"
    output = tmp_path / "labels.jsonl"
    source.write_text("".join(
        json.dumps({"item_id": f"PKU_test_{i}_r0", "prompt": f"prompt-{i}", "response": "r"}) + "\n"
        for i in range(2)
    ))
    monkeypatch.setenv("OPENROUTER_API_KEY", "fake-key")

    def request(payload, api_key, timeout=90):
        succeeds = any(
            f"PROMPT:\nprompt-{i}\n" in payload["messages"][1]["content"]
            for i in range(successful_items)
        )
        if not succeeds and failure == "request":
            raise RuntimeError("request failed")
        return {
            "choices": [{"message": {"content": '{"verdict":"SAFE"}' if succeeds else "not JSON"}}],
            "usage": {"prompt_tokens": 100, "completion_tokens": 20},
        }

    monkeypatch.setattr(G, "_post", request)
    status = G.main(["--input", str(source), "--out", str(output), "--concurrency", "1"])
    rows = [json.loads(line) for line in output.read_text().splitlines()]
    assert status == expected_status
    assert {r["item_id"] for r in rows} == {"PKU_test_0_r0", "PKU_test_1_r0"}
    assert sum(r["rubric_ystar"] is not None for r in rows) == successful_items
    assert G.load_done(output) == {f"PKU_test_{i}_r0" for i in range(successful_items)}


@pytest.mark.parametrize("rule", [
    "2.1_csam", "2.1_violence", "2.1_glorification", "2.1_crime", "2.1_hate",
    "2.1_sexual", "2.1_selfharm", "2.1_misinfo", "2.1_deception", "2.1_privacy",
    "2.2_commitment", "none", "2.1_content",
])
def test_label_one_preserves_rubric_and_legacy_tags_without_changing_verdict(monkeypatch, rule):
    raw = json.dumps({"verdict": "UNSAFE", "rule_fired": rule})
    monkeypatch.setattr(G, "label_real", lambda *args: (raw, 10, 20))
    row = G.label_one({"item_id": "example", "prompt": "p", "response": ""}, "k", G.MODEL, 8192)
    assert row["rubric_ystar"] == 1 and row["verdict"] == "UNSAFE"
    assert row["rule_fired"] == row["rule_fired_raw"] == rule
    assert (row["prompt_tokens"], row["completion_tokens"]) == (10, 20)
    assert G.parse_verdict(json.dumps({"verdict": "SAFE", "rule_fired": rule}))["rule_fired"] == "none"


@pytest.mark.parametrize(("n", "limit", "status", "pending"), [(9, None, 2, 1), (8, None, 0, 0), (9, 8, 0, 0)])
def test_budget_stop_and_resume_preserve_completed_records(tmp_path, monkeypatch, capsys, n, limit, status, pending):
    source, output = tmp_path / "items.jsonl", tmp_path / "labels.jsonl"
    source.write_text("".join(json.dumps({"item_id": str(i), "prompt": f"p{i}", "response": ""}) + "\n" for i in range(n)))
    calls = []

    def request(prompt, response, *args):
        calls.append((prompt, response))
        return '{"verdict":"SAFE"}', 100, 20

    monkeypatch.setattr(G, "label_real", request)
    monkeypatch.setattr(G, "load_api_key", lambda: "fake-key")
    args = ["--input", str(source), "--out", str(output), "--concurrency", "1"]
    if limit is not None:
        args += ["--limit", str(limit)]
    assert G.main([*args, "--max-cost", "0.000001"]) == status
    assert f"pending={pending}" in capsys.readouterr().out
    before = output.read_bytes()
    saved = [json.loads(line) for line in before.splitlines()]
    assert len(saved) == 8
    assert {r["item_id"] for r in saved} == {str(i) for i in range(8)}
    assert all(r["rubric_ystar"] == 0 and r["prompt_tokens"] == 100 and r["completion_tokens"] == 20 for r in saved)
    calls.clear()
    assert G.main(args) == 0
    assert calls == ([("p8", "")] if pending else [])
    assert output.read_bytes().startswith(before)


@pytest.mark.parametrize("rows", [
    [], [None], [{}],
    [{"item_id": "", "prompt": "p", "response": "r"}],
    [{"item_id": 1, "prompt": "p", "response": "r"}],
    [{"item_id": "i", "prompt": None, "response": "r"}],
    [{"item_id": "i", "prompt": "p", "response": None}],
    [{"item_id": "i", "prompt": "p", "response": "r"}, {"item_id": "i", "prompt": "different", "response": "r"}],
])
def test_invalid_input_rejected_before_requests_or_output(tmp_path, monkeypatch, rows):
    source, output = tmp_path / "items.jsonl", tmp_path / "labels.jsonl"
    source.write_text("".join(json.dumps(row) + "\n" for row in rows))
    monkeypatch.setattr(G, "load_api_key", lambda: "fake-key")
    monkeypatch.setattr(G, "_post", lambda *args: pytest.fail("Unexpected API request"))
    with pytest.raises(ValueError):
        G.main(["--input", str(source), "--out", str(output)])
    assert not output.exists()


@pytest.mark.parametrize(("flag", "value"), [
    ("--concurrency", "0"), ("--concurrency", "-1"), ("--progress-every", "0"),
    ("--max-tokens", "0"), ("--limit", "0"), ("--limit", "-1"),
    ("--max-cost", "0"), ("--max-cost", "-1"), ("--max-cost", "nan"), ("--max-cost", "inf"),
])
def test_invalid_numeric_arguments_rejected_before_input_or_key_access(monkeypatch, flag, value):
    monkeypatch.setattr(G, "load_api_key", lambda: pytest.fail("Key accessed before argument validation"))
    with pytest.raises(SystemExit) as exc:
        G.main(["--input", "absent.jsonl", "--out", "unused.jsonl", flag, value])
    assert exc.value.code == 2


@pytest.mark.parametrize("metadata", [
    {"model": "other", "prompt_version": G.PROMPT_VERSION},
    {"model": G.MODEL, "prompt_version": "other"},
    {"model": G.MODEL}, {"prompt_version": G.PROMPT_VERSION}, {},
])
def test_incompatible_resume_rejected_without_requests_or_mutation(tmp_path, monkeypatch, metadata):
    source, output = tmp_path / "items.jsonl", tmp_path / "labels.jsonl"
    source.write_text(json.dumps({"item_id": "i", "prompt": "p", "response": ""}) + "\n")
    before = json.dumps({"item_id": "i", "rubric_ystar": 1, **metadata}).encode()
    output.write_bytes(before)
    monkeypatch.setattr(G, "load_api_key", lambda: "fake-key")
    monkeypatch.setattr(G, "_post", lambda *args: pytest.fail("Unexpected API request"))
    with pytest.raises(ValueError, match="model|prompt_version"):
        G.main(["--input", str(source), "--out", str(output)])
    assert output.read_bytes() == before


def test_resume_retries_null_labels_with_matching_custom_model(tmp_path, monkeypatch):
    source, output = tmp_path / "items.jsonl", tmp_path / "labels.jsonl"
    source.write_text(json.dumps({"item_id": "i", "prompt": "p", "response": ""}) + "\n")
    output.write_text(json.dumps({"item_id": "i", "rubric_ystar": None}) + "\n")
    monkeypatch.setattr(G, "load_api_key", lambda: "fake-key")
    calls = []

    def request(prompt, response, key, model, tokens):
        calls.append((response, model))
        return '{"verdict":"SAFE"}', 1, 2

    monkeypatch.setattr(G, "label_real", request)
    args = ["--input", str(source), "--out", str(output), "--model", "custom"]
    assert G.main(args) == 0
    assert G.main(args) == 0
    assert calls == [("", "custom")]


@pytest.mark.parametrize(("reply", "label"), [("SAFE", 0), ("UNSAFE", 1), ("invalid", None)])
def test_custom_policy_request_and_record_identity(tmp_path, monkeypatch, reply, label):
    import hashlib

    policy_bytes = "Privé: reject disclosure.\r\nPermit refusals.\r\n".encode()
    policy_path = tmp_path / "policy.txt"
    policy_path.write_bytes(policy_bytes)
    source, output = tmp_path / "items.jsonl", tmp_path / "labels.jsonl"
    item = {"item_id": "customer-1", "prompt": "contact?", "response": ""}
    source.write_text(json.dumps(item) + "\n")
    requests = []

    def post(payload, key):
        requests.append(payload)
        return {"choices": [{"message": {"content": json.dumps({"verdict": reply})}}],
                "usage": {"prompt_tokens": 2, "completion_tokens": 1}}

    monkeypatch.setattr(G, "_post", post)
    monkeypatch.setattr(G, "load_api_key", lambda: "test")
    args = ["--input", str(source), "--out", str(output), "--policy-file", str(policy_path)]
    assert G.main(args) == (1 if label is None else 0)
    row = json.loads(output.read_text())
    assert row["rubric_ystar"] == label
    assert row["prompt_version"] == "custom-v1"
    assert row["policy_sha256"] == hashlib.sha256(policy_bytes).hexdigest()
    encoded = json.dumps({"prompt": "contact?", "response": ""}, ensure_ascii=False,
                         sort_keys=True, separators=(",", ":")).encode()
    assert row["input_sha256"] == hashlib.sha256(encoded).hexdigest()
    assert row["rule_fired"] == "none"
    assert policy_bytes.decode() in requests[0]["messages"][0]["content"]
    assert requests[0]["messages"] != G.build_messages(item["prompt"], item["response"])
    assert "two UNSAFE tests" not in requests[0]["messages"][1]["content"]
    count = len(requests)
    G.main(args)
    assert len(requests) == (count * 2 if label is None else count)


@pytest.mark.parametrize("label", [None, 1])
@pytest.mark.parametrize("field", ["model", "prompt_version", "policy_sha256", "input_sha256"])
def test_custom_resume_rejects_changed_identity_even_on_failed_records(
    tmp_path, monkeypatch, label, field
):
    policy = tmp_path / "policy.txt"
    policy.write_text("Policy A")
    source, output = tmp_path / "items.jsonl", tmp_path / "labels.jsonl"
    item = {"item_id": "item", "prompt": "p", "response": "r"}
    source.write_text(json.dumps(item) + "\n")
    row = {"item_id": "item", "rubric_ystar": label,
           **G.record_metadata(item, G.MODEL, "Policy A"), field: "different"}
    before = json.dumps(row).encode()
    output.write_bytes(before)
    monkeypatch.setattr(G, "load_api_key", lambda: pytest.fail("Unexpected key access"))
    with pytest.raises(ValueError, match=field):
        G.main(["--input", str(source), "--out", str(output), "--policy-file", str(policy)])
    assert output.read_bytes() == before


def test_custom_resume_accepts_output_covering_more_than_the_input_prefix(tmp_path):
    items = [{"item_id": str(i), "prompt": str(i), "response": ""} for i in range(2)]
    output = tmp_path / "labels.jsonl"
    output.write_text("".join(json.dumps({"item_id": item["item_id"], "rubric_ystar": 0,
                                          **G.record_metadata(item, G.MODEL, "policy")}) + "\n"
                              for item in items))
    assert G.load_done(output, policy="policy", items=items[:1]) == {"0", "1"}
    with pytest.raises(ValueError, match="prompt_version"):
        G.load_done(output)


def test_default_policy_cannot_adopt_failed_custom_policy_records(tmp_path):
    item = {"item_id": "i", "prompt": "p", "response": ""}
    output = tmp_path / "labels.jsonl"
    output.write_text(json.dumps({"item_id": "i", "rubric_ystar": None,
                                  **G.record_metadata(item, G.MODEL, "custom policy")}))
    with pytest.raises(ValueError, match="prompt_version"):
        G.load_done(output)


@pytest.mark.parametrize("content", [b" \n", b"\xff", None])
def test_invalid_policy_rejected_before_requests(tmp_path, monkeypatch, content):
    policy = tmp_path / "policy.txt"
    if content is not None:
        policy.write_bytes(content)
    monkeypatch.setattr(G, "load_api_key", lambda: pytest.fail("Unexpected key access"))
    with pytest.raises(SystemExit) as exc:
        G.main(["--input", "absent", "--out", "unused", "--policy-file", str(policy)])
    assert exc.value.code == 2


def test_default_request_messages_remain_frozen(monkeypatch):
    requests = []

    def post(payload, key):
        requests.append(payload)
        return {"choices": [{"message": {"content": '{"verdict":"SAFE"}'}}]}

    monkeypatch.setattr(G, "_post", post)
    G.label_real("p", "r", "k", G.MODEL, 8192)
    assert requests[0]["messages"] == G.build_messages("p", "r")
