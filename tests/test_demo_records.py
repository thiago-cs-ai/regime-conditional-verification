from __future__ import annotations

import hashlib
import json
import re
from importlib import resources

import pytest

from rcv.demo import records as R

PUBLISHED = {
    42: {"terminal": "tolerated", "labels": 2700, "depth": 4, "share": 0.544},
    789: {"terminal": "tolerated", "labels": 1800, "depth": 3, "share": 0.425},
    1024: {"terminal": "escalated", "labels": 3300, "depth": 3, "share": 0.383},
    456: {"terminal": "escalated", "labels": 3000, "depth": 2, "share": 0.316},
    123: {"terminal": "escalated", "labels": 1500, "depth": 0, "share": 0.000},
}


def _records_root():
    return resources.files("rcv.demo") / "records"


def _shipped_files():
    for seed in R.CHAIN_SEEDS:
        directory = _records_root() / f"chain_{seed}"
        for entry in sorted(directory.iterdir(), key=lambda item: item.name):
            if entry.name.endswith(".json"):
                yield entry


@pytest.mark.parametrize("seed", list(PUBLISHED))
def test_chain_matches_published_table(seed: int) -> None:
    chain = R.load_chain(seed)
    expected = PUBLISHED[seed]
    assert chain.terminal == expected["terminal"]
    assert chain.total_labels == expected["labels"]
    assert chain.depth == expected["depth"]
    assert round(chain.attack_share, 3) == expected["share"]


def test_chains_load_in_published_table_order() -> None:
    assert [chain.seed for chain in R.load_chains()] == list(R.CHAIN_SEEDS)


def test_every_shipped_file_matches_its_manifest_digest() -> None:
    manifest = (_records_root() / "MANIFEST.md").read_text(encoding="utf-8")
    rows = dict(re.findall(r"\| `(chain_\d+/\w+\.json)` \| `([0-9a-f]{64})` \|", manifest))
    assert len(rows) == 22, "the manifest must cover every shipped record"

    for entry in _shipped_files():
        relative = f"{entry.parent.name}/{entry.name}"
        digest = hashlib.sha256(entry.read_bytes()).hexdigest()
        assert digest == rows[relative], f"{relative} no longer matches its recorded digest"


def test_shipped_records_carry_only_allowlisted_keys() -> None:
    for entry in _shipped_files():
        record = json.loads(entry.read_text(encoding="utf-8"))
        if entry.name == "CHAIN.json":
            assert set(record) <= R.CHAIN_KEYS
            continue
        assert set(record) <= R.CYCLE_KEYS
        for event in record["loop_events"]:
            assert set(event) <= R.EVENT_KEYS
            assert event["kind"] in R.EVENT_KINDS


def test_the_decision_and_its_numbers_come_from_one_event() -> None:
    for chain in R.load_chains():
        for cycle in chain.cycles:
            if cycle.outcome == "repaired":
                assert cycle.attempts[-1].passed is True
            elif cycle.outcome == "escalated":
                assert cycle.attempts[-1].passed is False
            else:
                assert cycle.attempts == ()


def test_unknown_key_is_rejected() -> None:
    with pytest.raises(ValueError, match="unexpected keys"):
        R._check_keys({"chain_seed": 1, "surprise": 2}, R.CHAIN_KEYS, "chain 1")


def test_missing_field_is_rejected() -> None:
    with pytest.raises(ValueError, match="missing required field 'terminal'"):
        R._require({}, "terminal", "chain 1")


def test_unknown_cycle_outcome_is_rejected() -> None:
    with pytest.raises(ValueError, match="unknown cycle outcome"):
        R._parse_cycle({"family": "cyber", "outcome": "invented", "loop_events": []}, "cycle")


def _gate_event(**over):
    event = {"kind": "acceptance_evaluated", "audit_labels": 300, "gate_recall": 0.8,
             "gate_over_block": 0.02, "gate_passed": True}
    event.update(over)
    return event


_ALARM = {"kind": "alarm", "boundary": 0.5, "monitored_position": 1000, "regime": 0}


def _cycle_record(outcome="repaired", events=None, with_alarm=True):
    readings = [_gate_event()] if events is None else events
    return {"family": "cyber", "outcome": outcome,
            "loop_events": ([_ALARM] if with_alarm else []) + readings}


def _chain_record(**over):
    record = {"chain_seed": 9, "cycles_run": 1, "terminal": "escalated",
              "total_labels_billed": 600, "final_baseline": {"base": 0.5},
              "cycles": [{"family": "cyber", "outcome": "escalated", "labels_billed": 600}],
              "permutation": [], "provenance": {}}
    record.update(over)
    return record


def _write_chain(tmp_path, chain, cycles):
    directory = tmp_path / f"chain_{chain['chain_seed']}"
    directory.mkdir()
    (directory / "CHAIN.json").write_text(json.dumps(chain), encoding="utf-8")
    for index, cycle in enumerate(cycles, start=1):
        (directory / f"cycle_{index}.json").write_text(json.dumps(cycle), encoding="utf-8")


def test_repaired_cycle_whose_deciding_reading_failed_is_rejected() -> None:
    with pytest.raises(ValueError, match="disagrees with its deciding gate reading"):
        R._parse_cycle(_cycle_record("repaired", [_gate_event(gate_passed=False)]), "cycle")


def test_escalated_cycle_whose_deciding_reading_passed_is_rejected() -> None:
    events = [_gate_event(gate_passed=False), _gate_event(gate_passed=True)]
    with pytest.raises(ValueError, match="disagrees with its deciding gate reading"):
        R._parse_cycle(_cycle_record("escalated", events), "cycle")


def test_quiet_cycle_carrying_gate_readings_is_rejected() -> None:
    with pytest.raises(ValueError, match=r"outcome 'tolerated' has gate readings \(1\)"):
        R._parse_cycle(_cycle_record("tolerated", [_gate_event()]), "cycle")


def test_tolerated_cycle_with_an_alarm_but_no_gate_readings_is_rejected() -> None:
    with pytest.raises(ValueError, match="cycle: outcome 'tolerated' has an alarm event"):
        R._parse_cycle(_cycle_record("tolerated", events=[]), "cycle")


def test_decided_cycle_without_any_gate_reading_is_rejected() -> None:
    with pytest.raises(ValueError, match="outcome 'repaired' has no gate reading"):
        R._parse_cycle(_cycle_record("repaired", []), "cycle")


def _summary(family="cyber", outcome="repaired", labels=600):
    return {"family": family, "outcome": outcome, "labels_billed": labels}


def _cycle(family="cyber", outcome="repaired"):
    return R.Cycle(family=family, outcome=outcome, alarm_boundary=0.5,
                   attempts=(R.Attempt(300, 0.8, 0.02, outcome == "repaired"),))


def test_a_dropped_cycle_file_is_rejected() -> None:
    chain_record = {"cycles": [_summary(), _summary("fraud", "tolerated", 0)]}
    with pytest.raises(ValueError, match="cycles recorded in the chain but"):
        R._check_cycles_agree_with_the_chain(chain_record, (_cycle(),), 600, "chain 1")


def test_a_cycle_file_disagreeing_with_the_chain_is_rejected() -> None:
    chain_record = {"cycles": [_summary(family="fraud")]}
    with pytest.raises(ValueError, match="family or outcome differs from the chain summary"):
        R._check_cycles_agree_with_the_chain(chain_record, (_cycle(),), 600, "chain 1")


def test_label_totals_must_add_up() -> None:
    chain_record = {"cycles": [_summary(outcome="escalated", labels=600)], "terminal": "escalated"}
    with pytest.raises(ValueError, match="cycle labels sum to 600, chain totals 2700"):
        R._check_cycles_agree_with_the_chain(
            chain_record, (_cycle(outcome="escalated"),), 2700, "chain 1")


def test_gate_readings_without_an_alarm_event_are_rejected() -> None:
    with pytest.raises(ValueError, match="no alarm event"):
        R._parse_cycle(_cycle_record(with_alarm=False), "cycle")


def _failed_cycle():
    return _cycle_record("escalated", [_gate_event(gate_passed=False)])


def test_load_chain_accepts_a_consistent_chain(tmp_path, monkeypatch) -> None:
    _write_chain(tmp_path, _chain_record(), [_failed_cycle()])
    monkeypatch.setattr(R, "_records_root", lambda: tmp_path)
    loaded = R.load_chain(9)
    assert loaded.terminal == "escalated"
    assert loaded.depth == 0


def test_load_chain_rejects_a_chain_with_a_missing_cycle_file(tmp_path, monkeypatch) -> None:
    chain = _chain_record(cycles=[
        {"family": "cyber", "outcome": "escalated", "labels_billed": 600},
        {"family": "fraud", "outcome": "escalated", "labels_billed": 0},
    ])
    _write_chain(tmp_path, chain, [_failed_cycle()])
    monkeypatch.setattr(R, "_records_root", lambda: tmp_path)
    with pytest.raises(ValueError, match="cycles recorded in the chain but"):
        R.load_chain(9)


def test_load_chain_rejects_a_terminal_that_contradicts_the_last_cycle(tmp_path, monkeypatch) -> None:
    _write_chain(tmp_path, _chain_record(terminal="tolerated"), [_failed_cycle()])
    monkeypatch.setattr(R, "_records_root", lambda: tmp_path)
    with pytest.raises(ValueError, match="terminal 'tolerated' but the last cycle ended"):
        R.load_chain(9)
