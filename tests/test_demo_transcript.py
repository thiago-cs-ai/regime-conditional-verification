from __future__ import annotations

import json
from importlib import resources

from rcv.demo.records import CHAIN_SEEDS, Attempt, Chain, Cycle, load_chains
from rcv.demo.transcript import render


def test_quiet_cycle_prints_no_alarm_without_numbers() -> None:
    cycle = Cycle(family="fraud", outcome="tolerated", alarm_boundary=None, attempts=())
    assert _line(cycle) == "  fraud          no alarm"


def test_single_attempt_omits_the_attempts_count() -> None:
    cycle = Cycle(family="cyber", outcome="repaired", alarm_boundary=0.5,
                  attempts=(Attempt(300, 0.7941, 0.0474, True),))
    line = _line(cycle)
    assert "audit 300 ·" in line
    assert "attempts" not in line


def test_multiple_attempts_report_the_count_and_the_last_reading() -> None:
    cycle = Cycle(family="violence", outcome="escalated", alarm_boundary=0.95,
                  attempts=(Attempt(300, 0.8641975308641975, 0.0821917808219178, False),
                            Attempt(1200, 0.8765432098765432, 0.0867579908675799, False)))
    line = _line(cycle)
    assert "audit 1200 (2 attempts)" in line
    assert "recall 0.877 · over-block 0.087 · escalated" in line


def _line(cycle: Cycle) -> str:
    chain = Chain(seed=1, terminal="escalated", total_labels=0, attack_share=0.0,
                  cycles=(cycle,))
    return render((chain,))[1]


def _blocks() -> dict[int, list[str]]:
    blocks: dict[int, list[str]] = {}
    seed = 0
    for line in render(load_chains()):
        if line.startswith("chain "):
            seed = int(line.split()[1])
            blocks[seed] = [line]
        elif line:
            blocks[seed].append(line)
    return blocks


def test_every_printed_number_is_derived_from_its_record() -> None:
    root = resources.files("rcv.demo") / "records"
    blocks = _blocks()
    assert list(blocks) == list(CHAIN_SEEDS)

    for seed in CHAIN_SEEDS:
        chain = json.loads((root / f"chain_{seed}" / "CHAIN.json").read_text(encoding="utf-8"))
        header, *cycle_lines = blocks[seed]

        depth = sum(1 for summary in chain["cycles"] if summary["outcome"] == "repaired")
        assert header == (
            f"chain {seed}   {depth} accepted repairs · {chain['total_labels_billed']:,} labels"
        )
        assert len(cycle_lines) == len(chain["cycles"])

        for index, line in enumerate(cycle_lines, start=1):
            record = json.loads(
                (root / f"chain_{seed}" / f"cycle_{index}.json").read_text(encoding="utf-8"))
            assert line.strip().startswith(record["family"])
            readings = [e for e in record["loop_events"] if e["kind"] == "acceptance_evaluated"]
            if not readings:
                assert line.endswith("no alarm")
                continue
            alarm = next(e for e in record["loop_events"] if e["kind"] == "alarm")
            assert f"alarm {alarm['boundary']:.2f}" in line
            deciding = readings[-1]
            assert f"audit {deciding['audit_labels']}" in line
            assert f"recall {deciding['gate_recall']:.3f}" in line
            assert f"over-block {deciding['gate_over_block']:.3f}" in line
            assert line.endswith("repaired" if deciding["gate_passed"] else "escalated")
            if len(readings) > 1:
                assert f"({len(readings)} attempts)" in line
