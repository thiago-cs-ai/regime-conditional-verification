"""Load bundled chain records and check fields used by the demo."""
from __future__ import annotations

import json
from dataclasses import dataclass
from importlib import resources
from typing import Any

# Published Table T7 order.
CHAIN_SEEDS: tuple[int, ...] = (42, 789, 1024, 456, 123)

CHAIN_KEYS = frozenset({
    "chain_seed", "cycles", "cycles_run", "final_baseline", "permutation", "provenance",
    "terminal", "total_labels_billed",
})
CYCLE_KEYS = frozenset({
    "absorption", "acceptance_baseline", "alarms", "baseline_in", "chain_seed", "change_log",
    "cycle", "episode_seed", "family", "labels_billed", "loop_events", "monitor",
    "monitored_offset", "outcome", "pool_positions", "post_alarm_blocks", "probe_selection",
    "r_locked", "recipe", "reference_in", "refusal",
})
EVENT_KEYS = frozenset({
    "audit_labels", "boundary", "detail", "gate_over_block", "gate_passed", "gate_recall", "kind",
    "monitored_position", "regime", "stream_position",
})
EVENT_KINDS = frozenset({
    "acceptance_evaluated", "alarm", "audit", "escalation_demanded", "gate_block_composed",
    "repair_accepted", "watch_ended",
})

CYCLE_OUTCOMES = frozenset({"repaired", "escalated", "tolerated"})
TERMINALS = frozenset({"tolerated", "escalated"})


@dataclass(frozen=True)
class Attempt:
    """Metrics and decision from one acceptance_evaluated event."""

    audit_labels: int
    recall: float
    over_block: float
    passed: bool


@dataclass(frozen=True)
class Cycle:
    family: str
    outcome: str
    alarm_boundary: float | None
    attempts: tuple[Attempt, ...]


@dataclass(frozen=True)
class Chain:
    seed: int
    terminal: str
    total_labels: int
    attack_share: float
    cycles: tuple[Cycle, ...]

    @property
    def depth(self) -> int:
        """Accepted repairs, the quantity Table T7 calls depth."""
        return sum(1 for cycle in self.cycles if cycle.outcome == "repaired")


def _records_root():
    return resources.files(__package__) / "records"


def _read(path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _require(record: dict[str, Any], field: str, where: str) -> Any:
    if field not in record:
        raise ValueError(f"{where}: missing required field {field!r}.")
    return record[field]


def _check_keys(record: dict[str, Any], allowed: frozenset[str], where: str) -> None:
    unknown = sorted(set(record) - allowed)
    if unknown:
        raise ValueError(f"{where}: unexpected keys {unknown}.")


def _parse_cycle(record: dict[str, Any], where: str) -> Cycle:
    _check_keys(record, CYCLE_KEYS, where)
    outcome = _require(record, "outcome", where)
    if outcome not in CYCLE_OUTCOMES:
        raise ValueError(f"{where}: unknown cycle outcome {outcome!r}")

    boundary: float | None = None
    attempts: list[Attempt] = []
    for event in _require(record, "loop_events", where):
        _check_keys(event, EVENT_KEYS, where)
        kind = _require(event, "kind", where)
        if kind not in EVENT_KINDS:
            raise ValueError(f"{where}: unknown event kind {kind!r}")
        if kind == "alarm" and boundary is None:
            boundary = float(_require(event, "boundary", where))
        elif kind == "acceptance_evaluated":
            attempts.append(Attempt(
                audit_labels=int(_require(event, "audit_labels", where)),
                recall=float(_require(event, "gate_recall", where)),
                over_block=float(_require(event, "gate_over_block", where)),
                passed=bool(_require(event, "gate_passed", where)),
            ))

    if attempts and boundary is None:
        raise ValueError(f"{where}: gate readings have no alarm event.")
    _check_outcome_is_the_deciding_attempt(outcome, attempts, where)
    if outcome == "tolerated" and boundary is not None:
        raise ValueError(f"{where}: outcome 'tolerated' has an alarm event.")
    return Cycle(
        family=_require(record, "family", where),
        outcome=outcome,
        alarm_boundary=boundary,
        attempts=tuple(attempts),
    )


def _check_outcome_is_the_deciding_attempt(
    outcome: str, attempts: list[Attempt], where: str
) -> None:
    """Require no gate readings for 'tolerated', otherwise a matching final decision."""
    if outcome == "tolerated":
        if attempts:
            raise ValueError(f"{where}: outcome 'tolerated' has gate readings ({len(attempts)}).")
        return
    if not attempts:
        raise ValueError(f"{where}: outcome {outcome!r} has no gate reading.")
    decided = attempts[-1].passed
    if decided is not (outcome == "repaired"):
        raise ValueError(
            f"{where}: outcome {outcome!r} disagrees with its deciding gate reading "
            f"(gate_passed={decided})."
        )


def load_chain(seed: int) -> Chain:
    directory = _records_root() / f"chain_{seed}"
    chain_record = _read(directory / "CHAIN.json")
    where = f"chain {seed}"
    _check_keys(chain_record, CHAIN_KEYS, where)

    terminal = _require(chain_record, "terminal", where)
    if terminal not in TERMINALS:
        raise ValueError(f"{where}: unknown terminal {terminal!r}")

    cycles_run = int(_require(chain_record, "cycles_run", where))
    cycles = tuple(
        _parse_cycle(_read(directory / f"cycle_{index}.json"), f"{where} cycle {index}")
        for index in range(1, cycles_run + 1)
    )
    total_labels = int(_require(chain_record, "total_labels_billed", where))
    _check_cycles_agree_with_the_chain(chain_record, cycles, total_labels, where)

    base_share = float(_require(_require(chain_record, "final_baseline", where), "base", where))
    return Chain(
        seed=int(_require(chain_record, "chain_seed", where)),
        terminal=terminal,
        total_labels=total_labels,
        attack_share=1.0 - base_share,
        cycles=cycles,
    )


def _check_cycles_agree_with_the_chain(
    chain_record: dict[str, Any], cycles: tuple[Cycle, ...], total_labels: int, where: str
) -> None:
    """Check cycle count, families, outcomes, terminal, and the sum of summary label counts."""
    summaries = _require(chain_record, "cycles", where)
    if len(summaries) != len(cycles):
        raise ValueError(
            f"{where}: {len(summaries)} cycles recorded in the chain but {len(cycles)} read"
        )
    for index, (summary, cycle) in enumerate(zip(summaries, cycles, strict=True), start=1):
        if summary.get("family") != cycle.family or summary.get("outcome") != cycle.outcome:
            raise ValueError(f"{where} cycle {index}: family or outcome differs from the chain "
                             "summary.")

    terminal = _require(chain_record, "terminal", where)
    if cycles and terminal != cycles[-1].outcome:
        raise ValueError(
            f"{where}: terminal {terminal!r} but the last cycle ended {cycles[-1].outcome!r}"
        )

    billed = sum(int(_require(summary, "labels_billed", where)) for summary in summaries)
    if billed != total_labels:
        raise ValueError(f"{where}: cycle labels sum to {billed}, chain totals {total_labels}")


def load_chains(seeds: tuple[int, ...] = CHAIN_SEEDS) -> tuple[Chain, ...]:
    return tuple(load_chain(seed) for seed in seeds)
