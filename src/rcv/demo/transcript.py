"""Render chain records as the demo transcript."""
from __future__ import annotations

from rcv.demo.records import Chain, Cycle

FAMILY_WIDTH = 14

CYCLE_RESULT = {"repaired": "repaired", "escalated": "escalated", "tolerated": "no alarm"}


def _chain_header(chain: Chain) -> str:
    return f"chain {chain.seed}   {chain.depth} accepted repairs · {chain.total_labels:,} labels"


def _cycle_line(cycle: Cycle) -> str:
    family = f"{cycle.family:<{FAMILY_WIDTH}}"
    if not cycle.attempts:
        return f"  {family} no alarm"

    decision = cycle.attempts[-1]
    audit = f"audit {decision.audit_labels}"
    if len(cycle.attempts) > 1:
        audit += f" ({len(cycle.attempts)} attempts)"
    return (f"  {family} alarm {cycle.alarm_boundary:.2f} · {audit}"
            f" · recall {decision.recall:.3f} · over-block {decision.over_block:.3f}"
            f" · {CYCLE_RESULT[cycle.outcome]}")


def render(chains: tuple[Chain, ...]) -> list[str]:
    """Render validated chains in input order as lines without newline terminators."""
    lines: list[str] = []
    for chain in chains:
        if lines:
            lines.append("")
        lines.append(_chain_header(chain))
        lines.extend(_cycle_line(cycle) for cycle in chain.cycles)
    return lines
