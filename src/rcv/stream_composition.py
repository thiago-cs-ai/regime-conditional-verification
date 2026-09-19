"""Compose scheduled streams from base and drift item pools."""

from typing import NamedTuple

import numpy as np


class ComposedStream(NamedTuple):
    """A sampled stream in serving order."""
    item_id: np.ndarray
    is_drift_item: np.ndarray
    stream_lambda: np.ndarray


def linear_contamination_ramp(length, onset, maximum_rate):
    """Return a zero-prefix linear ramp ending at maximum_rate."""
    if not 0.0 <= maximum_rate < 1.0:
        raise ValueError(f"maximum_rate must be in [0, 1); got {maximum_rate!r}.")
    if not 0 < onset < length - 1:
        raise ValueError(f"onset must leave a prefix and a two-position ramp; got onset={onset}, "
                         f"length={length}.")
    schedule = np.zeros(length, dtype=np.float64)
    positions = np.arange(onset, length, dtype=np.float64)
    schedule[onset:] = maximum_rate * (positions - onset) / (length - 1 - onset)
    return schedule


def compose_stream(contamination_schedule, base_pool, drift_pool, seed):
    """Sample with replacement, choosing a drift item at each scheduled rate."""
    schedule = _as_schedule(contamination_schedule)
    base_pool = _as_pool(base_pool, "Base pool")
    drift_pool = _as_pool(drift_pool, "Drift pool")
    shared = np.intersect1d(base_pool, drift_pool)
    if len(shared):
        raise ValueError(f"Base and drift pools share {len(shared)} item IDs; first "
                         f"{shared[0]!r}. Item provenance would be ambiguous.")

    draw = np.random.default_rng(seed)
    coin = draw.random(len(schedule))
    base_pick = draw.integers(0, len(base_pool), len(schedule))
    drift_pick = draw.integers(0, len(drift_pool), len(schedule))

    is_drift_item = coin < schedule
    return ComposedStream(
        item_id=np.where(is_drift_item, drift_pool[drift_pick], base_pool[base_pick]),
        is_drift_item=is_drift_item,
        stream_lambda=schedule)


def _as_schedule(contamination_schedule):
    schedule = np.asarray(contamination_schedule, dtype=np.float64)
    if schedule.ndim != 1 or len(schedule) == 0:
        raise ValueError(f"Schedule must be a non-empty one-dimensional array; got shape "
                         f"{schedule.shape}.")
    if not np.all((schedule >= 0.0) & (schedule < 1.0)):
        raise ValueError("Schedule rates must be in [0, 1).")
    return schedule


def _as_pool(pool, what):
    pool = np.asarray(pool)
    if len(pool) == 0:
        raise ValueError(f"{what} must contain at least one item.")
    if len(np.unique(pool)) != len(pool):
        raise ValueError(f"{what} contains duplicate item IDs; sampling would weight them "
                         f"unevenly.")
    return pool
