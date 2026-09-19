"""Compose consecutive drift campaigns against a shared deployment."""

import hashlib
import math
from typing import NamedTuple

import numpy as np

from rcv.frames import concatenated, rows
from rcv.stream_composition import linear_contamination_ramp

BASE_COMPONENT = "base"
SERVING_POOL = "serving"
CHAIN_DRAW_TAG = 20260730

ITEM_COLUMNS = ("item_id", "representation", "verdict", "oracle", "human", "family")
POSITION_COLUMNS = ("is_drift_item", "is_stream", "stream_lambda", "pool")
SHARE_TOLERANCE = 1e-12


class CycleRecipe(NamedTuple):
    """Pool assignments, row indices, and schedule for one composed cycle."""
    pool: np.ndarray
    item_index: np.ndarray
    stream_lambda: np.ndarray
    attack_family: str


def initial_baseline():
    return {BASE_COMPONENT: 1.0}


def absorbed(baseline, family, rate):
    """Mix an attacked family into a baseline distribution."""
    assert_is_a_mixture(baseline)
    if not (isinstance(rate, (int, float)) and math.isfinite(rate) and 0.0 <= rate <= 1.0):
        raise ValueError(f"Absorption rate must be in [0, 1]; got {rate!r}.")
    if family == BASE_COMPONENT:
        raise ValueError(f"Cannot absorb {BASE_COMPONENT!r} into itself.")
    if family in baseline:
        raise ValueError(f"Baseline already contains family {family!r}.")
    thinned = {name: share * (1.0 - rate) for name, share in baseline.items()}
    return {**thinned, family: float(rate)}


def assert_is_a_mixture(baseline, what="the baseline"):
    """Validate a finite probability distribution over components."""
    if not baseline:
        raise ValueError(f"{what} must contain at least one component.")
    for name, share in baseline.items():
        if not (isinstance(share, (int, float)) and math.isfinite(share) and 0.0 <= share <= 1.0):
            raise ValueError(f"{what} gives {name!r} invalid share {share!r}.")
    total = math.fsum(baseline.values())
    if abs(total - 1.0) > SHARE_TOLERANCE:
        raise ValueError(f"{what} must sum to 1; got {total!r}.")


def replayed_baselines(absorptions):
    """Reconstruct baseline distributions after each absorption."""
    baseline = initial_baseline()
    replayed = [baseline]
    for family, rate in absorptions:
        baseline = absorbed(baseline, family, rate)
        replayed.append(baseline)
    return replayed


def assert_recursion_reproduces(recorded):
    """Validate recorded baselines against the absorption recursion."""
    replayed = initial_baseline()
    for index, cycle in enumerate(recorded, start=1):
        assert_mixtures_agree(cycle["baseline_in"], replayed, f"cycle {index}")
        absorption = cycle["absorption"]
        if absorption is not None:
            replayed = absorbed(replayed, *absorption)


def assert_mixtures_agree(recorded, replayed, what):
    if list(recorded) != list(replayed):
        raise ValueError(f"{what}: recorded components {list(recorded)} differ from "
                         f"replayed components {list(replayed)}.")
    departed = {name: (recorded[name], replayed[name]) for name in replayed
                if abs(recorded[name] - replayed[name]) > SHARE_TOLERANCE}
    if departed:
        raise ValueError(f"{what}: recorded and replayed shares differ: {departed}.")


def deduplicated_pool(block, what="the pool"):
    """Keep one row per item id and the item-level columns."""
    missing = [column for column in ITEM_COLUMNS if column not in block]
    if missing:
        raise ValueError(f"{what} is missing required item columns {missing}.")
    identity = np.asarray(block["item_id"])
    if len(identity) == 0:
        raise ValueError(f"{what} is empty.")
    _distinct, first_row = np.unique(identity, return_index=True)
    return {column: np.asarray(block[column])[first_row] for column in ITEM_COLUMNS}


def stream_block_of(frame):
    return rows(frame, np.asarray(frame["is_stream"]).astype(bool))


def serving_carve_of(frame):
    return rows(frame, ~np.asarray(frame["is_stream"]).astype(bool))


def base_pool_of(frame):
    """Return unique items from an uncontaminated null-frame stream."""
    stream = stream_block_of(frame)
    contaminated = (np.asarray(stream["is_drift_item"]).astype(bool)
                    | (np.asarray(stream["stream_lambda"]) > 0.0))
    if contaminated.any():
        raise ValueError(f"Base stream contains {int(contaminated.sum())} contaminated positions.")
    return deduplicated_pool(stream, "the base pool")


def attack_pool_of(frame, family="the attack pool"):
    """Return unique drift items from a frame's stream."""
    stream = stream_block_of(frame)
    drift = np.asarray(stream["is_drift_item"]).astype(bool)
    if not drift.any():
        raise ValueError(f"{family}: frame stream has no drift rows.")
    return deduplicated_pool(rows(stream, drift), f"the {family} pool")


def assert_serving_carves_match(carve, other, what, other_what):
    """Require matching shared serving carves."""
    if len(np.asarray(carve["item_id"])) != len(np.asarray(other["item_id"])):
        raise ValueError(f"{what} has {len(np.asarray(carve['item_id']))} serving rows; "
                         f"{other_what} has {len(np.asarray(other['item_id']))}.")
    for column in ("item_id", "verdict", "oracle", "human", "family"):
        if not np.array_equal(np.asarray(carve[column]), np.asarray(other[column])):
            departed = int(np.sum(np.asarray(carve[column]) != np.asarray(other[column])))
            raise ValueError(f"{what} and {other_what} differ in {departed} serving "
                             f"{column} values.")


def assert_pools_are_disjoint(pools):
    """Require each item id to appear in exactly one pool."""
    seen = {}
    for name, pool in pools.items():
        for identity in np.asarray(pool["item_id"]):
            if identity in seen:
                raise ValueError(f"Item {identity!r} appears in both {seen[identity]!r} and "
                                 f"{name!r} pools.")
            seen[identity] = name


def assert_pools_are_alike(pools):
    """Require common item-column shapes and dtypes across pools."""
    reference_name, reference = next(iter(pools.items()))
    for name, pool in pools.items():
        for column in ITEM_COLUMNS:
            here, there = np.asarray(pool[column]), np.asarray(reference[column])
            widths_differ_harmlessly = here.dtype.kind == there.dtype.kind == "U"
            if (here.shape[1:] != there.shape[1:] or here.dtype.kind != there.dtype.kind
                    or (here.dtype != there.dtype and not widths_differ_harmlessly)):
                raise ValueError(
                    f"Pool {name!r} column {column} is {here.dtype}{here.shape[1:]}; "
                    f"expected {there.dtype}{there.shape[1:]} from {reference_name!r}.")


def served_columns_of(pools, positions):
    return {column: np.empty(
        (positions, *np.asarray(next(iter(pools.values()))[column]).shape[1:]),
        dtype=np.result_type(*[np.asarray(pool[column]).dtype for pool in pools.values()]))
        for column in ITEM_COLUMNS}


def schedule_of(frame):
    return np.asarray(stream_block_of(frame)["stream_lambda"])


def assert_schedule_is_the_declared_ramp(schedule, run_in, monitored, cap):
    """Require a schedule equal to the declared linear ramp."""
    schedule = np.asarray(schedule)
    declared = linear_contamination_ramp(run_in + monitored, run_in, cap)
    if len(schedule) != len(declared):
        raise ValueError(f"Schedule has {len(schedule)} positions; expected {len(declared)}.")
    departed = np.flatnonzero(schedule != declared)
    if len(departed):
        first = int(departed[0])
        raise ValueError(f"Schedule differs from declared ramp at {len(departed)} positions; "
                         f"first {first}: {schedule[first]!r}, expected {declared[first]!r}.")
    return schedule


def rate_at(schedule, stream_position):
    """Return the scheduled attack rate at a stream position."""
    schedule = np.asarray(schedule)
    if not 0 <= stream_position < len(schedule):
        raise ValueError(f"Stream position {stream_position} is outside [0, {len(schedule)}).")
    return float(schedule[stream_position])


def cycle_generator(chain_seed, cycle_index):
    return np.random.default_rng([chain_seed, cycle_index, CHAIN_DRAW_TAG])


def chain_permutation(roster, chain_seed):
    """Return a deterministic permutation of the campaign roster."""
    families = [str(family) for family in roster]
    if len(set(families)) != len(families):
        raise ValueError(f"Roster repeats family names: {families}.")
    return [str(family) for family in np.random.default_rng(chain_seed).permutation(families)]


def drawn_cycle_recipe(baseline, attack_family, pool_sizes, schedule, rng):
    """Draw pool and item assignments for each scheduled position."""
    components = list(baseline)
    for name in (*components, attack_family):
        if name not in pool_sizes:
            raise ValueError(f"No pool named {name!r}; available pools: {sorted(pool_sizes)}.")
        if pool_sizes[name] < 1:
            raise ValueError(f"Pool {name!r} is empty.")
    schedule = np.asarray(schedule, dtype=np.float64)
    pool = drawn_pool_assignment(baseline, attack_family, schedule, rng)

    positions = len(schedule)
    pick = {name: rng.integers(0, pool_sizes[name], positions)
            for name in (*components, attack_family)}
    item_index = np.empty(positions, dtype=np.int64)
    for name in (*components, attack_family):
        drew_here = pool == name
        item_index[drew_here] = pick[name][drew_here]
    return CycleRecipe(pool=pool, item_index=item_index, stream_lambda=schedule,
                       attack_family=attack_family)


def drawn_pool_assignment(baseline, attack_name, schedule, rng):
    """Sample an attack or baseline pool at each position."""
    assert_is_a_mixture(baseline)
    if attack_name in baseline:
        raise ValueError(f"Attack {attack_name!r} is already in the baseline.")
    schedule = np.asarray(schedule, dtype=np.float64)
    if schedule.ndim != 1 or len(schedule) == 0:
        raise ValueError(f"Schedule must be a non-empty one-dimensional array; got {schedule.shape}.")
    if not np.all((schedule >= 0.0) & (schedule < 1.0)):
        raise ValueError("Scheduled rates must be in [0, 1).")
    components = list(baseline)
    positions = len(schedule)
    coin = rng.random(positions)
    component_coin = rng.random(positions)

    drew_the_attack = coin < schedule
    edges = np.cumsum([baseline[name] for name in components])
    chosen = np.minimum(np.searchsorted(edges, component_coin, side="right"),
                        len(components) - 1)

    pool = np.empty(positions, dtype=np.array([*components, attack_name]).dtype)
    pool[drew_the_attack] = attack_name
    for index, name in enumerate(components):
        pool[~drew_the_attack & (chosen == index)] = name
    return pool


def stream_of_recipe(recipe, pools):
    """Gather a cycle stream from recipe-selected pools."""
    drawn_from = [str(name) for name in np.unique(recipe.pool)]
    for name in drawn_from:
        if name not in pools:
            raise ValueError(f"Recipe draws from {name!r}; available pools: {sorted(pools)}.")
    drawn_pools = {name: pools[name] for name in drawn_from}
    assert_pools_are_alike(drawn_pools)

    positions = len(recipe.pool)
    served = served_columns_of(drawn_pools, positions)
    for name in drawn_from:
        drew_here = recipe.pool == name
        index = recipe.item_index[drew_here]
        available = len(np.asarray(pools[name]["item_id"]))
        if index.min() < 0 or index.max() >= available:
            raise ValueError(f"Recipe index for pool {name!r} is outside [0, {available}): "
                             f"maximum {int(index.max())}.")
        for column in ITEM_COLUMNS:
            served[column][drew_here] = np.asarray(pools[name][column])[index]

    served["is_drift_item"] = recipe.pool == recipe.attack_family
    served["is_stream"] = np.ones(positions, dtype=bool)
    served["stream_lambda"] = np.asarray(recipe.stream_lambda)
    served["pool"] = recipe.pool
    return served


def composed_cycle(baseline, attack_family, pools, schedule, rng):
    """Draw and gather one cycle stream."""
    sizes = {name: len(np.asarray(pool["item_id"])) for name, pool in pools.items()}
    recipe = drawn_cycle_recipe(baseline, attack_family, sizes, schedule, rng)
    return recipe, stream_of_recipe(recipe, pools)


def assembled_cycle_frame(serving, stream):
    """Combine a serving carve with a composed stream."""
    if np.asarray(serving["is_stream"]).any():
        raise ValueError("Serving carve contains stream rows.")
    if not np.asarray(stream["is_stream"]).all():
        raise ValueError("Composed stream contains serving rows.")
    carve = dict(serving)
    carve["pool"] = np.full(len(np.asarray(serving["item_id"])), SERVING_POOL)
    mismatched = set(carve) ^ set(stream)
    if mismatched:
        raise ValueError(f"Serving carve and stream have different columns: {sorted(mismatched)}.")
    return concatenated(carve, stream)


def assert_run_in_draws_no_attack(recipe, run_in):
    """Require run-in positions to be free of scheduled and sampled attack rows."""
    if not 0 < run_in < len(recipe.pool):
        raise ValueError(f"run_in must be in [1, {len(recipe.pool)}); got {run_in}.")
    contaminated = np.flatnonzero(recipe.pool[:run_in] == recipe.attack_family)
    if len(contaminated):
        raise ValueError(f"Run-in samples attack {len(contaminated)} times; first at position "
                         f"{int(contaminated[0])}.")
    scheduled = np.flatnonzero(np.asarray(recipe.stream_lambda[:run_in]) > 0.0)
    if len(scheduled):
        raise ValueError(f"Run-in schedule is nonzero at {len(scheduled)} positions; first at "
                         f"{int(scheduled[0])}.")


def assert_pool_shares_are_within_noise(recipe, baseline, sigmas):
    """Check sampled pool counts against the design within sigmas."""
    assert_is_a_mixture(baseline)
    schedule = np.asarray(recipe.stream_lambda)
    designed = {recipe.attack_family: schedule}
    for name, share in baseline.items():
        designed[name] = (1.0 - schedule) * share

    departed = {}
    for name, probability in designed.items():
        observed = int((recipe.pool == name).sum())
        expected = float(probability.sum())
        variance = float((probability * (1.0 - probability)).sum())
        if variance == 0.0:
            if observed != round(expected):
                departed[name] = {"observed": observed, "expected": expected, "sigmas": None}
            continue
        deviation = (observed - expected) / math.sqrt(variance)
        if abs(deviation) > sigmas:
            departed[name] = {"observed": observed, "expected": round(expected, 2),
                              "sigmas": round(deviation, 2)}
    if departed:
        raise ValueError(f"Sampled pool counts exceed {sigmas} standard deviations from the "
                         f"design: {departed}.")


def assert_columns_identical(first, second, what):
    """Require identical columns, dtypes, shapes, and values."""
    mismatched = set(first) ^ set(second)
    if mismatched:
        raise ValueError(f"{what}: the two blocks disagree on the columns {sorted(mismatched)}")
    for column in first:
        here, there = np.asarray(first[column]), np.asarray(second[column])
        if here.dtype != there.dtype or here.shape != there.shape:
            raise ValueError(f"{what}: {column} is {here.dtype}{here.shape} against "
                             f"{there.dtype}{there.shape}")
        if not np.array_equal(here, there):
            raise ValueError(f"{what}: {column} differs at "
                             f"{int(np.sum(here != there))} position(s)")


RECIPE_ARRAY_NAMES = ("pool", "item_index", "stream_lambda", "served_item_id", "attack_family")


def recipe_arrays(recipe, served_item_id):
    """Return persisted recipe arrays and their digest."""
    arrays = {"pool": np.asarray(recipe.pool),
              "item_index": np.asarray(recipe.item_index),
              "stream_lambda": np.asarray(recipe.stream_lambda),
              "served_item_id": np.asarray(served_item_id),
              "attack_family": np.asarray(recipe.attack_family)}
    return {**arrays, "sha256": np.asarray(recipe_digest(arrays))}


def recipe_digest(arrays):
    """Return a digest over ordered recipe names, dtypes, shapes, and bytes."""
    digest = hashlib.sha256()
    for name in RECIPE_ARRAY_NAMES:
        value = np.asarray(arrays[name])
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("utf-8"))
        digest.update(str(value.shape).encode("utf-8"))
        digest.update(value.tobytes())
    return digest.hexdigest()


def recipe_of_arrays(arrays):
    """Validate persisted recipe arrays and return the recipe with served item ids."""
    declared = str(arrays["sha256"])
    actual = recipe_digest(arrays)
    if declared != actual:
        raise ValueError(f"Recipe SHA-256 mismatch: declared {declared[:12]}…, "
                         f"computed {actual[:12]}….")
    recipe = CycleRecipe(pool=np.asarray(arrays["pool"]),
                         item_index=np.asarray(arrays["item_index"]),
                         stream_lambda=np.asarray(arrays["stream_lambda"]),
                         attack_family=str(arrays["attack_family"]))
    return recipe, np.asarray(arrays["served_item_id"])



ATTACK_COMPONENT = "attack"


def post_alarm_pools(frame):
    """Return deduplicated base and attack pools from a frame's stream."""
    stream = stream_block_of(frame)
    drift = np.asarray(stream["is_drift_item"]).astype(bool)
    if not drift.any():
        raise ValueError("Frame stream has no attack rows.")
    if drift.all():
        raise ValueError("Frame stream has no base rows.")
    return (deduplicated_pool(rows(stream, ~drift), "the post-alarm base pool"),
            deduplicated_pool(rows(stream, drift), "the post-alarm attack pool"))


def baseline_with_pools(baseline, pools):
    """Attach each baseline component to its source pool."""
    missing = [name for name in baseline if name not in pools]
    if missing:
        raise ValueError(f"No pools for baseline components {missing}; available pools: "
                         f"{sorted(pools)}.")
    return {name: (share, pools[name]) for name, share in baseline.items()}


def ladder_length(audit_budget, max_enlarging_retries):
    """Return the largest audit segment a bounded retry ladder can consume."""
    if audit_budget < 1 or max_enlarging_retries < 0:
        raise ValueError(f"audit_budget must be at least 1 and max_enlarging_retries nonnegative; "
                         f"got {audit_budget}, {max_enlarging_retries}.")
    return audit_budget * (1 + max_enlarging_retries)


def eligible_rows(pool, exclude_items):
    if not exclude_items:
        return pool
    return rows(pool, ~np.isin(np.asarray(pool["item_id"]), np.asarray(sorted(exclude_items))))


def unique_draw(size, needed, rng):
    """Draw indices without repetition until the pool is exhausted."""
    order = rng.permutation(size)
    if needed <= size:
        return order[:needed], 0
    passes = -(-needed // size)
    return np.tile(order, passes)[:needed], needed - size


def stationary_segment(baseline, attack_pool, rate, length, rng, exclude_items=frozenset()):
    """Compose a fixed-rate segment, preferring items outside exclude_items."""
    if not (isinstance(rate, (int, float)) and math.isfinite(rate) and 0.0 <= rate < 1.0):
        raise ValueError(f"Segment rate must be in [0, 1); got {rate!r}.")
    if length < 1:
        raise ValueError(f"Segment length must be at least 1; got {length}.")
    if ATTACK_COMPONENT in baseline:
        raise ValueError(f"Baseline component name {ATTACK_COMPONENT!r} is reserved for the attack.")
    shares = {name: share for name, (share, _pool) in baseline.items()}
    pools = {name: pool for name, (_share, pool) in baseline.items()}
    pools[ATTACK_COMPONENT] = attack_pool
    for name, pool in pools.items():
        if len(np.asarray(pool["item_id"])) == 0:
            raise ValueError(f"Pool {name!r} is empty.")

    drawn_from, exhausted = {}, {}
    for name, pool in pools.items():
        eligible = eligible_rows(pool, exclude_items)
        exhausted[name] = len(np.asarray(eligible["item_id"])) == 0
        drawn_from[name] = pool if exhausted[name] else eligible

    schedule = np.full(length, float(rate), dtype=np.float64)
    pool_of_position = drawn_pool_assignment(shares, ATTACK_COMPONENT, schedule, rng)
    item_index = np.empty(length, dtype=np.int64)
    repeated = {}
    for name in (*shares, ATTACK_COMPONENT):
        drew_here = np.flatnonzero(pool_of_position == name)
        size = len(np.asarray(drawn_from[name]["item_id"]))
        drawn, repeats = unique_draw(size, len(drew_here), rng)
        item_index[drew_here] = drawn
        repeated[name] = len(drew_here) if exhausted[name] else repeats

    recipe = CycleRecipe(pool=pool_of_position, item_index=item_index, stream_lambda=schedule,
                         attack_family=ATTACK_COMPONENT)
    segment = stream_of_recipe(recipe, drawn_from)
    provenance = {
        "rate": float(rate), "length": int(length),
        "eligible_attack": int(len(np.asarray(drawn_from[ATTACK_COMPONENT]["item_id"]))
                               * (not exhausted[ATTACK_COMPONENT])),
        "eligible_base": int(sum(len(np.asarray(drawn_from[name]["item_id"]))
                                 * (not exhausted[name]) for name in shares)),
        "reused_attack_rows": int(repeated[ATTACK_COMPONENT]),
        "reused_base_rows": int(sum(repeated[name] for name in shares)),
        "sha256": columns_digest(segment),
    }
    return segment, provenance


def columns_digest(columns):
    """Return a digest over sorted column names, dtypes, shapes, and bytes."""
    digest = hashlib.sha256()
    for name in sorted(columns):
        value = np.asarray(columns[name])
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("utf-8"))
        digest.update(str(value.shape).encode("utf-8"))
        digest.update(np.ascontiguousarray(value).tobytes())
    return digest.hexdigest()




ALL_FRESH = "all-fresh"
EXHAUST_FRESH_DATA = "exhaust-fresh-data"


class GateUncomposable(ValueError):
    """No permissible item can fill a required gate position."""


def audit_and_gate_blocks(baseline, attack_pool, rate, audit_length, gate_length, rng,
                          consumed_items=frozenset()):
    """Compose a gate block and a separate audit block at the alarm-time rate."""
    assert_composable_lengths(audit_length, gate_length, rate)
    shares = {name: share for name, (share, _pool) in baseline.items()}
    pools = {name: pool for name, (_share, pool) in baseline.items()}
    if ATTACK_COMPONENT in shares:
        raise ValueError(f"Baseline component name {ATTACK_COMPONENT!r} is reserved for the attack.")
    pools[ATTACK_COMPONENT] = attack_pool
    consumed = np.asarray(sorted(consumed_items), dtype=str)

    gate, gate_reuse = drawn_block(shares, pools, rate, gate_length, rng,
                                   forbidden=consumed, stale=consumed, refusal=GateUncomposable,
                                   what="the gate block")
    gate_items = np.unique(np.asarray(gate["item_id"]))
    stale = np.union1d(consumed, gate_items)
    if audit_length == ALL_FRESH:
        audit, audit_reuse = drawn_until_dry(shares, pools, rate, rng, stale)
    else:
        audit, audit_reuse = drawn_block(shares, pools, rate, audit_length, rng,
                                         forbidden=gate_items, stale=stale,
                                         refusal=ValueError, what="the audit block")
    return audit, gate, {
        "rate": float(rate), "audit_length": len(audit["verdict"]),
        "gate_length": int(gate_length),
        "eligible_attack": int(never_consumed_rows(pools[ATTACK_COMPONENT], consumed).sum()),
        "eligible_base": int(sum(never_consumed_rows(pools[name], consumed).sum()
                                 for name in shares)),
        "audit_reused_rows": int(sum(audit_reuse.values())),
        "gate_reused_rows": int(sum(gate_reuse.values())),
        "audit_sha256": columns_digest(audit), "gate_sha256": columns_digest(gate),
    }


def assert_composable_lengths(audit_length, gate_length, rate):
    if not (isinstance(rate, (int, float)) and math.isfinite(rate) and 0.0 <= rate < 1.0):
        raise ValueError(f"Block rate must be in [0, 1); got {rate!r}.")
    lengths = [("gate_length", gate_length)]
    if audit_length != ALL_FRESH:
        lengths.append(("audit_length", audit_length))
    for name, length in lengths:
        if not isinstance(length, (int, np.integer)) or length < 1:
            raise ValueError(f"{name} must be a positive integer; got {length!r}.")


def never_consumed_rows(pool, consumed):
    return ~np.isin(np.asarray(pool["item_id"]), consumed)


def drawn_block(shares, pools, rate, length, rng, forbidden, stale, refusal, what):
    """Compose one block, using fresh items before permitted reused items."""
    pool_of_position = drawn_pool_assignment(shares, ATTACK_COMPONENT,
                                             np.full(length, float(rate), dtype=np.float64), rng)
    item_index = np.empty(length, dtype=np.int64)
    reused = {}
    for name in (*shares, ATTACK_COMPONENT):
        identity = np.asarray(pools[name]["item_id"])
        permitted = np.flatnonzero(~np.isin(identity, forbidden))
        fresh = np.flatnonzero(~np.isin(identity, stale))
        positions = np.flatnonzero(pool_of_position == name)
        if len(positions) and len(permitted) == 0:
            raise refusal(f"{what} needs {len(positions)} positions from pool {name!r}, but all "
                          f"{len(identity)} items are forbidden.")
        drawn, repeats = fresh_first_draw(fresh, permitted, len(positions), rng)
        item_index[positions] = drawn
        reused[name] = repeats
    recipe = CycleRecipe(pool=pool_of_position, item_index=item_index,
                         stream_lambda=np.full(length, float(rate), dtype=np.float64),
                         attack_family=ATTACK_COMPONENT)
    return stream_of_recipe(recipe, pools), reused


def fresh_first_draw(fresh, permitted, needed, rng):
    """Draw fresh indices first, then repeat permitted indices as needed."""
    fresh_order = rng.permutation(len(fresh))
    permitted_order = rng.permutation(len(permitted))
    if needed <= len(fresh):
        return fresh[fresh_order[:needed]], 0
    shortfall = needed - len(fresh)
    passes = -(-shortfall // len(permitted))
    filler = np.tile(permitted[permitted_order], passes)[:shortfall]
    return np.concatenate([fresh[fresh_order], filler]), shortfall


def reference_after(change_log, standing):
    """Return the latest accepted repair's gate metrics, or the standing metrics."""
    accepted = [entry for entry in change_log if entry.what == "repair"]
    if not accepted:
        return standing
    latest = accepted[-1]
    if latest.gate_recall is None or latest.gate_over_block is None:
        raise ValueError("Accepted repair is missing gate recall or over-block metrics.")
    return {"recall": float(latest.gate_recall), "over_block": float(latest.gate_over_block)}


def assert_run_in_composes_the_alarm_time_distribution(recipe, run_in, baseline_in, previous,
                                                       sigmas):
    """Require the run-in baseline to equal the prior cycle's absorbed distribution."""
    expected = (initial_baseline() if previous is None
                else absorbed(previous["baseline_in"], previous["family"], previous["r_locked"]))
    try:
        assert_mixtures_agree(baseline_in, expected, "the run-in")
    except ValueError as departure:
        raise ValueError(f"Run-in baseline differs from the prior absorbed distribution: "
                         f"{departure}") from departure
    run_in_recipe = CycleRecipe(pool=recipe.pool[:run_in], item_index=recipe.item_index[:run_in],
                                stream_lambda=recipe.stream_lambda[:run_in],
                                attack_family=recipe.attack_family)
    assert_run_in_draws_no_attack(recipe, run_in)
    assert_pool_shares_are_within_noise(run_in_recipe, baseline_in, sigmas)


def drawn_until_dry(shares, pools, rate, rng, stale):
    """Compose fresh, unique audit rows until a required pool is exhausted."""
    fresh = {name: np.flatnonzero(~np.isin(np.asarray(pools[name]["item_id"]), stale))
             for name in (*shares, ATTACK_COMPONENT)}
    ceiling = sum(len(rows_) for rows_ in fresh.values())
    if ceiling == 0:
        raise ValueError("No pool holds a fresh item for the audit.")
    pool_of_position = drawn_pool_assignment(shares, ATTACK_COMPONENT,
                                             np.full(ceiling + 1, float(rate), dtype=np.float64),
                                             rng)
    order = {name: rng.permutation(len(rows_)) for name, rows_ in fresh.items()}

    realised = len(pool_of_position)
    for name, rows_ in fresh.items():
        drew_here = np.flatnonzero(pool_of_position == name)
        if len(drew_here) > len(rows_):
            realised = min(realised, int(drew_here[len(rows_)]))
    if realised == 0:
        raise ValueError(f"First audit position needs pool {str(pool_of_position[0])!r}, which "
                         f"has no fresh item.")

    item_index = np.empty(realised, dtype=np.int64)
    served = pool_of_position[:realised]
    for name, rows_ in fresh.items():
        drew_here = np.flatnonzero(served == name)
        item_index[drew_here] = rows_[order[name][:len(drew_here)]]
    recipe = CycleRecipe(pool=served, item_index=item_index,
                         stream_lambda=np.full(realised, float(rate), dtype=np.float64),
                         attack_family=ATTACK_COMPONENT)
    return stream_of_recipe(recipe, pools), {name: 0 for name in fresh}


def assert_bounded_ladder(max_enlarging_retries):
    """Require a nonnegative integer retry count for chained campaigns."""
    if max_enlarging_retries == EXHAUST_FRESH_DATA:
        raise ValueError(f"max_enlarging_retries {EXHAUST_FRESH_DATA!r} is unsupported for "
                         "chained campaigns.")
    if not isinstance(max_enlarging_retries, (int, np.integer)) or max_enlarging_retries < 0:
        raise ValueError(f"max_enlarging_retries must be a nonnegative integer; got "
                         f"{max_enlarging_retries!r}.")
    return int(max_enlarging_retries)
