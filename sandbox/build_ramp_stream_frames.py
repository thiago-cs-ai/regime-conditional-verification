
"""Run-once integration build: the RAMP-STREAM frames — a generated, continuously-ramping stream."""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

RCV = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RCV / "src"))
sys.path.insert(0, str(RCV / "sandbox"))

import build_stream_frames  # noqa: E402  (the module, so its FAMILY can be rebound)
from build_stream_frames import (  # noqa: E402
    CAMPAIGN,
    CLASSIFIER,
    CORPUS,
    EXPECTED_CORPUS_ROWS,
    EXPECTED_Z_DIM,
    FAMILY,
    OUT,
    PHASE_D,
    SERVING_POOLS,
    STREAM_POOL,
    WILDJAILBREAK,
    balance,
    block_rows,
    load_corpus,
    load_join,
    load_oracle,
    load_partition,
    load_wildjailbreak_seeds,
    read_jsonl,
    seed_for,
    source_anchors,
)

from rcv.frame_building import digest_of, family_codes, input_manifest, save_frame  # noqa: E402
from rcv.stream_composition import compose_stream, linear_contamination_ramp  # noqa: E402

STREAM_LENGTH = 21000
RAMP_ONSET = 2000
DEFAULT_RAMP_SEED = 20260727
RAMP_SEED = DEFAULT_RAMP_SEED
SEED_SUFFIX = ""
VARIANTS = (("null", 0.0), ("0.15", 0.15), ("0.30", 0.30))

FAMILY_POOLS = {
    "private_information_individual": ("privinfo", 1018),
    "sensitive_information_organization_government": ("sensinfo", 854),
    "toxic_language_hate_speech": ("toxic", 702),
    "disseminating_false_or_misleading_information_encouraging_disinformation_campaigns":
        ("disinfo", 955),
    "fraud_assisting_illegal_activities": ("fraud", 743),
    "cyberattack": ("cyber", 300),
    "defamation_encouraging_unethical_or_unsafe_actions": ("defamation", 505),
    "violence_and_physical_harm": ("violence", 519),
    "mental_health_over-reliance_crisis": ("mentalhealth", 388),
    "causing_material_harm_by_disseminating_misinformation": ("materialharm", 400),
    "social_stereotypes_and_unfair_discrimination": ("stereotypes", 2107),
}
DEFAULT_FAMILY = FAMILY
FAMILY_TAG, EXPECTED_DRIFT_POOL = FAMILY_POOLS[DEFAULT_FAMILY]
DRIFT_SET_ID = f"drift:{FAMILY}"
EXPECTED_BASE_POOL = 3499
DECILES = 10
SIGMA_BAND = 5.0


def select_family(family):
    """Rebind the drift family for this process — here and in `build_stream_frames`, whose `load_corpus`/`load_oracle` read its module global to know which judged slice to open."""
    if family not in FAMILY_POOLS:
        raise SystemExit(f"{family!r} is not a judged drift family; the panel's roster is "
                         f"{sorted(FAMILY_POOLS)}")
    for required in (CORPUS / "drift_candidates" / f"{family}.jsonl",
                     CORPUS / "judged" / f"{family}.judged.jsonl"):
        if not required.exists():
            raise SystemExit(f"the family {family!r} has no {required.name}; a frame cannot be "
                             f"built for a family the corpus never judged")
    global FAMILY, FAMILY_TAG, EXPECTED_DRIFT_POOL, DRIFT_SET_ID
    FAMILY = family
    FAMILY_TAG, EXPECTED_DRIFT_POOL = FAMILY_POOLS[family]
    DRIFT_SET_ID = f"drift:{family}"
    build_stream_frames.FAMILY = family
    return family


def select_stream_seed(seed):
    """Rebind the composition seed for this process, and with it the suffix a frame built off any other composition carries in its name."""
    global RAMP_SEED, SEED_SUFFIX
    RAMP_SEED = int(seed)
    SEED_SUFFIX = "" if RAMP_SEED == DEFAULT_RAMP_SEED else f"_s{RAMP_SEED}"
    return RAMP_SEED


def frame_path(family_tag, variant):
    """Where a frame lands."""
    return OUT / f"ramp_{CLASSIFIER}_{family_tag}_{variant}{SEED_SUFFIX}.npz"


def drift_pool_of(join, family_item_ids):
    """The drifting family, taken from the frozen join's own `set_id` and proven equal to the corpus slice the campaigns declare as the family's pool — so the pool does not depend on which of the two the build happened to read."""
    from_join = sorted(item for item, set_id in join["set_id"].items() if set_id == DRIFT_SET_ID)
    if len(from_join) != EXPECTED_DRIFT_POOL:
        raise ValueError(f"the join carries {len(from_join)} {DRIFT_SET_ID} items, expected "
                         f"{EXPECTED_DRIFT_POOL}")
    if from_join != sorted(family_item_ids):
        raise ValueError("the join's drift slice and the drift_candidates file name different "
                         "item sets; the drift pool would depend on which was read")
    return from_join


def base_pool_of(pools):
    base_pool = sorted(pools[STREAM_POOL])
    if len(base_pool) != EXPECTED_BASE_POOL:
        raise ValueError(f"the {STREAM_POOL} pool carries {len(base_pool)} ids, expected "
                         f"{EXPECTED_BASE_POOL}")
    return base_pool


def assert_stream_is_servable(pools, stream, join, tag):
    """Every position is the kind of position the composer says it is, drawn from the pool that kind of position may draw from — and serving stays material the stream never shows."""
    serving = sorted({item for pool in SERVING_POOLS for item in pools[pool]})
    traffic = set(pools[STREAM_POOL])
    order = stream.item_id.tolist()

    unknown = sorted({item for item in order + serving if item not in join["row_of"]})
    if unknown:
        raise ValueError(f"{len(unknown)} ids are absent from the phase-D join (first "
                         f"{unknown[0]!r}); the stream would be joining two corpora")

    for position, (item, drifting) in enumerate(zip(order, stream.is_drift_item)):
        expected = DRIFT_SET_ID if drifting else "pool"
        if join["set_id"][item] != expected:
            raise ValueError(f"ramp {tag} position {position}: item {item!r} is "
                             f"{join['set_id'][item]!r} but the draw flags it "
                             f"{'drift' if drifting else 'base'}")
        if not drifting and item not in traffic:
            raise ValueError(f"ramp {tag} position {position}: base item {item!r} is outside the "
                             f"{STREAM_POOL} pool; serving would not be drift-free-and-unseen")
    if stream.is_drift_item[:RAMP_ONSET].any():
        raise ValueError(f"ramp {tag}: a contaminated position inside the prefix, which the "
                         f"monitor's reference is taken from")
    if set(serving) & set(order):
        raise ValueError("serving and the stream share an item; the block the estimator is "
                         "fitted on would also be traffic it is monitored over")
    for item in serving:
        if join["set_id"][item] != "pool":
            raise ValueError(f"serving item {item!r} is {join['set_id'][item]!r}, not base pool")
    return serving


def assert_realized_follows_schedule(stream, tag):
    """The realized contamination is a draw, so it is checked against its own expectation and standard deviation, not against an exact count — overall and in each decile of the monitored region, which makes it a check on the ramp's SHAPE and not only on its total."""
    schedule = stream.stream_lambda
    edges = np.linspace(RAMP_ONSET, len(schedule), DECILES + 1).astype(int)
    bands = [("monitored", slice(RAMP_ONSET, len(schedule)))]
    bands += [(f"decile {index + 1}", slice(int(edges[index]), int(edges[index + 1])))
              for index in range(DECILES)]

    realized = []
    for name, band in bands:
        rates = schedule[band]
        expected, deviation = float(rates.sum()), float(np.sqrt((rates * (1.0 - rates)).sum()))
        observed = int(stream.is_drift_item[band].sum())
        if abs(observed - expected) > SIGMA_BAND * deviation:
            raise ValueError(f"ramp {tag} {name}: {observed} contaminated positions against an "
                             f"expectation of {expected:.1f} +/- {deviation:.1f}; the realized "
                             f"contamination does not follow the schedule")
        realized.append({"band": name, "positions": int(rates.size), "expected": expected,
                         "sd": deviation, "observed": observed})
    return realized


def build_ramp_frame(serving, stream, corpus, oracle, join, seed_of_adversarial, seeds,
                     representation):
    """Serving first, in canonical item order; then the stream, in composed order."""
    blocks = [block_rows(sorted(serving), False, corpus, oracle, join),
              block_rows(stream.item_id.tolist(), True, corpus, oracle, join)]
    prompt = [text for block in blocks for text in block["prompt"]]
    seed = [seed_for(text, seed_of_adversarial, seeds) for text in prompt]
    join_rows = np.concatenate([block["join_row"] for block in blocks])
    serving_rows = len(blocks[0]["item_id"])

    taken = np.ascontiguousarray(representation[join_rows])
    if taken.dtype != np.float32:
        raise ValueError(f"the representation is {taken.dtype}; a frame stores the dtype the "
                         f"classifier emitted")
    if not np.isfinite(taken).all():
        raise ValueError(f"{int((~np.isfinite(taken)).sum())} non-finite values in the joined "
                         f"representation")

    return {
        "item_id": np.concatenate([block["item_id"] for block in blocks]),
        "representation": taken,
        "verdict": np.concatenate([block["verdict"] for block in blocks]),
        "oracle": np.concatenate([block["oracle"] for block in blocks]),
        "human": np.concatenate([block["human"] for block in blocks]),
        "family": family_codes(prompt, seed),
        "is_stream": np.concatenate([block["is_stream"] for block in blocks]),
        "is_drift_item": np.concatenate([np.zeros(serving_rows, dtype=np.bool_),
                                         stream.is_drift_item]),
        "stream_lambda": np.concatenate([np.zeros(serving_rows, dtype=np.float64),
                                         stream.stream_lambda]),
    }


def ramp_statistics(frame, tag, maximum_rate, bands):
    serving, traffic = ~frame["is_stream"], frame["is_stream"]
    drifting = frame["is_drift_item"]
    prefix = np.zeros(len(traffic), dtype=bool)
    prefix[np.flatnonzero(traffic)[:RAMP_ONSET]] = True
    schedule = frame["stream_lambda"][traffic]

    def slice_stats(mask):
        verdict, oracle = frame["verdict"][mask], frame["oracle"][mask]
        agreement = (verdict == oracle)
        return {"rows": int(mask.sum()),
                "distinct_items": int(len(np.unique(frame["item_id"][mask]))),
                "verdict_unsafe": int(verdict.sum()),
                "oracle_unsafe": int(oracle.sum()),
                "human_unsafe": int(frame["human"][mask].sum()),
                "adherence": float(agreement.mean()) if mask.any() else None,
                "disagreement_by_regime": {
                    regime: (float((~agreement[verdict == regime]).mean())
                             if (verdict == regime).any() else None)
                    for regime in (0, 1)},
                "cases": {f"verdict{v}_agreement{a}":
                          int(((verdict == v) & (agreement == bool(a))).sum())
                          for v in (0, 1) for a in (0, 1)}}

    _, sizes = np.unique(frame["family"][serving], return_counts=True)
    return {
        "serving": slice_stats(serving),
        "stream": slice_stats(traffic),
        "stream_prefix": slice_stats(prefix),
        "stream_monitored": slice_stats(traffic & ~prefix),
        "stream_base": slice_stats(traffic & ~drifting),
        "stream_drift": slice_stats(traffic & drifting),
        "serving_families": {"families": int(len(sizes)), "largest": int(sizes.max()),
                             "rows_in_a_multi_row_family": int(sizes[sizes > 1].sum())},
        "families_total": int(len(np.unique(frame["family"]))),
        "schedule": {"tag": tag, "maximum_rate": maximum_rate, "length": STREAM_LENGTH,
                     "onset": RAMP_ONSET, "seed": RAMP_SEED,
                     "expected_contaminated": float(schedule.sum()),
                     "realized_contaminated": int(drifting.sum()),
                     "mean_rate_monitored": float(schedule[RAMP_ONSET:].mean()),
                     "bands": bands},
    }


def assert_frame_matches_stream(frame, stats, stream, anchors):
    traffic = int(frame["is_stream"].sum())
    if traffic != STREAM_LENGTH:
        raise ValueError(f"the frame carries {traffic} stream rows against a stream length of "
                         f"{STREAM_LENGTH}")
    if frame["is_drift_item"][~frame["is_stream"]].any():
        raise ValueError("a serving row is flagged a drift item; serving has no stream position")
    if frame["stream_lambda"][~frame["is_stream"]].any():
        raise ValueError("a serving row carries a scheduled rate; serving has no stream position")
    if not np.array_equal(frame["stream_lambda"][frame["is_stream"]], stream.stream_lambda):
        raise ValueError("the frame's stream_lambda column is not the schedule that drew it")
    if not np.array_equal(frame["is_drift_item"][frame["is_stream"]], stream.is_drift_item):
        raise ValueError("the frame's is_drift_item column is not the composer's")
    if frame["is_drift_item"].dtype != np.bool_:
        raise ValueError(f"is_drift_item is {frame['is_drift_item'].dtype}; a mask column is "
                         f"boolean, or `~` fancy-indexes instead of masking")
    if frame["stream_lambda"].dtype != np.float64:
        raise ValueError(f"stream_lambda is {frame['stream_lambda'].dtype}; the schedule is a "
                         f"float64 probability")
    if stats["stream_prefix"]["rows"] != RAMP_ONSET:
        raise ValueError(f"the prefix carries {stats['stream_prefix']['rows']} rows, not "
                         f"{RAMP_ONSET}")

    serving = stats["serving"]
    for name in ("rows", "verdict_unsafe", "oracle_unsafe"):
        if serving[name] != anchors[name]:
            raise ValueError(f"serving {name} {serving[name]} against the carve's own "
                             f"{anchors[name]}; the frame's columns are not the source's")
    if abs(serving["adherence"] - anchors["adherence"]) > 1e-12:
        raise ValueError("serving adherence differs from the independently recomputed value")
    for case, count in serving["cases"].items():
        if count == 0:
            raise ValueError(f"serving has no {case} rows; no split of it can carry all four "
                             f"cases")


def determinism_check(serving, stream, schedule, base_pool, drift_pool, corpus, oracle, join,
                      seed_of_adversarial, seeds, representation, shipped_digest, path):
    """The digest means something only if a rebuild reproduces it."""
    scratch = path.with_name(path.name + ".determinism-check")
    if scratch.exists():
        raise FileExistsError(f"{scratch} is left over from an earlier check")

    recomposed = compose_stream(schedule, base_pool, drift_pool, RAMP_SEED)
    if not np.array_equal(recomposed.item_id, stream.item_id) or \
            not np.array_equal(recomposed.is_drift_item, stream.is_drift_item):
        raise ValueError("a second composition at the same seed produced a different stream; the "
                         "draw is not seeded the way it claims")
    straight = save_frame(build_ramp_frame(serving, recomposed, corpus, oracle, join,
                                           seed_of_adversarial, seeds, representation), scratch)
    scratch.unlink()
    if straight != shipped_digest:
        raise ValueError(f"a straight rebuild digests {straight}, not {shipped_digest}")

    shuffled = list(np.random.default_rng(11).permutation(np.array(serving)))
    permuted = save_frame(build_ramp_frame(shuffled, stream, corpus, oracle, join,
                                           seed_of_adversarial, seeds, representation), scratch)
    scratch.unlink()
    if permuted != shipped_digest:
        raise ValueError(f"a rebuild from a permuted serving list digests {permuted}, not "
                         f"{shipped_digest}; the frame remembers its input order")
    return straight


def serving_family_overlap(frame):
    """The share of stream positions whose FAMILY also appears in the serving carve, split by pool side."""
    serving_families = set(np.asarray(frame["family"])[~frame["is_stream"]].tolist())
    stream_families = np.asarray(frame["family"])[frame["is_stream"]]
    seen = np.array([family in serving_families for family in stream_families.tolist()])
    drift = np.asarray(frame["is_drift_item"])[frame["is_stream"]].astype(bool)
    return {"all": float(np.mean(seen)),
            "base": float(np.mean(seen[~drift])) if (~drift).any() else None,
            "drift": float(np.mean(seen[drift])) if drift.any() else None}


def family_overlap_sentence(overlap):
    """How the build report states it."""
    def share(value):
        return "unmeasurable" if value is None else f"{value * 100:.2f}%"

    return (f"- serving n stream = 0 at the ITEM level (checked above), but the grouping unit is "
            f"the FAMILY and that axis straddles: {share(overlap['all'])} of stream positions sit "
            f"in a family the serving carve also carries — {share(overlap['base'])} of base "
            f"positions against {share(overlap['drift'])} of drift positions. Recorded, not "
            f"excluded: a probe fitted on the serving carve has seen material like the base side "
            f"of this stream and not like the drift side, which biases any base-versus-drift "
            f"contrast measured on it.")


def publish_after_gate(frame, path, gate):
    """Stage, gate, then move."""
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"{path} already exists; frames are written once and never "
                              f"overwritten")
    staging = path.with_name(path.name + ".staging")
    if staging.exists():
        raise FileExistsError(f"{staging} is left over from an interrupted build; inspect it "
                              f"before running again")
    sha256 = save_frame(frame, staging)
    try:
        gate(staging, sha256)
    except BaseException:
        staging.unlink(missing_ok=True)
        raise
    os.replace(staging, path)
    return sha256


def assert_variants_nest(streams):
    """The coupling the shared seed buys, stated as a property and checked on the real pools: the schedules are pointwise ordered and the composer's draws do not depend on their values, so the contaminated position sets nest and every position two variants both leave clean carries the same base item."""
    ordered = [tag for tag, _ in VARIANTS if tag in streams]
    if len(ordered) < 2:
        return None
    for gentler, steeper in zip(ordered, ordered[1:]):
        lower, upper = streams[gentler], streams[steeper]
        if np.any(lower.is_drift_item & ~upper.is_drift_item):
            raise ValueError(f"{gentler}'s contaminated positions are not a subset of "
                             f"{steeper}'s; the schedules are ordered, so the draws must nest")
        both_clean = ~lower.is_drift_item & ~upper.is_drift_item
        if not np.array_equal(lower.item_id[both_clean], upper.item_id[both_clean]):
            raise ValueError(f"{gentler} and {steeper} serve different base items at positions "
                             f"neither contaminates; the null is not a matched control")
    return " ⊆ ".join(ordered)


def nesting_sentence(nesting):
    """How the report names the nesting gate's outcome."""
    if nesting is None:
        return ("the cross-variant nesting gate was NOT CHECKED in this build — it compares "
                "variants pairwise and this run built one")
    return (f"the variants' contaminated positions NEST ({nesting}) and every position two "
            f"variants both leave clean carries the same base item — verified")


def append_report(built, partition, manifest, nesting):
    report = OUT / "BUILD_REPORT.md"
    lines = [
        "",
        "---",
        "",
        f"## Ramp-stream frames — {FAMILY}, generated continuous schedule, stream seed "
        f"{RAMP_SEED}",
        "",
        "",
        "",
        "",
        "",
        "| frame | maximum_rate | contaminated (expected) | distinct drift items | bytes | "
        "sha256 |",
        "|---|---:|---|---:|---:|---|",
    ]
    for entry in built:
        schedule, path = entry["stats"]["schedule"], entry["path"]
        lines.append(
            f"| `{path.name}` | {schedule['maximum_rate']} | "
            f"{schedule['realized_contaminated']} "
            f"({schedule['expected_contaminated']:.1f}) | "
            f"{entry['stats']['stream_drift']['distinct_items']} | {path.stat().st_size} | "
            f"`{entry['sha256']}` |")
    lines += [
        "",
        "",
        "### Per-frame blocks",
        "",
        "| frame | block | rows | distinct items | verdict | oracle | human | adherence |",
        "|---|---|---:|---:|---|---|---|---:|",
    ]
    for entry in built:
        frame, stats, path = entry["frame"], entry["stats"], entry["path"]
        for name, mask, block in (("serving", ~frame["is_stream"], stats["serving"]),
                                  ("stream", frame["is_stream"], stats["stream"])):
            lines.append(
                f"| `{path.name}` | {name} | {block['rows']} | {block['distinct_items']} | "
                f"{balance(frame['verdict'][mask])} | {balance(frame['oracle'][mask])} | "
                f"{balance(frame['human'][mask])} | {block['adherence']:.4f} |")
        for name, block in (("— prefix (t < onset)", stats["stream_prefix"]),
                            ("— monitored (t ≥ onset)", stats["stream_monitored"]),
                            ("— base positions", stats["stream_base"]),
                            ("— contaminated positions", stats["stream_drift"])):
            adherence = "—" if block["adherence"] is None else f"{block['adherence']:.4f}"
            lines.append(f"| `{path.name}` | {name} | {block['rows']} | "
                         f"{block['distinct_items']} | — | — | — | {adherence} |")
    lines += [
        "",
        "### The realized ramp, decile by decile",
        "",
        "",
        "| band | positions | " + " | ".join(f"`{tag}` obs (exp)" for tag, _ in VARIANTS) + " |",
        "|---|---:|" + "---:|" * len(VARIANTS),
    ]
    bands_by_tag = {entry["stats"]["schedule"]["tag"]: entry["stats"]["schedule"]["bands"]
                    for entry in built}
    for index, band in enumerate(built[0]["stats"]["schedule"]["bands"]):
        cells = []
        for tag, _ in VARIANTS:
            if tag not in bands_by_tag:
                cells.append("—")
                continue
            entry = bands_by_tag[tag][index]
            cells.append(f"{entry['observed']} ({entry['expected']:.1f})")
        lines.append(f"| {band['band']} | {band['positions']} | " + " | ".join(cells) + " |")
    lines += [
        "",
        "",
        family_overlap_sentence(serving_family_overlap(built[0]["frame"])),
        "",
        "",
        "",
        "### Inputs",
        "",
        "| file | bytes | sha256 |",
        "|---|---:|---|",
    ]
    for source, entry in manifest.items():
        lines.append(f"| `{source}` | {entry['bytes']} | `{entry['sha256']}` |")
    lines.append("")
    with open(report, "a") as stream:
        stream.write("\n".join(lines))
    return report


def main(wanted):
    OUT.mkdir(parents=True, exist_ok=True)
    partition, pools = load_partition()
    print(f"[carve] {partition['n_base_total']} base rows -> "
          f"{ {pool: len(ids) for pool, ids in pools.items()} }", flush=True)

    join = load_join()
    corpus = load_corpus()
    oracle = load_oracle()
    drift_pool = drift_pool_of(join, [row["item_id"] for row in read_jsonl(
        CORPUS / "drift_candidates" / f"{FAMILY}.jsonl")])
    base_pool = base_pool_of(pools)
    print(f"[pools] base {len(base_pool)} ({STREAM_POOL}), drift {len(drift_pool)} ({FAMILY})",
          flush=True)

    seed_of_adversarial, seeds = load_wildjailbreak_seeds()
    print(f"[wjb] {len(seed_of_adversarial)} adversarial->seed, {len(seeds)} seeds", flush=True)

    representation = np.load(PHASE_D / f"{CLASSIFIER}.Z.npy", mmap_mode="r")
    if representation.shape != (EXPECTED_CORPUS_ROWS, EXPECTED_Z_DIM):
        raise ValueError(f"Z has shape {representation.shape}, expected "
                         f"({EXPECTED_CORPUS_ROWS}, {EXPECTED_Z_DIM})")
    anchors = source_anchors(pools, join, oracle)

    built, streams = [], {}
    for tag, maximum_rate in VARIANTS:
        if wanted is not None and tag != wanted:
            continue
        schedule = linear_contamination_ramp(STREAM_LENGTH, RAMP_ONSET, maximum_rate)
        stream = compose_stream(schedule, base_pool, drift_pool, RAMP_SEED)
        streams[tag] = stream
        print(f"[ramp] {tag}: maximum_rate {maximum_rate} onset {RAMP_ONSET} length "
              f"{STREAM_LENGTH} seed {RAMP_SEED} -> {int(stream.is_drift_item.sum())} "
              f"contaminated positions (expected {schedule.sum():.1f})", flush=True)

        serving = assert_stream_is_servable(pools, stream, join, tag)
        bands = assert_realized_follows_schedule(stream, tag)
        print(f"[shape] serving {len(serving)} rows; every stream position placed; "
              f"{DECILES} decile bands inside {SIGMA_BAND:.0f} sd", flush=True)

        frame = build_ramp_frame(serving, stream, corpus, oracle, join, seed_of_adversarial,
                                 seeds, representation)
        stats = ramp_statistics(frame, tag, maximum_rate, bands)
        assert_frame_matches_stream(frame, stats, stream, anchors)

        path = frame_path(FAMILY_TAG, tag)
        print(f"[stats] {tag} {json.dumps(stats, indent=1)}", flush=True)

        sha256 = publish_after_gate(
            frame, path,
            gate=lambda staged, digest: determinism_check(
                serving, stream, schedule, base_pool, drift_pool, corpus, oracle, join,
                seed_of_adversarial, seeds, representation, digest, staged))
        print(f"[frame] {tag} determinism: straight rebuild and permuted-serving rebuild both "
              f"byte-identical", flush=True)
        print(f"[frame] wrote {path.name} ({path.stat().st_size} B) sha256={sha256}", flush=True)
        built.append({"tag": tag, "path": path, "sha256": sha256, "frame": frame, "stats": stats})

    nesting = assert_variants_nest(streams)
    print(f"[nesting] {nesting_sentence(nesting)}", flush=True)

    manifest = input_manifest(
        [CAMPAIGN / "partition_base_v31.json",
         CORPUS / "pool.jsonl", CORPUS / "drift_candidates" / f"{FAMILY}.jsonl",
         CORPUS / "judged" / "pool.judged.jsonl", CORPUS / "judged" / f"{FAMILY}.judged.jsonl",
         WILDJAILBREAK]
        + [PHASE_D / f"{CLASSIFIER}.{suffix}"
           for suffix in ("Z.npy", "arrays.npz", "items.jsonl")])
    report = append_report(built, partition, manifest, nesting)
    print(f"[report] appended to {report} sha256={digest_of(report)}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", default=None,
                        help="build one variant only (null, 0.15, 0.30); default builds all three")
    parser.add_argument("--family", default=DEFAULT_FAMILY,
                        help="the drifting family; defaults to the family every frame on disk "
                             "was built with")
    parser.add_argument("--stream-seed", type=int, default=DEFAULT_RAMP_SEED,
                        help="the composition seed — a second value is a second stream "
                             "realisation, named `_s<seed>`; defaults to the seed every frame on "
                             "disk was composed at")
    arguments = parser.parse_args()
    select_family(arguments.family)
    select_stream_seed(arguments.stream_seed)
    print(f"[build] family={FAMILY} tag={FAMILY_TAG} pool={EXPECTED_DRIFT_POOL} "
          f"variant={arguments.variant or 'all'} stream_seed={RAMP_SEED}"
          f"{' (default)' if RAMP_SEED == DEFAULT_RAMP_SEED else ''}", flush=True)
    main(arguments.variant)
