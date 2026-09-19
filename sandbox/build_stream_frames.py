
"""Run-once integration build: the first REPLAYED DRIFT-CAMPAIGN frame for Llama-Guard-3."""

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

RCV = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RCV / "src"))

from rcv.frame_building import (  # noqa: E402
    digest_of,
    family_codes,
    input_manifest,
    normalised_key,
    save_frame,
)

REPO = Path("$RCV_DATA")
RUN = REPO / "results" / "wgmix_secondbench_drift_20260703"
PHASE_D = RUN / "phase_d" / "joined"
CORPUS = RUN / "data"
CAMPAIGN = (RUN / "phase_f" / "enext" / "spike_runs_20260705b" / "S-E_privinfo_exhaustion"
            / "manifests")
WILDJAILBREAK = Path(
    "$SCRATCH/HuggingFace/huggingface/hub/"
    "datasets--allenai--wildjailbreak/snapshots/"
    "5ddc12a7894f842b0619b8e1c7ee496b198af009/train/train.tsv")

OUT = RCV / "frames"
CLASSIFIER = "lg3"
FAMILY = "private_information_individual"
DEFAULT_CAMPAIGN_TAG = "ep1-w1"
SERVING_POOLS = ("P0-train", "P0-cal", "RETENTION")
STREAM_POOL = "TRAFFIC"
EXPECTED_CORPUS_ROWS = 20199
EXPECTED_Z_DIM = 4096


def read_jsonl(path):
    with open(path) as stream:
        return [json.loads(line) for line in stream if line.strip()]


def sha_ids(ids):
    """The E-NEXT id digest, reimplemented so the manifest's claim about itself is checked, not read (`_f_common.sha_ids`)."""
    return hashlib.sha256("\n".join(map(str, ids)).encode()).hexdigest()


def load_manifest(path):
    """The Phase-E envelope {payload, payload_sha256, generated_utc}."""
    envelope = json.loads(Path(path).read_text())
    payload = envelope["payload"]
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    recomputed = hashlib.sha256(canonical.encode()).hexdigest()
    if recomputed != envelope["payload_sha256"]:
        raise ValueError(f"{path}: payload digest {recomputed[:12]}… does not match the "
                         f"envelope's {envelope['payload_sha256'][:12]}…; the manifest is not "
                         f"the manifest it says it is")
    return payload


def load_partition():
    partition = load_manifest(CAMPAIGN / "partition_base_v31.json")
    for pool, ids in partition["pools"].items():
        if sha_ids(ids) != partition["pool_sha256"][pool]:
            raise ValueError(f"base partition: pool {pool!r} does not hash to its stated "
                             f"pool_sha256")
    pools = {pool: list(map(str, ids)) for pool, ids in partition["pools"].items()}
    everywhere = [item for ids in pools.values() for item in ids]
    if len(set(everywhere)) != len(everywhere):
        raise ValueError("base partition: the four pools are not disjoint")
    return partition, pools


def load_stream(tag):
    stream = load_manifest(CAMPAIGN / f"stream_{tag}.json")
    order = list(map(str, stream["order_item_ids"]))
    is_drift = np.asarray(stream["is_drift"], dtype=np.int64)
    if sha_ids(order) != stream["order_sha256"]:
        raise ValueError(f"stream {tag}: the recorded order does not hash to its stated "
                         f"order_sha256; the replay would not be the recorded campaign")
    if len(order) != stream["length"] or len(is_drift) != stream["length"]:
        raise ValueError(f"stream {tag}: {len(order)} positions against a stated length of "
                         f"{stream['length']}")
    if int(is_drift.sum()) != stream["n_drift"]:
        raise ValueError(f"stream {tag}: {int(is_drift.sum())} drift positions against a stated "
                         f"n_drift of {stream['n_drift']}")
    realized = int(is_drift.sum()) / len(order)
    if abs(realized - stream["realized_dose"]) > 1e-12:
        raise ValueError(f"stream {tag}: realized dose {realized} against a stated "
                         f"{stream['realized_dose']}")
    families = [entry["family"] for entry in stream.get("families", [])]
    if families != [FAMILY]:
        raise ValueError(f"stream {tag}: drift families {families} — this build is scoped to the "
                         f"single-family ladder over {FAMILY!r}")
    if int(is_drift[:stream["t0"]].sum()):
        raise ValueError(f"stream {tag}: a drift position before t0={stream['t0']}; the recorded "
                         f"campaign declares the run-in drift-free")
    return stream, order, is_drift


def load_join():
    """The frozen phase-D join: one row per item, and the arrays positional against it."""
    items = read_jsonl(PHASE_D / f"{CLASSIFIER}.items.jsonl")
    if len(items) != EXPECTED_CORPUS_ROWS:
        raise ValueError(f"the join carries {len(items)} rows, expected {EXPECTED_CORPUS_ROWS}")
    item_id = [row["item_id"] for row in items]
    if len(set(item_id)) != len(item_id):
        raise ValueError("the join repeats an item id; a by-item join would be ambiguous")
    arrays = np.load(PHASE_D / f"{CLASSIFIER}.arrays.npz")
    verdict = arrays["verdict"].astype(np.int64)
    ystar = arrays["ystar"].astype(np.int64)
    if not np.isin(verdict, (0, 1)).all() or not np.isin(ystar, (0, 1)).all():
        raise ValueError("the stored verdict or ystar column carries a value other than 0 and 1")
    stored_agreement = arrays["agreement"].astype(np.int64)
    if not np.array_equal(stored_agreement, (verdict == ystar).astype(np.int64)):
        raise ValueError("the stored agreement column disagrees with int(verdict == ystar); "
                         "correctness has one home")
    return {"row_of": {item: row for row, item in enumerate(item_id)},
            "set_id": {row["item_id"]: row["set_id"] for row in items},
            "verdict": verdict, "ystar": ystar}


def load_corpus():
    """item_id -> prompt and the corpus's own human harm annotation, over the two slices this campaign touches: the base pool and the drifting family."""
    rows = {}
    for path in (CORPUS / "pool.jsonl",
                 CORPUS / "drift_candidates" / f"{FAMILY}.jsonl"):
        for row in read_jsonl(path):
            if int(row["ground_truth"]) != int(row["ai2_response_harm"]):
                raise ValueError(f"{row['item_id']}: ground_truth disagrees with "
                                 f"ai2_response_harm; the human column's identity is not what "
                                 f"this build assumed")
            if row["item_id"] in rows:
                raise ValueError(f"{row['item_id']}: appears in more than one corpus slice")
            rows[row["item_id"]] = {"prompt": row["prompt"],
                                    "human": int(row["ground_truth"])}
    return rows


def load_oracle():
    """item_id -> Y*, re-read from the judged files."""
    oracle = {}
    for path in (CORPUS / "judged" / "pool.judged.jsonl",
                 CORPUS / "judged" / f"{FAMILY}.judged.jsonl"):
        for row in read_jsonl(path):
            if row.get("new_label") is None:
                continue
            label = int(row["new_label"])
            standing = oracle.get(row["item_id"])
            if standing is not None and standing != label:
                raise ValueError(f"{row['item_id']}: the judged file supplies two disagreeing "
                                 f"labels ({standing} then {label}); last-OK-wins would be "
                                 f"choosing between them silently")
            oracle[row["item_id"]] = label
    return oracle


def load_wildjailbreak_seeds():
    """The exact-provenance branch of the family key, read from the same blob the EVAL builds read, so the grouping rule is one rule across every frame in this tree."""
    csv.field_size_limit(1 << 31)
    seed_of_adversarial, seeds = {}, set()
    with open(WILDJAILBREAK, newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream, delimiter="\t"):
            vanilla = (row["vanilla"] or "").strip()
            adversarial = (row["adversarial"] or "").strip()
            if not vanilla:
                continue
            seed = normalised_key(vanilla)
            seeds.add(seed)
            if adversarial:
                seed_of_adversarial[normalised_key(adversarial)] = seed
    return seed_of_adversarial, seeds


def seed_for(prompt, seed_of_adversarial, seeds):
    key = normalised_key(prompt)
    if key in seed_of_adversarial:
        return seed_of_adversarial[key]
    if key in seeds:
        return key
    return None


def assert_campaign_shape(pools, order, is_drift, join, tag):
    """Every id the replay names is one this corpus can serve, and every position is the kind of position the manifest says it is."""
    serving = sorted({item for pool in SERVING_POOLS for item in pools[pool]})
    traffic = set(pools[STREAM_POOL])

    unknown = sorted({item for item in order + serving if item not in join["row_of"]})
    if unknown:
        raise ValueError(f"{len(unknown)} campaign ids are absent from the phase-D join "
                         f"(first {unknown[0]!r}); the replay would be joining two corpora")

    drift_set_id = f"drift:{FAMILY}"
    for position, (item, drifting) in enumerate(zip(order, is_drift)):
        expected = drift_set_id if drifting else "pool"
        if join["set_id"][item] != expected:
            raise ValueError(f"stream {tag} position {position}: item {item!r} is "
                             f"{join['set_id'][item]!r} but the manifest flags it "
                             f"{'drift' if drifting else 'base'}")
        if not drifting and item not in traffic:
            raise ValueError(f"stream {tag} position {position}: base item {item!r} is outside "
                             f"the {STREAM_POOL} pool; serving would not be drift-free-and-unseen")
    if set(serving) & set(order):
        raise ValueError("serving and the stream share an item; the block the estimator is fitted "
                         "on would also be traffic it is monitored over")
    for item in serving:
        if join["set_id"][item] != "pool":
            raise ValueError(f"serving item {item!r} is {join['set_id'][item]!r}, not base pool")
    return serving


def block_rows(item_ids, is_stream, corpus, oracle, join):
    """One block's columns, joined by item."""
    missing = sorted({item for item in item_ids if item not in corpus})
    if missing:
        raise ValueError(f"{len(missing)} rows have no corpus text (first {missing[0]!r})")
    unlabelled = sorted({item for item in item_ids if item not in oracle})
    if unlabelled:
        raise ValueError(f"{len(unlabelled)} rows have no oracle label "
                         f"(first {unlabelled[0]!r})")

    rows = np.array([join["row_of"][item] for item in item_ids])
    frozen_ystar = join["ystar"][rows]
    judged = np.array([oracle[item] for item in item_ids], dtype=np.int64)
    disagreeing = np.flatnonzero(frozen_ystar != judged)
    if len(disagreeing):
        first = item_ids[int(disagreeing[0])]
        raise ValueError(f"the frozen ystar column disagrees with the judged file on "
                         f"{len(disagreeing)} rows (first {first!r}); the oracle has one value")
    return {"item_id": np.array(item_ids),
            "join_row": rows,
            "verdict": join["verdict"][rows],
            "oracle": judged,
            "human": np.array([corpus[item]["human"] for item in item_ids], dtype=np.int64),
            "prompt": [corpus[item]["prompt"] for item in item_ids],
            "is_stream": np.full(len(item_ids), is_stream, dtype=np.bool_)}


def build_campaign_frame(serving, order, corpus, oracle, join, seed_of_adversarial, seeds,
                         representation):
    """Serving first, in canonical item order; then the stream, in the order it was recorded."""
    blocks = [block_rows(sorted(serving), False, corpus, oracle, join),
              block_rows(order, True, corpus, oracle, join)]
    prompt = [text for block in blocks for text in block["prompt"]]
    seed = [seed_for(text, seed_of_adversarial, seeds) for text in prompt]
    join_rows = np.concatenate([block["join_row"] for block in blocks])

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
    }


def source_anchors(pools, join, oracle):
    """The serving block's verdict and oracle balances, recomputed straight off the carve and the judged files, so the frame's own columns are checked against a second derivation."""
    serving = sorted({item for pool in SERVING_POOLS for item in pools[pool]})
    rows = np.array([join["row_of"][item] for item in serving])
    verdict = join["verdict"][rows]
    ystar = np.array([oracle[item] for item in serving], dtype=np.int64)
    per_pool = {}
    for pool in SERVING_POOLS:
        pool_rows = np.array([join["row_of"][item] for item in pools[pool]])
        pool_ystar = np.array([oracle[item] for item in pools[pool]], dtype=np.int64)
        per_pool[pool] = {"rows": len(pool_rows),
                          "verdict_unsafe": int(join["verdict"][pool_rows].sum()),
                          "oracle_unsafe": int(pool_ystar.sum()),
                          "adherence": float((join["verdict"][pool_rows] == pool_ystar).mean())}
    return {"rows": len(serving), "verdict_unsafe": int(verdict.sum()),
            "oracle_unsafe": int(ystar.sum()),
            "adherence": float((verdict == ystar).mean()), "per_pool": per_pool}


def frame_statistics(frame, stream, is_drift):
    serving = ~frame["is_stream"]
    traffic = frame["is_stream"]
    drifting = np.zeros(len(frame["is_stream"]), dtype=bool)
    drifting[np.flatnonzero(traffic)] = is_drift.astype(bool)

    def slice_stats(mask):
        verdict, oracle = frame["verdict"][mask], frame["oracle"][mask]
        agreement = (verdict == oracle)
        return {"rows": int(mask.sum()),
                "distinct_items": int(len(np.unique(frame["item_id"][mask]))),
                "verdict_unsafe": int(verdict.sum()),
                "oracle_unsafe": int(oracle.sum()),
                "human_unsafe": int(frame["human"][mask].sum()),
                "adherence": float(agreement.mean()),
                "cases": {f"verdict{v}_agreement{a}":
                          int(((verdict == v) & (agreement == bool(a))).sum())
                          for v in (0, 1) for a in (0, 1)}}

    _, sizes = np.unique(frame["family"][serving], return_counts=True)
    return {
        "serving": slice_stats(serving),
        "stream": slice_stats(traffic),
        "stream_base": slice_stats(traffic & ~drifting),
        "stream_drift": slice_stats(traffic & drifting),
        "serving_families": {"families": int(len(sizes)), "largest": int(sizes.max()),
                             "rows_in_a_multi_row_family": int(sizes[sizes > 1].sum())},
        "families_total": int(len(np.unique(frame["family"]))),
        "manifest": {"length": stream["length"], "n_drift": stream["n_drift"],
                     "t0": stream["t0"], "dose": stream["total_dose"],
                     "realized_dose": stream["realized_dose"], "seed": stream["seed"],
                     "order_sha256": stream["order_sha256"]},
    }


def assert_frame_matches_manifest(frame, stats, stream, anchors):
    traffic = int(frame["is_stream"].sum())
    if traffic != stream["length"]:
        raise ValueError(f"the frame carries {traffic} stream rows against a manifest length of "
                         f"{stream['length']}")
    if stats["stream_drift"]["rows"] != stream["n_drift"]:
        raise ValueError(f"the frame carries {stats['stream_drift']['rows']} drift rows against "
                         f"a manifest n_drift of {stream['n_drift']}")
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


def determinism_check(serving, order, corpus, oracle, join, seed_of_adversarial, seeds,
                      representation, shipped_digest, path):
    """The digest means something only if a rebuild reproduces it."""
    scratch = path.with_name(path.name + ".determinism-check")
    if scratch.exists():
        raise FileExistsError(f"{scratch} is left over from an earlier check")
    shuffled = list(np.random.default_rng(11).permutation(np.array(serving)))
    rebuilt = save_frame(build_campaign_frame(shuffled, order, corpus, oracle, join,
                                              seed_of_adversarial, seeds, representation),
                         scratch)
    scratch.unlink()
    if rebuilt != shipped_digest:
        raise ValueError(f"a rebuild from a permuted serving list digests {rebuilt}, not "
                         f"{shipped_digest}; the frame remembers its input order")
    return rebuilt


def balance(column):
    counts = np.bincount(np.asarray(column, dtype=np.int64), minlength=2)
    return f"0={int(counts[0])} / 1={int(counts[1])}"


def append_report(tag, path, sha256, frame, stats, anchors, partition, manifest):
    report = OUT / "BUILD_REPORT.md"
    serving, stream_stats = stats["serving"], stats["stream"]
    lines = [
        "",
        "---",
        "",
        f"## Replayed drift-campaign frame — {FAMILY} @ dose "
        f"{stats['manifest']['dose']}",
        "",
        "",
        "| block | rows | distinct items | verdict | oracle | human | adherence |",
        "|---|---:|---:|---|---|---|---:|",
        f"| serving (`is_stream` False) | {serving['rows']} | {serving['distinct_items']} | "
        f"{balance(frame['verdict'][~frame['is_stream']])} | "
        f"{balance(frame['oracle'][~frame['is_stream']])} | "
        f"{balance(frame['human'][~frame['is_stream']])} | {serving['adherence']:.4f} |",
        f"| stream (`is_stream` True) | {stream_stats['rows']} | "
        f"{stream_stats['distinct_items']} | "
        f"{balance(frame['verdict'][frame['is_stream']])} | "
        f"{balance(frame['oracle'][frame['is_stream']])} | "
        f"{balance(frame['human'][frame['is_stream']])} | {stream_stats['adherence']:.4f} |",
        f"| — stream, base positions | {stats['stream_base']['rows']} | "
        f"{stats['stream_base']['distinct_items']} | — | — | — | "
        f"{stats['stream_base']['adherence']:.4f} |",
        f"| — stream, drift positions | {stats['stream_drift']['rows']} | "
        f"{stats['stream_drift']['distinct_items']} | — | — | — | "
        f"{stats['stream_drift']['adherence']:.4f} |",
        "",
        f"`{path.name}` — {path.stat().st_size} bytes, sha256 `{sha256}`. "
        f"Representation float32, {EXPECTED_Z_DIM}-dimensional. Polarity throughout: "
        f"1 = UNSAFE.",
        "",
        "### How serving and stream map onto the old carve",
        "",
        "",
        "| old pool | rows | role there | role here |",
        "|---|---:|---|---|",
    ]
    for pool in SERVING_POOLS:
        old_role = {"P0-train": "deployer's probe TRAIN base at t0",
                    "P0-cal": "probe CAL base at t0 (Platt anchor)",
                    "RETENTION": "era-invariant ruler; never streamed, never trained"}[pool]
        lines.append(f"| `{pool}` | {anchors['per_pool'][pool]['rows']} | {old_role} | "
                     f"serving block |")
    lines += [
        f"| `{STREAM_POOL}` | {len(partition['pools'][STREAM_POOL])} | the streamed base | "
        f"stream block, at the positions the manifest flags base |",
        "",
        "",
        "",
        "| pool | rows | verdict unsafe | oracle unsafe | adherence |",
        "|---|---:|---:|---:|---:|",
    ]
    for pool in SERVING_POOLS:
        entry = anchors["per_pool"][pool]
        lines.append(f"| `{pool}` | {entry['rows']} | {entry['verdict_unsafe']} | "
                     f"{entry['oracle_unsafe']} | {entry['adherence']:.4f} |")
    lines += [
        f"| **serving (union)** | {anchors['rows']} | {anchors['verdict_unsafe']} | "
        f"{anchors['oracle_unsafe']} | {anchors['adherence']:.4f} |",
        "",
        "",
        "",
        "",
        "",
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


def main(tag):
    OUT.mkdir(parents=True, exist_ok=True)
    partition, pools = load_partition()
    print(f"[carve] {partition['n_base_total']} base rows -> "
          f"{ {pool: len(ids) for pool, ids in pools.items()} }", flush=True)

    stream, order, is_drift = load_stream(tag)
    print(f"[stream] {tag}: length {stream['length']} dose {stream['total_dose']} "
          f"t0 {stream['t0']} n_drift {stream['n_drift']} seed {stream['seed']}", flush=True)

    join = load_join()
    serving = assert_campaign_shape(pools, order, is_drift, join, tag)
    print(f"[shape] serving {len(serving)} rows; every stream position placed", flush=True)

    corpus = load_corpus()
    oracle = load_oracle()
    seed_of_adversarial, seeds = load_wildjailbreak_seeds()
    print(f"[wjb] {len(seed_of_adversarial)} adversarial->seed, {len(seeds)} seeds", flush=True)

    representation = np.load(PHASE_D / f"{CLASSIFIER}.Z.npy", mmap_mode="r")
    if representation.shape != (EXPECTED_CORPUS_ROWS, EXPECTED_Z_DIM):
        raise ValueError(f"Z has shape {representation.shape}, expected "
                         f"({EXPECTED_CORPUS_ROWS}, {EXPECTED_Z_DIM})")

    frame = build_campaign_frame(serving, order, corpus, oracle, join, seed_of_adversarial,
                                 seeds, representation)
    anchors = source_anchors(pools, join, oracle)
    stats = frame_statistics(frame, stream, is_drift)
    assert_frame_matches_manifest(frame, stats, stream, anchors)

    path = OUT / f"stream_{CLASSIFIER}_{FAMILY}_{stream['total_dose']}.npz"
    sha256 = save_frame(frame, path)
    print(f"[frame] wrote {path.name} ({path.stat().st_size} B) sha256={sha256}", flush=True)
    print(f"[stats] {json.dumps(stats, indent=1)}", flush=True)

    determinism_check(serving, order, corpus, oracle, join, seed_of_adversarial, seeds,
                      representation, sha256, path)
    print("[frame] determinism: a permuted-serving rebuild is byte-identical", flush=True)

    manifest = input_manifest(
        [CAMPAIGN / "partition_base_v31.json", CAMPAIGN / f"stream_{tag}.json",
         CORPUS / "pool.jsonl", CORPUS / "drift_candidates" / f"{FAMILY}.jsonl",
         CORPUS / "judged" / "pool.judged.jsonl", CORPUS / "judged" / f"{FAMILY}.judged.jsonl",
         WILDJAILBREAK]
        + [PHASE_D / f"{CLASSIFIER}.{suffix}"
           for suffix in ("Z.npy", "arrays.npz", "items.jsonl")])
    report = append_report(tag, path, sha256, frame, stats, anchors, partition, manifest)
    print(f"[report] appended to {report} sha256={digest_of(report)}", flush=True)


SPIKE_ROOT = RUN / "phase_f" / "enext" / "spike_runs_20260705b"
DEFAULT_SPIKE = "S-E_privinfo_exhaustion"

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", default=DEFAULT_CAMPAIGN_TAG,
                        help="rung of the dose ladder (ep1-w1 … ep5-w1)")
    parser.add_argument("--spike", default=DEFAULT_SPIKE,
                        help="campaign directory under spike_runs_20260705b")
    parser.add_argument("--family", default=FAMILY,
                        help="the drifting family; must be the manifest's only one")
    arguments = parser.parse_args()
    CAMPAIGN = SPIKE_ROOT / arguments.spike / "manifests"
    FAMILY = arguments.family
    if not CAMPAIGN.is_dir():
        raise SystemExit(f"no manifests at {CAMPAIGN}")
    for required in (CORPUS / "drift_candidates" / f"{FAMILY}.jsonl",
                     CORPUS / "judged" / f"{FAMILY}.judged.jsonl"):
        if not required.exists():
            raise SystemExit(f"the family {FAMILY!r} has no {required.name}; a frame cannot be "
                             f"built for a family the corpus never judged")
    print(f"[build] spike={arguments.spike} family={FAMILY} rung={arguments.campaign}", flush=True)
    main(arguments.campaign)
