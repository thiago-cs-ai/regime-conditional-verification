"""Verify recorded payload and ID-list checksums, with optional universe membership checks.

Checksums are compared with values stored in the same manifest.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

from ._jsonl import RebuildError, read_jsonl


class ManifestVerificationError(RebuildError):
    """Manifest verification could not complete or a checked value did not match."""


def sha_ids(ids: Sequence[str]) -> str:
    """Hash IDs in order, newline-separated with no trailing newline."""
    return hashlib.sha256("\n".join(map(str, ids)).encode()).hexdigest()


def _sanitize(o: Any) -> Any:
    """Normalize NumPy values and replace non-finite floats with None for hashing."""
    if isinstance(o, dict):
        return {k: _sanitize(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_sanitize(v) for v in o]
    try:
        import numpy as np

        if isinstance(o, np.ndarray):
            return _sanitize(o.tolist())
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.bool_):
            return bool(o)
        if isinstance(o, np.floating):
            o = float(o)
    except ImportError:  # pragma: no cover - numpy is a core dependency
        pass
    if isinstance(o, float):
        return o if math.isfinite(o) else None
    return o


def canonical_payload_sha256(payload: dict[str, Any]) -> str:
    canon = json.dumps(_sanitize(payload), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canon.encode()).hexdigest()


def load_payload_verified(path: Path | str) -> dict[str, Any]:
    """Load the payload after comparing its canonical SHA-256 with the recorded checksum."""
    obj = json.loads(Path(path).read_text())
    if "payload" not in obj or "payload_sha256" not in obj:
        raise ManifestVerificationError(f"{path}: missing 'payload' or 'payload_sha256'")
    got = canonical_payload_sha256(obj["payload"])
    if got != obj["payload_sha256"]:
        raise ManifestVerificationError(
            f"Payload SHA-256 mismatch for {path} (computed {got[:12]} != recorded "
            f"{obj['payload_sha256'][:12]})"
        )
    return obj["payload"]


def iter_id_list_fields(payload: dict[str, Any]) -> Iterable[tuple[str, list[str], str]]:
    """Yield top-level ``(name, ids, recorded_sha)`` pairs.

    Match ``<name>_ids`` with ``<name>_sha256`` or ``<name>_ids_sha256``.
    Lists without a recognized string checksum are skipped.
    """
    for key, value in payload.items():
        if not key.endswith("_ids") or not isinstance(value, list):
            continue
        sha_key = key[: -len("_ids")] + "_sha256"
        alt_sha_key = key + "_sha256"
        recorded = payload.get(sha_key)
        if recorded is None:
            recorded = payload.get(alt_sha_key)
        if isinstance(recorded, str):
            yield key, value, recorded


def verify_carve_manifest(
    path: Path | str, universe_ids: set[str] | None = None
) -> dict[str, Any]:
    """Check the payload and all top-level ID-list checksums, returning per-list results.

    Require at least one ``*_ids`` field, each a list of strings with a paired checksum.
    When ``universe_ids`` is supplied, also check membership. Invalid metadata or a
    mismatch raises ``ManifestVerificationError``.
    """
    payload = load_payload_verified(path)
    id_fields = {name: ids for name, ids in payload.items() if name.endswith("_ids")}
    if not id_fields:
        raise ManifestVerificationError(f"{path}: no top-level ID lists")
    paired = list(iter_id_list_fields(payload))
    paired_names = {name for name, _, _ in paired}
    for name, ids in id_fields.items():
        if not isinstance(ids, list) or any(not isinstance(i, str) for i in ids):
            raise ManifestVerificationError(f"{path}: {name} must be a list of string IDs")
        if name not in paired_names:
            raise ManifestVerificationError(f"{path}: {name} is missing a string checksum")
    lists: dict[str, dict[str, Any]] = {}
    errors: list[str] = []
    for name, ids, recorded in paired:
        got = sha_ids(ids)
        sha_match = got == recorded
        if not sha_match:
            errors.append(f"{name}: ID-list SHA-256 mismatch (computed {got[:12]} != {recorded[:12]})")
        n_outside = None
        if universe_ids is not None:
            outside = [i for i in ids if i not in universe_ids]
            n_outside = len(outside)
            if outside:
                errors.append(f"{name}: {len(outside)} ID(s) are outside the supplied universe")
        lists[name] = {
            "n_ids": len(ids),
            "sha256": got,
            "recorded_sha256": recorded,
            "sha_match": sha_match,
            "n_outside_universe": n_outside,
        }
    if errors:
        raise ManifestVerificationError(
            f"Manifest verification failed for {path}:\n  - " + "\n  - ".join(errors)
        )
    seed = payload.get("seed")
    if seed is None:
        seed = (payload.get("config") or {}).get("seed")
    return {
        "manifest": str(path),
        "payload_sha256": canonical_payload_sha256(payload),
        "manifest_kind": payload.get("manifest_kind"),
        "seed": seed,
        "id_lists": lists,
        "verified": True,
    }


def load_universe_ids(rebuild_dir: Path | str) -> set[str]:
    """Collect IDs from available eval.jsonl, pool.jsonl, and drift_candidates/*.jsonl files."""
    rebuild_dir = Path(rebuild_dir)
    files = [rebuild_dir / "eval.jsonl", rebuild_dir / "pool.jsonl"]
    files += sorted((rebuild_dir / "drift_candidates").glob("*.jsonl"))
    present = [f for f in files if f.is_file()]
    if not present:
        raise ManifestVerificationError(f"no rebuilt corpus files found under {rebuild_dir}")
    ids: set[str] = set()
    for f in present:
        ids.update(row["item_id"] for row in read_jsonl(f))
    return ids


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Verify manifest checksums and optionally check item IDs against a rebuilt corpus."
    )
    parser.add_argument("--manifest", required=True, nargs="+", help="manifest JSON file(s)")
    parser.add_argument(
        "--universe", default=None, help="WildGuardMix rebuild directory for ID membership checks"
    )
    parser.add_argument("--report", default=None, help="optional verification-report JSON")
    args = parser.parse_args(argv)

    universe = load_universe_ids(args.universe) if args.universe else None
    reports = [verify_carve_manifest(m, universe) for m in args.manifest]
    summary = {
        "component": "rebuilders",
        "n_manifests": len(reports),
        "n_verified": sum(int(r["verified"]) for r in reports),
        "universe_dir": args.universe,
        "universe_n": len(universe) if universe is not None else None,
        "manifests": reports,
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    if args.report:
        p = Path(args.report)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(summary, indent=2) + "\n")
    for r in reports:
        lists = ", ".join(f"{k}({v['n_ids']})" for k, v in r["id_lists"].items())
        print(f"[rcv.rebuild.manifests] Recorded checksums match for {r['manifest']}: {lists}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
