"""Generate policy labels through OpenRouter using the frozen rubric or --policy-file.

Sends input prompts and responses to OpenRouter; requires OPENROUTER_API_KEY.
Appends labels (1=unsafe, 0=safe) and judge diagnostics to JSONL. On resume, skips
item IDs with non-null labels. Generated labels may differ from the published labels.
Exits with status 1 if any processed item remains unlabeled after a failure.
Exits with status 2 if cost stopping leaves unprocessed items and no labeling failures.
PKU-SafeRLHF inputs remain subject to CC-BY-NC-4.0.

    export OPENROUTER_API_KEY=...
    python -m rcv.rebuild.gen_rubric_ystar --input native_pku_test.jsonl --limit 1 --out ystar.jsonl
"""

from __future__ import annotations

import argparse
import json
import math
import os
import threading
import time
import urllib.error
import urllib.request
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from ._labeler_prompt import build_messages
from ._policy_prompt import (
    CUSTOM_PROMPT_VERSION,
    custom_messages,
    input_sha256,
    policy_sha256,
    read_policy,
)

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
MODEL = "openai/gpt-5-nano"
PROMPT_VERSION = "v1"

# Fixed USD-per-token rates used for every --model cost estimate.
NANO_PRICE_IN = 0.05 / 1_000_000
NANO_PRICE_OUT = 0.40 / 1_000_000

VALID_VERDICTS = {"SAFE", "UNSAFE"}
VALID_RULES = {
    "2.1_csam", "2.1_violence", "2.1_glorification", "2.1_crime", "2.1_hate",
    "2.1_sexual", "2.1_selfharm", "2.1_misinfo", "2.1_deception", "2.1_privacy",
    "2.2_commitment", "none",
    "2.1_content",  # Retain the legacy aggregate tag when reading older judge replies.
}
RETRYABLE_STATUS = {429, 500, 502, 503, 504}


class YStarKeyError(RuntimeError):
    """The OpenRouter API key is not available in the environment."""


def load_api_key() -> str:
    """Read a nonblank key from ``OPENROUTER_API_KEY`` or raise ``YStarKeyError``."""
    key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    if not key:
        raise YStarKeyError(
            "Set OPENROUTER_API_KEY to your OpenRouter API key."
        )
    return key


def _post(payload: dict[str, Any], api_key: str, timeout: int = 90) -> dict[str, Any]:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(OPENROUTER_URL, data=data, method="POST")  # noqa: S310 — https literal
    req.add_header("Authorization", f"Bearer {api_key}")
    req.add_header("Content-Type", "application/json")
    req.add_header("X-Title", "RCV Y-star labeler")
    with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310 — https literal
        return json.loads(resp.read().decode("utf-8"))


def parse_verdict(raw: str) -> dict[str, str] | None:
    """Normalize a SAFE/UNSAFE reply, returning None if no valid verdict is parsed.

    SAFE verdicts and tags outside ``VALID_RULES`` receive ``rule_fired='none'``.
    """
    import re

    if not raw or not raw.strip():
        return None
    text = raw.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\s*", "", text)
        text = re.sub(r"\s*```$", "", text).strip()
    obj: Any = None
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            try:
                obj = json.loads(match.group(0))
            except json.JSONDecodeError:
                return None
    if not isinstance(obj, dict):
        return None
    verdict = str(obj.get("verdict", "")).strip().upper()
    if verdict not in VALID_VERDICTS:
        return None
    rule = str(obj.get("rule_fired", "none")).strip()
    if rule not in VALID_RULES:
        rule = "none"
    if verdict == "SAFE":
        rule = "none"
    reason = str(obj.get("reason", "")).strip().replace("\n", " ")[:300]
    return {"verdict": verdict, "rule_fired": rule, "reason": reason}


def extract_raw_rule(raw: str) -> str | None:
    """Extract ``rule_fired`` as a stripped string before vocabulary normalization."""
    import re

    if not raw or not raw.strip():
        return None
    text = raw.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\s*", "", text)
        text = re.sub(r"\s*```$", "", text).strip()
    obj: Any = None
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if m:
            try:
                obj = json.loads(m.group(0))
            except json.JSONDecodeError:
                return None
    if not isinstance(obj, dict):
        return None
    rf = obj.get("rule_fired")
    return str(rf).strip() if rf is not None else None


def label_real(
    prompt: str, response: str, api_key: str, model: str, max_tokens: int, max_attempts: int = 5,
    *, policy: str | None = None,
) -> tuple[str, int, int]:
    """Return reply text and prompt/completion token counts from a successful request.

    Retries selected HTTP and network errors up to ``max_attempts`` total requests.
    HTTP 400/422 disables JSON mode for subsequent attempts. Temperature is omitted.
    """
    messages = (build_messages(prompt, response) if policy is None
                else custom_messages(policy, prompt, response))
    base = {"model": model, "messages": messages, "max_tokens": max_tokens}
    use_json, last = True, None
    for attempt in range(max_attempts):
        payload = dict(base)
        if use_json:
            payload["response_format"] = {"type": "json_object"}
        try:
            body = _post(payload, api_key)
            choice = body["choices"][0]["message"]["content"]
            usage = body.get("usage", {}) or {}
            return (
                choice,
                int(usage.get("prompt_tokens", 0) or 0),
                int(usage.get("completion_tokens", 0) or 0),
            )
        except urllib.error.HTTPError as e:
            status = e.code
            try:
                eb = e.read().decode("utf-8", "ignore")
            except Exception:  # noqa: BLE001
                eb = ""
            last = f"HTTP {status}: {eb[:200]}"
            if status in (400, 422) and use_json:
                use_json = False
                continue
            if status in RETRYABLE_STATUS:
                time.sleep(min(2**attempt, 30))
                continue
            raise RuntimeError(last) from e
        except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
            last = f"Network error: {e}"
            time.sleep(min(2**attempt, 30))
    raise RuntimeError(f"Label request attempts exhausted: {last}")


def record_metadata(item, model, policy):
    if policy is None:
        return {"model": model, "prompt_version": PROMPT_VERSION}
    return {"model": model, "prompt_version": CUSTOM_PROMPT_VERSION,
            "policy_sha256": policy_sha256(policy), "input_sha256": input_sha256(item)}


def label_one(item: dict[str, Any], api_key: str, model: str, max_tokens: int,
              *, policy: str | None = None) -> dict[str, Any]:
    """Label one item, retrying once if verdict parsing fails.

    Request or repeated parse failures yield a null label and ``PARSE_ERROR``.
    Token counts sum usage reported by completed responses, including parse retries.
    """
    raw, pin, pout = "", 0, 0
    failure_reason = ""
    policy_args = {} if policy is None else {"policy": policy}
    metadata = record_metadata(item, model, policy)
    for _ in range(2):
        try:
            raw, attempt_in, attempt_out = label_real(
                item["prompt"], item["response"], api_key, model, max_tokens, **policy_args
            )
        except Exception as exc:  # noqa: BLE001
            failure_reason = f"Labeling failed: {exc}"
            break
        pin += attempt_in
        pout += attempt_out
        p = parse_verdict(raw)
        if p is not None:
            nl = 1 if p["verdict"] == "UNSAFE" else 0
            return {
                "item_id": item["item_id"], "rubric_ystar": nl, "verdict": p["verdict"],
                "rule_fired": p["rule_fired"] if policy is None else "none",
                "rule_fired_raw": extract_raw_rule(raw),
                "reason": p.get("reason", ""), "prompt_tokens": pin, "completion_tokens": pout,
                **metadata,
            }
    return {
        "item_id": item["item_id"], "rubric_ystar": None, "verdict": "PARSE_ERROR",
        "rule_fired": "none", "rule_fired_raw": extract_raw_rule(raw), "reason": failure_reason,
        "prompt_tokens": pin, "completion_tokens": pout, **metadata,
    }


def load_items(input_path: Path, limit: int | None = None) -> list[dict[str, Any]]:
    """Read a nonempty input prefix with unique string IDs and string prompt/response fields."""
    if limit is not None and limit <= 0:
        raise ValueError("limit must be positive")
    items: list[dict[str, Any]] = []
    seen: set[str] = set()
    with open(input_path, encoding="utf-8") as fh:
        for line_number, line in enumerate(fh, 1):
            if limit is not None and len(items) >= limit:
                break
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if not isinstance(r, dict):
                raise ValueError(f"{input_path}:{line_number}: expected a JSON object")
            iid = r.get("item_id")
            if not isinstance(iid, str) or not iid.strip():
                raise ValueError(f"{input_path}:{line_number}: item_id must be a nonempty string")
            if iid in seen:
                raise ValueError(f"{input_path}:{line_number}: duplicate item_id {iid!r}")
            for field in ("prompt", "response"):
                if not isinstance(r.get(field), str):
                    raise ValueError(f"{input_path}:{line_number}: {field} must be a string")
            seen.add(iid)
            items.append({"item_id": r["item_id"], "prompt": r["prompt"], "response": r["response"]})
    if not items:
        raise ValueError(f"{input_path}: input is empty")
    return items


def load_done(out_path: Path, *, model: str = MODEL, policy: str | None = None,
              items: Sequence[dict[str, Any]] | None = None) -> set[str]:
    """Check resume identity and return completed IDs; skip malformed JSON lines.

    Custom policies require input items and validate failed records as well as successes.
    """
    if policy is not None and items is None:
        raise ValueError("Custom-policy resume requires input items.")
    expected = {"model": model, "prompt_version": PROMPT_VERSION}
    input_hashes = {}
    if policy is not None:
        expected.update(prompt_version=CUSTOM_PROMPT_VERSION, policy_sha256=policy_sha256(policy))
        input_hashes = {item["item_id"]: input_sha256(item) for item in items}
    ok: set[str] = set()
    if not out_path.exists():
        return ok
    with out_path.open(encoding="utf-8") as fh:
        for line_number, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(r, dict):
                raise ValueError(f"{out_path}:{line_number}: expected a JSON object")
            if (policy is not None or r.get("rubric_ystar") is not None
                    or r.get("prompt_version") == CUSTOM_PROMPT_VERSION or "policy_sha256" in r):
                identity = dict(expected)
                if r.get("item_id") in input_hashes:
                    identity["input_sha256"] = input_hashes[r["item_id"]]
                for field, value in identity.items():
                    if r.get(field) != value:
                        raise ValueError(
                            f"{out_path}:{line_number}: {field}={r.get(field)!r}; "
                            f"expected {value!r}. Use a new output path."
                        )
            if r.get("rubric_ystar") is not None:
                ok.add(r["item_id"])
    return ok


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True,
                    help="JSONL with item_id, prompt, and response; text is sent to OpenRouter")
    ap.add_argument("--out", required=True, help="output jsonl (append-only, resumable)")
    ap.add_argument("--policy-file", help="UTF-8 policy; omit to use the frozen v1 rubric")
    ap.add_argument("--limit", type=int, default=None,
                    help="read the first N input rows before skipping previously labeled IDs")
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--max-tokens", type=int, default=8192)
    ap.add_argument("--progress-every", type=int, default=200)
    ap.add_argument("--max-cost", type=float, default=None,
                    help="stop after a batch exceeds this USD estimate; fixed GPT-5-nano rates")
    args = ap.parse_args(argv)

    for flag in ("limit", "concurrency", "max_tokens", "progress_every"):
        value = getattr(args, flag)
        if value is not None and value <= 0:
            ap.error(f"--{flag.replace('_', '-')} must be positive")
    if args.max_cost is not None and (not math.isfinite(args.max_cost) or args.max_cost <= 0):
        ap.error("--max-cost must be finite and positive")

    policy = None
    if args.policy_file:
        try:
            policy = read_policy(args.policy_file)
        except (OSError, UnicodeError, ValueError) as exc:
            ap.error(str(exc))
    policy_args = {} if policy is None else {"policy": policy}
    items = load_items(Path(args.input), args.limit)
    out_path = Path(os.path.abspath(args.out))
    done = load_done(out_path, model=args.model, items=items, **policy_args)
    todo = [it for it in items if it["item_id"] not in done]
    print(f"Input rows: {len(items)} | labeled IDs in output: {len(done)} | pending: {len(todo)} | "
          f"model {args.model}, temperature omitted | rubric "
          f"{PROMPT_VERSION if policy is None else CUSTOM_PROMPT_VERSION}", flush=True)
    if not todo:
        print("No pending items.")
        return 0

    api_key = load_api_key()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
        with open(out_path, "rb") as _f:
            _f.seek(-1, os.SEEK_END)
            if _f.read(1) != b"\n":
                with open(out_path, "a") as _a:
                    _a.write("\n")

    lock = threading.Lock()
    t0 = time.monotonic()
    stats = {"done": 0, "fail": 0, "tin": 0, "tout": 0}
    fh = open(out_path, "a")

    def est_cost() -> float:
        return stats["tin"] * NANO_PRICE_IN + stats["tout"] * NANO_PRICE_OUT

    def emit(rec: dict[str, Any]) -> None:
        with lock:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fh.flush()
            os.fsync(fh.fileno())
            stats["done"] += 1
            if rec["rubric_ystar"] is None:
                stats["fail"] += 1
            stats["tin"] += rec.get("prompt_tokens", 0) or 0
            stats["tout"] += rec.get("completion_tokens", 0) or 0
            if stats["done"] % args.progress_every == 0 or stats["done"] == len(todo):
                el = time.monotonic() - t0
                rate = stats["done"] / el if el > 0 else 0
                eta = (len(todo) - stats["done"]) / rate if rate > 0 else 0
                print(f"  {stats['done']}/{len(todo)} records  without_label={stats['fail']}  "
                      f"estimated cost=${est_cost():.3f}  {el:.0f}s  ETA {eta:.0f}s", flush=True)

    stopped = False
    try:
        batch = max(args.concurrency * 8, args.concurrency)
        for start in range(0, len(todo), batch):
            with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
                futs = {ex.submit(label_one, it, api_key, args.model, args.max_tokens,
                                  **policy_args): it
                        for it in todo[start:start + batch]}
                for fut in as_completed(futs):
                    try:
                        rec = fut.result()
                    except Exception as e:  # noqa: BLE001
                        it = futs[fut]
                        rec = {"item_id": it["item_id"], "rubric_ystar": None,
                               "verdict": "PARSE_ERROR", "rule_fired": "none",
                               "rule_fired_raw": None, "reason": f"Labeling failed: {e}",
                               "prompt_tokens": 0, "completion_tokens": 0,
                               **record_metadata(it, args.model, policy)}
                    emit(rec)
            if args.max_cost is not None and est_cost() > args.max_cost:
                stopped = True
                break
    finally:
        fh.close()
    if stopped:
        print(f"Stopped after batch: cost threshold ${args.max_cost} exceeded "
              f"(estimate ${est_cost():.3f}).", flush=True)
    pending = len(todo) - stats["done"]
    print(f"Wrote {stats['done']} records: completed={stats['done'] - stats['fail']}, "
          f"failed={stats['fail']}, pending={pending}; "
          f"estimated cost=${est_cost():.3f}; out={out_path}",
          flush=True)
    if stats["fail"]:
        return 1
    return 2 if pending else 0


if __name__ == "__main__":
    raise SystemExit(main())
