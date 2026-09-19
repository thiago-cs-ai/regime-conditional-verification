"""Shared LG3 decision context and tokenizer fingerprints for extraction and fine-tuning."""

from __future__ import annotations

import hashlib
import json

MODEL_ID = "meta-llama/Llama-Guard-3-8B"
DECISION_NL = "\n\n"
EXPECTED_NL_ID = 271
EXPECTED_SAFE_ID = 19193
EXPECTED_UNSAFE_ID = 39257
TEMPLATE_HASH_ARM0 = "e105b5b9e02db8278a387006e35e9758e7c413ff4ff0a7d79920e1edda8d0a4a"


def build_lg3_decision_context(tok, prompt, response):
    """Return a ``(1, L)`` token tensor ending with the encoded ``"\\n\\n"`` suffix.

    Logits at ``L-1`` predict the next safe/unsafe token; extraction reads the
    hidden state there, and fine-tuning appends the target token at ``L``.
    """
    import torch

    full = [{"role": "user", "content": prompt}, {"role": "assistant", "content": response}]
    fi = tok.apply_chat_template(full, add_generation_prompt=True, return_tensors="pt")
    nl = tok.encode(DECISION_NL, add_special_tokens=False)
    fi = torch.cat([fi, torch.tensor([nl], dtype=fi.dtype, device=fi.device)], dim=1)
    return fi


def resolve_decision_tokens(tok, strict=True):
    """Return ``(safe_id, unsafe_id, nl_ids)`` from the tokenizer.

    Strict mode requires ``nl_ids == [271]`` and verdict IDs other than None or
    the unknown-token ID. It does not compare verdict IDs with the reference
    constants. Non-strict mode returns the values without these checks raising.
    """
    nl = tok.encode(DECISION_NL, add_special_tokens=False)
    safe_id = tok.convert_tokens_to_ids("safe")
    unsafe_id = tok.convert_tokens_to_ids("unsafe")
    unk = tok.unk_token_id
    problems = []
    if nl != [EXPECTED_NL_ID]:
        problems.append(f'encode("\\n\\n") returned {nl}; expected [{EXPECTED_NL_ID}]')
    if safe_id in (None, unk):
        problems.append(f"safe token ID is missing or unknown ({safe_id})")
    if unsafe_id in (None, unk):
        problems.append(f"unsafe token ID is missing or unknown ({unsafe_id})")
    if problems and strict:
        raise RuntimeError("Invalid LG3 decision tokens: " + "; ".join(problems))
    return safe_id, unsafe_id, nl


def target_token_id(safe_id, unsafe_id, ground_truth):
    """Map a binary target label to its token ID: 0 = safe, 1 = unsafe."""
    return unsafe_id if int(ground_truth) == 1 else safe_id


def _sha(*parts) -> str:
    """Hash each part followed by a NUL byte, including the final part."""
    h = hashlib.sha256()
    for p in parts:
        h.update(p if isinstance(p, (bytes, bytearray)) else str(p).encode())
        h.update(b"\x00")
    return h.hexdigest()


def template_hash(tok) -> str:
    """Fingerprint the chat template, newline IDs, and safe/unsafe IDs."""
    safe_id, unsafe_id, nl = resolve_decision_tokens(tok, strict=False)
    return _sha(tok.chat_template or "", json.dumps(nl), safe_id, unsafe_id)


def template_hash_record(tok) -> dict:
    """Return tokenizer fingerprints, token IDs, and version metadata for the FT manifest."""
    import transformers

    safe_id, unsafe_id, nl = resolve_decision_tokens(tok, strict=False)
    return {
        "template_hash": template_hash(tok),
        "chat_template_sha256": _sha(tok.chat_template or ""),
        "nl_ids": nl,
        "expected_nl_id": EXPECTED_NL_ID,
        "safe_id": safe_id,
        "unsafe_id": unsafe_id,
        "model_id": MODEL_ID,
        "transformers_version": transformers.__version__,
        "expected_template_hash_arm0": TEMPLATE_HASH_ARM0,
    }
