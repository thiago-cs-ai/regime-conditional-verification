from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from rcv.extraction import _lg3_context as C


class FakeTok:
    chat_template = "FAKE-CHAT-TEMPLATE-v1"
    unk_token_id = 0

    def __init__(self, ctx_ids=(10, 11, 12), nl=(271,), safe=19193, unsafe=39257):
        self._ctx = list(ctx_ids)
        self._nl = list(nl)
        self._safe = safe
        self._unsafe = unsafe

    def apply_chat_template(self, messages, add_generation_prompt=True, return_tensors="pt"):
        assert add_generation_prompt is True
        return torch.tensor([self._ctx], dtype=torch.long)

    def encode(self, s, add_special_tokens=False):
        assert s == "\n\n" and add_special_tokens is False
        return list(self._nl)

    def convert_tokens_to_ids(self, tok):
        return {"safe": self._safe, "unsafe": self._unsafe}.get(tok, self.unk_token_id)


def test_build_context_appends_decision_nl_at_L_minus_1():
    tok = FakeTok(ctx_ids=(10, 11, 12), nl=(271,))
    fi = C.build_lg3_decision_context(tok, "p", "r")
    assert fi.shape == (1, 4)
    assert int(fi[0, -1].item()) == 271
    assert fi[0, :3].tolist() == [10, 11, 12]


def test_resolve_decision_tokens_ok():
    safe, unsafe, nl = C.resolve_decision_tokens(FakeTok(), strict=True)
    assert (safe, unsafe, nl) == (19193, 39257, [271])


def test_resolve_decision_tokens_strict_raises_on_nl_drift():
    with pytest.raises(RuntimeError):
        C.resolve_decision_tokens(FakeTok(nl=(999,)), strict=True)


def test_resolve_decision_tokens_nonstrict_reports_without_raising():
    safe, unsafe, nl = C.resolve_decision_tokens(FakeTok(nl=(999,)), strict=False)
    assert nl == [999]


def test_target_token_id_polarity_1_is_unsafe():
    assert C.target_token_id(19193, 39257, 1) == 39257
    assert C.target_token_id(19193, 39257, 0) == 19193


def test_template_hash_deterministic_and_sensitive():
    h1 = C.template_hash(FakeTok())
    h2 = C.template_hash(FakeTok())
    assert h1 == h2 and len(h1) == 64
    assert C.template_hash(FakeTok(safe=1)) != h1


def test_expected_constants_frozen():
    assert C.EXPECTED_NL_ID == 271
    assert (C.EXPECTED_SAFE_ID, C.EXPECTED_UNSAFE_ID) == (19193, 39257)
    assert C.TEMPLATE_HASH_ARM0.startswith("e105b5b9")
