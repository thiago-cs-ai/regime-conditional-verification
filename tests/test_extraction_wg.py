from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from rcv.extraction import wg_extract as W
from rcv.extraction.templates import WG_COLON_ID, WG_NO_ID, WG_RESP_ID, WG_YES_ID


def test_locate_anchors_on_response_colon_marker():
    gen = [100, WG_RESP_ID, WG_COLON_ID, WG_YES_ID, 200]
    rel, verdict_id, fb = W.locate_harmresp_verdict(gen)
    assert (rel, verdict_id, fb) == (3, WG_YES_ID, False)


def test_locate_reads_position_regardless_of_emitted_token():
    gen = [WG_RESP_ID, WG_COLON_ID, 777]
    rel, verdict_id, fb = W.locate_harmresp_verdict(gen)
    assert rel == 2 and verdict_id == 777 and fb is False


def test_locate_fallback_to_last_yes_no_when_marker_absent():
    gen = [1, 2, WG_NO_ID, 3]
    rel, verdict_id, fb = W.locate_harmresp_verdict(gen)
    assert verdict_id == WG_NO_ID and fb is True


def test_locate_returns_none_when_no_marker_and_no_yes_no():
    assert W.locate_harmresp_verdict([1, 2, 3]) == (None, None, False)


def test_decision_pos_handoff_pass1_to_pass2():
    prompt_len = 40
    gen_ids = [100, WG_RESP_ID, WG_COLON_ID, WG_YES_ID]
    rel, _, _ = W.locate_harmresp_verdict(gen_ids)
    decision_pos = prompt_len + rel
    assert decision_pos == 43


def test_score_from_logits_index0_is_yes_unsafe():
    vocab = 6000
    logits = torch.full((1, 5, vocab), -10.0)
    decision_pos = 3
    logits[0, decision_pos - 1, WG_YES_ID] = 8.0
    logits[0, decision_pos - 1, WG_NO_ID] = -8.0
    score = W.score_from_logits(logits, decision_pos)
    assert score > 0.99
    assert int(score >= 0.5) == 1


def test_decision_vector_reads_position_minus_one_last_layer():
    D = 8
    T = 5
    hs = [torch.zeros((1, T, D)) for _ in range(3)]
    decision_pos = 4
    hs[-1][0, decision_pos - 1] = torch.arange(D, dtype=torch.float32)
    v = W.decision_vector(hs, decision_pos, layer=-1)
    assert v.shape == (D,) and v.dtype == np.float32
    assert np.array_equal(v, np.arange(D, dtype=np.float32))
