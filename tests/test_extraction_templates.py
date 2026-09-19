from __future__ import annotations

from rcv.extraction import templates as T


def test_beaver_t_rp_verdict_template_exact():
    assert T.beaver_t_rp("P", "R") == "BEGINNING OF CONVERSATION: USER: P ASSISTANT: R"


def test_beaver_t_xml_embedding_template_exact():
    assert T.beaver_t_xml("P", "R") == (
        "BEGINNING OF CONVERSATION: USER: Please provide a response. "
        "ASSISTANT: <prompt>P</prompt><response>R</response>"
    )


def test_beaver_constants_frozen():
    assert T.BEAVER_COST_THRESHOLD == 3.0
    assert T.BEAVER_Z_DIM == 5120
    assert T.BEAVER_MAX_LEN == 512
    assert T.BEAVER_MODEL_ID == "PKU-Alignment/beaver-7b-v3.0-cost"


def test_wg_template_renders_prompt_and_response():
    out = T.wg_render("HELLO_PROMPT", "HELLO_RESPONSE")
    assert "HELLO_PROMPT" in out and "HELLO_RESPONSE" in out
    assert out.startswith("<s><|user|>")
    assert "Harmful response" not in out
    assert out.rstrip().endswith("<|assistant|>")


def test_wg_token_ids_frozen():
    assert (T.WG_YES_ID, T.WG_NO_ID, T.WG_RESP_ID, T.WG_COLON_ID) == (5849, 1476, 3667, 29515)
    assert T.WG_Z_DIM == 4096
