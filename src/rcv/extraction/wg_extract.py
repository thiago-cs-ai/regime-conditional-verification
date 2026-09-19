"""Extract WildGuard harmful-response verdicts and decision-position representations.

Generate an assessment, then reread it to obtain the score and hidden state
immediately before the selected verdict token. Missing positions produce a
zero vector, score 0.5, verdict 1, and empty assessment text. These placeholders
and parse fallbacks are counted in the stats; neither causes the CLI to fail.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

from ._io import (
    check_extraction_start,
    flush_chunk,
    read_jsonl,
    sha_file,
    stitch_chunks,
    validate_labeled_rows,
)
from .templates import (
    WG_COLON_ID,
    WG_MODEL_ID,
    WG_NO_ID,
    WG_RESP_ID,
    WG_YES_ID,
    WG_Z_DIM,
    wg_render,
)


def enable_determinism(seed):
    """Seed the random generators and request deterministic Torch operations with warnings."""
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    import random

    import torch

    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)


def load_wildguard(model_id, device, dtype_str):
    """Return ``(model, tokenizer, hidden_size)`` after checking yes/no token IDs."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_id)
    tok.padding_side = "left"
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    td = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[dtype_str]
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=td, device_map={"": device} if device != "auto" else "auto"
    )
    model.eval()
    assert tok.convert_tokens_to_ids("▁yes") == WG_YES_ID, "WildGuard tokenizer: unexpected ID for '▁yes'."
    assert tok.convert_tokens_to_ids("▁no") == WG_NO_ID, "WildGuard tokenizer: unexpected ID for '▁no'."
    with torch.no_grad():
        probe = tok("<s>hi", return_tensors="pt", add_special_tokens=False).input_ids.to(model.device)
        out = model(probe, output_hidden_states=True, return_dict=True)
        D = int(out.hidden_states[-1].shape[-1])
    return model, tok, D


def locate_harmresp_verdict(gen_ids):
    """Return ``(index, token_id, used_fallback)`` within the generated tokens.

    Select the token after the first ``[▁response, :]`` marker with a following
    token, regardless of that token's value. Otherwise select the last yes/no
    token, which may belong to an earlier field. Return ``(None, None, False)``
    when neither position exists.
    """
    g = list(gen_ids)
    for i in range(len(g) - 2):
        if g[i] == WG_RESP_ID and g[i + 1] == WG_COLON_ID:
            return i + 2, g[i + 2], False
    for i in range(len(g) - 1, -1, -1):
        if g[i] in (WG_YES_ID, WG_NO_ID):
            return i, g[i], True
    return None, None, False


def decision_vector(hidden_states, decision_pos, layer=-1):
    """Return the float32 ``(D,)`` state before the absolute verdict-token position."""
    h = hidden_states[layer][0]
    return h[decision_pos - 1].float().cpu().numpy().astype(np.float32)


def score_from_logits(logits, decision_pos):
    """Return the yes-token softmax over yes/no logits at ``decision_pos - 1``."""
    import torch

    z = logits[0, decision_pos - 1, [WG_YES_ID, WG_NO_ID]].float()
    return float(torch.softmax(z, dim=-1)[0].item())


def process_one(model, tok, prompt, response, max_new_tokens=32):
    """Return ``(z[D], score, pred, assessment_text, used_fallback)`` from two passes.

    Score is the yes-token softmax over yes/no; ``pred = 1`` (unsafe) when
    score >= 0.5. Return None if no verdict position can be located.
    """
    import torch

    inp = tok(wg_render(prompt, response), return_tensors="pt", add_special_tokens=False).input_ids
    inp = inp.to(model.device)
    attn = torch.ones_like(inp)
    prompt_len = int(inp.shape[1])
    with torch.no_grad():
        gen = model.generate(
            inp, attention_mask=attn, max_new_tokens=max_new_tokens,
            do_sample=False, num_beams=1, pad_token_id=tok.eos_token_id,
        )
    full = gen[0]
    gen_ids = full[prompt_len:].tolist()
    rel, _verdict_id, used_fallback = locate_harmresp_verdict(gen_ids)
    if rel is None:
        return None
    decision_pos = prompt_len + rel
    with torch.no_grad():
        out = model(
            full.unsqueeze(0), attention_mask=torch.ones_like(full.unsqueeze(0)),
            output_hidden_states=True, return_dict=True,
        )
    z = decision_vector(out.hidden_states, decision_pos)
    score = score_from_logits(out.logits, decision_pos)
    pred = int(score >= 0.5)
    verdict_text = tok.decode(gen_ids, skip_special_tokens=True).strip()
    return z, score, pred, verdict_text, used_fallback


def _dry_run(a) -> int:
    print("[wg][dry-run] plan:")
    print(f"  model_id       = {a.model_id}")
    print(f"  input          = {a.input}  (exists={Path(a.input).exists()})")
    print(f"  output jsonl   = {a.output}")
    print(f"  embeddings npy = {a.embeddings}")
    print(f"  Z contract     = harmresp_decision_L-1, dim {WG_Z_DIM}, float32; model dtype={a.dtype}")
    print(f"  two-pass       = greedy generation (max_new_tokens={a.max_new_tokens}) + reread generated tokens")
    print(f"  chunk_size={a.chunk_size} seed={a.seed} device={a.device}")
    print("  Model access   = configured Hugging Face credentials")
    print("  No model loaded.")
    return 0


def run(a) -> int:
    if a.dry_run:
        return _dry_run(a)
    t0 = time.time()
    if not Path(a.input).exists():
        sys.exit(f"[wg] Input file not found: {a.input}")
    enable_determinism(a.seed)
    samples = read_jsonl(a.input, a.max_rows)
    validate_labeled_rows(samples)
    n = len(samples)
    emb_path, out_jsonl = Path(a.embeddings), Path(a.output)
    check_extraction_start(emb_path, out_jsonl, n)
    stats_path = Path(a.stats) if a.stats else out_jsonl.with_suffix(".stats.json")

    model, tok, D = load_wildguard(a.model_id, a.device, a.dtype)
    if D != WG_Z_DIM:
        sys.exit(f"[wg] Model hidden size {D} does not match expected {WG_Z_DIM}.")
    print(f"[wg] items={n} embedding_dim={D}", flush=True)

    chunk_rows, chunk_vecs, chunk_idx = [], [], 0
    scores, preds, n_degen, n_fallback = [], [], 0, 0
    for i, s in enumerate(samples):
        r = process_one(model, tok, s.get("prompt", ""), s.get("response", ""), a.max_new_tokens)
        if r is None:
            n_degen += 1
            z, sc, verdict_text = np.zeros(D, np.float32), 0.5, ""
        else:
            z, sc, _pred, verdict_text, fb = r
            n_fallback += int(fb)
        if z.shape != (D,) or not np.isfinite(z).all():
            raise ValueError(f"Invalid embedding at row {i}: expected ({D},) and finite values.")
        if np.ndim(sc) != 0 or not np.isfinite(sc) or not 0 <= sc <= 1:
            raise ValueError(f"Unsafe score at row {i} must be a finite scalar in [0, 1].")
        pred = int(sc >= 0.5)
        gt = int(s.get("ground_truth", 0))
        row = dict(s)
        row["wg_pred"] = pred
        row["wg_score"] = sc
        row["wg_agreement"] = int(pred == gt)
        row["wg_verdict_text"] = verdict_text
        if r is None:
            row["z_error"] = 1
        chunk_rows.append(row)
        chunk_vecs.append(z)
        scores.append(sc)
        preds.append(pred)
        if len(chunk_rows) >= a.chunk_size:
            flush_chunk(emb_path, out_jsonl, chunk_rows, chunk_vecs, chunk_idx)
            chunk_rows, chunk_vecs = [], []
            chunk_idx += 1
        if (i + 1) % 200 == 0:
            print(f"[wg][progress] {i + 1}/{n} placeholders={n_degen} parse_fallbacks={n_fallback}", flush=True)
    if chunk_rows:
        flush_chunk(emb_path, out_jsonl, chunk_rows, chunk_vecs, chunk_idx)
    nrow, shape = stitch_chunks(emb_path, out_jsonl, expected_rows=n)

    dt = time.time() - t0
    sa = np.array(scores) if scores else np.array([0.0])
    stats = {
        "n": n, "n_degenerate": n_degen, "n_parse_fallback": n_fallback, "n_rows": nrow,
        "shape": list(shape), "z_head": "harmresp_decision_L-1", "wall_s": dt,
        "model": a.model_id, "dtype": a.dtype, "device": a.device, "seed": a.seed,
        "score_dist": {
            "min": float(sa.min()), "max": float(sa.max()), "mean": float(sa.mean()),
            "frac_0p1_0p9": float(((sa > 0.1) & (sa < 0.9)).mean()),
        },
        "pred_frac_unsafe": float(np.mean(preds)) if preds else 0.0,
        "z_npy_sha256": sha_file(emb_path),
    }
    Path(stats_path).parent.mkdir(parents=True, exist_ok=True)
    Path(stats_path).write_text(json.dumps(stats, indent=2))
    print(f"[wg][done] {nrow} rows, Z {shape}, placeholders={n_degen}, parse_fallbacks={n_fallback}", flush=True)
    return 0


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--input", required=True, help="jsonl with {item_id, prompt, response, ground_truth}")
    ap.add_argument("--output", required=True)
    ap.add_argument("--embeddings", required=True, help="Z npy (n, 4096) float32")
    ap.add_argument("--stats", default=None)
    ap.add_argument("--chunk-size", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-rows", type=int, default=None)
    ap.add_argument("--max-new-tokens", type=int, default=32)
    ap.add_argument("--model-id", default=WG_MODEL_ID)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    ap.add_argument("--dry-run", action="store_true", help="print the plan without loading a model")
    return ap


def main(argv=None) -> int:
    return run(build_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
