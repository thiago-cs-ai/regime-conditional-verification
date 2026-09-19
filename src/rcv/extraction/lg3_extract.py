"""Extract LG3 verdicts and decision-position representations.

Row extraction failures use placeholders: verdict 0, score 0.5, a zero vector,
and ``z_error = 1``. Artifacts are written before checking ``--fb-abort`` and
``--expect-z-dim``; a failed check exits with an error and leaves those files.
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
from ._lg3_context import (
    MODEL_ID,
    build_lg3_decision_context,
    resolve_decision_tokens,
    template_hash,
)

Z_DIM = 4096


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


def load_model(model_id, adapter, device, dtype_str):
    """Return ``(model, tokenizer, safe_id, unsafe_id, hidden_size)`` in evaluation mode."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_id)
    td = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[dtype_str]
    print(f"[lg3] load {model_id} dtype={dtype_str} device={device} adapter={adapter}", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=td,
        device_map={"": device} if device != "auto" else "auto",
        output_hidden_states=True,
    )
    if adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter)
    model.eval()
    safe_id, unsafe_id, _ = resolve_decision_tokens(tok, strict=True)
    return model, tok, safe_id, unsafe_id, int(model.config.hidden_size)


def extract_row(model, tok, safe_id, unsafe_id, prompt, response):
    """Return ``(pred, score, z)`` from one forward pass over the shared LG3 context.

    Score is the unsafe-token softmax over the safe/unsafe pair; ``pred = 1``
    when score >= 0.5. The float32 vector ``z`` has shape ``(hidden_size,)`` and
    comes from the final layer at the context's last token, before the verdict.
    """
    import torch

    ids = build_lg3_decision_context(tok, prompt, response).to(model.device)
    with torch.no_grad():
        out = model(ids, output_hidden_states=True, return_dict=True)
    z_logits = out.logits[0, -1, [safe_id, unsafe_id]].float()
    score = float(torch.softmax(z_logits, dim=-1)[1].item())
    pred = int(score >= 0.5)
    z = out.hidden_states[-1][0, -1].float().cpu().numpy().astype(np.float32)
    return pred, score, z


def _dry_run(a) -> int:
    print("[lg3][dry-run] plan:")
    print(f"  model_id       = {a.model_id}")
    print(f"  adapter        = {a.adapter or '(base, no adapter)'}")
    print(f"  input          = {a.input}  (exists={Path(a.input).exists()})")
    print(f"  output jsonl   = {a.output}")
    print(f"  embeddings npy = {a.embeddings}")
    print(f"  Z contract     = decision_L-1, expected dim {a.expect_z_dim or 'unchecked'}, float32; model dtype={a.dtype}")
    print(f"  chunk_size={a.chunk_size} seed={a.seed} error_rate_limit={a.fb_abort} device={a.device}")
    print("  Model access   = configured Hugging Face credentials")
    print("  No model loaded.")
    return 0


def run(a) -> int:
    if a.dry_run:
        return _dry_run(a)
    t0 = time.time()
    if not Path(a.input).exists():
        sys.exit(f"[lg3] Input file not found: {a.input}")
    if a.adapter and not Path(a.adapter).exists():
        sys.exit(f"[lg3] Adapter directory not found: {a.adapter}")
    enable_determinism(a.seed)
    samples = read_jsonl(a.input, a.max_rows)
    validate_labeled_rows(samples)
    n = len(samples)
    slice_name = a.slice_name or Path(a.output).stem
    emb_path, out_jsonl = Path(a.embeddings), Path(a.output)
    check_extraction_start(emb_path, out_jsonl, n)
    stats_path = Path(a.stats) if a.stats else out_jsonl.with_suffix(".stats.json")

    model, tok, safe_id, unsafe_id, D = load_model(a.model_id, a.adapter, a.device, a.dtype)
    thash = template_hash(tok)
    print(
        f"[lg3] slice={slice_name} items={n} embedding_dim={D} template_hash={thash[:12]}",
        flush=True,
    )

    chunk_rows, chunk_vecs, chunk_idx = [], [], 0
    scores, preds, n_err = [], [], 0
    for i, s in enumerate(samples):
        prompt, response = s.get("prompt", ""), s.get("response", "")
        try:
            pred, sc, z = extract_row(model, tok, safe_id, unsafe_id, prompt, response)
            if z.shape != (D,) or not np.isfinite(z).all():
                raise RuntimeError(
                    f"Invalid embedding: shape={z.shape}; expected ({D},) and finite values."
                )
            if np.ndim(sc) != 0 or not np.isfinite(sc) or not 0 <= sc <= 1:
                raise ValueError("Unsafe score must be a finite scalar in [0, 1].")
            z_err = 0
        except Exception as e:  # noqa: BLE001
            n_err += 1
            print(
                f"[lg3][error] item {s.get('item_id', i)!r} at row {i}: {type(e).__name__}: {e}; using placeholders",
                flush=True,
            )
            pred, sc, z, z_err = 0, 0.5, np.zeros(D, dtype=np.float32), 1
        gt = int(s.get("ground_truth", 0))
        row = dict(s)
        row["lg3_pred"] = pred
        row["lg3_score"] = sc
        row["lg3_agreement"] = int(pred == gt)
        if z_err:
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
            print(f"[lg3][progress] {i + 1}/{n} errors={n_err}", flush=True)
    if chunk_rows:
        flush_chunk(emb_path, out_jsonl, chunk_rows, chunk_vecs, chunk_idx)
    nrow, shape = stitch_chunks(emb_path, out_jsonl, expected_rows=n)

    dt = time.time() - t0
    sa = np.array(scores) if scores else np.array([0.0])
    fb_rate = n_err / n if n else 0.0
    stats = {
        "slice": slice_name, "n": n, "n_err": n_err, "fb_rate": fb_rate, "n_rows": nrow,
        "shape": list(shape), "D": D, "wall_s": dt, "adapter": a.adapter, "model": a.model_id,
        "dtype": a.dtype, "device": a.device, "seed": a.seed, "template_hash": thash,
        "score_dist": {
            "min": float(sa.min()), "max": float(sa.max()), "mean": float(sa.mean()),
            "std": float(sa.std()), "frac_0p1_0p9": float(((sa > 0.1) & (sa < 0.9)).mean()),
        },
        "pred_frac_unsafe": float(np.mean(preds)) if preds else 0.0,
        "z_npy_sha256": sha_file(emb_path),
    }
    Path(stats_path).parent.mkdir(parents=True, exist_ok=True)
    Path(stats_path).write_text(json.dumps(stats, indent=2))

    if fb_rate > a.fb_abort:
        (out_jsonl.parent / f"{slice_name}._FB_ABORT").write_text(f"fb_rate {fb_rate} > {a.fb_abort}\n")
        sys.exit(f"[lg3] FB-ABORT: extraction error rate {fb_rate:.4f} exceeds {a.fb_abort}; artifacts retained.")
    if a.expect_z_dim and D != a.expect_z_dim:
        sys.exit(f"[lg3] Embedding dimension {D} does not match expected {a.expect_z_dim}; artifacts retained.")
    print(f"[lg3][done] {nrow} rows, Z {shape}, error_rate={fb_rate:.4f}", flush=True)
    return 0


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--input", required=True, help="jsonl with {item_id, prompt, response, ground_truth}")
    ap.add_argument("--output", required=True, help="verdicts jsonl (input rows + lg3_pred/score/agreement)")
    ap.add_argument("--embeddings", required=True, help="float32 .npy array (rows, model hidden size)")
    ap.add_argument("--stats", default=None)
    ap.add_argument("--adapter", default=None, help="LoRA adapter dir; omit for base LG3")
    ap.add_argument("--slice-name", default=None)
    ap.add_argument("--chunk-size", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-rows", type=int, default=None)
    ap.add_argument("--model-id", default=MODEL_ID)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    ap.add_argument("--fb-abort", type=float, default=0.02,
                    help="exit with an error after writing if the failed-row fraction exceeds this value")
    ap.add_argument("--expect-z-dim", type=int, default=Z_DIM,
                    help="expected model hidden size (default: 4096; 0 disables the check)")
    ap.add_argument("--dry-run", action="store_true", help="print the plan without loading a model")
    return ap


def main(argv=None) -> int:
    return run(build_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
