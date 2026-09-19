"""Extract Beaver costs, verdicts, and float32 representations.

Each batch uses separate forwards for T_rp verdicts and T_xml embeddings, each
truncated to 512 tokens. Non-finite costs or embeddings stop the run before
saving that batch; previously written chunks remain.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

from . import beaver_stub
from ._io import (
    check_extraction_start,
    flush_chunk,
    read_jsonl,
    sha_file,
    stitch_chunks,
    validate_labeled_rows,
)
from .templates import BEAVER_COST_THRESHOLD, BEAVER_MODEL_ID, BEAVER_Z_DIM


def _dry_run(a) -> int:
    print("[beaver][dry-run] plan:")
    print(f"  model_id       = {a.model_id}")
    print(f"  input          = {a.input}  (exists={Path(a.input).exists()})")
    print(f"  output jsonl   = {a.output}")
    print(f"  embeddings npy = {a.embeddings}")
    print(f"  verdict        = T_rp cost >= {BEAVER_COST_THRESHOLD} means unsafe (1)")
    print(f"  Z contract     = end_last_hidden_state under T_xml, dim {BEAVER_Z_DIM}, float32; model dtype={a.dtype}")
    print(f"  batch={a.batch} device={a.device}")
    print("  backend        = safe_rlhf.models.AutoModelForScore from github.com/PKU-Alignment/safe-rlhf")
    print("  No model loaded.")
    return 0


def run(a) -> int:
    if a.dry_run:
        return _dry_run(a)
    t0 = time.time()
    if not Path(a.input).exists():
        sys.exit(f"[beaver] Input file not found: {a.input}")
    samples = read_jsonl(a.input, a.max_rows)
    validate_labeled_rows(samples)
    n = len(samples)
    emb_path, out_jsonl = Path(a.embeddings), Path(a.output)
    check_extraction_start(emb_path, out_jsonl, n)
    stats_path = Path(a.stats) if a.stats else out_jsonl.with_suffix(".stats.json")

    tok, model = beaver_stub.load_beaver(a.model_id, a.device, a.dtype)
    print(f"[beaver] Model loaded; items={n}", flush=True)

    chunk_rows, chunk_vecs, chunk_idx = [], [], 0
    costs, preds = [], []
    for bstart in range(0, n, a.batch):
        batch = samples[bstart:bstart + a.batch]
        prompts = [s.get("prompt", "") for s in batch]
        responses = [s.get("response", "") for s in batch]
        cost, pred = beaver_stub.beaver_verdict(tok, model, prompts, responses, a.device)
        Z = beaver_stub.beaver_embedding(tok, model, prompts, responses, a.device)
        if cost.shape != (len(batch),) or pred.shape != (len(batch),) or Z.shape != (len(batch), BEAVER_Z_DIM):
            raise ValueError(f"Invalid batch shapes at row {bstart}: costs {cost.shape}, verdicts {pred.shape}, embeddings {Z.shape}.")
        if not np.isin(pred, (0, 1)).all():
            raise ValueError(f"Nonbinary verdict in batch starting at row {bstart}.")
        if not (np.isfinite(cost).all() and np.isfinite(Z).all()):
            sys.exit(f"[beaver] Non-finite cost or embedding in batch starting at row {bstart}.")
        for j, s in enumerate(batch):
            gt = int(s.get("ground_truth", 0))
            row = dict(s)
            row["beaver_cost"] = float(cost[j])
            row["beaver_pred"] = int(pred[j])
            row["beaver_agreement"] = int(int(pred[j]) == gt)
            chunk_rows.append(row)
            chunk_vecs.append(Z[j])
            costs.append(float(cost[j]))
            preds.append(int(pred[j]))
            if len(chunk_rows) >= a.chunk_size:
                flush_chunk(emb_path, out_jsonl, chunk_rows, chunk_vecs, chunk_idx)
                chunk_rows, chunk_vecs = [], []
                chunk_idx += 1
        if (bstart // a.batch) % 20 == 0:
            print(f"[beaver][progress] {min(bstart + a.batch, n)}/{n}", flush=True)
    if chunk_rows:
        flush_chunk(emb_path, out_jsonl, chunk_rows, chunk_vecs, chunk_idx)
    nrow, shape = stitch_chunks(emb_path, out_jsonl, expected_rows=n)

    dt = time.time() - t0
    ca = np.array(costs) if costs else np.array([0.0])
    stats = {
        "n": n, "n_rows": nrow, "shape": list(shape), "z_dim": BEAVER_Z_DIM, "wall_s": dt,
        "model": a.model_id, "dtype": a.dtype, "device": a.device, "threshold": BEAVER_COST_THRESHOLD,
        "cost_dist": {"min": float(ca.min()), "max": float(ca.max()), "mean": float(ca.mean())},
        "pred_frac_unsafe": float(np.mean(preds)) if preds else 0.0,
        "z_npy_sha256": sha_file(emb_path),
    }
    Path(stats_path).parent.mkdir(parents=True, exist_ok=True)
    Path(stats_path).write_text(json.dumps(stats, indent=2))
    print(f"[beaver][done] {nrow} rows, Z {shape}", flush=True)
    return 0


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--input", required=True, help="jsonl with {item_id, prompt, response, ground_truth}")
    ap.add_argument("--output", required=True)
    ap.add_argument("--embeddings", required=True, help="Z npy (n, 5120) float32")
    ap.add_argument("--stats", default=None)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--chunk-size", type=int, default=2000)
    ap.add_argument("--max-rows", type=int, default=None)
    ap.add_argument("--model-id", default=BEAVER_MODEL_ID)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    ap.add_argument("--dry-run", action="store_true", help="print the plan without loading a model")
    return ap


def main(argv=None) -> int:
    return run(build_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
