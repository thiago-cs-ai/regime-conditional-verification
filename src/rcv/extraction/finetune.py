"""Fine-tune Llama-Guard-3-8B with LoRA and one supervised verdict token per example.

Training uses the extraction context, followed by the safe/unsafe target token.
The default recipe uses a frozen bf16 base, fp32 adapters, and at most 30 optimizer
steps. Training options can be overridden on the CLI; there is no early stopping.

The adapter is saved before evaluation. A completed run writes ``ft_manifest.json``
and exits with 0 if the acceptance gate passes, otherwise 2. Rejected adapters
remain on disk.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from ._io import read_jsonl, validate_labeled_rows, write_jsonl
from ._lg3_context import (
    MODEL_ID,
    build_lg3_decision_context,
    resolve_decision_tokens,
    target_token_id,
    template_hash,
    template_hash_record,
)

TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

DEFAULTS = dict(
    lr=2e-5, lora_r=16, lora_alpha=32, lora_dropout=0.05, epochs=1, step_ceiling=30,
    micro_batch=16, grad_accum=1, warmup_frac=0.05, max_grad_norm=1.0,
)


def lora_config_dict(r=16, alpha=32, dropout=0.05, target_modules=TARGET_MODULES):
    return {
        "r": r,
        "lora_alpha": alpha,
        "lora_dropout": dropout,
        "bias": "none",
        "task_type": "CAUSAL_LM",
        "target_modules": list(target_modules),
    }


def binding_gate(val_delta, overfit_gap, degenerate, retention_flips):
    """Return ``(accepted, diagnostics)`` from the post-training readouts.

    ``val_delta`` is adapter minus base validation accuracy; ``overfit_gap`` is
    adapter training minus validation accuracy. ``retention_flips`` counts
    base-correct retention examples the adapter gets wrong. ``degenerate`` is
    the validation-score check.
    """
    checks = {
        "val_delta_ok": val_delta is not None and val_delta >= -0.02,
        "overfit_gap_ok": overfit_gap is not None and overfit_gap <= 0.15,
        "not_degenerate": not bool(degenerate),
        "retention_ok": retention_flips is not None and retention_flips <= 5,
    }
    return all(checks.values()), {
        "val_delta": val_delta, "overfit_gap": overfit_gap, "degenerate": bool(degenerate),
        "retention_flips": retention_flips, "checks": checks,
    }


def build_ft_example(tok, safe_id, unsafe_id, prompt, response, ground_truth):
    """Return token IDs and labels of shape ``(L+1,)`` for an extraction context of length L.

    Only the appended verdict token is supervised; other labels are -100.
    Ground truth uses 0 for safe and 1 for unsafe.
    """
    import torch

    ctx = build_lg3_decision_context(tok, prompt, response)[0]
    t = target_token_id(safe_id, unsafe_id, ground_truth)
    ids = torch.cat([ctx, torch.tensor([t], dtype=ctx.dtype)])
    labels = torch.full((ids.shape[0],), -100, dtype=torch.long)
    labels[-1] = t  # The causal-LM label shift supervises the final context logit.
    return ids, labels


def assert_single_decision_supervision(ids, labels, safe_id, unsafe_id, nl_id=271):
    """Require one supervised safe/unsafe target at the end, immediately after ``nl_id``."""
    sup = (labels != -100).nonzero()
    if sup.numel() != 1:
        raise AssertionError(f"Supervised positions: {sup.numel()}; expected 1")
    pos = int(sup[0, 0].item())
    if pos != ids.shape[0] - 1:
        raise AssertionError(f"Supervised position {pos} must be the last index ({ids.shape[0] - 1})")
    t = int(labels[pos].item())
    if t not in (int(safe_id), int(unsafe_id)):
        raise AssertionError(f"Supervised token {t} is neither safe nor unsafe")
    if int(ids[pos].item()) != t:
        raise AssertionError("Final input token differs from its supervised label")
    if int(ids[pos - 1].item()) != nl_id:
        raise AssertionError(f"Token before target is {int(ids[pos - 1].item())}; expected \\n\\n ({nl_id})")


def collate(examples, pad_id):
    """Right-pad to ``(batch, max_length)``; mask padding with attention 0 and label -100."""
    import torch

    B = len(examples)
    L = max(int(e[0].shape[0]) for e in examples)
    ids = torch.full((B, L), pad_id, dtype=torch.long)
    attn = torch.zeros((B, L), dtype=torch.long)
    labels = torch.full((B, L), -100, dtype=torch.long)
    for b, (x, lab) in enumerate(examples):
        sl = int(x.shape[0])
        ids[b, :sl] = x
        attn[b, :sl] = 1
        labels[b, :sl] = lab
    return ids, attn, labels


def _validate_splits(splits):
    for name in ("ft", "val", "retention"):
        rows = splits.get(name, [])
        if not rows:
            raise ValueError(f"Fine-tuning split {name!r} is empty.")
        validate_labeled_rows(rows)
    train_ids = {str(r["item_id"]) for r in splits["ft"]}
    for name in ("val", "retention"):
        overlap = train_ids & {str(r["item_id"]) for r in splits[name]}
        if overlap:
            raise ValueError(f"Training and {name} share {len(overlap)} item IDs.")


def materialize_ft_corpus(universe_jsonl, distribution_json, out_dir):
    """Write split rows in the distribution's item-ID order, with optional label overrides.

    The distribution has ``splits`` (ft/val/family/retention ID lists) and optional
    ``labels`` (item ID to target label). The output directory must not exist.
    Required splits are validated before any files are written.
    """
    from rcv.rebuild._jsonl import read_jsonl as rb_read

    out_dir = Path(out_dir)
    if out_dir.exists():
        raise ValueError(f"Corpus directory already exists: {out_dir}")
    universe = rb_read(universe_jsonl)
    by_id = {r["item_id"]: r for r in universe}
    if len({str(r["item_id"]) for r in universe}) != len(universe):
        raise ValueError("Universe contains duplicate item IDs.")
    dist = json.loads(Path(distribution_json).read_text())
    splits = dist["splits"]
    labels = dist.get("labels", {})
    prepared = {}
    for name in ("ft", "val", "family", "retention"):
        ids = splits.get(name, [])
        rows = []
        for iid in ids:
            if iid not in by_id:
                raise KeyError(f"Distribution item {iid!r} is missing from universe {universe_jsonl}")
            src = by_id[iid]
            y = labels[iid] if iid in labels else src.get("ground_truth")
            rows.append({
                "item_id": iid, "prompt": src.get("prompt"),
                "response": src.get("response"), "ground_truth": y,
            })
        validate_labeled_rows(rows)
        for row in rows:
            row["ground_truth"] = int(row["ground_truth"])
        prepared[name] = rows
    _validate_splits(prepared)
    written = {}
    for name, rows in prepared.items():
        path = out_dir / f"{name}.jsonl"
        if rows:
            write_jsonl(path, rows)
        written[name] = {"n": len(rows), "path": str(path)}
    return written


def load(lora_r, lora_alpha, lora_dropout, grad_ckpt=True):
    """Return ``(tokenizer, model, safe_id, unsafe_id)`` on CUDA: bf16 base, fp32 adapters."""
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    base = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16).to("cuda")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    base.config.use_cache = False
    cfg = lora_config_dict(lora_r, lora_alpha, lora_dropout)
    model = get_peft_model(base, LoraConfig(**cfg))
    for p in model.parameters():
        if p.requires_grad:
            p.data = p.data.float()
    if grad_ckpt:
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
    model.print_trainable_parameters()
    safe_id, unsafe_id, _ = resolve_decision_tokens(tok, strict=True)
    return tok, model, safe_id, unsafe_id


def _decisions(model, tok, safe_id, unsafe_id, rows):
    """Return aligned verdicts and unsafe scores through the LG3 extraction path."""
    from .lg3_extract import extract_row

    preds, scores = [], []
    for r in rows:
        pred, sc, _z = extract_row(model, tok, safe_id, unsafe_id, r.get("prompt", ""), r.get("response", ""))
        if not np.isfinite(sc):
            raise ValueError(f"Non-finite readout score for item {r['item_id']!r}.")
        preds.append(pred)
        scores.append(sc)
    return np.array(preds), np.array(scores)


def _readout(model, tok, safe_id, unsafe_id, train, val, retention):
    """Compare adapter and base agreement with target labels after training.

    Validation scores are degenerate if non-finite or if fewer than 5% or more
    than 60% reach the unsafe threshold (0.5). Missing split metrics are None.
    """
    was_train = model.training
    model.eval()

    def ft_base(rows):
        ft_p, _ = _decisions(model, tok, safe_id, unsafe_id, rows)
        with model.disable_adapter():
            b_p, _ = _decisions(model, tok, safe_id, unsafe_id, rows)
        gt = np.array([int(r.get("ground_truth", 0)) for r in rows])
        return ft_p, b_p, gt

    v_ft, v_base, v_gt = ft_base(val) if val else (np.array([]), np.array([]), np.array([]))
    val_ft_acc = float(np.mean(v_ft == v_gt)) if len(v_gt) else None
    val_base_acc = float(np.mean(v_base == v_gt)) if len(v_gt) else None
    val_delta = None if val_ft_acc is None else val_ft_acc - val_base_acc

    t_ft, _, t_gt = ft_base(train) if train else (np.array([]), None, np.array([]))
    train_ft_acc = float(np.mean(t_ft == t_gt)) if len(t_gt) else None
    overfit_gap = None if (train_ft_acc is None or val_ft_acc is None) else train_ft_acc - val_ft_acc

    ret_flips = None
    if retention:
        r_ft, r_base, r_gt = ft_base(retention)
        base_correct = r_base == r_gt
        ret_flips = int(np.sum(base_correct & (r_ft != r_gt)))

    degenerate = False
    if val:
        _, v_sc = _decisions(model, tok, safe_id, unsafe_id, val)
        pfu = float((v_sc >= 0.5).mean())
        degenerate = bool(not np.isfinite(v_sc).all() or pfu < 0.05 or pfu > 0.6)
    if was_train:
        model.train()
    return {
        "val_ft_acc": val_ft_acc, "val_base_acc": val_base_acc, "val_delta": val_delta,
        "train_ft_acc": train_ft_acc, "overfit_gap": overfit_gap,
        "retention_flips": ret_flips, "degenerate": degenerate,
    }


def _load_splits(corpus_dir):
    corpus_dir = Path(corpus_dir)
    out = {}
    for name in ("ft", "val", "family", "retention"):
        p = corpus_dir / f"{name}.jsonl"
        out[name] = read_jsonl(p) if p.exists() else []
    return out


def _dry_run(a) -> int:
    print("[finetune][dry-run] plan:")
    print(f"  seed                       = {a.seed}")
    print(f"  base model                 = {MODEL_ID} (gated — your HF auth)")
    print(f"  corpus_dir                 = {a.corpus_dir}")
    if a.rebuild_universe or a.distribution:
        print(f"  materialization inputs     = universe={a.rebuild_universe} dist={a.distribution} (both required)")
    print(f"  out (adapter)              = {a.out}")
    print(f"  recipe                     = lr={a.lr} r={a.lora_r} α={a.lora_alpha} dropout={a.lora_dropout}")
    print(f"                               epochs={a.epochs} step_ceiling={a.step_ceiling} "
          f"eff_batch={a.micro_batch * a.grad_accum} warmup_frac={a.warmup_frac} clip={a.max_grad_norm}")
    print("  objective                  = decision-token-only CE (1 supervised token/example)")
    print(f"  lora_config                = {json.dumps(lora_config_dict(a.lora_r, a.lora_alpha, a.lora_dropout), sort_keys=True)}")
    print("  binding gate               = val_delta>=-0.02 & overfit_gap<=0.15 & !degenerate & "
          "retention_flips<=5")
    splits = _load_splits(a.corpus_dir) if Path(a.corpus_dir).exists() else {}
    if splits:
        print("  corpus counts              = " + ", ".join(f"{k}={len(v)}" for k, v in splits.items()))
    print("  No model loaded.")
    return 0


def run(a) -> int:
    if a.dry_run:
        return _dry_run(a)
    import torch

    for name in ("epochs", "micro_batch", "grad_accum", "step_ceiling"):
        if getattr(a, name) <= 0:
            raise ValueError(f"{name} must be positive.")
    if bool(a.rebuild_universe) != bool(a.distribution):
        raise ValueError("Supply both --rebuild-universe and --distribution.")
    out = Path(a.out)
    if out.exists() and (not out.is_dir() or any(out.iterdir())):
        raise ValueError(f"Fine-tuning output directory must be empty: {out}")
    if a.rebuild_universe and a.distribution:
        materialize_ft_corpus(a.rebuild_universe, a.distribution, a.corpus_dir)
    splits = _load_splits(a.corpus_dir)
    _validate_splits(splits)
    train, val, retention = splits["ft"], splits["val"], splits["retention"]
    batches = (len(train) + a.micro_batch - 1) // a.micro_batch
    if batches % a.grad_accum and a.step_ceiling > batches // a.grad_accum:
        raise ValueError("Training would end an epoch with an incomplete gradient-accumulation group.")
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)

    tok, model, safe_id, unsafe_id = load(a.lora_r, a.lora_alpha, a.lora_dropout)
    thash = template_hash(tok)
    pad_id = tok.pad_token_id
    eff_batch = a.micro_batch * a.grad_accum
    nl_ids = tok.encode("\n\n", add_special_tokens=False)

    examples = [
        build_ft_example(tok, safe_id, unsafe_id, r["prompt"], r["response"], r["ground_truth"])
        for r in train
    ]
    for ids_e, lab_e in examples:
        assert_single_decision_supervision(ids_e, lab_e, safe_id, unsafe_id, nl_id=nl_ids[-1])
    print(f"[finetune] examples={len(examples)} eff_batch={eff_batch} template_hash={thash[:12]}", flush=True)

    trainable = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable, lr=a.lr, weight_decay=0.0)
    steps_per_epoch = max(1, int(np.ceil(len(examples) / eff_batch)))
    total_steps = min(a.step_ceiling, a.epochs * steps_per_epoch)
    warm = max(1, int(a.warmup_frac * total_steps))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt,
        lambda s: s / warm if s < warm else 0.5 * (1 + np.cos(np.pi * (s - warm) / max(1, total_steps - warm))),
    )

    t0, gstep = time.time(), 0
    for epoch in range(a.epochs):
        order = np.arange(len(examples))
        np.random.default_rng(a.seed + epoch).shuffle(order)
        model.train()
        opt.zero_grad()
        acc = 0
        for i in range(0, len(order), a.micro_batch):
            batch = [examples[int(j)] for j in order[i:i + a.micro_batch]]
            ids, attn, labels = collate(batch, pad_id)
            dev = next(model.parameters()).device
            fwd = dict(input_ids=ids.to(dev), attention_mask=attn.to(dev), labels=labels.to(dev))
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = model(**fwd).loss / a.grad_accum
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite training loss before optimizer step {gstep + 1}.")
            loss.backward()
            acc += 1
            if acc % a.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(trainable, a.max_grad_norm)
                opt.step()
                sched.step()
                opt.zero_grad()
                gstep += 1
                if gstep >= a.step_ceiling:
                    break
        if gstep >= a.step_ceiling:
            break

    (out / "adapter").mkdir(exist_ok=True)
    model.save_pretrained(str(out / "adapter"))

    readout = _readout(model, tok, safe_id, unsafe_id, train, val, retention)
    ok, gate = binding_gate(readout["val_delta"], readout["overfit_gap"], readout["degenerate"],
                            readout["retention_flips"])
    manifest = {
        "model_id": MODEL_ID, "seed": a.seed,
        "lora": lora_config_dict(a.lora_r, a.lora_alpha, a.lora_dropout),
        "optim": {"lr": a.lr, "epochs": a.epochs, "step_ceiling": a.step_ceiling,
                  "realized_steps": gstep, "micro_batch": a.micro_batch, "grad_accum": a.grad_accum,
                  "eff_batch": eff_batch, "warmup_frac": a.warmup_frac, "warmup_steps": warm,
                  "max_grad_norm": a.max_grad_norm, "schedule": "cosine_warmup"},
        "objective": "decision-token-only CE (1 supervised token/example at the \\n\\n position)",
        "template": template_hash_record(tok),
        "readout": readout, "binding_gate": gate, "binding_pass": ok,
        "timing_s": round(time.time() - t0, 1),
    }
    (out / "ft_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"[finetune][done] steps={gstep} binding_pass={ok} gate={gate['checks']}", flush=True)
    return 0 if ok else 2


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--seed", type=int, required=True, help="seed for initialization, dropout, and shuffling")
    ap.add_argument("--corpus-dir", required=True, help="dir with ft/val/family/retention.jsonl")
    ap.add_argument("--out", required=True, help="output dir for the adapter + ft_manifest.json")
    ap.add_argument("--rebuild-universe", default=None,
                    help="universe JSONL; with --distribution, create a new --corpus-dir")
    ap.add_argument("--distribution", default=None, help="JSON of split item IDs and optional label overrides")
    ap.add_argument("--lr", type=float, default=DEFAULTS["lr"])
    ap.add_argument("--lora-r", dest="lora_r", type=int, default=DEFAULTS["lora_r"])
    ap.add_argument("--lora-alpha", dest="lora_alpha", type=int, default=DEFAULTS["lora_alpha"])
    ap.add_argument("--lora-dropout", dest="lora_dropout", type=float, default=DEFAULTS["lora_dropout"])
    ap.add_argument("--micro-batch", type=int, default=DEFAULTS["micro_batch"])
    ap.add_argument("--grad-accum", type=int, default=DEFAULTS["grad_accum"])
    ap.add_argument("--epochs", type=int, default=DEFAULTS["epochs"])
    ap.add_argument("--step-ceiling", type=int, default=DEFAULTS["step_ceiling"])
    ap.add_argument("--warmup-frac", type=float, default=DEFAULTS["warmup_frac"])
    ap.add_argument("--max-grad-norm", type=float, default=DEFAULTS["max_grad_norm"])
    ap.add_argument("--dry-run", action="store_true", help="print the plan and any existing split counts without loading a model")
    return ap


def main(argv=None) -> int:
    return run(build_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
