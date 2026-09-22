# Extraction and fine-tuning

These tools extract classifier verdicts and internal representations, fine-tune
Llama-Guard-3, and compare the resulting classifier with its base model.

For a complete experiment connecting policy labels, LG3 extraction, verdict correction,
and monitoring, follow [Bring your policy to Llama-Guard-3](../../../docs/bring-your-policy.md).

For extraction, follow [steps 1–2](#1-prepare-the-input-rows). To also fine-tune LG3
and compare its verdicts with the base model, continue with
[steps 3–4](#3-fine-tune-llama-guard-3).

The [rebuild guide](../rebuild/README.md) prepares input rows using the
[published policy labels](../../../labels/README.md). For the project overview,
see the [root README](../../../README.md).

## Setup

Run from the repository root in an activated Python environment:

```bash
python -m pip install -e ".[gpu]"
```

The extraction commands default to CUDA; fine-tuning explicitly uses CUDA and
bfloat16. Configure Hugging Face access for the model weights. Beaver additionally
requires `safe_rlhf` from [PKU-Alignment's source repository](https://github.com/PKU-Alignment/safe-rlhf),
following its installation instructions.
For the paper's experimental environment, see Appendix D.3.

Joining and profiling run on CPU with the base installation (`pip install -e .`).
Each command has `--help`; `--dry-run` previews paths and settings.

## 1. Prepare the input rows

Use `work/policy_items.jsonl` produced by
[rebuild step 2](../rebuild/README.md#2-apply-the-published-labels-and-select-the-target),
or supply your own JSONL file with one item per line:

```json
{"item_id":"example_0","set_id":"eval","prompt":"...","response":"...","ground_truth":1}
```

Use unique item IDs and string prompt/response fields; empty responses are allowed.
`ground_truth` holds the selected target label: **0 = safe,
1 = unsafe**. Preserve row order throughout extraction and comparison. `set_id`
is optional for extraction; profiling gives separate summaries for `eval`, `pool`,
and `drift:*` sets.

Join labels to text by `item_id`, then assign
the desired label to `ground_truth` in a new input file:

| Source | Label to select for policy-based evaluation/training |
|---|---|
| Output of `rcv.rebuild.gen_rubric_ystar` | `rubric_ystar` |
| WildGuardMix policy-label map | `new_label` |
| PKU policy-label map | `rubric_ystar` |

After `label_map apply`, set `ground_truth` to the chosen field above. In rebuilt
WildGuardMix it initially contains the upstream human label; PKU text rows have no
`ground_truth`. The extractors validate IDs, binary targets, and text fields before inference.

## 2. Extract verdicts and representations

Use a new output directory for each run, including retries after an interruption;
extraction does not resume. Chunks are checked for consistent indices, shapes,
and row counts before final assembly.

```bash
python -m rcv.extraction.lg3_extract \
  --input work/policy_items.jsonl \
  --output work/base/lg3.items.jsonl --embeddings work/base/lg3.Z.npy
```

For the other classifiers, use `wg_extract` or `beaver_extract` with the same three
arguments and distinct output paths.

| Classifier | JSONL additions | Representation |
|---|---|---|
| LG3 | `lg3_pred`, `lg3_score`, `lg3_agreement` | `(N, 4096)` at the decision context's last token |
| WildGuard | `wg_pred`, `wg_score`, `wg_agreement`, `wg_verdict_text` | `(N, 4096)` at the harmful-response decision position |
| Beaver | `beaver_pred`, `beaver_cost`, `beaver_agreement` | `(N, 5120)` under the T_xml template |

Beaver assigns unsafe (`1`) at `beaver_cost ≥ 3.0`, as specified in the paper's
[Appendix D.1](https://arxiv.org/html/2608.14089#A4.SS1).

Representations are float32 and follow JSONL row order. Verdict 1 means unsafe.
LG3/WildGuard scores use their two verdict-token logits; Beaver costs are not
probabilities. A sibling `.stats.json` file records counts and run metadata.

Check output row count and order, embedding shape, finite values, and agreement
against the selected labels. LG3 and WildGuard can finish with placeholder rows:
check `n_err == 0` (LG3), `n_degenerate == 0` (WildGuard), and no `z_error` rows.
For WildGuard, review any `n_parse_fallback > 0`. Keep the stats with the outputs.

## 3. Fine-tune Llama-Guard-3

Prepare `work/ft-corpus/ft.jsonl`, `val.jsonl`, and `retention.jsonl` using the input
schema above. Record the split IDs and target labels. Each split must be nonempty,
with unique IDs; training items must be separate from validation and retention items.

```bash
python -m rcv.extraction.finetune \
  --seed 42 --corpus-dir work/ft-corpus --out work/ft-run
```

Defaults use LoRA rank 16, alpha 32, dropout 0.05, learning rate 2e-5, one epoch,
micro-batch size 16, and at most 30 optimizer steps. Keep `--grad-accum 1` for this recipe.
Training runs for the configured epoch or step limit, without early stopping.

Alternatively, supply both `--rebuild-universe` and `--distribution` to materialize
the corpus from item-ID lists. The distribution contains `splits` with `ft`, `val`, `retention`, and
optional `family` lists, plus an optional `labels` mapping from item ID to target.
Use a new corpus directory. The optional `family` split is not used in training or evaluation.

Use an empty output directory. The command writes `adapter/` and `ft_manifest.json`,
including the realized step count and acceptance readouts. Exit 0 with
`binding_pass: true` means acceptance; exit 2 means gate rejection, with the adapter
retained for inspection. The gate checks validation change, training/validation
gap, score degeneracy, and retention regressions. Non-finite loss or readout scores
stop the run.

## 4. Compare base and fine-tuned model outputs

Use the same ordered comparison inputs and label axis as the base extraction:

```bash
python -m rcv.extraction.lg3_extract \
  --adapter work/ft-run/adapter --input work/policy_items.jsonl \
  --output work/ft-extract/lg3ft.items.jsonl --embeddings work/ft-extract/lg3ft.Z.npy
```

Check the extraction as in step 2, then package the base JSONL's fields into the
NPZ reference expected by the comparison tools:

```bash
python - <<'PY'
import json
import numpy as np

with open("work/base/lg3.items.jsonl", encoding="utf-8") as f:
    rows = [json.loads(line) for line in f if line.strip()]
fields = {"verdict": "lg3_pred", "clf_score": "lg3_score",
          "ystar": "ground_truth", "agreement": "lg3_agreement"}
arrays = {key: np.asarray([r[field] for r in rows],
                         dtype=np.float64 if key == "clf_score" else np.int64)
          for key, field in fields.items()}
np.savez("work/base/lg3.arrays.npz", **arrays)
PY

python -m rcv.extraction.build_joined \
  --pulled work/ft-extract --base-items work/base/lg3.items.jsonl \
  --base-arrays work/base/lg3.arrays.npz --out-dir work/ft-joined

python -m rcv.extraction.diff_profile \
  --base-items work/base/lg3.items.jsonl --base-arrays work/base/lg3.arrays.npz \
  --ft-arrays work/ft-joined/lg3ft.arrays.npz --out work/comparison
```

`build_joined` checks item/set order, binary labels and verdicts, agreement, finite
scores, and embedding shape and dtype. It writes joined arrays, items, embeddings,
and a hash report.

`DIFF_PROFILE.json` reports flips, adherence, unsafe recall, and false-positive rate.
An absent target class produces `NaN` for its rate. To separate training/selection
items, pass `--exclusion` with a JSON object containing `ft_train_ids` and
`ft_test_ids` lists from the run. The reported `held_out_remainder` contains IDs
absent from both lists.
