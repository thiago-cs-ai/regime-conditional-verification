# Rebuild datasets and join policy labels

Use these tools to reconstruct WildGuardMix or PKU-SafeRLHF, join policy labels
to corpus text, and prepare inputs for [representation extraction](../extraction/README.md).
The [root README](../../../README.md) introduces RCV and the shipped experiment records.

The workflow is **rebuild → join the published labels → select the target label**.
The [published maps](../../../labels/README.md) supply the recorded targets and
clause annotations without new judge calls. You can also
[generate new policy labels](#optional-generate-new-policy-labels).

## Setup

Run from the repository root with Python 3.11 or later:

```bash
python -m pip install -e ".[rebuild]"
mkdir -p work
```

Rebuilding runs on CPU and downloads the upstream datasets. WildGuardMix requires
approved Hugging Face access. Use `--offline` with a populated Hugging Face cache;
each command provides `--help`. Dataset terms are linked in the
[root README](../../../README.md#license).

## 1. Rebuild one dataset

Choose one route below. Both set up `work/items.jsonl` for the remaining steps.
Use a fresh output directory. Both commands use pinned upstream revisions and
check the generated files against reference SHA-256 hashes. Exit 0 with
`all_match: true` confirms a match; a checksum mismatch returns exit 1 and leaves
the files and report for inspection.

**PKU-SafeRLHF:** both responses to each test prompt.

```bash
python -m rcv.rebuild.pku \
  --out-dir work/pku --report work/pku/rebuild-report.json
```

Select the rebuilt rows and published label map:

```bash
corpus=native
label_map=labels/native_pku_labels.jsonl
cp work/pku/native_pku_test.jsonl work/items.jsonl
```

The output contains 16,422 rows from 8,211 prompts, in prompt order with response
0 before response 1. IDs have the form `PKU_test_<index>_r0` or `_r1`. Blank responses
are retained. `native_pku_test.sidecar.jsonl` contains the upstream human labels
(`y_star_native`: 1 = unsafe, 0 = safe), harm metadata, and `group_key` for grouping
responses to the same prompt. Keep those groups together when creating data splits.

**WildGuardMix:** evaluation, pool, and drift candidates.

```bash
python -m rcv.rebuild.wgmix \
  --out-dir work/wgmix --report work/wgmix/rebuild-report.json
```

Select the evaluation rows and published label map:

```bash
corpus=wgmix
label_map=labels/wgmix_labels.jsonl
cp work/wgmix/eval.jsonl work/items.jsonl
```

The output has 1,709 evaluation rows, 9,999 pool rows, and 8,491 rows across
11 `drift_candidates/*.jsonl` files. The published map covers all of them; repeat
the joining steps with separate paths for pool or drift files.
The paper's 11,708-item frame combines eval and pool.
IDs (`it_` plus 12 hex digits) depend on text and position within each rebuilt file.
Preserve these IDs when selecting or combining rows. `ground_truth` initially holds
the upstream human label; step 2 replaces it with the policy label.

## 2. Apply the published labels and select the target

The map tools validate PKU/WildGuardMix item IDs. For your own IDs, use the
[hands-on guide's `prepare` command](../../../docs/bring-your-policy.md#use-your-own-policy-and-traffic).

The selected map uses `rubric_ystar` for PKU (`--corpus native`) or `new_label`
for WildGuardMix. Both encode **0 = safe, 1 = unsafe**. Check the maps and join
the selected one to the rebuilt rows:

```bash
shasum -a 256 -c labels/SHA256SUMS
python -m rcv.rebuild.label_map gate --corpus "$corpus" \
  --path "$label_map" --report work/label-map-report.json
python -m rcv.rebuild.label_map apply --corpus "$corpus" \
  --labels "$label_map" --items work/items.jsonl --out work/labeled-items.jsonl
```

`gate` validates the map's schema and IDs. `apply` joins by `item_id`, preserves
input order, and requires a label for every input item. A map may cover more items
than the selected input. Set `ground_truth` to the joined policy label for extraction:

```bash
python - "$corpus" <<'PY'
import sys
from rcv.rebuild._jsonl import read_jsonl, write_jsonl

field = {"native": "rubric_ystar", "wgmix": "new_label"}[sys.argv[1]]
rows = read_jsonl("work/labeled-items.jsonl")
for row in rows:
    row["ground_truth"] = row[field]
    row["set_id"] = "eval"
write_jsonl("work/policy_items.jsonl", rows)
PY
```

This example uses evaluation rows; use `pool` or `drift:<family>` as `set_id` for
those sets. Use new output paths when repeating these steps.
Continue with [extraction step 2](../extraction/README.md#2-extract-verdicts-and-representations)
using `work/policy_items.jsonl`; that guide includes GPU setup.

## Optional: generate new policy labels

To apply the [rubric](../../../judge_prompt.txt) to your own items or study the
labeling step, set `OPENROUTER_API_KEY` in your environment. The command sends
prompt–response pairs to OpenRouter using `openai/gpt-5-nano` by default and incurs
API charges.

```bash
python -m rcv.rebuild.gen_rubric_ystar \
  --input work/items.jsonl --out work/labels.raw.jsonl
```

Add `--policy-file path/to/policy.txt` to use your own written policy. The labeler
supplies the JSON output format and records compliance as `rubric_ystar=0` and a
violation as `rubric_ystar=1`. Custom-policy records use `rule_fired: "none"`.
For the complete labeling, extraction, and RCV experiment, follow
[Bring your policy to Llama-Guard-3](../../../docs/bring-your-policy.md).

Input rows need unique, nonempty string IDs and string `prompt`/`response` fields;
blank responses are allowed. Rerunning the same command skips completed IDs and
retries unfinished items, checking the saved model and prompt version.
Use a new output file when changing the input text or model.

Custom-policy resume also checks the policy's SHA-256, wrapper version, and each
requested item's text digest, including failed attempts. Changing those inputs
requires a new output file. Without `--policy-file`, the frozen v1 rubric is used.

Exit 0 means all selected IDs are labeled; exit 1 means labeling failures;
exit 2 means a cost stop with pending items, or invalid arguments. The summary
reports completed, failed, and pending counts. `--limit` selects an input prefix.
`--max-cost` checks estimated spending after each batch, using this invocation's
reported usage and fixed GPT-5-nano rates, including when another model is selected.

For PKU/WildGuardMix items, normalize the label field for the chosen corpus,
then export the map:

```bash
python - "$corpus" <<'PY'
import sys
from rcv.rebuild._jsonl import read_jsonl, write_jsonl

rows = read_jsonl("work/labels.raw.jsonl")
if sys.argv[1] == "wgmix":
    for row in rows:
        row["new_label"] = row.pop("rubric_ystar")
write_jsonl("work/labels.normalized.jsonl", rows)
PY

python -m rcv.rebuild.label_map export --corpus "$corpus" \
  --src work/labels.normalized.jsonl --out work/labels.jsonl
label_map=work/labels.jsonl
```

Export selects the map fields and retains the last non-null label for each ID.
The raw output retains model/version metadata, usage, and the original rule tag
(`rule_fired_raw`). The labeler normalizes recognized rules into `rule_fired`;
safe verdicts and unknown rules receive `none`.

For these corpus IDs, use `label_map` in
[step 2](#2-apply-the-published-labels-and-select-the-target) with new output paths.
For your own IDs, pass the raw labels to the
[example's `prepare` command](../../../docs/bring-your-policy.md#use-your-own-policy-and-traffic)
with `--label-field rubric_ystar`.
