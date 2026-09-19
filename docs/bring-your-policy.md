# Bring your policy to Llama-Guard-3

**Use RCV to adapt verdicts, detect drift, and evaluate repairs.**

Run a local experiment that compares Llama-Guard-3's original verdicts with RCV's
corrected verdicts, then processes a stream and records alarms, audit decisions,
probe updates, and label use. Start with the published policy and labels; the
[custom-policy route](#use-your-own-policy-and-traffic) substitutes your own inputs.

## Setup

Use Python 3.11 or later. Run commands from the repository root in an activated
environment:

```bash
python -m pip install -e ".[rebuild,gpu]"
mkdir -p work/policy
```

Rebuilding uses CPU and downloads WildGuardMix, which requires approved Hugging
Face access. LG3 extraction requires access to its model weights and a suitable
CUDA GPU; see [extraction setup](../src/rcv/extraction/README.md#setup).
RCV fitting and monitoring run on CPU after extraction.

The commands below select at most 6,000 responses for extraction. Without the two
size limits, the source files contain 18,490 responses before shared-family
exclusion. The worked example reuses the shipped labels without judge API calls.

## 1. Prepare the policy targets and examples

The [published policy](../judge_prompt.txt) includes an assistant's agreement to
help with a harmful request as a violation. The [frozen labels](../labels/README.md)
record the judge's targets under that policy. Rebuild their source text:

```bash
python -m rcv.rebuild.wgmix \
  --out-dir work/policy/wgmix --report work/policy/wgmix/rebuild-report.json
shasum -a 256 -c labels/SHA256SUMS

python examples/bring_your_policy.py prepare \
  --baseline work/policy/wgmix/pool.jsonl \
  --stream work/policy/wgmix/drift_candidates \
  --labels labels/wgmix_labels.jsonl --label-field new_label \
  --max-baseline-items 3000 --max-stream-items 3000 \
  --exclude-shared-groups \
  --out work/policy/items.jsonl
```

The rebuild checks its files against reference hashes; exit 0 with `all_match: true`
confirms a match. The preparation command reports the retained counts and joins
labels by item ID, setting `ground_truth` to **0=safe, 1=unsafe**.

The pool supplies baseline examples. Drift files form the stream in filename
order, retaining each file's row order; the size limit takes a prefix of that
stream. Baseline families that also appear in the selected stream are excluded.
The baseline cap keeps whole groups of related items, identified by normalized
prompts and optional `group_id` links.

This is a new experiment using published labels and an assembled stream. The
runner creates its own baseline splits; it does not recreate the historical
paper splits or the five published maintenance chains. Omit both size limits to
use the full pool and stream, or choose your own limits before extraction.

## 2. Extract verdicts and representations

```bash
python -m rcv.extraction.lg3_extract \
  --input work/policy/items.jsonl \
  --output work/policy/lg3/items.jsonl \
  --embeddings work/policy/lg3/Z.npy
```

The extractor adds LG3's verdict, score, and agreement with the policy target to
each JSONL row. `Z.npy` holds the corresponding internal representations;
`items.stats.json` records extraction statistics and the embedding checksum.
The next step checks their alignment and rejects failed extraction rows.
Use the complete prepared input; set size limits in `prepare` rather than
truncating this extraction with `--max-rows`.

## 3. Fit RCV and compare verdicts

```bash
python examples/bring_your_policy.py run \
  --inputs work/policy/items.jsonl \
  --extracted work/policy/lg3/items.jsonl \
  --embeddings work/policy/lg3/Z.npy \
  --config examples/bring_your_policy.yaml \
  --out work/policy/run
```

The runner splits baseline families into approximately 45% fitting, 35%
calibration, and 20% evaluation. It fits and calibrates a separate correctness
probe for each classifier verdict. The terminal summary reports:

- **Raw adherence:** the fraction of held-out verdicts matching the policy targets.
- **Corrected adherence:** that fraction after RCV's verdict corrections, with
  the change in percentage points.
- **Caught share:** among policy-unsafe examples LG3 passes, the fraction RCV
  changes to unsafe, alongside the number of examples in that population.

These measurements describe the held-out baseline before stream maintenance.
The same command then monitors the stream and evaluates repairs when alarms fire.

The run directory contains:

```text
frame.npz              Aligned inputs used by RCV
config.yaml            Resolved study settings and frame checksum
input_manifest.json    Hashes of the adapter's inputs and source
results/seed_42.json   Measurements, loop events, label use, and provenance
results/aggregate.json Summary across configured seeds
```

Use fresh output paths for a new preparation, extraction, or study run. A study
refusal is recorded in `results/REFUSALS.json` and returns a nonzero exit status.

## 4. Inspect monitoring and repair decisions

Open `results/seed_42.json`. Its `loop_events` show the sequence:

1. `alarm`: a correctness-score monitor fired.
2. `audit`: the loop sampled labels from subsequent traffic.
3. `acceptance_evaluated`: a cross-fitted audit evaluation measured the proposed
   update's recall and over-blocking.
4. `repair_accepted` or `escalation_demanded`: the update passed, or no attempt
   passed within the configured retries.

An accepted repair is followed by a fresh reference window for monitoring.
`reference_rederived` and `monitor.recalibrations` record that refresh. If too few
eligible items remain, `reference_window_exhausted` records why monitoring ended.
A stream with no alarm has no audit or repair events.

The [example configuration](../examples/bring_your_policy.yaml) uses 500-item
audit windows, up to 318 labels per draw, and two enlarging retries. Acceptance
allows recall to fall by at most five percentage points and over-blocking to rise
by at most five points relative to the initial corrected baseline. This continuous
workflow evaluates updates by cross-fitting the audit families; the published
episode replay uses a separate held-out gate block.

The monitor initially calibrates from baseline evaluation scores. The template's
`stream_length: null` sets its calibration horizon to the selected stream length.
An explicit positive integer selects another horizon; the quiet-horizon
calibration then applies to that interval.

## 5. Account for labels

The summary separates initial labels for fitting, calibration, and evaluation
from additional labels consumed by maintenance. The total and per-update ledger
are in `total_oracle_labels_spent` and `change_log`. Audit events show cumulative
batch sizes during retries, so adding those event counts would double-count labels.

This offline experiment has the stream's policy labels available in advance.
The monitor triggers alarms from classifier verdicts and correctness scores;
the loop consumes targets for audits after an alarm. Its audit-label count
describes the experiment's label use, not the API calls already spent labeling
the input data.

## 6. Continue from escalation

Follow [Fine-tune Llama-Guard-3](../src/rcv/extraction/README.md#3-fine-tune-llama-guard-3)
to prepare training, validation, and retention sets, then evaluate an adapter.
The loop records escalation; fine-tuning is a separate command. The extraction
guide also shows how to [compare base and fine-tuned outputs](../src/rcv/extraction/README.md#4-compare-base-and-fine-tuned-model-outputs).
After changing the classifier, extract its new outputs and fit/calibrate RCV
again before another monitored run.

## Use your own policy and traffic

Supply `work/custom/baseline.jsonl` with baseline examples and
`work/custom/stream.jsonl` with subsequent traffic in processing order. These are
your input files; they are not included in the repository. Each line has this shape:

```json
{"item_id":"conversation-001","prompt":"...","response":"...","group_id":"conversation-001"}
```

IDs must be unique across both files. Prompts must be nonblank strings; responses
may be empty. `group_id` is optional and links related examples. Matching normalized
prompts are grouped automatically. Keep baseline and stream groups separate.

Write your policy in `work/custom/policy.txt`. For example, a narrow disclosure
policy could say:

```text
A response violates this policy if it reveals a customer's private contact
information or agrees to provide it. Refusals and general discussion of
privacy comply with the policy.
```

The labeler supplies the output format and maps compliance to 0 and violations
to 1. Set `OPENROUTER_API_KEY` in your environment. These commands send the
prompt–response pairs to OpenRouter and incur API charges:

```bash
python -m rcv.rebuild.gen_rubric_ystar \
  --policy-file work/custom/policy.txt \
  --input work/custom/baseline.jsonl --out work/custom/baseline.labels.jsonl

python -m rcv.rebuild.gen_rubric_ystar \
  --policy-file work/custom/policy.txt \
  --input work/custom/stream.jsonl --out work/custom/stream.labels.jsonl

python examples/bring_your_policy.py prepare \
  --baseline work/custom/baseline.jsonl --stream work/custom/stream.jsonl \
  --labels work/custom/baseline.labels.jsonl work/custom/stream.labels.jsonl \
  --label-field rubric_ystar --out work/custom/items.jsonl
```

Reruns resume compatible labeling outputs. See the [labeler reference](../src/rcv/rebuild/README.md#optional-generate-new-policy-labels)
for resume checks, spending estimates, and exit codes. If you already have policy
annotations, skip the judge commands and pass your map paths and target column to
`prepare` instead. The example accepts your own string IDs.

Run extraction and RCV with separate custom output paths:

```bash
python -m rcv.extraction.lg3_extract \
  --input work/custom/items.jsonl \
  --output work/custom/lg3/items.jsonl --embeddings work/custom/lg3/Z.npy

python examples/bring_your_policy.py run \
  --inputs work/custom/items.jsonl \
  --extracted work/custom/lg3/items.jsonl --embeddings work/custom/lg3/Z.npy \
  --config examples/bring_your_policy.yaml --out work/custom/run
```

The grouped fitting, calibration, and evaluation splits each need correct and
incorrect decisions for both classifier verdicts; row count alone does not
establish that coverage. For a smaller stream, adjust `audit_sampling_window`,
`audit_budget`, and `post_repair_reference_window` together in a copy of the
configuration. Keep the acceptance tolerances tied to your policy requirements.
The default 500-item audit and reference windows assume enough subsequent
traffic to evaluate a repair and resume monitoring.
