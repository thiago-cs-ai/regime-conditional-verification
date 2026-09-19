# Regime-Conditional Verification (RCV)

[![CI](https://github.com/thiago-cs-ai/regime-conditional-verification/actions/workflows/ci.yml/badge.svg)](https://github.com/thiago-cs-ai/regime-conditional-verification/actions/workflows/ci.yml)
[![arXiv](https://img.shields.io/badge/arXiv-2608.14089-b31b1b.svg)](https://arxiv.org/abs/2608.14089)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**Your policy on the safety classifier you already run.**

Research code accompanying the RCV paper: the study implementation, acceptance tests, frozen policy
labels, dataset rebuilders, and a replay of five published maintenance chains.

Safety classifiers can disagree with a deployer's policy and lose accuracy as traffic changes. RCV
fits two small probes, one for safe verdicts and one for unsafe verdicts, to estimate whether each
decision agrees with the policy. It corrects likely mistakes and uses the same scores to detect drift
without incoming labels. After an alarm, an audit supplies labels for a probe update; a held-out check
determines whether the update restores the required recall and over-blocking rates. Updates that fail
within the label budget trigger escalation. RCV requires access to the classifier's internal
representations.

In our study, the [written policy](judge_prompt.txt) treats an assistant's agreement to help with a
harmful request as a violation, even before the response contains harmful content. Llama-Guard-3 and
Beaver were not trained to enforce that clause; WildGuard was. GPT-5-Nano applied the policy to
prompt–response pairs, producing the labels used to train and evaluate the probes.

Across three classifiers and two datasets, RCV improves agreement with the policy labels in all six
combinations, catching 29–81% of previously missed unsafe items without modifying the classifier. In
the maintenance experiment on Llama-Guard-3 with WildGuardMix, probe updates pass the repair gate in
79 of 100 drift episodes within the deployed label budget.

[Overview](https://rcv.tsandoval.com) · [Hands-on guide](docs/bring-your-policy.md) · [Paper](https://arxiv.org/abs/2608.14089) —
[Thiago Sandoval](https://www.linkedin.com/in/thiago-cs-ai/),
[Ufuk Topcu](https://ae.utexas.edu/person/ufuk-topcu/) (UT Austin)

## Run the replay

Requires Python 3.11 or later. From the repository root, in an activated virtual environment:

```bash
pip install -e .
python -m rcv demo
```

The command reads the saved records from five published maintenance chains and prints their outcomes.
Once installed, it runs offline without datasets, a GPU, or an API key. It does not rerun the
experiments—but their input-rebuild tools are here if you want to explore them: see [Rebuild
experiment inputs](#rebuild-experiment-inputs).

Each row reports one harm family introduced during a chain. `alarm 0.50` and `alarm 0.95` identify
the correctness-score boundary whose monitor fired. The recall and over-blocking values come from the
final attempted update's held-out gate. `repaired` means that update passed the gate; `no alarm` means
the monitor did not fire. Chain 42 contains four accepted repairs.

```
$ python -m rcv demo
chain 42   4 accepted repairs · 2,700 labels
  mentalhealth   alarm 0.95 · audit 300 · recall 0.794 · over-block 0.047 · repaired
  sensinfo       alarm 0.95 · audit 300 · recall 0.774 · over-block 0.025 · repaired
  cyber          alarm 0.50 · audit 600 (2 attempts) · recall 0.746 · over-block 0.035 · repaired
  stereotypes    alarm 0.95 · audit 300 · recall 0.796 · over-block 0.058 · repaired
  fraud          no alarm
```

The 2,700-label total includes 1,500 audit labels and four 300-label gate blocks. The [record
manifest](src/rcv/demo/records/MANIFEST.md) documents all five chains and their file hashes. These
records stop at escalation; the paper reports the fine-tuning experiment separately. View the [full
replay](https://rcv.tsandoval.com/#demo).

## Verify Table 1

```bash
python smoke/print_table1.py
```

This command recomputes the means and sample standard deviations in the paper's Table 1 from 60
shipped per-seed records. It requires all six classifier–dataset combinations and all ten seed
identities, checks each saved aggregate, and verifies the rounded values against
[arXiv:2608.14089v1](https://arxiv.org/abs/2608.14089v1). See
[`results/README.md`](results/README.md) for the metrics and provenance.

## Bring your policy to Llama-Guard-3

Follow the [hands-on guide](docs/bring-your-policy.md) to compare original and RCV-corrected
verdicts, monitor a stream, and inspect repair decisions and label use. Start with the published
policy labels, then substitute your own policy and traffic. LG3 extraction uses a GPU; RCV
fitting and monitoring run on CPU.

## Inspect the implementation

| What it enables | Start here |
|---|---|
| **Policy adaptation.** Correct likely verdict errors without retraining the classifier. | [`estimator.py`](src/rcv/estimator.py), [`flip.py`](src/rcv/flip.py) |
| **Label-free drift detection.** Monitor correctness scores using jointly calibrated alarm thresholds. | [`belief_bank.py`](src/rcv/belief_bank.py) |
| **Budgeted repair and escalation.** Audit alarms, evaluate probe updates, and resume monitoring or escalate. | [`loop.py`](src/rcv/loop.py) |
| **Representation extraction and fine-tuning.** Align representations with verdict decisions and check adapter performance and retention. | [`lg3_extract.py`](src/rcv/extraction/lg3_extract.py), [`finetune.py`](src/rcv/extraction/finetune.py) |
| **Traceable experiments.** Record inputs, code, settings, and environment; withhold aggregates when a seed fails. | [`runner.py`](src/rcv/runner.py) |

## Run tests and lint checks

Requires Python 3.11 or later.

```bash
python -m pip install -e ".[dev]"
python -m pytest
python -m ruff check .
```

For a pinned CPU environment, see [`requirements.lock`](requirements.lock).

## Rebuild experiment inputs

The [published policy-label maps](labels/README.md) are included. Rebuild the source corpora and join
the maps to recover the recorded targets without querying the judge:

1. [Rebuild datasets and prepare policy labels](src/rcv/rebuild/README.md). Reconstruct the
   corpora and join the frozen labels by item ID. Rebuilding runs on CPU with the `rebuild`
   extra. WildGuardMix requires approved Hugging Face access.
2. [Extract representations and fine-tune](src/rcv/extraction/README.md). Run classifiers on
   the prepared rows, validate their outputs, and optionally fine-tune Llama-Guard-3. These tools
   use the `gpu` extra and require access to the model weights.

Each guide provides installation, commands, input contracts, and output checks.
The paper's experimental environment is documented in its Appendix D.3.

Extracted representations and historical experiment split assets are not included. The guides also
cover optional generation of new policy labels through OpenRouter; that creates a new labeling run
and may produce different targets.

## Citation

```bibtex
@article{sandoval2026rcv,
  title   = {Regime-Conditional Verification: Correctness Estimation for Adapting and Monitoring Safety Classifiers},
  author  = {Sandoval, Thiago and Topcu, Ufuk},
  journal = {arXiv preprint arXiv:2608.14089},
  year    = {2026}
}
```

## License

The code and documentation are licensed under the [MIT License](LICENSE). Published label maps have
[separate data terms and attribution](labels/README.md#data-terms).

WildGuardMix ([ODC-BY 1.0](https://opendatacommons.org/licenses/by/1-0/), gated) and PKU-SafeRLHF
([CC-BY-NC-4.0](https://creativecommons.org/licenses/by-nc/4.0/)) are third-party datasets with
their own terms.
