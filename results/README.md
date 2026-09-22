# Published Table 1 records

This directory contains the per-seed records behind Table 1 of
[arXiv:2608.14089v2](https://arxiv.org/abs/2608.14089v2). The records cover three safety
classifiers on PKU-SafeRLHF and WildGuardMix, with ten fixed seeds for each combination. They contain
numeric results and provenance; they contain no corpus text.

## Verify the table

From the repository root:

```bash
python smoke/print_table1.py
```

The command requires all six combinations and all ten seed identities. It recomputes each mean and
sample standard deviation from the 60 seed records, checks the six saved aggregates, and compares the
rounded values with [`table1_reference.json`](table1_reference.json), which transcribes the table in
the paper. It reads saved experimental outputs; it does not rerun classifier inference or probe
training.

The table reports:

- **Raw adherence:** the share of evaluation items for which the classifier verdict matches the
  policy label.
- **Corrected adherence:** the same share after RCV applies the flip rule.
- **Caught share:** among policy-unsafe items the classifier passes, the share RCV flips to unsafe.

## Provenance

The files under [`final_steering_20260728/steering/`](final_steering_20260728/steering/) were copied
without modification from the paper's code and data supplement. Every seed record preserves the
configuration, software environment, frame digest, configuration digest, and code digest recorded
when the experiment ran. The aggregate beside each set of seeds preserves the same provenance.

`code_sha256` identifies the experimental `src/rcv/*.py` files that produced these results. It is
historical provenance and is not expected to match the evolving public checkout. Editing current
source comments does not change a saved result or its recorded digest.

[`SHA256SUMS`](SHA256SUMS) pins the 66 JSON files as imported. Check them from the repository root:

```bash
shasum -a 256 -c results/SHA256SUMS
```
