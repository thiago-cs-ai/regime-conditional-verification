# Published policy labels

These are the frozen GPT-5-Nano judgments used in
[arXiv:2608.14089v1](https://arxiv.org/abs/2608.14089v1). Rebuild the source corpora
and join these maps by `item_id` to recover the recorded policy targets. No API
call or new judge sampling is needed.

| Map | Coverage | Policy-label field |
|---|---|---|
| [`native_pku_labels.jsonl`](native_pku_labels.jsonl) | 16,422 responses from 8,211 PKU test prompts | `rubric_ystar` |
| [`wgmix_labels.jsonl`](wgmix_labels.jsonl) | 20,199 items: 1,709 eval, 9,999 pool, 8,491 drift | `new_label` |

Each JSONL row contains exactly `item_id`, the policy-label field, and `rule_fired`.
Labels use **0 = safe, 1 = unsafe**. `rule_fired` retains the saved diagnostic tag;
`2.2_commitment` identifies the added policy clause studied in Appendix B.2.
`none` is a historical normalized tag, not a separate safety label.

## Use the maps

Follow the [rebuild guide](../src/rcv/rebuild/README.md) to reconstruct text rows,
join labels, and select `ground_truth` for extraction. To validate the maps:

```bash
python -m rcv.rebuild.label_map gate --corpus native --path labels/native_pku_labels.jsonl
python -m rcv.rebuild.label_map gate --corpus wgmix --path labels/wgmix_labels.jsonl
shasum -a 256 -c labels/SHA256SUMS
```

Run these commands from the repository root. Join by `item_id` to align labels with text.
PKU IDs encode upstream test-prompt position and response slot; WGMIX IDs depend
on text and position within each carved file. The rebuilders use pinned revisions
and compare corpus hashes with the recorded references.

The paper's 11,708-item WildGuardMix frame is eval plus pool. The map also includes
all 11 drift families for the maintenance study.

## Provenance

The maps preserve the original saved labels and rule tags. PKU labels match all
16,422 entries in the submitted supplement's `labels/pku_labels.csv`. WGMIX labels match its
`labels/wgmix_big_labels.csv` and `labels/wgmix_eval_labels.csv`; the full universe
also matches the saved judge records.

Checked against the rebuilt native labels, the stored policy labels recover the
paper's agreement statistics: Cohen's κ = 0.838 on PKU and 0.766 on WGMIX eval plus
pool.

The rebuilders pin PKU revision `9421ffafec3fa40a1f1a7d567b4d525079477ecb` and WGMIX
revision `d29c47f41c8b51348b5c8e8c81c039b3132b66d1`.
[`SHA256SUMS`](SHA256SUMS) records the hashes of these published exports.

## Data terms

The repository's MIT license covers its code and documentation. These maps are
released as research data under the following dataset-specific terms:

- **PKU map:** [CC-BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/).
  Based on [PKU-SafeRLHF](https://huggingface.co/datasets/PKU-Alignment/PKU-SafeRLHF),
  Ji et al., *PKU-SafeRLHF: Towards Multi-Level Safety Alignment for LLMs with Human
  Preference* (2024), PKU-Alignment. The policy labels and rule tags are RCV annotations.
- **WildGuardMix map:** [ODC-BY 1.0](https://opendatacommons.org/licenses/by/1-0/).
  Contains information from [WildGuardMix](https://huggingface.co/datasets/allenai/wildguardmix),
  which is made available under the [ODC Attribution License](https://opendatacommons.org/licenses/by/1-0/).
  Source: Han et al., *WildGuard: Open One-Stop Moderation Tools for Safety Risks, Jailbreaks,
  and Refusals of LLMs* (2024), Allen Institute for AI. The policy labels and rule tags
  are RCV annotations. Follow the upstream research-use conditions and
  [Ai2 Responsible Use Guidelines](https://allenai.org/responsible-use).

RCV annotations: Thiago Sandoval and Ufuk Topcu, 2026. Corpus text and human
annotations are available from the official releases under their access conditions.
