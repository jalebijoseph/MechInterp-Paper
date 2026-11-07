# Full saved evidence export

This export preserves exact bytes from 22,870 captured artifacts, including 7,774 NPZ traces. The trace files use Git LFS. All saved prompts, subject responses, measured traces, factual references, grader judgments, calibration predictions, retry histories, and run provenance in the selected directories are retained. The capture window was 2026-11-07 23:20:29–23:20:41 UTC, or 18:20:29–18:20:41 Eastern time.

The earlier complete 8B and 32B Exp0 datasets remain in their existing `results/` directories. The earlier partial 32B suite snapshot is historical evidence. This directory adds the complete original 32B suite and corrected reruns.

| Run group | Captured files | NPZ traces | State at capture |
|---|---:|---:|---|
| Original Qwen3-32B experiments 1–4 | 5,982 | 5,802 | Target suite completed; includes 5,776 target traces and 26 smoke traces |
| Corrected Qwen3-32B follow-up | 2,011 | 1,972 | All four target portions completed |
| GPT-OSS Exp4 grader | 5,111 | 0 | All seven attempts terminal; final report has 45 unresolved judgments |
| Nemotron Exp4 grader | 9,518 | 0 | All four attempts terminal; final report has 53 unresolved judgments |
| Nemotron corrected Exp2 grader | 235 | 0 | Snapshot of grading in progress |
| Focused causal handoff | 13 | 0 | Snapshot of waiting controller; target traces pending |

## Contents

The original suite includes its complete manifest, 17 jobs, 5,800 turns, all target traces, smoke outputs, aggregate Exp4 report, configurations, validation reports, logs, model/lens revisions, and source-parity evidence.

The corrected follow-up includes 2,540 turns, all 1,972 referenced traces, summaries and validation reports for Exp1, Exp2, Exp3, and the NARCBench replay. Its 600 factual-review responses and 480 knowledge probes are preserved separately in the original Exp2 artifacts. The frozen external inputs, independent-unit primary family, endpoint identities, source configuration, and failed OLMo grader calibration are included. That calibration's raw replies are embedded in its predictions file.

The grading exports include every saved raw receipt, final model reply, usage field, failed judgment, calibration prediction, gate, scored response, unresolved response, and summary from the captured attempts. Earlier attempts and their transport/fallback histories remain attributable to their original source versions. Selected scientific source files and frozen fixtures are copied byte-for-byte and can be checked against the recorded hashes. Model weights, fitted lens weights, environments, dependency caches, lock files, and duplicate archive/source-checkout copies are outside this export.

The live Exp2 and causal groups contain the file census available at capture start. Each file was read once, preserving its exact individual version. These are partial snapshots of active jobs, with capture times and lifecycle metadata in the manifest. Final outputs produced after that window require a subsequent export.

## Integrity and paths

`manifest.json` records each artifact's original absolute path, exported relative path, size, and SHA-256. `SHA256SUMS` provides a standard checksum list. The manifest excludes itself and the checksum list. Local documentation and verification reports are additional export metadata.

Serialized paths inside raw data retain their original values. To resolve one locally, look up its `source_path` in the manifest and use the corresponding `path` below this directory. Relative trace references remain relative to their original `turns.jsonl` directory because the source layout is mirrored. No path or judgment normalization was applied to the evidence.

Fetch the trace contents and verify the export from the repository root.

```bash
git lfs pull
python3 scripts/verify_evidence_export.py --root results/cluster-evidence-20261107
```

The original source layout includes directories named `runs`, which match the repository's ignore rule. The export is staged explicitly from its manifest so every captured artifact is tracked. Model caches and local working runs remain outside the scientific export.

## Reproduce the Exp4 analysis

The analysis can consume the tracked raw files directly.

```bash
python3 scripts/analyze_exp4_specificity.py \
  --nemotron results/cluster-evidence-20261107/qwen3-32b-exp4-nemotron-candidate-20261105/attempt-004/grading/scored_rows.jsonl \
  --gpt-oss results/cluster-evidence-20261107/qwen3-32b-exp4-rescore-v3-20261105/attempt-007/grading/scored_rows.jsonl \
  --output /tmp/reproduced-exp4-specificity.json \
  --report /tmp/reproduced-exp4-specificity.md
```

Use fresh output paths for reviewer-packet regeneration. Human labels remain pending in the separately committed [blinded packet](../qwen3-32b-exp4-analysis-20261107/adjudication/reviewer_packet.html). The research findings remain exploratory under the failed original Exp0 gate.
