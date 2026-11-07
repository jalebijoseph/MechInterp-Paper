# Qwen3 cluster runs and Experiment 0 audit

The cluster helpers preserve the original experiment implementations and sample
sizes. All reported full runs use the original Hugging Face/PyTorch runtime in
BF16 with Jacobian-lens hooks. The optional vLLM adapter was investigated during
setup; it was not used for these full runs.

## Saved results

* `results/qwen3-8b-20261104/`: completed experiments 0–4, including 19 validated
  portions, 5,916 traces, and the combined Experiment 4 primary comparison.
* `results/qwen3-32b-exp0-20261104/`: completed Experiment 0, with 80 core items and
  60 Hi-ToM items, each with a trace.
* `results/qwen3-32b-suite-20261104/`: a dated snapshot of experiments 1–4 while the
  run is still in progress. Its manifest, embedded in `git-trace-inventory.json`,
  records which portions had completed.
  The complete plan is 17 jobs, 5,800 turns, and 5,776 experiment traces. Separate
  smoke traces check all 13 intervention presets on each of two replicas.
* `results/exp0-audit-20261104/`: post-hoc prompt diagnostics and trace-integrity
  checks. These do not change the original Experiment 0 results or gate.

Trace files (`results/**/*.npz`) are stored with Git LFS. With Git LFS installed,
run `git lfs pull` to obtain their contents. The trace format is documented in
`src/jpeers/trace.py`; each experiment's `turns.jsonl` references its trace files.
Model weights, caches, environments, and source-tree copies are excluded.

The [2026-11-07 evidence export](../results/cluster-evidence-20261107/README.md)
adds the complete original 32B suite, corrected follow-up prompts/responses/traces,
and both graders' full calibration and receipt histories. Its manifest preserves
exact source bytes and checksums. The original 32B snapshot above remains a dated
historical snapshot. Active corrected Exp2 grading and the queued causal study
are explicitly marked as partial snapshots in the new export.

`results/artifact-index.json` records source and exported checksums. Exported text
uses portable paths in place of machine-specific paths. Binary trace contents are
unchanged. Where an exported summary reference is resolvable, its checksum is
updated and the original checksum is retained as `source_summary_sha256`.
Historical source hashes identify the implementation used for the run; the cluster
helpers have since been given separate 8B/32B filenames for this repository.

## Interpretation

The original Experiment 0 belief-tracking gate failed for both 8B and 32B.
Experiments 1–4 must therefore be treated as exploratory under the existing
preregistration. An in-progress snapshot is not a final experiment result.

The audit found no incorrect core labels or behavioral scoring discrepancies in
160 saved core responses. Native Hugging Face replay reproduced all 160 strings;
recording enabled/disabled reproduced the same outputs in the 16-case 8B check.
All 280 saved Experiment 0 traces reproduced their recorded measurements.

The audit also found prompt sensitivity. Removing the `Answer:` prefill changed
32B people/contents controls from 0/10 to 10/10 while their false-belief counterparts
stayed 10/10. Those ten controls contain only nine unique prompts. Direct current
belief questions improved agent transfer performance. People transfer controls
continued to fail in most tested formats. These are post-hoc findings requiring
fresh, diverse held-out validation.

Hi-ToM's answer-start noun measurement is a separate limitation: responses begin
with option letters, and generated underscore-prefixed nouns tokenize differently
from the watched bare nouns. Noun-only scoring also merges some colored choices.
Its zero answer-start noun hit rate does not establish absent belief information.

The primary core gate reads the last story token before the question. Removing a
later assistant prefill cannot causally repair that earlier representation.

## Reproduction helpers

* `scripts/run_cluster_suite.py`: full 8B suite, 19 independently validated jobs.
* `scripts/cluster_manager.py`: owns the 8B servers and stops them after the suite.
* `scripts/queue_exp0_32b.py`: waits for the original 8B run and runs pinned 32B
  Experiment 0 on an idle pair of allowed GPUs.
* `scripts/serve_sharded.py`: BF16 32B serving across two GPUs.
* `scripts/run_cluster_suite_32b.py`: 32B experiments 1–4, excluding Experiment 0.
* `scripts/cluster_manager_32b.py`: selects idle physical GPUs 0–3 by UUID, checks
  all intervention presets, runs the suite, and stops only its owned servers.
* `scripts/smoke_32b_suite.py`: verifies finite recorded traces and expected edit
  counts across the 32B workspace band, layers 24–58.

Schedulers require already-running compatible `/generate` and `/info` endpoints:

```bash
python scripts/run_cluster_suite.py --ports 18080 --output-dir runs/8b-suite
python scripts/run_cluster_suite_32b.py --ports 18110 18111 --output-dir runs/32b-suite
```

Use `--prepare-only` to inspect the plan without requests. Reusing the output
directory resumes unfinished work; `--retry-failed` preserves earlier attempts.
Lifecycle managers expect a run root containing a source checkout at `source/`
and a `suite-command.json` argument list for the corresponding scheduler. The 32B
manager also requires `model-provenance.json` with pinned model/lens paths,
revisions, and `hf_home`. Configure these paths for the target machine rather than
executing exported historical configurations directly.

Pinned model revisions:

* Qwen3-8B: `b968826d9c46dd6066d109eabc6255188de91218`
* Qwen3-32B: `9216db5781bf21249d130ec9da846c4624c16137`
* Neuronpedia lens repository: `e166deb5ad540932100ef803c8c9bcc2a5453694`
* Jacobian-lens dependency: `581d398613e5602a5af361e1c34d3a92ea82ba8e`

Regenerate the behavioral audit from committed result files:

```bash
PYTHONPATH=src python scripts/exp0_audit/summarize_audit.py
```

`measurement_trace_audit.py` recomputes detailed checks on the LFS traces using
cached pinned tokenizers and the original Hi-ToM sample. It writes JSON to stdout
and accepts `--results-root`. Regenerate its compact report without overwriting
the saved evidence:

```bash
python scripts/exp0_audit/compact_measurement.py --output /tmp/measurement_summary.json
```

Use `--input` to compact a newly generated detailed report. GPU diagnostic entry points are
`diagnostic_generation.py` and `full_prompt_audit.py`; their output paths are
explicit arguments.

The integrated code passed 77 CPU tests. Six tokenizer checks were skipped because
the Qwen3-1.7B, Llama-3.1-8B-Instruct, and Gemma-3-12B-it tokenizers were not cached.
