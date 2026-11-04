These JSON configs are also valid YAML and preserve the original experiment sizes.
All inference uses the `qwen3-8b` HTTP endpoint. The service must implement the
repository's original HF/PyTorch `/generate` and `/info` routes (`python -m jpeers.server`).
A stock OpenAI-compatible completion endpoint does not provide the required traces or edits.

Run from the source directory after the service is ready:

```sh
python scripts/run_cluster_suite.py --ports 18080 --output-dir runs/full-suite
```

One subprocess uses each supplied port at a time. Add ports for independent servers.
The runner does not start or stop services; the enclosing lifecycle wrapper owns them.
Restart with the same output directory to resume pending/interrupted jobs. Existing
failures remain visible; `--retry-failed` creates a new attempt for failed/incomplete
portions without overwriting their original output. `--prepare-only` writes the plan
and provenance without making requests. A concurrent scheduler cannot own the same
output directory. A surviving old worker must finish or be stopped before resuming.

The 19 jobs cover five experiments:

| Experiment | Work |
|---|---|
| 0 core | 80 false/true-belief items, both framings |
| 0 Hi-ToM | 60 total items balanced across configured orders 0–3 |
| 1 | 96 conversation turns |
| 2 core | 360 reviews across all source/claim cells |
| 2 NARCBench | 20 configured transcript runs, all honest final prompts |
| 3 | 600 decisions plus 24 plan generations |
| 4 | 13 condition jobs with 360 reviews each, totaling 4,680 reviews |

External data portions are separate so a failed download cannot prevent core work.
There is no silent fallback: missing scenarios, missing summaries/traces, or omitted
Hi-ToM tokenizer items produce an incomplete/failed portion and nonzero suite exit.
The manifest records configured and observed counts, selected benchmark IDs/content
hashes, model info, package versions, source SHA/file hashes, logs, exit statuses,
effective configs, and exact result directories. Each experiment is complete only
when all its portions are complete. Experiments 0–3 retain their component summaries.
Exp4 fragments explicitly defer their primary comparison. After all 13 finish, the
runner saves `exp4-aggregate/summary.json` and `summary.md`, with the combined original
rows/live data, per-condition contrasts, recomputed baseline deltas, and the original
`ablation.primary_imposter` test over 180 matched false-claim pairs. Exp4 is complete
only after this aggregate validates all 4,680 unique reviews and its saved summary.

Experiment 3 self-preference compares deterministic and sampled plans from the same
Qwen3-8B model. It is explicitly a same-family comparison. The family-label variant
uses textual family/size descriptions and does not run models from other families.
