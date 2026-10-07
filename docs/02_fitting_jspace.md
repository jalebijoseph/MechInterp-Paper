# 2. Fitting a lens (building the J-space) for a new model

You only need this for a model without a Neuronpedia lens, a fine-tuned checkpoint,
or a lens fitted on a different corpus (e.g. multi-agent transcripts instead of
WikiText).

## Estimator

`scripts/fit_lens.py` wraps `jlens.fit`. For each prompt it runs one forward pass with the
prompt replicated `dim_batch` times, then `ceil(d_model / dim_batch)` backward passes. Each
pass injects one-hot cotangents at every valid target position and reads `dim_batch`
rows of every `J_l` at once. The result for a prompt is the sum over target positions
`t' ≥ t`, averaged over source positions `t` (ignoring the first 16 attention-sink
positions). The lens is the mean over prompts.

```bash
# one machine
python scripts/fit_lens.py --model Qwen/Qwen3-8B --n-prompts 300 --dim-batch 32 --out lenses/qwen3-8b.pt

# sharded across GPUs or cloud jobs, then merged
python scripts/fit_lens.py --model Qwen/Qwen3-8B --n-prompts 1000 --shard 0/4 --out lenses/p0.pt
python scripts/fit_lens.py --merge "lenses/p*.pt" --out lenses/qwen3-8b.pt

# use it
JPEERS_MODEL=Qwen/Qwen3-8B JPEERS_LENS_PATH=lenses/qwen3-8b.pt python -m jpeers.server
```

Checkpoints are written every 10 prompts and resume automatically.

## How many prompts, and what it costs

The paper uses 1000 × 128-token pretraining-like sequences and reports that quality
saturates quickly; about 100 prompts is already usable. We checked this locally: a GPT-2
lens fitted on **20** WikiText prompts (53 s on an RTX 3060) has cosine similarity
0.961 / 0.970 / 0.986 / 0.997 with Neuronpedia's 1000-prompt lens at layers 2 / 5 / 8 / 10.

Cost scales with `d_model / dim_batch` backward passes per prompt:

| model | d_model | passes/prompt (dim_batch 32) | rough A100-80GB time, 300 prompts |
|---|---|---|---|
| 1.7B | 2048 | 64 | ~20 min |
| 8B | 4096 | 128 | ~1 h |
| 32B | 5120 | 160 | ~4–5 h (shard it) |

These are estimates; time the first few prompts from the log (`prompt i/n ... Ns`).

## On the cloud

* **Azure ML**: `deploy/azure/fit_lens_job.yml` (command job on a GPU compute cluster).
* **Vertex AI**: `deploy/gcp/fit_lens_job.sh` (custom training job, writes to GCS).

Both use the same container image as the inference server.

## Agent-domain lenses (optional extension)

The lens is an average over a corpus. To ask whether J-space is different in multi-agent
dialogue, fit a second lens with `--prompts-file` on rendered transcripts from
`runs/*/turns.jsonl`, then compare readouts from the two lenses on the same traces.
