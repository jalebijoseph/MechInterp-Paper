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

## Agent-domain lenses: does the WikiText lens transfer to agent chat?

The lens is an average over a corpus (WikiText), while the experiments read it on
multi-agent chat. Check, per model, before the main runs (pre-registration, E):

```bash
# 1. transcript windows from recorded runs (fit set + held-out set, .jsonl)
python scripts/lens_agreement.py prompts --model qwen3-8b --runs runs/collab runs/imposter
# 2. fit a transcript lens (GPU job; ~100-300 prompts is enough)
python scripts/fit_lens.py --model Qwen/Qwen3-8B --prompts-file lenses/agent_prompts.fit.jsonl \
    --n-prompts 300 --out lenses/agent-qwen3-8b.pt
# 3. compare on held-out windows
python scripts/lens_agreement.py compare --model qwen3-8b --lens lenses/agent-qwen3-8b.pt
```

`compare` prints, per band layer, the cosine between the two Jacobians and how often the
two lenses agree on held-out transcript windows (top-1, top-10 overlap), next to the
same agreement between the WikiText lens and the plain logit lens as a reference. On
Qwen3-1.7B the logit lens agrees with the J-lens on only 1–2% of top tokens in the early
band and about 65% at the last band layer, so a transcript lens far above that reference
reads the same thing as the WikiText lens.

Fitting does not fit on a 6 GB laptop GPU: a 2-prompt timing run of Qwen3-1.7B filled
the card and did not finish in 10 minutes. Run step 2 with the cloud jobs below.
