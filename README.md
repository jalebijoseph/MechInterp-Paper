# afrj-jacobian-peers-analysis

Implementation of the research proposal **Internal Representations of Theory of Mind
Across Agents via the Jacobian Lens** ([proposal_formatted.md](proposal_formatted.md);
original draft in [PROPOSAL.md](PROPOSAL.md)).

We use the Jacobian lens (J-lens) from Anthropic's J-space work to read, and causally
edit, what LLM agents represent about each other while they collaborate, catch an
imposter, and delegate work. The package is `jpeers`.

## Layout

```
src/jpeers/
  registry.py        models with pre-fitted lenses (Qwen, Llama, Gemma, gpt-oss, ...)
  runtime.py         model + lens; token-by-token generation with J-space recording and edits
  schemas.py         request/response wire format (same for every backend)
  interventions.py   named J-space edit presets (ablate, random control, steer, swap, ...)
  server.py          HTTP server: local, Azure ML / Foundry (/score), Vertex AI (/predict)
  client.py          endpoint backends: local | http | azureml | vertex
  harness.py         multi-agent conversations; every turn's J-space saved to runs/
  analysis.py        rank / MRR / uptake / similarity metrics, permutation tests
  benchmarks.py      false-belief generator, Hi-ToM, NARCBench loaders
  experiments/       tom_validation, collab, imposter, peer_selection, ablation
  data/              scenario banks for Exp. 1-3
scripts/             fit_lens.py, lens_sanity.py, prefetch.py
configs/             one YAML per experiment + endpoints.example.yaml
deploy/              Dockerfile, azure/ (Foundry managed compute), gcp/ (Vertex AI, Cloud Run)
docs/                one guide per workstream (below)
tests/               CPU tests on a tiny model with a freshly fitted lens
```

## Guides

1. [Setting up the Jacobian lens](docs/01_jacobian_lens_setup.md)
2. [Fitting a lens / building J-space](docs/02_fitting_jspace.md)
3. [Benchmarks](docs/03_benchmarks.md)
4. [Experimental harness](docs/04_harness.md)
5. [Experiments 0-4](docs/05_experiments.md) and [pilot results](docs/results_pilot.md)
6. [Removing and editing J-space during generation](docs/06_interventions.md)
7. [Pre-registration: primary metrics, tests, sample sizes](docs/07_preregistration.md)

[Deployment on Azure AI Foundry and Google Cloud](docs/deployment.md)

## Quick start

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/Scripts/python.exe torch --index-url https://download.pytorch.org/whl/cu128
uv pip install --python .venv/Scripts/python.exe -e ".[server,cloud,data,dev]"
.venv/Scripts/python.exe -m pytest -q                        # 20 CPU tests

python -m jpeers.cli models                                  # registered models + lenses
python scripts/lens_sanity.py --model qwen3-1.7b             # lens behaves as expected?
python -m jpeers.cli run configs/exp2_imposter.yaml --set n_items=5   # local GPU pilot
```

For real runs, deploy endpoints (`deploy/azure/deploy.sh`, `deploy/gcp/deploy_vertex.sh`),
list them in `configs/endpoints.yaml`, and point the experiment configs at those names.

## Status

* Done: lens loading and fitting, recording and interventions, server and four client
  backends, harness, Exp. 0–4 runners, Azure and GCP deployment scripts, tests.
* Pilots of every experiment ran on a laptop RTX 3060 with Qwen3-1.7B; see
  [docs/results_pilot.md](docs/results_pilot.md). The cloud deployment scripts have not
  been executed yet; they need your subscription, project and GPU quota.
