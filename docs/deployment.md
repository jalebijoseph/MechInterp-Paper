# Deployment: Azure AI Foundry and Google Cloud GPU endpoints

## Why not the managed model APIs

The J-lens needs **white-box access**: residual activations at every workspace layer,
the unembedding matrix, and the ability to edit activations mid-forward-pass. Serverless
model APIs (Azure AI Foundry "serverless API" / Models-as-a-Service, Vertex AI Model
Garden MaaS, Gemini, the OpenAI API) return only text and logprobs, so they cannot be
used. Instead we run open-weight models in our own container on a dedicated GPU, using
each cloud's custom-container hosting:

| cloud | product | what we deploy | script |
|---|---|---|---|
| Azure | Azure AI Foundry **managed compute** = Azure ML **managed online endpoint** | custom container, 1 GPU VM | `deploy/azure/deploy.sh` |
| GCP | **Vertex AI online prediction** (custom container endpoint) | custom container, 1 GPU node | `deploy/gcp/deploy_vertex.sh` |
| GCP (alt.) | **Cloud Run with GPU** (NVIDIA L4) | same container, scales to zero | `deploy/gcp/deploy_cloudrun.sh` |

All three run the same image (`deploy/Dockerfile`, entrypoint `python -m jpeers.server`)
and serve the same request body. Each container holds one model and its lens, and you
deploy one endpoint per model. A cross-family experiment then talks to several
endpoints at once, e.g. Qwen on Azure and Llama on Vertex.

## Picking hardware

bf16 weights need about 2 bytes/param, plus the lens and activations. Approximate needs
are listed in the registry (`python -m jpeers.cli models`).

| model size | GPU | Azure SKU | GCP |
|---|---|---|---|
| ≤ 4B | 16–24 GB | `Standard_NC4as_T4_v3` (T4, use `JPEERS_DTYPE=fp16`) | `g2-standard-8` + `nvidia-l4`, or Cloud Run L4 |
| 7–9B | 24–48 GB | `Standard_NC24ads_A100_v4` (A100 80GB) | `g2-standard-12` + `nvidia-l4` (tight), or `a2-ultragpu-1g` + A100 80GB |
| 12–32B | 80 GB | `Standard_NC24ads_A100_v4` | `a2-ultragpu-1g` + `nvidia-a100-80gb` |
| 70B | 2 × 80 GB | `Standard_NC48ads_A100_v4` | `a2-ultragpu-2g` |

The server shards across all GPUs on the node with `device_map="auto"`. GPU quota is per
region and usually has to be requested first.

## Azure AI Foundry / Azure ML

Prereqs: `az` CLI with the `ml` extension; an Azure ML workspace or a **hub-based Foundry
project** (a hub project is an Azure ML workspace, so pass its name as `WS`); an Azure
Container Registry; GPU quota.

```bash
az login
az extension add -n ml
RG=my-rg WS=my-foundry-project ACR=myacr MODEL=qwen3-8b HF_TOKEN=hf_... ./deploy/azure/deploy.sh
```

The script:
1. Builds the image in ACR (`az acr build`, no local Docker needed).
2. Creates a key-auth managed online endpoint.
3. Creates deployment `main` with `inference_config` routes: liveness and readiness on
   `/health`, scoring on `/score`, port 8080. The probes allow 15 minutes for the first
   weight download.
4. Prints the `configs/endpoints.yaml` snippet and the command to fetch the key.

Settings to know:
* `request_timeout_ms: 180000` is the managed-endpoint maximum. Keep turns to a few
  hundred generated tokens. Long recordings fit because traces are compressed.
* `max_concurrent_requests_per_instance: 1`: requests are serialised per GPU. Run more
  instances for parallel throughput.
* `HF_TOKEN` is passed as a deployment environment variable. For anything beyond a
  personal project, store it in Key Vault and reference it instead.
* Endpoints bill while deployed. Tear down with
  `az ml online-endpoint delete -g $RG -w $WS -n <endpoint> --yes`.

Client side:

```yaml
qwen3-8b-azure:
  backend: azureml
  scoring_uri: https://<endpoint>.<region>.inference.ml.azure.com/score
  key_env: AZUREML_QWEN3_8B_KEY      # or `auth: aad` for Entra ID tokens
```

## Google Cloud: Vertex AI

Prereqs: `gcloud` CLI with `gcloud auth login` and `gcloud auth application-default login`;
the Vertex AI, Artifact Registry and Cloud Build APIs enabled; GPU quota for "custom model
serving".

```bash
PROJECT=my-proj REGION=us-central1 MODEL=llama3.1-8b-it HF_TOKEN=hf_... \
MACHINE=g2-standard-12 ACCEL=nvidia-l4 ./deploy/gcp/deploy_vertex.sh
```

The script builds with Cloud Build (`deploy/gcp/cloudbuild.yaml`) and uploads a Vertex
Model with `--container-health-route /health --container-predict-route /predict --container-ports 8080`.
It then creates an endpoint (dedicated if available) and deploys the model on one GPU.
The client calls `:rawPredict`, which passes our JSON body straight to `/predict`.
`:predict` with `{"instances": [...]}` also works.

Things to verify for your project (Vertex limits change; check current quotas):
* Shared public endpoints cap request/response payload sizes and timeouts. Dedicated
  endpoints raise both. Prefer dedicated endpoints for long traces, or reduce trace size
  (`stride`, `positions`).
* Deployed models bill per node-hour until undeployed (`gcloud ai endpoints undeploy-model`).

Client side:

```yaml
llama3.1-8b-it-vertex:
  backend: vertex
  project: my-proj
  region: us-central1
  endpoint_id: "123..."
  # dedicated_domain: <printed by the script if dedicated>
```

## Google Cloud: Cloud Run with GPU (simpler and cheaper for ≤ 8B)

```bash
PROJECT=my-proj REGION=us-central1 MODEL=qwen3-8b ./deploy/gcp/deploy_cloudrun.sh
```

This gives one L4 (24 GB), scale to zero, a 60-minute request timeout, IAM-authenticated
access (the client mints an ID token via ADC), and `concurrency 1`. Cold starts download
the weights, so bake them into the image (`--build-arg PREFETCH_MODEL=...`) if cold starts
matter.

## Local and dev

```bash
# in-process (any config: an endpoint name not in endpoints.yaml loads locally)
python -m jpeers.cli run configs/exp1_collab.yaml

# local server, same API as the clouds
JPEERS_MODEL=qwen3-1.7b python -m jpeers.server
docker build -f deploy/Dockerfile -t jpeers . && docker run --gpus all -p 8080:8080 -e JPEERS_MODEL=qwen3-1.7b jpeers
```

## Smoke test any endpoint

```bash
python -m jpeers.cli probe qwen3-8b-azure "Fact: The capital of the country shaped like a boot is" \
  --raw --last --watch Italy Rome --endpoints configs/endpoints.yaml
```

## Security notes

* Endpoints carry no data beyond what you send, but transcripts and traces in `runs/`
  may contain prompts; `runs/` is git-ignored.
* Self-hosted HTTP servers (VM / docker) should set `JPEERS_API_KEY`. Azure and Vertex
  authenticate in front of the container.
