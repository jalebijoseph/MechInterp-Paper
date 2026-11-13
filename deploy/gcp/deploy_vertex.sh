#!/usr/bin/env bash
# Deploy the jpeers server to a Vertex AI online endpoint (custom container, GPU).
#
# Prereqs: gcloud CLI, `gcloud auth login` + `gcloud auth application-default login`,
# Vertex AI + Artifact Registry + Cloud Build APIs enabled, and GPU quota for the
# accelerator in REGION ("Custom model serving <GPU> per region").
#
# Usage:
#   PROJECT=my-proj REGION=us-central1 MODEL=llama3.1-8b-it HF_TOKEN=... ./deploy/gcp/deploy_vertex.sh
#   Machine presets: MACHINE=g2-standard-12 ACCEL=nvidia-l4      (24GB: <=8B bf16)
#                    MACHINE=a2-ultragpu-1g ACCEL=nvidia-a100-80gb (80GB: <=32B bf16)
set -euo pipefail

: "${PROJECT:?set PROJECT}"
REGION="${REGION:-us-central1}"
MODEL="${MODEL:-qwen3-8b}"
MACHINE="${MACHINE:-g2-standard-12}"
ACCEL="${ACCEL:-nvidia-l4}"
REPO="${REPO:-jpeers}"
TAG="${TAG:-latest}"
IMAGE="${REGION}-docker.pkg.dev/${PROJECT}/${REPO}/jpeers:${TAG}"
NAME="jpeers-${MODEL//./-}"
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"

gcloud config set project "$PROJECT" >/dev/null
gcloud artifacts repositories describe "$REPO" --location "$REGION" >/dev/null 2>&1 || \
  gcloud artifacts repositories create "$REPO" --repository-format=docker --location "$REGION"

echo ">> building ${IMAGE} with Cloud Build"
(cd "$ROOT" && gcloud builds submit --region "$REGION" --config deploy/gcp/cloudbuild.yaml \
   --substitutions "_IMAGE=${IMAGE}" .)

ENV_VARS="JPEERS_MODEL=${MODEL},JPEERS_DTYPE=${DTYPE:-bf16},HF_HOME=/tmp/hf"
[[ -n "${HF_TOKEN:-}" ]] && ENV_VARS="${ENV_VARS},HF_TOKEN=${HF_TOKEN}"

echo ">> uploading model ${NAME}"
gcloud ai models upload --region "$REGION" --display-name "$NAME" \
  --container-image-uri "$IMAGE" \
  --container-ports 8080 \
  --container-health-route /health \
  --container-predict-route /predict \
  --container-env-vars "$ENV_VARS"
MODEL_ID=$(gcloud ai models list --region "$REGION" --filter "displayName=${NAME}" \
  --sort-by ~createTime --limit 1 --format 'value(name)' | awk -F/ '{print $NF}')

echo ">> endpoint (dedicated: larger payloads, longer timeouts)"
ENDPOINT_ID=$(gcloud ai endpoints list --region "$REGION" --filter "displayName=${NAME}" \
  --format 'value(name)' | awk -F/ '{print $NF}' | head -1)
if [[ -z "$ENDPOINT_ID" ]]; then
  gcloud beta ai endpoints create --region "$REGION" --display-name "$NAME" \
    --dedicated-endpoint-enabled 2>/dev/null || \
  gcloud ai endpoints create --region "$REGION" --display-name "$NAME"
  ENDPOINT_ID=$(gcloud ai endpoints list --region "$REGION" --filter "displayName=${NAME}" \
    --format 'value(name)' | awk -F/ '{print $NF}' | head -1)
fi

echo ">> deploying model ${MODEL_ID} to endpoint ${ENDPOINT_ID} on ${MACHINE} + ${ACCEL}"
gcloud ai endpoints deploy-model "$ENDPOINT_ID" --region "$REGION" --model "$MODEL_ID" \
  --display-name "$NAME" --machine-type "$MACHINE" \
  --accelerator "type=${ACCEL},count=1" \
  --min-replica-count 1 --max-replica-count 1 --traffic-split 0=100

DEDICATED=$(gcloud ai endpoints describe "$ENDPOINT_ID" --region "$REGION" \
  --format 'value(dedicatedEndpointDns)' 2>/dev/null || true)
cat <<EOF

Deployed. Add to configs/endpoints.yaml:

${MODEL}-vertex:
  backend: vertex
  project: ${PROJECT}
  region: ${REGION}
  endpoint_id: "${ENDPOINT_ID}"
$( [[ -n "$DEDICATED" ]] && echo "  dedicated_domain: ${DEDICATED}" )

Check: python -m jpeers.cli probe ${MODEL}-vertex "Hello" --endpoints configs/endpoints.yaml --last
Tear down (stops GPU billing):
  gcloud ai endpoints undeploy-model ${ENDPOINT_ID} --region ${REGION} --deployed-model-id <id from describe>
  gcloud ai endpoints delete ${ENDPOINT_ID} --region ${REGION}
EOF
