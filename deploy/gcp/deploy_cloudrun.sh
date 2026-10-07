#!/usr/bin/env bash
# Alternative GCP target: Cloud Run with a GPU (NVIDIA L4, 24GB). Scales to zero, bills
# per second, no payload-size or predict-route constraints. Good for models <= 8B.
#
#   PROJECT=my-proj REGION=us-central1 MODEL=qwen3-8b ./deploy/gcp/deploy_cloudrun.sh
#
# Store the HF token as a secret first (gated models):
#   printf %s "$HF_TOKEN" | gcloud secrets create hf-token --data-file=-
set -euo pipefail

: "${PROJECT:?set PROJECT}"
REGION="${REGION:-us-central1}"
MODEL="${MODEL:-qwen3-8b}"
REPO="${REPO:-jpeers}"
IMAGE="${REGION}-docker.pkg.dev/${PROJECT}/${REPO}/jpeers:${TAG:-latest}"
SERVICE="jpeers-${MODEL//./-}"

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
gcloud artifacts repositories describe "$REPO" --project "$PROJECT" --location "$REGION" \
  >/dev/null 2>&1 || gcloud artifacts repositories create "$REPO" --project "$PROJECT" \
  --repository-format=docker --location "$REGION"
[[ "${SKIP_BUILD:-0}" == 1 ]] || (cd "$ROOT" && gcloud builds submit --project "$PROJECT" \
  --region "$REGION" --config deploy/gcp/cloudbuild.yaml --substitutions "_IMAGE=${IMAGE}" .)

SECRETS=()
gcloud secrets describe hf-token >/dev/null 2>&1 && SECRETS=(--set-secrets "HF_TOKEN=hf-token:latest")

gcloud run deploy "$SERVICE" --project "$PROJECT" --region "$REGION" --image "$IMAGE" \
  --gpu 1 --gpu-type nvidia-l4 --no-gpu-zonal-redundancy \
  --cpu 8 --memory 32Gi --no-cpu-throttling \
  --concurrency 1 --max-instances 1 --timeout 3600 \
  --port 8080 --no-allow-unauthenticated \
  --set-env-vars "JPEERS_MODEL=${MODEL},HF_HOME=/tmp/hf" "${SECRETS[@]}"

URL=$(gcloud run services describe "$SERVICE" --region "$REGION" --format 'value(status.url)')
cat <<EOF

Deployed. Add to configs/endpoints.yaml:

${MODEL}-cloudrun:
  backend: http
  base_url: ${URL}
  id_token_audience: ${URL}

(Your account needs roles/run.invoker; the client mints an ID token via ADC.)
EOF
