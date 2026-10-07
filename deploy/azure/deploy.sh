#!/usr/bin/env bash
# Deploy the jpeers server to an Azure ML managed online endpoint
# (= Azure AI Foundry managed compute in a hub-based project).
#
# Prereqs: az CLI + ml extension (`az extension add -n ml`), `az login`, an Azure ML
# workspace or Foundry hub project, an Azure Container Registry attached to it, and GPU
# quota for INSTANCE_TYPE in the region.
#
# Usage:
#   RG=my-rg WS=my-foundry-project ACR=myacr MODEL=qwen3-8b ./deploy/azure/deploy.sh
#   (optional) HF_TOKEN=... for gated models, INSTANCE_TYPE=Standard_NC24ads_A100_v4
set -euo pipefail

: "${RG:?set RG (resource group)}"
: "${WS:?set WS (Azure ML workspace / Foundry project name)}"
: "${ACR:?set ACR (container registry name, without .azurecr.io)}"
MODEL="${MODEL:-qwen3-8b}"
INSTANCE_TYPE="${INSTANCE_TYPE:-Standard_NC24ads_A100_v4}"
TAG="${TAG:-latest}"
ENDPOINT="${ENDPOINT:-jpeers-${MODEL//./-}-$(printf '%s' "$WS" | md5sum | cut -c1-6)}"
IMAGE="${ACR}.azurecr.io/jpeers:${TAG}"
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"

echo ">> building ${IMAGE} in ACR (no local docker needed)"
az acr build -r "$ACR" -t "jpeers:${TAG}" -f "$ROOT/deploy/Dockerfile" "$ROOT"

echo ">> endpoint ${ENDPOINT}"
az ml online-endpoint create -g "$RG" -w "$WS" -f "$HERE/endpoint.yml" --name "$ENDPOINT" \
  || echo "(endpoint exists, continuing)"

EXTRA_ENV=()
if [[ -n "${HF_TOKEN:-}" ]]; then
  EXTRA_ENV=(--set "environment_variables.HF_TOKEN=${HF_TOKEN}")
fi

echo ">> deployment main (${MODEL} on ${INSTANCE_TYPE}); first start downloads weights"
az ml online-deployment create -g "$RG" -w "$WS" -f "$HERE/deployment.yml" --all-traffic \
  --set "endpoint_name=${ENDPOINT}" \
  --set "environment.image=${IMAGE}" \
  --set "environment_variables.JPEERS_MODEL=${MODEL}" \
  --set "instance_type=${INSTANCE_TYPE}" \
  "${EXTRA_ENV[@]}"

URI=$(az ml online-endpoint show -g "$RG" -w "$WS" -n "$ENDPOINT" --query scoring_uri -o tsv)
KEY_VAR="AZUREML_$(echo "${MODEL}" | tr 'a-z.-' 'A-Z__')_KEY"
cat <<EOF

Deployed. Add to configs/endpoints.yaml:

${MODEL}-azure:
  backend: azureml
  scoring_uri: ${URI}
  key_env: ${KEY_VAR}

and export the key:
  export ${KEY_VAR}=\$(az ml online-endpoint get-credentials -g ${RG} -w ${WS} -n ${ENDPOINT} --query primaryKey -o tsv)

Check: python -m jpeers.cli probe ${MODEL}-azure "Hello" --endpoints configs/endpoints.yaml --last
Tear down (stops GPU billing): az ml online-endpoint delete -g ${RG} -w ${WS} -n ${ENDPOINT} --yes
EOF
