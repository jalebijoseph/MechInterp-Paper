#!/usr/bin/env bash
# Fit a Jacobian lens as a Vertex AI custom training job (only for models without a
# pre-fitted lens). Writes lens.pt to a GCS bucket mounted at /gcs.
#   PROJECT=my-proj REGION=us-central1 BUCKET=my-bucket HF_MODEL=Qwen/Qwen3-8B ./deploy/gcp/fit_lens_job.sh
set -euo pipefail
: "${PROJECT:?}"; : "${BUCKET:?}"; : "${HF_MODEL:?}"
REGION="${REGION:-us-central1}"
IMAGE="${REGION}-docker.pkg.dev/${PROJECT}/jpeers/jpeers:${TAG:-latest}"
OUT="/gcs/${BUCKET}/lenses/$(echo "$HF_MODEL" | tr '/' '_').pt"

gcloud ai custom-jobs create --project "$PROJECT" --region "$REGION" \
  --display-name "jpeers-fit-$(echo "$HF_MODEL" | tr '/.' '--')" \
  --worker-pool-spec="machine-type=a2-ultragpu-1g,accelerator-type=NVIDIA_A100_80GB,accelerator-count=1,replica-count=1,container-image-uri=${IMAGE}" \
  --command=python \
  --args="scripts/fit_lens.py,--model,${HF_MODEL},--n-prompts,${N_PROMPTS:-300},--dim-batch,32,--out,${OUT}"
