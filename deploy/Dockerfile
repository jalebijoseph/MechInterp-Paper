# jpeers inference server: one model + Jacobian lens per container.
# Same image for Azure ML / Foundry managed online endpoints, Vertex AI endpoints,
# Cloud Run (GPU) and local docker. Torch wheels bundle CUDA; the host supplies the driver.
#
#   docker build -f deploy/Dockerfile -t jpeers:latest .
#   docker run --gpus all -p 8080:8080 -e JPEERS_MODEL=qwen3-1.7b jpeers:latest
#
# Optional: bake weights + lens into the image for fast cold starts (large image):
#   docker build -f deploy/Dockerfile --build-arg PREFETCH_MODEL=qwen3-8b \
#                --secret id=hf_token,env=HF_TOKEN -t jpeers:qwen3-8b .
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/models/hf \
    PORT=8080

WORKDIR /app
RUN pip install --index-url https://download.pytorch.org/whl/cu128 torch

COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install ".[server,cloud]"

COPY scripts/prefetch.py ./scripts/prefetch.py
ARG PREFETCH_MODEL=""
RUN --mount=type=secret,id=hf_token,required=false \
    if [ -n "$PREFETCH_MODEL" ]; then \
      HF_TOKEN="$(cat /run/secrets/hf_token 2>/dev/null || true)" \
      python scripts/prefetch.py "$PREFETCH_MODEL"; \
    fi
ENV JPEERS_MODEL=${PREFETCH_MODEL}

EXPOSE 8080
# Vertex AI sets AIP_HTTP_PORT; Cloud Run sets PORT; Azure uses the inference_config port.
CMD ["python", "-m", "jpeers.server"]
