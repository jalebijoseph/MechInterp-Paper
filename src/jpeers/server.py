"""HTTP inference server: one model + lens per process.

The same container serves three hosting targets:

* **Local / Cloud Run / any VM** -- ``POST /generate`` with a
  :class:`~jpeers.schemas.GenerateRequest` body.
* **Azure ML managed online endpoint (Azure AI Foundry managed compute)** -- the
  deployment's ``inference_config`` points scoring at ``POST /score`` (same body).
* **Vertex AI endpoint** -- Vertex sets ``AIP_HTTP_PORT``, ``AIP_HEALTH_ROUTE`` and
  ``AIP_PREDICT_ROUTE``. ``:predict`` sends ``{"instances": [request, ...]}`` and gets
  ``{"predictions": [response, ...]}``; ``:rawPredict`` forwards a bare request body.

Environment:
    JPEERS_MODEL      registry key or HF id (required)
    JPEERS_LENS_PATH  local lens file (optional; overrides the registry's Hub lens)
    JPEERS_BAND       band preset or "lo,hi" fractions (optional)
    JPEERS_DEVICE     auto | cuda | cpu (default auto)
    JPEERS_DTYPE      bf16 | fp16 | fp32 (default bf16)
    JPEERS_API_KEY    if set, require ``Authorization: Bearer <key>`` (self-hosted only;
                      Azure and Vertex authenticate in front of the container)
    HF_TOKEN          for gated models (Llama, Gemma)

Run locally:  ``JPEERS_MODEL=qwen3-1.7b uvicorn jpeers.server:app --port 8080``
"""

from __future__ import annotations

import asyncio
import logging
import os
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse

from jpeers.registry import get_model_spec
from jpeers.schemas import GenerateRequest

logger = logging.getLogger("jpeers.server")
STATE: dict = {"model": None, "error": None}


def _band_from_env() -> tuple[float, float] | str | None:
    raw = os.environ.get("JPEERS_BAND")
    if not raw:
        return None
    if "," in raw:
        lo, hi = (float(x) for x in raw.split(","))
        return (lo, hi)
    return raw


def load_model_from_env():
    from jpeers.runtime import LensedModel

    name = os.environ["JPEERS_MODEL"]
    spec = get_model_spec(name, lens_path=os.environ.get("JPEERS_LENS_PATH"),
                          band=_band_from_env())
    return LensedModel(spec, device=os.environ.get("JPEERS_DEVICE", "auto"),
                       dtype=os.environ.get("JPEERS_DTYPE", "bf16"))


@asynccontextmanager
async def lifespan(app: FastAPI):
    logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"))
    if STATE["model"] is None and os.environ.get("JPEERS_MODEL"):
        async def _load():
            try:
                STATE["model"] = await asyncio.to_thread(load_model_from_env)
            except Exception as exc:  # surface through /health instead of crash-looping
                logger.exception("model load failed")
                STATE["error"] = repr(exc)

        asyncio.get_running_loop().create_task(_load())
    yield


app = FastAPI(title="jpeers J-space inference", lifespan=lifespan)


def _auth(authorization: str | None = Header(default=None)) -> None:
    key = os.environ.get("JPEERS_API_KEY")
    if key and authorization != f"Bearer {key}":
        raise HTTPException(status_code=401, detail="bad or missing bearer token")


def _model():
    if STATE["model"] is None:
        raise HTTPException(status_code=503, detail=STATE["error"] or "model loading")
    return STATE["model"]


async def _run(payload: dict) -> dict:
    req = GenerateRequest.model_validate(payload)
    try:
        resp = await asyncio.to_thread(_model().generate, req)
    except ValueError as exc:  # bad request content, e.g. a watch word that is not one token
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return resp.model_dump()


@app.get("/health")
async def health():
    if STATE["model"] is None:
        return JSONResponse({"status": "error" if STATE["error"] else "loading",
                             "detail": STATE["error"]}, status_code=503)
    return {"status": "ok", "model": STATE["model"].spec.name}


@app.get("/info", dependencies=[Depends(_auth)])
async def info():
    m = _model()
    return {"model": m.spec.name, "hf_id": m.spec.hf_id, "family": m.spec.family,
            "params_b": m.spec.params_b, "n_layers": m.n_layers, "d_model": m.d_model,
            "band": m.band(), "lens_layers": m.lens.source_layers,
            "lens_prompts": m.lens.n_prompts}


@app.post("/generate", dependencies=[Depends(_auth)])
async def generate(request: Request):
    return await _run(await request.json())


@app.post("/score", dependencies=[Depends(_auth)])  # Azure ML scoring route
async def score(request: Request):
    return await _run(await request.json())


async def _vertex_predict(request: Request):
    body = await request.json()
    if isinstance(body, dict) and "instances" in body:
        return {"predictions": [await _run(inst) for inst in body["instances"]]}
    return await _run(body)  # rawPredict


_predict_route = os.environ.get("AIP_PREDICT_ROUTE", "/predict")
app.add_api_route(_predict_route, _vertex_predict, methods=["POST"],
                  dependencies=[Depends(_auth)])
if os.environ.get("AIP_HEALTH_ROUTE", "/health") != "/health":
    app.add_api_route(os.environ["AIP_HEALTH_ROUTE"], health, methods=["GET"])


def main() -> None:
    import uvicorn

    port = int(os.environ.get("AIP_HTTP_PORT", os.environ.get("PORT", "8080")))
    uvicorn.run("jpeers.server:app", host="0.0.0.0", port=port, workers=1)


if __name__ == "__main__":
    main()
