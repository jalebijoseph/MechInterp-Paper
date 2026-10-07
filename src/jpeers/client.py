"""Endpoint backends. The harness only sees :class:`Endpoint.generate`.

Backends:
    local   -- load the model in this process (dev box with a GPU, or CPU tests)
    http    -- any server running ``jpeers.server`` (docker locally, Cloud Run, a VM)
    azureml -- Azure ML managed online endpoint / Azure AI Foundry managed compute
    vertex  -- Vertex AI endpoint with the custom container (``:rawPredict``)

Endpoints are declared in a YAML file (see ``configs/endpoints.example.yaml``)::

    qwen-azure:
      backend: azureml
      scoring_uri: https://<endpoint>.<region>.inference.ml.azure.com/score
      key_env: AZUREML_QWEN_KEY        # or auth: aad
      deployment: qwen3-8b              # optional: pin a deployment
    llama-gcp:
      backend: vertex
      project: my-project
      region: us-central1
      endpoint_id: "1234567890"
"""

from __future__ import annotations

import os
import time
from abc import ABC, abstractmethod
from pathlib import Path

import httpx
import yaml

from jpeers.schemas import GenerateRequest, GenerateResponse


class Endpoint(ABC):
    name: str = "endpoint"

    @abstractmethod
    def generate(self, req: GenerateRequest) -> GenerateResponse: ...

    def info(self) -> dict:
        return {}


class LocalEndpoint(Endpoint):
    """In-process model. ``model`` is a registry key / HF id or a loaded LensedModel."""

    _cache: dict[tuple, object] = {}

    def __init__(self, model, *, name: str | None = None, lens_path: str | None = None,
                 device: str = "auto", dtype: str = "bf16", band=None) -> None:
        from jpeers.runtime import LensedModel

        if isinstance(model, LensedModel):
            self.model = model
        else:
            from jpeers.registry import get_model_spec

            key = (model, lens_path, device, dtype, str(band))
            if key not in self._cache:
                spec = get_model_spec(model, lens_path=lens_path, band=band)
                self._cache[key] = LensedModel(spec, device=device, dtype=dtype)
            self.model = self._cache[key]
        self.name = name or self.model.spec.name

    def generate(self, req: GenerateRequest) -> GenerateResponse:
        return self.model.generate(req)

    def info(self) -> dict:
        m = self.model
        return {"model": m.spec.name, "family": m.spec.family, "params_b": m.spec.params_b,
                "n_layers": m.n_layers, "band": m.band()}


class _HTTPEndpoint(Endpoint):
    """Shared retry / timeout logic. Subclasses supply URL and headers."""

    def __init__(self, name: str, timeout_s: float = 600.0, retries: int = 4) -> None:
        self.name = name
        self.timeout_s = timeout_s
        self.retries = retries

    @abstractmethod
    def _url(self) -> str: ...

    @abstractmethod
    def _headers(self) -> dict[str, str]: ...

    def _wrap(self, payload: dict) -> dict:
        return payload

    def _unwrap(self, body: dict) -> dict:
        return body

    def _post(self, url: str, payload: dict) -> dict:
        delay = 2.0
        for attempt in range(self.retries + 1):
            try:
                r = httpx.post(url, json=payload, headers=self._headers(), timeout=self.timeout_s)
                if r.status_code in (429, 502, 503, 504) and attempt < self.retries:
                    time.sleep(delay)
                    delay *= 2
                    continue
                r.raise_for_status()
                return r.json()
            except (httpx.TransportError, httpx.TimeoutException):
                if attempt >= self.retries:
                    raise
                time.sleep(delay)
                delay *= 2
        raise RuntimeError("unreachable")

    def generate(self, req: GenerateRequest) -> GenerateResponse:
        body = self._post(self._url(), self._wrap(req.model_dump()))
        return GenerateResponse.model_validate(self._unwrap(body))


class HTTPEndpoint(_HTTPEndpoint):
    def __init__(self, name: str, base_url: str, api_key_env: str | None = None,
                 id_token_audience: str | None = None, **kw) -> None:
        super().__init__(name, **kw)
        self.base_url = base_url.rstrip("/")
        self.api_key_env = api_key_env
        self.id_token_audience = id_token_audience  # Cloud Run with IAM auth

    def _url(self) -> str:
        return f"{self.base_url}/generate"

    def _headers(self) -> dict[str, str]:
        if self.api_key_env and os.environ.get(self.api_key_env):
            return {"Authorization": f"Bearer {os.environ[self.api_key_env]}"}
        if self.id_token_audience:
            import google.auth.transport.requests
            import google.oauth2.id_token

            tok = google.oauth2.id_token.fetch_id_token(
                google.auth.transport.requests.Request(), self.id_token_audience)
            return {"Authorization": f"Bearer {tok}"}
        return {}

    def info(self) -> dict:
        r = httpx.get(f"{self.base_url}/info", headers=self._headers(), timeout=60)
        r.raise_for_status()
        return r.json()


class AzureMLEndpoint(_HTTPEndpoint):
    """Azure ML managed online endpoint (what Azure AI Foundry "managed compute" uses).

    Auth: ``auth: key`` reads the endpoint key from ``key_env``; ``auth: aad`` uses
    ``azure-identity`` (``az login`` / managed identity) for an Entra token.
    """

    def __init__(self, name: str, scoring_uri: str, key_env: str | None = None,
                 auth: str = "key", deployment: str | None = None, **kw) -> None:
        super().__init__(name, **kw)
        self.scoring_uri = scoring_uri
        self.key_env = key_env
        self.auth = auth
        self.deployment = deployment
        self._cred = None

    def _url(self) -> str:
        return self.scoring_uri

    def _headers(self) -> dict[str, str]:
        if self.auth == "aad":
            if self._cred is None:
                from azure.identity import DefaultAzureCredential

                self._cred = DefaultAzureCredential()
            token = self._cred.get_token("https://ml.azure.com/.default").token
        else:
            token = os.environ.get(self.key_env or "AZUREML_ENDPOINT_KEY", "")
            if not token:
                raise RuntimeError(f"set {self.key_env or 'AZUREML_ENDPOINT_KEY'}")
        headers = {"Authorization": f"Bearer {token}"}
        if self.deployment:
            headers["azureml-model-deployment"] = self.deployment
        return headers


class VertexEndpoint(_HTTPEndpoint):
    """Vertex AI online endpoint serving the jpeers container. Uses ``:rawPredict`` so
    the request body reaches the container unchanged. ``dedicated_domain`` is required
    for endpoints created with a dedicated endpoint (recommended: no 1.5 MB limit on the
    shared domain's payloads, longer timeouts)."""

    def __init__(self, name: str, project: str, region: str, endpoint_id: str,
                 dedicated_domain: str | None = None, **kw) -> None:
        super().__init__(name, **kw)
        self.project, self.region, self.endpoint_id = project, region, endpoint_id
        self.dedicated_domain = dedicated_domain
        self._creds = None

    def _url(self) -> str:
        host = self.dedicated_domain or f"{self.region}-aiplatform.googleapis.com"
        return (f"https://{host}/v1/projects/{self.project}/locations/{self.region}"
                f"/endpoints/{self.endpoint_id}:rawPredict")

    def _headers(self) -> dict[str, str]:
        import google.auth
        import google.auth.transport.requests

        if self._creds is None:
            self._creds, _ = google.auth.default(
                scopes=["https://www.googleapis.com/auth/cloud-platform"])
        if not self._creds.valid:
            self._creds.refresh(google.auth.transport.requests.Request())
        return {"Authorization": f"Bearer {self._creds.token}"}


BACKENDS = {"local": LocalEndpoint, "http": HTTPEndpoint, "azureml": AzureMLEndpoint,
            "vertex": VertexEndpoint}


def make_endpoint(name: str, cfg: dict) -> Endpoint:
    cfg = dict(cfg)
    backend = cfg.pop("backend")
    if backend == "local":
        return LocalEndpoint(cfg.pop("model"), name=name, **cfg)
    return BACKENDS[backend](name=name, **cfg)


def load_endpoints(path: str | Path | None) -> dict[str, dict]:
    """Read an endpoints YAML. Values like ``${VAR}`` are expanded from the env."""
    if path is None or not Path(path).exists():
        return {}
    text = os.path.expandvars(Path(path).read_text(encoding="utf-8"))
    return yaml.safe_load(text) or {}


class EndpointPool:
    """Lazily constructs endpoints by name. A name not in the YAML is treated as a
    registry model to load locally (handy for dev runs: ``--agent-model qwen3-1.7b``)."""

    def __init__(self, configs: dict[str, dict] | None = None, local_defaults: dict | None = None
                 ) -> None:
        self.configs = configs or {}
        self.local_defaults = local_defaults or {}
        self._live: dict[str, Endpoint] = {}

    def get(self, name: str) -> Endpoint:
        if name not in self._live:
            if name in self.configs:
                self._live[name] = make_endpoint(name, self.configs[name])
            else:
                self._live[name] = LocalEndpoint(name, **self.local_defaults)
        return self._live[name]
