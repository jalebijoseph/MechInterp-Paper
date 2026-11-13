"""Pinned, lens-free Hugging Face endpoint on an externally selected GPU pair.

Example (the controller sets CUDA_VISIBLE_DEVICES to its owned pair):
    PYTHONPATH=src python scripts/serve_plain_hf.py \
      --model-id allenai/OLMo-2-0325-32B-Instruct --revision <40-character-commit> \
      --family olmo --name olmo2-32b --port 18102

Torch, Transformers and the model are loaded only by ``PlainHFBackend.load``.
The HTTP app can therefore be exercised with a CPU-only backend.
"""

from __future__ import annotations

import argparse
import asyncio
from contextlib import asynccontextmanager
from dataclasses import dataclass
import json
import logging
import os
from pathlib import Path
import re
import threading
import time

from jpeers.schemas import GenerateRequest, GenerateResponse


@dataclass(frozen=True)
class ModelConfig:
    model_id: str
    revision: str
    family: str
    model_path: str | None = None
    name: str | None = None
    params_b: float = 32.
    context_window: int = 4096

    def __post_init__(self):
        if not re.fullmatch(r"[a-fA-F0-9]{40}", self.revision):
            raise ValueError("revision must be a pinned 40-character commit SHA")
        if not self.model_id or not self.family:
            raise ValueError("model_id and family are required")
        if not 1 <= self.context_window <= 4096:
            raise ValueError("context_window must be between 1 and 4096")


def validate_request(req: GenerateRequest) -> None:
    if req.record.enabled:
        raise ValueError("Plain HF inference requires explicit record.enabled=false; no lens is loaded")
    if req.interventions:
        raise ValueError("Plain HF inference does not support interventions")
    if (req.prompt is None) == (not bool(req.messages)):
        raise ValueError("Supply exactly one of messages or prompt")
    if req.max_new_tokens < 0:
        raise ValueError("max_new_tokens must be nonnegative")
    if req.temperature < 0 or not 0 < req.top_p <= 1:
        raise ValueError("temperature must be nonnegative and top_p must be in (0, 1]")


def encode_request(tokenizer, req: GenerateRequest) -> list[int]:
    validate_request(req)
    if req.messages:
        if not getattr(tokenizer, "chat_template", None):
            raise ValueError("A native chat template is required for chat messages")
        text = tokenizer.apply_chat_template(req.messages, tokenize=False, add_generation_prompt=True)
        # Native templates already insert their required BOS and control tokens.
        # https://huggingface.co/docs/transformers/chat_templating
        add_special_tokens = False
    else:
        text, add_special_tokens = req.prompt, True
    encoded = tokenizer(text + req.assistant_prefill,
                        add_special_tokens=add_special_tokens, truncation=False)
    ids = encoded["input_ids"]
    if not isinstance(ids, list) or not ids or any(type(i) is not int for i in ids):
        raise ValueError("Tokenizer must return a nonempty, flat token-ID list")
    return ids


def validate_context(prompt_len: int, max_new_tokens: int, context_window: int) -> None:
    if prompt_len <= 0:
        raise ValueError("Prompt token count must be positive")
    if max_new_tokens < 0:
        raise ValueError("max_new_tokens must be nonnegative")
    if prompt_len + max_new_tokens > context_window:
        raise ValueError(f"Context budget exceeds {context_window} tokens: prompt={prompt_len}, "
                         f"requested_output={max_new_tokens}; input was not truncated")


class PlainHFBackend:
    def __init__(self, config: ModelConfig):
        self.config = config
        self.model = self.tokenizer = self.torch = None
        self.lock = threading.Lock()
        self.context_window = config.context_window
        self.gpus = []
        self.token_strings = {}

    def load(self):
        import torch
        import transformers

        if torch.cuda.device_count() != 2:
            raise RuntimeError("Plain 32B inference requires exactly two visible GPUs")
        source = self.config.model_path or self.config.model_id
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(
            source, revision=self.config.revision, trust_remote_code=False)
        if not getattr(self.tokenizer, "chat_template", None):
            raise RuntimeError("The pinned tokenizer has no native chat template")
        model = transformers.AutoModelForCausalLM.from_pretrained(
            source, revision=self.config.revision, trust_remote_code=False,
            dtype=torch.bfloat16, device_map="balanced",
            max_memory={0: "34GiB", 1: "34GiB", "cpu": 0}, low_cpu_mem_usage=True)
        placements = {str(v).replace("cuda:", "") for v in getattr(model, "hf_device_map", {}).values()}
        if placements != {"0", "1"}:
            raise RuntimeError(f"Expected GPU-only sharding across both visible devices: {placements}")
        loaded_revision = getattr(model.config, "_commit_hash", None)
        if loaded_revision and loaded_revision.lower() != self.config.revision.lower():
            raise RuntimeError(f"Loaded model commit {loaded_revision} differs from pinned {self.config.revision}")
        actual_context = getattr(model.config, "max_position_embeddings", None)
        if isinstance(actual_context, int) and actual_context > 0:
            self.context_window = min(self.context_window, actual_context)
        self.gpus = []
        for i in range(2):
            properties = torch.cuda.get_device_properties(i)
            self.gpus.append({"visible_index": i, "name": properties.name,
                              "total_memory_bytes": int(properties.total_memory),
                              "uuid": str(getattr(properties, "uuid", "unavailable"))})
        self.model, self.torch = model.eval(), torch
        logging.info("Loaded pinned %s at %s on the externally selected GPU pair", self.config.model_id,
                     self.config.revision)
        return self

    def info(self) -> dict:
        config = self.config
        hf_config = getattr(self.model, "config", None)
        return {"model": config.name or config.model_id, "hf_id": config.model_id,
                "model_id": config.model_id, "model_path": config.model_path or config.model_id,
                "revision": config.revision, "loaded_revision": getattr(hf_config, "_commit_hash", None),
                "family": config.family, "params_b": config.params_b, "dtype": "bfloat16",
                "context_window": self.context_window, "lens": None, "recording": False,
                "n_layers": getattr(hf_config, "num_hidden_layers", None),
                "d_model": getattr(hf_config, "hidden_size", None),
                "device_map": {str(k): str(v) for k, v in getattr(self.model, "hf_device_map", {}).items()},
                "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"), "gpus": self.gpus}

    def generate(self, req: GenerateRequest) -> GenerateResponse:
        validate_request(req)
        with self.lock:
            if self.model is None:
                raise RuntimeError("Model has not been loaded")
            started = time.perf_counter()
            ids = encode_request(self.tokenizer, req)
            validate_context(len(ids), req.max_new_tokens, self.context_window)
            tokenized = time.perf_counter()
            generated = []
            if req.max_new_tokens:
                torch = self.torch
                # Sharded models consume inputs on their embedding device, rather than
                # an assumed global model.device.
                device = self.model.get_input_embeddings().weight.device
                input_ids = torch.tensor([ids], dtype=torch.long, device=device)
                kwargs = {"input_ids": input_ids, "attention_mask": torch.ones_like(input_ids),
                          "max_new_tokens": req.max_new_tokens, "do_sample": req.temperature > 0,
                          "num_beams": 1, "num_return_sequences": 1, "use_cache": True}
                if req.temperature > 0:
                    kwargs.update(temperature=req.temperature, top_p=req.top_p)
                pad = self.tokenizer.pad_token_id
                if pad is None:
                    pad = self.tokenizer.eos_token_id
                if pad is not None:
                    kwargs["pad_token_id"] = pad
                if req.stop:
                    kwargs.update(stop_strings=req.stop, tokenizer=self.tokenizer)
                torch.manual_seed(req.seed)
                with torch.inference_mode():
                    output = self.model.generate(**kwargs)
                output_ids = output[0].tolist()
                if output_ids[:len(ids)] != ids:
                    raise RuntimeError("Causal model generation did not preserve the complete input")
                generated = output_ids[len(ids):]
            text = self.tokenizer.decode(generated, skip_special_tokens=True,
                                         clean_up_tokenization_spaces=False)
            cutoffs = [text.index(stop) for stop in req.stop if stop and stop in text]
            if cutoffs:
                text = text[:min(cutoffs)]
            all_ids = ids + generated
            for token in set(all_ids) - self.token_strings.keys():
                self.token_strings[token] = self.tokenizer.decode(
                    [token], skip_special_tokens=False, clean_up_tokenization_spaces=False)
            return GenerateResponse(
                model=self.config.name or self.config.model_id, text=text, prompt_len=len(ids),
                n_generated=len(generated), tokens=[self.token_strings[token] for token in all_ids],
                spans=[], layers=[], watch=[], watch_ids=[], vocab={}, trace_encoding="json", trace=None,
                interventions_applied=0, timings={"total_s": time.perf_counter() - started,
                                                "tokenize_s": tokenized - started})


def create_app(*, backend=None, loader=None):
    from fastapi import FastAPI, HTTPException
    from fastapi.responses import JSONResponse

    if backend is not None and loader is not None:
        raise ValueError("Supply either a loaded backend or a loader")
    state = {"backend": backend, "error": None}

    @asynccontextmanager
    async def lifespan(app):
        async def load_backend():
            try:
                state["backend"] = await asyncio.to_thread(loader)
            except Exception as exc:
                logging.exception("Plain HF model loading failed")
                state["error"] = repr(exc)
        task = asyncio.create_task(load_backend()) if state["backend"] is None and loader else None
        yield
        if task and not task.done():
            task.cancel()

    app = FastAPI(title="jpeers pinned plain HF inference", lifespan=lifespan)

    def model():
        if state["backend"] is None:
            raise HTTPException(status_code=503, detail=state["error"] or "model loading")
        return state["backend"]

    @app.get("/health")
    async def health():
        if state["backend"] is None:
            return JSONResponse({"status": "error" if state["error"] else "loading",
                                 "detail": state["error"]}, status_code=503)
        return {"status": "ok", "model": model().info()["model"]}

    @app.get("/info")
    async def info():
        return model().info()

    @app.post("/generate", response_model=GenerateResponse)
    async def generate(req: GenerateRequest):
        try:
            validate_request(req)
            return await asyncio.to_thread(model().generate, req)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    return app


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--revision", required=True, help="Pinned 40-character HF commit SHA")
    parser.add_argument("--family", required=True)
    parser.add_argument("--model-path", help="Optional predownloaded model snapshot path")
    parser.add_argument("--name")
    parser.add_argument("--params-b", type=float, default=32.)
    parser.add_argument("--context-window", type=int, default=4096)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18102)
    parser.add_argument("--provenance-output", type=Path)
    args = parser.parse_args()
    config = ModelConfig(model_id=args.model_id, revision=args.revision, family=args.family,
                         model_path=args.model_path, name=args.name, params_b=args.params_b,
                         context_window=args.context_window)

    def load():
        backend = PlainHFBackend(config).load()
        if args.provenance_output:
            args.provenance_output.parent.mkdir(parents=True, exist_ok=True)
            temp = args.provenance_output.with_suffix(args.provenance_output.suffix + ".tmp")
            temp.write_text(json.dumps(backend.info(), indent=2), encoding="utf-8")
            temp.replace(args.provenance_output)
        return backend

    import uvicorn
    logging.basicConfig(level=logging.INFO)
    uvicorn.run(create_app(loader=load), host=args.host, port=args.port, workers=1)


if __name__ == "__main__":
    main()
