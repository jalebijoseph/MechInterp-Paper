"""Serial Qwen3 inference with real vLLM execution and the jpeers trace schema.

The worker owns lens readouts and interventions. Native Qwen3 blocks defer the
last residual addition: their output is (MLP delta, residual), so hooks must read
their sum. Edited rows become (edited hidden, zero), preserving their exact sum
without low-precision cancellation; unselected rows retain both native tensors.
No HF model is loaded: the small HF RMSNorm used for *lens readouts* preserves
the pinned jlens unembedding arithmetic; actual generation stays in vLLM.

Run ``python -m jpeers.vllm_runtime --model qwen3-8b --port 18080``. This adapter
deliberately requires eager execution, one sequence, and TP/PP/DP size one.
"""

from __future__ import annotations

import argparse
import os
import threading
import time
from dataclasses import asdict
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F

from jpeers.registry import ModelSpec, get_model_spec
from jpeers.runtime import (LensedModel, _CHUNK, _Run, _eos_ids, _final_norm_gain,
                            _locate_spans, load_lens)
from jpeers.schemas import GenerateRequest, GenerateResponse, TokenSpan
from jpeers.trace import encode_arrays


def _positions(inputs, kwargs, *, index=0) -> torch.Tensor:
    positions = kwargs.get("positions")
    if positions is None and len(inputs) > index:
        positions = inputs[index]
    if not torch.is_tensor(positions) or positions.ndim != 1:
        raise RuntimeError("Qwen3 hooks require one-dimensional absolute positions")
    return positions.detach().to(device="cpu", dtype=torch.long)


def make_decoder_hook(run: _Run, layer: int):
    """Adapt the original HF post-block edit/record contract to vLLM Qwen3."""
    ivs = [iv for iv, layers in run.interventions if layer in layers]
    record = layer in run.buffers

    def hook(module, inputs, kwargs, output):
        if not isinstance(output, tuple) or len(output) != 2:
            raise RuntimeError("Expected Qwen3 decoder (delta, residual) output")
        delta, residual = output
        if delta.ndim != 2 or residual.shape != delta.shape:
            raise RuntimeError("Expected flattened Qwen3 residuals [tokens, hidden]")
        pos = _positions(inputs, kwargs)
        if len(pos) != len(delta):
            raise RuntimeError("Qwen3 positions and hidden rows are misaligned")
        original_hidden = delta + residual
        hidden = original_hidden
        if record and run.rec.when == "pre":
            run._record(layer, hidden, pos)
        changed = torch.zeros(len(pos), dtype=torch.bool)
        if ivs:
            edited = hidden.float().clone()
            for iv in ivs:
                mask = run._mask(iv.positions, pos)
                if mask.any():
                    idx = mask.nonzero(as_tuple=True)[0].to(hidden.device)
                    edited[idx] = run._apply(iv, layer, edited[idx], pos)
                    changed |= mask
                    run.n_edits += 1
            if changed.any():
                hidden = edited.to(delta.dtype)
        if record and run.rec.when == "post":
            run._record(layer, hidden, pos)
        # The next norm consumes only the sum. Re-splitting as (h' - r, r)
        # can erase a small edited coordinate in bf16 when r is large. Setting
        # selected rows to (h', 0) preserves the intended intervention exactly.
        # Keep all other rows in their original representation and arithmetic.
        # Identity edits retain the native split too: fused-add norm can retain
        # extra precision from delta+residual even when the recorded sum rounds.
        changed &= (hidden != original_hidden).any(dim=-1).cpu()
        if not changed.any():
            return None
        idx = changed.nonzero(as_tuple=True)[0].to(delta.device)
        next_delta, next_residual = delta.clone(), residual.clone()
        next_delta[idx], next_residual[idx] = hidden[idx], 0
        return next_delta, next_residual

    return hook


def record_output_baseline(run: _Run, normalized_hidden: torch.Tensor,
                           positions: torch.Tensor, compute_logits) -> None:
    """Compute genuine full-vocabulary output ranks only at recorded positions."""
    if not run.rec.enabled or not run.record_layers:
        return
    ranks = "output" in run.rec.baselines and bool(run.watch_ids)
    if not ranks and not run.rec.include_model_logits:
        return
    mask = run._mask(run.rec.positions, positions) & (positions >= run._record_floor)
    idx = mask.nonzero(as_tuple=True)[0]
    if not hasattr(run, "_output_positions"):
        run._output_positions = []
        run._selected_top1 = []
    for start in range(0, len(idx), _CHUNK):
        sub = idx[start:start + _CHUNK]
        logits = compute_logits(normalized_hidden[sub.to(normalized_hidden.device)])
        if logits is None or logits.shape != (len(sub), run.m.vocab_size):
            raise RuntimeError("Actual output baseline must contain the full vocabulary")
        run._output_positions.append(positions[sub].numpy())
        if ranks:
            run.output_ranks.append(run._watch_ranks(logits.float()))
        if run.rec.include_model_logits:
            run._selected_top1.append(logits.argmax(-1))


def finish_run(run: _Run) -> dict[str, np.ndarray] | None:
    """Use the original trace stacker, checking sparse baseline/layer alignment."""
    arrays = run.finish()
    if arrays is None:
        return None
    positions = arrays["positions"]
    for layer in run.record_layers:
        actual = np.concatenate(run.buffers[layer].positions)
        if not np.array_equal(actual, positions):
            raise RuntimeError(f"Trace positions differ at layer {layer}")
    needs_output = run.rec.include_model_logits or (
        "output" in run.rec.baselines and bool(run.watch_ids))
    if needs_output:
        chunks = getattr(run, "_output_positions", [])
        if not chunks or not np.array_equal(np.concatenate(chunks), positions):
            raise RuntimeError("Actual output baseline positions differ from lens trace")
    if run.rec.include_model_logits:
        arrays["model_top1"] = torch.cat(run._selected_top1).cpu().numpy()
    return arrays


def validate_trace_positions(arrays, *, prompt_len: int, n_generated: int) -> None:
    """The last sampled token has no activation: it was never fed back in."""
    if arrays is None:
        return
    positions = arrays["positions"]
    end = prompt_len + max(0, n_generated - 1)
    if np.any(positions < 0) or np.any(positions >= end):
        raise RuntimeError("Trace contains an unprocessed token position")
    if len(positions) > 1 and np.any(np.diff(positions) <= 0):
        raise RuntimeError("Trace positions must be unique and increasing")
    if any(len(a) != len(positions) for a in arrays.values()):
        raise RuntimeError("Trace array position axes are misaligned")


def _lens_facade(spec: ModelSpec, native, tokenizer) -> LensedModel:
    from transformers.models.qwen3.modeling_qwen3 import Qwen3RMSNorm

    model = object.__new__(LensedModel)
    model.spec, model.tokenizer = spec, tokenizer
    model.n_layers = native.config.num_hidden_layers
    model.d_model = native.config.hidden_size
    model.vocab_size = native.config.vocab_size
    model.lens = load_lens(spec)
    model._J_cache, model._word_like_dev = {}, {}
    model._vocab_strings, model._word_like = None, None
    model._W_U = native.lm_head.weight[:model.vocab_size]
    # Sharing existing weights adds no second model or unembedding allocation.
    norm = Qwen3RMSNorm(model.d_model, eps=native.config.rms_norm_eps)
    norm.weight = native.model.norm.weight
    model._gain = _final_norm_gain(norm, model.d_model).to(model._W_U.device)

    def unembed(x):
        x = x.to(device=model._W_U.device, dtype=model._W_U.dtype)
        return F.linear(norm(x), model._W_U)

    model.lm = SimpleNamespace(unembed=unembed, layers=native.model.layers,
                               input_device=torch.device("cpu"))
    missing = [layer for layer in model.band() if layer not in model.lens.jacobians]
    if missing:
        raise ValueError(f"Lens lacks requested layers: {missing}")
    return model


class JPeersWorkerExtension:
    """vLLM extends its GPU worker with these methods before model loading."""

    def jpeers_setup(self, spec_dict: dict) -> dict:
        from transformers import AutoTokenizer

        config = self.vllm_config
        parallel, scheduler = config.parallel_config, config.scheduler_config
        if (parallel.tensor_parallel_size, parallel.pipeline_parallel_size,
                parallel.data_parallel_size) != (1, 1, 1):
            raise RuntimeError("JPeers requires TP=PP=DP=1")
        if not config.model_config.enforce_eager or scheduler.max_num_seqs != 1:
            raise RuntimeError("JPeers requires eager execution and max_num_seqs=1")
        if config.cache_config.enable_prefix_caching or scheduler.enable_chunked_prefill:
            raise RuntimeError("JPeers requires disabled prefix caching/chunked prefill")
        if config.speculative_config is not None or scheduler.async_scheduling:
            raise RuntimeError("JPeers requires disabled speculation/async scheduling")
        native = self.model_runner.model
        if type(native).__name__ != "Qwen3ForCausalLM" or native.quant_config is not None:
            raise RuntimeError("JPeers adapter requires native, unquantized Qwen3ForCausalLM")
        spec = ModelSpec(**spec_dict)
        tokenizer = AutoTokenizer.from_pretrained(spec.hf_id)
        self._jpeers_model = _lens_facade(spec, native, tokenizer)
        self._jpeers_run, self._jpeers_handles = None, []
        model = self._jpeers_model
        return {"n_layers": model.n_layers, "d_model": model.d_model,
                "vocab_size": model.vocab_size, "band": model.band(),
                "lens_layers": list(model.lens.source_layers),
                "lens_prompts": model.lens.n_prompts}

    def jpeers_begin(self, payload: dict, prompt_len: int, spans: dict) -> dict:
        if self._jpeers_run is not None:
            raise RuntimeError("A JPeers request is already active")
        req = GenerateRequest.model_validate(payload)
        run = _Run(self._jpeers_model, req, prompt_len, spans)
        native = self.model_runner.model
        for layer in run.hooked_layers:
            if not 0 <= layer < self._jpeers_model.n_layers:
                raise ValueError(f"Invalid decoder layer {layer}")
            if layer not in self._jpeers_model.lens.jacobians:
                raise ValueError(f"Lens lacks decoder layer {layer}")
        self._jpeers_run = run
        self._jpeers_seen = []
        self._jpeers_prefill_s = 0.0
        self._jpeers_start = time.perf_counter()

        def before_model(module, inputs, kwargs):
            pos = _positions(inputs, kwargs, index=1)
            start = sum(len(p) for p in self._jpeers_seen)
            if not torch.equal(pos, torch.arange(start, start + len(pos))):
                raise RuntimeError("vLLM repeated, skipped, or reordered request positions")
            if (start == 0 and len(pos) != prompt_len) or (start > 0 and len(pos) != 1):
                raise RuntimeError("Expected one full prefill followed by single-token decode")
            self._jpeers_positions = pos
            self._jpeers_seen.append(pos.numpy())

        def after_norm(module, inputs, output):
            normalized = output[0] if isinstance(output, tuple) else output
            record_output_baseline(run, normalized, self._jpeers_positions,
                                   native.compute_logits)
            if self._jpeers_positions[0] == 0:
                torch.cuda.synchronize(normalized.device)
                self._jpeers_prefill_s = time.perf_counter() - self._jpeers_start

        try:
            # Qwen3Model's support_torch_compile wrapper overrides __call__ and
            # bypasses ordinary module pre-hooks even in eager mode. The outer
            # Qwen3ForCausalLM is a regular nn.Module and sees the same positions.
            self._jpeers_handles.append(native.register_forward_pre_hook(
                before_model, with_kwargs=True))
            for layer in run.hooked_layers:
                self._jpeers_handles.append(native.model.layers[layer].register_forward_hook(
                    make_decoder_hook(run, layer), with_kwargs=True))
            self._jpeers_handles.append(native.model.norm.register_forward_hook(after_norm))
        except BaseException:
            self.jpeers_abort()
            raise
        return {"layers": run.record_layers, "watch": run.watch_words,
                "watch_ids": run.watch_ids}

    def jpeers_abort(self) -> None:
        for handle in getattr(self, "_jpeers_handles", []):
            handle.remove()
        self._jpeers_handles = []
        self._jpeers_run = None
        self._jpeers_seen = []

    def jpeers_finish(self, n_generated: int) -> dict:
        run = self._jpeers_run
        if run is None:
            raise RuntimeError("No active JPeers request")
        try:
            expected = run.prompt_len + max(0, n_generated - 1)
            seen = np.concatenate(self._jpeers_seen) if self._jpeers_seen else np.array([])
            if not np.array_equal(seen, np.arange(expected)):
                raise RuntimeError(f"vLLM processed {len(seen)} positions; expected {expected}")
            arrays = finish_run(run)
            validate_trace_positions(arrays, prompt_len=run.prompt_len,
                                     n_generated=n_generated)
            vocab = {}
            if arrays is not None:
                ids = set(np.unique(arrays["topk_ids"]).tolist())
                ids.update(i for group in run.watch_ids for i in group)
                vocab = {int(i): run.m.vocab_strings[i] for i in ids}
            return {"trace": None if arrays is None else encode_arrays(
                        arrays, run.req.trace_encoding),
                    "vocab": vocab, "interventions_applied": run.n_edits,
                    "prefill_s": self._jpeers_prefill_s}
        finally:
            self.jpeers_abort()


class VLLMLensedModel:
    """One serial HTTP-facing owner of a real vLLM LLM and instrumented worker."""

    def __init__(self, spec: ModelSpec, *, dtype="bf16", max_model_len=16384,
                 gpu_memory_utilization=0.65):
        from transformers import AutoConfig, AutoTokenizer, GenerationConfig
        from vllm import LLM

        self.spec, self.lock = spec, threading.Lock()
        self.max_model_len = max_model_len
        self.tokenizer = AutoTokenizer.from_pretrained(spec.hf_id)
        try:
            generation = GenerationConfig.from_pretrained(spec.hf_id)
        except OSError:
            generation = GenerationConfig.from_model_config(AutoConfig.from_pretrained(spec.hf_id))
        self._eos_ids = _eos_ids(SimpleNamespace(generation_config=generation), self.tokenizer)
        self._prompt_model = object.__new__(LensedModel)
        self._prompt_model.spec, self._prompt_model.tokenizer = spec, self.tokenizer
        self._prompt_model.lm = SimpleNamespace(input_device=torch.device("cpu"))
        dtype = {"bf16": "bfloat16", "fp16": "float16", "fp32": "float32"}.get(dtype, dtype)
        self.engine = LLM(
            model=spec.hf_id, dtype=dtype, tensor_parallel_size=1, pipeline_parallel_size=1,
            data_parallel_size=1, enforce_eager=True, compilation_config=0,
            max_num_seqs=1, max_model_len=max_model_len,
            max_num_batched_tokens=max_model_len, gpu_memory_utilization=gpu_memory_utilization,
            enable_prefix_caching=False, enable_chunked_prefill=False, async_scheduling=False,
            speculative_config=None, generation_config="vllm",
            worker_extension_cls="jpeers.vllm_runtime.JPeersWorkerExtension")
        meta = self._rpc("jpeers_setup", asdict(spec))
        self.n_layers, self.d_model = meta["n_layers"], meta["d_model"]
        self._band = meta["band"]
        self.lens = SimpleNamespace(source_layers=meta["lens_layers"],
                                    n_prompts=meta["lens_prompts"])

    def _rpc(self, method, *args):
        results = self.engine.collective_rpc(method, args=args)
        if len(results) != 1:
            raise RuntimeError("JPeers requires exactly one vLLM worker")
        return results[0]

    def band(self):
        return self._band

    def generate(self, req: GenerateRequest) -> GenerateResponse:
        from vllm import SamplingParams

        with self.lock:
            started = time.perf_counter()
            if req.max_new_tokens < 0:
                raise ValueError("max_new_tokens must be nonnegative")
            text, add_special = self._prompt_model.render(req)
            ids, offsets = self._prompt_model.encode(text, add_special)
            prompt_ids = ids[0].tolist()
            prompt_len = len(prompt_ids)
            if not prompt_len:
                raise ValueError("Prompt must contain at least one token")
            if prompt_len + max(1, req.max_new_tokens) > self.max_model_len:
                raise ValueError(f"Prompt ({prompt_len}) plus generation exceeds context "
                                 f"length {self.max_model_len}")
            spans = _locate_spans(text, offsets, req, prompt_len)
            meta = self._rpc("jpeers_begin", req.model_dump(), prompt_len, spans)
            try:
                params = SamplingParams(
                    max_tokens=max(1, req.max_new_tokens), temperature=req.temperature,
                    top_p=req.top_p, seed=req.seed, stop=[s for s in req.stop if s],
                    stop_token_ids=sorted(self._eos_ids), skip_special_tokens=True)
                outputs = self.engine.generate({"prompt_token_ids": prompt_ids}, params,
                                               use_tqdm=False)
                if len(outputs) != 1 or len(outputs[0].outputs) != 1:
                    raise RuntimeError("Expected exactly one generated completion")
                if list(outputs[0].prompt_token_ids) != prompt_ids:
                    raise RuntimeError("vLLM changed the request's prompt tokenization")
                generated = list(outputs[0].outputs[0].token_ids) if req.max_new_tokens else []
                result = self._rpc("jpeers_finish", len(generated))
            except BaseException:
                self._rpc("jpeers_abort")
                raise
            gen_text = self.tokenizer.decode(
                [t for t in generated if t not in self._eos_ids], skip_special_tokens=True)
            for stop in req.stop:
                if stop and stop in gen_text:
                    gen_text = gen_text[:gen_text.index(stop)]
            return GenerateResponse(
                model=self.spec.name, text=gen_text, prompt_len=prompt_len,
                n_generated=len(generated),
                tokens=self.tokenizer.batch_decode([[i] for i in prompt_ids + generated]),
                spans=[TokenSpan(label=label, start=a, end=b)
                       for label, ranges in spans.items() for a, b in ranges],
                layers=meta["layers"], watch=meta["watch"], watch_ids=meta["watch_ids"],
                vocab=result["vocab"], trace_encoding=req.trace_encoding,
                trace=result["trace"], interventions_applied=result["interventions_applied"],
                timings={"prefill_s": result["prefill_s"],
                         "total_s": time.perf_counter() - started})


def load_model_from_env():
    from jpeers.server import _band_from_env

    spec = get_model_spec(os.environ["JPEERS_MODEL"],
                          lens_path=os.environ.get("JPEERS_LENS_PATH"), band=_band_from_env())
    return VLLMLensedModel(
        spec, dtype=os.environ.get("JPEERS_DTYPE", "bf16"),
        max_model_len=int(os.environ.get("JPEERS_MAX_MODEL_LEN", "16384")),
        gpu_memory_utilization=float(os.environ.get("JPEERS_GPU_MEMORY_UTILIZATION", ".65")))


# Reuse the existing HTTP schema, health, authorization, and worker-thread routes.
# The loader is resolved by its lifespan at startup, after this replacement.
from jpeers import server as _http  # noqa: E402
_http.load_model_from_env = load_model_from_env
app = _http.app


def main():
    import uvicorn

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=os.environ.get("JPEERS_MODEL", "qwen3-8b"))
    parser.add_argument("--lens-path", default=os.environ.get("JPEERS_LENS_PATH"))
    parser.add_argument("--dtype", default=os.environ.get("JPEERS_DTYPE", "bf16"))
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18080)
    parser.add_argument("--max-model-len", type=int, default=16384)
    parser.add_argument("--gpu-memory-utilization", type=float, default=.65)
    args = parser.parse_args()
    os.environ.update(JPEERS_MODEL=args.model, JPEERS_DTYPE=args.dtype,
                      JPEERS_MAX_MODEL_LEN=str(args.max_model_len),
                      JPEERS_GPU_MEMORY_UTILIZATION=str(args.gpu_memory_utilization))
    if args.lens_path:
        os.environ["JPEERS_LENS_PATH"] = args.lens_path
    uvicorn.run("jpeers.vllm_runtime:app", host=args.host, port=args.port, workers=1)


if __name__ == "__main__":
    main()
