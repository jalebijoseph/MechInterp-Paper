"""Frozen HF inference with append-only Qwen tokens and explicit replay perturbations.

Torch and Transformers are imported only when a backend is constructed. A call
replays the exact supplied token IDs; generation then keeps a native local cache.
Block-output edits affect downstream blocks, not that block's already-made KV.
"""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
import importlib.metadata
import json
import math

import numpy as np


MODEL_ID = "Qwen/Qwen3-32B"
REVISION = "9216db5781bf21249d130ec9da846c4624c16137"
_ASSISTANT = "<|im_start|>assistant\n<think>\n\n</think>\n\n"


class HFBackend:
    def __init__(self, model, tokenizer, *, model_id, revision, dtype="float32"):
        import torch

        self.torch, self.model, self.tokenizer = torch, model, tokenizer
        self.model_id, self.revision = str(model_id), str(revision)
        self.layers = model.model.layers
        parameter = next(model.parameters())
        self.device, self.model_dtype = parameter.device, parameter.dtype
        actual_dtype = str(parameter.dtype).removeprefix("torch.")
        aliases = {"fp32": "float32", "bf16": "bfloat16", "fp16": "float16"}
        if aliases.get(str(dtype), str(dtype)) != actual_dtype:
            raise ValueError(f"declared dtype {dtype!r} differs from model dtype {actual_dtype}")
        if getattr(model.config, "_attn_implementation", None) != "eager":
            raise ValueError("Experiment 5 requires eager attention for explicit source masks")
        self.dtype = actual_dtype
        self.hidden_size = int(model.config.hidden_size)
        self.max_context = int(model.config.max_position_embeddings)
        tokenizer_cap = getattr(tokenizer, "model_max_length", None)
        if tokenizer_cap is not None and 0 < tokenizer_cap < self.max_context:
            self.max_context = int(tokenizer_cap)
        self.synthetic = True  # Only the pinned local loader establishes target provenance.
        self._info_base = None
        self.model.eval()
        self.model.requires_grad_(False)
        self._end = self.encode_text("<|im_end|>")
        self._newline = self.encode_text("\n")
        if len(self._end) != 1:
            raise ValueError("tokenizer must recognize the Qwen <|im_end|> control token")
        self.stop_ids = {self._end[0]}
        eos = getattr(tokenizer, "eos_token_id", None)
        if eos is not None:
            self.stop_ids.update(eos if isinstance(eos, (list, tuple)) else [eos])

    @classmethod
    def from_pretrained(cls, config: dict, cache_dir: str | None = None):
        """Load a prepared, pinned local snapshot. This method never downloads."""
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        settings = config.get("model", config)
        model_id = settings.get("model_id", settings.get("id", MODEL_ID))
        revision = settings.get("revision", REVISION)
        if model_id != MODEL_ID or revision != REVISION:
            raise ValueError("real inference requires the pinned Qwen3-32B revision")
        if settings.get("enable_thinking", False):
            raise ValueError("the initial Experiment 5 backend requires enable_thinking=False")
        dtype_name = settings.get("dtype", "bfloat16")
        if dtype_name not in ("bfloat16", "bf16"):
            raise ValueError("the pinned initial runtime uses BF16")
        common = dict(revision=revision, cache_dir=cache_dir, local_files_only=True,
                      trust_remote_code=False)
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_id, **common)
            model = AutoModelForCausalLM.from_pretrained(
                model_id, torch_dtype=torch.bfloat16, attn_implementation="eager",
                device_map=settings.get("device_map", "auto"), **common)
        except (OSError, ImportError) as exc:
            raise RuntimeError("Pinned Qwen3 files/runtime are unavailable locally; prepare the "
                               "model cache and declared dependencies before real inference. "
                               "No download was attempted.") from exc
        result = cls(model, tokenizer, model_id=model_id, revision=revision, dtype="bfloat16")
        result.synthetic = False
        canonical = result._canonical_format_check()
        if canonical["passed"] is not True:
            raise RuntimeError("Pinned tokenizer canonical non-thinking format verification failed: "
                               + json.dumps(canonical, sort_keys=True))
        return result

    def info(self) -> dict:
        if self._info_base is not None:
            result = dict(self._info_base, synthetic=self.synthetic)
            result["fingerprint"] = hashlib.sha256(json.dumps(result, sort_keys=True).encode()).hexdigest()
            return result
        versions = {}
        for name in ("torch", "transformers"):
            try:
                versions[name] = importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError:
                versions[name] = None
        vocab = self.tokenizer.get_vocab() if hasattr(self.tokenizer, "get_vocab") else None
        tokenizer_hash = hashlib.sha256(json.dumps(
            dict(vocab=vocab, chat_template=getattr(self.tokenizer, "chat_template", None)),
            sort_keys=True, default=str).encode()).hexdigest()
        config = self.model.config.to_dict() if hasattr(self.model.config, "to_dict") else vars(self.model.config)
        config_hash = hashlib.sha256(json.dumps(config, sort_keys=True, default=str).encode()).hexdigest()
        # The revision pins real checkpoint identity. Sampled parameters distinguish
        # local injected fixtures without hashing or retaining a 32B weight copy.
        samples = hashlib.sha256()
        for name, parameter in self.model.named_parameters():
            samples.update(f"{name}:{tuple(parameter.shape)}:{parameter.dtype}".encode())
            flat = parameter.detach().reshape(-1)
            samples.update(flat[:8].float().cpu().numpy().tobytes())
            samples.update(flat[-8:].float().cpu().numpy().tobytes())
        result = dict(model_id=self.model_id, revision=self.revision,
                      n_layers=len(self.layers), hidden_size=self.hidden_size,
                      max_context=self.max_context, dtype=self.dtype, synthetic=self.synthetic,
                      device=str(self.device), attention_implementation="eager",
                      tokenizer_sha256=tokenizer_hash, versions=versions,
                      config_sha256=config_hash, sampled_parameters_sha256=samples.hexdigest(),
                      layer_numbering="zero-based block outputs", enable_thinking=False)
        self._info_base = {key: value for key, value in result.items() if key != "synthetic"}
        result["fingerprint"] = hashlib.sha256(json.dumps(result, sort_keys=True).encode()).hexdigest()
        return result

    def encode_text(self, text: str) -> list[int]:
        return list(self.tokenizer.encode(text, add_special_tokens=False))

    def decode(self, ids: list[int]) -> str:
        return self.tokenizer.decode(list(ids), skip_special_tokens=False,
                                     clean_up_tokenization_spaces=False)

    def append_message(self, prefix: list[int], role: str, content: str, *, close: bool = True):
        if role not in ("system", "user", "assistant", "tool"):
            raise ValueError(f"unsupported Qwen role: {role!r}")
        if not isinstance(content, str):
            raise TypeError("message content must be a string")
        header = _ASSISTANT if role == "assistant" else f"<|im_start|>{role}\n"
        result = list(prefix) + self.encode_text(header)
        start = len(result)
        result.extend(self.encode_text(content))
        end = len(result)
        if close:
            result.extend(self._end + self._newline)
        self._check_tokens(result)
        return result, (start, end)

    def finish_assistant(self, prefix: list[int]) -> list[int]:
        result = list(prefix)
        suffix = self._end + self._newline
        if result[-len(suffix):] == suffix:
            return result
        if result and result[-1] in self.stop_ids:
            result.extend(self._newline)
        else:
            result.extend(suffix)
        self._check_tokens(result)
        return result

    def _canonical_format_check(self):
        """Compare native fragments with the tokenizer's independent chat template."""
        apply_template = getattr(self.tokenizer, "apply_chat_template", None)
        required = not self.synthetic
        if not callable(apply_template):
            return dict(passed=False if required else None, supported=False, required=required,
                        reason="tokenizer has no callable apply_chat_template")
        cases = {}
        try:
            messages = [dict(role="system", content="You are a helpful assistant."),
                        dict(role="user", content="Reply briefly.")]
            prefix = []
            for message in messages:
                prefix, _ = self.append_message(prefix, message["role"], message["content"])
            opened, _ = self.append_message(prefix, "assistant", "", close=False)
            closed, _ = self.append_message(prefix, "assistant", "Recorded.")
            for name, history, native, generation_prompt in (
                    ("generation_prompt", messages, opened, True),
                    ("closed_assistant_receipt", messages + [dict(role="assistant", content="Recorded.")],
                     closed, False)):
                canonical = list(apply_template(history, tokenize=True,
                                                add_generation_prompt=generation_prompt,
                                                enable_thinking=False))
                cases[name] = dict(passed=canonical == native, canonical_tokens=len(canonical),
                                   native_tokens=len(native))
                if canonical != native:
                    cases[name]["first_difference"] = next(
                        (index for index, (actual, expected) in enumerate(zip(native, canonical))
                         if actual != expected), min(len(native), len(canonical)))
            return dict(passed=all(case["passed"] for case in cases.values()),
                        supported=True, required=required, cases=cases)
        except Exception as exc:
            return dict(passed=False, supported=True, required=required, cases=cases,
                        error=f"{type(exc).__name__}: {exc}")

    def _check_tokens(self, tokens, extra=0):
        if not tokens or any(isinstance(i, bool) or not isinstance(i, (int, np.integer)) or i < 0 for i in tokens):
            raise ValueError("a nonempty sequence of nonnegative integer native token IDs is required")
        if len(tokens) + extra > self.max_context:
            raise ValueError(f"context cap exceeded: {len(tokens)} + {extra} > {self.max_context}")

    def _prepare(self, tokens, capture, masks, edits, extra=0):
        self._check_tokens(tokens, extra)
        limit = len(tokens) + extra
        clean_capture = {}
        for layer, positions in capture.items():
            if not isinstance(layer, int) or not 0 <= layer < len(self.layers):
                raise ValueError("capture layer is outside zero-based block outputs")
            if any(not isinstance(p, int) or not 0 <= p < len(tokens) for p in positions):
                raise ValueError("capture position is outside the supplied prefix")
            clean_capture[layer] = sorted(set(positions))
        clean_masks = []
        for mask in masks:
            start = mask["start_query"]
            spans = [tuple(span) for span in mask["key_spans"]]
            if not isinstance(start, int) or not 0 <= start <= limit:
                raise ValueError("mask start_query is outside the requested context")
            if any(len(s) != 2 or any(not isinstance(p, int) for p in s) or
                   not 0 <= s[0] < s[1] <= len(tokens) for s in spans):
                raise ValueError("mask key spans must be nonempty prefix intervals")
            clean_masks.append(dict(key_spans=spans, start_query=start, tag=str(mask.get("tag", ""))))
        # Refuse an all-blocked row: a finite additive minimum otherwise yields
        # uniform attention, while negative infinity yields NaNs on some models.
        active_spans = []
        for mask in sorted(clean_masks, key=lambda value: value["start_query"]):
            active_spans.extend(mask["key_spans"])
            first_allowed = 0
            for start, end in sorted(active_spans):
                if start > first_allowed:
                    break
                first_allowed = max(first_allowed, end)
            if mask["start_query"] < limit and first_allowed > mask["start_query"]:
                raise ValueError("source masks would block every causal key for a query")
        clean_edits = []
        for edit in edits:
            entry = dict(edit)
            if entry["kind"] not in ("delete", "patch", "random_delete", "random_patch"):
                raise ValueError("unsupported block-output edit kind")
            if not isinstance(entry["layer"], int) or not 0 <= entry["layer"] < len(self.layers):
                raise ValueError("edit layer is outside zero-based block outputs")
            if not isinstance(entry["position"], int) or not 0 <= entry["position"] < limit:
                raise ValueError("edit position is outside the requested context")
            basis = np.asarray(entry["basis"], dtype=np.float32)
            if (basis.ndim != 2 or basis.shape[0] != self.hidden_size or
                    not 0 < basis.shape[1] <= self.hidden_size or not np.isfinite(basis).all() or
                    not np.allclose(basis.T @ basis, np.eye(basis.shape[1]), atol=1e-5, rtol=1e-5)):
                raise ValueError("basis must have finite orthonormal columns with shape [hidden_size, rank]")
            strength = float(entry.get("strength", 1))
            if not math.isfinite(strength) or not 0 <= strength <= 1:
                raise ValueError("edit strength must be finite and within [0,1]")
            entry.update(basis=basis, strength=strength, seed=int(entry.get("seed", 0)),
                         tag=str(entry.get("tag", "")))
            if entry["kind"].endswith("patch"):
                donor = np.asarray(entry["donor"], dtype=np.float32)
                if donor.shape != (self.hidden_size,) or not np.isfinite(donor).all():
                    raise ValueError("patch donor must be a finite full residual vector")
                entry["donor"] = donor
            clean_edits.append(entry)
        return clean_capture, clean_masks, clean_edits

    def _attention_mask(self, offset, count, masks):
        torch = self.torch
        queries = torch.arange(offset, offset + count, device=self.device)
        keys = torch.arange(offset + count, device=self.device)
        blocked = keys[None, :] > queries[:, None]
        for mask in masks:
            active = queries >= mask["start_query"]
            for start, end in mask["key_spans"]:
                blocked |= active[:, None] & ((keys >= start) & (keys < end))[None, :]
        result = torch.zeros((count, offset + count), dtype=self.model_dtype, device=self.device)
        return result.masked_fill(blocked, torch.finfo(self.model_dtype).min)[None, None]

    def _edit(self, hidden, edit):
        """Match geometry before dtype rounding, and report the actual displacement."""
        torch = self.torch
        h = hidden.detach().float()
        if not bool(torch.isfinite(h).all()):
            raise ValueError("edit recipient residual must be finite")
        basis = torch.as_tensor(edit["basis"], device=h.device)
        reference = -h if edit["kind"].endswith("delete") else torch.as_tensor(edit["donor"], device=h.device) - h
        targeted = edit["strength"] * (basis @ (basis.T @ reference))
        realized_target = (h + targeted).to(hidden.dtype).float() - h
        delta = targeted
        if edit["kind"].startswith("random_"):
            # Local NumPy generator leaves both NumPy and Torch global RNG alone.
            rng = np.random.default_rng(edit["seed"])
            hv = h.cpu().numpy().astype(np.float64)
            dv = realized_target.cpu().numpy().astype(np.float64)
            norm_h, norm_d = np.linalg.norm(hv), np.linalg.norm(dv)
            direction = rng.standard_normal(len(hv))
            if norm_h == 0:
                matched = norm_d * direction / np.linalg.norm(direction)
            else:
                unit = hv / norm_h
                along = float(dv @ unit)
                perpendicular_norm = math.sqrt(max(0., norm_d ** 2 - along ** 2))
                direction -= (direction @ unit) * unit
                norm_direction = np.linalg.norm(direction)
                if norm_direction < 1e-12 and perpendicular_norm > 1e-12:
                    raise ValueError("no perpendicular direction for matched random edit")
                matched = along * unit
                if perpendicular_norm:
                    matched += perpendicular_norm * direction / norm_direction
            delta = torch.as_tensor(matched, dtype=torch.float32, device=h.device)
        changed = (h + delta).to(hidden.dtype)
        if not bool(torch.isfinite(changed).all()):
            raise ValueError("edit produced a nonfinite model-dtype residual")
        actual = changed.float() - h
        actual_norm = float(actual.norm().item())
        actual_dot = float((actual * h).sum().item())
        reference_norm = float(realized_target.norm().item())
        reference_dot = float((realized_target * h).sum().item())
        receipt = {key: edit[key] for key in ("layer", "position", "kind", "strength", "seed", "tag")}
        receipt.update(rank=int(basis.shape[1]), dtype=self.dtype,
                       requested_norm=float(targeted.norm().item()),
                       requested_inner_product=float((targeted * h).sum().item()),
                       realized_norm=actual_norm, realized_inner_product=actual_dot,
                       reference_realized_norm=reference_norm,
                       reference_realized_inner_product=reference_dot,
                       norm_matching_error=actual_norm - reference_norm,
                       inner_product_matching_error=actual_dot - reference_dot)
        return changed, receipt

    @contextmanager
    def _hooks(self, capture, edits, offset, count, captures, receipt):
        handles = []
        try:
            for index in sorted(set(capture) | {entry["layer"] for entry in edits}):
                def hook(module, args, output, layer=index):
                    hidden = output[0] if isinstance(output, tuple) else output
                    relevant = [entry for entry in edits if entry["layer"] == layer and
                                offset <= entry["position"] < offset + count]
                    if relevant:
                        hidden = hidden.clone()
                        for entry in relevant:
                            position = entry["position"] - offset
                            changed, realized = self._edit(hidden[0, position], entry)
                            hidden[0, position] = changed
                            receipt["edits"].append(realized)
                    for position in capture.get(layer, ()):
                        if offset <= position < offset + count:
                            captures.setdefault(layer, {})[position] = hidden[0, position - offset].detach().float().cpu().numpy().copy()
                    return (hidden, *output[1:]) if isinstance(output, tuple) else hidden
                handles.append(self.layers[index].register_forward_hook(hook))
            yield
        finally:
            for handle in handles:
                handle.remove()

    def _receipt(self, count, masks):
        return dict(fingerprint=self.info()["fingerprint"], token_count=count,
                    mode="exact-token replay then native cache", masks=masks, edits=[], forward_calls=[])

    def _forward(self, tokens, capture, masks, edits, *, offset=0, past=None,
                 captures=None, receipt=None, output_attentions=False):
        torch = self.torch
        captures = {} if captures is None else captures
        receipt = self._receipt(offset + len(tokens), masks) if receipt is None else receipt
        with self._hooks(capture, edits, offset, len(tokens), captures, receipt), torch.inference_mode():
            output = self.model(input_ids=torch.tensor([tokens], dtype=torch.long, device=self.device),
                                attention_mask=self._attention_mask(offset, len(tokens), masks),
                                past_key_values=past, use_cache=True,
                                output_attentions=output_attentions, return_dict=True)
        receipt["forward_calls"].append(dict(offset=offset, query_tokens=len(tokens), key_tokens=offset + len(tokens)))
        return output, captures, receipt

    def observe(self, tokens: list[int], capture: dict[int, list[int]], *, masks=(), edits=()) -> dict:
        capture, masks, edits = self._prepare(tokens, capture, masks, edits)
        output, captures, receipt = self._forward(tokens, capture, masks, edits)
        return dict(captures=captures, logits=output.logits[0, -1].detach().float().cpu().numpy().copy(), receipt=receipt)

    def _sampler(self, sampler):
        temperature = float(sampler.get("temperature", .7))
        top_p, min_p = float(sampler.get("top_p", .8)), float(sampler.get("min_p", 0))
        top_k = sampler.get("top_k", 20)
        if (not all(math.isfinite(x) for x in (temperature, top_p, min_p)) or temperature < 0 or
                not 0 < top_p <= 1 or not 0 <= min_p <= 1 or
                not isinstance(top_k, int) or isinstance(top_k, bool) or top_k < 0):
            raise ValueError("invalid temperature/top_p/top_k/min_p sampler")
        return dict(temperature=temperature, top_p=top_p, top_k=top_k, min_p=min_p)

    def _sample(self, logits, sampler, generator):
        torch = self.torch
        logits = logits.detach().float().cpu().clone()
        if not torch.isfinite(logits).all():
            raise ValueError("model emitted nonfinite logits")
        if sampler["temperature"] == 0:
            return int(logits.argmax().item())
        logits /= sampler["temperature"]
        if sampler["top_k"]:
            cutoff = torch.topk(logits, min(sampler["top_k"], logits.numel())).values[-1]
            logits[logits < cutoff] = -float("inf")
        probabilities = torch.softmax(logits, dim=-1)
        if sampler["min_p"]:
            logits[probabilities < probabilities.max() * sampler["min_p"]] = -float("inf")
        if sampler["top_p"] < 1:
            ordered, indices = torch.sort(logits, descending=True)
            cumulative = torch.softmax(ordered, dim=-1).cumsum(dim=-1)
            remove = cumulative > sampler["top_p"]
            remove[1:] = remove[:-1].clone()
            remove[0] = False
            logits[indices[remove]] = -float("inf")
        return int(torch.multinomial(torch.softmax(logits, dim=-1), 1, generator=generator).item())

    def generate(self, tokens: list[int], *, sampler: dict, seed: int, max_new_tokens: int,
                 masks=(), edits=()) -> dict:
        if not isinstance(max_new_tokens, int) or isinstance(max_new_tokens, bool) or max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be a positive integer")
        capture, masks, edits = self._prepare(tokens, {}, masks, edits, max_new_tokens)
        sampler = self._sampler(sampler)
        generator = self.torch.Generator(device="cpu")
        generator.manual_seed(int(seed))
        output, _, receipt = self._forward(tokens, capture, masks, edits)
        generated = []
        for index in range(max_new_tokens):
            token = self._sample(output.logits[0, -1], sampler, generator)
            generated.append(token)
            if token in self.stop_ids or index == max_new_tokens - 1:
                break
            output, _, receipt = self._forward([token], capture, masks, edits,
                                              offset=len(tokens) + index, past=output.past_key_values,
                                              receipt=receipt)
        hit_cap = generated[-1] not in self.stop_ids
        receipt.update(generated_tokens=len(generated), hit_cap=hit_cap, seed=int(seed), sampler=sampler)
        text = self.tokenizer.decode(generated, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        return dict(text=text, token_ids=generated, hit_cap=hit_cap, receipt=receipt)

    def validate_runtime(self) -> dict:
        """Run real local forwards; a mismatch or unsupported attention never passes."""
        checks = {"canonical_non_thinking_format": self._canonical_format_check()}
        try:
            prefix, span = self.append_message([], "user", "x")
            native, receipt_span = self.append_message(prefix, "assistant", "Recorded.", close=False)
            finished = self.finish_assistant(native)
            expected = "<|im_start|>user\nx<|im_end|>\n" + _ASSISTANT + "Recorded.<|im_end|>\n"
            checks["non_thinking_format"] = dict(passed=(self.decode(finished) == expected and
                native[:len(prefix)] == prefix and self.decode(native[slice(*receipt_span)]) == "Recorded." and
                self.decode(prefix[slice(*span)]) == "x"))
            # Repeat a valid native fragment so BPE tokenizers and byte fixtures
            # both supply exactly ten positions, without assuming word lengths.
            tokens = (self.encode_text("Runtime replay") * 10)[:10]
            self._check_tokens(tokens)
            if len(tokens) < 8 or self.hidden_size < 2 or len(self.layers) < 2:
                raise ValueError("runtime validation requires >=8 tokens, width >=2, and >=2 blocks")
            width = self.hidden_size
            basis = np.zeros((width, 1), dtype=np.float32)
            basis[0, 0] = 1
            mask = dict(key_spans=[(0, 2)], start_query=3, tag="validation-source")
            changes = [dict(layer=0, position=2, kind="delete", basis=basis, strength=.5, seed=42, tag="validation-delete"),
                       dict(layer=len(self.layers) - 1, position=5, kind="random_patch", basis=basis,
                            donor=np.ones(width, dtype=np.float32), strength=.5, seed=43, tag="validation-patch")]
            capture = {0: [2, 5, 9], len(self.layers) - 1: [2, 5, 9]}
            for label, masks, edits in (("cache_replay", [], []),
                                        ("masked_edited_cache_replay", [mask], changes)):
                full, full_captures, full_receipt = self._forward(tokens, capture, masks, edits)
                segmented_captures, segmented_receipt, past = {}, self._receipt(len(tokens), masks), None
                final = None
                for start, end in ((0, 3), (3, 6), (6, 7), (7, 8), (8, 9), (9, 10)):
                    final, segmented_captures, segmented_receipt = self._forward(
                        tokens[start:end], capture, masks, edits, offset=start, past=past,
                        captures=segmented_captures, receipt=segmented_receipt)
                    past = final.past_key_values
                full_logits = full.logits[0, -1].detach().float().cpu().numpy()
                segmented_logits = final.logits[0, -1].detach().float().cpu().numpy()
                max_logit_error = float(np.max(np.abs(full_logits - segmented_logits)))
                max_capture_error = max(float(np.max(np.abs(full_captures[layer][pos] - segmented_captures[layer][pos])))
                                        for layer, positions in capture.items() for pos in positions)
                tolerance = .025 if self.dtype == "bfloat16" else (.005 if self.dtype == "float16" else 1e-5)
                passed = np.allclose(full_logits, segmented_logits, atol=tolerance, rtol=tolerance)
                passed &= max_capture_error <= tolerance * max(1., max(float(np.max(np.abs(v))) for vs in full_captures.values() for v in vs.values()))
                if edits:
                    passed &= len(full_receipt["edits"]) == len(segmented_receipt["edits"]) == len(edits)
                    passed &= all(e["realized_norm"] > 0 for e in full_receipt["edits"])
                checks[label] = dict(passed=bool(passed), max_logit_error=max_logit_error,
                                     max_capture_error=max_capture_error, atol=tolerance,
                                     segmented_forward_calls=len(segmented_receipt["forward_calls"]),
                                     realized_edits=full_receipt["edits"])
            attended, _, _ = self._forward(tokens, {}, [mask], [], output_attentions=True)
            attentions = getattr(attended, "attentions", None)
            if not attentions or any(value is None for value in attentions):
                checks["blocked_attention"] = dict(passed=False, supported=False,
                                                   reason="model did not expose actual eager attention weights")
            else:
                blocked_mass = max(float(value[..., 3:, :2].detach().float().abs().sum(-1).max().item()) for value in attentions)
                future_mass = max(float(value[0, :, 0, 1:].detach().float().abs().max().item()) for value in attentions)
                checks["blocked_attention"] = dict(passed=blocked_mass <= 1e-7 and future_mass <= 1e-7,
                                                   supported=True, max_blocked_key_mass=blocked_mass,
                                                   max_future_attention=future_mass)
        except Exception as exc:
            checks["runtime_error"] = dict(passed=False, error=f"{type(exc).__name__}: {exc}")
        passed = bool(checks) and all(row["passed"] is True or
            (row["passed"] is None and row.get("required") is False) for row in checks.values())
        return dict(passed=passed, checks=checks, fingerprint=self.info()["fingerprint"],
                    synthetic=self.synthetic, target_gate=(passed and not self.synthetic and
                        self.model_id == MODEL_ID and self.revision == REVISION and
                        checks["canonical_non_thinking_format"]["passed"] is True))
