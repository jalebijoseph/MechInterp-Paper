"""Load a model with its Jacobian lens and generate while reading / editing J-space.

The lens (Gurnee et al. 2026) transports a residual ``h`` at layer ``l`` into the
final-layer basis with the averaged Jacobian ``J_l`` and decodes it with the model's
own unembedding::

    lens_l(h) = unembed(J_l @ h)

The J-lens vector of vocabulary token ``t`` at layer ``l`` is the residual direction
that most raises that readout, ``v_{l,t} = J_l^T (g * W_U[t])``, where ``g`` is the final
norm's gain. "J-space" at a position is the handful of J-lens vectors with the highest
lens logits there.

:class:`LensedModel` wraps an HF model and a fitted :class:`jlens.JacobianLens` and runs
a token-by-token generation loop with forward hooks on the workspace layers that
(1) apply :class:`~jpeers.schemas.InterventionSpec` edits and (2) record the lens readout.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field

import numpy as np
import torch
from torch import nn

from jpeers.registry import BAND_PRESETS, ModelSpec, band_layers
from jpeers.schemas import (
    GenerateRequest,
    GenerateResponse,
    InterventionSpec,
    PositionSelector,
    RecordSpec,
    TokenSpan,
)
from jpeers.trace import encode_arrays

logger = logging.getLogger(__name__)

_CHUNK = 256  # positions per lens-logit chunk (bounds the [chunk, vocab] buffer)
_FALLBACK_TEMPLATE = (
    "{% for m in messages %}### {{ m['role'] | capitalize }}:\n{{ m['content'] }}\n\n"
    "{% endfor %}{% if add_generation_prompt %}### Assistant:\n{% endif %}"
)


def _dtype(name: str) -> torch.dtype:
    return {"bf16": torch.bfloat16, "bfloat16": torch.bfloat16, "fp16": torch.float16,
            "float16": torch.float16, "fp32": torch.float32, "float32": torch.float32}[name]


class LensedModel:
    """An HF causal LM plus a fitted Jacobian lens.

    Args:
        spec: Registry entry (see :mod:`jpeers.registry`).
        hf_model / tokenizer / lens: Pre-loaded objects (tests); loaded from ``spec``
            when omitted.
        device: ``"cuda"``, ``"cpu"`` or ``"auto"`` (``device_map="auto"`` sharding).
        dtype: Model weight dtype.
    """

    def __init__(self, spec: ModelSpec, *, hf_model=None, tokenizer=None, lens=None,
                 device: str = "auto", dtype: str = "bf16") -> None:
        import jlens
        import transformers

        self.spec = spec
        t0 = time.perf_counter()
        if tokenizer is None:
            tokenizer = transformers.AutoTokenizer.from_pretrained(spec.hf_id)
        if hf_model is None:
            kwargs: dict = {"dtype": _dtype(dtype)}
            if device == "auto":
                kwargs["device_map"] = "auto" if torch.cuda.is_available() else None
            hf_model = transformers.AutoModelForCausalLM.from_pretrained(spec.hf_id, **kwargs)
            if device not in ("auto",):
                hf_model = hf_model.to(device)
        self.hf = hf_model
        self.tokenizer = tokenizer
        self.lm = jlens.from_hf(hf_model, tokenizer, force_bos=not spec.chat)
        self.n_layers = self.lm.n_layers
        self.d_model = self.lm.d_model

        if lens is None:
            lens = load_lens(spec)
        missing = [layer for layer in self.band() if layer not in lens.jacobians]
        if missing:
            raise ValueError(f"lens for {spec.name} lacks band layers {missing}")
        self.lens = lens
        self._J_cache: dict[tuple[int, torch.device], torch.Tensor] = {}

        self._W_U: torch.Tensor = self.lm._lm_head.weight  # [V, d]
        self._gain = _final_norm_gain(self.lm._final_norm, self.d_model).to(self._W_U.device)
        self.vocab_size = self._W_U.shape[0]
        self._vocab_strings: list[str] | None = None
        self._word_like: torch.Tensor | None = None
        self._word_like_dev: dict[torch.device, torch.Tensor] = {}
        self._eos_ids = _eos_ids(hf_model, tokenizer)
        self.lock = threading.Lock()
        logger.info("loaded %s (%d layers, d=%d, band %s) in %.0fs", spec.name, self.n_layers,
                    self.d_model, self.band(), time.perf_counter() - t0)

    # ------------------------------------------------------------------ layers / vocab

    def band(self, which: str | list[int] = "band") -> list[int]:
        if isinstance(which, list):
            return sorted(set(which))
        band = self.spec.band if which == "band" else BAND_PRESETS[which]
        layers = band_layers(self.n_layers, band)
        return [layer for layer in layers if layer in self.lens.jacobians] if hasattr(
            self, "lens") else layers

    @property
    def vocab_strings(self) -> list[str]:
        if self._vocab_strings is None:
            self._vocab_strings = self.tokenizer.batch_decode(
                [[i] for i in range(self.vocab_size)])
        return self._vocab_strings

    @property
    def word_like(self) -> torch.Tensor:
        """Bool mask over the vocab: tokens that are (part of) a word, >= 2 letters."""
        if self._word_like is None:
            mask = [s.strip().isalpha() and len(s.strip()) >= 2 for s in self.vocab_strings]
            self._word_like = torch.tensor(mask, dtype=torch.bool)
        return self._word_like

    def word_like_on(self, device: torch.device) -> torch.Tensor:
        if device not in self._word_like_dev:
            self._word_like_dev[device] = self.word_like.to(device)
        return self._word_like_dev[device]

    def word_token_ids(self, word: str) -> list[int]:
        """Candidate token ids for ``word``: the first token of each of its common
        surface forms (with / without leading space, capitalised). The first entry is
        the primary id used to build J-lens directions."""
        ids: list[int] = []
        for form in (" " + word, word, " " + word.capitalize(), word.capitalize()):
            toks = self.tokenizer(form, add_special_tokens=False).input_ids
            if toks and toks[0] not in ids:
                ids.append(toks[0])
        if not ids:
            raise ValueError(f"cannot tokenize {word!r}")
        return ids

    # ------------------------------------------------------------------ lens math

    def J(self, layer: int, device: torch.device) -> torch.Tensor:
        key = (layer, device)
        if key not in self._J_cache:
            self._J_cache[key] = self.lens.jacobians[layer].to(device=device,
                                                               dtype=torch.float32)
        return self._J_cache[key]

    def lens_logits(self, layer: int, h: torch.Tensor, *, use_jacobian: bool = True
                    ) -> torch.Tensor:
        """``unembed(J_l h)`` for ``h`` of shape ``[n, d]`` -> ``[n, V]`` float32."""
        x = h.float()
        if use_jacobian:
            x = x @ self.J(layer, x.device).T
        return self.lm.unembed(x).float()

    def lens_directions(self, layer: int, token_ids: torch.Tensor | list[int],
                        device: torch.device) -> torch.Tensor:
        """Unit J-lens vectors ``v_{l,t}`` for ``token_ids`` (any shape) -> ``[..., d]``."""
        ids = torch.as_tensor(token_ids, device=self._W_U.device)
        rows = self._W_U[ids.reshape(-1)].float() * self._gain  # [n, d]
        v = rows.to(device) @ self.J(layer, device)  # (g*W_U[t]) @ J  ==  J^T (g*W_U[t])
        v = v / v.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        return v.reshape(*ids.shape, self.d_model)

    # ------------------------------------------------------------------ prompts

    def render(self, req: GenerateRequest) -> tuple[str, bool]:
        """Prompt text, and whether the tokenizer should add special tokens itself."""
        if req.prompt is not None:
            return req.prompt + req.assistant_prefill, True
        if not req.messages:
            raise ValueError("request needs `messages` or `prompt`")
        tok = self.tokenizer
        template = None if getattr(tok, "chat_template", None) else _FALLBACK_TEMPLATE
        kwargs = dict(tokenize=False, add_generation_prompt=True, chat_template=template,
                      **self.spec.template_kwargs)
        try:
            text = tok.apply_chat_template(req.messages, **kwargs)
        except Exception as exc:  # e.g. Gemma-2 rejects the system role
            if req.messages[0].get("role") != "system":
                raise
            logger.debug("template rejected system role (%s); folding into user turn", exc)
            msgs = _fold_system(req.messages)
            text = tok.apply_chat_template(msgs, **kwargs)
        return text + req.assistant_prefill, False

    def encode(self, text: str, add_special: bool) -> tuple[torch.Tensor, list[tuple[int, int]]]:
        enc = self.tokenizer(text, add_special_tokens=add_special, return_offsets_mapping=True)
        if add_special and self.tokenizer.bos_token_id is not None and (
                not enc.input_ids or enc.input_ids[0] != self.tokenizer.bos_token_id):
            enc_ids = [self.tokenizer.bos_token_id, *enc.input_ids]
            offsets = [(0, 0), *enc.offset_mapping]
        else:
            enc_ids, offsets = enc.input_ids, enc.offset_mapping
        ids = torch.tensor([enc_ids], device=self.lm.input_device)
        return ids, [tuple(o) for o in offsets]

    # ------------------------------------------------------------------ generation

    @torch.no_grad()
    def generate(self, req: GenerateRequest) -> GenerateResponse:
        with self.lock:
            return self._generate(req)

    def _generate(self, req: GenerateRequest) -> GenerateResponse:
        t0 = time.perf_counter()
        text, add_special = self.render(req)
        input_ids, offsets = self.encode(text, add_special)
        prompt_len = input_ids.shape[1]
        spans = _locate_spans(text, offsets, req, prompt_len)

        run = _Run(self, req, prompt_len, spans)
        gen = torch.Generator(device="cpu").manual_seed(req.seed)
        generated: list[int] = []
        t_prefill = 0.0
        with run.hooks():
            run.offset = 0
            out = self.hf(input_ids=input_ids, use_cache=True)
            t_prefill = time.perf_counter() - t0
            past = out.past_key_values
            logits = out.logits[0, -1]
            run.note_model_logits(out.logits[0])
            for _ in range(req.max_new_tokens):
                next_id = _sample(logits, req.temperature, req.top_p, gen)
                generated.append(next_id)
                if next_id in self._eos_ids or len(generated) == req.max_new_tokens:
                    break
                if req.stop and _hit_stop(self.tokenizer.decode(generated), req.stop):
                    break
                run.offset = prompt_len + len(generated) - 1
                step_ids = torch.tensor([[next_id]], device=input_ids.device)
                out = self.hf(input_ids=step_ids, past_key_values=past, use_cache=True)
                past = out.past_key_values
                logits = out.logits[0, -1]
                run.note_model_logits(out.logits[0])

        all_ids = input_ids[0].tolist() + generated
        gen_text = self.tokenizer.decode(
            [t for t in generated if t not in self._eos_ids], skip_special_tokens=True)
        for s in req.stop:
            if s and s in gen_text:
                gen_text = gen_text[: gen_text.index(s)]
        arrays = run.finish()
        vocab = {}
        if arrays is not None:
            for i in np.unique(arrays["topk_ids"]).tolist():
                vocab[int(i)] = self.vocab_strings[i]
        return GenerateResponse(
            model=self.spec.name,
            text=gen_text,
            prompt_len=prompt_len,
            n_generated=len(generated),
            tokens=[self.vocab_strings[i] for i in all_ids],
            spans=[TokenSpan(label=lbl, start=a, end=b) for lbl, rs in spans.items()
                   for a, b in rs],
            layers=run.record_layers,
            watch=run.watch_words,
            watch_ids=run.watch_ids,
            vocab=vocab,
            trace_encoding=req.trace_encoding,
            trace=None if arrays is None else encode_arrays(arrays, req.trace_encoding),
            interventions_applied=run.n_edits,
            timings={"prefill_s": t_prefill, "total_s": time.perf_counter() - t0},
        )


# ====================================================================== per-request state


@dataclass
class _LayerRecord:
    positions: list[np.ndarray] = field(default_factory=list)
    topk_ids: list[np.ndarray] = field(default_factory=list)
    topk_logits: list[np.ndarray] = field(default_factory=list)
    watch_ranks: list[np.ndarray] = field(default_factory=list)


class _Run:
    """Hooks + buffers for one generation call."""

    def __init__(self, model: LensedModel, req: GenerateRequest, prompt_len: int,
                 spans: dict[str, list[tuple[int, int]]]) -> None:
        self.m = model
        self.req = req
        self.rec: RecordSpec = req.record
        self.prompt_len = prompt_len
        self.spans = spans
        self.offset = 0
        self.n_edits = 0

        self.record_layers: list[int] = []
        if self.rec.enabled:
            self.record_layers = model.band(self.rec.layers)[:: max(1, self.rec.stride)]
        self.buffers = {layer: _LayerRecord() for layer in self.record_layers}
        self.model_top1: list[np.ndarray] = []

        self.watch_words = list(self.rec.watch)
        self.watch_ids = [model.word_token_ids(w) for w in self.watch_words]
        self._watch_flat: torch.Tensor | None = None
        self._watch_owner: torch.Tensor | None = None
        if self.watch_ids:
            flat, owner = [], []
            for wi, ids in enumerate(self.watch_ids):
                flat += ids
                owner += [wi] * len(ids)
            self._watch_flat = torch.tensor(flat)
            self._watch_owner = torch.tensor(owner)

        # prompt-position recording cap: keep the latest `max_positions`
        self._record_floor = max(0, prompt_len - self.rec.max_positions)

        self.interventions: list[tuple[InterventionSpec, set[int]]] = [
            (iv, set(model.band(iv.layers))) for iv in req.interventions]
        self._fixed_dirs: dict[tuple[int, int, torch.device], torch.Tensor] = {}
        self._layer_scale: dict[int, float] = {}
        self.hooked_layers = sorted(set(self.record_layers).union(
            *[layers for _, layers in self.interventions]) if self.interventions
            else set(self.record_layers))

    # ------------------------------------------------------------------ hooks

    def hooks(self):
        run = self

        class _Ctx:
            def __enter__(self_inner):
                self_inner.handles = [
                    run.m.lm.layers[layer].register_forward_hook(run._make_hook(layer))
                    for layer in run.hooked_layers]
                return run

            def __exit__(self_inner, *exc):
                for h in self_inner.handles:
                    h.remove()

        return _Ctx()

    def _make_hook(self, layer: int):
        ivs = [iv for iv, layers in self.interventions if layer in layers]
        record = layer in self.buffers

        def hook(module: nn.Module, inputs, output):
            is_tensor = torch.is_tensor(output)
            hidden = output if is_tensor else output[0]
            seq = hidden.shape[1]
            pos = torch.arange(self.offset, self.offset + seq)
            h = hidden[0]
            if record and self.rec.when == "pre":
                self._record(layer, h, pos)
            if ivs:
                h_new = h.float()
                changed = False
                for iv in ivs:
                    mask = self._mask(iv.positions, pos)
                    if mask.any():
                        idx = mask.nonzero(as_tuple=True)[0].to(h.device)
                        h_new[idx] = self._apply(iv, layer, h_new[idx], pos)
                        changed = True
                        self.n_edits += 1
                if changed:
                    h = h_new.to(hidden.dtype)
                    hidden = h.unsqueeze(0)
            if record and self.rec.when == "post":
                self._record(layer, h, pos)
            if not ivs:
                return None
            return hidden if is_tensor else (hidden, *output[1:])

        return hook

    # ------------------------------------------------------------------ positions

    def _mask(self, sel: PositionSelector, pos: torch.Tensor) -> torch.Tensor:
        P = self.prompt_len
        if sel.scope == "all":
            m = torch.ones_like(pos, dtype=torch.bool)
        elif sel.scope == "prompt":
            m = pos < P
        elif sel.scope == "generated":
            m = pos >= P
        elif sel.scope == "last":
            m = pos == P - 1
        elif sel.scope == "spans":
            m = torch.zeros_like(pos, dtype=torch.bool)
            for label in sel.labels:
                for a, b in self.spans.get(label, []):
                    m |= (pos >= a) & (pos < b)
            if sel.include_generated:
                m |= pos >= P
        else:
            raise ValueError(sel.scope)
        if sel.skip_first:
            m &= pos >= sel.skip_first
        return m

    # ------------------------------------------------------------------ recording

    def _record(self, layer: int, h: torch.Tensor, pos: torch.Tensor) -> None:
        mask = self._mask(self.rec.positions, pos) & (pos >= self._record_floor)
        if not mask.any():
            return
        idx = mask.nonzero(as_tuple=True)[0]
        buf = self.buffers[layer]
        use_j = self.rec.lens == "jacobian"
        K = self.rec.top_k
        for start in range(0, len(idx), _CHUNK):
            sub = idx[start:start + _CHUNK]
            logits = self.m.lens_logits(layer, h[sub.to(h.device)], use_jacobian=use_j)
            if self._watch_flat is not None:
                if self._watch_flat.device != logits.device:
                    self._watch_flat = self._watch_flat.to(logits.device)
                    self._watch_owner = self._watch_owner.to(logits.device)
                w_ids = self._watch_flat
                w_logit = logits[:, w_ids].contiguous()  # [n, C]
                asc = logits.sort(dim=-1).values
                # rank = number of vocab entries with a strictly higher logit
                ranks = logits.shape[-1] - torch.searchsorted(asc, w_logit, right=True)
                owner = self._watch_owner
                per_word = torch.full((len(sub), len(self.watch_ids)), self.m.vocab_size,
                                      device=logits.device, dtype=ranks.dtype)
                per_word = per_word.scatter_reduce(
                    1, owner.expand(len(sub), -1), ranks, reduce="amin")
                buf.watch_ranks.append(per_word.to(torch.int32))
            if self.rec.word_like_only:
                logits = logits.masked_fill(~self.m.word_like_on(logits.device), float("-inf"))
            top = logits.topk(K, dim=-1)
            buf.topk_ids.append(top.indices.to(torch.int32))
            buf.topk_logits.append(top.values.to(torch.float16))
            buf.positions.append(pos[sub].numpy())

    def note_model_logits(self, logits: torch.Tensor) -> None:
        if self.rec.enabled and self.rec.include_model_logits:
            self.model_top1.append(logits.argmax(-1))

    def finish(self) -> dict[str, np.ndarray] | None:
        """Stack the per-chunk buffers (kept on device during generation to avoid a
        host sync per layer per token) into ``[P, L, ...]`` numpy arrays."""
        if not self.record_layers or not self.buffers[self.record_layers[0]].positions:
            return None

        def stack(field: str) -> np.ndarray:
            per_layer = [torch.cat([t.cpu() for t in getattr(self.buffers[l], field)])
                         for l in self.record_layers]
            return torch.stack(per_layer, dim=1).numpy()

        out = {
            "positions": np.concatenate(self.buffers[self.record_layers[0]].positions),
            "topk_ids": stack("topk_ids"),
            "topk_logits": stack("topk_logits"),
        }
        if self.watch_ids:
            out["watch_ranks"] = stack("watch_ranks")
        else:
            out["watch_ranks"] = np.zeros((len(out["positions"]), len(self.record_layers), 0),
                                          dtype=np.int32)
        if self.model_top1:
            top1 = torch.cat([t.cpu() for t in self.model_top1]).numpy()
            out["model_top1"] = top1[out["positions"]]
        return out

    # ------------------------------------------------------------------ interventions

    def _scale(self, layer: int, h: torch.Tensor) -> float:
        """Typical residual norm at this layer: median over non-sink prompt positions,
        fixed at prefill so steering strength is comparable across positions."""
        if layer not in self._layer_scale:
            norms = h.float().norm(dim=-1)
            body = norms[4:] if len(norms) > 8 else norms
            self._layer_scale[layer] = float(body.median())
        return self._layer_scale[layer]

    def _dirs_for_words(self, layer: int, words: list[str], device) -> torch.Tensor:
        ids = [self.m.word_token_ids(w)[0] for w in words]
        return self.m.lens_directions(layer, ids, device)

    def _apply(self, iv: InterventionSpec, layer: int, h: torch.Tensor, pos) -> torch.Tensor:
        """Edit ``h`` ([n, d] float32, the selected positions) and return it."""
        dev = h.device
        if iv.kind in ("ablate_topk", "scale_jspace"):
            out = torch.empty_like(h)
            for s in range(0, len(h), _CHUNK):
                hs = h[s:s + _CHUNK]
                logits = self.m.lens_logits(layer, hs)
                if iv.word_like_only:
                    logits = logits.masked_fill(~self.m.word_like_on(logits.device),
                                                float("-inf"))
                top = logits.topk(iv.k, dim=-1).indices  # [n, k]
                V = self.m.lens_directions(layer, top, dev)  # [n, k, d]
                Q, _ = torch.linalg.qr(V.transpose(1, 2))  # [n, d, k]
                proj = torch.einsum("ndk,nk->nd", Q, torch.einsum("ndk,nd->nk", Q, hs))
                keep = 0.0 if iv.kind == "ablate_topk" else iv.strength
                out[s:s + _CHUNK] = hs - (1.0 - keep) * proj
            return out
        if iv.kind in ("ablate_tokens", "random_ablate"):
            key = (id(iv), layer, dev)
            if key not in self._fixed_dirs:
                if iv.kind == "ablate_tokens":
                    D = self._dirs_for_words(layer, iv.tokens, dev)
                else:
                    g = torch.Generator(device="cpu").manual_seed(iv.seed * 100_003 + layer)
                    D = torch.randn(iv.k, self.m.d_model, generator=g).to(dev)
                Q, _ = torch.linalg.qr(D.T)  # [d, m]
                self._fixed_dirs[key] = Q
            Q = self._fixed_dirs[key]
            return h - (h @ Q) @ Q.T
        if iv.kind == "steer":
            scale = self._scale(layer, h) * iv.strength
            D = self._dirs_for_words(layer, iv.tokens, dev)  # [m, d]
            return h + scale * D.sum(0, keepdim=True)
        if iv.kind == "swap":
            if not (iv.source and iv.target):
                raise ValueError("swap needs source and target")
            V = self._dirs_for_words(layer, [iv.source, iv.target], dev).T  # [d, 2]
            c = h @ torch.linalg.pinv(V).T  # [n, 2]
            return h + (c.flip(-1) - c) @ V.T
        raise ValueError(f"unknown intervention {iv.kind}")


# ====================================================================== helpers


def load_lens(spec: ModelSpec):
    import jlens

    if spec.lens_path:
        return jlens.JacobianLens.load(spec.lens_path)
    if not spec.lens_file:
        raise ValueError(f"no lens known for {spec.name}; fit one with scripts/fit_lens.py "
                         "and pass lens_path")
    return jlens.JacobianLens.from_pretrained(spec.lens_repo, filename=spec.lens_file,
                                              revision=spec.lens_revision)


def _final_norm_gain(norm: nn.Module, d_model: int) -> torch.Tensor:
    """Elementwise gain ``g`` of the final norm, so ``unembed(x) ~ W_U (g * x / rms(x))``.

    LayerNorm exposes it as ``weight``. For RMSNorm variants (incl. Gemma's ``1 + w``)
    the norm of an all-ones vector (rms = 1) is exactly ``g``.
    """
    with torch.no_grad():
        if isinstance(norm, nn.LayerNorm):
            return norm.weight.detach().float()
        p = next(norm.parameters(), None)
        ones = torch.ones(1, d_model, device=p.device if p is not None else "cpu",
                          dtype=p.dtype if p is not None else torch.float32)
        return norm(ones)[0].detach().float()


def _eos_ids(hf_model, tokenizer) -> set[int]:
    ids: set[int] = set()
    for src in (getattr(hf_model, "generation_config", None), tokenizer):
        val = getattr(src, "eos_token_id", None)
        if isinstance(val, int):
            ids.add(val)
        elif isinstance(val, (list, tuple)):
            ids.update(int(v) for v in val)
    for name in ("<|im_end|>", "<|eot_id|>", "<end_of_turn>", "<|return|>", "<|end|>"):
        tid = tokenizer.convert_tokens_to_ids(name) if name in tokenizer.get_vocab() else None
        if isinstance(tid, int):
            ids.add(tid)
    return ids


def _fold_system(messages: list[dict]) -> list[dict]:
    system, rest = messages[0]["content"], [dict(m) for m in messages[1:]]
    if rest and rest[0]["role"] == "user":
        rest[0]["content"] = f"{system}\n\n{rest[0]['content']}"
    else:
        rest.insert(0, {"role": "user", "content": system})
    return rest


def _sample(logits: torch.Tensor, temperature: float, top_p: float, gen: torch.Generator) -> int:
    if temperature <= 0:
        return int(logits.argmax())
    probs = torch.softmax(logits.float().cpu() / temperature, dim=-1)
    if top_p < 1.0:
        sp, si = probs.sort(descending=True)
        keep = sp.cumsum(0) - sp < top_p
        probs = torch.zeros_like(probs).scatter(0, si[keep], sp[keep])
        probs /= probs.sum()
    return int(torch.multinomial(probs, 1, generator=gen))


def _hit_stop(text: str, stops: list[str]) -> bool:
    return any(s and s in text for s in stops)


def _locate_spans(text: str, offsets: list[tuple[int, int]], req: GenerateRequest,
                  prompt_len: int) -> dict[str, list[tuple[int, int]]]:
    """Token spans for every message's content (``msg:{i}``, ``role:{role}``) and every
    requested :class:`~jpeers.schemas.SpanQuery`."""

    def char_to_tokens(a: int, b: int) -> tuple[int, int] | None:
        toks = [i for i, (s, e) in enumerate(offsets) if e > a and s < b and e > s]
        return (toks[0], toks[-1] + 1) if toks else None

    spans: dict[str, list[tuple[int, int]]] = {}
    cursor = 0
    for i, msg in enumerate(req.messages or []):
        content = msg.get("content") or ""
        at = text.find(content, cursor) if content else -1
        if at < 0:
            continue
        cursor = at + len(content)
        tok = char_to_tokens(at, cursor)
        if tok:
            spans.setdefault(f"msg:{i}", []).append(tok)
            spans.setdefault(f"role:{msg.get('role')}", []).append(tok)
    for q in req.spans:
        hits, start = [], 0
        while q.text and (at := text.find(q.text, start)) >= 0:
            hits.append(at)
            start = at + 1
        if q.occurrence == "first":
            hits = hits[:1]
        elif q.occurrence == "last":
            hits = hits[-1:]
        for at in hits:
            tok = char_to_tokens(at, at + len(q.text))
            if tok:
                spans.setdefault(q.label, []).append(tok)
    spans["prompt"] = [(0, prompt_len)]
    return spans
