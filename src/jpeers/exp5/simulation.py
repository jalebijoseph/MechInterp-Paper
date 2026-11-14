"""Explicitly synthetic Experiment 5 workflow backend using NumPy and stdlib.

The eight coordinates are four observed success rates followed by four exposure
counts, in CELL_ORDER. Layer outputs scale these coordinates by (layer+1)/64.
Edits carry coordinate displacements forward through the synthetic trajectory.
Masks remove overlapping JSON evidence events at active query positions; this is
a software fixture, not a transformer, cache, attention, or Qwen verification.
Generation plans JSON from the prompt-end state: edits to future generated
positions and masks that first activate during generation are rejected.
"""
from __future__ import annotations

import hashlib
import json
import math

import numpy as np

from .backend import HFBackend
from .protocol import CELL_ORDER


class SimulationBackend(HFBackend):
    """Reuse token append/validation contracts without constructing an HF model."""

    _controls = {"<|im_start|>": 256, "<|im_end|>": 257, "<|endoftext|>": 258}

    def __init__(self, config=None):
        # A study config may be supplied by the runner; hidden task labels and
        # configured model identity never determine this fixture's behavior.
        self.model_id, self.revision = "simulation/exp5-byte", "event-replay-v1"
        self.hidden_size, self.layers = 8, tuple(range(64))
        self.max_context, self.dtype, self.device = 4_194_304, "float32", "cpu"
        self._end, self._newline, self.stop_ids = [257], [10], {257, 258}

    @property
    def synthetic(self):
        return True

    def info(self) -> dict:
        result = dict(model_id=self.model_id, revision=self.revision,
                      n_layers=len(self.layers), hidden_size=self.hidden_size,
                      max_context=self.max_context, dtype=self.dtype, device=self.device,
                      synthetic=True, target_gate=False, enable_thinking=False,
                      attention_implementation="synthetic JSON event filtering",
                      actual_source_attention_verification=False,
                      layer_numbering="zero-based synthetic block-output labels",
                      coordinate_order=[f"rate:{p}:{d}" for p, d in CELL_ORDER] +
                                       [f"exposures:{p}:{d}" for p, d in CELL_ORDER],
                      layer_scale="(layer + 1) / 64", tokenizer="UTF-8 bytes plus 3 controls",
                      purpose="full workflow software testing; no target-model evidence")
        result["fingerprint"] = hashlib.sha256(json.dumps(result, sort_keys=True).encode()).hexdigest()
        return result

    def encode_text(self, text: str) -> list[int]:
        if not isinstance(text, str):
            raise TypeError("text must be a string")
        result, offset = [], 0
        while offset < len(text):
            special = next((s for s in self._controls if text.startswith(s, offset)), None)
            if special:
                result.append(self._controls[special])
                offset += len(special)
            else:
                result.extend(text[offset].encode("utf-8"))
                offset += 1
        return result

    def decode(self, ids: list[int]) -> str:
        inverse = {value: key for key, value in self._controls.items()}
        pieces, raw = [], bytearray()
        for token in ids:
            if isinstance(token, bool) or not isinstance(token, (int, np.integer)) or not 0 <= token < 259:
                raise ValueError("invalid synthetic byte/control token")
            if token in inverse:
                pieces.append(raw.decode("utf-8", errors="replace"))
                raw.clear()
                pieces.append(inverse[token])
            else:
                raw.append(token)
        pieces.append(raw.decode("utf-8", errors="replace"))
        return "".join(pieces)

    def _check_tokens(self, tokens, extra=0):
        super()._check_tokens(tokens, extra)
        if any(token >= 259 for token in tokens):
            raise ValueError("token ID is outside the synthetic byte/control vocabulary")

    def _events(self, tokens):
        """Read complete top-level JSON user/tool payloads with native intervals."""
        starts = [i for i, token in enumerate(tokens) if token == 256]
        events = []
        for index, start in enumerate(starts):
            stop = starts[index + 1] if index + 1 < len(starts) else len(tokens)
            end = next((i for i in range(start + 1, stop) if tokens[i] in self.stop_ids), stop)
            newline = next((i for i in range(start + 1, end) if tokens[i] == 10), None)
            if newline is None or self.decode(tokens[start + 1:newline]) not in ("user", "tool"):
                continue
            content_start = newline + 1
            try:
                value = json.loads(self.decode(tokens[content_start:end]))
            except (ValueError, RecursionError):
                continue
            if isinstance(value, dict) and isinstance(value.get("event"), str):
                events.append(dict(value=value, start=content_start, end=end))
        return events

    @staticmethod
    def _visible(events, position, masks):
        return [event for event in events if event["end"] <= position + 1 and not any(
            position >= mask["start_query"] and any(
                start < event["end"] and end > event["start"] for start, end in mask["key_spans"])
            for mask in masks)]

    def _state(self, events, position, masks):
        successes, exposures = np.zeros(4), np.zeros(4)
        domain = None
        for entry in self._visible(events, position, masks):
            event = entry["value"]
            if event["event"] in ("task", "delegation_probe", "peer_task"):
                domain = event.get("domain")
            if event["event"] not in ("feedback", "peer_feedback") or type(event.get("correct")) is not bool:
                continue
            cell = event.get("peer"), event.get("domain", domain)
            if cell not in CELL_ORDER:
                continue
            index = CELL_ORDER.index(cell)
            exposures[index] += 1
            successes[index] += event["correct"]
        rates = np.divide(successes, exposures, out=np.full(4, .5), where=exposures > 0)
        return np.concatenate((rates, exposures)).astype(np.float32)

    def _edit(self, hidden, edit):
        h, basis = np.asarray(hidden, dtype=np.float32), edit["basis"]
        reference = -h if edit["kind"].endswith("delete") else edit["donor"] - h
        targeted = edit["strength"] * (basis @ (basis.T @ reference))
        realized_target = (h + targeted).astype(np.float32) - h
        delta = targeted
        if edit["kind"].startswith("random_"):
            rng = np.random.default_rng(edit["seed"])
            hv, dv = h.astype(np.float64), realized_target.astype(np.float64)
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
            delta = matched.astype(np.float32)
        changed = (h + delta).astype(np.float32)
        if not np.isfinite(changed).all():
            raise ValueError("edit produced a nonfinite synthetic residual")
        actual = changed - h
        actual_norm, actual_dot = float(np.linalg.norm(actual)), float(actual @ h)
        reference_norm, reference_dot = float(np.linalg.norm(realized_target)), float(realized_target @ h)
        receipt = {key: edit[key] for key in ("layer", "position", "kind", "strength", "seed", "tag")}
        receipt.update(rank=int(basis.shape[1]), dtype=self.dtype,
                       requested_norm=float(np.linalg.norm(targeted)),
                       requested_inner_product=float(targeted @ h),
                       realized_norm=actual_norm, realized_inner_product=actual_dot,
                       reference_realized_norm=reference_norm,
                       reference_realized_inner_product=reference_dot,
                       norm_matching_error=actual_norm - reference_norm,
                       inner_product_matching_error=actual_dot - reference_dot)
        return changed, receipt

    def _receipt(self, count, masks):
        return dict(fingerprint=self.info()["fingerprint"], token_count=count,
                    mode="synthetic exact-token JSON evidence replay", masks=masks,
                    edits=[], forward_calls=[], synthetic=True, target_gate=False,
                    actual_source_attention_verification=False)

    def _replay(self, tokens, capture, masks, edits, *, final_position=None):
        events = self._events(tokens)
        receipt = self._receipt(len(tokens), masks)
        displacements = []

        def state_at(position, layer):
            state = self._state(events, position, masks)
            for prior, delta in displacements:
                if prior["position"] < position or (prior["position"] == position and prior["layer"] <= layer):
                    state += delta
            return state

        # ponytail: synthetic coordinate propagation, never a substitute for the
        # model's block/KV dependencies; real causal evidence uses HFBackend.
        for edit in sorted(edits, key=lambda entry: (entry["position"], entry["layer"])):
            if final_position is not None and edit["position"] > final_position:
                continue
            scale = (edit["layer"] + 1) / len(self.layers)
            hidden = state_at(edit["position"], edit["layer"]) * scale
            changed, realized = self._edit(hidden, edit)
            displacements.append((edit, (changed - hidden) / scale))
            receipt["edits"].append(realized)
        captures = {layer: {position: state_at(position, layer) * ((layer + 1) / len(self.layers))
                            for position in positions} for layer, positions in capture.items()}
        position = len(tokens) - 1 if final_position is None else final_position
        state = state_at(position, len(self.layers) - 1)
        receipt["forward_calls"].append(dict(offset=0, query_tokens=len(tokens), key_tokens=len(tokens), synthetic=True))
        return captures, state, events, receipt

    def observe(self, tokens: list[int], capture: dict[int, list[int]], *, masks=(), edits=()) -> dict:
        capture, masks, edits = self._prepare(tokens, capture, masks, edits)
        captures, state, _, receipt = self._replay(tokens, capture, masks, edits)
        logits = np.zeros(259, dtype=np.float32)
        logits[:8] = state  # Synthetic diagnostic scores, not a model vocabulary head.
        return dict(captures=captures, logits=logits, receipt=receipt)

    @staticmethod
    def _choice(rates, sampler, rng):
        rates = np.clip(np.asarray(rates, dtype=np.float64), 0, 1)
        if sampler["temperature"] == 0:
            return ("A", "B")[int(np.argmax(rates))]
        scores = 8 * (rates - rates.max()) / sampler["temperature"]
        probabilities = np.exp(scores)
        probabilities /= probabilities.sum()
        keep = np.ones(2, dtype=bool)
        if sampler["top_k"] == 1:
            keep[np.argmin(probabilities)] = False
        keep &= probabilities >= probabilities.max() * sampler["min_p"]
        ordered = np.argsort(-probabilities, kind="stable")
        cumulative = np.cumsum(probabilities[ordered])
        keep[ordered[1:]] &= cumulative[:-1] <= sampler["top_p"]
        probabilities *= keep
        probabilities /= probabilities.sum()
        return ("A", "B")[int(rng.choice(2, p=probabilities))]

    def _response(self, events, state, position, masks, sampler, rng):
        visible = self._visible(events, position, masks)
        requests = [entry["value"] for entry in visible if entry["value"]["event"] in
                    ("task", "delegation_probe", "peer_task", "capability_probe", "identity_probe")]
        if not requests:
            return {"choice": self._choice([.5, .5], sampler, rng)}
        request = requests[-1]
        kind = request["event"]
        if kind == "peer_task":
            facts = request.get("private_facts", [])
            values = {fact.get("value") for fact in facts if isinstance(fact, dict) and
                      fact.get("subject") == request.get("query_subject") and isinstance(fact.get("value"), str)}
            answer = next(iter(values)) if len(values) == 1 else "unknown"
            return {"answer": answer if answer in request.get("options", []) else "unknown"}
        if kind == "capability_probe":
            rates = np.clip(state[:4], 0, 1)
            return {peer: {domain: float(rates[CELL_ORDER.index((peer, domain))])
                           for domain in ("possession", "location")} for peer in ("A", "B")}
        if kind == "identity_probe":
            rosters = [entry["value"].get("roster", {}) for entry in visible if entry["value"]["event"] == "roster"]
            roster = rosters[-1] if rosters else {}
            choices = [peer for peer in ("A", "B") if roster.get(peer) == request.get("peer_name")]
            return {"choice": choices[0] if len(choices) == 1 else self._choice([.5, .5], sampler, rng)}
        domain = request.get("domain")
        rates = [state[CELL_ORDER.index((peer, domain))] if (peer, domain) in CELL_ORDER else .5
                 for peer in ("A", "B")]
        return {"choice": self._choice(rates, sampler, rng)}

    def generate(self, tokens: list[int], *, sampler: dict, seed: int, max_new_tokens: int,
                 masks=(), edits=()) -> dict:
        if not isinstance(max_new_tokens, int) or isinstance(max_new_tokens, bool) or max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be a positive integer")
        capture, masks, edits = self._prepare(tokens, {}, masks, edits, max_new_tokens)
        if any(edit["position"] >= len(tokens) for edit in edits):
            raise ValueError("synthetic generation does not support generated-position edits")
        if any(mask["start_query"] >= len(tokens) for mask in masks):
            raise ValueError("synthetic source masks must activate before, not during generation")
        sampler = self._sampler(sampler)
        _, state, events, receipt = self._replay(tokens, capture, masks, edits, final_position=len(tokens) - 1)
        response = self._response(events, state, len(tokens) - 1, masks, sampler, np.random.default_rng(int(seed)))
        intended = self.encode_text(json.dumps(response, ensure_ascii=False, separators=(",", ":"))) + self._end
        generated = intended[:max_new_tokens]
        hit_cap = generated[-1] not in self.stop_ids
        receipt.update(generated_tokens=len(generated), hit_cap=hit_cap, seed=int(seed), sampler=sampler,
                       generated_position_edits_supported=False)
        text_ids = generated[:-1] if generated[-1] in self.stop_ids else generated
        return dict(text=self.decode(text_ids), token_ids=generated, hit_cap=hit_cap, receipt=receipt)

    def validate_runtime(self) -> dict:
        """Exercise actual fixture operations, while declining model validation."""
        checks = {"blocked_attention": dict(passed=None, required=False, supported=False,
            reason="synthetic event filtering does not compute or verify attention weights")}
        try:
            prefix, span = self.append_message([], "user", "x")
            native, receipt_span = self.append_message(prefix, "assistant", "Recorded.", close=False)
            finished = self.finish_assistant(native)
            expected = "<|im_start|>user\nx<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\nRecorded.<|im_end|>\n"
            checks["non_thinking_format"] = dict(passed=(self.decode(finished) == expected and
                native[:len(prefix)] == prefix and self.decode(prefix[slice(*span)]) == "x" and
                self.decode(native[slice(*receipt_span)]) == "Recorded."))
            event = {"event": "peer_feedback", "peer": "A", "domain": "possession", "correct": True,
                     "response": '{"answer":"ball"}'}
            tokens, evidence_span = self.append_message([], "user", json.dumps(event))
            start = len(tokens)
            tokens, _ = self.append_message(tokens, "assistant", "Recorded.", close=False)
            position = len(tokens) - 1
            capture = {63: [position]}
            clean = self.observe(tokens, capture)
            repeated = self.observe(list(tokens), capture)
            checks["synthetic_replay"] = dict(passed=bool(np.array_equal(
                clean["captures"][63][position], repeated["captures"][63][position])))
            masked = self.observe(tokens, capture, masks=[dict(start_query=start, key_spans=[evidence_span])])
            checks["synthetic_source_filter"] = dict(passed=(
                float(clean["captures"][63][position][0]) == 1. and
                float(masked["captures"][63][position][0]) == .5))
            edited = self.observe(tokens, capture, edits=[dict(layer=63, position=position,
                basis=np.eye(8, dtype=np.float32)[:, :1], kind="delete", strength=.5)])
            checks["synthetic_edit"] = dict(passed=(float(edited["captures"][63][position][0]) == .5 and
                edited["receipt"]["edits"][0]["realized_norm"] == .5))
            generated = self.generate(tokens, sampler={}, seed=7, max_new_tokens=32)
            checks["native_generation"] = dict(passed=(not generated["hit_cap"] and
                generated["token_ids"][-1] in self.stop_ids and isinstance(json.loads(generated["text"]), dict)))
        except Exception as exc:
            checks["runtime_error"] = dict(passed=False, error=f"{type(exc).__name__}: {exc}")
        passed = all(row["passed"] is True or (row["passed"] is None and row.get("required") is False)
                     for row in checks.values())
        return dict(passed=passed, checks=checks, fingerprint=self.info()["fingerprint"],
                    synthetic=True, target_gate=False, actual_source_attention_verification=False)
