"""Native replay tests: local byte tokenizer and randomly initialized models only."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch

from jpeers.exp5.backend import HFBackend


class ByteTokenizer:
    controls = {"<|im_start|>": 256, "<|im_end|>": 257, "<|endoftext|>": 258}
    eos_token_id = 257
    model_max_length = 1024
    chat_template = "local Qwen non-thinking byte fixture"

    def encode(self, text, add_special_tokens=False):
        result = []
        while text:
            special = next((s for s in self.controls if text.startswith(s)), None)
            if special:
                result.append(self.controls[special])
                text = text[len(special):]
            else:
                char, text = text[0], text[1:]
                result.extend(char.encode("utf-8"))
        return result

    def decode(self, ids, skip_special_tokens=False, **kwargs):
        inverse = {v: k for k, v in self.controls.items()}
        pieces, raw = [], bytearray()
        for token in ids:
            if token in inverse:
                pieces.append(raw.decode("utf-8", errors="replace"))
                raw.clear()
                if not skip_special_tokens:
                    pieces.append(inverse[token])
            else:
                raw.append(token)
        pieces.append(raw.decode("utf-8", errors="replace"))
        return "".join(pieces)

    def get_vocab(self):
        return {**{str(i): i for i in range(256)}, **self.controls}


class CanonicalByteTokenizer(ByteTokenizer):
    """An independent local rendering fixture; never calls backend formatting."""
    def __init__(self):
        self.template_calls = []

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt, enable_thinking):
        self.template_calls.append(dict(messages=messages, tokenize=tokenize,
                                        add_generation_prompt=add_generation_prompt,
                                        enable_thinking=enable_thinking))
        text = ""
        for message in messages:
            text += "<|im_start|>" + message["role"] + "\n"
            if message["role"] == "assistant":
                text += "<think>\n\n</think>\n\n"
            text += message["content"] + "<|im_end|>\n"
        if add_generation_prompt:
            text += "<|im_start|>assistant\n"
            if not enable_thinking:
                text += "<think>\n\n</think>\n\n"
        return self.encode(text) if tokenize else text


class ToyBlock(torch.nn.Module):
    def __init__(self, width):
        super().__init__()
        self.proj = torch.nn.Linear(width, width, bias=False)
        self.attention = None

    def forward(self, hidden, *, old=None, attention_mask=None):
        keys = hidden if old is None else torch.cat((old, hidden), dim=1)
        scores = hidden @ keys.transpose(-1, -2) / hidden.shape[-1] ** .5
        self.attention = torch.softmax(scores[:, None] + attention_mask, dim=-1)
        output = hidden + self.proj(self.attention[:, 0] @ keys)
        return output, keys


class ToyModel(torch.nn.Module):
    def __init__(self, width=8, layers=3, context=1024):
        super().__init__()
        self.config = SimpleNamespace(hidden_size=width, max_position_embeddings=context,
                                      _attn_implementation="eager", model_type="exp5-toy")
        self.embed = torch.nn.Embedding(259, width)
        self.model = torch.nn.Module()
        self.model.layers = torch.nn.ModuleList([ToyBlock(width) for _ in range(layers)])
        self.lm_head = torch.nn.Linear(width, 259, bias=False)
        self.calls = []

    def forward(self, input_ids, attention_mask=None, past_key_values=None,
                use_cache=True, output_attentions=False, **kwargs):
        self.calls.append((input_ids.clone(), attention_mask.clone()))
        hidden, past, attentions = self.embed(input_ids), [], []
        for index, layer in enumerate(self.model.layers):
            output = layer(hidden, old=None if past_key_values is None else past_key_values[index],
                           attention_mask=attention_mask)
            hidden, keys = output
            past.append(keys)
            attentions.append(layer.attention)
        return SimpleNamespace(logits=self.lm_head(hidden), past_key_values=tuple(past),
                               attentions=tuple(attentions) if output_attentions else None)


def backend(model=None):
    if model is None:
        with torch.random.fork_rng():
            torch.manual_seed(91)
            model = ToyModel()
    return HFBackend(model, ByteTokenizer(), model_id="local/random-test", revision="fixture")


def edit(kind="delete", layer=0, position=2, **kwargs):
    basis = np.eye(8, dtype=np.float32)[:, :2]
    return dict(layer=layer, position=position, kind=kind, basis=basis,
                strength=.7, seed=19, tag=kind, **kwargs)


class BackendTests(unittest.TestCase):
    def test_append_preserves_native_prefix_and_exact_content_interval(self):
        b = backend()
        original = [1, 255, 2]
        tokens, span = b.append_message(original, "user", "scene")
        self.assertEqual(tokens[:len(original)], original)
        self.assertEqual(original, [1, 255, 2])
        self.assertEqual(b.decode(tokens[slice(*span)]), "scene")
        opened, receipt = b.append_message(tokens, "assistant", "Recorded.", close=False)
        self.assertEqual(opened[:len(tokens)], tokens)
        self.assertIn("assistant\n<think>\n\n</think>\n\nRecorded.", b.decode(opened))
        self.assertEqual(b.decode(opened[slice(*receipt)]), "Recorded.")
        finished = b.finish_assistant(opened)
        self.assertEqual(finished[:len(opened)], opened)
        self.assertTrue(b.decode(finished).endswith("Recorded.<|im_end|>\n"))
        self.assertEqual(b.finish_assistant(finished), finished)
        self.assertEqual(b.finish_assistant(finished[:-1]), finished)

    def test_captures_are_real_block_outputs_float32_and_hooks_are_removed(self):
        b = backend()
        seen = {}
        handle = b.layers[1].register_forward_hook(lambda m, a, out: seen.update(value=out[0].detach().clone()))
        result = b.observe([10, 11, 12, 13], {1: [0, 3]})
        handle.remove()
        np.testing.assert_allclose(result["captures"][1][3], seen["value"][0, 3].numpy())
        self.assertEqual(result["captures"][1][3].dtype, np.float32)
        self.assertEqual(result["logits"].shape, (259,))
        self.assertTrue(all(not layer._forward_hooks for layer in b.layers))
        self.assertFalse(any(p.requires_grad for p in b.model.parameters()))

    def test_masks_block_real_attention_and_persist_in_every_decode_call(self):
        b = backend()
        mask = dict(key_spans=[(0, 2)], start_query=2, tag="source")
        b.observe([10, 11, 12, 13], {}, masks=[mask])
        self.assertTrue(torch.equal(b.layers[0].attention[0, :, 2:, :2], torch.zeros(1, 2, 2)))
        b.model.calls.clear()
        result = b.generate([10, 11, 12, 13], sampler=dict(temperature=.7, top_k=20, top_p=.8, min_p=.05),
                            seed=5, max_new_tokens=5, masks=[mask])
        self.assertGreater(len(b.model.calls), 1)
        for inputs, attention in b.model.calls:
            if inputs.shape[1] == 1:
                self.assertTrue(torch.all(attention[..., :2] < -1000))
        self.assertEqual(result["receipt"]["masks"][0]["tag"], "source")

    def test_seeded_sampling_never_changes_global_rng(self):
        b = backend()
        sampler = dict(temperature=.9, top_k=13, top_p=.8, min_p=.1)
        torch_state, numpy_state = torch.random.get_rng_state().clone(), np.random.get_state()
        first = b.generate([10, 11, 12], sampler=sampler, seed=101, max_new_tokens=6)
        second = b.generate([10, 11, 12], sampler=sampler, seed=101, max_new_tokens=6)
        self.assertEqual(first["token_ids"], second["token_ids"])
        self.assertTrue(torch.equal(torch_state, torch.random.get_rng_state()))
        np.testing.assert_array_equal(numpy_state[1], np.random.get_state()[1])
        self.assertEqual(first["hit_cap"], len(first["token_ids"]) == 6 and first["token_ids"][-1] != 257)

    def test_selective_patch_delete_and_matched_random_geometry(self):
        b = backend()
        tokens = [10, 11, 12, 13]
        clean = b.observe(tokens, {0: [1, 2, 3]})["captures"][0]
        deleted = b.observe(tokens, {0: [1, 2, 3]}, edits=[edit()])
        np.testing.assert_allclose(deleted["captures"][0][2][:2], .3 * clean[2][:2], atol=1e-6)
        for pos in (1, 3):
            np.testing.assert_array_equal(clean[pos], deleted["captures"][0][pos])
        np.testing.assert_array_equal(clean[2][2:], deleted["captures"][0][2][2:])
        donor = np.arange(8, dtype=np.float32) + 2
        patch = edit("patch", donor=donor)
        patched = b.observe(tokens, {0: [2]}, edits=[patch])
        np.testing.assert_allclose(patched["captures"][0][2][:2], .3 * clean[2][:2] + .7 * donor[:2], atol=1e-6)
        for target, random in ((edit(), edit("random_delete")), (patch, edit("random_patch", donor=donor))):
            a = b.observe(tokens, {0: [2]}, edits=[target])
            c = b.observe(tokens, {0: [2]}, edits=[random])
            da, dc = a["captures"][0][2] - clean[2], c["captures"][0][2] - clean[2]
            self.assertAlmostEqual(float(np.linalg.norm(da)), float(np.linalg.norm(dc)), places=5)
            self.assertAlmostEqual(float(da @ clean[2]), float(dc @ clean[2]), places=5)
            self.assertFalse(np.allclose(da, dc))
            self.assertAlmostEqual(c["receipt"]["edits"][0]["realized_norm"], float(np.linalg.norm(dc)), places=6)

    def test_later_layer_rescue_and_future_position_decode_edit(self):
        b = backend()
        tokens = [10, 11, 12, 13]
        clean = b.observe(tokens, {2: [2]})["captures"][2][2]
        damaged = b.observe(tokens, {2: [2]}, edits=[edit()])["captures"][2][2]
        rescue = edit("patch", layer=2, donor=clean)
        rescued = b.observe(tokens, {2: [2]}, edits=[edit(), rescue])
        before, after = np.linalg.norm(clean[:2] - damaged[:2]), np.linalg.norm(clean[:2] - rescued["captures"][2][2][:2])
        self.assertLess(after, before)
        generation = b.generate(tokens, sampler=dict(temperature=0), seed=0, max_new_tokens=4,
                                edits=[edit(position=4)])
        self.assertEqual(generation["receipt"]["edits"][0]["position"], 4)
        self.assertEqual(sum(e["position"] == 4 for e in generation["receipt"]["edits"]), 1)

    def test_context_and_bad_geometry_rejected_before_forward(self):
        b = backend(ToyModel(context=8))
        for operation in (lambda: b.observe(list(range(9)), {}),
                          lambda: b.generate([1, 2, 3], sampler={}, seed=0, max_new_tokens=6),
                          lambda: b.observe([1, 2, 3], {}, edits=[dict(edit(), basis=np.ones((8, 2)))]),
                          lambda: b.observe([1, 2, 3], {}, masks=[dict(key_spans=[(0, 3)], start_query=0, tag="all")])):
            with self.assertRaises(ValueError):
                operation()
        self.assertFalse(b.model.calls)

    def test_hook_cleanup_after_failure(self):
        b = backend()
        def fail(*args, **kwargs):
            raise RuntimeError("fixture failure")
        original = b.model.forward
        b.model.forward = fail
        with self.assertRaisesRegex(RuntimeError, "fixture failure"):
            b.observe([1, 2, 3], {0: [2]}, edits=[edit()])
        b.model.forward = original
        self.assertTrue(all(not layer._forward_hooks for layer in b.layers))

    def test_runtime_validation_compares_cache_masks_edits_and_attention(self):
        b = backend()
        result = b.validate_runtime()
        self.assertTrue(result["passed"], result)
        self.assertTrue(result["checks"]["blocked_attention"]["passed"])
        self.assertTrue(result["checks"]["masked_edited_cache_replay"]["passed"])
        self.assertTrue(result["checks"]["non_thinking_format"]["passed"])
        self.assertTrue(result["fingerprint"])
        self.assertTrue(b.info()["synthetic"])

    def test_runtime_validation_cannot_pass_a_model_ignoring_masks(self):
        b = backend()
        forward = b.model.forward
        def ignore_source_masks(input_ids, attention_mask, **kwargs):
            count, total = attention_mask.shape[-2:]
            offset = total - count
            causal = b._attention_mask(offset, count, [])
            return forward(input_ids, attention_mask=causal, **kwargs)
        b.model.forward = ignore_source_masks
        result = b.validate_runtime()
        self.assertFalse(result["passed"])
        self.assertFalse(result["checks"]["blocked_attention"]["passed"])

    def test_fingerprint_identifies_different_injected_weights(self):
        first = backend()
        with torch.random.fork_rng():
            torch.manual_seed(92)
            second = backend(ToyModel())
        self.assertNotEqual(first.info()["fingerprint"], second.info()["fingerprint"])

    def test_bfloat16_edit_rounds_the_complete_float32_replacement_once(self):
        model = backend().model.to(torch.bfloat16)
        b = HFBackend(model, ByteTokenizer(), model_id="local/random-test", revision="fixture", dtype="bfloat16")
        clean = b.observe([10, 11, 12, 13], {0: [2]})["captures"][0][2]
        basis = np.ones((8, 1), dtype=np.float32) / np.sqrt(8)
        operation = dict(edit(), basis=basis, strength=.37)
        result = b.observe([10, 11, 12, 13], {0: [2]}, edits=[operation])
        displacement = -.37 * (basis @ (basis.T @ clean))
        expected = torch.tensor(clean + displacement).to(torch.bfloat16).float().numpy()
        np.testing.assert_array_equal(result["captures"][0][2], expected)
        actual = expected - clean
        self.assertAlmostEqual(result["receipt"]["edits"][0]["realized_norm"], float(np.linalg.norm(actual)), places=6)

    def test_sampling_filters_have_predictable_support_and_native_eos_is_retained(self):
        b = backend()
        logits = torch.tensor([0., 1., 2., 3.])
        for values, expected in ((dict(top_k=1, top_p=1., min_p=0.), 3),
                                 (dict(top_k=0, top_p=.1, min_p=0.), 3),
                                 (dict(top_k=0, top_p=1., min_p=.99), 3)):
            sampler = b._sampler(dict(temperature=1., **values))
            for seed in range(5):
                generator = torch.Generator().manual_seed(seed)
                self.assertEqual(b._sample(logits, sampler, generator), expected)
        original = b.model.forward
        def emit_end(*args, **kwargs):
            result = original(*args, **kwargs)
            result.logits[..., :] = -20
            result.logits[..., 257] = 20
            return result
        b.model.forward = emit_end
        result = b.generate([1, 2, 3], sampler=dict(temperature=0), seed=19, max_new_tokens=4)
        self.assertEqual(result["token_ids"], [257])
        self.assertEqual(result["text"], "")
        self.assertFalse(result["hit_cap"])

    def test_canonical_non_thinking_tokens_match_independent_template(self):
        tokenizer = CanonicalByteTokenizer()
        b = HFBackend(backend().model, tokenizer, model_id="local/random-test", revision="fixture")
        result = b.validate_runtime()
        check = result["checks"]["canonical_non_thinking_format"]
        self.assertTrue(result["passed"], result)
        self.assertTrue(check["supported"])
        self.assertTrue(check["passed"])
        self.assertFalse(check["required"])
        self.assertTrue(all(case["passed"] for case in check["cases"].values()))
        self.assertEqual(set(check["cases"]), {"generation_prompt", "closed_assistant_receipt"})
        self.assertTrue(all(call["tokenize"] and call["enable_thinking"] is False
                            for call in tokenizer.template_calls))
        self.assertTrue(any(call["add_generation_prompt"] for call in tokenizer.template_calls))
        self.assertFalse(result["target_gate"])

    def test_canonical_mismatch_fails_for_generation_and_closed_receipt(self):
        for mismatch_role in ("user", "assistant"):
            with self.subTest(mismatch_role=mismatch_role):
                tokenizer = CanonicalByteTokenizer()
                canonical = tokenizer.apply_chat_template
                def mismatch(messages, **kwargs):
                    tokens = canonical(messages, **kwargs)
                    if messages[-1]["role"] == mismatch_role:
                        tokens[-1] = 13
                    return tokens
                tokenizer.apply_chat_template = mismatch
                b = HFBackend(backend().model, tokenizer, model_id="local/random-test", revision="fixture")
                result = b.validate_runtime()
                check = result["checks"]["canonical_non_thinking_format"]
                self.assertFalse(check["passed"])
                case = "generation_prompt" if mismatch_role == "user" else "closed_assistant_receipt"
                self.assertFalse(check["cases"][case]["passed"])
                self.assertFalse(result["passed"])
                self.assertFalse(result["target_gate"])

    def test_missing_canonical_template_is_explicit_and_blocks_target_gate(self):
        b = backend()
        result = b.validate_runtime()
        check = result["checks"]["canonical_non_thinking_format"]
        self.assertFalse(check["supported"])
        self.assertIsNone(check["passed"])
        self.assertFalse(check["required"])
        self.assertTrue(result["passed"])
        self.assertFalse(result["target_gate"])
        # Exercise the target requirement without claiming this fixture is Qwen.
        b.synthetic = False
        result = b.validate_runtime()
        self.assertTrue(result["checks"]["canonical_non_thinking_format"]["required"])
        self.assertFalse(result["checks"]["canonical_non_thinking_format"]["passed"])
        self.assertFalse(result["passed"])
        self.assertFalse(result["target_gate"])


class TinyQwenTests(unittest.TestCase):
    def test_local_random_qwen3_runtime_validation(self):
        try:
            from transformers import Qwen3Config, Qwen3ForCausalLM
        except ImportError:
            self.skipTest("Qwen3 requires the isolated newer Transformers interpreter")
        with torch.random.fork_rng():
            torch.manual_seed(141)
            config = Qwen3Config(vocab_size=259, hidden_size=16, intermediate_size=24,
                                 num_hidden_layers=3, num_attention_heads=2, num_key_value_heads=1,
                                 head_dim=8, max_position_embeddings=1024,
                                 eos_token_id=257, attention_dropout=0.0)
            config._attn_implementation = "eager"
            model = Qwen3ForCausalLM(config)
        b = backend(model)
        result = b.validate_runtime()
        self.assertTrue(result["passed"], result)
        result = b.observe([10, 11, 12, 13], {0: [2], 2: [2]},
                           masks=[dict(key_spans=[(0, 1)], start_query=2, tag="R2")],
                           edits=[dict(edit(), basis=np.eye(16, dtype=np.float32)[:, :2])])
        self.assertEqual(result["captures"][2][2].shape, (16,))
        self.assertEqual(result["receipt"]["edits"][0]["position"], 2)


if __name__ == "__main__":
    unittest.main()
