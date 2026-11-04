"""CPU tests for plain HF serving: native prompts, context and recording contracts."""

import importlib.util
from contextlib import nullcontext
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from fastapi.testclient import TestClient

from jpeers.schemas import GenerateRequest, GenerateResponse, RecordSpec


REVISION = "b96024342a77a69aa0dda815c3454a671f477463"


def subject():
    name = "plain_hf_server_under_test"
    if name not in sys.modules:
        path = Path(__file__).resolve().parents[1] / "scripts" / "serve_plain_hf.py"
        if not path.exists():
            raise AssertionError("The plain HF server must exist before it can serve requests")
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


class Tokenizer:
    chat_template = "native template with BOS"
    eos_token_id = 2
    pad_token_id = 2

    def __init__(self):
        self.calls = []

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
        if tokenize or not add_generation_prompt:
            raise AssertionError("Use the native generation template as text")
        return "<BOS>User: hello\nAssistant:"

    def __call__(self, text, *, add_special_tokens, truncation):
        self.calls.append((text, add_special_tokens, truncation))
        ids = [1, 10, 11] if text.startswith("<BOS>") else [10, 11]
        if add_special_tokens:
            ids = [1, *ids]
        return {"input_ids": ids}


class FakeBackend:
    def __init__(self):
        self.requests = []

    def info(self):
        return {"model": "olmo2-32b", "model_id": "allenai/OLMo-2-0325-32B-Instruct",
                "model_path": "/cache/snapshots/" + REVISION,
                "revision": REVISION, "family": "olmo", "params_b": 32., "context_window": 4096}

    def generate(self, req):
        self.requests.append(req)
        return GenerateResponse(model="olmo2-32b", text='{"choice":"tie"}', prompt_len=5,
                                n_generated=7, tokens=["p"] * 5 + ["g"] * 7,
                                spans=[], layers=[], watch=[], watch_ids=[], vocab={},
                                trace_encoding="json", trace=None, interventions_applied=0,
                                timings={"total_s": .1})


class PromptTests(unittest.TestCase):
    def test_native_chat_bos_is_not_added_twice_and_prefill_is_preserved(self):
        tokenizer = Tokenizer()
        req = GenerateRequest(messages=[{"role": "user", "content": "hello"}],
                              assistant_prefill="{", record=RecordSpec(enabled=False))
        ids = subject().encode_request(tokenizer, req)
        self.assertEqual(ids, [1, 10, 11])
        self.assertEqual(tokenizer.calls, [("<BOS>User: hello\nAssistant:{", False, False)])

    def test_raw_prompt_adds_special_tokens_once_without_using_chat_template(self):
        tokenizer = Tokenizer()
        req = GenerateRequest(prompt="hello", record=RecordSpec(enabled=False))
        self.assertEqual(subject().encode_request(tokenizer, req), [1, 10, 11])

    def test_missing_native_template_is_rejected_without_fallback_format(self):
        tokenizer = Tokenizer()
        tokenizer.chat_template = None
        with self.assertRaisesRegex(ValueError, "native.*template"):
            subject().encode_request(tokenizer, GenerateRequest(messages=[{"role": "user", "content": "hi"}],
                                                                record=RecordSpec(enabled=False)))

    def test_context_reserves_the_entire_output_budget_including_384_grader_tokens(self):
        m = subject()
        m.validate_context(3712, 384, 4096)
        with self.assertRaisesRegex(ValueError, "4096"):
            m.validate_context(3713, 384, 4096)
        with self.assertRaisesRegex(ValueError, "positive"):
            m.validate_context(0, 384, 4096)

    def test_unpinned_revision_and_context_above_4096_are_rejected(self):
        m = subject()
        with self.assertRaisesRegex(ValueError, "commit"):
            m.ModelConfig(model_id="allenai/example", revision="main", family="olmo")
        with self.assertRaisesRegex(ValueError, "4096"):
            m.ModelConfig(model_id="allenai/example", revision=REVISION, family="olmo", context_window=8192)


class HttpTests(unittest.TestCase):
    def test_default_recording_and_interventions_fail_before_model_execution(self):
        backend = FakeBackend()
        with TestClient(subject().create_app(backend=backend)) as client:
            common = {"messages": [{"role": "user", "content": "hello"}]}
            for extra in ({}, {"record": {"enabled": True}},
                          {"record": {"enabled": False}, "interventions": [{"kind": "steer", "tokens": ["yes"]}]}):
                with self.subTest(extra=extra):
                    self.assertEqual(client.post("/generate", json={**common, **extra}).status_code, 422)
        self.assertEqual(backend.requests, [])

    def test_info_and_generation_preserve_the_wire_contract_without_traces(self):
        backend = FakeBackend()
        with TestClient(subject().create_app(backend=backend)) as client:
            self.assertEqual(client.get("/health").status_code, 200)
            info = client.get("/info").json()
            self.assertEqual(info["revision"], REVISION)
            self.assertEqual(info["model_path"], "/cache/snapshots/" + REVISION)
            response = client.post("/generate", json={"messages": [{"role": "user", "content": "hello"}],
                                                     "max_new_tokens": 384, "record": {"enabled": False}})
        self.assertEqual(response.status_code, 200)
        result = GenerateResponse.model_validate(response.json())
        self.assertEqual(result.text, '{"choice":"tie"}')
        self.assertIsNone(result.trace)
        self.assertEqual(result.interventions_applied, 0)
        self.assertEqual(backend.requests[0].max_new_tokens, 384)

    def test_loading_failures_surface_in_health_and_block_generation(self):
        def failed_loader():
            raise RuntimeError("fixture model load failed")
        with self.assertLogs(level="ERROR"), TestClient(subject().create_app(loader=failed_loader)) as client:
            # Calling /info yields execution to the model loading worker.
            for _ in range(20):
                health = client.get("/health")
                if health.json().get("status") == "error":
                    break
            self.assertEqual(health.status_code, 503)
            self.assertEqual(health.json()["status"], "error")
            self.assertIn("fixture model load failed", health.json()["detail"])
            response = client.post("/generate", json={"prompt": "hi", "record": {"enabled": False}})
            self.assertEqual(response.status_code, 503)


class BackendGenerationTests(unittest.TestCase):
    def test_overbudget_request_fails_before_allocating_or_generating(self):
        class LongTokenizer(Tokenizer):
            def __call__(self, text, *, add_special_tokens, truncation):
                return {"input_ids": [10] * 3713}
        m = subject()
        backend = m.PlainHFBackend(m.ModelConfig(model_id="allenai/example", revision=REVISION, family="olmo"))
        backend.model = Mock()
        backend.tokenizer = LongTokenizer()
        req = GenerateRequest(messages=[{"role": "user", "content": "long prompt"}],
                              max_new_tokens=384, record=RecordSpec(enabled=False))
        with self.assertRaisesRegex(ValueError, "prompt=3713.*requested_output=384"):
            backend.generate(req)
        backend.model.generate.assert_not_called()

    def test_generation_decodes_new_tokens_and_preserves_deterministic_policy(self):
        class Tensor:
            def __init__(self, values):
                self.values = values
            def __getitem__(self, item):
                return Tensor(self.values[item])
            def tolist(self):
                return self.values
        m = subject()
        backend = m.PlainHFBackend(m.ModelConfig(model_id="allenai/example", revision=REVISION,
                                                family="olmo", name="olmo2-32b"))
        backend.tokenizer = Tokenizer()
        strings = {1: "<BOS>", 10: "User", 11: "Assistant", 42: "AnswerEND", 2: "<EOS>"}
        backend.tokenizer.decode = lambda ids, skip_special_tokens, **kw: "".join(
            strings[i] for i in ids if not skip_special_tokens or i not in (1, 2))
        backend.model = SimpleNamespace(get_input_embeddings=lambda: SimpleNamespace(
            weight=SimpleNamespace(device="cuda:0")),
            generate=Mock(return_value=Tensor([[1, 10, 11, 42, 2]])))
        backend.torch = SimpleNamespace(tensor=lambda values, **kw: Tensor(values),
                                        long="int64", ones_like=lambda value: value,
                                        manual_seed=Mock(), inference_mode=nullcontext)
        response = backend.generate(GenerateRequest(
            messages=[{"role": "user", "content": "hello"}], max_new_tokens=384,
            stop=["END"], temperature=0., seed=17, record=RecordSpec(enabled=False)))
        self.assertEqual(response.text, "Answer")
        self.assertEqual(response.prompt_len, 3)
        self.assertEqual(response.n_generated, 2)
        self.assertEqual(len(response.tokens), 5)
        self.assertIsNone(response.trace)
        kwargs = backend.model.generate.call_args.kwargs
        self.assertFalse(kwargs["do_sample"])
        self.assertEqual(kwargs["max_new_tokens"], 384)
        self.assertEqual(kwargs["stop_strings"], ["END"])
        self.assertNotIn("temperature", kwargs)
        self.assertEqual(kwargs["input_ids"].tolist(), [[1, 10, 11]])
        backend.torch.manual_seed.assert_called_once_with(17)


class LoadingTests(unittest.TestCase):
    def modules(self, *, count=2, mapping=None):
        torch = SimpleNamespace(bfloat16="bf16", cuda=SimpleNamespace(
            device_count=lambda: count, get_device_properties=lambda i: SimpleNamespace(
                name=f"fixture GPU {i}", total_memory=48 * 1024 ** 3, uuid=f"GPU-{i}")))
        model = Mock()
        model.hf_device_map = mapping or {"model.embed_tokens": 0, "model.layers": 1, "lm_head": 1}
        model.config = SimpleNamespace(max_position_embeddings=4096, num_hidden_layers=64,
                                       hidden_size=5120, _commit_hash=REVISION)
        transformers = SimpleNamespace(AutoTokenizer=SimpleNamespace(from_pretrained=Mock(return_value=Tokenizer())),
                                       AutoModelForCausalLM=SimpleNamespace(from_pretrained=Mock(return_value=model)))
        return torch, transformers

    def test_model_loading_is_bf16_two_gpu_only_and_rejects_cpu_placement(self):
        torch, transformers = self.modules(mapping={"model.layers": 0, "lm_head": "cpu"})
        with patch.dict(sys.modules, {"torch": torch, "transformers": transformers}):
            backend = subject().PlainHFBackend(subject().ModelConfig(
                model_id="allenai/OLMo-2-0325-32B-Instruct", revision=REVISION, family="olmo"))
            with self.assertRaisesRegex(RuntimeError, "GPU-only"):
                backend.load()
        kwargs = transformers.AutoModelForCausalLM.from_pretrained.call_args.kwargs
        self.assertEqual(kwargs["dtype"], "bf16")
        self.assertEqual(kwargs["device_map"], "balanced")
        self.assertEqual(kwargs["max_memory"][0], "34GiB")
        self.assertEqual(kwargs["max_memory"][1], "34GiB")
        self.assertFalse(kwargs["trust_remote_code"])
        self.assertEqual(kwargs["revision"], REVISION)

    def test_wrong_visible_device_count_fails_before_loading_weights(self):
        torch, transformers = self.modules(count=1)
        with patch.dict(sys.modules, {"torch": torch, "transformers": transformers}):
            backend = subject().PlainHFBackend(subject().ModelConfig(
                model_id="allenai/example", revision=REVISION, family="olmo"))
            with self.assertRaisesRegex(RuntimeError, "two visible"):
                backend.load()
        transformers.AutoModelForCausalLM.from_pretrained.assert_not_called()


if __name__ == "__main__":
    unittest.main()
