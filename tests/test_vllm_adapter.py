"""CPU checks for vLLM's split residual and the experiment trace contract.

These use actual _Run recording/projection math; no model download or vLLM is needed.
"""
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from jpeers.registry import ModelSpec
from jpeers.runtime import LensedModel, _Run
from jpeers.schemas import GenerateRequest, InterventionSpec, PositionSelector, RecordSpec


class _Tokenizer:
    def __call__(self, text, **kwargs):
        return SimpleNamespace(input_ids=[{"Bob": 0, "Alice": 1, "Rome": 2, "Paris": 3}[text.strip()]])


def _model():
    model = object.__new__(LensedModel)
    model.spec = ModelSpec("toy", "toy", "qwen", 0, band=(0, 1))
    model.n_layers, model.d_model, model.vocab_size = 2, 4, 4
    model.tokenizer = _Tokenizer()
    model.lens = SimpleNamespace(jacobians={0: torch.eye(4)})
    model._J_cache, model._word_like_dev = {}, {}
    model._W_U, model._gain = torch.eye(4), torch.ones(4)
    model._vocab_strings = ["Bob", "Alice", "Rome", "Paris"]
    model._word_like = torch.ones(4, dtype=torch.bool)
    model.lm = SimpleNamespace(unembed=lambda h: h)
    return model


def _adapter():
    try:
        from jpeers import vllm_runtime
    except ImportError as exc:
        pytest.fail(f"vLLM experiment adapter is not implemented: {exc}")
    return vllm_runtime


def test_decoder_hook_records_full_residual_not_mlp_delta():
    adapter = _adapter()
    req = GenerateRequest(prompt="x", record=RecordSpec(layers=[0], watch=["Bob"], top_k=2))
    run = _Run(_model(), req, 2, {})
    delta = torch.tensor([[0., 0., 0., 1.], [0., 0., 0., 2.]])
    residual = torch.tensor([[4., 3., 2., 0.], [3., 4., 2., 0.]])
    output = (delta, residual)
    returned = adapter.make_decoder_hook(run, 0)(None, (torch.tensor([0, 1]),), {}, output)
    assert returned is None  # an unedited forward keeps native arithmetic exactly
    arrays = run.finish()
    assert arrays["positions"].tolist() == [0, 1]
    assert arrays["topk_ids"][:, 0, 0].tolist() == [0, 1]
    assert arrays["watch_ranks"][:, 0, 0].tolist() == [0, 1]
    assert torch.equal(delta, output[0]) and torch.equal(residual, output[1])


def test_decoder_projection_matches_hf_at_selected_absolute_positions():
    adapter = _adapter()
    iv = InterventionSpec(kind="ablate_tokens", layers=[0], tokens=["Bob"],
                          positions=PositionSelector(scope="generated"))
    req = GenerateRequest(prompt="x", interventions=[iv],
                          record=RecordSpec(layers=[0], watch=["Bob"], top_k=2))
    run = _Run(_model(), req, 7, {})
    delta = torch.tensor([[0., 1., 0., 0.], [1., 0., 0., 0.]])
    residual = torch.tensor([[4., 2., 1., 0.], [4., 3., 2., 1.]])
    output = adapter.make_decoder_hook(run, 0)(
        None, (), {"positions": torch.tensor([6, 7])}, (delta, residual))
    assert output is not None
    actual = output[0] + output[1]
    assert torch.equal(actual, torch.tensor([[4., 3., 1., 0.], [0., 3., 2., 1.]]))
    assert run.n_edits == 1
    assert run.finish()["positions"].tolist() == [6, 7]
    assert run.finish()["watch_ranks"][:, 0, 0].tolist() == [0, 3]


def test_output_baseline_aligns_with_sparse_record_positions():
    adapter = _adapter()
    req = GenerateRequest(prompt="x", record=RecordSpec(
        layers=[0], watch=["Bob"], top_k=2, include_model_logits=True,
        positions=PositionSelector(scope="spans", labels=["peer"], include_generated=True)))
    run = _Run(_model(), req, 4, {"peer": [(1, 3)]})
    positions = torch.tensor([0, 1, 2, 3])
    hidden = torch.tensor([[9., 0., 0., 0.], [1., 4., 3., 2.],
                           [4., 3., 2., 1.], [0., 1., 2., 3.]])
    hook = adapter.make_decoder_hook(run, 0)
    hook(None, (positions,), {}, (hidden, torch.zeros_like(hidden)))
    adapter.record_output_baseline(run, hidden, positions, lambda hs: hs)
    generated = torch.tensor([[0., 1., 2., 3.]])
    hook(None, (torch.tensor([4]),), {}, (generated, torch.zeros_like(generated)))
    adapter.record_output_baseline(run, generated, torch.tensor([4]), lambda hs: hs)
    arrays = adapter.finish_run(run)
    assert arrays["positions"].tolist() == [1, 2, 4]
    assert arrays["output_ranks"][:, 0].tolist() == [3, 0, 3]
    assert arrays["model_top1"].tolist() == [1, 0, 3]
    assert arrays["watch_ranks"].shape[:2] == (3, 1)


def test_trace_rejects_unprocessed_final_sample_and_layer_misalignment():
    adapter = _adapter()
    arrays = {"positions": np.array([0, 1, 2, 3]), "topk_ids": np.zeros((4, 1, 1))}
    adapter.validate_trace_positions(arrays, prompt_len=3, n_generated=2)
    with pytest.raises(RuntimeError, match="unprocessed"):
        adapter.validate_trace_positions(arrays, prompt_len=3, n_generated=1)
    adapter.validate_trace_positions({"positions": np.arange(3)}, prompt_len=3, n_generated=0)


def test_recording_uses_true_position_ids_instead_of_forward_counter():
    adapter = _adapter()
    req = GenerateRequest(prompt="x", record=RecordSpec(
        layers=[0], top_k=2, positions=PositionSelector(scope="last")))
    run = _Run(_model(), req, 10, {})
    hook = adapter.make_decoder_hook(run, 0)
    h = torch.ones(2, 4)
    hook(None, (torch.tensor([8, 9]),), {}, (h, h))
    assert run.finish()["positions"].tolist() == [9]


def test_edit_preserves_unselected_native_delta_in_bfloat16():
    adapter = _adapter()
    req = GenerateRequest(prompt="x", record=RecordSpec(enabled=False), interventions=[
        InterventionSpec(kind="ablate_tokens", layers=[0], tokens=["Bob"],
                         positions=PositionSelector(scope="generated"))])
    run = _Run(_model(), req, 1, {})
    delta = torch.tensor([[.001, .001, .001, .001], [1., 2., 3., 4.]], dtype=torch.bfloat16)
    residual = torch.ones_like(delta)
    output = adapter.make_decoder_hook(run, 0)(
        None, (torch.tensor([0, 1]),), {}, (delta, residual))
    assert torch.equal(output[0][0], delta[0])
    assert torch.equal(output[1][0], residual[0])


def test_bfloat16_edit_survives_large_deferred_residual_cancellation():
    adapter = _adapter()
    req = GenerateRequest(prompt="x", record=RecordSpec(enabled=False), interventions=[
        InterventionSpec(kind="swap", layers=[0], source="Bob", target="Paris")])
    run = _Run(_model(), req, 1, {})
    delta = torch.tensor([[.001, 3., 2., 1.]], dtype=torch.bfloat16)
    residual = torch.tensor([[1024., 0., 0., 0.]], dtype=torch.bfloat16)
    output = adapter.make_decoder_hook(run, 0)(
        None, (torch.tensor([0]),), {}, (delta, residual))
    expected = torch.tensor([[1., 3., 2., 1024.]], dtype=torch.bfloat16)
    assert torch.equal(output[0] + output[1], expected)


def test_output_baseline_does_not_compute_unrecorded_rows():
    adapter = _adapter()
    req = GenerateRequest(prompt="x", record=RecordSpec(
        layers=[0], watch=["Bob"], top_k=2, positions=PositionSelector(scope="last")))
    run = _Run(_model(), req, 5, {})
    seen = []

    def output_logits(hidden):
        seen.append(hidden.clone())
        return hidden

    hidden = torch.arange(20.).reshape(5, 4)
    adapter.record_output_baseline(run, hidden, torch.arange(5), output_logits)
    assert len(seen) == 1
    assert torch.equal(seen[0], hidden[-1:])


def test_identity_intervention_keeps_native_split_arithmetic():
    adapter = _adapter()
    req = GenerateRequest(prompt="x", record=RecordSpec(enabled=False), interventions=[
        InterventionSpec(kind="steer", layers=[0], tokens=["Bob"], strength=0)])
    run = _Run(_model(), req, 1, {})
    delta = torch.full((1, 4), .001, dtype=torch.bfloat16)
    residual = torch.ones_like(delta)
    output = adapter.make_decoder_hook(run, 0)(
        None, (torch.tensor([0]),), {}, (delta, residual))
    assert output is None
    assert run.n_edits == 1  # retain the repository's selected edit count
