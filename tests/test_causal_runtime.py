"""CPU-only causal-edit tests; no model weights or tokenizer downloads."""
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from pydantic import ValidationError

from jpeers.runtime import LensedModel, _Run
from jpeers.schemas import GenerateRequest, InterventionSpec, PositionSelector, RecordSpec
from jpeers.trace import JSpaceTrace, encode_arrays


class ToyModel(LensedModel):
    """Small deterministic residual/lens system, avoiding external model fixtures."""
    def __init__(self):
        self.d_model = 4
        self.n_layers = 3
        self.vocab_size = 4
        self._W_U = torch.diag(torch.tensor([1., 2., 3., 4.]))
        self._gain = torch.tensor([1., 2., 1., 1.])
        self._J_cache = {}
        self._vocab_strings = ["A", "B", "C", "D"]
        self.tokenizer = lambda word, **kw: SimpleNamespace(
            input_ids=[self.vocab_strings.index(word)])
        self._word_like_dev = {}
        self.lens = SimpleNamespace(jacobians={i: torch.eye(4) for i in range(3)})
        self.lens.jacobians[1][0] = torch.tensor([1., 1., 0., 0.])
        self.lm = SimpleNamespace(layers=[torch.nn.Identity() for _ in range(3)],
                                  unembed=lambda x: x @ self._W_U.T)

    def band(self, which="band"):
        return sorted(set(which)) if isinstance(which, list) else [0, 1, 2]

    def word_token_ids(self, word):
        return [self.vocab_strings.index(word)]


def fixed(kind="fixed_delete", **kwargs):
    data = dict(kind=kind, layers=[1], positions=PositionSelector(scope="last"),
                basis=[[1., 0., 0., 0.]], deletion_fraction=.5)
    data.update(kwargs)
    return InterventionSpec(**data)


def run_for(*ivs, record=None):
    return _Run(ToyModel(), GenerateRequest(prompt="toy", interventions=list(ivs),
                record=record or RecordSpec(enabled=False)), 3, {"neutral": [(2, 3)]})


def test_zero_dose_is_identity_and_partial_delete_has_known_coordinates():
    h = torch.tensor([[3., 4., 2., 1.]])
    zero = fixed(deletion_fraction=0.)
    run = run_for(zero)
    assert torch.equal(run._apply(zero, 1, h.clone(), None), h)
    iv = fixed()
    out = run._apply(iv, 1, h.clone(), None)
    torch.testing.assert_close(out, torch.tensor([[1.5, 4., 2., 1.]]))


def test_fixed_random_deletion_matches_partial_edit_norm_and_dot():
    h = torch.tensor([[3., 4., 2., 1.], [2., -1., 3., 2.]])
    iv, control = fixed(), fixed("fixed_random_matched", seed=19)
    run = run_for(iv, control)
    targeted = run._apply(iv, 1, h.clone(), None)
    random = run._apply(control, 1, h.clone(), None)
    torch.testing.assert_close((h-targeted).norm(dim=-1), (h-random).norm(dim=-1))
    torch.testing.assert_close(((h-targeted)*h).sum(-1), ((h-random)*h).sum(-1))
    torch.testing.assert_close(targeted.norm(dim=-1), random.norm(dim=-1))
    assert not torch.allclose(targeted, random)


def test_patch_replaces_only_basis_coordinates_and_wrong_donor_uses_same_api():
    h = torch.tensor([[3., 4., 2., 1.]])
    basis = [[0., 1., 0., 0.], [0., 0., 1., 0.]]
    clean = fixed("fixed_patch", basis=basis, donor_coefficients=[8., -2.])
    wrong = fixed("fixed_patch", basis=basis, donor_coefficients=[-8., 2.])
    run = run_for(clean, wrong)
    torch.testing.assert_close(run._apply(clean, 1, h.clone(), None),
                               torch.tensor([[3., 8., -2., 1.]]))
    torch.testing.assert_close(run._apply(wrong, 1, h.clone(), None),
                               torch.tensor([[3., -8., 2., 1.]]))


def test_random_restoration_matches_actual_donor_displacement_norm_and_dot():
    h = torch.tensor([[3., 4., 2., 1.], [2., -1., 3., 2.]])
    patch = fixed("fixed_patch", donor_coefficients=[7.])
    control = fixed("fixed_random_restore", donor_coefficients=[7.], seed=19)
    run = run_for(patch, control)
    target = run._apply(patch, 1, h.clone(), None)
    random = run._apply(control, 1, h.clone(), None)
    torch.testing.assert_close((target-h).norm(dim=-1), (random-h).norm(dim=-1))
    torch.testing.assert_close(((target-h)*h).sum(-1), ((random-h)*h).sum(-1))
    torch.testing.assert_close(target.norm(dim=-1), random.norm(dim=-1))
    assert not torch.allclose(target, random)


@pytest.mark.parametrize("changes", [
    {"basis": []}, {"basis": [[2., 0., 0., 0.]]},
    {"basis": [[1., 0.], [1., 0.]]}, {"basis": [[float("nan"), 0., 0., 0.]]},
    {"basis": [[float("inf"), 0., 0., 0.]]}, {"deletion_fraction": -.1},
    {"deletion_fraction": 1.1}, {"deletion_fraction": float("nan")},
    {"layers": "band"}, {"layers": [0, 1]},
    {"positions": PositionSelector(scope="generated")},
    {"kind": "fixed_patch"}, {"kind": "fixed_patch", "donor_coefficients": [1., 2.]},
    {"kind": "fixed_patch", "donor_coefficients": [float("inf")]},
])
def test_invalid_causal_request_is_rejected(changes):
    with pytest.raises((ValueError, ValidationError)):
        run_for(fixed(**changes))


def test_wrong_dimension_layer_or_nonunique_site_fails_closed():
    with pytest.raises(ValueError, match="dimension"):
        run_for(fixed(basis=[[1., 0.]]))
    with pytest.raises(ValueError, match="layer"):
        run_for(fixed(layers=[99]))
    with pytest.raises(ValueError, match="one prompt position"):
        run_for(fixed(positions=PositionSelector(scope="spans", labels=["missing"])))


def capture(when="post"):
    return RecordSpec(layers=[0, 1, 2], positions=PositionSelector(scope="last"),
                      when=when, top_k=2, watch=["A", "B"], word_like_only=False,
                      baselines=[], capture_residuals=True,
                      capture_directions=["jacobian", "logit"])


def forward_layers(run):
    states = []
    h = torch.tensor([[[1., 2., 3., 4.], [2., 3., 4., 5.], [3., 4., 5., 6.]]])
    with run.hooks():
        for layer in run.m.lm.layers:
            h = layer(h)
            states.append(h.clone())
    return states


def test_capture_respects_pre_post_selected_layer_site_and_isolation():
    iv = fixed()
    pre, post = run_for(iv, record=capture("pre")), run_for(iv, record=capture("post"))
    states = forward_layers(pre)
    forward_layers(post)
    a, b = pre.finish(), post.finish()
    assert a["residuals"].shape == (1, 3, 4)
    np.testing.assert_array_equal(a["positions"], [2])
    np.testing.assert_array_equal(a["residuals"][0, 1], [3., 4., 5., 6.])
    np.testing.assert_array_equal(b["residuals"][0, 1], [1.5, 4., 5., 6.])
    torch.testing.assert_close(states[0][0, 2], torch.tensor([3., 4., 5., 6.]))
    torch.testing.assert_close(states[1][0, :2], states[0][0, :2])
    clean = run_for(record=capture())
    forward_layers(clean)
    np.testing.assert_array_equal(clean.finish()["residuals"][0, 1], [3., 4., 5., 6.])
    assert clean.causal_edits == []
    assert len(post.causal_edits) == 1
    event = post.causal_edits[0]
    assert (event["layer"], event["position"], event["kind"]) == (1, 2, "fixed_delete")
    assert event["edit_norm"] == pytest.approx(1.5)
    assert event["h_dot_delta"] == pytest.approx(-4.5)


def test_direction_rows_use_recorded_primary_token_ids_and_correct_convention():
    run = run_for(record=capture())
    forward_layers(run)
    arrays = run.finish()
    assert arrays["directions_jacobian"].shape == (3, 2, 4)
    np.testing.assert_array_equal(arrays["direction_token_ids"], [0, 1])
    np.testing.assert_allclose(arrays["directions_jacobian"][1, 0],
                               [2**-.5, 2**-.5, 0., 0.], rtol=1e-6)
    np.testing.assert_array_equal(arrays["directions_logit"][1, 0], [1., 0., 0., 0.])
    assert run.capture_metadata["residual_site"] == "block_output"
    assert run.capture_metadata["when"] == "post"


@pytest.mark.parametrize("encoding", ["json", "npz_b64"])
def test_causal_arrays_and_metadata_survive_response_save_load(tmp_path, encoding):
    run = run_for(fixed(), record=capture())
    forward_layers(run)
    response = dict(trace=encode_arrays(run.finish(), encoding), trace_encoding=encoding,
                    spans=[], layers=[0, 1, 2], watch=["A", "B"], watch_ids=[[0], [1]],
                    tokens=[], vocab={}, prompt_len=3,
                    capture_metadata=run.capture_metadata, causal_edits=run.causal_edits)
    tr = JSpaceTrace.from_response(response)
    path = tmp_path / "capture.npz"
    tr.save(str(path))
    loaded = JSpaceTrace.load(str(path))
    for key in ("residuals", "directions_jacobian", "directions_logit", "direction_token_ids"):
        np.testing.assert_allclose(getattr(tr, key), getattr(loaded, key))
    assert loaded.capture_metadata == response["capture_metadata"]
    assert loaded.causal_edits == response["causal_edits"]


def test_legacy_trace_and_interventions_have_no_opt_in_capture():
    iv = InterventionSpec(kind="random_ablate", layers=[1], k=1, seed=2)
    run = run_for(iv, record=RecordSpec(layers=[1], top_k=2, word_like_only=False))
    forward_layers(run)
    assert "residuals" not in run.finish()
    assert run.capture_metadata == {} and run.causal_edits == []
    h = torch.tensor([[3., 4., 2., 1.]])
    edited = run._apply(iv, 1, h.clone(), None)
    torch.testing.assert_close(run._apply(iv, 1, edited.clone(), None), edited)


def test_span_endpoint_selector_keeps_shared_neutral_suffix_to_one_site():
    iv = fixed(positions=PositionSelector(scope="spans", labels=["suffix"], span_last=True))
    req = GenerateRequest(prompt="toy", interventions=[iv], record=capture())
    run = _Run(ToyModel(), req, 3, {"suffix": [(1, 3)]})
    states = forward_layers(run)
    torch.testing.assert_close(states[1][0, 1], torch.tensor([2., 3., 4., 5.]))
    torch.testing.assert_close(states[1][0, 2], torch.tensor([1.5, 4., 5., 6.]))


def test_bfloat16_edit_reports_desired_realized_and_quantization_geometry():
    run = run_for(fixed(deletion_fraction=.3), record=capture())
    h = torch.tensor([[[1., 2., 3., 4.], [2., 3., 4., 5.], [3., 4., 5., 6.]]],
                     dtype=torch.bfloat16)
    with run.hooks():
        run.m.lm.layers[1](h)
    event = run.causal_edits[0]
    assert event["desired_edit_norm"] == pytest.approx(.9)
    assert event["desired_h_dot_delta"] == pytest.approx(-2.7)
    assert event["edit_norm"] == pytest.approx(.90625)
    assert event["quantization_error_norm"] == pytest.approx(.00625, abs=1e-6)


def test_matched_controls_handle_zero_recipient_and_zero_displacement():
    zero = torch.zeros(1, 4)
    iv = fixed("fixed_random_restore", donor_coefficients=[2.])
    run = run_for(iv)
    out = run._apply(iv, 1, zero.clone(), None)
    torch.testing.assert_close(out.norm(), torch.tensor(2.))
    assert torch.isfinite(out).all()
    noop = fixed("fixed_random_restore", donor_coefficients=[0.])
    assert torch.equal(run._apply(noop, 1, zero.clone(), None), zero)
    with pytest.raises(ValueError, match="finite"):
        run._apply(iv, 1, torch.full((1, 4), float("nan")), None)


def test_direction_capture_supports_empty_watch_without_unrequested_residuals():
    run = run_for(record=RecordSpec(layers=[1], positions=PositionSelector(scope="last"),
                                  top_k=2, word_like_only=False,
                                  capture_directions=["jacobian", "logit"]))
    forward_layers(run)
    arrays = run.finish()
    assert arrays["directions_jacobian"].shape == (1, 0, 4)
    assert arrays["direction_token_ids"].shape == (0,)
    assert "residuals" not in arrays


def test_bare_label_direction_ids_and_tokenization_are_explicit():
    model = ToyModel()
    model.tokenizer = lambda word, **kw: SimpleNamespace(input_ids={"A": [2], "B": [3]}[word])
    model.word_token_ids = lambda word: {"A": [0, 2], "B": [1, 3]}[word]
    rec = capture()
    rec.direction_token_form = "bare"
    run = _Run(model, GenerateRequest(prompt="toy", record=rec), 3, {})
    forward_layers(run)
    arrays = run.finish()
    np.testing.assert_array_equal(arrays["direction_token_ids"], [2, 3])
    np.testing.assert_array_equal(arrays["directions_logit"][0, 0], [0., 0., 1., 0.])
    assert run.capture_metadata["label_tokenization"]["A"] == {
        "bare_ids": [2], "primary_id": 0, "direction_id": 2}
    model.tokenizer = lambda word, **kw: SimpleNamespace(input_ids=[0, 1])
    with pytest.raises(ValueError, match="bare.*single token"):
        _Run(model, GenerateRequest(prompt="toy", record=rec), 3, {})


@pytest.mark.parametrize("kind,extra,expected", [
    ("ablate_topk", {"k": 1}, [[3., 0., 2., 1.]]),
    ("scale_jspace", {"k": 1, "strength": .25}, [[3., 1., 2., 1.]]),
    ("ablate_tokens", {"tokens": ["A"]}, [[0., 4., 2., 1.]]),
    ("swap", {"source": "A", "target": "B"}, [[4., 3., 2., 1.]]),
])
def test_existing_intervention_coordinate_results_are_unchanged(kind, extra, expected):
    iv = InterventionSpec(kind=kind, layers=[0], **extra)
    run = run_for(iv)
    out = run._apply(iv, 0, torch.tensor([[3., 4., 2., 1.]]), None)
    torch.testing.assert_close(out, torch.tensor(expected))


def test_real_model_generate_capture_and_edit_are_request_isolated_without_downloads():
    import transformers
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from jpeers.registry import ModelSpec

    tokenizer = Tokenizer(WordLevel({"[UNK]": 0, "[BOS]": 1, "[EOS]": 2,
                                      "red": 3, "blue": 4, "notes": 5,
                                      "Red": 6, "Blue": 7}, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    tok = transformers.PreTrainedTokenizerFast(tokenizer_object=tokenizer,
            unk_token="[UNK]", bos_token="[BOS]", eos_token="[EOS]")
    torch.manual_seed(17)
    hf = transformers.GPT2LMHeadModel(transformers.GPT2Config(
        n_layer=3, n_embd=8, n_head=2, n_positions=32, vocab_size=8,
        bos_token_id=1, eos_token_id=2)).eval()
    lens = SimpleNamespace(jacobians={i: torch.eye(8) for i in (0, 1)})
    model = LensedModel(ModelSpec("cpu-wordlevel", "tiny", "test", 0., chat=False,
                                  band=(0., 1.)), hf_model=hf, tokenizer=tok,
                         lens=lens, device="cpu")
    rec = RecordSpec(layers=[0, 1], positions=PositionSelector(scope="last"),
            top_k=2, word_like_only=False, watch=["red", "blue"], capture_residuals=True,
            capture_directions=["jacobian", "logit"], direction_token_form="bare")
    request = dict(prompt="red blue notes", max_new_tokens=2, record=rec)
    baseline = model.generate(GenerateRequest(**request))
    iv = InterventionSpec(kind="fixed_delete", layers=[0],
            positions=PositionSelector(scope="last"), deletion_fraction=.5,
            basis=[[1., 0., 0., 0., 0., 0., 0., 0.]])
    edited = model.generate(GenerateRequest(**request, interventions=[iv]))
    later = model.generate(GenerateRequest(**request))
    first, changed, clean = [JSpaceTrace.from_response(r) for r in (baseline, edited, later)]
    np.testing.assert_array_equal(first.residuals, clean.residuals)
    assert baseline.text == later.text
    assert not np.array_equal(first.residuals, changed.residuals)
    assert len(edited.causal_edits) == 1 and baseline.causal_edits == []
    assert all(not layer._forward_hooks for layer in model.lm.layers)
    assert changed.direction_token_ids.tolist() == [3, 4]
    assert edited.capture_metadata["label_tokenization"]["red"]["bare_ids"] == [3]


@pytest.mark.parametrize("donor", [[float("nan")], [1., 2.]])
def test_unused_donor_payload_still_fails_closed(donor):
    with pytest.raises(ValueError, match="donor_coefficients"):
        fixed(donor_coefficients=donor)


def test_composed_fixed_edits_report_the_executed_bfloat16_intermediate():
    deletion = fixed(deletion_fraction=.3)
    patch = fixed("fixed_patch", donor_coefficients=[3.])
    run = run_for(deletion, patch)
    original = torch.tensor([[[1., 2., 3., 4.], [2., 3., 4., 5.], [3., 4., 5., 6.]]],
                            dtype=torch.bfloat16)
    with run.hooks():
        result = run.m.lm.layers[1](original.clone())
    assert torch.equal(result, original)
    first, second = run.causal_edits
    assert first["hidden_norm_after"] == second["hidden_norm_before"]
    assert second["edit_norm"] == pytest.approx(.90625)
    assert second["desired_edit_norm"] == pytest.approx(.90625)


def test_existing_positional_trace_constructor_keeps_metadata_order():
    tr = JSpaceTrace(np.array([2]), np.zeros((1, 1, 1), int),
                    np.zeros((1, 1, 1)), np.zeros((1, 1, 1), int),
                    None, None, None, [1], ["A"])
    assert tr.layers == [1] and tr.watch == ["A"]
    assert tr.residuals is None


def test_unrepresentable_donor_values_fail_before_editing():
    iv = fixed("fixed_patch", donor_coefficients=[1e300])
    run = run_for(iv)
    with pytest.raises(ValueError, match="finite"):
        run._apply(iv, 1, torch.ones(1, 4), None)


def test_finite_large_edit_geometry_does_not_overflow_reported_norms():
    run = run_for(fixed("fixed_patch", donor_coefficients=[1e20]))
    h = torch.ones(1, 3, 4)
    with run.hooks():
        run.m.lm.layers[1](h)
    assert run.causal_edits[0]["edit_norm"] == pytest.approx(1e20)
    assert run.causal_edits[0]["hidden_norm_after"] == pytest.approx(1e20)


def test_bfloat16_legacy_post_capture_matches_delivered_hidden_and_preserves_readouts():
    iv = InterventionSpec(kind="steer", layers=[1], tokens=["A"], strength=.3,
                          positions=PositionSelector(scope="last"))
    rec = capture("post")
    rec.layers = [1]
    captured = run_for(iv, record=rec)
    plain_rec = rec.model_copy(update={"capture_residuals": False, "capture_directions": []})
    plain = run_for(iv, record=plain_rec)
    hidden = torch.tensor([[[1., 2., 3., 4.], [2., 3., 4., 5.], [3., 4., 5., 6.]]],
                          dtype=torch.bfloat16)
    with captured.hooks():
        delivered = captured.m.lm.layers[1](hidden.clone())
    with plain.hooks():
        uncaptured = plain.m.lm.layers[1](hidden.clone())
    assert torch.equal(delivered, uncaptured)
    arrays, historical = captured.finish(), plain.finish()
    np.testing.assert_array_equal(arrays["residuals"][0, 0], delivered[0, 2].float().numpy())
    for key in ("topk_ids", "topk_logits", "watch_ranks"):
        np.testing.assert_array_equal(arrays[key], historical[key])
    assert captured.capture_metadata["residual_storage_dtype"] == "float32"
    assert captured.capture_metadata["residual_dtype_by_layer"] == {"1": "bfloat16"}
