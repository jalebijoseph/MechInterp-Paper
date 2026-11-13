import numpy as np
import pytest
import torch

from jpeers.runtime import _Run
from jpeers.schemas import (
    GenerateRequest,
    InterventionSpec,
    PositionSelector,
    RecordSpec,
    SpanQuery,
)
from jpeers.trace import JSpaceTrace

MSGS = [{"role": "system", "content": "You are Alice, working with Bob."},
        {"role": "user", "content": "[Bob]: The capital of France is Rome. Please check it."}]


def test_band_and_layers(tiny_model):
    assert tiny_model.band() == [0, 1, 2]  # final block has no lens


def test_generate_records_trace(tiny_model):
    req = GenerateRequest(messages=MSGS, max_new_tokens=5,
                          record=RecordSpec(watch=["Bob", "capital"], top_k=4),
                          spans=[SpanQuery(label="claim", text="The capital of France is Rome")])
    resp = tiny_model.generate(req)
    assert resp.n_generated == 5
    tr = JSpaceTrace.from_response(resp)
    P = resp.prompt_len + resp.n_generated - 1  # last sampled token is never fed back
    assert tr.topk_ids.shape == (P, 3, 4)
    assert tr.watch_ranks.shape == (P, 3, 2)
    assert list(tr.positions) == list(range(P))
    a, b = tr.spans["claim"][0]
    claim_tokens = "".join(resp.tokens[a:b])
    assert "capital" in claim_tokens and "Rome" in claim_tokens
    assert "msg:1" in tr.spans and "role:user" in tr.spans


def test_watch_rank_matches_direct_computation(tiny_model):
    req = GenerateRequest(prompt="Alice and Bob are working together", max_new_tokens=0,
                          record=RecordSpec(watch=["Bob"], word_like_only=False, top_k=3))
    resp = tiny_model.generate(req)
    tr = JSpaceTrace.from_response(resp)
    ids, _ = tiny_model.encode(req.prompt, True)
    from jlens import ActivationRecorder

    with torch.no_grad(), ActivationRecorder(tiny_model.lm.layers, at=[1]) as rec:
        tiny_model.lm.forward(ids)
    logits = tiny_model.lens_logits(1, rec.activations[1][0])
    cand = tiny_model.word_token_ids("Bob")
    expected = torch.stack([(logits > logits[:, c:c + 1]).sum(-1) for c in cand]).min(0).values
    li = tr.layers.index(1)
    assert np.array_equal(tr.watch_ranks[:, li, 0], expected.numpy())
    assert np.array_equal(tr.topk_ids[:, li, 0], logits.argmax(-1).numpy())


def test_baselines_match_direct_computation(tiny_model):
    req = GenerateRequest(prompt="Alice and Bob are working together", max_new_tokens=0,
                          record=RecordSpec(watch=["Bob"]))
    tr = JSpaceTrace.from_response(tiny_model.generate(req))
    ids, _ = tiny_model.encode(req.prompt, True)
    from jlens import ActivationRecorder

    with torch.no_grad(), ActivationRecorder(tiny_model.lm.layers, at=[1]) as rec:
        model_logits = tiny_model.hf(input_ids=ids).logits[0].float()
    cand = tiny_model.word_token_ids("Bob")

    def ranks(lg):
        return torch.stack([(lg > lg[:, c:c + 1]).sum(-1) for c in cand]).min(0).values.numpy()

    logit_lens = tiny_model.lens_logits(1, rec.activations[1][0], use_jacobian=False)
    li = tr.layers.index(1)
    assert np.array_equal(tr.watch_ranks_logit[:, li, 0], ranks(logit_lens))
    assert np.array_equal(tr.output_ranks[:, 0], ranks(model_logits))


def test_baselines_can_be_turned_off(tiny_model):
    req = GenerateRequest(prompt="Alice met Bob", max_new_tokens=2,
                          record=RecordSpec(watch=["Bob"], baselines=[]))
    tr = JSpaceTrace.from_response(tiny_model.generate(req))
    assert tr.watch_ranks_logit is None and tr.output_ranks is None


def test_upcoming_mask_and_lens_mrrs(tiny_model):
    from jpeers.analysis import lens_mrrs, upcoming_mask

    req = GenerateRequest(prompt="Alice and Bob are working together with Bob", max_new_tokens=0,
                          record=RecordSpec(watch=["Bob"]))
    tr = JSpaceTrace.from_response(tiny_model.generate(req))
    up = upcoming_mask(tr, ["Bob"], horizon=1)
    # exactly the positions whose next token is " Bob"
    expected = [tr.tokens[p + 1] == " Bob" if p + 1 < len(tr.tokens) else False
                for p in tr.positions]
    assert up.tolist() == expected and up.any()
    out = lens_mrrs(tr, ["Bob"], np.ones(len(tr.positions), bool), "bob")
    assert set(out) == {"bob", "bob_logit", "bob_output", "bob_beyond"}
    assert all(0 <= v <= 1 for v in out.values())


def test_position_selectors(tiny_model):
    req = GenerateRequest(messages=MSGS, max_new_tokens=3,
                          record=RecordSpec(positions=PositionSelector(scope="generated")))
    resp = tiny_model.generate(req)
    tr = JSpaceTrace.from_response(resp)
    assert (tr.positions >= resp.prompt_len).all() and len(tr.positions) == 2
    req.record.positions = PositionSelector(scope="last")
    tr = JSpaceTrace.from_response(tiny_model.generate(req))
    assert list(tr.positions) == [resp.prompt_len - 1]


def _run(model, req):
    text, add = model.render(req)
    ids, offsets = model.encode(text, add)
    return _Run(model, req, ids.shape[1], {})


def test_ablate_topk_projects_out_lens_vectors(tiny_model):
    iv = InterventionSpec(kind="ablate_topk", k=3)
    run = _run(tiny_model, GenerateRequest(prompt="hello there", interventions=[iv]))
    h = torch.randn(5, tiny_model.d_model)
    top = tiny_model.lens_logits(1, h).topk(3).indices
    out = run._apply(iv, 1, h.clone(), None)
    V = tiny_model.lens_directions(1, top, h.device)  # [5, 3, d]
    residual = torch.einsum("nkd,nd->nk", V, out)
    assert residual.abs().max() < 1e-4


def test_random_ablate_is_fixed_projection(tiny_model):
    iv = InterventionSpec(kind="random_ablate", k=4, seed=7)
    run = _run(tiny_model, GenerateRequest(prompt="hello", interventions=[iv]))
    h = torch.randn(3, tiny_model.d_model)
    once = run._apply(iv, 2, h.clone(), None)
    twice = run._apply(iv, 2, once.clone(), None)
    assert torch.allclose(once, twice, atol=1e-5)  # idempotent projector
    assert not torch.allclose(once, h)


def test_matched_control_removes_same_amount_as_jspace(tiny_model):
    h = torch.randn(6, tiny_model.d_model)
    jsp, ctl = (InterventionSpec(kind=k, k=3) for k in ("ablate_topk", "random_matched"))
    run = _run(tiny_model, GenerateRequest(prompt="hello", interventions=[jsp, ctl]))
    out_j = run._apply(jsp, 1, h.clone(), None)
    out_c = run._apply(ctl, 1, h.clone(), None)
    # same size of edit, same length afterwards, different content removed
    assert torch.allclose((h - out_j).norm(dim=-1), (h - out_c).norm(dim=-1), rtol=1e-3)
    assert torch.allclose(out_j.norm(dim=-1), out_c.norm(dim=-1), rtol=1e-3)
    assert not torch.allclose(out_j, out_c, atol=1e-3)
    # it is a true projection: the removed part is orthogonal to what is left
    assert ((h - out_c) * out_c).sum(-1).abs().max() < 1e-3 * h.norm(dim=-1).max() ** 2


def test_watch_words_must_be_whole_tokens(tiny_model):
    assert tiny_model.word_token_ids("Bob")
    with pytest.raises(ValueError, match="not a single token"):
        tiny_model.word_token_ids("Brasilia")  # splits into fragments in GPT-2


def test_colliding_watch_words_are_rejected(tiny_model):
    req = GenerateRequest(prompt="hello", max_new_tokens=0,
                          record=RecordSpec(watch=["bob", "Bob"]))  # " Bob" is shared
    with pytest.raises(ValueError, match="share the token"):
        tiny_model.generate(req)


def test_steer_raises_target_logit(tiny_model):
    iv = InterventionSpec(kind="steer", tokens=["Bob"], strength=0.5)
    run = _run(tiny_model, GenerateRequest(prompt="hello", interventions=[iv]))
    h = torch.randn(6, tiny_model.d_model)
    tid = tiny_model.word_token_ids("Bob")[0]
    before = tiny_model.lens_logits(1, h)[:, tid]
    after = tiny_model.lens_logits(1, run._apply(iv, 1, h.clone(), None))[:, tid]
    assert (after > before).all()


def test_swap_exchanges_coordinates(tiny_model):
    iv = InterventionSpec(kind="swap", source="Alice", target="Bob")
    run = _run(tiny_model, GenerateRequest(prompt="hello", interventions=[iv]))
    h = torch.randn(4, tiny_model.d_model)
    V = run._dirs_for_words(1, ["Alice", "Bob"], h.device).T
    pinv = torch.linalg.pinv(V)
    c_before = h @ pinv.T
    c_after = run._apply(iv, 1, h.clone(), None) @ pinv.T
    assert torch.allclose(c_after, c_before.flip(-1), atol=1e-4)


@pytest.mark.parametrize("kind", ["ablate_topk", "random_ablate", "random_matched",
                                  "scale_jspace", "ablate_tokens"])
def test_interventions_change_generation_path(tiny_model, kind):
    iv = InterventionSpec(kind=kind, k=5, tokens=["Bob"], strength=0.0,
                          positions=PositionSelector(scope="all"))
    base = tiny_model.generate(GenerateRequest(messages=MSGS, max_new_tokens=4))
    edited = tiny_model.generate(GenerateRequest(messages=MSGS, max_new_tokens=4,
                                                 interventions=[iv]))
    assert edited.interventions_applied > 0
    tb, te = (JSpaceTrace.from_response(r) for r in (base, edited))
    assert not np.array_equal(tb.topk_logits, te.topk_logits)


def test_span_scoped_intervention_only_touches_span(tiny_model):
    iv = InterventionSpec(kind="ablate_topk", k=5,
                          positions=PositionSelector(scope="spans", labels=["claim"]))
    req = dict(messages=MSGS, max_new_tokens=0,
               spans=[SpanQuery(label="claim", text="Rome. Please check it.")],
               record=RecordSpec(when="post"))
    base = JSpaceTrace.from_response(tiny_model.generate(GenerateRequest(**req)))
    edited = JSpaceTrace.from_response(tiny_model.generate(GenerateRequest(**req,
                                                                           interventions=[iv])))
    a, _ = base.spans["claim"][0]
    # causal model: positions before the span are untouched
    assert np.array_equal(base.topk_ids[:a], edited.topk_ids[:a])
    assert not np.array_equal(base.topk_logits[a:], edited.topk_logits[a:])
