"""Server, client, harness, trace encoding, analysis, benchmarks."""

import numpy as np
from fastapi.testclient import TestClient

from jpeers import server
from jpeers.analysis import (
    bag_similarity,
    content_uptake,
    group_ranks,
    jspace_bag,
    rank_summary,
)
from jpeers.benchmarks import false_belief_items, load_task_bank
from jpeers.client import EndpointPool, LocalEndpoint
from jpeers.harness import Agent, Conversation, RunWriter, _merge_same_role, iter_turns
from jpeers.interventions import PRESETS, resolve
from jpeers.schemas import GenerateRequest, GenerateResponse, RecordSpec
from jpeers.trace import JSpaceTrace, decode_arrays, encode_arrays


def test_trace_encodings_roundtrip():
    arrays = {"positions": np.arange(5), "topk_ids": np.ones((5, 2, 3), dtype=np.int32),
              "topk_logits": np.zeros((5, 2, 3), dtype=np.float16),
              "watch_ranks": np.full((5, 2, 1), 7, dtype=np.int32)}
    for enc in ("json", "npz_b64"):
        out = decode_arrays(encode_arrays(arrays, enc), enc)
        for k, v in arrays.items():
            assert np.array_equal(out[k], v)


def test_server_routes(tiny_model, monkeypatch):
    server.STATE["model"] = tiny_model
    client = TestClient(server.app)
    assert client.get("/health").json()["status"] == "ok"
    body = GenerateRequest(prompt="Alice met Bob", max_new_tokens=2,
                           record=RecordSpec(watch=["Bob"])).model_dump()
    for route in ("/generate", "/score", "/predict"):
        r = client.post(route, json=body)
        assert r.status_code == 200, r.text
        resp = GenerateResponse.model_validate(r.json())
        assert JSpaceTrace.from_response(resp).watch == ["Bob"]
    r = client.post("/predict", json={"instances": [body, body]})
    assert len(r.json()["predictions"]) == 2
    monkeypatch.setenv("JPEERS_API_KEY", "secret")
    assert client.post("/generate", json=body).status_code == 401
    assert client.post("/generate", json=body,
                       headers={"Authorization": "Bearer secret"}).status_code == 200


def test_merge_same_role():
    msgs = _merge_same_role([{"role": "system", "content": "s"},
                             {"role": "assistant", "content": "a"},
                             {"role": "user", "content": "u1"}, {"role": "user", "content": "u2"}])
    assert [m["role"] for m in msgs] == ["system", "user", "assistant", "user"]
    assert msgs[-1]["content"] == "u1\n\nu2"


def test_conversation_records_and_reloads(tiny_model, tmp_path):
    pool = EndpointPool()
    pool._live["tiny"] = LocalEndpoint(tiny_model, name="tiny")
    writer = RunWriter(tmp_path, "unit", run_id="r1")
    agents = [Agent("Alice", "tiny", "You are Alice.", max_new_tokens=3, watch=["Bob"]),
              Agent("Bob", "tiny", "You are Bob.", max_new_tokens=3, watch=["Alice"],
                    interventions=resolve("jspace_medium"))]
    conv = Conversation(agents, pool, writer=writer, tag="t")
    conv.say("Coordinator", "Plan the project.")
    conv.run(["Alice", "Bob"], rounds=2)
    assert len(conv.records) == 4
    assert "peer_last" in conv.traces[1].spans and "speaker:Alice" in conv.traces[1].spans
    turns = list(iter_turns(writer.dir))
    assert len(turns) == 4 and turns[1][0]["interventions"]
    assert turns[3][1].watch == ["Alice"]
    doc = Conversation(agents, pool, peer_framing="document")
    doc.say("Alice", "hi there")
    msgs, _ = doc.view_for(agents[1])
    assert msgs[-1]["content"].startswith("[Shared notes]")


def test_presets_resolve():
    for name in PRESETS:
        resolve(name, peer="Bob", self_name="Alice", words=["wrong"])
    sw = resolve("swap_peer_self", peer="Bob", self_name="Alice")[0]
    assert (sw.source, sw.target) == ("Alice", "Bob")
    assert resolve("suppress_words", words=["a", "b"])[0].tokens == ["a", "b"]


def test_analysis_on_real_trace(tiny_model):
    resp = tiny_model.generate(GenerateRequest(
        prompt="Alice and Bob discuss the quick brown fox", max_new_tokens=3,
        record=RecordSpec(watch=["fox", "Bob"], word_like_only=False)))
    tr = JSpaceTrace.from_response(resp)
    ranks = group_ranks(tr, ["fox", "Bob"])
    assert ranks.shape == (len(tr.positions),)
    s = rank_summary(ranks)
    assert 0 <= s["mrr"] <= 1
    bag = jspace_bag(tr)
    assert bag_similarity(bag, bag)["cosine"] > 0.999
    assert 0 <= content_uptake(tr, tr.mask(), "quick brown fox jumps")["uptake"] <= 1


def test_benchmarks_and_banks():
    items = false_belief_items(6, framing="people")
    fb = [i for i in items if not i.control]
    assert all(i.belief_answer != i.true_answer for i in fb)
    assert all(i.belief_answer == i.true_answer for i in items if i.control)
    agents = false_belief_items(4, framing="agents")
    assert all("Agent" in i.story for i in agents)
    for name in ("collab_tasks", "imposter_facts", "peer_selection"):
        assert load_task_bank(name)
