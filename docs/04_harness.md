# 4. Experimental harness: agents talking while their J-space is recorded

```
 ┌──────────── harness (your laptop / any CPU box) ─────────────┐
 │ Conversation(transcript)                                     │
 │   Agent "Alice" ──► Endpoint "qwen3-8b-azure"  ──HTTPS──► Azure ML endpoint ─┐
 │   Agent "Bob"   ──► Endpoint "llama3.1-8b-it-vertex" ──► Vertex AI endpoint ─┤ jpeers.server
 │   RunWriter ◄── text + J-space trace per turn                                │ (model + lens,
 └──────────────────────────────────────────────────────────────┘                │  hooks on band)
```

Each agent turn is **one stateless request**: the full chat history from that agent's
point of view, what to record, which interventions to apply, and which text spans to
label. The server renders the chat template, generates token by token with forward
hooks on the workspace layers, and returns the reply plus a compressed J-space trace.
Because requests are stateless, the agents of one conversation can live on different
clouds and model families.

## Main objects

| object | file | role |
|---|---|---|
| `GenerateRequest` / `GenerateResponse` | `schemas.py` | wire format (same for every backend) |
| `RecordSpec` | `schemas.py` | layers, positions, top-k, watch words, pre/post-intervention readout, J-lens or logit-lens baseline |
| `LensedModel.generate` | `runtime.py` | generation loop, hooks, recording, interventions |
| `Endpoint` (`local`, `http`, `azureml`, `vertex`) | `client.py` | where a model runs |
| `Agent`, `Conversation` | `harness.py` | agents, shared transcript, per-agent views |
| `RunWriter`, `iter_turns` | `harness.py` | `runs/<exp>/<run>/turns.jsonl` + `traces/*.npz` |
| `JSpaceTrace` | `trace.py` | arrays: `topk_ids[P,L,K]`, `topk_logits`, `watch_ranks[P,L,W]`, `positions` |
| metrics | `analysis.py` | ranks, MRR, hits, J-space bags, uptake, similarity, permutation tests |

## Two agents in ten lines

```python
from jpeers.client import EndpointPool, load_endpoints
from jpeers.harness import Agent, Conversation, RunWriter

pool = EndpointPool(load_endpoints("configs/endpoints.yaml"))   # unknown names load locally
alice = Agent("Alice", "qwen3-8b-azure", "You are Alice, a researcher working with Bob.",
              watch=["Bob", "team", "together"])
bob = Agent("Bob", "llama3.1-8b-it-vertex", "You are Bob, a developer working with Alice.",
            watch=["Alice", "team", "together"])
conv = Conversation([alice, bob], pool, writer=RunWriter("runs", "demo"))
conv.say("Coordinator", "Design a rate limiter for our API.")
conv.run(["Alice", "Bob"], rounds=3)
trace = conv.traces[-1]                 # Bob's last turn
print(trace.watch_ranks[trace.mask(generated=True)].min(axis=(0, 1)))
```

## How agents see each other

* An agent's own turns are `assistant` messages. Other speakers' turns are `user` messages
  of the form `[Name]: text`. Consecutive same-role messages are merged, because strict
  templates (Gemma, Llama) need alternating roles.
* `peer_framing="document"` shows peers as anonymous `[Shared notes]`. This is the
  matched no-peer control: same content, no social framing.
* `say(..., private_to=[...])` makes a message visible to only some agents (private
  channels, as in NARCBench).

## Span labels on every turn

`speaker:<name>` (each other speaker's latest message), `peer_last` (the most recent
peer message), `self`, `msg:<i>` / `role:<r>` (rendered chat messages) and `prompt`.
Spans are located on the server by searching for the message text in the rendered
prompt and mapping character offsets to tokens. Metrics (`trace.mask("peer_last")`) and
interventions (`PositionSelector(scope="spans", labels=["peer_last"])`) both use them.

## What is recorded

For every recorded (position, layer): the top-k J-lens tokens and their logits, and the
**full-vocabulary rank** of every watch word (best over its surface forms). Defaults:
every band layer, every position after the first 4 (attention sinks), k = 10, word-like
tokens only for the top-k display. Ranks are computed over the whole vocabulary,
unfiltered. By default every watch rank also comes with two baselines at the same
positions: `watch_ranks_logit` (logit lens) and `output_ranks` (the model's own
next-token prediction). Set `RecordSpec(baselines=[])` to skip them,
`RecordSpec(when="pre")` to read J-space before this layer's interventions, or
`lens="logit"` to make the logit lens the main readout.

Trace size: about `P × L × (2K + W)` numbers per turn, sent as compressed `.npz` (base64).
A 1500-token prompt with 16 band layers and K = 10 is roughly 0.3–0.6 MB. Raise `stride`
or narrow `positions` for very long contexts.

## Throughput

Recording computes `unembed(J_l h)` over the full vocabulary for every recorded
(position, layer), so it dominates cost. Measured on a laptop RTX 3060 with Qwen3-1.7B
(16 band layers): about 3.5 s prefill for 70 tokens and about 0.15 s per generated token.
On an A100 expect roughly 10–20× faster. To cut cost: use `stride: 2`, record only
`spans` and `generated`, and lower `max_positions`.
