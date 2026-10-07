# 6. Removing and editing J-space during generation

Interventions are applied **inside the forward pass**, at the workspace layers, on the
positions you select. This covers the prompt (so the KV cache holds edited states) and
every generated token. The same `InterventionSpec` works locally and on every cloud
endpoint, because it travels in the request.

## Kinds

| kind | operation on residual `h` at each selected (layer, position) | paper analogue |
|---|---|---|
| `ablate_topk` (k) | find the k J-lens vectors with the highest lens logits at this position; project `h` onto their orthogonal complement | "deleting the J-space ... removing its most active contents" (k = 10) |
| `scale_jspace` (k, strength) | scale the component inside that top-k subspace by `strength` (0 = ablate, 0.5 = halve, 2 = amplify) | graded version of the above |
| `ablate_tokens` (tokens) | project out the J-lens vectors of specific words (e.g. the peer's name, "suspicious") | targeted ablation |
| `random_ablate` (k, seed) | project out k fixed random directions | control for `ablate_topk` |
| `steer` (tokens, strength) | `h += strength · median‖h‖ · v̂_tok` (the median is over this layer's prompt positions) | "simple steering" `h ← h + α v_t` |
| `swap` (source, target) | coordinate patch: with `V = [v_s v_t]`, `c = V⁺h`, `h += V(swap(c) − c)` | "removed the Soccer pattern and added Rugby" |

`layers`: `"band"` (default), `"early"`, `"late"`, `"all"` or an explicit list.
`positions`: `all`, `prompt`, `generated`, `last`, or `spans` with `labels` (e.g.
`["peer_last"]`, `["speaker:Bob"]`), plus `skip_first`.

## Using them

Per request:

```python
from jpeers.schemas import InterventionSpec, PositionSelector
iv = InterventionSpec(kind="ablate_topk", k=10,
                      positions=PositionSelector(scope="spans", labels=["peer_last"]))
resp = endpoint.generate(GenerateRequest(messages=..., interventions=[iv]))
```

Per agent, for a whole conversation:

```python
from jpeers.interventions import resolve
alice = Agent("Alice", "qwen3-8b-azure", sys_prompt,
              interventions=resolve("jspace_medium"))           # a named preset
bob = Agent("Bob", ..., interventions=resolve("swap_peer_self", peer="Alice", self_name="Bob"))
```

Per turn: `conv.step("Alice", interventions=[...])`.

Per experiment (Exp. 4): `interventions: [none, jspace_medium, random_k10, ...]` in the
YAML. Every condition is then run on the same items, and `summary.md` reports each
metric's change against `none`.

## Presets (`jpeers.interventions.PRESETS`)

`none`, `jspace_light|medium|heavy` (k = 5/10/25), `random_k5|k10|k25`,
`jspace_peer_span`, `jspace_generated`, `jspace_early`, `jspace_late`, `halve_jspace`,
`double_jspace`, `suppress_peer`, `suppress_words`, `steer_suspicion`, `steer_trust`,
`steer_teamwork`, `swap_peer_self`. Placeholders `{peer}`, `{self}` and `{words}` are
filled per agent.

## Reading out while editing

`RecordSpec.when="post"` (default) records what flows downstream after the edit.
`"pre"` records the layer's J-space before its own edit. For example, with
`ablate_topk`, `pre` shows what was removed and `post` shows what remains. Every
response reports `interventions_applied`, the number of (layer, forward-step) edits that
fired.

## Strength guidance

Steering is relative to each layer's median residual norm and is added at **every** band
layer, so the effect compounds across layers. Measured on Qwen3-1.7B (16 band layers,
all positions), steering toward "suspicious"/"wrong" on a reviewer reading "The capital
of Australia is Sydney":

| strength | reply |
|---|---|
| 0 | "Your draft is accurate. The capital of Australia is indeed Sydney." |
| 0.01 | "I think the capital of Australia is Canberra. Let me check. Yes ... the capital is Canberra." |
| 0.02 | "[wrong] the capital of Australia is not Sydney. it is not. it is not. ..." |
| ≥ 0.04 | "wrong, wrong, wrong ..." (collapse) |

The presets therefore use 0.01. Re-sweep per model, since the right value scales with
the band width and the model. Restrict `layers` or `positions` to use larger values.
Ablation rank: k = 10 is the paper's default. Always compare against `random_k*` of the
same rank: removing any 10 directions perturbs the model somewhat. Text stays fluent at
k = 10 and is visibly degraded at k = 25 on 1.7B.

## Correctness checks (tests/test_runtime.py)

* After `ablate_topk`, the edited state has zero projection on the removed J-lens vectors.
* `random_ablate` is an idempotent projector.
* `steer` raises the target token's lens logit.
* `swap` exchanges the two coordinates exactly.
* A span-scoped edit leaves earlier positions bit-identical (causality) and changes the
  span.
