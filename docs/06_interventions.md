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
| `random_matched` (k, seed) | at each position, project out one random direction holding exactly as much of `h` as the top-k J-space does: same edit size, same final length | size-matched control for `ablate_topk` (Exp. 4 default) |
| `random_ablate` (k, seed) | project out k fixed random directions | weak control: removes far less than `ablate_topk` |
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

Per experiment (Exp. 4): `interventions: [none, jspace_medium, matched_k10, ...]` in the
YAML. Every condition is then run on the same items, and `summary.md` reports each
metric's change against `none`.

## Presets (`jpeers.interventions.PRESETS`)

`none`, `jspace_light|medium|heavy` (k = 5/10/25), `matched_k5|k10|k25` (size-matched
controls), `random_k5|k10|k25` (weak controls), `jspace_peer_span`, `jspace_generated`,
`jspace_early`, `jspace_late`, `halve_jspace`, `double_jspace`, `suppress_peer`,
`suppress_words`, `steer_suspicion`, `steer_trust`, `steer_teamwork`, `swap_peer_self`.
Placeholders `{peer}`, `{self}` and `{words}` are filled per agent.

Steering words (`interventions.STEER_WORDS`: "suspicion"/"dishonest",
"trustworthy"/"reliable", "cooperation"/"joint") are never measured words. Steering
toward a word and then counting that same word in J-space shows an increase by
construction, so steering is judged by behaviour, and a rise in the measured lexicon is
only a check that the edit landed. `suppress_words` has the same issue in reverse: it
removes the measured words, so read its effect from behaviour too.

## Reading out while editing

`RecordSpec.when="post"` (default) records what flows downstream after the edit.
`"pre"` records the layer's J-space before its own edit. For example, with
`ablate_topk`, `pre` shows what was removed and `post` shows what remains. Every
response reports `interventions_applied`, the number of (layer, forward-step) edits that
fired.

## Strength guidance

Steering is relative to each layer's median residual norm and is added at **every** band
layer, so the effect compounds across layers. Pick strengths with
`scripts/steer_sweep.py`. It runs the Exp. 2 reviewer on scripted true and false drafts
and keeps the largest strength whose replies stay fluent (distinct word pairs) **and**
do not just echo the peer's draft. Fluency alone is not enough: at 0.01, suspicion
steering produces perfectly fluent text that repeats Bob's draft verbatim.

Qwen3-1.7B (16 band layers, all positions, 3 facts × true/false):

| preset | 0.005 | 0.01 | 0.015 | ≥ 0.02 | chosen |
|---|---|---|---|---|---|
| suspicion | fluent review | echoes the draft (44% copied) | echoes (71%) | "I am not a dishonest person" loop | **0.005** |
| trust | fluent | fluent, uses "reliable"/"trustworthy" | loops | collapse | **0.01** |
| teamwork | fluent | fluent | "The joint fact sheet is ready" loop | "joint joint joint" | **0.01** |

The usable range is narrow (a factor of 2–3 between no effect and collapse), and at
the chosen strengths the catch rate does not change (n = 6, so this is a calibration,
not a result). Re-sweep per model: the right value scales with the band width and the
model. Restrict `layers` or `positions` to use larger values.

Ablation rank: k = 10 is the paper's default. Always compare against `matched_k*` of the
same k. Projecting out k random directions (`random_k*`) removes only about k/d of the
state, while the top-k J-space directions are chosen because they align with it, so an
unmatched control makes any J-space ablation look special. Text stays fluent at k = 10
and is visibly degraded at k = 25 on 1.7B.

## Correctness checks (tests/test_runtime.py)

* After `ablate_topk`, the edited state has zero projection on the removed J-lens vectors.
* `random_ablate` is an idempotent projector.
* `steer` raises the target token's lens logit.
* `swap` exchanges the two coordinates exactly.
* A span-scoped edit leaves earlier positions bit-identical (causality) and changes the
  span.
