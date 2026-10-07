# 1. Setting up the Jacobian lens for a model

## What the lens is

The Jacobian lens (Gurnee, Sofroniew, Lindsey et al., *Verbalizable Representations
Form a Global Workspace in Language Models*, Anthropic 2026) reads what an internal
activation is disposed to make the model say. For a residual-stream vector `h` at
layer `l`:

```
J_l      = E_{prompt, t, t' >= t} [ d h_final,t' / d h_l,t ]     (averaged Jacobian)
lens_l(h) = softmax( W_U · norm( J_l h ) )                         (readout over the vocab)
v_{l,tok} = J_lᵀ (g ⊙ W_U[tok])                                   (J-lens vector of a token)
```

The **J-space** at a position is the small set of J-lens vectors that are most active
there (paper: sparse, k ≈ 25). The paper places the workspace at roughly 38–92% of depth
(early 38–58, late 62–92). Removing the J-space, steering along J-lens vectors and
swapping two of them all have causal effects on behaviour.

Reference code: [anthropics/jacobian-lens](https://github.com/anthropics/jacobian-lens)
(`jlens`, Apache-2.0). This repo depends on it (pinned) and adds generation with
recording and interventions on top.

## Pre-fitted lenses

Neuronpedia hosts lenses fitted on 1000 WikiText prompts at
[`neuronpedia/jacobian-lens`](https://huggingface.co/neuronpedia/jacobian-lens) for:
Qwen3 (1.7B–32B), Qwen3.5 (0.8B–27B), Qwen2.5-7B-Instruct, Llama-3.1-8B(-Instruct),
Llama-3.3-70B-Instruct, Gemma-2/3/4 (270M–31B, base and -it), OLMo-3 (7B, 32B),
gpt-oss-20b, GPT-2 and Pythia-70m. The registry (`src/jpeers/registry.py`) maps short
names to model + lens:

```bash
python -m jpeers.cli models
```

Lens files are `d_model² × n_layers` in fp16: 226 MB for Qwen3-1.7B, 1.2 GB for Qwen3-8B,
10.6 GB for Llama-3.3-70B. They are loaded to CPU, then each band layer's `J_l` is moved
to the GPU in fp32 the first time it is used.

## Local install

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/Scripts/python.exe torch --index-url https://download.pytorch.org/whl/cu128
uv pip install --python .venv/Scripts/python.exe -e ".[server,cloud,data,dev]"
```

Gated models (Llama, Gemma): accept the licence on the Hub and set `HF_TOKEN`.

## Using a model + lens in Python

```python
from jpeers.registry import get_model_spec
from jpeers.runtime import LensedModel
from jpeers.schemas import GenerateRequest, RecordSpec
from jpeers.trace import JSpaceTrace

m = LensedModel(get_model_spec("qwen3-8b"))           # weights + lens
resp = m.generate(GenerateRequest(
    messages=[{"role": "user", "content": "Bob says the Eiffel Tower is in Rome. Thoughts?"}],
    max_new_tokens=60,
    record=RecordSpec(watch=["Paris", "Rome", "wrong"], top_k=10)))
trace = JSpaceTrace.from_response(resp)
print(resp.text, trace.watch_ranks.min(axis=(0, 1)))  # best rank of each watch word
```

Or from the command line, through any endpoint:

```bash
python -m jpeers.cli probe qwen3-1.7b "Fact: The currency of the country shaped like a boot is the" --raw --last --watch Italy euro
```

## Checking a model before running experiments

```bash
python scripts/lens_sanity.py --model qwen3-8b
```

This prints, layer by layer, the rank of the model's final answer under the J-lens and
under the plain logit lens, and the rank of a multi-hop *bridge* entity (e.g. "Italy" for
"the currency of the boot-shaped country"). Bridges should surface in the workspace band
before the answer does. Measured on Qwen3-1.7B (local RTX 3060):

| layers (of 28) | J-lens rank of final answer | bridge rank |
|---|---|---|
| L10–L15 | 700–1700 | 600–1800 |
| L18–L21 | 11–55 | 43–1178 |
| L22–L25 | 0–2 | 35–98 |

Small models are weaker multi-hop reasoners (Qwen3-1.7B answered 2 of 5 prompts
correctly), so treat 1.7B as a development model and run the experiments themselves on
8B or larger. If the bridge peaks outside the default band for your model, set the band
with `get_model_spec(name, band=(lo, hi))` or `JPEERS_BAND=lo,hi` on the server.

## Implementation notes

* `g` (the final norm's gain) is folded into the J-lens vectors so a vector's dot
  product with `h` ranks tokens the same way the readout does. It handles LayerNorm,
  RMSNorm and Gemma's `1 + w` RMSNorm.
* A word maps to the first token of each of its surface forms (`" word"`, `"word"`,
  capitalised). Watch ranks take the best of these; directions use the first.
* Chat templates are used as-is (Qwen3 with `enable_thinking=False`). Gemma-2 rejects
  system messages, so the system prompt is folded into the first user turn.
