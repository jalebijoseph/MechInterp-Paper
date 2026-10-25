# Do SAE Features Mean the Same Thing Across Prompts?

**Author:** Anjali Joseph

## Goal
Test whether SAE features that seem to represent a concept
(deception, uncertainty, math reasoning) fire consistently on new,
unseen prompts — or only on the examples they were found with.

## Research Question
Do features that appear to represent the same concept behave
consistently across different prompts?

## Hypothesis
Many features will look clean on the prompts used to find them but
will be less reliable on paraphrases, new formats, and tricky
negatives (e.g., "lie down" vs. "tell a lie").

## Setup
- Python 3.11, PyTorch
- Libraries: transformer_lens, sae_lens, numpy, pandas, scikit-learn, matplotlib
- Model: gemma-2-2b (small enough for one GPU)
- SAEs: Gemma Scope, residual stream, a few layers (e.g., 6, 12, 18), 16k width
- Optional: Neuronpedia labels to cross-check features

## Repo Structure
- `data/` — prompt sets (JSON), one file per concept
- `src/load.py` — load model + SAEs
- `src/activations.py` — run prompts, cache SAE feature activations
- `src/find_features.py` — find candidate features per concept
- `src/evaluate.py` — consistency metrics on held-out sets
- `src/steer.py` — causal tests (ablate / boost a feature)
- `notebooks/` — plots and analysis
- `results/` — saved metrics, figures, report.md

## Phase 1: Build Prompt Sets
For each concept, write ~50 positive and ~50 negative prompts.
- Deception: lying, hiding info vs. honest statements
- Uncertainty: hedging ("I think", "maybe") vs. confident claims
- Math: step-by-step reasoning vs. plain text with numbers

Split into: discovery set (to find features) and held-out set (to test).

**Status:** started small per the Claude Code instructions below — only
`deception` exists so far, with a 20-prompt discovery set and a small
`same_style` held-out set, just enough to exercise the full pipeline.
The other concepts and held-out categories (paraphrase, new formats,
tricky negatives, mixed prompts) come once Phase 1→6 work end to end.

## Phase 2: Find Candidate Features
- Run discovery prompts, record SAE activations per token
- For each feature: mean activation (positive) minus mean (negative)
- Keep top 10 features per concept per layer
- Save feature IDs + their top-activating examples

## Phase 3: Test on New Examples
Held-out test types:
- Same style, new prompts
- Paraphrases of discovery prompts
- New formats: dialogue, story, code comments, another language
- Tricky negatives: same words, different meaning
- Mixed prompts: two concepts at once

## Phase 4: Measure Consistency
For each feature, report:
- AUROC: does it separate concept vs. non-concept?
- Precision / recall at a fixed threshold
- Drop in score from discovery set → each held-out type
- Which tokens it fires on (is it the concept, or just a keyword?)
- Whether the same concept maps to similar features across layers

## Phase 5: Causal Check
- Ablate (zero out) top features → does concept-related output change?
- Boost (steer) features → does the model show more of the concept?
- Compare against random features as a baseline

## Phase 6: Analysis + Report
- Table: features ranked by consistency
- Plots: score drop by test type, per concept
- Failure cases: where features misfire and why
- Note known SAE issues: feature splitting, feature absorption
- Write `results/report.md` with findings + limitations

## Instructions for Claude Code
- Start small: 1 concept, 1 layer, 20 prompts — get it working end to end
- Cache activations to disk so runs aren't repeated
- Set random seeds; log all configs
- Write tests for activation caching and metric functions
- Ask before downloading large models or SAEs

## Current status

- **Phase 1:** only `deception` exists so far (`data/deception.json`):
  20 positive + 20 negative discovery prompts and a small `same_style`
  held-out set, enough to exercise the full pipeline.
- **Pipeline:** `src/` implements load → activations (cached to disk) →
  feature ranking → AUROC / precision / recall on held-out sets →
  steering/ablation helpers. `src/run_pipeline.py` runs the whole thing
  for one concept at one layer and writes
  `results/<concept>_layer<layer>_summary.json`.
- **Tests:** `pytest tests/` (15 tests) covers activation caching,
  feature ranking and the consistency metrics, with no model downloads.
- **First end-to-end run** (deception, layer 6) is in progress on a
  CPU-only laptop. An earlier attempt was silently killed for running out
  of memory while loading the model; see "Running it" below for the fix.

## Running it

### Locally (CPU)
```
python -m venv .venv
.venv\Scriptsctivate          # Windows; source .venv/bin/activate elsewhere
pip install -r requirements.txt
huggingface-cli login            # needs access to the gated google/gemma-2-2b repo
pytest tests/
python -m src.run_pipeline --concept deception --layer 6
```

Memory notes (16 GB machine, no GPU):
- The model loads in bfloat16 on CPU (~5 GB instead of ~10 GB in float32).
- It uses `HookedTransformer.from_pretrained_no_processing`. The default
  weight processing (fold_ln, center_writing_weights, ...) copies every
  weight, which pushed peak RAM past the limit. It also changes the
  residual stream, but Gemma Scope SAEs were trained on the unprocessed
  residual stream, so skipping it is also the more correct choice.
- Close browsers/editors before running if free RAM is low.

### On a cloud GPU (Azure, Colab, etc.)
The code picks CUDA automatically when it's available and uses float32
there (~10–11 GB for gemma-2-2b plus a few hundred MB for a 16k SAE).
A 16 GB GPU such as a T4 is enough; Colab's free T4 works for this stage.
1. Install a CUDA build of PyTorch first (see pytorch.org), then
   `pip install -r requirements.txt`.
2. `huggingface-cli login` with a token that has Gemma access.
3. Run the same commands as above.

## Next steps
1. Finish the deception / layer 6 run and check the summary JSON.
2. Expand the held-out sets: paraphrases, new formats (dialogue, story,
   code comments, another language), tricky negatives, mixed prompts.
3. Add the `uncertainty` and `math` prompt sets.
4. Run layers 12 and 18 and compare features across layers.
5. Causal checks with `src/steer.py` (ablate/boost vs. a random-feature
   baseline), then write `results/report.md`.
