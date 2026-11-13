# Causal decoder and readout audit

## Assessment

**Updated November 12: the actual development-trace reanalysis is complete.** All 18 original decoder fits reproduced, and stable solvers changed no predictions. A more general full-residual linear probe improves incoming decoding to **79.3%**, while probes restricted to the two-label J span remain at **49.2–53.1%**. Neither passes the unchanged 80% gate.

The current `residual_contrast` baseline does not test general linear decodability: it learns one factorial mean-difference direction and calibrates a scalar predictor on that coordinate. The same restriction is applied inside the two-label Jacobian and plain-logit spans. A reproducible counterexample scores 50% with the current implementation despite perfect linear recovery on the same development folds. This limitation explains why the general full-residual diagnostic is stronger, but changing the estimator alone does not rescue the recorded J-span search.

No additional runtime indexing, matrix-transposition, token-form, or capture-storage bug was demonstrated. The saved engineering evidence agrees with the intended setup. The chosen punctuation position and two-label span remain restrictive discovery choices whose functional relevance has not been established.

The current finding is therefore **failure of the declared localized feature search**. It does not measure J-space's causal necessity and does not establish that incoming information is absent from the captured residuals.

**Evidence availability:** The initial November 8 audit was limited by SSH timeouts. With access restored on November 12, all 256 development receipts and states were checked, all original fits were reproduced, and covariance-aware diagnostic probes were run on the same eight development-template folds. The complete attempt-004 output export now preserves all **289 receipts and 289 traces**, with per-file checksums. No alternate-position model generation or held-out evaluation was performed. See the [actual-trace audit](../results/qwen3-32b-causal-handoff-20261107/decoder-audit-20261112/actual-trace-audit.json) and [independent verification](../results/qwen3-32b-causal-handoff-20261107/decoder-audit-20261112/independent-verification.json).

## What actually stopped discovery

Attempt 004 completed 32 controls, one deterministic replay, and all 256 development baselines. Controls, replay, incoming-label accuracy, own-label accuracy, and normalized JSON validity passed. Every candidate incoming decoder was below the required 0.80 cross-validation accuracy.

| Layer index | Jacobian incoming successes / 256 | Incoming accuracy |
|---|---:|---:|
| 40 | 129 | 50.39% |
| 42 | 129 | 50.39% |
| 48 | 125 | 48.83% |
| 50 | 139 | 54.30% |
| 52 | 128 | 50.00% |
| 54 | 133 | 51.95% |

Passing requires at least 205/256. Thus all deletion/rescue pairs (40/42, 48/50, 52/54) fail before either dose is tried. Layer 54 also exceeds the nuisance-leakage ceiling: 161/256 = 62.89%, versus a maximum of 60%. The best incoming decoder across all three families is the layer-48 residual contrast at 62.89%.

The empty `settings` list is consistent with the source's gate logic. No deletion, random control, donor rescue, frozen selection, or held-out generation occurred. See the [original discovery report](../results/qwen3-32b-causal-handoff-20261107/attempt-004/results/discovery.json). These scores and thresholds were not changed by this audit.

## Confirmed estimator limitation

In `handoff._learn`, the design contains an intercept, own value, source, name pair, order, and incoming value. For these fully crossed tasks, each development training fold has 224 rows and satisfies `A.T @ A = 224 I`. Consequently, its incoming encoding coefficient is a scaled class mean difference. Projecting this coefficient into a lens span and normalizing it still produces a mean-difference direction. The subsequent scalar predictor cannot choose a different linear combination to cancel correlated variation.

This is a valid, deliberately restricted estimator. The name `residual_contrast` describes it correctly. Treating it as a general test for linearly decodable residual information would overstate what it measures.

### Reproduced false negative

Use the same 256 development rows and eight leave-template-out folds. Let `y` be the incoming label sign and `q = source_sign * order_sign`. Construct two residual coordinates:

```text
x1 = y + 10 q
x2 = 10 q
```

The incoming label is exactly `x1 - x2`. Both coordinates are available inside an identity two-dimensional lens span; no information is hidden outside the permitted span.

| Method on these synthetic states | Same-fold incoming accuracy |
|---|---:|
| Current `fit_component`, basis `[1, 0]` | 50% |
| Training-only discriminative ridge, fixed alpha 1 | 100% |
| Explicit subtraction `x1 - x2` | 100% |

The current own/nuisance leakage scores are also 50%. The nuisance adjustment includes main effects, so it does not remove the source-by-order interaction. A different linear combination cancels that variation and recovers the label. This proves a possible false negative in the discovery procedure; it does **not** establish that this interaction or covariance pattern explains the actual Qwen traces.

The existing decoder test uses perfectly separated, orthogonal synthetic coordinates. It verifies basic extraction and disjoint folds but does not test correlated nuisance variation or estimator adequacy.

Reproduce without a GPU or model download:

```bash
CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 PYTHONPATH=src:scripts \
  python results/qwen3-32b-causal-handoff-20261107/decoder-audit-20261108/reproduce_decoder_false_negative.py \
  --output /tmp/decoder-false-negative.json
```

The [saved reproduction](../results/qwen3-32b-causal-handoff-20261107/decoder-audit-20261108/decoder-false-negative.json) records the source hash, fold sizes, and fitted coefficients. It uses no held-out templates and no threshold or hyperparameter search.

## Actual development-trace reanalysis, November 12

All 256 development receipts pass request/receipt checksums, prompt/episode/manifest alignment, trace-versus-receipt array equality, label checks, and capture-site validation. The original-host replay reproduces all 18 complete bases and all 144 fold-basis hashes exactly. A separate local replay reproduces every original metric; its maximum basis difference across host libraries is `2.1e-15`.

The diagnostic classifier is a train-centered linear ridge probe with fixed alpha 1. No feature-variance rescaling, hyperparameter search, threshold tuning, or held-out templates are used. Centering and nuisance fits use training rows only.

| Available coordinates | Original contrast CV range | Raw fixed-alpha probe CV range | Nuisance-adjusted probe CV range |
|---|---:|---:|---:|
| Two-label Jacobian span | 48.8–54.3% | 49.2–53.1% | 50.0–53.1% |
| Two-label plain-logit span | 50.8–55.9% | 50.0–53.5% | 51.2–56.6% |
| Full residual | 52.3–62.9% | 71.5–79.3% | 73.0–78.5% |

The best raw full-residual probe is layer 50: **203/256**, below the **205/256** required by the 80% feasibility gate. Its raw own-label leakage is 49.6%, and its maximum raw nuisance leakage is 53.5%. The best nuisance-adjusted full-residual probe reaches 78.5% at layer 52 but has raw own leakage 62.9% and order leakage 71.5%, also failing the 60% leakage ceiling.

These are diagnostic probe comparisons, not replacements for the frozen discovery procedure or causal results. The full-residual probe uses a larger coordinate space than either two-dimensional lens probe. Its improvement establishes sensitivity to the estimator choice in the full-state baseline; it does not establish a J-specific feature or justify lowering the gate. A different readout site, broader J feature family, or further training-only probe validation remains a separate development experiment.

## Readout and runtime checks

**Source and inventory:** The initial audit matched all 112 launch source pins and 112 preflight pins to the then-current files. The actual-trace replay uses the frozen decoder source, whose hash remains `938cfca1d2184eaf1ef76a38269a36d7c5421c7f7f92d2fbbb7a285a937aba3a`. The regenerated 1,056-row inventory matches its saved hash: 256 development contexts, 768 held-out contexts, and 32 controls. Development has eight complete 32-cell factorial templates, balanced binary factors, and all four own/incoming combinations. All 144 saved fold records use the expected one validation template, 32 validation rows, and seven training templates. No label/order alignment defect was found in source or the actual development receipts.

**Archived actual traces:** All 32 attempt-003 controls pass receipt/request checksums, episode/prompt checks, turn projection, NPZ array equality, layer alignment, and capture-site validation. Each layer has 32 distinct residual states. Captures are finite `[1,6,5120]` arrays; their float32 storage exactly represents the delivered BF16 values. Directions are fixed across contexts and near unit length. The two-label J spans have rank two, with smaller singular values from 0.383 to 0.700. They have not collapsed to a zero or duplicate direction. The later actual-trace replay additionally verifies all 256 attempt-004 development contexts and fixed direction matrices.

**Position:** The recorded site is exactly the requested final `.` in `End of notes.`, after both notes. Nine chat-template tokens follow it. For the first control, this is position 134; the final prompt position is 143 and generation starts at 144. The recording-window repair is functioning as intended. However, correct placement at a declared position does not establish that this position contains the incoming value in a generalizable form or is a causal bottleneck for the later JSON answer. The run records no alternative pre-answer positions, so it cannot distinguish a weak estimator from an unsuitable site.

**Token forms:** Bare token IDs 1151/12203 correspond to `red`/`blue` in the generated JSON values. IDs 2518/6303 are their leading-space variants, which appear in the payload notes. Requesting bare output-label directions is intentional and matches the observed generated labels. Changing to leading-space directions is a possible development diagnostic, not a demonstrated correction to a token-ID bug.

**Layer and transport conventions:** Local capture uses block outputs, and row transport is `h @ J.T`. Decoder numerator rows use `(gain * W_U[token]) @ J`. These agree with the pinned reference's [block-output hooks](https://github.com/anthropics/jacobian-lens/blob/581d398613e5602a5af361e1c34d3a92ea82ba8e/jlens/hooks.py#L14-L18) and [transport orientation](https://github.com/anthropics/jacobian-lens/blob/581d398613e5602a5af361e1c34d3a92ea82ba8e/jlens/lens.py#L124-L132). Layer index 40 denotes the output of the 41st block; no one-block offset was demonstrated.

**Fixed directions and normalization:** The captured rows are fixed, normalized decoder-numerator directions. They are not local gradients through final normalization, as `runtime.lens_directions` documents. The upstream [unembedding](https://github.com/anthropics/jacobian-lens/blob/581d398613e5602a5af361e1c34d3a92ea82ba8e/jlens/hf.py#L149-L154) includes the model-dtype cast and final normalization. A separate synthetic reproduction confirms numerator equivalence while showing that moving along a fixed direction need not locally increase a normalized logit. The runtime module introduction's “most raises” description is too strong. This wording issue does not explain discovery failure, which occurred before interventions.

**Search scope:** The Jacobian search tests the span of two label rows, one punctuation position, and six recorded layers. This is substantially narrower than the [reference's vocabulary-frame description of J-space](https://transformer-circuits.pub/2026/workspace/index.html). A failure in this search cannot establish that all incoming representations or all relevant J-space content are unavailable. The ordinary output readout at this punctuation position also concerns the next prompt/chat token, not later JSON answer accuracy.

## Numerical assessment

The encoding design's condition number is exactly one in each original fold. On the actual traces, augmented least-squares and exact-centered scalar calculations change **zero predictions**; maximum stable-basis differences are about `2.9e-15`. The unpenalized-centered diagnostic also changes no reported accuracy. Numerical instability is therefore not an explanation for this recorded discovery failure.

Scalar calibration does penalize its intercept. Simply centering the scalar and leaving the intercept unpenalized changes the estimator; it is not an algebraically identical numerical repair. Synthetic extreme offsets can reveal translation sensitivity of that penalty.

The preliminary control-only projections had absolute means at most about 105 and standard deviations at least about 0.12. Those checks have now been supplemented by the exact development-fold comparisons above. Actual raw, adjusted, and own-coordinate scalar Gram condition maxima are approximately `4.97e5`, `653.5`, and `6.58e6`, respectively; stable predictions still match.

The separate `scalar-penalty-diagnostic.json` artifact uses `_cv` in its field names for synthetic resubstitution accuracies: the same toy points are used for fitting and prediction. Those values are not template cross-validation results. The dedicated false-negative reproduction above does use all eight original development folds.

## Next diagnostic, before another causal run

1. **Completed:** Recover the 256 development traces and reproduce all 18 original fits with verified key, label, and layer alignment.
2. **Completed:** Compare fixed-alpha train-only probes in the same J/logit spans and full residual coordinates. No diagnostic probe passes 80%; the narrow J span remains weak. Additional probe scaling/tuning or permutation validation would be new training-fold-only diagnostics, not results already obtained here.
3. Before another full causal run, compare a small, declared set of pre-answer positions and layers using development data only, with known-label positive controls. Later attention to the original payload remains a possible route; the punctuation anchor is not an established summary bottleneck. If a useful readout emerges, freeze its decoder and site before new intervention/held-out work.
4. Retain the current thresholds and keep held-out generations closed. A revised decoder or site requires a new attributed run, not rewriting attempt 004 or treating its discovery failure as an intervention result.

The necessary next step is diagnostic validation of the decoder/readout setup. Another identical GPU run would not resolve the estimator or site questions.

## Audit artifacts

The [initial audit evidence](../results/qwen3-32b-causal-handoff-20261107/decoder-audit-20261108/) contains the synthetic reproduction, partial integrity report, archived-control readout checks, numerical diagnostics, and SHA-256 inventory. The [actual-trace audit evidence](../results/qwen3-32b-causal-handoff-20261107/decoder-audit-20261112/) contains the CPU driver, complete diagnostic report, compact development-state dataset, and independent verification. Production code, model weights, scoring rules, original results, and scientific gates were not modified, and no model-serving process was started for either audit.
