# Qwen3-32B Experiment 4 specificity analysis

This is an exploratory analysis of 4,680 frozen subject responses graded independently by Nemotron and GPT-OSS. Each grid contains 60 facts, three repeated source framings, true and false peer claims, and 13 interventions. Every review remains in the rate denominators. No response or grader label was changed.

Both graders show substantially more correctness harm from the generated-token J-lens edit on false claims than on true claims. The all-fact false-minus-true bounds exclude zero even under adverse assignments of unresolved outcomes. False claims also show harm when the draft is presented as a document. The peer-versus-document comparisons have wide fact-bootstrap intervals and do not establish a larger effect specific to peer interaction.

The focal edit removes the ten most active J-lens directions at selected layers while generating. It tests this intervention and task. The model still contains other J-lens directions, and the peer-reading positions are outside this focal edit.

## Matched-control effects

Differences are J/generated minus its size- and position-matched random control, in percentage points. Negative correctness effects mean fewer correct final documents. The interaction is the false-claim effect minus the true-claim effect. A negative interaction means false claims show greater harm. Each fact contributes the mean of its three source differences. The fact bank supplies 60 clusters.

The point estimate and p value use complete fact clusters. Bounds use all 60 facts and every source pair, assigning each unresolved outcome its most adverse or most favorable binary value. They are identification bounds, separate from sampling uncertainty. All tests are two-sided sign flips under symmetry of fact differences and are unadjusted exploratory p values. Exact dynamic programming counts all sign assignments. The JSON also provides seeded percentile bootstrap intervals over complete fact clusters.

The complete-cluster estimate can fall outside the all-fact bounds because these quantities summarize different fact sets. Full-bank effect magnitude should be read from the bounds. The JSON also records complete source-pair estimates for comparison with the original grading summaries.

| Grader | Claim | Metric | Complete-fact effect, pp | All-fact bounds, pp | Facts | p |
|---|---|---|---:|---:|---:|---:|
| Nemotron | false | semantic_final_correct | -39.9 | [-41.1, -38.9] | 56/60 | 3.64e-12 |
| Nemotron | false | usable | -11.1 | [-13.3, -11.7] | 57/60 | 6.87e-05 |
| Nemotron | true | semantic_final_correct | -5.7 | [-6.1, -5.0] | 58/60 | 0.0405 |
| Nemotron | true | usable | -3.3 | [-3.3, -3.3] | 60/60 | 0.109 |
| Nemotron | false_minus_true | semantic_final_correct | -33.3 | [-36.1, -32.8] | 54/60 | 5.07e-07 |
| Nemotron | false_minus_true | usable | -8.2 | [-10.0, -8.3] | 57/60 | 0.0234 |
| GPT-OSS | false | semantic_final_correct | -40.5 | [-42.8, -40.6] | 56/60 | 1.82e-12 |
| GPT-OSS | false | usable | -11.9 | [-12.8, -12.2] | 59/60 | 6.1e-05 |
| GPT-OSS | true | semantic_final_correct | -4.0 | [-5.0, -3.9] | 58/60 | 0.191 |
| GPT-OSS | true | usable | -3.3 | [-3.3, -3.3] | 60/60 | 0.109 |
| GPT-OSS | false_minus_true | semantic_final_correct | -35.2 | [-38.9, -35.6] | 54/60 | 9.26e-08 |
| GPT-OSS | false_minus_true | usable | -8.5 | [-9.4, -8.9] | 59/60 | 0.019 |

## Source-specific correctness

Each source analysis still has 60 fact units. Source rows are repeated measurements of the same facts, so the combined analysis never treats 180 pairs as 180 independent facts.

| Grader | Source | Claim | Complete-fact effect, pp | All-fact bounds, pp | Facts | p |
|---|---|---|---:|---:|---:|---:|
| Nemotron | incentive | false | -43.9 | [-46.7, -41.7] | 57/60 | 4.17e-07 |
| Nemotron | incentive | true | -1.7 | [-3.3, -1.7] | 59/60 | 1 |
| Nemotron | incentive | false_minus_true | -41.1 | [-45.0, -38.3] | 56/60 | 5.65e-06 |
| Nemotron | no_incentive | false | -40.7 | [-41.7, -40.0] | 59/60 | 1.19e-07 |
| Nemotron | no_incentive | true | -10.0 | [-10.0, -10.0] | 60/60 | 0.146 |
| Nemotron | no_incentive | false_minus_true | -30.5 | [-31.7, -30.0] | 59/60 | 0.00195 |
| Nemotron | document | false | -35.0 | [-35.0, -35.0] | 60/60 | 5.72e-06 |
| Nemotron | document | true | -5.1 | [-5.0, -3.3] | 59/60 | 0.25 |
| Nemotron | document | false_minus_true | -30.5 | [-31.7, -30.0] | 59/60 | 0.000121 |
| GPT-OSS | incentive | false | -45.0 | [-45.0, -45.0] | 60/60 | 1.12e-07 |
| GPT-OSS | incentive | true | -1.7 | [-3.3, -1.7] | 59/60 | 1 |
| GPT-OSS | incentive | false_minus_true | -42.4 | [-43.3, -41.7] | 59/60 | 1.62e-06 |
| GPT-OSS | no_incentive | false | -43.1 | [-45.0, -41.7] | 58/60 | 5.96e-08 |
| GPT-OSS | no_incentive | true | -6.8 | [-8.3, -6.7] | 59/60 | 0.388 |
| GPT-OSS | no_incentive | false_minus_true | -36.8 | [-38.3, -33.3] | 57/60 | 0.000158 |
| GPT-OSS | document | false | -36.2 | [-38.3, -35.0] | 58/60 | 9.54e-07 |
| GPT-OSS | document | true | -3.3 | [-3.3, -3.3] | 60/60 | 0.625 |
| GPT-OSS | document | false_minus_true | -32.8 | [-35.0, -31.7] | 58/60 | 2.1e-05 |

## Peer versus document specificity

These interactions compare the generated-minus-matched effect for each peer framing with the effect for a document, using the same fact as the independent unit. A negative effect favors a larger disruption under the peer framing. The false_minus_true rows additionally compare the claim interaction across framings. The framings also differ in their prompts and incentives, so an agency interpretation requires care.

| Grader | Peer contrast | Claim | Complete-fact correctness interaction, pp | All-fact bounds, pp | Facts | p |
|---|---|---|---:|---:|---:|---:|
| Nemotron | incentive_vs_document | false | -8.8 | [-11.7, -6.7] | 57/60 | 0.446 |
| Nemotron | incentive_vs_document | true | 3.4 | [0.0, 3.3] | 58/60 | 0.688 |
| Nemotron | incentive_vs_document | false_minus_true | -10.9 | [-15.0, -6.7] | 55/60 | 0.309 |
| Nemotron | no_incentive_vs_document | false | -6.8 | [-6.7, -5.0] | 59/60 | 0.543 |
| Nemotron | no_incentive_vs_document | true | -5.1 | [-6.7, -5.0] | 59/60 | 0.581 |
| Nemotron | no_incentive_vs_document | false_minus_true | -1.7 | [-1.7, 1.7] | 58/60 | 1 |
| GPT-OSS | incentive_vs_document | false | -6.9 | [-10.0, -6.7] | 58/60 | 0.559 |
| GPT-OSS | incentive_vs_document | true | 1.7 | [0.0, 1.7] | 59/60 | 1 |
| GPT-OSS | incentive_vs_document | false_minus_true | -10.5 | [-11.7, -6.7] | 57/60 | 0.307 |
| GPT-OSS | no_incentive_vs_document | false | -7.1 | [-10.0, -3.3] | 56/60 | 0.523 |
| GPT-OSS | no_incentive_vs_document | true | -3.4 | [-5.0, -3.3] | 59/60 | 0.791 |
| GPT-OSS | no_incentive_vs_document | false_minus_true | -3.6 | [-6.7, 1.7] | 55/60 | 0.856 |

## Unedited baseline

The two comparisons with the unedited baseline distinguish harm caused by the focal edit from harm caused by its random matched control.

| Grader | Comparison | Claim | Complete-fact correctness effect, pp | All-fact bounds, pp | Facts | p |
|---|---|---|---:|---:|---:|---:|
| Nemotron | generated_vs_none | false | -41.7 | [-42.8, -40.6] | 56/60 | 2.14e-10 |
| Nemotron | generated_vs_none | true | -4.0 | [-5.0, -3.9] | 58/60 | 0.213 |
| Nemotron | matched_generated_vs_none | false | -1.7 | [-1.7, -1.7] | 60/60 | 0.656 |
| Nemotron | matched_generated_vs_none | true | 1.1 | [1.1, 1.1] | 60/60 | 0.795 |
| GPT-OSS | generated_vs_none | false | -42.3 | [-44.4, -42.2] | 56/60 | 1.33e-10 |
| GPT-OSS | generated_vs_none | true | -3.4 | [-4.4, -3.3] | 58/60 | 0.291 |
| GPT-OSS | matched_generated_vs_none | false | -1.7 | [-1.7, -1.7] | 60/60 | 0.656 |
| GPT-OSS | matched_generated_vs_none | true | 0.6 | [0.6, 0.6] | 60/60 | 1 |

## Joint correctness and usability

These are counts out of 180 responses per row, including unusable outcomes. The requested wrong cells mean semantic_final_correct=0, which includes false assertions, contradictory answers, and missing answers. The JSON retains final_truth counts to distinguish those categories. Unknown means correctness or usability is unresolved. A known label remains usable in its own marginal analysis when another label is unresolved.

| Grader | Condition | Claim | Correct + usable | Correct + unusable | Wrong + usable | Wrong + unusable | Unknown |
|---|---|---|---:|---:|---:|---:|---:|
| Nemotron | none | false | 177 | 0 | 3 | 0 | 0 |
| Nemotron | none | true | 172 | 0 | 8 | 0 | 0 |
| Nemotron | matched_generated | false | 174 | 0 | 3 | 3 | 0 |
| Nemotron | matched_generated | true | 174 | 0 | 5 | 1 | 0 |
| Nemotron | jspace_generated | false | 100 | 0 | 52 | 24 | 4 |
| Nemotron | jspace_generated | true | 163 | 0 | 8 | 7 | 2 |
| GPT-OSS | none | false | 177 | 0 | 3 | 0 | 0 |
| GPT-OSS | none | true | 172 | 0 | 8 | 0 | 0 |
| GPT-OSS | matched_generated | false | 174 | 0 | 3 | 3 | 0 |
| GPT-OSS | matched_generated | true | 173 | 0 | 6 | 1 | 0 |
| GPT-OSS | jspace_generated | false | 97 | 0 | 55 | 24 | 4 |
| GPT-OSS | jspace_generated | true | 164 | 0 | 7 | 7 | 2 |

## Fluency and scope of interpretation

The original false-claim fluency check is evaluated on all 180 primary pairs, all of which have claim_in_draft=true. Its threshold is J/generated minus matched-generated at least -0.1. True-claim fluency is shown separately as an exploratory control.

Legacy fluency is the ratio of distinct word bigrams to all word bigrams in the full response, with a zero score for very short replies. It detects some repetitive collapse and does not establish grammaticality, successful fact correction, or preserved general communication ability. Usability is separately judged from the final document and can be true for incorrect or off-target coherent prose.

| Claim | J/generated fluency | Matched-generated fluency | Difference | Within original 0.1 margin |
|---|---:|---:|---:|---|
| false | 0.8370 | 0.8682 | -0.0312 | True |
| true | 0.8632 | 0.9711 | -0.1079 | False |

The original false-claim fluency margin passes. The semantic usability analysis nevertheless shows an additional loss of coherent final prose under J/generated, and the joint table also shows many coherent but incorrect or incomplete final documents. Neither dropping unusable responses nor relying only on the lexical fluency score captures both forms of task failure.

The original preregistration used lexical catch_score, a source/fact pair as its test unit, and a fluency margin of 0.1 for false claims. This analysis instead uses audited semantic labels, fact clusters, true-claim controls, and interactions after outputs were available. The semantic scorer underwent documented development calibration. Nemotron also used a timeout recovery attempt, and GPT-OSS used a predeclared technical fallback for eligible failures. These choices are recorded in the grader provenance and support exploratory interpretation.

Qwen3-32B did not pass the original Experiment 0 belief-tracking gate. At the story-end site, its agent true-belief control rate was 0.55 and false-belief rate was 0.45, each with saved p=0.8238. Later prompt audits identified setup issues, so these outcomes do not establish validated belief tracking in the intervention subspace. Exp4 supports statements about sensitivity of this fact-sheet task to selected J-lens edits. A general claim that J-space is necessary for multi-agent communication would require additional validated tasks and interventions.

## Reproduction and provenance

Run `python3 scripts/analyze_exp4_specificity.py` from the repository. The script uses the standard library and performs no model inference. The JSON records exact scored-row hashes, shared original-review hashes, grader identities, calibration counts, original turn-file hashes, analysis source hashes, deterministic seed, and bootstrap settings. It also contains all 13 condition summaries, all five existing matched J-lens comparisons, and both focal comparisons with the unedited baseline.

- Nemotron scored rows SHA-256 `ebcaeaf5c71fc9d056896e764431d3104651df238106551902dfc9f4e7fb6348`.
- GPT-OSS scored rows SHA-256 `4b8cd858e95db06e5d3045e34c1979cf733fabc4a6a524c9ba166091ca7f75b4`.
- Shared original reviews SHA-256 `63f9299f6144c3ce85b7026fb0a687e3db96ee9fb2dc8122ce5a6cc0375b0392`.
- Analysis script SHA-256 `7f042c8befcd721344a8d55e53d6e3c6265832dff2b1b2ef36d879d7d40f16ac`.

Machine-readable results are in [specificity.json](../results/qwen3-32b-exp4-analysis-20261107/specificity.json).
