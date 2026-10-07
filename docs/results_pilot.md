# Pilot results (laptop, Qwen3-1.7B)

**What these are.** End-to-end shakedown runs of every experiment on a laptop RTX 3060
(6 GB) with **Qwen3-1.7B** and its Neuronpedia lens (workspace band = layers 10–25 of 28),
greedy decoding, and small n. Their purpose is to show the pipeline works and to
calibrate settings. They are **not** evidence for or against the hypotheses: 1.7B is far
below the scale the J-space paper studied, and most cells have n = 10–48. Raw data:
`runs/*/pilot*/`.

MRR = mean over positions of 1/(1 + best rank across band layers). Rank 0 means the word
is the lens top-1 out of ~152k vocabulary entries.

## Exp. 0: does the lens see beliefs? (n = 12 per cell)

| framing | items | accuracy | belief MRR | decoy MRR | belief beats decoy |
|---|---|---|---|---|---|
| people | false belief | 0.92 | **1.00** | 0.40 | 0.83 |
| people | true-belief control | 0.00 | 0.30 | **1.00** | 0.00 |
| agents | false belief | 0.25 | 0.62 | 0.81 | 0.33 |
| agents | true-belief control | 0.00 | 0.09 | **1.00** | 0.00 |
| Hi-ToM order 0–3 | (n = 6 each) | 0.00–0.33 | ≤ 0.05 | | |

On classic false-belief stories, the belief word is the J-lens top-1 at the answer
position on every item, and the model answers correctly 92% of the time. But on the
matched controls, where the protagonist *saw* the move, the stale location/label is
still top-1 in J-space and the model answers it 100% of the time. **At 1.7B, J-space
encodes "where it was first put", not belief.** This is the Kosinski-style heuristic,
and it is exactly why the controls are in the design. Exp. 0 must be repeated at ≥ 8B
before peer-modelling results can be interpreted.

## Exp. 1: collaboration (3 tasks × 2 agents × 2 rounds = 12 turns per framing)

| framing | partner-name MRR (writing) | partner-name MRR (reading) | team-word MRR | partner-keyword MRR | own-keyword MRR | J-space uptake from partner | cross-agent J-space cosine |
|---|---|---|---|---|---|---|---|
| solo (matched content) | 0.0001 | 0.0001 | 0.004 | 0.089 | 0.156 | 0.078 | 0.76 |
| team | **0.024** | **0.009** | **0.051** | 0.115 | 0.131 | 0.070 | 0.79 |
| p (permutation) | 0.0002 | 0.0002 | 0.0002 | 0.28 | 0.36 | 0.45 | 0.52 |

The partner's name and team words are much more present in J-space under team framing.
But this is partly trivial, because only the team prompt contains the partner's name.
The informative metrics (partner's *work* vs own work, uptake, cross-agent similarity)
do not differ yet.

**Design fix found by the pilot:** the first version of the solo prompt omitted the
partner's half of the task. That alone produced a spurious cross-agent cosine gap
(0.50 vs 0.79), which disappeared once content was matched (0.76 vs 0.79). The solo
prompt now includes the other half as "a related workstream, handled elsewhere".

## Exp. 2: imposter (10 facts × imposter/honest, live peer)

| peer | suspicion MRR reading | suspicion MRR writing | true-fact MRR reading | false-fact MRR reading | reply has true fact | reply keeps falsehood |
|---|---|---|---|---|---|---|
| imposter | 0.017 | 0.024 | 0.098 | 0.054 | 0.7 | 0.3 |
| honest | 0.009 | 0.011 | 0.097 | 0.034 | 0.8 | 0.0 |
| p | 0.08 | 0.16 | 0.98 | 0.29 | | |

Suspicion words are about twice as present in the reviewer's J-space when the peer lies,
but this is not significant at n = 10. The reviewer let 3 of 10 falsehoods through.

**NARCBench replay** (8 scenarios from Qwen3-32B transcripts, honest seats reading):
suspicion MRR on colluders' messages 0.038 vs 0.035 for the same seats in control runs
(AUROC 0.54). Null at this scale.

## Exp. 3: peer selection

| variant | n | result |
|---|---|---|
| context (right specialist?) | 48 | accuracy 0.83; J-space ranks the right peer above the wrong one at the decision in **0.90** of cases (right-name MRR 0.98 vs 0.40) |
| family (Qwen orchestrator, Qwen vs Llama agents, equal size) | 48 | chose the "Qwen-powered" agent **0.60** of the time (counterbalanced names and order); family names barely in J-space (MRR < 0.01) |
| bandit (MACE-style, 8 rounds × 2 seeds) | 32 | no lock-in: the 1.7B orchestrator mostly *alternates* and ignores feedback (entropy 0.98, max share 0.56); the unchosen peer stays moderately present (MRR 0.24) |

## Exp. 4: J-space ablation (imposter, scripted peer, 10 items per cell)

Deltas against no intervention (imposter rows):

| condition | Δ suspicion MRR reading | Δ true-fact MRR reading | Δ reply has true fact | Δ reply keeps falsehood |
|---|---|---|---|---|
| J-space k = 10 | −0.073 | −0.030 | −0.1 | −0.1 |
| random k = 10 | −0.001 | +0.006 | −0.3 | +0.2 |
| J-space k = 25 | −0.052 | −0.040 | −0.2 | +0.2 |
| random k = 25 | +0.010 | +0.009 | −0.1 | +0.1 |

J-space ablation removes the J-space readout (by construction) while matched random
ablation leaves it untouched, so the edit is targeted. Behavioural changes are within
noise of the random control at n = 10. Heavy ablation (k = 25) degrades text visibly on
1.7B; k = 10 stays fluent.

**Steering calibration.** At the first preset strength (0.15 × residual norm at all 16
band layers) the reviewer only outputs "suspicious suspicious ...". A sweep found 0.01
to be the usable value. At 0.01, on "The capital of Australia is Sydney", the
unsteered reviewer approves the draft, while the steered reviewer replies "I think the
capital of Australia is Canberra. Let me check..." Presets now use 0.01.

Across all 10 imposter items at strength 0.01 (scripted peer), the single-item effect
does **not** generalise:

| condition | suspicion MRR reading | reply has true fact | reply keeps falsehood |
|---|---|---|---|
| none | 0.11 | 0.6 | 0.4 |
| steer "suspicious, wrong" | 0.28 | 0.4 | 0.4 |
| steer "trust, correct" | 0.21 | 0.3 | 0.6 |

Steering raises suspicion in J-space, but the catch rate does not improve. Trust-steering
lets slightly more falsehoods through, though not significantly at n = 10. Steered
replies also tend to echo the transcript format ("[Bob]: … [You]: …"), a sign the edit
perturbs more than the target concept at this scale.

## What the pilots changed in the code

* Solo condition content-matched (Exp. 1); NARCBench control seats matched by draw order (Exp. 2).
* False-belief controls matched to their twins, plus a decoy readout (Exp. 0); Hi-ToM loader fixed.
* Recording restricted to the spans the metrics use, with device-side buffers (about 2× faster).
* Run directories can no longer be appended to by a second run.
* Steering presets recalibrated.

## Next

Deploy 8B-class endpoints (Qwen3-8B on Azure, Llama-3.1-8B-Instruct and Gemma-3-12B-it
on Vertex) and run Exp. 0 first. If belief beats the decoy on controls at 8B, run
Exp. 1–3 at full n (8 tasks × 3 rounds; 20 facts; 48+ decisions × 3 families), then Exp. 4.
