# Audited Qwen3-32B follow-up v2

The user authorized fixing the audited prompt/scoring issues, rerunning experiments
1–3 on Qwen3-32B, and rescoring experiment4's frozen outputs. This is a post-audit
follow-up, not an amendment that makes the original preregistered gate pass.
Original code paths/results and raw keyword scores remain available.

## Execution and resources

Use the existing HF/PyTorch BF16 Qwen3-32B checkpoint/lens revisions. Use actual
OLMo-2-0325-32B-Instruct as the second 32B family, pinned before execution, for
anonymous plan generation and independent fact grading. A plain HF endpoint needs
no lens. Restrict new GPU allocation to idle physical0–3 by UUID; never touch4–7.
A durable CPU controller waits for the original32B suite to finish and its servers
to stop, then starts Qwen on one pair and OLMo on another. Own and stop only child
sessions. Cache/download/CPU rescoring preparation can happen beforehand. Do not
interrupt or alter the original run. Use unique output roots, atomic checkpoints,
source/config/model/rubric hashes, exact expected counts, and explicit failures.

## Shared measurement and scoring

New followup modules use a common request writer and full saved messages. Record
original J/logit/output ranks and per-layer profiles. Primary follow-up readouts
average reciprocal rank across band layers, rather than maximizing over35 layers
and comparing with a single output layer. Compare role/condition differences and
cluster inference by independent task/fact/session/scenario. Keep original-style
readouts secondary for comparison. No assertion of unique J-space information
from unequal layer-pooling opportunities.

Fact grading separates final-document target truth, explicit rejection/correction,
intent accusations, and usable prose. An independent OLMo grader receives reference
true/false propositions and the frozen response, blinded to source, reward and
intervention. Require structured enum/boolean judgments and literal evidence spans;
invalid/uncertain outputs are not silently counted as success. Quoted, negated,
synonymous and plural answers require semantic treatment. Regression/calibration
cases cover Everest/Himalayas, Amsterdam/TheHague, literary doubt, dollars, quoted
false claims, mixed reviews/finals, ambiguity, repetition and instruction-like text.
A scorer calibration gate precedes authoritative scores; record abstentions and
coverage, and export unresolved cases for review with conservative score bounds.
Final target truth is the new semantic outcome, not average keyword inclusion and
error-word presence. Preserve legacy catch_score and fluency separately.

Before inference, freeze a five-test primary family: Exp1 role/framing contrast,
Exp2 reward-by-knowledge intent interaction, Exp3 contextual role contrast, Exp3
anonymous generated-plan preference, and Exp4 semantic final-truth intervention
contrast. Apply Holm correction across all five, retaining family size when a test
is unavailable. All tests remain exploratory after the original Exp0 gate failure.

## Experiment1

Use8 existing tasks and2 roles. Generate16 neutral frozen peer notes once from
that peer's private brief. Own prompts initially contain only own brief, task goal,
identical own/partner role metadata and own handoff code, not the partner brief.
Attach the same frozen peer note to team and named_solo conditions; change only
one collaboration instruction. Frozen notes include a task/role handoff code to
verify actual information transfer; a no-peer probe must admit missing peer info.
Run8 tasks×2roles×2framings=32 recorded reviews, plus16 unrecorded frozen-note
calls,32 structured transfer probes and16 no-peer probes. Score code transfer
exactly and keep semantic plan-quality claims out of this assay. Main readout:
team−named_solo of (partner keyword band-meanMRR−own keyword band-meanMRR),
paired by8tasks and averaged over roles; also report individual-layer profiles.

## Experiment2

Use all60 facts. Agent cells cross reward100/0, peer verified/unverified knowledge,
and true/false claims:480 recorded reviews. Add120 document reviews with a neutral
third-person draft:600 total. Reward notes have the same template, differing only
in100vs0; reward is for endorsing the complete proposition as true, explicitly not
mentioning/quoting/denying its answer. Peer knowledge refers to whether Bob checked
a reliable source, with no extra true-answer token leaked in only one reward arm.
Keep factual-error and strict-intent readout sets separate. Primary intent contrast
on false claims: (reward effect when verified)−(reward effect when unverified),
paired by fact. Grade final facts semantically and add480 short structured
knowledge probes after agent reviews; score against the stated knowledge state.
NARCBench reruns the same20 source runs with full author-header-anchored message
spans, saved prompts, descriptive AUROC and inference over10 matched scenarios.

## Experiment3

Context:12tasks×4namepairs×2roleassignments×2rosterorders=192 decisions. Use
no assistant prefill and parse an exact single roster name, never first substring.
Measure last-task-token per-name band-meanMRR margins; compare role swaps holding
task and literal roster order fixed, then cluster by12tasks. This cancels fixed
name priors. Preserve decision and per-layer readouts as secondary diagnostics.
Anonymous plans:12tasks, actual Qwen32BvsOLMo32B, same deterministic decoding and
length policy, bothorders=24 choices plus12identical-plan controls; accept strict
JSON choice A/B/tie without prefill, sufficient cap, invalid/tie counts explicit.
Report preference for generated plans, not family recognition independent of plan
quality. Main unit is12plan pairs, with order sensitivity and paired statistics.
Brand-label trials are explicitly hypothetical descriptive controls, with stated
sizes32/32,32/64,32/16, and no claim that hypothetical peers actually ran.
Bandit:8seeds×40rounds×2settings=640 decisions. Verify RNG/feedback/probability
mapping; report feedback-conditioned staying, per-session regret, early/late best
arm choices and entropy. Do not equate lack of lock-in with lack of learning.

## Experiment4 rescoring

Do not regenerate original intervention outputs. Rescore all13×360 frozen reviews
under the versioned semantic rubric, retaining condition/item/source/claim keys.
Resume from response+reference+rubric+grader-revision hashes. Require all4680 unique
reviews before final aggregation and180 matched false-claim pairs for the original
jspace_generated vs matched_generated contrast. Report semantic final truth,
coherence and legacy scores side by side, per-condition coverage and uncertainty.
Use the same frozen grader for original Exp2 and corrected Exp2 for comparison.
