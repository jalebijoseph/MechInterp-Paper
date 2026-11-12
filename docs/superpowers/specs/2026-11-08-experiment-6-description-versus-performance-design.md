# Experiment 6: peer self-description versus demonstrated performance

> **UNAPPROVED DRAFT — pending review, 2026-11-08.** Questions intentionally remain unanswered. This document proposes a scientific workflow; it authorizes no implementation, inference, launch, staging, or commit. User requirements and assistant proposals are distinguished below. Review decisions are recorded in [questioning.md — Experiment 6](questioning.md#experiment-6).

## 1. Question, scope, and scientific claim

**When a peer describes itself as capable, how does the recipient determine whether that claim is supported by actual results?** Experiment 6 develops historical Experiment B. Experiment 5 concerns learning capability associations; Experiment 7 concerns validation of belief/information-access representations. Historical Experiment D/8 is outside the implementation scope and is not a prerequisite.

The settled research target is a frozen recipient that makes its own sequential delegation decisions. No PeerMem, external peer summaries, controller-supplied capability estimates, or algorithmic exploration policy such as LinUCB selects its actions. Offline estimates are analysis artifacts only. Measure learning throughout trajectories, with structured peer-by-domain measurements and separate native-context versus carried-forward-state tests. These constraints are user requirements, not new proposals.

The proposed claim is bounded: the recipient distinguishes attributed self-description from demonstrated task performance, updates its assessment, and uses relevant evidence in subsequent choices. A claim–evidence factorial does **not** identify what the peer privately believes, whether it knows its claim is false, or whether it intends deception. Those require additional manipulations and validated labels. The historical labels “truthful” and “imposter” are therefore replaced by operational claim/performance descriptions.

## 2. Proposed construction and alternatives

Reuse [Experiment 5](2026-11-08-experiment-5-peer-capability-retention-design.md), especially §§5–9, for objective task checking, identity/domain controls, trajectory branching, readout discovery, and edit geometry. Its unresolved defaults are not approved by reference. Its capability-learning and measurement gates must transfer to this assay; prior success cannot be assumed.

The settled construction (G-005) is the **same peer checkpoint with different task-relevant private information**. Proposed implementation: two experimental peers, two objectively checked task domains, and fresh task-only peer requests. Informative versus restricted private material changes effective performance while peer weights, decoding policy, and formatting are matched. Candidate tasks use private rules or reference tables; the final domains and missing-information mechanism remain unresolved. Restricted information must produce coherent attempts, uncertainty, or abstention, rather than fabricated checker failures. Independently audit both regimes on held-out matched tasks. This measures **information-conditioned effective performance**, not a difference in intrinsic model capability.

Cross an attributed high/low capability claim independently with the high/low information regime for a target peer/domain cell. Assignment of claims to every studied peer/domain must vary independently of performance; do not make the second peer or domain an automatic complement. Hold the comparison peer’s information condition fixed within each paired target manipulation, and balance its condition across families. Include equal-quality conditions and independently counterbalance names, roster order, and claim/evidence presentation order.

| Assigned claim | Audited performance regime | Operational description |
|---|---|---|
| High | Higher | High claim, higher independently audited performance |
| High | Lower | High claim, lower independently audited performance |
| Low | Higher | Low claim, higher independently audited performance |
| Low | Lower | Low claim, lower independently audited performance |

Claims are proposed categorical statements about the named domain under the current setup, matched for length, tone, and format. Freeze wording after development checks confirm their intended ordering without extra clues. They are experimentally assigned attributed statements, not evidence that the generating model endorses a private self-belief. The recipient never sees regime labels, private material, source brands, future outcomes, or cumulative controller summaries.

Two delivery options remain open: live execution after each chosen delegation, or a frozen bank of genuine, verified peer responses indexed by the actual choice and task. The latter improves counterfactual matching but must preserve sequential choice and hide unrevealed outcomes. A scripted balanced history is useful for discovery, not a substitute for the main online study. Changing peer checkpoints would confound information and intrinsic source differences and is not the proposed main construction.

**Important distinction:** randomization crosses claim with the *assigned performance-generating regime*. Observed evidence depends on outcomes and endogenous exposure. Analyze the randomized regime effect as an assignment effect; do not relabel selected successes/failures as randomized evidence. Separate matched-history diagnostic branches supply the direct claim/evidence comparison below.

## 3. One complete trajectory and its diagnostic branches

1. Introduce opaque peer identities and their domain-specific claims. Record passive state and the initial actual delegation before any result. An equal-information, matched-claim baseline estimates name, order, and claim preferences; reuse compatible Experiment 5 baseline receipts only under identical conditions. It does not establish a uniquely correct peer.
2. Present tasks from a frozen domain-balanced schedule independent of peer choice. The recipient chooses; that peer executes; show its attributed answer and truthful checker outcome. Preserve the recipient’s native conversation/state. Never force balanced allocations, insert external learning estimates, or supply an unselected peer’s outcome to the main stream.
3. Continue over the full horizon, recording actions, all visible evidence, task difficulty, and exposure by peer/domain. Plot preference, concentration, switching, performance, and relational measurements throughout. A peer never selected remains unobserved, not known to be weak.
4. At frozen checkpoints, fork diagnostic questions and delayed continuations from the untouched state. Capability estimates, claim-recall answers, uncertainty questions, and intervention outputs never enter the main history or choose its later exposure. Capture passive activations every round where feasible; fix checkpoints and curve summaries before evaluation.
5. On separate matched branches, independently change only the target claim or the target cell’s coherent verified evidence package, retaining the other cells, tasks, nuisance settings, and next delegation problem. Preserve answer–checker consistency; never relabel an unchanged correct answer as a failure. Use genuine alternate receipts. These branches estimate controlled immediate effects, not counterfactual full online policies.
6. Add a prespecified confirming-versus-contradictory evidence branch at an update checkpoint, holding added text and observation opportunity comparable. Measure assessment change and subsequent choices. The main trajectory can also contain a prospectively scheduled information-regime change, but failure to notice an invisible improvement is not failure to use received evidence.

Proposed planning starting points are Experiment 5’s 40-round horizon, checkpoints after rounds 4, 8, 16, 32 and the final round, and immediate/128-token/512-token diagnostic delays. These remain assistant proposals, subject to power and resource planning. The 512-token site is the proposed primary delayed site. No unreviewed requirement to run every Cartesian combination is implied.

Use an independent quality reference for both peers on matched tasks, separate from the recipient’s selected outcomes. Report observed-evidence agreement separately from prediction of future performance. Regret or “best-peer routing” requires an identified quality advantage with uncertainty; otherwise report success, allocation, and evidence response. Log comparison-peer nonselection and abstention explicitly. An analysis-only observed-success reference is not the recipient’s actual belief.

## 4. Relational measurement, state, and retention

Proposed structured outputs distinguish, for every peer/domain cell: the attributed claim; demonstrated evidence and its exposure/unknown status; estimated future performance; and uncertainty. The first two have experimental/receipt labels. Future-performance calibration uses the independent audit. Uncertainty is elicited on forks and assessed against exposure and prediction error; it has no independently observed “true internal confidence” label. Self-belief and intent judgments are outside the main outcomes.

Capture residuals at the final token of a fixed neutral receipt immediately following the initial claim, each selected result, and each delayed continuation, before any diagnostic question. Freeze the exact receipt text, tokenizer span, layers, and hook location in development. At a next-choice checkpoint also capture a pre-decision state and a decision-site state separately. The decision-site signal cannot establish a representation available before the choice was requested.

Fit development-only structured readouts, transferring them without refitting to online checkpoints and unseen names/templates. Compare evidence decoding with prior, identity-blind outcome-total, and claim-only baselines; compare claim decoding with evidence-only baselines. Include constant-total peer-attribution swaps, domain swaps, and single-cell changes. A global “trust” score, name salience, or latest-outcome detector must not satisfy binding criteria. Distinguish decodability from causal use and from the recipient’s explicitly elicited judgment.

Reuse Experiment 5’s grouped fitting, fold-local preprocessing, bounded layer/rank discovery, residual/logit/J-lens comparisons, permutation controls, and unknown-cell scoring principles. The exact numerical search grid and transfer thresholds require a frozen Experiment 6 configuration. No candidate relational state is presumed to exist. Failure to find one limits the measurement; it is not proof of absent distributed information.

Run route tests separately for claim and performance sources: R0, ordinary native access; R1, question-time source blocking; and R2, blocking immediately after the post-result receipt through the delay, question, and answer. Include matched irrelevant-position masks. For the main carried-forward test block **both** original claim and evidence intervals, including task/attribution/answer/checker tokens and quoted repetitions; later summaries or diagnostic answers cannot leak them back. Freeze and save exact allowed/blocked spans and actual mask effects.

Native KV access can support retrieval of original messages. Blocking source positions can test information in later positions, but later KV entries may still encode literal content. Carried-forward success alone does not establish an abstract capability representation. Validate segmented native continuation or exact replay with all prior masks/edits against uninterrupted execution. A fresh stateless request does not inherit an earlier edit. Residual and direct-KV interventions require distinct provenance and cache-effect checks.

## 5. Conditional causal substudy

Proceed only after development identifies a reproducible evidence-related candidate with identity/domain binding and transfer to online states. A claim-related candidate is a separate discovery target; do not force orthogonal “claim” and “evidence” axes if the data support a shared or distributed representation.

At an online post-result checkpoint, hold the prefix, next task, and choice request fixed across forks. Reuse Experiment 5’s intervention structure: unedited execution; targeted partial deletion; matched random deletion; deletion followed by downstream clean-donor rescue; counterfactual donor patch without deletion; matched random restoration/patch; and an other-domain donor control. Test ordinary access and the carried-forward route separately.

An evidence donor changes only the target cell’s verified evidence package, keeping its claim fixed. A claim donor changes only its claim, keeping all observed receipts fixed. These are controlled-history donors, not assertions that an independently evolving online conversation would have produced that prefix. Clean rescue uses a separate unedited run of the identical episode. Freeze donor eligibility and nuisance matching; report unmatched fields rather than declaring perfect matching.

Proposed edit sites follow Experiment 5: deletion at the frozen receipt token/block output, rescue two blocks later with its own identified basis, and counterfactual patch at that later site. Verify actual downstream/cache effects; a block-output edit need not alter that layer’s already-computed keys and values. Match random controls on realized displacement norm, residual inner product, position, layer, and dtype effects, using the correct damaged or unedited recipient for each operation. Same-site algebraic cancellation is only an engineering check.

Require an **actual** loss relative to unedited execution and additional loss versus random deletion. Require a donor-directed behavioral change relative to both unedited execution and matched random patching. Rescue must improve on deletion and matched random restoration; full recovery requires its own noninferiority evidence. Test preservation of the other domain/cells, literal identity/fact retrieval, and output validity on independent behavioral forks. Proposed preservation margin is five percentage points, inherited as an unapproved starting point from Experiment 5. A nonsignificant side effect does not establish preservation.

Immediate forks identify effects on the next actual choice. Claims about altered long-run policy require separate branches that continue taking genuine sequential actions, with their own inventory and cost. R2-only effects support conditional use under restricted access; native-use claims require native-route evidence. A null local edit may reflect redundancy or a poor intervention.

## 6. Outcomes, statistical significance, and interpretation

Define `p(c,e,t)` as target-peer selection probability at time/checkpoint `t` in a paired controlled branch, with high/low claim `c` and evidence `e`. Report the evidence main effect averaged across claims, the claim main effect averaged across evidence, both simple effects, and the difference-in-differences interaction. In the online factorial, replace `e` with assigned regime and label the estimand accordingly. Do not adjust the primary assignment effect for post-assignment exposure as though exposure were randomized.

Evidence use can yield a positive evidence effect with **zero interaction**, for example additive integration of claims and results. Conflict behavior is assessed directly in high-claim/poor-evidence and low-claim/good-evidence branches. A significant interaction alone can reflect greater reliance on claims after failure; its direction and actual choices matter. No main or interaction contrast identifies deception.

Proposed primary candidates, each retained separately in a registry before evaluation, are:

- Controlled evidence sensitivity of next-choice behavior, including conflict cells, versus zero evidence effect.
- Online quality-adjusted routing improvement from frozen early to late windows, jointly requiring late performance above a matched random-routing reference where a quality advantage is identified.
- Held-out evidence-state prediction beyond identity-blind and claim-only baselines, jointly with constant-total binding transfer. Claim decoding is reported separately and is required only for a claim that both information types are represented.
- Native-route selective loss, donor-directed change, and rescue, registered as separate contrasts with the unedited/random comparisons specified above.
- Carried-forward retention and corresponding selective causal tests, separated from the native-route claim. Any noninferiority claim also requires performance above an appropriate chance/control baseline.

Freeze the exact scoring target, phase windows, update error reference, probability handling, direction, and conjunctive success rules before evaluation. For updating, require both an absolute reduction in assessment error toward revised observed evidence and evidence-aligned separation from the matched confirming branch; relative separation alone can conceal drift away from the evidence. Report behavioral and decoded updates independently. Avoid selecting a favorable checkpoint or aggregating components to hide a failed requirement.

The independent unit is a scenario family containing paired claim/regime assignments and its online trajectories/forks. Keep names, task/communication templates, and source-response provenance grouped across splits. Repeated turns, masks, donors, name arrangements, and random seeds are dependent observations, not additional independent samples. Analyze family-level paired contrasts, with equal family weights unless a different estimand is frozen prospectively.

Proposed inference reuses Experiment 5’s exact sign flips for at most 20 families, otherwise 50,000 seeded assignments, and 100,000 family-bootstrap resamples with seed `20261108`; state sign-flip symmetry assumptions and validate noninferiority procedures in simulation. Report effects and intervals, not p-values alone. Use family-wise 0.05 with Holm correction across the final frozen registry; reconcile its relationship to Experiments 5–7 before evaluation. Conjunction claims use the maximum component p-value and report every component. These numerical settings are proposals, not a completed preregistration.

Keep all assigned trials in end-to-end denominators. Invalid choices, malformed outputs, refusals, and cap hits remain explicit scored outcomes under a frozen rubric; scientific abstention and technical missingness differ. Do not condition primary effects on valid, baseline-correct, or probe-correct survivors. Freeze technical retry rules, retain receipts, and show conservative bounds for unresolved outcomes plus complete-family sensitivity analyses. Gated-off primaries remain visible in the declared family.

Positive behavior without a valid state assay supports evidence-sensitive routing only. Decoding without selective intervention supports available information, not causal mediation. Successful controlled donors without online transfer establish a controlled-assay mechanism only. Native retrieval without the carried-forward result supports contextual use. None establishes intrinsic capability differences, self-belief, deliberate deception, a uniquely social circuit, or J-space necessity.

## 7. Workflow, resources, and review boundary

Proceed through fixture/checker and information-manipulation audits; baseline and online behavioral development; matched-branch and relational discovery; route/intervention validation; power/resource/registry freeze; untouched evaluation; then receipt-based reconciliation. Each stage answers a different uncertainty: whether the manipulation works, whether the recipient uses results, whether relational information is measurable, whether its causal role is testable, and whether effects generalize. Preserve behavioral findings if mechanistic discovery fails. Never tune on evaluation and retain the same confirmation label.

Experiment 5’s candidate 16 development and 96 evaluation families are planning references only. Determine Experiment 6 size using development-family variance, endogenous exposure, audit uncertainty, multiplicity, missingness, and the full conjunction success rule. Specify a scientifically meaningful effect and precision target before power simulation. The historical 60–80-profile suggestion and simple standardized-interaction calculations are not a power guarantee for this longitudinal study.

Measure representative development latency/tokens before fixing a cost ceiling. Enumerate sessions × rounds, actual selected-peer calls, independent quality audits, controlled histories, checkpoint/delay/route forks, donor captures, causal/control conditions, and retries. Live execution ordinarily adds a selected-peer call per actual delegation; a response bank changes delivery cost, not the number of recipient decisions. Do not publish an exact budget until this inventory and backend are settled. Freeze stopping rules and resource limits before evaluation.

Retain protocol and family registry; task/private-information/claim manifests; independent audits; model/runtime hashes; all rendered prompts, spans, receipts, choices and visible-evidence inventories; native/replay checks; masks and intended/realized edits; grouped splits and fitted artifacts; donor maps; power/cost calculations; all primary effects, curves, coverage and missing-outcome bounds. Reuse existing Experiment 5 machinery where validated rather than build an external decision controller or rerun Experiment D.

The unresolved construction/backend, claim wording, task domains, horizon, sample size, estimands, discovery settings, causal scope, and resource ceiling remain proposals for later focused review. Knowing-deception, peer self-belief, matched non-social sources, and full edited-policy trajectories are optional extensions requiring separate design and budget decisions. No answers or approvals are inferred from this draft.

## Local grounding

- [Experiment 5 draft and shared methods](2026-11-08-experiment-5-peer-capability-retention-design.md).
- [Historical Experiment B](../../jeffrey-exp-plans-new/peer-modeling-next-experiments.md#proposed-experiment-b-prompted-state-versus-demonstrated-capability).
- [Audited results and localized causal-assay limitations](../../2026-11-07-causal-followup-design.md).
