# Experiment 7: renewed validation of peer-belief representations

> **UNAPPROVED DRAFT — pending focused review, 2026-11-08.** This document proposes a replacement validation study, not an implementation plan, preregistration, or authorization to run models. Unsettled choices remain assistant proposals. Questions are deliberately left unanswered for later review in [questioning.md — Experiment 7](questioning.md#experiment-7).

## 1. Goal and settled boundaries

Can a frozen recipient LLM maintain and use a peer-indexed representation that distinguishes what a peer believes from world truth, information access, stale evidence, recent wording, and the recipient's own assessment?

Experiment 7 instantiates historical Experiment C. Experiment 5 concerns capabilities; Experiment 6 concerns descriptions versus demonstrated results. Historical D/Experiment 8 remains outside this implementation scope. The original Experiment 0 gate remains failed: a successful new gate would validate this new assay, not repair the earlier preregistration or reinterpret its downstream results.

The agreed target is the recipient's native internal representations. No external PeerMem, controller-supplied belief table, memory offload, fine-tuning, or trained augmentation enters its computation. Offline probes are measurement instruments whose outputs never return to the recipient. Structured relational measurements are selected, but no belief direction, compact code, or identifiable causal component is presumed to exist.

Separate claims concern ordinary contextual/KV retention and stronger carry-forward after direct source access is restricted. Both concern one continuing native episode; neither establishes persistence across resets. A reset or text summary creates a different system and cannot serve as evidence that the original hidden state persisted.

## 2. Proposed approach and participants

The recommended core is a program-tracked event assay adapted from [ExploreToM](https://arxiv.org/abs/2412.12175), followed by an optional private-channel replication. ExploreToM supplies controlled story generation; its [belief tracker](https://github.com/facebookresearch/ExploreToM/blob/main/belief_tracker.py) maintains world, first-order, and second-order states. Its exact revision, subset, rendering, and unknown-state mapping require review and freezing. It is recommended, not approved as the exclusive backend.

Alternatives are a custom audited simulator, which offers tighter task control but greater validation burden, and an external conversational benchmark, which offers different surfaces but less convenient counterfactual control. FANToM or OmniToM are candidates for subsequent external validation; neither is required or selected here. A story-only result concerns tracking represented agents. Calling the story actors peers does not establish inference about a communicating LLM's actual internal beliefs.

Proposed recipient: the pinned Qwen3-32B checkpoint used in Experiment 5, native HF/PyTorch BF16 with its non-thinking template; exact tokenizer, runtime, decoding, and cache policy must be frozen. Proposed minimal inventory: two opaque peer identities and two independently variable facts, using physical-location and artifact-status domains with binary values plus epistemic unknown. This bounds the first assay, rather than establishing arbitrary-domain generality. First-order belief is primary; nested beliefs remain a separately registered extension.

If private source channels are adopted, the preferred control uses the same peer checkpoint and decoding policy with different experimentally assigned private information. A live peer, a frozen response receipt, and a simulated character are alternative participant formats awaiting review. In a source-LLM extension, probe that source on independent forks before it sends an answer; retain disagreements and uncertainty. These probes operationalize elicited beliefs, not transparent access to an otherwise unknowable mental state.

## 3. Ground truth and identifiable contrasts

Maintain separate analysis records for each event:

| Quantity | Operational label and limitation |
|---|---|
| World state `W` | Simulator's actual property value, whether or not the recipient sees it. |
| Peer access `A(peer, fact)` | Which observations/messages reached that peer, including timing and reliability assumptions. |
| Simulator-implied belief `B(peer, fact)` | Value or unknown under the simulator's update rules; not proof of an LLM peer's actual belief. |
| Recipient-available information `I` | Exact events and visibility information provided to the recipient. Hidden world truth need not be inferable from `I`. |
| Recipient self-assessment `S` | Separately elicited answer on an untouched fork; never silently equated with `W` or `I`. |
| Stale/recent candidates | Last peer-observed value, last mentioned value, and recency of mentions. |

Record whether each reference is inferable from `I`. A world question about hidden truth requires an information-appropriate unknown/set-valued reference; decoding inaccessible `W` is not a success requirement. Labels naturally coincide: a true-belief peer often agrees with world truth, and a false belief often equals its last observation. Construct a registered contrast graph separating competing explanations across the inventory; mark contrasts unidentifiable where labels coincide. Simple last-observation retrieval can remain a valid explanation on some items. Communication, re-observation, uncertain access, and recipient-private information provide additional diagnostic cases without claiming perfect factorization.

Use matched true-belief, false-belief, and unknown branches. True and false refer to correspondence with `W`; unknown means the stipulated information does not determine either value. Negation, absence of a tracker property, malformed data, and an LLM refusal are different states. Audit the backend's negation and missing-property semantics before mapping them to unknown. Exclude unsupported semantics before evaluation and publish that scope, rather than turning arbitrary missing labels into correct uncertainty.

## 4. One complete event workflow

1. Generate a symbolic event family and valid counterfactual branches before recipient inference. Check every visibility transition, belief update, and rendered implication; keep hidden tables out of model inputs.
2. Introduce identities, facts, and the information-access rules. Present an initial observation to designated peers and the recipient. Capture an immediate neutral checkpoint without asking a belief question.
3. Change the world. In matched branches the target peer observes the change, misses it, or lacks the information needed to determine the value. The recipient receives the same world event where compatible; access information changes truthfully with the branch. A world-only branch changes `W` while preserving the target's information history.
4. Insert balanced irrelevant events and value mentions. Capture after the change, after new peer evidence, and at a neutral story endpoint before any question. Use a proposed immediate, 128-token, and 512-token continuation, with 512 tokens the primary delayed endpoint. Continuations must be unrelated to target facts, matched across branches, and checked for leakage.
5. Fork the exact pre-question state independently for peer belief, world answer, access judgment, recipient self-assessment, and a peer-belief-dependent action. Proposed action: select the message/correction needed for the peer's next task, with separate correct actions for stale and unknown information. Score reference answers against information actually available to the recipient. Probe outputs never enter another branch or the main episode.
6. Reveal a further observation or reliable message to the peer, paired with a matched confirming/no-information continuation. Capture the next passive state and repeat independent diagnostic forks. Measure movement toward the updated label, not merely any change over time.
7. Replay the frozen family under name/role swaps, entity/value changes, paraphrases, access changes, and question formats. Keep decision-site captures secondary; they cannot establish a state that existed before a question.
8. Apply the frozen route and causal assays on separate forks. Save clean, edited, random-control, donor, and rescue receipts, including all failures. None of these diagnostic branches modifies a subsequent main-stream episode.

## 5. Matched controls and passive measurements

Counterbalance identity, roster order, fact/domain binding, value frequency, event length, and mention recency independently where semantic validity permits. Include swaps preserving the event/value multiset while changing who had access, and fact-assignment swaps preserving each peer's overall exposure. Change one peer/fact cell while holding the other cells fixed. Hold `W` fixed for peer-belief donor pairs; use separate world-changing pairs to test whether a readout simply follows reality. Explicitly record unavoidable dependencies rather than describing every factor as orthogonal.

Propose a regularized linear structured decoder over passive residual captures as the first measurement family. Its output is a distribution over value 0, value 1, and unknown for each peer/fact cell, with separate world and access readouts. Shared entity/role features must be fitted without evaluation labels; an unseen literal name cannot require its own trained classifier. Compare J-lens/semantic projection, raw-state decoding, identity-blind aggregate predictors, last-observation/last-mention rules, and question-time logit/output baselines on the same grouped units. Watched token ranks remain descriptive.

Use grouped development folds with complete event skeletons, paraphrases, swaps, and donor relatives kept together. Reserve untouched event/template families, literal identities/entities, and rendering/question formats for confirmation. Select preprocessing, regularization, sites, layers, subspace rank, and any nonlinear alternative only within development. Bound and record that search; freeze one candidate assay or report that discovery found none. Raw residual success with J-lens failure is a method comparison, not proof that J-space uniquely carries belief.

## 6. Retention and the stronger carry-forward test

Proposed routes are ordinary native context/KV access `R0`, direct source-position masking `R1`, and a stronger route `R2` that also blocks subsequently created positions from repeatedly retrieving the original diagnostic evidence during the delay. The allowed relay/carrier positions, timing, attention edges, and sham mask must be fixed in development. Native continuation or exact verified replay is required; dropping source text and recomputing from scratch is not the same intervention.

Route validation must demonstrate that designated source tokens cannot be retrieved through the intended blocked edges, while matched unmasked facts, identity retrieval, formatting, and task execution remain usable. Compare `R2` to a matched route control and its frozen task baseline. A proposed five-percentage-point noninferiority margin requires an uncertainty bound above the loss margin and adequate absolute performance. A nonsignificant degradation test does not establish retention.

Masks restrict specified routes, not every redundant copy or inference path. Successful `R2` behavior supports availability through the stated surviving routes; it does not prove an abstract, unique memory register. Failure can reflect mask damage or removal of necessary computation. Cross-check those explanations before interpreting it as absence of a carried-forward state.

## 7. Selective causal assay

Find a candidate basis using conditional development pairs that differ in target `B(peer, fact)` while preserving world truth and non-target cells. Source text/access may necessarily differ to produce the target belief; match their surfaces and audit remaining nuisances. Use independent access/belief combinations where possible. If source-belief and access effects remain inseparable, report that limitation rather than calling the basis belief-specific.

Proposed conditions are unedited execution; targeted deletion; geometry-matched random deletion; counterfactual target-belief patch; matched random patch; deletion followed by clean downstream rescue; and matched random restoration after deletion. Add a non-target-cell donor changing a separate fact while preserving the target task, world, and other peer's relevant state. Require a realized nonzero edit. Match layer, position, displacement norm, and relevant residual geometry; record intended and realized BF16 effects.

Choose acquisition/retention sites, basis rank, dose, and a rescue site two blocks downstream during development. Rescue uses separately fitted later-site coordinates and an independent clean donor. Same-site subtraction/addition is an engineering identity check. Counterfactual donors must preserve `W`, identities, non-target facts, task, and question; do not patch a false belief with a different-world donor and then attribute its effect to belief alone. If a conditional donor inventory cannot be constructed, the relevant causal claim is unavailable.

Require deletion to reduce the target behavior against both the actual unedited baseline and random deletion. Require patches to change behavior toward donor content against both unedited execution and random patching. Require downstream rescue to improve over damaged execution and random restoration. Measure non-target judgments, world answers, accessible literal facts, access judgments where separable, and format validity independently. Their proposed five-point preservation margins require uncertainty-based tests; preserved coordinates explicitly constrained by an edit do not establish specificity. An `R2` effect alone establishes conditional use under restricted access; ordinary-use claims need `R0` effects.

## 8. Metrics and proposed significance registry

Unknown is a scored third class. Report balanced accuracy, unknown recall/false-certainty rate, multiclass Brier score, per-state confusion, and joint all-cell correctness. Invalid outputs and unsupported abstentions count as end-to-end errors; correct unknown does not mean arbitrary indecision. Evaluate belief, world, access, self-assessment, and behavior separately. Prefer reference distributions/sets where the protocol permits uncertainty; no probability target is inferred merely from the simulator's binary truth.

The independent unit is an event/template family containing all paired states, counterbalances, forks, and donors. Repeated tokens, questions, cells, and stochastic generations are within-family measurements. Proposed primaries are:

| ID | Required contrast | Scientific significance |
|---|---|---|
| P1 | Pre-question decoding passes frozen condition-balanced reference thresholds in each true/false/unknown stratum, plus gains over identity-blind predictors on identifiable disagreement/binding contrasts. | Candidate peer-indexed information beyond world/recency shortcuts. |
| P2 | Independent-fork belief answers and belief-dependent actions each exceed frozen condition-balanced reference policies in all three strata. | The assay supports behavioral belief attribution and uncertainty handling. |
| P3 | Updating reduces error relative to the new target state and separates from confirming/no-information continuations in the evidence-predicted direction. | Evidence-sensitive updating, beyond generic drift. |
| P4 | `R2` behavior is noninferior to its matched route control at five points and exceeds its frozen task baseline. | Bounded carry-forward availability. |
| P5 | Targeted deletion causes loss versus unedited and random deletion, under both `R0` and `R2`. | Candidate information contributes to behavior in both routes. |
| P6 | Donor-directed patches exceed unedited and random-patch behavior, under both routes. | Intervention content, beyond nonspecific damage, controls behavior. |
| P7 | Downstream clean rescue exceeds damaged execution and matched random restoration under `R2`, with preservation gates. | Partial selective recovery through the identified later component. |

These seven entries are proposals. Freeze exact estimands, baseline fitting, thresholds, diagnostic contrasts, and specificity gates before confirmation. Baselines receive no hidden state/stratum labels; do not demand an improvement over a definitionally correct rule on aliased items. Conjunctive entries require every component; use their maximum valid component p-value, never an average that hides a failed control. Decoder evidence alone does not establish causal use; successful behavior alone does not validate the decoder.

Propose family-wise error 0.05 with Holm adjustment across the frozen primary registry. Its relationship to Experiments 5-6 must be prospectively settled. Use paired family-level inference: exact sign flips through 20 families, otherwise 50,000 seeded assignments, with symmetry assumptions explicit; 100,000 family-bootstrap resamples for intervals, proposed seed `20261108`. Validate the one-sided noninferiority procedure under null simulation. Specify simultaneous/adjusted uncertainty for preservation claims. Report effect sizes, component intervals, all failed/gated tests, and sensitivity to unresolved outputs. Retain every assigned family; never condition primaries on baseline-correct or fluent survivors.

## 9. Stage gates, power, and deliverables

| Stage | Gate and consequence |
|---|---|
| Semantic/engineering audit | Valid world/access/belief labels, unknown mapping, leak checks, replay equivalence, exact masks/edits; failure blocks dependent claims. |
| Development | Behavioral feasibility, grouped relational discovery, diagnostic controls, route integrity, and selective intervention; no candidate is a reportable outcome. |
| Freeze | Fixed protocol, primary family, splits, power analysis, complete call inventory, and compute ceiling before opening evaluation. |
| Untouched confirmation | Execute every registered cell without retuning; incomplete runs retain explicit coverage and conservative bounds. |
| Reconciliation | Recompute from immutable receipts; distinguish behavioral, representational, route, and causal successes and failures. |

Provisional planning counts are 16 development families and 144 evaluation families, each containing matched true/false/unknown branches. These are budgeting placeholders, not an 80%-power guarantee. Simulate power for the entire joint success rule, including multiplicity, paired-family variance, unknown prevalence, preservation margins, missingness, and correlations. Select final counts and any model replication before confirmation. Historical single-binomial counts do not establish power for this conjunction.

Count all event versions, checkpoints, retention routes, independent probe/action forks, donor captures, causal/random/rescue conditions, source audits, repetitions, and retries. Time representative development workloads and freeze token, call, and wall-time ceilings. No evaluation-driven significance stopping or silent replacement of failed families is permitted.

Required artifacts are pinned source/model/configuration hashes; symbolic and rendered event manifests; visibility and label audits; all grouped splits; complete raw receipts; passive states; decoder/basis files; donor mappings; realized masks/edit geometry; registry and power/resource report; per-family effects; uncertainty/coverage tables; and reports of all gates. Existing serving/capture/edit infrastructure is reusable, but the event adapter, semantic validator, independent branching, routes, and analysis require implementation after approval.

A positive conjunction supports a bounded functional peer-belief account for this frozen model, task, and intervention. A valid decoder without selective intervention establishes decodability; a failed candidate limits this measurement, not all possible distributed beliefs. Similar effects for matched document sources would favor source-general tracking. Without that optional comparison, do not claim a uniquely social circuit, general theory of mind, or direct knowledge of a source LLM's true beliefs.

Review remains open at [questioning.md — Experiment 7](questioning.md#experiment-7); no proposed setting above is user-approved by inclusion in this draft.

## References and prior evidence

- [Historical Experiment C and failed Experiment 0](../../jeffrey-exp-plans-new/peer-modeling-next-experiments.md).
- [Audited results and localized causal methodology](../../2026-11-07-causal-followup-design.md).
- [Experiment 5 specification](2026-11-08-experiment-5-peer-capability-retention-design.md), for shared native-state and statistical constraints.
- [ExploreToM primary paper](https://arxiv.org/abs/2412.12175) and [programmatic belief tracker](https://github.com/facebookresearch/ExploreToM/blob/main/belief_tracker.py), motivating the proposed event backend; these do not validate this assay or the recipient's hidden state.
