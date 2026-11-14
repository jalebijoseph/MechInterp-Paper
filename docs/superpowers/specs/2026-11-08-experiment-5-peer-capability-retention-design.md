# Experiment 5: acquisition, retention, and causal use of peer-capability associations

Status: draft 0.3; focused review active, 2026-11-08. This is a scientific design specification, not an implementation plan or a completed preregistration. No new model runs are authorized by this document alone. Decisions and unresolved questions are maintained in the temporary [question log](questioning.md#experiment-5); a draft default is not user approval.

## 1. Research question and scope

Can an unmodified recipient LLM infer an association between a particular peer and a task capability from communication and verified performance, retain and update that association, and use it to make later delegation decisions?

The user selected the recipient/orchestrator's knowledge of its peers as the first study's measurement subject (E5-050) and accepted this as the initial readout approach (E5-018). The goal is mechanistic understanding of internal peer-skill representations, their updating, and their causal use. Mutual skill knowledge across all conversing agents is not required for this initial study, and an orchestrator-only result would not establish it. Exact decoder labels and scoring still require refinement.

The user emphasized the emergence of a preference, or bias, toward one peer over another. Track its direction and development, how it follows domain-specific evidence, and the internal representations contributing to it. Compare preference with independently verified peer quality and the accepted equal-capability baseline. This emphasis complements the accepted learning criterion and mechanistic tests; an aggregate allocation share alone cannot establish their claims.

The study distinguishes two forms of retention:

1. Native contextual retention: later decisions use information retained in the model's ordinary attention/KV state, potentially by retrieving the original peer messages.
2. Carried-forward representation: later states support the peer-capability association and its causal use when direct access to the original evidence is restricted.

A positive result at the first level does not automatically establish the second. A representation may be distributed across positions, heads, or layers; this study does not assume that a single persistent vector must exist.

The target LLM remains frozen. There is no PeerMem store, controller-maintained capability estimate supplied to the recipient, external per-peer summary, fine-tuning of the recipient, or hidden capability label in its prompt. An offline probe is an analysis instrument; its predictions never become memory or input for the recipient.

Experiment 5 concerns learned capability associations. Experiment 6 separately tests stated capability against demonstrated results. Experiment 7 separately validates belief and information-access representations. Experiment D/8 remains in the historical planning document and is outside this implementation run. No Experiment 8 result is a prerequisite here.

## 2. What the study adds to Experiment 3

| Existing evidence | Remaining question addressed here |
|---|---|
| Experiment 3 context trials route from supplied work histories; corrected trials show a small role-sensitive readout. | Does a capability association form from observed results without a supplied specialty description? |
| Experiment 3 bandit trials expose feedback and measure allocation over time. | What internal information supports the allocation, where is it retained, and does manipulating its content change later decisions? |
| Existing name swaps address fixed name preferences. | Does the association follow both peer identity and domain through independent assignment changes, delays, and updates? |

Repeating static profiles, name ranks, or a longer bandit alone would not satisfy this specification.

The original Experiment 0 belief gate remains failed. This study cannot retrospectively repair that gate or establish theory of mind from a capability result.

## 3. Agreed constraints and proposed implementation choices

Agreed constraints are the internal-memory research target, separate tests of the two retention levels, maintenance of agent-capability associations, structured relational measurements, and an implementation scope of Experiments 5-7. Subsequent review established that the orchestrator makes the actual delegation choices, learning is measured throughout the interaction, the proposed longitudinal/relational/causal metric families are appropriate, and an equal-capability preference/lock-in investigation runs early as a baseline. The user has now selected copies of the same peer-model checkpoint with different task-relevant information for the main condition (G-005); different peer checkpoints are not the primary construction. This revision separates that online study from controlled histories used only for measurement discovery.

The table distinguishes settled requirements from settings still proposed for review:

| Choice | Draft default | Reason |
|---|---|---|
| Model checkpoint: settled | Pinned Qwen3-32B for the orchestrator and both peers | User accepted the initial single-checkpoint, three-role configuration (G-010/G-017/G-019). Roles retain separate contexts and private information. |
| Inference mode and sampling policy: settled | Non-thinking mode and seeded sampling for all three roles; matching sampler parameters for the two peers | User accepted these parts of G-011. Numerical sampler values, seed schedule, and native HF/PyTorch BF16 remain proposals. |
| Interaction: settled | Live online delegation: the orchestrator chooses the next peer, that peer generates a fresh answer, and the orchestrator receives the response and verified result before choosing again | User selected live peer responses (E5-005); prerecorded responses do not service the main trajectory. |
| Peer construction: settled | Copies of one peer checkpoint, with private task-relevant information varied in the main condition; matched information in the early baseline | Tests the orchestrator's inference about effective capability under different information access, not differences between trained weight sets. |
| Initial task construction: settled | Controlled private world information in possession and location tasks; exact instances and packet controls remain open | User accepted E5-054. Private rulebooks are retained as an additional family (E5-055); realistic briefings remain a further experiment idea. |
| Peer/domain count: settled | One orchestrator, two named peers, and two objectively gradable domains: possession and location | User accepted E5-004's counts and E5-054's first domains. The design tests identity-by-domain binding rather than a single global favorite. |
| Main trajectory | Proposed 40 rounds with independently balanced task domains; free choice from the first task without advance demonstrations is settled (E5-007) | Reveals how initial preference and internal state develop. Do not force equal exposure; actual observations per peer/domain are endogenous and must be logged. |
| Controlled discovery panels | Ten observations per cell, with 9/10 and 3/10 success histories | Calibrates structured measurements and counterfactual edits; these panels do not replace the online learning study. |
| Measurement frequency: settled | Examine candidate peer-skill representations at every turn; exact readouts, sites, and causal-fork frequency remain proposed | User requires turn-by-turn mechanistic investigation (E5-010). Structured readouts and causal tests need separate validation; token-word ranks remain secondary. |
| Retention strategy: settled | Bounded raw development captures, then live analysis with compact saved measurements throughout the main study | User accepted E5-047. Exact record fields and category-specific capture limits remain open. |
| Initial result-data allowance: settled | 1 GiB for one logical copy of new Experiment 5 run artifacts | User accepted E5-048 and requested a pre-run storage estimate (E5-049). Internal allocations remain proposed; extra physical copies and model/lens downloads are reported separately. |
| Delay | Immediate, 128-token, and 512-token neutral continuations; 512 is the primary delayed site | Distinguishes immediate availability from delayed accessibility. |
| Sample planning | 16 development scenario families; candidate 96 untouched evaluation families | Provisional planning counts, subject to development-based power and compute checks before evaluation. |

Selected model revision for all three roles: `9216db5781bf21249d130ec9da846c4624c16137` (G-010/G-017). Same checkpoint means the same architecture, tokenizer, and saved learned weights; role-specific contexts and private observations remain separate. The optional J-lens baseline remains proposed: `neuronpedia/jacobian-lens`, revision `e166deb5ad540932100ef803c8c9bcc2a5453694`, with its Qwen3-32B lens. Exact runtime, tokenizer, dependency, precision, template, and decoding identities must still be frozen.

The user accepted the pinned Qwen3-32B checkpoint, non-thinking mode, and seeded sampling for the orchestrator and both peers in the initial study, with matching sampler parameters for the two peers. This reuses the existing measurement/intervention infrastructure. Numerical sampler values, precision, and resource settings remain separate review choices. Non-thinking disables an explicit generated reasoning segment; activation measurement remains available, and conclusions are conditional on this mode.

Seeded sampling supersedes the original deterministic-decoding default for all three initial roles (G-011). The two peers' sampler parameters match and are fixed across information comparisons; they need not match the orchestrator's numerical settings. Sampling adds controlled randomness and needs a frozen policy, adequate independent sessions, and verified replay where claimed; numerical parameters and seed counts remain open. The assistant proposes recorded session seeds with separate random streams or derived per-turn seeds, preserving the main trajectory's random stream when diagnostics run. Do not restart an identical default seed each turn. Causal forks must test the registered policy or state the scope of any different one; paired random draws are a proposed variance control, not a guarantee of identical sampled outputs after an intervention.

**Additional experiment idea: thinking orchestrator.** At the user's request, retain a comparison enabling thinking for the orchestrator (E5-057). The assistant proposes holding the peer configuration fixed and leaving peers non-thinking, with matched task families/starting histories and a prospectively specified decoding/token-budget comparison. More generated computation can change the trajectory and complicate causal attribution; variability should be measured rather than assumed. Timing, scope, measurement sites, and resources remain open; this is not automatically part of the initial run.

The initial study uses one checkpoint (G-019), with a separately planned second-model replication retained as a later option. Its model, workload, and scheduling remain open. One-checkpoint development simplifies the initial assay but limits conclusions to that model; replication needs its own readout/intervention validation and resource estimate.

## 4. Approaches considered

The structured relational approach was selected in discussion. The main experiment is now an online delegation study. Controlled histories remain a separate calibration assay for identifying and validating candidate representations; they are not the primary behavioral workflow.

| Approach | Assessment |
|---|---|
| Supplied profiles plus name ranking | Useful historical baseline; repeats the central limitation of Experiment 3. |
| Live peer execution after each orchestrator choice | Selected by the user for the main study. Requires objective scoring and an independent quality reference; exact peer runtime and task setup remain open. |
| Online choices serviced from a frozen verified response bank | Considered but not selected for the main trajectory. Genuine prerecorded text could preserve sequential choice without full-state snapshots; reconstructing internal computation is a separate concern. |
| Separate controlled panels plus structured readouts and interventions | Retained for measurement discovery and causal calibration; successful transfer to online states must be demonstrated. |
| External memory or a newly trained peer-model module | Excluded: changes the system being studied rather than identifying what the frozen recipient already does. |

In the controlled calibration panels, frozen messages may originate from real peer-model calls, but their experimental identity and success schedule are controlled. The resulting claim concerns the recipient's inference from demonstrated performance, not the intrinsic competence of the peer-producing checkpoint. This distinction must remain in the paper.

## 5. Stimuli and ground truth

### 5.0 Early equal-capability baseline: agreed scope

Run this investigation before introducing capability differences. Its purpose is to measure presentation bias, concentration, and path dependence when neither peer has an established quality advantage. It is a control within Experiment 5, not Experiment 8 and not the primary capability-learning result.

- Initial equal-expected-quality baseline proposal: same peer checkpoint, system instructions, current task information, matching sampled-decoding parameters, and comparable source context; opaque identity and roster position are counterbalanced. Seeded peer sampling is accepted (G-011); independently recorded random streams and fresh task-only peer requests remain proposed ways to match the source setup while the orchestrator retains its interaction history. Verify equivalence rather than assuming shared weights guarantee equal effective inputs. Different realized outcomes permit an early-luck analysis and do not alone establish different underlying capabilities.
- Optional strict-clone assay: retain the earlier deterministic construction only as a separately labelled policy check requiring its own planned scope. It is not the initial sampled-peer default (E5-001).
- The orchestrator makes every actual delegation. Record its initial choice before feedback, then the entire sequence. Do not enforce a 50/50 allocation or a LinUCB exploration policy.
- Counterbalance literal names, roster order, and the attribution of matched early performance evidence across independent sessions. For a causal early-history comparison, fork consistent histories; do not relabel a verified success as a failure while retaining the same answer.
- Report preference by time window and domain, maximum share, entropy, streaks/switching, and success/failure-conditioned staying. Separate within-session concentration from the distribution of favored identities across sessions. A pooled 50/50 allocation can hide strong commitment in every session.
- A repeated preference is not automatically a mistake when expected quality is equal. Call it harmful lock-in only when a later condition supplies credible contrary evidence or a genuine quality disadvantage and the policy fails to respond. Even then, distinguish not receiving evidence from ignoring evidence it received.
- Preserve baseline findings even if the later capability/representation discovery fails. Do not tune name sets, starting exposure, or sampling to erase an inconvenient baseline before evaluation.

The user has approved an early baseline, not its exact sampling temperature, session count, or round horizon. Those remain proposed settings to freeze through development and resource planning. Numerical baseline tests are secondary/descriptive unless prospectively added to the primary registry; no claim of individually independent turns is permitted.

### 5.1 Task and communication construction

The user selected possession and location as the first two domains (E5-054), with one orchestrator choosing between two peers (E5-004). Use controlled private world information: which object a person holds, and where a person or object is. Exact entities, world constraints, and task templates remain open. Keep peer weights and decoding policy matched and independently vary task-relevant information available to each peer in each domain. The receiver observes performance rather than receiving the private packets or labels such as expert. Exact packet contents, information coverage, length-matching controls, and persistence policy remain proposals for focused review (E5-008). Verify on independent task instances that the information manipulation produces the intended performance difference; an intended advantage is not automatically an actual advantage. Each domain needs multiple task templates, fresh instances, and held-out template families. Generalization to arbitrary domains is outside this two-domain study.

The user accepted E5-051's complete-versus-partial correct-information construction, initially framed as private rules and now applied to the selected world-fact tasks. This incomplete-information comparison uses no false facts; misinformation is tested separately under E5-053. The early equal-capability baseline has matched coverage. Independently varied peer/domain assignments still need specification. Verify the actual performance effect and retain uncertainty or abstention as explicit outcomes rather than forcing incorrect answers. Exact coverage fractions and matching controls remain open.

For the development pilot, the user accepted E5-008's rates: correct target information on every scheduled reference instance, correct target information on half the incomplete-condition instances with the target omitted on the rest, and correct versus false target information on half each of the incorrect-information instances. They describe private inputs, not forced answer accuracy or guaranteed observed exposure under free choice. Fix input schedules prospectively by peer/domain and measure actual responses and quality. False-value generation, packet matching, and final evaluation rates remain open. Scripted calibration panels do not define these live input rates, and pilot approval does not freeze confirmation settings.

The user clarified that the chair/ball alternative means supplying false ball information to a peer while chair is world truth (E5-052), and also considers correct-but-incomplete information useful. This can be an Experiment 5 reliability manipulation if actual performance differences and learned preferences are the outcomes; it does not itself establish a source peer's belief. Belief-attribution claims would need the additional validation in Experiment 7. Exact task content and any single-object world constraint remain open. Repeated fresh instances and defined task families are needed to test learned peer-specific reliability rather than only a single fact.

The user approved E5-053's separate complete-correct versus incomplete-correct and complete-correct versus incorrect-information comparisons within the same two-peer/two-domain framework. Keep actual answer quality, preference, internal readouts, and causal outcomes distinct. Both comparisons must enter the final workload, registry/power planning, and pre-run storage estimate within the approved allowance or a prospectively revised budget. Their exact counts and task identities are not selected.

The possession/location selection adapts the earlier private-rule proposal toward controlled private facts. Fresh instances, independently manipulated peer/domain information, and objective reference answers support the information-conditioned reliability question. Findings remain bounded to these task families rather than establishing broad reasoning expertise.

**Additional experiment family: private rulebooks.** At the user's request, preserve tasks using private rulebooks, such as controlled transformations or rule-governed problems (E5-055). The assistant proposes investigating this as an extension after the first possession/location assay. Exact tasks, timing, counts, and resource requirements remain open; this does not silently add more domains to the initial run.

**Further experiment idea: realistic task briefings.** At the user's request, retain a future version using realistic project documentation or task briefings to test whether findings from the controlled tasks transfer to naturalistic collaboration. This is outside the initial study. Exact tasks, scoring, and resource requirements remain undecided.

For the controlled calibration panels, build a candidate response bank before recipient inference. Retain exact peer-generation provenance and verify every answer programmatically. Correct and incorrect answers must be coherent attempts with matched output format; empty, malformed, and nonsensical outputs do not stand in for low capability.

The user accepted selected-peer feedback (E5-017): the recipient sees the current task, chooses, then receives its selected peer's attributed response and verified correctness outcome. The unselected peer's answer/outcome stays hidden. In the separate calibration panels it sees the prescribed individual observations. It does not receive accumulated counts, the words strong/weak/expert, a capability posterior, source model brands, or the hidden assignment table. Source-family, wording, answer length, and formatting must not identify the capability condition.

World truth is hidden before the relevant delegation, but correctness feedback can reveal the queried fact afterward. In an exclusive binary task, even an incorrect label can identify the other answer. Do not describe this feedback as leaving all world facts unknown. The user accepted fresh scenes with new hidden target facts each round while the same orchestrator conversation and peer identities continue (E5-056). Verify that prior feedback and public task metadata do not already determine the new answer. The intended transfer is learned peer/domain reliability across new questions, rather than repeated delegation for an already-known fact. Exact scene-generation parameters remain open; changing scenes does not reset the recipient or remove its history.

Freeze the candidate bank, sampling rules, correctness checks, and any length matching before subject inference. If the bank cannot support the stated controls, report stimulus-construction failure and revise before evaluation; do not silently substitute new stimulus rules after results exist.

### 5.2 Capability matrices and binding controls

Represent the experimental state as four cells: `(peer 1, domain 1)`, `(peer 1, domain 2)`, `(peer 2, domain 1)`, and `(peer 2, domain 2)`.

The controlled calibration bank contains all 16 independent strong/weak assignments across these cells. It includes equal-capability, globally stronger-peer, complementary-specialty, and reversed-specialty cases. Complementary profiles alone are insufficient: they would allow one cell to be inferred from another.

Within each scenario family, use the same task/answer bank and nuisance settings across the 16 assignments. Evidence ordering is balanced so latest outcome, longest answer, and first speaker do not identify the stronger peer.

Use these diagnostic comparisons:

- Swap peer attribution while preserving the total task/answer/outcome multiset. A binding measure must follow the changed attribution.
- Swap domain-specific assignments while preserving each peer's overall success total where possible. A global trust score cannot explain this comparison.
- Toggle one cell while keeping the other three cells fixed. This supplies conditional discovery pairs and content-specific donors.
- Include equal-state trials. Correct handling includes ties; the recipient must not always invent a preferred peer.

The user accepted E5-003's principle of independently varying peer-name labels, roster order, and peer/domain information assignments across sessions while keeping identities stable within a session. This makes name-linked, position-linked, and evidence-dependent preferences separately interpretable; report the initial biases as outcomes rather than removing them through pooled averaging. The orchestrator remains free to choose throughout. Do not relabel identities midway through a main session or reorder feedback generated by its choices.

The proposed numerical allocation is two name arrangements, two roster orders, and two evidence-order patterns balanced across eight-family blocks. Evidence-order controls concern controlled panels or prospectively scheduled inputs, not retroactive edits to online feedback. Each family receives one frozen combination, held fixed for its controlled counterfactual histories. This bounds cost and supports average effects across these settings; it does not power every nuisance interaction. Literal name-pair pools and communication/task template families are disjoint between development and evaluation under the proposed split policy. The control principle is accepted; numerical allocation and exact split construction remain under review.

### 5.3 Causal anchor profiles

For calibration add eight causal profiles per family: four target peer/domain cells, each in a strong and weak version. Set the other peer in the queried domain to 6/10 successes. The target's 9/10 versus 3/10 therefore reverses the evidence-supported delegation while the other three cells stay fixed. Balance the other-domain states independently.

These anchor profiles avoid a donor swap that merely changes a strict preference into an ambiguous tie. They supplement, rather than replace, the independently varied core matrices and constant-total binding controls. Include their middle-valued capability cells in development calibration of the continuous readouts.

### 5.4 Reference quality and information available to the orchestrator

For controlled panels, labels follow demonstrated, objectively checked performance. For online trajectories distinguish three quantities: the peer's independently assessed capability, the evidence actually observed for that peer, and the recipient's elicited assessment. Do not equate them.

Log counts and successes per peer/domain for analysis only. An unselected peer can remain unobserved; represent that status explicitly, not as zero competence or an incorrectly decoded hidden label. A uniform-prior success posterior may be used as an external analysis reference for observed evidence, but it is not supplied as memory and is not assumed to be the model's true belief.

Task reward is objectively checked correctness. A regret/reference-best metric requires known controlled response probabilities or an independent, held-out assessment of both peers under matched tasks/conditions. Never define the best peer only from the peers the orchestrator happened to select. If the reference difference is unresolved, report success and preference with quality uncertainty; do not invent a known best peer. Equal-quality trials have no uniquely correct preferred identity.

The user accepted independent quality audits of both peers for the live study on 2026-11-09 (E5-009): fresh matched tasks under the same private-information, context, and sampled-decoding conditions, scored objectively by domain with uncertainty. Results remain analysis references, separate from the orchestrator's selected-peer feedback. Audit requests do not advance main histories or supply extra evidence to the orchestrator. Development and prospectively frozen evaluation audit pools are separate; counts, repetition, matching, and resource costs remain proposals for the pilot plan.

### 5.5 Longitudinal measurements accepted in review

| Metric family | Measurement | Interpretation |
|---|---|---|
| Preference trajectory | Rolling per-peer allocation by domain, including the initial no-feedback choice | Emergence and evolution of preference. |
| Concentration | Entropy, maximum share, streaks, and switching | Allocation concentration; alone it does not establish learning or a failure. |
| Feedback response | Staying/switching after verified success versus failure, controlling domain/task difficulty; matched counterfactual-history contrasts | Evidence sensitivity. Observational win-stay/lose-switch is not itself a causal mechanism. |
| Learning/adaptation | Accepted primary structure: early-to-late quality-adjusted routing improvement plus late performance above random delegation; full curves, identifiable regret, and response after contrary evidence remain reported | Whether decisions improve and update; whole-session quality alone is insufficient for the user's decision-process question. Equal-quality baselines do not support a best-peer learning score. |
| Relational state | Held-out peer/domain decoding, uncertainty/coverage, and changes after each result and delay | Information available in the native state; no inference of causal use from decoding alone. |
| Causal choice effect | Donor-directed change and selective loss/rescue against unedited and matched random controls | Whether the identified association contributes to subsequent actual choices. |

Plot these over time, with uncertainty across independent sessions/families. Freeze window widths, analysis checkpoints, phase definitions, and curve summaries before evaluation. Do not search for the round at which a p-value happens to cross a threshold.

## 6. One complete online trajectory and its diagnostic branches

1. Introduce the peers and task formats without capability descriptions or advance performance demonstrations. The user selected free choice from the first task (E5-007). Record a passive checkpoint and the first actual delegation before any performance feedback. Do not require equal initial exposure. A first choice is evidence of preference under the tested prompt/settings; identity/order controls and neural/causal evidence are needed for stronger interpretations.
2. Supply a fresh scene with a new hidden target fact, using a frozen, domain-balanced task schedule independent of the orchestrator's chosen identity. Retain the same orchestrator history and peer identities. Make a passive pre-decision measurement and let the orchestrator select which peer acts. The selected peer executes the actual task live using its assigned private world information and generates a fresh response.
3. Show the selected response and its truthful checker result. Append both to the ongoing conversation, measure a neutral post-feedback state, and continue to the next round. The next delegation is genuinely the orchestrator's own choice. No controller forces its allocations or supplies accumulated capability estimates.
4. Repeat throughout the session, logging which evidence each choice makes available. Never assume ten observations for every peer/domain in an adaptive session. Proposed horizon is 40 rounds; the final horizon is frozen after development, not selected from evaluation learning curves.
5. Examine candidate peer-skill information at every turn, as required by the user. Passive pre-decision and post-feedback sites are the proposed timing; exact capture and readout details remain open. Separately, at fixed checkpoints, proposed after rounds 4, 8, 16, 32 and the final round, fork delayed and diagnostic continuations from the untouched state. Record immediate, 128-token and 512-token retention sites; ask domain-specific delegation and four-cell capability questions on separate forks. These diagnostic outputs never enter the main conversation or determine later main-run exposure. The sparse fork schedule does not limit the every-turn mechanistic measurements; full causal testing at every turn has not been approved.
6. In a separate updating phase, change a predeclared peer/domain condition or provide predeclared contrary evidence, and measure the entire subsequent adaptation trajectory. Verify the performance manipulation independently. Do not declare failed updating merely because an unselected peer improved invisibly. Matched confirming/counterfactual branches isolate evidence effects from time and additional text.
7. Use actual online checkpoints for causal transfer tests. Keep the next task and pre-edit prefix fixed across edit/control forks. These forks estimate immediate next-choice effects. Claims about an edited policy's entire future trajectory require a separate rollout estimand because later choices alter its future evidence.

At a diagnostic fork, capture the internal association before asking the assessment question. Training labels/readout selection must not be based on the eventual selected identity alone; that would risk rediscovering answer preparation. Compare with task, name/order, last-choice, recent-feedback, and ordinary output baselines.

The user selected A/B-only delegation for the initial study on 2026-11-11 (E5-040): the orchestrator chooses one roster peer each round. Proposed failure handling remains zero task reward and no selected-peer feedback for invalid model choices, retaining the failure without fallback identities or replacement reprompts. That failure policy is not approved by the action-space answer. Report invalid choices separately and allocation over all scheduled rounds, plus valid-conditioned shares if useful. Selection entropy is computed over valid peer choices, always alongside validity coverage, and is undefined if no valid choices exist. A selected peer's uncertainty or abstention is a distinct task-response outcome; this action policy does not prohibit it. Infrastructure retry rules remain separate proposals.

Strict JSON applies to diagnostic probes. The online action schema names one roster peer. Abstention is deferred, including the earlier proposal for three-action diagnostic delegation branches. A diagnostic capability `tie` is meaningful in equal-evidence assessments; delegation choices remain A/B. Save malformed replies and cap hits, count them in end-to-end outcomes, and never silently replace them with a successful policy's choice. No model grader is required for primary task scoring.

The fixed-history acquisition, reversal, and anchor panels remain separate calibration experiments. Their precise counts are known by construction, but their success does not establish that the free-delegating orchestrator spontaneously forms the same state. Demonstrate online generalization without refitting on evaluation trajectories.

## 7. Native computation and retention-route tests

Each online trajectory preserves the recipient's native computation and cache across actual delegation, evidence, neutral continuations, and diagnostic branches. For a chosen diagnostic checkpoint, C2 means its post-evidence neutral state and C3 means the delayed branch from that state; it no longer implies completion of a fixed ten-observation history. Offline activations are read-only artifacts, not feedback supplied to the recipient.

Keeping the current execution state available during a session is distinct from saving a full state snapshot at every turn. The user prefers avoiding routine full-state snapshots because of storage cost. Exact retained measurements are under review in E5-047; this preference does not authorize dropping information needed for a claimed causal comparison or assuming replay equivalence without validation.

The existing stateless API is useful for reproducible replay, but a fresh request does not inherit an earlier intervention. Implementation must either support segmented continuation/forking of the native computation or replay the exact prefix with all prior edits and masks reapplied. Verify equivalence against uninterrupted native execution before interpreting any memory result.

The user accepted attention-blocking diagnostic branches and irrelevant-text controls for the contextual-versus-carry-forward question on 2026-11-09 (E5-029). These run alongside the main online trajectory. Blocking original evidence does not erase computation already carried into later state; that later availability is what the branches test. Extra model computation and perturbation sensitivity are tradeoffs. Exact delays, checkpoints, token intervals, and matching tolerances remain proposals. The accepted route design comprises these five conditions:

| Route | Condition | Scientific purpose |
|---|---|---|
| R0 | Ordinary access to all preceding messages | Establishes native contextual performance. |
| R1 | Block attention to original evidence-message positions only when the final question is processed | Tests whether later representations can support the answer without direct question-time retrieval. Earlier rereading during the delay remains possible. |
| R1-control | Matched question-time blocking of irrelevant positions | Measures generic sensitivity to the mask operation. |
| R2 | Block original evidence positions immediately after C2 and throughout the delay and questions | Stronger test of information carried forward from the acquisition checkpoint. |
| R2-control | Matched early blocking of irrelevant positions | Controls early masking and reduced available context. |

The original evidence interval includes every task prompt, peer attribution, peer response, verifier outcome, and any repeated quotation of those outcomes; blocking only response tokens would leak the labels through the checker text. R2 retains the neutral C2 receipt positions, normal subsequent positions, roster information, and question. R1/R1-control masks begin with the first question token and remain active for the entire question prefill and every generated answer token, including the complete JSON response. R2/R2-control masks remain active from immediately after C2 through every delay, question, and answer token. Forking or replay must never reopen blocked evidence positions during decoding. No performance summary or diagnostic answer is inserted into the retained interval. Exact allowed/blocked token intervals are frozen and verified. Irrelevant controls must match the best feasible number, age, and attention opportunity of removed positions; use additional irrelevant history if necessary. Save intended masks, actual masks, and realized attention effects. A position count match alone does not guarantee a perfect control.

These masks are experimental perturbations, not changes to the unedited model's architecture. If controls themselves cause substantial failures, or usable matching cannot be achieved, label this route assay inconclusive rather than declaring memory absent.

R1 success supports later availability. R2 success is stronger evidence of carry-forward information, but is not alone proof of an abstract peer model: later KV entries may redundantly encode message content. Relational generalization and selective interventions must provide the additional evidence.

## 8. Discovering the representation before claiming it exists

### 8.1 Structured readouts

A readout is an analysis measurement computed from selected internal activations; it is not an external memory supplied to the subject. The user accepted an orchestrator-focused initial readout of peer-skill representations. Initial means a candidate instrument developed and validated during the bounded pilot, not a readout confined to the first interaction turns. Once frozen, apply it during every main-study turn at the registered sites and retain compact results. Designated causal forks also execute while the relevant state is available. After the run, analyze retained scores, choices, outcomes, and intervention/control results; the compact archive does not support arbitrary new neural readouts of uncaptured main-trajectory activity without new model computation. The user accepted observed success rates as the initial calibration reference in E5-018; exact aggregation and missing-output scoring remain open. Recovering observed performance is not sufficient evidence of functional skill knowledge; identity/domain binding, updating, held-out use, and selective causal effects provide the stronger tests.

Fit the structured readout on development/calibration families and evaluate it without refitting at passive online checkpoints. For controlled panels, labels are observed success fractions, with balanced accuracy on their binary profiles. In online histories, label only the observed peer/domain evidence with its exposure/unknown status; never treat an unobserved hidden capability as knowledge the recipient ought to possess. Report alignment with an observed-evidence reference separately from calibration to independently measured future performance. Fit the linear decoder without probability constraints, clip estimates to [0,1] for primary scores and unknown-outcome bounds, and retain raw estimates as diagnostics. Only for the controlled binary core panels use the frozen observed-rate midpoint 0.6 as the classification threshold; this threshold is not a universal classifier for arbitrary online histories; do not optimize it on evaluation data. The decoder sees captured activations only, not the transcript, checker outcomes, task answers, hidden capability table, or future question.

The user accepted observed successes divided by observed attempts for each peer/domain as the first numerical online calibration reference on 2026-11-09, retaining exposure separately (E5-018). A 1/1 history has n=1 and is not a confident competence estimate. With no observations, the rate target is undefined; retain the cell in exposure/unknown coverage and the complete trial inventory rather than assigning zero ability. This is a calibration reference for observed evidence, not ground truth for the model's private belief. A smoothed reference remains a proposed optional secondary analysis; exact uncertainty, aggregation, and missing-output scoring remain open. Neither an unobserved cell nor an initial preference requires an abstention from the free-choice orchestrator.

Output heads refer to peer slots defined by the initial introduction and to the two domains. The frozen name map is used only to score later literal-name answers. Reversing a later choice roster does not redefine peer identity. Evaluation on new names checks transfer of the association rather than memorization of a small name bank.

The user accepted a linear probe as the first instrument (E5-020), fitted offline on the bounded development captures and frozen before the main study. Only the analysis instrument is fitted; Qwen remains frozen and receives no probe scores. Exact inputs, regularization, clipping/scoring, and validation thresholds remain proposals. Start with proposed single-position, single-layer residual readouts at the neutral checkpoint. This limits decoder capacity but may miss nonlinear or distributed information. Proposed development search: layers `{24,32,40,48,56}`, ridge penalties `{1e-6,1e-3,1}`, and causal ranks `{1,2,4}`. Any wider or multi-position search is a documented development amendment before evaluation; failure of this bounded search does not establish absence of a distributed state.

The default candidate causal bases come from conditional activation differences between matched cell assignments. Distributed Alignment Search is a documented alternative, not part of the initial implementation; activating it requires a prospective development amendment before evaluation. A probe's separating weight is not automatically an encoding direction. Select bases by held-out development behavior and specificity, not by interpretability of their token names.

### 8.2 Baselines and leakage controls

Use grouped development folds that keep an entire scenario family together. Fit centering, regularization, any dimension reduction, and conditional contrasts inside each training fold. The proposed first validation checks transfer to held-out scenario/name families and identity/domain binding under preserved outcome totals (E5-025). Targets are available for supervised fitting and scoring, never as extra decoder inputs at measurement time. Decodability is evidence of accessible information; later selective interventions test whether it contributes to delegation.

Required comparisons are:

- A label-prior baseline, an identity-blind performance-total baseline, and cold-start name/position measurements.
- Matched linear logit/J-lens feature baselines and untransformed residual measurements, with equal checkpoint positions and comparable fitted capacity. Higher raw MRR alone is not an advantage claim.
- Peer-attribution and domain-assignment controls with preserved global outcome totals.
- Permuted-assignment control fits and text-only task/evidence baselines, clearly distinguished from activation probes.
- Unseen template families and literal names; separate reporting of all four cells and joint correctness.

Native J-lens watch words are secondary diagnostics. A collection of names, domain words, or intent terms is not a validated relational decoder. Multi-token verbalizers are optional secondary analyses, not a replacement for structured ground truth.

### 8.3 Development gates

Draft calibration feasibility thresholds are at least 95% valid behavioral outputs, at least 80% evidence-supported routing on unequal controlled-panel trials, and at least 75% macro balanced accuracy for the delayed controlled-panel readout, exceeding prior and identity-blind controls by at least 10 percentage points. Online state-transfer performance and the resulting action effects require their own frozen development checks; calibration thresholds do not automatically establish spontaneous online learning. These are proposed engineering/discovery thresholds, not significance tests or established power assumptions.

Select a causal site only if the target-cell association transfers across development families and the attribution controls work. Select the smallest effective dose preserving non-target task behavior and output validity. Freeze the selected layer, checkpoint position(s), basis, dose, donor rule, route masks, decoder, and all analysis code before opening evaluation outcomes.

If no candidate passes, record discovery failure. Do not manufacture a peer direction by choosing the best evaluation example, a peer-name token, or a larger unregistered search. Behavioral evaluation may still proceed under its frozen analysis; the mechanistic hypotheses remain unavailable or unsupported.

## 9. Causal intervention substudy

The user accepted removal, downstream clean restoration, and counterfactual donor patching to test whether candidate peer/domain information contributes to delegation on 2026-11-09, including unedited, similarly sized random-edit, and other-domain comparisons (E5-031/E5-033). Removal alone could cause generic disruption; a selective donor-directed shift and recovery against matched controls provide complementary evidence. Exact bases, sites, offsets, ranks, doses, donor inventories, matching geometry, frequency, and numerical success criteria remain proposals. First validate the following operational conditions on controlled causal anchor episodes under R0 and R2. Then apply the frozen method at preregistered online checkpoints; calibration and online effects are reported separately, not pooled into one mechanism success:

| Condition | Operation | What it tests |
|---|---|---|
| I0 | Unedited | Reference behavior and readouts. |
| I1 | Partial deletion of the development-identified target-cell component at C2 | Whether that pathway contributes to later use. |
| I2 | Matched random deletion | Whether disruption exceeds a matched perturbation. |
| I3 | I1 followed by a clean-donor component patch two blocks later | Whether a nontrivial downstream rescue restores use. |
| I4 | Counterfactual donor patch for the target cell, without preceding deletion | Whether changing represented content shifts delegation toward the donor's evidence-supported choice. |
| I5 | I1 followed by matched random restoration | Whether selective clean rescue exceeds matched energy restoration. |
| I6 | Matched random patch without preceding deletion | Whether the counterfactual donor effect exceeds a matched displacement. |
| I7 | Patch the same peer in the other domain, using a donor that changes only that other-domain cell | Whether the targeted-domain decision remains stable when a distinct association changes. |

I3 uses a clean donor from the identical episode, with no capability-cell changes. I4 changes only the target capability cell; the other three cells, task instances, names, order, and future question are held fixed. I7 uses a separate donor inventory: if the target is peer 1/domain 1, change only peer 1/domain 2, preserving both peers in domain 1. A donor for the competing peer in the queried domain is not an unrelated-state control because it could legitimately change the delegation. Require a nonzero realized I7 edit, and separately report whether the other-domain assessment follows its donor while the targeted-domain decision stays stable. For I3 use a separate clean run. For I4 report effects against both I0 and I6. Literal retrieval controls for intervention specificity use separately designated facts in unmasked coordinator text and compare edits within the same route condition; do not demand access to a source fact the route mask deliberately removes. All non-target behavior is measured independently; preservation of a readout explicitly constrained by the edit is not evidence of specificity.

Deletion I1 occurs at the frozen layer l and acquisition checkpoint C2. Rescue I3 occurs at layer l+2 at the same checkpoint, using a separately identified basis for that later layer. Counterfactual patch I4 also uses the later-layer basis, without prior deletion. Freeze the exact coordinate-replacement rule during development. Match I5 to the realized rescue displacement in the damaged recipient and I6 to the realized counterfactual patch displacement in the unedited recipient; these are separate control operations.

Match removal/patch norm, residual inner product, positions, and layers as appropriate, and record realized BF16 displacement. Same-site subtract-and-add cancellation is an engineering check, not scientific rescue. A mask or residual edit does not erase every redundant copy of the state; a null local effect limits that intervention's claim.

Required specificity outcomes are non-target cell judgments, non-target-domain routing, literal fact/identity retrieval controls, and JSON validity. Five percentage points is the proposed maximum acceptable loss against the relevant matched control. Preservation requires uncertainty bounds compatible with that margin, not merely a nonsignificant degradation test.

## 10. Metrics, hypotheses, and statistical significance

The independent unit is an online session/trajectory, or a matched scenario family containing those trajectories, assignments, and intervention forks. Sessions that share one source scenario form one family for inference. Controlled calibration panels retain their own family units and are not counted as additional independent online replications. Tokens, turns, four output cells, counterbalances, and repeated questions are measurements within a family. Aggregate the prespecified within-family contrasts before inference. Claims about generalization remain limited to the sampled task/template/name regime.

| ID | Primary contrast in the draft registry, not yet frozen | Expected direction | Meaning if supported |
|---|---|---|---|
| P1 | Held-out online relational-state prediction improvement over the frozen identity-blind baseline across registered checkpoints, plus attribution controls | Positive | Native states carry peer/domain-specific information beyond global performance or selection salience. |
| P2 | Online learning: late-minus-early quality-adjusted routing improvement in stationary unequal-quality phases, and late routing above the random reference | Both positive | Delegation improves with interaction rather than merely reflecting an initial preference. Equal-quality baselines are excluded from this claim. |
| P3 | Joint updating test: reversal reduces target-cell error relative to its revised observed rate, and reversal-versus-confirmation readings separate in the evidence-predicted direction | Both improvements positive | The measured association moves toward revised evidence rather than merely differing across branches. |
| P4 | R2 routing minus R2-control routing | Greater than -0.05, and R2 above chance | Carry-forward information remains sufficient within the stated loss margin when direct evidence access is blocked. |
| P5 | I1 routing accuracy minus I0 and minus I2, separately under R0 and R2; all four components required | All negative | Targeted deletion causes an actual loss and exceeds the matched random effect under ordinary and restricted access. |
| P6 | Donor-consistent routing under I4 minus I0 and minus I6, separately under R0 and R2; all four components required | All positive | The donor changes behavior toward its content relative to unedited execution and matched perturbation under both routes. |
| P7 | Routing accuracy under I3 minus I1 and under I3 minus I5, both under R2 | Both positive | Clean restoration produces partial recovery and exceeds matched random restoration. |

The user accepted P2's early-to-late improvement plus late-above-random behavioral criterion (E5-036). Exact windows, independent quality references, and numerical/statistical definitions still require freezing. Retain full turn-by-turn trajectories and their relation to internal readouts and causal tests; whole-session quality/regret is secondary and cannot establish the decision-process claim alone.

These are proposed revised registry entries for held-out online trajectories and their prespecified diagnostic forks, not a frozen registry. Controlled-panel results validate the measurement but cannot substitute for an unavailable online primary. Before evaluation, P1 must fix the online evidence-state target and unknown-cell scoring, and P2 must fix the independently measured quality reference, phase windows, and curve estimand. They replace the previous endpoint-only primary; the old 0.6 binary threshold applies only to calibration panels.

For P1 the identity-blind baseline is fitted on development data using task/domain format and aggregate outcomes without peer attribution. It is fixed before evaluation. The attribution gate requires the decoder to track relabelled associations on constant-total pairs; numeric gate thresholds must be frozen with the final registry. All four cells remain in the scoring inventory.

For P3 let `q_initial`, `q_reversal`, and `q_confirming` be the observed target-cell success fractions, and let `h_initial`, `h_reversal`, and `h_confirming` be the corresponding frozen decoder estimates. Compute two within-history quantities:

- Absolute improvement: `abs(h_initial - q_reversal) - abs(h_reversal - q_reversal)`.
- Evidence-aligned branch separation: `sign(q_reversal - q_confirming) * (h_reversal - h_confirming)`.

Both family-mean quantities must be positive for the primary updating claim. The first prevents both branches drifting away from the revised evidence while still producing a relative difference. The second controls nonspecific effects of processing more material. Report both effects and intervals, and report behavioral estimate changes and delegation updates separately.

P5 explicitly requires loss against I0, not just inferiority to I2; P6 explicitly requires a donor-directed change against I0, not just superiority to I6. P7 supports partial rescue, not full recovery to I0 unless a separately frozen noninferiority comparison establishes that stronger claim.

P2, P3, P4, P5, P6, and P7 are conjunctions: all stated component tests must meet the direction and significance requirement. Use the maximum component p-value as the valid intersection-union p-value for that registry entry, and retain all component estimates and intervals. Do not average the components into a score that lets one failure disappear. R2-only causal effects support conditional use under restricted access; causal use during ordinary unmasked execution requires the corresponding R0 evidence. Redundant retrieval can compensate for a local R0 edit, so an R0 null does not establish nonuse, but the joint stronger claim remains unsupported.

For P4 use a one-sided family-level noninferiority analysis at the stated margin and a conjunctive above-chance requirement. For the other mean contrasts use paired family-level sign flips, exact for at most 20 families and otherwise 50,000 seeded assignments, with the symmetry assumption explicit. Report family-bootstrap confidence intervals using 100,000 resamples and seed `20261108`; preflight the noninferiority procedure under null simulations before freezing it.

Control multiplicity at family-wise 0.05 across the frozen primary registry. Seven Experiment 5 entries are proposed here. The final registry must also state how primary claims across Experiments 5-7 are combined or separated into prospectively defined studies; this decision follows review of all three specs and precedes all evaluation runs. Missing or gated-off primaries remain in the declared family. Do not insert later successful tests while continuing to call the original family frozen.

Additional metrics are four-cell probability error, joint association accuracy, tie accuracy, invalid/cap-hit rates, retention curves, per-domain performance, predicted utility/regret based on demonstrated rates, literal-content retrieval, and pre/post-edit readouts. They are descriptive or explicitly labelled secondary unless added to the registry before evaluation. A p-value alone is not a positive mechanism result.

All assigned evaluation trials stay in the end-to-end denominators. Do not keep only baseline-correct, fluent, probe-correct, or valid survivors. Runtime failures are retried only under frozen technical rules, retain all receipts, and remain explicit unresolved outcomes if exhausted. Give conservative bounds for unresolved cells; report complete-cluster estimates separately.

## 11. Workflow, gates, and significance of each stage

The user clarified that the immediate task is experimental setup, with experiment execution deferred (E5-042, 2026-11-11). Prepare configuration, task/checker fixtures, prompt contracts, offline validation, and the implementation plan. Compute ceilings and actual pilot/evaluation workloads are future execution decisions, not prerequisites for this setup. The stages below describe later model-based development and evaluation; offline setup does not establish that their runtime, readout, or causal gates have passed.

The first offline milestone is described in the [setup guide](../../experiment-5-setup.md) and [implementation plan](../plans/2026-11-11-experiment-5-offline-setup.md). Its binary, equal-fact-count world fixtures and preview settings are assistant proposals recorded in E5-058 through E5-062; they are not frozen evaluation stimuli or evidence that neural gates have passed.

The user subsequently authorized the remaining implementation (E5-063). The [runtime guide](../../experiment-5-runtime.md) and [runtime implementation ledger](../plans/2026-11-11-experiment-5-runtime.md) describe the executable development/freeze/evaluation/report pipeline and offline checks. Native replay, controlled measurement, live transfer, and causal discovery still require validation on the actual pinned model. Runtime defaults are proposals recorded in E5-064 through E5-074. The current planning forecast exceeds the approved total storage cap; revise and freeze the workload/resource plan before launch rather than increasing the allowance implicitly.

| Stage | Work | Gate/result | Scientific significance |
|---|---|---|---|
| 0: protocol and fixture validation | Freeze model mode, task checkers, communication bank, splits, identity maps, masks, and trace sites. | Correct inventories, no answer/label leakage, native/replay equivalence. | Establishes that later effects concern the intended manipulation. |
| 1a: early baseline | Run equal-capability online sessions with counterbalanced identities/order. | Preference/concentration trajectories and stimulus equivalence checks. | Separates presentation bias and path dependence from genuine capability advantage. |
| 1b: behavioral discovery | Run active development trajectories and independent diagnostic forks. | Learning curves, exposure inventories, and explicit failures. | Establishes whether decisions adapt to the performance the recipient actually observes. |
| 2: relational discovery | Fit/check the structured readouts and assignment controls. | Freeze a valid candidate or report no identifiable candidate. | Locates candidate information while separating identity binding from global preferences. |
| 3: route and causal discovery | Test masks, doses, donor rules, random controls, and unrelated-state preservation. | Freeze a selective assay or report engineering/discovery failure. | Determines whether a carry-forward claim is testable with this intervention. |
| 4: power and resource freeze | Simulate the full primary success rule using independent-family variation; time representative development workloads. | Fixed evaluation N, registry, precision targets, compute cap, and source hashes. | Prevents unsupported sample-size promises and evaluation-driven stopping. |
| 5: untouched evaluation | Execute the frozen behavioral, update, route, and causal cells. | Complete inventory, or explicit incomplete-run bounds. | Tests generalization without retuning on the reported outcomes. |
| 6: reconciliation/reporting | Recompute every metric from immutable inputs, report all primaries and failures. | Reproducible analysis and bounded interpretation. | Supports a claim proportional to the evidence, including useful negative results. |

No held-out run begins until all gates required for its hypotheses and the combined primary registry are frozen. Extending a failed method after opening evaluation requires a new labelled exploratory study and fresh confirmation data.

## 12. Sample planning and compute accounting

The 16 development families and candidate 96 evaluation families remain proposals, not an 80%-power guarantee. Baseline, active trajectories, controlled measurement panels, and causal transfer tests need separately enumerated workloads. Estimate power and precision using variation across complete independent trajectories/families, including endogenous exposure and the entire primary success rule.

The earlier draft's 44,544-continuation estimate applied to its fixed-history assay and is superseded. It is not the budget for this active-delegation revision. Live peer responses and mechanistic measurement every turn are settled. Before evaluation, freeze the number and length of sessions, exact selected-peer runtime, task schedule, passive-measurement workload, diagnostic checkpoint count, all fork/control counts, retained activation inventory, donor-state extraction, retry rules, source-audit calls, and hard storage/token/time/call ceilings.

For a provisional T=40 session, the live main stream alone has T orchestrator action generations and T selected-peer response generations, plus any separately defined feedback/protocol calls. Every-turn passive measurements add activation/readout and retention work. Diagnostic capability probes, retention forks, alternate-response donors, source-quality audits, calibration panels, and causal conditions are additional calls. A calibration response bank does not replace live peer responses in the main trajectory.

Independent source-quality audits can assess both peers without showing unselected responses to the orchestrator. They are scoring artifacts, not an external memory or decision policy. Reuse only exact identical receipts and report that reuse. Freeze the final N, horizon, stopping rule, primary registry and compute cap before evaluation; do not stop when a preferred learning curve becomes significant.

The user approved E5-048's initial 1 GiB ceiling for one logical copy of new Experiment 5 artifacts across development, main sessions, and diagnostic/causal branches. The internal allocation remains an assistant proposal: 256 MiB for raw development captures, capacity for 500,000 compact records at 1 KiB each, 128 MiB for text/provenance, and 32 MiB for shared fitted artifacts and reports. This allocates about 904 MiB with about 120 MiB headroom; it is not a measured forecast, a final workload count, or a requirement to fill the allowance. Existing model/lens files and extra physical copies such as LFS caches are separate. The complete required case inventory and actual record serialization must fit before confirmation; needed cases or artifacts cannot be silently dropped. Lossless sharing of repeated text is proposed, with exact rendering verified. The detailed assumptions are in [E5-048](questioning.md#e5-048).

Before launching a run, display the storage estimate requested by the user (E5-049). Derive it from the scheduled workload and measured serialization, including all required output categories. Proposed display fields are expected new result bytes with a breakdown, the approved allowance, additional physical disk requirements, available space, output destination, and uncertainty. Proposed behavior is an informational notice for an already-authorized run within the allowance; an estimate above the ceiling requires a revised storage plan or approval of a larger budget before launch. Insufficient physical space prevents launch. The notice must distinguish estimated usage from the allowance. Its exact format remains proposed; no additional confirmation on every within-budget launch is assumed.

## 13. Interpretation of possible outcomes

| Outcome | Interpretation |
|---|---|
| Good delegation, no valid relational readout | Behavioral use of observed performance; the tested measurement did not identify its representation. |
| Decodable association, no selective causal effect | Information is available to the analysis, but its causal use is unestablished by these interventions. |
| R0 succeeds, R2 fails with stable masking controls | Consistent with dependence on access to original evidence; no support for the tested compact carry-forward route. |
| R2 succeeds, binding/causal controls fail | Later states retain useful information, but an abstract peer-specific capability representation is not established. |
| Binding, updating, R2 preservation, and the joint R0/R2 selective-loss and donor effects plus rescue generalize | Supports a bounded causal account of a retained peer-capability association in this model and assay. |
| Peer and matched non-social source versions behave similarly | Supports a source-general association mechanism; does not establish a privileged social mechanism. |
| Broad language/format degradation accompanies the edit | General disruption limits the proposed capability-specific interpretation. |

The final source comparison is an optional prospectively frozen extension, not a prerequisite for basic identity-by-capability binding. Without a matched non-social comparison, use peer-specific to mean information bound to a particular experimental peer; do not claim an exclusively social circuit.

A concentrated preference in the early baseline is a descriptive finding, not proof of irrationality or harmful lock-in. A successful controlled-panel mechanism that fails to transfer online is a controlled-assay result only. No outcome here alone establishes general collaboration competence, theory of mind, intrinsic source-model capability, or that J-space is uniquely necessary. A positive structured residual result with a null J-lens result is a measurement comparison, not a contradiction.

## 14. Required architecture work and artifacts

The Experiment 5 implementation is in `src/jpeers/exp5/`, with entry point `scripts/run_exp5.py`. It provides the capability-task generator, grouped relational analysis, exact native-token replay, source-position attention masks, fixed-basis/donor interventions, and runner/report. Its CPU and synthetic checks establish software behavior; they do not establish that the actual Qwen assay or representation has been validated. See the runtime guide for remaining deployment and scientific gates.

The runner must verify that an edit at a block output affects the intended downstream computation/cache; it must not assume it changes the same layer's already-computed keys/values. Any direct KV edits need their own capture, geometry, and equivalence checks. Native-cache and residual effects must be separately attributable.

Save a frozen protocol/configuration; task/checker and source-response manifests; every rendered prompt and token span; exact state labels and scoring identities; development folds; decoder/basis files; source and model hashes; selected capture arrays under the agreed retention policy; intended/realized masks and edits; donor identities; all raw outputs and retries; progress inventories; the primary registry; power/resource calculations; per-family effects; coverage/unknown bounds; and the final report.

The user approved E5-047's strategy: a bounded development collection of selected raw activations to discover and validate a readout, followed by compact measurements computed live at every main-study turn. Proposed record fields contain peer/domain scores, exposure/unknown status, optional word-rank diagnostics, and provenance; decoder/basis files are shared artifacts. Raw vectors are not routinely archived for every main-study turn. Necessary causal-study artifacts need their own bounded retention policy, with live use of donor states where feasible. Every-turn measurement remains required. Word lists alone do not establish peer-skill binding, and compact records limit retrospective analysis of uncaptured activity. The initial total result-data ceiling is 1 GiB; exact capture sites, record fields, category limits, and causal artifacts remain open and must be settled before evaluation. These offline records never become external memory supplied to the orchestrator.

The report must include acquisition/update/retention curves, association confusion tables, route comparisons, joint correctness/validity outcomes, and targeted/random/donor/rescue effects. A headline aggregate cannot replace these diagnostics.

## 15. Focused review status

The user has accepted the early equal-capability baseline, active orchestrator delegation, longitudinal preference/evidence-response metrics, and structured relational/causal investigation. These choices are incorporated here; the research goals need not be revalidated as a whole.

The peer-weight construction is answered: same checkpoint, with different task-relevant information. The user also approved separate correct-versus-incomplete and correct-versus-incorrect comparisons, one orchestrator with two peers, possession and location as the first domains, live peer responses, free choice from the first task without advance demonstrations, fresh scenes with continuing orchestrator history and peer identities, selected-peer answer/correctness feedback, mechanistic examination at every turn, an orchestrator-focused initial peer-skill readout, early-to-late behavioral improvement plus late performance above random, bounded raw development captures followed by compact live-analysis records, an initial 1 GiB result-data allowance, and a pre-run storage estimate. Private rulebooks are retained as an additional experiment family with timing/workload open; realistic task briefings remain a further experiment idea. The user emphasized emerging peer preference as a behavioral goal and prefers avoiding routine full-state snapshots. Do not re-ask those choices. Remaining focused choices include feedback formatting, exact world instances and information contents, coverage fractions, precise readout/scoring definitions, early/late windows and quality references, internal storage allocations and retained fields, matching controls, diagnostic/intervention schedules, numerical horizon/sample sizes, and the final primary registry and compute envelope. They are recorded in [questioning.md](questioning.md#experiment-5); none is a blanket request to approve this document. The exact mechanism must first be discovered and validated; this draft does not presume a peer-state representation already exists. Experiment 6 and 7 specifications follow their own reviews, with the final cross-experiment registry reconciled before execution.

## References and local evidence

- [Current MACE delegation and homogeneous-agent controls](https://arxiv.org/html/2607.11250v2). Motivation for the early preference baseline; its algorithm does not replace the orchestrator here.
- [Historical A-D planning document](../../jeffrey-exp-plans-new/peer-modeling-next-experiments.md).
- [Audited completed results and causal limitations](../../2026-11-07-causal-followup-design.md).
- [Experiment 4 specificity analysis](../../2026-11-07-exp4-specificity.md).
- [Native generation runtime](../../../src/jpeers/runtime.py) and [recording/intervention schema](../../../src/jpeers/schemas.py).
- Feng and Steinhardt, [How do Language Models Bind Entities in Context?](https://arxiv.org/abs/2310.17191). Prior evidence for causally testable entity-attribute binding; not a result about this assay.
- Geiger et al., [Finding Alignments Between Interpretable Causal Variables and Distributed Neural Representations](https://proceedings.mlr.press/v236/geiger24a.html). Methodological basis for optional development-only interchange alignment.
- Todd et al., [Function Vectors in Large Language Models](https://arxiv.org/abs/2310.15213). Related evidence for abstract representations derived from demonstrations; not proof of a peer model.
- Hewitt and Liang, [Designing and Interpreting Probes with Control Tasks](https://aclanthology.org/D19-1275/). Motivation for probe controls and limits of decoding alone.
