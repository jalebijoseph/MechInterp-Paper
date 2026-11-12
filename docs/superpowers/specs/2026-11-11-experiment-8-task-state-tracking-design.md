# Experiment 8: partner contribution and task-state tracking

> **UNAPPROVED DRAFT — pending focused review, 2026-11-11.** This document proposes a study design, not a preregistration or authorization to run models. Unsettled choices remain assistant proposals. A CPU-tested pilot of the behavioral and passive-readout layer exists (§9); its existence approves none of the settings below. Questions are left for later review in [questioning.md — Experiment 8](questioning.md#experiment-8).

## 1. Question and settled boundaries

**Besides J-space, does an orchestrator maintain any form of information on who owns specific tasks and what each peer contributed?**

Experiment 8 instantiates historical Experiment D. Experiment 5 concerns learned capabilities, Experiment 6 descriptions versus demonstrated results, and Experiment 7 peer beliefs; Experiment 8 is the most observable layer beneath them: who was assigned what, who supplied which contribution, who has seen it, and what state the work is in. It is not a prerequisite for Experiments 5–7 and does not reuse their unapproved defaults by reference.

"Besides J-space" sets the measurement scope. The question is answered first by behavior (separately elicited structured reports and next actions) and by representation measurements that are not J-space (raw residual decoding, logit-lens and output baselines), with J-lens readouts as one compared instrument. A J-lens null does not answer the question negatively; a J-lens success does not establish anything a residual or behavioral measurement does not also show.

The target is the orchestrator's native computation in one continuing episode. No external task tracker, controller-supplied state table, or summary enters its context. Ordinary contextual/KV access and carry-forward after source access is restricted are separate claims (§6). A text summary or reset creates a different system and is not evidence that the original state persisted.

Collaborative state tracking is not theory of mind. Knowing that a peer was assigned a task does not mean modelling what the peer believes; access labels here record which messages reached a peer, not the peer's mental state.

## 2. Proposed approach and participants

Proposed recipient: the pinned Qwen3-32B checkpoint and lens used by the audited follow-up, native HF/PyTorch BF16 with its non-thinking template. Qwen3-8B is a proposed replication; tokenizer, runtime, decoding, and cache policy must be frozen.

Proposed core: two opaque peers, two subtasks per project (a target and a matched distractor), and frozen scripted peer messages carrying exact contribution codes. Scripted receipts give exact, objective labels for content, provenance, ownership, and access, and they make counterfactual branches identical outside the manipulated event. The cost is ecological: the study measures tracking of stated workflow events, not of contributions peers actually produced.

Alternatives awaiting review: live peers generating contributions from private briefs (as in the audited Experiment 1 handoff assay), with codes appended as coordinator metadata; or a frozen bank of genuine peer receipts indexed by assignment. Both raise label-audit burden. Live peers are a proposed extension once the scripted assay passes its gates.

## 3. Ground truth and identifiable contrasts

Maintain separate labels for the target subtask after each event:

| Quantity | Operational label and limitation |
|---|---|
| Owner `O` | Peer currently responsible, under the latest assignment or reassignment message. |
| Content `C` | Set of contribution codes submitted for the subtask. |
| Provenance `P(code)` | Peer who submitted the code; null when it appears only in a neutral document. |
| Access `A(code)` | Peers a message carrying the code reached (all peer messages are private to the orchestrator unless forwarded). |
| Status `S` | assigned, in progress, submitted (unverified), complete (verified), blocked, or conflict. |
| Next action `N` | wait, query, verify, integrate, reassign, or resolve, under a frozen rubric mapping from `S`. |
| Recency/frequency | Last peer speaker and per-name mention counts, recorded per branch. |

These labels coincide in ordinary workflow: the owner usually submitted the code and spoke last. The design therefore includes dissociation conditions:

| Condition | Separates |
|---|---|
| reassigned (blocked first peer leaves a partial code; task given to the second) | owner ≠ provenance; owner = last speaker |
| conflicting (non-owner also submits a code) | provenance of the probed code ≠ owner; last speaker ≠ owner |
| document (code appears in neutral shared notes) | content present, provenance null |
| assigned silent / in progress / blocked | owner with no content |
| unverified vs completed | identical content and provenance; status differs only by a verification receipt |

`N` is a rubric label, not an independently true action; report it separately from the five factual outcomes. Contrasts where labels remain aliased (for example owner and last speaker in `reassigned`) are marked unidentifiable for that explanation rather than counted as evidence against it.

## 4. One complete episode

1. Introduce identities, the project, and both subtasks. Capture a neutral checkpoint before any assignment.
2. The orchestrator assigns the target to one peer and the distractor to the other. The distractor is completed and verified, matching name exposure across peers.
3. Scripted peer and system receipts put the target into its condition (§3). Capture a passive state at the final token of a fixed neutral receipt after each event, before any question.
4. Insert balanced irrelevant messages (proposed: none, 3 messages, and a 512-token continuation, the last being the proposed primary delayed endpoint). Fillers name no peer, code, state, or action, and are matched across branches.
5. Fork the exact pre-question state independently for each outcome: content, provenance, owner, access, status, next action. Each fork asks one strict-JSON question; no answer enters another fork or the main episode. A no-peer control asks owner and content before any assignment.
6. Continue with an update branch (for example, a verification receipt arrives, or the owner is reassigned) and a matched no-information continuation; capture and fork again. Measure movement toward the new label, not any change.
7. Replay the family under name/role swaps, roster order, subtask swaps, paraphrased receipts, and question formats. Decision-site captures remain secondary.

## 5. Matched controls and passive measurements

Cross literal names with roles so that every name is owner and non-owner within a swap pair; averaging an owner margin over the pair cancels fixed name priors. Rotate name pairs across projects and keep roster order fixed within a pair. Counterbalance target/distractor order and receipt length where semantics permit, and record dependencies that cannot be removed (reassignment necessarily mentions the new owner last).

Measurement families, compared on the same grouped units:

- **Behavioral:** the six independent forks, scored exactly against §3 labels.
- **Residual decoding (proposed primary representational instrument):** a regularized linear structured decoder over passive captures, predicting owner, provenance, and status per subtask; shared entity features fitted without evaluation labels so unseen names do not need their own classifier.
- **J-lens, logit-lens, and output:** band-mean reciprocal ranks of each roster name at the passive site, as in the audited follow-up. Owner margin = owner minus other peer.
- **Shortcut baselines:** last-speaker, most-mentioned-name, and last-assigned-name rules, plus question-time logit/output.

Select sites, layers, rank, and regularization within development only; freeze one candidate or report that none was found. Raw-residual success with J-lens failure is a method comparison, not evidence that J-space lacks the information.

## 6. Retention and the stronger carry-forward test

Use the route definitions of Experiment 7: ordinary native access `R0`; masking of the source receipt positions at question time `R1`; and `R2`, which also blocks later positions from retrieving the source receipts during the delay. Allowed carrier positions, mask timing, and a sham mask are fixed in development. Native continuation or exact verified replay is required.

A code that must be reported literally can survive only through retrievable copies, so `R2` is applied to ownership, provenance, and status, not to literal content recall. Validate that blocked receipts are unreachable through the intended edges while unmasked facts and output formatting remain usable. A proposed five-point noninferiority margin against a matched route control requires an uncertainty bound and adequate absolute performance; a nonsignificant degradation test does not establish retention.

## 7. Selective causal assay (conditional)

Proceed only if development finds a reproducible ownership or provenance candidate that transfers to unseen names and templates. Build donor pairs that change only the target cell (for example, owner A versus owner B with identical content and status, or provenance A versus B with identical owner), holding the distractor fixed.

Conditions follow Experiment 7: unedited; targeted deletion; geometry-matched random deletion; donor-directed patch; matched random patch; deletion with downstream clean rescue; matched random restoration. Require loss versus unedited and random deletion, donor-directed change versus unedited and random patching, and rescue over damaged and random-restoration runs. Preservation checks cover the distractor's owner, literal code recall, status of the other cell, and format validity, with proposed five-point margins tested with uncertainty. An `R2`-only effect establishes conditional use under restricted access.

## 8. Metrics and proposed significance registry

Score every assigned fork; invalid JSON, refusals, and cap hits count as errors under a frozen rubric. Report per-outcome accuracy by condition and delay, joint all-outcome correctness, confusion matrices for owner versus provenance in the dissociation conditions, and no-peer null rates.

The independent unit is a project family containing all conditions, swaps, delays, forks, and donors. Proposed primaries:

| ID | Required contrast | Scientific significance |
|---|---|---|
| P1 | Owner and provenance fork accuracy exceed the best shortcut baseline in `reassigned` and `conflicting`, at the primary delay. | Behavioral tracking beyond recency and frequency. |
| P2 | Passive pre-question residual decoding of owner exceeds identity-blind and shortcut baselines on held-out families, with swap-averaged margins. | Ownership is available before it is asked, without J-space. |
| P3 | In dissociation conditions, decoded owner and decoded provenance point to different peers as labelled. | Ownership and contribution are separately represented. |
| P4 | Update branches move status/owner decoding toward the new label beyond the no-information continuation. | Evidence-sensitive updating. |
| P5 | `R2` owner/provenance behavior is noninferior to its route control at five points and above its task baseline. | Bounded carry-forward availability. |
| P6 | Targeted deletion and donor patch effects with rescue, as in §7, under `R0` and `R2`. | Causal use. |

J-lens owner margins (and their logit/output baselines) are reported as secondary descriptive measurements at the same sites, comparable with Experiments 1–3. Propose family-wise 0.05 with Holm correction over the frozen registry; reconcile its relationship to Experiments 5–7 before evaluation. Inference follows Experiment 7: exact sign flips up to 20 families, otherwise 50,000 seeded assignments, 100,000 family-bootstrap resamples, proposed seed `20261108`. Conjunctive entries use the maximum component p-value.

## 9. Stage gates, pilot, and deliverables

| Stage | Gate and consequence |
|---|---|
| Fixture audit | Labels, dissociation coverage, filler leak checks, and probe parsers verified; failure blocks dependent claims. |
| Behavioral pilot (`R0`) | Forks are parseable and above shortcut baselines in at least the non-dissociated conditions; if not, the representational study is not informative. |
| Development | Residual discovery, route validation, candidate causal basis; no candidate is a reportable outcome. |
| Freeze | Protocol, registry, splits, power simulation, call inventory, compute ceiling. |
| Untouched confirmation and reconciliation | As in Experiment 7. |

**Existing pilot (CPU-tested, not run on a model).** `configs/exp8_task_state.yaml` with protocol `src/jpeers/followup/task_state.py` implements the scripted fixture, the eight conditions, name/role swaps, a none/3-message filler delay, the six strict-JSON forks, the no-peer control, and J-lens/logit/output band-mean owner margins at the passive pre-question site and the decision site (`R0` only). Its default inventory is 16 projects × 8 conditions × 2 swaps × 2 delays = 512 cells, 2,624 forks (512 recorded) plus 32 no-peer forks; `tests/test_task_state.py` checks labels, dissociations, leak-free fillers, parsers, swap cancellation of name priors, and exact counts. Not yet implemented: residual decoding, the 512-token delay, update branches, `R1`/`R2` masking, the causal assay, live peers, subtask-order counterbalancing.

Provisional sizes are 16 development and 64 evaluation project families; these are budgeting placeholders, not a power guarantee. Simulate power for the joint success rules, including multiplicity, paired-family variance, and missingness, before confirmation.

A positive P1–P3 supports a bounded functional account: this model tracks ownership and provenance of scripted workflow events, separately, before being asked. Without P5–P6 it supports contextual availability, not maintained or causally used state. None of the outcomes supports claims about peer beliefs, intentions, or theory of mind.

## References and prior evidence

- [Historical Experiment D](../../jeffrey-exp-plans-new/peer-modeling-next-experiments.md#proposed-experiment-d-partner-contribution-and-task-state-tracking).
- [Audited follow-up v2](2026-11-04-audited-followup-v2-design.md), for private briefs, handoff codes, frozen messages, and no-peer probes.
- [Experiment 7 specification](2026-11-08-experiment-7-peer-belief-validation-design.md), for route, causal, and statistical conventions reused here.
- [Exp. 8 pilot description](../../05_experiments.md).
