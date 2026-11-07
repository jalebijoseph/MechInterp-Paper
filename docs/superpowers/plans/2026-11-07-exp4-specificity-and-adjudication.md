# Exp4 specificity and adjudication implementation plan

> **For agentic workers:** Use subagent-driven development for the independent analysis, review-packet, and follow-up-design workstreams. The user authorized execution on 2026-11-07.

**Goal** Establish which causal claims the frozen Qwen3-32B outputs support, prepare blinded human adjudication, and specify the next focused intervention study.

**Architecture** Analyze immutable scored responses from Nemotron attempt-004 and GPT-OSS attempt-007. Preserve each model's labels and unknowns. Keep reviewer materials separate from the identity and prediction key.

**Tech stack** Python, existing JSONL artifacts, existing fact-cluster sign-flip statistics, CSV and a readable review packet.

**Spec** The user-approved next steps in this conversation are an audit of unresolved judgments and sampled disagreements, specificity analysis with existing controls and corrected Experiments 1–3, and a focused peer-information intervention with a rescue and a no-peer factual control.

## Global constraints

- Frozen outputs and grader labels remain attributable to their pinned source artifacts.
- The independent statistical unit is the fact, with 60 facts and repeated source conditions.
- Retain uncertain judgments and propagate worst-case bounds.
- Report joint writing usability and correctness without filtering on post-intervention usability.
- Describe all current effects as exploratory because the original Experiment 0 gate failed.
- Human review labels stay empty until a human supplies them.
- Any GPU execution must use the authorized idle physical GPUs 0–3 and stop owned serving processes on exit.
- Any new follow-up must identify peer-relevant information without selecting directions on held-out outcome labels.

## Review focus

- Invalid or abstained judgments contribute unknowns, rather than false labels.
- Missing conditions, duplicate row identities, or mismatched response text prevent paired comparisons.
- Three source conditions for one fact never count as three independent facts.
- Reviewer materials exclude grader labels, intervention names, and sampling reasons.
- Clustering, resampling seeds, sampled review identities, and source hashes are reproducible.
- Restoring all removed coordinates is an intervention reversal control, which alone does not isolate peer information.

## Task 1 — Import and verify frozen evidence

**Files** Existing ignored `runs/cluster-qwen3-32b-exp4-*` directories.

- [x] Download both complete `scored_rows.jsonl` artifacts and record checksums and remote attempt identifiers.
- [x] Verify 4,680 unique identities in each set, identical references and responses, pinned summaries and passing gates.
- [x] Locate completed corrected Experiments 1–3 reports and retain the Experiment 0 gate limitation.

## Task 2 — Analyze specificity

**Files** `scripts/analyze_exp4_specificity.py`, `results/qwen3-32b-exp4-analysis-20261107/specificity.json`, and `docs/2026-11-07-exp4-specificity.md`.

**Consumes** Both verified scored-response files.

**Produces** Per-source and per-claim contrasts, interactions, joint outcomes, missingness bounds, and deterministic inference at the fact level.

- [x] Implement input validation and reuse existing statistical conventions.
- [x] Compare generated-position J-space and matched interventions, false versus true claims, and peer versus document source conditions.
- [x] Count joint correctness and usability outcomes including unknowns.
- [x] Verify results against the saved primary summary and targeted synthetic missing-data and clustering cases.

## Task 3 — Prepare blinded adjudication

**Files** `scripts/prepare_exp4_adjudication.py` and `results/qwen3-32b-exp4-analysis-20261107/adjudication/`.

**Consumes** Both verified scored-response files and their shared rubric.

**Produces** An unlabeled reviewer packet, separate provenance and prediction key, complete grader agreement analysis, and sampling instructions.

- [x] Include the union of unresolved cases, a deterministic sample of disagreements, and agreement controls.
- [x] Hide model, condition, identity, prediction, and selection information from reviewer materials.
- [x] Retain exact reference propositions and saved response text.
- [x] Check packet/key correspondence, empty human labels, sampling reproducibility, and complete inventory coverage.

## Task 4 — Synthesize and specify the causal follow-up

**Files** `docs/2026-11-07-causal-followup-design.md`.

**Consumes** Specificity results, grader comparison, corrected Experiments 1–3, and current intervention APIs.

**Produces** A narrowly scoped, implementable protocol with discovery/confirmation separation, matched controls, rescue, technical feasibility, and explicit remaining dependencies.

- [x] State the supported claim and any limits imposed by the document control, fluency damage, and failed original gate.
- [x] Specify peer-information direction identification and an independently held-out test.
- [x] Distinguish a selective rescue from restoring the entire original state.
- [x] Evaluate whether the current implementation and evidence support meaningful GPU execution.

## Task 5 — Integrate and verify

- [x] Review both analysis scripts and all generated artifacts independently of their implementation agents.
- [x] Run reproduction commands and appropriate focused verification.
- [ ] Report concrete findings, human adjudication still pending, and any required implementation or evidence before a new GPU study.

## Completion discovered during evidence reconciliation

The corrected Exp2 run generated all 600 reviews, but its original semantic grader failed calibration. Completing those grades is part of the previously authorized corrected reruns and is necessary for the synthesis.

**Files** `scripts/run_exp2_rescore_v3.py` and a minimal explicit dataset branch in `scripts/queue_exp4_rescore_v3.py`.

- [x] Verify the pinned corrected summary contains exactly 600 review cells and excludes all 480 knowledge probes.
- [x] Reuse the unchanged Nemotron profile and rubric, replaying its 84 integrity-checked calibration receipts.
- [x] Verify inventory, provenance, metadata preservation, and no-network preflight before launch.
- [x] Launch managed grading on idle GPUs 0–3 with a 1,800-second request timeout and automatic server cleanup.
- [x] Record the live run location and clearly distinguish grading in progress from completed analysis.

## Execute the focused follow-up

The evidence reconciliation identified implementation requirements for the approved causal follow-up. The new assay tests incoming-information mediation and includes an agency-framing interaction. Its own and incoming values are independently crossed, including equal-value cases.

**Files** Opt-in additions to `schemas.py`, `runtime.py`, and `trace.py`; `followup/handoff.py`; `scripts/run_causal_handoff.py`; `scripts/queue_causal_handoff.py`; `configs/causal_handoff_32b.json`; and focused tests.

- [x] Add localized residual/direction capture and fixed-basis deletion, donor patching, and matched random controls while preserving original defaults.
- [x] Verify actual BF16 edit geometry and isolate every request's state.
- [x] Build 8 development and 24 held-out templates with independent value, source, name, and order factors and strict JSON scoring.
- [x] Implement fold-local discovery fitting and development specificity gates.
- [x] Implement and test a frozen selection barrier before any held-out requests.
- [x] Preserve durable responses across interruption, revalidate cached responses, prevent concurrent output writers, and retain unknown denominators.
- [x] Independently review the implementation and run the complete available offline CPU suite.
- [x] Queue the study behind corrected Exp2 grading with tested idle-GPU and owned-server lifecycle controls.
- [x] Report queued execution separately from actual discovery or held-out results. Failed discovery gates stop the study without opening held-out outputs.

## Execution

The user has already approved proceeding. Independent workstreams run in this conversation and share the existing branch. No additional execution-method approval is needed.

## Live status at handoff

Analysis, the unfilled human packet, implementation, independent review, and CPU preflight are complete. Corrected Exp2 semantic grading is running. The causal controller is waiting for that grader to release its server and for an allowed GPU pair to be idle. Human adjudication and actual discovery/held-out results remain pending. The offline suite passed 242 tests with 8 optional tokenizer skips and 86 subtests passing. Live locations are recorded under the corresponding ignored `runs/cluster-qwen3-32b-*` directories and in the saved results index.
