# Causal follow-up: evidence, claim, and implementation dependencies

Date: 2026-11-07. This document records the approved follow-up and its implementation. The user authorized the blinded audit, specificity analysis, and a focused causal study. Numerical settings below are implementation choices made during this work. Discovery and held-out outcomes remain pending until the queued model run executes.

## Implementation and execution status

The runtime now supports opt-in residual and decoder-direction capture, partial fixed-basis deletion, downstream donor-coordinate patches, and random controls matched in removal norm and inner product. Trace files retain the added arrays and edit metadata. Model-dtype rounding is recorded separately from intended geometry, and original intervention defaults remain intact.

The staged assay is implemented in [handoff.py](../src/jpeers/followup/handoff.py), with [the command-line runner](../scripts/run_causal_handoff.py), [frozen configuration](../configs/causal_handoff_32b.json), and [lifecycle controller](../scripts/queue_causal_handoff.py). It fits one incoming-value direction inside the two label-token direction span. Development uses 256 contexts and 32 assay controls plus one clean replay. Held-out evaluation has 768 contexts and six conditions, yielding at most 4,608 held-out generations. The deployment freezes all source/configuration hashes after CPU verification.

The controller waits for the current 600-review corrected Exp2 Nemotron grading job to stop its server, then selects an idle GPU pair among physical GPUs 0–3. It serves pinned BF16 Qwen3-32B with the Hugging Face runtime, runs discovery first, and opens the held-out stage only after every development gate passes. It stops owned serving processes on completion or failure. A discovery failure remains explicitly recorded as that outcome.

Human adjudication of the existing Exp4 results remains pending in the separately prepared blinded packet. New handoff outcomes use exact JSON scoring and require no model grader.

## Claim supported now

> In exploratory Qwen3-32B assays, Jacobian-lens readouts change with collaboration framing, an author's stated incentive and source access, and the association between agent names and specialties. Broad intervention during generation reduces factual correction relative to an edit-size-matched random control. The present evidence does not establish a peer-specific communication mechanism, a representation of another agent's beliefs, or a privileged role for the Jacobian readout over output predictions.

The last qualifications matter. The original Experiment 0 gate failed and remains failed. Corrected Experiments 1–3 are completed target runs under an explicitly exploratory post-audit protocol; they do not retrospectively satisfy that gate. Current Experiment 4 specificity analysis finds little evidence that the correction impairment is larger for named peers than anonymous documents. Fluency and usability also deteriorate. A mechanistic communication claim needs selective loss of incoming information and a nontrivial content-specific rescue, rather than another broad generation ablation.

## Corrected Experiments 1–3: frozen results

The completed evidence is local under [completed-evidence](../runs/cluster-qwen3-32b-followup-v2-20261104/completed-evidence/). The November 4 `cluster-snapshot` is stale. The manager's final `incomplete` status reflects failure of the old independent semantic-grading calibration; [worker-status.json](../runs/cluster-qwen3-32b-followup-v2-20261104/completed-evidence/worker-status.json) records completed Exp1, Exp2, Exp3, and NARCBench target jobs.

| Frozen primary | Estimate and scale | Independent units | Raw p | Frozen Holm p | Descriptive 95% cluster bootstrap CI |
|---|---:|---:|---:|---:|---:|
| Exp1 team-minus-named-solo change in partner-minus-own vocabulary readout | +0.00759066 reciprocal-rank units | 8 tasks, averaging the two role contrasts | 0.0234375 | 0.046875 | [0.00284503, 0.01275334] |
| Exp2 false-draft reward effect when verified minus reward effect when unverified, strict-intent vocabulary | +0.00000475719 reciprocal-rank units | 60 facts | 0.000939981 | 0.00469991 | [0.00000193896, 0.00000811886] |
| Exp3 completed-task-token role-swap contrast at fixed literal names and roster order | +0.0000295767 reciprocal-rank units | 12 tasks, averaging eight role-swap pairs each | 0.00244141 | 0.00732422 | [0.0000176560, 0.0000407780] |
| Exp3 anonymous Qwen-versus-OLMo generated-plan preference | +0.833333 signed preference units | 12 paired tasks | 0.000976563 | 0.00469991 | [0.625, 1.000] |

Sources: [Exp1 summary](../runs/cluster-qwen3-32b-followup-v2-20261104/completed-evidence/results/exp1/followup-v2/summary.json), [Exp2 summary](../runs/cluster-qwen3-32b-followup-v2-20261104/completed-evidence/results/exp2/followup-v2/summary.json), [Exp3 summary](../runs/cluster-qwen3-32b-followup-v2-20261104/completed-evidence/results/exp3/followup-v2/summary.json), and [frozen primary family](../runs/cluster-qwen3-32b-followup-v2-20261104/completed-evidence/primary-family.json). Exp1/Exp3 p-values are exact task-level sign flips; Exp2 uses 50,000 fact-level sign flips, seed 0. Frozen Holm correction retains a family of five, with the original Exp4 semantic test unavailable. Do not silently replace that missing test with a later grader result and continue calling the family frozen.

The intervals were added for this design, not supplied by the frozen reports: percentile bootstrap of the independent-unit effects, 100,000 resamples, NumPy `default_rng(20261107)` reset independently for each metric. They are post hoc, unadjusted, and especially fragile for eight tasks. They do not replace the frozen randomization tests.

Interpretation and denominator checks:

- **Exp1:** 32 recorded reviews use 16 frozen handoff notes. Both role and partner metadata and the note itself are held constant across framing. Explicit handoff-code retrieval is correct in 31/32 probes: team 15/16, named-solo 16/16. No-peer probes correctly return the available own code and unavailable-partner null in 16/16. This is retrieval, not a demonstrated increase in plan quality or cooperative competence.
- **Exp2:** 600 reviews cover 60 facts × two truth conditions × four agent incentive/source-access combinations plus one document condition. All 480 knowledge probes parse; 439/480 recover the stated source-access condition, including 199/240 verified and 240/240 unverified cases. The primary measures changes in the readout of `lie`, `lying`, and `deceive`, not verified deception detection. Actual final-text truth and accusations in these corrected responses have not yet been semantically graded.
- **Exp3 context:** all 192 choices parse and delegate correctly. Four name pairs, specialty swaps, and roster reversals support the readout contrast; 192 is not the inferential sample size.
- **Exp3 plans:** the two actual sources are Qwen3-32B and OLMo-2-0325-32B-Instruct, both using temperature 0 and a 128-token plan limit; saved plans use 18–76 generated tokens, with no cap hit. Across 24 order-balanced judgments there are 21 Qwen choices, one OLMo choice, and two explicit ties. Thus Qwen is chosen in 21/24 of all judgments, 21/22 non-ties; tie-half-credit gives 22/24. The primary averages signed scores (+1 Qwen, −1 OLMo, 0 tie) within task, producing +0.833333. Identical-plan controls return explicit ties in 12/12. Plan quality, style, and source family remain confounded; family recognition and intrinsic self-preference are not established.
- **Bandit:** the corrected design has eight 40-round sessions per setting, not 640 independent replications. In the second-better setting mean max-share is 0.575, regret per round 0.1275, and late-minus-early best-arm selection is +0.0125, with descriptive session-bootstrap CI [−0.0625, 0.0875]. Feedback-conditioned staying and allocation concentration answer different questions. The equal setting has no unique best arm, so its recorded `best_arm_rate=1` is a coding convention and not evidence of learning.
- **External intent check:** corrected [NARCBench](../runs/cluster-qwen3-32b-followup-v2-20261104/completed-evidence/results/narcbench/followup-v2/summary.json) strict-intent J readout has descriptive AUROC 0.21125 and a 10-scenario paired difference −0.0000298437, exact p=0.123047. It does not corroborate a general colluder detector.

Output and plain-logit baselines limit a unique-J claim. Post hoc paired comparisons of the same independent-unit contrasts give Exp1 J-minus-output −0.00516582 (p=0.242188); Exp2 J-minus-output −0.00000842515 (p=0.0213996). The Exp2 output baseline itself has an interaction +0.0000131823 (p=0.00299994). Baseline differences were not declared primary and require multiplicity caution; the relevant point is that positive J scores alone do not establish information unavailable to the ordinary output readout.

## Corrected Exp2 behavior grading

This is completion of the authorized corrected reruns, not a request to generate new target responses. Use the exact 600 `reviews` from the local Exp2 summary linked above, preserving `(condition, source, claim, item, reward, verified)`; exclude all 480 knowledge probes. Hash the complete source summary and the serialized selected rows. As of this design, a separate workstream is implementing and preflighting the v3 adapter for a managed Nemotron grading launch. These 600 responses remain ungraded here; pending launch is not a completed behavioral result.

The old failure is specific to OLMo-2-0325-32B-Instruct with `semantic-fact-v2.0.1`: calibration coverage 0.65, agreement 0.35, and four critical false positives. It does not invalidate the later Nemotron `semantic-fact-v3.4.0` profile, which passed the frozen 40-case development and 44-case holdout gates.

The new **v3 dataset adapter** implements these requirements:

1. Reuse `rescore._validate_rows(rows, "exp2_v2", facts)` and `_reference` for the complete 600-cell inventory and exact fact references.
2. Reuse the current v3 runner's `request_payload`, `grade_batch`, raw-receipt integrity checks, evidence-ID parser, and pinned Nemotron profile/checklist. Bind the rubric, model revision, generation profile, input hash, and gate hashes in a new output root; use the already selected 1,800-second request timeout.
3. Preserve every factorial key and grade status. Report final truth, usable text, explicit correction, accusation of intent, and refusals separately; aggregate fact-level reward × source-access behavior contrasts with unknown bounds. Do not copy the readout contrast into a behavioral result.
4. Reuse gates only under identical model/rubric/profile identities; otherwise run the unchanged frozen calibration checks first. Retain old failed results unchanged.

The adapter is [run_exp2_rescore_v3.py](../scripts/run_exp2_rescore_v3.py). Its ten focused tests pass, and remote CPU preflight replayed all 84 pinned calibration receipts with zero model calls. The managed grading job has started at `/home/vinli0921/mechinterp-runs/qwen3-32b-exp2-nemotron-v3-20261107/attempt-001`. Its location and source pins are saved in [queue-location.json](../runs/cluster-qwen3-32b-exp2-nemotron-v3-20261107/queue-location.json). Behavioral contrasts will be analyzed after the new grades are available.

## Focused causal question

The next mechanistic question is: **does a localized, independently identified representation of incoming information selectively mediate its later use, and can restoring the correct content reverse that loss?** Agent specificity is an additional test, not an assumption.

Two alternatives are deliberately rejected. Ablating a token direction for `Bob` does not identify a representation of Bob's knowledge. Restoring a component immediately after subtracting that exact component at the same site is an algebraic identity, not a meaningful rescue. Existing broad top-k generation edits have already answered the broad perturbation question and should not be relabeled as a narrow mechanism test.

### Proposed assay and fixed controls

Use a short private-handoff assay based on the corrected Exp1 retrieval structure. Each new episode contains one recipient-private value and one independently assigned value in an incoming handoff. **Fully cross the two own-value labels with the two incoming-value labels**, including both equal-value cases: `(A,A)`, `(A,B)`, `(B,A)`, `(B,B)`. Opposite-only assignments are invalid because the recipient could infer the incoming value from its own value. Request both values in strict JSON. Use balanced, preflighted single-token value labels so accidental pretrained knowledge cannot supply the answer; do not substitute the old multi-token hexadecimal codes into a whole-token lens assay. Exact JSON scoring needs no grader. This is incoming-information use; calling the field a peer field does not establish agency-specific processing.

For every episode, keep both payloads and the requested answer identical across named-peer and anonymous-document framing. Cross source framing, two literal name arrangements, and two note orders independently of the four value assignments; do not encode name and order in one coupled reversal. Measure tokenization and actual spans. Include no-handoff trials, where unavailable incoming information must yield null, and an own-only extraction control. Failure to transfer with the information available and failure to abstain when unavailable are different outcomes.

**Proposed bounded scope:** eight development templates and 24 new held-out templates. Freeze their text, generation provenance, disjoint split, and hashes before inspecting development intervention outcomes; keep held-out responses unopened until settings are frozen. Full crossing gives 24 templates × four value assignments × two source framings × two name arrangements × two note orders = **768 contexts**, or **4,608 short JSON generations** for the six conditions below. This larger proposed ceiling preserves independent name/order control. A smaller version with one frozen name pair and crossed note order has 384 contexts/2,304 generations but cannot establish generalization over literal names. No-handoff and own-only checks are additional assay gates. Cluster inference by template, averaging repeated cells; neither context count nor generation count is the independent sample size. Twenty-four templates is an exploratory feasibility proposal, not an established sample size for five-point noninferiority. A variance/power calculation and final held-out size must be frozen before a confirmatory run; if the budget cannot support the specificity margins, report that limitation rather than claim noninferiority.

### Discovery before intervention

Identify a candidate peer-value component using development data in which own value and incoming value vary independently. Append the shared neutral suffix `End of notes.` after both payload notes in every note order. Capture residuals at the final token of that unique suffix, before any value answer is generated. Resolve the suffix through an explicit span and verify its token endpoint and placement after both payloads during tokenizer preflight. Counterfactual donors preserve the template, names, order, own value, and token lengths while changing only the incoming value. A conditional decoder must predict incoming value out of development-template folds while controlling for own value, name, order, and source framing. Fit an own-value decoder separately as a specificity check.

For a J-specific candidate, work within a frozen low-rank span of J-lens directions for the balanced payload labels and derive the incoming-value contrast from the factorial residual differences. Compare with the same construction using plain unembedding directions and an unconstrained residual contrast. A name direction or a collection of domain keywords is not sufficient. If no reproducible incoming-value component can be isolated, report that discovery failure; do not proceed to an alleged peer-representation ablation.

Suggested bounded development search: deletion layer in {40, 48, 52}, rescue two blocks later, both within the recorded Qwen workspace band; the suffix endpoint defined above; rank at most two; deletion fraction in {0.25, 0.5}. These are proposed engineering defaults. Select the weakest development setting that changes incoming-value use while preserving own-value accuracy and JSON validity within five percentage points of baseline. Freeze layer, site, basis, dose, donor rule, task split, and analysis before opening held-out results. The development thresholds are feasibility checks, not statistical evidence of a mechanism.

### Six causal conditions

1. Unedited baseline.
2. Partial deletion of the development-identified incoming-value component at the earlier site.
3. Random-direction deletion with the **same realized removal norm**, same positions and layers, and equal post-edit residual norm. The present `random_matched` implementation is tied to dynamic top-k projection, so it must be generalized before it can honestly control a fixed learned component.
4. Earlier deletion followed by downstream restoration of only the identified component from a separate clean run of the same episode.
5. The same deletion followed by a donor component from the opposite incoming-value counterfactual, with own value and all nuisance fields held fixed. A selective change toward that donor value is stronger evidence about content than recovery alone.
6. The same deletion followed by a norm-matched random-component restoration, controlling for adding energy at the rescue site.

For partial deletion, matching just the removal norm is insufficient. With `δ = λ QQᵀh`, match both `r = ||δ||` and `d = hᵀδ`; a random edit can be constructed as `δ_random = a h_hat + sqrt(r²−a²) u_perp`, where `a = d/||h||` and `u_perp` is a unit vector orthogonal to `h`. Then `h−δ_random` matches the targeted edit's removal norm and final residual norm. The existing full-projection helper's `a=r²/||h||` is not correct for a general partial deletion. Apply the corresponding inner-product/norm check to restoration controls too, using the actual shared damaged recipient state before patching. Freeze random-control seeds before the run and report their limited directional replication; adding more seeds later is a labeled robustness analysis, not new independent task replication.

The target model uses BF16. Record both intended floating-point edit geometry and the realized displacement after casting back to the model dtype. The development engineering check requires a nonzero realized edit and a maximum paired geometry discrepancy of 5% relative to the intended edit magnitude before held-out execution. Record the exact quantities and denominators in the frozen configuration. A failed rounding or geometry check is an engineering failure to report, rather than evidence about the proposed information component.

For downstream patching use `h_patched = h_damaged + Q(Qᵀh_donor − Qᵀh_damaged)`, with a frozen orthonormal basis `Q` at the later site; only those one or two coordinates may be replaced. The damage occurs two blocks earlier. Same-site deletion-plus-addition cancellation is reserved for an engineering identity test, never the scientific rescue result. Record actual deletion and restoration norms for every layer/site/cell; match random restoration norm to the realized clean-donor restoration, not an average from another condition.

Primary held-out contrasts are incoming-value accuracy under targeted-minus-random deletion and under clean-donor-rescue-minus-deletion. Treat own-value retention and JSON validity as required specificity checks, with predeclared five-point noninferiority margins and uncertainty intervals. Test named-peer minus document differences as a separate, multiplicity-adjusted interaction. Equal effects in peer and document conditions support a source-general information-use mechanism, not a peer-specific one. A wrong-donor edit changing the own field or merely damaging JSON fails the intended specificity interpretation.

All malformed outputs and cap hits count as failures in the primary end-to-end analysis. Also report the joint outcome table for incoming correctness, own correctness, and format validity. Never condition the primary on post-intervention fluent or valid survivors. Use paired template-level effects, 50,000 sign flips with seed 0, 100,000 cluster-bootstrap intervals with a frozen seed, and Holm correction for the two causal contrasts plus the source-framing interaction. Preserve unknown/missing cells and show worst-case bounds. No single successful example or favorable layer chosen from held-out data establishes the claim.

## Implementation requirements addressed

The existing layer hooks and explicit position selectors supplied the foundation for the localized assay. This follow-up added capture, fixed-basis edits, donor coefficients, and exact suffix-endpoint selection as opt-in features.

The following gaps were identified before implementation and covered by the new code and tests:

- Original traces lacked hidden residuals and donor coefficients. New optional arrays are captured from fresh runs and preserved through response encoding and trace save/load.
- New `fixed_delete`, `fixed_random_matched`, `fixed_patch`, and `fixed_random_restore` operations accept explicit orthonormal basis rows and donor coefficients at one layer and prompt site.
- Partial deletion uses its own validated fraction. The original `ablate_tokens` behavior stays attributable to the original experiments.
- Fixed-component controls match the intended displacement norm and inner product, with separate receipts for realized model-dtype geometry.
- The new runner persists actual interventions directly in `TurnRecord` and saves an exact durable request/response receipt before validation and derived trace projection. Interrupted machine-generated turn views can be rebuilt from those receipts without another generation.
- Per-layer tensors remain request-local, model generation is serialized, and the command-line study uses an output lock to prevent concurrent writers.

The implementation includes a deterministic factorial generator and exact scorer, captured donor/basis artifacts with hashes, the bounded discovery search, and held-out template-level analysis. The proposed feature remains a scientific candidate until development decoding, behavior, specificity, replay, and realized-geometry gates pass.

CPU and tiny-model checks cover exact sites and layers, zero-dose identity, basis validation, deletion and restoration geometry, donor patches, independent payloads, wrong-donor identity, fold-local fitting, serialized state, truthful persisted interventions, and interruption recovery. Independent reviews covered the runtime, assay, and lifecycle controller. GPU execution begins with capture/anchor checks and a deterministic clean replay. Scientific discovery can fail even when the implementation passes its engineering checks, so the held-out stage remains conditional.

## Provenance to retain

- Protocol: `audited-followup-v2`; frozen source commit recorded as `dba3a42c3bb932b2b4b34bff800e5c997279d675`, source-base SHA `5ce32c4a75a96f682a3677710e20e37dab2673a0`; per-file hashes are in [followup-config.json](../runs/cluster-qwen3-32b-followup-v2-20261104/completed-evidence/followup-config.json).
- Qwen target: `Qwen/Qwen3-32B`, revision `9216db5781bf21249d130ec9da846c4624c16137`; 64 blocks, width 5,120; J-lens repository `neuronpedia/jacobian-lens`, revision `e166deb5ad540932100ef803c8c9bcc2a5453694`, file `qwen3-32b/jlens/Salesforce-wikitext/Qwen3-32B_jacobian_lens.pt`; workspace band 24–58 inclusive. The remote `qwen-server/model-provenance.json` records the snapshot and lens paths; its SHA-256 is `6a3e67dbb3dbb75a1a6d56d7c227deec6faa6de4fca1c8b976c60e8a4f34301f`.
- Actual plan comparator and old failed grader: `allenai/OLMo-2-0325-32B-Instruct`, revision `b96024342a77a69aa0dda815c3454a671f477463`; endpoint identities in [endpoint-identities.json](../runs/cluster-qwen3-32b-followup-v2-20261104/completed-evidence/endpoint-identities.json).
- Summary SHA-256: Exp1 `54de362b0ec0a076a648c4d1c5d25cd0366e8044684e7311977e9e51c44c4d37`; Exp2 `010689c97477e908d213939ff01b96cf780f9bf92720c24a1f636fa7fe8bdb9b`; Exp3 `3ea5030205d405b05ee9bacbcb6a3f187d16757de6f2446a640105bcf6983c37`; frozen family `9c8163c674b01136d6bff3f789ef18bb9bd2d446c7531f10961e8c4ee658c962`.
- Current validated Nemotron grader: revision `da5b2650a835292c678994a1fa3fb653250fb508`, scorer `semantic-fact-v3.4.0`, rubric SHA-256 `3af84c4c0430e9dc1d7689623fc879fbdd988d175ff961ef57bf2e93b20a6120`, generation profile `nemotron-3.5-super-vl-v2`, `enable_thinking=true`, `medium_effort=false`, temperature 1, top-p 0.95, seed 0, 12,288-token completion limit. Its [Exp4 summary](../runs/cluster-qwen3-32b-exp4-nemotron-20261106/summary.json) remains explicitly partial with 53 unresolved grades; do not describe calibration success as complete result coverage.

Current outputs are the blinded adjudication packet, specificity report, validated Exp2 grading adapter, and implemented staged handoff assay. Grading and discovery status remain separate from completed statistical results. The held-out causal stage requires a frozen discovered component and passing engineering and specificity gates.
