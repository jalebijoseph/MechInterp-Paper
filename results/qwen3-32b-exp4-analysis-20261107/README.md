# Qwen3-32B evidence reconciliation and follow-up

The completed specificity analysis uses all 4,680 original Exp4 responses and both calibrated graders. Raw responses and existing grader labels are preserved. All current findings are exploratory because the original Exp0 gate failed and the scoring and interaction analyses were developed after the original runs.

## Findings

The generated-position J-space intervention reduces false-claim correction by about 40 percentage points relative to the matched random edit. Nemotron's all-fact missingness bounds are −41.1 to −38.9 points, and GPT-OSS's are −42.8 to −40.6 points. The deficit is larger for false than true claims under both graders. Peer-versus-document interactions provide limited statistical evidence of agency specificity. Joint tables retain unusable and uncertain responses.

Read the [specificity report](../../docs/2026-11-07-exp4-specificity.md) and [machine-readable analysis](specificity.json). The [source manifest](source-manifest.json) records exact frozen input paths and checksums. The report distinguishes full-bank unknown-outcome bounds from confidence intervals and complete-cluster point estimates.

## Human review

The [reviewer packet](adjudication/reviewer_packet.html) contains 149 cases with empty human labels. It includes the union of 74 responses unresolved by either grader, 50 sampled resolved disagreements, and 25 agreement controls. Give an independent reviewer only the packet HTML, [label-entry CSV](adjudication/reviewer_packet.csv), packet JSONL, and rubric. Keep the coordinator key and comparison inventories separate until labels are frozen. Human adjudication is pending.

## Corrected Exp2 grading

The corrected Exp2 run generated 600 reviews, but its original grader failed calibration. The validated Nemotron v3 adapter now grades those saved reviews after replaying its 84 integrity-checked calibration receipts with zero new calibration calls. It excludes all 480 knowledge probes and preserves the complete factorial identity.

The active run is `/home/vinli0921/mechinterp-runs/qwen3-32b-exp2-nemotron-v3-20261107/attempt-001` on the authorized cluster. The local [run pointer](../../runs/cluster-qwen3-32b-exp2-nemotron-v3-20261107/queue-location.json) identifies its live progress file. At 18:00 ET on 2026-11-07, 58 of 600 reviews had been processed with zero invalid grades.

## Focused causal follow-up

The [implemented protocol](../../docs/2026-11-07-causal-followup-design.md) uses fresh independently varied own/incoming labels, named-peer/document framing, name pairs, and note order. It captures a common suffix endpoint after both notes, identifies a conditional incoming-value direction on development templates, and tests deletion, matched random deletion, clean donor rescue, counterfactual donor patching, and matched random restoration.

The managed study is queued behind corrected Exp2 grading at `/home/vinli0921/mechinterp-runs/qwen3-32b-causal-handoff-20261107/attempt-001`. It waits for grader shutdown and an idle pair among physical GPUs 0–3. Discovery gates precede any held-out requests. Failed gates produce an explicit discovery-failure report. Both jobs stop owned serving processes when they exit.

The complete staged source passed 242 offline CPU tests, with 8 optional tokenizer checks skipped and 86 subtests passing. Independent reviews covered statistics, reviewer blinding, runtime edits, staged assay, and lifecycle controls. GPU discovery and held-out results are pending. The [live study pointer](../../runs/cluster-qwen3-32b-causal-handoff-20261107/queue-location.json) identifies status, progress, and result files.
