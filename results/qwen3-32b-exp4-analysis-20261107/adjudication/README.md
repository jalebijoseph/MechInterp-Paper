# Human review packet

Human review is pending. All human label, evidence, rationale, and reviewer ID cells are empty.

Give the reviewer only `reviewer_packet.csv`, `reviewer_packet.html`, `reviewer_packet.jsonl`, and `rubric.txt`. Keep this guide, the coordinator key, comparison summary, and comparison inventories away from reviewers until labels are frozen. The HTML provides readable exact saved text and numbered evidence excerpts. Enter labels and supporting evidence in the CSV. Use `null` for an uncertain Boolean and leave fields empty until reviewed. `final_truth` accepts true, false, both, missing, or unclear. Save reviewer ID and rationale with each completed case. Do not substitute grader predictions for human judgments.

The CSV prefixes potentially formula-interpreted cells with an apostrophe. The JSONL and HTML retain exact text, so consult those if a CSV cell starts with a protection prefix. Exact response content can reveal contextual clues even though experiment metadata, predictions, failures, selection reasons, and model identities are omitted. The common frozen rubric is included verbatim. Each Boolean is judged independently. An invalid machine judgment stays invalid in the original artifacts.

## Sampling and coordinator instructions

The population has 4,680 paired row identities. Both graders returned structurally valid judgments on 4,664 rows. There are 199 rows with a difference in at least one of the five rubric labels among those valid pairs. The per-field matrices include valid unclear and null judgments and retain their categories.

The packet contains 149 rows. It includes all 74 rows unresolved by either grader, 50 of 160 otherwise-resolved disagreements, and 25 of 4446 otherwise-resolved agreements. Requested sample maxima are 50 and 25. Sampling is without replacement over original review cells. Identical texts in different cells remain separate and are mapped in the key.

The seed is `exp4-adjudication-20261107-v1`. Separate SHA-256 rankings select disagreement and control samples. An independent presentation ranking mixes every selected case before blind review IDs are assigned. Canonical JSON hashing uses the repo's `_digest`. Neither the desired effect, direction of disagreement, truth label, nor condition is used for sample ranking. This enriched audit sample cannot be treated as a simple random sample of all outputs or used as an unweighted estimate of grader accuracy.

For internal scheduling, sort `coordinator_key.jsonl` by `coordinator_priority`, then `review_id`. Priority 0 contains unresolved rows in the original primary false-claim pair, priority 1 other unresolved rows, priority 2 sampled disagreements, and priority 3 agreement controls. Do not show these priorities or selection reasons to the reviewer. Keep the combined blinded packet order for review where practical. If assigning smaller batches, mix cases before delivery to preserve blinding.

The key contains original identities, both grader identities, labels, exact evidence, raw outputs, errors, provenance, hashes, and selection reasons. It does not contain an adjudicated answer. `comparison_rows.jsonl` and CSV retain every population cell, including invalid pairs. `disagreements.jsonl` and CSV contain all valid pairs with at least one label difference. Evidence wording differences alone do not define a label disagreement. `comparison_summary.json` records all inputs and artifact hashes. These are agreement statistics, and human adjudication is required to evaluate semantic correctness.

## Reproduce

From the repository root, run `PYTHONPATH=src python scripts/prepare_exp4_adjudication.py` in a Python environment with the repo's pydantic dependency. The default paths refer to the frozen downloaded runs. The script validates the complete identity grid, identical original inputs, frozen references, rubric version and hash, stored raw valid judgments, and original unresolved/invalid schemas. It performs no GPU or model calls. Generation stops if existing packet CSV or JSONL files contain human entries. Use a new output directory to preserve completed reviews.
