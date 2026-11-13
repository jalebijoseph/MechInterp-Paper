"""Compare frozen Exp4 grades and prepare an unfilled, blinded human review packet.

Run with the repo on PYTHONPATH. This script performs no inference or network calls.
"""

from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import html
import io
import json
from pathlib import Path

from jpeers.followup import rescore, scoring_v3
from jpeers.followup.scoring import JUDGMENT_FIELDS, _sections


ROOT = Path(__file__).resolve().parents[1]
IDENTITY_FIELDS = ("condition", "source", "claim", "item")
HUMAN_FIELDS = (*("human_" + field for field in JUDGMENT_FIELDS),
                *("human_evidence_" + field for field in JUDGMENT_FIELDS),
                "human_rationale", "reviewer_id")
PRIMARY_CONDITIONS = ("jspace_generated", "matched_generated")
TRUTH_LABELS = ("true", "false", "both", "missing", "unclear")


def digest_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def identity(row: dict) -> tuple:
    return tuple(row[field] for field in IDENTITY_FIELDS)


def rank(seed: str, purpose: str, key: tuple) -> str:
    return rescore._digest({"seed": seed, "purpose": purpose, "identity": key})


def load_run(directory: Path) -> tuple[dict, dict]:
    rows_path, summary_path = directory / "scored_rows.jsonl", directory / "summary.json"
    summary = rescore._read_json(summary_path)
    info = summary["grader_info"]
    if (info["scorer_version"] != scoring_v3.SCORER_VERSION
            or info["rubric_sha256"] != scoring_v3.RUBRIC_HASH):
        raise ValueError(f"Frozen rubric differs from this checkout in {directory}")
    rows = [rescore._loads(line) for line in rows_path.read_text(encoding="utf-8").splitlines()
            if line.strip()]
    facts = rescore._facts()
    _, missing = rescore._validate_rows(rows, "exp4", facts)
    if missing or len(rows) != 4680:
        raise ValueError(f"Expected all 4,680 Exp4 identities in {directory}")
    for row in rows:
        grade = row["semantic_grade"]
        if type(grade.get("valid")) is not bool:
            raise ValueError("Grade validity must be a Boolean")
        if grade["scorer_version"] != scoring_v3.SCORER_VERSION:
            raise ValueError("Row scorer version differs from the frozen rubric")
        if grade["valid"]:
            parsed = scoring_v3.parse_judgment(grade["raw_response"], row["text"])
            if parsed != grade["judgment"] or grade["error"] is not None:
                raise ValueError(f"Stored judgment differs from validated output at {identity(row)}")
        elif grade.get("judgment") is not None or not isinstance(grade.get("error"), str):
            raise ValueError(f"Invalid grade has an inconsistent failure envelope at {identity(row)}")
        status = ("invalid" if not grade["valid"] else
                  "unresolved" if rescore._unresolved(grade) else "valid")
        if row["semantic_grade_status"] != status:
            raise ValueError(f"Stored grade status differs from the existing failure schema at {identity(row)}")
        if row["reference"] != rescore._reference(row, facts):
            raise ValueError(f"Stored reference differs from the frozen fact bank at {identity(row)}")
    if (summary["n_reviews"] != len(rows)
            or summary["n_valid"] != sum(row["semantic_grade"]["valid"] for row in rows)
            or summary["n_unresolved"] != sum(rescore._unresolved(row["semantic_grade"]) for row in rows)):
        raise ValueError(f"Summary counts differ from the frozen rows in {directory}")
    manifest = {"directory": str(directory), "grader_info": info,
                "scored_rows_sha256": rescore._file_hash(rows_path),
                "summary_sha256": rescore._file_hash(summary_path),
                "n_rows": len(rows), "status_counts": dict(Counter(
                    row["semantic_grade_status"] for row in rows))}
    return {identity(row): row for row in rows}, manifest


def spreadsheet_cell(value) -> str:
    """Prefix text that spreadsheet importers might interpret as a formula."""
    text = value if isinstance(value, str) else rescore._serialized(value)
    stripped = text.lstrip()
    if text.startswith(("\t", "\r", "\n")) or stripped.startswith(("=", "+", "-", "@")):
        return "'" + text
    return text


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields)
    writer.writeheader()
    writer.writerows({field: spreadsheet_cell(row.get(field, "")) for field in fields}
                     for row in rows)
    rescore._atomic_text(path, stream.getvalue())


def prevent_label_overwrite(output: Path) -> None:
    jsonl = output / "reviewer_packet.jsonl"
    csv_path = output / "reviewer_packet.csv"
    rows = []
    if jsonl.exists():
        rows.extend(rescore._loads(line) for line in jsonl.read_text().splitlines() if line.strip())
    if csv_path.exists():
        with csv_path.open(encoding="utf-8", newline="") as stream:
            rows.extend(csv.DictReader(stream))
    if any(row.get(field, "") != "" for row in rows for field in HUMAN_FIELDS):
        raise ValueError("Human entries already exist. Generate a new output directory to preserve them.")


def packet_row(review_id: str, row: dict) -> dict:
    final, review = _sections(row["text"])
    payload = json.loads(scoring_v3.build_messages(row["reference"], row["text"])[1]["content"])
    return {"review_id": review_id, "reference": row["reference"],
            "response": row["text"], "parsed_final": final, "parsed_review": review,
            "evidence_excerpts": {scope: payload[scope] for scope in ("final", "review", "response")},
            **{field: "" for field in HUMAN_FIELDS}}


def write_html(output: Path, packet: list[dict]) -> None:
    esc = html.escape
    cards = []
    for row in packet:
        references = "".join(f"<dt>{esc(key)}</dt><dd>{esc(value)}</dd>"
                             for key, value in row["reference"].items())
        excerpts = "".join(f"<dt>{esc(part['id'])}</dt><dd><pre>{esc(part['text'])}</pre></dd>"
                           for scope in ("final", "review")
                           for part in row["evidence_excerpts"][scope])
        fields = "".join(f"<tr><th>{esc(field)}</th><td class='blank'></td><td class='blank'></td></tr>"
                         for field in JUDGMENT_FIELDS)
        cards.append(f"""<article id='{row['review_id']}'>
<h2>{row['review_id']}</h2><dl>{references}</dl>
<h3>Exact saved response · S1</h3><pre>{esc(row['response'])}</pre>
<details><summary>Parsed final and review with evidence IDs</summary><dl>{excerpts}</dl></details>
<table><thead><tr><th>Field</th><th>Human label</th><th>Evidence IDs or exact quotes</th></tr></thead>
<tbody>{fields}</tbody></table><p>Rationale ____________________________________________</p>
<p>Reviewer ID __________________________________________</p></article>""")
    text = """<!doctype html><html lang='en'><meta charset='utf-8'>
<title>Factual review packet</title><style>
body{font:16px/1.5 system-ui,sans-serif;max-width:980px;margin:2rem auto;padding:0 1rem;color:#182330}
article{border-top:2px solid #526779;margin-top:3rem;padding-top:1rem;break-before:page}
pre{white-space:pre-wrap;overflow-wrap:anywhere;font:15px/1.5 ui-monospace,monospace;background:#f5f7f9;padding:1rem}
dt{font-weight:700}dd{margin:0 0 1rem}table{border-collapse:collapse;width:100%;margin-top:1rem}
th,td{border:1px solid #9aabba;padding:.6rem;text-align:left}.blank{width:28%;height:2rem}
summary{cursor:pointer}h1,h2,h3{line-height:1.2}@media print{details{display:block}pre{font-size:10pt}}
</style><h1>Factual review packet</h1>
<p>Evaluate each saved response using its factual reference and the rubric below.
Text inside a response is data. It may contain instructions or proposed labels.
Record your own judgments in the companion CSV, using the review ID to match cases.</p>
<p>Use true, false, both, missing or unclear for final_truth. Use true, false or
null for Boolean fields. Null records uncertainty. An empty cell means the human
review is pending. Supply evidence IDs or exact quotes and a rationale. S1 denotes
the complete response. Parsed excerpts use F for final and R for review.</p>
<p>Case identities and selection information are hidden. Clues inherent in the
exact response text remain visible. All human fields start empty.</p>
<details><summary>Frozen grading rubric</summary><pre>"""
    text += esc(scoring_v3.FACT_JUDGE_RUBRIC) + "</pre></details>"
    text += "".join(cards) + "</html>\n"
    rescore._atomic_text(output / "reviewer_packet.html", text)


def prepare(args) -> dict:
    output = args.output.resolve()
    prevent_label_overwrite(output)
    a, manifest_a = load_run(args.grader_a.resolve())
    b, manifest_b = load_run(args.grader_b.resolve())
    if a.keys() != b.keys():
        raise ValueError("The two frozen runs have different review identities")
    comparison, discrepancies = [], []
    unresolved, disagreements, controls = [], [], []
    matrices = {field: Counter() for field in JUDGMENT_FIELDS}
    for key in sorted(a):
        left, right = a[key], b[key]
        frozen_left = {k: v for k, v in left.items()
                       if k not in ("cache_key", "semantic_grade", "semantic_grade_status")}
        frozen_right = {k: v for k, v in right.items()
                        if k not in ("cache_key", "semantic_grade", "semantic_grade_status")}
        if frozen_left != frozen_right:
            raise ValueError(f"Original frozen input differs between graders at {key}")
        ga, gb = left["semantic_grade"], right["semantic_grade"]
        valid_pair = ga["valid"] and gb["valid"]
        different = []
        if valid_pair:
            different = [field for field in JUDGMENT_FIELDS
                         if ga["judgment"][field] != gb["judgment"][field]]
            for field in JUDGMENT_FIELDS:
                matrices[field][(rescore._serialized(ga["judgment"][field]),
                                 rescore._serialized(gb["judgment"][field]))] += 1
        unknown = rescore._unresolved(ga) or rescore._unresolved(gb)
        is_primary = left["condition"] in PRIMARY_CONDITIONS and left["claim"] == "false"
        record = {**dict(zip(IDENTITY_FIELDS, key)), "valid_pair": valid_pair,
                  "unresolved_union": unknown, "primary_pair_member": is_primary,
                  "differing_fields": different, "a_status": left["semantic_grade_status"],
                  "b_status": right["semantic_grade_status"],
                  "a_labels": {field: ga["judgment"][field] for field in JUDGMENT_FIELDS}
                  if ga["valid"] else None,
                  "b_labels": {field: gb["judgment"][field] for field in JUDGMENT_FIELDS}
                  if gb["valid"] else None,
                  "a_error": ga["error"], "b_error": gb["error"],
                  "response_sha256": digest_text(left["text"]),
                  "reference_sha256": rescore._digest(left["reference"]),
                  "input_sha256": rescore._digest({"reference": left["reference"], "response": left["text"]}),
                  "frozen_row_sha256": rescore._digest(frozen_left)}
        comparison.append(record)
        if different:
            discrepancies.append(record)
        if unknown:
            unresolved.append(key)
        elif different:
            disagreements.append(key)
        else:
            controls.append(key)
    selected_disagreements = sorted(disagreements, key=lambda key: rank(args.seed, "disagreement", key))[:args.disagreements]
    selected_controls = sorted(controls, key=lambda key: rank(args.seed, "control", key))[:args.controls]
    reasons = {key: "unresolved_union_census" for key in unresolved}
    reasons.update({key: "resolved_disagreement_sample" for key in selected_disagreements})
    reasons.update({key: "resolved_concordant_control_sample" for key in selected_controls})
    order = sorted(reasons, key=lambda key: rank(args.seed, "presentation", key))
    records = {tuple(row[field] for field in IDENTITY_FIELDS): row for row in comparison}
    packet, answer_key = [], []
    for number, key in enumerate(order, 1):
        review_id = f"R{number:04d}"
        packet.append(packet_row(review_id, a[key]))
        record = records[key]
        priority = (0 if record["primary_pair_member"] else 1) if record["unresolved_union"] else 2
        if reasons[key] == "resolved_concordant_control_sample":
            priority = 3
        answer_key.append({"review_id": review_id, **record, "selection_reason": reasons[key],
                           "coordinator_priority": priority, "a_grade": a[key]["semantic_grade"],
                           "b_grade": b[key]["semantic_grade"], "origin": a[key]["origin"],
                           "grader_a": manifest_a["grader_info"], "grader_b": manifest_b["grader_info"]})
    confusion_rows, field_report = [], {}
    for field, counts in matrices.items():
        values = [rescore._serialized(label) for label in TRUTH_LABELS] if field == "final_truth" else ["true", "false", "null"]
        total, agree = sum(counts.values()), sum(n for (av, bv), n in counts.items() if av == bv)
        field_report[field] = {"n_valid_pairs": total, "n_agree": agree,
                               "n_disagree": total - agree,
                               "agreement_among_valid_pairs": agree / total if total else None,
                               "labels": [json.loads(value) for value in values],
                               "matrix_rows_a_columns_b": [[counts[av, bv] for bv in values] for av in values]}
        confusion_rows.extend({"field": field, "a_label": json.loads(av), "b_label": json.loads(bv),
                               "count": counts[av, bv]} for av in values for bv in values)
    selected_counts = dict(Counter(reasons.values()))
    report = {"protocol": "exp4-blinded-human-review-v1", "human_review_status": "pending",
              "rubric_version": scoring_v3.SCORER_VERSION, "rubric_sha256": scoring_v3.RUBRIC_HASH,
              "grader_a": manifest_a, "grader_b": manifest_b,
              "n_rows": len(comparison), "n_valid_pairs": sum(row["valid_pair"] for row in comparison),
              "n_any_field_disagreements_among_valid_pairs": len(discrepancies),
              "n_all_fields_agree_among_valid_pairs": sum(row["valid_pair"] and not row["differing_fields"] for row in comparison),
              "n_unresolved_union": len(unresolved),
              "per_field": field_report,
              "sampling": {"seed": args.seed, "unit": "original (condition, source, claim, item) review cell",
                           "deduplicated": False,
                           "unresolved_definition": "existing _unresolved: invalid grade, final_truth=unclear, or any null Boolean, from either grader",
                           "population_counts": {"unresolved_union": len(unresolved),
                                                 "resolved_disagreements": len(disagreements),
                                                 "resolved_concordant_controls": len(controls)},
                           "requested_sample_counts": {"resolved_disagreements": args.disagreements,
                                                       "resolved_concordant_controls": args.controls},
                           "selected_counts": selected_counts, "n_selected": len(packet),
                           "selection": "census of unresolved union; lowest SHA-256 ranks separately within otherwise-resolved disagreement and concordant-control populations",
                           "rank_definition": "SHA-256 of canonical JSON {seed,purpose,identity}; purposes disagreement, control, presentation",
                           "presentation": "all selected cells mixed by independent presentation rank before sequential blind IDs are assigned",
                           "primary_priority": "only in coordinator key: unresolved false-claim jspace_generated/matched_generated rows first",
                           "selection_uses_label_direction_or_effect": False},
              "interpretation": "Agreement compares graders and does not establish semantic accuracy. Human labels are pending.",
              "csv_formula_protection": "A leading apostrophe is added to cells with a leading formula marker or control character. JSONL and HTML preserve exact response text.",
              "metadata_blinding_limit": "The exact response itself may contain source, agent, reward, or other contextual clues."}
    rescore._atomic_jsonl(output / "comparison_rows.jsonl", comparison)
    rescore._atomic_jsonl(output / "disagreements.jsonl", discrepancies)
    rescore._atomic_jsonl(output / "coordinator_key.jsonl", answer_key)
    rescore._atomic_jsonl(output / "reviewer_packet.jsonl", packet)
    write_csv(output / "per_field_confusion.csv", confusion_rows, ["field", "a_label", "b_label", "count"])
    write_csv(output / "comparison_rows.csv", comparison, list(comparison[0]))
    write_csv(output / "disagreements.csv", discrepancies, list(comparison[0]))
    packet_fields = ["review_id", "reference", "response", "parsed_final", "parsed_review",
                     "evidence_excerpts", *HUMAN_FIELDS]
    write_csv(output / "reviewer_packet.csv", packet, packet_fields)
    write_html(output, packet)
    rescore._atomic_text(output / "rubric.txt", scoring_v3.FACT_JUDGE_RUBRIC)
    rescore._atomic_text(output / "README.md", guide(report))
    report["implementation_files_sha256"] = {
        str(path.relative_to(ROOT)): rescore._file_hash(path) for path in (
            Path(__file__).resolve(), ROOT / "src/jpeers/followup/scoring_v3.py",
            ROOT / "src/jpeers/followup/scoring.py", ROOT / "src/jpeers/followup/rescore.py",
            ROOT / "src/jpeers/data/imposter_facts.json")}
    report["artifacts_sha256"] = {path.name: rescore._file_hash(path)
                                  for path in sorted(output.iterdir()) if path.is_file()
                                  and path.name != "comparison_summary.json"}
    rescore._atomic_json(output / "comparison_summary.json", report)
    return report


def guide(report: dict) -> str:
    sampling = report["sampling"]
    populations, selected = sampling["population_counts"], sampling["selected_counts"]
    return f"""# Human review packet

Human review is pending. All human label, evidence, rationale, and reviewer ID cells are empty.

Give the reviewer only `reviewer_packet.csv`, `reviewer_packet.html`, `reviewer_packet.jsonl`, and `rubric.txt`. Keep this guide, the coordinator key, comparison summary, and comparison inventories away from reviewers until labels are frozen. The HTML provides readable exact saved text and numbered evidence excerpts. Enter labels and supporting evidence in the CSV. Use `null` for an uncertain Boolean and leave fields empty until reviewed. `final_truth` accepts true, false, both, missing, or unclear. Save reviewer ID and rationale with each completed case. Do not substitute grader predictions for human judgments.

The CSV prefixes potentially formula-interpreted cells with an apostrophe. The JSONL and HTML retain exact text, so consult those if a CSV cell starts with a protection prefix. Exact response content can reveal contextual clues even though experiment metadata, predictions, failures, selection reasons, and model identities are omitted. The common frozen rubric is included verbatim. Each Boolean is judged independently. An invalid machine judgment stays invalid in the original artifacts.

## Sampling and coordinator instructions

The population has {report['n_rows']:,} paired row identities. Both graders returned structurally valid judgments on {report['n_valid_pairs']:,} rows. There are {report['n_any_field_disagreements_among_valid_pairs']:,} rows with a difference in at least one of the five rubric labels among those valid pairs. The per-field matrices include valid unclear and null judgments and retain their categories.

The packet contains {sampling['n_selected']} rows. It includes all {populations['unresolved_union']} rows unresolved by either grader, {selected.get('resolved_disagreement_sample', 0)} of {populations['resolved_disagreements']} otherwise-resolved disagreements, and {selected.get('resolved_concordant_control_sample', 0)} of {populations['resolved_concordant_controls']} otherwise-resolved agreements. Requested sample maxima are {sampling['requested_sample_counts']['resolved_disagreements']} and {sampling['requested_sample_counts']['resolved_concordant_controls']}. Sampling is without replacement over original review cells. Identical texts in different cells remain separate and are mapped in the key.

The seed is `{sampling['seed']}`. Separate SHA-256 rankings select disagreement and control samples. An independent presentation ranking mixes every selected case before blind review IDs are assigned. Canonical JSON hashing uses the repo's `_digest`. Neither the desired effect, direction of disagreement, truth label, nor condition is used for sample ranking. This enriched audit sample cannot be treated as a simple random sample of all outputs or used as an unweighted estimate of grader accuracy.

For internal scheduling, sort `coordinator_key.jsonl` by `coordinator_priority`, then `review_id`. Priority 0 contains unresolved rows in the original primary false-claim pair, priority 1 other unresolved rows, priority 2 sampled disagreements, and priority 3 agreement controls. Do not show these priorities or selection reasons to the reviewer. Keep the combined blinded packet order for review where practical. If assigning smaller batches, mix cases before delivery to preserve blinding.

The key contains original identities, both grader identities, labels, exact evidence, raw outputs, errors, provenance, hashes, and selection reasons. It does not contain an adjudicated answer. `comparison_rows.jsonl` and CSV retain every population cell, including invalid pairs. `disagreements.jsonl` and CSV contain all valid pairs with at least one label difference. Evidence wording differences alone do not define a label disagreement. `comparison_summary.json` records all inputs and artifact hashes. These are agreement statistics, and human adjudication is required to evaluate semantic correctness.

## Reproduce

From the repository root, run `PYTHONPATH=src python scripts/prepare_exp4_adjudication.py` in a Python environment with the repo's pydantic dependency. The default paths refer to the frozen downloaded runs. The script validates the complete identity grid, identical original inputs, frozen references, rubric version and hash, stored raw valid judgments, and original unresolved/invalid schemas. It performs no GPU or model calls. Generation stops if existing packet CSV or JSONL files contain human entries. Use a new output directory to preserve completed reviews.
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grader-a", type=Path, default=ROOT / "runs/cluster-qwen3-32b-exp4-nemotron-20261106")
    parser.add_argument("--grader-b", type=Path, default=ROOT / "runs/cluster-qwen3-32b-exp4-rescore-v3-20261105")
    parser.add_argument("--output", type=Path, default=ROOT / "results/qwen3-32b-exp4-analysis-20261107/adjudication")
    parser.add_argument("--seed", default="exp4-adjudication-20261107-v1")
    parser.add_argument("--disagreements", type=int, default=50)
    parser.add_argument("--controls", type=int, default=25)
    args = parser.parse_args()
    if args.disagreements < 0 or args.controls < 0:
        parser.error("Sample counts must be nonnegative")
    report = prepare(args)
    print(json.dumps({"human_review_status": report["human_review_status"],
                      "n_valid_pairs": report["n_valid_pairs"],
                      "n_disagreements": report["n_any_field_disagreements_among_valid_pairs"],
                      "sampling": report["sampling"], "output": str(args.output.resolve())}, indent=2))


if __name__ == "__main__":
    main()
