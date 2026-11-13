"""Analyze frozen Exp4 judgments with fact clusters and unresolved-outcome bounds.

This script uses only the Python standard library. Its sign-flip statistic follows
followup.common.paired_stats, with exact integer dynamic programming for the binary
outcome lattice instead of Monte Carlo when there are more than 20 facts.
"""

from __future__ import annotations

import argparse
from collections import Counter
from fractions import Fraction
from functools import lru_cache
import hashlib
import itertools
import json
import math
from pathlib import Path
import random


ROOT = Path(__file__).resolve().parents[1]
SOURCES = ("incentive", "no_incentive", "document")
CONDITIONS = (
    "none", "jspace_light", "matched_k5", "jspace_medium", "matched_k10",
    "jspace_heavy", "matched_k25", "jspace_generated", "matched_generated",
    "jspace_peer_span", "matched_peer_span", "suppress_words", "steer_suspicion",
)
COMPARISONS = (
    ("generated_vs_matched", "jspace_generated", "matched_generated"),
    ("generated_vs_none", "jspace_generated", "none"),
    ("matched_generated_vs_none", "matched_generated", "none"),
    ("light_vs_matched", "jspace_light", "matched_k5"),
    ("medium_vs_matched", "jspace_medium", "matched_k10"),
    ("heavy_vs_matched", "jspace_heavy", "matched_k25"),
    ("peer_span_vs_matched", "jspace_peer_span", "matched_peer_span"),
)
METRICS = ("semantic_final_correct", "usable")
JOINT = ("correct_usable", "correct_unusable", "wrong_usable", "wrong_unusable")
FIELDS = ("review_rejects_false", "intent_accusation", "usable", "refusal")
TRUTH = ("true", "false", "both", "missing", "unclear")
COMPANION_NAMES = (
    "queue-location.json", "run.json", "gate.json", "passing-gate.json",
    "transport-retry.json", "artifact-receipt.json", "inventory.json",
)


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def digest(value):
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def recorded_path(path):
    path = path.resolve()
    return str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)


def row_key(row):
    return tuple(row[name] for name in ("condition", "source", "claim", "item"))


def outcome(row, metric):
    if not row["semantic_grade"]["valid"]:
        return None
    return row["semantic_grade"]["judgment"][metric]


def unresolved(row):
    grade = row["semantic_grade"]
    if not grade["valid"]:
        return True
    judgment = grade["judgment"]
    return judgment["final_truth"] == "unclear" or any(judgment[f] is None for f in FIELDS)


def load_grader(path, facts):
    path = path.resolve()
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    index = {}
    for row in rows:
        key = row_key(row)
        if key in index:
            raise ValueError(f"Duplicate review identity in {path}: {key}")
        index[key] = row
        grade = row["semantic_grade"]
        if type(grade["valid"]) is not bool:
            raise ValueError(f"Invalid grade envelope: {key}")
        if grade["valid"]:
            judgment = grade["judgment"]
            if judgment["final_truth"] not in TRUTH:
                raise ValueError(f"Unknown truth category: {key}")
            correct = None if judgment["final_truth"] == "unclear" else int(
                judgment["final_truth"] == "true")
            if judgment["semantic_final_correct"] != correct:
                raise ValueError(f"Saved correctness conflicts with truth category: {key}")
            if any(judgment[f] is not None and type(judgment[f]) is not bool for f in FIELDS):
                raise ValueError(f"Invalid saved Boolean judgment: {key}")
        elif grade.get("judgment") is not None:
            raise ValueError(f"Invalid grade contains a parsed judgment: {key}")
        if row["legacy"]["claim_in_draft"] is not True:
            raise ValueError(f"Claim absent from draft; explicit exclusion analysis required: {key}")
    expected = set(itertools.product(CONDITIONS, SOURCES, ("false", "true"), facts))
    if set(index) != expected or len(rows) != 4680:
        raise ValueError(f"Expected the complete 13 x 3 x 2 x 60 frozen review grid: {path}")
    summary_path = path.with_name("summary.json")
    summary = read_json(summary_path)
    if (summary["n_reviews"] != len(rows)
            or summary["n_valid"] != sum(r["semantic_grade"]["valid"] for r in rows)
            or summary["n_unresolved"] != sum(unresolved(r) for r in rows)):
        raise ValueError(f"Saved summary disagrees with its scored rows: {summary_path}")
    gate = summary["calibration"]
    if (not gate["passed"] or any(not gate[p]["passed"] for p in ("development", "holdout"))
            or any(gate[p]["grader_info"] != summary["grader_info"]
                   for p in ("development", "holdout"))):
        raise ValueError(f"Both frozen calibration gates must pass for the same grader: {path}")
    # The shared outputs, references, metadata and origin hashes must match exactly
    # between graders. Only the grader-specific envelopes are excluded.
    excluded = {"cache_key", "semantic_grade", "semantic_grade_status"}
    frozen_rows = [{k: v for k, v in index[key].items() if k not in excluded}
                   for key in sorted(index)]
    provenance = {
        "scored_rows": recorded_path(path), "scored_rows_sha256": file_hash(path),
        "summary": recorded_path(summary_path), "summary_sha256": file_hash(summary_path),
        "frozen_reviews_sha256": digest(frozen_rows), "grader_info": summary["grader_info"],
        "n_reviews": len(rows), "n_valid": summary["n_valid"],
        "n_unresolved_any_label": summary["n_unresolved"],
        "calibration": {p: {k: gate[p][k] for k in (
            "passed", "n_cases", "n_valid", "n_agree", "fixture_sha256",
            "critical_false_positive_ids")} for p in ("development", "holdout")},
        "origin_files": {r["origin"]["turns_file"]: r["origin"]["turns_sha256"] for r in rows},
    }
    companions = {}
    for name in COMPANION_NAMES:
        companion = path.with_name(name)
        if companion.is_file():
            companions[recorded_path(companion)] = file_hash(companion)
    provenance["companion_files"] = companions
    return rows, index, provenance


def rate(values):
    values = list(values)
    known = [int(v) for v in values if v is not None]
    n, positive, unknown = len(values), sum(known), len(values) - len(known)
    return {"n_total": n, "n_known": len(known), "n_positive": positive,
            "n_negative": len(known) - positive, "n_unknown": unknown,
            "mean_among_known": positive / len(known) if known else None,
            "bounds": {"lower": positive / n, "upper": (positive + unknown) / n}}


def joint_outcome(correct, usable):
    return ("correct" if correct else "wrong") + ("_usable" if usable else "_unusable")


def describe(rows):
    joint = Counter({label: 0 for label in (*JOINT, "unknown")})
    possible = Counter({label: 0 for label in JOINT})
    for row in rows:
        correct, usable = (outcome(row, m) for m in METRICS)
        if correct is None or usable is None:
            joint["unknown"] += 1
        else:
            joint[joint_outcome(correct, usable)] += 1
        for c, u in itertools.product((0, 1) if correct is None else (correct,),
                                      (False, True) if usable is None else (usable,)):
            possible[joint_outcome(c, u)] += 1
    fluency = [row["legacy"]["fluency"] for row in rows]
    if any(type(v) not in (int, float) or not math.isfinite(v) for v in fluency):
        raise ValueError("Legacy fluency values must be finite")
    return {
        "n_reviews": len(rows), "n_invalid": sum(not r["semantic_grade"]["valid"] for r in rows),
        "n_unresolved_any_label": sum(unresolved(r) for r in rows),
        "final_truth_counts": dict(Counter(r["semantic_grade"]["judgment"]["final_truth"]
                                         if r["semantic_grade"]["valid"] else "invalid" for r in rows)),
        **{m: rate(outcome(row, m) for row in rows) for m in METRICS},
        "joint_counts": dict(joint),
        "joint_rates": {label: count / len(rows) for label, count in joint.items()},
        "joint_bounds": {label: {"lower": joint[label] / len(rows),
                                 "upper": possible[label] / len(rows)} for label in JOINT},
        "legacy_fluency": {"n": len(fluency), "mean": sum(fluency) / len(fluency)},
    }


@lru_cache(maxsize=None)
def paired_stats(values, bootstrap_samples=10000, permutations=50000, seed=0):
    """Use the same two-sided sign-flip null and missing-cluster policy as common.py."""
    known = [v for v in values if v is not None]
    n = len(known)
    result = {"n": n, "n_total": len(values), "n_missing": len(values) - n,
              "diff": float(sum(known) / n) if n else None, "p": None,
              "p_exact": None, "method": "no_valid_pairs", "seed": seed}
    if not known:
        return result
    # Binary outcomes averaged across one or three sources lie on this small
    # lattice. Counting signed sums gives the full 2**n null without enumeration.
    denominator = math.lcm(*(v.denominator for v in known))
    weights = [int(v * denominator) for v in known]
    if sum(abs(w) for w in weights) <= 10000:
        distribution = {0: 1}
        for weight in weights:
            updated = Counter()
            for total, count in distribution.items():
                updated[total - weight] += count
                updated[total + weight] += count
            distribution = updated
        observed = abs(sum(weights))
        hits = sum(count for total, count in distribution.items() if abs(total) >= observed)
        probability = hits / (2 ** n)
        result.update(p=probability, p_exact=probability, method="exact_cluster_sign_flip_dp",
                      null_assignments=2 ** n, null_hits=hits)
    else:
        rng = random.Random(seed)
        observed = abs(sum(known))
        hits = sum(abs(sum(v if rng.getrandbits(1) else -v for v in known)) >= observed
                   for _ in range(permutations))
        result.update(p=(hits + 1) / (permutations + 1),
                      method="monte_carlo_cluster_sign_flip", permutations=permutations)
    rng = random.Random(seed)
    numbers = [float(v) for v in known]
    means = sorted(sum(rng.choices(numbers, k=n)) / n for _ in range(bootstrap_samples))
    result["bootstrap_ci95"] = [means[int(0.025 * (bootstrap_samples - 1))],
                                means[int(0.975 * (bootstrap_samples - 1))]]
    result["bootstrap_samples"] = bootstrap_samples
    result["bootstrap_unit"] = "fact, complete clusters only"
    return result


def contrast(index, facts, condition, control, sources, claim, metric, settings):
    claims = (("false", 1), ("true", -1)) if claim == "false_minus_true" else ((claim, 1),)
    differences, lower, upper, complete_pairs = [], [], [], []
    fact_rows = []
    for fact in facts:
        cluster = []
        for source in sources:
            terms = []
            for state, sign in claims:
                for cond, coefficient in ((condition, sign), (control, -sign)):
                    terms.append((coefficient, outcome(index[(cond, source, state, fact)], metric)))
            low = sum(sign * value if value is not None else min(0, sign) for sign, value in terms)
            high = sum(sign * value if value is not None else max(0, sign) for sign, value in terms)
            difference = None if any(v is None for _, v in terms) else sum(s * v for s, v in terms)
            cluster.append((difference, low, high))
            if difference is not None:
                complete_pairs.append(difference)
        value = (Fraction(sum(d for d, _, _ in cluster), len(sources))
                 if all(d is not None for d, _, _ in cluster) else None)
        lo = Fraction(sum(v for _, v, _ in cluster), len(sources))
        hi = Fraction(sum(v for _, _, v in cluster), len(sources))
        differences.append(value)
        lower.append(lo)
        upper.append(hi)
        fact_rows.append({"fact": fact, "difference": float(value) if value is not None else None,
                          "lower": float(lo), "upper": float(hi)})
    result = {
        "n_expected_pairs": len(facts) * len(sources), "n_complete_pairs": len(complete_pairs),
        "difference_among_complete_pairs": (sum(complete_pairs) / len(complete_pairs)
                                            if complete_pairs else None),
        "bounds": {"lower": float(sum(lower) / len(facts)),
                   "upper": float(sum(upper) / len(facts))},
        "fact_cluster_stats": paired_stats(tuple(differences), **settings),
    }
    return result, fact_rows


def analyze(index, facts, settings):
    scopes = {"all": SOURCES, **{source: (source,) for source in SOURCES}}
    conditions = {}
    for condition in CONDITIONS:
        conditions[condition] = {
            scope: {claim: describe([index[(condition, source, claim, fact)]
                                    for source in sources for fact in facts])
                    for claim in ("false", "true")}
            for scope, sources in scopes.items()}
    comparisons, by_fact = {}, {}
    for name, condition, control in COMPARISONS:
        comparison = {"condition": condition, "control": control, "by_source": {}}
        for scope, sources in scopes.items():
            comparison["by_source"][scope] = {}
            for claim in ("false", "true", "false_minus_true"):
                comparison["by_source"][scope][claim] = {}
                for metric in METRICS:
                    result, facts_result = contrast(index, facts, condition, control, sources,
                                                   claim, metric, settings)
                    comparison["by_source"][scope][claim][metric] = result
                    if name == "generated_vs_matched" and scope == "all":
                        by_fact.setdefault(claim, {})[metric] = facts_result
        comparisons[name] = comparison
    source_interactions = {}
    for source in SOURCES[:2]:
        source_interactions[source + "_vs_document"] = {}
        for claim in ("false", "true", "false_minus_true"):
            source_interactions[source + "_vs_document"][claim] = {}
            for metric in METRICS:
                _, a = contrast(index, facts, "jspace_generated", "matched_generated",
                                (source,), claim, metric, settings)
                _, b = contrast(index, facts, "jspace_generated", "matched_generated",
                                ("document",), claim, metric, settings)
                differences = [None if x["difference"] is None or y["difference"] is None
                               else Fraction(x["difference"] - y["difference"])
                               for x, y in zip(a, b, strict=True)]
                source_interactions[source + "_vs_document"][claim][metric] = {
                    "definition": f"J-minus-matched effect for {source} minus document",
                    "bounds": {
                        "lower": sum(x["lower"] - y["upper"] for x, y in zip(a, b, strict=True)) / len(facts),
                        "upper": sum(x["upper"] - y["lower"] for x, y in zip(a, b, strict=True)) / len(facts)},
                    "fact_cluster_stats": paired_stats(tuple(differences), **settings),
                }
    return {"conditions": conditions, "comparisons": comparisons,
            "focal_fact_differences": by_fact, "source_interactions": source_interactions}


def percentage(value):
    return "unknown" if value is None else f"{100 * value:.1f}"


def effect_text(result):
    stats, bounds = result["fact_cluster_stats"], result["bounds"]
    p = "unknown" if stats["p"] is None else f"{stats['p']:.3g}"
    return (f"{percentage(stats['diff'])} | [{percentage(bounds['lower'])}, "
            f"{percentage(bounds['upper'])}] | {stats['n']}/60 | {p}")


def report_markdown(report):
    lines = [
        "# Qwen3-32B Experiment 4 specificity analysis", "",
        "This is an exploratory analysis of 4,680 frozen subject responses graded independently "
        "by Nemotron and GPT-OSS. Each grid contains 60 facts, three repeated source framings, "
        "true and false peer claims, and 13 interventions. Every review remains in the rate "
        "denominators. No response or grader label was changed.", "",
        "Both graders show substantially more correctness harm from the generated-token "
        "J-lens edit on false claims than on true claims. The all-fact false-minus-true "
        "bounds exclude zero even under adverse assignments of unresolved outcomes. "
        "False claims also show harm when the draft is presented as a document. The "
        "peer-versus-document comparisons have wide fact-bootstrap intervals and "
        "do not establish a larger effect specific to peer interaction.", "",
        "The focal edit removes the ten most active J-lens directions at selected layers while "
        "generating. It tests this intervention and task. The model still contains other "
        "J-lens directions, and the peer-reading positions are outside this focal edit.", "",
        "## Matched-control effects", "",
        "Differences are J/generated minus its size- and position-matched random control, in "
        "percentage points. Negative correctness effects mean fewer correct final documents. "
        "The interaction is the false-claim effect minus the true-claim effect. A negative "
        "interaction means false claims show greater harm. Each fact contributes the mean "
        "of its three source differences. The fact bank supplies 60 clusters.", "",
        "The point estimate and p value use complete fact clusters. Bounds use all 60 facts "
        "and every source pair, assigning each unresolved outcome its most adverse or most "
        "favorable binary value. They are identification bounds, separate from sampling "
        "uncertainty. All tests are two-sided sign flips under symmetry of fact differences "
        "and are unadjusted exploratory p values. Exact dynamic programming counts all sign "
        "assignments. The JSON also provides seeded percentile bootstrap intervals over "
        "complete fact clusters.", "",
        "The complete-cluster estimate can fall outside the all-fact bounds because "
        "these quantities summarize different fact sets. Full-bank effect magnitude "
        "should be read from the bounds. The JSON also records complete source-pair "
        "estimates for comparison with the original grading summaries.", "",
        "| Grader | Claim | Metric | Complete-fact effect, pp | All-fact bounds, pp | Facts | p |",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    for name, data in report["graders"].items():
        focal = data["analysis"]["comparisons"]["generated_vs_matched"]["by_source"]["all"]
        for claim in ("false", "true", "false_minus_true"):
            for metric in METRICS:
                lines.append(f"| {name} | {claim} | {metric} | {effect_text(focal[claim][metric])} |")
    lines += ["", "## Source-specific correctness", "",
              "Each source analysis still has 60 fact units. Source rows are repeated "
              "measurements of the same facts, so the combined analysis never treats 180 "
              "pairs as 180 independent facts.", "",
              "| Grader | Source | Claim | Complete-fact effect, pp | All-fact bounds, pp | Facts | p |",
              "|---|---|---|---:|---:|---:|---:|"]
    for name, data in report["graders"].items():
        focal = data["analysis"]["comparisons"]["generated_vs_matched"]["by_source"]
        for source in SOURCES:
            for claim in ("false", "true", "false_minus_true"):
                result = focal[source][claim]["semantic_final_correct"]
                lines.append(f"| {name} | {source} | {claim} | {effect_text(result)} |")
    lines += ["", "## Peer versus document specificity", "",
              "These interactions compare the generated-minus-matched effect for each "
              "peer framing with the effect for a document, using the same fact as the "
              "independent unit. A negative effect favors a larger disruption under "
              "the peer framing. The false_minus_true rows additionally compare the "
              "claim interaction across framings. The framings also differ in their "
              "prompts and incentives, so an agency interpretation requires care.", "",
              "| Grader | Peer contrast | Claim | Complete-fact correctness interaction, pp | All-fact bounds, pp | Facts | p |",
              "|---|---|---|---:|---:|---:|---:|"]
    for name, data in report["graders"].items():
        for source, values in data["analysis"]["source_interactions"].items():
            for claim in ("false", "true", "false_minus_true"):
                lines.append(f"| {name} | {source} | {claim} | "
                             f"{effect_text(values[claim]['semantic_final_correct'])} |")
    lines += ["", "## Unedited baseline", "",
              "The two comparisons with the unedited baseline distinguish harm caused by "
              "the focal edit from harm caused by its random matched control.", "",
              "| Grader | Comparison | Claim | Complete-fact correctness effect, pp | All-fact bounds, pp | Facts | p |",
              "|---|---|---|---:|---:|---:|---:|"]
    for name, data in report["graders"].items():
        for comparison in ("generated_vs_none", "matched_generated_vs_none"):
            values = data["analysis"]["comparisons"][comparison]["by_source"]["all"]
            for claim in ("false", "true"):
                lines.append(f"| {name} | {comparison} | {claim} | "
                             f"{effect_text(values[claim]['semantic_final_correct'])} |")
    lines += ["", "## Joint correctness and usability", "",
              "These are counts out of 180 responses per row, including unusable outcomes. "
              "The requested wrong cells mean semantic_final_correct=0, which includes "
              "false assertions, contradictory answers, and missing answers. The JSON "
              "retains final_truth counts to distinguish those categories. Unknown means "
              "correctness or usability is unresolved. A known label remains usable in "
              "its own marginal analysis when another label is unresolved.", "",
              "| Grader | Condition | Claim | Correct + usable | Correct + unusable | Wrong + usable | Wrong + unusable | Unknown |",
              "|---|---|---|---:|---:|---:|---:|---:|"]
    for name, data in report["graders"].items():
        for condition in ("none", "matched_generated", "jspace_generated"):
            for claim in ("false", "true"):
                counts = data["analysis"]["conditions"][condition]["all"][claim]["joint_counts"]
                cells = " | ".join(str(counts[key]) for key in (*JOINT, "unknown"))
                lines.append(f"| {name} | {condition} | {claim} | {cells} |")
    lines += ["", "## Fluency and scope of interpretation", "",
              "The original false-claim fluency check is evaluated on all 180 primary "
              "pairs, all of which have claim_in_draft=true. Its threshold is "
              "J/generated minus matched-generated at least -0.1. True-claim fluency "
              "is shown separately as an exploratory control.", "",
              "Legacy fluency is the ratio of distinct word bigrams to all word bigrams "
              "in the full response, with a zero score for very short replies. It detects "
              "some repetitive collapse and does not establish grammaticality, successful "
              "fact correction, or preserved general communication ability. Usability is "
              "separately judged from the final document and can be true for incorrect "
              "or off-target coherent prose.", "",
              "| Claim | J/generated fluency | Matched-generated fluency | Difference | Within original 0.1 margin |",
              "|---|---:|---:|---:|---|"]
    first = next(iter(report["graders"].values()))["analysis"]["conditions"]
    for claim in ("false", "true"):
        a = first["jspace_generated"]["all"][claim]["legacy_fluency"]["mean"]
        b = first["matched_generated"]["all"][claim]["legacy_fluency"]["mean"]
        lines.append(f"| {claim} | {a:.4f} | {b:.4f} | {a-b:.4f} | {a-b >= -0.1} |")
    lines += ["", "The original false-claim fluency margin passes. The semantic usability "
              "analysis nevertheless shows an additional loss of coherent final prose "
              "under J/generated, and the joint table also shows many coherent but "
              "incorrect or incomplete final documents. Neither dropping unusable "
              "responses nor relying only on the lexical fluency score captures both "
              "forms of task failure.", "",
              "The original preregistration used lexical catch_score, a source/fact "
              "pair as its test unit, and a fluency margin of 0.1 for false claims. This "
              "analysis instead uses audited semantic labels, fact clusters, true-claim "
              "controls, and interactions after outputs were available. The semantic "
              "scorer underwent documented development calibration. Nemotron also used "
              "a timeout recovery attempt, and GPT-OSS used a predeclared technical "
              "fallback for eligible failures. These choices are recorded in the grader "
              "provenance and support exploratory interpretation.", "",
              "Qwen3-32B did not pass the original Experiment 0 belief-tracking gate. "
              "At the story-end site, its agent true-belief control rate was 0.55 and "
              "false-belief rate was 0.45, each with saved p=0.8238. Later prompt audits "
              "identified setup issues, so these outcomes do not establish validated "
              "belief tracking in the intervention subspace. Exp4 supports statements "
              "about sensitivity of this fact-sheet task to selected J-lens edits. A "
              "general claim that J-space is necessary for multi-agent communication "
              "would require additional validated tasks and interventions.", "",
              "## Reproduction and provenance", "",
              "Run `python3 scripts/analyze_exp4_specificity.py` from the repository. "
              "The script uses the standard library and performs no model inference. "
              "The JSON records exact scored-row hashes, shared original-review hashes, "
              "grader identities, calibration counts, original turn-file hashes, analysis "
              "source hashes, deterministic seed, and bootstrap settings. It also contains "
              "all 13 condition summaries, all five existing matched J-lens comparisons, "
              "and both focal comparisons with the unedited baseline.", ""]
    for name, provenance in report["inputs"].items():
        lines += [f"- {name} scored rows SHA-256 `{provenance['scored_rows_sha256']}`."]
    lines += [f"- Shared original reviews SHA-256 `{report['shared_frozen_reviews_sha256']}`.",
              f"- Analysis script SHA-256 `{report['source_files']['scripts/analyze_exp4_specificity.py']}`.",
              "", "Machine-readable results are in "
              "[specificity.json](../results/qwen3-32b-exp4-analysis-20261107/specificity.json).", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nemotron", type=Path, default=ROOT / "runs/cluster-qwen3-32b-exp4-nemotron-20261106/scored_rows.jsonl")
    parser.add_argument("--gpt-oss", type=Path, default=ROOT / "runs/cluster-qwen3-32b-exp4-rescore-v3-20261105/scored_rows.jsonl")
    parser.add_argument("--output", type=Path, default=ROOT / "results/qwen3-32b-exp4-analysis-20261107/specificity.json")
    parser.add_argument("--report", type=Path, default=ROOT / "docs/2026-11-07-exp4-specificity.md")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--permutations", type=int, default=50000)
    args = parser.parse_args()
    if args.bootstrap_samples < 100 or args.permutations < 1:
        parser.error("Use at least 100 bootstrap samples and one permutation")
    facts_path = ROOT / "src/jpeers/data/imposter_facts.json"
    facts = [item["id"] for item in read_json(facts_path)["items"]]
    if len(facts) != 60 or len(set(facts)) != 60:
        raise ValueError("Expected 60 unique independent fact IDs")
    source_paths = [Path(__file__), facts_path, ROOT / "src/jpeers/followup/common.py",
                    ROOT / "src/jpeers/followup/scoring_v3.py", ROOT / "src/jpeers/followup/scoring.py",
                    ROOT / "src/jpeers/interventions.py", ROOT / "src/jpeers/analysis.py",
                    ROOT / "docs/07_preregistration.md",
                    ROOT / "results/qwen3-32b-exp0-20261104/tom_validation/qwen3-32b-exp0-attempt-2-1793812722733295233/summary.json"]
    source_hashes = {str(p.relative_to(ROOT)): file_hash(p) for p in source_paths}
    settings = {"seed": args.seed, "bootstrap_samples": args.bootstrap_samples,
                "permutations": args.permutations}
    protected = {path.resolve() for path in source_paths}
    grading_directories = {path.resolve().parent for path in (args.nemotron, args.gpt_oss)}
    for path in (args.nemotron, args.gpt_oss):
        protected.add(path.resolve())
        protected.update(path.with_name(name).resolve() for name in ("summary.json", *COMPANION_NAMES))
    inputs, loaded = {}, {}
    for name, path in (("Nemotron", args.nemotron), ("GPT-OSS", args.gpt_oss)):
        _, index, provenance = load_grader(path, facts)
        inputs[name] = provenance
        loaded[name] = index
        protected.update((ROOT / path).resolve() for path in provenance["companion_files"])
        protected.update((ROOT / path).resolve() for path in provenance["origin_files"])
    destinations = (args.output.resolve(), args.report.resolve())
    if (any(path in protected or any(path.is_relative_to(root) for root in grading_directories)
            for path in destinations) or destinations[0] == destinations[1]):
        raise ValueError("Analysis outputs must be separate from frozen inputs and source files")
    fingerprints = {p["frozen_reviews_sha256"] for p in inputs.values()}
    if len(fingerprints) != 1:
        raise ValueError("Graders did not judge exactly the same frozen original reviews")
    graders = {name: {"analysis": analyze(index, facts, settings)} for name, index in loaded.items()}
    report = {
        "protocol": "exploratory-exp4-specificity-v1", "date": "2026-11-07",
        "independent_unit": "fact; source framings are repeated measurements",
        "n_facts": len(facts), "sources": list(SOURCES), "settings": settings,
        "metric_definitions": {
            "semantic_final_correct": "Saved final_truth=true is 1, unclear is unknown; false/both/missing are 0",
            "usable": "Saved coherent final-prose label, independently of factual correctness",
            "wrong_joint_label": "semantic_final_correct=0, including missing and contradictory answers",
            "interaction": "(J - control on false claims) - (J - control on true claims)",
            "bounds": "All expected fact/source cells; each unknown binary outcome ranges from 0 to 1",
            "inference": "Complete fact clusters only; two-sided sign-flip symmetry test, unadjusted exploratory p values",
            "bootstrap": "Seeded percentile 95% interval over complete fact clusters; distinct from unresolved bounds",
        },
        "inputs": inputs, "shared_frozen_reviews_sha256": next(iter(fingerprints)),
        "source_files": source_hashes, "graders": graders,
    }
    for relative, before in source_hashes.items():
        if file_hash(ROOT / relative) != before:
            raise ValueError(f"Analysis source changed during the run: {relative}")
    for provenance in inputs.values():
        for path_key, hash_key in (("scored_rows", "scored_rows_sha256"), ("summary", "summary_sha256")):
            if file_hash(ROOT / provenance[path_key]) != provenance[hash_key]:
                raise ValueError("Frozen grading input changed during analysis")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    args.report.write_text(report_markdown(report))
    print(json.dumps({"output": str(args.output), "report": str(args.report),
                      "n_facts": len(facts), "seed": args.seed}))


if __name__ == "__main__":
    main()
