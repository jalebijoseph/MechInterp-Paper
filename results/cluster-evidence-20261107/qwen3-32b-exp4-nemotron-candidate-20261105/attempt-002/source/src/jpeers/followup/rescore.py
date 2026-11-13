"""Calibrate one frozen grader and rescore saved outputs without regeneration.

All writes are confined to the supplied output/cache directories. A content key
includes the exact response, factual references, rubric and grader provenance.
Unknown grades remain unknown; missing input cells widen the reported bounds.
"""

from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import tempfile

from jpeers.followup.scoring import (
    CALIBRATION_GATE, JUDGMENT_FIELDS, RUBRIC_HASH, SCORER_VERSION,
    calibration_report, grade_response, parse_fact_judgment, scorer_cache_key,
)


DATA_DIR = Path(__file__).resolve().parents[1] / "data"
CALIBRATION_FILE = DATA_DIR / "fact_grader_calibration_v2.json"
FACTS_FILE = DATA_DIR / "imposter_facts.json"
EXP4_CONDITIONS = (
    "none", "jspace_light", "matched_k5", "jspace_medium", "matched_k10",
    "jspace_heavy", "matched_k25", "jspace_generated", "matched_generated",
    "jspace_peer_span", "matched_peer_span", "suppress_words", "steer_suspicion",
)
ORIGINAL_SOURCES = ("incentive", "no_incentive", "document")
_LEGACY_METRICS = ("catch_score", "fluency", "copies_peer", "reply_has_true",
                   "reply_flags", "reply_keeps_false")
_REFERENCE_NOTES = {
    "everest_continent": "Everest in the Himalayas on the Nepal-China border entails Asia.",
    "capital_nl": "Amsterdam is the capital; The Hague is the distinct seat of government.",
}


def _serialized(value) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False,
                      separators=(",", ":"))


def _digest(value) -> str:
    return hashlib.sha256(_serialized(value).encode("utf-8")).hexdigest()


def _file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key {key}")
        result[key] = value
    return result


def _loads(text: str):
    return json.loads(text, object_pairs_hook=_unique_object)


def _read_json(path: Path):
    return _loads(path.read_text(encoding="utf-8"))


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                         prefix="." + path.name + ".", suffix=".tmp",
                                         delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _atomic_json(path: Path, value) -> None:
    _atomic_text(path, json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def _atomic_jsonl(path: Path, rows: list[dict]) -> None:
    _atomic_text(path, "".join(_serialized(row) + "\n" for row in rows))


def _grader_info(grader_info: dict) -> dict:
    if not isinstance(grader_info, dict) or not grader_info:
        raise ValueError("grader_info must contain the frozen grader provenance")
    return _loads(_serialized(grader_info))


def _fixtures() -> tuple[dict, str]:
    fixture = _read_json(CALIBRATION_FILE)
    if fixture.get("scorer_version") != SCORER_VERSION or fixture.get("gate") != CALIBRATION_GATE:
        raise ValueError("calibration fixture version/gate does not match the scorer")
    return fixture, _file_hash(CALIBRATION_FILE)


def _require_calibration(calibration: dict, grader_info: dict) -> None:
    fixture, fixture_hash = _fixtures()
    if (not isinstance(calibration, dict) or calibration.get("passed") is not True
            or calibration.get("scorer_version") != SCORER_VERSION
            or calibration.get("rubric_sha256") != RUBRIC_HASH
            or calibration.get("fixture_sha256") != fixture_hash
            or calibration.get("gate") != CALIBRATION_GATE
            or calibration.get("grader_info") != grader_info):
        raise ValueError("calibration must pass the current frozen rubric, fixtures and same grader")
    total = len(fixture["cases"])
    valid, agree = calibration.get("n_valid"), calibration.get("n_agree")
    critical = calibration.get("critical_false_positive_ids")
    if (calibration.get("n_cases") != total or type(valid) is not int or type(agree) is not int
            or not 0 <= agree <= valid <= total
            or calibration.get("coverage") != valid / total
            or calibration.get("grade_agreement") != agree / total
            or valid / total < CALIBRATION_GATE["minimum_coverage"]
            or agree / total < CALIBRATION_GATE["minimum_grade_agreement"]
            or not isinstance(critical, list)
            or len(critical) > CALIBRATION_GATE["max_critical_false_positives"]):
        raise ValueError("calibration pass flag is inconsistent with its frozen gate statistics")


def _cache_grade(endpoint, reference: dict, response: str, grader_info: dict,
                 cache_dir: Path) -> tuple[dict, str, bool]:
    key = scorer_cache_key(reference, response, grader_info)
    path = cache_dir / (key + ".json")
    if path.exists():
        saved = _read_json(path)
        if (saved.get("cache_key") != key or saved.get("scorer_version") != SCORER_VERSION
                or saved.get("rubric_sha256") != RUBRIC_HASH
                or saved.get("grader_info") != grader_info
                or saved.get("response_sha256") != hashlib.sha256(response.encode()).hexdigest()):
            raise ValueError(f"cache provenance mismatch: {path}")
        grade = saved.get("grade")
        if (not isinstance(grade, dict) or type(grade.get("valid")) is not bool
                or grade.get("scorer_version") != SCORER_VERSION):
            raise ValueError(f"cache grade envelope is invalid: {path}")
        raw = grade.get("raw_response")
        if grade["valid"]:
            try:
                parsed = parse_fact_judgment(raw, response)
            except ValueError as exc:
                raise ValueError(f"cache contains invalid raw judgment: {path}") from exc
            if parsed != grade.get("judgment") or grade.get("error") is not None:
                raise ValueError(f"cache judgment differs from raw evidence: {path}")
            return grade, key, True
        if grade.get("judgment") is not None or not isinstance(grade.get("error"), str):
            raise ValueError(f"cache invalid-grade envelope is inconsistent: {path}")
        if raw is not None:
            try:
                parse_fact_judgment(raw, response)
            except ValueError:
                return grade, key, True
            raise ValueError(f"cache marks a valid raw judgment invalid: {path}")
        # Transport failures have no frozen model judgment. Retain the record but
        # retry them on a later invocation, unlike malformed model outputs.
    grade = grade_response(endpoint, reference, response)
    _atomic_json(path, {"cache_key": key, "scorer_version": SCORER_VERSION,
                       "rubric_sha256": RUBRIC_HASH, "grader_info": grader_info,
                       "response_sha256": hashlib.sha256(response.encode()).hexdigest(),
                       "grade": grade})
    return grade, key, False


def run_calibration(endpoint, output_dir, grader_info, *, cache_dir=None) -> dict:
    """Run the frozen controlled cases; persist actual judgments and the gate."""
    output = Path(output_dir).resolve()
    cache = Path(cache_dir).resolve() if cache_dir is not None else output / "cache"
    info = _grader_info(grader_info)
    fixture, fixture_hash = _fixtures()
    predictions = []
    cache_hits = 0
    for case in fixture["cases"]:
        grade, key, hit = _cache_grade(endpoint, case["item"], case["response"], info, cache)
        predictions.append({"id": case["id"], "cache_key": key, **grade})
        cache_hits += hit
        _atomic_json(output / "progress.json", {
            "kind": "calibration", "status": "running", "expected": len(fixture["cases"]),
            "processed": len(predictions), "cache_hits": cache_hits,
            "grader_calls": len(predictions) - cache_hits,
        })
    report = {**calibration_report(predictions, fixture["cases"]),
              "grader_info": info, "fixture_sha256": fixture_hash,
              "cache_hits": cache_hits, "grader_calls": len(predictions) - cache_hits,
              "predictions_artifact": str(output / "calibration_predictions.jsonl")}
    _atomic_jsonl(output / "calibration_predictions.jsonl", predictions)
    _atomic_json(output / "calibration.json", report)
    _atomic_json(output / "progress.json", {
        "kind": "calibration", "status": "complete" if report["passed"] else "failed",
        "expected": len(fixture["cases"]), "processed": len(predictions),
        "cache_hits": cache_hits, "grader_calls": len(predictions) - cache_hits,
    })
    return report


def _facts() -> dict[str, dict]:
    items = _read_json(FACTS_FILE)["items"]
    facts = {item["id"]: item for item in items}
    if len(items) != 60 or len(facts) != 60:
        raise ValueError("rescoring requires the original 60 unique fact references")
    return facts


def _reference(row: dict, facts: dict) -> dict:
    item = facts.get(row.get("item"))
    if item is None:
        raise ValueError(f"unknown original fact ID: {row.get('item')}")
    reference = {key: item[key] for key in ("question", "true_claim", "false_claim")}
    for key in ("true_claim", "false_claim"):
        if key in row and row[key] != reference[key]:
            raise ValueError(f"inline {key} differs from the frozen reference for {item['id']}")
    if item["id"] in _REFERENCE_NOTES:
        reference["reference_notes"] = _REFERENCE_NOTES[item["id"]]
    return reference


def _key(row: dict, kind: str) -> tuple:
    keys = ("condition", "source", "claim", "item")
    if kind == "exp2_v2":
        keys += ("reward", "verified")
    if any(key not in row for key in keys):
        raise ValueError("review metadata is missing required identity fields")
    return tuple(row[key] for key in keys)


def _expected_keys(kind: str, facts: dict) -> list[tuple]:
    if kind == "exp2_v2":
        return [("none", source, claim, item, reward, verified)
                for item in facts for claim in ("false", "true")
                for source, reward, verified in (
                    [("agent", reward, verified) for reward in (100, 0) for verified in (True, False)]
                    + [("document", None, None)])]
    conditions = EXP4_CONDITIONS if kind == "exp4" else ("none",)
    return list(itertools.product(conditions, ORIGINAL_SOURCES, ("false", "true"), facts))


def _validate_rows(rows: list[dict], kind: str, facts: dict) -> tuple[list[tuple], list[tuple]]:
    if kind not in ("exp4", "exp2", "exp2_v2"):
        raise ValueError(f"unsupported rescoring kind: {kind}")
    expected = _expected_keys(kind, facts)
    expected_set = set(expected)
    seen = set()
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get("text"), str):
            raise ValueError("every review requires its exact saved response text")
        if kind == "exp2_v2":
            if row.get("kind", "review") != "review":
                raise ValueError("grade_rows accepts review rows, never knowledge probes")
            if row.get("source") == "agent":
                if type(row.get("verified")) is not bool or type(row.get("reward")) is not int:
                    raise ValueError("agent review verified/reward metadata has invalid types")
            elif row.get("reward") is not None or row.get("verified") is not None:
                raise ValueError("document review reward/verified metadata must be null")
        key = _key(row, kind)
        if key not in expected_set:
            raise ValueError(f"unexpected review metadata cell: {key}")
        if key in seen:
            raise ValueError(f"Duplicate reviewed response identity: {key}")
        seen.add(key)
        _reference(row, facts)
    return expected, [key for key in expected if key not in seen]


def _discover(input_root: Path, kind: str) -> list[Path]:
    if not input_root.is_dir():
        raise ValueError("input_root must be a frozen results directory")
    direct = input_root / "turns.jsonl"
    candidates = [direct] if direct.exists() else sorted(input_root.rglob("turns.jsonl"))
    selected = []
    for path in candidates:
        config_path = path.parent / "config.json"
        config = _read_json(config_path) if config_path.exists() else {}
        if kind == "exp4":
            matches = config.get("experiment") == "ablation" and config.get("base") == "imposter"
            fallback = any(part.startswith(("exp4-", "ablation-exp4-")) for part in path.parts)
        else:
            matches = config.get("experiment") == "imposter" and config.get("live", True) is not False
            fallback = "exp2-core" in path.parts
        if matches or (not config and (fallback or path == direct)):
            selected.append(path)
    return selected


def _inventory(input_root: Path, kind: str) -> tuple[list[dict], dict]:
    rows = []
    files = {}
    for path in _discover(input_root, kind):
        text = path.read_text(encoding="utf-8")
        files[str(path)] = hashlib.sha256(text.encode("utf-8")).hexdigest()
        config_path = path.parent / "config.json"
        if config_path.exists():
            files[str(config_path)] = _file_hash(config_path)
        summary_path = path.parent / "summary.json"
        legacy_by_key = {}
        if summary_path.exists():
            summary = _read_json(summary_path)
            files[str(summary_path)] = _file_hash(summary_path)
            for legacy in summary.get("live", []):
                key = _key(legacy, kind)
                if key in legacy_by_key:
                    raise ValueError(f"Duplicate legacy summary identity: {key}")
                legacy_by_key[key] = legacy
        for number, line in enumerate(text.splitlines(), 1):
            if not line.strip():
                continue
            turn = _loads(line)
            extra = turn.get("extra", {})
            if not isinstance(extra, dict):
                raise ValueError(f"review metadata is not an object: {path}:{number}")
            metadata = {key: extra.get(key) for key in ("condition", "source", "claim", "item")}
            tag = turn.get("tag", "").split("/")
            if len(tag) == 4:
                for key, value in zip(metadata, tag):
                    if key in extra and extra[key] != value:
                        raise ValueError(f"tag/metadata mismatch in {path}:{number}")
                    metadata[key] = value
            if any(value is None for value in metadata.values()):
                raise ValueError(f"missing review identity metadata: {path}:{number}")
            key = _key(metadata, kind)
            legacy = legacy_by_key.get(key)
            rows.append({**turn, **metadata, "legacy": legacy,
                         "origin": {"turns_file": str(path), "line": number,
                                    "turns_sha256": files[str(path)]}})
    return rows, {"input_files": files, "selected_turn_files": len(_discover(input_root, kind))}


def _unresolved(grade: dict) -> bool:
    judgment = grade.get("judgment")
    return (not grade.get("valid") or judgment["final_truth"] == "unclear"
            or any(judgment[key] is None for key in JUDGMENT_FIELDS[1:]))


def _value(row: dict | None, metric: str):
    if row is None or row.get("semantic_grade", {}).get("valid") is not True:
        return None
    judgment = row["semantic_grade"]["judgment"]
    if metric == "semantic_final_correct":
        return judgment[metric]
    if metric == "semantic_ambiguous":
        return judgment[metric]
    return judgment[metric]


def _rate(rows: list[dict], metric: str, total: int) -> dict:
    values = [_value(row, metric) for row in rows]
    known = [float(value) for value in values if value is not None]
    positives = sum(known)
    unknown = total - len(known)
    return {"n_total": total, "n_known": len(known), "n_unknown": unknown,
            "coverage": len(known) / total if total else None,
            "mean_among_known": positives / len(known) if known else None,
            "bounds": {"lower": positives / total if total else None,
                       "upper": (positives + unknown) / total if total else None}}


def _conditions(rows: list[dict], expected: list[tuple]) -> dict:
    expected_counts = Counter(key[0] for key in expected)
    result = {}
    for condition, total in expected_counts.items():
        selected = [row for row in rows if row["condition"] == condition]
        n_valid = sum(row["semantic_grade"]["valid"] for row in selected)
        legacy = {}
        for metric in _LEGACY_METRICS:
            values = [(row.get("legacy") or {}).get(metric) for row in selected]
            values = [float(value) for value in values if isinstance(value, (int, float))
                      and math.isfinite(value)]
            legacy[metric] = {"n_known": len(values),
                              "mean_among_known": sum(values) / len(values) if values else None}
        result[condition] = {
            "n_reviews": len(selected), "expected_reviews": total,
            "missing_reviews": total - len(selected), "n_valid": n_valid,
            "valid_coverage": n_valid / total,
            "final_truth_counts": dict(Counter(
                row["semantic_grade"]["judgment"]["final_truth"]
                if row["semantic_grade"]["valid"] else "invalid" for row in selected)),
            "legacy": legacy,
            **{metric: _rate(selected, metric, total) for metric in (
                "semantic_final_correct", "usable", "semantic_ambiguous",
                "review_rejects_false", "intent_accusation", "refusal")},
        }
    return result


def _primary(rows: list[dict], facts: dict) -> dict:
    from jpeers.followup.common import paired_stats

    condition, control = "jspace_generated", "matched_generated"
    a = {(row["source"], row["item"]): row for row in rows
         if row["condition"] == condition and row["claim"] == "false"}
    b = {(row["source"], row["item"]): row for row in rows
         if row["condition"] == control and row["claim"] == "false"}
    units = list(itertools.product(ORIGINAL_SOURCES, facts))
    n_pairs = sum(unit in a and unit in b for unit in units)
    result = {"condition": condition, "control": control, "expected_pairs": 180,
              "n_pairs": n_pairs, "missing_pairs": 180 - n_pairs,
              "independent_unit": "fact; average its three source differences"}
    for metric in ("semantic_final_correct", "usable"):
        complete, lower, upper = [], [], []
        by_fact = defaultdict(list)
        for source, item in units:
            av, bv = _value(a.get((source, item)), metric), _value(b.get((source, item)), metric)
            lower.append((float(av) if av is not None else 0.) -
                         (float(bv) if bv is not None else 1.))
            upper.append((float(av) if av is not None else 1.) -
                         (float(bv) if bv is not None else 0.))
            if av is not None and bv is not None:
                difference = float(av) - float(bv)
                complete.append(difference)
                by_fact[item].append(difference)
        clusters = [sum(by_fact[item]) / 3 if len(by_fact[item]) == 3 else None for item in facts]
        result[metric] = {"n_complete_pairs": len(complete),
                          "difference": sum(complete) / len(complete) if complete else None,
                          "bounds": {"lower": sum(lower) / 180, "upper": sum(upper) / 180},
                          "fact_cluster_stats": paired_stats(clusters)}
    result["n_matched"] = n_pairs
    result["n_facts"] = result["semantic_final_correct"]["fact_cluster_stats"]["n"]
    result["paired_stats"] = result["semantic_final_correct"]["fact_cluster_stats"]
    return result


def _source_provenance() -> dict:
    paths = [Path(__file__), Path(__file__).with_name("scoring.py"), FACTS_FILE, CALIBRATION_FILE]
    return {str(path): _file_hash(path) for path in paths}


def _grade_rows(rows: list[dict], output: Path, endpoint, info: dict, calibration: dict,
                kind: str, inventory: dict, *, cache_dir=None) -> dict:
    _require_calibration(calibration, info)
    facts = _facts()
    expected, missing = _validate_rows(rows, kind, facts)
    cache = Path(cache_dir).resolve() if cache_dir is not None else output / "cache"
    inventory = {**inventory, "kind": kind, "expected_reviews": len(expected),
                 "n_reviews": len(rows), "missing_keys": [list(key) for key in missing],
                 "missing_reviews": len(missing)}
    _atomic_json(output / "inventory.json", inventory)
    scored = []
    cache_hits = 0
    for row in rows:
        reference = _reference(row, facts)
        grade, key, hit = _cache_grade(endpoint, reference, row["text"], info, cache)
        cache_hits += hit
        scored.append({**row, "reference": reference, "cache_key": key,
                       "semantic_grade": grade,
                       "semantic_grade_status": "invalid" if not grade["valid"] else
                           "unresolved" if _unresolved(grade) else "valid"})
        _atomic_json(output / "progress.json", {
            "kind": kind, "status": "running", "expected": len(expected),
            "inventory_reviews": len(rows), "processed": len(scored),
            "cache_hits": cache_hits, "grader_calls": len(scored) - cache_hits,
        })
    unresolved = [row for row in scored if _unresolved(row["semantic_grade"])]
    changed = [path for path, digest in inventory.get("input_files", {}).items()
               if not Path(path).exists() or _file_hash(Path(path)) != digest]
    complete = not missing and not unresolved and not changed
    artifacts = {"rows": str(output / "scored_rows.jsonl"),
                 "unresolved": str(output / "unresolved.jsonl"),
                 "inventory": str(output / "inventory.json"),
                 "summary": str(output / "summary.json")}
    report = {
        "protocol": "audited-followup-v2", "kind": kind,
        "status": "complete" if complete else "partial", "authoritative": complete,
        "expected_reviews": len(expected), "n_reviews": len(scored),
        "missing_reviews": len(missing), "n_valid": sum(row["semantic_grade"]["valid"] for row in scored),
        "n_unresolved": len(unresolved), "cache_hits": cache_hits,
        "grader_calls": len(scored) - cache_hits, "input_changed_files": changed,
        "scorer_version": SCORER_VERSION, "rubric_sha256": RUBRIC_HASH,
        "grader_info": info, "calibration": calibration,
        "conditions": _conditions(scored, expected),
        "primary": _primary(scored, facts) if kind == "exp4" else None,
        "provenance": {"source_files": _source_provenance(),
                       "input_files": inventory.get("input_files", {}),
                       "input_rows_sha256": inventory.get("input_rows_sha256"),
                       "original_outputs_regenerated": False},
        "rows_artifact": artifacts["rows"], "artifacts": artifacts,
    }
    _atomic_jsonl(output / "scored_rows.jsonl", scored)
    _atomic_jsonl(output / "unresolved.jsonl", unresolved)
    _atomic_json(output / "summary.json", report)
    _atomic_json(output / "progress.json", {
        "kind": kind, "status": report["status"], "expected": len(expected),
        "inventory_reviews": len(rows), "processed": len(scored),
        "cache_hits": cache_hits, "grader_calls": len(scored) - cache_hits,
    })
    return report


def run_rescore(input_root, output_dir, endpoint, grader_info, calibration, *, kind="exp4",
                cache_dir=None) -> dict:
    """Rescore original Exp2/Exp4 result directories, with a fixed complete inventory.

    Missing reviews produce an explicitly partial report. Duplicated or foreign
    cells and inconsistent tags raise before grading; no attempt is silently
    selected when multiple original attempts overlap.
    """
    if kind not in ("exp4", "exp2"):
        raise ValueError("run_rescore supports original exp4 or exp2; use grade_rows for exp2_v2")
    root, output = Path(input_root).resolve(), Path(output_dir).resolve()
    if output == root or root in output.parents:
        raise ValueError("rescoring output must be outside the original input directory")
    info = _grader_info(grader_info)
    _require_calibration(calibration, info)
    rows, inventory = _inventory(root, kind)
    return _grade_rows(rows, output, endpoint, info, calibration, kind, inventory, cache_dir=cache_dir)


def grade_rows(rows, output_dir, endpoint, grader_info, calibration, *, kind="exp2_v2",
               cache_dir=None) -> dict:
    """Grade already generated v2 review rows, preserving factorial metadata."""
    info = _grader_info(grader_info)
    _require_calibration(calibration, info)
    # Freeze the caller's mutable data before a long-running grade loop.
    frozen = _loads(_serialized(list(rows)))
    return _grade_rows(frozen, Path(output_dir).resolve(), endpoint, info, calibration, kind,
                       {"input_files": {}, "input_rows_sha256": _digest(frozen)}, cache_dir=cache_dir)
