"""Replay frozen calibration, then grade exactly 600 saved corrected Exp2 reviews."""
from __future__ import annotations

import argparse
import fcntl
import hashlib
from pathlib import Path
import re

import httpx

from jpeers.followup import rescore, scoring_v3
import run_exp4_rescore_v3 as shared


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input-summary", type=Path, required=True)
    ap.add_argument("--input-summary-sha256", required=True)
    ap.add_argument("--calibration-source", type=Path, required=True)
    ap.add_argument("--calibration-source-sha256", required=True, help="Prior run.json SHA-256")
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--model-revision", required=True)
    ap.add_argument("--generation-profile", choices=("nemotron-3.5-super-vl",), required=True)
    ap.add_argument("--fixture", type=Path, required=True)
    ap.add_argument("--fixture-sha256", required=True)
    ap.add_argument("--concurrency", type=int, default=16)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--prepare-only", action="store_true",
                    help="Validate inputs and replay cached gates without any HTTP calls")
    return ap


def grader_identity(args) -> dict:
    return {"model": args.model, "model_revision": args.model_revision,
            "revision_source": "operator-supplied pinned model commit",
            "scorer_version": scoring_v3.SCORER_VERSION,
            "rubric_sha256": scoring_v3.RUBRIC_HASH,
            "generation_profile": "nemotron-3.5-super-vl-v2",
            "generation": shared.NEMOTRON_GENERATION, "checklist": shared.NEMOTRON_CHECKLIST}


def source_files() -> dict:
    return {path.name: rescore._file_hash(path) for path in (
        Path(shared.__file__), Path(scoring_v3.__file__), Path(rescore.__file__),
        Path(scoring_v3.__file__).with_name("scoring.py"),
        Path(scoring_v3.__file__).with_name("common.py"), rescore.FACTS_FILE)}


def load_reviews(path: Path, pin: str) -> tuple[list[dict], list[tuple], dict]:
    if rescore._file_hash(path) != pin:
        raise ValueError("Saved Exp2 summary differs from its supplied SHA-256")
    summary = rescore._read_json(path)
    rows = summary.get("reviews")
    if (summary.get("protocol") != "audited-followup-v2" or not isinstance(rows, list)
            or len(rows) != 600
            or any(not isinstance(row, dict) or row.get("kind") != "review" for row in rows)):
        raise ValueError("Corrected Exp2 requires exactly 600 review rows and no knowledge probes")
    expected, missing = rescore._validate_rows(rows, "exp2_v2", rescore._facts())
    if missing or len(expected) != 600 or summary.get("counts", {}).get("recorded_reviews") != 600:
        raise ValueError("Corrected Exp2 summary does not cover the complete 600-cell review grid")
    if rescore._file_hash(path) != pin:
        raise ValueError("Saved Exp2 summary changed during loading")
    return rows, expected, {"kind": "exp2_v2", "input_summary": str(path),
                            "input_summary_sha256": pin, "expected_reviews": 600,
                            "n_reviews": len(rows), "missing_reviews": 0, "missing_keys": [],
                            "knowledge_probes_graded": 0}


def calibration_receipts(source: Path, pin: str, frozen: dict,
                         development: list[dict], holdout: list[dict]) -> tuple[dict, dict]:
    prior_path = source / "run.json"
    if rescore._file_hash(prior_path) != pin:
        raise ValueError("Prior calibration run.json differs from its supplied SHA-256")
    prior = rescore._read_json(prior_path)
    for field in ("grader_info", "development_fixture_sha256", "holdout_fixture_sha256",
                  "holdout_fixture_scorer_version", "development_only", "source_files"):
        if prior.get(field) != frozen[field]:
            raise ValueError(f"Prior calibration provenance differs in {field}")
    identity = frozen["grader_info"]
    seeds, hashes = {}, {}
    for phase, cases in (("calibration", development), ("holdout", holdout)):
        for case in cases:
            payload = shared.request_payload(identity["model"], case["item"], case["response"],
                                             identity["generation"], identity["checklist"])
            key = rescore._digest({"phase": phase, "grader": identity, "request": payload})
            path = source / "cache" / (key + ".json")
            raw = path.read_text(encoding="utf-8")
            # Validate the key, receipt digest and freshly parsed stored grade.
            receipt = shared.prior_receipt(path, key, case["response"])
            if receipt.get("received") is not True:
                raise ValueError("Calibration reuse requires received receipts and never retries transport failures")
            if rescore._file_hash(path) != hashlib.sha256(raw.encode("utf-8")).hexdigest():
                raise ValueError("Prior calibration receipt changed during validation")
            seeds[key] = raw
            hashes[key] = rescore._file_hash(path)
    if len(seeds) != 84 or rescore._file_hash(prior_path) != pin:
        raise ValueError("Expected all 84 distinct, unchanged frozen calibration receipts")
    return seeds, hashes


def score_reviews(rows: list[dict], expected: list[tuple], inventory: dict,
                  input_path: Path, gate: dict, batch: dict) -> int:
    if gate.get("passed") is not True:
        raise ValueError("Exp2 rescoring requires both unchanged calibration gates to pass")
    facts, output = rescore._facts(), batch["output"]
    items = [(rescore._reference(row, facts), row["text"]) for row in rows]
    grades, progress = shared.grade_batch(items, "rescore_exp2_v2", **batch)
    scored, unresolved = [], []
    for row, (reference, _), grade in zip(rows, items, grades, strict=True):
        unknown = rescore._unresolved(grade)
        result = {**row, "reference": reference, "cache_key": grade["cache_key"],
                  "semantic_grade": grade, "semantic_grade_status":
                  "invalid" if not grade["valid"] else "unresolved" if unknown else "valid"}
        scored.append(result)
        if unknown:
            unresolved.append(result)
    changed = (not input_path.is_file()
               or rescore._file_hash(input_path) != inventory["input_summary_sha256"])
    complete = not unresolved and not changed
    report = {"protocol": "audited-exp2-v2-rescore-v3", "status": "complete" if complete else "partial",
              "authoritative": complete, "kind": "exp2_v2", "calibration": gate,
              "grader_info": batch["identity"], "expected_reviews": 600, "n_reviews": len(scored),
              "n_valid": sum(row["semantic_grade"]["valid"] for row in scored),
              "n_unresolved": len(unresolved), "missing_reviews": 0,
              "input_changed_files": [str(input_path)] if changed else [],
              "cache_hits": progress["cache_hits"], "grader_calls": progress["grader_calls"],
              "conditions": rescore._conditions(scored, expected), "knowledge_probes_graded": 0,
              "behavioral_contrasts": {"status": "deferred",
                                        "reason": "Requires a separate fact-cluster analysis of the scored review grid"},
              "provenance": {"input_files": {str(input_path): inventory["input_summary_sha256"]},
                             "original_outputs_regenerated": False}}
    rescore._atomic_jsonl(output / "scored_rows.jsonl", scored)
    rescore._atomic_jsonl(output / "unresolved.jsonl", unresolved)
    rescore._atomic_json(output / "summary.json", report)
    rescore._atomic_json(output / "progress.json", {**progress, "status": report["status"],
                         "n_unresolved": len(unresolved), "authoritative": complete})
    return 0 if complete else 1


def run(args, client: httpx.Client) -> int:
    if not 1 <= args.concurrency <= 256 or not re.fullmatch(r"[a-f0-9]{40}", args.model_revision):
        raise ValueError("Require concurrency 1–256 and a pinned 40-character model revision")
    if args.generation_profile != "nemotron-3.5-super-vl":
        raise ValueError("Exp2 receipt reuse requires the unchanged Nemotron profile")
    for pin in (args.input_summary_sha256, args.calibration_source_sha256, args.fixture_sha256):
        if not re.fullmatch(r"[a-f0-9]{64}", pin or ""):
            raise ValueError("Input, calibration-source and fixture SHA-256 pins are required")
    output, input_path = args.output_dir.resolve(), args.input_summary.resolve()
    source = args.calibration_source.resolve()
    if (output in input_path.parents or input_path.parent in output.parents or output == source
            or output in source.parents or source in output.parents):
        raise ValueError("Use an output directory separate from all frozen input and calibration artifacts")
    rows, expected, inventory = load_reviews(input_path, args.input_summary_sha256)
    development, development_hash = rescore._fixtures()
    holdout, holdout_hash = shared.load_holdout(args.fixture, args.fixture_sha256,
                                              development["cases"])
    identity = grader_identity(args)
    frozen = {"grader_info": identity, "development_fixture_sha256": development_hash,
              "holdout_fixture_sha256": holdout_hash,
              "holdout_fixture_scorer_version": shared.HOLDOUT_FIXTURE_VERSION,
              "development_only": False, "source_files": source_files()}
    seeds, receipt_hashes = calibration_receipts(source, args.calibration_source_sha256,
                                                frozen, development["cases"], holdout)
    frozen.update(kind="exp2_v2", input_summary=str(input_path),
                  input_summary_sha256=args.input_summary_sha256,
                  calibration_source=str(source), calibration_source_sha256=args.calibration_source_sha256,
                  calibration_receipts_sha256=receipt_hashes)
    frozen["source_files"] = {**frozen["source_files"], Path(__file__).name: rescore._file_hash(Path(__file__))}
    output.mkdir(parents=True, exist_ok=True)
    with (output / ".run.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        provenance = output / "run.json"
        if any(path.name != ".run.lock" for path in output.iterdir()):
            if not args.resume or not provenance.is_file() or rescore._read_json(provenance) != frozen:
                raise ValueError("Use a new output directory or --resume with exactly matching provenance")
        rescore._atomic_json(provenance, frozen)
        rescore._atomic_json(output / "inventory.json", inventory)
        for key, raw in seeds.items():
            path = output / "cache" / (key + ".json")
            if path.exists() and rescore._file_hash(path) != receipt_hashes[key]:
                raise ValueError("A seeded calibration cache entry differs from the pinned source receipt")
            rescore._atomic_text(path, raw)
        url = shared.endpoint_url(args.base_url)
        batch = dict(client=client, url=url, output=output, identity=identity,
                     concurrency=args.concurrency)
        old = shared.calibrate(development["cases"], "calibration", development_hash,
                               development["scorer_version"], batch)
        fresh = shared.calibrate(holdout, "holdout", holdout_hash, shared.HOLDOUT_FIXTURE_VERSION, batch)
        if old["grader_calls"] or fresh["grader_calls"]:
            raise ValueError("Calibration replay unexpectedly attempted new model calls")
        gate = {"passed": old["passed"] and fresh["passed"], "development_only": False,
                "development": old, "holdout": fresh,
                "status": "passed" if old["passed"] and fresh["passed"] else "blocked_by_replayed_calibration"}
        rescore._atomic_json(output / "gate.json", gate)
        if not gate["passed"]:
            return 2
        if args.prepare_only:
            rescore._atomic_json(output / "progress.json", {"phase": "preflight", "status": "prepared",
                                 "expected_reviews": 600, "calibration_cache_hits": 84, "grader_calls": 0})
            return 0
        observed = shared.verify_endpoint(client, url, args.model)
        rescore._atomic_json(output / "endpoint.json", {"base_url": url, "observed_model": observed,
                                                        "grader_info": identity})
        return score_reviews(rows, expected, inventory, input_path, gate, batch)


def main() -> int:
    args = parser().parse_args()
    with httpx.Client(timeout=httpx.Timeout(1800, connect=15), trust_env=False) as client:
        return run(args, client)


if __name__ == "__main__":
    raise SystemExit(main())
