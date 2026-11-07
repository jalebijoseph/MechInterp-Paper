"""Calibrate a pinned vLLM grader, then rescore the 4,680 frozen Exp4 reviews."""
from __future__ import annotations

import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import fcntl
from pathlib import Path
import re
from urllib.parse import urlsplit

import httpx

from jpeers.followup import rescore
from jpeers.followup.scoring import CALIBRATION_GATE, calibration_report


GENERATION = {"reasoning_effort": "medium", "max_completion_tokens": 4096,
              "temperature": 0, "seed": 0, "stream": False}
HOLDOUT_FIXTURE_VERSION = "semantic-fact-v3.0.0"


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input-root", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--base-url", required=True, help="vLLM server URL, with optional /v1")
    ap.add_argument("--model", required=True, help="Model ID advertised by /v1/models")
    ap.add_argument("--model-revision", required=True, help="Pinned Hugging Face commit SHA")
    ap.add_argument("--fixture", type=Path, help="Frozen fresh holdout JSON")
    ap.add_argument("--fixture-sha256", help="Expected holdout SHA-256, required for full grading")
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--development-only", action="store_true", help="Grade only the old 40 cases")
    ap.add_argument("--resume", action="store_true", help="Reuse this runner's matching output root")
    return ap


def endpoint_url(url: str) -> str:
    url = url.rstrip("/")
    parsed = urlsplit(url)
    if (parsed.scheme not in ("http", "https") or not parsed.hostname or parsed.username
            or parsed.password or parsed.query or parsed.fragment):
        raise ValueError("Base URL must be an HTTP(S) API URL without credentials, query or fragment")
    return url if url.endswith("/v1") else url + "/v1"


def request_payload(model: str, reference: dict, response: str) -> dict:
    from jpeers.followup import scoring_v3

    return {"model": model, "messages": scoring_v3.build_messages(reference, response),
            **GENERATION, "response_format": {"type": "json_schema", "json_schema": {
                "name": "semantic_fact_v3", "strict": True,
                "schema": scoring_v3.schema_for_response(response)}}}


def verify_endpoint(client: httpx.Client, url: str, model: str) -> dict:
    response = client.get(url + "/models")
    response.raise_for_status()
    matches = [row for row in response.json()["data"] if row.get("id") == model]
    if len(matches) != 1:
        raise ValueError(f"The vLLM endpoint does not advertise exactly one model named {model}")
    return {key: matches[0].get(key) for key in ("id", "root", "parent")}


def completion(client: httpx.Client, url: str, payload: dict) -> dict:
    """Keep the final channel and token counts. The reasoning text is not stored."""
    result = {"received": False, "raw_response": None, "finish_reason": None,
              "response_id": None, "reasoning_tokens": None, "completion_tokens": None,
              "prompt_tokens": None, "protocol_error": None, "transport_error": None}
    try:
        response = client.post(url + "/chat/completions", json=payload)
        response.raise_for_status()
    except httpx.HTTPError as exc:
        result["transport_error"] = f"{type(exc).__name__}: {exc}"
        if isinstance(exc, httpx.HTTPStatusError):
            result["http_error_body"] = exc.response.text[:2000]
        return result
    result["received"] = True
    try:
        data = response.json()
        choice, = data["choices"]
        message = choice["message"]
        raw = message.get("content")
        result["raw_response"] = raw if isinstance(raw, str) else None
        result["finish_reason"] = choice.get("finish_reason")
        result["response_id"] = data.get("id")
        usage = data.get("usage") or {}
        counts = {key: usage.get(key) for key in ("completion_tokens", "prompt_tokens")}
        counts["reasoning_tokens"] = (usage.get("completion_tokens_details") or {}).get("reasoning_tokens")
        result.update({key: value for key, value in counts.items() if type(value) is int and value >= 0})
        if data.get("model") != payload["model"]:
            raise ValueError(f"Returned model ID differs from requested {payload['model']}")
        if result["finish_reason"] != "stop":
            raise ValueError(f"Incomplete final output: finish_reason={result['finish_reason']!r}")
        if not isinstance(raw, str) or not raw.strip():
            raise ValueError("No final message content returned")
    except (AttributeError, KeyError, TypeError, ValueError) as exc:
        result["protocol_error"] = f"{type(exc).__name__}: {exc}"
    return result


def parse_completion(receipt: dict, response: str) -> dict:
    from jpeers.followup import scoring_v3

    error = receipt["protocol_error"] or receipt["transport_error"]
    judgment = None
    if error is None:
        try:
            judgment = scoring_v3.parse_judgment(receipt["raw_response"], response)
        except (ValueError, TypeError) as exc:
            error = f"{type(exc).__name__}: {exc}"
    return {**receipt, "scorer_version": scoring_v3.SCORER_VERSION, "valid": error is None,
            "judgment": judgment, "error": error}


def cached_grade(client, url, cache, key, payload, response):
    path = cache / (key + ".json")
    attempts = []
    if path.is_file():
        saved = rescore._read_json(path)
        receipt = saved.get("receipt")
        if (saved.get("cache_key") != key or not isinstance(receipt, dict)
                or saved.get("receipt_sha256") != rescore._digest(receipt)):
            raise ValueError(f"Cached grader response failed integrity check: {path}")
        attempts = saved.get("transport_attempts", [])
        if receipt["received"]:
            grade = parse_completion(receipt, response)
            if grade != saved.get("grade"):
                raise ValueError(f"Cached grade differs from its raw final output: {path}")
            return grade, True
    receipt = completion(client, url, payload)
    if receipt["transport_error"] is not None:
        attempts = [*attempts, receipt]
    grade = parse_completion(receipt, response)
    rescore._atomic_json(path, {"cache_key": key, "receipt": receipt,
                               "receipt_sha256": rescore._digest(receipt),
                               "transport_attempts": attempts, "grade": grade})
    return grade, False


def grade_batch(items, phase, *, client, url, output, identity, concurrency):
    groups = defaultdict(list)
    for index, (reference, response) in enumerate(items):
        payload = request_payload(identity["model"], reference, response)
        key = rescore._digest({"phase": phase, "grader": identity, "request": payload})
        groups[key].append((index, payload, response))
    results = [None] * len(items)
    progress = {"phase": phase, "status": "running", "expected": len(items), "processed": 0,
                "cache_hits": 0, "grader_calls": 0, "invalid": 0}
    progress_file = output / "progress.json"
    rescore._atomic_json(progress_file, progress)
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        pending = {executor.submit(cached_grade, client, url, output / "cache", key,
                                   group[0][1], group[0][2]): key for key, group in groups.items()}
        try:
            for future in as_completed(pending):
                key = pending[future]
                grade, hit = future.result()
                group = groups[key]
                for index, _, _ in group:
                    results[index] = {"cache_key": key, **grade}
                progress["processed"] += len(group)
                progress["cache_hits"] += len(group) - int(not hit)
                progress["grader_calls"] += int(not hit)
                progress["invalid"] += len(group) * int(not grade["valid"])
                rescore._atomic_json(progress_file, progress)
        except BaseException:
            for future in pending:
                future.cancel()
            raise
    return results, progress


def calibrate(cases, phase, fixture_hash, fixture_version, batch):
    grades, progress = grade_batch([(case["item"], case["response"]) for case in cases], phase, **batch)
    predictions = [{"id": case["id"], **grade} for case, grade in zip(cases, grades, strict=True)]
    output, identity = batch["output"], batch["identity"]
    report = {**calibration_report(predictions, cases), "scorer_version": identity["scorer_version"],
              "rubric_sha256": identity["rubric_sha256"], "fixture_sha256": fixture_hash,
              "fixture_scorer_version": fixture_version,
              "grader_info": identity, "cache_hits": progress["cache_hits"],
              "grader_calls": progress["grader_calls"]}
    rescore._atomic_jsonl(output / phase / "calibration_predictions.jsonl", predictions)
    rescore._atomic_json(output / phase / "calibration.json", report)
    rescore._atomic_json(output / "progress.json", {**progress,
                         "status": "complete" if report["passed"] else "failed"})
    return report


def load_holdout(path: Path, expected_hash: str, development: list[dict]) -> tuple[list[dict], str]:
    from jpeers.followup import scoring_v3

    observed_hash = rescore._file_hash(path)
    if observed_hash != expected_hash:
        raise ValueError("Fresh holdout SHA-256 does not match the pinned fixture")
    fixture = rescore._read_json(path)
    # The frozen labels and thresholds belong to the v3 rubric family. Prompt
    # and output-schema revisions do not require relabeling or re-freezing them.
    if (not isinstance(fixture, dict) or fixture.get("scorer_version") != HOLDOUT_FIXTURE_VERSION
            or not re.fullmatch(r"semantic-fact-v3\.\d+\.\d+", scoring_v3.SCORER_VERSION)):
        raise ValueError("Frozen v3.0.0 holdout version requires a compatible v3.x scorer")
    if fixture.get("gate") != CALIBRATION_GATE:
        raise ValueError("Fresh holdout must use the unchanged calibration thresholds")
    cases = fixture.get("cases")
    if not isinstance(cases, list) or len(cases) < 40 or any(not isinstance(case, dict) for case in cases):
        raise ValueError("Fresh holdout must contain at least 40 labeled cases")
    if (any(type(case.get("critical")) is not bool for case in cases)
            or not any(case["critical"] for case in cases)):
        raise ValueError("Fresh holdout must identify a nonempty set of critical cases")
    calibration_report([], cases)
    development_keys = {rescore._digest(scoring_v3.build_messages(case["item"], case["response"]))
                        for case in development}
    fresh_keys = [rescore._digest(scoring_v3.build_messages(case["item"], case["response"]))
                  for case in cases]
    if development_keys.intersection(fresh_keys) or len(set(fresh_keys)) != len(fresh_keys):
        raise ValueError("Fresh holdout contains a reused development case or a duplicate grader input")
    if rescore._file_hash(path) != observed_hash:
        raise ValueError("Fresh holdout changed during loading")
    return cases, observed_hash


def score_exp4(input_root, gate, batch):
    if gate.get("passed") is not True:
        raise ValueError("Exp4 scoring requires passing both frozen calibration gates")
    output = batch["output"]
    rows, inventory = rescore._inventory(input_root, "exp4")
    facts = rescore._facts()
    expected, missing = rescore._validate_rows(rows, "exp4", facts)
    inventory = {**inventory, "expected_reviews": len(expected), "n_reviews": len(rows),
                 "missing_keys": [list(key) for key in missing], "missing_reviews": len(missing)}
    path = output / "inventory.json"
    if path.exists():
        previous = rescore._read_json(path)
        if previous["missing_reviews"] == 0 and previous != inventory:
            raise ValueError("Frozen Exp4 inventory differs from the one already graded in this output root")
    rescore._atomic_json(path, inventory)
    if missing or len(expected) != 4680:
        rescore._atomic_json(output / "progress.json", {"phase": "rescore", "status": "blocked_by_inventory",
                             "expected": 4680, "inventory_reviews": len(rows), "processed": 0})
        raise ValueError(f"Exp4 requires all 4,680 unique frozen reviews; found {len(rows)}")
    items = [(rescore._reference(row, facts), row["text"]) for row in rows]
    grades, progress = grade_batch(items, "rescore", **batch)
    scored, unresolved = [], []
    for row, (reference, _), grade in zip(rows, items, grades, strict=True):
        unknown = rescore._unresolved(grade)
        result = {**row, "reference": reference, "cache_key": grade["cache_key"],
                  "semantic_grade": grade, "semantic_grade_status":
                  "invalid" if not grade["valid"] else "unresolved" if unknown else "valid"}
        scored.append(result)
        if unknown:
            unresolved.append(result)
    changed = [name for name, digest in inventory["input_files"].items()
               if not Path(name).is_file() or rescore._file_hash(Path(name)) != digest]
    primary = None if changed else rescore._primary(scored, facts)
    if primary is not None and primary["n_pairs"] != 180:
        raise ValueError("Expected 180 matched false-claim intervention/control pairs")
    complete = not unresolved and not changed
    report = {"protocol": "audited-exp4-rescore-v3", "status": "complete" if complete else "partial",
              "authoritative": complete, "kind": "exp4", "calibration": gate,
              "grader_info": batch["identity"], "expected_reviews": 4680,
              "n_reviews": len(scored), "n_valid": sum(row["semantic_grade"]["valid"] for row in scored),
              "n_unresolved": len(unresolved), "input_changed_files": changed,
              "cache_hits": progress["cache_hits"], "grader_calls": progress["grader_calls"],
              "conditions": rescore._conditions(scored, expected), "primary": primary,
              "provenance": {"input_files": inventory["input_files"],
                             "original_outputs_regenerated": False}}
    rescore._atomic_jsonl(output / "scored_rows.jsonl", scored)
    rescore._atomic_jsonl(output / "unresolved.jsonl", unresolved)
    rescore._atomic_json(output / "summary.json", report)
    rescore._atomic_json(output / "progress.json", {**progress, "status": report["status"],
                         "n_unresolved": len(unresolved), "authoritative": complete})
    return 0 if complete else 1


def run(args, client: httpx.Client) -> int:
    from jpeers.followup import scoring_v3

    if not 1 <= args.concurrency <= 256:
        raise ValueError("Concurrency must be between 1 and 256")
    if not re.fullmatch(r"[a-f0-9]{40}", args.model_revision):
        raise ValueError("Model revision must be a pinned 40-character lowercase commit SHA")
    if not args.development_only and (args.fixture is None or
                                      not re.fullmatch(r"[a-f0-9]{64}", args.fixture_sha256 or "")):
        raise ValueError("Full grading requires --fixture and its pinned --fixture-sha256")
    output, input_root = args.output_dir.resolve(), args.input_root.resolve()
    if output == input_root or input_root in output.parents:
        raise ValueError("Output directory must be separate from the frozen original inputs")
    url = endpoint_url(args.base_url)
    development, development_hash = rescore._fixtures()
    holdout = None if args.development_only else load_holdout(args.fixture, args.fixture_sha256,
                                                              development["cases"])
    identity = {"model": args.model, "model_revision": args.model_revision,
                "revision_source": "operator-supplied pinned model commit",
                "scorer_version": scoring_v3.SCORER_VERSION, "rubric_sha256": scoring_v3.RUBRIC_HASH,
                "generation": GENERATION}
    frozen = {"grader_info": identity, "input_root": str(input_root),
              "development_fixture_sha256": development_hash,
              "holdout_fixture_sha256": holdout[1] if holdout else None,
              "holdout_fixture_scorer_version": HOLDOUT_FIXTURE_VERSION if holdout else None,
              "development_only": args.development_only,
              "source_files": {path.name: rescore._file_hash(path) for path in
                               [Path(__file__), Path(scoring_v3.__file__), Path(rescore.__file__),
                                Path(scoring_v3.__file__).with_name("scoring.py"),
                                Path(scoring_v3.__file__).with_name("common.py"),
                                rescore.FACTS_FILE]}}
    output.mkdir(parents=True, exist_ok=True)
    with (output / ".run.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        provenance = output / "run.json"
        if any(path != provenance and path.name != ".run.lock" for path in output.iterdir()) or provenance.exists():
            if not args.resume or not provenance.is_file() or rescore._read_json(provenance) != frozen:
                raise ValueError("Use a new output directory or --resume with exactly matching provenance")
        observed = verify_endpoint(client, url, args.model)
        rescore._atomic_json(provenance, frozen)
        rescore._atomic_json(output / "endpoint.json", {"base_url": url, "observed_model": observed,
                                                        "grader_info": identity})
        batch = dict(client=client, url=url, output=output, identity=identity, concurrency=args.concurrency)
        old = calibrate(development["cases"], "calibration", development_hash,
                        development["scorer_version"], batch)
        gate = {"passed": False, "development_only": args.development_only,
                "development": old, "holdout": None}
        if not old["passed"] or args.development_only:
            gate["status"] = "development_only_complete" if old["passed"] else "blocked_by_development"
            rescore._atomic_json(output / "gate.json", gate)
            return 0 if old["passed"] else 2
        fresh = calibrate(holdout[0], "holdout", holdout[1], HOLDOUT_FIXTURE_VERSION, batch)
        gate.update(passed=fresh["passed"], holdout=fresh,
                    status="passed" if fresh["passed"] else "blocked_by_holdout")
        rescore._atomic_json(output / "gate.json", gate)
        if not gate["passed"]:
            return 2
        return score_exp4(input_root, gate, batch)


def main() -> int:
    args = parser().parse_args()
    with httpx.Client(timeout=httpx.Timeout(600, connect=15), trust_env=False) as client:
        return run(args, client)


if __name__ == "__main__":
    raise SystemExit(main())
