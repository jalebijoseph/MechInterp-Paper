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


GENERATION = {"reasoning_effort": "high", "max_completion_tokens": 6144,
              "thinking_token_budget": 5000, "temperature": 0, "seed": 0, "stream": False}
NEMOTRON_GENERATION = {
    "chat_template_kwargs": {"enable_thinking": True, "medium_effort": True},
    "max_completion_tokens": 12288, "temperature": 1.0, "top_p": 0.95,
    "seed": 0, "stream": False,
}
HOLDOUT_FIXTURE_VERSION = "semantic-fact-v3.0.0"
FALLBACK_POLICY = {
    "eligibility": "http_5xx_without_final_or_length_without_parseable_judgment",
    "max_attempts": 1,
    "generation": {**{k: v for k, v in GENERATION.items() if k != "thinking_token_budget"},
                   "reasoning_effort": "medium"},
}


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input-root", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--base-url", required=True, help="vLLM server URL, with optional /v1")
    ap.add_argument("--model", required=True, help="Model ID advertised by /v1/models")
    ap.add_argument("--model-revision", required=True, help="Pinned Hugging Face commit SHA")
    ap.add_argument("--generation-profile", choices=("nemotron-3.5-super-vl",),
                    help="Use Nemotron native medium reasoning instead of the GPT-OSS defaults")
    ap.add_argument("--fixture", type=Path, help="Frozen fresh holdout JSON")
    ap.add_argument("--fixture-sha256", help="Expected holdout SHA-256, required for full grading")
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--development-only", action="store_true", help="Grade only the old 40 cases")
    ap.add_argument("--resume", action="store_true", help="Reuse this runner's matching output root")
    ap.add_argument("--fallback-from", type=Path, help="Read-only grading directory from the primary attempt")
    ap.add_argument("--fallback-source-sha256", help="SHA-256 pin of the prior grading run.json")
    return ap


def endpoint_url(url: str) -> str:
    url = url.rstrip("/")
    parsed = urlsplit(url)
    if (parsed.scheme not in ("http", "https") or not parsed.hostname or parsed.username
            or parsed.password or parsed.query or parsed.fragment):
        raise ValueError("Base URL must be an HTTP(S) API URL without credentials, query or fragment")
    return url if url.endswith("/v1") else url + "/v1"


def request_payload(model: str, reference: dict, response: str, generation: dict = GENERATION) -> dict:
    from jpeers.followup import scoring_v3

    return {"model": model, "messages": scoring_v3.build_messages(reference, response),
            **generation, "response_format": {"type": "json_schema", "json_schema": {
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
            result["http_status"] = exc.response.status_code
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


def prior_receipt(path, key, response):
    saved = rescore._read_json(path)
    receipt = saved.get("receipt")
    if (saved.get("cache_key") != key or not isinstance(receipt, dict)
            or saved.get("receipt_sha256") != rescore._digest(receipt)
            or saved.get("grade") != parse_completion(receipt, response)):
        raise ValueError(f"Original cache receipt failed integrity check: {path}")
    return receipt


def fallback_eligible(receipt, response):
    """Select on transport/truncation only, independently of expected labels."""
    if not receipt["received"] and receipt["raw_response"] is None:
        status = receipt.get("http_status")
        return (type(status) is int and 500 <= status < 600) or bool(re.match(
            r"HTTPStatusError: Server error '5\d\d(?: [^']*)?'(?: |$)",
            receipt["transport_error"] or ""))
    if (receipt["finish_reason"] != "length"
            or receipt["protocol_error"] != "ValueError: Incomplete final output: finish_reason='length'"):
        return False
    from jpeers.followup import scoring_v3

    try:
        scoring_v3.parse_judgment(receipt["raw_response"], response)
    except (ValueError, TypeError):
        return True
    # A complete parseable judgment at the length boundary gets no alternate.
    return False


def fallback_grade(client, url, cache, key, payload, response, reuse, phase):
    path = cache / (key + ".json")
    original_key = rescore._digest({"phase": phase, "grader": reuse["identity"], "request": payload})
    primary_path = reuse["output"] / "cache" / (original_key + ".json")
    calls = 0
    if path.exists():
        saved = rescore._read_json(path)
        if (saved.get("cache_key") != key or saved.get("original_key") != original_key
                or saved.get("receipts_sha256") != rescore._digest(saved.get("receipts"))):
            raise ValueError(f"Fallback cache receipt failed integrity check: {path}")
        receipts = saved["receipts"]
        from_source = saved["primary_from_source"]
    else:
        from_source = primary_path.is_file()
        if from_source:
            primary = prior_receipt(primary_path, original_key, response)
        else:
            primary = completion(client, url, payload)
            calls += 1
        receipts = {"primary": primary, "fallback": None}

    def save():
        rescore._atomic_json(path, {"cache_key": key, "original_key": original_key,
                                   "primary_from_source": from_source, "receipts": receipts,
                                   "receipts_sha256": rescore._digest(receipts)})

    eligible = fallback_eligible(receipts["primary"], response)
    if receipts["fallback"] is not None and not eligible:
        raise ValueError(f"Ineligible fallback receipt in cache: {path}")
    if eligible and receipts["fallback"] is None:
        save()  # A restart must recover the exact primary even if retry was interrupted.
        alternate = {k: v for k, v in payload.items() if k != "thinking_token_budget"}
        alternate["reasoning_effort"] = "medium"
        receipts["fallback"] = completion(client, url, alternate)
        calls += 1
        save()
    elif not path.exists():
        save()
    chosen = receipts["fallback"] if receipts["fallback"] is not None else receipts["primary"]
    return {**parse_completion(chosen, response), "fallback_used": receipts["fallback"] is not None,
            "primary_from_source": from_source, "primary_cache_key": original_key}, calls


def validate_reuse(source, pin, output, frozen, primary_identity, cases):
    source = source.resolve()
    if source == output or source in output.parents or output in source.parents:
        raise ValueError("Fallback outputs and the prior run must use separate roots")
    prior_path = source / "run.json"
    if rescore._file_hash(prior_path) != pin:
        raise ValueError("Prior run provenance SHA-256 differs from the supplied pin")
    prior = rescore._read_json(prior_path)
    expected = {**frozen, "grader_info": primary_identity}
    # Only the runner orchestration changes. The pinned prior run.json still
    # records the original runner hash and all unchanged scorer/input hashes.
    old_runner_hash = prior.get("source_files", {}).get("run_exp4_rescore_v3.py", "")
    if not re.fullmatch(r"[a-f0-9]{64}", old_runner_hash):
        raise ValueError("Prior provenance does not record the original runner hash")
    expected["source_files"] = {**frozen["source_files"], "run_exp4_rescore_v3.py": old_runner_hash}
    if prior != expected or rescore._file_hash(prior_path) != pin:
        raise ValueError("Prior provenance differs from the exact primary request configuration")
    prior_report = rescore._read_json(source / "holdout" / "calibration.json")
    if (prior_report.get("fixture_sha256") != frozen["holdout_fixture_sha256"]
            or prior_report.get("grader_info") != primary_identity
            or prior_report.get("n_cases") != len(cases)):
        raise ValueError("Prior holdout report does not match the pinned run")
    n_valid, n_agree = prior_report.get("n_valid"), prior_report.get("n_agree")
    if type(n_valid) is not int or type(n_agree) is not int or not 0 <= n_agree <= n_valid <= len(cases):
        raise ValueError("Prior holdout has invalid aggregate judgment counts")
    # Keep the frozen 95% threshold. Only technically invalid receipts can be
    # replaced, so existing semantic disagreements still consume this budget.
    required_agreement = (95 * len(cases) + 99) // 100
    if (n_valid - n_agree > len(cases) - required_agreement
            or prior_report.get("critical_false_positive_ids") != []):
        raise ValueError("Prior holdout has a semantic gate failure. Use a new frozen holdout.")
    n_valid = 0
    for case in cases:
        response = case["response"]
        payload = request_payload(primary_identity["model"], case["item"], response)
        key = rescore._digest({"phase": "holdout", "grader": primary_identity, "request": payload})
        receipt = prior_receipt(source / "cache" / (key + ".json"), key, response)
        valid = parse_completion(receipt, response)["valid"]
        n_valid += valid
        if not valid and not fallback_eligible(receipt, response):
            raise ValueError("Prior holdout has a non-transport invalid judgment. Use a new frozen holdout.")
    if n_valid != prior_report["n_valid"]:
        raise ValueError("Prior holdout report does not match its original receipts")
    return {"output": source, "identity": primary_identity}


def grade_batch(items, phase, *, client, url, output, identity, concurrency, reuse=None):
    groups = defaultdict(list)
    for index, (reference, response) in enumerate(items):
        payload = request_payload(identity["model"], reference, response, identity["generation"])
        key = rescore._digest({"phase": phase, "grader": identity, "request": payload})
        groups[key].append((index, payload, response))
    results = [None] * len(items)
    progress = {"phase": phase, "status": "running", "expected": len(items), "processed": 0,
                "cache_hits": 0, "grader_calls": 0, "invalid": 0}
    progress_file = output / "progress.json"
    rescore._atomic_json(progress_file, progress)
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        pending = {}
        for key, group in groups.items():
            args = (client, url, output / "cache", key, group[0][1], group[0][2])
            future = (executor.submit(fallback_grade, *args, reuse, phase) if reuse else
                      executor.submit(cached_grade, *args))
            pending[future] = key
        try:
            for future in as_completed(pending):
                key = pending[future]
                grade, result = future.result()
                calls = result if reuse else int(not result)
                group = groups[key]
                for index, _, _ in group:
                    results[index] = {"cache_key": key, **grade}
                progress["processed"] += len(group)
                progress["cache_hits"] += len(group) - int(calls > 0)
                progress["grader_calls"] += calls
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
    if args.generation_profile == "nemotron-3.5-super-vl":
        if args.fallback_from is not None or args.fallback_source_sha256 is not None:
            raise ValueError("The existing fallback policy applies only to GPT-OSS")
        identity.update(generation_profile="nemotron-3.5-super-vl-v1", generation=NEMOTRON_GENERATION)
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
    reuse = None
    if args.fallback_from is not None or args.fallback_source_sha256 is not None:
        if (args.development_only or args.fallback_from is None
                or not re.fullmatch(r"[a-f0-9]{64}", args.fallback_source_sha256 or "")):
            raise ValueError("Fallback requires a full run, a prior output root and its pinned run.json SHA-256")
        reuse = validate_reuse(args.fallback_from, args.fallback_source_sha256, output,
                               frozen, identity, holdout[0])
        identity = {**identity, "fallback": FALLBACK_POLICY}
        frozen = {**frozen, "grader_info": identity,
                  "fallback_source": str(reuse["output"]),
                  "fallback_source_sha256": args.fallback_source_sha256}
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
        batch = dict(client=client, url=url, output=output, identity=identity,
                     concurrency=args.concurrency, reuse=reuse)
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
