#!/usr/bin/env python3
"""Durable full-size Qwen3-8B suite; inference servers are managed externally.

Example: python scripts/run_cluster_suite.py --ports 18080 --output-dir runs/full-suite
Add ports to run independent jobs concurrently, one request stream per server.
Restart the same command to resume pending/interrupted jobs; --retry-failed retries failures.
Each attempt keeps its own config, endpoint, log, benchmark audit, results, and exit status.
Hi-ToM and NARCBench are separate portions, so unavailable external data cannot block
the core tasks. No portion is complete without validated counts and a saved summary.
"""

from __future__ import annotations

import argparse
from collections import Counter, deque
import copy
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import importlib
import importlib.metadata
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback
import urllib.request

SOURCE = Path(__file__).resolve().parents[1]
CONFIGS = SOURCE / "configs" / "cluster-qwen3-8b"
SOURCE_SHA = "1f97522907c33cdb557163769b599088f8e7936b"
CONFIG_FILES = ["exp0_tom_validation.json", "exp1_collab.json", "exp2_imposter.json",
                "exp3_peer_selection.json", "exp4_ablation.json"]


def now():
    return datetime.now(timezone.utc).isoformat()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def atomic_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    with temp.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, allow_nan=False)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(temp, path)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()


def versions():
    packages = ["jpeers", "vllm", "torch", "transformers", "jlens", "numpy", "pydantic",
                "huggingface_hub", "pyyaml", "httpx", "fastapi", "uvicorn"]
    result = {"python": sys.version, "executable": sys.executable}
    for name in packages:
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = None
    return result


def source_hashes():
    paths = [SOURCE / "pyproject.toml"]
    for folder in ("src", "scripts", "configs/cluster-qwen3-8b"):
        paths.extend(p for p in (SOURCE / folder).rglob("*")
                     if p.is_file() and "__pycache__" not in p.parts)
    return {str(p.relative_to(SOURCE)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(paths)}


def build_jobs():
    configs = [read_json(CONFIGS / name) for name in CONFIG_FILES]
    jobs = []
    for exp, base in enumerate(configs):
        parts = ["core", "hitom"] if exp == 0 else ["core", "narcbench"] if exp == 2 else ["full"]
        if exp == 4:
            parts = base["interventions"]
        for part in parts:
            cfg = copy.deepcopy(base)
            if exp == 0:
                cfg["include_hitom"] = part == "hitom"
                if part == "hitom":
                    cfg["n_items"] = 0
            if exp == 2:
                if part == "core":
                    cfg.pop("narcbench")
                else:
                    cfg["live"] = False
            if exp == 4:
                cfg["interventions"] = [part]
                cfg["cluster_fragment"] = {"condition": part,
                    "primary_status": "deferred until all 13 conditions are aggregated"}
            jobs.append({"id": f"exp{exp}-{part}", "experiment": exp, "portion": part,
                         "source_config": str(CONFIGS / CONFIG_FILES[exp]),
                         "config": cfg, "config_sha256": digest(cfg), "status": "pending",
                         "attempts": []})
    return jobs


def bank(name, key):
    return read_json(SOURCE / "src/jpeers/data" / f"{name}.json")[key]


def planned_counts(cfg):
    """Expected successful responses; NARC replay count is resolved by data preflight."""
    exp = cfg["experiment"]
    if exp == "tom_validation":
        core = cfg["n_items"] * 2 * len(cfg["framings"])
        hitom = cfg["hitom_items"] if cfg["include_hitom"] else 0
        return {"core_items": core, "hitom_items": hitom, "requests": core + hitom}
    if exp == "collab":
        if len(bank("collab_tasks", "tasks")) < cfg["n_tasks"]:
            raise ValueError("Collaboration bank is smaller than the requested task count")
        return {"turns": cfg["n_tasks"] * len(cfg["conditions"]) * 2 * cfg["rounds"]}
    if exp in ("imposter", "ablation"):
        if len(bank("imposter_facts", "items")) < cfg["n_items"]:
            raise ValueError("Fact bank is smaller than the requested item count")
        reviews = (cfg["n_items"] * len(cfg.get("sources", ["incentive", "no_incentive", "document"]))
                   * 2 * len(cfg.get("interventions", ["none"]))) if cfg.get("live", True) else 0
        return {"reviews": reviews, "narcbench_scenarios": cfg.get("narcbench", {}).get("max_scenarios", 0)}
    if exp == "peer_selection":
        tasks = len(bank("peer_selection", "tasks"))
        counts = {"context": tasks * 4,
                  "family": tasks * len(cfg["other_families"]) * len(cfg["size_conditions"]) * 4,
                  "self_preference": tasks * 2, "bandit": 2 * cfg["n_seeds"] * cfg["rounds"]}
        counts = {k: counts[k] for k in cfg["variants"]}
        return {"decisions": counts, "plans": tasks * 2 if "self_preference" in counts else 0,
                "self_preference_scope": cfg["self_preference_scope"]}
    raise ValueError(f"Unsupported experiment {exp}")


def benchmark_preflight(cfg, audit):
    from jpeers.benchmarks import load_hitom, list_narcbench, load_narcbench

    if cfg.get("include_hitom"):
        items = load_hitom(orders=tuple(cfg["hitom_orders"]), max_items=cfg["hitom_items"])
        audit["hitom"] = {"configured": cfg["hitom_items"], "actual": len(items),
                          "ids": [it.id for it in items], "orders": dict(Counter(it.order for it in items)),
                          "content_sha256": digest([asdict(it) for it in items])}
        if len(items) != cfg["hitom_items"]:
            raise ValueError(f"Requested {cfg['hitom_items']} Hi-ToM items, found {len(items)}")
    if cfg.get("narcbench"):
        nb = cfg["narcbench"]
        selected = list_narcbench(nb["source_model"], nb["tier"])[:nb["max_scenarios"]]
        items = load_narcbench(nb["source_model"], nb["tier"], nb["max_scenarios"])
        actual_ids = [it.run_id for it in items]
        audit["narcbench"] = {"configured": nb["max_scenarios"], "selected_ids": selected,
                             "actual_ids": actual_ids, "modes": dict(Counter(it.mode for it in items)),
                             "content_sha256": digest([asdict(it) for it in items]),
                             "final_prompts": {it.run_id: len(it.final_prompts(it.honest_seats)) for it in items}}
        if len(selected) != nb["max_scenarios"] or actual_ids != selected:
            raise ValueError(f"NARCBench requested {nb['max_scenarios']} scenarios; selected "
                             f"{len(selected)}, loaded {len(items)}; missing {sorted(set(selected)-set(actual_ids))}")
        empty = [key for key, count in audit["narcbench"]["final_prompts"].items() if not count]
        if empty:
            raise ValueError(f"NARCBench scenarios lack honest final prompts: {empty}")
    return audit


def validate_summary(cfg, summary, audit, turns):
    expected = planned_counts(cfg)
    errors = []
    actual = {"turns": len(turns)}

    def check(label, got, want):
        actual[label] = got
        if got != want:
            errors.append(f"{label}: expected {want}, got {got}")

    exp = cfg["experiment"]
    if not summary.get("rows"):
        errors.append("Summary has no aggregate rows")
    if exp == "tom_validation":
        items = summary.get("items", [])
        hitom = [it for it in items if it.get("framing", "").startswith("hitom-")]
        check("core_items", len(items) - len(hitom), expected["core_items"])
        check("hitom_items", len(hitom), expected["hitom_items"])
        check("turns", len(turns), expected["requests"])
        if cfg.get("include_hitom") and {it["item"] for it in hitom} != set(audit["hitom"]["ids"]):
            errors.append("Hi-ToM output item IDs differ from benchmark preflight")
        skipped = summary.get("hitom_skipped_not_single_token", [])
        actual["hitom_skipped_not_single_token"] = skipped
        if skipped:
            errors.append(f"Hi-ToM omitted {len(skipped)} items; see tokenizer-skip list")
    elif exp == "collab":
        check("summary_turns", len(summary.get("turns", [])), expected["turns"])
        check("turns", len(turns), expected["turns"])
    elif exp in ("imposter", "ablation"):
        rows = summary.get("live", [])
        check("reviews", len(rows), expected["reviews"])
        expected_turns = expected["reviews"]
        if expected["reviews"]:
            cells = Counter((r["condition"], r["source"], r["claim"]) for r in rows)
            conditions = cfg.get("interventions", ["none"])
            for cond in conditions:
                for source in cfg.get("sources", ["incentive", "no_incentive", "document"]):
                    for claim in ("true", "false"):
                        check(f"cell/{cond}/{source}/{claim}", cells[cond, source, claim], cfg["n_items"])
        if cfg.get("narcbench"):
            replay = summary.get("narcbench", {}).get("rows", [])
            got_ids = {r["run_id"] for r in replay}
            expected_ids = set(audit["narcbench"]["selected_ids"])
            actual["narcbench_scenario_ids"] = sorted(got_ids)
            if got_ids != expected_ids:
                errors.append(f"NARCBench summary missing scenarios: {sorted(expected_ids-got_ids)}")
            counts = Counter(t.get("extra", {}).get("run_id") for t in turns if t.get("tag", "").startswith("narcbench/"))
            for run_id, count in audit["narcbench"]["final_prompts"].items():
                check(f"narcbench_prompts/{run_id}", counts[run_id], count)
            expected_turns += sum(audit["narcbench"]["final_prompts"].values())
        check("turns", len(turns), expected_turns)
        if exp == "ablation":
            if cfg.get("cluster_fragment"):
                if summary.get("cluster_fragment") != cfg["cluster_fragment"]:
                    errors.append("Ablation fragment lacks its explicit deferred-primary label")
            elif not summary.get("primary"):
                errors.append("Ablation summary lacks the original paired primary comparison")
    elif exp == "peer_selection":
        counts = Counter(r["variant"] for r in summary.get("decisions", []))
        for variant, count in expected["decisions"].items():
            check(f"decisions/{variant}", counts[variant], count)
        check("turns", len(turns), sum(expected["decisions"].values()) + expected["plans"])
        actual["self_preference_scope"] = expected["self_preference_scope"]
    # Every recorded request must have a durable trace; plan-generation calls are unrecorded.
    recorded = [t for t in turns if not t.get("tag", "").startswith("plan/")]
    # Plan generation is unrecorded; decision count still requires every expected trace.
    if exp == "peer_selection":
        recorded = [t for t in turns if t.get("trace_file")]
        check("recorded_decisions", len(recorded), sum(expected["decisions"].values()))
    missing = [i for i, t in enumerate(recorded) if not t.get("trace_file")]
    if missing:
        errors.append(f"Recorded requests missing trace references: {len(missing)}")
    return {"expected": expected, "actual": actual, "errors": errors}


def worker(attempt_dir):
    """One isolated experiment portion. Parent survives errors and schedules the next."""
    attempt_dir = Path(attempt_dir)
    cfg = read_json(attempt_dir / "config.json")
    result = {"started_at": now(), "status": "failed", "versions": versions(),
              "source_file_sha256": source_hashes()}
    try:
        info_url = read_json(cfg["endpoints_file"])["qwen3-8b"]["base_url"] + "/info"
        with urllib.request.urlopen(info_url, timeout=60) as response:
            result["endpoint_info"] = json.load(response)
        if result["endpoint_info"].get("model", "").lower() not in ("qwen3-8b", "qwen/qwen3-8b"):
            raise ValueError(f"Endpoint is not Qwen3-8B: {result['endpoint_info']}")
        atomic_json(attempt_dir / "worker_result.json", result)
        audit = {}
        try:
            benchmark_preflight(cfg, audit)
        finally:
            atomic_json(attempt_dir / "benchmark_audit.json", audit)
            result["benchmark_audit"] = audit
        mod = importlib.import_module(f"jpeers.experiments.{cfg['experiment']}")
        returned = mod.run(cfg)
        run_dir = Path(returned["run_dir"]).resolve()
        result["run_dirs"] = [str(run_dir)]
        summary_path = run_dir / "summary.json"
        summary = read_json(summary_path)
        if cfg.get("cluster_fragment"):
            summary["cluster_fragment"] = cfg["cluster_fragment"]
            atomic_json(summary_path, summary)
            md = run_dir / "summary.md"
            md.write_text("**Condition fragment: primary comparison is deferred until all 13 "
                          "conditions are aggregated.**\n\n" + md.read_text(encoding="utf-8"),
                          encoding="utf-8")
        turns = [json.loads(line) for line in (run_dir / "turns.jsonl").read_text(encoding="utf-8").splitlines() if line]
        result["summary"] = str(summary_path)
        result["summary_sha256"] = hashlib.sha256(summary_path.read_bytes()).hexdigest()
        result["validation"] = validate_summary(cfg, summary, audit, turns)
        missing = [t["trace_file"] for t in turns if t.get("trace_file") and not (run_dir / t["trace_file"]).is_file()]
        if missing:
            result["validation"]["errors"].append(f"Missing saved trace files: {len(missing)}")
        if result["validation"]["errors"]:
            result["status"] = "incomplete"
            raise ValueError("; ".join(result["validation"]["errors"]))
        result["status"] = "completed"
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
        result["traceback"] = traceback.format_exc()
        traceback.print_exc()
    finally:
        result["finished_at"] = now()
        result["run_dirs"] = sorted(str(p.parent.resolve()) for p in (attempt_dir / "results").rglob("config.json"))
        atomic_json(attempt_dir / "worker_result.json", result)
    return 0 if result["status"] == "completed" else 1


def summarize(manifest):
    experiments = {}
    for exp in range(5):
        parts = [job for job in manifest["jobs"] if job["experiment"] == exp]
        complete = all(job["status"] == "completed" for job in parts)
        if exp == 4:
            aggregate = manifest.get("exp4_aggregate", {})
            complete = complete and aggregate.get("status") == "completed" and saved_summary_valid(aggregate)
        experiments[f"exp{exp}"] = {"status": "completed" if complete else "incomplete",
                                    "portions": {job["portion"]: job["status"] for job in parts},
                                    "summaries": [job["attempts"][-1].get("summary") for job in parts if job["attempts"]]}
        if exp == 4:
            experiments["exp4"]["aggregate"] = manifest.get("exp4_aggregate", {"status": "pending"})
    manifest["experiments"] = experiments
    manifest["updated_at"] = now()


def saved_summary_valid(attempt):
    path = Path(attempt.get("summary") or "/nonexistent-summary")
    try:
        summary = read_json(path)
        return (bool(summary.get("rows"))
                and hashlib.sha256(path.read_bytes()).hexdigest() == attempt.get("summary_sha256"))
    except (OSError, ValueError, AttributeError):
        return False


def discovered_run_dirs(directory):
    return sorted(str(p.parent.resolve()) for p in (Path(directory) / "results").rglob("config.json"))


def combine_exp4(fragments, cfg, primary_fn):
    """Combine unchanged per-condition statistics and run the original paired test."""
    conditions = cfg["interventions"]
    if [cond for cond, _ in fragments] != conditions:
        raise ValueError("Exp4 aggregation requires exactly all 13 configured conditions in order")
    sources = cfg.get("sources", ["incentive", "no_incentive", "document"])
    item_ids = [item["id"] for item in bank("imposter_facts", "items")[:cfg["n_items"]]]
    live, rows, contrasts = [], [], {}
    for condition, summary in fragments:
        values = summary.get("live", [])
        expected_keys = {(condition, source, claim, item) for source in sources
                         for claim in ("true", "false") for item in item_ids}
        counts = Counter((r["condition"], r["source"], r["claim"], r["item"]) for r in values)
        if set(counts) != expected_keys or any(n != 1 for n in counts.values()):
            raise ValueError(f"Exp4 fragment {condition} lacks exactly one review for each of its 360 cells/items")
        if summary.get("cluster_fragment", {}).get("condition") != condition:
            raise ValueError(f"Exp4 fragment {condition} lacks its fragment label")
        if not summary.get("rows") or any(r.get("condition") != condition for r in summary["rows"]):
            raise ValueError(f"Exp4 fragment {condition} has missing or foreign aggregate rows")
        live.extend(values)
        rows.extend(summary["rows"])
        contrasts.update(summary.get("contrasts", {}))
    expected_reviews = cfg["n_items"] * len(sources) * 2 * len(conditions)
    if len(live) != expected_reviews:
        raise ValueError(f"Exp4 expected {expected_reviews} total reviews, got {len(live)}")
    primary = primary_fn(live, cfg["primary_condition"], cfg["primary_control"], cfg["max_fluency_drop"])
    expected_pairs = cfg["n_items"] * len(sources)
    if primary.get("n_pairs") != expected_pairs:
        raise ValueError(f"Exp4 primary expected {expected_pairs} pairs, got {primary.get('n_pairs')}")

    # Identical baseline-delta rule to experiments.ablation.run, applied after concatenation.
    def key(row):
        return tuple(sorted((k, v) for k, v in row.items() if k != "condition" and isinstance(v, str)))

    def numeric(value):
        return isinstance(value, (int, float)) and not isinstance(value, bool)

    baseline = {key(row): row for row in rows if row.get("condition") == "none"}
    deltas = []
    for row in rows:
        base = baseline.get(key(row))
        if row.get("condition") == "none" or base is None:
            continue
        delta = {"condition": row["condition"], **dict(key(row))}
        delta.update({f"d_{k}": value - base[k] for k, value in row.items()
                      if k != "n" and numeric(value) and numeric(base.get(k))})
        deltas.append(delta)
    return {"live": live, "rows": rows, "contrasts": contrasts,
            "primary": primary, "deltas_vs_none": deltas}


def aggregate_exp4(manifest, output):
    jobs = [j for j in manifest["jobs"] if j["experiment"] == 4]
    if not all(j["status"] == "completed" for j in jobs):
        manifest["exp4_aggregate"] = {"status": "pending", "reason": "all 13 fragments must complete"}
        return
    aggregate = {"status": "failed", "started_at": now()}
    manifest["exp4_aggregate"] = aggregate
    try:
        for job in jobs:
            if not saved_summary_valid(job["attempts"][-1]):
                raise ValueError(f"Missing or changed fragment summary: {job['id']}")
        # Import only when actual inference results exist; the scheduler otherwise needs stdlib only.
        sys.path.insert(0, str(SOURCE / "src"))
        from jpeers.experiments.ablation import primary_imposter
        from jpeers.experiments.common import write_summary

        cfg = read_json(CONFIGS / CONFIG_FILES[4])
        fragments = [(j["portion"], read_json(j["attempts"][-1]["summary"])) for j in jobs]
        summary = combine_exp4(fragments, cfg, primary_imposter)
        directory = Path(output) / "exp4-aggregate"
        directory.mkdir(parents=True, exist_ok=True)
        summary["fragments"] = [{"condition": j["portion"], "summary": j["attempts"][-1]["summary"],
                                 "summary_sha256": j["attempts"][-1]["summary_sha256"]} for j in jobs]
        summary["aggregation"] = {"n_conditions": len(jobs), "n_reviews": len(summary["live"]),
                                  "n_primary_pairs": summary["primary"]["n_pairs"],
                                  "primary_implementation": "jpeers.experiments.ablation.primary_imposter",
                                  "ordering": "original condition/source/claim/item order preserved"}
        atomic_json(directory / "config.json", cfg)
        write_summary(directory, summary, summary["rows"] + summary["deltas_vs_none"],
                      "Exp. 4 ablation of imposter: metrics and deltas vs none")
        summary_path = directory / "summary.json"
        aggregate.update(status="completed", summary=str(summary_path),
                         summary_sha256=hashlib.sha256(summary_path.read_bytes()).hexdigest(),
                         run_dir=str(directory), n_reviews=len(summary["live"]),
                         n_primary_pairs=summary["primary"]["n_pairs"])
    except Exception as exc:
        aggregate.update(error=f"{type(exc).__name__}: {exc}", traceback=traceback.format_exc())
        traceback.print_exc()
    aggregate["finished_at"] = now()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ports", nargs="+", type=int, default=[18080])
    ap.add_argument("--host", default="localhost")
    ap.add_argument("--output-dir", type=Path, default=SOURCE / "runs" / "cluster-suite")
    ap.add_argument("--source-sha", default=SOURCE_SHA)
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--retry-failed", action="store_true")
    ap.add_argument("--prepare-only", action="store_true", help="write plan/provenance without starting requests")
    ap.add_argument("--worker", type=Path, help=argparse.SUPPRESS)
    args = ap.parse_args(argv)
    if args.worker:
        return worker(args.worker)
    if len(set(args.ports)) != len(args.ports) or any(p < 1 or p > 65535 for p in args.ports):
        ap.error("ports must be unique valid TCP port numbers")
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    # A process lock prevents two resumed schedulers from dispatching duplicate work.
    import fcntl
    lock = (output / ".scheduler.lock").open("w")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        ap.error(f"another scheduler owns {output}")
    manifest_path = output / "manifest.json"
    wanted = build_jobs()
    if manifest_path.exists():
        manifest = read_json(manifest_path)
        if [(j["id"], j["config_sha256"]) for j in wanted] != [(j["id"], j["config_sha256"]) for j in manifest["jobs"]]:
            ap.error("configs changed since this suite was created; use a new output directory")
        for job in manifest["jobs"]:
            if job["status"] == "running":
                pid = job["attempts"][-1].get("pid")
                try:
                    if pid:
                        os.kill(pid, 0)
                        ap.error(f"prior worker {pid} may still be active; stop it before resuming")
                except ProcessLookupError:
                    pass
                job["status"] = "interrupted"
                attempt = job["attempts"][-1]
                result_path = Path(attempt["directory"]) / "worker_result.json"
                if result_path.is_file():
                    result = read_json(result_path)
                    if result.get("status") == "completed" and saved_summary_valid(result):
                        attempt.update(result, recovered_after_scheduler_exit=True,
                                       exit_status_note="Worker result saved; exit code unavailable after scheduler exit")
                        job["status"] = "completed"
            if job["status"] == "completed":
                if not saved_summary_valid(job["attempts"][-1]):
                    job["status"] = "incomplete"
    else:
        manifest = {"created_at": now(), "source_branch": "origin/jeffrey/initial_experiment",
                    "source_branch_sha": args.source_sha, "source_directory": str(SOURCE),
                    "versions": versions(), "source_file_sha256": source_hashes(),
                    "model": "Qwen/Qwen3-8B", "backend_requirement": "original HF/PyTorch HTTP serving",
                    "config_overrides": ["all inference endpoints: qwen3-1.7b -> qwen3-8b",
                        "HTTP endpoint replaces local loading", "full repository default sample sizes preserved",
                        "Exp0 core/Hi-ToM and Exp2 core/NARCBench run as independent portions",
                        "Exp4 runs as 13 single-condition fragments with original statistics aggregated afterward",
                        "self_preference remains same-family deterministic versus sampled Qwen plans"],
                    "jobs": wanted}
    for job in manifest["jobs"]:
        job["planned_counts"] = planned_counts(job["config"])
    manifest["scheduler"] = {"pid": os.getpid(), "ports": args.ports, "host": args.host,
                             "started_at": now(), "python": args.python}
    manifest["status"] = "prepared" if args.prepare_only else "running"
    summarize(manifest)
    atomic_json(manifest_path, manifest)
    if args.prepare_only:
        print(manifest_path, flush=True)
        return 0
    queue = deque(job for job in manifest["jobs"] if job["status"] in ("pending", "interrupted")
                  or (args.retry_failed and job["status"] in ("failed", "incomplete")))
    active = {}
    stopped = []

    def on_signal(signum, _frame):
        stopped.append(signum)

    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, on_signal)
    try:
        while queue or active:
            if stopped:
                break
            for port in args.ports:
                if port in active or not queue:
                    continue
                job = queue.popleft()
                directory = output / "jobs" / job["id"] / f"attempt-{len(job['attempts']) + 1:03d}"
                directory.mkdir(parents=True, exist_ok=False)
                cfg = copy.deepcopy(job["config"])
                cfg.update(out_dir=str(directory / "results"), run_id=job["id"],
                           endpoints_file=str(directory / "endpoints.json"))
                endpoint = {"qwen3-8b": {"backend": "http", "base_url": f"http://{args.host}:{port}",
                                         "timeout_s": 3600, "retries": 2}}
                atomic_json(directory / "config.json", cfg)
                atomic_json(directory / "endpoints.json", endpoint)
                attempt = {"started_at": now(), "directory": str(directory), "port": port,
                           "log": str(directory / "process.log"), "status": "running",
                           "config": str(directory / "config.json"), "exit_code": None}
                job["attempts"].append(attempt)
                job["status"] = "running"
                log = (directory / "process.log").open("a", encoding="utf-8")
                env = dict(os.environ)
                env["PYTHONPATH"] = str(SOURCE / "src") + os.pathsep + env.get("PYTHONPATH", "")
                command = [args.python, "-u", str(Path(__file__).resolve()), "--worker", str(directory)]
                attempt["command"] = command
                try:
                    process = subprocess.Popen(command, cwd=SOURCE, env=env, stdout=log,
                                               stderr=subprocess.STDOUT, start_new_session=True)
                    attempt["pid"] = process.pid
                    active[port] = (process, log, job, attempt)
                    print(f"{now()} started {job['id']} on {port}", flush=True)
                except Exception as exc:
                    log.close()
                    job["status"] = attempt["status"] = "failed"
                    attempt.update(error=str(exc), finished_at=now())
                summarize(manifest)
                atomic_json(manifest_path, manifest)
            for port, (process, log, job, attempt) in list(active.items()):
                if process.poll() is None:
                    continue
                log.close()
                result_path = Path(attempt["directory"]) / "worker_result.json"
                try:
                    result = read_json(result_path)
                except (OSError, ValueError) as exc:
                    result = {"status": "failed", "error": f"Worker exited without readable result: {exc}"}
                attempt.update(result, exit_code=process.returncode, finished_at=now())
                attempt["run_dirs"] = discovered_run_dirs(attempt["directory"])
                if attempt["status"] == "completed" and (process.returncode != 0 or not saved_summary_valid(attempt)):
                    attempt["status"] = "failed"
                job["status"] = attempt["status"]
                print(f"{now()} {job['id']}: {job['status']} (exit {process.returncode})", flush=True)
                del active[port]
                summarize(manifest)
                atomic_json(manifest_path, manifest)
            if active:
                time.sleep(0.5)
    finally:
        for process, _, _, _ in active.values():
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
        deadline = time.monotonic() + 10
        for process, log, job, attempt in active.values():
            try:
                process.wait(timeout=max(0.01, deadline - time.monotonic()))
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            log.close()
            job["status"] = attempt["status"] = "interrupted"
            attempt.update(exit_code=process.returncode, finished_at=now(),
                           run_dirs=discovered_run_dirs(attempt["directory"]))
        aggregate_exp4(manifest, output)
        summarize(manifest)
        complete = all(exp["status"] == "completed" for exp in manifest["experiments"].values())
        manifest.update(status="completed" if complete else "interrupted" if stopped else "incomplete",
                        finished_at=now())
        atomic_json(manifest_path, manifest)
    print(f"Suite {manifest['status']}: {manifest_path}", flush=True)
    return 0 if manifest["status"] == "completed" else 130 if stopped else 1


if __name__ == "__main__":
    raise SystemExit(main())
