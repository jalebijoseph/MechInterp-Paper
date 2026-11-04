#!/usr/bin/env python3
"""Wait for the successful 8B suite, then run original Exp0 on pinned BF16 Qwen3-32B.

This controller is CPU-only until prerequisites succeed and two GPUs among 0–3 are
idle. It owns only its child process groups and never stops prerequisite servers.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import fcntl
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
import traceback
import urllib.error
import urllib.request

MODEL = "Qwen/Qwen3-32B"
MODEL_REVISION = "9216db5781bf21249d130ec9da846c4624c16137"
LENS_REPO = "neuronpedia/jacobian-lens"
LENS_REVISION = "e166deb5ad540932100ef803c8c9bcc2a5453694"
LENS_FILE = "qwen3-32b/jlens/Salesforce-wikitext/Qwen3-32B_jacobian_lens.pt"
SOURCE_SHA = "1f97522907c33cdb557163769b599088f8e7936b"
TERMINAL_FAILURES = {"failed", "incomplete", "interrupted", "error", "cancelled", "canceled"}


def now():
    return datetime.now(timezone.utc).isoformat()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def atomic_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def prerequisite_state(root):
    paths = {"manager": Path(root) / "manager-status.json",
             "manifest": Path(root) / "source/runs/full-suite/manifest.json"}
    observed = {}
    for name, path in paths.items():
        try:
            observed[name] = read_json(path)
        except (OSError, ValueError) as exc:
            observed[name] = {"status": "unavailable", "detail": str(exc)}
    for name, value in observed.items():
        if value.get("status") in TERMINAL_FAILURES:
            return "failed", f"Prerequisite {name} is {value['status']}; 32B will not start", observed
    if any(value.get("status") != "completed" for value in observed.values()):
        return "waiting", "Waiting for successful manager and full-suite manifest", observed
    if not observed["manager"].get("servers_stopped_at"):
        return "waiting", "Waiting for prerequisite server shutdown record", observed
    return "ready", "8B completed successfully and its servers stopped", observed


def select_idle_gpus(gpu_csv, process_csv):
    """Fail closed on malformed inventory; never select a GPU outside physical 0–3."""
    busy = set()
    for line in process_csv.splitlines():
        if not line.strip():
            continue
        fields = [part.strip() for part in line.split(",")]
        if len(fields) != 2 or not fields[0].startswith("GPU-") or not fields[1].isdigit():
            raise ValueError(f"Unrecognized compute process inventory: {line}")
        busy.add(fields[0])
    available = []
    inventory = []
    for line in gpu_csv.splitlines():
        if not line.strip():
            continue
        fields = [part.strip() for part in line.split(",")]
        if len(fields) != 3:
            raise ValueError(f"Unrecognized GPU inventory: {line}")
        index, uuid, memory = int(fields[0]), fields[1], float(fields[2])
        if not uuid.startswith("GPU-") or memory < 0:
            raise ValueError(f"Invalid GPU inventory: {line}")
        if index not in range(4):
            continue
        row = {"index": index, "uuid": uuid, "memory_used_mib": memory,
               "has_compute_process": uuid in busy}
        inventory.append(row)
        if memory < 256 and uuid not in busy:
            available.append(index)
    return sorted(available)[:2] if len(available) >= 2 else [], inventory


def gpu_snapshot():
    def query(fields, kind):
        return subprocess.run(["nvidia-smi", f"--query-{kind}={fields}",
                               "--format=csv,noheader,nounits"], check=True,
                              capture_output=True, text=True, timeout=30).stdout
    gpu_csv = query("index,uuid,memory.used", "gpu")
    process_csv = query("gpu_uuid,pid", "compute-apps")
    return select_idle_gpus(gpu_csv, process_csv)


def selected_gpu_uuids(selected, inventory):
    rows = {row['index']: row for row in inventory}
    if len(selected) != 2 or len(set(selected)) != 2 or any(gpu not in range(4) for gpu in selected):
        raise ValueError('32B requires two distinct allowed physical GPUs among 0-3')
    uuids = []
    for gpu in selected:
        row = rows[gpu]
        if row['has_compute_process'] or row['memory_used_mib'] >= 256 or not row['uuid'].startswith('GPU-'):
            raise ValueError(f'Physical GPU {gpu} is not idle or has an invalid UUID')
        uuids.append(row['uuid'])
    return uuids


def terminate_process_group(process, grace=15):
    """Stop only a subprocess launched by this controller with start_new_session=True."""
    if process is None or process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        process.wait(timeout=grace)
    except subprocess.TimeoutExpired:
        if process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        process.wait(timeout=grace)


def assert_port_free(port=18100):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        try:
            probe.bind(("127.0.0.1", port))
        except OSError as exc:
            raise RuntimeError(f"Port {port} is unavailable; refusing to connect to another server") from exc


def download_pinned(root, hf_home):
    """Called in the requested Python environment only after the prerequisite gate."""
    os.environ["HF_HOME"] = str(hf_home)
    from huggingface_hub import hf_hub_download, snapshot_download

    record = {"model": MODEL, "model_revision": MODEL_REVISION,
              "lens_repo": LENS_REPO, "lens_revision": LENS_REVISION,
              "lens_file": LENS_FILE, "hf_home": str(hf_home), "started_at": now(),
              "status": "downloading"}
    path = root / "model-provenance.json"
    atomic_json(path, record)
    model_path = snapshot_download(MODEL, revision=MODEL_REVISION,
                                  allow_patterns=["*.json", "*.safetensors", "*.model", "*.txt", "*.jinja", "*.py"])
    lens_path = hf_hub_download(LENS_REPO, LENS_FILE, revision=LENS_REVISION)
    record.update(model_path=str(Path(model_path).resolve()), lens_path=str(Path(lens_path).resolve()),
                  status="completed", finished_at=now())
    atomic_json(path, record)


def exp0_config(root):
    return {"experiment": "tom_validation", "out_dir": str(root / "results"),
            "run_id": "qwen3-32b-exp0", "endpoints_file": str(root / "endpoints.json"),
            "model": "qwen3-32b", "n_items": 20, "framings": ["people", "agents"],
            "include_hitom": True, "hitom_items": 60, "hitom_orders": [0, 1, 2, 3],
            "max_new_tokens": 6, "record": {"top_k": 10}}


def validate_results(run_dir):
    run_dir = Path(run_dir)
    summary_path = run_dir / "summary.json"
    summary = read_json(summary_path)
    turns = [json.loads(line) for line in (run_dir / "turns.jsonl").read_text(encoding="utf-8").splitlines() if line]
    items = summary.get("items", [])
    core = [item for item in items if item.get("framing") in ("people", "agents")]
    hitom = [item for item in items if item.get("framing", "").startswith("hitom-o")]
    cells = Counter((item.get("framing"), item.get("false_belief")) for item in core)
    errors = []
    for framing in ("people", "agents"):
        for false_belief in (False, True):
            if cells[framing, false_belief] != 20:
                errors.append(f"{framing}/false_belief={false_belief}: expected20, got{cells[framing, false_belief]}")
    if len(core) != 80 or len(hitom) != 60 or len(items) != 140 or len(turns) != 140:
        errors.append(f"Expected80core+60HiToM/140turns; got{len(core)}+{len(hitom)}/{len(turns)}")
    if len({(item.get("framing"), item.get("item")) for item in items}) != 140:
        errors.append("Missing or duplicate item identities")
    if Counter(item.get("framing") for item in hitom) != Counter({f"hitom-o{o}": 15 for o in range(4)}):
        errors.append("Hi-ToM must contain15items for each order0–3")
    if summary.get("hitom_skipped_not_single_token"):
        errors.append("Hi-ToM tokenizer exclusions occurred; full requested sample did not complete")
    if not summary.get("rows"):
        errors.append("Saved summary has no aggregate rows")
    trace_files = [turn.get("trace_file") for turn in turns]
    if len(set(trace_files)) != 140:
        errors.append("Expected140distinct trace references")
    for turn in turns:
        path = run_dir / (turn.get("trace_file") or "missing-trace")
        if not path.is_file() or path.stat().st_size == 0:
            errors.append(f"Missing saved trace: {turn.get('trace_file')}")
        if turn.get("model", "").lower() not in ("qwen3-32b", "qwen/qwen3-32b"):
            errors.append(f"Unexpected result model: {turn.get('model')}")
    result = {"core_items": len(core), "hitom_items": len(hitom), "turns": len(turns),
              "traces": len(set(trace_files)), "summary": str(summary_path),
              "summary_sha256": hashlib.sha256(summary_path.read_bytes()).hexdigest(), "errors": errors}
    return result


def get_json_url(url, timeout=5):
    try:
        response = urllib.request.urlopen(url, timeout=timeout)
    except urllib.error.HTTPError as exc:
        response = exc
    with response:
        return json.load(response)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--prerequisite-root", type=Path)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--hf-home", type=Path)
    parser.add_argument("--poll-seconds", type=float, default=15)
    parser.add_argument("--startup-timeout", type=float, default=3600)
    parser.add_argument("--experiment-timeout", type=float, default=21600)
    parser.add_argument("--download-timeout", type=float, default=21600)
    parser.add_argument("--retry-failed", action="store_true")
    parser.add_argument("--download", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    root = args.root.resolve()
    if args.download:
        if not args.hf_home:
            parser.error("download worker requires --hf-home")
        download_pinned(root, args.hf_home.resolve())
        return 0
    if not args.prerequisite_root:
        parser.error("--prerequisite-root is required")
    if not 0 < args.poll_seconds <= 60:
        parser.error("--poll-seconds must be between0 and60")
    prerequisite = args.prerequisite_root.resolve()
    hf_home = (args.hf_home or prerequisite / "hf-cache").resolve()
    source = root / "source"
    root.mkdir(parents=True, exist_ok=True)
    lock = (root / ".queue.lock").open("w")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        parser.error("another controller already owns this queue")
    status_path = root / "queue-status.json"
    prior = read_json(status_path) if status_path.exists() else None
    if prior:
        if prior.get("status") == "completed":
            validation = validate_results(Path(prior["validation"]["summary"]).parent)
            if not validation["errors"]:
                print("32B Exp0 already completed and remains validated", flush=True)
                return 0
            parser.error("completed results no longer validate; preserve this run and use a new root")
        if prior.get("status") in TERMINAL_FAILURES and not args.retry_failed:
            print(f"Queue remains{prior['status']}; use --retry-failed for a new preserved attempt", flush=True)
            return 1
        for name in ("server", "experiment", "download"):
            process_info = prior.get(name, {})
            pid = process_info.get("pid")
            if pid and process_info.get("exit_code") is None:
                try:
                    os.kill(pid, 0)
                except ProcessLookupError:
                    continue
                parser.error(f"previous owned{name}process{pid}may still be alive; refusing duplicate launch")
        archive = root / "attempt-history" / f"queue-status-{time.time_ns()}.json"
        atomic_json(archive, prior)
    state = {"status": "queued", "created_at": now(), "pid": os.getpid(),
             "attempt": (prior.get("attempt", 1) + 1) if prior else 1,
             "prerequisite_root": str(prerequisite), "python": args.python, "hf_home": str(hf_home),
             "source_sha": SOURCE_SHA, "backend": "original HF/PyTorch sharded BF16",
             "model": MODEL, "model_revision": MODEL_REVISION, "lens_revision": LENS_REVISION,
             "allowed_physical_gpus": [0, 1, 2, 3], "required_idle_gpu_count": 2,
             "selected_gpus": [], "gpu_allocation_started": False,
             "events": [], "server": {"started": False}}
    stopped = []
    processes = {}
    logs = []

    def save(status=None, **updates):
        if status is not None and state["status"] != status:
            state["events"].append({"at": now(), "status": status})
            print(f"{now()} {status}", flush=True)
            state["status"] = status
        state.update(updates, updated_at=now())
        atomic_json(status_path, state)

    def check_stop():
        if stopped:
            raise InterruptedError(f"Controller received signal{stopped[-1]}")

    def pause():
        # Short polling keeps SIGTERM responsive without ever allocating a GPU.
        deadline = time.monotonic() + args.poll_seconds
        while time.monotonic() < deadline:
            check_stop()
            time.sleep(min(0.5, deadline - time.monotonic()))

    def launch(name, command, env):
        log_path = root / f"{name}-{time.time_ns()}.log"
        log = log_path.open("w", encoding="utf-8")
        logs.append(log)
        process = subprocess.Popen(command, cwd=source, env=env, stdout=log,
                                   stderr=subprocess.STDOUT, start_new_session=True)
        processes[name] = process
        state[name] = {"pid": process.pid, "command": command, "log": str(log_path),
                       "started_at": now(), "exit_code": None, "started": True}
        save()
        return process

    def wait_process(name, timeout):
        process = processes[name]
        deadline = time.monotonic() + timeout
        while process.poll() is None:
            check_stop()
            if time.monotonic() >= deadline:
                raise TimeoutError(f"{name}exceeded{timeout}seconds")
            pause()
        state[name].update(exit_code=process.returncode, finished_at=now())
        save()
        if process.returncode:
            raise RuntimeError(f"{name}exited{process.returncode}; see{state[name]['log']}")

    def wait_capacity(stage):
        while True:
            check_stop()
            selected, inventory = gpu_snapshot()
            save(stage, gpu_inventory=inventory, candidate_gpus=selected)
            if len(selected) == 2:
                return selected
            pause()

    signal.signal(signal.SIGTERM, lambda sig, frame: stopped.append(sig))
    signal.signal(signal.SIGINT, lambda sig, frame: stopped.append(sig))
    save()
    outcome = "failed"
    try:
        while True:
            check_stop()
            gate, reason, observed = prerequisite_state(prerequisite)
            save("waiting_for_8b", prerequisite_reason=reason,
                 prerequisite_status={name: {k: value.get(k) for k in ("status", "servers_stopped_at")}
                                      for name, value in observed.items()})
            if gate == "failed":
                raise RuntimeError(reason)
            if gate == "ready":
                save(prerequisite_satisfied_at=now())
                break
            pause()
        wait_capacity("waiting_capacity_before_download")
        env = dict(os.environ)
        env.update(HF_HOME=str(hf_home), PYTHONPATH=str(source / "src"),
                   OMP_NUM_THREADS="4", MKL_NUM_THREADS="4", OPENBLAS_NUM_THREADS="4",
                   CUDA_VISIBLE_DEVICES="")
        save("downloading")
        launch("download", [args.python, "-u", str(Path(__file__).resolve()), "--root", str(root),
                             "--hf-home", str(hf_home), "--download"], env)
        wait_process("download", args.download_timeout)
        provenance = read_json(root / "model-provenance.json")
        if provenance.get("status") != "completed" or provenance.get("model_revision") != MODEL_REVISION or provenance.get("lens_revision") != LENS_REVISION:
            raise ValueError("Downloaded model/lens provenance does not match the pinned revisions")
        selected = wait_capacity("waiting_capacity_after_download")
        selected_uuids = selected_gpu_uuids(selected, state["gpu_inventory"])
        assert_port_free()
        env.update(CUDA_VISIBLE_DEVICES=",".join(selected_uuids), JPEERS_MODEL="qwen3-32b",
                   JPEERS_MODEL_PATH=provenance["model_path"], JPEERS_MODEL_REVISION=MODEL_REVISION,
                   JPEERS_LENS_PATH=provenance["lens_path"], JPEERS_DTYPE="bf16",
                   JPEERS_DEVICE="auto", JPEERS_PORT="18100", PORT="18100")
        save("starting_server", selected_gpus=selected, selected_gpu_uuids=selected_uuids,
             gpu_selected_at=now())
        server = launch("server", [args.python, "-u", "scripts/serve_sharded.py", "--root", str(root), "--port", "18100"], env)
        save(gpu_allocation_started=True)
        deadline = time.monotonic() + args.startup_timeout
        while True:
            check_stop()
            if server.poll() is not None:
                raise RuntimeError(f"Server exited during startup with{server.returncode}")
            if time.monotonic() >= deadline:
                raise TimeoutError("Server startup exceeded configured timeout")
            try:
                health = get_json_url("http://127.0.0.1:18100/health")
            except (OSError, ValueError):
                health = {"status": "unreachable"}
            save(server_health=health)
            if health.get("status") == "error":
                raise RuntimeError(f"Permanent model startup error: {health.get('detail')}")
            if health.get("status") == "ok":
                info = get_json_url("http://127.0.0.1:18100/info")
                if info.get("model") != "qwen3-32b" or info.get("n_layers") != 64 or info.get("d_model") != 5120:
                    raise ValueError(f"Unexpected endpoint identity: {info}")
                save(endpoint_info=info, server_ready_at=now())
                break
            pause()
        config = exp0_config(root)
        if state["attempt"] > 1:
            config["run_id"] += f"-attempt-{state['attempt']}-{time.time_ns()}"
        atomic_json(root / "exp0-config.json", config)
        atomic_json(root / "endpoints.json", {"qwen3-32b": {"backend": "http", "base_url": "http://localhost:18100",
                                                             "timeout_s": 3600, "retries": 0}})
        save("running_experiment", config=str(root / "exp0-config.json"))
        experiment_env = dict(env, CUDA_VISIBLE_DEVICES="")
        launch("experiment", [args.python, "-u", "-m", "jpeers.cli", "run", str(root / "exp0-config.json")], experiment_env)
        wait_process("experiment", args.experiment_timeout)
        run_dir = root / "results/tom_validation" / config["run_id"]
        if not run_dir.is_dir():
            raise ValueError("Experiment exited without a result directory")
        save("validating", run_dirs=[str(run_dir)])
        validation = validate_results(run_dir)
        save(validation=validation)
        if validation["errors"]:
            raise ValueError("; ".join(validation["errors"]))
        outcome = "completed"
    except InterruptedError as exc:
        outcome = "interrupted"
        save(error=str(exc))
    except Exception as exc:
        save(error=f"{type(exc).__name__}: {exc}", traceback=traceback.format_exc())
        traceback.print_exc()
    finally:
        save("stopping_owned_processes")
        for name in ("experiment", "server", "download"):
            if name in processes:
                terminate_process_group(processes[name])
                state[name].update(exit_code=processes[name].returncode, stopped_at=now())
        for log in logs:
            log.close()
        if "server" in processes:
            state["servers_stopped_at"] = now()
            state["gpu_allocation_started"] = False
        state["run_dirs"] = sorted(str(path) for path in (root / "results/tom_validation").glob("qwen3-32b-exp0*"))
        save(outcome, finished_at=now())
    return 0 if outcome == "completed" else 130 if outcome == "interrupted" else 1


if __name__ == "__main__":
    raise SystemExit(main())
