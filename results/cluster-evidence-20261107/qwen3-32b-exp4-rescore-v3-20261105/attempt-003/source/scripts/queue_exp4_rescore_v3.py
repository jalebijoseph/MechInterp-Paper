"""Serve the pinned independent grader, run the gates and rescore, then shut down."""

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request

from queue_exp0_32b import assert_port_free, atomic_json
from queue_followup import gpu_pairs, now, predecessor_released


MODEL = "openai/gpt-oss-120b"
SERVED_MODEL = "gpt-oss-120b-exp4-v3"


def stop_owned_group(process):
    if process is None:
        return
    # vLLM's workers can outlive the session leader during a startup failure.
    for sig, grace in ((signal.SIGTERM, 30), (signal.SIGKILL, 10)):
        try:
            os.killpg(process.pid, sig)
        except ProcessLookupError:
            process.poll()
            return
        deadline = time.monotonic() + grace
        while time.monotonic() < deadline:
            process.poll()
            groups = subprocess.check_output(["ps", "-eo", "pgid=,stat="], text=True, timeout=5)
            members = [line.split() for line in groups.splitlines() if line.strip()]
            if not any(pgid == str(process.pid) and not stat.startswith("Z") for pgid, stat in members):
                return
            time.sleep(0.25)
    raise TimeoutError(f"Owned process group {process.pid} did not stop")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--predecessor", type=Path, required=True)
    parser.add_argument("--port", type=int, default=18310)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--fixture-sha256", required=True)
    parser.add_argument("--concurrency", type=int, default=24)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    source = Path(__file__).resolve().parents[1]
    fixture = source / "src/jpeers/data/fact_grader_holdout_v3.json"
    runner = source / "scripts/run_exp4_rescore_v3.py"
    if hashlib.sha256(fixture.read_bytes()).hexdigest() != args.fixture_sha256:
        raise ValueError("Holdout differs from its frozen hash")
    if not runner.is_file():
        raise ValueError("Scoring worker is missing")
    root.mkdir(parents=True, exist_ok=True)
    lock = (root / ".manager.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    status_path = root / "manager-status.json"
    if status_path.exists() and not args.resume:
        raise ValueError("A manager attempt is already recorded; use --resume to recover it")
    state = {"pid": os.getpid(), "started_at": now(), "status": "waiting",
             "model": MODEL, "model_revision": args.revision,
             "holdout_sha256": args.fixture_sha256, "predecessor": str(args.predecessor)}
    server = worker = None

    def update(**values):
        state.update(values)
        atomic_json(status_path, state)

    def interrupted(signum, frame):
        raise KeyboardInterrupt(f"Received signal {signum}")

    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    try:
        deadline = time.monotonic() + 24 * 3600
        while True:
            provenance_path = root / "model-provenance.json"
            model = json.loads(provenance_path.read_text()) if provenance_path.exists() else {}
            if model.get("status") == "failed":
                raise RuntimeError(f"Model download failed: {model.get('error')}")
            if model and (model.get("model") != MODEL or model.get("revision", args.revision) != args.revision):
                raise ValueError("Downloaded model does not match the pinned grader")
            released, predecessor = predecessor_released(args.predecessor)
            if released and predecessor["manifest_status"] != "completed":
                raise RuntimeError("Original suite did not complete")
            pairs = gpu_pairs() if released and model.get("status") == "completed" else []
            update(checked_at=now(), model_download=model.get("status"), predecessor_state=predecessor)
            if pairs:
                break
            if time.monotonic() > deadline:
                raise TimeoutError("Waiting for downloaded weights and idle GPUs 0–3 timed out")
            time.sleep(10)

        assert_port_free(args.port)
        devices = [gpu for pair in pairs for gpu in pair]
        model_path = Path(model["model_path"]).resolve(strict=True)
        if model_path.name != args.revision:
            raise ValueError("The model snapshot path does not match the pinned revision")
        base_url = f"http://127.0.0.1:{args.port}"
        command = [sys.executable, "-m", "vllm.entrypoints.cli.main", "serve", str(model_path),
                   "--served-model-name", SERVED_MODEL, "--host", "127.0.0.1", "--port", str(args.port),
                   "--tensor-parallel-size", str(len(devices)), "--max-model-len", "8192",
                   "--max-num-seqs", str(args.concurrency), "--gpu-memory-utilization", "0.86",
                   "--reasoning-parser", "openai_gptoss", "--moe-backend", "marlin",
                   "--generation-config", "vllm", "--enforce-eager"]
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=",".join(gpu["uuid"] for gpu in devices),
                   PYTHONPATH=str(source / "src"), PYTHONUNBUFFERED="1", HF_HUB_OFFLINE="1")
        with (root / "server.log").open("a") as log:
            server = subprocess.Popen(command, cwd=source, env=env, stdin=subprocess.DEVNULL,
                                      stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        update(status="starting_grader", server_pid=server.pid, server_command=command,
               physical_gpus=[gpu["index"] for gpu in devices], port=args.port)
        deadline = time.monotonic() + 1800
        while True:
            if server.poll() is not None:
                raise RuntimeError(f"Grader server exited {server.returncode}; see server.log")
            try:
                with urllib.request.urlopen(f"{base_url}/v1/models", timeout=5) as reply:
                    advertised = {entry["id"] for entry in json.load(reply)["data"]}
                if SERVED_MODEL in advertised:
                    break
                raise RuntimeError(f"Unexpected server model IDs: {advertised}")
            except (urllib.error.URLError, TimeoutError):
                if time.monotonic() > deadline:
                    raise TimeoutError("Grader server startup timed out")
                time.sleep(3)

        worker_command = [sys.executable, str(runner),
                          "--input-root", str(args.predecessor / "source/runs/full-suite"),
                          "--output-dir", str(root / "grading"), "--base-url", base_url,
                          "--model", SERVED_MODEL, "--model-revision", args.revision,
                          "--fixture", str(fixture), "--fixture-sha256", args.fixture_sha256,
                          "--concurrency", str(args.concurrency)]
        if args.resume and (root / "grading").exists():
            worker_command.append("--resume")
        with (root / "worker.log").open("a") as log:
            worker = subprocess.Popen(worker_command, cwd=source, env=dict(env, CUDA_VISIBLE_DEVICES=""),
                                      stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                                      start_new_session=True)
        update(status="running", worker_pid=worker.pid, worker_command=worker_command)
        deadline = time.monotonic() + 3 * 86400
        while worker.poll() is None:
            if server.poll() is not None:
                raise RuntimeError(f"Grader server exited {server.returncode}; see server.log")
            if time.monotonic() > deadline:
                raise TimeoutError("Scoring worker exceeded its runtime limit")
            time.sleep(3)
        progress_path = root / "grading/progress.json"
        progress = json.loads(progress_path.read_text()) if progress_path.exists() else {}
        update(status="completed" if worker.returncode == 0 else "incomplete",
               worker_exit_code=worker.returncode, worker_progress=progress)
    except BaseException as exc:
        update(status="failed", error=repr(exc))
    finally:
        shutdown = {"finished_at": now()}
        for name, process in (("worker", worker), ("server", server)):
            try:
                stop_owned_group(process)
                shutdown[f"{name}_stopped_at"] = now()
            except Exception as exc:
                shutdown.update(status="failed", **{f"{name}_cleanup_error": repr(exc)})
        update(**shutdown)
    return 0 if state["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
