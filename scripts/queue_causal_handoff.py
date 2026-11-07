"""Wait for grader shutdown, run the frozen Qwen handoff study, and stop its server."""
from __future__ import annotations

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

from queue_exp0_32b import (
    MODEL_REVISION, LENS_REVISION, assert_port_free, atomic_json,
    gpu_snapshot, now, selected_gpu_uuids,
)
from queue_exp4_rescore_v3 import stop_owned_group


def grader_released(root: Path) -> tuple[bool, dict]:
    path = root / "manager-status.json"
    if not path.is_file():
        return False, {"status": "unavailable"}
    state = json.loads(path.read_text())
    released = (state.get("status") in {"completed", "incomplete", "failed"}
                and bool(state.get("server_stopped_at"))
                and not state.get("server_cleanup_error"))
    return released, {key: state.get(key) for key in (
        "status", "server_stopped_at", "server_cleanup_error")}


def verify_preflight(root: Path, model_path: Path) -> tuple[dict, dict]:
    source = (root / "source").resolve()
    preflight = json.loads((root / "preflight.json").read_text())
    if preflight.get("status") != "passed" or not preflight.get("source_files"):
        raise ValueError("The frozen CPU preflight must pass before GPU execution")
    for relative, expected in preflight["source_files"].items():
        path = (source / relative).resolve()
        if not path.is_relative_to(source) or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f"Frozen study source differs from its preflight hash: {relative}")
    config = source / "configs/causal_handoff_32b.json"
    if hashlib.sha256(config.read_bytes()).hexdigest() != preflight.get("config_sha256"):
        raise ValueError("Study configuration differs from its frozen preflight hash")
    model = json.loads(model_path.read_text())
    if (model.get("model") != "Qwen/Qwen3-32B" or model.get("model_revision") != MODEL_REVISION
            or model.get("lens_revision") != LENS_REVISION):
        raise ValueError("The study requires the pinned Qwen3-32B model and Jacobian lens")
    checkpoint = Path(model["model_path"]).resolve(strict=True)
    lens = Path(model["lens_path"]).resolve(strict=True)
    if checkpoint.name != MODEL_REVISION or not lens.is_file():
        raise ValueError("Pinned model or lens artifact is unavailable")
    return preflight, model


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--grader-root", type=Path, required=True)
    parser.add_argument("--qwen-provenance", type=Path, required=True)
    parser.add_argument("--port", type=int, default=18130)
    parser.add_argument("--wait-timeout", type=int, default=7 * 86400)
    parser.add_argument("--run-timeout", type=int, default=3 * 86400)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    source = root / "source"
    root.mkdir(parents=True, exist_ok=True)
    status_path = root / "manager-status.json"
    if status_path.exists() and not args.resume:
        parser.error("Use a new attempt directory or --resume to preserve an existing study")
    state = {"status": "waiting", "pid": os.getpid(), "started_at": now(),
             "grader_root": str(args.grader_root), "protocol": "causal-handoff-v1"}
    server = worker = None

    def update(**values):
        state.update(values)
        atomic_json(status_path, state)

    def interrupted(signum, frame):
        raise KeyboardInterrupt(f"Received signal {signum}")

    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    with (root / ".manager.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            preflight, model = verify_preflight(root, args.qwen_provenance)
            update(preflight=preflight)
            deadline = time.monotonic() + args.wait_timeout
            while True:
                released, predecessor = grader_released(args.grader_root)
                selected, inventory = gpu_snapshot() if released else ([], [])
                update(checked_at=now(), predecessor=predecessor, gpu_inventory=inventory)
                if selected:
                    break
                if time.monotonic() > deadline:
                    raise TimeoutError("Waiting for grader shutdown and idle GPUs 0–3 expired")
                time.sleep(20)
            verify_preflight(root, args.qwen_provenance)
            uuids = selected_gpu_uuids(selected, inventory)
            assert_port_free(args.port)
            atomic_json(root / "model-provenance.json", model)
            base_env = dict(os.environ, HF_HOME=model["hf_home"], HF_HUB_OFFLINE="1",
                            PYTHONPATH=f"{source / 'src'}:{source / 'scripts'}",
                            PYTHONUNBUFFERED="1", TOKENIZERS_PARALLELISM="false",
                            OMP_NUM_THREADS="4")
            server_env = dict(base_env, CUDA_VISIBLE_DEVICES=",".join(uuids),
                              JPEERS_MODEL_PATH=model["model_path"],
                              JPEERS_MODEL_REVISION=model["model_revision"],
                              JPEERS_LENS_PATH=model["lens_path"])
            server_command = [sys.executable, str(source / "scripts/serve_sharded.py"),
                              "--root", str(root), "--port", str(args.port)]
            with (root / "server.log").open("a") as log:
                server = subprocess.Popen(server_command, cwd=source, env=server_env,
                                          stdin=subprocess.DEVNULL, stdout=log,
                                          stderr=subprocess.STDOUT, start_new_session=True)
            update(status="starting_model", physical_gpus=selected, gpu_uuids=uuids,
                   server_pid=server.pid, server_command=server_command, port=args.port)
            url = f"http://127.0.0.1:{args.port}"
            deadline = time.monotonic() + 1800
            while True:
                if server.poll() is not None:
                    raise RuntimeError(f"Owned Qwen server exited {server.returncode}")
                try:
                    with urllib.request.urlopen(url + "/health", timeout=5) as reply:
                        health = json.load(reply)
                    if health.get("status") != "ok":
                        raise RuntimeError(f"Unexpected server health: {health}")
                    with urllib.request.urlopen(url + "/info", timeout=5) as reply:
                        info = json.load(reply)
                    if (info.get("hf_id") != "Qwen/Qwen3-32B" or info.get("n_layers") != 64
                            or info.get("d_model") != 5120):
                        raise RuntimeError("Owned server does not advertise the expected Qwen model")
                    atomic_json(root / "endpoint-identity.json", info)
                    break
                except urllib.error.HTTPError as exc:
                    if exc.code != 503:
                        raise
                    detail = exc.read().decode(errors="replace")
                    if '"error"' in detail:
                        raise RuntimeError(f"Qwen loading failed: {detail}") from exc
                except (urllib.error.URLError, TimeoutError):
                    pass
                if time.monotonic() > deadline:
                    raise TimeoutError("Qwen loading exceeded its startup limit")
                time.sleep(3)
            worker_command = [sys.executable, str(source / "scripts/run_causal_handoff.py"),
                              "--config", str(source / "configs/causal_handoff_32b.json"),
                              "--base-url", url, "--output-dir", str(root / "results"),
                              "--stage", "all"]
            if args.resume:
                worker_command.append("--resume")
            with (root / "worker.log").open("a") as log:
                worker = subprocess.Popen(worker_command, cwd=source,
                                          env=dict(base_env, CUDA_VISIBLE_DEVICES=""),
                                          stdin=subprocess.DEVNULL, stdout=log,
                                          stderr=subprocess.STDOUT, start_new_session=True)
            update(status="running", worker_pid=worker.pid, worker_command=worker_command)
            deadline = time.monotonic() + args.run_timeout
            while worker.poll() is None:
                if server.poll() is not None:
                    raise RuntimeError(f"Owned Qwen server exited {server.returncode}")
                if time.monotonic() > deadline:
                    raise TimeoutError("Handoff study exceeded its runtime limit")
                time.sleep(3)
            report_path = root / "results/status.json"
            report = json.loads(report_path.read_text()) if report_path.is_file() else {}
            result = report.get("status")
            terminal = "failed"
            if result == "discovery_failed":
                terminal = "discovery_failed"
            elif worker.returncode == 0 and result == "completed":
                terminal = "completed"
            update(status=terminal,
                   worker_exit_code=worker.returncode, worker_status=report)
        except BaseException as exc:
            update(status="failed", error=repr(exc))
        finally:
            for name, process in (("worker", worker), ("server", server)):
                try:
                    stop_owned_group(process)
                    update(**{f"{name}_stopped_at": now()})
                except Exception as exc:
                    update(status="failed", **{f"{name}_cleanup_error": repr(exc)})
            update(finished_at=now())
    return 0 if state["status"] in {"completed", "discovery_failed"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
