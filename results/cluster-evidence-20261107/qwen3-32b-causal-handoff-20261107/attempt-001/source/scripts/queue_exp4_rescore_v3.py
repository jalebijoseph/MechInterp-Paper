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
NEMOTRON_PROFILE = "nemotron-3.5-super-vl"
NEMOTRON_MODEL = "nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4"


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
    parser.add_argument("--request-timeout-seconds", type=int, default=600)
    parser.add_argument("--model-profile", choices=("gpt-oss", NEMOTRON_PROFILE), default="gpt-oss")
    parser.add_argument("--fallback-from", type=Path)
    parser.add_argument("--fallback-source-sha256")
    parser.add_argument("--input-summary", type=Path,
                        help="Select the corrected Exp2 worker for this frozen summary")
    parser.add_argument("--input-summary-sha256")
    parser.add_argument("--calibration-source", type=Path,
                        help="Read-only grading directory containing all 84 prior calibration receipts")
    parser.add_argument("--calibration-source-sha256", help="SHA-256 pin of prior calibration run.json")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.request_timeout_seconds <= 3600:
        parser.error("--request-timeout-seconds must be between 1 and 3600")
    if (args.fallback_from is None) != (args.fallback_source_sha256 is None):
        parser.error("--fallback-from and --fallback-source-sha256 must be supplied together")
    nemotron = args.model_profile == NEMOTRON_PROFILE
    if nemotron and args.fallback_from is not None:
        parser.error("GPT-OSS fallback receipts cannot be used for Nemotron")
    exp2_options = (args.input_summary, args.input_summary_sha256,
                    args.calibration_source, args.calibration_source_sha256)
    if any(value is not None for value in exp2_options) and not all(value is not None for value in exp2_options):
        parser.error("Exp2 requires both input-summary and calibration-source paths and SHA-256 pins")
    exp2 = args.input_summary is not None
    if exp2 and not nemotron:
        parser.error("The corrected Exp2 worker requires the Nemotron model profile")
    model_id = NEMOTRON_MODEL if nemotron else MODEL
    served_model = f"{NEMOTRON_PROFILE}-exp4-v3" if nemotron else SERVED_MODEL
    root = args.root.resolve()
    source = Path(__file__).resolve().parents[1]
    fixture = source / "src/jpeers/data/fact_grader_holdout_v3.json"
    runner = source / ("scripts/run_exp2_rescore_v3.py" if exp2 else "scripts/run_exp4_rescore_v3.py")
    if hashlib.sha256(fixture.read_bytes()).hexdigest() != args.fixture_sha256:
        raise ValueError("Holdout differs from its frozen hash")
    if not runner.is_file():
        raise ValueError("Scoring worker is missing")
    if exp2:
        if hashlib.sha256(args.input_summary.read_bytes()).hexdigest() != args.input_summary_sha256:
            raise ValueError("Saved Exp2 summary differs from its frozen hash")
        if hashlib.sha256((args.calibration_source / "run.json").read_bytes()).hexdigest() != args.calibration_source_sha256:
            raise ValueError("Prior calibration provenance differs from its frozen hash")
    root.mkdir(parents=True, exist_ok=True)
    lock = (root / ".manager.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    status_path = root / "manager-status.json"
    if status_path.exists() and not args.resume:
        raise ValueError("A manager attempt is already recorded; use --resume to recover it")
    state = {"pid": os.getpid(), "started_at": now(), "status": "waiting",
             "model": model_id, "model_revision": args.revision,
             "holdout_sha256": args.fixture_sha256, "predecessor": str(args.predecessor)}
    if exp2:
        state.update(kind="exp2_v2", input_summary=str(args.input_summary),
                     input_summary_sha256=args.input_summary_sha256,
                     calibration_source=str(args.calibration_source),
                     calibration_source_sha256=args.calibration_source_sha256,
                     worker_script_sha256=hashlib.sha256(runner.read_bytes()).hexdigest())
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
            if model and (model.get("model") != model_id or model.get("revision", args.revision) != args.revision):
                raise ValueError("Downloaded model does not match the pinned grader")
            released, predecessor = predecessor_released(args.predecessor)
            if released and predecessor["manifest_status"] != "completed":
                raise RuntimeError("Original suite did not complete")
            pairs = gpu_pairs() if released and model.get("status") in ("completed", "cached") else []
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
                   "--served-model-name", served_model, "--host", "127.0.0.1", "--port", str(args.port),
                   "--tensor-parallel-size", str(len(devices)), "--max-model-len", "16384" if nemotron else "8192",
                   "--max-num-seqs", str(args.concurrency), "--gpu-memory-utilization", "0.86",
                   "--moe-backend", "marlin",
                   "--generation-config", "vllm", "--enforce-eager"]
        if nemotron:
            command += ["--trust-remote-code", "--language-model-only", "--reasoning-parser", "nemotron_v3",
                        "--chat-template-content-format", "string", "--attention-backend", "TRITON_ATTN"]
        else:
            reasoning_config = json.dumps({
                "reasoning_start_str": "<|channel|>analysis<|message|>",
                "reasoning_end_str": "<|end|><|start|>assistant<|channel|>final<|message|>",
            })
            command += ["--reasoning-parser", "openai_gptoss", "--reasoning-config", reasoning_config]
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=",".join(gpu["uuid"] for gpu in devices),
                   PYTHONPATH=str(source / "src"), PYTHONUNBUFFERED="1", HF_HUB_OFFLINE="1")
        if nemotron:
            env["VLLM_USE_FLASHINFER_SAMPLER"] = "0"
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
                if served_model in advertised:
                    break
                raise RuntimeError(f"Unexpected server model IDs: {advertised}")
            except (urllib.error.URLError, TimeoutError):
                if time.monotonic() > deadline:
                    raise TimeoutError("Grader server startup timed out")
                time.sleep(3)

        worker_inputs = (["--input-summary", str(args.input_summary),
                          "--input-summary-sha256", args.input_summary_sha256,
                          "--calibration-source", str(args.calibration_source),
                          "--calibration-source-sha256", args.calibration_source_sha256] if exp2 else
                         ["--input-root", str(args.predecessor / "source/runs/full-suite")])
        worker_command = [sys.executable, str(runner), *worker_inputs,
                          "--output-dir", str(root / "grading"), "--base-url", base_url,
                          "--model", served_model, "--model-revision", args.revision,
                          "--fixture", str(fixture), "--fixture-sha256", args.fixture_sha256,
                          "--concurrency", str(args.concurrency)]
        if nemotron:
            worker_command += ["--generation-profile", NEMOTRON_PROFILE]
        if args.fallback_from is not None:
            worker_command += ["--fallback-from", str(args.fallback_from),
                               "--fallback-source-sha256", args.fallback_source_sha256]
        if args.resume and (root / "grading").exists():
            worker_command.append("--resume")
        if exp2 or args.request_timeout_seconds != 600:
            transport_entrypoint = (
                ("import sys\n" + f"sys.path.insert(0, {str(runner.parent)!r})\n" if exp2 else "") +
                "import httpx, runpy\n"
                f"runner = runpy.run_path({str(runner)!r})\n"
                "args = runner['parser']().parse_args()\n"
                f"timeout = httpx.Timeout({args.request_timeout_seconds}, connect=15)\n"
                "with httpx.Client(timeout=timeout, trust_env=False) as client:\n"
                "    raise SystemExit(runner['run'](args, client))\n"
            )
            worker_command = [sys.executable, "-c", transport_entrypoint, *worker_command[2:]]
        with (root / "worker.log").open("a") as log:
            worker = subprocess.Popen(worker_command, cwd=source, env=dict(env, CUDA_VISIBLE_DEVICES=""),
                                      stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                                      start_new_session=True)
        update(status="running", worker_pid=worker.pid, worker_command=worker_command,
               request_timeout_seconds=args.request_timeout_seconds)
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
