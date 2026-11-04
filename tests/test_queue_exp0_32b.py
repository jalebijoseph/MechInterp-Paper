"""CPU-only tests: no model download, GPU access, or existing server mutation."""
import importlib.util
import json
from pathlib import Path
import signal
import socket
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/queue_exp0_32b.py"
SPEC = importlib.util.spec_from_file_location("queue_exp0_32b", SCRIPT)
queue = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(queue)


class QueueTests(unittest.TestCase):
    def test_prerequisite_requires_both_successes_and_shutdown(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.assertEqual(queue.prerequisite_state(root)[0], "waiting")
            queue.atomic_json(root / "manager-status.json", {"status": "completed"})
            queue.atomic_json(root / "source/runs/full-suite/manifest.json", {"status": "completed"})
            self.assertEqual(queue.prerequisite_state(root)[0], "waiting")
            queue.atomic_json(root / "manager-status.json", {"status": "completed", "servers_stopped_at": "timestamp"})
            self.assertEqual(queue.prerequisite_state(root)[0], "ready")
            queue.atomic_json(root / "source/runs/full-suite/manifest.json", {"status": "incomplete"})
            self.assertEqual(queue.prerequisite_state(root)[0], "failed")

    def test_gpu_selection_uses_only_idle_allowed_devices(self):
        inventory = "0, GPU-a, 0\n1, GPU-b, 0\n2, GPU-c, 255\n3, GPU-d, 256\n4, GPU-e, 0\n"
        selected, details = queue.select_idle_gpus(inventory, "GPU-b, 123\n")
        self.assertEqual(selected, [0, 2])
        self.assertEqual([row["index"] for row in details], [0, 1, 2, 3])
        selected, _ = queue.select_idle_gpus(inventory, "GPU-a, 124\nGPU-b, 123\n")
        self.assertEqual(selected, [])
        with self.assertRaises(ValueError):
            queue.select_idle_gpus(inventory, "unrecognized output")

    def test_config_retains_full_original_exp0(self):
        config = queue.exp0_config(Path("/tmp/test32b"))
        self.assertEqual(config["model"], "qwen3-32b")
        self.assertEqual(config["n_items"], 20)
        self.assertEqual(config["framings"], ["people", "agents"])
        self.assertTrue(config["include_hitom"])
        self.assertEqual(config["hitom_items"], 60)
        self.assertEqual(config["hitom_orders"], [0, 1, 2, 3])
        self.assertEqual(config["max_new_tokens"], 6)
        self.assertEqual(queue.MODEL_REVISION, "9216db5781bf21249d130ec9da846c4624c16137")
        self.assertEqual(queue.LENS_REVISION, "e166deb5ad540932100ef803c8c9bcc2a5453694")

    def fixture_results(self, root):
        items = [{"item": f"fb{i}{'c' if not false_belief else ''}", "framing": framing,
                  "false_belief": false_belief}
                 for framing in ("people", "agents") for false_belief in (False, True)
                 for i in range(20)]
        items += [{"item": f"hitom-{i}", "framing": f"hitom-o{i % 4}", "false_belief": i % 4 > 0}
                  for i in range(60)]
        queue.atomic_json(root / "summary.json", {"rows": [{"n": 140}], "items": items,
                                                  "hitom_skipped_not_single_token": []})
        turns = []
        for i in range(140):
            filename = f"traces/{i:05d}.npz"
            trace = root / filename
            trace.parent.mkdir(exist_ok=True)
            trace.write_bytes(b"trace-fixture")
            turns.append({"model": "qwen3-32b", "trace_file": filename})
        (root / "turns.jsonl").write_text("\n".join(map(json.dumps, turns)) + "\n")

    def test_exact_counts_and_saved_trace_validation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.fixture_results(root)
            result = queue.validate_results(root)
            self.assertEqual(result["errors"], [])
            self.assertEqual((result["core_items"], result["hitom_items"], result["turns"], result["traces"]),
                             (80, 60, 140, 140))
            (root / "traces/00000.npz").unlink()
            self.assertTrue(any("Missing saved trace" in error for error in queue.validate_results(root)["errors"]))

    def test_skipped_or_duplicate_items_cannot_complete(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.fixture_results(root)
            summary = queue.read_json(root / "summary.json")
            summary["items"][-1] = summary["items"][-2]
            summary["hitom_skipped_not_single_token"] = ["skipped"]
            queue.atomic_json(root / "summary.json", summary)
            errors = queue.validate_results(root)["errors"]
            self.assertTrue(any("duplicate" in error for error in errors))
            self.assertTrue(any("exclusions" in error for error in errors))

    def test_cleanup_stops_an_actual_owned_subprocess(self):
        child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"],
                                 start_new_session=True)
        try:
            queue.terminate_process_group(child, grace=2)
            self.assertIsNotNone(child.poll())
            self.assertLess(child.returncode, 0)
        finally:
            if child.poll() is None:
                child.kill()
                child.wait()

    def test_cleanup_never_signals_completed_processes(self):
        child = subprocess.Popen([sys.executable, "-c", "pass"], start_new_session=True)
        child.wait(timeout=5)
        with patch.object(queue.os, "killpg") as killpg:
            queue.terminate_process_group(child)
            killpg.assert_not_called()

    def test_occupied_port_is_rejected_before_server_launch(self):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
            try:
                listener.bind(("127.0.0.1", 0))
            except PermissionError:
                self.skipTest("This test environment disallows binding localhost sockets")
            listener.listen()
            with self.assertRaisesRegex(RuntimeError, "another server"):
                queue.assert_port_free(listener.getsockname()[1])

    def test_failed_prerequisite_never_downloads_or_launches_server(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            prerequisite, root = base / "eight", base / "thirtytwo"
            queue.atomic_json(prerequisite / "manager-status.json", {"status": "failed"})
            result = subprocess.run([sys.executable, str(SCRIPT), "--root", str(root),
                                     "--prerequisite-root", str(prerequisite)],
                                    capture_output=True, text=True, timeout=10)
            self.assertEqual(result.returncode, 1)
            state = queue.read_json(root / "queue-status.json")
            self.assertEqual(state["status"], "failed")
            self.assertIn("Prerequisite manager is failed", state["error"])
            self.assertFalse(state["gpu_allocation_started"])
            self.assertFalse(state["server"]["started"])
            self.assertNotIn("download", state)
            self.assertFalse((root / "model-provenance.json").exists())

    def test_waiter_is_cpu_only_and_interruptible(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "thirtytwo"
            process = subprocess.Popen([sys.executable, str(SCRIPT), "--root", str(root),
                                        "--prerequisite-root", str(Path(tmp) / "eight"),
                                        "--poll-seconds", "0.05"], stdout=subprocess.PIPE,
                                       stderr=subprocess.PIPE, text=True, start_new_session=True)
            try:
                deadline = time.monotonic() + 5
                state = {}
                while time.monotonic() < deadline:
                    try:
                        state = queue.read_json(root / "queue-status.json")
                    except OSError:
                        pass
                    if state.get("status") == "waiting_for_8b":
                        break
                    time.sleep(0.02)
                self.assertEqual(state["status"], "waiting_for_8b")
                self.assertEqual(state["selected_gpus"], [])
                self.assertFalse(state["gpu_allocation_started"])
                self.assertNotIn("download", state)
                self.assertFalse(state["server"]["started"])
                process.send_signal(signal.SIGTERM)
                process.communicate(timeout=5)
                self.assertEqual(process.returncode, 130)
                self.assertEqual(queue.read_json(root / "queue-status.json")["status"], "interrupted")
            finally:
                if process.poll() is None:
                    queue.terminate_process_group(process, grace=2)
                process.communicate()


if __name__ == "__main__":
    unittest.main()
