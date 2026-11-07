"""CPU-only lifecycle and frozen-input checks for the handoff controller."""
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import MagicMock, patch


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
spec = importlib.util.spec_from_file_location("queue_causal_handoff", SCRIPTS / "queue_causal_handoff.py")
queue = importlib.util.module_from_spec(spec)
spec.loader.exec_module(queue)


class CausalQueueTests(unittest.TestCase):
    def fixture(self, directory):
        root = directory / "study"
        source = root / "source"
        (source / "configs").mkdir(parents=True)
        (source / "entry.py").write_text("# frozen test source\n")
        config = source / "configs/causal_handoff_32b.json"
        config.write_text("{}")
        preflight = {"status": "passed", "source_files": {
            "entry.py": hashlib.sha256((source / "entry.py").read_bytes()).hexdigest()},
            "config_sha256": hashlib.sha256(config.read_bytes()).hexdigest()}
        (root / "preflight.json").write_text(json.dumps(preflight))
        checkpoint = directory / queue.MODEL_REVISION
        checkpoint.mkdir()
        lens = directory / "lens.pt"
        lens.write_bytes(b"test lens")
        provenance = directory / "qwen.json"
        model = {"model": "Qwen/Qwen3-32B", "model_revision": queue.MODEL_REVISION,
                 "lens_revision": queue.LENS_REVISION, "model_path": str(checkpoint),
                 "lens_path": str(lens), "hf_home": str(directory / "cache")}
        provenance.write_text(json.dumps(model))
        grader = directory / "grader"
        grader.mkdir()
        (grader / "manager-status.json").write_text(json.dumps({
            "status": "incomplete", "server_stopped_at": "recorded"}))
        return root, provenance, grader

    def test_predecessor_requires_terminal_status_and_cleanup(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.assertFalse(queue.grader_released(root)[0])
            path = root / "manager-status.json"
            for value in ({"status": "running", "server_stopped_at": "x"},
                          {"status": "incomplete"},
                          {"status": "completed", "server_stopped_at": "x",
                           "server_cleanup_error": "still owns workers"}):
                path.write_text(json.dumps(value))
                self.assertFalse(queue.grader_released(root)[0])
            path.write_text(json.dumps({"status": "incomplete", "server_stopped_at": "x"}))
            self.assertTrue(queue.grader_released(root)[0])

    def test_changed_sources_config_and_model_fail_before_gpu_use(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, provenance, _ = self.fixture(Path(tmp))
            queue.verify_preflight(root, provenance)
            source = root / "source/entry.py"
            original = source.read_text()
            source.write_text("changed")
            with self.assertRaisesRegex(ValueError, "source"):
                queue.verify_preflight(root, provenance)
            source.write_text(original)
            config = root / "source/configs/causal_handoff_32b.json"
            config.write_text('{"changed":true}')
            with self.assertRaisesRegex(ValueError, "configuration"):
                queue.verify_preflight(root, provenance)
            config.write_text("{}")
            model = json.loads(provenance.read_text())
            model["model_revision"] = "another revision"
            provenance.write_text(json.dumps(model))
            with self.assertRaisesRegex(ValueError, "pinned"):
                queue.verify_preflight(root, provenance)

    def test_controller_uses_selected_uuids_and_stops_only_owned_groups(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, provenance, grader = self.fixture(Path(tmp))
            (root / "results").mkdir()
            (root / "results/status.json").write_text('{"status":"completed"}')
            inventory = [{"index": i, "uuid": f"GPU-{i}", "memory_used_mib": 0,
                          "has_compute_process": False} for i in range(4)]
            server = MagicMock(pid=1001)
            server.poll.return_value = None
            worker = MagicMock(pid=1002, returncode=0)
            worker.poll.return_value = 0
            replies = [io.BytesIO(b'{"status":"ok"}'), io.BytesIO(json.dumps({
                "hf_id": "Qwen/Qwen3-32B", "n_layers": 64, "d_model": 5120}).encode())]
            argv = ["queue", "--root", str(root), "--grader-root", str(grader),
                    "--qwen-provenance", str(provenance)]
            with patch.object(sys, "argv", argv), \
                    patch.object(queue, "gpu_snapshot", return_value=([0, 2], inventory)), \
                    patch.object(queue, "assert_port_free"), \
                    patch.object(queue.subprocess, "Popen", side_effect=[server, worker]) as launch, \
                    patch.object(queue.urllib.request, "urlopen", side_effect=replies), \
                    patch.object(queue, "stop_owned_group") as stop:
                self.assertEqual(queue.main(), 0)
            server_env = launch.call_args_list[0].kwargs["env"]
            worker_env = launch.call_args_list[1].kwargs["env"]
            self.assertEqual(server_env["CUDA_VISIBLE_DEVICES"], "GPU-0,GPU-2")
            self.assertEqual(worker_env["CUDA_VISIBLE_DEVICES"], "")
            self.assertEqual([c.args[0] for c in stop.call_args_list], [worker, server])
            state = json.loads((root / "manager-status.json").read_text())
            self.assertEqual(state["status"], "completed")
            self.assertEqual(state["physical_gpus"], [0, 2])
            self.assertIn("server_stopped_at", state)

    def test_failed_loading_cleans_up_without_launching_worker(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, provenance, grader = self.fixture(Path(tmp))
            inventory = [{"index": i, "uuid": f"GPU-{i}", "memory_used_mib": 0,
                          "has_compute_process": False} for i in range(2)]
            server = MagicMock(pid=1001, returncode=9)
            server.poll.return_value = 9
            argv = ["queue", "--root", str(root), "--grader-root", str(grader),
                    "--qwen-provenance", str(provenance)]
            with patch.object(sys, "argv", argv), \
                    patch.object(queue, "gpu_snapshot", return_value=([0, 1], inventory)), \
                    patch.object(queue, "assert_port_free"), \
                    patch.object(queue.subprocess, "Popen", return_value=server) as launch, \
                    patch.object(queue, "stop_owned_group") as stop:
                self.assertEqual(queue.main(), 1)
            self.assertEqual(launch.call_count, 1)
            self.assertEqual([c.args[0] for c in stop.call_args_list], [None, server])
            state = json.loads((root / "manager-status.json").read_text())
            self.assertEqual(state["status"], "failed")
            self.assertIn("server_stopped_at", state)

    def test_zero_exit_does_not_hide_discovery_failure_or_missing_status(self):
        for worker_status, expected in (("discovery_failed", "discovery_failed"),
                                        ("failed", "failed"), ("unexpected", "failed")):
            with self.subTest(worker_status=worker_status), tempfile.TemporaryDirectory() as tmp:
                root, provenance, grader = self.fixture(Path(tmp))
                (root / "results").mkdir()
                (root / "results/status.json").write_text(json.dumps({"status": worker_status}))
                inventory = [{"index": i, "uuid": f"GPU-{i}", "memory_used_mib": 0,
                              "has_compute_process": False} for i in range(2)]
                server = MagicMock(pid=1001)
                server.poll.return_value = None
                worker = MagicMock(pid=1002, returncode=0)
                worker.poll.return_value = 0
                replies = [io.BytesIO(b'{"status":"ok"}'), io.BytesIO(json.dumps({
                    "hf_id": "Qwen/Qwen3-32B", "n_layers": 64, "d_model": 5120}).encode())]
                argv = ["queue", "--root", str(root), "--grader-root", str(grader),
                        "--qwen-provenance", str(provenance)]
                with patch.object(sys, "argv", argv), \
                        patch.object(queue, "gpu_snapshot", return_value=([0, 1], inventory)), \
                        patch.object(queue, "assert_port_free"), \
                        patch.object(queue.subprocess, "Popen", side_effect=[server, worker]), \
                        patch.object(queue.urllib.request, "urlopen", side_effect=replies), \
                        patch.object(queue, "stop_owned_group"):
                    self.assertEqual(queue.main(), 0 if expected == "discovery_failed" else 1)
                state = json.loads((root / "manager-status.json").read_text())
                self.assertEqual(state["status"], expected)


if __name__ == "__main__":
    unittest.main()
