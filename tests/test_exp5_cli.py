"""Command-line handoff and explicit real-model execution boundaries."""
import json
import contextlib
import io
from pathlib import Path
import runpy
import subprocess
import sys
import types
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]


class CLITests(unittest.TestCase):
    def invoke(self, *args):
        return subprocess.run([sys.executable, str(ROOT / "scripts/run_exp5.py"), *args],
                              cwd=ROOT, capture_output=True, text=True)

    def test_help_lists_complete_workflow(self):
        result = self.invoke("--help")
        self.assertEqual(result.returncode, 0, result.stderr)
        for command in ["preflight", "simulate", "validate-runtime", "develop", "freeze", "evaluate", "report"]:
            self.assertIn(command, result.stdout)

    def test_preflight_is_model_free(self):
        result = self.invoke("preflight")
        self.assertEqual(result.returncode, 0, result.stderr)
        data = json.loads(result.stdout)
        self.assertEqual(data["model_calls"], 0)
        self.assertFalse(data["ready_for_execution"])
        self.assertIn("storage", data)

    def test_real_commands_require_explicit_execution_before_loading_weights(self):
        for command, extra in [("develop", ["--out", "unused-exp5-output"]),
                               ("validate-runtime", [])]:
            result = self.invoke(command, *extra)
            self.assertEqual(result.returncode, 2, result.stderr)
            self.assertIn("--execute", result.stderr)
            self.assertNotIn("locally", result.stderr)
        self.assertFalse((ROOT / "unused-exp5-output").exists())

    def test_simulate_dispatches_both_phases_as_explicit_synthetic_execution(self):
        calls = []
        runner = types.ModuleType("jpeers.exp5.runner")
        runner.smoke_config = lambda cfg: cfg
        def phase(name):
            def invoke(*args, **kwargs):
                calls.append((name, kwargs))
                return {"status": "complete", "synthetic": True}
            return invoke
        runner.develop, runner.freeze, runner.evaluate = (phase(n) for n in ("develop", "freeze", "evaluate"))
        script = runpy.run_path(str(ROOT / "scripts/run_exp5.py"))
        with mock.patch.dict(sys.modules, {"jpeers.exp5.runner": runner}), \
             mock.patch.object(sys, "argv", ["run_exp5.py", "simulate", "--out", str(ROOT / "runs/cli-dispatch-only")]), \
             contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(script["main"](), 0)
        self.assertTrue(next(k for n, k in calls if n == "develop").get("execute"))
        evaluation = next(k for n, k in calls if n == "evaluate")
        self.assertTrue(evaluation.get("execute"))
        self.assertTrue(evaluation.get("simulation"))

    def test_failed_storage_preflight_prevents_weight_loading(self):
        sys.path.insert(0, str(ROOT / "src"))
        from jpeers.exp5.backend import HFBackend
        runner = types.ModuleType("jpeers.exp5.runner")
        runner.preflight = lambda *a, **k: {"checks": {"estimated_storage_within_allowance": False,
            "physical_space_for_estimate": True, "output_directory_available": True},
            "forecast": {"raw_estimate_within_allowance": True}, "model_calls": 0}
        script = runpy.run_path(str(ROOT / "scripts/run_exp5.py"))
        with mock.patch.dict(sys.modules, {"jpeers.exp5.runner": runner}), \
             mock.patch.object(HFBackend, "from_pretrained", side_effect=AssertionError("weights loaded")) as loader, \
             mock.patch.object(sys, "argv", ["run_exp5.py", "develop", "--execute", "--max-model-calls", "1000", "--out", "never-created"]), \
             contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as raised:
                script["main"]()
            self.assertEqual(raised.exception.code, 2)
            loader.assert_not_called()

    def test_failed_combined_study_forecast_prevents_weight_loading(self):
        sys.path.insert(0, str(ROOT / "src"))
        from jpeers.exp5.backend import HFBackend
        runner = types.ModuleType("jpeers.exp5.runner")
        runner.preflight = lambda *a, **k: {"checks": {"estimated_storage_within_allowance": True,
            "planned_study_storage_within_allowance": False, "physical_space_for_estimate": True,
            "output_directory_available": True}, "forecast": {"raw_estimate_within_allowance": True}}
        script = runpy.run_path(str(ROOT / "scripts/run_exp5.py"))
        with mock.patch.dict(sys.modules, {"jpeers.exp5.runner": runner}), \
             mock.patch.object(HFBackend, "from_pretrained", side_effect=AssertionError("weights loaded")) as loader, \
             mock.patch.object(sys, "argv", ["run_exp5.py", "develop", "--execute", "--max-model-calls", "1000", "--out", "never-created"]), \
             contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as raised:
                script["main"]()
            self.assertEqual(raised.exception.code, 2)
            loader.assert_not_called()


if __name__ == "__main__":
    unittest.main()
