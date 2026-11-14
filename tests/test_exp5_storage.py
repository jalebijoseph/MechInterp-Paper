"""Bounded, lossless artifacts for long Experiment 5 trajectories."""
import json
from pathlib import Path
import shutil
import sys
import unittest
import uuid

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from jpeers.exp5.storage import ArtifactStore, BudgetExceeded


class StorageTests(unittest.TestCase):
    def folder(self):
        path = (ROOT / "runs" / ("exp5-storage-test-" + uuid.uuid4().hex)).resolve()
        self.assertTrue(path.is_relative_to((ROOT / "runs").resolve()))
        self.addCleanup(lambda: shutil.rmtree(path) if path.exists() else None)
        return path

    def limits(self):
        return {"total_bytes": 1000000, "raw_development_bytes": 100000,
                "text_bytes": 100000, "shared_bytes": 100000,
                "compact_bytes": 500000, "manifest_reserve_bytes": 10000}

    def test_lossless_histories_and_strict_json(self):
        store = ArtifactStore(self.folder(), self.limits(), metadata={"synthetic": True})
        tokens = [151644, 123, 0, 123, 151645]
        first = store.history(tokens)
        spent = store.bytes_written
        self.assertEqual(first, store.history(tokens))
        self.assertEqual(store.bytes_written, spent)
        self.assertEqual(store.read_history(first), tokens)
        store.append({"type": "readout", "prediction": np.array([.1, np.nan]), "x": np.int64(2)})
        row = json.loads((store.root / "records.jsonl").read_text(encoding="utf-8"))
        self.assertEqual(row["prediction"], [.1, None])
        manifest = store.finish("complete")
        self.assertEqual(manifest["status"], "complete")
        self.assertEqual(manifest["total_bytes"], sum(p.stat().st_size for p in store.root.rglob("*") if p.is_file()))
        self.assertFalse(manifest.get("ready_for_execution", False))

    def test_budget_stop_preserves_receipts_and_manifest(self):
        limits = self.limits()
        limits["text_bytes"] = 5
        store = ArtifactStore(self.folder(), limits)
        store.write("text", "keep.txt", b"keep")
        with self.assertRaises(BudgetExceeded):
            store.write("text", "extra.txt", b"xx")
        self.assertFalse((store.root / "extra.txt").exists())
        self.assertEqual((store.root / "keep.txt").read_bytes(), b"keep")
        result = store.finish("incomplete", reason="storage limit")
        self.assertEqual(result["status"], "incomplete")
        self.assertEqual(result["reason"], "storage limit")

    def test_existing_outputs_and_path_escape_are_rejected(self):
        path = self.folder()
        store = ArtifactStore(path, self.limits())
        with self.assertRaises(FileExistsError):
            ArtifactStore(path, self.limits())
        for filename in ["../outside.txt", str(ROOT / "outside.txt"), "manifest.json"]:
            with self.assertRaises(ValueError):
                store.write("text", filename, b"bad")

    def test_numeric_archives_reject_pickle_and_round_trip(self):
        store = ArtifactStore(self.folder(), self.limits())
        store.npz("raw_development", "raw/example.npz", captures=np.arange(12).reshape(3, 4))
        with np.load(store.root / "raw/example.npz", allow_pickle=False) as archive:
            self.assertEqual(archive["captures"].shape, (3, 4))
        with self.assertRaises(ValueError):
            store.npz("raw_development", "raw/bad.npz", values=np.array([object()]))


if __name__ == "__main__":
    unittest.main()
