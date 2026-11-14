"""Exact result budgets, compact records and deduplicated native token prefixes."""
from __future__ import annotations

import gzip
import hashlib
import io
import json
import math
from pathlib import Path

import numpy as np


class BudgetExceeded(RuntimeError):
    """A declared resource allowance would be exceeded; never discard required cases."""


def jsonable(value):
    if isinstance(value, np.ndarray):
        return jsonable(value.tolist())
    if isinstance(value, np.generic):
        return jsonable(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    return value


def json_bytes(value) -> bytes:
    return (json.dumps(jsonable(value), ensure_ascii=True, sort_keys=True,
                       separators=(",", ":"), allow_nan=False) + "\n").encode()


class ArtifactStore:
    def __init__(self, root: Path, limits: dict, *, metadata: dict | None = None):
        self.root = Path(root).resolve()
        self.limits = dict(limits)
        self.metadata = dict(metadata or {})
        self.used = dict.fromkeys(["raw_development", "text", "shared", "compact"], 0)
        self.files = {}
        self.record_count = 0
        if not 0 < limits["total_bytes"] <= 1073741824:
            raise ValueError("Artifact allowance must be positive and no larger than 1 GiB")
        if not 0 < limits["manifest_reserve_bytes"] < limits["total_bytes"]:
            raise ValueError("A final manifest needs a reserved allowance")
        self.root.mkdir(parents=True, exist_ok=False)

    @property
    def bytes_written(self):
        return sum(self.used.values())

    def _path(self, name):
        relative = Path(name)
        path = (self.root / relative).resolve()
        if relative.is_absolute() or not path.is_relative_to(self.root) or relative.name == "manifest.json":
            raise ValueError("Artifact paths must remain inside the run and cannot replace its manifest")
        return path

    def _check(self, category, delta):
        if category not in self.used:
            raise ValueError(f"Unknown artifact category: {category}")
        if self.used[category] + delta > self.limits[category + "_bytes"]:
            raise BudgetExceeded(f"{category} artifact allowance exceeded")
        if self.bytes_written + delta + self.limits["manifest_reserve_bytes"] > self.limits["total_bytes"]:
            raise BudgetExceeded("Total result allowance exceeded")

    def write(self, category: str, name: str, data: bytes):
        path = self._path(name)
        prior = self.files.get(name, (category, 0))
        if prior[0] != category:
            raise ValueError("An existing artifact cannot change budget categories")
        delta = len(data) - prior[1]
        self._check(category, delta)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        self.used[category] += delta
        self.files[name] = (category, len(data))
        return {"path": name, "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}

    def json(self, category, name, value):
        return self.write(category, name, json_bytes(value))

    def npz(self, category, name, **arrays):
        arrays = {key: np.asarray(value) for key, value in arrays.items()}
        if any(value.dtype.hasobject for value in arrays.values()):
            raise ValueError("Object/pickle arrays are not allowed")
        stream = io.BytesIO()
        np.savez_compressed(stream, **arrays)
        return self.write(category, name, stream.getvalue())

    def append(self, record):
        line = json_bytes(dict(record, record_index=self.record_count))
        self._check("compact", len(line))
        path = self.root / "records.jsonl"
        with path.open("ab") as handle:
            handle.write(line)
        previous = self.files.get("records.jsonl", ("compact", 0))[1]
        self.files["records.jsonl"] = ("compact", previous + len(line))
        self.used["compact"] += len(line)
        self.record_count += 1

    def history(self, tokens):
        values = np.asarray(tokens)
        if values.ndim != 1 or not np.issubdtype(values.dtype, np.integer) or np.any(values < 0) or np.any(values > 2 ** 32 - 1):
            raise ValueError("Native history must be a one-dimensional unsigned token sequence")
        raw = values.astype("<u4").tobytes()
        digest = hashlib.sha256(raw).hexdigest()
        name = f"histories/{digest}.bin.gz"
        if name not in self.files:
            self.write("text", name, gzip.compress(raw, mtime=0))
        return {"path": name, "sha256": digest, "length": len(tokens)}

    def read_history(self, reference):
        raw = gzip.decompress(self._path(reference["path"]).read_bytes())
        if hashlib.sha256(raw).hexdigest() != reference["sha256"]:
            raise ValueError("History content hash mismatch")
        values = np.frombuffer(raw, dtype="<u4").tolist()
        if len(values) != reference["length"]:
            raise ValueError("History length mismatch")
        return values

    def finish(self, status: str, **metadata):
        index = {name: {"category": category, "bytes": size,
                        "sha256": hashlib.sha256((self.root / name).read_bytes()).hexdigest()}
                 for name, (category, size) in self.files.items() if name != "artifact-index.json"}
        index_complete = True
        try:
            self.json("compact", "artifact-index.json", index)
        except BudgetExceeded:
            index_complete = False
            status = "incomplete"
        manifest = dict(self.metadata, **metadata)
        manifest.update(status=status, record_count=self.record_count, artifact_index_complete=index_complete,
                        category_bytes=self.used, total_bytes=0, ready_for_execution=False)
        for _ in range(8):
            data = json_bytes(manifest)
            expected = self.bytes_written + len(data)
            if expected == manifest["total_bytes"]:
                break
            manifest["total_bytes"] = expected
        data = json_bytes(manifest)
        if len(data) > self.limits["manifest_reserve_bytes"]:
            raise BudgetExceeded("Final manifest exceeds its reserved allowance")
        (self.root / "manifest.json").write_bytes(data)
        return manifest
