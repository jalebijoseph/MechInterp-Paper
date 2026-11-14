"""Workspace-owned test directories with inherited Windows sandbox permissions."""
from contextlib import contextmanager
from pathlib import Path
import shutil
import uuid

ROOT = Path(__file__).resolve().parents[1] / "runs"


@contextmanager
def temporary_directory(*, dir=None):
    base = Path(dir).resolve() if dir is not None else ROOT.resolve()
    if not base.is_relative_to(ROOT.resolve()):
        raise ValueError("Experiment 5 test directories must stay under workspace runs")
    base.mkdir(parents=True, exist_ok=True)
    path = (base / ("exp5-unit-" + uuid.uuid4().hex)).resolve()
    path.mkdir()
    try:
        yield str(path)
    finally:
        if path.exists():
            if not path.resolve().is_relative_to(ROOT.resolve()):
                raise ValueError("Refusing to remove a test directory outside workspace runs")
            shutil.rmtree(path)
