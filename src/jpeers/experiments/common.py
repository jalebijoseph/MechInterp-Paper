"""Shared plumbing for experiment runners."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from jpeers.client import EndpointPool, load_endpoints
from jpeers.harness import RunWriter
from jpeers.interventions import resolve
from jpeers.schemas import PositionSelector, RecordSpec


def make_pool(cfg: dict) -> EndpointPool:
    local = cfg.get("local", {})
    return EndpointPool(load_endpoints(cfg.get("endpoints_file")), local_defaults=local)


def make_writer(cfg: dict, experiment: str, suffix: str = "") -> RunWriter:
    run_id = cfg.get("run_id")
    if run_id and suffix:
        run_id = f"{run_id}-{suffix}"
    return RunWriter(cfg.get("out_dir", "runs"), experiment, run_id=run_id, config=cfg,
                     save_messages=cfg.get("save_messages", True))


def record_spec(cfg: dict, watch: list[str], *, scope: str = "all",
                labels: list[str] | None = None) -> RecordSpec:
    """Recording settings from ``cfg["record"]``. ``scope="spans"`` with ``labels`` records
    only those spans plus generated tokens (what most metrics read), which keeps cost
    flat as transcripts grow."""
    r = dict(cfg.get("record", {}))
    r.setdefault("top_k", 10)
    default = {"scope": scope, "skip_first": 4}
    if scope == "spans":
        default.update(labels=labels or ["peer_last"], include_generated=True)
    positions = r.pop("positions", default)
    return RecordSpec(watch=list(dict.fromkeys(watch)), positions=PositionSelector(**positions),
                      **r)


def intervention_conditions(cfg: dict) -> dict[str, list]:
    """``{"condition_name": preset-or-list}``; default is just the unedited baseline."""
    ivs = cfg.get("interventions")
    if not ivs:
        return {"none": []}
    if isinstance(ivs, list):
        return {name: name for name in ivs}
    return dict(ivs)


def resolved(spec, **ctx):
    return resolve(spec, **ctx)


def _clean(x):
    if isinstance(x, dict):
        return {str(k): _clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_clean(v) for v in x]
    if isinstance(x, (np.floating, float)):
        return None if math.isnan(float(x)) else round(float(x), 4)
    if isinstance(x, np.integer):
        return int(x)
    return x


def write_summary(run_dir: Path, summary: dict, table_rows: list[dict] | None = None,
                  title: str = "") -> dict:
    summary = _clean({**summary, "run_dir": str(run_dir)})
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if table_rows:
        cols = list(dict.fromkeys(k for row in table_rows for k in row))
        lines = [f"# {title}", "", "| " + " | ".join(cols) + " |",
                 "|" + "|".join("---" for _ in cols) + "|"]
        for row in _clean(table_rows):
            lines.append("| " + " | ".join("" if row.get(c) is None else str(row.get(c))
                                           for c in cols) + " |")
        (run_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary


def mean(values) -> float:
    vals = [v for v in values if v is not None and not (isinstance(v, float) and math.isnan(v))]
    return float(np.mean(vals)) if vals else float("nan")
