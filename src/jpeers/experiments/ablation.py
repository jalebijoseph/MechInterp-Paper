"""Experiment 4: the same experiments with J-space removed or edited.

Runs ``base`` (collab | imposter | peer_selection | tom_validation) once per intervention
condition and reports each metric's change against the unedited baseline. The default
condition set pairs every J-space ablation with a random-direction control of the same
rank, which is what separates "J-space matters" from "any k-dim edit hurts":

    none, jspace_light, random_k5, jspace_medium, random_k10, jspace_heavy, random_k25,
    jspace_peer_span

Config: ``base`` plus that experiment's own keys; ``interventions`` overrides the list
(names from :data:`jpeers.interventions.PRESETS`, or ``{name: [raw spec dicts]}``).
"""

from __future__ import annotations

import datetime as _dt
import importlib
from pathlib import Path

from jpeers.experiments.common import write_summary

DEFAULT_CONDITIONS = ["none", "jspace_light", "random_k5", "jspace_medium", "random_k10",
                      "jspace_heavy", "random_k25", "jspace_peer_span"]


def run(cfg: dict) -> dict:
    base = cfg["base"]
    mod = importlib.import_module(f"jpeers.experiments.{base}")
    sub = dict(cfg)
    sub.setdefault("interventions", DEFAULT_CONDITIONS)
    # keep ablation runs apart from plain runs of the same base experiment
    stamp = cfg.get("run_id") or _dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    sub["run_id"] = f"ablation-{stamp}"
    summary = mod.run(sub)
    rows = summary.get("rows", [])

    def key(r: dict) -> tuple:  # the row's identity apart from the condition
        return tuple(sorted((k, v) for k, v in r.items() if k != "condition" and isinstance(v, str)))

    def numeric(v) -> bool:
        return isinstance(v, (int, float)) and not isinstance(v, bool)

    baseline = {key(r): r for r in rows if r.get("condition") == "none"}
    deltas = []
    for r in rows:
        b = baseline.get(key(r))
        if r.get("condition") == "none" or b is None:
            continue
        d = {"condition": r["condition"], **dict(key(r))}
        d.update({f"d_{k}": v - b[k] for k, v in r.items()
                  if k != "n" and numeric(v) and numeric(b.get(k))})
        deltas.append(d)

    return write_summary(Path(summary["run_dir"]), {**summary, "deltas_vs_none": deltas},
                         rows + deltas, f"Exp. 4 ablation of {base}: metrics and deltas vs none")
