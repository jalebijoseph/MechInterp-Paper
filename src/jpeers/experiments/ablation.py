"""Experiment 4: the same experiments with J-space removed or edited.

Runs ``base`` (collab | imposter | peer_selection | tom_validation) once per intervention
condition and reports each metric's change against the unedited baseline. The default
condition set pairs every J-space ablation with a size-matched random control
(``matched_k*``: the same amount removed, along a random direction), which is what
separates "J-space content matters" from "any edit this large hurts":

    none, jspace_light, matched_k5, jspace_medium, matched_k10, jspace_heavy, matched_k25,
    jspace_generated, matched_generated, jspace_peer_span, matched_peer_span

``jspace_generated`` (remove J-space only while writing) is the main Exp. 4 condition;
``jspace_peer_span`` (only while reading the peer) is its comparison. Each has a control
that edits the same positions by the same amount.

Pre-registered primary test (docs/07_preregistration.md), for ``base: imposter``: on
false claims, ``catch_score`` under ``primary_condition`` minus ``primary_control``,
paired by (source, fact) with a sign-flip test. ``fluency`` and ``copies_peer`` are
paired the same way: "still communicates" needs catch_score to hold *and* fluency to
stay within ``max_fluency_drop`` of the control.

Config: ``base`` plus that experiment's own keys; ``interventions`` overrides the list
(names from :data:`jpeers.interventions.PRESETS`, or ``{name: [raw spec dicts]}``);
``primary_condition`` / ``primary_control`` / ``max_fluency_drop``.
"""

from __future__ import annotations

import datetime as _dt
import importlib
from pathlib import Path

from jpeers.analysis import paired_permutation_test
from jpeers.experiments.common import mean, write_summary

DEFAULT_CONDITIONS = ["none", "jspace_light", "matched_k5", "jspace_medium", "matched_k10",
                      "jspace_heavy", "matched_k25", "jspace_generated", "matched_generated",
                      "jspace_peer_span", "matched_peer_span"]


def primary_imposter(live: list[dict], condition: str, control: str,
                     max_fluency_drop: float) -> dict:
    """Pre-registered Exp. 4 test on Exp. 2 rows (see module docstring)."""
    def by_unit(cond: str) -> dict[tuple, dict]:
        return {(r["source"], r["item"]): r for r in live
                if r["condition"] == cond and r["claim"] == "false" and r["claim_in_draft"]}

    a, b = by_unit(condition), by_unit(control)
    units = sorted(set(a) & set(b))
    out: dict = {"condition": condition, "control": control, "n_pairs": len(units)}
    for m in ("catch_score", "fluency", "copies_peer"):
        xa, xb = [float(a[u][m]) for u in units], [float(b[u][m]) for u in units]
        out[m] = {"condition": mean(xa), "control": mean(xb),
                  "diff": mean(x - y for x, y in zip(xa, xb)),
                  "p": paired_permutation_test(xa, xb)}
    out["fluency_ok"] = (bool(units) and
                         out["fluency"]["diff"] >= -max_fluency_drop)
    return out


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

    extra = {"deltas_vs_none": deltas}
    if base == "imposter" and summary.get("live"):
        extra["primary"] = primary_imposter(
            summary["live"], cfg.get("primary_condition", "jspace_generated"),
            cfg.get("primary_control", "matched_generated"),
            cfg.get("max_fluency_drop", 0.1))
    return write_summary(Path(summary["run_dir"]), {**summary, **extra},
                         rows + deltas, f"Exp. 4 ablation of {base}: metrics and deltas vs none")
