"""Experiment 8 (Experiment D): partner contribution and task-state tracking.

CLI entry point for :mod:`jpeers.followup.task_state`, which holds the protocol, the
frozen scripts, parsers and analysis. See docs/05_experiments.md, "Exp. 8".
"""

from __future__ import annotations

from jpeers.experiments.common import make_pool, make_writer, write_summary
from jpeers.followup.task_state import OUTCOMES, run_task_state


def run(cfg: dict) -> dict:
    pool = make_pool(cfg)
    writer = make_writer(cfg, "task_state")
    result = run_task_state(cfg, pool, writer)
    summary = result["summary"]
    table = []
    for key, by_outcome in summary["behavior"].items():
        condition, delay = key.split("/")
        table.append({"condition": condition, "delay": delay,
                      **{f"acc_{o}": by_outcome[o]["accuracy"] for o in OUTCOMES
                         if o in by_outcome}})
    for key, value in summary["jspace"].items():
        table.append({"condition": f"jspace:{key}", "n_projects": value["stats"]["n"],
                      "effect": value["stats"]["diff"], "p": value["stats"]["p"]})
    return write_summary(writer.dir, result, table,
                         "Exp. 8 task-state tracking: probe accuracy and owner margins")
