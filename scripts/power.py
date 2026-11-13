"""Sample size for the pre-registered primary metrics (docs/07_preregistration.md).

Reads a finished run's ``summary.json``, extracts the primary metric, and prints the
observed effect and the n needed for the target power. Paired metrics (Exp. 1, 2, 4)
use the normal approximation for a paired test, n = ((z_a + z_b) * sd / mean)^2; rate
metrics (Exp. 0, 3) use the one-sample binomial approximation against chance.

``alpha`` defaults to 0.01: 0.05 split over the 5 Holm-corrected primary tests in the
worst case, so an n that is enough at 0.01 is enough after correction.

    python scripts/power.py runs/imposter/<run_id>          # experiment from the path
    python scripts/power.py runs/collab/<run_id> --power 0.9

Effects measured on a small model (or a smoke run) are only a rough guide: re-run this
on the 8B pilot before fixing n.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from statistics import NormalDist, mean, stdev

Z = NormalDist().inv_cdf


def n_paired(diffs: list[float], alpha: float, power: float) -> dict:
    diffs = [d for d in diffs if not math.isnan(d)]
    if len(diffs) < 2:
        return {"n_units": len(diffs), "note": "need >= 2 paired units"}
    m, sd = mean(diffs), stdev(diffs)
    if m == 0 or sd == 0:
        return {"n_units": len(diffs), "mean_diff": m, "sd": sd,
                "note": "no effect or no spread observed; n cannot be estimated"}
    n = ((Z(1 - alpha / 2) + Z(power)) * sd / abs(m)) ** 2
    return {"n_units": len(diffs), "mean_diff": m, "sd": sd, "n_needed": math.ceil(n)}


def n_rate(successes: int, n: int, p0: float, alpha: float, power: float) -> dict:
    if n == 0:
        return {"n_units": 0}
    p1 = successes / n
    if p1 == p0:
        return {"n_units": n, "rate": p1, "chance": p0, "note": "rate equals chance"}
    num = Z(1 - alpha / 2) * math.sqrt(p0 * (1 - p0)) + Z(power) * math.sqrt(p1 * (1 - p1))
    return {"n_units": n, "rate": p1, "chance": p0,
            "n_needed": math.ceil((num / (p1 - p0)) ** 2)}


def exp0(s: dict, a: float, pw: float) -> dict:
    out = {}
    for fb, label in ((True, "false-belief"), (False, "control")):
        v = [r["beats_story_end"] for r in s["items"] if r.get("false_belief") is fb
             and r.get("beats_story_end") is not None and r.get("true_rank") is not None]
        out[f"beats_story_end/{label}"] = n_rate(sum(v), len(v), 0.5, a, pw)
    return out


def exp1(s: dict, a: float, pw: float) -> dict:
    turns = [t for t in s["turns"] if t["condition"] == "none"]
    def task_mean(framing: str, task: str) -> float:
        v = [t.get("partner_kw_mrr_beyond") for t in turns
             if t["framing"] == framing and t["task"] == task]
        v = [x for x in v if x is not None and not math.isnan(x)]
        return mean(v) if v else float("nan")
    tasks = sorted({t["task"] for t in turns})
    d = [task_mean("team", t) - task_mean("named_solo", t) for t in tasks]
    return {"partner_kw_mrr_beyond: team - named_solo (per task)": n_paired(d, a, pw)}


def exp2(s: dict, a: float, pw: float, m: str = "susp_mrr_reading_beyond") -> dict:
    live = [r for r in s["live"] if r["condition"] == "none" and r["claim_in_draft"]]
    cell = {(r["source"], r["claim"], r["item"]): r[m] for r in live}
    items = sorted({r["item"] for r in live})
    d = [cell[("incentive", "false", i)] - cell[("no_incentive", "false", i)] for i in items
         if ("incentive", "false", i) in cell and ("no_incentive", "false", i) in cell]
    return {f"{m}: tom_false (per fact)": n_paired(d, a, pw)}


def exp3(s: dict, a: float, pw: float) -> dict:
    dec = [r for r in s["decisions"] if r["condition"] == "none"]
    out = {}
    v = [r["task_prefers_right"] for r in dec
         if r["variant"] == "context" and r.get("task_prefers_right") is not None]
    out["task_prefers_right (context)"] = n_rate(sum(v), len(v), 0.5, a, pw)
    v = [r["chose_own"] for r in dec
         if r["variant"] == "self_preference" and r.get("chose_own") is not None]
    out["chose_own (self_preference)"] = n_rate(sum(v), len(v), 0.5, a, pw)
    return out


def exp4(s: dict, a: float, pw: float) -> dict:
    p = s.get("primary")
    if not p:
        return {"note": "no primary block (base must be imposter)"}
    live = [r for r in s["live"] if r["claim"] == "false" and r["claim_in_draft"]]
    by = {(r["condition"], r["source"], r["item"]): r["catch_score"] for r in live}
    units = {(src, it) for (_, src, it) in by}
    d = [by[(p["condition"], *u)] - by[(p["control"], *u)] for u in sorted(units)
         if (p["condition"], *u) in by and (p["control"], *u) in by]
    return {f"catch_score: {p['condition']} - {p['control']} (per fact)": n_paired(d, a, pw)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--experiment", help="tom_validation | collab | imposter | "
                    "peer_selection | ablation (default: from the path)")
    ap.add_argument("--alpha", type=float, default=0.01)
    ap.add_argument("--power", type=float, default=0.8)
    args = ap.parse_args()
    run = Path(args.run_dir)
    s = json.loads((run / "summary.json").read_text(encoding="utf-8"))
    exp = args.experiment or ("ablation" if run.name.startswith("ablation")
                              else run.parent.name)
    fn = {"tom_validation": exp0, "collab": exp1, "imposter": exp2,
          "peer_selection": exp3, "ablation": exp4}[exp]
    print(f"{exp}: alpha={args.alpha}, power={args.power}")
    for metric, res in fn(s, args.alpha, args.power).items():
        print(f"  {metric}: " + ", ".join(
            f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}" for k, v in res.items()))


if __name__ == "__main__":
    main()
