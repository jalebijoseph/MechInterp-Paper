"""Sweep steering strength for each steering preset and recommend one per model.

Steering adds ``strength x median residual norm`` along the steering words' J-lens
vectors at every band layer, so a usable strength depends on the model and band. For
each preset and strength, a reviewer reads a scripted draft stating a true or a false
fact (the Exp. 2 setting) and we record:

    fluency    distinct word bigrams / all bigrams in the reply (repetitive collapse -> low)
    copies_peer  share of the reply's 4-grams copied from the peer's message (echoing the
               draft is fluent but is not a review)
    check_mrr  J-lens MRR of the preset's *measured* lexicon while writing. Steering words
               are never measured words, so a rise here shows the edit reached the
               concept rather than the literal steered tokens.
    behaviour  reply corrects a false claim (has_true), keeps it (keeps_false), or flags an
               honest draft (false_alarm)

Recommended strength: the largest whose mean fluency stays >= ``--min-fluency`` x the
unsteered fluency and whose copying stays within ``--max-extra-copy`` of unsteered.
Behaviour is reported, not optimised: Exp. 4 tests it.

    python scripts/steer_sweep.py --model qwen3-1.7b --items 3
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
from pathlib import Path

from jpeers.analysis import copies, fluency, group_ranks, mentions, rank_summary
from jpeers.benchmarks import load_task_bank
from jpeers.client import EndpointPool
from jpeers.experiments.collab import TEAM_WORDS
from jpeers.experiments.common import mean
from jpeers.experiments.imposter import PEER, REVIEWER, _reviewer_system, _scripted
from jpeers.harness import Agent, Conversation
from jpeers.interventions import STEER_WORDS
from jpeers.schemas import InterventionSpec, PositionSelector, RecordSpec

STRENGTHS = [0.0, 0.005, 0.01, 0.015, 0.02, 0.03, 0.04]


def measured_lexicon(preset: str, bank: dict) -> list[str]:
    return {"suspicion": bank["suspicion_words"], "trust": bank["agreement_words"],
            "teamwork": TEAM_WORDS}[preset]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen3-1.7b")
    ap.add_argument("--items", type=int, default=3, help="facts per strength (x true/false)")
    ap.add_argument("--presets", nargs="+", default=list(STEER_WORDS))
    ap.add_argument("--strengths", nargs="+", type=float, default=STRENGTHS)
    ap.add_argument("--max-new-tokens", type=int, default=80)
    ap.add_argument("--min-fluency", type=float, default=0.9)
    ap.add_argument("--max-extra-copy", type=float, default=0.15,
                    help="allowed rise in copies_peer over the unsteered reply")
    ap.add_argument("--out", default="runs/steer_sweep")
    args = ap.parse_args()

    bank = load_task_bank("imposter_facts")
    items = bank["items"][: args.items]
    pool = EndpointPool({})
    rows, examples = [], {}
    for preset in args.presets:
        lexicon = measured_lexicon(preset, bank)
        for s in args.strengths:
            ivs = [] if s == 0 else [InterventionSpec(
                kind="steer", tokens=STEER_WORDS[preset], strength=s,
                positions=PositionSelector(scope="all", skip_first=4))]
            per = []
            for item in items:
                for imposter in (True, False):
                    reviewer = Agent(REVIEWER, args.model, _reviewer_system(item),
                                     max_new_tokens=args.max_new_tokens, interventions=ivs)
                    rec_spec = RecordSpec(watch=lexicon, baselines=[],
                                          positions=PositionSelector(scope="generated"))
                    conv = Conversation([reviewer], pool, record_defaults=rec_spec)
                    conv.say("Coordinator", f"Task: {item['task']} {PEER}, please draft it.")
                    draft = _scripted(item, imposter)
                    conv.say(PEER, draft)
                    text = conv.step(REVIEWER).text
                    tr = conv.traces[-1]
                    per.append({
                        "fluency": fluency(text),
                        "copies_peer": copies(text, draft),
                        "check_mrr": rank_summary(group_ranks(tr, lexicon))["mrr"],
                        "has_true": mentions(text, [item["true"]]) if imposter else None,
                        "keeps_false": (mentions(text, [item["false"]])
                                        and not mentions(text, [item["true"]]))
                        if imposter else None,
                        "false_alarm": None if imposter else mentions(
                            text, bank["suspicion_words"]),
                    })
                    if item is items[0] and imposter:
                        examples[(preset, s)] = text
            row = {"preset": preset, "strength": s, "n": len(per),
                   **{k: mean(p[k] for p in per) for k in per[0]}}
            rows.append(row)
            print(f"{preset:10s} {s:<6} fluency={row['fluency']:.2f} "
                  f"copies={row['copies_peer']:.2f} check={row['check_mrr']:.3f} "
                  f"| {examples[(preset, s)][:90]!r}", flush=True)

    recommended = {}
    for preset in args.presets:
        rs = [r for r in rows if r["preset"] == preset]
        base = next(r for r in rs if r["strength"] == 0)
        ok = [r["strength"] for r in rs if r["strength"] > 0
              and r["fluency"] >= args.min_fluency * base["fluency"]
              and r["copies_peer"] <= base["copies_peer"] + args.max_extra_copy]
        recommended[preset] = max(ok) if ok else None

    out = Path(args.out) / f"{args.model}-{_dt.datetime.now():%Y%m%d-%H%M%S}"
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(
        {"rows": rows, "recommended": recommended, "args": vars(args),
         "examples": {f"{p}@{s}": t for (p, s), t in examples.items()}}, indent=2),
        encoding="utf-8")
    cols = ["preset", "strength", "n", "fluency", "copies_peer", "check_mrr", "has_true",
            "keeps_false", "false_alarm"]
    lines = [f"# Steering sweep: {args.model}", "",
             f"Recommended (fluency >= {args.min_fluency} x unsteered, copies_peer <= "
             f"unsteered + {args.max_extra_copy}): "
             + ", ".join(f"{p} = {s}" for p, s in recommended.items()), "",
             "| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for r in rows:
        lines.append("| " + " | ".join(
            f"{r[c]:.3f}" if isinstance(r[c], float) else str(r[c]) for c in cols) + " |")
    lines += ["", "## Example replies (first item, false claim)", ""]
    lines += [f"* **{p} @ {s}**: {t[:300]!r}" for (p, s), t in examples.items()]
    (out / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nrecommended: {recommended}\nresults: {out}")


if __name__ == "__main__":
    main()
