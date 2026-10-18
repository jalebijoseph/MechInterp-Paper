"""Benchmark validation: does the J-lens read out *belief* states?

Before asking whether agents model each other, check that the lens can see ToM content
at all. For false-belief items the protagonist's belief (e.g. "box") differs from the
world state (e.g. "basket"). We read the lens rank of the belief word and the decoy
(reality on false-belief items, the stale answer on controls) at three sites:

    story_end  last story token: the story is read, the question not yet asked
    question   the question tokens
    answer     the decision position after "Answer:" (also read with the logit-lens and
               output baselines)

At ``answer`` the model is about to say its answer, so J-space agreeing with the answer
there shows little: ``beats_answer`` close to ``beats_answer_output`` means the J-lens
only restates the output. Belief tracking in J-space predicts ``beats_story_end`` and
``beats_question`` > 0.5 on false-belief items *and* controls.

``framing: agents`` runs the same items rewritten as LLM agents sharing a workspace.
Hi-ToM items add higher-order (2nd-4th order) beliefs.

Config keys: model (endpoint name), n_items, framings, include_hitom, hitom_items,
hitom_orders, max_new_tokens, interventions.
"""

from __future__ import annotations

import logging
import re

import numpy as np

from jpeers.analysis import rank_summary, word_ranks
from jpeers.benchmarks import false_belief_items, load_hitom
from jpeers.experiments.common import (
    intervention_conditions,
    make_pool,
    make_writer,
    mean,
    record_spec,
    resolved,
    write_summary,
)
from jpeers.harness import TurnRecord
from jpeers.schemas import GenerateRequest, SpanQuery
from jpeers.trace import JSpaceTrace

logger = logging.getLogger(__name__)

SYSTEM = "Read the story and answer the question with a single word."


SITES = ("story_end", "question", "answer")


def _ask(endpoint, writer, rec_cfg, story: str, question: str, watch: list[str], ivs,
         max_new_tokens: int, tag: str, extra: dict, suffix: str = ""):
    # Record the story, the question and the answer prefill. Only at the answer position
    # is the model about to say the belief word; the earlier sites test whether J-space
    # holds the belief before the model is asked for it. ``suffix`` (e.g. Hi-ToM's list
    # of choices) follows the question but is outside the "question" span, so the
    # question site is not scored on a list that spells out every candidate.
    # Only the story's last token is read, so record just its tail (long Hi-ToM stories).
    spans = [SpanQuery(label="story_tail", text=story[-40:]),
             SpanQuery(label="question", text=question),
             SpanQuery(label="answer", text="Answer:")]
    req = GenerateRequest(
        messages=[{"role": "system", "content": SYSTEM},
                  {"role": "user", "content": f"{story}\n\nQuestion: {question}{suffix}"}],
        assistant_prefill="Answer:", max_new_tokens=max_new_tokens, spans=spans,
        record=record_spec(rec_cfg, watch, scope="spans",
                           labels=["story_tail", "question", "answer"]),
        interventions=ivs)
    resp = endpoint.generate(req)
    trace = JSpaceTrace.from_response(resp)
    writer.write(TurnRecord(index=0, agent="subject", endpoint=endpoint.name, model=resp.model,
                            text=resp.text, prompt_len=resp.prompt_len,
                            n_generated=resp.n_generated, timings=resp.timings,
                            interventions=[iv.model_dump() for iv in ivs], trace_file=None,
                            tag=tag, messages=req.messages, extra=extra), trace)
    return resp, trace


def _best(trace: JSpaceTrace, word: str, mask: np.ndarray, lens: str = "jlens"
          ) -> int | None:
    ranks = word_ranks(trace, word, mask, lens)
    if ranks.size == 0 or np.isnan(ranks.astype(float)).all():
        return None
    return int(np.nanmin(ranks))


def _site_masks(trace: JSpaceTrace, last: np.ndarray) -> dict[str, np.ndarray]:
    """story_end: last story token (story read, question not yet asked); question: the
    question tokens; answer: the decision position after the "Answer:" prefill."""
    story = trace.spans.get("story_tail", [])
    story_end = trace.mask((story[-1][1] - 1, story[-1][1])) if story else last & False
    return {"story_end": story_end, "question": trace.mask("question"), "answer": last}


def _beats_all(trace: JSpaceTrace, target: str, competitors: list[str], mask: np.ndarray,
               lens: str) -> bool | None:
    """Does ``target`` outrank every competitor at this site (None if not readable)?"""
    b = _best(trace, target, mask, lens)
    ds = [d for c in competitors if (d := _best(trace, c, mask, lens)) is not None]
    return None if b is None or not ds else b < min(ds)


def _beats(trace: JSpaceTrace, it, mask: np.ndarray, lens: str) -> bool | None:
    """Does the belief word outrank the decoy at this site (None if not readable)?"""
    return _beats_all(trace, it.belief_answer, [it.decoy], mask, lens)


def hitom_nouns(choices: str) -> list[str]:
    """Container nouns of Hi-ToM choices ("A. blue_drawer, B. green_treasure_chest" ->
    ["drawer", "chest"]), deduplicated in order."""
    return list(dict.fromkeys(c.split("_")[-1]
                              for c in re.findall(r"[a-z]+(?:_[a-z]+)+", choices)))


def run(cfg: dict) -> dict:
    pool = make_pool(cfg)
    endpoint = pool.get(cfg["model"])
    writer = make_writer(cfg, "tom_validation")
    rows, per_item = [], []
    skipped: list[str] = []
    max_new = cfg.get("max_new_tokens", 6)

    for cond, iv_spec in intervention_conditions(cfg).items():
        ivs = resolved(iv_spec)
        for framing in cfg.get("framings", ["people", "agents"]):
            items = false_belief_items(cfg.get("n_items", 20), framing=framing,
                                       seed=cfg.get("seed", 0))
            for it in items:
                watch = [it.belief_answer, it.true_answer, it.decoy]
                resp, trace = _ask(endpoint, writer, cfg, it.story, it.question, watch, ivs,
                                   max_new, f"{cond}/{framing}", {"item": it.id})
                last = trace.mask((resp.prompt_len - 1, resp.prompt_len))
                b = _best(trace, it.belief_answer, last)
                t = _best(trace, it.true_answer, last)
                d = _best(trace, it.decoy, last)
                row = {"condition": cond, "framing": framing, "item": it.id,
                       "false_belief": not it.control, "belief_rank": b,
                       "true_rank": t, "decoy_rank": d,
                       "correct": it.belief_answer.lower() in resp.text.lower(),
                       "answer": resp.text}
                for site, mask in _site_masks(trace, last).items():
                    row[f"beats_{site}"] = _beats(trace, it, mask, "jlens")
                for lens in ("logit", "output"):
                    row[f"beats_answer_{lens}"] = _beats(trace, it, last, lens)
                per_item.append(row)

        if cfg.get("include_hitom", False):
            for it in load_hitom(orders=tuple(cfg.get("hitom_orders", [0, 1, 2])),
                                 max_items=cfg.get("hitom_items", 40)):
                # Decoys: the other container nouns among the choices. Choices that share
                # the answer's noun (blue_drawer vs green_drawer) cannot be told apart by
                # single-word readouts, so the test is "answer noun beats every other
                # noun" (chance ~ 1 / number of distinct nouns).
                noun = it.answer.split("_")[-1]
                others = [n for n in hitom_nouns(it.choices) if n != noun]
                suffix = f"\nChoices: {it.choices}\nAnswer with the exact choice text."
                try:
                    resp, trace = _ask(endpoint, writer, cfg, it.story, it.question,
                                       [noun, *others], ivs, 12, f"{cond}/hitom",
                                       {"item": it.id}, suffix=suffix)
                except ValueError as exc:  # a noun is not a single token
                    logger.warning("skipping %s: %s", it.id, exc)
                    skipped.append(it.id)
                    continue
                last = trace.mask((resp.prompt_len - 1, resp.prompt_len))
                row = {"condition": cond, "framing": f"hitom-o{it.order}", "item": it.id,
                       "false_belief": it.order > 0, "belief_rank": _best(trace, noun, last),
                       "true_rank": None, "n_competitors": len(others),
                       "correct": it.answer.replace("_", " ") in resp.text.replace("_", " "),
                       "answer": resp.text}
                for site, mask in _site_masks(trace, last).items():
                    row[f"beats_{site}"] = _beats_all(trace, noun, others, mask, "jlens")
                for lens in ("logit", "output"):
                    row[f"beats_answer_{lens}"] = _beats_all(trace, noun, others, last, lens)
                per_item.append(row)

    def ranks(rs: list[dict], key: str) -> np.ndarray:
        return np.array([r[key] for r in rs if r[key] is not None], dtype=float)

    groups: dict[tuple, list[dict]] = {}
    for r in per_item:
        groups.setdefault((r["condition"], r["framing"], r["false_belief"]), []).append(r)
    for (cond, framing, fb), rs in sorted(groups.items()):
        b = ranks(rs, "belief_rank")
        row = {"condition": cond, "framing": framing,
               "items": "false-belief" if fb else "true-belief control", "n": len(rs),
               "accuracy": mean(r["correct"] for r in rs),
               "belief_mrr": rank_summary(b)["mrr"], "belief_hit@10": rank_summary(b)["hit@10"]}
        if rs[0]["true_rank"] is not None:
            row["true_mrr"] = rank_summary(ranks(rs, "true_rank"))["mrr"]
            row["decoy_mrr"] = rank_summary(ranks(rs, "decoy_rank"))["mrr"]
        else:  # Hi-ToM: chance of beating every other noun
            row["chance"] = mean(1 / (r["n_competitors"] + 1) for r in rs)
        # belief beats decoy(s) at each site; real belief tracking predicts above chance
        # on false-belief items AND controls, and already at story_end / question
        for key in [*(f"beats_{s}" for s in SITES), "beats_answer_logit",
                    "beats_answer_output"]:
            row[key] = mean(r[key] for r in rs if r[key] is not None)
        rows.append(row)
    return write_summary(writer.dir, {"rows": rows, "items": per_item,
                                      "hitom_skipped_not_single_token": skipped}, rows,
                         "ToM validation: belief vs decoy before and at the answer")
