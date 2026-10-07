"""Benchmark validation: does the J-lens read out *belief* states?

Before asking whether agents model each other, check that the lens can see ToM content
at all. For false-belief items the protagonist's belief (e.g. "box") differs from the
world state (e.g. "basket"). At the answer position we read the lens rank of both words
across the workspace band:

    belief_rank < true_rank on false-belief items  -> J-space represents the belief
    accuracy                                       -> does the model answer correctly

``framing: agents`` runs the same items rewritten as LLM agents sharing a workspace.
Hi-ToM items add higher-order (2nd-4th order) beliefs.

Config keys: model (endpoint name), n_items, framings, include_hitom, hitom_items,
hitom_orders, max_new_tokens, interventions.
"""

from __future__ import annotations

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
from jpeers.schemas import GenerateRequest
from jpeers.trace import JSpaceTrace

SYSTEM = "Read the story and answer the question with a single word."


def _ask(endpoint, writer, rec_cfg, story: str, question: str, watch: list[str], ivs,
         max_new_tokens: int, tag: str, extra: dict):
    req = GenerateRequest(
        messages=[{"role": "system", "content": SYSTEM},
                  {"role": "user", "content": f"{story}\n\nQuestion: {question}"}],
        assistant_prefill="Answer:", max_new_tokens=max_new_tokens,
        record=record_spec(rec_cfg, watch, scope="last"), interventions=ivs)
    resp = endpoint.generate(req)
    trace = JSpaceTrace.from_response(resp)
    writer.write(TurnRecord(index=0, agent="subject", endpoint=endpoint.name, model=resp.model,
                            text=resp.text, prompt_len=resp.prompt_len,
                            n_generated=resp.n_generated, timings=resp.timings,
                            interventions=[iv.model_dump() for iv in ivs], trace_file=None,
                            tag=tag, messages=req.messages, extra=extra), trace)
    return resp, trace


def run(cfg: dict) -> dict:
    pool = make_pool(cfg)
    endpoint = pool.get(cfg["model"])
    writer = make_writer(cfg, "tom_validation")
    rows, per_item = [], []
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
                b = int(word_ranks(trace, it.belief_answer, last).min())
                t = int(word_ranks(trace, it.true_answer, last).min())
                d = int(word_ranks(trace, it.decoy, last).min())
                correct = it.belief_answer.lower() in resp.text.lower()
                per_item.append({"condition": cond, "framing": framing, "item": it.id,
                                 "false_belief": not it.control, "belief_rank": b,
                                 "true_rank": t, "decoy_rank": d,
                                 "belief_beats_decoy": b < d, "correct": correct,
                                 "answer": resp.text})

        if cfg.get("include_hitom", False):
            for it in load_hitom(orders=tuple(cfg.get("hitom_orders", [0, 1, 2])),
                                 max_items=cfg.get("hitom_items", 40)):
                noun = it.answer.split("_")[-1]
                q = f"{it.question}\nChoices: {it.choices}\nAnswer with the exact choice text."
                resp, trace = _ask(endpoint, writer, cfg, it.story, q, [noun], ivs, 12,
                                   f"{cond}/hitom", {"item": it.id})
                last = trace.mask((resp.prompt_len - 1, resp.prompt_len))
                per_item.append({"condition": cond, "framing": f"hitom-o{it.order}",
                                 "item": it.id, "false_belief": it.order > 0,
                                 "belief_rank": int(word_ranks(trace, noun, last).min()),
                                 "true_rank": None, "belief_beats_decoy": None,
                                 "correct": it.answer.replace("_", " ") in resp.text.replace(
                                     "_", " "), "answer": resp.text})

    groups: dict[tuple, list[dict]] = {}
    for r in per_item:
        groups.setdefault((r["condition"], r["framing"], r["false_belief"]), []).append(r)
    for (cond, framing, fb), rs in sorted(groups.items()):
        b = np.array([r["belief_rank"] for r in rs])
        row = {"condition": cond, "framing": framing,
               "items": "false-belief" if fb else "true-belief control", "n": len(rs),
               "accuracy": mean(r["correct"] for r in rs),
               "belief_mrr": rank_summary(b)["mrr"], "belief_hit@10": rank_summary(b)["hit@10"]}
        if rs[0]["true_rank"] is not None:
            t = np.array([r["true_rank"] for r in rs])
            row["true_mrr"] = rank_summary(t)["mrr"]
            row["decoy_mrr"] = rank_summary(np.array([r["decoy_rank"] for r in rs]))["mrr"]
            row["belief_beats_decoy"] = mean(r["belief_beats_decoy"] for r in rs)
        rows.append(row)
    return write_summary(writer.dir, {"rows": rows, "items": per_item}, rows,
                         "ToM validation: J-lens readout of beliefs at the answer position")
