"""Experiment 1: do collaborating agents represent each other in J-space?

Two agents split a task (research vs implementation) and talk for ``rounds`` rounds.

Conditions:
    team  -- each agent is told it works with the named partner; partner messages are
             shown as "[Bob]: ...".
    solo  -- control. Same task halves and the *same partner text*, but each agent is
             told it works alone and partner text is shown as anonymous "[Shared notes]".
             Content exposure is matched; only the social framing differs.

Readouts per turn (workspace band, recorded on the reading span and on generation):
    partner_name_mrr     partner's name in J-space while writing
    team_mrr             team words (team, together, partner, colleague, we)
    partner_kw_mrr       keywords of the partner's half of the task
    own_kw_mrr           keywords of the agent's own half (reference)
    uptake               share of J-space tokens taken from the partner's last message
    cross_agent_cosine   similarity of the two agents' J-space bags in the same round

Each MRR also comes as ``_logit`` (logit-lens baseline), ``_output`` (the model's own
next-token prediction) and ``_beyond`` (J-lens at positions where the word is not among
the next 3 tokens). Best case for the hypothesis: team > solo on partner_name /
partner_kw while writing, with own_kw unchanged, and the gap holding in ``_beyond`` and
exceeding the ``_output`` gap.

Config keys: agents.{research,implementation} (endpoint names), names, n_tasks, rounds,
conditions, max_new_tokens, interventions (Exp. 4).
"""

from __future__ import annotations

from jpeers.analysis import (
    bag_similarity,
    content_uptake,
    jspace_bag,
    lens_mrrs,
    permutation_test,
)
from jpeers.benchmarks import load_task_bank
from jpeers.experiments.common import (
    intervention_conditions,
    make_pool,
    make_writer,
    mean,
    record_spec,
    resolved,
    write_summary,
)
from jpeers.harness import Agent, Conversation

TEAM_WORDS = ["team", "together", "partner", "colleague", "we", "collaborate"]


def _system(name: str, partner: str, role: str, other_role: str, task: dict, team: bool) -> str:
    mine, theirs = task[role]["brief"], task[other_role]["brief"]
    if team:
        return (f"You are {name}, the {role} specialist on a two-agent team with {partner}, "
                f"the {other_role} specialist.\nTeam goal: {task['goal']}\n"
                f"Your part: {mine}\n{partner}'s part: {theirs}\n"
                f"Work with {partner}: build on their messages and keep the joint plan "
                "consistent. Reply in under 120 words.")
    # Same information as the team prompt (incl. the other half of the task); only the
    # social framing differs: no named partner, the other half is "handled elsewhere".
    return (f"You are {name}, a {role} specialist working alone.\nGoal: {task['goal']}\n"
            f"Your part: {mine}\nRelated workstream, handled elsewhere: {theirs}\n"
            "You may see shared notes from a document; use them if relevant. "
            "Reply in under 120 words.")


def run(cfg: dict) -> dict:
    pool = make_pool(cfg)
    bank = load_task_bank("collab_tasks")["tasks"][: cfg.get("n_tasks", 8)]
    names = cfg.get("names", {"research": "Alice", "implementation": "Bob"})
    endpoints = cfg["agents"]
    rounds = cfg.get("rounds", 2)
    writer = make_writer(cfg, "collab")
    per_turn: list[dict] = []

    for cond_iv, iv_spec in intervention_conditions(cfg).items():
        for framing in cfg.get("conditions", ["team", "solo"]):
            team = framing == "team"
            for task in bank:
                agents = []
                for role, other in (("research", "implementation"),
                                    ("implementation", "research")):
                    name, partner = names[role], names[other]
                    watch = [partner, name, *TEAM_WORDS, *task[role]["keywords"],
                             *task[other]["keywords"]]
                    agents.append(Agent(
                        name=name, endpoint=endpoints[role],
                        system_prompt=_system(name, partner, role, other, task, team),
                        max_new_tokens=cfg.get("max_new_tokens", 160), watch=watch,
                        interventions=resolved(iv_spec, peer=partner, self_name=name)))
                tag = f"{cond_iv}/{framing}/{task['id']}"
                conv = Conversation(agents, pool, writer=writer,
                                    peer_framing="agent" if team else "document",
                                    record_defaults=record_spec(cfg, [], scope="spans"), tag=tag,
                                    extra={"condition": cond_iv, "framing": framing,
                                           "task": task["id"]})
                conv.say("Coordinator", f"Kick-off: {task['goal']} Please start.")
                order = [names["research"], names["implementation"]]
                for rnd in range(rounds):
                    bags = {}
                    for name in order:
                        rec = conv.step(name, extra={"round": rnd})
                        trace = conv.traces[-1]
                        role = "research" if name == names["research"] else "implementation"
                        other = "implementation" if role == "research" else "research"
                        partner = names[other]
                        gen = trace.mask(generated=True)
                        peer_texts = [t.content for t in conv.transcript[:-1]
                                      if t.speaker == partner]
                        own_ctx = task["goal"] + task[role]["brief"]
                        row = {
                            "condition": cond_iv, "framing": framing, "task": task["id"],
                            "round": rnd, "agent": name, "role": role,
                            **lens_mrrs(trace, [partner], gen, "partner_name_mrr"),
                            **lens_mrrs(trace, TEAM_WORDS, gen, "team_mrr"),
                            **lens_mrrs(trace, task[other]["keywords"], gen, "partner_kw_mrr"),
                            **lens_mrrs(trace, task[role]["keywords"], gen, "own_kw_mrr"),
                            "text_mentions_partner": partner.lower() in rec.text.lower(),
                        }
                        if peer_texts:
                            row["uptake"] = content_uptake(trace, gen, peer_texts[-1],
                                                           own_ctx)["uptake"]
                            row.update(lens_mrrs(trace, [partner], trace.mask("peer_last"),
                                                 "partner_name_mrr_reading"))
                        bags[name] = jspace_bag(trace, gen)
                        per_turn.append(row)
                    if len(bags) == 2:
                        sim = bag_similarity(*bags.values())
                        per_turn[-1]["cross_agent_cosine"] = sim["cosine"]
                        per_turn[-2]["cross_agent_cosine"] = sim["cosine"]

    metrics = [f"{base}{sfx}" for base in ("partner_name_mrr", "partner_name_mrr_reading",
                                           "team_mrr", "partner_kw_mrr", "own_kw_mrr")
               for sfx in ("", "_logit", "_output", "_beyond")]
    metrics += ["uptake", "cross_agent_cosine", "text_mentions_partner"]
    rows = []
    keys = sorted({(r["condition"], r["framing"]) for r in per_turn})
    for cond, framing in keys:
        rs = [r for r in per_turn if r["condition"] == cond and r["framing"] == framing]
        rows.append({"condition": cond, "framing": framing, "turns": len(rs),
                     **{m: mean(r.get(m) for r in rs) for m in metrics}})
    tests = {}
    for cond in {c for c, _ in keys}:
        team = [r for r in per_turn if r["condition"] == cond and r["framing"] == "team"]
        solo = [r for r in per_turn if r["condition"] == cond and r["framing"] == "solo"]
        if team and solo:
            tests[cond] = {m: permutation_test([r.get(m, float("nan")) for r in team],
                                               [r.get(m, float("nan")) for r in solo])
                           for m in metrics}
    return write_summary(writer.dir, {"rows": rows, "p_team_vs_solo": tests, "turns": per_turn},
                         rows, "Exp. 1 collaboration: peer representation in J-space")
