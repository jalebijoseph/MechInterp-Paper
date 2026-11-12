"""Exp. 8 (Experiment D): partner contribution and task-state tracking.

Question: besides J-space, does an orchestrator maintain information on who owns each
subtask and what each peer contributed? An orchestrator assigns two subtasks to two
peers; scripted (frozen) peer messages put the target subtask into one of eight states;
then, optionally after unrelated filler messages, six separate strict-JSON probes ask
for one outcome each:

    content      which contribution codes were submitted for the subtask
    provenance   who submitted a given code (null when only a neutral document has it)
    ownership    who currently owns the subtask
    access       which peers have seen a given code
    status       assigned | in_progress | submitted | complete | blocked | conflict
    action       wait | query | verify | integrate | reassign | resolve

The probes are the non-J-space answer to the question. J-space is read on the owner
probe at two sites: ``pre_query`` (last token of the transcript, before any probe text,
so identical for every probe of a cell) and ``decision`` (last prompt token). Each
roster name is scored with band-mean reciprocal ranks under the J-lens and the logit /
output baselines. Literal names are crossed with roles (``role_swap``), so averaging
the owner margin over a swap pair cancels fixed name priors.

Dissociations built into the design: in ``reassigned`` the owner (the second peer) is
not the contributor of the only code (the first peer); in ``conflicting`` the probed
code's submitter (the second peer) is not the owner. ``document`` puts a code in
neutral shared notes, so provenance must be null. A no-peer control asks the owner and
content probes before anything has been assigned.

Peer messages are scripted, so only the orchestrator endpoint runs. Scripted messages
are a limitation: they measure tracking of stated information, not of information the
peers generated themselves.
"""

from __future__ import annotations

import hashlib
import json

from jpeers.benchmarks import load_task_bank
from jpeers.followup.common import paired_stats, parse_json_object, request, site_scores
from jpeers.followup.peer_selection import NAME_PAIRS
from jpeers.schemas import SpanQuery

PROTOCOL = "exp8-task-state-v1"
CONDITIONS = ("assigned_silent", "in_progress", "unverified", "completed", "blocked",
              "reassigned", "conflicting", "document")
DELAYS = ("none", "filler")
STATUSES = {
    "assigned": "assigned, with no update from the owner yet",
    "in_progress": "the owner says they are working on it",
    "submitted": "a contribution was received but has not been verified",
    "complete": "a contribution was received and verified",
    "blocked": "the owner cannot continue",
    "conflict": "competing contributions were received",
}
ACTIONS = {
    "wait": "let the owner keep working",
    "query": "ask the owner for an update",
    "verify": "check a contribution that has not been verified",
    "integrate": "merge verified work into the project",
    "reassign": "give the subtask to another team member",
    "resolve": "settle competing contributions",
}
# condition -> (status, next action)
STATE_LABELS = {
    "assigned_silent": ("assigned", "query"),
    "in_progress": ("in_progress", "wait"),
    "unverified": ("submitted", "verify"),
    "completed": ("complete", "integrate"),
    "blocked": ("blocked", "reassign"),
    "reassigned": ("in_progress", "wait"),
    "conflicting": ("conflict", "resolve"),
    "document": ("submitted", "verify"),
}
OUTCOMES = ("owner", "content", "provenance", "access", "status", "action")


def _digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()


def contribution_code(seed: int, project: str, subtask: str, peer_slot: str) -> str:
    """Deterministic 8-character code, unique per project, subtask and contributor slot."""
    return _digest([PROTOCOL, seed, project, subtask, peer_slot])[:8].upper()


def _peer(sender: str, text: str) -> dict:
    return {"speaker": sender, "kind": "peer",
            "text": f"[{sender} -> Orchestrator, private]: {text}"}


def _system(text: str) -> dict:
    return {"speaker": "System", "kind": "system", "text": f"[System]: {text}"}


def _orchestrator(text: str) -> dict:
    return {"speaker": "Orchestrator", "kind": "orchestrator", "text": text}


def cells(bank: dict, *, conditions=CONDITIONS, delays=DELAYS, seed: int = 0) -> list[dict]:
    """Every (project, condition, role swap, delay) cell with its frozen script and labels.

    Literal names come from ``NAME_PAIRS`` by project index and are crossed with roles,
    so ``first`` (the target subtask's original owner) is each name once per swap pair.
    """
    if set(conditions) - set(CONDITIONS) or set(delays) - set(DELAYS):
        raise ValueError("unknown task-state condition or delay")
    out = []
    for index, project in enumerate(bank["projects"]):
        names = list(NAME_PAIRS[index % len(NAME_PAIRS)])
        target, other = project["subtasks"]
        for condition in conditions:
            for role_swap in (False, True):
                first, second = names[::-1] if role_swap else names
                for delay in delays:
                    out.append(_cell(bank, index, project, names, first, second, target,
                                     other, condition, role_swap, delay, seed))
    return out


def _cell(bank, index, project, names, first, second, target, other, condition, role_swap,
          delay, seed) -> dict:
    x, y = target["label"], other["label"]
    code_first = contribution_code(seed, project["id"], target["id"], "first")
    code_second = contribution_code(seed, project["id"], target["id"], "second")
    code_other = contribution_code(seed, project["id"], other["id"], "second")
    events = [
        _system("Project kickoff. Assign the two subtasks."),
        _orchestrator(f"Assigning the {x} to {first} and the {y} to {second}."),
        _peer(second, f"The {y} is done. Contribution code: {code_other}."),
        _system(f"Check passed: contribution {code_other} for the {y} is verified."),
    ]
    owner, codes, probe_code, submitter, seen_by = first, [], None, None, None
    if condition == "in_progress":
        events.append(_peer(first, f"I have started the {x}. No contribution code yet; "
                                   "I will send it when I am done."))
    elif condition in ("unverified", "completed"):
        events.append(_peer(first, f"The {x} is ready for review. "
                                   f"Contribution code: {code_first}."))
        if condition == "completed":
            events.append(_system(f"Check passed: contribution {code_first} for the {x} "
                                  "is verified."))
        codes, probe_code, submitter, seen_by = [code_first], code_first, first, [first]
    elif condition == "blocked":
        events.append(_peer(first, f"I am blocked on the {x}: I cannot get the input data "
                                   "and cannot continue without it. No contribution code."))
    elif condition == "reassigned":
        events += [
            _peer(first, f"I am blocked on the {x}: I cannot get the input data and cannot "
                         f"continue without it. Partial contribution code: {code_first}."),
            _orchestrator(f"Reassigning the {x} to {second}. {second}, here is {first}'s "
                          f"partial contribution code: {code_first}."),
            _peer(second, f"Got it. I have taken over the {x} and I am working on it now."),
        ]
        owner = second
        codes, probe_code, submitter, seen_by = [code_first], code_first, first, \
            sorted([first, second])
    elif condition == "conflicting":
        events += [
            _peer(first, f"The {x} is ready for review. Contribution code: {code_first}."),
            _peer(second, f"I also made a version of the {x}. "
                          f"Contribution code: {code_second}."),
        ]
        codes = [code_first, code_second]
        probe_code, submitter, seen_by = code_second, second, [second]
    elif condition == "document":
        events.append({"speaker": None, "kind": "document",
                       "text": f"[Shared notes]: A draft of the {x} was uploaded. "
                               f"Contribution code: {code_first}."})
        codes, probe_code, submitter = [code_first], code_first, None
    if delay == "filler":
        filler = bank["filler"]
        events += [_system(filler[(index + k) % len(filler)]) for k in range(3)]
    status, action = STATE_LABELS[condition]
    roster = list(names)
    return {"project": project["id"], "goal": project["goal"], "target": target,
            "other": other, "condition": condition, "role_swap": role_swap, "delay": delay,
            "names": list(names), "roster": roster, "first": first, "second": second,
            "events": events,
            "labels": {"owner": owner, "codes": codes, "probe_code": probe_code,
                       "submitter": submitter, "seen_by": seen_by,
                       "status": status, "action": action},
            "last_speaker": next((e["speaker"] for e in reversed(events)
                                  if e["kind"] == "peer"), None),
            "mentions": {n: sum(e["text"].count(n) for e in events) for n in names}}


def system_prompt(cell: dict) -> str:
    target, other = cell["target"], cell["other"]
    return (f"You are the orchestrator of a two-person team: {cell['roster'][0]} and "
            f"{cell['roster'][1]}.\nProject goal: {cell['goal']}\n"
            f"Subtasks:\n- the {target['label']}: {target['description']}\n"
            f"- the {other['label']}: {other['description']}\n"
            "You assign subtasks, keep track of who owns each one, and integrate finished "
            "work. Team members report privately to you. A contribution code is the only "
            "record of a contribution.")


def transcript_messages(cell: dict, *, include_events: bool = True) -> tuple[list[dict], str]:
    """Chat messages for a cell and the text of its final transcript event.

    Orchestrator turns are assistant messages; everything else is user content, merged
    so roles alternate.
    """
    messages = [{"role": "system", "content": system_prompt(cell)}]
    events = cell["events"] if include_events else cell["events"][:1]
    for event in events:
        role = "assistant" if event["kind"] == "orchestrator" else "user"
        if messages[-1]["role"] == role:
            messages[-1] = {"role": role, "content": messages[-1]["content"] + "\n" + event["text"]}
        else:
            messages.append({"role": role, "content": event["text"]})
    return messages, events[-1]["text"]


def probe_question(outcome: str, cell: dict) -> str:
    x = f"the {cell['target']['label']}"
    names = " or ".join(cell["roster"])
    code = cell["labels"]["probe_code"]
    statuses = "; ".join(f"{k} ({v})" for k, v in STATUSES.items())
    actions = "; ".join(f"{k} ({v})" for k, v in ACTIONS.items())
    tail = " Return only the JSON object, with no explanation. Do not guess."
    return {
        "owner": f'Who currently owns {x}? Return {{"owner": "<name>"}} with exactly '
                 f'{names}, or {{"owner": null}} if nobody has been assigned it.',
        "content": f'Which contribution codes have been submitted for {x}? Return '
                   f'{{"codes": [...]}} listing every literal code, or {{"codes": []}} if none.',
        "provenance": f'Who submitted contribution code {code}? Return {{"submitted_by": '
                      f'"<name>"}} with exactly {names}, or {{"submitted_by": null}} if the '
                      'conversation does not say who submitted it.',
        "access": f'Besides you, which team members have seen contribution code {code}? '
                  f'Return {{"seen_by": [...]}} listing names from {names}, or '
                  '{"seen_by": []} if none.',
        "status": f'What is the current status of {x}? Return {{"status": "<status>"}} with '
                  f'one of: {statuses}.',
        "action": f'What should you do next about {x}? Return {{"next_action": "<action>"}} '
                  f'with one of: {actions}.',
    }[outcome] + tail


def probes_for(cell: dict) -> list[str]:
    """Outcomes that have a ground truth in this cell (no code -> no provenance/access)."""
    labels = cell["labels"]
    out = ["owner", "content"]
    if labels["probe_code"] is not None:
        out.append("provenance")
        if labels["seen_by"] is not None:
            out.append("access")
    return out + ["status", "action"]


def _name_or_null(value, roster) -> tuple[bool, str | None]:
    if value is None:
        return True, None
    return (True, value) if isinstance(value, str) and value in roster else (False, None)


def parse_probe(outcome: str, text: str, roster) -> dict:
    """Strict: one JSON object with exactly the outcome's key and a valid value."""
    key = {"owner": "owner", "content": "codes", "provenance": "submitted_by",
           "access": "seen_by", "status": "status", "action": "next_action"}[outcome]
    invalid = {"valid": False, "value": None}
    try:
        data = parse_json_object(text)
    except (TypeError, ValueError):
        return invalid
    if set(data) != {key}:
        return invalid
    value = data[key]
    if outcome in ("owner", "provenance"):
        ok, value = _name_or_null(value, roster)
        return {"valid": ok, "value": value} if ok else invalid
    if outcome in ("content", "access"):
        if not isinstance(value, list) or not all(isinstance(v, str) for v in value) \
                or len(set(value)) != len(value):
            return invalid
        if outcome == "access" and any(v not in roster for v in value):
            return invalid
        return {"valid": True, "value": sorted(value)}
    allowed = STATUSES if outcome == "status" else ACTIONS
    return {"valid": True, "value": value} if isinstance(value, str) and value in allowed \
        else invalid


def truth(outcome: str, cell: dict):
    labels = cell["labels"]
    return {"owner": labels["owner"], "content": sorted(labels["codes"]),
            "provenance": labels["submitter"], "access": labels["seen_by"],
            "status": labels["status"], "action": labels["action"]}[outcome]


def score_probe(outcome: str, text: str, cell: dict) -> dict:
    parsed = parse_probe(outcome, text, cell["roster"])
    expected = truth(outcome, cell)
    correct = parsed["valid"] and parsed["value"] == expected
    return {"outcome": outcome, "text": text, **parsed, "expected": expected,
            "correct": bool(correct)}


def _mean(values) -> float | None:
    valid = [float(v) for v in values if v is not None]
    return sum(valid) / len(valid) if valid else None


def owner_margins(row: dict, site: str, lens: str = "jlens") -> dict:
    """Owner-minus-other, and contributor-minus-owner where they differ."""
    scores = row["scores"][site]
    first, second, owner = row["first"], row["second"], row["labels"]["owner"]
    other = second if owner == first else first

    def diff(a, b):
        sa, sb = scores[a][lens], scores[b][lens]
        return None if sa is None or sb is None else sa - sb

    submitter = row["labels"]["submitter"]
    return {"owner": diff(owner, other),
            "submitter_vs_owner": diff(submitter, owner)
            if submitter is not None and submitter != owner else None}


def swap_effects(rows: list[dict], site: str = "pre_query", lens: str = "jlens",
                 key: str = "owner") -> list[dict]:
    """Average each margin over its role-swap pair (cancels fixed name priors), then
    average within project over conditions. Projects are the independent units."""
    pairs: dict = {}
    for row in rows:
        k = (row["project"], row["condition"], row["delay"])
        pair = pairs.setdefault(k, {})
        if row["role_swap"] in pair:
            raise ValueError(f"duplicate task-state cell: {k}/{row['role_swap']}")
        pair[row["role_swap"]] = row
    by_project: dict = {}
    for (project, condition, delay), pair in pairs.items():
        if set(pair) != {False, True}:
            raise ValueError(f"incomplete role-swap pair: {project}/{condition}/{delay}")
        margins = [owner_margins(pair[s], site, lens)[key] for s in (False, True)]
        if any(m is None for m in margins):
            continue
        by_project.setdefault(project, []).append(_mean(margins))
    return [{"project": p, "effect": _mean(v), "n_pairs": len(v)}
            for p, v in sorted(by_project.items())]


def _record_sites(trace, rec, end_text_span: str) -> dict:
    if trace is None or not trace.spans.get(end_text_span):
        raise ValueError("owner probe is missing its trace or transcript-end span")
    end = trace.spans[end_text_span][-1][1]
    masks = {"pre_query": trace.mask((end - 1, end)),
             "decision": trace.mask((rec.prompt_len - 1, rec.prompt_len))}
    if any(not m.any() for m in masks.values()):
        raise ValueError("owner probe is missing a required token site")
    return masks


def probe_messages(messages: list[dict], question: str) -> list[dict]:
    """The transcript with the probe question as (or appended to) the final user turn."""
    out = json.loads(json.dumps(messages))
    if out[-1]["role"] == "user":
        out[-1]["content"] += "\n\n" + question
    else:
        out.append({"role": "user", "content": question})
    return out


def _probe(endpoint, writer, cell, outcome, messages, end_text, *, record, extra):
    probe_messages_ = probe_messages(messages, probe_question(outcome, cell))
    spans = [SpanQuery(label="transcript_end", text=end_text, occurrence="last")] if record else []
    rec, trace = request(endpoint, writer, messages=probe_messages_,
                         tag=f"task_state/{cell['project']}/{cell['condition']}/{outcome}",
                         extra={**extra, "kind": f"probe:{outcome}"},
                         watch=cell["roster"] if record else (), spans=spans,
                         max_new_tokens=64, record=record)
    result = score_probe(outcome, rec.text, cell)
    if record:
        masks = _record_sites(trace, rec, "transcript_end")
        result["scores"] = {site: {n: site_scores(trace, [n], mask) for n in cell["roster"]}
                            for site, mask in masks.items()}
        result["trace_file"] = rec.trace_file
    return result


def expected_counts(n_projects: int, conditions=CONDITIONS, delays=DELAYS,
                    no_peer: bool = True) -> dict:
    with_code = sum(c not in ("assigned_silent", "in_progress", "blocked") for c in conditions)
    with_access = sum(c in ("unverified", "completed", "reassigned", "conflicting")
                      for c in conditions)
    per_swap_delay = 2 * len(delays) * n_projects
    probes = per_swap_delay * (4 * len(conditions) + with_code + with_access)
    return {"cells": per_swap_delay * len(conditions), "probes": probes,
            "recorded": per_swap_delay * len(conditions),
            "no_peer_probes": 2 * n_projects if no_peer else 0}


def run_task_state(cfg: dict, pool, writer) -> dict:
    """Run every cell's probes and return rows, summary and validated counts."""
    bank = load_task_bank("task_state")
    projects = bank["projects"][: cfg.get("n_projects", len(bank["projects"]))]
    bank = {**bank, "projects": projects}
    conditions = tuple(cfg.get("conditions", CONDITIONS))
    delays = tuple(cfg.get("delays", DELAYS))
    seed = cfg.get("seed", 0)
    endpoint = pool.get(cfg["orchestrator"])
    rows, probes, no_peer = [], [], []
    for cell in cells(bank, conditions=conditions, delays=delays, seed=seed):
        messages, end_text = transcript_messages(cell)
        extra = {"project": cell["project"], "condition": cell["condition"],
                 "role_swap": cell["role_swap"], "delay": cell["delay"],
                 "agent": "Orchestrator", "protocol": PROTOCOL}
        results = {o: _probe(endpoint, writer, cell, o, messages, end_text,
                             record=o == "owner", extra=extra) for o in probes_for(cell)}
        probes += [{**extra, **r} for r in results.values()]
        rows.append({**{k: cell[k] for k in ("project", "condition", "role_swap", "delay",
                                                "names", "roster", "first", "second",
                                                "labels", "last_speaker", "mentions")},
                     "messages_sha256": _digest(messages),
                     "scores": results["owner"]["scores"],
                     "trace_file": results["owner"]["trace_file"],
                     "correct": {o: r["correct"] for o, r in results.items()},
                     "valid": {o: r["valid"] for o, r in results.items()}})
    if cfg.get("no_peer", True):
        # Nothing assigned yet: the owner must be null and there must be no codes.
        for project_cell in cells(bank, conditions=("assigned_silent",), delays=("none",),
                                  seed=seed):
            if project_cell["role_swap"]:
                continue
            control = {**project_cell, "labels": {**project_cell["labels"], "owner": None,
                                                  "codes": []}}
            messages, end_text = transcript_messages(control, include_events=False)
            extra = {"project": control["project"], "condition": "no_peer",
                     "agent": "Orchestrator", "protocol": PROTOCOL}
            for outcome in ("owner", "content"):
                no_peer.append({**extra, **_probe(endpoint, writer, control, outcome, messages,
                                                  end_text, record=False, extra=extra)})
    counts = {"cells": len(rows), "probes": len(probes),
              "recorded": sum(r["trace_file"] is not None for r in rows),
              "no_peer_probes": len(no_peer)}
    expected = expected_counts(len(projects), conditions, delays, cfg.get("no_peer", True))
    if counts != expected:
        raise ValueError(f"task-state counts differ: observed={counts}, expected={expected}")
    return {"protocol": PROTOCOL, "experiment": "task_state", "rows": rows, "probes": probes,
            "no_peer_probes": no_peer, "summary": summarize(rows, probes, no_peer),
            "counts": counts, "expected_counts": expected}


def summarize(rows: list[dict], probes: list[dict], no_peer: list[dict]) -> dict:
    """Behavioral accuracy per outcome and condition, and the J-space owner effects."""
    behavior = {}
    for condition in dict.fromkeys(r["condition"] for r in rows):
        for delay in dict.fromkeys(r["delay"] for r in rows):
            cell_probes = [p for p in probes
                           if p["condition"] == condition and p["delay"] == delay]
            behavior[f"{condition}/{delay}"] = {
                o: {"n": len(ps), "accuracy": _mean(p["correct"] for p in ps),
                    "invalid": sum(not p["valid"] for p in ps)}
                for o in OUTCOMES if (ps := [p for p in cell_probes if p["outcome"] == o])}
    jspace = {}
    for delay in dict.fromkeys(r["delay"] for r in rows):
        rs = [r for r in rows if r["delay"] == delay]
        for site in ("pre_query", "decision"):
            for lens in ("jlens", "logit", "output"):
                effects = swap_effects(rs, site, lens)
                jspace[f"{delay}/{site}/{lens}"] = {
                    "project_effects": effects,
                    "stats": paired_stats([e["effect"] for e in effects])}
        # The dissociation conditions: is the code's submitter favoured over the owner?
        dissociation = [r for r in rs if r["condition"] in ("reassigned", "conflicting")]
        if dissociation:
            effects = swap_effects(dissociation, "pre_query", "jlens", "submitter_vs_owner")
            jspace[f"{delay}/pre_query/jlens/submitter_vs_owner"] = {
                "project_effects": effects,
                "stats": paired_stats([e["effect"] for e in effects])}
    primary_key = "filler/pre_query/jlens" if "filler/pre_query/jlens" in jspace \
        else next(iter(jspace), None)
    recency = {}
    for condition in dict.fromkeys(r["condition"] for r in rows):
        rs = [r for r in rows if r["condition"] == condition]
        recency[condition] = _mean(r["last_speaker"] == r["labels"]["owner"] for r in rs
                                   if r["last_speaker"] is not None)
    return {
        "behavior": behavior,
        "behavior_overall": {o: _mean(p["correct"] for p in probes if p["outcome"] == o)
                             for o in OUTCOMES},
        "no_peer": {o: _mean(p["correct"] for p in no_peer if p["outcome"] == o)
                    for o in ("owner", "content")},
        "jspace": jspace,
        "primary": {"key": primary_key,
                    "stats": jspace[primary_key]["stats"] if primary_key else None,
                    "unit": "project; owner margin averaged over each role-swap pair",
                    "readout": "band-mean reciprocal rank, owner minus other peer, at the last "
                               "transcript token before any probe text"},
        "owner_is_last_peer_speaker": recency,
        "interpretation": "Probe accuracy shows what the orchestrator can report; the "
                          "pre-query J-space margin shows whether the owner is represented "
                          "before it is asked. Decision-site margins mostly restate the "
                          "imminent answer. Scripted peer messages; not a theory-of-mind test.",
    }
