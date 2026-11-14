"""Fresh private-world tasks and separately labelled controlled observations.

World truth and input-condition labels belong exclusively to ``analysis``.
The two rendering functions explicitly select the fields visible to each role.
"""
from __future__ import annotations

import itertools
import json
import math
import random

from .protocol import CELL_ORDER, derived_seed

PEERS = ("A", "B")
DOMAINS = ("possession", "location")
COMPARISONS = ("equal_information", "correct_vs_incomplete", "correct_vs_incorrect")
SPLITS = ("development", "evaluation")
TEMPLATES = {
    "development": {
        "possession": (
            "Fresh scene. Each person holds exactly one kind of object, {first} or {second}; people may hold the same kind. Which kind does {subject} hold?",
            "In this fresh scene, {subject} holds one object kind: {first} or {second}. Different people can have the same kind. What does {subject} hold?",
        ),
        "location": (
            "Fresh scene. Each person is in exactly one place, {first} or {second}; people may share a place. Where is {subject}?",
            "In this fresh scene, {subject} is at one of two locations: {first} or {second}. Locations can be shared. Which location contains {subject}?",
        ),
    },
    "evaluation": {
        "possession": (
            "A fresh scene begins. Identify the kind of object held by {subject}. Every person has one kind, either {first} or {second}, and kinds need not differ between people.",
            "Consider a fresh scene with object kinds {first} and {second}. A person has exactly one kind; several people may have that kind. Report the kind in {subject}'s possession.",
        ),
        "location": (
            "A fresh scene begins. Identify the whereabouts of {subject}. Every person occupies one location, either {first} or {second}, and locations need not differ between people.",
            "Consider a fresh scene with locations {first} and {second}. A person occupies exactly one location; several people may occupy it. Report the location of {subject}.",
        ),
    },
}


def _integer(value: object, name: str, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _labels(value: object, name: str, *, minimum: int, exact: int | None = None) -> list[str]:
    if (not isinstance(value, list) or len(value) < minimum
            or (exact is not None and len(value) != exact)
            or any(not isinstance(v, str) or not v.strip() or v != v.strip() for v in value)
            or len(set(value)) != len(value) or "unknown" in value):
        raise ValueError(f"{name} requires distinct nonempty labels")
    return value


def _settings(config: dict) -> dict:
    if not isinstance(config, dict) or not isinstance(config.get("tasks"), dict):
        raise ValueError("config.tasks is required")
    settings = config["tasks"]
    _integer(settings.get("seed"), "tasks.seed")
    rounds = _integer(settings.get("rounds"), "tasks.rounds", 4)
    reversal = settings.get("reversal_round")
    if rounds % 4:
        raise ValueError("tasks.rounds must be divisible by four for exact domain/half schedules")
    if reversal is not None:
        _integer(reversal, "tasks.reversal_round", 4)
        if reversal >= rounds or reversal % 4:
            raise ValueError("reversal_round is a zero-based boundary inside the session, divisible by four")
    _labels(settings.get("objects"), "tasks.objects", minimum=2, exact=2)
    _labels(settings.get("locations"), "tasks.locations", minimum=2, exact=2)
    for key, minimum in (("peer_names", 2), ("subjects", 3)):
        pools = settings.get(key)
        if not isinstance(pools, dict):
            raise ValueError(f"tasks.{key} must contain development and evaluation pools")
        for split in SPLITS:
            pool = _labels(pools.get(split), f"tasks.{key}.{split}", minimum=minimum)
            if key == "peer_names" and len(pool) % 2:
                raise ValueError("peer-name pools must consist of pairs")
        if set(pools["development"]) & set(pools["evaluation"]):
            raise ValueError(f"tasks.{key} development/evaluation pools must be disjoint")
    return settings


def _request(config: dict, split: str, family: int) -> dict:
    settings = _settings(config)
    if split not in SPLITS:
        raise ValueError(f"split must be one of {SPLITS}")
    _integer(family, "family")
    return settings


def _rng(seed: int, *parts: object) -> random.Random:
    return random.Random(derived_seed(seed, *parts))


def roster(config: dict, split: str, family: int) -> dict[str, str]:
    """Fixed slot identities; insertion order is the counterbalanced roster order."""
    settings = _request(config, split, family)
    names = settings["peer_names"][split]
    pair = (family // 8) % (len(names) // 2)
    assigned = names[pair * 2:pair * 2 + 2]
    if family % 2:
        assigned = assigned[::-1]
    identity = dict(zip(PEERS, assigned))
    order = PEERS if (family // 2) % 2 == 0 else PEERS[::-1]
    return {peer: identity[peer] for peer in order}


def _references(settings: dict, split: str, family: int, phase: str) -> dict[str, str]:
    reference = {
        domain: _rng(settings["seed"], split, family, "reference", domain).choice(PEERS)
        for domain in DOMAINS
    }
    if phase == "reversed":
        reference = {domain: PEERS[1 - PEERS.index(peer)] for domain, peer in reference.items()}
    return reference


def _scene(settings: dict, split: str, family: int, stream: str, phase: str,
           domain: str, index: int, comparison: str, supplied: dict[str, bool]) -> dict:
    parts = (split, family, stream, phase, domain, index)
    seed = settings["seed"]
    values = settings["objects" if domain == "possession" else "locations"]
    subjects = settings["subjects"][split]
    world_rng = _rng(seed, *parts, "world")
    world = {subject: world_rng.choice(values) for subject in subjects}
    query = _rng(seed, *parts, "query").choice(subjects)
    distractors = [subject for subject in subjects if subject != query]
    packet_rng = _rng(seed, *parts, "packet")
    packet_rng.shuffle(distractors)
    target_first = packet_rng.choice((False, True))
    reference = _references(settings, split, family, phase)
    conditions, packets = {}, {}
    for peer in PEERS:
        condition = ("correct" if comparison == "equal_information" or peer == reference[domain]
                     else comparison.removeprefix("correct_vs_"))
        conditions[peer] = condition
        target_subject, target_value = query, world[query]
        if not supplied[peer]:
            if condition == "incomplete":
                target_subject, target_value = distractors[1], world[distractors[1]]
            elif condition == "incorrect":
                target_value = next(value for value in values if value != world[query])
        facts = [{"subject": distractors[0], "value": world[distractors[0]]},
                 {"subject": target_subject, "value": target_value}]
        packets[peer] = facts[::-1] if target_first else facts
    options = list(values)
    _rng(seed, *parts, "options").shuffle(options)
    template = _rng(seed, *parts, "template").choice(TEMPLATES[split][domain])
    question = template.format(subject=query, first=options[0], second=options[1])
    return {
        "id": f"{stream}:{split}:{family}:{comparison}:{phase}:{domain}:{index}",
        "domain": domain, "question": question, "query_subject": query,
        "options": options, "private_facts": packets,
        "analysis": {"true_answer": world[query], "world": world, "conditions": conditions,
                     "reference_by_domain": reference, "phase": phase},
    }


def session_tasks(config: dict, split: str, family: int, comparison: str, *,
                  audit: bool = False) -> list[dict]:
    """Build exogenous tasks; audits have their own count, worlds, and streams.

    A reversal boundary is zero based. Audits contain ``cases_per_domain`` for
    each represented phase, rather than using the main trajectory's horizon.
    Half schedules are private inputs, never promises about live peer answers.
    """
    settings = _request(config, split, family)
    if comparison not in COMPARISONS:
        raise ValueError(f"comparison must be one of {COMPARISONS}")
    if type(audit) is not bool:
        raise ValueError("audit must be boolean")
    reversal = settings.get("reversal_round")
    phases = [("initial", settings["rounds"] if reversal is None else reversal)]
    if reversal is not None:
        phases.append(("reversed", settings["rounds"] - reversal))
    stream = "audit" if audit else "main"
    if audit:
        audits = config.get("audits")
        if not isinstance(audits, dict):
            raise ValueError("config.audits must be an object")
        count = _integer(audits.get("cases_per_domain"), "audits.cases_per_domain", 2)
        if count % 2:
            raise ValueError("audit cases_per_domain must be even for exact half schedules")
        phases = [(phase, count * 2) for phase, _ in phases]
    rows = []
    for phase, length in phases:
        count = length // 2
        domains = list(DOMAINS) * count
        _rng(settings["seed"], split, family, stream, phase, "domain_schedule").shuffle(domains)
        supplied = {
            (peer, domain): set(_rng(settings["seed"], split, family, stream, phase,
                                     "private_schedule", peer, domain).sample(range(count), count // 2))
            for peer, domain in CELL_ORDER
        }
        indices = dict.fromkeys(DOMAINS, 0)
        for domain in domains:
            index = indices[domain]
            rows.append(_scene(settings, split, family, stream, phase, domain, index, comparison,
                               {peer: index in supplied[peer, domain] for peer in PEERS}))
            indices[domain] += 1
    return rows


def _calibration_settings(config: dict) -> tuple[int, float, float, float]:
    settings = config.get("development")
    if not isinstance(settings, dict):
        raise ValueError("config.development must be an object")
    count = _integer(settings.get("observations_per_cell"), "development.observations_per_cell", 2)
    rates = [settings.get(key) for key in ("low_rate", "high_rate", "anchor_rate")]
    _success_counts(rates, count)
    low, high, anchor = rates
    if not 0 < low < anchor < high < 1:
        raise ValueError("calibration rates must satisfy 0 < low < anchor < high < 1")
    return count, low, high, anchor


def _success_counts(profile: list[float], count: int) -> list[int]:
    if not isinstance(profile, list) or any(
        isinstance(rate, bool) or not isinstance(rate, (int, float))
        or not math.isfinite(rate) or not 0 <= rate <= 1
        or not math.isclose(rate * count, round(rate * count), rel_tol=0, abs_tol=1e-9)
        for rate in profile
    ):
        raise ValueError("rates must be finite probabilities representable as integer success counts")
    return [round(rate * count) for rate in profile]


def calibration_profiles(config: dict) -> list[dict]:
    """Independent four-cell matrices plus paired, preference-reversing anchors."""
    _settings(config)
    _, low, high, anchor = _calibration_settings(config)
    profiles = [{"id": "binary:" + "".join(map(str, bits)),
                 "rates": [high if bit else low for bit in bits]}
                for bits in itertools.product((0, 1), repeat=4)]
    for target in range(4):
        rates = [low, high, high, low] if target < 2 else [high, low, low, high]
        rates[(target + 2) % 4] = anchor
        for label, rate in (("low", low), ("high", high)):
            changed = list(rates)
            changed[target] = rate
            profiles.append({"id": f"anchor:{target}:{label}", "rates": changed})
    return profiles


def calibration_tasks(config: dict, family: int, profile: list[float]) -> list[dict]:
    """Construct scripted, checked observations, never main-study source replies."""
    settings = _request(config, "development", family)
    count, _, _, _ = _calibration_settings(config)
    if not isinstance(profile, list) or len(profile) != len(CELL_ORDER):
        raise ValueError("profile must contain four rates in CELL_ORDER")
    counts = _success_counts(profile, count)
    success_indices = {}
    for cell, successes in zip(CELL_ORDER, counts):
        indices = list(range(count - 1))
        _rng(settings["seed"], "development", family, "calibration", "success_order", *cell).shuffle(indices)
        # Match the latest outcome across profile counterfactuals; balance it by family/cell.
        if (family + CELL_ORDER.index(cell)) % 2:
            indices.insert(0, count - 1)
        else:
            indices.append(count - 1)
        success_indices[cell] = set(indices[:successes])
    rows = []
    for index in range(count):
        order = list(CELL_ORDER)
        _rng(settings["seed"], "development", family, "calibration", "observation_order", index).shuffle(order)
        for peer, domain in order:
            # Each cell has its own fresh task bank, held fixed across all profiles.
            row = _scene(settings, "development", family, f"calibration-{peer}", "initial",
                         domain, index, "equal_information", {"A": True, "B": True})
            answer = row["analysis"]["true_answer"]
            if index not in success_indices[peer, domain]:
                answer = next(value for value in row["options"] if value != answer)
            response = json.dumps({"answer": answer}, ensure_ascii=False, separators=(",", ":"))
            row.update(peer=peer, response=response, correct=grade_peer(row, response)["correct"])
            rows.append(row)
    return rows


def public_task(task: dict) -> dict:
    """The explicit orchestrator-visible task schema, with no private state."""
    return {"event": "task", "domain": task["domain"], "question": task["question"],
            "query_subject": task["query_subject"], "options": list(task["options"])}


def peer_messages(task: dict, peer: str) -> list[dict]:
    """Fresh task-only request; the peer sees exactly its own private packet."""
    if peer not in PEERS:
        raise ValueError("peer must be A or B")
    payload = public_task(task)
    payload.update(event="peer_task", private_facts=[dict(fact) for fact in task["private_facts"][peer]])
    return [
        {"role": "system", "content": "Answer the current fresh scene using only its supplied private facts. "
         "Other people may have the same object kind or location; do not infer uniqueness. "
         'Return only a JSON object with exactly one key, "answer", whose value is an exact '
         'option label, or "unknown" if the facts do not determine the answer.'},
        {"role": "user", "content": json.dumps(payload, ensure_ascii=False, separators=(",", ":"))},
    ]


def grade_peer(task: dict, text: str) -> dict:
    """Strict single-object JSON and exact labels; unknown is valid but unrewarded."""
    invalid = {"answer": None, "valid": False, "correct": False, "unknown": False}
    if not isinstance(text, str):
        return invalid

    def unique_object(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate key")
            result[key] = value
        return result

    def reject_constant(value):
        raise ValueError(f"invalid JSON constant {value}")

    try:
        response = json.loads(text, object_pairs_hook=unique_object, parse_constant=reject_constant)
    except (ValueError, RecursionError):
        return invalid
    if not isinstance(response, dict) or set(response) != {"answer"}:
        return invalid
    answer = response["answer"]
    if not isinstance(answer, str) or answer not in [*task["options"], "unknown"]:
        return invalid
    return {"answer": answer, "valid": True,
            "correct": answer == task["analysis"]["true_answer"], "unknown": answer == "unknown"}


def counterfactual_feedback(task: dict, peer: str, correct: bool) -> str:
    """Controlled donor-history observation with a coherently changed answer."""
    if peer not in PEERS or type(correct) is not bool:
        raise ValueError("counterfactual feedback requires peer A/B and boolean correctness")
    answer = task["analysis"]["true_answer"]
    if not correct:
        answer = next(value for value in task["options"] if value != answer)
    response = json.dumps({"answer": answer}, ensure_ascii=False, separators=(",", ":"))
    checked = grade_peer(task, response)
    return json.dumps({"event": "peer_feedback", "peer": peer, "response": response,
                       "correct": checked["correct"]}, ensure_ascii=False, separators=(",", ":"))
