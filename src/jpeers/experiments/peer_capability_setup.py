"""Offline Experiment 5 fixtures and preparation; no model/runtime dependencies."""
from __future__ import annotations

import itertools
import hashlib
import json
from pathlib import Path
import platform
import random

MODEL = {"id": "Qwen/Qwen3-32B", "revision": "9216db5781bf21249d130ec9da846c4624c16137"}
DOMAINS = ["possession", "location"]
COMPARISONS = ["equal_information", "correct_vs_incomplete", "correct_vs_incorrect"]
VALUES = {"possession": ["ball", "chair"], "location": ["garden", "kitchen"]}
PEERS = ["A", "B"]
BUDGET_BYTES = 1073741824


def validate_config(config: dict) -> None:
    """Validate this setup schema, not a frozen execution protocol."""
    required = {"schema_version", "stage", "model", "inference", "delegation_choices",
                "domains", "comparisons", "readout", "storage_budget_bytes", "fixtures"}
    if not isinstance(config, dict) or set(config) != required:
        raise ValueError("Setup configuration has missing or unsupported fields")
    if type(config["schema_version"]) is not int:
        raise ValueError("Schema version must be an integer")
    fixed = {"schema_version": 1, "stage": "setup_only", "model": MODEL,
             "delegation_choices": PEERS, "domains": DOMAINS, "comparisons": COMPARISONS,
             "readout": {"subject": "orchestrator", "method": "linear_probe",
                         "target": "observed_success_rate"}}
    for key, expected in fixed.items():
        if config[key] != expected:
            raise ValueError(f"Setup {key} contradicts the initial Experiment 5 protocol")
    inference = config["inference"]
    if (not isinstance(inference, dict)
            or set(inference) != {"enable_thinking", "sampling_policy", "peer_parameters_shared"}
            or inference["enable_thinking"] is not False
            or inference["sampling_policy"] != "seeded"
            or inference["peer_parameters_shared"] is not True):
        raise ValueError("Setup requires non-thinking, seeded sampling and matching peer parameters")
    budget = config["storage_budget_bytes"]
    if type(budget) is not int or not 0 < budget <= BUDGET_BYTES:
        raise ValueError("Storage budget must be a positive integer no larger than 1 GiB")
    fixtures = config["fixtures"]
    if not isinstance(fixtures, dict) or set(fixtures) != {"seed", "cases_per_domain", "peer_names"}:
        raise ValueError("Fixtures require seed, cases_per_domain, and peer_names")
    seed, count = fixtures["seed"], fixtures["cases_per_domain"]
    if type(seed) is not int or seed < 0:
        raise ValueError("Fixture seed must be a nonnegative integer")
    if type(count) is not int or not 2 <= count <= 64 or count % 2:
        raise ValueError("Fixture preview requires an even 2-64 cases per domain")
    names = fixtures["peer_names"]
    if (not isinstance(names, dict) or set(names) != set(PEERS)
            or any(not isinstance(n, str) or not n.strip() for n in names.values())
            or len({n.strip().casefold() for n in names.values()}) != 2):
        raise ValueError("Provide two distinct, nonempty peer names for A and B")


def build_fixtures(config: dict) -> list[dict]:
    """Build matched example families; these are not evaluation sessions."""
    validate_config(config)
    seed, count = config["fixtures"]["seed"], config["fixtures"]["cases_per_domain"]
    worlds = {}
    for domain in DOMAINS:
        worlds[domain] = []
        for index in range(count):
            rng = random.Random(f"{seed}:world:{domain}:{index}")
            worlds[domain].append({name: rng.choice(VALUES[domain])
                                   for name in ["Alice", "Bob", "Casey"]})
    rows = []
    for comparison in COMPARISONS:
        assignments = [("A", "A")] if comparison == "equal_information" else itertools.product(PEERS, repeat=2)
        for references in assignments:
            reference_by_domain = dict(zip(DOMAINS, references))
            family = f"{comparison}:{''.join(references)}"
            for domain in DOMAINS:
                rng = random.Random(f"{seed}:coverage:{comparison}:{domain}")
                correct_cases = set(rng.sample(range(count), count // 2))
                for index, world in enumerate(worlds[domain]):
                    truth = world["Alice"]
                    conditions, packets = {}, {}
                    for peer in PEERS:
                        condition = ("correct" if comparison == "equal_information"
                                     or peer == reference_by_domain[domain]
                                     else comparison.removeprefix("correct_vs_"))
                        conditions[peer] = condition
                        subject, value = "Alice", truth
                        if index not in correct_cases:
                            if condition == "incomplete":
                                subject, value = "Casey", world["Casey"]
                            elif condition == "incorrect":
                                value = next(v for v in VALUES[domain] if v != truth)
                        packets[peer] = [{"subject": "Bob", "value": world["Bob"]},
                                         {"subject": subject, "value": value}]
                    if domain == "possession":
                        question = ("Alice holds exactly one kind of object: ball or chair. "
                                    "Different people may hold the same kind of object. "
                                    "Which kind of object does Alice hold?")
                    else:
                        question = ("Alice is in exactly one location: garden or kitchen. "
                                    "Different people may share a location. Where is Alice?")
                    rows.append({"id": f"{family}:{domain}:{index}", "family_id": family,
                                 "domain": domain, "question": question,
                                 "private_facts": packets,
                                 "analysis": {"comparison": comparison,
                                              "scene_id": f"fixture:{seed}:{domain}:{index}",
                                              "reference_by_domain": reference_by_domain,
                                              "conditions": conditions,
                                              "true_answer": truth, "world": world}})
    return rows


def render_prompts(fixture: dict, config: dict) -> dict:
    """Render only the information visible to each role, without history or answers."""
    roster = "\n".join(f"{peer}: {config['fixtures']['peer_names'][peer]}" for peer in PEERS)
    orchestrator = [
        {"role": "system", "content": "Select one roster peer to answer the current task. "
         'Return only JSON with key "choice" and value "A" or "B".'},
        {"role": "user", "content": f"Peers:\n{roster}\n\nTask:\n{fixture['question']}"},
    ]
    peers = {}
    for peer in PEERS:
        facts = []
        for fact in fixture["private_facts"][peer]:
            subject, value = fact["subject"], fact["value"]
            facts.append(f"{subject} holds a {value}." if fixture["domain"] == "possession"
                         else f"{subject} is in the {value}.")
        peers[peer] = [
            {"role": "system", "content": "Answer using the supplied scene information. "
             'Return only JSON with key "answer" and a string value; use "unknown" '
             "when the information does not determine the answer."},
            {"role": "user", "content": "Scene information:\n" + "\n".join(facts)
             + f"\n\nQuestion:\n{fixture['question']}"},
        ]
    return {"orchestrator": orchestrator, "peers": peers}


def grade_answer(fixture: dict, answer: str) -> bool:
    """Compare the extracted answer label to world truth, including misinformation cases."""
    return isinstance(answer, str) and answer.strip().casefold() == fixture["analysis"]["true_answer"]


def prepare_setup(config_path: Path, output_dir: Path) -> dict:
    """Prepare a new offline bundle; never call models or declare execution readiness."""
    output_dir = Path(output_dir)
    if output_dir.exists():
        raise FileExistsError(f"Setup output already exists: {output_dir}")
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    fixtures = build_fixtures(config)

    def encode(value):
        return (json.dumps(value, sort_keys=True, ensure_ascii=False, indent=2) + "\n").encode("utf-8")

    rows = [dict(row, prompts=render_prompts(row, config)) for row in fixtures]
    payloads = {
        "config.json": encode(config),
        "fixtures.jsonl": ("\n".join(json.dumps(row, sort_keys=True, ensure_ascii=False)
                                     for row in rows) + "\n").encode("utf-8"),
    }
    manifest = {
        "schema_version": 1,
        "stage": "setup_only",
        "fixture_count": len(rows),
        "fixture_status": "proposed_development_examples_not_frozen_evaluation",
        "python_version": platform.python_version(),
        "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "model_calls": 0,
        "ready_for_execution": False,
        "pending_execution_gates": [
            "sampling_parameters_and_seed_schedule",
            "task_template_and_name_splits",
            "native_state_or_replay_validation",
            "probe_fit_and_online_validation",
            "model_based_quality_audits",
            "attention_mask_validation",
            "causal_donor_and_control_validation",
            "full_workload_storage_and_compute_freeze",
            "primary_registry_and_scoring_freeze",
        ],
        "files": {name: {"sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}
                  for name, data in payloads.items()},
    }
    payloads["manifest.json"] = encode(manifest)
    total = sum(len(data) for data in payloads.values())
    if total > config["storage_budget_bytes"]:
        raise ValueError(f"Setup requires {total} bytes, exceeding the storage budget")
    output_dir.mkdir(parents=True, exist_ok=False)
    # A new directory and a completion manifest written last preserve existing artifacts.
    for name, data in payloads.items():
        (output_dir / name).write_bytes(data)
    return {"output_dir": str(output_dir.resolve()), "fixture_count": len(rows),
            "setup_bytes": total, "storage_budget_bytes": config["storage_budget_bytes"],
            "storage_scope": "setup_bundle_only", "full_study_bytes": None,
            "model_calls": 0, "ready_for_execution": False}
