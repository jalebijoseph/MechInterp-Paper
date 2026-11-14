"""Explicit, bounded Experiment 5 phases; importing this module loads no model.

All model calls use named independent streams. Analysis labels stay in Python
records; the native conversation is assembled exclusively from public payloads.
"""
from __future__ import annotations

import copy
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import time

import numpy as np

from .analysis import family_test, noninferiority_test, report_payloads, summarize_records
from .diagnostics import run_checkpoint
from .probes import (_blind, artifact_payloads, conditional_bases, fit_readout,
                     load_artifact, predict_readout, select_readout)
from .protocol import CELL_ORDER, MODEL_ID, REVISION, derived_seed, parse_choice, protocol_hash, validate_protocol
from .storage import ArtifactStore, BudgetExceeded, json_bytes, jsonable
from .tasks import (COMPARISONS, calibration_profiles, calibration_tasks, grade_peer,
                    peer_messages, public_task, roster, session_tasks)
from .validation import noninferiority_preflight


def _json(value):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _sha(data):
    return hashlib.sha256(data).hexdigest()


def _implementation():
    root = Path(__file__).parent
    return {p.name: _sha(p.read_bytes()) for p in sorted(root.glob("*.py"))}


def smoke_config(config: dict) -> dict:
    """Explicit software-test protocol, never a shortened target-model study."""
    value = copy.deepcopy(config)
    value["development"].update(families=4, layers=[0], ridges=[.1], ranks=[1],
                                doses=[1.], rescue_offset=2)
    value["evaluation"].update(families=2, early_rounds=1, late_rounds=1)
    value["tasks"].update(rounds=4, reversal_round=None)
    value["audits"]["cases_per_domain"] = 2
    value["diagnostics"].update(checkpoints=[4], delays=[0, 2, 4], primary_delay=4)
    value["statistics"].update(permutations=128, bootstrap=128)
    value["software_test_protocol"] = True
    return value


def _forecast(config, phase, info):
    families, rounds = config[phase]["families"], config["tasks"]["rounds"]
    sessions = families*len(COMPARISONS)
    phases = 1+int(config["tasks"].get("reversal_round") is not None)
    diagnostics, dev = config["diagnostics"], config["development"]
    checkpoints = sum(1 <= r <= rounds for r in diagnostics["checkpoints"])
    routes, delays = diagnostics["routes"], diagnostics["delays"]
    condition_forks = sum(len(diagnostics["conditions"]) if d == diagnostics["primary_delay"] and r in ("R0", "R2") else 1
                          for d in delays for r in routes)
    # Four independent generation calls per fork: routing and three specificity
    # controls. Donor/update attempts can stop early; these are planned bounds.
    donor_limit = diagnostics["max_donor_attempts"]
    checkpoint_nominal = len(routes)*len(delays)+4*condition_forks+2*rounds+8+5
    checkpoint_upper = len(routes)*len(delays)+4*condition_forks+(2*rounds+8)*donor_limit+5
    main_calls, audit_calls = sessions*rounds*4, sessions*phases*4*config["audits"]["cases_per_domain"]
    nominal = main_calls+audit_calls+sessions*checkpoints*checkpoint_nominal
    upper = main_calls+audit_calls+sessions*checkpoints*checkpoint_upper
    width = info.get("hidden_size", 5120)
    capture_layers = set(dev["layers"]) | {l+dev["rescue_offset"] for l in dev["layers"]}
    profiles = 24
    raw = 0
    calibration_calls = candidate_calls = 0
    if phase == "development":
        captures = families*profiles*len(delays)
        raw = captures*(len(capture_layers)*width*4+2048)
        calibration_calls = captures*3
        anchor_contexts = (families-max(2, families//2))*8
        candidate_count = len(dev["ranks"])*len(dev["doses"])
        candidate_calls = anchor_contexts*candidate_count
        nominal += calibration_calls+candidate_calls*(2+4*16+2*4*dev["observations_per_cell"]+13)
        upper += calibration_calls+candidate_calls*(2+4*16+(2*4*dev["observations_per_cell"]+8)*donor_limit+5)
    # Explicit planning assumptions rather than empirical compression claims.
    # All actual bytes, including duplicate phase copies, remain hard bounded.
    estimated = raw+audit_calls*1024+(nominal-audit_calls)*4096+1048576
    context_ceiling = info.get("max_context", 32768)
    worst = raw+upper*(context_ceiling*4+65536)+1048576
    return dict(families=families, sessions=sessions, main_turns=sessions*rounds,
                audit_requests=audit_calls, checkpoints=sessions*checkpoints,
                calibration_requests=calibration_calls, candidate_anchor_assays=candidate_calls), dict(
        model_calls_nominal=nominal, model_calls_upper_bound=upper,
        estimated_new_bytes=estimated, upper_bound_bytes=max(estimated, worst),
        raw_development_bytes=raw, configured_allowance_bytes=config["storage"]["total_bytes"],
        estimate_within_allowance=estimated <= config["storage"]["total_bytes"],
        raw_estimate_within_allowance=raw <= config["storage"]["raw_development_bytes"],
        assumptions=["all configured diagnostics enabled; valid main choices call one peer",
            "24 controlled profiles and 8 held-out anchors per validation family",
            "all rank/dose candidates; donor/update retry ceiling retained",
            "runtime validation internal forwards are additional and implementation dependent",
            "NPZ captures budget 4 bytes per coordinate plus 2048 bytes per archive",
            "nominal audit call artifacts 1024 bytes; other calls 4096 bytes; metadata 1 MiB",
            "upper bound permits one full uint32 context plus 64 KiB receipt per call",
            "compression, prefix deduplication and early unavailable forks can reduce actual bytes",
            "width uses backend metadata, otherwise proposed 5120; context ceiling defaults 32768"],
        uncertainty="planning forecast, not an empirical measurement or permission to exceed caps")


def preflight(config: dict, phase="development", backend_info=None, *, output_dir=None, ancestor_dir=None) -> dict:
    validate_protocol(config)
    if phase not in ("development", "evaluation"):
        raise ValueError("phase must be development or evaluation")
    limits = config["limits"]
    budget = any(limits.get(k) is not None and limits[k] > 0
                 for k in ("max_model_calls", "max_seconds"))
    info = backend_info or {}
    inventory, forecast = _forecast(config, phase, info)
    if ancestor_dir is not None and phase != "evaluation":
        raise ValueError("Only evaluation preflight accepts a frozen ancestor")
    retained = _verify(ancestor_dir)[3]["total_bytes"] if ancestor_dir is not None else 0
    evaluation_bytes = _forecast(config, "evaluation", info)[1]["estimated_new_bytes"] if phase == "development" else 0
    study_forecast = dict(estimated_minimum_bytes=retained+forecast["estimated_new_bytes"]+evaluation_bytes,
                          retained_ancestor_bytes=retained, planned_evaluation_bytes=evaluation_bytes,
                          scope="development plus planned evaluation" if phase == "development" else
                                "evaluation plus verified ancestors" if ancestor_dir is not None else
                                "evaluation only; supply the frozen ancestor for study accounting",
                          excludes=["future frozen handoff copies", "future reports"])
    destination = Path(output_dir).resolve() if output_dir is not None else Path.cwd()
    volume = destination
    while not volume.exists():
        volume = volume.parent
    free_bytes = shutil.disk_usage(volume).free
    checks = dict(protocol=True, execution_budget=budget,
                  sufficient_fit_and_validation_families=config["development"]["families"] >= 4,
                  target_backend=bool(info and not info.get("synthetic", True)
                                      and info.get("model_id") == MODEL_ID and info.get("revision") == REVISION),
                  runtime_validation=False)
    checks.update(estimated_storage_within_allowance=forecast["estimate_within_allowance"],
                  planned_study_storage_within_allowance=study_forecast["estimated_minimum_bytes"] <= config["storage"]["total_bytes"],
                  physical_space_for_estimate=free_bytes >= forecast["estimated_new_bytes"],
                  output_directory_available=output_dir is None or not destination.exists())
    if info:
        layers = config["development"]["layers"]
        checks["layer_geometry"] = bool(layers and min(layers) >= 0 and
            max(layers) + config["development"]["rescue_offset"] < info.get("n_layers", 0))
    return dict(phase=phase, protocol_hash=protocol_hash(config), checks=checks,
                ready_for_execution=False, requires_explicit_execute=True,
                model_calls=0, storage=copy.deepcopy(config["storage"]),
                planned_inventory=inventory, forecast=forecast, study_forecast=study_forecast, physical_free_bytes=free_bytes,
                output_directory=str(destination) if output_dir is not None else None,
                synthetic=bool(info.get("synthetic", False)),
                ni_preflight_passed=False,
                reason="Runtime validation and declared budgets are required for real execution.")


def _safe_file(root, name):
    path = (root / name).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError("Artifact path escapes its run directory")
    return path


def _verify(root, seen=None):
    """Verify every indexed byte and native history before fitting or analysis."""
    root = Path(root).resolve()
    seen = set() if seen is None else seen
    if root in seen:
        raise ValueError("Repeated/cyclic ancestor reference")
    seen.add(root)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("artifact_index_complete") is not True:
        raise ValueError("Artifact index is incomplete")
    index_bytes = (root / "artifact-index.json").read_bytes()
    if _sha(index_bytes) != manifest.get("artifact_index_sha256"):
        raise ValueError("Artifact index hash mismatch")
    index = json.loads(index_bytes)
    for name, entry in index.items():
        data = _safe_file(root, name).read_bytes()
        if len(data) != entry["bytes"] or _sha(data) != entry["sha256"]:
            raise ValueError(f"Artifact hash/size mismatch: {name}")
    config = json.loads((root / "config.json").read_text(encoding="utf-8"))
    validate_protocol(config)
    if protocol_hash(config) != manifest.get("protocol_hash"):
        raise ValueError("Protocol hash mismatch")
    records = []
    if (root / "records.jsonl").exists():
        records = [json.loads(line) for line in (root / "records.jsonl").read_text().splitlines()]
    if len(records) != manifest["record_count"]:
        raise ValueError("Record count mismatch")
    def references(value):
        if isinstance(value, dict):
            if {"path", "sha256", "length"} <= set(value) and str(value["path"]).startswith("histories/"):
                name = value["path"]
                if name not in index:
                    raise ValueError("Unindexed native history")
                raw = gzip.decompress(_safe_file(root, name).read_bytes())
                if _sha(raw) != value["sha256"] or len(raw) != 4 * value["length"]:
                    raise ValueError("Native history hash/length mismatch")
            for child in value.values():
                references(child)
        elif isinstance(value, list):
            for child in value:
                references(child)
    for name in index:
        if name.startswith("receipts/"):
            references(json.loads(gzip.decompress((root / name).read_bytes())))
    references(records)
    own_bytes = sum(entry["bytes"] for entry in index.values()) + len(index_bytes) + (root / "manifest.json").stat().st_size
    if own_bytes != manifest["total_bytes"]:
        raise ValueError("Run byte accounting mismatch")
    totals = dict(total_bytes=own_bytes, category_bytes=dict(manifest["category_bytes"]))
    for reference in manifest.get("ancestors", []):
        ancestor = (root / reference["path"]).resolve()
        if _sha((ancestor / "manifest.json").read_bytes()) != reference["manifest_sha256"]:
            raise ValueError("Ancestor manifest hash mismatch")
        _, _, _, counts = _verify(ancestor, seen)
        totals["total_bytes"] += counts["total_bytes"]
        for key, amount in counts["category_bytes"].items():
            totals["category_bytes"][key] += amount
    if totals["total_bytes"] > config["storage"]["total_bytes"]:
        raise ValueError("Logical-study artifact allowance exceeded")
    return manifest, config, records, totals


def _store(config, out_dir, phase, ancestors=()):
    limits = copy.deepcopy(config["storage"])
    refs = []
    for ancestor in ancestors:
        ancestor = Path(ancestor).resolve()
        _, _, _, counts = _verify(ancestor)
        limits["total_bytes"] -= counts["total_bytes"]
        for key, amount in counts["category_bytes"].items():
            limits[key + "_bytes"] -= amount
        refs.append(dict(path=os.path.relpath(ancestor, Path(out_dir).resolve()),
                         manifest_sha256=_sha((ancestor / "manifest.json").read_bytes())))
    if limits["total_bytes"] <= limits["manifest_reserve_bytes"] or any(v < 0 for v in limits.values()):
        raise BudgetExceeded("Ancestors exhaust this logical study's allowance")
    result = ArtifactStore(Path(out_dir), limits, metadata=dict(phase=phase,
        protocol_hash=protocol_hash(config), ancestors=refs, implementation=_implementation()))
    result.json("shared", "config.json", config)
    return result


def _finish(store, status, **metadata):
    index = {name: dict(category=category, bytes=size,
                        sha256=_sha((store.root / name).read_bytes()))
             for name, (category, size) in store.files.items() if name != "artifact-index.json"}
    return store.finish(status, artifact_index_sha256=_sha(json_bytes(index)), **metadata)


class Engine:
    """One bounded phase; call counters identify receipts, never random streams."""
    def __init__(self, config, backend, store, artifact=None, runtime_validation=None):
        self.config, self.backend, self.store = config, backend, store
        self.artifact = artifact
        self.runtime_validation = runtime_validation or {}
        self.records = []
        self.calls, self.started = 0, time.monotonic()

    def _budget(self):
        limits = self.config["limits"]
        if limits.get("max_model_calls") is not None and self.calls >= limits["max_model_calls"]:
            raise BudgetExceeded("Declared model-call allowance reached")
        if limits.get("max_seconds") is not None and time.monotonic() - self.started >= limits["max_seconds"]:
            raise BudgetExceeded("Declared wall-time allowance reached")

    def _call(self, tokens, *, stream, method, role="orchestrator", **kwargs):
        self._budget()
        seed = derived_seed(self.config["tasks"]["seed"], role, *stream)
        self.calls += 1
        call_id = f"call-{self.calls:08d}"
        started = time.monotonic()
        if method == "generate":
            result = self.backend.generate(list(tokens), sampler=self.config["inference"][role], seed=seed, **kwargs)
        else:
            result = self.backend.observe(list(tokens), **kwargs)
        receipt = dict(call_id=call_id, method=method, role=role, stream=list(stream), seed=seed,
                       history=self.store.history(tokens), receipt=result.get("receipt", {}),
                       seconds=time.monotonic()-started)
        if method == "generate":
            receipt.update(output=result.get("text"), token_ids=result.get("token_ids"), hit_cap=result.get("hit_cap"))
        reference = self.store.write("text", f"receipts/{call_id}.json.gz", gzip.compress(json_bytes(receipt), mtime=0))
        result = dict(result, call_id=call_id)
        result["receipt"] = dict(result.get("receipt", {}), artifact_reference=reference)
        return result

    def generate(self, tokens, *, stream, role="orchestrator", masks=(), edits=(), max_new_tokens=None):
        maximum = max_new_tokens or self.config["inference"]["max_peer_tokens" if role == "peer" else "max_orchestrator_tokens"]
        return self._call(tokens, stream=stream, method="generate", role=role,
                          max_new_tokens=maximum, masks=masks, edits=edits)

    def observe(self, tokens, capture, *, stream, masks=(), edits=()):
        return self._call(tokens, stream=stream, method="observe", capture=capture, masks=masks, edits=edits)

    def peer_reply(self, task, peer, *, stream):
        tokens = []
        for message in peer_messages(task, peer):
            tokens, _ = self.backend.append_message(tokens, message["role"], message["content"])
        tokens, _ = self.backend.append_message(tokens, "assistant", "", close=False)
        result = self.generate(tokens, stream=stream, role="peer")
        checked = grade_peer(task, result.get("text"))
        if result.get("hit_cap"):
            checked = dict(checked, raw_valid=checked["valid"], valid=False, correct=False)
        return dict(result, **checked)

    def record(self, record):
        def compact(value):
            if isinstance(value, dict):
                if "artifact_reference" in value:
                    return {k: jsonable(v) for k, v in value.items() if k != "forward_calls"}
                return {k: compact(v) for k, v in value.items()}
            if isinstance(value, (list, tuple)):
                return [compact(v) for v in value]
            return jsonable(value)
        value = compact(record)
        self.store.append(value)
        self.records.append(value)

    def replay(self, events):
        tokens, source, control = [], [], []
        terminator = self.backend.finish_assistant([])
        for event in events:
            start = len(tokens)
            if event["kind"] == "message":
                fragment, relative = self.backend.append_message([], event["role"], event["content"], close=event.get("close", True))
                tokens.extend(fragment)
                span = start+relative[0], start+relative[1]
            elif event["kind"] == "tokens":
                tokens += list(event["tokens"])
                span = start, len(tokens)
            elif event["kind"] == "finish":
                tail = tokens[-len(terminator):]
                tokens.extend(self.backend.finish_assistant(tail)[len(tail):])
                span = start, len(tokens)
            else:
                raise ValueError("Unknown native event kind")
            if span[1] > span[0]:
                if event.get("source"):
                    source.append(span)
                if event.get("control"):
                    control.append(span)
        return dict(tokens=tokens, source_spans=source, control_spans=control,
                    checkpoint_position=len(tokens)-1)


def _message(role, content, *, source=False, purpose=None, close=True, **metadata):
    return dict(kind="message", role=role, content=content, source=source,
                purpose=purpose, close=close, **metadata)


def _initial_events(config, split, family):
    return [_message("system", 'Delegate each fresh task to A or B. Return only {"choice":"A"} or {"choice":"B"}. '
                     'Learn from selected-peer checker feedback. Acknowledge feedback with Recorded.', purpose="instructions"),
            _message("user", _json(dict(event="roster", roster=roster(config, split, family))), purpose="roster")]


def _control_events(backend, count):
    neutral = backend.encode_text(" Neutral context continues.")
    padding = (neutral * ((count+len(neutral)-1)//len(neutral)))[:count]
    empty, span = backend.append_message([], "user", "")
    return [_message("user", "", close=False, purpose="control"),
            dict(kind="tokens", tokens=padding, control=True, purpose="control"),
            dict(kind="tokens", tokens=empty[span[1]:], purpose="control_close")]


def _receipt(events):
    events.append(_message("assistant", "Recorded.", close=False, purpose="receipt"))


def _phase(task, comparison):
    return "baseline" if comparison == "equal_information" else task["analysis"]["phase"]


def _quality_interval(successes, exposures):
    # Wilson interval: uncertainty is retained, including ties and absent cells.
    if not exposures:
        return None, [0., 1.]
    q, z = successes/exposures, 1.959963984540054
    den = 1+z*z/exposures
    center = (q+z*z/(2*exposures))/den
    half = z*math.sqrt(q*(1-q)/exposures+z*z/(4*exposures**2))/den
    return q, [max(0., center-half), min(1., center+half)]


def _audits(engine, split, family, comparison):
    groups = {}
    for index, task in enumerate(session_tasks(engine.config, split, family, comparison, audit=True)):
        phase = _phase(task, comparison)
        for peer in ("A", "B"):
            reply = engine.peer_reply(task, peer, stream=("audit", split, family, comparison, phase, index, peer))
            key = phase, peer, task["domain"]
            group = groups.setdefault(key, [0, 0])
            group[0] += int(reply["correct"])
            group[1] += 1
            engine.record(dict(type="audit", family=str(family), session=comparison, split=split, phase=phase,
                domain=task["domain"], peer=peer, valid=reply["valid"], correct=reply["correct"],
                unknown=reply["unknown"], hit_cap=reply["hit_cap"], call_id=reply["call_id"],
                independent=True, task_id=task["id"], synthetic=engine.backend.info()["synthetic"]))
    quality = {}
    for (phase, peer, domain), (s, n) in groups.items():
        q, interval = _quality_interval(s, n)
        quality.setdefault(phase, {})[peer, domain] = q
        engine.record(dict(type="audit_quality", family=str(family), session=comparison, split=split,
            phase=phase, peer=peer, domain=domain, successes=s, exposures=n, quality=q,
            confidence_interval=interval, method="Wilson 95%", synthetic=engine.backend.info()["synthetic"]))
    return quality


def _readout(engine, context, *, site, stream):
    layer = engine.artifact["layer"]
    position = len(context["tokens"])-1
    out = engine.observe(context["tokens"], {layer: [position]}, stream=stream)
    target = [s/n if n else None for s, n in zip(context["successes"], context["exposures"])]
    mask = [n > 0 for n in context["exposures"]]
    predicted = predict_readout(engine.artifact, out["captures"][layer][position])
    blind = predict_readout(engine.artifact["identity_blind"], _blind(dict(target=target, mask=mask)))
    engine.record(dict(type="readout", family=context["family"], session=context["session"],
        split=context["split"], round=context["round"], site=site, target=target, mask=mask,
        prediction=predicted["prediction"], raw=predicted["raw"],
        identity_blind_prediction=blind["prediction"], call_id=out["call_id"],
        synthetic=engine.backend.info()["synthetic"]))


def _run_sessions(engine, split, diagnostics):
    for family in range(engine.config[split]["families"]):
        for comparison in COMPARISONS:
            quality = _audits(engine, split, family, comparison)
            events = _initial_events(engine.config, split, family)
            successes, exposures, unknown = [0]*4, [0]*4, [0]*4
            tasks = session_tasks(engine.config, split, family, comparison)
            for index, task in enumerate(tasks):
                round_number = index+1
                context = dict(engine.replay(events), family=str(family), session=comparison, split=split,
                    round=round_number, successes=successes, exposures=exposures, events=events)
                if index:
                    _readout(engine, context, site="pre_task_neutral_receipt",
                             stream=("main_readout", split, family, comparison, round_number, "pre"))
                else:
                    first = events + [_message("assistant", "Recorded.", close=False)]
                    _readout(engine, dict(context, **engine.replay(first)), site="pre_task_neutral_receipt",
                             stream=("main_readout", split, family, comparison, round_number, "pre"))
                events.append(dict(kind="finish")) if index else None
                turn_start = len(events)
                events.append(_message("user", _json(public_task(task)), source=True, purpose="task", task=task))
                events.append(_message("assistant", "", close=False, purpose="choice_prefix"))
                prefix = engine.replay(events)["tokens"]
                choice = engine.generate(prefix, stream=("main", split, family, comparison, round_number, "choice"))
                peer = parse_choice(choice.get("text", ""))
                valid = peer is not None and not choice.get("hit_cap", False)
                if not valid:
                    peer = None
                events.append(dict(kind="tokens", tokens=choice["token_ids"], source=True, purpose="choice", task=task, peer=peer))
                events.append(dict(kind="finish", source=True))
                reply = None
                if valid:
                    reply = engine.peer_reply(task, peer, stream=("main", split, family, comparison, round_number, "peer", peer))
                    cell = CELL_ORDER.index((peer, task["domain"]))
                    exposures[cell] += 1
                    successes[cell] += int(reply["correct"])
                    unknown[cell] += int(reply["unknown"])
                    feedback = dict(event="peer_feedback", domain=task["domain"], peer=peer,
                                    response=reply["text"], correct=reply["correct"])
                    events.append(_message("user", _json(feedback), source=True, purpose="feedback",
                                           task=task, peer=peer, correct=reply["correct"]))
                else:
                    events.append(_message("user", _json(dict(event="invalid_delegation", reward=0,
                        instruction="This task has ended. Continue to the next task.")), source=True, purpose="feedback", task=task))
                source_count = sum(b-a for a, b in engine.replay(events[turn_start:])["source_spans"])
                events.extend(_control_events(engine.backend, source_count))
                _receipt(events)
                phase = _phase(task, comparison)
                context = dict(engine.replay(events), family=str(family), session=comparison, split=split,
                    round=round_number, successes=list(successes), exposures=list(exposures), events=copy.deepcopy(events),
                    audit_quality=quality.get(phase, {}))
                engine.record(dict(type="turn", family=str(family), session=comparison, split=split,
                    round=round_number, phase=phase, stationary=phase != "reversed", domain=task["domain"],
                    peer=peer, valid=valid, correct=bool(reply and reply["correct"]),
                    reward=int(bool(reply and reply["correct"])), unknown=bool(reply and reply["unknown"]),
                    peer_valid=reply["valid"] if reply else None, choice_output=choice["text"],
                    choice_hit_cap=choice["hit_cap"], peer_hit_cap=reply["hit_cap"] if reply else None,
                    choice_call_id=choice["call_id"], peer_call_id=reply["call_id"] if reply else None,
                    observed_successes=list(successes), observed_exposures=list(exposures), observed_unknown=list(unknown),
                    history=engine.store.history(context["tokens"]), synthetic=engine.backend.info()["synthetic"]))
                _readout(engine, context, site="post_feedback_neutral_receipt",
                         stream=("main_readout", split, family, comparison, round_number, "post"))
                if diagnostics and round_number in engine.config["diagnostics"]["checkpoints"]:
                    terminal = index+1 == len(tasks)
                    if terminal:
                        # Predetermined audit worlds preserve this family's reference
                        # assignment, while remaining outside the main trajectory.
                        next_task = next(row for row in session_tasks(engine.config, split, family, comparison, audit=True)
                            if row["domain"] == task["domain"] and row["analysis"]["phase"] == task["analysis"]["phase"])
                    else:
                        next_task = tasks[index+1]
                    context["terminal_continuation"] = terminal
                    engine.record(dict(type="checkpoint_assignment", family=str(family), session=comparison,
                        split=split, round=round_number, task_id=next_task["id"], terminal_continuation=terminal,
                        synthetic=engine.backend.info()["synthetic"]))
                    # Exogenous family/checkpoint schedule, never baseline correctness filtering.
                    target = CELL_ORDER.index((("A", "B")[(family+index)//2 % 2], next_task["domain"]))
                    run_checkpoint(engine, context, next_task, target)


def _calibration(engine):
    config = engine.config
    layers = sorted(set(config["development"]["layers"] +
                        [l+config["development"]["rescue_offset"] for l in config["development"]["layers"]]))
    records, anchors = [], []
    profiles = calibration_profiles(config)
    fit_count = max(2, config["development"]["families"]//2)
    for family in range(config["development"]["families"]):
        for profile in profiles:
            events = _initial_events(config, "development", family)
            tasks = calibration_tasks(config, family, profile["rates"])
            successes, exposures = [0]*4, [0]*4
            for index, task in enumerate(tasks):
                if index:
                    events.append(dict(kind="finish"))
                turn_start = len(events)
                peer = task["peer"]
                checked = grade_peer(task, task["response"])
                if not checked["valid"] or checked["correct"] != task["correct"]:
                    raise ValueError("Controlled calibration response/checker mismatch")
                events.append(_message("user", _json(public_task(task)), source=True, purpose="task", task=task))
                events.append(_message("assistant", "", close=False, purpose="choice_prefix"))
                events.append(dict(kind="tokens", tokens=engine.backend.encode_text(_json(dict(choice=peer))),
                                   source=True, purpose="choice", task=task, peer=peer))
                events.append(dict(kind="finish", source=True))
                events.append(_message("user", _json(dict(event="peer_feedback", domain=task["domain"], peer=peer,
                    response=task["response"], correct=checked["correct"])), source=True, purpose="feedback",
                    task=task, peer=peer, correct=checked["correct"]))
                cell = CELL_ORDER.index((peer, task["domain"]))
                successes[cell] += int(checked["correct"])
                exposures[cell] += 1
                source_count = sum(b-a for a, b in engine.replay(events[turn_start:])["source_spans"])
                events.extend(_control_events(engine.backend, source_count))
                _receipt(events)
            context = dict(engine.replay(events), family=str(family), session="calibration:"+profile["id"],
                split="development", round=len(tasks), successes=successes, exposures=exposures,
                events=events, controlled=True)
            history = engine.store.history(context["tokens"])
            for delay in config["diagnostics"]["delays"]:
                neutral = engine.backend.encode_text(" Neutral context continues.")
                tokens = context["tokens"] + (neutral*((delay+len(neutral)-1)//len(neutral)))[:delay]
                pos = len(tokens)-1
                observed = engine.observe(tokens, {layer: [pos] for layer in layers},
                    stream=("calibration", family, profile["id"], delay, "capture"))
                capture = {layer: observed["captures"][layer][pos] for layer in layers}
                name = f"calibration/{family}/{profile['id'].replace(':', '-')}/{delay}"
                engine.store.npz("raw_development", name+".npz", **{f"layer_{l}": v for l, v in capture.items()})
                for domain in ("possession", "location"):
                    rates = [profile["rates"][CELL_ORDER.index((peer, domain))] for peer in ("A", "B")]
                    winner = None if np.isclose(*rates) else ("A", "B")[int(rates[1] > rates[0])]
                    query = engine.backend.finish_assistant(tokens)
                    query, _ = engine.backend.append_message(query, "user", _json(dict(event="delegation_probe", domain=domain)))
                    query, _ = engine.backend.append_message(query, "assistant", "", close=False)
                    choice = engine.generate(query, stream=("calibration", family, profile["id"], delay, domain, "routing"))
                    parsed = parse_choice(choice["text"])
                    row = dict(id=f"{name}:{domain}", family=str(family), captures=capture, target=list(profile["rates"]),
                        mask=[True]*4, profile=profile["id"], controlled=True, delay=delay, site="neutral_receipt",
                        match_id=f"calibration:{family}:{domain}", split="development", valid=parsed is not None and not choice["hit_cap"],
                        routing_correct=bool(parsed == winner and not choice["hit_cap"]), routing_applicable=winner is not None,
                        synthetic=engine.backend.info()["synthetic"])
                    records.append(row)
                    engine.record(dict(type="calibration", **{k:v for k,v in row.items() if k != "captures"},
                        capture_path=name+".npz", history=history, call_id=observed["call_id"], choice_call_id=choice["call_id"]))
            if family >= fit_count and profile["id"].startswith("anchor:"):
                cell = int(profile["id"].split(":")[1])
                task = next(t for t in reversed(tasks) if t["domain"] == CELL_ORDER[cell][1])
                anchors.append((context, task, cell))
    return records, anchors, fit_count


def _heldout_validation(engine, rows, fit_count):
    heldout = [r for r in rows if int(r["family"]) >= fit_count and r["delay"] == engine.config["diagnostics"]["primary_delay"]]
    if not heldout:
        return dict(passed=False, reason="No untouched development validation families")
    errors, blind_errors = [], []
    for row in heldout:
        target = np.asarray(row["target"])
        pred = predict_readout(engine.artifact, row["captures"][engine.artifact["layer"]])["prediction"]
        blind = predict_readout(engine.artifact["identity_blind"], _blind(row))["prediction"]
        errors.append(float(np.mean(abs(target-pred))))
        blind_errors.append(float(np.mean(abs(target-blind))))
    validity = sum(r.get("valid") is True for r in heldout)/len(heldout)
    applicable = [r for r in heldout if r.get("routing_applicable") is not False]
    routing = sum(r.get("routing_correct") is True for r in applicable)/len(applicable) if applicable else 0.
    limits = engine.config["development"]["gates"]
    return dict(passed=bool(np.mean(errors) < np.mean(blind_errors) and
                           validity >= limits["validity"] and routing >= limits["routing"]),
                validity=validity, routing_accuracy=routing, n_routing_applicable=len(applicable),
                mae=float(np.mean(errors)),
                identity_blind_mae=float(np.mean(blind_errors)), families=sorted({r["family"] for r in heldout}),
                status="held-out controlled validation; online transfer measured separately")


def _candidate_score(records, config):
    """Paired-family uncertainty and specificity, retaining failed candidate forks."""
    rows = [r for r in records if r.get("type") == "causal"]
    groups = {}
    for row in rows:
        groups.setdefault((row["family"], row["session"], row["route"]), {})[row["condition"]] = row
    tests, passed = {}, bool(groups)
    settings = config["statistics"]
    sampling = dict(seed=settings["seed"], n_permutations=settings["permutations"], n_bootstrap=settings["bootstrap"])
    contrasts = (("damage_vs_clean", "I0", "I1", False), ("damage_vs_random", "I2", "I1", False),
                               ("donor_vs_clean", "I4", "I0", True), ("donor_vs_random", "I4", "I6", True),
                               ("rescue_vs_damage", "I3", "I1", False), ("rescue_vs_random", "I3", "I5", False))
    families = {r["family"] for r in rows}
    for route in ("R0", "R2"):
        for label, a, b, donor in contrasts:
            by_family, missing = {}, 0
            for (family, _, assigned_route), conditions in groups.items():
                if assigned_route != route:
                    continue
                if a not in conditions or b not in conditions:
                    missing += 1
                    continue
                left, right = conditions[a], conditions[b]
                if any(r.get("status") != "available" for r in (left, right)):
                    missing += 1
                    continue
                target = left.get("donor_choice") if donor else None
                if donor and target not in ("A", "B"):
                    missing += 1
                    continue
                value = lambda r: float(bool(r.get("valid")) and (r.get("choice") == target if donor else r.get("correct", False)))
                by_family.setdefault(family, []).append(value(left)-value(right))
            missing += len(families-set(by_family))
            result = family_test([float(np.mean(v)) for v in by_family.values()], **sampling)
            supported = bool(not missing and result["status"] == "available" and result["estimate"] > 0 and result["p"] <= .05)
            tests[f"{route}:{label}"] = dict(result, passed=supported, missing=missing)
            passed = passed and supported
    preservation = {}
    margin = config["development"]["gates"]["preservation_margin"]
    for route in ("R0", "R2"):
        for condition in [f"I{i}" for i in range(1, 8)]:
            for field in ("valid", "literal_correct", "other_domain_correct", "non_target_error"):
                by_family, missing = {}, 0
                for (family, _, assigned_route), conditions in groups.items():
                    if assigned_route != route:
                        continue
                    baseline, row = conditions.get("I0", {}), conditions.get(condition, {})
                    if row.get(field) is None or baseline.get(field) is None:
                        missing += 1
                        continue
                    delta = float(row[field])-float(baseline[field])
                    by_family.setdefault(family, []).append(-delta if field == "non_target_error" else delta)
                missing += len(families-set(by_family))
                result = noninferiority_test([float(np.mean(v)) for v in by_family.values()], margin=margin, **sampling)
                supported = bool(not missing and result["status"] == "available" and result["p"] <= .05)
                preservation[f"{route}:{condition}:{field}"] = dict(result, passed=supported, missing=missing)
                passed = passed and supported
    geometry = bool(rows and all(r.get("control_match_valid") is True and r.get("route_match_valid") is True for r in rows))
    attention = bool(rows and all(r.get("source_attention_valid") is True for r in rows))
    return dict(passed=bool(passed and geometry and attention), contrasts=tests, preservation=preservation,
                geometry_passed=geometry, source_attention_passed=attention,
                n_assigned=len(rows), reason="Candidate must satisfy every paired causal and preservation check")


def _discover_causal(engine, calibration, anchors, fit_count):
    fitting = [r for r in calibration if int(r["family"]) < fit_count and r["delay"] == 0]
    layer = engine.artifact["layer"]
    later = layer+engine.config["development"]["rescue_offset"]
    candidates = []
    selected = None
    for rank in sorted(engine.config["development"]["ranks"]):
        bases = dict(delete=conditional_bases(fitting, layer=layer, rank=rank),
                     restore=conditional_bases(fitting, layer=later, rank=rank))
        for dose in sorted(engine.config["development"]["doses"]):
            settings = dict(layer=layer, later_layer=later, rank=rank, strength=dose)
            engine.artifact.update(bases=bases, causal_settings=settings)
            start = len(engine.records)
            for context, task, cell in anchors:
                run_checkpoint(engine, context, task, cell, conditions=[f"I{i}" for i in range(8)],
                               routes=["R0", "R2"], delays=[engine.config["diagnostics"]["primary_delay"]])
            result = dict(settings=settings, **_candidate_score(engine.records[start:], engine.config))
            candidates.append(result)
            engine.record(dict(type="causal_candidate", split="development", synthetic=engine.backend.info()["synthetic"], **result))
            if result["passed"] and selected is None:
                selected = dict(settings=settings, bases=copy.deepcopy(bases))
        if selected is not None:
            break
    if selected:
        engine.artifact.update(causal_settings=selected["settings"], bases=selected["bases"])
    # Failed candidates remain usable for descriptive software diagnostics only.
    return dict(passed=selected is not None, selected=selected["settings"] if selected else None,
                descriptive_settings=engine.artifact.get("causal_settings"), candidates=candidates,
                selection="lowest rank, then weakest supported dose on untouched development anchors")


def _gates(engine, selection, heldout, causal, fit_count, ni):
    expected = set(map(str, range(fit_count, engine.config["development"]["families"])))
    live = [r for r in engine.records if str(r.get("family")) in expected and r.get("session") in COMPARISONS]
    readouts = [r for r in live if r.get("type") == "readout"]
    by_family = {}
    for row in readouts:
        target, pred, blind = (np.asarray(row[k], dtype=float) for k in ("target", "prediction", "identity_blind_prediction"))
        mask = np.asarray(row["mask"], dtype=bool)
        if mask.any():
            by_family.setdefault(str(row["family"]), []).append(float(np.mean(abs(target[mask]-blind[mask])-abs(target[mask]-pred[mask]))))
    online = bool(expected and set(by_family) == expected and
                  all(np.isfinite(v).all() and np.mean(v) > 0 for v in by_family.values()))
    attribution = [r for r in live if r.get("diagnostic") == "attribution" and
                   r.get("attribution_applicable") is not False]
    online = online and {str(r["family"]) for r in attribution} == expected and all(
        r.get("attribution_applicable") is True and r.get("status") == "available" and
        np.isfinite(r.get("attribution_error_improvement", np.nan)) and
        r.get("attribution_error_improvement", -1) > 0 for r in attribution)
    probe = all(v for k, v in selection["gates"].items() if k not in ("online_transfer", "target_model_evidence"))
    return dict(runtime_validation_passed=bool(engine.runtime_validation.get("passed") is True and
                                               engine.runtime_validation.get("target_gate") is True),
                probe_validation_passed=bool(probe and heldout["passed"]), online_transfer_passed=bool(online),
                causal_discovery_passed=bool(causal["passed"]), ni_preflight_passed=bool(ni["passed"]))


def _execution(config, backend, *, phase="development", out_dir=None, ancestor_dir=None):
    if backend is None:
        raise ValueError("An explicitly constructed backend is required after --execute")
    info = backend.info()
    planned = preflight(config, phase=phase, backend_info=info, output_dir=out_dir, ancestor_dir=ancestor_dir)
    if config["development"]["families"] < 4:
        raise ValueError("Need >=4 development families for disjoint fitting and validation")
    if not info.get("synthetic"):
        for name in ("estimated_storage_within_allowance", "planned_study_storage_within_allowance",
                     "physical_space_for_estimate", "output_directory_available"):
            if planned["checks"].get(name) is not True:
                raise ValueError(f"Real execution preflight failed: {name}")
        if planned["forecast"]["raw_estimate_within_allowance"] is not True:
            raise ValueError("Raw development forecast exceeds its category allowance")
        if config.get("software_test_protocol"):
            raise ValueError("Software-test protocols cannot launch target-model studies")
        if not any(config["limits"].get(k) is not None for k in ("max_model_calls", "max_seconds")):
            raise ValueError("Real execution requires a positive declared model-call or time budget")
        if info.get("model_id") != MODEL_ID or info.get("revision") != REVISION:
            raise ValueError("Real execution requires the pinned target model")
    return info


def develop(config, out_dir, backend, *, execute=False, diagnostics=True) -> dict:
    if not execute:
        return preflight(config, backend_info=backend.info() if backend else None, output_dir=out_dir)
    info = _execution(config, backend, out_dir=out_dir)
    planned = preflight(config, backend_info=info, output_dir=out_dir)
    store = _store(config, out_dir, "development")
    store.json("shared", "preflight.json", planned)
    engine = Engine(config, backend, store)
    try:
        engine._budget()
        engine.runtime_validation = backend.validate_runtime()
        store.json("shared", "runtime.json", engine.runtime_validation)
        store.json("shared", "backend.json", info)
        if not info.get("synthetic") and not (engine.runtime_validation.get("passed") is True and engine.runtime_validation.get("target_gate") is True):
            raise ValueError("Actual target-model runtime validation failed")
        calibration, anchors, fit_count = _calibration(engine)
        fitting = [r for r in calibration if int(r["family"]) < fit_count]
        engine.artifact, selection = select_readout(fitting, config["development"]["layers"],
            config["development"]["ridges"], primary_delay=config["diagnostics"]["primary_delay"])
        heldout = _heldout_validation(engine, calibration, fit_count)
        causal = _discover_causal(engine, calibration, anchors, fit_count)
        _run_sessions(engine, "development", diagnostics)
        engine._budget()
        ni = noninferiority_preflight(config)
        store.json("shared", "noninferiority-preflight.json", ni)
        gates = _gates(engine, selection, heldout, causal, fit_count, ni)
        validation = dict(gates=gates, selection=selection, controlled_validation=heldout, causal=causal,
                          fit_families=list(map(str, range(fit_count))),
                          validation_families=list(map(str, range(fit_count, config["development"]["families"]))))
        engine.artifact["development_validation"] = validation
        for name, payload in artifact_payloads(engine.artifact).items():
            store.write("shared", name, payload)
        store.json("shared", "validation.json", validation)
        return _finish(store, "complete", synthetic=bool(info["synthetic"]), model_calls=engine.calls,
                       gates=gates, elapsed_seconds=time.monotonic()-engine.started)
    except (BudgetExceeded, ValueError, RuntimeError) as exc:
        try:
            engine.record(dict(type="run_status", incomplete=True, status="incomplete", reason=str(exc),
                               split="development", synthetic=bool(info["synthetic"])))
        except BudgetExceeded:
            pass
        return _finish(store, "incomplete", synthetic=bool(info["synthetic"]), model_calls=engine.calls,
                       incomplete_reason=str(exc), elapsed_seconds=time.monotonic()-engine.started)


def freeze(development_dir, out_dir, *, simulation=False, allow_inconclusive=False,
           evaluation_families=None) -> dict:
    root = Path(development_dir).resolve()
    manifest, config, _, _ = _verify(root)
    if manifest.get("phase") != "development" or manifest["status"] != "complete":
        raise ValueError("Only a complete development phase can be frozen")
    if manifest.get("implementation") != _implementation():
        raise ValueError("Implementation changed after development")
    synthetic = bool(manifest.get("synthetic"))
    if synthetic and not simulation:
        raise ValueError("Synthetic development requires an explicitly simulation-only freeze")
    source_protocol_hash = protocol_hash(config)
    validation = json.loads((root / "validation.json").read_text())
    gates = validation["gates"]
    ni = json.loads((root / "noninferiority-preflight.json").read_text())
    if evaluation_families is not None:
        if type(evaluation_families) is not int or evaluation_families < 1:
            raise ValueError("Final evaluation family count must be a positive integer")
        config = copy.deepcopy(config)
        config["evaluation"]["families"] = evaluation_families
        validate_protocol(config)
        try:
            ni = noninferiority_preflight(config)
        except ValueError as exc:
            ni = dict(passed=False, reason=str(exc), model_calls=0, n_families=evaluation_families)
        gates["ni_preflight_passed"] = ni["passed"] is True
        validation["final_evaluation_families"] = evaluation_families
        validation["source_development_protocol_hash"] = source_protocol_hash
    required = ("runtime_validation_passed", "probe_validation_passed", "online_transfer_passed", "causal_discovery_passed", "ni_preflight_passed")
    ready = bool(not synthetic and all(gates.get(k) is True for k in required))
    if not synthetic and not ready and not allow_inconclusive:
        raise ValueError("Required scientific/runtime gates failed; use explicit inconclusive behavioral handoff")
    if not synthetic and gates.get("runtime_validation_passed") is not True:
        raise ValueError("Inconclusive behavioral handoff still requires actual runtime validation")
    store = _store(config, out_dir, "frozen", [root])
    for name in ("probe.json", "probe.npz", "runtime.json", "backend.json"):
        store.write("shared", name, (root / name).read_bytes())
    store.json("shared", "validation.json", validation)
    store.json("shared", "noninferiority-preflight.json", ni)
    frozen = dict(schema_version=1, synthetic=synthetic, simulation_only=synthetic or simulation,
        ready_for_execution=ready, allow_inconclusive=bool(allow_inconclusive),
        permitted_interpretation="synthetic software test" if synthetic else
                                 "full registered analysis" if ready else "inconclusive behavioral/descriptive analysis",
        gates=gates, protocol_hash=protocol_hash(config), implementation=_implementation(),
        source_development_protocol_hash=source_protocol_hash,
        evaluation_families=config["evaluation"]["families"],
        probe_sha256=_sha((root / "probe.npz").read_bytes()),
        runtime_sha256=_sha((root / "runtime.json").read_bytes()))
    store.json("shared", "frozen.json", frozen)
    return _finish(store, "complete", synthetic=synthetic, gates=gates)


def evaluate(config, frozen_dir, out_dir, backend, *, execute=False, simulation=False, diagnostics=True) -> dict:
    frozen_root = Path(frozen_dir).resolve()
    manifest, frozen_config, _, _ = _verify(frozen_root)
    if manifest.get("phase") != "frozen" or manifest["status"] != "complete":
        raise ValueError("A complete frozen handoff is required")
    frozen = json.loads((frozen_root / "frozen.json").read_text())
    if protocol_hash(config) != frozen["protocol_hash"]:
        raise ValueError("Evaluation protocol differs from the frozen development protocol")
    if frozen["implementation"] != _implementation():
        raise ValueError("Frozen implementation fingerprint mismatch")
    if frozen["simulation_only"] and not simulation:
        raise ValueError("This frozen handoff permits simulation only")
    if not execute:
        return preflight(config, phase="evaluation", backend_info=backend.info() if backend else None,
                         output_dir=out_dir, ancestor_dir=frozen_root)
    info = _execution(config, backend, phase="evaluation", out_dir=out_dir, ancestor_dir=frozen_root)
    if bool(info["synthetic"]) != bool(frozen["synthetic"]):
        raise ValueError("Frozen and evaluation backend kinds differ")
    old_info = json.loads((frozen_root / "backend.json").read_text())
    if info.get("fingerprint") != old_info.get("fingerprint"):
        raise ValueError("Frozen backend fingerprint mismatch")
    planned = preflight(config, phase="evaluation", backend_info=info, output_dir=out_dir, ancestor_dir=frozen_root)
    store = _store(config, out_dir, "evaluation", [frozen_root])
    store.json("shared", "preflight.json", planned)
    engine = Engine(config, backend, store, load_artifact(frozen_root / "probe.json"))
    try:
        engine._budget()
        engine.runtime_validation = backend.validate_runtime()
        if engine.runtime_validation.get("fingerprint") != old_info.get("fingerprint"):
            raise ValueError("Runtime fingerprint mismatch")
        if not info["synthetic"] and not (engine.runtime_validation.get("passed") is True and engine.runtime_validation.get("target_gate") is True):
            raise ValueError("Evaluation runtime validation failed")
        store.json("shared", "runtime.json", engine.runtime_validation)
        store.json("shared", "backend.json", info)
        store.json("shared", "frozen.json", frozen)
        _run_sessions(engine, "evaluation", diagnostics)
        return _finish(store, "complete", synthetic=bool(info["synthetic"]), model_calls=engine.calls,
                       gates=frozen["gates"], elapsed_seconds=time.monotonic()-engine.started)
    except (BudgetExceeded, ValueError, RuntimeError) as exc:
        try:
            engine.record(dict(type="run_status", incomplete=True, status="incomplete", reason=str(exc),
                               split="evaluation", synthetic=bool(info["synthetic"])))
        except BudgetExceeded:
            pass
        return _finish(store, "incomplete", synthetic=bool(info["synthetic"]), model_calls=engine.calls,
                       gates=frozen["gates"], incomplete_reason=str(exc), elapsed_seconds=time.monotonic()-engine.started)


def report(run_dir, out_dir=None) -> dict:
    root = Path(run_dir).resolve()
    manifest, config, records, _ = _verify(root)
    analysis_config = copy.deepcopy(config)
    analysis_config["analysis"] = dict(seed=config["statistics"]["seed"],
        primary_delay=config["diagnostics"]["primary_delay"],
        n_permutations=config["statistics"]["permutations"], n_bootstrap=config["statistics"]["bootstrap"],
        early_count=config["evaluation"]["early_rounds"], late_count=config["evaluation"]["late_rounds"],
        ni_margin=config["development"]["gates"]["preservation_margin"],
        **{k: bool(manifest.get("gates", {}).get(k) is True) for k in
           ("runtime_validation_passed", "probe_validation_passed", "online_transfer_passed", "causal_discovery_passed", "ni_preflight_passed")})
    phase = manifest.get("phase")
    expected = list(map(str, range(config[phase]["families"]))) if phase in ("development", "evaluation") else []
    observed = {str(r["family"]) for r in records if "family" in r}
    assigned = {(session, number) for session in COMPARISONS for number in range(1, config["tasks"]["rounds"]+1)}
    completed = {}
    for family in expected:
        completed[family] = {(r.get("session"), r.get("round")) for r in records
                             if r.get("type") == "turn" and str(r.get("family")) == family}
    incomplete_families = [family for family in expected if completed[family] != assigned]
    inventory = dict(planned_families=expected, observed_families=sorted(observed),
        unstarted_families=[family for family in expected if family not in observed],
        incomplete_families=incomplete_families, n_planned=len(expected),
        n_complete=len(expected)-len(incomplete_families), expected_turns_per_family=len(assigned),
        recorded_turns_per_family={family:len(completed[family]) for family in expected},
        unresolved_family_effect_bounds={family:[-1., 1.] for family in incomplete_families})
    execution_complete = manifest["status"] == "complete" and not incomplete_families
    for family in expected:
        if family not in observed:
            records.append(dict(type="family_status", family=family, split=phase, incomplete=True,
                                status="incomplete", synthetic=manifest.get("synthetic", False)))
    if not execution_complete:
        records.append(dict(type="run_status", incomplete=True, synthetic=manifest.get("synthetic", False)))
    summary = summarize_records(records, analysis_config)
    for primary in summary["primaries"].values():
        primary["validation_gates"]["execution_complete"] = execution_complete
        if not execution_complete:
            primary.update(status="unavailable", p=1., adjusted_p=1., reject=False, target_model_support=False,
                           reason="Execution is incomplete; assigned families/trials retained with unresolved bounds")
            primary.setdefault("gate_failures", []).append("execution_complete")
    summary.update(run_status=manifest["status"], source_manifest_sha256=_sha((root / "manifest.json").read_bytes()),
                   validation_gates=analysis_config["analysis"], verified_artifacts=True, family_inventory=inventory)
    destination = Path(out_dir) if out_dir is not None else root / "report"
    store = _store(config, destination, "report", [root])
    for name, payload in report_payloads(summary).items():
        store.write("compact", name, payload)
    _finish(store, "complete", synthetic=bool(manifest.get("synthetic")), source_run_status=manifest["status"])
    return summary
