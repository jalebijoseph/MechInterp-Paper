"""Independent, token-native Experiment 5 checkpoint assays.

Every continuation starts from a copied checkpoint. Only checker-verified live
source calls may construct donor/update evidence; none enters the main session.
Synthetic behavior is retained, but cannot validate source-attention blocking.
"""
from __future__ import annotations

import copy
import json
import math
import re

import numpy as np

from .probes import predict_readout
from .protocol import CELL_ORDER, derived_seed, parse_choice, strict_object
from .storage import BudgetExceeded
from .tasks import grade_peer, public_task, session_tasks


def _json(value):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _plain(value):
    if isinstance(value, np.ndarray):
        return _plain(value.tolist())
    if isinstance(value, np.generic):
        return _plain(value.item())
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _stream(context, *parts):
    return ("diagnostic", context["family"], context["session"], context["round"], *parts)


def _base(engine, context):
    return {**{k: context[k] for k in ("family", "session", "split", "round")},
            "synthetic": bool(engine.backend.info().get("synthetic", False))}


def _rates(context):
    return [s / n if n else None for s, n in zip(context["successes"], context["exposures"])]


def _winner(rates, domain):
    indices = [CELL_ORDER.index((p, domain)) for p in ("A", "B")]
    a, b = (rates[i] for i in indices)
    if a is None or b is None or not np.isfinite([a, b]).all() or np.isclose(a, b):
        return None
    return "A" if a > b else "B"


def _reference(context, domain):
    quality = context.get("audit_quality", {})
    a, b = (quality.get((p, domain)) for p in ("A", "B"))
    if a is not None and b is not None:
        if not np.isfinite([a, b]).all() or np.isclose(a, b):
            return None, "audited_tie_or_invalid"
        return ("A" if a > b else "B"), "independent_audit"
    if context.get("controlled") is True:
        return _winner(_rates(context), domain), "observed_unique_profile_conditional"
    return None, "independent_audit_unavailable"


def _positions(spans):
    return sorted({p for start, end in spans for p in range(start, end)})


def _geometry(source_spans, control_spans, checkpoint_position, max_age_gap_tokens):
    source, control = _positions(source_spans), _positions(control_spans)
    gaps = [abs(a-b) for a, b in zip(source, control)]
    valid_intervals = all(0 <= start < end <= checkpoint_position + 1
                          for start, end in [*source_spans, *control_spans])
    valid = bool(valid_intervals and source and len(source) == len(control)
                 and max(gaps, default=math.inf) <= max_age_gap_tokens)
    return dict(valid=valid, source_count=len(source), control_count=len(control),
                source_mean_age=float(np.mean([checkpoint_position-p for p in source])) if source else None,
                control_mean_age=float(np.mean([checkpoint_position-p for p in control])) if control else None,
                max_paired_age_gap=max(gaps, default=None), max_age_gap_tokens=max_age_gap_tokens,
                source_control_disjoint=not bool(set(source) & set(control)))


def control_geometry_valid(source_spans, control_spans, checkpoint_position, max_age_gap_tokens):
    """Require equal unique token counts and bounded sorted-position age gaps."""
    geometry = _geometry(source_spans, control_spans, checkpoint_position, max_age_gap_tokens)
    return bool(geometry["valid"] and geometry["source_control_disjoint"])


def route_masks(context: dict, route: str, question_start: int) -> list[dict]:
    """Absolute masks persist through question prefill and all decoding steps."""
    route = route.replace("-control", "_control")
    if route == "R0":
        return []
    if route not in ("R1", "R1_control", "R2", "R2_control"):
        raise ValueError(f"unknown retention route {route}")
    control = route.endswith("_control")
    return [dict(key_spans=[tuple(s) for s in context["control_spans" if control else "source_spans"]],
                 start_query=question_start if route.startswith("R1") else context["checkpoint_position"] + 1,
                 tag="irrelevant_control" if control else "source_evidence")]


def _actual_attention(engine, result):
    check = engine.runtime_validation.get("checks", {}).get("blocked_attention", {})
    receipt = result.get("receipt", {})
    return bool(not engine.backend.info().get("synthetic", False)
                and engine.runtime_validation.get("passed") is True
                and engine.runtime_validation.get("target_gate") is True
                and check.get("supported") is True and check.get("passed") is True
                and receipt.get("actual_source_attention_verification", True) is True)


def _same_masks(intended, realized):
    def canonical(masks):
        return [(int(mask["start_query"]), tuple(tuple(span) for span in mask["key_spans"]),
                 str(mask.get("tag", ""))) for mask in masks]
    try:
        return realized is not None and canonical(intended) == canonical(realized)
    except (TypeError, KeyError, ValueError):
        return False


def _reply_for_outcome(engine, context, task, peer, correct, stream):
    """Obtain a genuine answer using changed private input and the original world."""
    private = copy.deepcopy(task)
    answer = task["analysis"]["true_answer"]
    if not correct:
        answer = next((v for v in task["options"] if v != answer), None)
    if answer is None:
        return dict(available=False, reason="no false option", attempts=[])
    facts = private["private_facts"][peer]
    queried = [i for i, fact in enumerate(facts) if fact["subject"] == task["query_subject"]]
    if queried:
        for index in queried:
            facts[index]["value"] = answer
    elif facts:
        facts[-1] = dict(subject=task["query_subject"], value=answer)
    else:
        return dict(available=False, reason="empty packet cannot preserve fact count", attempts=[])
    attempts = []
    for attempt in range(engine.config["diagnostics"]["max_donor_attempts"]):
        response = engine.peer_reply(private, peer, stream=(*stream, attempt))
        checked = grade_peer(task, response.get("text"))
        attempts.append(dict(call_id=response.get("call_id"), response=response.get("text"),
                             checker=checked, hit_cap=bool(response.get("hit_cap", False))))
        if checked["valid"] and not checked["unknown"] and checked["correct"] == correct and not response.get("hit_cap", False):
            return dict(available=True, response=response["text"], checked=checked,
                        call_id=response.get("call_id"), private_facts=private["private_facts"][peer], attempts=attempts)
    return dict(available=False, reason="required actual source answer unavailable", attempts=attempts,
                private_facts=private["private_facts"][peer])


def build_donor(engine, context: dict, target_cell: int) -> dict:
    """Change one observed cell using new live replies; preserve all other events."""
    peer, domain = CELL_ORDER[target_cell]
    rates = _rates(context)
    competitor = (target_cell + 2) % 4
    unavailable = dict(available=False, target_cell=target_cell, donor_choice=None, recipes=[])
    if rates[target_cell] is None or rates[competitor] is None:
        return dict(unavailable, reason="donor target or competing cell unobserved")
    desired = rates[target_cell] <= rates[competitor]
    events = copy.deepcopy(context["events"])
    successes = list(context["successes"])
    recipes, changed = [], 0
    for index, event in enumerate(events):
        if event.get("purpose") != "feedback" or event.get("peer") != peer:
            continue
        task = event.get("task")
        if not task or task.get("domain") != domain:
            continue
        if event.get("kind") != "message":
            return dict(unavailable, recipes=recipes, reason="feedback is not an append-message event")
        try:
            payload = strict_object(event["content"])
        except ValueError:
            return dict(unavailable, recipes=recipes, reason="feedback JSON unavailable")
        original = grade_peer(task, payload.get("response"))
        if original["correct"] != event.get("correct") or payload.get("correct") != original["correct"]:
            return dict(unavailable, recipes=recipes, reason="original feedback/checker mismatch")
        if original["correct"] == desired and original["valid"] and not original["unknown"]:
            continue
        response = _reply_for_outcome(engine, context, task, peer, desired,
                                      _stream(context, "donor", target_cell, index))
        recipes.append(dict(event_index=index, task_id=task["id"], requested_correct=desired, **response))
        if not response["available"]:
            return dict(unavailable, recipes=recipes, reason=response["reason"])
        payload.update(response=response["response"], correct=response["checked"]["correct"])
        event.update(content=_json(payload), correct=response["checked"]["correct"])
        successes[target_cell] += int(response["checked"]["correct"]) - int(original["correct"])
        changed += 1
    revised = [s / n if n else None for s, n in zip(successes, context["exposures"])]
    choice = _winner(revised, domain)
    if not changed or choice is None:
        return dict(unavailable, recipes=recipes, reason="donor does not produce a unique revised preference")
    try:
        replay = engine.replay(events)
    except BudgetExceeded:
        raise
    except (ValueError, KeyError, TypeError, IndexError, RuntimeError) as exc:
        return dict(unavailable, recipes=recipes, reason=f"donor replay failed: {exc}")
    return dict(replay, available=True, target_cell=target_cell, donor_choice=choice,
                events=events, successes=successes, exposures=list(context["exposures"]),
                rates=revised, recipes=recipes, changed_feedback=changed)


def _delayed(engine, context, delay):
    tokens = list(context["tokens"])
    neutral = engine.backend.encode_text(" Neutral context continues.")
    if delay:
        tokens += (neutral * ((delay + len(neutral)-1)//len(neutral)))[:delay]
    return tokens


def _capture(engine, context, tokens, layer, position, *, stream, masks=(), edits=()):
    out = engine.observe(tokens, {layer: [position]}, stream=stream, masks=masks, edits=edits)
    return np.asarray(out["captures"][layer][position]), out


def _prediction(engine, context, tokens, *, stream, masks=(), edits=()):
    artifact = engine.artifact
    vector, out = _capture(engine, context, tokens, int(artifact["layer"]), len(tokens)-1,
                           stream=stream, masks=masks, edits=edits)
    return predict_readout(artifact, vector), out


def _question(engine, context, tokens, payload, *, stream, route="R0", edits=()):
    tokens = engine.backend.finish_assistant(list(tokens))
    tokens, span = engine.backend.append_message(tokens, "user", _json(payload))
    tokens, _ = engine.backend.append_message(tokens, "assistant", "", close=False)
    masks = route_masks(context, route, span[0])
    out = engine.generate(tokens, stream=stream, masks=masks, edits=edits)
    return out, masks


def _choice(engine, context, tokens, domain, *, stream, route="R0", edits=(), task=None):
    payload = public_task(task) if task else {"domain": domain}
    payload.update(event="delegation_probe", domain=domain,
                   instruction='Choose the more reliable peer for this domain. Return only {"choice":"A"} or {"choice":"B"}.')
    out, masks = _question(engine, context, tokens, payload, stream=stream, route=route, edits=edits)
    choice = parse_choice(out.get("text", ""))
    return dict(out, choice=choice, valid=choice is not None and not out.get("hit_cap", False)), masks


def _basis(engine, category, cell):
    bases = engine.artifact["bases"][category]
    basis = np.asarray(bases[cell] if cell in bases else bases[str(cell)], dtype=np.float32)
    rank = int(engine.artifact["causal_settings"]["rank"])
    if basis.ndim != 2 or basis.shape[1] < rank or rank <= 0:
        raise ValueError("selected cell basis/rank unavailable")
    return basis[:, :rank]


def _edits(engine, context, condition, target_cell, clean_later, donor_vectors):
    if condition == "I0":
        return []
    settings = engine.artifact["causal_settings"]
    if condition in ("I3", "I4", "I5", "I6", "I7") and settings["later_layer"] <= settings["layer"]:
        raise ValueError("restoration requires a distinct downstream layer")
    common = dict(position=context["checkpoint_position"], strength=float(settings["strength"]),
                  seed=derived_seed(engine.config["tasks"]["seed"], *_stream(context, "edit", target_cell, condition)))
    delete = dict(common, layer=int(settings["layer"]), kind="delete",
                  basis=_basis(engine, "delete", target_cell), tag="target_delete")
    if condition in ("I1", "I2"):
        return [dict(delete, kind="random_delete" if condition == "I2" else "delete")]
    other = target_cell ^ 1
    cell = other if condition == "I7" else target_cell
    donor = clean_later if condition in ("I3", "I5") else donor_vectors[cell]
    patch = dict(common, layer=int(settings["later_layer"]),
                 kind="random_patch" if condition in ("I5", "I6") else "patch",
                 basis=_basis(engine, "restore", cell), donor=donor,
                 tag="clean_rescue" if condition in ("I3", "I5") else f"counterfactual_cell_{cell}")
    return [delete, patch] if condition in ("I3", "I5") else [patch]


def _matching(receipt, condition, tolerance):
    edits = receipt.get("edits", [])
    if condition == "I0":
        return True, []
    if len(edits) != (2 if condition in ("I3", "I5") else 1):
        return False, []
    metrics = []
    for edit in edits:
        if not edit.get("kind", "").startswith("random_"):
            continue
        try:
            norm = abs(float(edit["norm_matching_error"])) / max(abs(float(edit["reference_realized_norm"])), 1e-12)
            dot = abs(float(edit["inner_product_matching_error"])) / max(abs(float(edit["reference_realized_inner_product"])), 1e-12)
            metrics.append(dict(relative_norm_error=norm, relative_inner_product_error=dot,
                                valid=bool(np.isfinite([norm, dot]).all() and max(norm, dot) <= tolerance)))
        except (KeyError, TypeError, ValueError):
            metrics.append(dict(valid=False, reason="missing realized matching receipt"))
    valid = all(m["valid"] for m in metrics)
    if condition == "I7":
        valid = valid and bool(edits[-1].get("realized_norm", 0) > 0)
    return bool(valid), metrics


def _specificity(engine, context, tokens, condition, target_cell, route, delay, edits):
    other_domain = CELL_ORDER[target_cell ^ 1][1]
    other, _ = _choice(engine, context, tokens, other_domain,
                       stream=_stream(context, "specificity", target_cell, delay, route, "other_domain"),
                       route=route, edits=edits)
    expected, _ = _reference(context, other_domain)
    capability, _ = _question(engine, context, tokens, dict(event="capability_probe", instruction=
        'Return A and B, each with possession and location success-rate probabilities as JSON.'),
        stream=_stream(context, "specificity", target_cell, delay, route, "capability"), route=route, edits=edits)
    try:
        parsed = strict_object(capability.get("text", ""))
        if set(parsed) != {"A", "B"} or any(set(parsed[p]) != {"possession", "location"} for p in ("A", "B")):
            raise ValueError("four-cell schema")
        values = [parsed[p][d] for p, d in CELL_ORDER]
        valid = bool(all(type(v) in (int, float) and np.isfinite(v) and 0 <= v <= 1 for v in values)
                     and not capability.get("hit_cap", False))
    except (ValueError, TypeError, KeyError):
        values, valid = [None]*4, False
    rates = _rates(context)
    non_target = [i for i in range(4) if i != target_cell and rates[i] is not None]
    error = float(np.mean([abs(values[i]-rates[i]) for i in non_target])) if valid and non_target else None
    roster = {}
    for event in context["events"]:
        if event.get("purpose") == "roster" and event.get("kind") == "message":
            try:
                roster = strict_object(event["content"]).get("roster", roster)
            except ValueError:
                pass
    peer = CELL_ORDER[target_cell][0]
    identity = dict(text=None, choice=None, valid=False, hit_cap=False, call_id=None)
    if peer in roster:
        out, _ = _question(engine, context, tokens, dict(event="identity_probe", peer_name=roster[peer],
            instruction='Return the slot for this name as {"choice":"A"} or {"choice":"B"}.'),
            stream=_stream(context, "specificity", target_cell, delay, route, "identity"), route=route, edits=edits)
        identity = dict(out, choice=parse_choice(out.get("text", "")))
        identity["valid"] = identity["choice"] is not None and not out.get("hit_cap", False)
    return dict(other_domain_choice=other["choice"], other_domain_valid=other["valid"],
                other_domain_correct=bool(other["valid"] and other["choice"] == expected) if expected else None,
                non_target_prediction=values, non_target_valid=valid, non_target_error=error,
                literal_correct=bool(identity["valid"] and identity["choice"] == peer) if peer in roster else None,
                identity_valid=identity["valid"], specificity_calls={"other_domain": other.get("call_id"),
                "capability": capability.get("call_id"), "identity": identity.get("call_id")},
                specificity_outputs={"other_domain": other.get("text"), "capability": capability.get("text"),
                                     "identity": identity.get("text")},
                specificity_hit_cap={"other_domain": other.get("hit_cap", False),
                    "capability": capability.get("hit_cap", False), "identity": identity.get("hit_cap", False)})


def _swap_choice_tokens(backend, tokens, old, new):
    """Retokenize only tokens covering the changed label; retain native suffix."""
    text = backend.decode(tokens)
    payload, _ = json.JSONDecoder().raw_decode(text.lstrip())
    if payload != {"choice": old}:
        raise ValueError("past native choice disagrees with its peer metadata")
    match = re.search(r'"choice"\s*:\s*"([AB])"', text)
    if not match:
        raise ValueError("past choice label unavailable")
    offset = match.start(1)
    start, end, prefix_size = 0, len(tokens), 0
    for index in range(len(tokens)+1):
        prefix = backend.decode(tokens[:index])
        if text.startswith(prefix) and len(prefix) <= offset:
            start, prefix_size = index, len(prefix)
        if text.startswith(prefix) and len(prefix) >= offset + 1:
            end = index
            break
    segment = backend.decode(tokens[start:end])
    local = offset-prefix_size
    changed = segment[:local] + new + segment[local+1:]
    result = list(tokens[:start]) + backend.encode_text(changed) + list(tokens[end:])
    if backend.decode(result) != text[:offset] + new + text[offset+1:]:
        raise ValueError("native label replacement could not preserve surrounding text")
    return result


def _attribution(engine, context, domain, emit):
    record = dict(type="diagnostic", diagnostic="attribution", domain=domain,
                  attribution_applicable=True, status="unavailable")
    rates = _rates(context)
    cells = [CELL_ORDER.index((p, domain)) for p in ("A", "B")]
    if any(rates[i] is None for i in cells) or np.isclose(rates[cells[0]], rates[cells[1]]):
        emit(dict(record, attribution_applicable=False, status="not_applicable",
                  reason="attribution cells unobserved or equal"))
        return
    events, changed = copy.deepcopy(context["events"]), 0
    try:
        for event in events:
            task = event.get("task")
            if not task or task.get("domain") != domain or event.get("peer") not in ("A", "B"):
                continue
            if event.get("purpose") in ("feedback", "choice_header", "choice"):
                old = event["peer"]
                new = "B" if old == "A" else "A"
                if event["kind"] == "message" and event.get("purpose") != "choice_header":
                    payload = strict_object(event["content"])
                    if "peer" in payload:
                        payload["peer"] = new
                    if "choice" in payload:
                        payload["choice"] = new
                    event["content"] = _json(payload)
                elif event.get("purpose") == "choice":
                    event["tokens"] = _swap_choice_tokens(engine.backend, event["tokens"], old, new)
                event["peer"] = new
                changed += event.get("purpose") == "feedback"
    except BudgetExceeded:
        raise
    except (ValueError, KeyError, TypeError, IndexError, RuntimeError) as exc:
        emit(dict(record, reason=str(exc)))
        return
    if not changed:
        emit(dict(record, reason="no attributable feedback events"))
        return
    swapped = list(rates)
    swapped[cells[0]], swapped[cells[1]] = swapped[cells[1]], swapped[cells[0]]
    try:
        replay = engine.replay(events)
        original, _ = _prediction(engine, context, context["tokens"], stream=_stream(context, "attribution", "original"))
        revised, _ = _prediction(engine, replay, replay["tokens"], stream=_stream(context, "attribution", "swapped"))
        indices = [i for i, q in enumerate(swapped) if q is not None]
        old, new = original["prediction"], revised["prediction"]
        if not np.isfinite(np.asarray([old[indices], new[indices]])).all():
            raise ValueError("nonfinite frozen predictions")
        improvement = float(np.mean([abs(old[i]-swapped[i])-abs(new[i]-swapped[i]) for i in indices]))
        emit(dict(record, status="available", target=swapped, original_target=rates,
                  prediction=new, unchanged_prediction=old, attribution_error_improvement=improvement,
                  constant_total=True, changed_feedback=changed,
                  history=engine.store.history(replay["tokens"])))
    except BudgetExceeded:
        raise
    except (ValueError, KeyError, TypeError, IndexError, RuntimeError) as exc:
        emit(dict(record, reason=str(exc)))


def _updating(engine, context, task, target_cell, emit):
    q = _rates(context)[target_cell]
    peer, domain = CELL_ORDER[target_cell]
    record = dict(type="update", target_cell=target_cell, domain=domain, controlled_diagnostic=True,
                  q_initial=q, q_reversal=None, q_confirming=None,
                  h_initial=None, h_reversal=None, h_confirming=None, status="unavailable")
    if q is None or np.isclose(q, .5):
        emit(dict(record, reason="initial cell unobserved or has no directional evidence"))
        return
    try:
        initial, _ = _prediction(engine, context, context["tokens"], stream=_stream(context, "update", target_cell, "initial"))
        record["h_initial"] = float(initial["prediction"][target_cell])
        fresh_config = copy.deepcopy(engine.config)
        fresh_config["tasks"]["seed"] = derived_seed(engine.config["tasks"]["seed"],
            *_stream(context, "update_worlds", target_cell))
        fresh_config["audits"]["cases_per_domain"] = max(4, fresh_config["audits"]["cases_per_domain"])
        bank = [row for row in session_tasks(fresh_config, context["split"], 0,
                    "equal_information", audit=True) if row["domain"] == domain][:4]
        recipes = {}
        for branch, desired in (("reversal", q < .5), ("confirming", q > .5)):
            events = copy.deepcopy(context["events"])
            branch_recipes = []
            for index in range(4):
                fresh = bank[index]
                response = _reply_for_outcome(engine, context, fresh, peer, desired,
                    _stream(context, "update", target_cell, branch, index))
                branch_recipes.append(response)
                if not response["available"]:
                    emit(dict(record, reason=response["reason"], recipes={**recipes, branch: branch_recipes}))
                    return
                events.append(dict(kind="finish", source=False))
                events.append(dict(kind="message", role="user", content=_json(public_task(fresh)),
                                   close=True, source=True, purpose="task", task=fresh))
                events.append(dict(kind="message", role="user", content=_json(dict(event="peer_feedback",
                    peer=peer, domain=domain, response=response["response"], correct=response["checked"]["correct"])),
                    close=True, source=True, purpose="feedback", peer=peer, task=fresh,
                    correct=response["checked"]["correct"]))
                events.append(dict(kind="message", role="assistant", content="Recorded.", close=False,
                                   source=False, purpose="receipt"))
            replay = engine.replay(events)
            prediction, _ = _prediction(engine, replay, replay["tokens"],
                                       stream=_stream(context, "update", target_cell, branch, "readout"))
            record[f"q_{branch}"] = (context["successes"][target_cell] + 4*int(desired)) / (context["exposures"][target_cell]+4)
            record[f"h_{branch}"] = float(prediction["prediction"][target_cell])
            record[f"history_{branch}"] = engine.store.history(replay["tokens"])
            recipes[branch] = branch_recipes
        if not np.isfinite([record[k] for k in ("h_initial", "h_reversal", "h_confirming")]).all():
            raise ValueError("nonfinite frozen updating estimates")
        emit(dict(record, status="available", recipes=recipes, observations_per_branch=4,
                  fresh_task_policy="independently seeded fresh audit worlds matched across branches",
                  world_seed=fresh_config["tasks"]["seed"], task_ids=[row["id"] for row in bank]))
    except (ValueError, KeyError) as exc:
        emit(dict(record, reason=str(exc)))


def run_checkpoint(engine, context: dict, task: dict, target_cell: int | None = None, *,
                   conditions=None, delays=None, routes=None) -> dict:
    """Execute registered forks without advancing or modifying the main state.

    Full I0-I7 is assigned at primary delay under R0/R2. Remaining requested
    route/delay combinations retain I0. All unavailable/capped cases are logged.
    """
    settings = engine.config["diagnostics"]
    conditions = list(settings["conditions"] if conditions is None else conditions)
    delays = list(settings["delays"] if delays is None else delays)
    routes = [r.replace("-control", "_control") for r in (settings["routes"] if routes is None else routes)]
    target_cell = CELL_ORDER.index(("A", task["domain"])) if target_cell is None else target_cell
    if type(target_cell) is not int or not 0 <= target_cell < 4:
        raise ValueError("target_cell must be an index in CELL_ORDER")
    if any(c not in {f"I{i}" for i in range(8)} for c in conditions):
        raise ValueError("unknown causal condition")
    if any(type(d) is not int or d < 0 for d in delays):
        raise ValueError("delays must be nonnegative token counts")
    domain = CELL_ORDER[target_cell][1]
    records = []

    def emit(record):
        plain = _plain(dict(_base(engine, context), **record))
        engine.record(plain)
        records.append(plain)

    geometry = _geometry(context["source_spans"], context["control_spans"],
                         context["checkpoint_position"], settings["max_age_gap_tokens"])
    route_valid = bool(geometry["valid"] and geometry["source_control_disjoint"])
    reference, reference_source = _reference(context, domain)
    donors, donor_vectors = {}, {}
    clean_later = None
    need_target = any(c in conditions for c in ("I4", "I6"))
    for cell in ([target_cell] if need_target else []) + ([target_cell ^ 1] if "I7" in conditions else []):
        donor = build_donor(engine, context, cell)
        donors[cell] = donor
        emit(dict(type="donor", target_cell=cell, status="available" if donor["available"] else "unavailable",
                  donor_choice=donor.get("donor_choice"), recipes=donor.get("recipes", []),
                  reason=donor.get("reason"), successes=donor.get("successes"), exposures=donor.get("exposures"),
                  history=engine.store.history(donor["tokens"]) if donor["available"] else None))
        if donor["available"]:
            try:
                layer = int(engine.artifact["causal_settings"]["later_layer"])
                vector, _ = _capture(engine, donor, donor["tokens"], layer, donor["checkpoint_position"],
                                      stream=_stream(context, "donor_capture", cell))
                donor_vectors[cell] = vector
            except (ValueError, KeyError) as exc:
                donor.update(available=False, reason=str(exc))
    if any(c in conditions for c in ("I3", "I5")):
        try:
            layer = int(engine.artifact["causal_settings"]["later_layer"])
            clean_later, _ = _capture(engine, context, context["tokens"], layer, context["checkpoint_position"],
                                      stream=_stream(context, "clean_rescue_capture", target_cell))
        except (ValueError, KeyError):
            pass
    rates = _rates(context)
    for delay in delays:
        tokens = _delayed(engine, context, delay)
        for route in routes:
            route_masks(context, route, len(tokens))  # Validate even if no conditions assigned.
            assigned = conditions if delay == settings["primary_delay"] and route in ("R0", "R2") else ["I0"]
            fork_id = f"{context['session']}:{context['round']}:{target_cell}:{delay}"
            retention = dict(type="retention", fork_id=fork_id, target_cell=target_cell, delay=delay,
                             route=route, prediction=None, raw=None, target=rates,
                             mask=[q is not None for q in rates], status="unavailable")
            try:
                read_masks = route_masks(context, route, len(tokens))
                prediction, observed = _prediction(engine, context, tokens, masks=read_masks,
                    stream=_stream(context, "retention", target_cell, delay, route))
                retention.update(prediction=prediction["prediction"], raw=prediction["raw"],
                                 status="available", receipt=observed.get("receipt"))
            except (ValueError, KeyError) as exc:
                retention["reason"] = str(exc)
            emit(retention)
            for condition in assigned:
                specificity = dict(other_domain_choice=None, other_domain_valid=False,
                    other_domain_correct=None, non_target_prediction=[None]*4, non_target_valid=False,
                    non_target_error=None, literal_correct=None, identity_valid=False)
                record = dict(type="causal", fork_id=fork_id, target_cell=target_cell, domain=domain,
                    delay=delay, route=route, condition=condition, valid=False, choice=None,
                    correct=False, reference_choice=reference, reference_source=reference_source,
                    donor_choice=donors.get(target_cell, {}).get("donor_choice"),
                    other_domain_donor_choice=donors.get(target_cell ^ 1, {}).get("donor_choice"),
                    control_match_valid=False, route_match_valid=route_valid,
                    source_attention_valid=False, geometry=geometry, status="unavailable",
                    runtime_attention_validation=engine.runtime_validation.get("checks", {}).get("blocked_attention"))
                try:
                    donor_cell = target_cell ^ 1 if condition == "I7" else target_cell
                    if condition in ("I4", "I6", "I7") and donor_cell not in donor_vectors:
                        raise ValueError(donors.get(donor_cell, {}).get("reason", "counterfactual donor unavailable"))
                    if condition in ("I3", "I5") and clean_later is None:
                        raise ValueError("independent clean later-layer donor unavailable")
                    edits = _edits(engine, context, condition, target_cell, clean_later, donor_vectors)
                    # CRN pairing deliberately excludes condition from answer streams.
                    result, masks = _choice(engine, context, tokens, domain,
                        stream=_stream(context, "causal_choice", target_cell, delay, route),
                        route=route, edits=edits, task=task)
                    match, metrics = _matching(result.get("receipt", {}), condition,
                                              engine.config["development"]["gates"]["matching_error"])
                    masks_match = _same_masks(masks, result.get("receipt", {}).get("masks"))
                    record.update(choice=result["choice"], valid=result["valid"],
                        correct=bool(result["valid"] and result["choice"] == reference),
                        status="available" if reference else "unavailable", call_id=result.get("call_id"),
                        hit_cap=bool(result.get("hit_cap", False)), output=result.get("text"),
                        receipt=result.get("receipt"), masks=masks, matching=metrics,
                        control_match_valid=match, route_match_valid=bool(route_valid and masks_match),
                        source_attention_valid=bool(masks_match and _actual_attention(engine, result)),
                        edit_recipe=[{k:v for k,v in e.items() if k not in ("basis", "donor")} for e in edits])
                    if not match:
                        record.update(status="inconclusive", reason="realized edit matching or I7 nonzero check failed")
                    elif reference is None:
                        record["reason"] = "unique independently audited quality reference unavailable"
                    if not record["route_match_valid"]:
                        record.update(status="inconclusive", reason="source/control geometry or realized route mask mismatch")
                    specificity = _specificity(engine, context, tokens, condition, target_cell, route, delay, edits)
                except (ValueError, KeyError, IndexError) as exc:
                    record.update(reason=str(exc))
                record.update(specificity)
                record["other_domain_donor_consistent"] = bool(specificity["other_domain_valid"] and
                    specificity["other_domain_choice"] == record["other_domain_donor_choice"]) if record["other_domain_donor_choice"] else None
                emit(dict(type="specificity", fork_id=fork_id, target_cell=target_cell,
                          domain=domain, route=route, delay=delay, condition=condition,
                          status="available" if specificity.get("specificity_calls") else "unavailable",
                          **specificity))
                emit(record)
    _attribution(engine, context, domain, emit)
    _updating(engine, context, task, target_cell, emit)
    return dict(target_cell=target_cell, n_records=len(records), records=records,
                available=sum(r.get("status") == "available" for r in records))
