"""Family-level Experiment 5 inference and compact, explicit reports.

Event schema (all events retain family/session/split/synthetic provenance):
* turn: round, phase, domain, peer, valid, correct, reward; optional four-cell
  observed_successes/observed_exposures/observed_unknown arrays.
* audit: independent phase/domain/peer, valid/correct (one fresh audit trial).
* readout: target/prediction/mask and frozen identity_blind_prediction arrays;
  attribution_error_improvement is a separately matched constant-total control.
* update: q_initial/q_reversal/q_confirming and h_initial/h_reversal/h_confirming.
* causal: fork_id (identical across matched forks), route R0/R2/R2_control,
  condition I0..I7, target_cell, valid, choice, correct, donor_choice; independent
  literal_correct/other_domain_correct/non_target_error specificity outcomes.

Missing primary components remain unavailable (p=1). Invalid actions score zero;
missing matched forks do not disappear through survivor-only averaging.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import json
from pathlib import Path

import numpy as np

PRIMARY_IDS = tuple(f"P{i}" for i in range(1, 8))


def _unavailable(reason, **fields):
    return dict(status="unavailable", p=1., estimate=None, ci=None,
                n_families=0, reason=reason, **fields)


def family_test(values: list[float], *, seed: int = 20261108,
                n_permutations: int = 50000, n_bootstrap: int = 100000,
                margin: float = 0) -> dict:
    """One-sided paired-family mean sign-flip test against mean <= margin.

    Exactness requires independent family contrasts with symmetric distributions
    about the null boundary. This is not a distribution-free test of any mean.
    Bootstrap CIs resample whole families, never turns or four cells.
    """
    x = np.asarray(values, dtype=float)
    if x.ndim != 1 or not np.isfinite(x).all() or not np.isfinite(margin):
        raise ValueError("family contrasts and null boundary must be finite")
    if n_permutations < 1 or n_bootstrap < 1:
        raise ValueError("resampling counts must be positive")
    n = len(x)
    if n < 2:
        return _unavailable("at least two independent families are required", observed_families=n)
    centered = x - margin
    observed = float(centered.mean())
    tolerance = 1e-12 * max(1., float(np.max(np.abs(centered))))
    rng = np.random.default_rng(seed)
    exact = n <= 20
    assignments = 2 ** n if exact else n_permutations
    batch = max(1, min(4096, 1_000_000 // n))
    extreme = 0
    for start in range(0, assignments, batch):
        size = min(batch, assignments - start)
        if exact:
            integers = np.arange(start, start + size, dtype=np.uint64)
            bits = (integers[:, None] >> np.arange(n, dtype=np.uint64)) & 1
            signs = bits.astype(float) * 2 - 1
        else:
            signs = rng.integers(0, 2, (size, n)) * 2 - 1
        statistics = np.mean(signs * centered, axis=1)
        extreme += int(np.count_nonzero(statistics >= observed - tolerance))
    p = extreme / assignments if exact else (extreme + 1) / (assignments + 1)
    # Independent stream keeps CI draws stable when permutation count changes.
    boot_rng = np.random.default_rng(np.random.SeedSequence([seed, 1]))
    boot = np.empty(n_bootstrap)
    for start in range(0, n_bootstrap, batch):
        size = min(batch, n_bootstrap - start)
        boot[start:start + size] = x[boot_rng.integers(0, n, (size, n))].mean(axis=1)
    return dict(status="available", estimate=float(x.mean()), ci=np.quantile(boot, [.025, .975]).tolist(),
                p=float(p), n_families=n, method="exact_sign_flip" if exact else "monte_carlo_sign_flip",
                assignments=assignments, bootstrap_resamples=n_bootstrap, seed=seed,
                null_boundary=float(margin), alternative="mean > null_boundary",
                assumptions=["independent source families", "paired contrasts symmetric at null boundary"],
                interval="95% percentile whole-family bootstrap")


def noninferiority_test(values: list[float], *, margin: float = .05, **kwargs) -> dict:
    """Separate one-sided boundary-shift NI procedure under stated symmetry.

    Calling this procedure is not evidence that study-specific null calibration
    has passed. The registry requires that preflight before P4 can be available.
    A nonsignificant inferiority test is never interpreted as preservation.
    """
    if not np.isfinite(margin) or margin <= 0:
        raise ValueError("noninferiority loss margin must be positive")
    result = family_test(values, margin=-margin, **kwargs)
    result.update(procedure="paired_boundary_sign_flip_noninferiority", loss_margin=margin,
                  preflight_required=True)
    return result


def holm(p_values: dict[str, float], alpha: float = .05) -> dict:
    """Holm step-down correction over the fixed seven-entry primary family."""
    if set(p_values) - set(PRIMARY_IDS) or not 0 < alpha < 1:
        raise ValueError("only frozen P1..P7 entries and 0<alpha<1 are supported")
    p = {key: float(p_values.get(key, 1.)) for key in PRIMARY_IDS}
    if any(not np.isfinite(v) or not 0 <= v <= 1 for v in p.values()):
        raise ValueError("p values must be finite in [0,1]")
    ordered = sorted(p, key=lambda k: (p[k], k))
    out, maximum, continuing = {}, 0., True
    for i, key in enumerate(ordered):
        maximum = max(maximum, (len(ordered) - i) * p[key])
        reject = continuing and p[key] <= alpha / (len(ordered) - i)
        continuing = bool(reject)
        out[key] = dict(p=p[key], adjusted_p=min(1., maximum), reject=bool(reject), alpha=alpha)
    return {key: out[key] for key in PRIMARY_IDS}


def _event(record):
    return record.get("type", record.get("kind"))


def _mean(values):
    return float(np.mean(values)) if values else None


def _jsonable(value):
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(v) for v in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def summarize_records(records: list[dict], config: dict) -> dict:
    """Aggregate every matched contrast within source family before inference.

    Proposed defaults: first/last quarter windows (configured counts override;
    explicit round lists take precedence), absolute probability error, and
    5-point NI loss margin. These must be frozen
    prospectively. No binary .6 threshold is used on spontaneous online records.
    """
    settings = config.get("analysis", {})
    sampling = dict(seed=int(settings.get("seed", 20261108)),
                    n_permutations=int(settings.get("n_permutations", 50000)),
                    n_bootstrap=int(settings.get("n_bootstrap", 100000)))
    online = [r for r in records if r.get("split") not in ("train", "development", "dev", "calibration")
              and not r.get("controlled", False)]
    observed_families = sorted({str(r["family"]) for r in online if "family" in r
                                and _event(r) not in ("family_status", "run_status")})
    if "expected_families" in settings:
        families = list(dict.fromkeys(str(f) for f in settings["expected_families"]))
    else:
        families = sorted({str(r["family"]) for r in online if "family" in r})
        expected_count = int(config.get("evaluation", {}).get("families", len(families)))
        unknown = 1
        while len(families) < expected_count:
            placeholder = f"__unknown_family_{unknown}__"
            if placeholder not in families:
                families.append(placeholder)
            unknown += 1
    family_set = set(families)
    missing_families = [f for f in families if f not in observed_families]
    inference_records = [r for r in online if str(r.get("family")) in family_set]
    primary_delay = int(settings.get("primary_delay", config.get("diagnostics", {}).get("primary_delay", 512)))
    synthetic = any(r.get("synthetic", False) for r in records)
    execution_incomplete = bool(missing_families) or any(
        _event(r) in ("run_status", "family_status")
        and (r.get("incomplete") is True or r.get("uncompleted") is True or r.get("completed") is False
             or r.get("status") in ("incomplete", "error", "uncompleted", "failed")) for r in records)
    incomplete = any(r.get("incomplete", False) or r.get("status") in ("incomplete", "unavailable", "error") for r in records)

    def infer(grouped, reason="required matched observations unavailable", missing=0, ni=False):
        # A family with no retained measurement must not vanish from the registry.
        missing += sum(not grouped.get(family) for family in families)
        effects = {str(k): float(np.mean(v)) for k, v in grouped.items() if v and str(k) in family_set}
        if missing:
            return _unavailable(reason, missing_measurements=missing,
                                family_effects=effects,
                                unresolved_effect_bounds=[-1., 1.])
        result = (noninferiority_test(list(effects.values()), margin=float(settings.get("ni_margin", .05)), **sampling)
                  if ni else family_test(list(effects.values()), **sampling))
        result["family_effects"] = effects
        return result

    def conjunction(components):
        available = all(c["status"] == "available" for c in components.values())
        return dict(status="available" if available else "unavailable",
                    p=max(c["p"] for c in components.values()), components=components)

    # P1: all cells retained; unobserved cells excluded from numerical error only.
    prediction, attribution = defaultdict(list), defaultdict(list)
    missing_prediction = missing_attribution = 0
    readouts = [r for r in inference_records if _event(r) == "readout"]
    for r in readouts:
        f = str(r["family"])
        if not all(k in r for k in ("target", "prediction", "mask", "identity_blind_prediction")):
            missing_prediction += 1
        else:
            target, pred, blind = (np.asarray(r[k], float) for k in ("target", "prediction", "identity_blind_prediction"))
            mask = np.asarray(r["mask"], bool)
            if target.shape != (4,) or pred.shape != (4,) or blind.shape != (4,) or mask.shape != (4,):
                missing_prediction += 1
            elif mask.any() and np.isfinite(np.stack([target, pred, blind])[:, mask]).all():
                prediction[f].append(float(np.mean(np.abs(blind[mask] - target[mask]) - np.abs(np.clip(pred[mask], 0, 1) - target[mask]))))
            elif mask.any():
                missing_prediction += 1
    # Attribution diagnostics can be separate matched constant-total assays.
    attribution_assays = [r for r in inference_records
                          if (_event(r) == "diagnostic" and r.get("diagnostic") == "attribution")
                          or (_event(r) == "readout" and
                              ("attribution_error_improvement" in r or "attribution_applicable" in r))]
    for r in attribution_assays:
        if r.get("attribution_applicable") is False:
            continue
        try:
            improvement = float(r.get("attribution_error_improvement"))
        except (TypeError, ValueError):
            missing_attribution += 1
            continue
        if not np.isfinite(improvement):
            missing_attribution += 1
        else:
            attribution[str(r["family"])].append(improvement)
    primaries = {"P1": conjunction({"prediction_improvement": infer(prediction, missing=missing_prediction),
                                    "attribution_control": infer(attribution, missing=missing_attribution)})}

    # P2: independently audited quality; no main observed-rate proxy is accepted.
    audit_counts = defaultdict(lambda: [0., 0.])
    for r in online:
        if _event(r) != "audit" or r.get("peer") not in ("A", "B"):
            continue
        key = (str(r["family"]), str(r.get("session", "")),
               str(r.get("phase", "stationary")), r.get("domain"), r["peer"])
        if "successes" in r and "exposures" in r:
            successes, exposures = float(r["successes"]), float(r["exposures"])
            if not np.isfinite([successes, exposures]).all() or not 0 <= successes <= exposures:
                raise ValueError("audit counts must satisfy 0 <= successes <= exposures")
            audit_counts[key][0] += successes
            audit_counts[key][1] += exposures
        else:
            audit_counts[key][0] += bool(r.get("valid", False)) and bool(r.get("correct", False))
            audit_counts[key][1] += 1
    audit_quality = {key: count[0] / count[1] for key, count in audit_counts.items() if count[1] > 0}
    turns = [r for r in online if _event(r) == "turn"]
    turn_groups = defaultdict(list)
    for r in turns:
        if str(r["family"]) in family_set:
            turn_groups[(str(r["family"]), str(r.get("session", "")), str(r.get("phase", "stationary")))].append(r)
    improvements, late_random = defaultdict(list), defaultdict(list)
    learning_missing = 0
    for (family, session, phase), rows in turn_groups.items():
        if phase.lower() in ("equal", "baseline", "reversal", "reversed", "post_reversal") or any(r.get("stationary") is False for r in rows):
            continue
        rounds = sorted({int(r["round"]) for r in rows if "round" in r})
        window = max(1, len(rounds) // 4)
        early = set(settings.get("early_rounds", rounds[:int(settings.get("early_count", window))]))
        late = set(settings.get("late_rounds", rounds[-int(settings.get("late_count", window)):]))
        scores = {"early": [], "late": []}
        group_missing = False
        for r in rows:
            memberships = [name for name, selected in (("early", early), ("late", late)) if int(r.get("round", -1)) in selected]
            if not memberships:
                continue
            qa = audit_quality.get((family, session, phase, r.get("domain"), "A"))
            qb = audit_quality.get((family, session, phase, r.get("domain"), "B"))
            if qa is None or qb is None:
                group_missing = True
                continue
            if np.isclose(qa, qb):
                continue
            chosen = qa if r.get("peer") == "A" else qb
            score = (chosen - min(qa, qb)) / abs(qa - qb) if r.get("valid", False) and r.get("peer") in ("A", "B") else 0.
            for name in memberships:
                scores[name].append(float(score))
        if group_missing or not scores["early"] or not scores["late"] or early & late:
            learning_missing += 1
        else:
            improvements[family].append(float(np.mean(scores["late"]) - np.mean(scores["early"])))
            late_random[family].append(float(np.mean(scores["late"]) - .5))
    primaries["P2"] = conjunction({"late_minus_early": infer(improvements, missing=learning_missing),
                                   "late_above_random": infer(late_random, missing=learning_missing)})

    # P3: absolute convergence and signed branch separation are distinct tests.
    absolute, separation = defaultdict(list), defaultdict(list)
    update_missing = 0
    for r in inference_records:
        if _event(r) != "update":
            continue
        keys = ("q_initial", "q_reversal", "q_confirming", "h_initial", "h_reversal", "h_confirming")
        if not all(k in r and r[k] is not None and np.isfinite(r[k]) for k in keys):
            update_missing += 1
            continue
        _, qr, qc, hi, hr, hc = (float(r[k]) for k in keys)
        f = str(r["family"])
        absolute[f].append(abs(hi - qr) - abs(hr - qr))
        separation[f].append(float(np.sign(qr - qc) * (hr - hc)))
    primaries["P3"] = conjunction({"absolute_improvement": infer(absolute, missing=update_missing),
                                   "branch_separation": infer(separation, missing=update_missing)})

    causal = [r for r in online if _event(r) == "causal"]
    primary_causal = [r for r in causal if str(r["family"]) in family_set and r.get("delay") == primary_delay]
    forks = defaultdict(lambda: defaultdict(list))
    for r in primary_causal:
        fork_id = r.get("fork_id", r.get("checkpoint", r.get("round", "unspecified")))
        key = (str(r["family"]), str(r.get("session", "")), str(fork_id),
               str(r.get("target_cell", "")), str(r.get("domain", "")))
        route = r.get("route")
        if route in ("R2-control", "R2C") or (route == "R2" and r.get("route_control")):
            route = "R2_control"
        forks[key][(route, r.get("condition"))].append(r)

    def accuracy(row, donor=None):
        if not row.get("valid", False):
            return 0.
        if donor is not None:
            return float(row.get("choice") == donor)
        return float(bool(row.get("correct", False)))

    def contrast(route, left, right, *, direction=1., donor=False):
        groups, missing = defaultdict(list), 0
        for key, cells in forks.items():
            # Each registered fork must provide both specified conditions.
            a, b = cells.get((route, left), []), cells.get((route, right), [])
            if not a or not b:
                missing += 1
                continue
            target = next((r.get("donor_choice") for r in a if r.get("donor_choice") in ("A", "B")), None) if donor else None
            if donor and target is None:
                missing += 1
                continue
            groups[key[0]].append(direction * (np.mean([accuracy(r, target) for r in a]) -
                                               np.mean([accuracy(r, target) for r in b])))
        return infer(groups, missing=missing)

    route_difference, route_chance = defaultdict(list), defaultdict(list)
    route_missing = 0
    for key, cells in forks.items():
        a, b = cells.get(("R2", "I0"), []), cells.get(("R2_control", "I0"), [])
        if not a or not b:
            route_missing += 1
            continue
        route_difference[key[0]].append(float(np.mean([accuracy(r) for r in a]) - np.mean([accuracy(r) for r in b])))
        route_chance[key[0]].append(float(np.mean([accuracy(r) for r in a]) - .5))
    ni = infer(route_difference, missing=route_missing, ni=True)
    if settings.get("ni_preflight_passed") is not True:
        ni = _unavailable("noninferiority null-simulation preflight not frozen", descriptive_test=ni)
    primaries["P4"] = conjunction({"route_noninferiority": ni,
                                   "route_above_chance": infer(route_chance, missing=route_missing)})
    primaries["P5"] = conjunction({f"{route}_loss_vs_{control}": contrast(route, "I1", control, direction=-1)
                                   for route in ("R0", "R2") for control in ("I0", "I2")})
    primaries["P6"] = conjunction({f"{route}_donor_vs_{control}": contrast(route, "I4", control, donor=True)
                                   for route in ("R0", "R2") for control in ("I0", "I6")})
    primaries["P7"] = conjunction({"rescue_vs_damaged": contrast("R2", "I3", "I1"),
                                   "rescue_vs_random": contrast("R2", "I3", "I5")})
    # Realized dtype/mask receipts govern interpretation, not intended geometry.
    # A failed control stays in the inventory and invalidates its primary; it is
    # never discarded so that the remaining controls can become significant.
    required_conditions = {"P4": {"I0"}, "P5": {"I0", "I1", "I2"},
                           "P6": {"I0", "I4", "I6"}, "P7": {"I1", "I3", "I5"}}
    for primary, conditions in required_conditions.items():
        rows = [r for r in primary_causal if r.get("condition") in conditions
                and (primary not in ("P4", "P7") or str(r.get("route", "")).startswith("R2"))]
        failed = []
        for field in ("control_match_valid", "route_match_valid", "source_attention_valid"):
            if any(r.get(field) is not True for r in rows):
                failed.append(field)
        if failed:
            entry = primaries[primary]
            entry.update(status="unavailable", p=1., gate_failures=failed,
                         reason="realized intervention/control or route validation failed; all cases retained")
    prerequisite_fields = {
        "P1": ("runtime_validation_passed", "probe_validation_passed", "online_transfer_passed"),
        "P2": ("runtime_validation_passed",),
        "P3": ("runtime_validation_passed", "probe_validation_passed", "online_transfer_passed"),
        "P4": ("runtime_validation_passed", "ni_preflight_passed"),
        **{p: ("runtime_validation_passed", "probe_validation_passed", "online_transfer_passed",
               "causal_discovery_passed") for p in ("P5", "P6", "P7")},
    }
    for primary, fields in prerequisite_fields.items():
        entry = primaries[primary]
        entry["validation_gates"] = {field: settings.get(field) is True for field in fields}
        failed = [field for field, passed in entry["validation_gates"].items() if not passed]
        if failed:
            entry.update(status="unavailable", p=1.,
                         gate_failures=entry.get("gate_failures", []) + failed,
                         reason="required validation prerequisites missing or failed; descriptive components retained")
    if execution_incomplete:
        for entry in primaries.values():
            entry.update(status="unavailable", p=1., execution_incomplete=True,
                         unresolved_effect_bounds=[-1., 1.],
                         reason="execution incomplete; assigned families/trials unresolved; descriptive components retained")
    correction = holm({key: value["p"] for key, value in primaries.items()})
    for key in PRIMARY_IDS:
        primaries[key].update(correction[key])
        primaries[key]["synthetic"] = synthetic
        primaries[key]["target_model_support"] = bool(correction[key]["reject"] and not synthetic and primaries[key]["status"] == "available")
    incomplete = incomplete or execution_incomplete or any(p["status"] != "available" for p in primaries.values())

    counts = Counter(r.get("peer") if r.get("valid", False) and r.get("peer") in ("A", "B") else "invalid" for r in turns)
    n = len(turns)
    allocation = {"n_turns": n, "unconditional": {p: counts[p] / n if n else None for p in ("A", "B", "invalid")},
                  "conditional_valid": {p: counts[p] / (counts["A"] + counts["B"]) if counts["A"] + counts["B"] else None for p in ("A", "B")}}
    snapshots = {}
    trajectories = []
    for r in turns:
        key = (str(r["family"]), str(r.get("session", "")))
        if key not in snapshots or int(r.get("round", -1)) >= int(snapshots[key].get("round", -1)):
            snapshots[key] = r
        trajectories.append({k: r.get(k) for k in ("family", "session", "round", "phase", "domain", "peer", "valid", "correct", "reward", "synthetic",
                                                     "observed_successes", "observed_exposures", "observed_unknown")})
    cells = []
    for (family, session), r in snapshots.items():
        successes = r.get("observed_successes", [0] * 4)
        exposures = r.get("observed_exposures", [0] * 4)
        unknown = r.get("observed_unknown", [0] * 4)
        for c in range(4):
            exp = exposures[c]
            cells.append(dict(family=family, session=session, cell=c, successes=successes[c], exposures=exp,
                              unknown=unknown[c], rate=successes[c] / exp if exp else None,
                              bounds=[successes[c] / exp, min(1., (successes[c] + unknown[c]) / exp)] if exp else [0., 1.],
                              synthetic=bool(r.get("synthetic", False))))
    coverage = dict(cells=cells, unobserved_cells=sum(c["exposures"] == 0 for c in cells),
                    unknown_outcomes=sum(c["unknown"] for c in cells),
                    observed_cells=sum(c["exposures"] > 0 for c in cells))
    specificity = {}
    for condition in sorted({str(r.get("condition")) for r in causal}):
        rs = [r for r in causal if str(r.get("condition")) == condition]
        specificity[condition] = {key: dict(mean=_mean([float(r[key]) for r in rs if r.get(key) is not None]),
                                           n_reported=sum(r.get(key) is not None for r in rs),
                                           n_assigned=len(rs))
                                  for key in ("valid", "literal_correct", "other_domain_correct", "non_target_error")}
    return _jsonable(dict(schema_version=1, synthetic=synthetic, target_model_evidence=not synthetic and bool(records),
                         incomplete=incomplete, execution_incomplete=execution_incomplete,
                         n_records=len(records), n_families=sum(f in observed_families for f in families),
                         primary_delay=primary_delay,
                         family_inventory=dict(expected=families, observed=observed_families, missing=missing_families,
                                               unregistered=sorted(set(observed_families) - family_set),
                                               unresolved_family_effect_bounds={f: [-1., 1.] for f in missing_families}),
                         causal_inventory=dict(n_primary_records=len(primary_causal),
                                               n_secondary_records=len(causal) - len(primary_causal),
                                               n_missing_delay=sum("delay" not in r for r in causal)),
                         primaries=primaries, primary_registry=list(PRIMARY_IDS), allocation=allocation,
                         coverage=coverage, trajectories=trajectories, specificity=specificity,
                         audits=[dict(family=k[0], session=k[1], phase=k[2], domain=k[3], peer=k[4], quality=v,
                                      n=audit_counts[k][1], synthetic=synthetic) for k, v in audit_quality.items()],
                         retention=[{k: r.get(k) for k in ("family", "session", "round", "delay", "prediction", "raw", "target", "mask", "synthetic")}
                                    for r in records if _event(r) in ("readout", "retention")],
                         assumptions=["source family is the independent inference unit", "sign-flip contrasts symmetric at null boundary",
                                      "audit worlds/streams independent of main history", "invalid outputs retained as zero end-to-end success",
                                      "missing matched comparisons unavailable; all seven primaries retained"],
                         proposed_estimands={"P1": "masked absolute observed-rate error improvement; separate constant-total attribution control",
                                              "P2": "(audited chosen quality-min)/(max-min), configured first/last counts (quarter defaults); explicit round lists override",
                                              "noninferiority_margin": settings.get("ni_margin", .05)},
                         interpretation="Synthetic pipeline validation only; no evidence about Qwen mechanisms." if synthetic else
                         "Observational decoding alone does not establish causal use; inspect all gates and components."))


def report_payloads(summary: dict) -> dict[str, bytes]:
    """Build report bytes so runners can enforce artifact budgets before writes."""
    summary = _jsonable(summary)
    payloads = {"report.json": json.dumps(summary, indent=2, allow_nan=False).encode("utf-8")}
    lines = ["# Experiment 5 report", "", "Synthetic pipeline validation" if summary.get("synthetic") else "Model-run analysis",
             "", summary.get("interpretation", ""), "", f"Incomplete: {summary.get('incomplete')}",
             f"Independent families: {summary.get('n_families', 0)}", "", "| Primary | Status | p | Holm p | Reject |",
             "|---|---|---:|---:|---|"]
    for key, p in summary.get("primaries", {}).items():
        lines.append(f"| {key} | {p['status']} | {p['p']:.6g} | {p['adjusted_p']:.6g} | {p['reject']} |")
    lines += ["", "All components, family effects, unknown bounds, retention measurements, audits and trajectories are retained in report.json.",
              "", "## Trajectory", "", "| Family | Session | Round | Domain | Peer | Valid | Correct |", "|---|---|---:|---|---|---|---|"]
    for r in summary.get("trajectories", []):
        lines.append("| " + " | ".join(str(r.get(k)) for k in ("family", "session", "round", "domain", "peer", "valid", "correct")) + " |")
    payloads["report.md"] = ("\n".join(lines) + "\n").encode("utf-8")
    return payloads


def write_report(output_dir: Path, summary: dict):
    """Write a portable report; bounded runners should use report_payloads."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, payload in report_payloads(summary).items():
        (output_dir / name).write_bytes(payload)
