"""Offline linear instruments for Experiment 5 (NumPy only).

Heads score *observed* evidence, never hidden peer quality. Unknown heads are
NaN, rather than zero capability. This module never changes model parameters.
"""
from __future__ import annotations

import hashlib
from io import BytesIO
import json
from pathlib import Path

import numpy as np

VERSION = 1
CELLS = (("A", "possession"), ("A", "location"),
         ("B", "possession"), ("B", "location"))


def _validate_records(records, layer):
    if not records:
        raise ValueError("at least one capture is required")
    x = np.stack([np.asarray(r["captures"][layer], dtype=float) for r in records])
    y = np.asarray([r["target"] for r in records], dtype=float)
    mask = np.asarray([r["mask"] for r in records], dtype=bool)
    if x.ndim != 2 or not np.isfinite(x).all() or y.shape != (len(records), 4) or mask.shape != y.shape:
        raise ValueError("finite vector captures and four targets/masks are required")
    if not np.isfinite(y[mask]).all() or np.any((y[mask] < 0) | (y[mask] > 1)):
        raise ValueError("observed targets must be rates in [0,1]")
    return x, y, mask


def fit_readout(records: list[dict], *, ridge: float, layer: int) -> dict:
    """Fit independent masked ridge heads with training-only preprocessing."""
    if not np.isfinite(ridge) or ridge < 0:
        raise ValueError("ridge must be finite and nonnegative")
    if any(r.get("split") in ("eval", "evaluation", "test") for r in records):
        raise ValueError("evaluation records cannot fit a frozen instrument")
    x, y, mask = _validate_records(records, layer)
    site = records[0].get("site")
    if any(r.get("site") != site for r in records):
        raise ValueError("a frozen readout requires one consistent capture site")
    mean, scale = x.mean(axis=0), x.std(axis=0)
    scale[scale < 1e-12] = 1
    z = (x - mean) / scale
    coef = np.zeros((x.shape[1], 4))
    intercept = np.full(4, np.nan)
    counts = mask.sum(axis=0)
    for cell in range(4):
        keep = mask[:, cell]
        if not keep.any():
            continue
        a, b = z[keep], y[keep, cell]
        am, bm = a.mean(axis=0), b.mean()
        a = a - am
        # Dual solve avoids a hidden-size squared matrix when n << d.
        if ridge > 0 and len(a) < a.shape[1]:
            weight = a.T @ np.linalg.solve(a @ a.T + ridge * np.eye(len(a)), b - bm)
        elif ridge > 0:
            weight = np.linalg.solve(a.T @ a + ridge * np.eye(a.shape[1]), a.T @ (b - bm))
        else:
            weight = np.linalg.lstsq(a, b - bm, rcond=None)[0]
        coef[:, cell] = weight
        intercept[cell] = bm - am @ weight
    return dict(kind="exp5-linear-readout", version=VERSION, layer=int(layer), site=site,
                ridge=float(ridge), mean=mean, scale=scale, coef=coef,
                intercept=intercept, counts=counts,
                cell_order=[list(c) for c in CELLS], evidence_target="observed_successes/observed_attempts",
                provenance={"families": sorted({str(r["family"]) for r in records}),
                            "record_ids": [str(r.get("id", i)) for i, r in enumerate(records)],
                            "delays": sorted({int(r["delay"]) for r in records}),
                            "n_records": len(records),
                            "synthetic": any(bool(r.get("synthetic")) for r in records)})


def predict_readout(artifact: dict, vector: np.ndarray) -> dict:
    x = np.asarray(vector, dtype=float)
    if x.shape != np.asarray(artifact["mean"]).shape or not np.isfinite(x).all():
        raise ValueError("capture shape/finite values do not match frozen instrument")
    raw = ((x - artifact["mean"]) / artifact["scale"]) @ artifact["coef"] + artifact["intercept"]
    return {"raw": raw, "prediction": np.clip(raw, 0, 1), "mask": np.asarray(artifact["counts"]) > 0}


def _balanced(y, predicted, mask):
    scores = []
    for cell in range(4):
        recalls = []
        for high in (False, True):
            keep = mask[:, cell] & ((y[:, cell] >= .6) == high)
            if keep.any():
                recalls.append(float(np.mean((predicted[keep, cell] >= .6) == high)))
        if len(recalls) == 2:
            scores.append(float(np.mean(recalls)))
    return float(np.mean(scores)) if scores else None


def _blind(row):
    # Outcome totals are allowed evidence features, but attribution is removed.
    y, m = np.asarray(row["target"], float), np.asarray(row["mask"], bool)
    totals = []
    for cells in ((0, 2), (1, 3)):
        keep = [i for i in cells if m[i]]
        totals.extend((float(np.mean(y[keep])) if keep else 0., len(keep) / 2))
    return np.asarray(totals)


def _controlled(row):
    profile = row.get("profile", row.get("profile_id", ""))
    return (bool(row.get("controlled", "profile" in row)) and not bool(row.get("anchor", False))
            and not str(profile).startswith("anchor")
            and not np.any(np.isclose(np.asarray(row["target"], float), .6)))


def _binding(y, p, mask):
    scores = []
    for i in range(len(y)):
        for j in range(i + 1, len(y)):
            changed = ~np.isclose(y[i], y[j])
            if (changed.sum() >= 2 and mask[i].all() and mask[j].all()
                    and np.isclose(y[i].sum(), y[j].sum())):
                scores.extend((np.sign(p[j, changed] - p[i, changed]) ==
                               np.sign(y[j, changed] - y[i, changed])).tolist())
    return float(np.mean(scores)) if scores else None


def select_readout(records: list[dict], layers: list[int], ridges: list[float], *,
                   primary_delay: int = 512) -> tuple[dict, dict]:
    """Leave-family-out development selection; no online threshold optimization.

    The returned gates are separate: a calibrated decoder alone cannot establish
    behavioral validity, spontaneous online transfer, or a causal mechanism.
    """
    if any(r.get("split") in ("eval", "evaluation", "test") for r in records):
        raise ValueError("selection is restricted to development families")
    rows = [r for r in records if int(r["delay"]) == primary_delay and _controlled(r)]
    families = sorted({r["family"] for r in rows}, key=str)
    if len(families) < 2 or not layers or not ridges:
        raise ValueError("selection requires controlled primary-delay captures from >=2 families")
    candidates = []
    for layer in layers:
        for ridge in ridges:
            all_y, all_p, all_m, priors, blind_preds, permuted, folds = [], [], [], [], [], [], []
            for family in families:
                train = [r for r in rows if r["family"] != family]
                test = [r for r in rows if r["family"] == family]
                fitted = fit_readout(train, ridge=ridge, layer=layer)
                _, y, mask = _validate_records(test, layer)
                p = np.stack([predict_readout(fitted, r["captures"][layer])["prediction"] for r in test])
                prior = np.array([np.mean([r["target"][c] for r in train if r["mask"][c]])
                                  if any(r["mask"][c] for r in train) else np.nan for c in range(4)])
                blind_train = [dict(r, captures={0: _blind(r)}) for r in train]
                blind_fit = fit_readout(blind_train, ridge=ridge, layer=0)
                bp = np.stack([predict_readout(blind_fit, _blind(r))["prediction"] for r in test])
                seed = int.from_bytes(hashlib.sha256(str(family).encode()).digest()[:8], "little")
                rng = np.random.default_rng(seed)
                # Permute complete target/mask rows together inside each training family.
                perm_train = []
                for train_family in families:
                    group = [r for r in train if r["family"] == train_family]
                    for r, idx in zip(group, rng.permutation(len(group))):
                        perm_train.append(dict(r, target=group[idx]["target"], mask=group[idx]["mask"]))
                pf = fit_readout(perm_train, ridge=ridge, layer=layer)
                pp = np.stack([predict_readout(pf, r["captures"][layer])["prediction"] for r in test])
                all_y.extend(y); all_p.extend(p); all_m.extend(mask)
                priors.extend(np.tile(prior, (len(test), 1))); blind_preds.extend(bp); permuted.extend(pp)
                folds.append(dict(test_family=family, train_families=[f for f in families if f != family],
                                  mean=fitted["mean"].tolist(), scale=fitted["scale"].tolist(),
                                  attribution_accuracy=_binding(y, p, mask),
                                  balanced_accuracy=_balanced(y, p, mask)))
            y, p, m = np.asarray(all_y), np.asarray(all_p), np.asarray(all_m)
            candidates.append(dict(layer=layer, ridge=ridge, balanced_accuracy=_balanced(y, p, m),
                                   prior_accuracy=_balanced(y, np.asarray(priors), m),
                                   identity_blind_accuracy=_balanced(y, np.asarray(blind_preds), m),
                                   permuted_accuracy=_balanced(y, np.asarray(permuted), m), folds=folds))
    best = max(candidates, key=lambda c: c["balanced_accuracy"] if c["balanced_accuracy"] is not None else -1)
    score = best["balanced_accuracy"]
    behavioral_coverage = {}
    for field in ("valid", "routing_correct"):
        applicable = [r for r in rows if field == "valid" or r.get("routing_applicable") is not False]
        passed = sum(r.get(field) is True for r in applicable)
        behavioral_coverage[field] = dict(n_assigned=len(rows), n_applicable=len(applicable),
                                          n_nonapplicable=len(rows) - len(applicable),
                                          n_reported=sum(field in r for r in applicable), n_passed=passed,
                                          rate=passed / len(applicable) if applicable else None)
    gates = {"readout_accuracy": score is not None and score >= .75,
             "prior_improvement": score is not None and best["prior_accuracy"] is not None and score - best["prior_accuracy"] >= .1,
             "identity_blind_improvement": score is not None and best["identity_blind_accuracy"] is not None and score - best["identity_blind_accuracy"] >= .1,
             "unseen_family_transfer": all(f["balanced_accuracy"] is not None and f["balanced_accuracy"] >= .75 for f in best["folds"]),
             "attribution_binding": all(f["attribution_accuracy"] is not None and f["attribution_accuracy"] >= .75 for f in best["folds"]),
             "behavioral_validity": behavioral_coverage["valid"]["rate"] >= .95,
             "evidence_supported_routing": (behavioral_coverage["routing_correct"]["rate"] is not None
                                            and behavioral_coverage["routing_correct"]["rate"] >= .8),
             "online_transfer": False,
             "target_model_evidence": not any(r.get("synthetic", False) for r in rows)}
    gates = {key: bool(value) for key, value in gates.items()}
    report = dict(best, candidates=[{k: v for k, v in c.items() if k != "folds"} for c in candidates],
                  gates=gates, behavioral_coverage=behavioral_coverage,
                  passed=all(gates.values()), primary_delay=primary_delay,
                  threshold=.6, score_population="controlled binary profiles only",
                  online_transfer_status="requires separate frozen development validation",
                  baseline_features="identity-blind observed per-domain totals/coverage",
                  selection="leave-one-source-family-out development; no evaluation refit")
    artifact = fit_readout(rows, ridge=best["ridge"], layer=best["layer"])
    artifact["identity_blind"] = fit_readout([dict(r, captures={0: _blind(r)}) for r in rows],
                                            ridge=best["ridge"], layer=0)
    artifact["validation"] = report
    return artifact, report


def conditional_bases(records: list[dict], *, layer: int, rank: int) -> dict[int, np.ndarray]:
    """SVD of matched one-cell encoding differences (never decoder weights)."""
    if rank < 1:
        raise ValueError("rank must be positive")
    if any(r.get("split") in ("eval", "evaluation", "test") for r in records):
        raise ValueError("evaluation records cannot identify frozen encoding bases")
    x, y, mask = _validate_records(records, layer)
    results = {}
    for cell in range(4):
        differences = []
        for i, left in enumerate(records):
            if any(not isinstance(left.get(field), str) or not left[field].strip()
                   for field in ("site", "match_id")):
                continue
            for j in range(i + 1, len(records)):
                right = records[j]
                if (left["family"] != right["family"] or left["delay"] != right["delay"]
                        or left.get("site") != right.get("site")
                        or left.get("match_id") != right.get("match_id")):
                    continue
                if not mask[i].all() or not mask[j].all():
                    continue
                other = [c for c in range(4) if c != cell]
                if not np.allclose(y[i, other], y[j, other], atol=1e-10, rtol=0) or np.isclose(y[i, cell], y[j, cell], atol=1e-10, rtol=0):
                    continue
                differences.append((x[j] - x[i]) * np.sign(y[j, cell] - y[i, cell]))
        if differences:
            _, singular, vt = np.linalg.svd(np.stack(differences), full_matrices=False)
            n = min(rank, int(np.sum(singular > max(singular[0] * 1e-10, 1e-12))))
            results[cell] = vt[:n].T.copy()
        else:
            results[cell] = np.empty((x.shape[1], 0))
    return results


def artifact_payloads(artifact: dict, stem: str = "probe") -> dict[str, bytes]:
    """Build versioned JSON/NPZ bytes before a runner enforces write budgets."""
    arrays = {}
    def encode(value):
        if isinstance(value, np.ndarray):
            if value.dtype.hasobject:
                raise ValueError("object arrays/pickle are forbidden")
            key = f"array_{len(arrays)}"
            arrays[key] = value
            return {"array": key, "shape": list(value.shape), "dtype": str(value.dtype)}
        if isinstance(value, dict):
            return {"mapping": [[encode(k), encode(v)] for k, v in value.items()]}
        if isinstance(value, (list, tuple)):
            return [encode(v) for v in value]
        if isinstance(value, np.generic):
            return value.item()
        return value
    metadata = dict(version=VERSION, format="exp5-numeric-tree", artifact=encode(artifact))
    manifest = json.dumps(metadata, indent=2, allow_nan=False).encode("utf-8")
    archive = BytesIO()
    np.savez_compressed(archive, **arrays)
    # Sidecar precedes the manifest when a writer consumes this mapping in order.
    return {f"{stem}.npz": archive.getvalue(), f"{stem}.json": manifest}


def save_artifact(path: Path, artifact: dict):
    """Versioned JSON tree + numeric NPZ; integer mapping keys round-trip."""
    path = Path(path).with_suffix(".json")
    payloads = artifact_payloads(artifact, stem=path.stem)
    path.parent.mkdir(parents=True, exist_ok=True)
    for name, payload in payloads.items():
        (path.parent / name).write_bytes(payload)


def load_artifact(path: Path) -> dict:
    path = Path(path).with_suffix(".json")
    meta = json.loads(path.read_text(encoding="utf-8"))
    if meta.get("version") != VERSION or meta.get("format") != "exp5-numeric-tree":
        raise ValueError("unsupported readout artifact version/format")
    with np.load(path.with_suffix(".npz"), allow_pickle=False) as arrays:
        def decode(value):
            if isinstance(value, dict) and "array" in value:
                array = arrays[value["array"]].copy()
                if list(array.shape) != value["shape"] or str(array.dtype) != value["dtype"]:
                    raise ValueError("array metadata mismatch")
                return array
            if isinstance(value, dict) and "mapping" in value:
                return {decode(k): decode(v) for k, v in value["mapping"]}
            if isinstance(value, list):
                return [decode(v) for v in value]
            return value
        artifact = decode(meta["artifact"])
    if artifact.get("version") != VERSION:
        raise ValueError("unsupported instrument version")
    d = len(artifact["mean"])
    if artifact["coef"].shape != (d, 4) or artifact["scale"].shape != (d,) or artifact["intercept"].shape != (4,) or np.any(artifact["scale"] <= 0):
        raise ValueError("invalid instrument geometry")
    return artifact
