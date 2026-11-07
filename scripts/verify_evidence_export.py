"""Verify saved cluster evidence without inference, network access, or file edits."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path, PurePosixPath
import posixpath
import re


SOURCE_FILES = {
    "run_exp4_rescore_v3.py": "scripts/run_exp4_rescore_v3.py",
    "run_exp2_rescore_v3.py": "scripts/run_exp2_rescore_v3.py",
    **{name: "src/jpeers/followup/" + name for name in (
        "scoring_v3.py", "scoring.py", "common.py", "rescore.py")},
    "imposter_facts.json": "src/jpeers/data/imposter_facts.json",
}
CORRECTED_COUNTS = {"exp1": (96, 32), "exp2": (1080, 600),
                    "exp3": (1324, 1300), "narcbench": (40, 40)}
LFS_HEADER = b"version https://git-lfs.github.com/spec/v1"


def digest(value):
    raw = json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False,
                     separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def hash_file(path):
    checksum, size, header = hashlib.sha256(), 0, b""
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            if not header:
                header = chunk[:128]
            checksum.update(chunk)
            size += len(chunk)
    return checksum.hexdigest(), size, header


def safe_path(root, value):
    if not isinstance(value, str) or not value or "\\" in value:
        raise ValueError(f"Invalid export path {value!r}")
    relative = PurePosixPath(value)
    if (not relative.parts or relative.is_absolute() or ".." in relative.parts
            or str(relative) != value):
        raise ValueError(f"Export path escapes or aliases its root: {value!r}")
    path = root.joinpath(*relative.parts)
    if not path.resolve().is_relative_to(root):
        raise ValueError(f"Export path escapes its root: {value!r}")
    current = root
    for part in relative.parts:
        current = current / part
        if current.is_symlink():
            raise ValueError(f"Symlink in export path: {value!r}")
    return path


def completion_checks(group, info, documents, errors):
    """Check recorded target completion independently of historical grader failures."""
    checked = 0
    if info["kind"] == "completed_original_target_suite":
        key = group + "/source/runs/full-suite/manifest.json"
        value = documents.get(key)
        if (not isinstance(value, dict) or value.get("status") != "completed"
                or any(value.get("experiments", {}).get(f"exp{i}", {}).get("status") != "completed"
                       for i in range(1, 5))
                or len(value.get("jobs", [])) != 17
                or any(job.get("status") != "completed" for job in value.get("jobs", []))):
            errors.append(f"Original suite lacks its completed Exp1–4, 17-job manifest: {key}")
        checked += 1
        if (info.get("expected_target_traces"), info.get("expected_smoke_traces")) != (5776, 26):
            errors.append(f"Original suite must declare 5776 target and 26 smoke traces: {group}")
    if info["kind"] == "completed_corrected_targets":
        if info.get("expected_target_traces") != 1972:
            errors.append(f"Corrected targets must declare 1972 traces: {group}")
        for job, (turns, traces) in CORRECTED_COUNTS.items():
            key = f"{group}/results/{job}/followup-v2/validation.json"
            value = documents.get(key)
            if (not isinstance(value, dict) or value.get("status") != "completed"
                    or value.get("n_turns") != turns or value.get("n_traces") != traces
                    or value.get("validation_errors") != []):
                errors.append(f"Corrected target validation is missing or incomplete: {key}")
            checked += 1
    return checked


def verify(manifest_path):
    manifest_path = Path(manifest_path).resolve()
    root = manifest_path.parent
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    groups, entries = manifest["groups"], manifest["artifacts"]
    errors, notes = [], []
    artifacts, source_index, documents, observed = {}, {}, {}, {}
    verified, bytes_hashed = set(), 0
    totals = defaultdict(Counter)
    receipt_checks = source_checks = metadata_checks = 0
    for entry in entries:
        name, group = entry.get("path"), entry.get("group")
        if group not in groups:
            errors.append(f"Unknown artifact group: {group!r}")
            continue
        try:
            path = safe_path(root, name)
        except (ValueError, OSError) as exc:
            errors.append(str(exc))
            continue
        if PurePosixPath(name).parts[0] != group:
            errors.append(f"Artifact path is outside its declared group: {name}")
            continue
        if name in artifacts:
            errors.append(f"Duplicate artifact path: {name}")
            continue
        source = entry.get("source_path")
        if not isinstance(source, str) or not PurePosixPath(source).is_absolute():
            errors.append(f"Artifact lacks an absolute source path: {name}")
            continue
        if (group, source) in source_index:
            errors.append(f"Duplicate source mapping in group {group}: {source}")
            continue
        artifacts[name] = entry
        source_index[group, source] = name
        totals[group]["files"] += 1
        totals[group]["npz_files"] += int(path.suffix == ".npz")
        if entry.get("kind") not in ("trace", "scientific_source", "saved_evidence"):
            errors.append(f"Unknown artifact kind: {name}")
        if (type(entry.get("bytes")) is not int or entry["bytes"] < 0
                or not isinstance(entry.get("sha256"), str)
                or not re.fullmatch(r"[a-f0-9]{64}", entry["sha256"])):
            errors.append(f"Invalid artifact size or SHA-256 declaration: {name}")
            continue
        totals[group]["bytes"] += entry["bytes"]
        if not path.is_file():
            errors.append(f"Missing exported file: {name}")
            continue
        actual, size, header = hash_file(path)
        observed[name] = actual
        bytes_hashed += size
        if header.startswith(LFS_HEADER):
            errors.append(f"Git LFS pointer replaces saved evidence: {name}. "
                          "Run git lfs pull from the repository root, then retry.")
            continue
        if actual != entry["sha256"] or size != entry["bytes"]:
            errors.append(f"Checksum or byte-size mismatch: {name}")
        if path.suffix == ".npz" and (entry.get("kind") != "trace" or not header.startswith(b"PK")):
            errors.append(f"NPZ is not a declared saved ZIP trace: {name}")
        elif actual == entry["sha256"] and size == entry["bytes"]:
            verified.add(name)
        if path.suffix == ".json":
            try:
                document = json.loads(path.read_text(encoding="utf-8"))
                documents[name] = document
                if isinstance(document, dict):
                    for field, receipt in (("receipt_sha256", "receipt"),
                                           ("receipts_sha256", "receipts")):
                        if field in document:
                            receipt_checks += 1
                            if receipt not in document or digest(document[receipt]) != document[field]:
                                errors.append(f"Raw receipt digest mismatch ({field}): {name}")
            except (ValueError, TypeError) as exc:
                errors.append(f"Invalid saved JSON in {name}: {exc}")

    references = Counter()
    turn_files = 0
    for name, entry in artifacts.items():
        if PurePosixPath(name).name != "turns.jsonl" or name not in observed:
            continue
        turn_files += 1
        group = entry["group"]
        try:
            with (root / name).open(encoding="utf-8") as stream:
                for number, line in enumerate(stream, 1):
                    if not line.strip():
                        continue
                    turn = json.loads(line)
                    reference = turn.get("trace_file")
                    if reference is None:
                        continue
                    if not isinstance(reference, str) or not reference:
                        errors.append(f"Invalid trace reference: {name}:{number}")
                        continue
                    ref = PurePosixPath(reference)
                    sources = ([str(ref)] if ref.is_absolute() else [
                        posixpath.normpath(str(PurePosixPath(entry["source_path"]).parent / ref)),
                        posixpath.normpath(str(PurePosixPath(groups[group]["source_root"]) / ref))])
                    candidates = {source_index[group, source] for source in sources
                                  if (group, source) in source_index}
                    if not ref.is_absolute():
                        local = posixpath.normpath(str(PurePosixPath(name).parent / ref))
                        if local in artifacts and artifacts[local]["group"] == group:
                            candidates.add(local)
                    if len(candidates) != 1:
                        errors.append(f"Missing or ambiguous exported trace {reference!r}: {name}:{number}")
                        continue
                    target = candidates.pop()
                    if (artifacts[target].get("kind") != "trace" or target not in observed):
                        errors.append(f"Trace reference lacks a saved trace file: {name}:{number} -> {target}")
                        continue
                    references[target] += 1
        except (ValueError, TypeError, AttributeError) as exc:
            errors.append(f"Invalid turns JSONL in {name}: {exc}")

    for name, entry in artifacts.items():
        if entry.get("kind") != "trace" or "smoke" in PurePosixPath(name).parts:
            continue
        if references[name] > 1:
            errors.append(f"Duplicate target trace references ({references[name]}): {name}")
        if groups[entry["group"]].get("expected_target_traces") is not None and not references[name]:
            errors.append(f"Target trace is not referenced by any saved turn: {name}")

    for name, run in documents.items():
        path = PurePosixPath(name)
        if path.parts[-2:] != ("grading", "run.json") or not isinstance(run, dict):
            continue
        source = path.parent.parent / "source"
        pins = {}
        for basename, expected in run.get("source_files", {}).items():
            if basename not in SOURCE_FILES:
                errors.append(f"Unmapped scientific source basename in {name}: {basename}")
                continue
            pins[str(source / SOURCE_FILES[basename])] = expected
        for field, filename in (("development_fixture_sha256", "fact_grader_calibration_v2.json"),
                                ("holdout_fixture_sha256", "fact_grader_holdout_v3.json")):
            if run.get(field) is not None:
                pins[str(source / "src/jpeers/data" / filename)] = run[field]
        for target, expected in pins.items():
            source_checks += 1
            if observed.get(target) != expected:
                errors.append(f"Frozen scientific source pin missing or mismatched: {name} -> {target}")

    summary = {}
    for group, info in groups.items():
        if type(info.get("live")) is not bool:
            errors.append(f"Group must declare live as a Boolean: {group}")
        if info.get("live"):
            status = info.get("capture_status", "")
            if status not in ("active_snapshot", "live_snapshot", "partial_snapshot"):
                errors.append(f"Live group claims a final or unrecognized capture status: {group}")
            notes.append(f"{group} is a partial live snapshot; individual files can reflect "
                         "different instants and do not establish final job completion.")
        for field in ("files", "bytes", "npz_files"):
            if type(info.get(field)) is not int or info[field] != totals[group][field]:
                errors.append(f"Group {field} differs from its artifact census: {group}")
        target = sum(e.get("kind") == "trace" and "smoke" not in PurePosixPath(n).parts
                     for n, e in artifacts.items() if e["group"] == group)
        smoke = sum(e.get("kind") == "trace" and "smoke" in PurePosixPath(n).parts
                    for n, e in artifacts.items() if e["group"] == group)
        for field, actual in (("expected_target_traces", target), ("expected_smoke_traces", smoke)):
            if field in info and (type(info[field]) is not int or info[field] != actual):
                errors.append(f"Group {field} mismatch: {group}; expected {info[field]}, found {actual}")
        metadata_checks += completion_checks(group, info, documents, errors)
        summary[group] = {**dict(totals[group]), "target_traces": target, "smoke_traces": smoke,
                          "referenced_target_traces": sum(references[n] == 1 for n, e in artifacts.items()
                              if e["group"] == group and e.get("kind") == "trace"
                              and "smoke" not in PurePosixPath(n).parts),
                          **{key: info.get(key) for key in ("kind", "live", "capture_status",
                              "lifecycle_before", "lifecycle_after")}}
    return {"protocol": "cluster-evidence-verifier-v1", "ok": not errors,
            "manifest": str(manifest_path), "manifest_sha256": hash_file(manifest_path)[0],
            "files_hashed": len(observed), "bytes_hashed": bytes_hashed,
            "files_verified": len(verified), "bytes_verified": sum(
                artifacts[name]["bytes"] for name in verified),
            "turn_files_checked": turn_files, "receipt_digests_checked": receipt_checks,
            "scientific_pins_checked": source_checks, "completion_metadata_checked": metadata_checks,
            "groups": summary, "notes": notes, "errors": errors}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1] /
                        "results/cluster-evidence-20261107")
    parser.add_argument("--report", type=Path, help="Optional JSON report outside the exported evidence")
    args = parser.parse_args()
    if args.report is not None and args.report.resolve().is_relative_to(args.root.resolve()):
        parser.error("The report must be outside the exported evidence directory")
    try:
        result = verify(args.root / "manifest.json")
    except (OSError, ValueError, KeyError, TypeError) as exc:
        result = {"protocol": "cluster-evidence-verifier-v1", "ok": False, "errors": [str(exc)]}
    text = json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(text, encoding="utf-8")
    print(text, end="")
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
