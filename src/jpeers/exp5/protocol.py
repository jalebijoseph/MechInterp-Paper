"""Frozen-choice validation and deterministic independent random streams."""
from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path

CELL_ORDER = (("A", "possession"), ("A", "location"),
              ("B", "possession"), ("B", "location"))
MODEL_ID = "Qwen/Qwen3-32B"
REVISION = "9216db5781bf21249d130ec9da846c4624c16137"
GIB = 1073741824


def derived_seed(seed: int, *parts: object) -> int:
    payload = json.dumps([seed, *parts], sort_keys=True, separators=(",", ":"), default=str)
    return int.from_bytes(hashlib.sha256(payload.encode()).digest()[:8], "big") % (2 ** 63 - 1)


def strict_object(text: str) -> dict:
    # Reuse the repository's duplicate-key/fenced-JSON checks, with lazy imports.
    from jpeers.followup.common import parse_json_object
    return parse_json_object(text)


def parse_choice(text: str) -> str | None:
    try:
        value = strict_object(text)
        if set(value) == {"choice"} and value["choice"] in ("A", "B"):
            return value["choice"]
    except (ValueError, TypeError):
        pass
    return None


def protocol_hash(config: dict) -> str:
    science = copy.deepcopy(config)
    science.pop("limits", None)
    return hashlib.sha256(json.dumps(science, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def load_protocol(path: Path | str) -> dict:
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    validate_protocol(config)
    return config


def validate_protocol(config: dict) -> None:
    try:
        if type(config["schema_version"]) is not int or config["schema_version"] != 1:
            raise ValueError("Unsupported protocol schema")
        if config["model"]["id"] != MODEL_ID or config["model"]["revision"] != REVISION:
            raise ValueError("Experiment 5 requires the pinned Qwen3-32B checkpoint")
        if config["delegation_choices"] != ["A", "B"]:
            raise ValueError("Initial delegation is A/B only")
        inference = config["inference"]
        if inference["enable_thinking"] is not False:
            raise ValueError("Initial roles must use non-thinking mode")
        for role in ["orchestrator", "peer"]:
            sampler = inference[role]
            if (not math.isfinite(sampler["temperature"]) or sampler["temperature"] <= 0
                    or not 0 < sampler["top_p"] <= 1 or not 0 <= sampler["min_p"] < 1
                    or type(sampler["top_k"]) is not int or sampler["top_k"] < 0):
                raise ValueError("Invalid seeded sampler parameters")
        tasks = config["tasks"]
        rounds, reversal = tasks["rounds"], tasks.get("reversal_round")
        if type(rounds) is not int or rounds < 4 or rounds % 4:
            raise ValueError("Rounds must be divisible by four for exact input schedules")
        if reversal is not None and (type(reversal) is not int or reversal <= 0
                                     or reversal >= rounds or reversal % 4):
            raise ValueError("Reversal boundary must preserve balanced phase schedules")
        if type(tasks["seed"]) is not int or tasks["seed"] < 0:
            raise ValueError("Protocol seed must be a nonnegative integer")
        if (config["development"]["families"] < 2 or config["evaluation"]["families"] < 1
                or config["audits"]["cases_per_domain"] < 2
                or config["audits"]["cases_per_domain"] % 2):
            raise ValueError("Insufficient family/audit counts")
        dev = config["development"]
        for rate in [dev["high_rate"], dev["low_rate"], dev["anchor_rate"]]:
            if not 0 <= rate <= 1 or abs(rate * dev["observations_per_cell"]
                                        - round(rate * dev["observations_per_cell"])) > 1e-8:
                raise ValueError("Calibration rates need exact integer observation counts")
        storage = config["storage"]
        if type(storage["total_bytes"]) is not int or not 0 < storage["total_bytes"] <= GIB:
            raise ValueError("Total result storage cannot exceed 1 GiB")
        if sum(storage[k] for k in ["raw_development_bytes", "text_bytes", "shared_bytes",
                                    "compact_bytes", "manifest_reserve_bytes"]) > storage["total_bytes"]:
            raise ValueError("Storage allocations exceed total allowance")
        if config["diagnostics"]["primary_delay"] not in config["diagnostics"]["delays"]:
            raise ValueError("Primary delay is missing from the registered delays")
        if any(type(x) is not int or x < 0 for x in config["diagnostics"]["delays"]):
            raise ValueError("Delays must be nonnegative token counts")
        if config["readout"] != {"method": "linear_probe", "target": "observed_success_rate"}:
            raise ValueError("Readout must retain the accepted method and target")
        for limit in config["limits"].values():
            if limit is not None and (isinstance(limit, bool) or not math.isfinite(limit) or limit <= 0):
                raise ValueError("Execution limits must be positive or unset")
    except (KeyError, TypeError) as error:
        raise ValueError(f"Malformed Experiment 5 protocol: {error}") from error
