"""Experiment 5: preflight, simulation, local-model development, freeze and evaluation."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ["preflight", "simulate", "validate-runtime", "develop", "evaluate"]:
        sub = commands.add_parser(name)
        sub.add_argument("--config", type=Path, default=ROOT / "configs/exp5_study.json")
        if name in ("simulate", "develop", "evaluate"):
            sub.add_argument("--out", type=Path, required=True)
            sub.add_argument("--no-diagnostics", action="store_true")
        if name == "simulate":
            sub.add_argument("--full-config", action="store_true")
        if name == "preflight":
            sub.add_argument("--out", type=Path)
            sub.add_argument("--frozen", type=Path)
            sub.add_argument("--phase", choices=["development", "evaluation"], default="development")
            sub.add_argument("--statistical-check", action="store_true")
        if name in ("validate-runtime", "develop", "evaluate"):
            sub.add_argument("--execute", action="store_true")
            sub.add_argument("--cache-dir")
            sub.add_argument("--max-model-calls", type=int)
            sub.add_argument("--max-seconds", type=float)
        if name == "evaluate":
            sub.add_argument("--frozen", type=Path, required=True)
    frozen = commands.add_parser("freeze")
    frozen.add_argument("--development", type=Path, required=True)
    frozen.add_argument("--out", type=Path, required=True)
    frozen.add_argument("--allow-inconclusive", action="store_true")
    frozen.add_argument("--evaluation-families", type=int)
    reports = commands.add_parser("report")
    reports.add_argument("--run", type=Path, required=True)
    reports.add_argument("--out", type=Path)
    args = parser.parse_args()
    if args.command in ("validate-runtime", "develop", "evaluate") and not args.execute:
        parser.error("Real-model commands require --execute; use preflight or simulate for offline setup")

    from jpeers.exp5.protocol import load_protocol, validate_protocol
    from jpeers.exp5.storage import BudgetExceeded, jsonable
    try:
        if args.command == "freeze":
            from jpeers.exp5.runner import freeze
            result = freeze(args.development, args.out, allow_inconclusive=args.allow_inconclusive,
                            evaluation_families=args.evaluation_families)
        elif args.command == "report":
            from jpeers.exp5.runner import report
            result = report(args.run, args.out)
        else:
            config = load_protocol(args.config)
            for key in ("max_model_calls", "max_seconds"):
                value = getattr(args, key, None)
                if value is not None:
                    config["limits"][key] = value
            validate_protocol(config)
            if args.command == "preflight":
                from jpeers.exp5.runner import preflight
                result = preflight(config, phase=args.phase, output_dir=args.out, ancestor_dir=args.frozen)
                if args.statistical_check:
                    from jpeers.exp5.validation import noninferiority_preflight
                    result["noninferiority_preflight"] = noninferiority_preflight(config)
            elif args.command == "simulate":
                from jpeers.exp5.runner import develop, evaluate, freeze, smoke_config
                from jpeers.exp5.simulation import SimulationBackend
                if not args.full_config:
                    config = smoke_config(config)
                if args.out.exists():
                    raise FileExistsError(f"Simulation output already exists: {args.out}")
                dev, frz, evaluation = (args.out / name for name in ("development", "frozen", "evaluation"))
                development = develop(config, dev, SimulationBackend(config), execute=True,
                                      diagnostics=not args.no_diagnostics)
                if development.get("status") != "complete":
                    result = {"status": "incomplete", "synthetic": True, "development": development}
                else:
                    frozen_result = freeze(dev, frz, simulation=True, allow_inconclusive=True)
                    evaluated = evaluate(config, frz, evaluation, SimulationBackend(config), execute=True,
                                         simulation=True, diagnostics=not args.no_diagnostics)
                    result = {"status": evaluated.get("status"), "synthetic": True,
                              "ready_for_execution": False, "development": development,
                              "freeze": frozen_result, "evaluation": evaluated}
            else:
                if all(value is None for value in config["limits"].values()):
                    raise ValueError("Declare --max-model-calls or --max-seconds before loading a real model")
                if args.command in ("develop", "evaluate"):
                    from jpeers.exp5.runner import preflight
                    notice = preflight(config, phase="evaluation" if args.command == "evaluate" else "development",
                                       output_dir=args.out, ancestor_dir=args.frozen if args.command == "evaluate" else None)
                    print(json.dumps({"pre_run_notice": jsonable(notice)}, sort_keys=True), file=sys.stderr)
                    failed = [key for key in ("estimated_storage_within_allowance", "planned_study_storage_within_allowance", "physical_space_for_estimate",
                                              "output_directory_available") if notice["checks"].get(key) is not True]
                    if notice.get("forecast", {}).get("raw_estimate_within_allowance") is not True:
                        failed.append("raw_development_estimate_within_allowance")
                    if failed:
                        raise ValueError("Preflight prevents model loading: " + ", ".join(failed))
                from jpeers.exp5.backend import HFBackend
                backend = HFBackend.from_pretrained(config, cache_dir=args.cache_dir)
                if args.command == "validate-runtime":
                    result = backend.validate_runtime()
                    result["backend"] = backend.info()
                elif args.command == "develop":
                    from jpeers.exp5.runner import develop
                    result = develop(config, args.out, backend, execute=True,
                                     diagnostics=not args.no_diagnostics)
                else:
                    from jpeers.exp5.runner import evaluate
                    result = evaluate(config, args.frozen, args.out, backend, execute=True,
                                      diagnostics=not args.no_diagnostics)
        print(json.dumps(jsonable(result), sort_keys=True, allow_nan=False, default=str))
        return 1 if result.get("status") in ("error", "incomplete") or result.get("passed") is False else 0
    except (ValueError, OSError, RuntimeError, BudgetExceeded) as error:
        parser.error(str(error))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
