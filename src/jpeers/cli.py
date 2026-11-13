"""Command line: ``python -m jpeers.cli <command> ...``

    run CONFIG [--set key=value ...]   run an experiment config (configs/*.yaml)
    serve [--model NAME]               start the inference server (port 8080 / $AIP_HTTP_PORT)
    models                             list registered models and their lenses
    probe ENDPOINT "text" [--watch w]  one-off J-space readout through any endpoint
"""

from __future__ import annotations

import argparse
import importlib
import json
import logging
import os
import sys

import yaml


def _set(cfg: dict, assignment: str) -> None:
    key, _, raw = assignment.partition("=")
    node = cfg
    *parents, leaf = key.split(".")
    for p in parents:
        node = node.setdefault(p, {})
    node[leaf] = yaml.safe_load(raw)


def cmd_run(args) -> None:
    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    for s in args.set or []:
        _set(cfg, s)
    mod = importlib.import_module(f"jpeers.experiments.{cfg['experiment']}")
    summary = mod.run(cfg)
    print(json.dumps(summary.get("rows", summary), indent=2)[:6000])
    print(f"\nresults: {summary.get('run_dir')}")


def cmd_serve(args) -> None:
    if args.model:
        os.environ["JPEERS_MODEL"] = args.model
    from jpeers.server import main

    main()


def cmd_models(args) -> None:
    from jpeers.registry import MODELS

    for spec in MODELS.values():
        print(f"{spec.name:18s} {spec.hf_id:38s} {spec.family:8s} {spec.params_b:>6}B "
              f"~{spec.min_gpu_gb:>4}GB  chat={spec.chat}")


def cmd_probe(args) -> None:
    from jpeers.client import EndpointPool, load_endpoints
    from jpeers.schemas import GenerateRequest, PositionSelector, RecordSpec
    from jpeers.trace import JSpaceTrace

    pool = EndpointPool(load_endpoints(args.endpoints))
    ep = pool.get(args.endpoint)
    req = GenerateRequest(
        messages=[{"role": "user", "content": args.text}] if not args.raw else None,
        prompt=args.text if args.raw else None, max_new_tokens=args.max_new_tokens,
        record=RecordSpec(watch=args.watch or [], top_k=args.top_k,
                          positions=PositionSelector(scope="last" if args.last else "all")))
    resp = ep.generate(req)
    tr = JSpaceTrace.from_response(resp)
    print(f"model: {resp.model}\nreply: {resp.text!r}\n")
    for row in range(max(0, len(tr.positions) - args.show), len(tr.positions)):
        p = int(tr.positions[row])
        tops = " | ".join(f"L{l}:{tr.vocab[int(tr.topk_ids[row, j, 0])].strip()}"
                          for j, l in enumerate(tr.layers))
        print(f"{p:5d} {tr.tokens[p]!r:>14}  {tops}")
    if tr.watch:
        print("\nbest rank over band:", dict(zip(tr.watch, tr.watch_ranks.min(axis=(0, 1)).tolist())))


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"),
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    ap = argparse.ArgumentParser(prog="jpeers")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("config")
    r.add_argument("--set", action="append", help="override a config key, e.g. n_tasks=2")
    r.set_defaults(fn=cmd_run)
    s = sub.add_parser("serve")
    s.add_argument("--model")
    s.set_defaults(fn=cmd_serve)
    m = sub.add_parser("models")
    m.set_defaults(fn=cmd_models)
    p = sub.add_parser("probe")
    p.add_argument("endpoint")
    p.add_argument("text")
    p.add_argument("--endpoints", default=os.environ.get("JPEERS_ENDPOINTS"))
    p.add_argument("--watch", nargs="*")
    p.add_argument("--raw", action="store_true", help="raw text prompt, no chat template")
    p.add_argument("--last", action="store_true", help="only the final prompt position")
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument("--show", type=int, default=12)
    p.add_argument("--max-new-tokens", type=int, default=0)
    p.set_defaults(fn=cmd_probe)
    args = ap.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    args.fn(args)


if __name__ == "__main__":
    main()
