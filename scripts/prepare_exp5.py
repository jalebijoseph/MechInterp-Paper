"""Prepare Experiment 5 inputs offline; this command cannot launch an experiment."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from jpeers.experiments.peer_capability_setup import prepare_setup


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/exp5_setup.json")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    try:
        report = prepare_setup(args.config, args.out)
    except (ValueError, OSError) as error:
        parser.error(str(error))
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
