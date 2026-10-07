"""Collect the summary tables of the latest (or a named) run of each experiment.

    python scripts/report.py [--run-id pilot] [--out docs/results_latest.md]
"""

from __future__ import annotations

import argparse
from pathlib import Path

ORDER = ["tom_validation", "collab", "imposter", "peer_selection"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--run-id")
    ap.add_argument("--out")
    args = ap.parse_args()
    parts = []
    for exp_dir in sorted(Path(args.runs).iterdir(), key=lambda p: (
            ORDER.index(p.name) if p.name in ORDER else 99, p.name)):
        runs = [p for p in exp_dir.iterdir() if (p / "summary.md").exists()]
        if args.run_id:
            runs = [p for p in runs if p.name.startswith(args.run_id)]
        for run in sorted(runs, key=lambda p: p.stat().st_mtime)[-1:]:
            parts.append(f"<!-- {run} -->\n" + (run / "summary.md").read_text(encoding="utf-8")
                         .replace("\n# ", "\n## ", 1).replace("# ", "## ", 1))
    text = "\n\n".join(parts)
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
