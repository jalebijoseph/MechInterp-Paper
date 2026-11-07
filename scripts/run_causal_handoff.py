"""Run opt-in discovery, then the frozen held-out incoming-information assay."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from jpeers.client import HTTPEndpoint
from jpeers.followup.handoff import HandoffRunner, output_lock


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--base-url', required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--stage', choices=['discovery', 'heldout', 'all'], default='all')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    endpoint = HTTPEndpoint('causal-handoff-target', args.base_url, timeout_s=1800, retries=0)
    with output_lock(args.output_dir):
        runner = HandoffRunner(endpoint, args.output_dir, config, resume=args.resume)
        result = runner.run(args.stage)
        status = json.loads((args.output_dir / 'status.json').read_text())
    print(json.dumps({'status': status['status'], 'output_dir': str(args.output_dir),
                      'reason': result.get('reason')}))


if __name__ == '__main__':
    main()
