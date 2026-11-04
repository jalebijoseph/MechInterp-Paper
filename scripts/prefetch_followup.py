"""CPU-only pinned model prefetch for the independent follow-up endpoint."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path


def main():
    from huggingface_hub import HfApi, snapshot_download
    from transformers import AutoConfig

    ap = argparse.ArgumentParser()
    ap.add_argument('--root', type=Path, required=True)
    ap.add_argument('--model-id', default='allenai/OLMo-2-0325-32B-Instruct')
    ap.add_argument('--revision')
    args = ap.parse_args()
    args.root.mkdir(parents=True, exist_ok=True)
    info = HfApi().model_info(args.model_id, revision=args.revision)
    record = {'model': args.model_id, 'family': 'olmo', 'revision': info.sha,
              'status': 'downloading', 'started_at': datetime.now(timezone.utc).isoformat()}
    output = args.root / 'olmo-provenance.json'
    output.write_text(json.dumps(record, indent=2))
    config = AutoConfig.from_pretrained(args.model_id, revision=info.sha, trust_remote_code=False)
    record['model_type'] = config.model_type
    record['model_path'] = snapshot_download(args.model_id, revision=info.sha,
        allow_patterns=['*.json', '*.safetensors', '*.model', '*.txt', '*.jinja', 'tokenizer*'],
        max_workers=4)
    record.update(status='completed', finished_at=datetime.now(timezone.utc).isoformat())
    output.write_text(json.dumps(record, indent=2))
    print('Completed', info.sha, flush=True)


if __name__ == '__main__':
    main()
