"""Download a registered model's weights and lens into the HF cache (``$HF_HOME``).

    python scripts/prefetch.py qwen3-8b [llama3.1-8b-it ...]
"""

from __future__ import annotations

import sys

from huggingface_hub import hf_hub_download, snapshot_download

from jpeers.registry import get_model_spec


def main() -> None:
    for name in sys.argv[1:]:
        spec = get_model_spec(name)
        snapshot_download(spec.hf_id, allow_patterns=["*.json", "*.safetensors", "*.model",
                                                      "*.txt", "*.jinja", "tokenizer*"])
        if spec.lens_file:
            hf_hub_download(spec.lens_repo, spec.lens_file, revision=spec.lens_revision)
        print(f"prefetched {name}")


if __name__ == "__main__":
    main()
