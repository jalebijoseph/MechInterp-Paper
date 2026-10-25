"""Shared config: seeds, paths, model/SAE identifiers, logging.

Every script in this project should import SEED from here and seed
everything with it (random, numpy, torch) so runs are reproducible, and
should log which config values it ran with.
"""

import logging
import random
from dataclasses import dataclass, field
from pathlib import Path

SEED = 42

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
CACHE_DIR = ROOT_DIR / "cache"
RESULTS_DIR = ROOT_DIR / "results"

MODEL_NAME = "google/gemma-2-2b"

# Gemma Scope residual-stream SAEs, 16k width, via sae_lens.
# Confirmed against the google/gemma-scope-2b-pt-res repo's file tree on
# Hugging Face (huggingface.co/google/gemma-scope-2b-pt-res), which lists
# several average_l0 (sparsity) variants per layer/width. Picked the
# variant closest to average_l0~=70 at each layer, matching the L0=71
# variant the model card names as the one used in the Neuronpedia demo
# (for layer 20, not one of ours, but the same selection rule).
SAE_RELEASE = "gemma-scope-2b-pt-res"
SAE_WIDTH = "16k"
LAYERS = [6, 12, 18]
LAYER_AVERAGE_L0 = {6: 70, 12: 82, 18: 74}

for _dir in (DATA_DIR, CACHE_DIR, RESULTS_DIR):
    _dir.mkdir(parents=True, exist_ok=True)


def seed_everything(seed: int = SEED) -> None:
    """Seed every RNG this project touches. Call once at the start of a run."""
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def setup_logging(level: int = logging.INFO) -> logging.Logger:
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    return logging.getLogger("mech_interp_sae")


@dataclass
class RunConfig:
    """A single run's config, for logging alongside its results."""

    concept: str
    layer: int
    model_name: str = MODEL_NAME
    sae_release: str = SAE_RELEASE
    sae_width: str = SAE_WIDTH
    seed: int = SEED
    top_k_features: int = 10
    extra: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "concept": self.concept,
            "layer": self.layer,
            "model_name": self.model_name,
            "sae_release": self.sae_release,
            "sae_width": self.sae_width,
            "seed": self.seed,
            "top_k_features": self.top_k_features,
            **self.extra,
        }
