"""Run prompts through the model and cache SAE feature activations to disk.

The caching logic (`cached_compute`) is plain Python/stdlib and has no
dependency on torch/transformer_lens/sae_lens, so it's unit-testable
without the heavy ML stack installed. `get_feature_activations` is the
only function here that actually needs a loaded model + SAE.
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Callable

import numpy as np

from src.config import CACHE_DIR

logger = logging.getLogger("mech_interp_sae.activations")


def cache_key(*parts: object) -> str:
    """Stable hash for a tuple of (mostly-string) parts, used as a cache filename."""
    raw = json.dumps(parts, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]


def cached_compute(
    key: str,
    compute_fn: Callable[[], np.ndarray],
    cache_dir: Path = CACHE_DIR,
) -> np.ndarray:
    """Return `compute_fn()`'s result, from disk if it's been computed before.

    Stores results as .npy files under `cache_dir`, named by `key`
    (use `cache_key` to build one from the inputs that determine the
    result, e.g. model name + layer + sae id + prompt text).
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"{key}.npy"

    if path.exists():
        logger.debug("Cache hit: %s", key)
        return np.load(path)

    logger.debug("Cache miss: %s", key)
    result = compute_fn()
    np.save(path, result)
    return result


def get_feature_activations(
    model,
    sae,
    prompts: list[str],
    layer: int,
    reduce: str = "mean",
    cache_dir: Path = CACHE_DIR,
) -> np.ndarray:
    """SAE feature activations for a list of prompts at one layer.

    Returns an array of shape (len(prompts), n_features): one feature
    vector per prompt, reduced over the token dimension (`reduce`:
    "mean" or "max"). Each prompt's result is cached independently so
    adding new prompts later doesn't require recomputing old ones.
    """
    hook_name = f"blocks.{layer}.hook_resid_post"

    def compute_one(prompt: str) -> np.ndarray:
        tokens = model.to_tokens(prompt)
        _, cache = model.run_with_cache(tokens, names_filter=hook_name)
        resid = cache[hook_name][0]  # (seq, d_model)
        # The model may run in bfloat16 (see load.py) while the SAE's own
        # weights are typically float32 — cast to match before encoding.
        resid = resid.to(next(sae.parameters()).dtype)
        feats = sae.encode(resid)  # (seq, n_features)
        feats = feats.detach().cpu().numpy()
        if reduce == "mean":
            return feats.mean(axis=0)
        if reduce == "max":
            return feats.max(axis=0)
        raise ValueError(f"Unknown reduce mode: {reduce}")

    rows = []
    for prompt in prompts:
        key = cache_key("feature_activations", model.cfg.model_name, layer, reduce, prompt)
        rows.append(cached_compute(key, lambda p=prompt: compute_one(p), cache_dir=cache_dir))
    return np.stack(rows)


def load_prompt_set(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
