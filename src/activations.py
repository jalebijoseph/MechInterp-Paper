"""Run prompts through the model and cache SAE feature activations to disk.

The caching logic (`cached_compute`, `TokenActivations`) is plain
numpy/stdlib and has no dependency on torch/transformer_lens/sae_lens,
so it's unit-testable without the heavy ML stack installed.
`compute_token_activations` is the only function here that actually
needs a loaded model + SAE.

What gets cached is each prompt's *per-token* SAE activations, stored
sparsely (Gemma Scope SAEs have L0 ~70 active features per token out
of 16k, so this is a few KB per prompt). Per-prompt mean/max vectors are
derived from that on load, and the same cache answers "which tokens does
this feature fire on?" without another forward pass.

The BOS token is dropped: Gemma's residual stream at position 0 has a
huge norm and SAE features fire on it regardless of the prompt, so
including it would add a prompt-independent offset to every vector.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

from src.config import CACHE_DIR

logger = logging.getLogger("mech_interp_sae.activations")

# Bump when the cached format or how activations are computed changes,
# so stale cache files are never silently reused.
CACHE_VERSION = "token_acts_v2_nobos"


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


@dataclass
class TokenActivations:
    """Sparse per-token SAE activations for one prompt (BOS excluded).

    Nonzero entries are stored as parallel (position, feature, value)
    arrays; `tokens[i]` is the string for position i.
    """

    tokens: list[str]
    positions: np.ndarray
    features: np.ndarray
    values: np.ndarray
    n_features: int

    @classmethod
    def from_dense(cls, tokens: list[str], dense: np.ndarray) -> "TokenActivations":
        positions, features = np.nonzero(dense)
        return cls(
            tokens=list(tokens),
            positions=positions.astype(np.int32),
            features=features.astype(np.int32),
            values=dense[positions, features].astype(np.float32),
            n_features=int(dense.shape[1]),
        )

    def dense(self) -> np.ndarray:
        out = np.zeros((len(self.tokens), self.n_features), dtype=np.float32)
        out[self.positions, self.features] = self.values
        return out

    def reduce(self, mode: str = "mean") -> np.ndarray:
        """One (n_features,) vector for the prompt, reduced over tokens."""
        if len(self.tokens) == 0:
            return np.zeros(self.n_features, dtype=np.float32)
        if mode == "mean":
            out = np.zeros(self.n_features, dtype=np.float32)
            np.add.at(out, self.features, self.values)
            return out / len(self.tokens)
        if mode == "max":
            out = np.zeros(self.n_features, dtype=np.float32)
            np.maximum.at(out, self.features, self.values)
            return out
        raise ValueError(f"Unknown reduce mode: {mode}")

    def feature_trace(self, feature_id: int) -> np.ndarray:
        """One feature's activation at every token position."""
        out = np.zeros(len(self.tokens), dtype=np.float32)
        mask = self.features == feature_id
        out[self.positions[mask]] = self.values[mask]
        return out

    def save(self, path: Path) -> None:
        np.savez(
            path,
            tokens=np.array(self.tokens, dtype=np.str_),
            positions=self.positions,
            features=self.features,
            values=self.values,
            n_features=np.array(self.n_features),
        )

    @classmethod
    def load(cls, path: Path) -> "TokenActivations":
        with np.load(path) as f:
            return cls(
                tokens=[str(t) for t in f["tokens"]],
                positions=f["positions"],
                features=f["features"],
                values=f["values"],
                n_features=int(f["n_features"]),
            )


def cached_token_activations(
    key: str,
    compute_fn: Callable[[], TokenActivations],
    cache_dir: Path = CACHE_DIR,
) -> TokenActivations:
    """Like `cached_compute`, for `TokenActivations` (stored as .npz)."""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"{key}.npz"

    if path.exists():
        logger.debug("Cache hit: %s", key)
        return TokenActivations.load(path)

    logger.debug("Cache miss: %s", key)
    result = compute_fn()
    result.save(path)
    return result


def compute_token_activations(model, sae, prompt: str, layer: int) -> TokenActivations:
    """Forward one prompt up to `layer` and SAE-encode its residual stream."""
    import torch

    hook_name = f"blocks.{layer}.hook_resid_post"
    tokens = model.to_tokens(prompt)
    with torch.inference_mode():
        # Only blocks 0..layer are needed: stopping there skips the rest
        # of the model and the 256k-vocab unembedding.
        _, cache = model.run_with_cache(tokens, names_filter=hook_name, stop_at_layer=layer + 1)
        resid = cache[hook_name][0, 1:]  # (seq - 1, d_model), BOS dropped
        # The model may run in bfloat16 (see load.py) while the SAE's own
        # weights are typically float32 — cast to match before encoding.
        resid = resid.to(next(sae.parameters()).dtype)
        feats = sae.encode(resid).float().cpu().numpy()  # (seq - 1, n_features)
    # Decode from the same ids the forward pass used (to_str_tokens on a
    # raw string errors in the installed transformer_lens version).
    str_tokens = [model.tokenizer.decode([t]) for t in tokens[0, 1:].tolist()]
    return TokenActivations.from_dense(str_tokens, feats)


def get_token_activations(
    model,
    sae,
    prompts: list[str],
    layer: int,
    sae_id: str,
    cache_dir: Path = CACHE_DIR,
) -> list[TokenActivations]:
    """Per-token SAE activations for each prompt, cached per prompt.

    `sae_id` is part of the cache key, so different SAE variants at the
    same layer never share cache entries.
    """
    out = []
    for i, prompt in enumerate(prompts):
        key = cache_key(CACHE_VERSION, model.cfg.model_name, sae_id, layer, prompt)
        out.append(
            cached_token_activations(
                key, lambda p=prompt: compute_token_activations(model, sae, p, layer), cache_dir=cache_dir
            )
        )
        logger.debug("Activations %d/%d", i + 1, len(prompts))
    return out


def stack_reduced(token_acts: list[TokenActivations], reduce: str = "mean") -> np.ndarray:
    """(n_prompts, n_features) array from a list of `TokenActivations`."""
    return np.stack([t.reduce(reduce) for t in token_acts])


def get_feature_activations(
    model,
    sae,
    prompts: list[str],
    layer: int,
    sae_id: str,
    reduce: str = "mean",
    cache_dir: Path = CACHE_DIR,
) -> np.ndarray:
    """SAE feature activations for a list of prompts at one layer.

    Returns an array of shape (len(prompts), n_features): one feature
    vector per prompt, reduced over the token dimension (`reduce`:
    "mean" or "max"), BOS excluded.
    """
    return stack_reduced(get_token_activations(model, sae, prompts, layer, sae_id, cache_dir), reduce)


def load_prompt_set(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
