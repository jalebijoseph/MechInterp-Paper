"""Causal checks: ablate or boost an SAE feature and see what changes.

Needs a loaded model + SAE (see load.py) — nothing here is unit-testable
without them, so this is exercised manually / in notebooks once the
pipeline is running end to end, not in tests/.
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger("mech_interp_sae.steer")


def _reconstruction_with_edit(sae, resid, feature_id: int, scale: float):
    """Decode `resid` through the SAE with one feature's activation scaled.

    `scale=0.0` ablates the feature; `scale>1.0` boosts it;
    `scale=1.0` is a no-op (same as the SAE's own reconstruction).
    """
    feats = sae.encode(resid)
    feats = feats.clone()
    feats[..., feature_id] = feats[..., feature_id] * scale
    return sae.decode(feats)


def make_edit_hook(sae, feature_id: int, scale: float):
    """transformer_lens hook: replace resid_post with an SAE-edited version."""

    def hook_fn(resid, hook):
        edited = _reconstruction_with_edit(sae, resid, feature_id, scale)
        return edited

    return hook_fn


def run_with_feature_edit(
    model,
    sae,
    layer: int,
    feature_id: int,
    prompts: list[str],
    scale: float,
    max_new_tokens: int = 30,
) -> list[str]:
    """Generate completions for `prompts` with one SAE feature ablated/boosted."""
    hook_name = f"blocks.{layer}.hook_resid_post"
    hook_fn = make_edit_hook(sae, feature_id, scale)

    outputs = []
    for prompt in prompts:
        tokens = model.to_tokens(prompt)
        with model.hooks(fwd_hooks=[(hook_name, hook_fn)]):
            out = model.generate(tokens, max_new_tokens=max_new_tokens, do_sample=False)
        outputs.append(model.to_string(out[0]))
    return outputs


def compare_ablation_vs_baseline_vs_random(
    model,
    sae,
    layer: int,
    feature_id: int,
    prompts: list[str],
    n_features: int,
    rng: np.random.Generator,
    max_new_tokens: int = 30,
) -> dict[str, list[str]]:
    """Baseline (scale=1), feature ablated (scale=0), and a random feature ablated.

    The random-feature comparison is the control: if ablating a random
    feature changes outputs just as much as ablating the candidate
    feature, the candidate isn't doing anything special.
    """
    random_feature_id = int(rng.integers(0, n_features))
    while random_feature_id == feature_id:
        random_feature_id = int(rng.integers(0, n_features))

    return {
        "baseline": run_with_feature_edit(model, sae, layer, feature_id, prompts, scale=1.0, max_new_tokens=max_new_tokens),
        "ablated": run_with_feature_edit(model, sae, layer, feature_id, prompts, scale=0.0, max_new_tokens=max_new_tokens),
        "random_ablated": run_with_feature_edit(
            model, sae, layer, random_feature_id, prompts, scale=0.0, max_new_tokens=max_new_tokens
        ),
    }
