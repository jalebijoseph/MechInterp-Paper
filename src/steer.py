"""Causal checks: ablate or boost an SAE feature and see what changes.

Needs a loaded model + SAE (see load.py); tests/test_causal.py exercises
the edit hook on a tiny random model. For quantitative causal checks
(KL, token/component attribution, random baselines) see causal.py; this
module is for qualitative side-by-side generations.
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger("mech_interp_sae.steer")


def _residual_with_edit(sae, resid, feature_id: int, scale: float):
    """`resid` with one feature's contribution (act × decoder row) scaled.

    `scale=0.0` ablates the feature; `scale>1.0` boosts it; `scale=1.0`
    is an exact no-op. Only the feature's own contribution changes, so
    the SAE's reconstruction error stays in the residual stream (decoding
    the edited features instead would also drop that error, changing the
    model's output even at scale=1.0).
    """
    w = sae.W_dec[feature_id]
    acts = sae.encode(resid.to(w.dtype))[..., feature_id : feature_id + 1]
    return resid + ((scale - 1.0) * acts * w).to(resid.dtype)


def make_edit_hook(sae, feature_id: int, scale: float):
    """transformer_lens hook: scale one feature's contribution to resid_post."""

    def hook_fn(resid, hook):
        return _residual_with_edit(sae, resid, feature_id, scale)

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
