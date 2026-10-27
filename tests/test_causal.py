"""Causal-experiment helpers, on a tiny random transformer + fake SAE (no downloads)."""

import numpy as np
import pytest
import torch

from src.activations import TokenActivations
from src.causal import (
    add_vector_hook,
    component_attribution,
    components_for_layer,
    feature_acts,
    firing_frequency,
    matched_random_features,
    output_effects,
    token_attribution,
)
from src.steer import _residual_with_edit

D_MODEL, N_FEATURES, LAYER = 16, 8, 1


class FakeSAE(torch.nn.Module):
    def __init__(self):
        super().__init__()
        g = torch.Generator().manual_seed(0)
        self.W_enc = torch.nn.Parameter(torch.randn(D_MODEL, N_FEATURES, generator=g))
        self.b_enc = torch.nn.Parameter(torch.full((N_FEATURES,), 5.0))  # every feature fires
        self.W_dec = torch.nn.Parameter(torch.randn(N_FEATURES, D_MODEL, generator=g))

    def encode(self, x):
        return torch.relu(x @ self.W_enc + self.b_enc)


@pytest.fixture(scope="module")
def tiny():
    from transformer_lens import HookedTransformer, HookedTransformerConfig

    torch.manual_seed(0)
    cfg = HookedTransformerConfig(n_layers=3, d_model=D_MODEL, n_ctx=64, d_head=4, n_heads=2,
                                  d_mlp=32, d_vocab=50, act_fn="relu", normalization_type="LN")
    model = HookedTransformer(cfg)
    model.to_tokens = lambda s: torch.tensor([[0] + [1 + ord(c) % 49 for c in s]])  # 0 = BOS
    return model, FakeSAE()


# --- pure helpers -----------------------------------------------------------


def test_firing_frequency():
    a = TokenActivations.from_dense(["a", "b"], np.array([[1.0, 0.0, 0.0], [1.0, 2.0, 0.0]]))
    np.testing.assert_allclose(firing_frequency([a]), [1.0, 0.5, 0.0])


def test_matched_random_features_excludes_and_matches():
    freq = np.array([0.5, 0.49, 0.51, 0.0, 0.01, 0.9, 0.48])
    picked = matched_random_features(freq, target=0, n=2, rng=np.random.default_rng(0), exclude={1})
    assert len(picked) == 2
    assert 0 not in picked and 1 not in picked and 3 not in picked  # target, excluded, never fires
    assert all(abs(freq[i] - 0.5) < 0.5 for i in picked)


def test_components_for_layer():
    comps = components_for_layer(2, n_heads=4)
    assert len(comps) == 3 * (4 + 1)
    assert comps[-1] == (2, "mlp", None)


# --- hooks on the tiny model -----------------------------------------------


def test_steer_edit_scale_one_is_identity():
    sae = FakeSAE()
    resid = torch.randn(1, 5, D_MODEL)
    torch.testing.assert_close(_residual_with_edit(sae, resid, 3, 1.0), resid)


def test_steer_edit_ablation_removes_feature_contribution():
    sae = FakeSAE()
    resid = torch.randn(1, 5, D_MODEL)
    acts = sae.encode(resid)[..., 3:4]
    expected = resid - acts * sae.W_dec[3]
    torch.testing.assert_close(_residual_with_edit(sae, resid, 3, 0.0), expected)


def test_output_effects_noop_edit_has_zero_kl(tiny):
    model, _ = tiny
    effects = output_effects(model, LAYER, "abcdef", {"noop": add_vector_hook(torch.zeros(D_MODEL))})
    assert effects["noop"].kl == pytest.approx(0.0, abs=1e-6)
    assert np.abs(effects["noop"].delta).max() < 1e-5


def test_output_effects_real_edit_changes_output(tiny):
    model, _ = tiny
    effects = output_effects(model, LAYER, "abcdef", {"push": add_vector_hook(10 * torch.ones(D_MODEL))})
    assert effects["push"].kl > 0


def test_token_attribution_covers_tokens_up_to_peak(tiny):
    model, sae = tiny
    ta = token_attribution(model, sae, LAYER, 2, "abcdef")
    assert ta is not None
    assert len(ta.tokens) == len(ta.drops) == ta.target_pos
    clean = feature_acts(model, sae, model.to_tokens("abcdef"), LAYER, 2)[0]
    assert ta.clean_act == pytest.approx(float(clean[1:].max()))


def test_component_attribution_has_every_component(tiny):
    model, sae = tiny
    drops = component_attribution(model, sae, LAYER, 2, "abcdef", batch_size=3)  # forces chunking
    assert set(drops) == {f"L{l}H{h}" for l in range(LAYER + 1) for h in range(2)} | {
        f"L{l}MLP" for l in range(LAYER + 1)
    }
    assert any(abs(v) > 0 for v in drops.values())


def test_component_ablation_only_touches_its_row(tiny):
    """Batched ablations must not leak between rows: a lone row-0 ablation
    gives the same result as row 0 of a bigger batch."""
    from src.causal import _component_hooks

    model, sae = tiny
    tokens = model.to_tokens("abcdef")
    store = {}

    def capture(x, hook):
        store[hook.name] = x[0, 1:].mean(dim=0).clone()

    names = [f"blocks.{l}.attn.hook_z" for l in range(LAYER + 1)] + [f"blocks.{l}.hook_mlp_out" for l in range(LAYER + 1)]
    feature_acts(model, sae, tokens, LAYER, 2, [(n, capture) for n in names])
    comps = components_for_layer(LAYER, 2)
    batched = feature_acts(model, sae, tokens.repeat(len(comps), 1), LAYER, 2,
                           _component_hooks(list(enumerate(comps)), store))
    single = feature_acts(model, sae, tokens, LAYER, 2, _component_hooks([(0, comps[3])], store))
    np.testing.assert_allclose(batched[3], single[0], rtol=1e-5, atol=1e-5)
