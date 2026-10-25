import numpy as np
import pytest

from src.find_features import rank_features, top_k_features


def test_rank_features_orders_by_mean_difference():
    # Feature 0: fires on positives only. Feature 1: fires on negatives only.
    # Feature 2: fires on both equally (should rank last).
    positive_acts = np.array([[1.0, 0.0, 0.5], [1.0, 0.0, 0.5]])
    negative_acts = np.array([[0.0, 1.0, 0.5], [0.0, 1.0, 0.5]])

    ranked = rank_features(positive_acts, negative_acts)

    assert ranked[0].feature_id == 0
    assert ranked[0].score == pytest.approx(1.0)
    assert ranked[-1].feature_id == 1


def test_top_k_features_respects_k():
    positive_acts = np.random.default_rng(0).normal(size=(5, 20))
    negative_acts = np.random.default_rng(1).normal(size=(5, 20))
    top = top_k_features(positive_acts, negative_acts, k=3)
    assert len(top) == 3


def test_rank_features_requires_matching_feature_count():
    positive_acts = np.zeros((3, 5))
    negative_acts = np.zeros((3, 6))
    with pytest.raises(ValueError):
        rank_features(positive_acts, negative_acts)


from src.activations import TokenActivations
from src.find_features import token_profile


def test_token_profile_finds_keyword_feature():
    # Feature 0 fires only on " lie" tokens -> top token share of 1.0.
    a = TokenActivations.from_dense([" a", " lie"], np.array([[0.0], [3.0]]))
    b = TokenActivations.from_dense([" Lie", " here"], np.array([[1.0], [0.0]]))
    c = TokenActivations.from_dense([" nothing"], np.array([[0.0]]))
    profile = token_profile(0, [a, b, c])
    assert profile.top_tokens[0] == ("lie", 4.0)
    assert profile.top_token_share == pytest.approx(1.0)
    assert profile.n_prompts_firing == 2
    assert profile.n_prompts == 3


def test_token_profile_never_fires():
    a = TokenActivations.from_dense([" x"], np.array([[0.0]]))
    profile = token_profile(0, [a])
    assert profile.top_tokens == []
    assert profile.top_token_share == 0.0
