"""Find candidate features for a concept: rank by mean(positive) - mean(negative)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class CandidateFeature:
    feature_id: int
    score: float  # mean(positive activation) - mean(negative activation)
    mean_positive: float
    mean_negative: float


def rank_features(
    positive_acts: np.ndarray,
    negative_acts: np.ndarray,
) -> list[CandidateFeature]:
    """Rank every feature by how much more it fires on positive vs. negative prompts.

    `positive_acts` / `negative_acts`: arrays of shape (n_prompts, n_features),
    as returned by `activations.get_feature_activations`.
    """
    if positive_acts.shape[1] != negative_acts.shape[1]:
        raise ValueError("positive_acts and negative_acts must have the same number of features")

    mean_pos = positive_acts.mean(axis=0)
    mean_neg = negative_acts.mean(axis=0)
    scores = mean_pos - mean_neg

    order = np.argsort(scores)[::-1]
    return [
        CandidateFeature(
            feature_id=int(i),
            score=float(scores[i]),
            mean_positive=float(mean_pos[i]),
            mean_negative=float(mean_neg[i]),
        )
        for i in order
    ]


def top_k_features(
    positive_acts: np.ndarray,
    negative_acts: np.ndarray,
    k: int = 10,
) -> list[CandidateFeature]:
    return rank_features(positive_acts, negative_acts)[:k]
