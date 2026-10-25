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


@dataclass
class TokenProfile:
    """Where one feature fires, aggregated over a set of prompts."""

    feature_id: int
    top_tokens: list[tuple[str, float]]  # (token, summed activation), descending
    top_token_share: float  # fraction of all activation mass on the single top token
    n_prompts_firing: int
    n_prompts: int


def token_profile(feature_id: int, token_acts: list, top_n: int = 10) -> TokenProfile:
    """Which tokens a feature fires on across prompts (`TokenActivations` list).

    A high `top_token_share` means the feature is mostly tracking one
    token (e.g. " lie") rather than the concept spread across a sentence.
    Tokens are compared after stripping whitespace and lowercasing.
    """
    totals: dict[str, float] = {}
    n_firing = 0
    for acts in token_acts:
        trace = acts.feature_trace(feature_id)
        if np.any(trace > 0):
            n_firing += 1
        for tok, val in zip(acts.tokens, trace):
            if val > 0:
                norm = tok.strip().lower() or "␣"  # whitespace-only tokens, e.g. Gemma's space before digits
                totals[norm] = totals.get(norm, 0.0) + float(val)

    ranked = sorted(totals.items(), key=lambda kv: kv[1], reverse=True)
    mass = sum(totals.values())
    return TokenProfile(
        feature_id=feature_id,
        top_tokens=ranked[:top_n],
        top_token_share=(ranked[0][1] / mass) if mass > 0 else 0.0,
        n_prompts_firing=n_firing,
        n_prompts=len(token_acts),
    )
