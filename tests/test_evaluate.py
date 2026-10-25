import numpy as np
import pytest

from src.evaluate import (
    best_threshold,
    evaluate_feature,
    feature_auroc,
    precision_recall_at_threshold,
    score_drop,
)


def test_auroc_perfect_separation():
    # Positives all higher than negatives -> perfect separation.
    values = np.array([0.1, 0.2, 0.3, 0.8, 0.9, 1.0])
    labels = np.array([0, 0, 0, 1, 1, 1])
    assert feature_auroc(values, labels) == pytest.approx(1.0)


def test_auroc_chance_level_for_random_feature():
    rng = np.random.default_rng(0)
    values = rng.normal(size=200)
    labels = rng.integers(0, 2, size=200)
    auroc = feature_auroc(values, labels)
    # Not a tight bound (it's random), just confirm it's near chance, not near 0 or 1.
    assert 0.35 < auroc < 0.65


def test_auroc_requires_both_classes():
    values = np.array([0.1, 0.2, 0.3])
    labels = np.array([1, 1, 1])
    with pytest.raises(ValueError):
        feature_auroc(values, labels)


def test_precision_recall_at_threshold_known_case():
    values = np.array([0.1, 0.9, 0.2, 0.8])
    labels = np.array([0, 1, 1, 1])
    # threshold 0.5: predicts positive for indices 1 and 3 only
    precision, recall = precision_recall_at_threshold(values, labels, threshold=0.5)
    assert precision == pytest.approx(1.0)  # both predicted positives are true positives
    assert recall == pytest.approx(2 / 3)  # missed one of the three true positives


def test_best_threshold_is_within_range():
    values = np.array([0.1, 0.2, 0.3, 0.8, 0.9, 1.0])
    labels = np.array([0, 0, 0, 1, 1, 1])
    thr = best_threshold(values, labels)
    assert values.min() <= thr <= values.max()


def test_evaluate_feature_returns_expected_fields():
    values = np.array([0.1, 0.2, 0.3, 0.8, 0.9, 1.0])
    labels = np.array([0, 0, 0, 1, 1, 1])
    result = evaluate_feature(feature_id=7, feature_values=values, labels=labels)
    assert result.feature_id == 7
    assert result.auroc == pytest.approx(1.0)
    assert 0.0 <= result.precision_at_threshold <= 1.0
    assert 0.0 <= result.recall_at_threshold <= 1.0


def test_score_drop_positive_means_worse_on_held_out():
    discovery = evaluate_feature(0, np.array([0.1, 0.2, 0.8, 0.9]), np.array([0, 0, 1, 1]))
    held_out = evaluate_feature(
        0, np.array([0.1, 0.2, 0.8, 0.9]), np.array([0, 0, 1, 1]), threshold=discovery.threshold
    )
    # Same data twice -> no drop.
    assert score_drop(discovery, held_out) == pytest.approx(0.0)

    worse_held_out = evaluate_feature(
        0, np.array([0.5, 0.5, 0.5, 0.5]), np.array([0, 0, 1, 1]), threshold=discovery.threshold
    )
    assert score_drop(discovery, worse_held_out) > 0
