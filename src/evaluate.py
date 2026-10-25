"""Consistency metrics for a candidate feature, on discovery vs. held-out sets.

Only needs numpy + scikit-learn, not the rest of the ML stack, so this
is unit-testable with synthetic data (see tests/test_evaluate.py).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import precision_recall_curve, roc_auc_score


@dataclass
class ConsistencyResult:
    feature_id: int
    auroc: float
    precision_at_threshold: float
    recall_at_threshold: float
    threshold: float


def feature_auroc(feature_values: np.ndarray, labels: np.ndarray) -> float:
    """AUROC for one feature separating positive (1) from negative (0) prompts.

    `feature_values`: 1D array, that feature's activation per prompt.
    `labels`: 1D array of 0/1, same length.
    """
    if len(np.unique(labels)) < 2:
        raise ValueError("Need both positive and negative examples to compute AUROC")
    return float(roc_auc_score(labels, feature_values))


def best_threshold(feature_values: np.ndarray, labels: np.ndarray) -> float:
    """Threshold maximizing F1, from the precision-recall curve."""
    precision, recall, thresholds = precision_recall_curve(labels, feature_values)
    # precision/recall have one more element than thresholds; align them.
    precision, recall = precision[:-1], recall[:-1]
    f1 = np.where((precision + recall) > 0, 2 * precision * recall / (precision + recall + 1e-12), 0.0)
    if len(thresholds) == 0:
        return float(np.median(feature_values))
    return float(thresholds[int(np.argmax(f1))])


def precision_recall_at_threshold(
    feature_values: np.ndarray, labels: np.ndarray, threshold: float
) -> tuple[float, float]:
    preds = feature_values >= threshold
    labels_bool = labels.astype(bool)
    tp = int(np.sum(preds & labels_bool))
    fp = int(np.sum(preds & ~labels_bool))
    fn = int(np.sum(~preds & labels_bool))
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    return precision, recall


def evaluate_feature(
    feature_id: int,
    feature_values: np.ndarray,
    labels: np.ndarray,
    threshold: float | None = None,
) -> ConsistencyResult:
    """Full evaluation of one feature on one (discovery or held-out) set.

    If `threshold` isn't given, it's picked to maximize F1 on this same
    set — pass the discovery-set threshold in when evaluating a
    held-out set, so the comparison is apples to apples.
    """
    auroc = feature_auroc(feature_values, labels)
    thr = threshold if threshold is not None else best_threshold(feature_values, labels)
    precision, recall = precision_recall_at_threshold(feature_values, labels, thr)
    return ConsistencyResult(
        feature_id=feature_id,
        auroc=auroc,
        precision_at_threshold=precision,
        recall_at_threshold=recall,
        threshold=thr,
    )


def score_drop(discovery_result: ConsistencyResult, held_out_result: ConsistencyResult) -> float:
    """AUROC drop from the discovery set to a held-out set (positive = got worse)."""
    return discovery_result.auroc - held_out_result.auroc
