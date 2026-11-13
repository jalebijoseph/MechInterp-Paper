"""CPU-only counterexample for the current factorial-contrast decoder.

Run from the repository root with PYTHONPATH=src:scripts. This uses synthetic
states and development templates only. It does not alter the causal experiment.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from jpeers.followup import handoff


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = [r for r in handoff.make_episodes() if r.split == "development"]
    labels = handoff._signed([r.incoming for r in rows])
    nuisance = handoff._nuisance(rows)
    interaction = nuisance[:, 2] * nuisance[:, 4]  # source x note order
    states = np.column_stack([labels + 10 * interaction, 10 * interaction])
    original = handoff.fit_component(rows, states, np.eye(2), ridge=1e-6)
    templates = np.array([r.template_id for r in rows])
    predictions, folds = [], []
    for template in sorted(set(templates)):
        train, test = templates != template, templates == template
        mean = states[train].mean(axis=0)
        centered = states[train] - mean
        # Fixed diagnostic alpha=1. No held-out templates or hyperparameter search.
        weights = np.linalg.solve(
            centered.T @ centered + np.eye(2), centered.T @ labels[train])
        predictions.extend(
            (((states[test] - mean) @ weights >= 0) == (labels[test] > 0)).tolist())
        design = np.column_stack([nuisance[train], labels[train]])
        assert np.allclose(design.T @ design, 224 * np.eye(6))
        folds.append({"test_template": template, "n_train": int(train.sum()),
                      "n_test": int(test.sum()), "weights": weights.tolist()})
    trained_accuracy = float(np.mean(predictions))
    explicit_accuracy = float(np.mean(
        (states @ np.array([1., -1.]) >= 0) == (labels > 0)))
    assert original["incoming_cv"] == .5
    assert trained_accuracy == explicit_accuracy == 1.
    assert original["incoming_basis_own_cv"] == .5
    assert original["incoming_basis_nuisance_cv"] == .5
    source = Path(handoff.__file__)
    report = {
        "kind": "synthetic estimator counterexample; not an actual Qwen probe result",
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "n_development_rows": len(rows), "n_template_folds": len(folds),
        "formula": "x1 = y + 10*(source*order); x2 = 10*(source*order)",
        "available_span": "entire two-dimensional space (identity directions)",
        "current_contrast_incoming_cv": original["incoming_cv"],
        "current_basis": original["basis"],
        "current_own_leakage_cv": original["incoming_basis_own_cv"],
        "current_nuisance_leakage_cv": original["incoming_basis_nuisance_cv"],
        "train_only_discriminative_ridge_cv": trained_accuracy,
        "explicit_x1_minus_x2_accuracy": explicit_accuracy,
        "folds": folds,
        "scope": "Proves a possible false negative; does not establish actual-run cause."
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "folds"}, indent=2))


if __name__ == "__main__":
    main()
