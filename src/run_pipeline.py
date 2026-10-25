"""End-to-end pipeline for one concept at one layer: the "get it working
end to end" milestone from the project instructions, before expanding to
more concepts/layers/held-out types.

This is the one entry point that actually loads the model + SAE and
therefore triggers the large downloads described in load.py — it does
nothing on import, only when run, and running it is a deliberate choice
you make, not something that happens automatically.

Usage:
    python -m src.run_pipeline --concept deception --layer 6
"""

from __future__ import annotations

import argparse
import json

from src.activations import get_feature_activations, load_prompt_set
from src.config import DATA_DIR, RESULTS_DIR, RunConfig, seed_everything, setup_logging
from src.evaluate import evaluate_feature
from src.find_features import top_k_features
from src.load import load_model, load_sae

logger = setup_logging()


def run(concept: str, layer: int, top_k: int = 10) -> dict:
    seed_everything()
    config = RunConfig(concept=concept, layer=layer, top_k_features=top_k)
    logger.info("Run config: %s", config.as_dict())

    prompt_set = load_prompt_set(DATA_DIR / f"{concept}.json")
    discovery = prompt_set["discovery"]

    logger.info("Loading model and SAE (layer %d) — this downloads weights the first time.", layer)
    model = load_model()
    sae = load_sae(layer)

    logger.info("Computing discovery-set activations (%d positive, %d negative prompts)",
                len(discovery["positive"]), len(discovery["negative"]))
    pos_acts = get_feature_activations(model, sae, discovery["positive"], layer)
    neg_acts = get_feature_activations(model, sae, discovery["negative"], layer)

    candidates = top_k_features(pos_acts, neg_acts, k=top_k)
    logger.info("Top candidate feature: id=%d score=%.4f", candidates[0].feature_id, candidates[0].score)

    import numpy as np

    discovery_values = np.concatenate([pos_acts, neg_acts])[:, candidates[0].feature_id]
    discovery_labels = np.array([1] * len(discovery["positive"]) + [0] * len(discovery["negative"]))
    discovery_result = evaluate_feature(candidates[0].feature_id, discovery_values, discovery_labels)
    logger.info("Discovery-set AUROC for top feature: %.4f", discovery_result.auroc)

    held_out_results = {}
    for name, split in prompt_set.get("held_out", {}).items():
        pos = get_feature_activations(model, sae, split["positive"], layer)
        neg = get_feature_activations(model, sae, split["negative"], layer)
        values = np.concatenate([pos, neg])[:, candidates[0].feature_id]
        labels = np.array([1] * len(split["positive"]) + [0] * len(split["negative"]))
        result = evaluate_feature(
            candidates[0].feature_id, values, labels, threshold=discovery_result.threshold
        )
        held_out_results[name] = result
        logger.info("Held-out[%s] AUROC: %.4f (drop: %.4f)", name, result.auroc, discovery_result.auroc - result.auroc)

    summary = {
        "config": config.as_dict(),
        "top_feature_id": candidates[0].feature_id,
        "discovery_auroc": discovery_result.auroc,
        "held_out": {name: r.auroc for name, r in held_out_results.items()},
    }

    out_path = RESULTS_DIR / f"{concept}_layer{layer}_summary.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    logger.info("Wrote summary to %s", out_path)

    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--concept", default="deception")
    parser.add_argument("--layer", type=int, default=6)
    parser.add_argument("--top-k", type=int, default=10)
    args = parser.parse_args()

    run(concept=args.concept, layer=args.layer, top_k=args.top_k)
