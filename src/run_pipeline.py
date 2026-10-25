"""End-to-end pipeline for one concept at one layer: the "get it working
end to end" milestone from the project instructions, before expanding to
more concepts/layers/held-out types.

This is the one entry point that actually loads the model + SAE and
therefore triggers the large downloads described in load.py — it does
nothing on import, only when run, and running it is a deliberate choice
you make, not something that happens automatically.

For each of the top-k candidate features (ranked on the discovery set),
reports AUROC + precision/recall on the discovery set and on every
held-out split (using the discovery-set threshold), plus which tokens
the feature fires on.

Usage:
    python -m src.run_pipeline --concept deception --layer 6
"""

from __future__ import annotations

import argparse
import json

import numpy as np

from src.activations import get_token_activations, load_prompt_set, stack_reduced
from src.config import DATA_DIR, RESULTS_DIR, RunConfig, seed_everything, setup_logging
from src.evaluate import evaluate_feature
from src.find_features import token_profile, top_k_features
from src.load import load_model, load_sae, sae_id_for_layer

logger = setup_logging()


def _split_arrays(token_acts_pos, token_acts_neg, reduce: str):
    values = np.concatenate([stack_reduced(token_acts_pos, reduce), stack_reduced(token_acts_neg, reduce)])
    labels = np.array([1] * len(token_acts_pos) + [0] * len(token_acts_neg))
    return values, labels


def run(concept: str, layer: int, top_k: int = 10, reduce: str = "mean") -> dict:
    seed_everything()
    sae_id = sae_id_for_layer(layer)
    config = RunConfig(concept=concept, layer=layer, top_k_features=top_k, extra={"sae_id": sae_id, "reduce": reduce})
    logger.info("Run config: %s", config.as_dict())

    prompt_set = load_prompt_set(DATA_DIR / f"{concept}.json")
    splits = {"discovery": prompt_set["discovery"], **prompt_set.get("held_out", {})}

    logger.info("Loading model and SAE (layer %d) — this downloads weights the first time.", layer)
    model = load_model()
    sae = load_sae(layer)

    token_acts = {}
    for name, split in splits.items():
        logger.info("Computing activations for %s (%d positive, %d negative prompts)",
                    name, len(split["positive"]), len(split["negative"]))
        token_acts[name] = (
            get_token_activations(model, sae, split["positive"], layer, sae_id),
            get_token_activations(model, sae, split["negative"], layer, sae_id),
        )

    disc_pos, disc_neg = token_acts["discovery"]
    candidates = top_k_features(stack_reduced(disc_pos, reduce), stack_reduced(disc_neg, reduce), k=top_k)

    arrays = {name: _split_arrays(pos, neg, reduce) for name, (pos, neg) in token_acts.items()}
    all_pos_acts = [a for pos, _ in token_acts.values() for a in pos]
    all_neg_acts = [a for _, neg in token_acts.values() for a in neg]

    features = []
    for cand in candidates:
        fid = cand.feature_id
        values, labels = arrays["discovery"]
        disc = evaluate_feature(fid, values[:, fid], labels)
        entry = {
            "feature_id": fid,
            "score": cand.score,
            "mean_positive": cand.mean_positive,
            "mean_negative": cand.mean_negative,
            "discovery": {"auroc": disc.auroc, "precision": disc.precision_at_threshold,
                          "recall": disc.recall_at_threshold, "threshold": disc.threshold},
            "held_out": {},
        }
        for name in splits:
            if name == "discovery":
                continue
            values, labels = arrays[name]
            res = evaluate_feature(fid, values[:, fid], labels, threshold=disc.threshold)
            entry["held_out"][name] = {"auroc": res.auroc, "precision": res.precision_at_threshold,
                                       "recall": res.recall_at_threshold, "auroc_drop": disc.auroc - res.auroc}
        pos_profile = token_profile(fid, all_pos_acts)
        neg_profile = token_profile(fid, all_neg_acts)
        entry["tokens"] = {
            "positive_top": pos_profile.top_tokens,
            "positive_top_token_share": pos_profile.top_token_share,
            "positive_prompts_firing": f"{pos_profile.n_prompts_firing}/{pos_profile.n_prompts}",
            "negative_top": neg_profile.top_tokens,
            "negative_prompts_firing": f"{neg_profile.n_prompts_firing}/{neg_profile.n_prompts}",
        }
        features.append(entry)

        held = ", ".join(f"{n}={v['auroc']:.2f}" for n, v in entry["held_out"].items())
        top_tok = pos_profile.top_tokens[0][0] if pos_profile.top_tokens else "-"
        logger.info("feature %5d  disc AUROC=%.2f  held-out: %s  top token=%r (%.0f%% of mass)",
                    fid, disc.auroc, held, top_tok, 100 * pos_profile.top_token_share)

    summary = {"config": config.as_dict(), "features": features}
    out_path = RESULTS_DIR / f"{concept}_layer{layer}_{reduce}_summary.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    logger.info("Wrote summary to %s", out_path)

    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--concept", default="deception")
    parser.add_argument("--layer", type=int, default=6)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--reduce", choices=["mean", "max"], default="mean")
    args = parser.parse_args()

    run(concept=args.concept, layer=args.layer, top_k=args.top_k, reduce=args.reduce)
