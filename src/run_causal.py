"""Causal experiments for a concept's candidate features at one layer.

Needs `results/<concept>_layer<layer>_mean_summary.json` from
run_pipeline (to pick features and reuse cached activations). By
default tests the top-ranked feature on the discovery set (what a naive
analysis would pick) plus the two most consistent on held-out data.

For each feature, on the positive prompts where it fires most:
1. Token attribution: which input tokens cause its peak activation.
2. Component attribution: which attention heads / MLPs cause it.
3. Output effects: KL divergence of next-token predictions when the
   feature is ablated or steered, vs. random features with a similar
   firing rate (the control); and which tokens it promotes.

Usage:
    python -m src.run_causal --concept deception --layer 6
    python -m src.run_causal --concept deception --layer 6 --features 5938 6058
"""

from __future__ import annotations

import argparse
import json

import numpy as np

from src.activations import get_token_activations, load_prompt_set
from src.causal import (
    add_vector_hook,
    component_attribution,
    decoder_logit_lens,
    firing_frequency,
    matched_random_features,
    output_effects,
    token_attribution,
    top_tokens,
    ablate_feature_hook,
)
from src.config import DATA_DIR, RESULTS_DIR, SEED, seed_everything, setup_logging
from src.load import load_model, load_sae, sae_id_for_layer
from src.report import plot_component_heatmap

logger = setup_logging()


def pick_features(summary: dict, n_consistent: int = 2) -> dict[int, str]:
    """Top-ranked-on-discovery feature + the most consistent on held-out sets."""
    feats = summary["features"]
    roles = {feats[0]["feature_id"]: "top-ranked on discovery set"}
    held = lambda f: float(np.mean([v["auroc"] for v in f["held_out"].values()]))
    for f in sorted(feats, key=held, reverse=True):
        if len(roles) > n_consistent:
            break
        roles.setdefault(f["feature_id"], f"most consistent on held-out (mean AUROC {held(f):.2f})")
    return roles


def run(concept: str, layer: int, features: list[int] | None = None, n_prompts: int = 6,
        n_random: int = 3, steer_scale: float = 4.0, model=None, sae=None) -> dict:
    seed_everything()
    rng = np.random.default_rng(SEED)
    sae_id = sae_id_for_layer(layer)

    with open(RESULTS_DIR / f"{concept}_layer{layer}_mean_summary.json", encoding="utf-8") as f:
        summary = json.load(f)
    roles = {fid: "chosen manually" for fid in features} if features else pick_features(summary)
    logger.info("Causal run: concept=%s layer=%d features=%s", concept, layer, roles)

    model = model or load_model()
    sae = sae or load_sae(layer)

    prompt_set = load_prompt_set(DATA_DIR / f"{concept}.json")
    splits = {"discovery": prompt_set["discovery"], **prompt_set.get("held_out", {})}
    positives = [p for s in splits.values() for p in s["positive"]]
    negatives = [p for s in splits.values() for p in s["negative"]]
    pos_acts = get_token_activations(model, sae, positives, layer, sae_id)  # cache hits after run_pipeline
    neg_acts = get_token_activations(model, sae, negatives, layer, sae_id)
    freq = firing_frequency(pos_acts + neg_acts)

    results = {"config": {"concept": concept, "layer": layer, "sae_id": sae_id, "n_prompts": n_prompts,
                          "n_random": n_random, "steer_scale": steer_scale, "seed": SEED},
               "features": []}

    for fid, role in roles.items():
        peak = np.array([a.reduce("max")[fid] for a in pos_acts])
        order = [i for i in np.argsort(peak)[::-1] if peak[i] > 0][:n_prompts]
        prompts = [positives[i] for i in order]
        if not prompts:
            logger.warning("Feature %d never fires on a positive prompt; skipping", fid)
            continue
        logger.info("Feature %d (%s): %d prompts", fid, role, len(prompts))

        # 1. Token attribution
        tok_attr = [ta for p in prompts if (ta := token_attribution(model, sae, layer, fid, p)) is not None]
        self_share = []
        for ta in tok_attr:
            pos_drops = np.clip(ta.drops, 0, None)
            if pos_drops.sum() > 0:
                self_share.append(float(pos_drops[-1] / pos_drops.sum()))

        # 2. Component attribution (relative to the clean activation)
        comp_rel: dict[str, list[float]] = {}
        for ta in tok_attr:
            drops = component_attribution(model, sae, layer, fid, ta.prompt)
            if drops is None:
                continue
            for name, d in drops.items():
                comp_rel.setdefault(name, []).append(d / ta.clean_act)
        comp_mean = {name: float(np.mean(v)) for name, v in comp_rel.items()}

        # 3. Output effects vs. matched random features
        randoms = matched_random_features(freq, fid, n_random, rng, exclude=set(roles))
        w = sae.W_dec.detach()
        typical = float(peak[order].mean())
        steer_vec = steer_scale * typical * w[fid]
        hooks = {"ablate": ablate_feature_hook(sae, fid), "steer": add_vector_hook(steer_vec)}
        for r in randoms:
            hooks[f"ablate_rand{r}"] = ablate_feature_hook(sae, r)
            hooks[f"steer_rand{r}"] = add_vector_hook(w[r] / w[r].norm() * steer_vec.norm())
        effects = [output_effects(model, layer, p, hooks) for p in prompts]
        kl = {name: float(np.mean([e[name].kl for e in effects])) for name in hooks}
        mean_delta = {name: np.mean([e[name].delta for e in effects], axis=0) for name in ("ablate", "steer")}
        rand_ablate = [kl[f"ablate_rand{r}"] for r in randoms]
        rand_steer = [kl[f"steer_rand{r}"] for r in randoms]

        entry = {
            "feature_id": fid,
            "role": role,
            "prompts": prompts,
            "token_attribution": [ta.__dict__ for ta in tok_attr],
            "target_token_share_of_cause": float(np.mean(self_share)) if self_share else None,
            "components_top": sorted(comp_mean.items(), key=lambda kv: kv[1], reverse=True)[:10],
            "components_all": comp_mean,
            "kl_ablate": kl["ablate"],
            "kl_ablate_random": rand_ablate,
            "kl_steer": kl["steer"],
            "kl_steer_random": rand_steer,
            "random_features": randoms,
            # Removing the feature lowers these tokens' log-prob most -> what it normally promotes.
            "promotes_when_present": top_tokens(model, -mean_delta["ablate"]),
            "promoted_by_steering": top_tokens(model, mean_delta["steer"]),
            "suppressed_by_steering": top_tokens(model, mean_delta["steer"], largest=False),
            "decoder_logit_lens": decoder_logit_lens(model, sae, fid),
        }
        results["features"].append(entry)
        plot_component_heatmap(comp_mean, layer, model.cfg.n_heads,
                               f"Feature {fid} ({concept}, layer {layer}): what causes it",
                               RESULTS_DIR / "figures" / f"causal_{concept}_layer{layer}_f{fid}_components.png")
        logger.info(
            "feature %d  ablate KL=%.4f (random %.4f)  steer KL=%.4f (random %.4f)  self-token share=%s  top comps=%s",
            fid, kl["ablate"], float(np.mean(rand_ablate)) if rand_ablate else float("nan"),
            kl["steer"], float(np.mean(rand_steer)) if rand_steer else float("nan"),
            f"{entry['target_token_share_of_cause']:.2f}" if self_share else "-",
            [n for n, _ in entry["components_top"][:3]],
        )

    stem = f"causal_{concept}_layer{layer}"
    with open(RESULTS_DIR / f"{stem}.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=float)
    (RESULTS_DIR / f"{stem}.md").write_text(causal_markdown(results), encoding="utf-8")
    logger.info("Wrote results/%s.json and .md", stem)
    return results


def _fmt_tokens(pairs, n=6) -> str:
    return ", ".join(f"`{t.strip() or '␣'}`" for t, _ in pairs[:n])


def causal_markdown(results: dict) -> str:
    cfg = results["config"]
    lines = [f"### Causal experiments: {cfg['concept']}, layer {cfg['layer']}", ""]
    for f in results["features"]:
        rand_a = np.mean(f["kl_ablate_random"]) if f["kl_ablate_random"] else float("nan")
        rand_s = np.mean(f["kl_steer_random"]) if f["kl_steer_random"] else float("nan")
        share = f["target_token_share_of_cause"]
        lines += [
            f"#### Feature {f['feature_id']} ({f['role']})",
            "",
            "**What causes it**",
            "",
            f"- Share of the input-token effect coming from the token it fires on: "
            f"{'–' if share is None else f'{100 * share:.0f}%'} "
            "(high = a keyword detector; low = it integrates context).",
            "- Top components (mean fraction of the activation lost when ablated): "
            + ", ".join(f"{n} ({100 * v:.0f}%)" for n, v in f["components_top"][:5]),
            "- Example prompts (top input-token causes, fraction of peak activation lost):",
        ]
        for ta in f["token_attribution"][:3]:
            ranked = sorted(zip(ta["tokens"], ta["drops"]), key=lambda kv: kv[1], reverse=True)[:3]
            causes = ", ".join(f"`{t.strip() or '␣'}` ({100 * d / ta['clean_act']:.0f}%)" for t, d in ranked)
            lines.append(f"  - \"{ta['prompt']}\" → peaks on `{ta['target_token'].strip() or '␣'}`; causes: {causes}")
        lines += [
            "",
            "**What it causes**",
            "",
            f"- Ablating it: KL {f['kl_ablate']:.4f} vs. {rand_a:.4f} for matched random features "
            f"({f['kl_ablate'] / rand_a:.1f}×)" if rand_a > 0 else f"- Ablating it: KL {f['kl_ablate']:.4f}",
            f"- Steering along it: KL {f['kl_steer']:.4f} vs. {rand_s:.4f} for random directions of the same norm",
            f"- Tokens it promotes (log-prob drop when ablated): {_fmt_tokens(f['promotes_when_present'])}",
            f"- Steering promotes: {_fmt_tokens(f['promoted_by_steering'])}",
            f"- Decoder logit lens (rough): {_fmt_tokens(f['decoder_logit_lens'])}",
            "",
        ]
    return "\n".join(lines)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--concept", nargs="+", default=["deception"])
    parser.add_argument("--layer", type=int, nargs="+", default=[6])
    parser.add_argument("--features", type=int, nargs="*", help="feature ids (default: auto-pick 3)")
    parser.add_argument("--n-prompts", type=int, default=6)
    parser.add_argument("--n-random", type=int, default=3)
    parser.add_argument("--steer-scale", type=float, default=4.0,
                        help="steering vector = scale × typical peak activation × decoder row")
    args = parser.parse_args()

    model = load_model()
    for layer in args.layer:
        sae = load_sae(layer)
        for concept in args.concept:
            run(concept, layer, args.features, args.n_prompts, args.n_random, args.steer_scale, model=model, sae=sae)
        del sae
