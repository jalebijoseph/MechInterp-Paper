"""Turn run_pipeline summaries into a markdown table + AUROC heatmap.

Reads results/<concept>_layer<layer>_<reduce>_summary.json and writes
results/<same stem>.md and results/figures/<same stem>_auroc.png.

Usage:
    python -m src.report results/deception_layer6_mean_summary.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from src.config import RESULTS_DIR

# Diverging blue <-> red around AUROC 0.5 (chance), neutral gray midpoint.
BLUE, MID, RED = "#1c5cab", "#f0efec", "#e34948"
TEXT_PRIMARY, TEXT_SECONDARY, SURFACE = "#0b0b0b", "#52514e", "#fcfcfb"


def auroc_matrix(summary: dict) -> tuple[np.ndarray, list[int], list[str]]:
    """(n_features, n_splits) AUROC matrix, discovery first."""
    feats = summary["features"]
    splits = ["discovery"] + list(feats[0]["held_out"]) if feats else ["discovery"]
    rows = [[f["discovery"]["auroc"]] + [f["held_out"][s]["auroc"] for s in splits[1:]] for f in feats]
    return np.array(rows, dtype=float), [f["feature_id"] for f in feats], splits


def markdown_table(summary: dict) -> str:
    matrix, ids, splits = auroc_matrix(summary)
    cfg = summary["config"]
    lines = [
        f"### {cfg['concept']} — layer {cfg['layer']} ({cfg.get('sae_id', '')}, {cfg.get('reduce', 'mean')} over tokens)",
        "",
        "AUROC per held-out split (threshold from discovery set). Mean = mean over held-out splits.",
        "",
        "| feature | " + " | ".join(splits) + " | mean held-out | top + token (share) |",
        "|---" * (len(splits) + 3) + "|",
    ]
    for row, fid, f in zip(matrix, ids, summary["features"]):
        top = f["tokens"]["positive_top"]
        tok = f"`{top[0][0] or '␣'}` ({100 * f['tokens']['positive_top_token_share']:.0f}%)" if top else "–"
        held = row[1:].mean() if len(row) > 1 else float("nan")
        cells = " | ".join(f"{v:.2f}" for v in row)
        lines.append(f"| {fid} | {cells} | {held:.2f} | {tok} |")
    return "\n".join(lines) + "\n"


def plot_heatmap(summary: dict, out_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    matrix, ids, splits = auroc_matrix(summary)
    cmap = LinearSegmentedColormap.from_list("auroc", [RED, MID, BLUE])
    cfg = summary["config"]

    fig, ax = plt.subplots(figsize=(1.3 * len(splits) + 2, 0.45 * len(ids) + 1.6), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    im = ax.imshow(matrix, cmap=cmap, vmin=0.0, vmax=1.0, aspect="auto")
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            # Light text on the darkest cells so values stay readable.
            color = SURFACE if abs(matrix[i, j] - 0.5) > 0.35 else TEXT_PRIMARY
            ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center", fontsize=9, color=color)
    # 2px surface gap between cells.
    ax.set_xticks(np.arange(-0.5, len(splits)), minor=True)
    ax.set_yticks(np.arange(-0.5, len(ids)), minor=True)
    ax.grid(which="minor", color=SURFACE, linewidth=2)
    ax.tick_params(which="minor", length=0)
    ax.set_xticks(range(len(splits)), [s.replace("_", "\n") for s in splits], fontsize=9, color=TEXT_SECONDARY)
    ax.set_yticks(range(len(ids)), [str(i) for i in ids], fontsize=9, color=TEXT_SECONDARY)
    ax.tick_params(which="major", length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_ylabel("SAE feature", color=TEXT_SECONDARY)
    ax.set_title(f"{cfg['concept'].capitalize()} features, layer {cfg['layer']}: AUROC by prompt set",
                 loc="left", fontsize=11, color=TEXT_PRIMARY)
    cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cbar.set_ticks([0, 0.5, 1])
    cbar.set_ticklabels(["0 (inverted)", "0.5 (chance)", "1 (separates)"])
    cbar.ax.tick_params(labelsize=8, colors=TEXT_SECONDARY, length=0)
    cbar.outline.set_visible(False)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, facecolor=SURFACE)
    plt.close(fig)


def plot_component_heatmap(comp_mean: dict[str, float], layer: int, n_heads: int, title: str, out_path: Path) -> None:
    """Layers x (heads + MLP) heatmap of the fraction of a feature's activation
    lost when each component is mean-ablated. Blue = component helps cause
    the activation, red = component suppresses it."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    cols = [f"H{h}" for h in range(n_heads)] + ["MLP"]
    matrix = np.array([[comp_mean.get(f"L{l}{c}", np.nan) for c in cols] for l in range(layer + 1)])
    lim = max(0.05, float(np.nanmax(np.abs(matrix)))) if np.isfinite(matrix).any() else 1.0
    cmap = LinearSegmentedColormap.from_list("drop", [RED, MID, BLUE])

    fig, ax = plt.subplots(figsize=(0.62 * len(cols) + 2.2, 0.32 * (layer + 1) + 1.6), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    im = ax.imshow(matrix, cmap=cmap, vmin=-lim, vmax=lim, aspect="auto")
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            v = matrix[i, j]
            if np.isfinite(v) and abs(v) >= 0.1:
                color = SURFACE if abs(v) > 0.7 * lim else TEXT_PRIMARY
                ax.text(j, i, f"{100 * v:.0f}", ha="center", va="center", fontsize=7, color=color)
    ax.set_xticks(np.arange(-0.5, len(cols)), minor=True)
    ax.set_yticks(np.arange(-0.5, layer + 1), minor=True)
    ax.grid(which="minor", color=SURFACE, linewidth=2)
    ax.tick_params(which="minor", length=0)
    ax.set_xticks(range(len(cols)), cols, fontsize=8, color=TEXT_SECONDARY)
    ax.set_yticks(range(layer + 1), [str(l) for l in range(layer + 1)], fontsize=8, color=TEXT_SECONDARY)
    ax.tick_params(which="major", length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_ylabel("layer", color=TEXT_SECONDARY)
    ax.set_title(title, loc="left", fontsize=10, color=TEXT_PRIMARY)
    cbar = fig.colorbar(im, ax=ax, fraction=0.05, pad=0.02)
    cbar.set_ticks([-lim, 0, lim])
    cbar.set_ticklabels([f"{-100 * lim:.0f}% (suppresses)", "0", f"{100 * lim:.0f}% (causes)"])
    cbar.ax.tick_params(labelsize=7, colors=TEXT_SECONDARY, length=0)
    cbar.outline.set_visible(False)
    fig.text(0.01, 0.01, "Cell = % of the feature's peak activation lost when that head/MLP is mean-ablated (|value| ≥ 10% labeled).",
             fontsize=7, color=TEXT_SECONDARY)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, facecolor=SURFACE)
    plt.close(fig)


# Categorical slots 1-3 of the validated palette (pass all-pairs CVD checks).
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]


def layer_table(summaries: list[dict]) -> dict[str, dict[int, dict[str, float]]]:
    """concept -> layer -> {top10_mean, best, tricky_top10_mean} held-out AUROCs."""
    out: dict[str, dict[int, dict[str, float]]] = {}
    for s in summaries:
        feats = s["features"]
        held = [np.mean([v["auroc"] for v in f["held_out"].values()]) for f in feats]
        tricky = [f["held_out"]["tricky_negatives"]["auroc"] for f in feats if "tricky_negatives" in f["held_out"]]
        out.setdefault(s["config"]["concept"], {})[s["config"]["layer"]] = {
            "top10_mean": float(np.mean(held)),
            "best": float(np.max(held)),
            "tricky_top10_mean": float(np.mean(tricky)) if tricky else float("nan"),
        }
    return out


def plot_layer_comparison(table: dict, out_path: Path) -> None:
    """Small multiples: held-out AUROC vs. layer, one line per concept."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    panels = [("top10_mean", "Top-10 features, mean held-out AUROC"),
              ("best", "Best feature, mean held-out AUROC"),
              ("tricky_top10_mean", "Top-10 features, tricky negatives")]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), sharey=True, facecolor=SURFACE)
    for ax, (key, title) in zip(axes, panels):
        ax.set_facecolor(SURFACE)
        ax.axhline(0.5, color=TEXT_SECONDARY, linewidth=0.8, linestyle=(0, (3, 3)))
        for color, concept in zip(SERIES, sorted(table)):
            layers = sorted(table[concept])
            ys = [table[concept][l][key] for l in layers]
            ax.plot(layers, ys, color=color, linewidth=2, marker="o", markersize=6,
                    markeredgecolor=SURFACE, markeredgewidth=2, label=concept)
            ax.annotate(concept, (layers[-1], ys[-1]), xytext=(6, 0), textcoords="offset points",
                        va="center", fontsize=8, color=TEXT_SECONDARY)
        all_layers = sorted({l for c in table.values() for l in c})
        ax.set_xticks(all_layers)
        ax.set_xlim(min(all_layers) - 1, max(all_layers) + 4)
        ax.set_ylim(0, 1.02)
        ax.set_title(title, loc="left", fontsize=10, color=TEXT_PRIMARY)
        ax.set_xlabel("layer", color=TEXT_SECONDARY, fontsize=9)
        ax.tick_params(colors=TEXT_SECONDARY, labelsize=8, length=0)
        ax.grid(axis="y", color="#e6e5e1", linewidth=0.8)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color("#c3c2b7")
    axes[0].set_ylabel("AUROC (0.5 = chance)", color=TEXT_SECONDARY, fontsize=9)
    axes[0].legend(frameon=False, fontsize=8, loc="lower left", labelcolor=TEXT_SECONDARY)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, facecolor=SURFACE)
    plt.close(fig)


def layer_markdown(table: dict) -> str:
    lines = ["| concept | layer | top-10 mean held-out | best feature held-out | top-10 mean tricky negatives |",
             "|---|---|---|---|---|"]
    for concept in sorted(table):
        for layer in sorted(table[concept]):
            r = table[concept][layer]
            lines.append(f"| {concept} | {layer} | {r['top10_mean']:.2f} | {r['best']:.2f} | {r['tricky_top10_mean']:.2f} |")
    return "\n".join(lines) + "\n"


def main(paths: list[Path]) -> None:
    summaries = []
    for path in paths:
        with open(path, encoding="utf-8") as f:
            summary = json.load(f)
        stem = path.stem.removesuffix("_summary")
        (RESULTS_DIR / f"{stem}.md").write_text(markdown_table(summary), encoding="utf-8")
        plot_heatmap(summary, RESULTS_DIR / "figures" / f"{stem}_auroc.png")
        print(f"Wrote results/{stem}.md and results/figures/{stem}_auroc.png")
        summaries.append(summary)
    layers = {s["config"]["layer"] for s in summaries}
    if len(layers) > 1:
        table = layer_table(summaries)
        (RESULTS_DIR / "layer_comparison.md").write_text(layer_markdown(table), encoding="utf-8")
        plot_layer_comparison(table, RESULTS_DIR / "figures" / "layer_comparison.png")
        print("Wrote results/layer_comparison.md and results/figures/layer_comparison.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("summaries", nargs="+", type=Path)
    main(parser.parse_args().summaries)
