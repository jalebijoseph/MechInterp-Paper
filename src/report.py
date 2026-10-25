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
        tok = f"`{top[0][0]}` ({100 * f['tokens']['positive_top_token_share']:.0f}%)" if top else "–"
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
            ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center", fontsize=9, color=TEXT_PRIMARY)
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


def main(paths: list[Path]) -> None:
    for path in paths:
        with open(path, encoding="utf-8") as f:
            summary = json.load(f)
        stem = path.stem.removesuffix("_summary")
        (RESULTS_DIR / f"{stem}.md").write_text(markdown_table(summary), encoding="utf-8")
        plot_heatmap(summary, RESULTS_DIR / "figures" / f"{stem}_auroc.png")
        print(f"Wrote results/{stem}.md and results/figures/{stem}_auroc.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("summaries", nargs="+", type=Path)
    main(parser.parse_args().summaries)
