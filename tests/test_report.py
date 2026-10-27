import pytest

from src.report import auroc_matrix, markdown_table, plot_heatmap


def _summary():
    feat = lambda fid, d, a, b: {
        "feature_id": fid,
        "discovery": {"auroc": d},
        "held_out": {"same_style": {"auroc": a}, "paraphrase": {"auroc": b}},
        "tokens": {"positive_top": [[" lie", 3.0]], "positive_top_token_share": 0.5},
    }
    return {"config": {"concept": "deception", "layer": 6}, "features": [feat(1, 0.9, 0.8, 0.4), feat(2, 0.8, 0.7, 0.6)]}


def test_auroc_matrix_shape_and_order():
    matrix, ids, splits = auroc_matrix(_summary())
    assert matrix.shape == (2, 3)
    assert ids == [1, 2]
    assert splits == ["discovery", "same_style", "paraphrase"]


def test_markdown_table_has_row_per_feature():
    table = markdown_table(_summary())
    assert "| 1 | 0.90 | 0.80 | 0.40 | 0.60 |" in table
    assert table.count("\n| 2 |") == 1


def test_plot_heatmap_writes_file(tmp_path):
    out = tmp_path / "fig.png"
    plot_heatmap(_summary(), out)
    assert out.stat().st_size > 0


from src.report import layer_markdown, layer_table, plot_layer_comparison


def _layer_summary(concept, layer, auroc):
    f = {"feature_id": 1, "discovery": {"auroc": 0.9},
         "held_out": {"tricky_negatives": {"auroc": auroc}, "paraphrase": {"auroc": auroc + 0.1}},
         "tokens": {"positive_top": [], "positive_top_token_share": 0.0}}
    return {"config": {"concept": concept, "layer": layer}, "features": [f]}


def test_layer_table_and_plot(tmp_path):
    table = layer_table([_layer_summary("math", 6, 0.6), _layer_summary("math", 12, 0.8),
                         _layer_summary("deception", 6, 0.5)])
    assert table["math"][12]["top10_mean"] == pytest.approx(0.85)
    assert table["math"][6]["tricky_top10_mean"] == pytest.approx(0.6)
    assert "| math | 12 | 0.85 |" in layer_markdown(table)
    plot_layer_comparison(table, tmp_path / "layers.png")
    assert (tmp_path / "layers.png").stat().st_size > 0
