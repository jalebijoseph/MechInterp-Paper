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
