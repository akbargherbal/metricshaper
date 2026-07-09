import json

import numpy as np
import pandas as pd
import pytest

from dataset_loader import (
    COLOR_PALETTE,
    Dataset,
    DatasetConfigError,
    _apply_normalize_space,
    _validate_config,
    load_datasets,
)

# ---------------------------------------------------------------------------
# Dataset._assign_colors
# ---------------------------------------------------------------------------


def test_assign_colors_defaults(sample_df, sample_config):
    cfg = json.loads(json.dumps(sample_config))
    for ch in cfg["channels"]:
        ch.pop("color", None)
        ch.pop("normalize_scope", None)
    ds = Dataset("test", sample_df.copy(), cfg)
    for ch in ds.config["channels"]:
        assert "color" in ch
        assert ch["normalize_scope"] == "global"
        assert ch["label"]  # defaults to column name if missing
    for f in ds.config["filters"]:
        assert f["label"]


def test_assign_colors_preserves_explicit_values(sample_df, sample_config):
    cfg = json.loads(json.dumps(sample_config))
    cfg["channels"][0]["color"] = "custom-color"
    ds = Dataset("test", sample_df.copy(), cfg)
    assert ds.config["channels"][0]["color"] == "custom-color"


def test_assign_colors_label_defaults_to_column(sample_df, sample_config):
    cfg = json.loads(json.dumps(sample_config))
    del cfg["channels"][0]["label"]
    ds = Dataset("test", sample_df.copy(), cfg)
    assert ds.config["channels"][0]["label"] == ds.config["channels"][0]["column"]


def test_assign_colors_cycles_through_palette_with_wraparound():
    # 10 numeric channels > len(COLOR_PALETTE)=9, forces the modulo wraparound
    n = 10
    df = pd.DataFrame({f"col{i}": [1.0, 2.0, 3.0] for i in range(n)})
    cfg = {
        "label": "Many channels",
        "identifier_column": "col0",
        "channels": [{"column": f"col{i}"} for i in range(n)],
        "filters": [],
    }
    ds = Dataset("many", df, cfg)
    colors = [ch["color"] for ch in ds.config["channels"]]
    assert colors[9] == COLOR_PALETTE[9 % len(COLOR_PALETTE)]
    assert colors[9] == colors[0]  # wraps back to the first palette color


def test_assign_colors_filter_label_defaults_to_column(sample_df, sample_config):
    cfg = json.loads(json.dumps(sample_config))
    del cfg["filters"][0]["label"]
    ds = Dataset("test", sample_df.copy(), cfg)
    assert ds.config["filters"][0]["label"] == ds.config["filters"][0]["column"]


# ---------------------------------------------------------------------------
# Dataset._compute_global_bounds
# ---------------------------------------------------------------------------


def test_compute_global_bounds_minmax_vs_log_minmax(sample_df, sample_config):
    ds = Dataset("test", sample_df.copy(), json.loads(json.dumps(sample_config)))
    rating_lo, rating_hi = ds.channel_bounds["rating"]
    assert rating_lo == pytest.approx(sample_df["rating"].min())
    assert rating_hi == pytest.approx(sample_df["rating"].max())

    enroll_lo, enroll_hi = ds.channel_bounds["enrollments"]
    expected_vals = np.log1p(
        np.clip(sample_df["enrollments"].to_numpy(dtype=float), 0, None)
    )
    assert enroll_lo == pytest.approx(expected_vals.min())
    assert enroll_hi == pytest.approx(expected_vals.max())


def test_compute_global_bounds_empty_dataframe_fallback():
    df = pd.DataFrame({"rating": pd.Series([], dtype=float)})
    cfg = {
        "label": "Empty",
        "identifier_column": "rating",
        "channels": [{"column": "rating", "normalize": "minmax"}],
        "filters": [],
    }
    ds = Dataset("empty", df, cfg)
    assert ds.channel_bounds["rating"] == (0.0, 1.0)


# ---------------------------------------------------------------------------
# Dataset.filter_options
# ---------------------------------------------------------------------------


def test_filter_options_categorical_sorted_unique(dataset):
    opts = dataset.filter_options()
    assert opts["category"] == sorted(set(dataset.df["category"].astype(str)))


def test_filter_options_range_min_max(dataset):
    opts = dataset.filter_options()
    assert opts["price"] == [
        float(dataset.df["price"].min()),
        float(dataset.df["price"].max()),
    ]


def test_filter_options_empty_categorical():
    df = pd.DataFrame(
        {"id": pd.Series([], dtype=object), "cat": pd.Series([], dtype=object)}
    )
    cfg = {
        "label": "Empty",
        "identifier_column": "id",
        "channels": [],
        "filters": [{"column": "cat", "type": "categorical"}],
    }
    ds = Dataset("empty", df, cfg)
    assert ds.filter_options()["cat"] == []


def test_filter_options_empty_range():
    df = pd.DataFrame(
        {"id": pd.Series([], dtype=object), "price": pd.Series([], dtype=float)}
    )
    cfg = {
        "label": "Empty",
        "identifier_column": "id",
        "channels": [],
        "filters": [{"column": "price", "type": "range"}],
    }
    ds = Dataset("empty", df, cfg)
    assert ds.filter_options()["price"] == [0.0, 1.0]


# ---------------------------------------------------------------------------
# Dataset.to_public_config
# ---------------------------------------------------------------------------


def test_to_public_config_contains_expected_keys(dataset):
    pub = dataset.to_public_config()
    expected_keys = {
        "name",
        "label",
        "identifier_column",
        "channels",
        "filters",
        "filter_options",
        "row_count",
    }
    assert expected_keys.issubset(pub.keys())


def test_to_public_config_row_count_matches_df(dataset):
    pub = dataset.to_public_config()
    assert pub["row_count"] == len(dataset.df)


def test_to_public_config_label_defaults_to_name_when_missing(sample_df, sample_config):
    cfg = json.loads(json.dumps(sample_config))
    del cfg["label"]
    ds = Dataset("fallback_name", sample_df.copy(), cfg)
    assert ds.to_public_config()["label"] == "fallback_name"


# ---------------------------------------------------------------------------
# _apply_normalize_space
# ---------------------------------------------------------------------------


def test_apply_normalize_space_log_minmax():
    raw = np.array([0.0, 1.0, np.e - 1])
    result = _apply_normalize_space(raw, "log_minmax")
    expected = np.log1p(np.clip(raw, 0, None))
    np.testing.assert_allclose(result, expected)


@pytest.mark.parametrize("normalize", ["minmax", "none", "anything_else"])
def test_apply_normalize_space_passthrough_for_other_values(normalize):
    raw = np.array([1.0, 2.0, 3.0])
    result = _apply_normalize_space(raw, normalize)
    np.testing.assert_array_equal(result, raw)


# ---------------------------------------------------------------------------
# _validate_config
# ---------------------------------------------------------------------------


def test_validate_config_happy_path_raises_nothing(sample_df, sample_config):
    _validate_config("ok", sample_df, sample_config)  # should not raise


def test_validate_config_missing_top_level_keys(sample_df):
    with pytest.raises(DatasetConfigError, match="missing keys"):
        _validate_config("bad", sample_df, {})


def test_validate_config_identifier_column_not_in_df(sample_df, sample_config):
    cfg = json.loads(json.dumps(sample_config))
    cfg["identifier_column"] = "does_not_exist"
    with pytest.raises(DatasetConfigError, match="identifier_column"):
        _validate_config("bad", sample_df, cfg)


def test_validate_config_empty_channels_list(sample_df, sample_config):
    cfg = json.loads(json.dumps(sample_config))
    cfg["channels"] = []
    with pytest.raises(DatasetConfigError, match="at least one channel"):
        _validate_config("bad", sample_df, cfg)


def test_validate_config_channel_column_not_in_df(sample_df, sample_config):
    cfg = json.loads(json.dumps(sample_config))
    cfg["channels"][0]["column"] = "nonexistent_col"
    with pytest.raises(DatasetConfigError, match="not in dataset"):
        _validate_config("bad", sample_df, cfg)


def test_validate_config_channel_column_non_numeric(sample_df, sample_config):
    cfg = json.loads(json.dumps(sample_config))
    cfg["channels"][0]["column"] = "category"  # a string column
    with pytest.raises(DatasetConfigError, match="must be numeric"):
        _validate_config("bad", sample_df, cfg)


def test_validate_config_invalid_normalize(sample_df, sample_config):
    cfg = json.loads(json.dumps(sample_config))
    cfg["channels"][0]["normalize"] = "not_a_real_mode"
    with pytest.raises(DatasetConfigError, match="invalid normalize"):
        _validate_config("bad", sample_df, cfg)


def test_validate_config_invalid_normalize_scope(sample_df, sample_config):
    cfg = json.loads(json.dumps(sample_config))
    cfg["channels"][0]["normalize_scope"] = "not_a_real_scope"
    with pytest.raises(DatasetConfigError, match="invalid normalize_scope"):
        _validate_config("bad", sample_df, cfg)


def test_validate_config_filter_column_not_in_df(sample_df, sample_config):
    cfg = json.loads(json.dumps(sample_config))
    cfg["filters"][0]["column"] = "nonexistent_col"
    with pytest.raises(DatasetConfigError, match="not in dataset"):
        _validate_config("bad", sample_df, cfg)


def test_validate_config_invalid_filter_type(sample_df, sample_config):
    cfg = json.loads(json.dumps(sample_config))
    cfg["filters"][0]["type"] = "not_a_real_type"
    with pytest.raises(DatasetConfigError, match="invalid type"):
        _validate_config("bad", sample_df, cfg)


def test_dataset_config_error_is_a_value_error():
    assert issubclass(DatasetConfigError, ValueError)


# ---------------------------------------------------------------------------
# load_datasets
# ---------------------------------------------------------------------------


def test_load_datasets_nonexistent_dir_returns_empty(tmp_path):
    result = load_datasets(tmp_path / "does_not_exist")
    assert result == {}


def test_load_datasets_pkl_without_config_is_skipped(tmp_path, sample_df, capsys):
    sample_df.to_pickle(tmp_path / "orphan.pkl")
    result = load_datasets(tmp_path)
    assert result == {}
    captured = capsys.readouterr()
    assert "orphan.pkl" in captured.out


def test_load_datasets_invalid_config_is_skipped_others_still_load(
    tmp_path, sample_df, sample_config, capsys
):
    # valid pair
    sample_df.to_pickle(tmp_path / "good.pkl")
    (tmp_path / "good.config.json").write_text(json.dumps(sample_config))

    # invalid pair (bad config -> validation failure)
    bad_cfg = json.loads(json.dumps(sample_config))
    bad_cfg["channels"][0]["column"] = "nonexistent_col"
    sample_df.to_pickle(tmp_path / "bad.pkl")
    (tmp_path / "bad.config.json").write_text(json.dumps(bad_cfg))

    result = load_datasets(tmp_path)
    assert "good" in result
    assert "bad" not in result
    captured = capsys.readouterr()
    assert "bad.pkl" in captured.out


def test_load_datasets_multiple_valid_pairs_all_load(
    tmp_path, sample_df, sample_config
):
    for stem in ("alpha", "beta"):
        sample_df.to_pickle(tmp_path / f"{stem}.pkl")
        (tmp_path / f"{stem}.config.json").write_text(json.dumps(sample_config))

    result = load_datasets(tmp_path)
    assert set(result.keys()) == {"alpha", "beta"}
    for name, ds in result.items():
        assert isinstance(ds, Dataset)
        assert ds.name == name


def test_load_datasets_uses_data_dir_fixture(data_dir):
    result = load_datasets(data_dir)
    assert "sample_courses" in result
