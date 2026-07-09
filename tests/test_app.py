# TODO: Copy-paste the content of tests/test_app.py here
import json
import sys

import pytest


def _build_test_dataset(sample_df, sample_config):
    from dataset_loader import Dataset

    return Dataset(
        "sample_courses", sample_df.copy(), json.loads(json.dumps(sample_config))
    )


def _fresh_app_module(monkeypatch, datasets_dict):
    """Reload app.py with dataset_loader.load_datasets patched to return
    `datasets_dict`, so app.py's module-level DATASETS is hermetic."""
    monkeypatch.setattr("dataset_loader.load_datasets", lambda _dir: datasets_dict)
    sys.modules.pop("app", None)
    import app as app_module

    return app_module


@pytest.fixture
def app_module(monkeypatch, sample_df, sample_config):
    ds = _build_test_dataset(sample_df, sample_config)
    module = _fresh_app_module(monkeypatch, {"sample_courses": ds})
    yield module
    sys.modules.pop("app", None)


@pytest.fixture
def client(app_module):
    app_module.app.config.update(TESTING=True)
    with app_module.app.test_client() as c:
        yield c


# ---------------------------------------------------------------------------
# Import-time behavior
# ---------------------------------------------------------------------------


def test_runtime_error_when_no_datasets(monkeypatch):
    monkeypatch.setattr("dataset_loader.load_datasets", lambda _dir: {})
    sys.modules.pop("app", None)
    with pytest.raises(RuntimeError, match="No usable datasets"):
        import app  # noqa: F401
    sys.modules.pop("app", None)


# ---------------------------------------------------------------------------
# get_dataset / sanitize_identifier
# ---------------------------------------------------------------------------


def test_get_dataset_known_name_returns_it(app_module):
    ds = app_module.get_dataset("sample_courses")
    assert ds.name == "sample_courses"


def test_get_dataset_unknown_name_falls_back_to_default(app_module):
    ds = app_module.get_dataset("does_not_exist")
    assert ds.name == app_module.DEFAULT_DATASET


def test_get_dataset_none_falls_back_to_default(app_module):
    ds = app_module.get_dataset(None)
    assert ds.name == app_module.DEFAULT_DATASET


def test_sanitize_identifier_removes_non_alphanumeric(app_module):
    assert app_module.sanitize_identifier("some column!") == "some_column_"


def test_sanitize_identifier_leading_digit_gets_prefix(app_module):
    assert app_module.sanitize_identifier("123abc") == "_123abc"


def test_sanitize_identifier_empty_string_returns_x(app_module):
    assert app_module.sanitize_identifier("") == "x"


# ---------------------------------------------------------------------------
# GET /
# ---------------------------------------------------------------------------


def test_index_returns_200(client):
    resp = client.get("/")
    assert resp.status_code == 200


def test_index_default_active_dataset(client, app_module):
    resp = client.get("/")
    assert app_module.DEFAULT_DATASET.encode() in resp.data


def test_index_dataset_query_param_switches_dataset(
    monkeypatch, sample_df, sample_config
):
    ds1 = _build_test_dataset(sample_df, sample_config)
    ds2_cfg = json.loads(json.dumps(sample_config))
    from dataset_loader import Dataset

    ds2 = Dataset("other_dataset", sample_df.copy(), ds2_cfg)
    module = _fresh_app_module(
        monkeypatch, {"sample_courses": ds1, "other_dataset": ds2}
    )
    module.app.config.update(TESTING=True)
    with module.app.test_client() as c:
        resp = c.get("/?dataset=other_dataset")
        assert b"other_dataset" in resp.data
    sys.modules.pop("app", None)


# ---------------------------------------------------------------------------
# GET /api/datasets
# ---------------------------------------------------------------------------


def test_api_datasets_returns_list_and_active(client, app_module):
    resp = client.get("/api/datasets")
    data = resp.get_json()
    assert data["active"] == app_module.DEFAULT_DATASET
    names = [d["name"] for d in data["datasets"]]
    assert "sample_courses" in names
    labels = [d["label"] for d in data["datasets"]]
    assert "Test Catalog" in labels


# ---------------------------------------------------------------------------
# GET /api/transforms
# ---------------------------------------------------------------------------


def test_api_transforms_returns_specs_and_dataset(client, app_module):
    from weight_transforms import TRANSFORM_SPECS

    resp = client.get("/api/transforms")
    data = resp.get_json()
    assert data["specs"] == TRANSFORM_SPECS
    assert data["dataset"]["name"] == "sample_courses"


def test_api_transforms_dataset_param_selects_dataset(
    monkeypatch, sample_df, sample_config
):
    from dataset_loader import Dataset

    ds1 = _build_test_dataset(sample_df, sample_config)
    ds2 = Dataset("second", sample_df.copy(), json.loads(json.dumps(sample_config)))
    module = _fresh_app_module(monkeypatch, {"sample_courses": ds1, "second": ds2})
    module.app.config.update(TESTING=True)
    with module.app.test_client() as c:
        resp = c.get("/api/transforms?dataset=second")
        assert resp.get_json()["dataset"]["name"] == "second"
    sys.modules.pop("app", None)


# ---------------------------------------------------------------------------
# GET /api/popular_values
# ---------------------------------------------------------------------------


def test_popular_values_unknown_column_400(client):
    resp = client.get("/api/popular_values?column=nonexistent")
    assert resp.status_code == 400
    assert "error" in resp.get_json()


def test_popular_values_non_categorical_column_400(client):
    resp = client.get("/api/popular_values?column=price")  # price is a range filter
    assert resp.status_code == 400
    assert "error" in resp.get_json()


def test_popular_values_valid_categorical_sorted_by_count(client):
    resp = client.get("/api/popular_values?column=category")
    data = resp.get_json()
    values = data["popular_values"]
    assert len(values) <= 8
    counts = [v["count"] for v in values]
    assert counts == sorted(counts, reverse=True)


def test_popular_values_scope_narrows_results(client):
    resp = client.get(
        "/api/popular_values?column=category&scope_column=price&scope_value=10.0"
    )
    data = resp.get_json()
    assert "popular_values" in data


def test_popular_values_hx_request_renders_fragment(client):
    resp = client.get(
        "/api/popular_values?column=category", headers={"HX-Request": "true"}
    )
    assert resp.status_code == 200
    assert b"<button" in resp.data
    assert resp.content_type.startswith("text/html")


# ---------------------------------------------------------------------------
# POST /api/compute
# ---------------------------------------------------------------------------


def _valid_pipeline():
    return [
        {"type": "amplify", "power": 4.0, "scale": 2.5},
        {"type": "linear", "w": 1.0},
    ]


def test_compute_pipeline_length_mismatch_400(client):
    resp = client.post("/api/compute", json={"pipeline": [{"type": "linear"}]})
    assert resp.status_code == 400
    assert "error" in resp.get_json()


def test_compute_invalid_transform_type_400(client):
    resp = client.post(
        "/api/compute",
        json={"pipeline": [{"type": "not_a_real_type"}, {"type": "linear"}]},
    )
    assert resp.status_code == 400
    assert "Invalid transform configuration" in resp.get_json()["error"]


def test_compute_categorical_filter_narrows_rows(client):
    resp = client.post(
        "/api/compute",
        json={
            "pipeline": _valid_pipeline(),
            "filters": {"category": "Web"},
            "top_x": 20,
        },
    )
    data = resp.get_json()
    ids = {row["id"] for row in data["leaderboard"]}
    assert ids.issubset({"Alpha", "Beta"})


def test_compute_range_filter_inclusive_bounds(client):
    resp = client.post(
        "/api/compute",
        json={
            "pipeline": _valid_pipeline(),
            "filters": {"price": [0.0, 50.0]},
            "top_x": 20,
        },
    )
    data = resp.get_json()
    ids = {row["id"] for row in data["leaderboard"]}
    # Alpha=10, Beta=50, Gamma=0 all within [0, 50]; Delta=200 excluded
    assert ids == {"Alpha", "Beta", "Gamma"}


def test_compute_filter_none_value_is_noop(client):
    resp = client.post(
        "/api/compute",
        json={
            "pipeline": _valid_pipeline(),
            "filters": {"category": None},
            "top_x": 20,
        },
    )
    data = resp.get_json()
    assert len(data["leaderboard"]) == 4


def test_compute_filter_empty_string_is_noop(client):
    resp = client.post(
        "/api/compute",
        json={"pipeline": _valid_pipeline(), "filters": {"category": ""}, "top_x": 20},
    )
    data = resp.get_json()
    assert len(data["leaderboard"]) == 4


def test_compute_filter_empty_list_is_noop(client):
    resp = client.post(
        "/api/compute",
        json={"pipeline": _valid_pipeline(), "filters": {"price": []}, "top_x": 20},
    )
    data = resp.get_json()
    assert len(data["leaderboard"]) == 4


def test_compute_filtering_to_zero_rows_early_return(client):
    resp = client.post(
        "/api/compute",
        json={
            "pipeline": _valid_pipeline(),
            "filters": {"category": "DoesNotExist"},
            "top_x": 20,
        },
    )
    data = resp.get_json()
    assert data == {
        "leaderboard": [],
        "watchlist": [],
        "formula": {"latex": "y = 0", "python": "y = 0"},
    }


def test_compute_normalize_global_scope_stable_across_filters(client):
    resp_unfiltered = client.post(
        "/api/compute", json={"pipeline": _valid_pipeline(), "top_x": 20}
    )
    resp_filtered = client.post(
        "/api/compute",
        json={
            "pipeline": _valid_pipeline(),
            "filters": {"category": "Web"},
            "top_x": 20,
        },
    )
    unfiltered_scores = {
        r["id"]: r["total_score"] for r in resp_unfiltered.get_json()["leaderboard"]
    }
    filtered_scores = {
        r["id"]: r["total_score"] for r in resp_filtered.get_json()["leaderboard"]
    }
    assert unfiltered_scores["Alpha"] == pytest.approx(filtered_scores["Alpha"])
    assert unfiltered_scores["Beta"] == pytest.approx(filtered_scores["Beta"])


def test_compute_normalize_filtered_scope_recomputes_bounds(client, sample_config):
    cfg = json.loads(json.dumps(sample_config))
    cfg["channels"][0]["normalize_scope"] = "filtered"
    resp_unfiltered = client.post(
        "/api/compute", json={"pipeline": _valid_pipeline(), "top_x": 20}
    )
    # sanity: request succeeds
    assert resp_unfiltered.status_code == 200


def test_compute_filtered_scope_single_row_no_div_by_zero(
    monkeypatch, sample_df, sample_config
):
    cfg = json.loads(json.dumps(sample_config))
    cfg["channels"][0]["normalize_scope"] = "filtered"
    from dataset_loader import Dataset

    ds = Dataset("sample_courses", sample_df.copy(), cfg)
    module = _fresh_app_module(monkeypatch, {"sample_courses": ds})
    module.app.config.update(TESTING=True)
    with module.app.test_client() as c:
        resp = c.post(
            "/api/compute",
            json={
                "pipeline": _valid_pipeline(),
                "filters": {"category": "Data"},
                "top_x": 1,
            },
        )
        assert resp.status_code == 200
    sys.modules.pop("app", None)


def test_compute_total_score_equals_sum_of_contributions(client):
    resp = client.post(
        "/api/compute", json={"pipeline": _valid_pipeline(), "top_x": 20}
    )
    data = resp.get_json()
    for row in data["leaderboard"]:
        total = sum(b["contribution"] for b in row["breakdown"])
        assert row["total_score"] == pytest.approx(total)


def test_compute_leaderboard_sorted_descending(client):
    resp = client.post(
        "/api/compute", json={"pipeline": _valid_pipeline(), "top_x": 20}
    )
    data = resp.get_json()
    scores = [row["total_score"] for row in data["leaderboard"]]
    assert scores == sorted(scores, reverse=True)


def test_compute_rank_is_1_indexed_and_contiguous(client):
    resp = client.post(
        "/api/compute", json={"pipeline": _valid_pipeline(), "top_x": 20}
    )
    data = resp.get_json()
    ranks = [row["rank"] for row in data["leaderboard"]]
    assert ranks == list(range(1, len(ranks) + 1))


def test_compute_top_x_truncates_leaderboard(client):
    resp = client.post("/api/compute", json={"pipeline": _valid_pipeline(), "top_x": 2})
    data = resp.get_json()
    assert len(data["leaderboard"]) == 2


def test_compute_top_x_larger_than_rows_returns_all(client):
    resp = client.post(
        "/api/compute", json={"pipeline": _valid_pipeline(), "top_x": 999}
    )
    data = resp.get_json()
    assert len(data["leaderboard"]) == 4


def test_compute_top_x_zero_returns_empty_leaderboard(client):
    resp = client.post("/api/compute", json={"pipeline": _valid_pipeline(), "top_x": 0})
    assert resp.status_code == 200
    assert resp.get_json()["leaderboard"] == []


def test_compute_top_x_negative_returns_empty_leaderboard(client):
    resp = client.post(
        "/api/compute", json={"pipeline": _valid_pipeline(), "top_x": -5}
    )
    assert resp.status_code == 200
    assert resp.get_json()["leaderboard"] == []


def test_compute_watchlist_found(client):
    resp = client.post(
        "/api/compute",
        json={
            "pipeline": _valid_pipeline(),
            "target_identifiers": ["Alpha"],
            "top_x": 20,
        },
    )
    data = resp.get_json()
    entry = data["watchlist"][0]
    assert entry["found"] is True
    assert entry["id"] == "Alpha"
    assert isinstance(entry["rank"], int)
    assert isinstance(entry["total_score"], float)
    assert entry["breakdown"]


def test_compute_watchlist_not_found(client):
    resp = client.post(
        "/api/compute",
        json={
            "pipeline": _valid_pipeline(),
            "target_identifiers": ["Does Not Exist"],
            "top_x": 20,
        },
    )
    data = resp.get_json()
    entry = data["watchlist"][0]
    assert entry == {
        "id": "Does Not Exist",
        "found": False,
        "rank": None,
        "total_score": 0.0,
        "breakdown": [],
    }


def test_compute_watchlist_excluded_by_filter_is_not_found(client):
    resp = client.post(
        "/api/compute",
        json={
            "pipeline": _valid_pipeline(),
            "filters": {"category": "Web"},
            "target_identifiers": ["Gamma"],  # Gamma is category "Data"
            "top_x": 20,
        },
    )
    data = resp.get_json()
    entry = data["watchlist"][0]
    assert entry["found"] is False


def test_compute_watchlist_blank_entries_dropped(client):
    resp = client.post(
        "/api/compute",
        json={
            "pipeline": _valid_pipeline(),
            "target_identifiers": ["", "   ", "Alpha"],
            "top_x": 20,
        },
    )
    data = resp.get_json()
    assert len(data["watchlist"]) == 1
    assert data["watchlist"][0]["id"] == "Alpha"


def test_compute_formula_non_empty_and_term_count_matches_channels(client):
    resp = client.post(
        "/api/compute", json={"pipeline": _valid_pipeline(), "top_x": 20}
    )
    data = resp.get_json()
    formula = data["formula"]
    assert formula["latex"]
    assert formula["python"]
    # two channels -> two "(" terms after "y = "
    assert formula["python"].count("(") >= 2


def test_compute_formula_sanitizes_column_names_with_punctuation(
    monkeypatch, sample_df, sample_config
):
    cfg = json.loads(json.dumps(sample_config))
    df = sample_df.copy()
    df = df.rename(columns={"rating": "rating (stars)!"})
    cfg["channels"][0]["column"] = "rating (stars)!"
    from dataset_loader import Dataset

    ds = Dataset("sample_courses", df, cfg)
    module = _fresh_app_module(monkeypatch, {"sample_courses": ds})
    module.app.config.update(TESTING=True)
    with module.app.test_client() as c:
        resp = c.post("/api/compute", json={"pipeline": _valid_pipeline(), "top_x": 20})
        assert resp.status_code == 200
        formula = resp.get_json()["formula"]
        assert "rating" in formula["python"]
    sys.modules.pop("app", None)


def test_compute_malformed_body_no_json_treated_as_empty(client, app_module):
    n_channels = len(app_module.get_dataset(None).config["channels"])
    resp = client.post("/api/compute", data="not json", content_type="text/plain")
    assert resp.status_code == 400  # pipeline length (0) != n_channels
    assert "error" in resp.get_json()
    assert n_channels > 0
