import sys
import pathlib

# Explicitly add the project root to the sys.path so modules like
# 'dataset_loader' and 'weight_transforms' are always discoverable.
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent.resolve()))

import json
import pandas as pd
import pytest


@pytest.fixture
def sample_df():
    return pd.DataFrame(
        {
            "title": ["Alpha", "Beta", "Gamma", "Delta"],
            "category": ["Web", "Web", "Data", "Data"],
            "price": [10.0, 50.0, 0.0, 200.0],
            "rating": [4.9, 3.5, 4.2, 4.99],
            "enrollments": [10, 5000, 25, 100000],
        }
    )


@pytest.fixture
def sample_config():
    return {
        "label": "Test Catalog",
        "identifier_column": "title",
        "channels": [
            {
                "column": "rating",
                "label": "Rating",
                "normalize": "minmax",
                "default_transform": {"type": "amplify", "power": 4.0, "scale": 2.5},
            },
            {
                "column": "enrollments",
                "label": "Enrollments",
                "normalize": "log_minmax",
                "default_transform": {"type": "linear", "w": 1.0},
            },
        ],
        "filters": [
            {"column": "category", "label": "Category", "type": "categorical"},
            {"column": "price", "label": "Price", "type": "range"},
        ],
    }


@pytest.fixture
def dataset(sample_df, sample_config):
    from dataset_loader import Dataset

    return Dataset(
        "sample_courses", sample_df.copy(), json.loads(json.dumps(sample_config))
    )


@pytest.fixture
def data_dir(tmp_path, sample_df, sample_config):
    """A temp data/ dir with one valid <name>.pkl + <name>.config.json pair."""
    sample_df.to_pickle(tmp_path / "sample_courses.pkl")
    (tmp_path / "sample_courses.config.json").write_text(json.dumps(sample_config))
    return tmp_path
