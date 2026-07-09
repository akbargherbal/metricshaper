# Paste dataset loader class here (dataset_loader.py)
"""
Loads datasets for the signal shaper from `data/*.pkl`, each paired with an
explicit `<name>.config.json` sitting next to it.

Design choice: config is REQUIRED, not auto-detected. A pickle with no
matching config is skipped (with a warning) rather than guessed at. This
keeps the mapping from "raw column" to "scoring channel" / "filter"
intentional and documented, instead of magic.

Config schema (see data/sample_courses.config.json for a worked example):

{
  "label": "Human-readable dataset name",
  "identifier_column": "column used to identify/rank/watch individual rows",
  "channels": [
    {
      "column": "numeric column name",
      "label": "Display label",
      "color": "teal",                 // one of COLOR_PALETTE below
      "normalize": "minmax" | "log_minmax" | "none",
      "normalize_scope": "global" | "filtered",   // optional, defaults to "global"
      "default_transform": {"type": "amplify", "power": 4.0, "scale": 2.5},
      "warning": "optional UI hint string"
    },\n    ...\n  ],
  "filters": [
    {"column": "category column", "label": "Category", "type": "categorical"},
    {"column": "numeric column", "label": "Price ($)", "type": "range"},
    ...\n  ]\n}
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

COLOR_PALETTE = [
    "teal",
    "lime",
    "amber",
    "purple",
    "pink",
    "sky",
    "rose",
    "orange",
    "cyan",
]

REQUIRED_TOP_LEVEL_KEYS = {"identifier_column", "channels", "filters"}
VALID_NORMALIZE = {"minmax", "log_minmax", "none"}
VALID_NORMALIZE_SCOPE = {"global", "filtered"}
VALID_FILTER_TYPES = {"categorical", "range"}


class DatasetConfigError(ValueError):
    """Raised when a dataset's config.json is missing, malformed, or doesn't
    match its pickle's columns."""


class Dataset:
    def __init__(self, name: str, df: pd.DataFrame, config: dict):
        self.name = name
        self.df = df
        self.config = config
        self._assign_colors()
        self.channel_bounds = self._compute_global_bounds()

    def _assign_colors(self) -> None:
        for i, ch in enumerate(self.config["channels"]):
            ch.setdefault("color", COLOR_PALETTE[i % len(COLOR_PALETTE)])
            ch.setdefault("normalize", "minmax")
            ch.setdefault("normalize_scope", "global")
            ch.setdefault("label", ch["column"])
        for f in self.config["filters"]:
            f.setdefault("label", f["column"])

    def _compute_global_bounds(self) -> dict[str, tuple[float, float]]:
        bounds = {}
        for ch in self.config["channels"]:
            col = ch["column"]
            raw = self.df[col].to_numpy(dtype=float)
            vals = _apply_normalize_space(raw, ch["normalize"])
            if len(vals) == 0:
                bounds[col] = (0.0, 1.0)
            else:
                bounds[col] = (float(np.min(vals)), float(np.max(vals)))
        return bounds

    def filter_options(self) -> dict:
        opts = {}
        for f in self.config["filters"]:
            col = f["column"]
            if f["type"] == "categorical":
                opts[col] = sorted(
                    str(v) for v in self.df[col].dropna().unique().tolist()
                )
            else:
                raw = self.df[col].to_numpy(dtype=float)
                if len(raw) == 0:
                    opts[col] = [0.0, 1.0]
                else:
                    opts[col] = [float(np.min(raw)), float(np.max(raw))]
        return opts

    def to_public_config(self) -> dict:
        """Config as sent to the frontend: adds filter_options, keeps the rest."""
        return {
            "name": self.name,
            "label": self.config.get("label", self.name),
            "identifier_column": self.config["identifier_column"],
            "channels": self.config["channels"],
            "filters": self.config["filters"],
            "filter_options": self.filter_options(),
            "row_count": int(len(self.df)),
        }


def _apply_normalize_space(raw: np.ndarray, normalize: str) -> np.ndarray:
    """Maps raw values into the space they'll be min-max scaled in."""
    if normalize == "log_minmax":
        return np.log1p(np.clip(raw, a_min=0, a_max=None))
    return raw


def _validate_config(name: str, df: pd.DataFrame, config: dict) -> None:
    missing = REQUIRED_TOP_LEVEL_KEYS - config.keys()
    if missing:
        raise DatasetConfigError(f"[{name}] config missing keys: {sorted(missing)}")

    id_col = config["identifier_column"]
    if id_col not in df.columns:
        raise DatasetConfigError(
            f"[{name}] identifier_column {id_col!r} not found in dataset columns"
        )

    if not config["channels"]:
        raise DatasetConfigError(f"[{name}] must define at least one channel")

    for ch in config["channels"]:
        col = ch.get("column")
        if col not in df.columns:
            raise DatasetConfigError(f"[{name}] channel column {col!r} not in dataset")
        if not pd.api.types.is_numeric_dtype(df[col]):
            raise DatasetConfigError(f"[{name}] channel column {col!r} must be numeric")
        norm = ch.get("normalize", "minmax")
        if norm not in VALID_NORMALIZE:
            raise DatasetConfigError(
                f"[{name}] channel {col!r} has invalid normalize={norm!r}"
            )
        scope = ch.get("normalize_scope", "global")
        if scope not in VALID_NORMALIZE_SCOPE:
            raise DatasetConfigError(
                f"[{name}] channel {col!r} has invalid normalize_scope={scope!r}"
            )

    for f in config["filters"]:
        col = f.get("column")
        if col not in df.columns:
            raise DatasetConfigError(f"[{name}] filter column {col!r} not in dataset")
        ftype = f.get("type")
        if ftype not in VALID_FILTER_TYPES:
            raise DatasetConfigError(
                f"[{name}] filter {col!r} has invalid type={ftype!r}"
            )


def load_datasets(data_dir: Path | str) -> dict[str, Dataset]:
    """Scans `data_dir` for `<name>.pkl` files with a matching
    `<name>.config.json` beside them. Pickles without a config are skipped
    with a printed warning rather than raising, so one bad dataset doesn't
    take down the whole app."""
    data_dir = Path(data_dir)
    datasets: dict[str, Dataset] = {}

    if not data_dir.exists():
        return datasets

    for pkl_path in sorted(data_dir.glob("*.pkl")):
        name = pkl_path.stem
        config_path = pkl_path.with_suffix("").with_suffix(".config.json")
        if not config_path.exists():
            print(
                f"[dataset_loader] skipping {pkl_path.name}: no {config_path.name} found"
            )
            continue

        try:
            df = pd.read_pickle(pkl_path)
            config = json.loads(config_path.read_text())
            _validate_config(name, df, config)
        except Exception as exc:
            print(f"[dataset_loader] skipping {pkl_path.name}: {exc}")
            continue

        datasets[name] = Dataset(name, df, config)
        print(
            f"[dataset_loader] loaded {name!r}: {len(df)} rows, "
            f"{len(config['channels'])} channels, {len(config['filters'])} filters"
        )

    return datasets
