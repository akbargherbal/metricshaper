"""
Flask app for the configurable weight-transform pipeline — generalized to
work with any dataset that ships a `<name>.pkl` + `<name>.config.json` pair
in `data/`.

Endpoints
---------
GET  /                    - interactive console UI
GET  /api/datasets        - list available datasets
GET  /api/transforms      - transform specs + active dataset's channel/filter config
GET  /api/popular_values  - top values for a categorical filter column
POST /api/compute         - filter, normalize, score, and rank a dataset's rows
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
from flask import Flask, jsonify, render_template, request

from dataset_loader import Dataset, load_datasets
from weight_transforms import TRANSFORM_SPECS, compile_pipeline_formulas, from_config

DATA_DIR = Path(__file__).parent / "data"
DATASETS = load_datasets(DATA_DIR)

if not DATASETS:
    raise RuntimeError(
        f"No usable datasets found in {DATA_DIR}. Add a <name>.pkl file with a "
        f"matching <name>.config.json beside it (see README for the schema), "
        f"or run `python scripts/generate_sample_data.py` for a demo dataset."
    )

DEFAULT_DATASET = next(iter(DATASETS))

app = Flask(__name__)


def get_dataset(name: str | None) -> Dataset:
    if name and name in DATASETS:
        return DATASETS[name]
    return DATASETS[DEFAULT_DATASET]


def sanitize_identifier(s: str) -> str:
    """Turns an arbitrary column name into a safe LaTeX/Python variable name."""
    s = re.sub(r"[^0-9a-zA-Z_]", "_", s)
    if re.match(r"^[0-9]", s):
        s = "_" + s
    return s or "x"


@app.route("/")
def index():
    ds_name = request.args.get("dataset", DEFAULT_DATASET)
    ds = get_dataset(ds_name)
    return render_template(
        "index.html",
        dataset_names=list(DATASETS.keys()),
        active_dataset=ds.name,
        specs=TRANSFORM_SPECS,
    )


@app.route("/api/datasets")
def api_datasets():
    return jsonify(
        {
            "datasets": [
                {"name": d.name, "label": d.config.get("label", d.name)}
                for d in DATASETS.values()
            ],
            "active": DEFAULT_DATASET,
        }
    )


@app.route("/api/transforms")
def api_transforms():
    ds = get_dataset(request.args.get("dataset"))
    return jsonify({"specs": TRANSFORM_SPECS, "dataset": ds.to_public_config()})


@app.route("/api/popular_values")
def api_popular_values():
    """Top values for a categorical filter column, optionally scoped to
    another filter's active value (e.g. top categories within a level)."""
    ds = get_dataset(request.args.get("dataset"))
    column = request.args.get("column")
    scope_column = request.args.get("scope_column")
    scope_value = request.args.get("scope_value")

    filter_cfg = next((f for f in ds.config["filters"] if f["column"] == column), None)
    if not filter_cfg or filter_cfg["type"] != "categorical":
        return jsonify({"error": "unknown or non-categorical filter column"}), 400

    df = ds.df
    if scope_column and scope_value:
        if scope_column not in df.columns:
            return jsonify({"error": f"unknown scope_column {scope_column!r}"}), 400
        df = df[df[scope_column].astype(str) == scope_value]

    top = df[column].dropna().astype(str).value_counts().head(8)
    result = [{"value": k, "count": int(v)} for k, v in top.items()]

    if request.headers.get("HX-Request"):
        return render_template(
            "_popular_tags.html", column=column, popular_values=result
        )

    return jsonify({"popular_values": result})


@app.route("/api/compute", methods=["POST"])
def api_compute():
    """Filter, normalize, score, and rank rows for the active dataset.

    Request body:
        {
          "dataset": "sample_courses",
          "filters": {"category": "Web Development", "price": [0, 100]},
          "target_identifiers": ["JavaScript Intermediate and Advanced Concepts"],
          "pipeline": [ {"type": "amplify", "power": 4, "scale": 2.5}, ... ],
          "top_x": 20
        }

    The pipeline must contain exactly one transform per configured channel,
    in the same order as the dataset's `channels` config.
    """
    data = request.get_json(silent=True) or {}
    ds = get_dataset(data.get("dataset"))
    channels_cfg = ds.config["channels"]
    filters_cfg = {f["column"]: f for f in ds.config["filters"]}
    id_col = ds.config["identifier_column"]

    pipeline_specs = data.get("pipeline", [])
    if not isinstance(pipeline_specs, list) or len(pipeline_specs) != len(channels_cfg):
        return (
            jsonify(
                {
                    "error": (
                        f"Pipeline must contain exactly {len(channels_cfg)} transforms "
                        f"(one per configured channel, in order)."
                    )
                }
            ),
            400,
        )

    try:
        knobs = [from_config(spec) for spec in pipeline_specs]
    except Exception as exc:
        return jsonify({"error": f"Invalid transform configuration: {exc}"}), 400

    try:
        top_x = int(data.get("top_x", 20))
    except (TypeError, ValueError):
        return jsonify({"error": "top_x must be an integer"}), 400

    raw_targets = data.get("target_identifiers", [])
    if not isinstance(raw_targets, list):
        return jsonify({"error": "target_identifiers must be a list of strings"}), 400
    targets = [str(t).strip() for t in raw_targets if str(t).strip()]

    # 1. Filter
    filtered_df = ds.df
    active_filters = data.get("filters") or {}
    if not isinstance(active_filters, dict):
        return jsonify({"error": "filters must be an object mapping column -> value"}), 400

    for col, value in active_filters.items():
        fc = filters_cfg.get(col)
        if not fc or value in (None, "", []):
            continue
        if fc["type"] == "categorical":
            filtered_df = filtered_df[filtered_df[col].astype(str) == str(value)]
        elif (
            fc["type"] == "range"
            and isinstance(value, (list, tuple))
            and len(value) == 2
        ):
            try:
                lo, hi = float(value[0]), float(value[1])
            except (TypeError, ValueError):
                return (
                    jsonify({"error": f"filter {col!r} range bounds must be numeric"}),
                    400,
                )
            filtered_df = filtered_df[
                (filtered_df[col] >= lo) & (filtered_df[col] <= hi)
            ]

    if filtered_df.empty:
        return jsonify(
            {
                "leaderboard": [],
                "watchlist": [],
                "formula": {"latex": "y = 0", "python": "y = 0"},
            }
        )

    # 2. Normalize + transform each channel, tracking per-channel contributions
    work_df = filtered_df.copy()
    contrib_cols = []
    for ch, knob in zip(channels_cfg, knobs):
        col = ch["column"]
        norm = ch.get("normalize", "minmax")
        scope = ch.get("normalize_scope", "global")

        raw = work_df[col].to_numpy(dtype=float)
        if norm == "log_minmax":
            vals = np.log1p(np.clip(raw, a_min=0, a_max=None))
        else:
            vals = raw

        if scope == "global":
            lo, hi = ds.channel_bounds[col]
        elif len(vals):
            lo, hi = float(np.min(vals)), float(np.max(vals))
        else:
            lo, hi = 0.0, 1.0

        span = (hi - lo) or 1.0
        norm_vals = (vals - lo) / span if norm in ("minmax", "log_minmax") else vals

        contrib_col = f"__contrib__{col}"
        work_df[contrib_col] = knob(norm_vals)
        contrib_cols.append((col, ch, contrib_col))

    work_df["total_score"] = sum(work_df[c] for _, _, c in contrib_cols)
    work_df = work_df.sort_values("total_score", ascending=False).reset_index(drop=True)
    work_df["rank"] = work_df.index + 1

    def build_breakdown(row) -> list[dict]:
        return [
            {
                "column": col,
                "label": ch.get("label", col),
                "color": ch.get("color", "teal"),
                "contribution": float(row[contrib_col]),
            }
            for col, ch, contrib_col in contrib_cols
        ]

    # 3. Leaderboard (top_x rows)
    top_df = work_df.head(max(top_x, 0))

    leaderboard = []
    for _, row in top_df.iterrows():
        leaderboard.append(
            {
                "id": row[id_col],
                "rank": int(row["rank"]),
                "total_score": float(row["total_score"]),
                "breakdown": build_breakdown(row),
            }
        )

    # 4. Watchlist — resolved against the currently filtered+scored set
    watchlist = []
    for target in targets:
        match = work_df[work_df[id_col].astype(str) == target]
        if not match.empty:
            row = match.iloc[0]
            watchlist.append(
                {
                    "id": target,
                    "found": True,
                    "rank": int(row["rank"]),
                    "total_score": float(row["total_score"]),
                    "breakdown": build_breakdown(row),
                }
            )
        else:
            watchlist.append(
                {
                    "id": target,
                    "found": False,
                    "rank": None,
                    "total_score": 0.0,
                    "breakdown": [],
                }
            )

    # 5. Formula, using sanitized column names as variables
    python_vars = [
        sanitize_identifier(ch["column"]) for ch, _ in zip(channels_cfg, knobs)
    ]
    latex_vars = [f"\\\\text{{{v}}}" for v in python_vars]
    formula = compile_pipeline_formulas(
        knobs, latex_vars=latex_vars, python_vars=python_vars
    )

    return jsonify(
        {"leaderboard": leaderboard, "watchlist": watchlist, "formula": formula}
    )


if __name__ == "__main__":  # pragma: no cover
    app.run(debug=True)
