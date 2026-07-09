# MetricSynth (MetricShaper)

A configurable, non-linear weight-transform scoring and ranking engine designed to explore, shape, and prioritize features across arbitrary tabular datasets.

Instead of evaluating metrics using static linear weights:
$$y = \sum (w_i \cdot x_i)$$

MetricSynth implements a dynamic, config-driven transform pipeline:
$$y = \sum (f_i(x_i))$$

Where each $f_i(x_i)$ is a modular, composable mathematical transform (or "knob") such as a clamp, logarithmic compression, sigmoid saturation, hard threshold, or amplification power curve. This allows you to define custom scoring philosophies (e.g., "prioritizing quality over raw scale") and see how rankings shift in real time.

---

## System Architecture

```text
├── data/
│   ├── sample_courses.pkl         # Tabular dataset (pandas DataFrame pickle)
│   └── sample_courses.config.json # Dataset-specific configuration mapping
├── templates/
│   ├── index.html                 # Front-end dashboard (Jinja + Tailwind shell)
│   └── _popular_tags.html         # HTMX fragment for filter quick-select badges
├── static/
│   └── app.js                     # Front-end state manager & KaTeX rendering math
├── scripts/
│   └── generate_sample_data.py    # Generates the default course dataset
├── app.py                         # Flask server serving the REST API
├── dataset_loader.py              # Ingests, validates, and normalizes datasets
└── weight_transforms.py           # Core algebraic transform library & formula compiler
```

---

## Concept & Mathematical Knobs

The application maps numeric dataset columns into **Scoring Channels**. Each channel is run through a chosen mathematical transform, and the result is summed to produce a unified score:

- **Linear:** $f(x) = w \cdot x$. Simple scaling.
- **Clamp:** $f(x) = \text{clip}(x, \min, \max)$. Prevents extreme outliers from skewing the ranking pool.
- **Amplify:** $f(x) = \text{sign}(x) \cdot |x|^p$. Stretches out closely-packed distributions (e.g., rating fields clustered between 4.2 and 4.9).
- **Log Compress:** $f(x) = \ln(1 + \max(0, x))$. Compresses wide, highly right-skewed ranges (e.g., student counts or view metrics) to prevent raw volume from drowning out quality signals.
- **Sigmoid:** Smooth saturation curve with customizable midpoint and sharpness. Bounds skewed ranges into a soft, normalized $0 \text{ to } 1$ signal.
- **Threshold:** A hard step function returning a baseline value below a boundary and an elevated value above it.
- **Passthrough:** Passes the normalized value through without modification.

---

## Dataset Configuration Schema (`<name>.config.json`)

To load a new dataset, place your pandas DataFrame as a `.pkl` file inside the `data/` folder, and write an accompanying `.config.json` file with the exact same base name.

The engine uses this configuration to dynamically build the user interface and validate incoming pipeline payloads.

### Schema Blueprint

```json
{
  "label": "Display Label for the Dataset Selector",
  "identifier_column": "name_of_the_unique_row_identifier_column",
  "channels": [
    {
      "column": "dataframe_numeric_column_name",
      "label": "User-facing label for this channel card",
      "normalize": "minmax" | "log_minmax" | "none",
      "normalize_scope": "global" | "filtered",
      "default_transform": {
        "type": "linear" | "clamp" | "amplify" | "log_compress" | "sigmoid" | "threshold" | "passthrough",
        "weight_or_param_key": 1.0
      },
      "warning": "Optional warning/tip displayed below the channel card"
    }
  ],
  "filters": [
    {
      "column": "dataframe_categorical_or_numeric_column",
      "label": "Label in filter panel",
      "type": "categorical" | "range"
    }
  ]
}
```

### Options Breakdown

- **`normalize`**:
  - `minmax`: Scales values linearly between $[0, 1]$.
  - `log_minmax`: Log-transforms values before performing a min-max scale. Highly recommended for power-law distributions.
  - `none`: Bypasses scaling.
- **`normalize_scope`** (Defaults to `"global"`):
  - `global`: Normalization boundaries ($\min/\max$) are computed once at startup across the _entire_ dataset. Scores remain stable and comparable when changing filters.
  - `filtered`: Boundaries are recomputed dynamically using _only_ the currently filtered subset. Maximizes contrast within small sub-populations, but shifts absolute scores when filters change.
- **`filters` Types**:
  - `categorical`: Renders an exact-match dropdown menu accompanied by HTMX-driven high-frequency value selection badges.
  - `range`: Renders a pair of minimum and maximum numerical boundary inputs.

---

## Setup & Execution

### 1. Prerequisites

Ensure you have Python 3.8+ installed.

### 2. Install Dependencies

Install Flask, pandas, numpy, and testing libraries:

```bash
pip install flask pandas numpy pytest pytest-cov
```

### 3. Generate Sample Dataset

Run the bootstrapping script to generate a synthetic dataset representing online courses:

```bash
python scripts/generate_sample_data.py
```

This creates:

- `data/sample_courses.pkl` (41 synthetic course rows)
- `data/sample_courses.config.json` (A 7-channel, 4-filter configuration)

### 4. Run the Application

Launch the Flask development server:

```bash
python app.py
```

Open [http://127.0.0.1:5000](http://127.0.0.1:5000) in your browser.

---

## API Reference

### 1. `GET /api/datasets`

Lists all validated dataset configurations discovered inside the `data/` folder.

- **Response:**
  ```json
  {
    "datasets": [
      { "name": "sample_courses", "label": "Online Course Catalog" }
    ],
    "active": "sample_courses"
  }
  ```

### 2. `GET /api/transforms?dataset=<name>`

Returns the mathematical parameters for all available transform types alongside the structural definitions and filter ranges of the selected dataset.

- **Response snippet:**
  ```json
  {
    "specs": {
      "amplify": {
        "label": "Amplify",
        "description": "scale × sign(x) × |x|^power",
        "params": {
          "power": { "label": "power", "default": 2.0, "step": 0.1 },
          "scale": { "label": "scale", "default": 1.0, "step": 0.1 }
        }
      }
    },
    "dataset": {
      "name": "sample_courses",
      "label": "Online Course Catalog",
      "identifier_column": "course_title",
      "channels": [ ... ],
      "filters": [ ... ],
      "filter_options": {
        "category": ["Business", "Data Science", "Design", "Mobile Development", "Web Development"],
        "price": [0.0, 199.99]
      },
      "row_count": 41
    }
  }
  ```

### 3. `GET /api/popular_values?dataset=<name>&column=<col>`

Returns high-frequency occurrences in categorical columns to populate quick-select UI elements.

- **Response:**
  ```json
  {
    "popular_values": [
      { "value": "Web Development", "count": 12 },
      { "value": "Data Science", "count": 10 }
    ]
  }
  ```

### 4. `POST /api/compute`

Filters, normalizes, executes transform curves, aggregates scores, and ranks outputs.

- **Headers:** `Content-Type: application/json`
- **Payload:**
  ```json
  {
    "dataset": "sample_courses",
    "filters": {
      "category": "Web Development",
      "price": [0, 100]
    },
    "target_identifiers": ["JavaScript Intermediate and Advanced Concepts"],
    "pipeline": [
      { "type": "amplify", "power": 4.0, "scale": 2.5 },
      { "type": "amplify", "power": 2.0, "scale": 2.2 },
      { "type": "linear", "w": 0.6 },
      { "type": "linear", "w": 0.4 },
      { "type": "linear", "w": 0.4 },
      { "type": "clamp", "max_v": 0.7, "scale": 0.8 },
      { "type": "linear", "w": 1.0 }
    ],
    "top_x": 20
  }
  ```
- **Response snippet:**
  ```json
  {
    "leaderboard": [
      {
        "id": "JavaScript Intermediate and Advanced Concepts",
        "rank": 1,
        "total_score": 6.65,
        "breakdown": [
          { "column": "avg_rating", "label": "Average rating", "color": "teal", "contribution": 2.5 },
          { "column": "completion_pct", "label": "Completion rate", "color": "lime", "contribution": 2.2 }
        ]
      }
    ],
    "watchlist": [
      {
        "id": "JavaScript Intermediate and Advanced Concepts",
        "found": true,
        "rank": 1,
        "total_score": 6.65,
        "breakdown": [ ... ]
      }
    ],
    "formula": {
      "latex": "y = \\left( 2.50 \\cdot \\operatorname{sgn}(\\text{avg_rating}) \\cdot |\\text{avg_rating}|^{4.00} \\right) + ...",
      "python": "y = (2.50 * np.sign(avg_rating) * (np.abs(avg_rating) ** 4.00)) + ..."
    }
  }
  ```
