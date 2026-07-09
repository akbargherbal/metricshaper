# Testing Strategy — MetricSynth (MetricShaper)

This document defines how an AI agent (or engineer) should write and organize tests for this
repository. It covers the Python backend (pytest) and the JavaScript frontend (vitest).

## Goals

- **Backend line/branch coverage ≥ 95%**, enforced in CI via `pytest-cov`.
- **Frontend**: full coverage of every pure/exported function in `static/app.js`, plus DOM-behavior
  tests for the app's rendering and event-wiring logic using `jsdom`.
- Tests must be deterministic, fast, and independent of the real `data/` pickle files shipped with
  the app — everything is built from fixtures created in-test.
- No network calls, no reliance on `scripts/generate_sample_data.py` output existing on disk.

---

## 1. Backend (pytest)

### 1.1 Tooling & layout

```
pip install pytest pytest-cov pytest-mock
```

```
tests/
├── conftest.py
├── test_weight_transforms.py
├── test_dataset_loader.py
└── test_app.py
```

`pytest.ini` (or `pyproject.toml [tool.pytest.ini_options]`):

```ini
[pytest]
testpaths = tests
addopts = -q --cov=. --cov-report=term-missing --cov-report=html --cov-fail-under=95
          --cov-config=.coveragerc
```

`.coveragerc` should omit non-application code:

```ini
[run]
omit =
    scripts/*
    */site-packages/*
    tests/*
```

> `scripts/generate_sample_data.py` is a one-off data-generation utility, not application logic —
> excluding it from the coverage denominator keeps the 95% target meaningful and focused on
> `app.py`, `dataset_loader.py`, and `weight_transforms.py`.

### 1.2 `conftest.py` fixtures

Build everything from in-memory DataFrames — never depend on `data/*.pkl` on disk.

```python
import json
import pandas as pd
import pytest
from pathlib import Path

@pytest.fixture
def sample_df():
    return pd.DataFrame({
        "title":       ["Alpha", "Beta", "Gamma", "Delta"],
        "category":    ["Web", "Web", "Data", "Data"],
        "price":       [10.0, 50.0, 0.0, 200.0],
        "rating":      [4.9, 3.5, 4.2, 4.99],
        "enrollments": [10, 5000, 25, 100000],
    })

@pytest.fixture
def sample_config():
    return {
        "label": "Test Catalog",
        "identifier_column": "title",
        "channels": [
            {"column": "rating", "label": "Rating", "normalize": "minmax",
             "default_transform": {"type": "amplify", "power": 4.0, "scale": 2.5}},
            {"column": "enrollments", "label": "Enrollments", "normalize": "log_minmax",
             "default_transform": {"type": "linear", "w": 1.0}},
        ],
        "filters": [
            {"column": "category", "label": "Category", "type": "categorical"},
            {"column": "price", "label": "Price", "type": "range"},
        ],
    }

@pytest.fixture
def dataset(sample_df, sample_config):
    from dataset_loader import Dataset
    return Dataset("sample_courses", sample_df.copy(), json.loads(json.dumps(sample_config)))

@pytest.fixture
def data_dir(tmp_path, sample_df, sample_config):
    """A temp data/ dir with one valid <name>.pkl + <name>.config.json pair."""
    sample_df.to_pickle(tmp_path / "sample_courses.pkl")
    (tmp_path / "sample_courses.config.json").write_text(json.dumps(sample_config))
    return tmp_path
```

### 1.3 `weight_transforms.py` — pure math engine

This module is the highest-value target: pure functions, no I/O, easy to hit 100%.

| Area | What to test |
|---|---|
| Each factory (`linear`, `clamp`, `amplify`, `log_compress`, `sigmoid`, `threshold`, `passthrough`) | Correct output for scalar and `np.ndarray` input; default params; each explicit kwarg. |
| `clamp` | `min_v` only, `max_v` only, both, neither (falls back to `scale * x`). |
| `amplify` | Negative input preserves sign (`sign(x) * |x|^power`). |
| `threshold` | Value exactly at `level` goes to `above` (boundary is inclusive per `np.where(x < level, ...)`). |
| `Transform.__call__` / `__repr__` | Repr contains name and formatted params. |
| `Transform.to_latex` / `to_python` | One test per transform `name` branch, including the `else` fallback for an unrecognized name (construct a `Transform` directly with a bogus name). Assert on exact string content, not just non-emptiness, since these strings are user-facing. |
| `compose` | Chains 2+ steps left-to-right; name string is `"a -> b -> c"`. |
| `from_config` | Valid spec builds correct `Transform`; unknown `type` raises `ValueError`; stray/unknown keys in the spec are silently dropped (don't reach the factory) — verify via a spec with an extra bogus key. |
| `weighted_sum` | Correct total + per-input contribution list; mismatched `len(inputs)` vs `len(transforms)` raises `ValueError`. |
| `vectorized_weighted_sum` | Same as above but with array inputs; confirm it returns arrays, not scalars. |
| `compile_pipeline_formulas` | Zero knobs → `{"latex": "y = 0", "python": "y = 0"}`; default `x_i`/`xN` var names when `latex_vars`/`python_vars` omitted; custom var names substituted correctly; mismatched var-list length raises `ValueError`. |
| `TRANSFORM_SPECS` / `REGISTRY` | Assert every key in `REGISTRY` has a matching entry in `TRANSFORM_SPECS` (keeps the two tables from drifting apart silently). |

Use `pytest.mark.parametrize` to drive the factory/to_latex/to_python tests off a single table of
`(transform_name, kwargs, sample_input)` rather than one function per knob — keeps this file
maintainable and is the fastest path to 95%+ coverage of the file.

### 1.4 `dataset_loader.py` — config validation & loading

| Area | What to test |
|---|---|
| `Dataset._assign_colors` | Missing `color`/`normalize`/`normalize_scope`/`label` on a channel get sensible defaults; colors cycle through `COLOR_PALETTE` by index (test with >9 channels to hit the modulo wraparound). |
| `Dataset._compute_global_bounds` | `minmax` vs `log_minmax` bounds differ as expected; empty DataFrame → `(0.0, 1.0)` fallback. |
| `Dataset.filter_options` | Categorical → sorted unique string values; range → `[min, max]` floats; empty DataFrame edge case for both types. |
| `Dataset.to_public_config` | Contains all expected top-level keys; `row_count` matches `len(df)`. |
| `_apply_normalize_space` | `log_minmax` applies `log1p(clip(x, 0, None))`; anything else passes through unchanged. |
| `_validate_config` | One test per failure mode, each asserting `DatasetConfigError` (subclass of `ValueError`) with a message mentioning the offending field: missing top-level keys; `identifier_column` not in df; empty `channels` list; channel `column` not in df; channel column non-numeric; invalid `normalize`; invalid `normalize_scope`; filter `column` not in df; invalid filter `type`. Also one **happy-path** test that a fully valid config raises nothing. |
| `load_datasets` | (a) Non-existent `data_dir` → `{}`. (b) `.pkl` with no matching `.config.json` is skipped, not raised (use `tmp_path`, write only the pickle). (c) `.pkl` + config that fails validation is skipped, not raised — assert the *other* valid dataset in the same dir still loads (use `capsys` to assert the warning was printed, not raised). (d) Multiple valid pairs all load and dict keys equal file stems. |

Use `tmp_path` for every `load_datasets` test — never touch the real `data/` directory.

### 1.5 `app.py` — Flask routes

Use Flask's test client. Because `app.py` builds `DATASETS` at import time from `DATA_DIR = Path(__file__).parent / "data"`, **monkeypatch before import** so tests don't depend on the real dataset shipped in the repo:

```python
import importlib
import sys

@pytest.fixture
def client(monkeypatch, data_dir):
    monkeypatch.setattr("dataset_loader.load_datasets", lambda _dir: {
        "sample_courses": _build_test_dataset()  # or reuse the `dataset` fixture's Dataset
    })
    sys.modules.pop("app", None)
    import app as app_module
    app_module.app.config.update(TESTING=True)
    with app_module.app.test_client() as c:
        yield c
```

(Adjust based on whichever monkeypatch strategy is cleanest — the key requirement is that
`app.py`'s module-level `DATASETS = load_datasets(DATA_DIR)` must resolve to fixture data, not the
real `data/sample_courses.pkl`, so tests are hermetic and don't break if that file is ever removed
or regenerated.)

| Route | What to test |
|---|---|
| `RuntimeError` at import | If `load_datasets` returns `{}`, importing `app` raises `RuntimeError` (test in isolation, reloading the module with an empty-dict patch). |
| `get_dataset` | Known name returns that dataset; unknown/`None` name falls back to `DEFAULT_DATASET`. |
| `sanitize_identifier` | Removes non-alphanumeric chars → `_`; leading digit gets `_` prefix; empty string → `"x"`. |
| `GET /` | 200; `active_dataset` defaults correctly; `?dataset=` query param switches dataset in the rendered context (check via `response.data` containing the dataset name, or mock `render_template` and inspect call kwargs). |
| `GET /api/datasets` | Returns `datasets` list with `name`/`label` per dataset and correct `active`. |
| `GET /api/transforms` | Returns `specs` (equal to `TRANSFORM_SPECS`) and `dataset` (equal to `to_public_config()`); `?dataset=` param selects the right one. |
| `GET /api/popular_values` | Unknown column → 400 with `error` key; non-categorical (range) column → 400; valid categorical column → top values sorted by count, capped at 8; `scope_column`/`scope_value` narrows the value_counts; `HX-Request` header → renders `_popular_tags.html` fragment instead of JSON (assert `response.content_type` and/or fragment markup, not JSON). |
| `POST /api/compute` — pipeline length validation | Pipeline length ≠ number of configured channels → 400 with descriptive error. |
| `POST /api/compute` — invalid transform | A pipeline entry with unknown `type` → 400, error message includes `"Invalid transform configuration"`. |
| `POST /api/compute` — filtering | Categorical filter narrows rows; range filter `[lo, hi]` narrows rows (both inclusive bounds); a filter value of `None`/`""`/`[]` is a no-op (must not exclude everything); filtering to zero rows → early-return payload `{"leaderboard": [], "watchlist": [], "formula": {...}}`. |
| `POST /api/compute` — normalization | `normalize_scope: "global"` uses `ds.channel_bounds` even after filtering (score for a fixed row is stable regardless of active filters); `normalize_scope: "filtered"` recomputes bounds from the filtered subset only; a `filtered` scope with a single remaining row (`span == 0`) doesn't divide by zero (span falls back to `1.0`). |
| `POST /api/compute` — scoring & ranking | `total_score` equals the sum of per-channel contributions; `leaderboard` is sorted descending by `total_score`; `rank` is 1-indexed and contiguous; `top_x` truncates the leaderboard to that many rows; `top_x` larger than the row count returns all rows; `top_x: 0` or negative → empty leaderboard, not an error. |
| `POST /api/compute` — watchlist | Identifier present in the filtered/scored set → `found: true` with correct `rank`/`total_score`/`breakdown`; identifier absent (e.g. because a filter excluded it, or it never existed) → `found: false`, `rank: null`, `total_score: 0.0`, `breakdown: []`; blank/whitespace-only entries in `target_identifiers` are dropped before matching. |
| `POST /api/compute` — formula | Response `formula.latex`/`formula.python` are non-empty strings whose term count matches the number of channels; sanitized variable names (spaces/punctuation in column names) don't break the formula string. |
| `POST /api/compute` — malformed body | No JSON body / `Content-Type` not JSON → treated as `{}` (via `silent=True`), doesn't 500. |

Because `/api/compute` is the core business-logic route, prefer several small, targeted requests
over one giant end-to-end test — each row in the table above should be its own test function so
failures are easy to localize.

### 1.6 Reaching 95%

After the above, run `pytest --cov-report=term-missing` and inspect the "Missing" line numbers per
file. Common gaps to check for:
- Both branches of every `if norm == "log_minmax"` / `if scope == "global"` style condition in
  `app.py`'s `/api/compute`.
- The `else` fallback branches in `Transform.to_latex`/`to_python`.
- The `except Exception` branches in `load_datasets` and the `from_config` try/except in
  `api_compute` (feed genuinely invalid input, don't just assert the happy path).
- The `if __name__ == "__main__":` blocks in `app.py` and `weight_transforms.py` are conventionally
  excluded from coverage — add `# pragma: no cover` to those blocks rather than writing
  process-spawning tests for them.

---

## 2. Frontend (vitest)

### 2.1 Tooling & layout

```
npm install -D vitest jsdom @vitest/coverage-v8
```

`vite.config.js` / `vitest.config.js`:

```js
export default {
  test: {
    environment: "jsdom",
    coverage: { provider: "v8", reporter: ["text", "html"] },
  },
};
```

```
static/
└── app.js
tests/
├── pure-helpers.test.js
└── dom-behavior.test.js
```

### 2.2 Why `app.js` is split into two testing tiers

The module guards its side-effecting `init()` call behind `typeof document !== "undefined"`, and
exports four pure helpers explicitly for testability:

```js
export function computeBreakdownWidths(breakdown) { ... }
export function buildPipelinePayload(channelState) { ... }
export function parseWatchlist(text) { ... }
export function sanitizeNumber(value, fallback) { ... }
```

Everything else (`init`, `renderFilters`, `renderChannels`, `compute`, `renderLeaderboard`, the
transform-reference modal, etc.) is DOM-driven, module-private, and only reachable by triggering
`init()` inside a real `document`. Test these in two separate files/tiers rather than one, since
they need different setup.

### 2.3 Tier 1 — pure helpers (`pure-helpers.test.js`)

No DOM needed at all; these should run identically under Node or jsdom.

| Function | Cases |
|---|---|
| `computeBreakdownWidths` | Normal breakdown with 2+ entries sums to 100%; entry with `contribution: 0`/`undefined`/negative value (uses `Math.abs`); empty array doesn't divide by zero (denominator falls back to `1`); single-entry breakdown → 100%. |
| `buildPipelinePayload` | Spreads `params` alongside `type` for each channel; empty `channelState` → `[]`; params object with overlapping keys to `type` doesn't get clobbered (params spread after `type`, confirm the order in the source: `{ type: ch.type, ...ch.params }` — a `params.type` would win; add a case asserting this if it's a real risk). |
| `parseWatchlist` | Splits on `\n` and `\r\n`; trims whitespace per line; drops blank lines; empty string → `[]`; single line without trailing newline still parses. |
| `sanitizeNumber` | Valid numeric string → number; `"abc"` → fallback; `""` → fallback; `null`/`undefined` → fallback (per `Number(null) === 0`, verify actual behavior — `Number(null)` is `0`, which is finite, so document/test that `null` returns `0`, not the fallback); `Infinity`/`NaN` → fallback. |

### 2.4 Tier 2 — DOM behavior (`dom-behavior.test.js`)

Since `init()` only runs when `document` exists and immediately does `fetch(...)`, these tests need:
1. A DOM fixture resembling `templates/index.html`'s key element IDs (`dataset-tagline`,
   `channel-shaper-title`, `filters-container`, `channels`, `watchlist-input`, `top-x-input`,
   `copy-latex-btn`, `copy-python-btn`, `transform-reference-btn`, `close-modal-btn`,
   `transform-modal`, `error-banner`, `leaderboard-body`, `watchlist-body`, `score-summary`,
   `code-display`, `latex-display`, and a `<script id="transform-specs">` tag with the JSON specs).
2. `global.fetch` mocked (`vi.fn()`) to return canned `/api/transforms` and `/api/compute`
   responses.
3. A stubbed global `katex` object (`{ render: vi.fn() }`) since `app.js` calls
   `katex.render(...)` unconditionally in `renderFormula`/`renderModalContent` and the real KaTeX
   library isn't loaded in tests.
4. Because `app.js` runs `init()` as a side effect at import time (guarded on `document`), set up
   the DOM fixture and mocks **before** importing the module in each test file (or use
   `vi.resetModules()` + dynamic `await import("../static/app.js")` per test so `init()` re-runs
   against a fresh DOM each time).

| Area | What to test |
|---|---|
| `init()` bootstrap | After awaiting init, `dataset-tagline` and `channel-shaper-title` text content reflect the fetched dataset's `label`/`row_count`/`channels.length`/`filters.length`; a `compute()` POST is fired once during bootstrap. |
| `renderFilters` | Categorical filter renders a `<select>` with an "All X" option plus one `<option>` per `filter_options` value, and triggers `loadPopularTags` (assert the second `fetch` call for `/api/popular_values`); range filter renders min/max number inputs pre-filled with the option bounds. |
| `collectFilters` (indirectly, via triggering `compute`) | Selecting a categorical value and changing a range input changes the JSON body of the next `/api/compute` `fetch` call; blank categorical selection is omitted from the filters payload. |
| `loadPopularTags` → tag buttons | Clicking a rendered `.popular-tag` button sets the corresponding `<select>` value and triggers a recompute (`fetch` called again). |
| `renderChannels` / transform `<select>` | Changing a channel's transform-type dropdown updates `CHANNEL_STATE` (indirectly verify via the next compute payload's `pipeline` entry) and re-renders that channel's param controls with the new type's defaults. |
| `renderParamControls` — slider input | Moving a param `<input type="range">` updates the adjacent readout `<span>` text and schedules a recompute. |
| `compute()` — success path | Given a canned `/api/compute` JSON response, `leaderboard-body`/`watchlist-body`/`score-summary`/`code-display` all get populated; `error-banner` stays hidden. |
| `compute()` — error path | Response body containing `{"error": "..."}` populates `error-banner` with that message and does **not** touch leaderboard/watchlist. |
| `compute()` — network failure | `fetch` rejecting (mock `vi.fn().mockRejectedValue(...)`) shows `"Failed to compute: <message>"` in `error-banner`. |
| `renderLeaderboard` — empty state | `rows: []` renders the "No rows match" placeholder row and sets `score-summary` to `"-"`. |
| `renderWatchlist` — empty & mixed | `rows: []` renders "No items on watchlist"; a mix of `found: true`/`false` rows renders each variant's markup (rank/score vs "not found"). |
| `copyText` / `fallbackCopy` | With `navigator.clipboard.writeText` mocked to resolve, button text flips to `"copied"` then back after the timeout (use `vi.useFakeTimers()`); with `navigator.clipboard` undefined, falls back to the hidden-textarea + `document.execCommand("copy")` path (mock `document.execCommand`). |
| Modal open/close | `transform-reference-btn` click opens the modal (`transform-modal` loses `hidden`, gains `flex`) and renders tabs/content for the current channel-0 type; `close-modal-btn` click and clicking the modal backdrop (the `#transform-modal` element itself, not its children) both close it; clicking a modal tab switches `renderModalContent` to that type. |
| `renderCurveSVG` (via modal content) | Rendered SVG `<polyline>` point count matches the sampled resolution (41 points) — can assert via a simple string/DOM query rather than pixel values. |

### 2.5 Coverage note

Frontend coverage is not held to a hard percentage the way the backend is, since large parts of
`renderModalContent`/`renderCurveSVG` are presentational SVG string-building. Prioritize:
1. 100% of the four exported pure helpers (cheap, high value).
2. Every state-changing user interaction path (filter change, transform change, param change,
   watchlist/top-x input, copy buttons, modal open/close) — these are the most likely to silently
   break during refactors and are what a regression suite is actually for.

---

## 3. Running everything

```bash
# Backend
pytest

# Frontend
npx vitest run --coverage
```

Both commands should be wired into CI as separate jobs (Python + Node), each failing the build if
coverage drops below the thresholds above.
