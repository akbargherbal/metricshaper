// Paste frontend module logic here (app.js)
// Safe DOM extraction for Node test runner compatibility
const specsEl =
  typeof document !== "undefined"
    ? document.getElementById("transform-specs")
    : null;
const SPECS = specsEl ? JSON.parse(specsEl.textContent) : {};
const ACTIVE_DATASET =
  typeof window !== "undefined" ? window.ACTIVE_DATASET : null;

let DATASET = null; // public dataset config from /api/transforms
let CHANNEL_STATE = []; // [{ column, type, params }]
let computeTimer = null;
let LATEST_FORMULA = { latex: "", python: "" };

// ---------------------------------------------------------------------------
// Pure, exported helpers (covered by Vitest — no DOM access)
// ---------------------------------------------------------------------------

export function computeBreakdownWidths(breakdown) {
  const abs = breakdown.map((b) => Math.abs(b.contribution || 0));
  const total = abs.reduce((a, b) => a + b, 0) || 1;
  return breakdown.map((b, i) => ({
    column: b.column,
    color: b.color,
    pct: (abs[i] / total) * 100,
  }));
}

export function buildPipelinePayload(channelState) {
  return channelState.map((ch) => ({ type: ch.type, ...ch.params }));
}

export function parseWatchlist(text) {
  return text
    .split(/\r?\n/)
    .map((s) => s.trim())
    .filter(Boolean);
}

export function sanitizeNumber(value, fallback) {
  if (value === "" || (typeof value === "string" && value.trim() === "")) {
    return fallback;
  }
  const n = Number(value);
  return Number.isFinite(n) ? n : fallback;
}

// ---------------------------------------------------------------------------
// DOM-dependent app logic (skipped in the Vitest/Node environment)
// ---------------------------------------------------------------------------

if (typeof document !== "undefined") {
  init();
}

async function init() {
  const datasetSelect = document.getElementById("dataset-select");
  if (datasetSelect) {
    datasetSelect.addEventListener("change", (e) => {
      window.location.search = "?dataset=" + encodeURIComponent(e.target.value);
    });
  }

  const res = await fetch(
    `/api/transforms?dataset=${encodeURIComponent(ACTIVE_DATASET || "")}`,
  );
  const data = await res.json();
  DATASET = data.dataset;

  document.getElementById("dataset-tagline").textContent =
    `${DATASET.label} · ${DATASET.row_count} rows · ${DATASET.channels.length} channels · ${DATASET.filters.length} filters`;
  document.getElementById("channel-shaper-title").textContent =
    `CHANNEL SHAPER — ${DATASET.channels.length} OF ${DATASET.channels.length} ACTIVE`;

  CHANNEL_STATE = DATASET.channels.map((ch) => ({
    column: ch.column,
    type: ch.default_transform?.type || "linear",
    params: extractParams(ch.default_transform),
  }));

  renderFilters();
  renderChannels();
  bindGlobalControls();
  await compute();
}

function extractParams(transformSpec) {
  const { type, ...rest } = transformSpec || {};
  return rest;
}

function bindGlobalControls() {
  document
    .getElementById("watchlist-input")
    .addEventListener("input", scheduleCompute);
  document
    .getElementById("top-x-input")
    .addEventListener("input", scheduleCompute);
  document
    .getElementById("copy-latex-btn")
    .addEventListener("click", (e) => copyText(e.target, LATEST_FORMULA.latex));
  document
    .getElementById("copy-python-btn")
    .addEventListener("click", (e) =>
      copyText(e.target, LATEST_FORMULA.python),
    );
  document
    .getElementById("transform-reference-btn")
    .addEventListener("click", () =>
      openTransformModal(CHANNEL_STATE[0]?.type || "linear"),
    );
  document
    .getElementById("close-modal-btn")
    .addEventListener("click", closeTransformModal);
  document.getElementById("transform-modal").addEventListener("click", (e) => {
    if (e.target.id === "transform-modal") closeTransformModal();
  });
}

function scheduleCompute() {
  clearTimeout(computeTimer);
  computeTimer = setTimeout(compute, 250);
}

// ---------------------------------------------------------------------------
// Filters
// ---------------------------------------------------------------------------

function renderFilters() {
  const container = document.getElementById("filters-container");
  container.innerHTML = "";

  DATASET.filters.forEach((f) => {
    const wrap = document.createElement("div");
    wrap.className = "flex flex-col gap-1.5";

    if (f.type === "categorical") {
      wrap.innerHTML = `
        <label class="font-mono text-[10px] uppercase tracking-wider text-slate-500 font-bold">${f.label}</label>
        <select id="filter-${f.column}" class="bg-[#0e121a] border border-slate-850 text-slate-200 rounded-lg px-3 py-2 text-xs font-mono focus:outline-none focus:ring-1 focus:ring-teal-500/40 shadow-inner">
          <option value="">All ${f.label.toLowerCase()}</option>
          ${DATASET.filter_options[f.column].map((v) => `<option value="${v}">${v}</option>`).join("")}
        </select>
        <div id="popular-${f.column}" class="flex flex-wrap gap-1.5 pt-1"></div>
      `;
      container.appendChild(wrap);
      const select = wrap.querySelector("select");
      select.addEventListener("change", scheduleCompute);
      loadPopularTags(f.column);
    } else if (f.type === "range") {
      const [lo, hi] = DATASET.filter_options[f.column];
      wrap.innerHTML = `
        <label class="font-mono text-[10px] uppercase tracking-wider text-slate-500 flex justify-between font-bold">
          <span>${f.label}</span>
          <span class="text-slate-600 font-light lowercase font-normal">${lo} &ndash; ${hi}</span>
        </label>
        <div class="flex items-center gap-2">
          <input type="number" id="filter-${f.column}-min" value="${lo}" step="any" class="w-1/2 bg-[#0e121a] border border-slate-850 text-slate-200 rounded-lg px-2 py-1.5 text-xs font-mono shadow-inner">
          <span class="text-slate-600 text-xs">to</span>
          <input type="number" id="filter-${f.column}-max" value="${hi}" step="any" class="w-1/2 bg-[#0e121a] border border-slate-850 text-slate-200 rounded-lg px-2 py-1.5 text-xs font-mono shadow-inner">
        </div>
      `;
      container.appendChild(wrap);
      wrap
        .querySelectorAll("input")
        .forEach((el) => el.addEventListener("input", scheduleCompute));
    }
  });
}

async function loadPopularTags(column) {
  const res = await fetch(
    `/api/popular_values?dataset=${encodeURIComponent(DATASET.name)}&column=${encodeURIComponent(column)}`,
  );
  const data = await res.json();
  const el = document.getElementById(`popular-${column}`);
  if (!el || !data.popular_values) return;
  el.innerHTML = data.popular_values
    .map(
      (item) =>
        `<button type="button" data-column="${column}" data-value="${item.value}" class="popular-tag px-2 py-1 text-[11px] bg-[#080b11] hover:bg-slate-900 border border-slate-900 text-slate-300 rounded-md font-mono">${item.value} <span class="text-[9px] text-slate-500">(${item.count})</span></button>`,
    )
    .join("");
  el.querySelectorAll(".popular-tag").forEach((btn) => {
    btn.addEventListener("click", () => {
      const select = document.getElementById(`filter-${btn.dataset.column}`);
      if (select) {
        select.value = btn.dataset.value;
        scheduleCompute();
      }
    });
  });
}

function collectFilters() {
  const filters = {};
  DATASET.filters.forEach((f) => {
    if (f.type === "categorical") {
      const el = document.getElementById(`filter-${f.column}`);
      if (el && el.value) filters[f.column] = el.value;
    } else if (f.type === "range") {
      const minEl = document.getElementById(`filter-${f.column}-min`);
      const maxEl = document.getElementById(`filter-${f.column}-max`);
      if (minEl && maxEl) {
        filters[f.column] = [
          sanitizeNumber(minEl.value, 0),
          sanitizeNumber(maxEl.value, 0),
        ];
      }
    }
  });
  return filters;
}

// ---------------------------------------------------------------------------
// Channels
// ---------------------------------------------------------------------------

function renderChannels() {
  const container = document.getElementById("channels");
  container.innerHTML = "";

  DATASET.channels.forEach((ch, idx) => {
    const card = document.createElement("div");
    card.className =
      "bg-[#0e121a] border border-slate-900 rounded-xl p-4 space-y-3";
    card.innerHTML = `
      <div class="flex items-center justify-between">
        <span class="text-[10px] font-mono uppercase tracking-wider font-bold text-${ch.color}-400">${ch.label}</span>
        <div class="flex items-center gap-1.5">
          <button data-transform-help="${idx}" class="text-slate-600 hover:text-teal-400 text-[10px] font-mono w-4 h-4 flex items-center justify-center border border-slate-800 rounded-full" title="View formula">?</button>
          <span class="w-2 h-2 rounded-full bg-${ch.color}-500"></span>
        </div>
      </div>
      <select data-transform-select="${idx}" class="w-full bg-[#080b11] border border-slate-850 text-slate-200 rounded-lg px-2 py-1.5 text-[11px] font-mono">
        ${Object.keys(SPECS)
          .map(
            (key) =>
              `<option value="${key}" ${key === CHANNEL_STATE[idx].type ? "selected" : ""}>${SPECS[key].label}</option>`,
          )
          .join("")}
      </select>\n      <div data-param-container="${idx}" class="space-y-1.5"></div>
      ${ch.warning ? `<p class="text-[9px] text-${ch.color}-500/70 font-mono leading-snug">${ch.warning}</p>` : ""}
    `;
    container.appendChild(card);

    renderParamControls(idx);

    card
      .querySelector(`[data-transform-select="${idx}"]`)
      .addEventListener("change", (e) => {
        CHANNEL_STATE[idx].type = e.target.value;
        CHANNEL_STATE[idx].params = defaultParamsFor(e.target.value);
        renderParamControls(idx);
        scheduleCompute();
      });

    card
      .querySelector(`[data-transform-help="${idx}"]`)
      .addEventListener("click", () => {
        openTransformModal(CHANNEL_STATE[idx].type);
      });
  });
}

function defaultParamsFor(type) {
  const params = {};
  const spec = SPECS[type];
  if (!spec) return params;
  Object.entries(spec.params).forEach(([key, meta]) => {
    params[key] = meta.default;
  });
  return params;
}

function renderParamControls(idx) {
  const container = document.querySelector(`[data-param-container="${idx}"]`);
  const type = CHANNEL_STATE[idx].type;
  const spec = SPECS[type];
  container.innerHTML = "";
  if (!spec) return;

  Object.entries(spec.params).forEach(([key, meta]) => {
    const currentValue = CHANNEL_STATE[idx].params[key] ?? meta.default;
    const row = document.createElement("div");
    row.className = "space-y-1";
    row.innerHTML = `
      <div class="flex justify-between text-[10px] font-mono text-slate-500">
        <span>${meta.label}</span>
        <span class="text-slate-300" data-param-readout="${idx}-${key}">${currentValue}</span>
      </div>
      <input type="range" min="${meta.nullable ? 0 : (meta.min ?? 0)}" max="${meta.max ?? 10}" step="${meta.step ?? 0.1}" value="${currentValue ?? meta.default ?? 0}" class="w-full bg-[#080b11] border border-slate-850 rounded-lg h-1" data-param-slider="${idx}-${key}">
    `;
    container.appendChild(row);

    row
      .querySelector(`[data-param-slider="${idx}-${key}"]`)
      .addEventListener("input", (e) => {
        const val = sanitizeNumber(e.target.value, meta.default ?? 0);
        CHANNEL_STATE[idx].params[key] = val;
        document.querySelector(
          `[data-param-readout="${idx}-${key}"]`,
        ).textContent = val;
        scheduleCompute();
      });
  });
}

// ---------------------------------------------------------------------------
// Compute + render results
// ---------------------------------------------------------------------------

async function compute() {
  const payload = {
    dataset: DATASET.name,
    filters: collectFilters(),
    target_identifiers: parseWatchlist(
      document.getElementById("watchlist-input").value,
    ),
    pipeline: buildPipelinePayload(CHANNEL_STATE),
    top_x: sanitizeNumber(document.getElementById("top-x-input").value, 20),
  };

  const errorBanner = document.getElementById("error-banner");
  try {
    const res = await fetch("/api/compute", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    const data = await res.json();
    if (data.error) {
      errorBanner.textContent = data.error;
      errorBanner.classList.remove("hidden");
      return;
    }
    errorBanner.classList.add("hidden");
    renderLeaderboard(data.leaderboard);
    renderWatchlist(data.watchlist);
    renderFormula(data.formula);
  } catch (err) {
    errorBanner.textContent = "Failed to compute: " + err.message;
    errorBanner.classList.remove("hidden");
  }
}

function renderLeaderboard(rows) {
  const body = document.getElementById("leaderboard-body");
  if (!rows.length) {
    body.innerHTML = `<tr><td colspan="4" class="px-4 py-8 text-center text-slate-500 font-mono italic">No rows match the current filters.</td></tr>`;
    document.getElementById("score-summary").textContent = "-";
    return;
  }
  body.innerHTML = rows.map((row) => rowToHtml(row, row.rank === 1)).join("");
  const scores = rows.map((r) => r.total_score);
  document.getElementById("score-summary").textContent =
    `${Math.min(...scores).toFixed(2)} – ${Math.max(...scores).toFixed(2)}`;
}

function renderWatchlist(rows) {
  const body = document.getElementById("watchlist-body");
  if (!rows.length) {
    body.innerHTML = `<tr><td colspan="3" class="px-4 py-6 text-center text-slate-500 font-mono italic">No items on watchlist.</td></tr>`;
    return;
  }
  body.innerHTML = rows
    .map((row) => {
      if (!row.found) {
        return `<tr><td class="px-4 py-2.5 text-slate-400 font-sans">${row.id}</td><td class="px-3 py-2.5 text-center text-slate-600">&mdash;</td><td class="px-4 py-2.5 text-right text-slate-600">not found</td></tr>`;
      }
      return `<tr class="bg-teal-500/5"><td class="px-4 py-2.5 text-slate-200 font-sans">${row.id}</td><td class="px-3 py-2.5 text-center text-teal-400 font-bold">${row.rank}</td><td class="px-4 py-2.5 text-right text-teal-400 font-bold">${row.total_score.toFixed(2)}</td></tr>`;
    })
    .join("");
}

function rowToHtml(row, isTop) {
  const widths = computeBreakdownWidths(row.breakdown);
  const bars = widths
    .map(
      (w) =>
        `<div class="bg-${w.color}-500" style="width:${w.pct.toFixed(1)}%"></div>`,
    )
    .join("");
  return `
    <tr class="${isTop ? "bg-teal-500/5" : ""}">
      <td class="px-3 py-3 text-center ${isTop ? "text-teal-400" : "text-slate-400"} font-bold">${row.rank}</td>
      <td class="px-4 py-3 ${isTop ? "text-slate-100" : "text-slate-300"} font-sans">${row.id}</td>
      <td class="px-4 py-3"><div class="flex h-2 rounded-full overflow-hidden bg-slate-900">${bars}</div></td>
      <td class="px-4 py-3 text-right ${isTop ? "text-teal-400" : "text-slate-300"} font-bold">${row.total_score.toFixed(2)}</td>
    </tr>
  `;
}

function renderFormula(formula) {
  LATEST_FORMULA = formula;
  document.getElementById("code-display").textContent = formula.python;
  try {
    // eslint-disable-next-line no-undef
    katex.render(formula.latex, document.getElementById("latex-display"), {
      throwOnError: false,
    });
  } catch (e) {
    document.getElementById("latex-display").textContent = formula.latex;
  }
}

function copyText(btn, text) {
  const original = btn.textContent;
  const done = () => {
    btn.textContent = "copied";
    setTimeout(() => (btn.textContent = original), 1200);
  };
  if (navigator.clipboard && navigator.clipboard.writeText) {
    navigator.clipboard
      .writeText(text)
      .then(done)
      .catch(() => fallbackCopy(text, done));
  } else {
    fallbackCopy(text, done);
  }
}

function fallbackCopy(text, done) {
  const ta = document.createElement("textarea");
  ta.value = text;
  document.body.appendChild(ta);
  ta.select();
  try {
    document.execCommand("copy");
  } catch (e) {
    /* no-op */
  }
  document.body.removeChild(ta);
  done();
}

// ---------------------------------------------------------------------------
// Transform reference modal (educational, kept intentionally separate from
// the main scoring path — one data object + one render function, so adding
// a new knob later is a one-line addition here, mirroring its factory in
// weight_transforms.py).
// ---------------------------------------------------------------------------

const CURVE_FNS = {
  linear: (x) => x,
  clamp: (x) => Math.min(x, 0.6),
  amplify: (x) => Math.pow(x, 3),
  log_compress: (x) => Math.log1p(x * 10) / Math.log1p(10),
  sigmoid: (x) => 1 / (1 + Math.exp(-10 * (x - 0.5))),
  threshold: (x) => (x < 0.5 ? 0.2 : 0.9),
  passthrough: (x) => x,
};

const LATEX_BY_TYPE = {
  linear: "f(x) = w \\cdot x",
  clamp: "f(x) = s \\cdot \\min(\\max(x, m), M)",
  amplify: "f(x) = s \\cdot \\operatorname{sgn}(x) \\cdot |x|^{p}",
  log_compress: "f(x) = s \\cdot \\ln(1 + \\max(0, x))",
  sigmoid: "f(x) = \\dfrac{s}{1 + e^{-k(x-c)}}",
  threshold: "f(x) = \\begin{cases} b & x < l \\\\ a & x \\ge l \\end{cases}",
  passthrough: "f(x) = x",
};

function renderCurveSVG(fn) {
  const n = 40;
  const ys = [];
  for (let i = 0; i <= n; i++) ys.push(fn(i / n));
  const ymin = Math.min(...ys);
  const ymax = Math.max(...ys);
  const range = ymax - ymin || 1;
  const pts = ys.map((y, i) => {
    const px = 10 + (i / n) * 180;
    const py = 90 - ((y - ymin) / range) * 75;
    return `${px.toFixed(1)},${py.toFixed(1)}`;
  });
  return `<svg viewBox="0 0 200 100" class="w-full h-24">
    <line x1="10" y1="90" x2="190" y2="90" stroke="#1e293b" stroke-width="1"/>
    <line x1="10" y1="10" x2="10" y2="90" stroke="#1e293b" stroke-width="1"/>
    <polyline points="${pts.join(" ")}" fill="none" stroke="#2dd4bf" stroke-width="2"/>
  </svg>`;
}

function renderModalTabs(activeKey) {
  const tabsEl = document.getElementById("modal-tabs");
  tabsEl.innerHTML = Object.keys(SPECS)
    .map((key) => {
      const active = key === activeKey;
      return `<button data-modal-tab="${key}" class="text-left px-4 py-2 text-[11px] font-mono ${
        active
          ? "text-teal-400 bg-teal-500/5 border-r-2 border-teal-500"
          : "text-slate-500 hover:text-slate-300"
      }">${SPECS[key].label}</button>`;
    })
    .join("");
  tabsEl.querySelectorAll("[data-modal-tab]").forEach((btn) => {
    btn.addEventListener("click", () =>
      openTransformModal(btn.dataset.modalTab),
    );
  });
}

function renderModalContent(key) {
  const spec = SPECS[key];
  const el = document.getElementById("modal-content");
  const pythonSnippet = `def ${key}(x${Object.keys(spec.params).length ? ", " : ""}${Object.entries(
    spec.params,
  )
    .map(([k, m]) => `${k}=${m.default}`)
    .join(", ")}):\n    ...`;
  el.innerHTML = `
    <div class="bg-[#0e121a] border border-slate-900 rounded-lg p-4 flex items-center justify-center min-h-[60px]" id="modal-latex"></div>
    <p class="text-[13px] text-slate-400 leading-relaxed">${spec.description}</p>
    <div class="bg-[#0e121a] border border-slate-900 rounded-lg p-3">${renderCurveSVG(CURVE_FNS[key] || ((x) => x))}</div>
    <div class="space-y-1">
      <div class="flex justify-between items-center">
        <span class="text-[9px] font-mono text-slate-500 uppercase tracking-wider font-extrabold">Signature</span>
        <button id="modal-copy-btn" class="copy-btn px-2 py-1 text-[10px] font-mono text-slate-400 hover:text-teal-400 hover:bg-slate-900 border border-slate-800 rounded-md">copy</button>
      </div>
      <pre class="bg-[#121620] rounded-lg p-3 border border-slate-800 text-[11px] text-teal-400 font-mono overflow-x-auto">${pythonSnippet}</pre>
    </div>
  `;
  // eslint-disable-next-line no-undef
  katex.render(
    LATEX_BY_TYPE[key] || "f(x) = x",
    document.getElementById("modal-latex"),
    { throwOnError: false },
  );
  document
    .getElementById("modal-copy-btn")
    .addEventListener("click", (e) => copyText(e.target, pythonSnippet));
}

function openTransformModal(key) {
  renderModalTabs(key);
  renderModalContent(key);
  const modal = document.getElementById("transform-modal");
  modal.classList.remove("hidden");
  modal.classList.add("flex");
}

function closeTransformModal() {
  const modal = document.getElementById("transform-modal");
  modal.classList.add("hidden");
  modal.classList.remove("flex");
}
