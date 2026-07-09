import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";

// Mock specs mimicking weight_transforms.py registry
const mockSpecs = {
  linear: {
    label: "Linear",
    description: "Simple scaling f(x) = w * x",
    params: {
      w: { label: "Weight", default: 1.0, step: 0.1 },
    },
  },
  clamp: {
    label: "Clamp",
    description: "Capped values",
    params: {
      min_v: { label: "Min", default: null, step: 0.05, nullable: true },
      max_v: { label: "Max", default: 1.0, step: 0.05, nullable: true },
      scale: { label: "Scale", default: 1.0, step: 0.1 },
    },
  },
  passthrough: {
    label: "Passthrough",
    description: "No modification",
    params: {},
  },
};

// Mock public dataset configuration
const mockDatasetConfig = {
  name: "sample_courses",
  label: "Online Course Catalog",
  identifier_column: "course_title",
  channels: [
    {
      column: "avg_rating",
      label: "Average rating",
      color: "teal",
      normalize: "minmax",
      normalize_scope: "global",
      default_transform: { type: "linear", w: 1.0 },
    },
  ],
  filters: [
    { column: "category", label: "Category", type: "categorical" },
    { column: "price", label: "Price", type: "range" },
  ],
  filter_options: {
    category: ["Web Development", "Data Science"],
    price: [0.0, 100.0],
  },
  row_count: 6,
};

// Mock responsive compute data
const mockComputeSuccess = {
  leaderboard: [
    {
      id: "JavaScript Intermediate",
      rank: 1,
      total_score: 2.5,
      breakdown: [
        {
          column: "avg_rating",
          label: "Average rating",
          color: "teal",
          contribution: 2.5,
        },
      ],
    },
  ],
  watchlist: [
    {
      id: "JavaScript Intermediate",
      found: true,
      rank: 1,
      total_score: 2.5,
      breakdown: [],
    },
    {
      id: "Python for Everybody",
      found: false,
      rank: null,
      total_score: 0.0,
      breakdown: [],
    },
  ],
  formula: { latex: "y = w \\cdot x", python: "y = w * x" },
};

const makeDomFixture = (specs) => {
  return `
    <script id="transform-specs" type="application/json">${JSON.stringify(specs)}</script>
    <select id="dataset-select">
      <option value="sample_courses" selected>sample_courses</option>
      <option value="other_dataset">other_dataset</option>
    </select>
    <p id="dataset-tagline">loading...</p>
    <h2 id="channel-shaper-title">loading...</h2>
    <div id="filters-container"></div>
    <div id="channels"></div>
    <textarea id="watchlist-input"></textarea>
    <input id="top-x-input" type="number" value="20" />
    <button id="copy-latex-btn">copy LaTeX</button>
    <button id="copy-python-btn">copy Python</button>
    <button id="transform-reference-btn">Transform reference</button>
    <div id="transform-modal" class="hidden">
      <button id="close-modal-btn">close</button>
      <div id="modal-tabs"></div>
      <div id="modal-content"></div>
    </div>
    <div id="error-banner" class="hidden"></div>
    <table>
      <tbody id="leaderboard-body"></tbody>
    </table>
    <table>
      <tbody id="watchlist-body"></tbody>
    </table>
    <span id="score-summary">-</span>
    <code id="code-display"></code>
    <div id="latex-display"></div>
  `;
};

describe("Frontend Dashboard UI (JSDOM Environment)", () => {
  let originalFetch;
  let originalKatex;
  let fetchMock;
  let katexMock;

  beforeEach(async () => {
    vi.useFakeTimers();
    vi.resetModules();

    // Attach JSDOM environment fixtures
    document.body.innerHTML = makeDomFixture(mockSpecs);
    window.ACTIVE_DATASET = "sample_courses";

    // Clipboard and layout mocks
    Object.assign(navigator, {
      clipboard: {
        writeText: vi.fn().mockResolvedValue(undefined),
      },
    });

    // Mock global document features
    document.execCommand = vi.fn().mockReturnValue(true);

    // Mock Katex globally
    katexMock = { render: vi.fn() };
    originalKatex = global.katex;
    global.katex = katexMock;

    // Mock fetch with custom endpoints
    fetchMock = vi.fn().mockImplementation((url) => {
      if (url.includes("/api/transforms")) {
        return Promise.resolve({
          ok: true,
          json: () =>
            Promise.resolve({ specs: mockSpecs, dataset: mockDatasetConfig }),
        });
      }
      if (url.includes("/api/popular_values")) {
        return Promise.resolve({
          ok: true,
          json: () =>
            Promise.resolve({
              popular_values: [{ value: "Web Development", count: 4 }],
            }),
        });
      }
      if (url.includes("/api/compute")) {
        return Promise.resolve({
          ok: true,
          json: () => Promise.resolve(mockComputeSuccess),
        });
      }
      return Promise.reject(new Error("Unknown route: " + url));
    });
    originalFetch = global.fetch;
    global.fetch = fetchMock;

    // Static import for app.js; resetModules() handles re-evaluation automatically
    await import("../static/app.js");
    // Flush initial dynamic microtask loops (for fetches in init())
    await vi.runAllTimersAsync();
  });

  afterEach(() => {
    global.fetch = originalFetch;
    global.katex = originalKatex;
    vi.useRealTimers();
  });

  describe("init() bootstrapping", () => {
    it("sets tagline and shaper title correctly from dataset data", () => {
      const tagline = document.getElementById("dataset-tagline").textContent;
      const shaperTitle = document.getElementById(
        "channel-shaper-title",
      ).textContent;

      expect(tagline).toContain("Online Course Catalog");
      expect(tagline).toContain("6 rows");
      expect(tagline).toContain("1 channels");
      expect(shaperTitle).toBe("CHANNEL SHAPER — 1 OF 1 ACTIVE");
    });

    it("triggers an initial compute request on bootstrap", () => {
      const calls = fetchMock.mock.calls.map((c) => c[0]);
      const computeCalls = calls.filter((c) => c.includes("/api/compute"));
      expect(computeCalls.length).toBe(1);
    });

    it("changes location.search when selecting a different dataset", () => {
      const mockLocation = { search: "" };
      Object.defineProperty(window, "location", {
        value: mockLocation,
        writable: true,
      });

      const select = document.getElementById("dataset-select");
      select.value = "other_dataset";
      select.dispatchEvent(new Event("change"));

      expect(window.location.search).toBe("?dataset=other_dataset");
    });
  });

  describe("Filter Panel Integration", () => {
    it("renders both categorical elements and range sliders based on configuration", () => {
      const catSelect = document.getElementById("filter-category");
      expect(catSelect).not.toBeNull();
      expect(catSelect.options[1].value).toBe("Web Development");

      const minInput = document.getElementById("filter-price-min");
      const maxInput = document.getElementById("filter-price-max");
      expect(minInput.value).toBe("0");
      expect(maxInput.value).toBe("100");
    });

    it("fetches popular value tags and appends them to DOM", () => {
      const tagButton = document.querySelector(".popular-tag");
      expect(tagButton).not.toBeNull();
      expect(tagButton.textContent).toContain("Web Development");
      expect(tagButton.textContent).toContain("(4)");
    });

    it("clicks popular values tags to select option and triggers compute schedule", async () => {
      fetchMock.mockClear();
      const tagButton = document.querySelector(".popular-tag");
      tagButton.click();

      const catSelect = document.getElementById("filter-category");
      expect(catSelect.value).toBe("Web Development");

      // Advance timers past the 250ms debouncer threshold
      await vi.advanceTimersByTimeAsync(250);
      const computeCalls = fetchMock.mock.calls.filter((c) =>
        c[0].includes("/api/compute"),
      );
      expect(computeCalls.length).toBe(1);

      // Verify selected filter payload reflects changes
      const requestBody = JSON.parse(computeCalls[0][1].body);
      expect(requestBody.filters.category).toBe("Web Development");
    });
  });

  describe("Channel Configuration Interactions", () => {
    it("updates channel dropdown selections and schedules recomputation", async () => {
      fetchMock.mockClear();
      const select = document.querySelector('[data-transform-select="0"]');
      select.value = "clamp";
      select.dispatchEvent(new Event("change"));

      await vi.advanceTimersByTimeAsync(250);
      const computeCalls = fetchMock.mock.calls.filter((c) =>
        c[0].includes("/api/compute"),
      );
      expect(computeCalls.length).toBe(1);

      const requestBody = JSON.parse(computeCalls[0][1].body);
      expect(requestBody.pipeline[0].type).toBe("clamp");
    });

    it("renders nested parameter controllers on transform shift", () => {
      const select = document.querySelector('[data-transform-select="0"]');
      select.value = "clamp";
      select.dispatchEvent(new Event("change"));

      const maxSlider = document.querySelector('[data-param-slider="0-max_v"]');
      expect(maxSlider).not.toBeNull();
      expect(maxSlider.value).toBe("1");
    });

    it("binds slider movement directly to layout values and trigger timelines", async () => {
      const select = document.querySelector('[data-transform-select="0"]');
      select.value = "clamp";
      select.dispatchEvent(new Event("change"));

      fetchMock.mockClear();
      const maxSlider = document.querySelector('[data-param-slider="0-max_v"]');
      maxSlider.value = "0.75";
      maxSlider.dispatchEvent(new Event("input"));

      // Readout updates instantly in rendering loop
      const readout = document.querySelector('[data-param-readout="0-max_v"]');
      expect(readout.textContent).toBe("0.75");

      // Advance timers to confirm scheduled debounce
      await vi.advanceTimersByTimeAsync(250);
      const computeCalls = fetchMock.mock.calls.filter((c) =>
        c[0].includes("/api/compute"),
      );
      expect(computeCalls.length).toBe(1);
    });
  });

describe("Formula Rendering & Copy Actions", () => {
    it("renders rendered formula inside display utilizing katex block", () => {
      expect(katexMock.render).toHaveBeenCalledWith(
        "y = w \\cdot x",
        document.getElementById("latex-display"),
        expect.any(Object),
      );
    });

    it("handles katex rendering exception gracefully by outputting raw latex text", async () => {
      katexMock.render.mockImplementationOnce(() => {
        throw new Error("Katex failure");
      });

      // trigger manual rerender by calling custom mock compute payload
      const select = document.querySelector('[data-transform-select="0"]');
      select.dispatchEvent(new Event("change"));

      // Advance timers past the 250ms debouncer threshold to run compute()
      await vi.advanceTimersByTimeAsync(250);

      expect(document.getElementById("latex-display").textContent).toBe(
        "y = w \\cdot x",
      );
    });

    it("utilizes standard writeText for LaTeX copying interactions", async () => {
      const btn = document.getElementById("copy-latex-btn");
      btn.click();

      expect(navigator.clipboard.writeText).toHaveBeenCalledWith(
        "y = w \\cdot x",
      );

      // Flush promise microtasks so that the .then(done) callback executes
      await Promise.resolve();

      expect(btn.textContent).toBe("copied");

      // Check recovery timer restoring text
      await vi.advanceTimersByTimeAsync(1200);
      expect(btn.textContent).toBe("copy LaTeX");
    });

    it("utilizes standard writeText for Python copying interactions", async () => {
      const btn = document.getElementById("copy-python-btn");
      btn.click();

      expect(navigator.clipboard.writeText).toHaveBeenCalledWith(
        "y = w * x",
      );

      // Flush promise microtasks so that the .then(done) callback executes
      await Promise.resolve();

      expect(btn.textContent).toBe("copied");

      // Check recovery timer restoring text
      await vi.advanceTimersByTimeAsync(1200);
      expect(btn.textContent).toBe("copy Python");
    });

    it("reverts to browser legacy commands if navigator clipboard is unavailable", () => {
      const oldClipboard = navigator.clipboard;
      Object.defineProperty(navigator, "clipboard", {
        value: undefined,
        configurable: true,
      });

      const btn = document.getElementById("copy-latex-btn");
      btn.click();

      expect(document.execCommand).toHaveBeenCalledWith("copy");
      expect(btn.textContent).toBe("copied");

      Object.defineProperty(navigator, "clipboard", {
        value: oldClipboard,
        configurable: true,
      });
    });
  });

  describe("API Compute Responses (Success vs Failure Paths)", () => {
    it("renders rows correctly on empty responses with placeholders", async () => {
      fetchMock.mockImplementationOnce(() =>
        Promise.resolve({
          ok: true,
          json: () =>
            Promise.resolve({
              leaderboard: [],
              watchlist: [],
              formula: { latex: "", python: "" },
            }),
        }),
      );

      const input = document.getElementById("watchlist-input");
      input.dispatchEvent(new Event("input"));
      await vi.advanceTimersByTimeAsync(250);

      expect(document.getElementById("leaderboard-body").innerHTML).toContain(
        "No rows match",
      );
      expect(document.getElementById("score-summary").textContent).toBe("-");
    });

    it("displays error messages directly in layout banner when compute endpoint fails", async () => {
      fetchMock.mockImplementationOnce(() =>
        Promise.resolve({
          ok: true,
          json: () => Promise.resolve({ error: "Configuration failed" }),
        }),
      );

      const input = document.getElementById("watchlist-input");
      input.dispatchEvent(new Event("input"));
      await vi.advanceTimersByTimeAsync(250);

      const errorBanner = document.getElementById("error-banner");
      expect(errorBanner.classList.contains("hidden")).toBe(false);
      expect(errorBanner.textContent).toBe("Configuration failed");
    });

    it("displays generic fallback message on fetch rejection", async () => {
      fetchMock.mockImplementationOnce(() =>
        Promise.reject(new Error("Network disconnect")),
      );

      const input = document.getElementById("watchlist-input");
      input.dispatchEvent(new Event("input"));
      await vi.advanceTimersByTimeAsync(250);

      const errorBanner = document.getElementById("error-banner");
      expect(errorBanner.textContent).toBe(
        "Failed to compute: Network disconnect",
      );
    });
  });

  describe("Reference Modal and Curves Rendering", () => {
    it("opens transform details on trigger clicks", () => {
      const btn = document.getElementById("transform-reference-btn");
      const modal = document.getElementById("transform-modal");

      btn.click();
      expect(modal.classList.contains("hidden")).toBe(false);
      expect(modal.classList.contains("flex")).toBe(true);
    });

    it("closes transform details on clicking backdrop or close buttons", () => {
      const modal = document.getElementById("transform-modal");
      modal.classList.add("flex");
      modal.classList.remove("hidden");

      const closeBtn = document.getElementById("close-modal-btn");
      closeBtn.click();
      expect(modal.classList.contains("hidden")).toBe(true);

      modal.classList.add("flex");
      modal.classList.remove("hidden");

      // Click backdrop container directly
      modal.dispatchEvent(new MouseEvent("click", { bubbles: true }));
      expect(modal.classList.contains("hidden")).toBe(true);
    });

    it("renders custom mathematical curves as inline SVGs", () => {
      const btn = document.getElementById("transform-reference-btn");
      btn.click();

      const polyline = document.querySelector("polyline");
      expect(polyline).not.toBeNull();

      const pts = polyline.getAttribute("points").split(" ");
      // SVG curve calculation matches 41 distinct points
      expect(pts.length).toBe(41);
    });
  });
});
