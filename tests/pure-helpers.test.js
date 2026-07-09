// @vitest-environment node
import { describe, it, expect } from "vitest";
import {
  computeBreakdownWidths,
  buildPipelinePayload,
  parseWatchlist,
  sanitizeNumber,
} from "../static/app.js";

describe("computeBreakdownWidths", () => {
  it("sums to 100% for a normal breakdown with 2+ entries", () => {
    const breakdown = [
      { column: "a", color: "teal", contribution: 3 },
      { column: "b", color: "lime", contribution: 1 },
    ];
    const result = computeBreakdownWidths(breakdown);
    const total = result.reduce((acc, r) => acc + r.pct, 0);
    expect(total).toBeCloseTo(100, 5);
    expect(result[0].pct).toBeCloseTo(75, 5);
    expect(result[1].pct).toBeCloseTo(25, 5);
  });

  it("uses Math.abs for negative contributions", () => {
    const breakdown = [
      { column: "a", color: "teal", contribution: -3 },
      { column: "b", color: "lime", contribution: 1 },
    ];
    const result = computeBreakdownWidths(breakdown);
    expect(result[0].pct).toBeCloseTo(75, 5);
    expect(result[1].pct).toBeCloseTo(25, 5);
  });

  it("treats contribution: 0 as zero, not falsy-skip", () => {
    const breakdown = [
      { column: "a", color: "teal", contribution: 0 },
      { column: "b", color: "lime", contribution: 5 },
    ];
    const result = computeBreakdownWidths(breakdown);
    expect(result[0].pct).toBe(0);
    expect(result[1].pct).toBe(100);
  });

  it("treats contribution: undefined as zero via the || 0 fallback", () => {
    const breakdown = [
      { column: "a", color: "teal", contribution: undefined },
      { column: "b", color: "lime", contribution: 4 },
    ];
    const result = computeBreakdownWidths(breakdown);
    expect(result[0].pct).toBe(0);
    expect(result[1].pct).toBe(100);
  });

  it("empty array doesn't divide by zero (denominator falls back to 1)", () => {
    const result = computeBreakdownWidths([]);
    expect(result).toEqual([]);
  });

  it("single-entry breakdown is 100%", () => {
    const result = computeBreakdownWidths([
      { column: "a", color: "teal", contribution: 42 },
    ]);
    expect(result[0].pct).toBe(100);
  });
});

describe("buildPipelinePayload", () => {
  it("spreads params alongside type for each channel", () => {
    const state = [
      { column: "rating", type: "amplify", params: { power: 4, scale: 2.5 } },
      { column: "enrollments", type: "linear", params: { w: 1.0 } },
    ];
    const result = buildPipelinePayload(state);
    expect(result).toEqual([
      { type: "amplify", power: 4, scale: 2.5 },
      { type: "linear", w: 1.0 },
    ]);
  });

  it("empty channelState returns []", () => {
    expect(buildPipelinePayload([])).toEqual([]);
  });

  it("params spread after type: a params.type key would win", () => {
    const state = [{ column: "x", type: "linear", params: { type: "sneaky" } }];
    const result = buildPipelinePayload(state);
    // Source is `{ type: ch.type, ...ch.params }` -- params spread AFTER type,
    // so params.type overwrites ch.type in the resulting object.
    expect(result[0].type).toBe("sneaky");
  });
});

describe("parseWatchlist", () => {
  it("splits on \\n", () => {
    expect(parseWatchlist("a\nb\nc")).toEqual(["a", "b", "c"]);
  });

  it("splits on \\r\\n", () => {
    expect(parseWatchlist("a\r\nb\r\nc")).toEqual(["a", "b", "c"]);
  });

  it("trims whitespace per line", () => {
    expect(parseWatchlist("  a  \n  b  ")).toEqual(["a", "b"]);
  });

  it("drops blank lines", () => {
    expect(parseWatchlist("a\n\n\nb\n")).toEqual(["a", "b"]);
  });

  it("empty string returns []", () => {
    expect(parseWatchlist("")).toEqual([]);
  });

  it("single line without trailing newline still parses", () => {
    expect(parseWatchlist("solo")).toEqual(["solo"]);
  });
});

describe("sanitizeNumber", () => {
  it("valid numeric string returns a number", () => {
    expect(sanitizeNumber("42", 0)).toBe(42);
  });

  it("'abc' returns the fallback", () => {
    expect(sanitizeNumber("abc", 7)).toBe(7);
  });

  it("empty string returns the fallback", () => {
    expect(sanitizeNumber("", 7)).toBe(7);
  });

  it("null returns 0, not the fallback (Number(null) === 0, which is finite)", () => {
    expect(sanitizeNumber(null, 99)).toBe(0);
  });

  it("undefined returns the fallback (Number(undefined) is NaN)", () => {
    expect(sanitizeNumber(undefined, 99)).toBe(99);
  });

  it("Infinity returns the fallback", () => {
    expect(sanitizeNumber(Infinity, 5)).toBe(5);
  });

  it("NaN returns the fallback", () => {
    expect(sanitizeNumber(NaN, 5)).toBe(5);
  });
});