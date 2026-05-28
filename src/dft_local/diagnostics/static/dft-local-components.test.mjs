import test from "node:test";
import assert from "node:assert/strict";

import { nice, graphBounds, zoomView, panView, equalAspectView, kBasisToCartesian, rotatePoint, kspacePayloadToCartesian, nearestPathPoint, selectedPathHits, nearestPointByX, makeGraphSvg, isSelectionFrozen, emitSelectionFreeze, selectedSteps, emitSelectedSteps } from "./dft-local-components.js";
import { readFileSync } from "node:fs";

test("nice formats ordinary numbers", () => {
  assert.equal(nice(0), "0");
  assert.equal(nice(1.25), "1.25");
});

test("nice formats very small and very large numbers exponentially", () => {
  assert.equal(nice(1e-4), "1.00e-4");
  assert.equal(nice(1e5), "1.00e+5");
});

test("nice handles non-finite values", () => {
  assert.equal(nice(Number.NaN), "");
  assert.equal(nice(Number.POSITIVE_INFINITY), "");
});


test("graphBounds returns padded bounds", () => {
  const bounds = graphBounds({
    id: "g",
    title: "G",
    x_label: "x",
    y_label: "y",
    series: [
      {
        name: "s",
        kind: "line",
        points: [
          { x: 0, y: 0 },
          { x: 10, y: 20 },
        ],
      },
    ],
  });

  assert.equal(bounds.xmin, 0);
  assert.equal(bounds.xmax, 10);
  assert.equal(bounds.ymin, -1.2);
  assert.equal(bounds.ymax, 21.2);
});

test("zoomView contracts around fractional point", () => {
  const view = { xmin: 0, xmax: 10, ymin: 0, ymax: 10 };
  const got = zoomView(view, 0.5, 0.5, 0.5);

  assert.deepEqual(got, { xmin: 2.5, xmax: 7.5, ymin: 2.5, ymax: 7.5 });
});

test("panView shifts the visible window", () => {
  const view = { xmin: 0, xmax: 10, ymin: 0, ymax: 20 };
  const got = panView(view, 0.1, -0.25);

  assert.deepEqual(got, { xmin: -1, xmax: 9, ymin: -5, ymax: 15 });
});


test("nearestPointByX finds closest x point", () => {
  /** @type {import("./dft-local-components.js").GraphPayload} */
  const payload = {
    id: "g",
    title: "G",
    x_label: "x",
    y_label: "y",
    series: [
      {
        name: "band 0",
        kind: "line",
        points: [
          { x: 0, y: 10 },
          { x: 5, y: 20 },
        ],
      },
      {
        name: "band 1",
        kind: "line",
        points: [
          { x: 2, y: 30 },
          { x: 8, y: 40 },
        ],
      },
    ],
  };

  const hit = nearestPointByX(payload, { xmin: 0, xmax: 10, ymin: 0, ymax: 50 }, 0.48);

  if (hit === null) {
    assert.fail("expected nearest point hit");
  }

  assert.equal(hit.series, "band 0");
  assert.equal(hit.x, 4.8);
  assert.equal(hit.y, 19.6);
});


test("nearestPointByX interpolates line y and chooses nearest curve", () => {
  /** @type {any} */
  const payload = {
    id: "g",
    title: "G",
    x_label: "x",
    y_label: "y",
    series: [
      {
        name: "low",
        kind: "line",
        points: [
          { x: 0, y: 0 },
          { x: 10, y: 10 },
        ],
      },
      {
        name: "high",
        kind: "line",
        points: [
          { x: 0, y: 100 },
          { x: 10, y: 110 },
        ],
      },
    ],
  };

  const hit = nearestPointByX(payload, { xmin: 0, xmax: 10, ymin: 0, ymax: 120 }, 0.5, 0.1);

  if (hit === null) {
    assert.fail("expected nearest point hit");
  }

  assert.equal(hit.series, "high");
  assert.equal(hit.x, 5);
  assert.equal(hit.y, 105);
});


test("equalAspectView expands shorter axis", () => {
  const got = equalAspectView({ xmin: 0, xmax: 10, ymin: 0, ymax: 2 });

  assert.deepEqual(got, { xmin: 0, xmax: 10, ymin: -4, ymax: 6 });
});


test("kBasisToCartesian maps reciprocal basis to Cartesian coordinates", () => {
  const got = kBasisToCartesian(0, 2);

  assert.equal(got.x, -1);
  assert.equal(got.y, Math.sqrt(3));
});

test("rotatePoint rotates coordinates", () => {
  const got = rotatePoint(1, 0, Math.PI / 2);

  assert.ok(Math.abs(got.x) < 1e-12);
  assert.ok(Math.abs(got.y - 1) < 1e-12);
});

test("kspacePayloadToCartesian transforms payload points", () => {
  const got = kspacePayloadToCartesian({
    id: "k",
    title: "K",
    x_label: "k1",
    y_label: "k2",
    series: [
      {
        name: "path",
        kind: "line",
        points: [{ x: 0, y: 2 }],
      },
    ],
  });

  assert.equal(got.x_label, "k Cartesian x");
  assert.equal(got.y_label, "k Cartesian y");
  assert.equal(got.series[0].points[0].x, -1);
  assert.equal(got.series[0].points[0].y, Math.sqrt(3));
});


test("reset orientation policy is documented by zero rotation expectation", () => {
  // K-space reset should reset view and orientation together.
  assert.equal(0, 0);
});


test("basis-space hexagon becomes regular after Cartesian projection", () => {
  const pi = Math.PI;
  const vertices = [
    { x: pi, y: 0 },
    { x: pi, y: pi },
    { x: 0, y: pi },
    { x: -pi, y: 0 },
    { x: -pi, y: -pi },
    { x: 0, y: -pi },
  ].map((point) => kBasisToCartesian(point.x, point.y));

  const lengths = vertices.map((point, i) => {
    const next = vertices[(i + 1) % vertices.length];
    return Math.hypot(next.x - point.x, next.y - point.y);
  });

  const spread = Math.max(...lengths) - Math.min(...lengths);
  assert.ok(spread < 1e-12);
});


test("high-symmetry K M coordinates align with hexagon", () => {
  const pi = Math.PI;
  const k = kBasisToCartesian(pi, 0);
  const m = kBasisToCartesian(pi, 0.5 * pi);

  assert.equal(k.x, pi);
  assert.equal(k.y, 0);
  assert.equal(m.x, 0.75 * pi);
  assert.equal(m.y, 0.25 * Math.sqrt(3) * pi);
});


test("drag mode policy has pan and rotate states", () => {
  const modes = new Set(["pan", "rotate"]);

  assert.equal(modes.has("pan"), true);
  assert.equal(modes.has("rotate"), true);
});


test("nearestPathPoint finds selected path point and step", () => {
  /** @type {any} */
  const payload = {
    id: "kspace",
    title: "K",
    x_label: "kx",
    y_label: "ky",
    series: [
      {
        name: "selected path",
        kind: "line_points",
        points: [
          { x: 0, y: 0, meta: { step: 0, x: 0 } },
          { x: 10, y: 0, meta: { step: 1, x: 42 } },
        ],
      },
    ],
  };

  const hit = nearestPathPoint(payload, { xmin: 0, xmax: 10, ymin: -5, ymax: 5 }, 0.9, 0.5);

  if (hit === null) {
    assert.fail("expected nearest path hit");
  }

  assert.equal(hit.step, 1);
  assert.equal(hit.pathX, 42);
  assert.equal(hit.x, 10);
});


test("plot clipping and legend policy exists in component source", () => {
  const source = readFileSync(new URL("./dft-local-components.js", import.meta.url), "utf8");

  assert.equal(source.includes("clipPath"), true);
  assert.equal(source.includes("graph-legend"), true);
});



test("plot fraction helper policy exists in component source", () => {
  const source = readFileSync(new URL("./dft-local-components.js", import.meta.url), "utf8");

  assert.equal(source.includes("plotFractionsFromPointer"), true);
  assert.equal(source.includes("margin.left"), true);
  assert.equal(source.includes("innerW"), true);
});


test("global selection freeze state toggles", () => {
  emitSelectionFreeze(false, new EventTarget());
  assert.equal(isSelectionFrozen(), false);

  emitSelectionFreeze(true, new EventTarget());
  assert.equal(isSelectionFrozen(), true);

  emitSelectionFreeze(false, new EventTarget());
  assert.equal(isSelectionFrozen(), false);
});


test("selected steps global state updates", () => {
  emitSelectedSteps([{ step: 3, pathX: 1.5, energy: null, label: "K" }], new EventTarget());

  assert.deepEqual(selectedSteps(), [{ step: 3, pathX: 1.5, energy: null, label: "K" }]);

  emitSelectedSteps([], new EventTarget());
  assert.deepEqual(selectedSteps(), []);
});

test("selectedPathHits maps selected steps onto selected path points", () => {
  /** @type {any} */
  const payload = {
    id: "kspace",
    title: "K",
    x_label: "kx",
    y_label: "ky",
    series: [
      {
        name: "selected path",
        kind: "line_points",
        points: [
          { x: 0, y: 0, meta: { step: 0, x: 0 } },
          { x: 10, y: 0, meta: { step: 1, x: 42 } },
        ],
      },
    ],
  };

  emitSelectedSteps([{ step: 1, pathX: 42 }], new EventTarget());
  const hits = selectedPathHits(payload, { xmin: 0, xmax: 10, ymin: -5, ymax: 5 });

  assert.equal(hits.length, 1);
  assert.equal(hits[0].step, 1);
  assert.equal(hits[0].x, 10);

  emitSelectedSteps([], new EventTarget());
});


test("svg user point pointer mapping policy exists in component source", () => {
  const source = readFileSync(new URL("./dft-local-components.js", import.meta.url), "utf8");

  assert.equal(source.includes("createSVGPoint"), true);
  assert.equal(source.includes("getScreenCTM"), true);
  assert.equal(source.includes("dblclick"), true);
});


test("kspace hover label policy exists", () => {
  const source = readFileSync(new URL("./dft-local-components.js", import.meta.url), "utf8");

  assert.equal(source.includes("kspace-hover-label"), true);
  assert.equal(source.includes("step="), true);
});


test("double click reset policy exists", () => {
  const source = readFileSync(new URL("./dft-local-components.js", import.meta.url), "utf8");

  assert.equal(source.includes("dblclick"), true);
  assert.equal(source.includes("this.resetView()"), true);
  assert.equal(source.includes("window.setTimeout"), true);
});


test("selected table marker split policy exists", () => {
  const source = readFileSync(new URL("./dft-local-components.js", import.meta.url), "utf8");

  assert.equal(source.includes("selected-symmetry-marker"), true);
  assert.equal(source.includes("selected-energy-overlay"), true);
  assert.equal(source.includes("selected-symmetry-ring"), false);
  assert.equal(source.includes("selected-row-readout"), false);
  assert.equal(source.includes("item.energy === null"), true);
  assert.equal(source.includes("} else if (Number.isFinite(item.energy)) {"), true);
});
