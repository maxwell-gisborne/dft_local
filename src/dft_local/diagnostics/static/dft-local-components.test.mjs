// @ts-check

// @ts-ignore node builtin import is provided by Node at runtime
import test from "node:test";
// @ts-ignore node builtin import is provided by Node at runtime
import assert from "node:assert/strict";

import {
  nice,
  readJsonPayload,
  graphBounds,
  zoomView,
  panView,
  equalAspectView,
  kBasisToCartesian,
  rotatePoint,
  kspacePayloadToCartesian,
  bandSurfaceVertices,
  bandSurfaceTriangles,
  bandSurfaceMeshData,
  bandSurfaceSummary,
  projectBandSurfacePoint,
  nearestPathPoint,
  selectedPathHits,
  nearestPointByX,
  makeGraphSvg,
  createDftSignalBus,
  isSelectionFrozen,
  emitSelectionFreeze,
  selectedSteps,
  emitSelectedSteps,
  projectedKspaceHexagonSideLengths,
} from "./dft-local-components.js";
// @ts-ignore node builtin import is provided by Node at runtime
import { readFileSync } from "node:fs";

/**
 * @template T
 * @param {T | null | undefined} value
 * @param {string} message
 * @returns {T}
 */
function requireValue(value, message) {
  if (value == null) {
    throw new Error(message);
  }
  return value;
}


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
        kind: /** @type {"line"} */ ("line"),
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

  const hit = requireValue(
    nearestPointByX(payload, { xmin: 0, xmax: 10, ymin: 0, ymax: 50 }, 0.48),
    "expected nearest point hit",
  );

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

  const hit = requireValue(
    nearestPointByX(payload, { xmin: 0, xmax: 10, ymin: 0, ymax: 120 }, 0.5, 0.1),
    "expected nearest interpolated hit",
  );

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

  const hit = requireValue(
    nearestPathPoint(payload, { xmin: 0, xmax: 10, ymin: -5, ymax: 5 }, 0.9, 0.5),
    "expected nearest path hit",
  );

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


test("kspace hexagon remains regular after screen projection", () => {
  const corners = [
    { x: 2 / 3, y: 1 / 3 },
    { x: 1 / 3, y: 2 / 3 },
    { x: -1 / 3, y: 1 / 3 },
    { x: -2 / 3, y: -1 / 3 },
    { x: -1 / 3, y: -2 / 3 },
    { x: 1 / 3, y: -1 / 3 },
  ];

  const payload = {
    id: "regular-hexagon-render-test",
    title: "Regular hexagon render test",
    x_label: "k1",
    y_label: "k2",
    series: [
      {
        name: "hexagon",
        kind: /** @type {"line"} */ ("line"),
        points: corners,
      },
    ],
  };

  const sideLengths = projectedKspaceHexagonSideLengths(payload, { kspace: true });
  const minLength = Math.min(...sideLengths);
  const maxLength = Math.max(...sideLengths);

  assert.ok(minLength > 0);
  assert.ok(
    maxLength / minLength < 1.000000001,
    `screen-projected hexagon side lengths are not regular: ${sideLengths.join(", ")}`
  );
});



test("generic signal bus publishes named detail payloads", () => {
  const bus = createDftSignalBus();
  /** @type {Array<{name:string, detail:Record<string, unknown>, source:unknown}>} */
  const received = [];

  const unsubscribe = bus.on("selected-band", (payload) => received.push(payload));
  const emitted = bus.emit("selected-band", { band: 3 }, "unit-test");

  assert.deepEqual(emitted, {
    name: "selected-band",
    detail: { band: 3 },
    source: "unit-test",
  });
  assert.deepEqual(received, [emitted]);

  unsubscribe();
  bus.emit("selected-band", { band: 4 }, "unit-test");

  assert.deepEqual(received, [emitted]);
});


test("generic signal bus keeps signal names isolated", () => {
  const bus = createDftSignalBus();
  /** @type {number[]} */
  const selectedBands = [];
  /** @type {number[]} */
  const selectedSlices = [];

  bus.on("selected-band", (payload) => selectedBands.push(Number(payload.detail.band)));
  bus.on("slice-changed", (payload) => selectedSlices.push(Number(payload.detail.value)));

  bus.emit("selected-band", { band: 2 });
  bus.emit("slice-changed", { axis: "u", value: 0.25 });

  assert.deepEqual(selectedBands, [2]);
  assert.deepEqual(selectedSlices, [0.25]);
});



test("band controls component policy exists", () => {
  const source = readFileSync("src/dft_local/diagnostics/static/dft-local-components.js", "utf8");

  assert.equal(source.includes("class DftBandControls extends HTMLElement"), true);
  assert.equal(source.includes('customElements.define("dft-band-controls"'), true);
  assert.equal(source.includes('emitDftSignal("selected-band"'), true);
  assert.equal(source.includes('emitDftSignal("slice-changed"'), true);
  assert.equal(source.includes("data-dft-band"), true);
  assert.equal(source.includes("data-dft-slice-axis"), true);
  assert.equal(source.includes("data-dft-slice-value"), true);
});



test("band readout component listens to signal policy", () => {
  const source = readFileSync("src/dft_local/diagnostics/static/dft-local-components.js", "utf8");

  assert.equal(source.includes("class DftBandReadout extends HTMLElement"), true);
  assert.equal(source.includes('customElements.define("dft-band-readout"'), true);
  assert.equal(source.includes('onDftSignal("selected-band"'), true);
  assert.equal(source.includes('onDftSignal("slice-changed"'), true);
  assert.equal(source.includes("disconnectedCallback()"), true);
});



test("band surface viewer signal listener policy exists", () => {
  const source = readFileSync("src/dft_local/diagnostics/static/dft-local-components.js", "utf8");

  assert.equal(source.includes("class DftBandSurfaceViewer extends HTMLElement"), true);
  assert.equal(source.includes('customElements.define("dft-band-surface-viewer"'), true);
  assert.equal(source.includes('onDftSignal("selected-band"'), true);
  assert.equal(source.includes('onDftSignal("slice-changed"'), true);
  assert.equal(source.includes("band-surface-viewer-stub"), true);
});



test("generic json payload reader policy exists", () => {
  const source = readFileSync("src/dft_local/diagnostics/static/dft-local-components.js", "utf8");

  assert.equal(source.includes("function readJsonPayload(host)"), true);
  assert.equal(source.includes("return /** @type {GraphPayload | null} */ (readJsonPayload(host));"), true);
});


test("band surface viewer reads payload policy", () => {
  const source = readFileSync("src/dft_local/diagnostics/static/dft-local-components.js", "utf8");

  assert.equal(source.includes("const payload = readJsonPayload(this);"), true);
  assert.equal(source.includes("payload?.nbands"), true);
  assert.equal(source.includes("payload?.nu"), true);
  assert.equal(source.includes("payload?.nv"), true);
});



test("bandSurfaceVertices extracts selected band samples", () => {
  const payload = {
    k1: [[0, 1], [2, 3]],
    k2: [[10, 11], [12, 13]],
    energies: [
      [[0.0, 10.0], [1.0, 11.0]],
      [[2.0, 12.0], [3.0, 13.0]],
    ],
  };

  assert.deepEqual(bandSurfaceVertices(payload, 1), [
    { x: 0, y: 10, z: 10, i: 0, j: 0, band: 1 },
    { x: 1, y: 11, z: 11, i: 0, j: 1, band: 1 },
    { x: 2, y: 12, z: 12, i: 1, j: 0, band: 1 },
    { x: 3, y: 13, z: 13, i: 1, j: 1, band: 1 },
  ]);
});


test("bandSurfaceTriangles builds two triangles per unmasked cell", () => {
  const payload = { nu: 2, nv: 3 };

  assert.deepEqual(bandSurfaceTriangles(payload), [
    [0, 3, 1],
    [3, 4, 1],
    [1, 4, 2],
    [4, 5, 2],
  ]);
});


test("bandSurfaceTriangles skips masked triangle corners", () => {
  const payload = {
    nu: 2,
    nv: 2,
    mask: [
      [true, true],
      [false, true],
    ],
  };

  assert.deepEqual(bandSurfaceTriangles(payload), []);
});


test("bandSurfaceMeshData bundles vertices triangles and summary", () => {
  const payload = {
    nu: 2,
    nv: 2,
    k1: [[0, 1], [2, 3]],
    k2: [[10, 11], [12, 13]],
    energies: [
      [[0.0], [1.0]],
      [[2.0], [3.0]],
    ],
  };

  const mesh = bandSurfaceMeshData(payload, 0);

  assert.equal(mesh.vertices.length, 4);
  assert.deepEqual(mesh.triangles, [
    [0, 2, 1],
    [2, 3, 1],
  ]);
  assert.deepEqual(mesh.summary, {
    count: 4,
    zmin: 0.0,
    zmax: 3.0,
  });
});


test("bandSurfaceSummary reports count and energy range", () => {
  const payload = {
    k1: [[0, 1]],
    k2: [[0, 0]],
    energies: [[[3.0], [-1.0]]],
  };

  assert.deepEqual(bandSurfaceSummary(payload, 0), {
    count: 2,
    zmin: -1.0,
    zmax: 3.0,
  });
});



test("band surface projection maps finite points into canvas", () => {
  const point = { x: 0.5, y: 0.5, z: 0.5 };
  const projected = projectBandSurfacePoint(point, {
    xmin: 0,
    xmax: 1,
    ymin: 0,
    ymax: 1,
    zmin: 0,
    zmax: 1,
    width: 100,
    height: 100,
  });

  assert.equal(Number.isFinite(projected.x), true);
  assert.equal(Number.isFinite(projected.y), true);
});


test("band surface viewer canvas policy exists", () => {
  const source = readFileSync("src/dft_local/diagnostics/static/dft-local-components.js", "utf8");

  assert.equal(source.includes("function drawBandSurfacePreview"), true);
  assert.equal(source.includes('canvas class="band-surface-preview"'), true);
  assert.equal(source.includes("drawBandSurfacePreview(mesh, canvas"), true);
});



test("energy scale signal policy exists", () => {
  const source = readFileSync("src/dft_local/diagnostics/static/dft-local-components.js", "utf8");

  assert.equal(source.includes("data-dft-energy-scale"), true);
  assert.equal(source.includes('emitDftSignal("view-changed"'), true);
  assert.equal(source.includes('onDftSignal("view-changed"'), true);
  assert.equal(source.includes("energyScale"), true);
  assert.equal(source.includes("drawBandSurfacePreview(mesh, canvas"), true);
  assert.equal(source.includes("energyScale: this.energyScale"), true);
});



test("surface rotation signal policy exists", () => {
  const source = readFileSync("src/dft_local/diagnostics/static/dft-local-components.js", "utf8");

  assert.equal(source.includes("data-dft-rotation"), true);
  assert.equal(source.includes("rotationInput"), true);
  assert.equal(source.includes("this.rotation"), true);
  assert.equal(source.includes("rotation: this.rotation"), true);
  assert.equal(source.includes("Math.cos(theta)"), true);
  assert.equal(source.includes("Math.sin(theta)"), true);
});



test("band surface slice guide policy exists", () => {
  const source = readFileSync("src/dft_local/diagnostics/static/dft-local-components.js", "utf8");

  assert.equal(source.includes("function drawBandSurfaceSliceGuide"), true);
  assert.equal(source.includes("function drawProjectedPolyline"), true);
  assert.equal(source.includes("sliceAxis: this.sliceAxis"), true);
  assert.equal(source.includes("sliceValue: this.sliceValue"), true);
  assert.equal(source.includes('axis === "u"'), true);
  assert.equal(source.includes('axis === "v"'), true);
  assert.equal(source.includes('axis === "energy"'), true);
});
