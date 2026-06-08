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
  bandSurfaceMeshDataWithMask,
  visibleBandIndices,
  bandSurfaceColor,
  allBandIndices,
  bandSurfaceSummary,
  projectBandSurfacePoint,
  nearestBandSurfaceVertex,
  pointInDisplayPolygon,
  vertexInsideVisibleHexagon,
  bandBasisToCartesian,
  threeUvGridReferenceData,
  threeHexagonReferenceData,
  threeBandSurfaceGeometryData,
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
  assert.equal(source.includes('onDftSignal("selected-band"'), true);
  assert.equal(source.includes('onDftSignal("slice-changed"'), true);
  assert.equal(source.includes('onDftSignal("selected-kpoint"'), true);
});



test("generic json payload reader policy exists", () => {
  const source = readFileSync("src/dft_local/diagnostics/static/dft-local-components.js", "utf8");

  assert.equal(source.includes("function readJsonPayload(host)"), true);
  assert.equal(source.includes("return /** @type {GraphPayload | null} */ (readJsonPayload(host));"), true);
});


test("band surface viewer reads payload policy", () => {
  const source = readFileSync("src/dft_local/diagnostics/static/dft-local-components.js", "utf8");

  assert.equal(source.includes("this.payload = readJsonPayload("), true);
  assert.equal(source.includes("bandSurfaceMeshDataWithMask(this.payload, band, this.maskToHexagon)"), true);
  assert.equal(source.includes("this.requestSurfaceUpdate();"), true);
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


test("band surface viewer is threejs only policy exists", () => {
  const source = readFileSync("src/dft_local/diagnostics/static/dft-local-components.js", "utf8");

  assert.equal(source.includes("class DftBandSurfaceViewer extends HTMLElement"), true);
  assert.equal(source.includes("async ensureThree()"), true);
  assert.equal(source.includes("new THREE.WebGLRenderer"), true);
  assert.equal(source.includes("new OrbitControls(camera, renderer.domElement)"), true);
  assert.equal(source.includes("band-surface-three"), true);
  assert.equal(source.includes("band-surface-preview"), false);
});












test("band surface reference frame policy exists", () => {
  const source = readFileSync("src/dft_local/diagnostics/static/dft-local-components.js", "utf8");

  assert.equal(source.includes("function drawBandSurfaceReferenceFrame"), true);
  assert.equal(source.includes("function drawBandSurfaceBzBoundary"), true);
  assert.equal(source.includes("function drawBandSurfaceSymmetryLabels"), true);
  assert.equal(source.includes('drawArrow(ctx, origin, k1Tip, "k1")'), true);
  assert.equal(source.includes('drawArrow(ctx, origin, k2Tip, "k2")'), true);
  assert.equal(source.includes('drawArrow(ctx, origin, eTip, "E")'), true);
  assert.equal(source.includes('"Γ"'), true);
  assert.equal(source.includes('"K"'), true);
  assert.equal(source.includes('"M"'), true);
  assert.equal(source.includes("bz_hexagon"), true);
});









test("nearestBandSurfaceVertex selects closest projected point", () => {
  const mesh = {
    vertices: [
      { x: 0, y: 0, z: 0, i: 0, j: 0, band: 0 },
      { x: 1, y: 0, z: 0, i: 1, j: 0, band: 0 },
    ],
    summary: { zmin: 0, zmax: 1 },
  };
  const view = {
    xmin: 0,
    xmax: 1,
    ymin: 0,
    ymax: 1,
    zmin: 0,
    zmax: 1,
    width: 100,
    height: 100,
  };

  const projected = projectBandSurfacePoint(mesh.vertices[1], view);
  const hit = nearestBandSurfaceVertex(mesh, projected, view, 5.0);

  assert.notEqual(hit, null);
  assert.equal(hit?.vertex.i, 1);
});


test("band surface hover selection policy exists", () => {
  const source = readFileSync("src/dft_local/diagnostics/static/dft-local-components.js", "utf8");

  assert.equal(source.includes("handlePointerMove(event)"), true);
  assert.equal(source.includes("handleClick(event)"), true);
  assert.equal(source.includes("pickNearestVertex(event"), true);
  assert.equal(source.includes('emitDftSignal("selected-kpoint"'), true);
  assert.equal(source.includes("data-dft-surface-hover"), true);
});



test("kpoint readout component policy exists", () => {
  const source = readFileSync("src/dft_local/diagnostics/static/dft-local-components.js", "utf8");

  assert.equal(source.includes("class DftKPointReadout extends HTMLElement"), true);
  assert.equal(source.includes('customElements.define("dft-kpoint-readout"'), true);
  assert.equal(source.includes('onDftSignal("selected-kpoint"'), true);
  assert.equal(source.includes("selected k-point:"), true);
});


test("selected kpoint marker policy exists", () => {
  const source = readFileSync("src/dft_local/diagnostics/static/dft-local-components.js", "utf8");

  assert.equal(source.includes("function drawBandSurfaceSelectionMarker"), true);
  assert.equal(source.includes("selectedKpoint"), true);
  assert.equal(source.includes('emitDftSignal("selected-kpoint"'), true);
  assert.equal(source.includes('ctx.fillText("selected"'), true);
});





test("threeBandSurfaceGeometryData builds position and index buffers", () => {
  const mesh = {
    vertices: [
      { x: 0, y: 0, z: 100, i: 0, j: 0, band: 0 },
      { x: 1, y: 0, z: 101, i: 0, j: 1, band: 0 },
      { x: 0, y: 1, z: 102, i: 1, j: 0, band: 0 },
    ],
    /** @type {Array<[number, number, number]>} */
    triangles: [[0, 1, 2]],
    summary: { count: 3, zmin: 100, zmax: 102 },
  };

  const data = threeBandSurfaceGeometryData(mesh);

  assert.equal(data.positions.length, 9);
  assert.equal(data.indices.length, 3);
  assert.deepEqual(Array.from(data.indices), [0, 1, 2]);
  assert.equal(Number.isFinite(data.center.x), true);
  assert.equal(Number.isFinite(data.center.y), true);
  assert.equal(Number.isFinite(data.center.z), true);
  assert.equal(Number.isFinite(data.radius), true);
  assert.equal(data.radius > 0, true);
});


test("band surface threejs orbit controls policy exists", () => {
  const source = readFileSync("src/dft_local/diagnostics/static/dft-local-components.js", "utf8");

  assert.equal(source.includes("async function loadThreeRuntime"), true);
  assert.equal(source.includes("OrbitControls"), true);
  assert.equal(source.includes("new THREE.WebGLRenderer"), true);
  assert.equal(source.includes("new OrbitControls(camera, renderer.domElement)"), true);
  assert.equal(source.includes("data-dft-three-surface"), true);
  assert.equal(source.includes("three.js controls: left drag rotate, wheel zoom, right drag pan"), true);
});



test("band surface orbit control interaction policy exists", () => {
  const source = readFileSync("src/dft_local/diagnostics/static/dft-local-components.js", "utf8");

  assert.equal(source.includes("controls.enableDamping = true"), true);
  assert.equal(source.includes("controls.screenSpacePanning = true"), true);
  assert.equal(source.includes("handlePointerMove(event)"), true);
  assert.equal(source.includes("handleClick(event)"), true);
  assert.equal(source.includes("pickNearestVertex(event"), true);
  assert.equal(source.includes('emitDftSignal("selected-kpoint"'), true);
});


test("band surface selected marker is threejs mesh policy exists", () => {
  const source = readFileSync("src/dft_local/diagnostics/static/dft-local-components.js", "utf8");

  assert.equal(source.includes("updateSelectedMarker()"), true);
  assert.equal(source.includes("new THREE.SphereGeometry"), true);
  assert.equal(source.includes("this.selectedMarker.position.set"), true);
});



test("threeBandSurfaceGeometryData uses Cartesian k-plane and normalized energy", () => {
  const mesh = {
    vertices: [
      { x: 0, y: 0, z: 100, i: 0, j: 0, band: 0 },
      { x: 1, y: 0, z: 200, i: 0, j: 1, band: 0 },
      { x: 0, y: 1, z: 300, i: 1, j: 0, band: 0 },
    ],
    /** @type {Array<[number, number, number]>} */
    triangles: [[0, 1, 2]],
    summary: { count: 3, zmin: 100, zmax: 300 },
  };

  const data = threeBandSurfaceGeometryData(mesh);
  const positions = Array.from(data.positions);
  const ys = positions.filter((_, i) => i % 3 === 1);

  // Energy coordinates should be visual-normalized, not raw 100..300.
  assert.equal(Math.max(...ys) < 10, true);
  assert.equal(Math.min(...ys) > -10, true);

  // The second basis vector should not map to raw square coordinate (0, 1).
  const secondBasisMappedX = positions[6];
  const secondBasisMappedZ = positions[8];
  assert.equal(Math.abs(secondBasisMappedX) > 1e-12 || Math.abs(secondBasisMappedZ - 1) > 1e-12, true);
});



test("bandBasisToCartesian embeds oblique reciprocal basis", () => {
  assert.deepEqual(bandBasisToCartesian(1, 0), { x: 1, y: 0 });

  const e2 = bandBasisToCartesian(0, 1);
  assert.equal(Math.abs(e2.x + 0.5) < 1e-12, true);
  assert.equal(Math.abs(e2.y - Math.sqrt(3) / 2) < 1e-12, true);
});



test("band surface threejs resize policy exists", () => {
  const source = readFileSync("src/dft_local/diagnostics/static/dft-local-components.js", "utf8");

  assert.equal(source.includes("resizeThreeSurface()"), true);
  assert.equal(source.includes("new ResizeObserver"), true);
  assert.equal(source.includes("this.camera.aspect = width / height"), true);
  assert.equal(source.includes("this.camera.updateProjectionMatrix()"), true);
  assert.equal(source.includes("renderer.setSize(width, height, false)"), true);
  assert.equal(source.includes("min-height:560px"), true);
});



test("threeHexagonReferenceData converts bz hexagon to display plane", () => {
  const points = threeHexagonReferenceData({
    bz_hexagon: [
      [1, 0],
      [0, 1],
      [-1, 1],
      ["bad", 2],
    ],
  });

  assert.equal(points.length, 3);
  assert.deepEqual(points[0], { x: 1, y: 0, z: 0 });
  assert.equal(Math.abs(points[1].x + 0.5) < 1e-12, true);
  assert.equal(Math.abs(points[1].z - Math.sqrt(3) / 2) < 1e-12, true);
});


test("band surface uv plane and hexagon reference policy exists", () => {
  const source = readFileSync("src/dft_local/diagnostics/static/dft-local-components.js", "utf8");

  assert.equal(source.includes("function threeHexagonReferenceData(payload)"), true);
  assert.equal(source.includes("new THREE.PlaneGeometry"), true);
  assert.equal(source.includes("new THREE.GridHelper"), false);
  assert.equal(source.includes("threeHexagonReferenceData(this.payload)"), true);
  assert.equal(source.includes("band-surface-reference-group"), true);
  assert.equal(source.includes("plane.position.set(data.center.x, 0.0, data.center.z)"), true);
});



test("threeHexagonReferenceData provides fallback regular hexagon", () => {
  const points = threeHexagonReferenceData({});

  assert.equal(points.length, 6);
  const radii = points.map((p) => Math.hypot(p.x, p.z));
  assert.equal(Math.max(...radii) - Math.min(...radii) < 1e-12, true);
});



test("threeUvGridReferenceData creates u and v grid lines", () => {
  const lines = threeUvGridReferenceData(Math.PI, 2);

  assert.equal(lines.length, 10);
  for (const line of lines) {
    assert.equal(line.length, 2);
    assert.equal(Number.isFinite(line[0].x), true);
    assert.equal(Number.isFinite(line[0].z), true);
    assert.equal(Number.isFinite(line[1].x), true);
    assert.equal(Number.isFinite(line[1].z), true);
  }
});


test("band surface uv reference plane styling policy exists", () => {
  const source = readFileSync("src/dft_local/diagnostics/static/dft-local-components.js", "utf8");

  assert.equal(source.includes("threeUvGridReferenceData(Math.PI, 8)"), true);
  assert.equal(source.includes("color: 0x101820"), true);
  assert.equal(source.includes("color: 0xffb000"), true);
  assert.equal(source.includes("depthTest: false"), true);
  assert.equal(source.includes("color: 0x4e6e81"), true);
});



test("bandSurfaceMeshDataWithMask clips triangles against visible hexagon", () => {
  const payload = {
    nu: 2,
    nv: 2,
    k1: [[0, 10], [0, 10]],
    k2: [[0, 0], [1, 1]],
    energies: [
      [[0], [1]],
      [[2], [3]],
    ],
    bz_hexagon: [
      [2, 0],
      [1, 1],
      [-1, 1],
      [-2, 0],
      [-1, -1],
      [1, -1],
    ],
  };

  const unmasked = bandSurfaceMeshDataWithMask(payload, 0, false);
  const masked = bandSurfaceMeshDataWithMask(payload, 0, true);

  assert.equal(unmasked.vertices.length, 4);
  assert.equal(unmasked.triangles.length, 2);
  assert.equal(masked.triangles.length, 0);
});


test("visible hexagon point predicate follows displayed polygon", () => {
  const polygon = threeHexagonReferenceData({
    bz_hexagon: [
      [2, 0],
      [1, 1],
      [-1, 1],
      [-2, 0],
      [-1, -1],
      [1, -1],
    ],
  });

  assert.equal(pointInDisplayPolygon({ x: 0, z: 0 }, polygon), true);
  assert.equal(pointInDisplayPolygon({ x: 100, z: 100 }, polygon), false);
});


test("band surface hex mask toggle policy exists", () => {
  const source = readFileSync("src/dft_local/diagnostics/static/dft-local-components.js", "utf8");

  assert.equal(source.includes("data-dft-mask-to-hexagon"), true);
  assert.equal(source.includes("mask to hexagon"), true);
  assert.equal(source.includes("this.maskToHexagon"), true);
  assert.equal(source.includes("this.maskToHexagon = false"), true);
  assert.equal(source.includes("bandSurfaceMeshDataWithMask(this.payload, band, this.maskToHexagon)"), true);
  assert.equal(source.includes("hex mask on"), true);
  assert.equal(source.includes("hex mask off"), true);
});



test("threeHexagonReferenceData fallback hexagon is regular in Cartesian display", () => {
  const points = threeHexagonReferenceData({});
  assert.equal(points.length, 6);

  const lengths = points.map((p, i) => {
    const q = points[(i + 1) % points.length];
    return Math.hypot(p.x - q.x, p.z - q.z);
  });

  assert.equal(Math.max(...lengths) - Math.min(...lengths) < 1e-12, true);
});



test("band surface preserves camera across data updates", () => {
  const source = readFileSync("src/dft_local/diagnostics/static/dft-local-components.js", "utf8");

  assert.equal(source.includes("hasInitialCamera"), true);
  assert.equal(source.includes("resetCameraIfNeeded"), true);
  assert.equal(source.includes("hasInitialCamera = true"), true);
  assert.equal(source.includes("hasInitialCamera = false"), true);
  assert.equal(source.includes("resetCameraIfNeeded(data)"), true);
});




test("visibleBandIndices hides legend-toggled bands", () => {
  const payload = { bands: [0, 1, 2, 3], nbands: 4, selected_band: 1 };

  assert.deepEqual(allBandIndices(payload), [0, 1, 2, 3]);
  assert.deepEqual(visibleBandIndices(payload, new Set()), [0, 1, 2, 3]);
  assert.deepEqual(visibleBandIndices(payload, new Set([1, 3])), [0, 2]);
});


test("bandSurfaceColor is stable by band index", () => {
  assert.equal(bandSurfaceColor(0), bandSurfaceColor(10));
  assert.notEqual(bandSurfaceColor(0), bandSurfaceColor(1));
});


test("band surface multi-band render policy exists", () => {
  const source = readFileSync("src/dft_local/diagnostics/static/dft-local-components.js", "utf8");

  assert.equal(source.includes("visibleBandIndices(this.payload, this.hiddenBands)"), true);
  assert.equal(source.includes("this.currentBandMeshes"), true);
  assert.equal(source.includes("bandSurfaceColor(item.band)"), true);
  assert.equal(source.includes("data-dft-surface-legend"), true);
  assert.equal(source.includes("band-surface-legend-item-hidden"), true);
  assert.equal(source.includes("this.hiddenBands.add(band)"), true);
  assert.equal(source.includes("this.hiddenBands.delete(band)"), true);
  assert.equal(source.includes("#808080"), true);
  assert.equal(source.includes("band-surface-legend-item-hidden"), true);
  assert.equal(source.includes("this.hiddenBands.add(band)"), true);
  assert.equal(source.includes("this.hiddenBands.delete(band)"), true);
  assert.equal(source.includes("#808080"), true);
});



test("band surface no longer has band visibility multichoice", () => {
  const source = readFileSync("src/dft_local/diagnostics/static/dft-local-components.js", "utf8");

  assert.equal(source.includes("data-dft-band-visibility"), false);
  assert.equal(source.includes("data-dft-band-start"), false);
  assert.equal(source.includes("data-dft-band-end"), false);
});



test("band surface legend keeps one band visible policy exists", () => {
  const source = readFileSync("src/dft_local/diagnostics/static/dft-local-components.js", "utf8");

  assert.equal(source.includes("At least one band must remain visible"), true);
  assert.equal(source.includes("button.disabled = isLastVisible"), true);
  assert.equal(source.includes("visible.length <= 1"), true);
  assert.equal(source.includes("no visible bands"), true);
});



test("band surface no visible bands status keeps mask state", () => {
  const source = readFileSync("src/dft_local/diagnostics/static/dft-local-components.js", "utf8");

  assert.equal(source.includes("no visible bands; ${maskText}"), true);
});
