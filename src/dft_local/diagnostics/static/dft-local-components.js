// @ts-check

/**
 * @typedef {{x:number, y:number, entity_id?:string|null, label?:string, meta?:Record<string, unknown>}} GraphPoint
 * @typedef {{name:string, kind:"line"|"points"|"line_points", points:GraphPoint[]}} GraphSeries
 * @typedef {{id:string, title:string, x_label:string, y_label:string, series:GraphSeries[]}} GraphPayload
 * @typedef {Record<string, unknown>} JsonPayload
 * @typedef {{xmin:number, xmax:number, ymin:number, ymax:number}} GraphView
 * @typedef {{series:string, x:number, y:number, sx:number, sy:number, step?:number|null, pathX?:number|null, label?:string|null}} CursorHit
 * @typedef {{name:string, detail:Record<string, unknown>, source:unknown}} DftSignalPayload
 * @typedef {(payload:DftSignalPayload) => void} DftSignalListener
 */

/**
 * @param {number} value
 * @returns {string}
 */
function nice(value) {
  if (!Number.isFinite(value)) return "";
  if (value === 0) return "0";
  const a = Math.abs(value);
  if (a < 1e-3 || a >= 1e4) return value.toExponential(2);
  return value.toPrecision(4).replace(/\.?0+$/, "");
}

const DFT_SIGNAL_EVENT = "dft-local-signal";

/**
 * @param {EventTarget | null} target
 */
function createDftSignalBus(target = null) {
  /** @type {Map<string, Set<DftSignalListener>>} */
  const listeners = new Map();

  /**
   * @param {string} name
   * @returns {Set<DftSignalListener>}
   */
  function listenersFor(name) {
    const key = String(name);
    if (!listeners.has(key)) listeners.set(key, new Set());
    return /** @type {Set<DftSignalListener>} */ (listeners.get(key));
  }

  /**
   * @param {string} name
   * @param {DftSignalListener} listener
   * @returns {() => void}
   */
  function on(name, listener) {
    if (typeof listener !== "function") {
      throw new TypeError("signal listener must be a function");
    }

    const bucket = listenersFor(name);
    bucket.add(listener);

    return () => {
      bucket.delete(listener);
      if (bucket.size === 0) listeners.delete(String(name));
    };
  }

  /**
   * @param {string} name
   * @param {Record<string, unknown>} detail
   * @param {unknown} source
   * @returns {DftSignalPayload}
   */
  function emit(name, detail = {}, source = null) {
    const key = String(name);
    const payload = { name: key, detail, source };

    for (const listener of Array.from(listenersFor(key))) {
      listener(payload);
    }

    if (target && typeof CustomEvent !== "undefined") {
      target.dispatchEvent(new CustomEvent(DFT_SIGNAL_EVENT, {
        detail: payload,
        bubbles: true,
        composed: true,
      }));
    }

    return payload;
  }

  return { on, emit };
}

const THREE_MODULE_URL = "https://unpkg.com/three@0.160.0/build/three.module.js";
const ORBIT_CONTROLS_MODULE_URL = "https://unpkg.com/three@0.160.0/examples/jsm/controls/OrbitControls.js";

/** @type {null | Promise<{THREE:any, OrbitControls:any}>} */
let threeRuntimePromise = null;

async function loadThreeRuntime() {
  if (threeRuntimePromise) return threeRuntimePromise;

  threeRuntimePromise = Promise.all([
    import(THREE_MODULE_URL),
    import(ORBIT_CONTROLS_MODULE_URL),
  ]).then(([THREE, controlsModule]) => ({
    THREE,
    OrbitControls: controlsModule.OrbitControls,
  }));

  return threeRuntimePromise;
}

/**
 * Convert oblique reciprocal-basis coordinates into a Euclidean display plane.
 *
 * The payload uses basis coordinates (k1, k2). The central BZ condition
 * |k1| <= pi, |k2| <= pi, |k1 - k2| <= pi corresponds to an oblique
 * hexagonal coordinate system, so display basis vectors should not be
 * drawn as a square grid.
 *
 * @param {number} k1
 * @param {number} k2
 * @returns {{x:number, y:number}}
 */
function bandBasisToCartesian(k1, k2) {
  return {
    x: k1 - 0.5 * k2,
    y: (Math.sqrt(3.0) / 2.0) * k2,
  };
}

/**
 * Convert payload BZ hexagon vertices into the three.js display plane.
 *
 * @param {JsonPayload | null} payload
 * @returns {Array<{x:number, y:number, z:number}>}
 */
/**
 * Build oblique u/v grid line segments on the display k-plane.
 *
 * @param {number} limit
 * @param {number} steps
 * @returns {Array<[{x:number, y:number, z:number}, {x:number, y:number, z:number}]>}
 */
function threeUvGridReferenceData(limit = Math.PI, steps = 8) {
  /** @type {Array<[{x:number, y:number, z:number}, {x:number, y:number, z:number}]>} */
  const lines = [];

  for (let i = -steps; i <= steps; i += 1) {
    const t = (limit * i) / steps;

    const u0 = bandBasisToCartesian(t, -limit);
    const u1 = bandBasisToCartesian(t, limit);
    lines.push([
      { x: u0.x, y: 0.0, z: u0.y },
      { x: u1.x, y: 0.0, z: u1.y },
    ]);

    const v0 = bandBasisToCartesian(-limit, t);
    const v1 = bandBasisToCartesian(limit, t);
    lines.push([
      { x: v0.x, y: 0.0, z: v0.y },
      { x: v1.x, y: 0.0, z: v1.y },
    ]);
  }

  return lines;
}


/**
 * Convert payload BZ hexagon vertices into the three.js display plane.
 *
 * @param {JsonPayload | null} payload
 * @returns {Array<{x:number, y:number, z:number}>}
 */
function threeHexagonReferenceData(payload) {
  const raw = Array.isArray(payload?.bz_hexagon) ? payload.bz_hexagon : [];
  /** @type {Array<{x:number, y:number, z:number}>} */
  const out = [];

  for (const point of raw) {
    if (!Array.isArray(point) || point.length < 2) continue;

    const k1 = Number(point[0]);
    const k2 = Number(point[1]);

    if (!Number.isFinite(k1) || !Number.isFinite(k2)) continue;

    const p = bandBasisToCartesian(k1, k2);
    out.push({ x: p.x, y: 0.0, z: p.y });
  }

  if (out.length >= 3) return out;

  return [
    [Math.PI, 0.0],
    [Math.PI, Math.PI],
    [0.0, Math.PI],
    [-Math.PI, 0.0],
    [-Math.PI, -Math.PI],
    [0.0, -Math.PI],
  ].map(([k1, k2]) => {
    const p = bandBasisToCartesian(k1, k2);
    return { x: p.x, y: 0.0, z: p.y };
  });
}


/**
 * @param {{vertices:Array<{x:number, y:number, z:number, i:number, j:number, band:number}>, triangles:Array<[number, number, number]>, summary:{count:number, zmin:number|null, zmax:number|null}}} mesh
 * @returns {{positions:Float32Array, indices:Uint32Array, center:{x:number,y:number,z:number}, radius:number}}
 */
/**
 * @param {{x:number, z:number}} point
 * @param {Array<{x:number, y:number, z:number}>} polygon
 * @returns {boolean}
 */
function pointInDisplayPolygon(point, polygon) {
  if (polygon.length < 3) return true;

  const tol = 1e-10;
  let inside = false;

  for (let i = 0, j = polygon.length - 1; i < polygon.length; j = i, i += 1) {
    const pi = polygon[i];
    const pj = polygon[j];

    const dx = pj.x - pi.x;
    const dz = pj.z - pi.z;
    const cross = (point.x - pi.x) * dz - (point.z - pi.z) * dx;
    const dot = (point.x - pi.x) * dx + (point.z - pi.z) * dz;
    const len2 = dx * dx + dz * dz;

    if (Math.abs(cross) <= tol && dot >= -tol && dot <= len2 + tol) {
      return true;
    }

    const crosses = (
      (pi.z > point.z) !== (pj.z > point.z)
      && point.x < ((pj.x - pi.x) * (point.z - pi.z)) / ((pj.z - pi.z) || 1e-30) + pi.x
    );

    if (crosses) inside = !inside;
  }

  return inside;
}

/**
 * @param {JsonPayload | null} payload
 * @param {{x:number, y:number, z:number, i:number, j:number, band:number}} vertex
 * @returns {boolean}
 */
function vertexInsideVisibleHexagon(payload, vertex) {
  const polygon = threeHexagonReferenceData(payload);
  const p = bandBasisToCartesian(vertex.x, vertex.y);
  return pointInDisplayPolygon({ x: p.x, z: p.y }, polygon);
}


/**
 * @param {{vertices:Array<{x:number, y:number, z:number, i:number, j:number, band:number}>, triangles:Array<[number, number, number]>, summary:{count:number, zmin:number|null, zmax:number|null}}} mesh
 * @returns {{positions:Float32Array, indices:Uint32Array, center:{x:number,y:number,z:number}, radius:number}}
 */
function threeBandSurfaceGeometryData(mesh) {
  const positions = new Float32Array(mesh.vertices.length * 3);
  const indices = new Uint32Array(mesh.triangles.length * 3);

  let kxmin = Infinity;
  let kxmax = -Infinity;
  let kymin = Infinity;
  let kymax = -Infinity;
  let emin = Infinity;
  let emax = -Infinity;

  const cartesian = mesh.vertices.map((v) => {
    const p = bandBasisToCartesian(v.x, v.y);
    kxmin = Math.min(kxmin, p.x);
    kxmax = Math.max(kxmax, p.x);
    kymin = Math.min(kymin, p.y);
    kymax = Math.max(kymax, p.y);
    emin = Math.min(emin, v.z);
    emax = Math.max(emax, v.z);
    return { kx: p.x, ky: p.y, energy: v.z };
  });

  const kSpan = Math.max(kxmax - kxmin, kymax - kymin, 1.0);
  const eSpan = Math.max(emax - emin, 1e-12);
  const energyVisualHeight = 0.9 * kSpan;

  let xmin = Infinity;
  let xmax = -Infinity;
  let ymin = Infinity;
  let ymax = -Infinity;
  let zmin = Infinity;
  let zmax = -Infinity;

  for (let i = 0; i < cartesian.length; i += 1) {
    const p = cartesian[i];

    const x = p.kx;
    const y = ((p.energy - emin) / eSpan - 0.5) * energyVisualHeight;
    const z = p.ky;

    positions[3 * i + 0] = x;
    positions[3 * i + 1] = y;
    positions[3 * i + 2] = z;

    xmin = Math.min(xmin, x);
    xmax = Math.max(xmax, x);
    ymin = Math.min(ymin, y);
    ymax = Math.max(ymax, y);
    zmin = Math.min(zmin, z);
    zmax = Math.max(zmax, z);
  }

  for (let i = 0; i < mesh.triangles.length; i += 1) {
    const tri = mesh.triangles[i];
    indices[3 * i + 0] = tri[0];
    indices[3 * i + 1] = tri[1];
    indices[3 * i + 2] = tri[2];
  }

  const center = {
    x: 0.5 * (xmin + xmax),
    y: 0.5 * (ymin + ymax),
    z: 0.5 * (zmin + zmax),
  };

  const radius = Math.max(
    xmax - xmin,
    ymax - ymin,
    zmax - zmin,
    1.0,
  );

  return { positions, indices, center, radius };
}


const dftSignals = createDftSignalBus(
  typeof window === "undefined" ? null : window,
);

/**
 * @param {string} name
 * @param {Record<string, unknown>} detail
 * @param {unknown} source
 * @returns {DftSignalPayload}
 */
function emitDftSignal(name, detail = {}, source = null) {
  return dftSignals.emit(name, detail, source);
}

/**
 * @param {string} name
 * @param {DftSignalListener} listener
 * @returns {() => void}
 */
function onDftSignal(name, listener) {
  return dftSignals.on(name, listener);
}


/**
 * @param {Element} host
 * @returns {JsonPayload | null}
 */
function readJsonPayload(host) {
  const source = host.getAttribute("data-source");
  if (!source) return null;

  const script = document.getElementById(source);
  if (!script) return null;

  try {
    return /** @type {Record<string, unknown>} */ (JSON.parse(script.textContent || ""));
  } catch {
    return null;
  }
}

/**
 * @param {Element} host
 * @returns {GraphPayload | null}
 */
/**
 * @param {Element} host
 * @returns {GraphPayload | null}
 */
function readGraphPayload(host) {
  return /** @type {GraphPayload | null} */ (readJsonPayload(host));
}

/**
 * @param {number | null} step
 * @param {number | null} pathX
 * @param {EventTarget} source
 */
function emitPathHover(step, pathX, source) {
  window.dispatchEvent(new CustomEvent("dft-local-path-hover", {
    detail: { step, pathX, source },
  }));
}

/**
 * @param {unknown} value
 * @returns {number | null}
 */
function numberOrNull(value) {
  return typeof value === "number" && Number.isFinite(value) ? value : null;
}

const globalSelectionState = {
  frozen: false,
  /** @type {Map<number, {pathX:number|null, energy:number|null, label:string|null}>} */
  selectedSteps: new Map(),
};

/**
 * @param {boolean} frozen
 * @param {EventTarget} source
 */
function emitSelectionFreeze(frozen, source) {
  globalSelectionState.frozen = frozen;

  if (typeof window !== "undefined") {
    window.dispatchEvent(new CustomEvent("dft-local-selection-freeze", {
      detail: { frozen, source },
    }));
  }
}

/**
 * @returns {boolean}
 */
function isSelectionFrozen() {
  return globalSelectionState.frozen;
}

/**
 * @returns {Array<{step:number, pathX:number|null, energy:number|null, label:string|null}>}
 */
function selectedSteps() {
  return Array.from(globalSelectionState.selectedSteps, ([step, value]) => {
    if (typeof value === "number" || value === null) {
      return { step, pathX: value, energy: null, label: null };
    }

    return {
      step,
      pathX: value.pathX,
      energy: value.energy,
      label: value.label ?? null,
    };
  });
}

/**
 * @param {Array<{step:number, pathX:number|null, energy?:number|null, label?:string|null}>} values
 * @param {EventTarget} source
 */
function emitSelectedSteps(values, source) {
  globalSelectionState.selectedSteps = new Map(
    values.map((item) => [
      item.step,
      { pathX: item.pathX, energy: item.energy ?? null, label: item.label ?? null },
    ]),
  );

  if (typeof window !== "undefined") {
    window.dispatchEvent(new CustomEvent("dft-local-selected-steps", {
      detail: { values: selectedSteps(), source },
    }));
  }
}

/**
 * @param {string | null | undefined} value
 * @returns {number | null}
 */
function parseNumericAttribute(value) {
  if (value === null || value === undefined || value === "") return null;
  const number = Number(value);
  return Number.isFinite(number) ? number : null;
}

/**
 * @param {GraphPayload} payload
 * @returns {{xmin:number, xmax:number, ymin:number, ymax:number}}
 */
function graphBounds(payload) {
  /** @type {GraphPoint[]} */
  const all = [];

  for (const series of payload.series) {
    for (const point of series.points) {
      if (Number.isFinite(point.x) && Number.isFinite(point.y)) {
        all.push(point);
      }
    }
  }

  if (all.length === 0) {
    return { xmin: -1, xmax: 1, ymin: -1, ymax: 1 };
  }

  let xmin = Math.min(...all.map((p) => p.x));
  let xmax = Math.max(...all.map((p) => p.x));
  let ymin = Math.min(...all.map((p) => p.y));
  let ymax = Math.max(...all.map((p) => p.y));

  if (xmin === xmax) {
    xmin -= 1;
    xmax += 1;
  }
  if (ymin === ymax) {
    ymin -= 1;
    ymax += 1;
  }

  const ypad = 0.06 * (ymax - ymin);
  ymin -= ypad;
  ymax += ypad;

  return { xmin, xmax, ymin, ymax };
}

/**
 * @param {{xmin:number, xmax:number, ymin:number, ymax:number}} view
 * @param {number} fx
 * @param {number} fy
 * @param {number} factor
 * @returns {{xmin:number, xmax:number, ymin:number, ymax:number}}
 */
function zoomView(view, fx, fy, factor) {
  const x = view.xmin + fx * (view.xmax - view.xmin);
  const y = view.ymin + fy * (view.ymax - view.ymin);

  return {
    xmin: x - (x - view.xmin) * factor,
    xmax: x + (view.xmax - x) * factor,
    ymin: y - (y - view.ymin) * factor,
    ymax: y + (view.ymax - y) * factor,
  };
}

/**
 * @param {{xmin:number, xmax:number, ymin:number, ymax:number}} view
 * @param {number} dxFraction
 * @param {number} dyFraction
 * @returns {{xmin:number, xmax:number, ymin:number, ymax:number}}
 */
function panView(view, dxFraction, dyFraction) {
  const dx = dxFraction * (view.xmax - view.xmin);
  const dy = dyFraction * (view.ymax - view.ymin);

  return {
    xmin: view.xmin - dx,
    xmax: view.xmax - dx,
    ymin: view.ymin + dy,
    ymax: view.ymax + dy,
  };
}

/**
 * @param {GraphView} view
 * @returns {GraphView}
 */
function equalAspectView(view) {
  const xMid = 0.5 * (view.xmin + view.xmax);
  const yMid = 0.5 * (view.ymin + view.ymax);
  const span = Math.max(view.xmax - view.xmin, view.ymax - view.ymin);

  return {
    xmin: xMid - 0.5 * span,
    xmax: xMid + 0.5 * span,
    ymin: yMid - 0.5 * span,
    ymax: yMid + 0.5 * span,
  };
}

/**
 * Expand a data view so one data unit has the same pixel size in x and y.
 *
 * @param {GraphView} view
 * @param {number} innerW
 * @param {number} innerH
 * @returns {GraphView}
 */
function equalPixelAspectView(view, innerW, innerH) {
  const xMid = 0.5 * (view.xmin + view.xmax);
  const yMid = 0.5 * (view.ymin + view.ymax);
  const xSpan = view.xmax - view.xmin;
  const ySpan = view.ymax - view.ymin;
  const pixelAspect = innerW / innerH;
  const dataAspect = xSpan / ySpan;

  if (!Number.isFinite(pixelAspect) || !Number.isFinite(dataAspect) || pixelAspect <= 0 || dataAspect <= 0) {
    return view;
  }

  if (dataAspect < pixelAspect) {
    const expandedXSpan = ySpan * pixelAspect;
    return {
      xmin: xMid - 0.5 * expandedXSpan,
      xmax: xMid + 0.5 * expandedXSpan,
      ymin: view.ymin,
      ymax: view.ymax,
    };
  }

  const expandedYSpan = xSpan / pixelAspect;
  return {
    xmin: view.xmin,
    xmax: view.xmax,
    ymin: yMid - 0.5 * expandedYSpan,
    ymax: yMid + 0.5 * expandedYSpan,
  };
}

/**
 * @param {number} k1
 * @param {number} k2
 * @returns {{x:number, y:number}}
 */
function kBasisToCartesian(k1, k2) {
  return {
    x: k1 - 0.5 * k2,
    y: (Math.sqrt(3) / 2) * k2,
  };
}

/**
 * @param {number} x
 * @param {number} y
 * @param {number} angle
 * @returns {{x:number, y:number}}
 */
function rotatePoint(x, y, angle) {
  const c = Math.cos(angle);
  const s = Math.sin(angle);
  return { x: c * x - s * y, y: s * x + c * y };
}

/**
 * @param {GraphPayload} payload
 * @param {number} angle
 * @returns {GraphPayload}
 */
/**
 * @param {JsonPayload | null} payload
 * @param {number} band
 * @returns {Array<{x:number, y:number, z:number, i:number, j:number, band:number}>}
 */
function bandSurfaceVertices(payload, band) {
  if (!payload) return [];

  const k1 = /** @type {unknown[][] | undefined} */ (payload.k1);
  const k2 = /** @type {unknown[][] | undefined} */ (payload.k2);
  const energies = /** @type {unknown[][][] | undefined} */ (payload.energies);

  if (!Array.isArray(k1) || !Array.isArray(k2) || !Array.isArray(energies)) return [];

  const vertices = [];

  for (let i = 0; i < energies.length; i += 1) {
    const row = energies[i];
    if (!Array.isArray(row)) continue;

    for (let j = 0; j < row.length; j += 1) {
      const energyBands = row[j];
      if (!Array.isArray(energyBands)) continue;

      const x = Number(k1[i]?.[j]);
      const y = Number(k2[i]?.[j]);
      const z = Number(energyBands[band]);

      if (!Number.isFinite(x) || !Number.isFinite(y) || !Number.isFinite(z)) continue;

      vertices.push({ x, y, z, i, j, band });
    }
  }

  return vertices;
}

/**
 * @param {JsonPayload | null} payload
 * @param {boolean} useMask
 * @returns {Array<[number, number, number]>}
 */
function bandSurfaceTriangles(payload, useMask = true) {
  if (!payload) return [];

  const nu = Number(payload.nu);
  const nv = Number(payload.nv);
  const mask = /** @type {unknown[][] | undefined} */ (payload.mask);

  if (!Number.isInteger(nu) || !Number.isInteger(nv) || nu < 2 || nv < 2) return [];

  /**
   * @param {number} i
   * @param {number} j
   */
  function vertexIndex(i, j) {
    return i * nv + j;
  }

  /**
   * @param {number} i
   * @param {number} j
   */
  function enabled(i, j) {
    if (!useMask || !Array.isArray(mask)) return true;
    return Boolean(mask[i]?.[j]);
  }

  /** @type {Array<[number, number, number]>} */
  const triangles = [];

  for (let i = 0; i < nu - 1; i += 1) {
    for (let j = 0; j < nv - 1; j += 1) {
      const a = enabled(i, j);
      const b = enabled(i + 1, j);
      const c = enabled(i, j + 1);
      const d = enabled(i + 1, j + 1);

      if (a && b && c) {
        triangles.push([vertexIndex(i, j), vertexIndex(i + 1, j), vertexIndex(i, j + 1)]);
      }

      if (b && d && c) {
        triangles.push([vertexIndex(i + 1, j), vertexIndex(i + 1, j + 1), vertexIndex(i, j + 1)]);
      }
    }
  }

  return triangles;
}

/**
 * @param {JsonPayload | null} payload
 * @param {number} band
 * @returns {{count:number, zmin:number|null, zmax:number|null}}
 */
function bandSurfaceSummary(payload, band) {
  const vertices = bandSurfaceVertices(payload, band);

  if (vertices.length === 0) {
    return { count: 0, zmin: null, zmax: null };
  }

  let zmin = Infinity;
  let zmax = -Infinity;

  for (const vertex of vertices) {
    zmin = Math.min(zmin, vertex.z);
    zmax = Math.max(zmax, vertex.z);
  }

  return { count: vertices.length, zmin, zmax };
}

/**
 * @param {JsonPayload | null} payload
 * @param {number} band
 * @returns {{
 *   vertices:Array<{x:number, y:number, z:number, i:number, j:number, band:number}>,
 *   triangles:Array<[number, number, number]>,
 *   summary:{count:number, zmin:number|null, zmax:number|null}
 * }}
 */
function bandSurfaceMeshData(payload, band) {
  const vertices = bandSurfaceVertices(payload, band);
  const triangles = bandSurfaceTriangles(payload);
  const summary = bandSurfaceSummary(payload, band);

  return { vertices, triangles, summary };
}


/**
 * @param {JsonPayload | null} payload
 * @param {number} band
 * @param {boolean} useMask
 * @returns {{
 *   vertices:Array<{x:number, y:number, z:number, i:number, j:number, band:number}>,
 *   triangles:Array<[number, number, number]>,
 *   summary:{count:number, zmin:number|null, zmax:number|null}
 * }}
 */
/**
 * @param {number} band
 * @returns {number}
 */
function bandSurfaceColor(band) {
  const palette = [
    0x4e79a7,
    0xf28e2b,
    0xe15759,
    0x76b7b2,
    0x59a14f,
    0xedc949,
    0xaf7aa1,
    0xff9da7,
    0x9c755f,
    0xbab0ab,
  ];

  return palette[Math.abs(Math.trunc(band)) % palette.length];
}

/**
 * @param {JsonPayload | null} payload
 * @returns {number[]}
 */
function allBandIndices(payload) {
  const bands = Array.isArray(payload?.bands)
    ? payload.bands.map((band) => Number(band)).filter((band) => Number.isInteger(band))
    : [];

  const nbands = Number(payload?.nbands ?? bands.length);
  const fallback = Number.isInteger(nbands) && nbands > 0
    ? Array.from({ length: nbands }, (_, band) => band)
    : bands;

  return bands.length > 0 ? bands : fallback;
}


/**
 * @param {JsonPayload | null} payload
 * @param {Set<number>} hiddenBands
 * @returns {number[]}
 */
function visibleBandIndices(payload, hiddenBands) {
  const bands = Array.isArray(payload?.bands)
    ? payload.bands.map((band) => Number(band)).filter((band) => Number.isInteger(band))
    : [];

  const nbands = Number(payload?.nbands ?? bands.length);
  const fallback = Number.isInteger(nbands) && nbands > 0
    ? Array.from({ length: nbands }, (_, band) => band)
    : bands;

  const available = bands.length > 0 ? bands : fallback;

  return available.filter((band) => !hiddenBands.has(band));
}




/**
 * @param {JsonPayload | null} payload
 * @param {number} band
 * @param {boolean} useMask
 * @returns {{
 *   vertices:Array<{x:number, y:number, z:number, i:number, j:number, band:number}>,
 *   triangles:Array<[number, number, number]>,
 *   summary:{count:number, zmin:number|null, zmax:number|null}
 * }}
 */
function bandSurfaceMeshDataWithMask(payload, band, useMask) {
  const rawVertices = bandSurfaceVertices(payload, band);
  const rawTriangles = bandSurfaceTriangles(payload, false);

  if (!useMask) {
    const summary = bandSurfaceSummary(payload, band);
    return { vertices: rawVertices, triangles: rawTriangles, summary };
  }

  /** @type {Map<number, number>} */
  const indexMap = new Map();
  /** @type {Array<{x:number, y:number, z:number, i:number, j:number, band:number}>} */
  const vertices = [];
  /** @type {Array<[number, number, number]>} */
  const triangles = [];

  for (const tri of rawTriangles) {
    const triVertices = tri.map((rawIndex) => rawVertices[rawIndex]);

    if (
      triVertices.length !== 3
      || triVertices.some((vertex) => !vertex || !vertexInsideVisibleHexagon(payload, vertex))
    ) {
      continue;
    }

    /** @type {number[]} */
    const remapped = [];

    for (const rawIndex of tri) {
      let mapped = indexMap.get(rawIndex);

      if (mapped === undefined) {
        const vertex = rawVertices[rawIndex];
        if (!vertex) break;

        mapped = vertices.length;
        indexMap.set(rawIndex, mapped);
        vertices.push(vertex);
      }

      remapped.push(mapped);
    }

    if (remapped.length === 3) {
      triangles.push(/** @type {[number, number, number]} */ ([remapped[0], remapped[1], remapped[2]]));
    }
  }

  if (vertices.length === 0) {
    return { vertices, triangles, summary: { count: 0, zmin: null, zmax: null } };
  }

  let zmin = Infinity;
  let zmax = -Infinity;

  for (const vertex of vertices) {
    zmin = Math.min(zmin, vertex.z);
    zmax = Math.max(zmax, vertex.z);
  }

  return { vertices, triangles, summary: { count: vertices.length, zmin, zmax } };
}



/**
 * @param {{x:number, y:number, z:number}} point
 * @param {{xmin:number, xmax:number, ymin:number, ymax:number, zmin:number, zmax:number, width:number, height:number, energyScale?:number, rotation?:number, pitch?:number, viewZoom?:number}} view
 * @returns {{x:number, y:number}}
 */
function projectBandSurfacePoint(point, view) {
  const xRange = view.xmax - view.xmin || 1.0;
  const yRange = view.ymax - view.ymin || 1.0;
  const zRange = view.zmax - view.zmin || 1.0;

  const viewZoom = view.viewZoom ?? 1.0;
  const px0 = ((point.x - view.xmin) / xRange - 0.5) * viewZoom;
  const py0 = ((point.y - view.ymin) / yRange - 0.5) * viewZoom;
  const pz = ((point.z - view.zmin) / zRange) * (view.energyScale ?? 1.0);
  const theta = view.rotation ?? 0.0;
  const pitch = view.pitch ?? 0.65;
  const c = Math.cos(theta);
  const s = Math.sin(theta);
  const px = c * px0 - s * py0 + 0.5;
  const py = s * px0 + c * py0 + 0.5;
  const planarY = py * (view.height - 48) * Math.cos(pitch);
  const energyY = 0.75 * pz * (view.height - 48) * Math.sin(pitch);

  return {
    x: 24 + px * (view.width - 48) + 0.22 * (py - 0.5) * (view.width - 48),
    y: view.height - 24 - planarY - energyY,
  };
}

/**
 * @param {{vertices:Array<{x:number, y:number, z:number, i:number, j:number, band:number}>, summary:{zmin:number|null, zmax:number|null}}} mesh
 * @param {{x:number, y:number}} pointer
 * @param {{xmin:number, xmax:number, ymin:number, ymax:number, zmin:number, zmax:number, width:number, height:number, energyScale?:number, rotation?:number, pitch?:number, viewZoom?:number}} view
 * @param {number} maxDistance
 * @returns {null | {vertex:{x:number, y:number, z:number, i:number, j:number, band:number}, sx:number, sy:number, distance:number}}
 */
function nearestBandSurfaceVertex(mesh, pointer, view, maxDistance = 20.0) {
  let best = null;
  let bestDistance = maxDistance;

  for (const vertex of mesh.vertices) {
    const p = projectBandSurfacePoint(vertex, view);
    const distance = Math.hypot(pointer.x - p.x, pointer.y - p.y);

    if (distance <= bestDistance) {
      bestDistance = distance;
      best = { vertex, sx: p.x, sy: p.y, distance };
    }
  }

  return best;
}

/**
 * @param {{vertices:Array<{x:number, y:number, z:number, i:number, j:number, band:number}>, triangles:Array<[number, number, number]>, summary:{count:number, zmin:number|null, zmax:number|null}}} mesh
 * @param {HTMLCanvasElement} canvas
 * @param {{energyScale?:number, rotation?:number, pitch?:number, viewZoom?:number, sliceAxis?:string|null, sliceValue?:number|null, payload?:JsonPayload|null, selectedKpoint?:Record<string, unknown>|null}} options
 * @returns {null | {xmin:number, xmax:number, ymin:number, ymax:number, zmin:number, zmax:number, width:number, height:number, energyScale?:number, rotation?:number}}
 */
function drawBandSurfacePreview(mesh, canvas, options = {}) {
  const ctx = canvas.getContext("2d");
  if (!ctx) return null;

  const width = canvas.width;
  const height = canvas.height;
  ctx.clearRect(0, 0, width, height);

  if (mesh.vertices.length === 0) {
    ctx.fillText("no surface data", 16, 24);
    return null;
  }

  let xmin = Infinity;
  let xmax = -Infinity;
  let ymin = Infinity;
  let ymax = -Infinity;

  for (const v of mesh.vertices) {
    xmin = Math.min(xmin, v.x);
    xmax = Math.max(xmax, v.x);
    ymin = Math.min(ymin, v.y);
    ymax = Math.max(ymax, v.y);
  }

  const view = {
    xmin,
    xmax,
    ymin,
    ymax,
    zmin: mesh.summary.zmin ?? 0.0,
    zmax: mesh.summary.zmax ?? 1.0,
    width,
    height,
    energyScale: options.energyScale ?? 1.0,
    rotation: options.rotation ?? 0.0,
    pitch: options.pitch ?? 0.65,
    viewZoom: options.viewZoom ?? 1.0,
  };

  ctx.lineWidth = 0.6;
  ctx.globalAlpha = 0.45;

  for (const tri of mesh.triangles) {
    const a = mesh.vertices[tri[0]];
    const b = mesh.vertices[tri[1]];
    const c = mesh.vertices[tri[2]];
    if (!a || !b || !c) continue;

    const pa = projectBandSurfacePoint(a, view);
    const pb = projectBandSurfacePoint(b, view);
    const pc = projectBandSurfacePoint(c, view);

    ctx.beginPath();
    ctx.moveTo(pa.x, pa.y);
    ctx.lineTo(pb.x, pb.y);
    ctx.lineTo(pc.x, pc.y);
    ctx.closePath();
    ctx.stroke();
  }

  drawBandSurfaceReferenceFrame(options.payload ?? null, ctx, view);
  drawBandSurfaceSliceGuide(mesh, ctx, view, options);
  drawBandSurfaceSelectionMarker(mesh, ctx, view, options.selectedKpoint ?? null);
  ctx.globalAlpha = 1.0;
  return view;
}

/**
 * @param {{vertices:Array<{x:number, y:number, z:number, i:number, j:number}>, triangles:Array<[number, number, number]>, summary:{count:number, zmin:number|null, zmax:number|null}}} mesh
 * @param {CanvasRenderingContext2D} ctx
 * @param {{xmin:number, xmax:number, ymin:number, ymax:number, zmin:number, zmax:number, width:number, height:number, energyScale?:number, rotation?:number, pitch?:number, viewZoom?:number}} view
 * @param {{sliceAxis?:string|null, sliceValue?:number|null}} options
 */
function drawBandSurfaceSliceGuide(mesh, ctx, view, options) {
  const axis = options.sliceAxis ?? null;
  const value = options.sliceValue ?? null;

  if (axis === null || value === null || !Number.isFinite(value)) return;

  const clamped = Math.max(0.0, Math.min(1.0, value));
  ctx.save();
  ctx.lineWidth = 2.0;
  ctx.globalAlpha = 0.9;
  ctx.setLineDash([6, 4]);

  if (axis === "u") {
    const iValues = mesh.vertices.map((v) => v.i);
    const imax = Math.max(...iValues);
    const target = Math.round(clamped * imax);
    const points = mesh.vertices.filter((v) => v.i === target);

    if (points.length >= 2) {
      drawProjectedPolyline(points, ctx, view);
    }
  } else if (axis === "v") {
    const jValues = mesh.vertices.map((v) => v.j);
    const jmax = Math.max(...jValues);
    const target = Math.round(clamped * jmax);
    const points = mesh.vertices.filter((v) => v.j === target);

    if (points.length >= 2) {
      drawProjectedPolyline(points, ctx, view);
    }
  } else if (axis === "energy") {
    const z = view.zmin + clamped * (view.zmax - view.zmin);
    const points = mesh.vertices
      .slice()
      .sort((a, b) => a.x - b.x || a.y - b.y)
      .map((v) => ({ ...v, z }));

    if (points.length >= 2) {
      drawProjectedPolyline(points, ctx, view);
    }
  }

  ctx.restore();
}

/**
 * @param {Array<{x:number, y:number, z:number}>} points
 * @param {CanvasRenderingContext2D} ctx
 * @param {{xmin:number, xmax:number, ymin:number, ymax:number, zmin:number, zmax:number, width:number, height:number, energyScale?:number, rotation?:number, pitch?:number, viewZoom?:number}} view
 */
function drawProjectedPolyline(points, ctx, view) {
  ctx.beginPath();

  for (let index = 0; index < points.length; index += 1) {
    const p = projectBandSurfacePoint(points[index], view);
    if (index === 0) ctx.moveTo(p.x, p.y);
    else ctx.lineTo(p.x, p.y);
  }

  ctx.stroke();
}


/**
 * @param {JsonPayload | null} payload
 * @param {CanvasRenderingContext2D} ctx
 * @param {{xmin:number, xmax:number, ymin:number, ymax:number, zmin:number, zmax:number, width:number, height:number, energyScale?:number, rotation?:number, pitch?:number, viewZoom?:number}} view
 */
function drawBandSurfaceReferenceFrame(payload, ctx, view) {
  ctx.save();
  ctx.globalAlpha = 1.0;
  ctx.lineWidth = 1.4;
  ctx.setLineDash([]);

  const origin = projectBandSurfacePoint({ x: 0, y: 0, z: view.zmin }, view);
  const k1Tip = projectBandSurfacePoint({ x: view.xmax, y: 0, z: view.zmin }, view);
  const k2Tip = projectBandSurfacePoint({ x: 0, y: view.ymax, z: view.zmin }, view);
  const eTip = projectBandSurfacePoint({ x: 0, y: 0, z: view.zmax }, view);

  drawArrow(ctx, origin, k1Tip, "k1");
  drawArrow(ctx, origin, k2Tip, "k2");
  drawArrow(ctx, origin, eTip, "E");

  drawBandSurfaceBzBoundary(payload, ctx, view);
  drawBandSurfaceSymmetryLabels(ctx, view);

  ctx.restore();
}

/**
 * @param {CanvasRenderingContext2D} ctx
 * @param {{x:number, y:number}} a
 * @param {{x:number, y:number}} b
 * @param {string} label
 */
function drawArrow(ctx, a, b, label) {
  ctx.beginPath();
  ctx.moveTo(a.x, a.y);
  ctx.lineTo(b.x, b.y);
  ctx.stroke();

  const dx = b.x - a.x;
  const dy = b.y - a.y;
  const len = Math.hypot(dx, dy) || 1.0;
  const ux = dx / len;
  const uy = dy / len;
  const size = 8;

  ctx.beginPath();
  ctx.moveTo(b.x, b.y);
  ctx.lineTo(b.x - size * ux - 0.45 * size * uy, b.y - size * uy + 0.45 * size * ux);
  ctx.lineTo(b.x - size * ux + 0.45 * size * uy, b.y - size * uy - 0.45 * size * ux);
  ctx.closePath();
  ctx.fill();

  ctx.fillText(label, b.x + 6, b.y - 6);
}

/**
 * @param {JsonPayload | null} payload
 * @param {CanvasRenderingContext2D} ctx
 * @param {{xmin:number, xmax:number, ymin:number, ymax:number, zmin:number, zmax:number, width:number, height:number, energyScale?:number, rotation?:number, pitch?:number, viewZoom?:number}} view
 */
function drawBandSurfaceBzBoundary(payload, ctx, view) {
  const bz = /** @type {unknown[][] | undefined} */ (payload?.bz_hexagon);
  if (!Array.isArray(bz) || bz.length < 3) return;

  ctx.save();
  ctx.lineWidth = 2.0;
  ctx.setLineDash([3, 3]);

  ctx.beginPath();
  for (let index = 0; index < bz.length; index += 1) {
    const point = bz[index];
    if (!Array.isArray(point) || point.length < 2) continue;

    const p = projectBandSurfacePoint({
      x: Number(point[0]),
      y: Number(point[1]),
      z: view.zmin,
    }, view);

    if (index === 0) ctx.moveTo(p.x, p.y);
    else ctx.lineTo(p.x, p.y);
  }
  ctx.closePath();
  ctx.stroke();

  ctx.restore();
}

/**
 * @param {CanvasRenderingContext2D} ctx
 * @param {{xmin:number, xmax:number, ymin:number, ymax:number, zmin:number, zmax:number, width:number, height:number, energyScale?:number, rotation?:number, pitch?:number, viewZoom?:number}} view
 */
function drawBandSurfaceSymmetryLabels(ctx, view) {
  const labels = [
    ["Γ", 0.0, 0.0],
    ["K", (2.0 * Math.PI) / 3.0, -(2.0 * Math.PI) / 3.0],
    ["M", Math.PI, 0.0],
  ];

  ctx.save();
  ctx.setLineDash([]);
  ctx.lineWidth = 1.0;

  for (const [label, x, y] of labels) {
    const p = projectBandSurfacePoint({ x: Number(x), y: Number(y), z: view.zmin }, view);

    ctx.beginPath();
    ctx.arc(p.x, p.y, 3.5, 0, 2.0 * Math.PI);
    ctx.fill();
    ctx.fillText(String(label), p.x + 6, p.y - 6);
  }

  ctx.restore();
}


/**
 * @param {{vertices:Array<{x:number, y:number, z:number, i:number, j:number, band:number}>}} mesh
 * @param {CanvasRenderingContext2D} ctx
 * @param {{xmin:number, xmax:number, ymin:number, ymax:number, zmin:number, zmax:number, width:number, height:number, energyScale?:number, rotation?:number, pitch?:number, viewZoom?:number}} view
 * @param {null | Record<string, unknown>} selected
 */
function drawBandSurfaceSelectionMarker(mesh, ctx, view, selected) {
  if (!selected) return;

  const i = Number(selected.i);
  const j = Number(selected.j);
  const band = Number(selected.band);

  const vertex = mesh.vertices.find((v) => v.i === i && v.j === j && v.band === band);
  if (!vertex) return;

  const p = projectBandSurfacePoint(vertex, view);

  ctx.save();
  ctx.setLineDash([]);
  ctx.lineWidth = 2.5;
  ctx.globalAlpha = 1.0;

  ctx.beginPath();
  ctx.arc(p.x, p.y, 7.0, 0, 2.0 * Math.PI);
  ctx.stroke();

  ctx.beginPath();
  ctx.arc(p.x, p.y, 3.0, 0, 2.0 * Math.PI);
  ctx.fill();

  ctx.fillText("selected", p.x + 10, p.y - 10);
  ctx.restore();
}


/**
 * @param {GraphPayload} payload
 * @param {number} angle
 * @returns {GraphPayload}
 */
function kspacePayloadToCartesian(payload, angle = 0) {
  return {
    ...payload,
    x_label: "k Cartesian x",
    y_label: "k Cartesian y",
    series: payload.series.map((/** @type {GraphSeries} */ series) => ({
      ...series,
      points: series.points.map((/** @type {GraphPoint} */ point) => {
        const cart = kBasisToCartesian(point.x, point.y);
        const rotated = rotatePoint(cart.x, cart.y, angle);
        return { ...point, x: rotated.x, y: rotated.y };
      }),
    })),
  };
}

/**
 * @param {{showAxes?: boolean, kspace?: boolean}} [options]
 * @returns {{width:number, height:number, margin:{left:number, right:number, top:number, bottom:number}, innerW:number, innerH:number}}
 */
function graphLayout(options = {}) {
  const showAxes = options.showAxes !== false;
  const width = showAxes ? 1000 : 720;
  const height = showAxes ? 520 : 720;
  const margin = showAxes
    ? { left: 78, right: 150, top: 28, bottom: 62 }
    : { left: 24, right: 24, top: 24, bottom: 24 };
  const innerW = width - margin.left - margin.right;
  const innerH = height - margin.top - margin.bottom;

  return { width, height, margin, innerW, innerH };
}

/**
 * @param {GraphView} view
 * @param {{left:number, right:number, top:number, bottom:number}} margin
 * @param {number} innerW
 * @param {number} innerH
 * @returns {{sx:(x:number)=>number, sy:(y:number)=>number}}
 */
function graphProjector(view, margin, innerW, innerH) {
  const { xmin, xmax, ymin, ymax } = view;

  return {
    sx: (x) => margin.left + ((x - xmin) / (xmax - xmin)) * innerW,
    sy: (y) => margin.top + ((ymax - y) / (ymax - ymin)) * innerH,
  };
}

/**
 * Side lengths, in rendered SVG pixels, for the first six k-space points
 * after Cartesian conversion, active-view expansion, and graph projection.
 *
 * @param {GraphPayload} payload
 * @param {{showAxes?: boolean, kspace?: boolean}} [options]
 * @returns {number[]}
 */
export function projectedKspaceHexagonSideLengths(payload, options = { kspace: true }) {
  const displayPayload = kspacePayloadToCartesian(payload, 0);
  const { margin, innerW, innerH } = graphLayout(options);
  const bounds = graphBounds(displayPayload);
  const activeView = options.kspace ? equalPixelAspectView(bounds, innerW, innerH) : bounds;
  const { sx, sy } = graphProjector(activeView, margin, innerW, innerH);

  const points = displayPayload.series[0].points.slice(0, 6).map((/** @type {GraphPoint} */ point) => ({
    x: sx(point.x),
    y: sy(point.y),
  }));

  return points.map((/** @type {{x:number, y:number}} */ point, /** @type {number} */ index) => {
    const next = points[(index + 1) % points.length];
    return Math.hypot(point.x - next.x, point.y - next.y);
  });
}

/**
 * @param {GraphPayload} payload
 * @param {GraphView} view
 * @param {number} fx
 * @param {number} fy
 * @returns {CursorHit | null}
 */
function nearestPathPoint(payload, view, fx, fy) {
  const targetX = view.xmin + fx * (view.xmax - view.xmin);
  const targetY = view.ymax - fy * (view.ymax - view.ymin);
  const selected = payload.series.find((series) => series.name === "selected path");

  if (!selected) return null;

  let best = null;
  let bestDistance = Number.POSITIVE_INFINITY;

  for (const point of selected.points) {
    if (!Number.isFinite(point.x) || !Number.isFinite(point.y)) continue;

    const distance = Math.hypot(point.x - targetX, point.y - targetY);
    if (distance < bestDistance) {
      bestDistance = distance;
      best = {
        series: selected.name,
        x: point.x,
        y: point.y,
        sx: (point.x - view.xmin) / (view.xmax - view.xmin),
        sy: (view.ymax - point.y) / (view.ymax - view.ymin),
        step: numberOrNull(point.meta?.step),
        pathX: numberOrNull(point.meta?.x),
        label: point.label ?? null,
      };
    }
  }

  return best;
}

/**
 * @param {GraphPayload} payload
 * @param {GraphView} view
 * @returns {CursorHit[]}
 */
function selectedPathHits(payload, view) {
  const selected = payload.series.find((series) => series.name === "selected path");
  if (!selected) return [];

  const selectedStepSet = new Set(selectedSteps().map((item) => item.step));
  /** @type {CursorHit[]} */
  const hits = [];

  for (const point of selected.points) {
    const step = numberOrNull(point.meta?.step);
    if (step === null || !selectedStepSet.has(step)) continue;

    hits.push({
      series: selected.name,
      x: point.x,
      y: point.y,
      sx: (point.x - view.xmin) / (view.xmax - view.xmin),
      sy: (view.ymax - point.y) / (view.ymax - view.ymin),
      step,
      pathX: numberOrNull(point.meta?.x),
    });
  }

  return hits;
}

/**
 * @param {GraphPayload} payload
 * @param {GraphView} view
 * @param {number} fx
 * @param {number} [fy]
 * @returns {CursorHit | null}
 */
function nearestPointByX(payload, view, fx, fy = 0.5) {
  const targetX = view.xmin + fx * (view.xmax - view.xmin);
  const targetY = view.ymax - fy * (view.ymax - view.ymin);

  let best = null;
  let bestDistance = Number.POSITIVE_INFINITY;

  for (const series of payload.series) {
    const points = series.points
      .filter((point) => Number.isFinite(point.x) && Number.isFinite(point.y))
      .slice()
      .sort((a, b) => a.x - b.x);

    if (points.length === 0) continue;

    /** @type {{x:number, y:number, step?:number|null} | null} */
    let candidate = null;

    for (let i = 0; i < points.length - 1; i++) {
      const a = points[i];
      const b = points[i + 1];

      const xmin = Math.min(a.x, b.x);
      const xmax = Math.max(a.x, b.x);
      if (targetX < xmin || targetX > xmax) continue;

      const dx = b.x - a.x;
      if (dx === 0) {
        const stepA = numberOrNull(a.meta?.step);
        candidate = { x: targetX, y: 0.5 * (a.y + b.y), step: stepA };
      } else {
        const t = (targetX - a.x) / dx;
        const stepA = numberOrNull(a.meta?.step);
        const stepB = numberOrNull(b.meta?.step);
        const step = stepA !== null && stepB !== null
          ? Math.round(stepA + t * (stepB - stepA))
          : stepA;
        candidate = { x: targetX, y: a.y + t * (b.y - a.y), step };
      }
      break;
    }

    if (candidate === null) {
      let nearest = points[0];
      let nearestDistance = Math.abs(nearest.x - targetX);

      for (const point of points.slice(1)) {
        const distance = Math.abs(point.x - targetX);
        if (distance < nearestDistance) {
          nearest = point;
          nearestDistance = distance;
        }
      }

      candidate = { x: nearest.x, y: nearest.y, step: numberOrNull(nearest.meta?.step) };
    }

    const distance = Math.abs(candidate.y - targetY);
    if (distance < bestDistance) {
      bestDistance = distance;
      best = {
        series: series.name,
        x: candidate.x,
        y: candidate.y,
        sx: (candidate.x - view.xmin) / (view.xmax - view.xmin),
        sy: (view.ymax - candidate.y) / (view.ymax - view.ymin),
        step: candidate.step ?? null,
        pathX: candidate.x,
      };
    }
  }

  return best;
}

/**
 * @param {string} name
 * @param {Record<string, string>} attrs
 * @returns {SVGElement}
 */
function svgEl(name, attrs = {}) {
  const el = document.createElementNS("http://www.w3.org/2000/svg", name);
  for (const [key, value] of Object.entries(attrs)) {
    el.setAttribute(key, value);
  }
  return el;
}

/**
 * @param {SVGSVGElement} svg
 * @param {PointerEvent | WheelEvent} event
 * @returns {{x:number, y:number}}
 */
function svgUserPoint(svg, event) {
  const point = svg.createSVGPoint();
  point.x = event.clientX;
  point.y = event.clientY;

  const matrix = svg.getScreenCTM();
  if (matrix === null) {
    return { x: 0, y: 0 };
  }

  const transformed = point.matrixTransform(matrix.inverse());
  return { x: transformed.x, y: transformed.y };
}

/**
 * @param {SVGSVGElement} svg
 * @param {PointerEvent | WheelEvent} event
 * @param {boolean} showAxes
 * @returns {{fx:number, fy:number}}
 */
function plotFractionsFromPointer(svg, event, showAxes) {
  const width = showAxes ? 1000 : 720;
  const height = showAxes ? 520 : 720;
  const margin = showAxes
    ? { left: 78, right: 150, top: 28, bottom: 62 }
    : { left: 24, right: 24, top: 24, bottom: 24 };
  const innerW = width - margin.left - margin.right;
  const innerH = height - margin.top - margin.bottom;

  const point = svgUserPoint(svg, event);
  return {
    fx: Math.min(1, Math.max(0, (point.x - margin.left) / innerW)),
    fy: Math.min(1, Math.max(0, (point.y - margin.top) / innerH)),
  };
}

/**
 * @param {GraphPayload} payload
 * @param {GraphView | null} [view]
 * @param {CursorHit | null} [cursor]
 * @param {{showAxes?: boolean, kspace?: boolean}} [options]
 * @returns {SVGSVGElement}
 */
function makeGraphSvg(payload, view = null, cursor = null, options = {}) {
  const showAxes = options.showAxes !== false;
  const width = showAxes ? 1000 : 720;
  const height = showAxes ? 520 : 720;
  const margin = showAxes
    ? { left: 78, right: 150, top: 28, bottom: 62 }
    : { left: 24, right: 24, top: 24, bottom: 24 };
  const innerW = width - margin.left - margin.right;
  const innerH = height - margin.top - margin.bottom;

  const svg = /** @type {SVGSVGElement} */ (svgEl("svg", {
    class: options.kspace ? "graph-svg graph-svg-component kspace-svg" : "graph-svg graph-svg-component",
    width: String(width),
    height: String(height),
    viewBox: `0 0 ${width} ${height}`,
    preserveAspectRatio: "xMidYMid meet",
    role: "img",
    "aria-label": payload.title || "Graph",
  }));

  svg.appendChild(svgEl("rect", {
    class: "graph-bg",
    x: "0",
    y: "0",
    width: String(width),
    height: String(height),
    fill: "#fffdf8",
  }));

  const bounds = graphBounds(payload);
  const rawView = view || bounds;
  const activeView = options.kspace ? equalPixelAspectView(rawView, innerW, innerH) : rawView;
  const { xmin, xmax, ymin, ymax } = activeView;

  const { sx, sy } = graphProjector(activeView, margin, innerW, innerH);

  const clipId = `plot-clip-${payload.id}`;
  const defs = svgEl("defs");
  const clipPath = svgEl("clipPath", { id: clipId });
  clipPath.appendChild(svgEl("rect", {
    x: String(margin.left),
    y: String(margin.top),
    width: String(innerW),
    height: String(innerH),
  }));
  defs.appendChild(clipPath);
  svg.appendChild(defs);

  const dataLayer = showAxes ? svgEl("g", { "clip-path": `url(#${clipId})` }) : svgEl("g");

  if (showAxes) {
  for (let i = 0; i <= 5; i++) {
      const t = i / 5;
      const x = margin.left + t * innerW;
      const xv = xmin + t * (xmax - xmin);
  
      svg.appendChild(svgEl("line", {
        class: "grid",
        x1: String(x),
        y1: String(margin.top),
        x2: String(x),
        y2: String(margin.top + innerH),
      }));
  
      const label = svgEl("text", {
        class: "axis-label",
        x: String(x),
        y: String(height - 24),
        "text-anchor": "middle",
      });
      label.textContent = nice(xv);
      svg.appendChild(label);
    }
  
    for (let i = 0; i <= 5; i++) {
      const t = i / 5;
      const y = margin.top + t * innerH;
      const yv = ymax - t * (ymax - ymin);
  
      svg.appendChild(svgEl("line", {
        class: "grid",
        x1: String(margin.left),
        y1: String(y),
        x2: String(margin.left + innerW),
        y2: String(y),
      }));
  
      const label = svgEl("text", {
        class: "axis-label",
        x: String(margin.left - 10),
        y: String(y + 4),
        "text-anchor": "end",
      });
      label.textContent = nice(yv);
      svg.appendChild(label);
    }
  
    svg.appendChild(svgEl("line", {
      class: "axis",
      x1: String(margin.left),
      y1: String(margin.top + innerH),
      x2: String(margin.left + innerW),
      y2: String(margin.top + innerH),
    }));
  
    svg.appendChild(svgEl("line", {
      class: "axis",
      x1: String(margin.left),
      y1: String(margin.top),
      x2: String(margin.left),
      y2: String(margin.top + innerH),
    }));
  
    const xTitle = svgEl("text", {
      class: "axis-title",
      x: String(margin.left + innerW / 2),
      y: String(height - 6),
      "text-anchor": "middle",
    });
    xTitle.textContent = payload.x_label || "x";
    svg.appendChild(xTitle);
  
    const yTitle = svgEl("text", {
      class: "axis-title",
      x: "18",
      y: String(margin.top + innerH / 2),
      "text-anchor": "middle",
      transform: `rotate(-90 18 ${margin.top + innerH / 2})`,
    });
    yTitle.textContent = payload.y_label || "y";
    svg.appendChild(yTitle);
  
    }

  const colours = [
    "#2563eb",
    "#dc2626",
    "#16a34a",
    "#9333ea",
    "#ea580c",
    "#0891b2",
    "#4f46e5",
    "#be123c",
    "#65a30d",
    "#7c3aed",
    "#0f766e",
    "#b45309",
  ];

  payload.series.forEach((series, seriesIndex) => {
    const colour = colours[seriesIndex % colours.length];
    const points = series.points
      .filter((p) => Number.isFinite(p.x) && Number.isFinite(p.y))
      .map((p) => [sx(p.x), sy(p.y)]);

    if (points.length === 0) return;

    if (series.kind === "line" || series.kind === "line_points") {
      const d = points.map(([x, y], i) => `${i === 0 ? "M" : "L"} ${x.toFixed(3)} ${y.toFixed(3)}`).join(" ");
      dataLayer.appendChild(svgEl("path", {
        class: "series-line",
        d,
        fill: "none",
        stroke: colour,
        "stroke-width": "1.8",
        "vector-effect": "non-scaling-stroke",
      }));
    }

    if (series.kind === "points" || series.kind === "line_points") {
      for (const [x, y] of points) {
        dataLayer.appendChild(svgEl("circle", {
          cx: String(x),
          cy: String(y),
          r: showAxes ? "2.5" : "4",
          fill: colour,
        }));
      }
    }

    const [lx, ly] = points[points.length - 1];
    const label = svgEl("text", {
      class: options.kspace ? "series-label kspace-label" : "series-label",
      x: String(lx + 10),
      y: String(ly - 8),
      fill: colour,
      stroke: "#ffffff",
      "stroke-width": options.kspace ? "5" : "3",
      "paint-order": "stroke",
    });
    label.textContent = series.name;
    svg.appendChild(label);
  });

  svg.appendChild(dataLayer);

  if (showAxes) {
    const legend = svgEl("g", { class: "graph-legend" });
    payload.series.forEach((series, seriesIndex) => {
      const colour = colours[seriesIndex % colours.length];
      const x = margin.left + innerW + 8;
      const y = margin.top + 18 + 18 * seriesIndex;

      legend.appendChild(svgEl("line", {
        x1: String(x),
        y1: String(y - 4),
        x2: String(x + 18),
        y2: String(y - 4),
        stroke: colour,
        "stroke-width": "2",
      }));

      const label = svgEl("text", {
        class: "legend-label",
        x: String(x + 24),
        y: String(y),
        fill: colour,
      });
      label.textContent = series.name;
      legend.appendChild(label);
    });
    svg.appendChild(legend);
  }

  const selected = selectedSteps();

  if (showAxes) {
    const markerLayer = svgEl("g", { "clip-path": `url(#${clipId})` });

    for (const item of selected) {
      if (item.pathX === null) continue;

      const markerX = sx(item.pathX);

      if (item.energy === null) {
        markerLayer.appendChild(svgEl("line", {
          class: "selected-symmetry-marker",
          x1: String(markerX),
          y1: String(margin.top),
          x2: String(markerX),
          y2: String(margin.top + innerH),
          stroke: "#ef4444",
          "stroke-width": "3",
          "stroke-dasharray": "2 4",
          "pointer-events": "none",
        }));
      } else if (Number.isFinite(item.energy)) {
        markerLayer.appendChild(svgEl("circle", {
          class: "selected-energy-overlay",
          cx: String(markerX),
          cy: String(sy(item.energy)),
          r: "8",
          fill: "#f59e0b",
          stroke: "#000000",
          "stroke-width": "2",
          "pointer-events": "none",
        }));
      }
    }

    svg.appendChild(markerLayer);
  }

  if (!showAxes) {
    for (const hit of selectedPathHits(payload, activeView)) {
      svg.appendChild(svgEl("circle", {
        class: "selected-kpoint-marker",
        cx: String(sx(hit.x)),
        cy: String(sy(hit.y)),
        r: "10",
        fill: "rgba(245, 158, 11, 0.18)",
        stroke: "#f59e0b",
        "stroke-width": "4",
        "pointer-events": "none",
      }));
    }
  }

  if (cursor) {
    const cx = sx(cursor.x);
    const cy = sy(cursor.y);

    svg.appendChild(svgEl("line", {
      class: "cursor-line",
      stroke: "#111827",
      "stroke-width": "1.2",
      "stroke-dasharray": "4 4",
      x1: String(cx),
      y1: String(margin.top),
      x2: String(cx),
      y2: String(margin.top + innerH),
    }));

    svg.appendChild(svgEl("line", {
      class: "cursor-line",
      stroke: "#111827",
      "stroke-width": "1.2",
      "stroke-dasharray": "4 4",
      x1: String(margin.left),
      y1: String(cy),
      x2: String(margin.left + innerW),
      y2: String(cy),
    }));

    svg.appendChild(svgEl("circle", {
      class: "cursor-point",
      fill: "#111827",
      stroke: "#ffffff",
      "stroke-width": "1.5",
      cx: String(cx),
      cy: String(cy),
      r: "4",
    }));

    const label = svgEl("text", {
      class: showAxes ? "cursor-label" : "cursor-label kspace-hover-label",
      fill: "#111827",
      x: showAxes ? String(Math.min(cx + 10, margin.left + innerW - 260)) : String(margin.left + 12),
      y: showAxes ? String(Math.max(cy - 10, margin.top + 18)) : String(margin.top + 30),
      stroke: "#ffffff",
      "stroke-width": showAxes ? "0" : "5",
      "paint-order": "stroke",
    });
    label.textContent = showAxes
      ? `${cursor.series}: x=${nice(cursor.x)}, y=${nice(cursor.y)}`
      : `${cursor.label ?? cursor.series}: step=${cursor.step ?? ""}, kx=${nice(cursor.x)}, ky=${nice(cursor.y)}`;
    svg.appendChild(label);
  }

  return svg;
}

function collectSelectedTableSteps() {
  const checked = Array.from(document.querySelectorAll(".table-step-select:checked"));
  return checked
    .map((input) => ({
      step: parseNumericAttribute(input.getAttribute("data-step")),
      pathX: parseNumericAttribute(input.getAttribute("data-path-x")),
      energy: parseNumericAttribute(input.getAttribute("data-energy")),
      label: input.getAttribute("data-label"),
    }))
    .filter((item) => item.step !== null)
    .map((item) => ({
      step: /** @type {number} */ (item.step),
      pathX: item.pathX,
      energy: item.energy,
      label: item.label,
    }));
}

function syncInitialTableSelection() {
  emitSelectedSteps(collectSelectedTableSteps(), document);
}

function installTableSelectionControls() {
  document.addEventListener("change", (event) => {
    const target = event.target;
    if (!(target instanceof HTMLInputElement)) return;
    if (!target.classList.contains("table-step-select")) return;

    emitSelectedSteps(collectSelectedTableSteps(), target);
  });

  document.addEventListener("click", (event) => {
    const target = event.target;
    if (!(target instanceof HTMLElement)) return;

    const action = target.getAttribute("data-table-select");
    if (action !== "all" && action !== "none") return;

    const controls = target.closest(".table-select-controls");
    const tableId = controls?.getAttribute("data-table-id");
    if (!tableId) return;

    const boxes = Array.from(document.querySelectorAll(`.table-step-select[data-table-id="${CSS.escape(tableId)}"]`));
    for (const box of boxes) {
      if (box instanceof HTMLInputElement) {
        box.checked = action === "all";
      }
    }

    emitSelectedSteps(collectSelectedTableSteps(), target);
  });
}

if (typeof document !== "undefined") {
  installTableSelectionControls();
  syncInitialTableSelection();
}

if (typeof HTMLElement !== "undefined" && typeof customElements !== "undefined") {
  class DftBandControls extends HTMLElement {
    connectedCallback() {
      if (this.dataset.bound === "1") return;
      this.dataset.bound = "1";

      const bandInput = this.querySelector("[data-dft-band]");
      const sliceAxisInput = this.querySelector("[data-dft-slice-axis]");
      const sliceValueInput = this.querySelector("[data-dft-slice-value]");
      const energyScaleInput = this.querySelector("[data-dft-energy-scale]");
      const rotationInput = this.querySelector("[data-dft-rotation]");

      if (bandInput) {
        bandInput.addEventListener("change", () => {
          emitDftSignal("selected-band", {
            band: Number(/** @type {HTMLInputElement | HTMLSelectElement} */ (bandInput).value),
          }, this);
        });
      }

      const emitSlice = () => {
        const axis = sliceAxisInput
          ? String(/** @type {HTMLInputElement | HTMLSelectElement} */ (sliceAxisInput).value)
          : "u";
        const value = sliceValueInput
          ? Number(/** @type {HTMLInputElement | HTMLSelectElement} */ (sliceValueInput).value)
          : 0.0;

        emitDftSignal("slice-changed", { axis, value }, this);
      };

      if (sliceAxisInput) sliceAxisInput.addEventListener("change", emitSlice);
      if (sliceValueInput) sliceValueInput.addEventListener("input", emitSlice);
      if (sliceValueInput) sliceValueInput.addEventListener("change", emitSlice);

      const emitView = () => {
        const energyScale = energyScaleInput
          ? Number(/** @type {HTMLInputElement | HTMLSelectElement} */ (energyScaleInput).value)
          : 1.0;
        const rotation = rotationInput
          ? Number(/** @type {HTMLInputElement | HTMLSelectElement} */ (rotationInput).value)
          : 0.0;

        emitDftSignal("view-changed", { energyScale, rotation }, this);
      };

      if (energyScaleInput) energyScaleInput.addEventListener("input", emitView);
      if (energyScaleInput) energyScaleInput.addEventListener("change", emitView);
      if (rotationInput) rotationInput.addEventListener("input", emitView);
      if (rotationInput) rotationInput.addEventListener("change", emitView);
    }
  }


  class DftBandReadout extends HTMLElement {
    constructor() {
      super();
      /** @type {number | null} */
      this.selectedBand = null;
      /** @type {string | null} */
      this.sliceAxis = null;
      /** @type {number | null} */
      this.sliceValue = null;
      /** @type {number} */
      this.energyScale = 1.0;
      /** @type {Array<() => void>} */
      this.unsubscribers = [];
    }

    connectedCallback() {
      if (this.dataset.bound === "1") return;
      this.dataset.bound = "1";

      this.unsubscribers.push(onDftSignal("selected-band", (payload) => {
        this.selectedBand = Number(payload.detail.band);
        this.render();
      }));

      this.unsubscribers.push(onDftSignal("slice-changed", (payload) => {
        this.sliceAxis = String(payload.detail.axis ?? "");
        this.sliceValue = Number(payload.detail.value);
        this.render();
      }));

      this.unsubscribers.push(onDftSignal("view-changed", (payload) => {
        const energyScale = Number(payload.detail.energyScale);
        const rotation = Number(payload.detail.rotation);
        const pitch = Number(payload.detail.pitch);
        const viewZoom = Number(payload.detail.viewZoom);
        this.energyScale = Number.isFinite(energyScale) && energyScale > 0 ? energyScale : this.energyScale;
        this.rotation = Number.isFinite(rotation) ? rotation : this.rotation;
        this.pitch = Number.isFinite(pitch) ? Math.max(0.05, Math.min(1.45, pitch)) : this.pitch;
        this.viewZoom = Number.isFinite(viewZoom) && viewZoom > 0 ? Math.max(0.2, Math.min(5.0, viewZoom)) : this.viewZoom;
        this.render();
      }));

      this.unsubscribers.push(onDftSignal("selected-kpoint", (payload) => {
        this.selectedKpoint = payload.detail;
        this.render();
      }));

      this.render();
    }

    disconnectedCallback() {
      for (const unsubscribe of this.unsubscribers) unsubscribe();
      this.unsubscribers = [];
      delete this.dataset.bound;
    }

    render() {
      const band = this.selectedBand === null || !Number.isFinite(this.selectedBand)
        ? "none"
        : String(this.selectedBand);
      const slice = this.sliceAxis === null || this.sliceValue === null || !Number.isFinite(this.sliceValue)
        ? "none"
        : `${this.sliceAxis}=${nice(this.sliceValue)}`;

      this.textContent = `band: ${band}; slice: ${slice}`;
    }
  }


  class DftKPointReadout extends HTMLElement {
    constructor() {
      super();
      /** @type {Array<() => void>} */
      this.unsubscribers = [];
      /** @type {null | Record<string, unknown>} */
      this.selected = null;
    }

    connectedCallback() {
      if (this.dataset.bound === "1") return;
      this.dataset.bound = "1";

      this.unsubscribers.push(onDftSignal("selected-kpoint", (payload) => {
        this.selected = payload.detail;
        this.render();
      }));

      this.render();
    }

    disconnectedCallback() {
      for (const unsubscribe of this.unsubscribers) unsubscribe();
      this.unsubscribers = [];
      delete this.dataset.bound;
    }

    render() {
      if (!this.selected) {
        this.textContent = "selected k-point: none";
        return;
      }

      const i = Number(this.selected.i);
      const j = Number(this.selected.j);
      const band = Number(this.selected.band);
      const k1 = Number(this.selected.k1);
      const k2 = Number(this.selected.k2);
      const energy = Number(this.selected.energy);

      this.textContent = `selected k-point: i=${i}, j=${j}, band=${band}, k1=${nice(k1)}, k2=${nice(k2)}, E=${nice(energy)}`;
    }
  }


  class DftBandSurfaceViewer extends HTMLElement {
    constructor() {
      super();

      /** @type {number | null} */
      this.selectedBand = null;
      /** @type {string | null} */
      this.sliceAxis = null;
      /** @type {number | null} */
      this.sliceValue = null;
      /** @type {null | Record<string, unknown>} */
      this.selectedKpoint = null;

      /** @type {ResizeObserver | null} */
      this.resizeObserver = null;
      /** @type {any} */
      this.referenceGroup = null;
      /** @type {boolean} */
      this.maskToHexagon = false;
      /** @type {boolean} */
      this.hasInitialCamera = false;
      /** @type {Set<number>} */
      this.hiddenBands = new Set();
      /** @type {Array<() => void>} */
      this.unsubscribers = [];

      /** @type {JsonPayload | null} */
      this.payload = null;
      /** @type {null | ReturnType<typeof bandSurfaceMeshData>} */
      this.currentMesh = null;
      /** @type {Array<{band:number, mesh:ReturnType<typeof bandSurfaceMeshData>}>} */
      this.currentBandMeshes = [];

      /** @type {HTMLElement | null} */
      this.statusEl = null;
      /** @type {HTMLElement | null} */
      this.hoverEl = null;
      /** @type {HTMLElement | null} */
      this.legendEl = null;
      /** @type {HTMLElement | null} */
      this.threeHost = null;

      /** @type {any} */
      this.THREE = null;
      /** @type {any} */
      this.OrbitControls = null;
      /** @type {any} */
      this.renderer = null;
      /** @type {any} */
      this.scene = null;
      /** @type {any} */
      this.camera = null;
      /** @type {any} */
      this.controls = null;
      /** @type {any[]} */
      this.surfaceMeshes = [];
      /** @type {any[]} */
      this.wireMeshes = [];
      /** @type {any} */
      this.selectedMarker = null;
      /** @type {number | null} */
      this.animationFrame = null;
    }

    connectedCallback() {
      if (this.dataset.bound === "1") return;
      this.dataset.bound = "1";

      this.payload = readJsonPayload(/** @type {Element} */ (this));
      this.buildStaticDom();

      this.unsubscribers.push(onDftSignal("selected-band", (payload) => {
        this.selectedBand = Number(payload.detail.band);
        this.requestSurfaceUpdate();
      }));

      this.unsubscribers.push(onDftSignal("slice-changed", (payload) => {
        this.sliceAxis = String(payload.detail.axis ?? "");
        this.sliceValue = Number(payload.detail.value);
        this.updateStatus();
      }));

      this.unsubscribers.push(onDftSignal("selected-kpoint", (payload) => {
        this.selectedKpoint = payload.detail;
        this.updateSelectedMarker();
        this.updateStatus();
      }));

      this.requestSurfaceUpdate();
    }

    disconnectedCallback() {
      for (const unsubscribe of this.unsubscribers) unsubscribe();
      this.unsubscribers = [];
      this.disposeThree();
      delete this.dataset.bound;
    }

    buildStaticDom() {
      this.innerHTML = `
        <div class="band-surface-viewer-three-only">
          <div class="band-surface-status" data-dft-surface-status></div>
          <div class="band-surface-controls">
            <label class="band-surface-mask-toggle">
              <input type="checkbox" data-dft-mask-to-hexagon>
              mask to hexagon
            </label>
          </div>
          <div class="band-surface-legend" data-dft-surface-legend></div>
          <div class="band-surface-hover" data-dft-surface-hover>hover: none</div>
          <div class="band-surface-three" data-dft-three-surface style="width:100%; min-height:560px;"></div>
          <p class="band-surface-help">three.js controls: left drag rotate, wheel zoom, right drag pan.</p>
        </div>
      `;

      this.statusEl = this.querySelector("[data-dft-surface-status]");
      this.hoverEl = this.querySelector("[data-dft-surface-hover]");
      this.legendEl = this.querySelector("[data-dft-surface-legend]");
      this.threeHost = this.querySelector("[data-dft-three-surface]");
      this.bindMaskToggle();
    }

    bindMaskToggle() {
      const input = this.querySelector("[data-dft-mask-to-hexagon]");
      if (!(input instanceof HTMLInputElement)) return;
      if (input.dataset.bound === "1") return;

      input.checked = this.maskToHexagon;
      input.dataset.bound = "1";
      const update = () => {
        this.maskToHexagon = input.checked;
        this.requestSurfaceUpdate();
      };

      input.addEventListener("input", update);
      input.addEventListener("change", update);
    }

selectedBandIndex() {
      if (this.selectedBand !== null && Number.isFinite(this.selectedBand)) {
        return this.selectedBand;
      }

      return Number(this.payload?.selected_band ?? 0);
    }

    async ensureThree() {
      if (this.renderer && this.scene && this.camera && this.controls) return;
      if (!(this.threeHost instanceof HTMLElement)) return;

      const { THREE, OrbitControls } = await loadThreeRuntime();
      if (!this.isConnected) return;

      this.THREE = THREE;
      this.OrbitControls = OrbitControls;

      const width = Math.max(360, this.threeHost.clientWidth || 800);
      const height = 560;

      const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
      renderer.setSize(width, height);
      renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
      renderer.domElement.style.width = "100%";
      renderer.domElement.style.height = `${height}px`;

      this.threeHost.replaceChildren(renderer.domElement);

      const scene = new THREE.Scene();
      const camera = new THREE.PerspectiveCamera(45, width / height, 0.001, 100000);
      const controls = new OrbitControls(camera, renderer.domElement);

      controls.enableDamping = true;
      controls.dampingFactor = 0.08;
      controls.screenSpacePanning = true;

      scene.add(new THREE.AmbientLight(0xffffff, 0.55));

      const light = new THREE.DirectionalLight(0xffffff, 1.2);
      light.position.set(1, 2, 2);
      scene.add(light);

      this.renderer = renderer;
      this.scene = scene;
      this.camera = camera;
      this.controls = controls;

      this.resizeThreeSurface();

      if (this.resizeObserver) this.resizeObserver.disconnect();
      this.resizeObserver = new ResizeObserver(() => this.resizeThreeSurface());
      this.resizeObserver.observe(this.threeHost);

      renderer.domElement.addEventListener("pointermove", (/** @type {PointerEvent} */ event) => this.handlePointerMove(event));
      renderer.domElement.addEventListener("click", (/** @type {MouseEvent} */ event) => this.handleClick(event));

      this.startThreeLoop();
    }

    startThreeLoop() {
      if (!this.renderer || !this.scene || !this.camera || !this.controls) return;

      this.controls.update();
      this.renderer.render(this.scene, this.camera);
      this.animationFrame = requestAnimationFrame(() => this.startThreeLoop());
    }

    resizeThreeSurface() {
      if (!this.renderer || !this.camera || !(this.threeHost instanceof HTMLElement)) return;

      const rect = this.threeHost.getBoundingClientRect();
      const width = Math.max(320, Math.floor(rect.width || this.threeHost.clientWidth || 720));
      const height = 560;

      this.renderer.setSize(width, height, false);
      this.camera.aspect = width / height;
      this.camera.updateProjectionMatrix();
    }

    disposeThree() {
      if (this.animationFrame !== null) {
        cancelAnimationFrame(this.animationFrame);
        this.animationFrame = null;
      }

      if (this.resizeObserver) {
        this.resizeObserver.disconnect();
        this.resizeObserver = null;
      }

      if (this.controls && typeof this.controls.dispose === "function") this.controls.dispose();
      if (this.renderer && typeof this.renderer.dispose === "function") this.renderer.dispose();

      this.renderer = null;
      this.scene = null;
      this.camera = null;
      this.controls = null;
      this.surfaceMeshes = [];
      this.wireMeshes = [];
      this.selectedMarker = null;
      this.hasInitialCamera = false;
    }

    clearBandSurfaceMeshes() {
      if (!this.scene) return;

      for (const mesh of this.surfaceMeshes) this.scene.remove(mesh);
      for (const mesh of this.wireMeshes) this.scene.remove(mesh);

      this.surfaceMeshes = [];
      this.wireMeshes = [];
    }

    updateLegend() {
      if (!this.legendEl) return;

      const bands = allBandIndices(this.payload);
      if (bands.length === 0) {
        this.legendEl.textContent = "bands: none";
        return;
      }

      this.legendEl.replaceChildren();

      for (const band of bands) {
        const hidden = this.hiddenBands.has(band);
        const button = document.createElement("button");
        button.type = "button";
        button.className = hidden
          ? "band-surface-legend-item band-surface-legend-item-hidden"
          : "band-surface-legend-item";
        button.dataset.band = String(band);
        button.style.marginRight = "0.4rem";
        const visible = visibleBandIndices(this.payload, this.hiddenBands);
        const isLastVisible = !hidden && visible.length <= 1;
        button.disabled = isLastVisible;
        button.title = isLastVisible ? "At least one band must remain visible" : "";
        button.style.opacity = hidden ? "0.55" : "1.0";
        button.textContent = `band ${band}`;

        const swatch = document.createElement("span");
        swatch.className = "band-surface-legend-swatch";
        swatch.style.display = "inline-block";
        swatch.style.width = "0.9em";
        swatch.style.height = "0.9em";
        swatch.style.marginRight = "0.35em";
        swatch.style.verticalAlign = "-0.1em";
        swatch.style.background = hidden
          ? "#808080"
          : `#${bandSurfaceColor(band).toString(16).padStart(6, "0")}`;

        button.prepend(swatch);
        button.addEventListener("click", () => {
          if (this.hiddenBands.has(band)) {
            this.hiddenBands.delete(band);
          } else {
            const visible = visibleBandIndices(this.payload, this.hiddenBands);
            if (visible.length <= 1 && visible.includes(band)) {
              return;
            }
            this.hiddenBands.add(band);
          }
          this.requestSurfaceUpdate();
        });

        this.legendEl.appendChild(button);
      }
    }



    requestSurfaceUpdate() {
      void this.updateSurface().catch((error) => {
        const message = error instanceof Error ? error.message : String(error);
        if (this.statusEl) {
          this.statusEl.textContent = `surface update failed: ${message}`;
        }
        throw error;
      });
    }

    async updateSurface() {
      const maskInput = this.querySelector("[data-dft-mask-to-hexagon]");
      if (maskInput instanceof HTMLInputElement) {
        this.maskToHexagon = maskInput.checked;
      }

      const visibleBands = visibleBandIndices(this.payload, this.hiddenBands);

      this.currentBandMeshes = visibleBands.map((band) => ({
        band,
        mesh: bandSurfaceMeshDataWithMask(this.payload, band, this.maskToHexagon),
      }));
      this.currentMesh = this.currentBandMeshes[0]?.mesh ?? null;

      this.updateStatus();
      this.updateLegend();

      const drawable = this.currentBandMeshes.filter((item) => (
        item.mesh.vertices.length > 0 && item.mesh.triangles.length > 0
      ));

      if (drawable.length === 0) {
        this.clearBandSurfaceMeshes();
        this.updateLegend();
        if (this.statusEl) {
          const allBands = allBandIndices(this.payload);
          this.statusEl.textContent = `visible 0; hidden ${this.hiddenBands.size}; bands ${allBands.length}; no visible bands`;
        }
        return;
      }

      await this.ensureThree();
      if (!this.THREE || !this.scene || !this.camera || !this.controls) return;

      const THREE = this.THREE;
      this.clearBandSurfaceMeshes();

      let firstData = null;

      for (const item of drawable) {
        const data = threeBandSurfaceGeometryData(item.mesh);
        if (!firstData) firstData = data;

        const geometry = new THREE.BufferGeometry();
        geometry.setAttribute("position", new THREE.BufferAttribute(data.positions, 3));
        geometry.setIndex(new THREE.BufferAttribute(data.indices, 1));
        geometry.computeVertexNormals();

        const color = bandSurfaceColor(item.band);
        const material = new THREE.MeshStandardMaterial({
          color,
          side: THREE.DoubleSide,
          transparent: true,
          opacity: drawable.length === 1 ? 0.82 : 0.52,
          roughness: 0.72,
          metalness: 0.0,
        });

        const wireMaterial = new THREE.MeshBasicMaterial({
          color,
          wireframe: true,
          transparent: true,
          opacity: drawable.length === 1 ? 0.22 : 0.30,
        });

        const surface = new THREE.Mesh(geometry, material);
        const wire = new THREE.Mesh(geometry, wireMaterial);

        this.surfaceMeshes.push(surface);
        this.wireMeshes.push(wire);
        this.scene.add(surface);
        this.scene.add(wire);
      }

      if (firstData) {
        this.addReferenceObjects(firstData);
        this.resetCameraIfNeeded(firstData);
      }

      this.updateSelectedMarker();
    }


    /**
     * @param {{center:{x:number,y:number,z:number}, radius:number}} data
     */
    addReferenceObjects(data) {
      const THREE = this.THREE;
      if (!THREE || !this.scene) return;

      if (this.referenceGroup) {
        this.scene.remove(this.referenceGroup);
        this.referenceGroup = null;
      }

      const radius = Number.isFinite(data.radius) && data.radius > 0 ? data.radius : 1.0;
      const group = new THREE.Group();
      group.name = "band-surface-reference-group";

      // Reference plane is the uv/k plane.  It is deliberately at y=0, so
      // it passes through the OrbitControls target and is easy to compare
      // against the band surface.
      const planeSize = 2.4 * radius;
      const plane = new THREE.Mesh(
        new THREE.PlaneGeometry(planeSize, planeSize),
        new THREE.MeshBasicMaterial({
          color: 0x101820,
          transparent: true,
          opacity: 0.22,
          side: THREE.DoubleSide,
          depthWrite: false,
        }),
      );
      plane.rotation.x = -Math.PI / 2.0;
      plane.position.set(data.center.x, 0.0, data.center.z);
      group.add(plane);

      const uvGridLines = threeUvGridReferenceData(Math.PI, 8);
      const uvGridMaterial = new THREE.LineBasicMaterial({
        color: 0x4e6e81,
        transparent: true,
        opacity: 0.42,
        depthTest: false,
      });

      for (const line of uvGridLines) {
        const lineGeom = new THREE.BufferGeometry().setFromPoints([
          new THREE.Vector3(line[0].x, 0.006 * radius, line[0].z),
          new THREE.Vector3(line[1].x, 0.006 * radius, line[1].z),
        ]);
        group.add(new THREE.Line(lineGeom, uvGridMaterial));
      }

      const hex = threeHexagonReferenceData(this.payload);
      if (hex.length >= 3) {
        const pts = hex.map((p) => new THREE.Vector3(p.x, 0.0, p.z));
        pts.push(new THREE.Vector3(hex[0].x, 0.0, hex[0].z));

        const hexGeom = new THREE.BufferGeometry().setFromPoints(pts);
        const hexLine = new THREE.Line(
          hexGeom,
          new THREE.LineBasicMaterial({
            color: 0xffb000,
            transparent: true,
            opacity: 1.0,
            depthTest: false,
          }),
        );
        group.add(hexLine);

        // radial spokes make the hexagon visibly non-square
        for (const p of hex) {
          const spoke = new THREE.BufferGeometry().setFromPoints([
            new THREE.Vector3(0.0, 0.0, 0.0),
            new THREE.Vector3(p.x, 0.0, p.z),
          ]);
          group.add(new THREE.Line(
            spoke,
            new THREE.LineBasicMaterial({
              color: 0xffd166,
              transparent: true,
              opacity: 0.45,
              depthTest: false,
            }),
          ));
        }
      }

      const axes = new THREE.AxesHelper(0.8 * radius);
      axes.position.set(0.0, 0.0, 0.0);
      group.add(axes);

      this.referenceGroup = group;
      this.scene.add(group);
    }


/**
     * @param {{center:{x:number,y:number,z:number}, radius:number}} data
     */
    resetCameraIfNeeded(data) {
      if (this.hasInitialCamera) return;
      if (!this.camera || !this.controls) return;

      this.camera.position.set(
        data.center.x + 1.3 * data.radius,
        data.center.y + 0.9 * data.radius,
        data.center.z + 1.3 * data.radius,
      );
      this.camera.lookAt(data.center.x, 0.0, data.center.z);

      this.controls.target.set(data.center.x, 0.0, data.center.z);
      this.controls.update();
      this.hasInitialCamera = true;
    }

    updateStatus() {
      const mesh = this.currentMesh;
      const band = this.selectedBandIndex();
      const visibleCount = this.currentBandMeshes.length;
      const totalVertices = this.currentBandMeshes.reduce((sum, item) => sum + item.mesh.vertices.length, 0);
      const totalTriangles = this.currentBandMeshes.reduce((sum, item) => sum + item.mesh.triangles.length, 0);
      const bands = /** @type {unknown[] | undefined} */ (this.payload?.bands);
      const nbands = Number(this.payload?.nbands ?? NaN);
      const nu = Number(this.payload?.nu ?? NaN);
      const nv = Number(this.payload?.nv ?? NaN);

      const gridText = Number.isFinite(nu) && Number.isFinite(nv) ? `${nu}×${nv}` : "unknown";
      const bandsText = Number.isFinite(nbands)
        ? String(nbands)
        : Array.isArray(bands) ? String(bands.length) : "unknown";
      const energyText = !mesh || mesh.summary.zmin === null || mesh.summary.zmax === null
        ? "unknown"
        : `${nice(mesh.summary.zmin)} to ${nice(mesh.summary.zmax)}`;
      const slice = this.sliceAxis === null || this.sliceValue === null || !Number.isFinite(this.sliceValue)
        ? "none"
        : `${this.sliceAxis}=${nice(this.sliceValue)}`;

      if (this.statusEl) {
        const maskText = this.maskToHexagon ? "hex mask on" : "hex mask off";
        const hiddenCount = this.hiddenBands.size;
        this.statusEl.textContent = `band ${band}; visible ${visibleCount}; hidden ${hiddenCount}; grid ${gridText}; bands ${bandsText}; vertices ${totalVertices}; triangles ${totalTriangles}; energy ${energyText}; slice ${slice}; ${maskText}`;
      }
    }

    /**
     * @param {PointerEvent} event
     */
    handlePointerMove(event) {
      const hit = this.pickNearestVertex(event, 0.06);
      if (!this.hoverEl) return;

      if (!hit) {
        this.hoverEl.textContent = "hover: none";
        return;
      }

      const v = hit.vertex;
      this.hoverEl.textContent = `hover: i=${v.i}, j=${v.j}, band=${v.band}, k1=${nice(v.x)}, k2=${nice(v.y)}, E=${nice(v.z)}`;
    }

    /**
     * @param {MouseEvent} event
     */
    handleClick(event) {
      const hit = this.pickNearestVertex(event, 0.08);
      if (!hit) return;

      const v = hit.vertex;
      emitDftSignal("selected-kpoint", {
        i: v.i,
        j: v.j,
        band: v.band,
        k1: v.x,
        k2: v.y,
        energy: v.z,
      }, this);
    }

    /**
     * @param {MouseEvent | PointerEvent} event
     * @param {number} maxDistance
     */
    pickNearestVertex(event, maxDistance = 0.08) {
      if (this.currentBandMeshes.length === 0 || !this.renderer || !this.camera || !this.THREE) return null;

      const rect = this.renderer.domElement.getBoundingClientRect();
      const pointer = {
        x: ((event.clientX - rect.left) / rect.width) * 2 - 1,
        y: -(((event.clientY - rect.top) / rect.height) * 2 - 1),
      };

      let best = null;
      let bestDistance = maxDistance;

      for (const item of this.currentBandMeshes) {
        for (const vertex of item.mesh.vertices) {
          const display = bandBasisToCartesian(vertex.x, vertex.y);
          const p = new this.THREE.Vector3(display.x, vertex.z, display.y);
          p.project(this.camera);

          const distance = Math.hypot(pointer.x - p.x, pointer.y - p.y);
          if (distance <= bestDistance) {
            bestDistance = distance;
            best = { vertex, distance };
          }
        }
      }

      return best;
    }

    updateSelectedMarker() {
      if (!this.THREE || !this.scene || this.currentBandMeshes.length === 0) return;

      if (this.selectedMarker) {
        this.scene.remove(this.selectedMarker);
        this.selectedMarker = null;
      }

      if (!this.selectedKpoint) return;

      const i = Number(this.selectedKpoint.i);
      const j = Number(this.selectedKpoint.j);
      const band = Number(this.selectedKpoint.band);

      let vertex = null;
      for (const item of this.currentBandMeshes) {
        vertex = item.mesh.vertices.find((v) => v.i === i && v.j === j && v.band === band) ?? null;
        if (vertex) break;
      }
      if (!vertex) return;

      const THREE = this.THREE;
      const display = bandBasisToCartesian(vertex.x, vertex.y);
      const geometry = new THREE.SphereGeometry(0.035 * Math.max(1, Math.abs(vertex.z) ** 0.2), 16, 16);
      const material = new THREE.MeshBasicMaterial({ color: bandSurfaceColor(vertex.band) });

      this.selectedMarker = new THREE.Mesh(geometry, material);
      this.selectedMarker.position.set(display.x, vertex.z, display.y);
      this.scene.add(this.selectedMarker);
    }
  }

  class DftLineGraph extends HTMLElement {
    constructor() {
      super();
      /** @type {GraphPayload | null} */
      this.payload = null;
      /** @type {GraphView | null} */
      this.view = null;
      /** @type {{x:number, y:number, angle?:number, mode?:"pan"|"rotate"} | null} */
      this.dragStart = null;
      this.inspectMode = true;
      /** @type {CursorHit | null} */
      this.cursor = null;
    }

    connectedCallback() {
      this.payload = readGraphPayload(this);
      if (!this.payload) return;

      this.view = graphBounds(this.payload);
      this.render();

      window.addEventListener("dft-local-selection-freeze", () => {
        this.render();
      });

      window.addEventListener("dft-local-selected-steps", () => {
        this.render();
      });

      window.addEventListener("dft-local-path-hover", (event) => {
        const customEvent = /** @type {CustomEvent<{step:number|null, pathX:number|null, source:EventTarget|null}>} */ (event);
        const detail = customEvent.detail;
        if (detail.source === this || !this.payload || !this.view || isSelectionFrozen()) return;

        if (detail.pathX === null) {
          this.cursor = null;
        } else {
          const fx = (detail.pathX - this.view.xmin) / (this.view.xmax - this.view.xmin);
          this.cursor = nearestPointByX(this.payload, this.view, fx, 0.5);
        }

        this.render();
      });

      this.setAttribute("data-ready", "true");
    }

    resetView() {
      if (!this.payload) return;
      this.view = graphBounds(this.payload);
      this.cursor = null;
      this.dragStart = null;
      this.render();
    }

    /**
     * @param {SVGSVGElement} svg
     */
    bindViewport(svg) {
      svg.addEventListener("wheel", (event) => {
        event.preventDefault();
        event.stopPropagation();
        if (!this.view) return;

        const { fx, fy } = plotFractionsFromPointer(svg, event, true);
        const factor = event.deltaY < 0 ? 0.82 : 1.22;

        this.view = zoomView(this.view, fx, 1.0 - fy, factor);
        if (!isSelectionFrozen()) {
          this.cursor = null;
        }
        this.render();
      }, { passive: false });

      svg.addEventListener("pointerdown", (event) => {
        event.preventDefault();
        event.stopPropagation();
        svg.setPointerCapture(event.pointerId);
        this.dragStart = { x: event.clientX, y: event.clientY, mode: "pan" };
      });

      svg.addEventListener("pointermove", (event) => {
        if (!this.view || !this.payload) return;

        const rect = svg.getBoundingClientRect();

        if (this.dragStart) {
          event.preventDefault();
          event.stopPropagation();

          const dx = ((event.clientX - this.dragStart.x) / rect.width);
          const dy = ((event.clientY - this.dragStart.y) / rect.height);

          this.view = panView(this.view, dx, dy);
          this.dragStart = { x: event.clientX, y: event.clientY, mode: "pan" };
          if (!isSelectionFrozen()) {
            this.cursor = null;
          }
          this.render();
          return;
        }

        if (!isSelectionFrozen()) {
          const { fx, fy } = plotFractionsFromPointer(svg, event, true);
          this.cursor = nearestPointByX(this.payload, this.view, fx, fy);
          emitPathHover(this.cursor?.step ?? null, this.cursor?.pathX ?? this.cursor?.x ?? null, this);
          this.render();
        }
      });

      svg.addEventListener("pointerup", (event) => {
        event.preventDefault();
        event.stopPropagation();
        try {
          svg.releasePointerCapture(event.pointerId);
        } catch {
          // Ignore if capture was already released.
        }
        this.dragStart = null;
      });

      svg.addEventListener("pointerleave", () => {
        this.dragStart = null;
        if (isSelectionFrozen()) return;
        this.cursor = null;
        emitPathHover(null, null, this);
        this.render();
      });

      svg.addEventListener("click", (event) => {
        event.preventDefault();
        event.stopPropagation();

        if (event.detail >= 2) {
          if (this.clickTimer !== null) {
            window.clearTimeout(this.clickTimer);
            this.clickTimer = null;
          }
          emitSelectionFreeze(false, this);
          this.resetView();
          return;
        }

        if (this.clickTimer !== null) {
          window.clearTimeout(this.clickTimer);
          this.clickTimer = null;
        }

        this.clickTimer = window.setTimeout(() => {
          this.clickTimer = null;
          emitSelectionFreeze(!isSelectionFrozen(), this);
        }, 360);
      });

      svg.addEventListener("dblclick", (event) => {
        event.preventDefault();
        event.stopPropagation();
      });
    }

    render() {
      if (!this.payload || !this.view) return;

      const reset = document.createElement("button");
      reset.type = "button";
      reset.textContent = "Reset view";
      reset.addEventListener("pointerdown", (event) => event.stopPropagation());
      reset.addEventListener("click", (event) => {
        event.preventDefault();
        event.stopPropagation();
        this.resetView();
      });

      const inspect = document.createElement("button");
      inspect.type = "button";
      inspect.textContent = this.inspectMode ? "Inspect cursor: on" : "Inspect cursor: off";
      inspect.addEventListener("pointerdown", (event) => event.stopPropagation());
      inspect.addEventListener("click", (event) => {
        event.preventDefault();
        event.stopPropagation();
        this.inspectMode = !this.inspectMode;
        this.cursor = null;
        this.render();
      });

      const lock = document.createElement("span");
      lock.className = "graph-control-help";
      lock.textContent = isSelectionFrozen() ? "selection frozen" : "selection live";

      const controls = document.createElement("div");
      controls.className = "graph-controls";
      controls.appendChild(reset);
      controls.appendChild(lock);

      const svg = makeGraphSvg(this.payload, this.view, this.cursor);
      svg.style.touchAction = "none";
      this.bindViewport(svg);

      const shell = document.createElement("div");
      shell.className = "graph-component";
      shell.appendChild(controls);
      shell.appendChild(svg);

      this.replaceChildren(shell);
    }
  }

  class DftKSpacePlot extends HTMLElement {
    constructor() {
      super();
      /** @type {GraphPayload | null} */
      this.payload = null;
      /** @type {GraphPayload | null} */
      this.displayPayload = null;
      /** @type {GraphView | null} */
      this.view = null;
      /** @type {{x:number, y:number, angle?:number, mode?:"pan"|"rotate"} | null} */
      this.dragStart = null;
      this.rotation = 0;
      this.dragMode = "pan";
      /** @type {CursorHit | null} */
      this.cursor = null;
    }

    connectedCallback() {
      this.payload = readGraphPayload(this);
      if (!this.payload) return;

      this.rebuildDisplayPayload(true);
      this.render();

      window.addEventListener("dft-local-selection-freeze", () => {
        this.render();
      });

      window.addEventListener("dft-local-selected-steps", () => {
        this.render();
      });

      window.addEventListener("dft-local-path-hover", (event) => {
        const customEvent = /** @type {CustomEvent<{step:number|null, pathX:number|null, source:EventTarget|null}>} */ (event);
        const detail = customEvent.detail;
        if (detail.source === this || !this.displayPayload || isSelectionFrozen()) return;

        if (detail.step === null) {
          this.cursor = null;
        } else {
          const selected = this.displayPayload.series.find((series) => series.name === "selected path");
          const point = selected?.points.find((p) => p.meta?.step === detail.step);
          this.cursor = point
            ? {
                series: "selected path",
                x: point.x,
                y: point.y,
                sx: this.view ? (point.x - this.view.xmin) / (this.view.xmax - this.view.xmin) : 0,
                sy: this.view ? (this.view.ymax - point.y) / (this.view.ymax - this.view.ymin) : 0,
                step: detail.step,
                pathX: detail.pathX,
              }
            : null;
        }

        this.render();
      });

      this.setAttribute("data-ready", "true");
    }

    /**
     * @param {boolean} resetView
     */
    rebuildDisplayPayload(resetView) {
      if (!this.payload) return;

      this.displayPayload = kspacePayloadToCartesian(this.payload, this.rotation);

      if (resetView) {
        if (this.displayPayload === null) return;
        this.view = equalAspectView(graphBounds(this.displayPayload));
      }
    }

    resetView() {
      this.rotation = 0;
      this.rebuildDisplayPayload(true);
      this.dragStart = null;
      this.cursor = null;
      this.render();
    }

    /**
     * @param {SVGSVGElement} svg
     */
    bindViewport(svg) {
      svg.addEventListener("wheel", (event) => {
        event.preventDefault();
        event.stopPropagation();

        if (event.shiftKey) {
          this.rotation += event.deltaY < 0 ? Math.PI / 60 : -Math.PI / 60;
          this.rebuildDisplayPayload(false);
          this.render();
          return;
        }

        if (!this.view) return;

        const { fx, fy } = plotFractionsFromPointer(svg, event, false);
        const factor = event.deltaY < 0 ? 0.82 : 1.22;

        this.view = zoomView(this.view, fx, 1.0 - fy, factor);
        this.render();
      }, { passive: false });

      svg.addEventListener("pointerdown", (event) => {
        event.preventDefault();
        event.stopPropagation();
        svg.setPointerCapture(event.pointerId);
        this.dragStart = { x: event.clientX, y: event.clientY };
      });

      svg.addEventListener("pointermove", (event) => {
        if (!this.dragStart || !this.view) return;

        event.preventDefault();
        event.stopPropagation();

        const rect = svg.getBoundingClientRect();

        if (this.dragStart.mode === "rotate") {
          const cx = rect.left + 0.5 * rect.width;
          const cy = rect.top + 0.5 * rect.height;
          const angle = Math.atan2(event.clientY - cy, event.clientX - cx);
          const previous = this.dragStart.angle ?? angle;
          this.rotation += angle - previous;
          this.dragStart = {
            x: event.clientX,
            y: event.clientY,
            angle,
            mode: "rotate",
          };
          this.rebuildDisplayPayload(false);
          this.render();
          return;
        }

        const dx = ((event.clientX - this.dragStart.x) / rect.width);
        const dy = ((event.clientY - this.dragStart.y) / rect.height);

        this.view = panView(this.view, dx, dy);
        this.dragStart = { x: event.clientX, y: event.clientY, mode: "pan" };
        if (!isSelectionFrozen()) {
          this.cursor = null;
        }
        this.render();
      });

      svg.addEventListener("pointerup", (event) => {
        event.preventDefault();
        event.stopPropagation();
        try {
          svg.releasePointerCapture(event.pointerId);
        } catch {
          // Ignore if capture was already released.
        }
        this.dragStart = null;
      });

      svg.addEventListener("pointermove", (event) => {
        if (isSelectionFrozen() || this.dragStart || !this.view || !this.displayPayload) return;

        const { fx, fy } = plotFractionsFromPointer(svg, event, false);

        this.cursor = nearestPathPoint(this.displayPayload, this.view, fx, fy);
        emitPathHover(this.cursor?.step ?? null, this.cursor?.pathX ?? null, this);
        this.render();
      });

      svg.addEventListener("pointerleave", () => {
        this.dragStart = null;
        if (isSelectionFrozen()) return;
        this.cursor = null;
        emitPathHover(null, null, this);
        this.render();
      });

      svg.addEventListener("click", (event) => {
        event.preventDefault();
        event.stopPropagation();

        if (event.detail >= 2) {
          if (this.clickTimer !== null) {
            window.clearTimeout(this.clickTimer);
            this.clickTimer = null;
          }
          emitSelectionFreeze(false, this);
          this.resetView();
          return;
        }

        if (this.clickTimer !== null) {
          window.clearTimeout(this.clickTimer);
          this.clickTimer = null;
        }

        this.clickTimer = window.setTimeout(() => {
          this.clickTimer = null;
          emitSelectionFreeze(!isSelectionFrozen(), this);
        }, 360);
      });

      svg.addEventListener("dblclick", (event) => {
        event.preventDefault();
        event.stopPropagation();
      });
    }

    render() {
      if (!this.displayPayload || !this.view) return;

      const reset = document.createElement("button");
      reset.type = "button";
      reset.textContent = "Reset view";
      reset.addEventListener("pointerdown", (event) => event.stopPropagation());
      reset.addEventListener("click", (event) => {
        event.preventDefault();
        event.stopPropagation();
        this.resetView();
      });

      const rotateLeft = document.createElement("button");
      rotateLeft.type = "button";
      rotateLeft.textContent = "Rotate left";
      rotateLeft.addEventListener("pointerdown", (event) => event.stopPropagation());
      rotateLeft.addEventListener("click", (event) => {
        event.preventDefault();
        event.stopPropagation();
        this.rotation -= Math.PI / 24;
        this.rebuildDisplayPayload(false);
        this.render();
      });

      const rotateRight = document.createElement("button");
      rotateRight.type = "button";
      rotateRight.textContent = "Rotate right";
      rotateRight.addEventListener("pointerdown", (event) => event.stopPropagation());
      rotateRight.addEventListener("click", (event) => {
        event.preventDefault();
        event.stopPropagation();
        this.rotation += Math.PI / 24;
        this.rebuildDisplayPayload(false);
        this.render();
      });

      const dragMode = document.createElement("button");
      dragMode.type = "button";
      dragMode.textContent = this.dragMode === "rotate" ? "Drag mode: rotate" : "Drag mode: pan";
      dragMode.addEventListener("pointerdown", (event) => event.stopPropagation());
      dragMode.addEventListener("click", (event) => {
        event.preventDefault();
        event.stopPropagation();
        this.dragMode = this.dragMode === "rotate" ? "pan" : "rotate";
        this.dragStart = null;
        this.render();
      });

      const help = document.createElement("span");
      help.className = "graph-control-help";
      help.textContent = "Shift+wheel rotates; drag mode controls plain drag";

      const lock = document.createElement("span");
      lock.className = "graph-control-help";
      lock.textContent = isSelectionFrozen() ? "selection frozen" : "selection live";

      const controls = document.createElement("div");
      controls.className = "graph-controls";
      controls.appendChild(reset);
      controls.appendChild(rotateLeft);
      controls.appendChild(rotateRight);
      controls.appendChild(dragMode);
      controls.appendChild(help);
      controls.appendChild(lock);

      const svg = makeGraphSvg(this.displayPayload, this.view, this.cursor, { showAxes: false, kspace: true });
      svg.style.touchAction = "none";
      this.bindViewport(svg);

      const shell = document.createElement("div");
      shell.className = "graph-component graph-component-kspace";
      shell.appendChild(controls);
      shell.appendChild(svg);

      this.replaceChildren(shell);
    }
  }

  if (!customElements.get("dft-band-controls")) {
    customElements.define("dft-band-controls", DftBandControls);
  }

  if (!customElements.get("dft-band-readout")) {
    customElements.define("dft-band-readout", DftBandReadout);
  }

  if (!customElements.get("dft-kpoint-readout")) {
    customElements.define("dft-kpoint-readout", DftKPointReadout);
  }

  if (!customElements.get("dft-band-surface-viewer")) {
    customElements.define("dft-band-surface-viewer", DftBandSurfaceViewer);
  }

  if (!customElements.get("dft-line-graph")) {
    customElements.define("dft-line-graph", DftLineGraph);
  }

  if (!customElements.get("dft-kspace-plot")) {
    customElements.define("dft-kspace-plot", DftKSpacePlot);
  }
}

export {nice, readJsonPayload, readGraphPayload, makeGraphSvg, graphBounds, zoomView, panView, equalAspectView, kBasisToCartesian, rotatePoint, kspacePayloadToCartesian, bandSurfaceVertices, bandSurfaceTriangles, bandSurfaceMeshData, bandSurfaceMeshDataWithMask, bandSurfaceColor, allBandIndices, visibleBandIndices, bandSurfaceSummary, projectBandSurfacePoint, nearestBandSurfaceVertex, bandBasisToCartesian, vertexInsideVisibleHexagon, pointInDisplayPolygon, threeUvGridReferenceData, threeHexagonReferenceData, threeBandSurfaceGeometryData, drawBandSurfacePreview, drawBandSurfaceReferenceFrame, drawBandSurfaceSliceGuide, drawBandSurfaceSelectionMarker, plotFractionsFromPointer, createDftSignalBus, emitDftSignal, onDftSignal, isSelectionFrozen, emitSelectionFreeze, selectedSteps, emitSelectedSteps, nearestPathPoint, selectedPathHits, nearestPointByX };
