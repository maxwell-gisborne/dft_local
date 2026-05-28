// @ts-check

/**
 * @typedef {{x:number, y:number, entity_id?:string|null, label?:string, meta?:Record<string, unknown>}} GraphPoint
 * @typedef {{name:string, kind:"line"|"points"|"line_points", points:GraphPoint[]}} GraphSeries
 * @typedef {{id:string, title:string, x_label:string, y_label:string, series:GraphSeries[]}} GraphPayload
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

/**
 * @param {Element} host
 * @returns {GraphPayload | null}
 */
function readGraphPayload(host) {
  const source = host.getAttribute("data-source");
  if (!source) return null;

  const script = document.getElementById(source);
  if (!script) return null;

  try {
    return /** @type {GraphPayload} */ (JSON.parse(script.textContent || ""));
  } catch {
    return null;
  }
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
 * @param {GraphPayload} payload
 * @returns {SVGSVGElement}
 */
function makeGraphSvg(payload) {
  const width = 1000;
  const height = 520;
  const margin = { left: 78, right: 24, top: 28, bottom: 62 };
  const innerW = width - margin.left - margin.right;
  const innerH = height - margin.top - margin.bottom;

  /** @type {GraphPoint[]} */
  const all = [];
  for (const series of payload.series) {
    for (const point of series.points) {
      if (Number.isFinite(point.x) && Number.isFinite(point.y)) {
        all.push(point);
      }
    }
  }

  const svg = /** @type {SVGSVGElement} */ (svgEl("svg", {
    class: "graph-svg graph-svg-component",
    viewBox: `0 0 ${width} ${height}`,
    role: "img",
    "aria-label": payload.title || "Graph",
  }));

  svg.appendChild(svgEl("rect", {
    class: "graph-bg",
    x: "0",
    y: "0",
    width: String(width),
    height: String(height),
  }));

  if (all.length === 0) return svg;

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

  const sx = (/** @type {number} */ x) => margin.left + ((x - xmin) / (xmax - xmin)) * innerW;
  const sy = (/** @type {number} */ y) => margin.top + ((ymax - y) / (ymax - ymin)) * innerH;

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
      svg.appendChild(svgEl("path", {
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
        svg.appendChild(svgEl("circle", {
          cx: String(x),
          cy: String(y),
          r: "2.5",
          fill: colour,
        }));
      }
    }

    const [lx, ly] = points[points.length - 1];
    const label = svgEl("text", {
      class: "series-label",
      x: String(lx + 5),
      y: String(ly + 4),
      fill: colour,
    });
    label.textContent = series.name;
    svg.appendChild(label);
  });

  return svg;
}

if (typeof HTMLElement !== "undefined" && typeof customElements !== "undefined") {
  class DftLineGraph extends HTMLElement {
    connectedCallback() {
      const payload = readGraphPayload(this);
      if (!payload) return;

      this.replaceChildren(makeGraphSvg(payload));
      this.setAttribute("data-ready", "true");
    }
  }

  class DftKSpacePlot extends HTMLElement {
    connectedCallback() {
      const payload = readGraphPayload(this);
      if (!payload) return;

      this.replaceChildren(makeGraphSvg(payload));
      this.setAttribute("data-ready", "true");
    }
  }

  if (!customElements.get("dft-line-graph")) {
    customElements.define("dft-line-graph", DftLineGraph);
  }

  if (!customElements.get("dft-kspace-plot")) {
    customElements.define("dft-kspace-plot", DftKSpacePlot);
  }
}

export { nice, readGraphPayload, makeGraphSvg };
