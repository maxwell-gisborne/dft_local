from dataclasses import dataclass
from html import escape
from pathlib import Path
from typing import Any
import json
from html import escape
import numpy as np


import numpy as np
from fastapi import FastAPI, Query
from fastapi.responses import HTMLResponse, JSONResponse

from . import (
    SparseDataset,
    NearestNeighbourGraph,
    EdgeDirections,
    EdgeGroupLabels,
    GdKernelArrays,
    SymbolPair,
    DenseMatrixDiagnostics,
    LocalPath,
    hermitian_part,
)
def polyline(points: list[tuple[float, float]], *, stroke: str = "#254f7a", width: float = 1.4) -> str:
    pts = " ".join(f"{x:.3f},{y:.3f}" for x, y in points)
    return f'<polyline points="{pts}" fill="none" stroke="{stroke}" stroke-width="{width}" />'


def svg_band_plot(
    x: np.ndarray,
    E: np.ndarray,
    labels: tuple[tuple[int, str], ...],
    *,
    width: int = 920,
    height: int = 520,
    pad_left: int = 72,
    pad_right: int = 24,
    pad_top: int = 28,
    pad_bottom: int = 56,
) -> str:
    x = np.asarray(x, dtype=float)
    E = np.asarray(E, dtype=float)

    if E.ndim != 2:
        raise ValueError(f"E must have shape (nk, nbands), got {E.shape}")

    xmin = float(np.min(x))
    xmax = float(np.max(x))
    emin = float(np.min(E))
    emax = float(np.max(E))

    # Add small vertical margin.
    margin = 0.06 * max(emax - emin, 1.0)
    emin -= margin
    emax += margin

    plot_w = width - pad_left - pad_right
    plot_h = height - pad_top - pad_bottom

    def sx(v: float) -> float:
        if xmax == xmin:
            return pad_left
        return pad_left + (v - xmin) / (xmax - xmin) * plot_w

    def sy(v: float) -> float:
        if emax == emin:
            return pad_top + plot_h / 2
        return pad_top + (emax - v) / (emax - emin) * plot_h

    lines = []

    # Background and borderless plotting area.
    lines.append(f'<rect x="0" y="0" width="{width}" height="{height}" fill="#ffffff" />')

    # Axes.
    x0 = pad_left
    y0 = pad_top + plot_h
    lines.append(f'<line x1="{x0}" y1="{pad_top}" x2="{x0}" y2="{y0}" stroke="#222" stroke-width="1" />')
    lines.append(f'<line x1="{x0}" y1="{y0}" x2="{pad_left + plot_w}" y2="{y0}" stroke="#222" stroke-width="1" />')

    # Horizontal energy grid ticks.
    nticks = 6
    for i in range(nticks):
        t = i / (nticks - 1)
        val = emin + t * (emax - emin)
        yy = sy(val)
        lines.append(f'<line x1="{x0}" y1="{yy:.3f}" x2="{pad_left + plot_w}" y2="{yy:.3f}" stroke="#e8e8e8" stroke-width="1" />')
        lines.append(
            f'<text x="{pad_left - 10}" y="{yy + 4:.3f}" text-anchor="end" '
            f'font-size="12" fill="#444">{val:.2f}</text>'
        )

    # High-symmetry vertical lines and labels.
    for idx, label in labels:
        xx = sx(float(x[idx]))
        lines.append(f'<line x1="{xx:.3f}" y1="{pad_top}" x2="{xx:.3f}" y2="{y0}" stroke="#d0d0d0" stroke-width="1" />')
        lines.append(
            f'<text x="{xx:.3f}" y="{height - 24}" text-anchor="middle" '
            f'font-size="14" fill="#222">{escape(label)}</text>'
        )

    # Band lines.
    nbands = E.shape[1]
    for band in range(nbands):
        pts = [(sx(float(x[i])), sy(float(E[i, band]))) for i in range(len(x))]
        lines.append(polyline(pts, stroke="#254f7a", width=1.35))

    # Axis label.
    lines.append(
        f'<text x="18" y="{pad_top + plot_h / 2:.3f}" text-anchor="middle" '
        f'font-size="13" fill="#222" transform="rotate(-90 18,{pad_top + plot_h / 2:.3f})">Energy / eV</text>'
    )

    return (
        f'<svg viewBox="0 0 {width} {height}" width="100%" height="auto" '
        f'role="img" aria-label="Band structure plot">'
        + "\n".join(lines)
        + "</svg>"
    )

def svg_k_path_plot(
    k1: np.ndarray,
    k2: np.ndarray,
    labels: tuple[tuple[int, str], ...],
    *,
    width: int = 420,
    height: int = 420,
    pad: int = 48,
) -> str:
    k1 = np.asarray(k1, dtype=float)
    k2 = np.asarray(k2, dtype=float)

    xmin = float(np.min(k1))
    xmax = float(np.max(k1))
    ymin = float(np.min(k2))
    ymax = float(np.max(k2))

    # Make plot square in k-space.
    cx = 0.5 * (xmin + xmax)
    cy = 0.5 * (ymin + ymax)
    span = max(xmax - xmin, ymax - ymin, 1.0)
    xmin = cx - 0.55 * span
    xmax = cx + 0.55 * span
    ymin = cy - 0.55 * span
    ymax = cy + 0.55 * span

    plot_w = width - 2 * pad
    plot_h = height - 2 * pad

    def sx(v: float) -> float:
        return pad + (v - xmin) / (xmax - xmin) * plot_w

    def sy(v: float) -> float:
        return pad + (ymax - v) / (ymax - ymin) * plot_h

    lines = []
    lines.append(f'<rect x="0" y="0" width="{width}" height="{height}" fill="#ffffff" />')

    # Axes through origin if visible.
    if xmin <= 0 <= xmax:
        x0 = sx(0.0)
        lines.append(f'<line x1="{x0:.3f}" y1="{pad}" x2="{x0:.3f}" y2="{height-pad}" stroke="#e0e0e0" stroke-width="1" />')
    if ymin <= 0 <= ymax:
        y0 = sy(0.0)
        lines.append(f'<line x1="{pad}" y1="{y0:.3f}" x2="{width-pad}" y2="{y0:.3f}" stroke="#e0e0e0" stroke-width="1" />')

    # Border.
    lines.append(
        f'<rect x="{pad}" y="{pad}" width="{plot_w}" height="{plot_h}" '
        f'fill="none" stroke="#222" stroke-width="1" />'
    )

    # Path polyline.
    pts = " ".join(f"{sx(a):.3f},{sy(b):.3f}" for a, b in zip(k1, k2))
    lines.append(
        f'<polyline points="{pts}" fill="none" stroke="#254f7a" stroke-width="2" />'
    )

    # Direction markers: a few small points along path.
    step = max(1, len(k1) // 12)
    for i in range(0, len(k1), step):
        lines.append(
            f'<circle cx="{sx(k1[i]):.3f}" cy="{sy(k2[i]):.3f}" r="2.2" fill="#254f7a" />'
        )

    # Label special points.
    for index, label in labels:
        kk1 = float(k1[index])
        kk2 = float(k2[index])
        x = sx(kk1)
        y = sy(kk2)

        lines.append(f'<circle cx="{x:.3f}" cy="{y:.3f}" r="5" fill="#e74c3c" />')
        lines.append(
            f'<text x="{x + 8:.3f}" y="{y - 8:.3f}" '
            f'font-size="14" fill="#222">{escape(label)}</text>'
        )

    # Axis labels.
    lines.append(
        f'<text x="{width/2:.3f}" y="{height - 12}" text-anchor="middle" '
        f'font-size="13" fill="#222">k1</text>'
    )
    lines.append(
        f'<text x="15" y="{height/2:.3f}" text-anchor="middle" '
        f'font-size="13" fill="#222" transform="rotate(-90 15,{height/2:.3f})">k2</text>'
    )

    # Corner tick labels.
    lines.append(
        f'<text x="{pad}" y="{height - pad + 18}" text-anchor="middle" font-size="11" fill="#666">{xmin:.2f}</text>'
    )
    lines.append(
        f'<text x="{width - pad}" y="{height - pad + 18}" text-anchor="middle" font-size="11" fill="#666">{xmax:.2f}</text>'
    )
    lines.append(
        f'<text x="{pad - 8}" y="{height - pad + 4}" text-anchor="end" font-size="11" fill="#666">{ymin:.2f}</text>'
    )
    lines.append(
        f'<text x="{pad - 8}" y="{pad + 4}" text-anchor="end" font-size="11" fill="#666">{ymax:.2f}</text>'
    )

    return (
        f'<svg viewBox="0 0 {width} {height}" width="100%" height="auto" '
        f'role="img" aria-label="k-space path plot">'
        + "\n".join(lines)
        + "</svg>"
    )

def svg_band_plot_interactive(
    x: np.ndarray,
    E: np.ndarray,
    labels: tuple[tuple[int, str], ...],
    *,
    width: int = 920,
    height: int = 520,
) -> str:
    x = np.asarray(x, dtype=float)
    E = np.asarray(E, dtype=float)

    if E.ndim != 2:
        raise ValueError(f"E must have shape (nk, nbands), got {E.shape}")

    xmin = float(np.min(x))
    xmax = float(np.max(x))
    emin = float(np.min(E))
    emax = float(np.max(E))

    margin = 0.06 * max(emax - emin, 1.0)
    ymin = emin - margin
    ymax = emax + margin

    data_w = xmax - xmin
    data_h = ymax - ymin

    def pts_for_band(band: int) -> str:
        # SVG y-axis goes downward, so use -E as display y.
        return " ".join(
            f"{float(x[i]):.10g},{-float(E[i, band]):.10g}"
            for i in range(len(x))
        )

    # Use y display bounds as -ymax .. -ymin.
    view_x = xmin
    view_y = -ymax
    view_w = data_w if data_w > 0 else 1.0
    view_h = data_h if data_h > 0 else 1.0

    lines = []

    # Horizontal zero-energy line if visible.
    if ymin <= 0 <= ymax:
        lines.append(
            f'<line x1="{xmin}" y1="0" x2="{xmax}" y2="0" '
            f'stroke="#bbbbbb" stroke-width="{0.002 * view_h}" vector-effect="non-scaling-stroke" />'
        )

    # High-symmetry vertical lines.
    for idx, label in labels:
        xx = float(x[idx])
        lines.append(
            f'<line x1="{xx}" y1="{-ymax}" x2="{xx}" y2="{-ymin}" '
            f'stroke="#d0d0d0" stroke-width="{0.002 * view_h}" vector-effect="non-scaling-stroke" />'
        )
        lines.append(
            f'<text x="{xx}" y="{-ymin + 0.04 * data_h}" '
            f'text-anchor="middle" font-size="{0.045 * data_h}" fill="#222">'
            f'{escape(label)}</text>'
        )

    # Bands.
    for band in range(E.shape[1]):
        lines.append(
            f'<polyline points="{pts_for_band(band)}" fill="none" '
            f'stroke="#254f7a" stroke-width="1.4" vector-effect="non-scaling-stroke" />'
        )

    return f"""
<div class="zoom-shell">
  <div class="zoom-toolbar">
    <button type="button" data-zoom="in">Zoom in</button>
    <button type="button" data-zoom="out">Zoom out</button>
    <button type="button" data-reset="1">Reset</button>
    <span class="small">Wheel to zoom, drag to pan.</span>
  </div>

  <svg
    id="band-svg"
    class="interactive-svg"
    viewBox="{view_x} {view_y} {view_w} {view_h}"
    data-initial-viewbox="{view_x} {view_y} {view_w} {view_h}"
    preserveAspectRatio="none"
    width="100%"
    height="{height}"
    role="img"
    aria-label="Interactive band plot"
  >
    <rect x="{view_x}" y="{view_y}" width="{view_w}" height="{view_h}" fill="white" />
    {''.join(lines)}
  </svg>
</div>
"""


@dataclass(frozen=True)
class DiagnosticsState:
    data: SparseDataset
    geom: NearestNeighbourGraph
    edges: EdgeDirections
    labels: EdgeGroupLabels

    KH: GdKernelArrays
    KS: GdKernelArrays

    KH_avg: GdKernelArrays
    KS_avg: GdKernelArrays

    KH_avg_star: GdKernelArrays
    KS_avg_star: GdKernelArrays


def band_plot_payload(
    x: np.ndarray,
    E: np.ndarray,
    labels: tuple[tuple[int, str], ...],
) -> dict:
    return {
        "x": [float(v) for v in x],
        "energies": [
            [float(E[i, j]) for j in range(E.shape[1])]
            for i in range(E.shape[0])
        ],
        "labels": [
            {
                "index": int(i),
                "label": str(label),
                "x": float(x[i]),
            }
            for i, label in labels
        ],
    }

def build_state(root: str | Path) -> DiagnosticsState:
    data = SparseDataset.load(Path(root))

    geom = NearestNeighbourGraph.from_positions(data.metadata.positions)
    edges = EdgeDirections.from_geometry(geom)
    labels = EdgeGroupLabels.from_geometry(geom, edges)

    KH = GdKernelArrays.from_anchored(data.H, labels, matrix_name="H anchored")
    KS = GdKernelArrays.from_anchored(data.S, labels, matrix_name="S anchored")

    KH_avg = GdKernelArrays.from_average(data.H, labels, matrix_name="H average")
    KS_avg = GdKernelArrays.from_average(data.S, labels, matrix_name="S average")

    KH_avg_star = KH_avg.star_symmetrised(matrix_name="H average star")
    KS_avg_star = KS_avg.star_symmetrised(matrix_name="S average star")

    return DiagnosticsState(
        data=data,
        geom=geom,
        edges=edges,
        labels=labels,
        KH=KH,
        KS=KS,
        KH_avg=KH_avg,
        KS_avg=KS_avg,
        KH_avg_star=KH_avg_star,
        KS_avg_star=KS_avg_star,
    )


def safe_json_for_script(data: dict) -> str:
    return (
        json.dumps(data, ensure_ascii=False)
        .replace("&", "\\u0026")
        .replace("<", "\\u003c")
        .replace(">", "\\u003e")
    )


def canvas_band_plot(payload: dict) -> str:
    payload_json = safe_json_for_script(payload)

    return f"""
<div class="band-canvas-wrap">
  <div class="band-toolbar">
    <button type="button" id="band-zoom-in">Zoom in</button>
    <button type="button" id="band-zoom-out">Zoom out</button>
    <button type="button" id="band-reset">Reset</button>
    <span class="small">Wheel to zoom around cursor. Drag to pan.</span>
  </div>

  <canvas id="band-canvas"></canvas>
  <script id="band-data" type="application/json">{payload_json}</script>
</div>
"""

CSS = """
:root {
  --bg: #fafafa;
  --fg: #161616;
  --muted: #666;
  --line: #d9d9d9;
  --card: #ffffff;
  --accent: #254f7a;
  --bad: #8a1f1f;
  --good: #236b3a;
  font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
}

body {
  margin: 0;
  background: var(--bg);
  color: var(--fg);
}

header {
  border-bottom: 1px solid var(--line);
  background: var(--card);
  padding: 1rem 1.5rem;
}

main {
  max-width: 1100px;
  margin: 0 auto;
  padding: 1.25rem;
}

nav a {
  display: inline-block;
  margin-right: 1rem;
  color: var(--accent);
  text-decoration: none;
  font-weight: 600;
}

nav a:hover {
  text-decoration: underline;
}

h1, h2, h3 {
  line-height: 1.2;
}

.grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(230px, 1fr));
  gap: 1rem;
}

.card {
  background: var(--card);
  border: 1px solid var(--line);
  border-radius: 10px;
  padding: 1rem;
}

.metric {
  font-size: 1.55rem;
  font-weight: 700;
}

.label {
  color: var(--muted);
  font-size: 0.9rem;
}

table {
  border-collapse: collapse;
  width: 100%;
  background: var(--card);
  margin: 1rem 0;
}

th {
  text-align: left;
  border-bottom: 2px solid var(--fg);
  padding: 0.45rem 0.55rem;
}

td {
  border-bottom: 1px solid var(--line);
  padding: 0.45rem 0.55rem;
  vertical-align: top;
}

td.num, th.num {
  text-align: right;
  font-variant-numeric: tabular-nums;
}

code {
  background: #eee;
  padding: 0.1rem 0.25rem;
  border-radius: 4px;
}

.good { color: var(--good); font-weight: 700; }
.bad { color: var(--bad); font-weight: 700; }

.small {
  color: var(--muted);
  font-size: 0.9rem;
}

form.inline {
  display: flex;
  flex-wrap: wrap;
  gap: 0.75rem;
  align-items: end;
  margin: 1rem 0;
}

input {
  padding: 0.35rem 0.45rem;
  border: 1px solid var(--line);
  border-radius: 5px;
}

button {
  padding: 0.4rem 0.75rem;
  border: 1px solid var(--accent);
  background: var(--accent);
  color: white;
  border-radius: 5px;
}

.zoom-shell {
  width: 100%;
}

.zoom-toolbar {
  display: flex;
  gap: 0.5rem;
  align-items: center;
  margin-bottom: 0.6rem;
  flex-wrap: wrap;
}

.zoom-toolbar button {
  padding: 0.35rem 0.65rem;
  border: 1px solid var(--accent);
  background: var(--card);
  color: var(--accent);
  border-radius: 5px;
  cursor: pointer;
}

.zoom-toolbar button:hover {
  background: #eef4fa;
}

.band-canvas-wrap {
  width: 100%;
}

.band-toolbar {
  display: flex;
  gap: 0.5rem;
  align-items: center;
  margin-bottom: 0.6rem;
  flex-wrap: wrap;
}

.band-toolbar button {
  padding: 0.35rem 0.65rem;
  border: 1px solid var(--accent);
  background: var(--card);
  color: var(--accent);
  border-radius: 5px;
  cursor: pointer;
}

.band-toolbar button:hover {
  background: #eef4fa;
}

#band-canvas {
  width: 100%;
  height: 680px;
  display: block;
  border: 1px solid var(--line);
  border-radius: 8px;
  background: white;
  cursor: grab;
}


#band-canvas.dragging {
  cursor: grabbing;
}


.interactive-svg {
  display: block;
  width: 100%;
  height: 520px;
  border: 1px solid var(--line);
  border-radius: 8px;
  background: white;
  cursor: grab;
  touch-action: none;
}

.interactive-svg.dragging {
  cursor: grabbing;
}

.plot-grid {
  display: grid;
  grid-template-columns: 1fr;
  gap: 1rem;
  align-items: stretch;
}

.plot-grid .card {
  min-width: 0;
}

.plot-grid svg {
  max-width: 100%;
}

.kpath-card svg {
  max-height: 420px;
}


@media (max-width: 900px) {
  .plot-grid {
    grid-template-columns: 1fr;
  }
}

"""

VIEWER_CSS = """
html, body {
  margin: 0;
  height: 100%;
  overflow: hidden;
  font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
}

#app {
  display: grid;
  grid-template-columns: 1fr 320px;
  height: 100vh;
}

#viewer {
  width: 100%;
  height: 100%;
  background: #101216;
}

#panel {
  border-left: 1px solid #ddd;
  padding: 1rem;
  background: #fafafa;
  overflow: auto;
}

h1 {
  font-size: 1.1rem;
  margin: 0 0 0.75rem 0;
}

dl {
  display: grid;
  grid-template-columns: auto 1fr;
  gap: 0.35rem 0.75rem;
}

dt {
  color: #666;
}

dd {
  margin: 0;
  font-variant-numeric: tabular-nums;
}

a {
  color: #254f7a;
  font-weight: 600;
  text-decoration: none;
}

a:hover {
  text-decoration: underline;
}

.small {
  color: #666;
  font-size: 0.9rem;
}
"""

BAND_ZOOM_JS = """
<script>
(() => {
  const svg = document.getElementById("band-svg");
  if (!svg) return;

  const initial = svg.dataset.initialViewbox.split(" ").map(Number);
  let viewBox = [...initial];
  let dragging = false;
  let last = null;

  function setViewBox() {
    svg.setAttribute("viewBox", viewBox.join(" "));
  }

  function svgPoint(event) {
    const rect = svg.getBoundingClientRect();
    const x = viewBox[0] + (event.clientX - rect.left) / rect.width * viewBox[2];
    const y = viewBox[1] + (event.clientY - rect.top) / rect.height * viewBox[3];
    return [x, y];
  }

  function zoomAt(clientX, clientY, factor) {
    const rect = svg.getBoundingClientRect();
    const px = (clientX - rect.left) / rect.width;
    const py = (clientY - rect.top) / rect.height;

    const x = viewBox[0] + px * viewBox[2];
    const y = viewBox[1] + py * viewBox[3];

    const newW = viewBox[2] * factor;
    const newH = viewBox[3] * factor;

    viewBox[0] = x - px * newW;
    viewBox[1] = y - py * newH;
    viewBox[2] = newW;
    viewBox[3] = newH;

    setViewBox();
  }

  svg.addEventListener("wheel", (event) => {
    event.preventDefault();
    const factor = event.deltaY < 0 ? 0.85 : 1.18;
    zoomAt(event.clientX, event.clientY, factor);
  }, { passive: false });

  svg.addEventListener("pointerdown", (event) => {
    dragging = true;
    last = [event.clientX, event.clientY];
    svg.classList.add("dragging");
    svg.setPointerCapture(event.pointerId);
  });

  svg.addEventListener("pointermove", (event) => {
    if (!dragging || !last) return;

    const rect = svg.getBoundingClientRect();
    const dx = (event.clientX - last[0]) / rect.width * viewBox[2];
    const dy = (event.clientY - last[1]) / rect.height * viewBox[3];

    viewBox[0] -= dx;
    viewBox[1] -= dy;

    last = [event.clientX, event.clientY];
    setViewBox();
  });

  svg.addEventListener("pointerup", (event) => {
    dragging = false;
    last = null;
    svg.classList.remove("dragging");
    svg.releasePointerCapture(event.pointerId);
  });

  svg.addEventListener("pointercancel", () => {
    dragging = false;
    last = null;
    svg.classList.remove("dragging");
  });

  document.querySelector('[data-zoom="in"]')?.addEventListener("click", () => {
    const rect = svg.getBoundingClientRect();
    zoomAt(rect.left + rect.width / 2, rect.top + rect.height / 2, 0.8);
  });

  document.querySelector('[data-zoom="out"]')?.addEventListener("click", () => {
    const rect = svg.getBoundingClientRect();
    zoomAt(rect.left + rect.width / 2, rect.top + rect.height / 2, 1.25);
  });

  document.querySelector("[data-reset]")?.addEventListener("click", () => {
    viewBox = [...initial];
    setViewBox();
  });
})();
</script>
"""

BAND_CANVAS_JS = """
<script>
(() => {
  const canvas = document.getElementById("band-canvas");
  const raw = document.getElementById("band-data");
  if (!canvas || !raw) return;

  const data = JSON.parse(raw.textContent);
  const ctx = canvas.getContext("2d");

  const xData = data.x;
  const energies = data.energies;
  const labels = data.labels;

  const nk = xData.length;
  const nbands = energies[0].length;

  const allE = energies.flat();

  const full = {
    xmin: Math.min(...xData),
    xmax: Math.max(...xData),
    ymin: Math.min(...allE),
    ymax: Math.max(...allE),
  };

  const emargin = 0.06 * Math.max(full.ymax - full.ymin, 1.0);
  full.ymin -= emargin;
  full.ymax += emargin;

  let view = { ...full };
  let dragging = false;
  let last = null;

  const pad = {
    left: 70,
    right: 20,
    top: 24,
    bottom: 52,
  };

  function resizeCanvas() {
    const dpr = window.devicePixelRatio || 1;
    const rect = canvas.getBoundingClientRect();

    canvas.width = Math.round(rect.width * dpr);
    canvas.height = Math.round(rect.height * dpr);

    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    draw();
  }

  function plotRect() {
    const rect = canvas.getBoundingClientRect();
    return {
      x0: pad.left,
      y0: pad.top,
      w: rect.width - pad.left - pad.right,
      h: rect.height - pad.top - pad.bottom,
    };
  }

  function sx(x) {
    const r = plotRect();
    return r.x0 + (x - view.xmin) / (view.xmax - view.xmin) * r.w;
  }

  function sy(y) {
    const r = plotRect();
    return r.y0 + (view.ymax - y) / (view.ymax - view.ymin) * r.h;
  }

  function invx(px) {
    const r = plotRect();
    return view.xmin + (px - r.x0) / r.w * (view.xmax - view.xmin);
  }

  function invy(py) {
    const r = plotRect();
    return view.ymax - (py - r.y0) / r.h * (view.ymax - view.ymin);
  }

  function niceStep(span, targetTicks = 6) {
    const raw = span / targetTicks;
    const power = Math.pow(10, Math.floor(Math.log10(raw)));
    const scaled = raw / power;

    let step;
    if (scaled < 1.5) step = 1;
    else if (scaled < 3) step = 2;
    else if (scaled < 7) step = 5;
    else step = 10;

    return step * power;
  }

  function tickValues(min, max, targetTicks = 6) {
    const step = niceStep(max - min, targetTicks);
    const start = Math.ceil(min / step) * step;
    const out = [];

    for (let v = start; v <= max + 0.5 * step; v += step) {
      out.push(v);
    }

    return out;
  }

  function fmt(v) {
    const av = Math.abs(v);
    if (av > 0 && (av < 1e-3 || av >= 1e4)) return v.toExponential(2);
    return v.toFixed(2);
  }

  function drawAxes() {
    const r = plotRect();

    ctx.save();

    ctx.fillStyle = "#ffffff";
    ctx.fillRect(0, 0, canvas.clientWidth, canvas.clientHeight);

    // Grid.
    ctx.strokeStyle = "#e8e8e8";
    ctx.lineWidth = 1;

    ctx.font = "12px system-ui, sans-serif";
    ctx.fillStyle = "#444";
    ctx.textBaseline = "middle";

    const yTicks = tickValues(view.ymin, view.ymax, 7);
    for (const y of yTicks) {
      const py = sy(y);
      ctx.beginPath();
      ctx.moveTo(r.x0, py);
      ctx.lineTo(r.x0 + r.w, py);
      ctx.stroke();

      ctx.textAlign = "right";
      ctx.fillText(fmt(y), r.x0 - 8, py);
    }

    const xTicks = tickValues(view.xmin, view.xmax, 7);
    ctx.textBaseline = "top";
    for (const x of xTicks) {
      const px = sx(x);
      ctx.beginPath();
      ctx.moveTo(px, r.y0);
      ctx.lineTo(px, r.y0 + r.h);
      ctx.stroke();

      ctx.textAlign = "center";
      ctx.fillText(fmt(x), px, r.y0 + r.h + 8);
    }

    // High symmetry verticals and labels.
    ctx.strokeStyle = "#cfcfcf";
    ctx.fillStyle = "#111";
    ctx.font = "14px system-ui, sans-serif";

    for (const item of labels) {
      const px = sx(item.x);
      if (px < r.x0 - 20 || px > r.x0 + r.w + 20) continue;

      ctx.beginPath();
      ctx.moveTo(px, r.y0);
      ctx.lineTo(px, r.y0 + r.h);
      ctx.stroke();

      ctx.textAlign = "center";
      ctx.textBaseline = "bottom";
      ctx.fillText(item.label, px, canvas.clientHeight - 10);
    }

    // Axes frame.
    ctx.strokeStyle = "#222";
    ctx.lineWidth = 1.2;
    ctx.strokeRect(r.x0, r.y0, r.w, r.h);

    // Y label.
    ctx.save();
    ctx.translate(18, r.y0 + r.h / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    ctx.fillStyle = "#222";
    ctx.font = "13px system-ui, sans-serif";
    ctx.fillText("Energy / eV", 0, 0);
    ctx.restore();

    ctx.restore();
  }

  function drawBands() {
    const r = plotRect();

    ctx.save();

    // Clip to plot region.
    ctx.beginPath();
    ctx.rect(r.x0, r.y0, r.w, r.h);
    ctx.clip();

    ctx.strokeStyle = "#254f7a";
    ctx.lineWidth = 1.35;

    for (let b = 0; b < nbands; b++) {
      ctx.beginPath();

      let started = false;

      for (let i = 0; i < nk; i++) {
        const x = xData[i];
        const y = energies[i][b];

        // Skip far outside current view for speed/cleanliness.
        if (x < view.xmin || x > view.xmax) {
          started = false;
          continue;
        }

        const px = sx(x);
        const py = sy(y);

        if (!started) {
          ctx.moveTo(px, py);
          started = true;
        } else {
          ctx.lineTo(px, py);
        }
      }

      ctx.stroke();
    }

    ctx.restore();
  }

  function draw() {
    drawAxes();
    drawBands();
  }

  function zoomAt(clientX, clientY, factor) {
    const rect = canvas.getBoundingClientRect();
    const px = clientX - rect.left;
    const py = clientY - rect.top;

    const x = invx(px);
    const y = invy(py);

    const newW = (view.xmax - view.xmin) * factor;
    const newH = (view.ymax - view.ymin) * factor;

    const rx = (x - view.xmin) / (view.xmax - view.xmin);
    const ry = (view.ymax - y) / (view.ymax - view.ymin);

    view.xmin = x - rx * newW;
    view.xmax = view.xmin + newW;

    view.ymax = y + ry * newH;
    view.ymin = view.ymax - newH;

    draw();
  }

  canvas.addEventListener("wheel", (event) => {
    event.preventDefault();
    const factor = event.deltaY < 0 ? 0.85 : 1.18;
    zoomAt(event.clientX, event.clientY, factor);
  }, { passive: false });

  canvas.addEventListener("pointerdown", (event) => {
    dragging = true;
    last = [event.clientX, event.clientY];
    canvas.classList.add("dragging");
    canvas.setPointerCapture(event.pointerId);
  });

  canvas.addEventListener("pointermove", (event) => {
    if (!dragging || !last) return;

    const rect = canvas.getBoundingClientRect();

    const dx = (event.clientX - last[0]) / rect.width * (view.xmax - view.xmin);
    const dy = (event.clientY - last[1]) / rect.height * (view.ymax - view.ymin);

    view.xmin -= dx;
    view.xmax -= dx;

    view.ymin += dy;
    view.ymax += dy;

    last = [event.clientX, event.clientY];
    draw();
  });

  canvas.addEventListener("pointerup", (event) => {
    dragging = false;
    last = null;
    canvas.classList.remove("dragging");
    canvas.releasePointerCapture(event.pointerId);
  });

  canvas.addEventListener("pointercancel", () => {
    dragging = false;
    last = null;
    canvas.classList.remove("dragging");
  });

  document.getElementById("band-zoom-in")?.addEventListener("click", () => {
    const rect = canvas.getBoundingClientRect();
    zoomAt(rect.left + rect.width / 2, rect.top + rect.height / 2, 0.8);
  });

  document.getElementById("band-zoom-out")?.addEventListener("click", () => {
    const rect = canvas.getBoundingClientRect();
    zoomAt(rect.left + rect.width / 2, rect.top + rect.height / 2, 1.25);
  });

  document.getElementById("band-reset")?.addEventListener("click", () => {
    view = { ...full };
    draw();
  });

  window.addEventListener("resize", resizeCanvas);

  resizeCanvas();
})();
</script>
"""


def fmt(x: Any, digits: int = 4) -> str:
    if x is None:
        return "—"
    if isinstance(x, (np.integer, int)):
        return str(int(x))
    if isinstance(x, (np.floating, float)):
        x = float(x)
        if not np.isfinite(x):
            return "—"
        if x == 0:
            return "0"
        ax = abs(x)
        if ax < 1e-3 or ax >= 1e4:
            return f"{x:.{digits}e}"
        return f"{x:.{digits}g}"
    return escape(str(x))


def page(title: str, body: str) -> HTMLResponse:
    html = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>{escape(title)}</title>
  <style>{CSS}</style>
</head>
<body>
<header>
  <h1>{escape(title)}</h1>
  <nav>
    <a href="/">Overview</a>
    <a href="/geometry">Geometry</a>
    <a href="/kernels">Kernels</a>
    <a href="/symbol?k1=0.1&k2=0.2">Symbol</a>
    <a href="/viewer">Viewer</a>
    <a href="/bands">Bands</a>
  </nav>
</header>
<main>
{body}
</main>
</body>
</html>"""
    return HTMLResponse(html)


def metric_card(label: str, value: Any, note: str = "") -> str:
    return f"""
<div class="card">
  <div class="label">{escape(label)}</div>
  <div class="metric">{fmt(value)}</div>
  <div class="small">{escape(note)}</div>
</div>
"""


def dict_table(d: dict[str, Any]) -> str:
    rows = []
    for key, value in d.items():
        rows.append(
            f"<tr><th>{escape(str(key))}</th><td class='num'>{fmt(value)}</td></tr>"
        )
    return "<table>" + "\n".join(rows) + "</table>"


def rows_table(headers: list[str], rows: list[list[Any]], numeric: set[int] | None = None) -> str:
    numeric = numeric or set()
    hs = "".join(
        f"<th class='{'num' if i in numeric else ''}'>{escape(h)}</th>"
        for i, h in enumerate(headers)
    )
    body = []
    for row in rows:
        tds = "".join(
            f"<td class='{'num' if i in numeric else ''}'>{fmt(v)}</td>"
            for i, v in enumerate(row)
        )
        body.append(f"<tr>{tds}</tr>")
    return f"<table><thead><tr>{hs}</tr></thead><tbody>{''.join(body)}</tbody></table>"


def kernel_summary_row(name: str, K: GdKernelArrays) -> list[Any]:
    d = K.diagnostics()
    s = K.star_defect()

    return [
        name,
        d["support_size"],
        d["num_even"],
        d["num_odd"],
        s["num_missing_inverse"],
        s["star_defect_mean"],
        s["star_defect_max"],
    ]


def symbol_diag_row(name: str, K: GdKernelArrays, k1: float, k2: float) -> list[Any]:
    A = K.symbol_generic(k1, k2)
    d = DenseMatrixDiagnostics.from_dense_matrix(A, name=name)

    return [
        name,
        d.shape,
        d.norm,
        d.hermitian_defect_rel,
    ]


def local_symbol_rows(state: DiagnosticsState, k1: float, k2: float) -> list[list[Any]]:
    return [
        symbol_diag_row("H anchored", state.KH, k1, k2),
        symbol_diag_row("H average", state.KH_avg, k1, k2),
        symbol_diag_row("H average star", state.KH_avg_star, k1, k2),
        symbol_diag_row("S anchored", state.KS, k1, k2),
        symbol_diag_row("S average", state.KS_avg, k1, k2),
        symbol_diag_row("S average star", state.KS_avg_star, k1, k2),
    ]

def viewer_payload(state: DiagnosticsState) -> dict:
    geom = state.geom
    labels = state.labels

    atoms = []
    for a in range(geom.natoms):
        atoms.append({
            "id": int(a),
            "symbol": str(state.data.metadata.symbols[a]),
            "x": float(geom.positions[a, 0]),
            "y": float(geom.positions[a, 1]),
            "z": float(geom.positions[a, 2]),
            "m": int(labels.m[a]),
            "n": int(labels.n[a]),
            "eps": int(labels.eps[a]),
            "degree": int(geom.degree[a]),
        })

    bonds = []
    seen = set()
    for a in range(geom.natoms):
        for b in geom.neighbours(a):
            b = int(b)
            key = tuple(sorted((a, b)))
            if key in seen:
                continue
            seen.add(key)
            bonds.append({"a": int(a), "b": int(b)})

    return {
        "natoms": int(geom.natoms),
        "atoms": atoms,
        "bonds": bonds,
        "anchor_atom": int(labels.anchor_atom),
    }

def create_app(root: str | Path = "test_run/run_dir/data") -> FastAPI:
    app = FastAPI(title="DFT local diagnostics")
    state = build_state(root)
    app.state.diagnostics_state = state

    @app.get("/", response_class=HTMLResponse)
    def overview():
        s: DiagnosticsState = app.state.diagnostics_state
        gdiag = s.geom.diagnostics()
        ldiag = s.labels.diagnostics()

        body = f"""
<section class="grid">
  {metric_card("Atoms", s.data.metadata.natoms)}
  {metric_card("Basis size", s.data.metadata.nbasis)}
  {metric_card("Channels / atom", s.data.basis.nchannels)}
  {metric_card("Anchor atom", s.labels.anchor_atom)}
  {metric_card("Bulk atoms", gdiag["num_bulk_atoms"])}
  {metric_card("Core bulk atoms", gdiag["num_core_bulk_atoms"])}
</section>

<h2>Label diagnostics</h2>
{dict_table({
    "visited": ldiag["visited_count"],
    "unvisited": ldiag["unvisited_count"],
    "m_min": ldiag["m_min"],
    "m_max": ldiag["m_max"],
    "n_min": ldiag["n_min"],
    "n_max": ldiag["n_max"],
})}

<h2>Kernel summary</h2>
{rows_table(
    ["kernel", "support", "even", "odd", "missing inverses", "mean star defect", "max star defect"],
    [
        kernel_summary_row("H anchored", s.KH),
        kernel_summary_row("H average", s.KH_avg),
        kernel_summary_row("H average star", s.KH_avg_star),
    ],
    numeric={1,2,3,4,5,6},
)}
"""
        return page("DFT local diagnostics", body)

    @app.get("/geometry", response_class=HTMLResponse)
    def geometry():
        s: DiagnosticsState = app.state.diagnostics_state
        gdiag = s.geom.diagnostics()
        ediag = s.edges.diagnostics(s.geom)
        ldiag = s.labels.diagnostics()

        degree_rows = [
            [degree, count]
            for degree, count in sorted(gdiag["degree_counts"].items())
        ]

        pos_err = ldiag.get("position_reconstruction_errors") or ldiag.get("positino_reconstruction_errors", {})

        body = f"""
<h2>Nearest-neighbour graph</h2>
<section class="grid">
  {metric_card("Bond length a0", gdiag["a0"], "Estimated median nearest-neighbour distance")}
  {metric_card("Cutoff", gdiag["cutoff"])}
  {metric_card("Bulk atoms", gdiag["num_bulk_atoms"])}
  {metric_card("Core bulk atoms", gdiag["num_core_bulk_atoms"])}
</section>

<h3>Degree counts</h3>
{rows_table(["degree", "count"], degree_rows, numeric={0,1})}

<h2>Edge directions</h2>
{dict_table({
    "anchor_atom": ediag["anchor_atom"],
    "alignment_min": ediag["alignment_min"],
    "alignment_median": ediag["alignment_median"],
    "alignment_max": ediag["alignment_max"],
})}

<h2>Group labels</h2>
{dict_table({
    "visited_count": ldiag["visited_count"],
    "unvisited_count": ldiag["unvisited_count"],
    "m_min": ldiag["m_min"],
    "m_max": ldiag["m_max"],
    "n_min": ldiag["n_min"],
    "n_max": ldiag["n_max"],
    "position_error_max": pos_err.get("max"),
    "position_error_mean": pos_err.get("mean"),
})}
"""
        return page("Geometry diagnostics", body)

    @app.get("/kernels", response_class=HTMLResponse)
    def kernels():
        s: DiagnosticsState = app.state.diagnostics_state

        rows = [
            kernel_summary_row("H anchored", s.KH),
            kernel_summary_row("H average", s.KH_avg),
            kernel_summary_row("H average star", s.KH_avg_star),
            kernel_summary_row("S anchored", s.KS),
            kernel_summary_row("S average", s.KS_avg),
            kernel_summary_row("S average star", s.KS_avg_star),
        ]

        worst = s.KH_avg.star_defect_table_filtered(min_norm=1e-2, max_radius=3).head(20)
        worst_rows = [
            [
                int(r.m), int(r.n), int(r.eps),
                r.norm, r.inv_norm, r.star_error, r.star_relative_error,
            ]
            for r in worst.itertuples()
        ]

        body = f"""
<h2>Kernel star defects</h2>
{rows_table(
    ["kernel", "support", "even", "odd", "missing inverses", "mean star defect", "max star defect"],
    rows,
    numeric={1,2,3,4,5,6},
)}

<h2>Worst local star defects for averaged H</h2>
<p class="small">Filtered by <code>min_norm=1e-2</code> and <code>max_radius=3</code>.</p>
{rows_table(
    ["m", "n", "eps", "norm", "inv norm", "star error", "relative"],
    worst_rows,
    numeric={0,1,2,3,4,5,6},
)}
"""
        return page("Kernel diagnostics", body)

    @app.get("/symbol", response_class=HTMLResponse)
    def symbol(
        k1: float = Query(0.1),
        k2: float = Query(0.2),
    ):
        s: DiagnosticsState = app.state.diagnostics_state

        pair = SymbolPair(s.KH_avg_star, s.KS_avg_star, k1, k2, name="average star")
        local = pair.form()
        E = local.energies()

        Hdiag = DenseMatrixDiagnostics.from_dense_matrix(local.Hk, name="H(k)")
        Sdiag = DenseMatrixDiagnostics.from_dense_matrix(local.Sk, name="S(k)", check_eigenvalues=True)

        energy_rows = [[i, e] for i, e in enumerate(E)]

        body = f"""
<form class="inline" method="get" action="/symbol">
  <label>k1<br><input name="k1" value="{fmt(k1)}"></label>
  <label>k2<br><input name="k2" value="{fmt(k2)}"></label>
  <button type="submit">Update</button>
</form>

<h2>Symbol diagnostics</h2>
{rows_table(
    ["symbol", "shape", "norm", "Hermitian defect"],
    local_symbol_rows(s, k1, k2),
    numeric={2,3},
)}

<h2>Effective problem used for energies</h2>
{dict_table({
    "H hermitian defect": Hdiag.hermitian_defect_rel,
    "S hermitian defect": Sdiag.hermitian_defect_rel,
    "S eig min": Sdiag.eig_min,
    "S eig max": Sdiag.eig_max,
    "S condition": Sdiag.condition_number_abs,
})}

<h2>Energies</h2>
{rows_table(["band", "energy / eV"], energy_rows, numeric={0,1})}
"""
        return page("Symbol diagnostics", body)


    @app.get("/viewer/data")
    def viewer_data():
        s: DiagnosticsState = app.state.diagnostics_state
        return JSONResponse(viewer_payload(s))


    @app.get("/viewer", response_class=HTMLResponse)
    def viewer():
        html = """<!doctype html>
    <html>
    <head>
      <meta charset="utf-8">
      <title>Graphene viewer</title>
      <style>""" + f'{VIEWER_CSS}' + """</style>

      <script type="importmap">
      {
        "imports": {
          "three": "https://cdn.jsdelivr.net/npm/three@0.184.0/build/three.module.js",
          "three/addons/": "https://cdn.jsdelivr.net/npm/three@0.184.0/examples/jsm/"
        }
      }
      </script>
    </head>
    <body>
      <div id="app">
        <div id="viewer"></div>
        <aside id="panel">
          <h1>Graphene viewer</h1>
          <p class="small">Drag to rotate. Scroll to zoom. Click an atom.</p>
          <div id="atom-info">No atom selected.</div>
          <p><a href="/">Back to diagnostics</a></p>
        </aside>
      </div>

      <script type="module">
        import * as THREE from "three";
        import { OrbitControls } from "three/addons/controls/OrbitControls.js";

        const container = document.getElementById("viewer");
        const info = document.getElementById("atom-info");

        const scene = new THREE.Scene();
        scene.background = new THREE.Color(0x101216);

        const camera = new THREE.PerspectiveCamera(
          50,
          container.clientWidth / container.clientHeight,
          0.1,
          10000
        );

        const renderer = new THREE.WebGLRenderer({ antialias: true });
        renderer.setPixelRatio(window.devicePixelRatio);
        renderer.setSize(container.clientWidth, container.clientHeight);
        container.appendChild(renderer.domElement);

        const controls = new OrbitControls(camera, renderer.domElement);
        controls.enableDamping = true;

        const light = new THREE.DirectionalLight(0xffffff, 1.0);
        light.position.set(0, 0, 100);
        scene.add(light);

        const ambient = new THREE.AmbientLight(0xffffff, 0.6);
        scene.add(ambient);

        const data = await fetch("/viewer/data").then(r => r.json());

        const positions = data.atoms.map(a => new THREE.Vector3(a.x, a.y, a.z));
        const center = new THREE.Vector3();
        for (const p of positions) center.add(p);
        center.multiplyScalar(1 / positions.length);

        for (const p of positions) p.sub(center);

        const atomGroup = new THREE.Group();
        const bondGroup = new THREE.Group();
        scene.add(bondGroup);
        scene.add(atomGroup);

        const atomMaterialA = new THREE.MeshStandardMaterial({ color: 0x5dade2, roughness: 0.45 });
        const atomMaterialB = new THREE.MeshStandardMaterial({ color: 0xf5b041, roughness: 0.45 });
        const anchorMaterial = new THREE.MeshStandardMaterial({ color: 0xe74c3c, roughness: 0.35 });
        const selectedMaterial = new THREE.MeshStandardMaterial({ color: 0x2ecc71, roughness: 0.25 });
        const bondMaterial = new THREE.LineBasicMaterial({ color: 0x888888 });

        const sphereGeometry = new THREE.SphereGeometry(0.22, 16, 16);
        const atomMeshes = new Map();
        let selected = null;
        let selectedOldMaterial = null;

        for (let i = 0; i < data.atoms.length; i++) {
          const atom = data.atoms[i];
          const material =
            atom.id === data.anchor_atom ? anchorMaterial :
            atom.eps === 0 ? atomMaterialA : atomMaterialB;

          const mesh = new THREE.Mesh(sphereGeometry, material);
          mesh.position.copy(positions[i]);
          mesh.userData.atom = atom;
          atomGroup.add(mesh);
          atomMeshes.set(atom.id, mesh);
        }

        const bondPositions = [];
        for (const bond of data.bonds) {
          const pa = positions[bond.a];
          const pb = positions[bond.b];
          bondPositions.push(pa.x, pa.y, pa.z, pb.x, pb.y, pb.z);
        }

        const bondGeometry = new THREE.BufferGeometry();
        bondGeometry.setAttribute(
          "position",
          new THREE.Float32BufferAttribute(bondPositions, 3)
        );
        const bondLines = new THREE.LineSegments(bondGeometry, bondMaterial);
        bondGroup.add(bondLines);

        camera.position.set(0, 0, 85);
        controls.target.set(0, 0, 0);
        controls.update();

        const raycaster = new THREE.Raycaster();
        const mouse = new THREE.Vector2();

        let hovered = null;
        let hoveredOldMaterial = null;

        const hoveredMaterial = new THREE.MeshStandardMaterial({
          color: 0xffffff,
          roughness: 0.2
        });

        function atomHtml(atom, title = "Atom") {
          return `
            <h1>${title} ${atom.id}</h1>
            <dl>
              <dt>symbol</dt><dd>${atom.symbol}</dd>
              <dt>degree</dt><dd>${atom.degree}</dd>
              <dt>eps</dt><dd>${atom.eps}</dd>
              <dt>m</dt><dd>${atom.m}</dd>
              <dt>n</dt><dd>${atom.n}</dd>
              <dt>x</dt><dd>${atom.x.toFixed(6)}</dd>
              <dt>y</dt><dd>${atom.y.toFixed(6)}</dd>
              <dt>z</dt><dd>${atom.z.toFixed(6)}</dd>
            </dl>
            <p><a href="/atom/${atom.id}">Open atom diagnostics</a></p>
          `;
        }
        function setPanelForAtom(atom, mode) {
          const title = mode === "hover" ? "Hovered atom" : "Selected atom";
          info.innerHTML = atomHtml(atom, title);
        }

        function clearHoverPanel() {
          if (selected) {
            setPanelForAtom(selected.userData.atom, "selected");
          } else {
            info.innerHTML = "No atom selected.";
          }
        }

        function setHovered(mesh) {
          if (hovered === mesh) {
            return;
          }

          if (hovered && hovered !== selected) {
            hovered.material = baseMaterialForAtom(hovered.userData.atom);
          }

          hovered = mesh;

          if (hovered) {
            if (hovered !== selected) {
              hovered.material = hoveredMaterial;
            }
            setPanelForAtom(hovered.userData.atom, "hover");
            renderer.domElement.style.cursor = "pointer";
          } else {
            clearHoverPanel();
            renderer.domElement.style.cursor = "default";
          }
        }


        function pickAtom(event) {
          const rect = renderer.domElement.getBoundingClientRect();

          mouse.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
          mouse.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;

          raycaster.setFromCamera(mouse, camera);
          const hits = raycaster.intersectObjects(atomGroup.children, false);

          if (hits.length > 0) {
            return hits[0].object;
          }

          return null;
        }

        function baseMaterialForAtom(atom) {
          if (atom.id === data.anchor_atom) {
            return anchorMaterial;
          }
          return atom.eps === 0 ? atomMaterialA : atomMaterialB;
        }

        function selectAtom(mesh) {
          if (selected && selected !== hovered) {
            selected.material = baseMaterialForAtom(selected.userData.atom);
          }

          selected = mesh;
          selectedOldMaterial = baseMaterialForAtom(mesh.userData.atom);

          mesh.material = selectedMaterial;

          const atom = mesh.userData.atom;
          setPanelForAtom(atom, "selected");
        }
        

        renderer.domElement.addEventListener("click", event => {
          const rect = renderer.domElement.getBoundingClientRect();

          mouse.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
          mouse.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;

          raycaster.setFromCamera(mouse, camera);
          const hits = raycaster.intersectObjects(atomGroup.children, false);

          if (hits.length > 0) {
            selectAtom(hits[0].object);
          }
        });

        window.addEventListener("resize", () => {
          camera.aspect = container.clientWidth / container.clientHeight;
          camera.updateProjectionMatrix();
          renderer.setSize(container.clientWidth, container.clientHeight);
        });

        renderer.domElement.addEventListener("mousemove", event => {
          const mesh = pickAtom(event);
          setHovered(mesh);
        });

        renderer.domElement.addEventListener("click", event => {
          const mesh = pickAtom(event);

          if (mesh) {
            selectAtom(mesh);
          }
        });

        function animate() {
          requestAnimationFrame(animate);
          controls.update();
          renderer.render(scene, camera);
        }

        animate();
      </script>
    </body>
    </html>"""
        return HTMLResponse(html)

    @app.get("/atom/{atom_id}", response_class=HTMLResponse)
    def atom_detail(atom_id: int):
        s: DiagnosticsState = app.state.diagnostics_state

        if atom_id < 0 or atom_id >= s.geom.natoms:
            return page("Atom not found", f"<p>No atom {atom_id}</p>")

        g = s.labels.element(atom_id)
        ns = s.geom.neighbours(atom_id)

        neighbour_rows = []
        for b in ns:
            b = int(b)
            h = s.labels.relative(atom_id, b)
            neighbour_rows.append([
                b,
                int(s.labels.eps[b]),
                h.m,
                h.n,
                h.eps,
                float(np.linalg.norm(s.geom.positions[b] - s.geom.positions[atom_id])),
            ])

        body = f"""
    <h2>Atom {atom_id}</h2>
    <section class="grid">
      {metric_card("symbol", s.data.metadata.symbols[atom_id])}
      {metric_card("degree", s.geom.degree[atom_id])}
      {metric_card("m", g.m)}
      {metric_card("n", g.n)}
      {metric_card("eps", g.eps)}
    </section>

    <h2>Neighbours</h2>
    {rows_table(
        ["atom", "eps", "rel m", "rel n", "rel eps", "distance"],
        neighbour_rows,
        numeric={0,1,2,3,4,5},
    )}

    <p><a href="/viewer">Back to viewer</a></p>
    """
        return page(f"Atom {atom_id}", body)

    @app.get("/bands", response_class=HTMLResponse)
    def bands(
        points_per_segment: int = Query(80, ge=4, le=1000),
    ):
        s: DiagnosticsState = app.state.diagnostics_state

        points = [
            ("Γ", 0.0, 0.0),
            ("K", 2 * np.pi / 3, -2 * np.pi / 3),
            ("M", np.pi, 0.0),
            ("Γ", 0.0, 0.0),
        ]

        path = LocalPath.from_points(
            s.KH_avg_star,
            s.KS_avg_star,
            points,
            points_per_segment=points_per_segment,
            name="bulk averaged star-symmetrised",
        )

        E = path.energies()
        plot = canvas_band_plot(band_plot_payload(path.x, E, path.labels))
        kplot = svg_k_path_plot(path.k1, path.k2, path.labels)

        energy_rows = [
            ["min", float(np.min(E))],
            ["max", float(np.max(E))],
            ["band count", int(E.shape[1])],
            ["k-points", int(E.shape[0])],
        ]

        label_rows = [
            [
                label,
                int(index),
                float(path.x[index]),
                float(path.k1[index]),
                float(path.k2[index]),
            ]
            for index, label in path.labels
        ]

        

        body = f"""
<h2>Band path</h2>

<form class="inline" method="get" action="/bands">
  <label>points per segment<br>
    <input name="points_per_segment" value="{int(points_per_segment)}">
  </label>
  <button type="submit">Update</button>
</form>

<p class="small">
  Kernel: averaged star-symmetrised H and S. Path: Γ → K → M → Γ.
</p>

<div class="plot-grid">
  <div class="card kpath-card">
    <h3>k-space path</h3>
    {kplot}
  </div>

  <div class="card">
    <h3>Band energies</h3>
    {plot}
  </div>
</div>

{BAND_CANVAS_JS}

<h2>Summary</h2>
{rows_table(["quantity", "value"], energy_rows, numeric={1})}

<h2>Path labels</h2>
{rows_table(["label", "index", "x", "k1", "k2"], label_rows, numeric={1,2,3,4})}
"""

        return page("Band diagnostics", body)

    return app


app = create_app()
