"""Minimal HTML renderer for diagnostics."""

from __future__ import annotations

from html import escape
from typing import Any
import math

from dft_local.diagnostics.models import DiagnosticResult, Graph2D


def fmt(value: Any) -> str:
    """Format values for compact diagnostic display."""

    if value is None:
        return "—"

    if isinstance(value, dict):
        if {"real", "imag", "abs"} <= set(value):
            return (
                f"real={fmt(value['real'])}; "
                f"imag={fmt(value['imag'])}; "
                f"abs={fmt(value['abs'])}"
            )
        return ", ".join(f"{key}={fmt(val)}" for key, val in value.items())

    if isinstance(value, float):
        if value == 0.0:
            return "0"
        if abs(value) < 1e-4 or abs(value) > 1e5:
            return f"{value:.6e}"
        return f"{value:.6g}"

    if isinstance(value, complex):
        return f"{value.real:.6e} + {value.imag:.6e} i"

    return str(value)


def _nice(value: float) -> str:
    if not math.isfinite(value):
        return ""
    if value == 0:
        return "0"
    if abs(value) < 1e-3 or abs(value) >= 1e4:
        return f"{value:.2e}"
    return f"{value:.4g}"


def _render_graph_svg(graph: Graph2D) -> str:
    """Render a Graph2D as static inline SVG."""

    width = 1000.0
    height = 520.0
    left = 78.0
    right = 24.0
    top = 28.0
    bottom = 62.0
    inner_w = width - left - right
    inner_h = height - top - bottom

    all_points = [
        point
        for series in graph.series
        for point in series.points
        if math.isfinite(point.x) and math.isfinite(point.y)
    ]

    if not all_points:
        return "<p>No graph data.</p>"

    xmin = min(point.x for point in all_points)
    xmax = max(point.x for point in all_points)
    ymin = min(point.y for point in all_points)
    ymax = max(point.y for point in all_points)

    if xmin == xmax:
        xmin -= 1.0
        xmax += 1.0
    if ymin == ymax:
        ymin -= 1.0
        ymax += 1.0

    ypad = 0.06 * (ymax - ymin)
    ymin -= ypad
    ymax += ypad

    def sx(x: float) -> float:
        return left + (x - xmin) / (xmax - xmin) * inner_w

    def sy(y: float) -> float:
        return top + (ymax - y) / (ymax - ymin) * inner_h

    colours = (
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
    )

    parts: list[str] = [
        (
            "<svg class='graph-svg' "
            f"viewBox='0 0 {width:.0f} {height:.0f}' "
            "role='img' "
            f"aria-label='{escape(graph.title, quote=True)}'>"
        ),
        "<rect class='graph-bg' x='0' y='0' width='1000' height='520' />",
    ]

    for i in range(6):
        t = i / 5.0
        x = left + t * inner_w
        xv = xmin + t * (xmax - xmin)
        parts.append(f"<line class='grid' x1='{x:.3f}' y1='{top:.3f}' x2='{x:.3f}' y2='{top + inner_h:.3f}' />")
        parts.append(f"<text class='axis-label' x='{x:.3f}' y='{height - 24:.3f}' text-anchor='middle'>{escape(_nice(xv))}</text>")

    for i in range(6):
        t = i / 5.0
        y = top + t * inner_h
        yv = ymax - t * (ymax - ymin)
        parts.append(f"<line class='grid' x1='{left:.3f}' y1='{y:.3f}' x2='{left + inner_w:.3f}' y2='{y:.3f}' />")
        parts.append(f"<text class='axis-label' x='{left - 10:.3f}' y='{y + 4:.3f}' text-anchor='end'>{escape(_nice(yv))}</text>")

    parts.append(f"<line class='axis' x1='{left:.3f}' y1='{top + inner_h:.3f}' x2='{left + inner_w:.3f}' y2='{top + inner_h:.3f}' />")
    parts.append(f"<line class='axis' x1='{left:.3f}' y1='{top:.3f}' x2='{left:.3f}' y2='{top + inner_h:.3f}' />")
    parts.append(f"<text class='axis-title' x='{left + inner_w / 2:.3f}' y='{height - 6:.3f}' text-anchor='middle'>{escape(graph.x_label)}</text>")
    parts.append(
        f"<text class='axis-title' x='18' y='{top + inner_h / 2:.3f}' text-anchor='middle' "
        f"transform='rotate(-90 18 {top + inner_h / 2:.3f})'>{escape(graph.y_label)}</text>"
    )

    for series_index, series in enumerate(graph.series):
        colour = colours[series_index % len(colours)]
        points = [
            (sx(point.x), sy(point.y))
            for point in series.points
            if math.isfinite(point.x) and math.isfinite(point.y)
        ]

        if not points:
            continue

        if series.kind in {"line", "line_points"}:
            d = " ".join(
                f"{'M' if i == 0 else 'L'} {x:.3f} {y:.3f}"
                for i, (x, y) in enumerate(points)
            )
            parts.append(
                f"<path class='series-line' d='{d}' fill='none' "
                f"stroke='{colour}' stroke-width='1.8' vector-effect='non-scaling-stroke' />"
            )

        if series.kind in {"points", "line_points"}:
            for x, y in points:
                parts.append(f"<circle cx='{x:.3f}' cy='{y:.3f}' r='2.5' fill='{colour}' />")

        lx, ly = points[-1]
        parts.append(
            f"<text class='series-label' x='{lx + 5:.3f}' y='{ly + 4:.3f}' "
            f"fill='{colour}'>{escape(series.name)}</text>"
        )

    parts.append("</svg>")
    return "\n".join(parts)


def render_result(result: DiagnosticResult) -> str:
    """Render a diagnostic result body as simple HTML."""

    parts: list[str] = []

    parts.append(f"<h1>{escape(result.title)}</h1>")
    parts.append(f"<p>{escape(result.summary)}</p>")

    if result.cards:
        parts.append("<section class='cards'>")
        for card in result.cards:
            parts.append(
                "<article class='card'>"
                f"<h3>{escape(card.label)}</h3>"
                f"<p class='{escape(card.status)}'>{escape(fmt(card.value))}</p>"
                f"<small>{escape(card.help)}</small>"
                "</article>"
            )
        parts.append("</section>")

    for matrix in result.matrices:
        parts.append(f"<section><h2>{escape(matrix.title)}</h2>")
        parts.append(f"<p>{escape(matrix.description)}</p>")
        parts.append("<table><thead><tr><th></th>")
        for label in matrix.col_labels:
            parts.append(f"<th>{escape(label)}</th>")
        parts.append("</tr></thead><tbody>")

        cells = {(cell.i, cell.j): cell.value for cell in matrix.cells}
        for i, row_label in enumerate(matrix.row_labels):
            parts.append(f"<tr><th>{escape(row_label)}</th>")
            for j, _col_label in enumerate(matrix.col_labels):
                parts.append(f"<td>{escape(fmt(cells.get((i, j))))}</td>")
            parts.append("</tr>")

        parts.append("</tbody></table></section>")

    for graph in result.graphs:
        parts.append(f"<section class='graph-panel'><h2>{escape(graph.title)}</h2>")
        parts.append(f"<p>{escape(graph.description)}</p>")
        parts.append(_render_graph_svg(graph))
        parts.append("</section>")

    for table in result.tables:
        parts.append(f"<section><h2>{escape(table.title)}</h2>")
        parts.append(f"<p>{escape(table.description)}</p>")
        parts.append("<table><thead><tr>")
        for header in table.headers:
            parts.append(f"<th>{escape(header)}</th>")
        parts.append("</tr></thead><tbody>")

        for row in table.rows:
            parts.append("<tr>")
            for cell in row.cells:
                parts.append(f"<td>{escape(fmt(cell))}</td>")
            parts.append("</tr>")

        parts.append("</tbody></table></section>")

    if result.notes:
        parts.append("<section><h2>Notes</h2>")
        for note in result.notes:
            parts.append(f"<p>{escape(note)}</p>")
        parts.append("</section>")

    return "\n".join(parts)


def render_page(title: str, body: str) -> str:
    """Render a full diagnostic HTML page."""

    return f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>{escape(title)}</title>
  <style>
    body {{ font-family: system-ui, sans-serif; margin: 2rem; line-height: 1.45; background: #f6f7f9; color: #111827; }}
    body > h1, body > p, body > section, body > form {{ max-width: 1200px; margin-left: auto; margin-right: auto; }}
    table {{ border-collapse: collapse; width: 100%; margin: 1rem 0 2rem; background: white; }}
    th, td {{ border: 1px solid #ddd; padding: 0.35rem 0.5rem; text-align: left; }}
    th {{ background: #f5f5f5; }}
    .cards {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(12rem, 1fr)); gap: 1rem; }}
    .card, section, form {{ border: 1px solid #ddd; border-radius: 0.75rem; padding: 1rem; background: white; margin-bottom: 1rem; }}
    .ok {{ color: #166534; }}
    .warn {{ color: #92400e; }}
    .bad {{ color: #991b1b; }}
    .neutral {{ color: #374151; }}
    nav a {{ margin-right: 1rem; }}
    input, select, button {{ font: inherit; padding: 0.35rem 0.5rem; margin: 0.15rem; }}
    button {{ font-weight: 700; }}
    .graph-svg {{ display: block; width: 100%; height: 32rem; border: 1px solid #d1d5db; border-radius: 0.5rem; background: white; }}
    .graph-bg {{ fill: white; }}
    .grid {{ stroke: #e5e7eb; stroke-width: 1; }}
    .axis {{ stroke: #6b7280; stroke-width: 1.2; }}
    .axis-label, .axis-title, .series-label {{ font-size: 12px; fill: #6b7280; }}
  </style>
</head>
<body>
{body}
</body>
</html>"""
