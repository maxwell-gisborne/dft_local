"""Minimal HTML renderer for diagnostics."""

from __future__ import annotations

from html import escape
import re
from typing import Any
import math
import json

from dft_local.diagnostics.models import DiagnosticSection, MarkdownBlock, DiagnosticResult, Graph2D


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


def render_markdown_block(block: MarkdownBlock) -> str:
    return (
        f"<section id='{escape(block.id)}'>"
        f"<h2>{escape(block.title)}</h2>"
        + DiagnosticApp.render_markdown(block.markdown)
        + "</section>"
    )


def render_markdown_text(markdown: str) -> str:
    lines = markdown.splitlines()
    html: list[str] = []
    in_list = False
    paragraph: list[str] = []

    def flush_paragraph() -> None:
        nonlocal paragraph
        if paragraph:
            html.append("<p>" + " ".join(escape(line.strip()) for line in paragraph) + "</p>")
            paragraph = []

    def close_list() -> None:
        nonlocal in_list
        if in_list:
            html.append("</ul>")
            in_list = False

    for line in lines:
        stripped = line.rstrip()

        if not stripped:
            flush_paragraph()
            close_list()
            continue

        heading = re.match(r"^(#{1,6})\s+(.*)$", stripped)
        if heading:
            flush_paragraph()
            close_list()
            level = len(heading.group(1))
            html.append(f"<h{level}>{escape(heading.group(2).strip())}</h{level}>")
            continue

        if stripped.startswith("- "):
            flush_paragraph()
            if not in_list:
                html.append("<ul>")
                in_list = True
            html.append(f"<li>{escape(stripped[2:].strip())}</li>")
            continue

        paragraph.append(stripped)

    flush_paragraph()
    close_list()
    return "\n".join(html)


def render_markdown_block(block: MarkdownBlock) -> str:
    return (
        f"<section id='{escape(block.id)}' class='markdown-block'>"
        f"<h2>{escape(block.title)}</h2>"
        + render_markdown_text(block.markdown)
        + "</section>"
    )


def render_card(card) -> str:
    title = getattr(card, "title", getattr(card, "label", ""))
    value = getattr(card, "value", "")
    subtitle = getattr(card, "subtitle", getattr(card, "kind", ""))

    return (
        "<div class='card'>"
        f"<strong>{escape(str(title))}</strong>"
        f"<div>{escape(str(value))}</div>"
        f"<small>{escape(str(subtitle))}</small>"
        "</div>"
    )


def render_table(table) -> str:
    headers = "".join(f"<th>{escape(str(header))}</th>" for header in table.headers)

    rows = []
    for row in table.rows:
        cells = getattr(row, "cells", row)
        rows.append(
            "<tr>"
            + "".join(f"<td>{escape(str(cell))}</td>" for cell in cells)
            + "</tr>"
        )

    description = ""
    if getattr(table, "description", ""):
        description = f"<p><small>{escape(str(table.description))}</small></p>"

    return (
        f"<section id='{escape(str(table.id))}'>"
        f"<h3>{escape(str(table.title))}</h3>"
        + description
        + "<table><thead><tr>"
        + headers
        + "</tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table></section>"
    )


def render_diagnostic_section(section: DiagnosticSection) -> str:
    open_attr = "" if section.collapsed else " open"
    body: list[str] = []

    if section.description:
        body.append(f"<p>{escape(section.description)}</p>")

    for block in section.markdowns:
        body.append(render_markdown_block(block))

    for card in section.cards:
        body.append(render_card(card))

    for table in section.tables:
        body.append(render_table(table))

    for child in section.sections:
        body.append(render_diagnostic_section(child))

    return (
        f"<details id='{escape(section.id)}' class='diagnostic-section'{open_attr}>"
        f"<summary>{escape(section.title)}</summary>"
        + "".join(body)
        + "</details>"
    )


def render_result(result: DiagnosticResult) -> str:
    rendered_markdowns = ''.join(render_markdown_block(block) for block in result.markdowns)
    rendered_sections = ''.join(render_diagnostic_section(section) for section in result.sections)
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
        payload = json.dumps(graph.payload()).replace("</", "<\\/")
        data_id = f"data-{graph.id}"
        component = "dft-kspace-plot" if "kspace" in graph.id else "dft-line-graph"

        parts.append(f"<section class='graph-panel'><h2>{escape(graph.title)}</h2>")
        parts.append(f"<p>{escape(graph.description)}</p>")
        parts.append(
            f"<script type='application/json' id='{escape(data_id)}'>"
            f"{payload}"
            "</script>"
        )
        parts.append(
            f"<{component} data-source='{escape(data_id)}'>"
            f"{_render_graph_svg(graph)}"
            f"</{component}>"
        )
        parts.append("</section>")

    for table in result.tables:
        has_step = "step" in table.headers
        step_index = table.headers.index("step") if has_step else -1
        x_index = table.headers.index("x") if "x" in table.headers else -1
        energy_index = table.headers.index("energy") if "energy" in table.headers else -1
        row_label_index = 0
        energy_index = table.headers.index("energy") if "energy" in table.headers else -1
        row_label_index = 0

        parts.append(f"<section><h2>{escape(table.title)}</h2>")
        parts.append(f"<p>{escape(table.description)}</p>")

        if has_step:
            parts.append(
                f"<div class='table-select-controls' data-table-id='{escape(table.id)}'>"
                "<button type='button' data-table-select='all'>Select all</button>"
                "<button type='button' data-table-select='none'>Clear all</button>"
                "</div>"
            )

        parts.append("<table><thead><tr>")
        if has_step:
            parts.append("<th>select</th>")
        for header in table.headers:
            parts.append(f"<th>{escape(header)}</th>")
        parts.append("</tr></thead><tbody>")

        for row in table.rows:
            step_value = row.cells[step_index] if has_step else None
            x_value = row.cells[x_index] if x_index >= 0 else step_value
            energy_value = row.cells[energy_index] if energy_index >= 0 else ""
            row_label_value = row.cells[row_label_index] if row.cells else step_value
            energy_value = row.cells[energy_index] if energy_index >= 0 else ""
            row_label_value = row.cells[row_label_index] if row.cells else step_value

            attrs = ""
            if has_step:
                attrs = (
                    f" data-step='{escape(str(step_value))}'"
                    f" data-path-x='{escape(str(x_value))}'"
                    f" data-energy='{escape(str(energy_value))}'"
                    f" data-label='{escape(str(row_label_value))}'"
                    f" data-table-id='{escape(table.id)}'"
                )

            parts.append(f"<tr{attrs}>")

            if has_step:
                parts.append(
                    "<td>"
                    f"<input type='checkbox' class='table-step-select'"
                    f" data-step='{escape(str(step_value))}'"
                    f" data-path-x='{escape(str(x_value))}'"
                    f" data-energy='{escape(str(energy_value))}'"
                    f" data-label='{escape(str(row_label_value))}'"
                    f" data-table-id='{escape(table.id)}'>"
                    "</td>"
                )

            for cell in row.cells:
                parts.append(f"<td>{escape(fmt(cell))}</td>")
            parts.append("</tr>")

        parts.append("</tbody></table></section>")

    if result.notes:
        parts.append("<section><h2>Notes</h2>")
        for note in result.notes:
            parts.append(f"<p>{escape(note)}</p>")
        parts.append("</section>")

    return rendered_markdowns + rendered_sections + "\n".join(parts)


ACADEMIC_STYLE = """
<style>
:root {
  --paper: #fbfaf7;
  --ink: #1f2933;
  --muted: #606f7b;
  --rule: #d7d0c4;
  --soft-rule: #ebe4d8;
  --accent: #284b63;
  --accent-soft: #eef4f7;
  --mono-bg: #f4f1eb;
  --shadow: 0 1px 2px rgba(31, 41, 51, 0.08);
}

html {
  background: #ece7dd;
  color: var(--ink);
  font-size: 16px;
}

body {
  max-width: 980px;
  margin: 0 auto;
  padding: 3rem 2.25rem 5rem;
  background: var(--paper);
  font-family: "Latin Modern Roman", "Computer Modern Serif", "CMU Serif", Georgia, "Times New Roman", serif;
  line-height: 1.62;
  box-shadow: 0 0 0 1px rgba(80, 70, 55, 0.10), 0 10px 30px rgba(80, 70, 55, 0.12);
  min-height: 100vh;
}

.diagnostic-paper h1, .diagnostic-paper h2, .diagnostic-paper h3, .diagnostic-paper h4 {
  color: #111827;
  font-weight: 600;
  line-height: 1.2;
  letter-spacing: -0.015em;
}

.diagnostic-paper h1 {
  margin: 0 0 1.2rem;
  font-size: 2.25rem;
  text-align: center;
  border-bottom: 1px solid var(--rule);
  padding-bottom: 0.8rem;
}

.diagnostic-paper h2 {
  margin-top: 2.4rem;
  font-size: 1.45rem;
  border-bottom: 1px solid var(--soft-rule);
  padding-bottom: 0.25rem;
}

.diagnostic-paper h3 {
  margin-top: 1.8rem;
  font-size: 1.15rem;
}

.diagnostic-paper p {
  margin: 0.75rem 0 1rem;
}

.diagnostic-paper a {
  color: var(--accent);
  text-decoration-thickness: 0.08em;
  text-underline-offset: 0.14em;
}

.diagnostic-paper nav {
  margin: -1rem 0 2rem;
  padding: 0.7rem 0;
  border-bottom: 1px solid var(--rule);
  color: var(--muted);
  font-size: 0.92rem;
}

.diagnostic-paper code, .diagnostic-paper pre, .diagnostic-paper kbd {
  font-family: "Latin Modern Mono", "Computer Modern Typewriter", "CMU Typewriter Text", ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
}

.diagnostic-paper code {
  background: var(--mono-bg);
  border: 1px solid var(--soft-rule);
  border-radius: 3px;
  padding: 0.08rem 0.25rem;
  font-size: 0.92em;
}

.diagnostic-paper pre {
  overflow-x: auto;
  background: var(--mono-bg);
  border: 1px solid var(--rule);
  border-radius: 4px;
  padding: 0.85rem 1rem;
}

pre .diagnostic-paper code {
  border: 0;
  background: transparent;
  padding: 0;
}

.diagnostic-paper table {
  width: 100%;
  border-collapse: collapse;
  margin: 1rem 0 1.6rem;
  font-size: 0.92rem;
  line-height: 1.35;
}

.diagnostic-paper thead {
  border-top: 1.5px solid #2f3640;
  border-bottom: 1px solid #2f3640;
}

.diagnostic-paper tbody {
  border-bottom: 1.5px solid #2f3640;
}

.diagnostic-paper th, .diagnostic-paper td {
  padding: 0.42rem 0.55rem;
  text-align: right;
  vertical-align: top;
  border: 0;
}

.diagnostic-paper th:first-child, .diagnostic-paper td:first-child {
  text-align: left;
}

.diagnostic-paper tbody tr:nth-child(even) {
  background: rgba(40, 75, 99, 0.035);
}

.diagnostic-paper small {
  color: var(--muted);
}

.diagnostic-paper ul, .diagnostic-paper ol {
  padding-left: 1.5rem;
}

.diagnostic-paper .card {
  display: inline-block;
  vertical-align: top;
  min-width: 11rem;
  max-width: 18rem;
  margin: 0.35rem 0.45rem 0.55rem 0;
  padding: 0.65rem 0.8rem;
  border: 1px solid var(--rule);
  border-radius: 3px;
  background: #fffdf8;
  box-shadow: var(--shadow);
}

.diagnostic-paper .card strong {
  display: block;
  font-variant: small-caps;
  letter-spacing: 0.035em;
}

.diagnostic-paper .card div {
  font-size: 1.05rem;
  margin-top: 0.15rem;
}

.card .diagnostic-paper small {
  display: block;
  margin-top: 0.2rem;
}

.diagnostic-paper .markdown-block {
  margin: 1.2rem 0 1.4rem;
  padding: 0.95rem 1.15rem;
  border-left: 3px solid var(--accent);
  background: var(--accent-soft);
}

.markdown-block .diagnostic-paper h2 {
  margin-top: 0;
  border-bottom: 0;
  font-size: 1.18rem;
}

.diagnostic-paper details.diagnostic-section {
  margin: 1.35rem 0;
  border: 1px solid var(--rule);
  border-radius: 4px;
  background: #fffdf8;
  box-shadow: var(--shadow);
}

.diagnostic-paper details.diagnostic-section > summary {
  cursor: pointer;
  padding: 0.75rem 0.95rem;
  font-weight: 600;
  color: #111827;
  background: linear-gradient(#fffdf8, #f7f1e8);
  border-bottom: 1px solid transparent;
}

.diagnostic-paper details.diagnostic-section[open] > summary {
  border-bottom-color: var(--rule);
}

.diagnostic-paper details.diagnostic-section > *:not(summary) {
  margin-left: 1rem;
  margin-right: 1rem;
}

details.diagnostic-section .diagnostic-paper details.diagnostic-section {
  margin-left: 0.5rem;
  margin-right: 0.5rem;
  background: #fbfaf7;
}

@media (max-width: 760px) {
  body {
    padding: 1.4rem 1rem 3rem;
    box-shadow: none;
  }

  .diagnostic-paper h1 {
    font-size: 1.8rem;
  }

  .diagnostic-paper table {
    display: block;
    overflow-x: auto;
    white-space: nowrap;
  }

  .diagnostic-paper .card {
    display: block;
    max-width: none;
  }
}


/* Interactive graph components own their own geometry and colour rules.
   Keep the paper theme from leaking into their layout calculations. */
.diagnostic-paper dft-line-graph,
.diagnostic-paper dft-kspace-graph,
.diagnostic-paper dft-band-graph {
  display: block;
  font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
  line-height: normal;
  color: #111827;
  background: transparent;
  box-sizing: content-box;
  contain: layout style;
}


/* Graph SVG layout only: do not style internal geometry. */
.diagnostic-paper svg.graph-svg-component {
  max-width: 100%;
  height: auto;
}

.diagnostic-paper svg.kspace-svg {
  width: min(720px, 100%);
  height: auto;
  aspect-ratio: 1 / 1;
}

</style>
"""


def render_page(title: str, body: str) -> str:
    return (
        "<!doctype html>\n"
        "<html>\n"
        "<head>\n"
        "  <meta charset='utf-8'>\n"
        f"  <title>{escape(title)}</title>\n"
        f"{ACADEMIC_STYLE}\n"
        "  <script type='module' src='/static/dft-local-components.js'></script>\n"
        "</head>\n"
        "<body>\n"
        "<main class='diagnostic-paper'>\n"
        f"{body}\n"
        "</main>\n"
        "</body>\n"
        "</html>\n"
    )

    return f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>{escape(title)}</title>
  <script type='module' src='/static/dft-local-components.js'></script>
</head>
<body>
{body}
</body>
</html>"""
