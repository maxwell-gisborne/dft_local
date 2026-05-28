"""Minimal HTML renderer for diagnostics.

This renderer is intentionally small.  It renders the structured diagnostic
model used by the dft_local package and does not depend on the old diagnostics
panel package.
"""

from __future__ import annotations

from html import escape
from typing import Any

from dft_local.diagnostics.models import DiagnosticResult


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
        parts.append(f"<section><h2>{escape(graph.title)}</h2>")
        parts.append(f"<p>{escape(graph.description)}</p>")
        parts.append(
            f"<p><small>Graph payload: {len(graph.series)} series, "
            f"x={escape(graph.x_label)}, y={escape(graph.y_label)}</small></p>"
        )
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
    body {{ font-family: system-ui, sans-serif; margin: 2rem; line-height: 1.45; }}
    table {{ border-collapse: collapse; width: 100%; margin: 1rem 0 2rem; }}
    th, td {{ border: 1px solid #ddd; padding: 0.35rem 0.5rem; text-align: left; }}
    th {{ background: #f5f5f5; }}
    .cards {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(12rem, 1fr)); gap: 1rem; }}
    .card {{ border: 1px solid #ddd; border-radius: 0.5rem; padding: 1rem; }}
    .ok {{ color: #166534; }}
    .warn {{ color: #92400e; }}
    .bad {{ color: #991b1b; }}
    .neutral {{ color: #374151; }}
    nav a {{ margin-right: 1rem; }}
  </style>
</head>
<body>
{body}
</body>
</html>"""
