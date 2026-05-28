"""Formatting helpers for diagnostics."""

from __future__ import annotations

import json
from markupsafe import Markup


def fmt(value) -> str:
    """Format common diagnostic scalar values for display."""

    if value is None:
        return "—"

    if isinstance(value, bool):
        return "true" if value else "false"

    if isinstance(value, int):
        return str(value)

    if isinstance(value, float):
        if value == 0:
            return "0"
        if abs(value) < 1e-4 or abs(value) >= 1e5:
            return f"{value:.6e}"
        return f"{value:.6g}"

    return str(value)


def safe_json_for_script(value) -> Markup:
    """JSON encode data for embedding inside a script tag."""

    text = json.dumps(value)
    text = (
        text.replace("&", "\\u0026")
        .replace("<", "\\u003c")
        .replace(">", "\\u003e")
        .replace("\u2028", "\\u2028")
        .replace("\u2029", "\\u2029")
    )
    return Markup(text)
