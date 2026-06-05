"""Small guarded Typst rendering helpers for diagnostics HTML."""

from __future__ import annotations

import hashlib
import html
import re
import subprocess
import tempfile
from pathlib import Path


class TypstRenderError(RuntimeError):
    """Raised when a Typst snippet cannot be rendered."""


_CACHE_DIR = Path(".diagnostics-cache") / "typst-svg"
_MAX_SOURCE_CHARS = 4096
_TIMEOUT_SECONDS = 10.0

DIAGNOSTIC_TYPST_PRELUDE = """
#import "@preview/physica:0.9.5": *
"""


def _wrap_math(source: str, *, display: bool) -> str:
    # Keep this deliberately tiny.  Full Typst documents can come later.
    body = source.strip()
    if not body.startswith("$"):
        body = f"$ {body} $"

    if display:
        body = f"#align(center)[{body}]"

    prelude = DIAGNOSTIC_TYPST_PRELUDE.strip()
    return (
        "#set page(width: auto, height: auto, margin: 0pt, fill: none)\n"
        "#set text(size: 11pt)\n"
        f"{prelude}\n"
        f"{body}\n"
    )


def _sanitise_svg(svg: str) -> str:
    # Typst-generated SVG is trusted enough for local development, but strip the
    # most obvious active content before embedding.
    svg = re.sub(r"<script\b.*?</script>", "", svg, flags=re.IGNORECASE | re.DOTALL)
    svg = re.sub(r"\son[a-zA-Z]+\s*=\s*(['\"]).*?\1", "", svg)
    return svg


def render_typst_math_to_svg(source: str, *, display: bool = False) -> str:
    """Compile a small Typst math snippet to inline SVG."""

    if len(source) > _MAX_SOURCE_CHARS:
        raise TypstRenderError(
            f"Typst snippet too long: {len(source)} > {_MAX_SOURCE_CHARS} characters"
        )

    key_material = f"typst-0.14|display={display}|{source}".encode("utf-8")
    key = hashlib.sha256(key_material).hexdigest()
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cached = _CACHE_DIR / f"{key}.svg"
    if cached.exists():
        return cached.read_text(encoding="utf-8")

    with tempfile.TemporaryDirectory(prefix="dft-local-typst-") as tmp:
        tmp_path = Path(tmp)
        input_path = tmp_path / "snippet.typ"
        output_path = tmp_path / "snippet.svg"
        input_path.write_text(_wrap_math(source, display=display), encoding="utf-8")

        try:
            proc = subprocess.run(
                ["typst", "compile", str(input_path), str(output_path)],
                cwd=tmp_path,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=_TIMEOUT_SECONDS,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise TypstRenderError("Typst compile timed out") from exc
        except FileNotFoundError as exc:
            raise TypstRenderError("typst executable not found") from exc

        if proc.returncode != 0:
            message = (proc.stderr or proc.stdout or "Typst compile failed").strip()
            raise TypstRenderError(message)

        svg = _sanitise_svg(output_path.read_text(encoding="utf-8"))
        cached.write_text(svg, encoding="utf-8")
        return svg


def render_typst_error(source: str, error: Exception) -> str:
    return (
        "<span class='typst-math typst-error' title='"
        + html.escape(str(error), quote=True)
        + "'>"
        + html.escape(source)
        + "</span>"
    )
