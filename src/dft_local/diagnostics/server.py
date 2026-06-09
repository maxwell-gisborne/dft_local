"""Small diagnostic server for the dft_local package.

This is independent of the old `dft_local.diagnostics_pannel` package.  It
uses explicit diagnostic discovery and the local structured model copy.
"""

from __future__ import annotations

from html import escape
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, quote
from wsgiref.simple_server import make_server
import os

from dft_local.diagnostics.discovery import load_diagnostics
from dft_local.diagnostics.models import InputParseError, parse_inputs
from dft_local.diagnostics.render import render_page, render_result

STATIC_ROOT = Path(__file__).resolve().parent / "static"
DOCS_ROOT = Path(__file__).resolve().parents[1]


def load_default_context(root: str | Path = "test_run/run_dir/data") -> Any:
    """Compatibility wrapper returning dft_local-local diagnostic context."""

    from dft_local.diagnostics.context import (
        DiagnosticContext,
        DiagnosticsState,
    )

    return DiagnosticContext(DiagnosticsState.from_root(root))


class DiagnosticApp:
    """Tiny WSGI app for diagnostics."""

    def __init__(self, *, ctx: Any = None) -> None:
        self.ctx = ctx
        self.specs = {spec.id: spec for spec in load_diagnostics()}

    def __call__(self, environ, start_response):
        path = environ.get("PATH_INFO", "/")
        query = parse_qs(environ.get("QUERY_STRING", ""), keep_blank_values=True)
        raw_inputs = {key: values[-1] for key, values in query.items()}

        content_type = "text/html; charset=utf-8"

        try:
            if path.startswith("/static/"):
                rel = path.removeprefix("/static/")
                target = (STATIC_ROOT / rel).resolve()
                static_root = STATIC_ROOT.resolve()

                if not str(target).startswith(str(static_root)) or not target.is_file():
                    body = "not found"
                    status = "404 Not Found"
                    content_type = "text/plain; charset=utf-8"
                else:
                    body = target.read_text()
                    status = "200 OK"
                    if target.suffix == ".js":
                        content_type = "text/javascript; charset=utf-8"
                    elif target.suffix == ".css":
                        content_type = "text/css; charset=utf-8"
                    else:
                        content_type = "text/plain; charset=utf-8"

            elif path == "/":
                body = self.index()
                status = "200 OK"
            elif path == "/docs" or path.startswith("/docs/"):
                doc_id = path.removeprefix("/docs").strip("/")
                body = self.docs_page(doc_id)
                status = "200 OK"
            elif path.startswith("/d-run/"):
                diagnostic_id = path.removeprefix("/d-run/")
                body = self.diagnostic_run_stream(diagnostic_id, raw_inputs)
                status = "200 OK"
                content_type = "text/event-stream; charset=utf-8"
            elif path.startswith("/d/"):
                diagnostic_id = path.removeprefix("/d/")
                body = self.diagnostic_page(diagnostic_id, raw_inputs)
                status = "200 OK"
            else:
                body = render_page("Not found", "<h1>Not found</h1>")
                status = "404 Not Found"
        except Exception as exc:  # noqa: BLE001 - diagnostic server should show failures
            body = render_page("Error", f"<h1>Error</h1><pre>{type(exc).__name__}: {exc}</pre>")
            status = "500 Internal Server Error"

        data = body.encode("utf-8")
        start_response(status, [("Content-Type", content_type), ("Content-Length", str(len(data)))])
        return [data]

    def index(self) -> str:
        def render_tree(tree: dict[str, object]) -> str:
            parts: list[str] = ["<ul>"]

            for key in sorted(tree):
                value = tree[key]

                if isinstance(value, dict):
                    parts.append(f"<li><strong>{key}</strong>")
                    parts.append(render_tree(value))
                    parts.append("</li>")
                else:
                    spec = value
                    parts.append(
                        f"<li><a href='/d/{spec.id}'>{spec.title}</a>"
                        f"<br><small><code>{spec.id}</code> · {spec.description}</small></li>"
                    )

            parts.append("</ul>")
            return "\n".join(parts)

        tree: dict[str, object] = {}

        for spec in sorted(self.specs.values(), key=lambda s: s.id):
            parts = spec.id.split(".")
            cursor = tree

            for part in parts[:-1]:
                child = cursor.setdefault(part, {})
                if not isinstance(child, dict):
                    raise TypeError(f"Diagnostic namespace collision at {spec.id}")
                cursor = child

            cursor[parts[-1]] = spec

        body = "<h1>dft_local diagnostics</h1><nav><a href='/docs/'>docs</a></nav>" + render_tree(tree)
        return render_page("dft_local diagnostics", body)

    def docs_page(self, doc_id: str) -> str:
        docs = self.discover_docs()

        if not doc_id:
            return render_page("dft_local docs", self.docs_index(docs))

        if doc_id not in docs:
            return render_page(
                "Document not found",
                "<nav><a href='/docs/'>docs</a></nav>"
                f"<h1>Document not found</h1><p><code>{escape(doc_id)}</code></p>",
            )

        path = docs[doc_id]
        matching_specs = [
            spec for spec in sorted(self.specs.values(), key=lambda item: item.id)
            if spec.id == doc_id or spec.id.startswith(doc_id + ".")
        ]
        diagnostic_links = ""
        if matching_specs:
            diagnostic_links = (
                "<section><h2>Diagnostics</h2><ul>"
                + "\n".join(
                    f"<li><a href='/d/{spec.id}'>{escape(spec.title)}</a>"
                    f"<br><small><code>{escape(spec.id)}</code></small></li>"
                    for spec in matching_specs
                )
                + "</ul></section>"
            )

        body = (
            "<nav><a href='/'>diagnostics</a> · <a href='/docs/'>docs</a></nav>"
            f"<h1><code>{escape(doc_id)}</code></h1>"
            f"<p><small>{escape(str(path.relative_to(DOCS_ROOT)))}</small></p>"
            + diagnostic_links
            + self.render_markdown(path.read_text())
        )
        return render_page(f"docs · {doc_id}", body)

    @staticmethod
    def discover_docs() -> dict[str, Path]:
        docs: dict[str, Path] = {}

        for path in sorted(DOCS_ROOT.rglob("docs.md")):
            if "__pycache__" in path.parts:
                continue

            rel = path.parent.relative_to(DOCS_ROOT)
            doc_id = ".".join(rel.parts)

            if doc_id == ".":
                continue

            docs[doc_id] = path

        return docs

    @staticmethod
    def docs_index(docs: dict[str, Path]) -> str:
        def render_tree(tree: dict[str, object]) -> str:
            parts: list[str] = ["<ul>"]

            for key in sorted(tree):
                value = tree[key]

                if isinstance(value, dict):
                    parts.append(f"<li><strong>{escape(key)}</strong>")
                    parts.append(render_tree(value))
                    parts.append("</li>")
                else:
                    doc_id, path = value
                    parts.append(
                        f"<li><a href='/docs/{doc_id}'>{escape(key)}</a>"
                        f"<br><small><code>{escape(doc_id)}</code> · "
                        f"{escape(str(path.relative_to(DOCS_ROOT)))}</small></li>"
                    )

            parts.append("</ul>")
            return "\n".join(parts)

        tree: dict[str, object] = {}

        for doc_id, path in docs.items():
            parts = doc_id.split(".")
            cursor = tree

            for part in parts[:-1]:
                child = cursor.setdefault(part, {})
                if not isinstance(child, dict):
                    raise TypeError(f"Document namespace collision at {doc_id}")
                cursor = child

            cursor[parts[-1]] = (doc_id, path)

        return (
            "<h1>dft_local docs</h1>"
            "<p>Documentation mirrors the source/domain hierarchy. "
            "Each <code>docs.md</code> file is exposed at its package-like path.</p>"
            + render_tree(tree)
        )

    @staticmethod
    def render_markdown(markdown: str) -> str:
        markdown_exe = shutil.which("markdown")

        if markdown_exe is not None:
            with tempfile.NamedTemporaryFile("w", suffix=".md", encoding="utf-8") as handle:
                handle.write(markdown)
                handle.flush()

                result = subprocess.run(
                    [markdown_exe, "-G", "-html5", handle.name],
                    text=True,
                    capture_output=True,
                    check=True,
                )

            return result.stdout

        lines = markdown.splitlines()
        html: list[str] = []
        in_code = False
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

            if stripped.startswith("```"):
                flush_paragraph()
                close_list()

                if in_code:
                    html.append("</code></pre>")
                    in_code = False
                else:
                    html.append("<pre><code>")
                    in_code = True

                continue

            if in_code:
                html.append(escape(stripped))
                continue

            if not stripped:
                flush_paragraph()
                close_list()
                continue

            heading = re.match(r"^(#{1,6})\s+(.*)$", stripped)
            if heading:
                flush_paragraph()
                close_list()
                level = len(heading.group(1))
                title = escape(heading.group(2).strip())
                html.append(f"<h{level}>{title}</h{level}>")
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

        if in_code:
            html.append("</code></pre>")

        return "\n".join(html)


    def diagnostic_page(self, diagnostic_id: str, raw_inputs: dict[str, str]) -> str:
        spec = self.specs[diagnostic_id]
        inputs = parse_inputs(spec, raw_inputs)
        result = spec.compute(self.ctx, inputs)

        doc_id = ".".join(diagnostic_id.split(".")[:-1])
        doc_link = ""
        if doc_id in self.discover_docs():
            doc_link = f" · <a href='/docs/{escape(doc_id)}'>docs</a>"

        form = self.form(spec, inputs)
        body = (
            f"<nav><a href='/'>index</a>{doc_link}</nav>"
            f"{form}"
            f"{self.result_outlet(render_result(result))}"
        )
        return render_page(spec.title, body)

    def diagnostic_run_stream(self, diagnostic_id: str, raw_inputs: dict[str, str]) -> str:
        spec = self.specs[diagnostic_id]
        inputs = parse_inputs(spec, raw_inputs)
        result = spec.compute(self.ctx, inputs)
        html = render_result(result)
        return self.datastar_patch_result_blocks(html)

    @staticmethod
    def result_outlet(html: str) -> str:
        return (
            "<section id='diagnostic-result' data-dft-diagnostic-result>"
            f"{html}"
            "</section>"
        )

    @staticmethod
    def datastar_patch_element(selector: str, html: str, *, mode: str = "outer") -> str:
        lines = [
            "event: datastar-patch-elements",
            f"data: selector {selector}",
            f"data: mode {mode}",
        ]
        lines.extend(f"data: elements {line}" for line in html.splitlines())
        lines.append("")
        return "\n".join(lines) + "\n"

    @staticmethod
    def datastar_execute_script(script: str) -> str:
        lines = [
            "event: datastar-execute-script",
            f"data: script {script}",
            "",
        ]
        return "\n".join(lines) + "\n"

    @classmethod
    def datastar_patch_result_blocks(cls, html: str) -> str:
        """Return Datastar SSE patches for the rendered diagnostic result.

        The renderer has already marked every important block with
        data-dft-block-kind.  We patch by representation kind:

        - json-rendered: patch only model islands, then refresh components.
        - stateful-html: patch the HTML block, preserving table state.
        - static-html: patch the HTML block.
        """

        blocks = cls.extract_dft_blocks(html)
        if not blocks:
            return (
                cls.datastar_patch_element(
                    "#diagnostic-result",
                    cls.result_outlet(html),
                )
                + cls.datastar_execute_script("window.dftRefreshModels?.(document)")
            )

        stream = []
        needs_model_refresh = False
        needs_table_restore = False

        has_stateful_html = any(block["kind"] == "stateful-html" for block in blocks)
        if has_stateful_html:
            stream.append(
                cls.datastar_execute_script(
                    "window.__dftTableState = window.captureDftTableState?.(document)"
                )
            )

        for block in blocks:
            block_id = block["id"]
            kind = block["kind"]
            block_html = block["html"]

            if kind == "json-rendered":
                for model_id, model_html in cls.extract_dft_models(block_html):
                    stream.append(cls.datastar_patch_element(f"#{model_id}", model_html))
                    needs_model_refresh = True
            elif kind == "stateful-html":
                needs_table_restore = True
                stream.append(cls.datastar_patch_element(f"#dft-block-{block_id}", block_html))
            else:
                stream.append(cls.datastar_patch_element(f"#dft-block-{block_id}", block_html))

        if needs_table_restore:
            stream.append(
                cls.datastar_execute_script(
                    "window.restoreDftTableState?.(window.__dftTableState, document)"
                )
            )

        if needs_model_refresh:
            stream.append(cls.datastar_execute_script("window.dftRefreshModels?.(document)"))

        return "".join(stream)

    @staticmethod
    def extract_dft_blocks(html: str) -> list[dict[str, str]]:
        import re

        pattern = re.compile(
            r"(<section id='dft-block-([^']+)'[^>]*data-dft-block-kind='([^']+)'[^>]*>.*?</section>)",
            re.S,
        )

        return [
            {"html": match.group(1), "id": match.group(2), "kind": match.group(3)}
            for match in pattern.finditer(html)
        ]

    @staticmethod
    def extract_dft_models(html: str) -> list[tuple[str, str]]:
        import re

        pattern = re.compile(
            r"(<script type='application/json' id='([^']+)'[^>]*>.*?</script>)",
            re.S,
        )

        return [(match.group(2), match.group(1)) for match in pattern.finditer(html)]

    @staticmethod
    def form(spec: DiagnosticSpec, inputs: dict[str, object]) -> str:
        fields: list[str] = []

        for input_spec in spec.inputs:
            value = inputs[input_spec.name]
            label = escape(input_spec.label)
            name = escape(input_spec.name)

            if input_spec.kind == "select":
                options = []
                for raw_value, raw_label in input_spec.options:
                    selected = " selected" if raw_value == value else ""
                    options.append(
                        f"<option value='{escape(str(raw_value))}'{selected}>"
                        f"{escape(str(raw_label))}"
                        "</option>"
                    )

                control = (
                    f"<label>{label}<br>"
                    f"<select name='{name}'>"
                    + "".join(options)
                    + "</select></label>"
                )
            else:
                control = (
                    f"<label>{label}<br>"
                    f"<input name='{name}' value='{escape(str(value))}'></label>"
                )

            if input_spec.help:
                control += f"<br><small>{escape(input_spec.help)}</small>"

            fields.append(control)

        action = f"/d-run/{quote(spec.id)}"
        return (
            f"<form method='get' action='/d/{escape(spec.id)}' "
            f"data-on:submit=\"@get('{action}', {{contentType: 'form'}})\">"
            "<p>"
            + "</p><p>".join(fields)
            + "</p><button type='submit'>Run</button></form>"
        )


class DiagnosticASGI:
    def __init__(self, *, ctx: Any = None) -> None:
        self.wsgi = DiagnosticApp(ctx=ctx)

    async def __call__(self, scope, receive, send) -> None:
        if scope["type"] != "http":
            await send({
                "type": "http.response.start",
                "status": 404,
                "headers": [(b"content-type", b"text/plain; charset=utf-8")],
            })
            await send({
                "type": "http.response.body",
                "body": b"not found",
            })
            return

        path = scope.get("path", "/")
        query_string = scope.get("query_string", b"").decode("utf-8")

        environ = {
            "PATH_INFO": path,
            "QUERY_STRING": query_string,
        }

        captured: dict[str, object] = {}

        def start_response(status: str, headers: list[tuple[str, str]]) -> None:
            captured["status"] = status
            captured["headers"] = headers

        chunks = self.wsgi(environ, start_response)
        body = b"".join(chunks)

        status_text = str(captured.get("status", "500 Internal Server Error"))
        status_code = int(status_text.split()[0])

        headers = [
            (name.lower().encode("latin1"), value.encode("latin1"))
            for name, value in captured.get("headers", [])
            if name.lower() != "content-length"
        ]
        headers.append((b"content-length", str(len(body)).encode("ascii")))

        await send({
            "type": "http.response.start",
            "status": status_code,
            "headers": headers,
        })
        await send({
            "type": "http.response.body",
            "body": body,
        })


def default_data_root() -> str:
    """Return data root from environment or the development default."""

    return os.environ.get("DFT_LOCAL_DATA_ROOT", "test_run/run_dir/data")


def create_app(root: str | Path | None = None) -> DiagnosticASGI:
    """Create an ASGI app for uvicorn."""

    if root is None:
        root = default_data_root()

    return DiagnosticASGI(ctx=load_default_context(root))

_app: DiagnosticASGI | None = None


def get_app() -> DiagnosticASGI:
    """Return the default diagnostics app, creating it lazily."""
    global _app

    if _app is None:
        _app = create_app()

    return _app


def __getattr__(name: str):
    """Keep `dft_local.diagnostics.server.app` compatibility without import-time data loading."""
    if name == "app":
        return get_app()

    raise AttributeError(name)


def run(
    host: str = "127.0.0.1",
    port: int = 8765,
    root: str | Path = "test_run/run_dir/data",
) -> None:
    """Run dft_local diagnostic server with uvicorn."""

    import uvicorn

    uvicorn.run(
        create_app(root),
        host=host,
        port=port,
    )


if __name__ == "__main__":
    run()
