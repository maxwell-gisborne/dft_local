"""Small diagnostic server for the dft_local package.

This is independent of the old `dft_local.diagnostics_pannel` package.  It
uses explicit diagnostic discovery and the local structured model copy.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from urllib.parse import parse_qs
from wsgiref.simple_server import make_server
import os

from dft_local.diagnostics.discovery import load_diagnostics
from dft_local.diagnostics.models import InputParseError, parse_inputs
from dft_local.diagnostics.render import render_page, render_result


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

        try:
            if path == "/":
                body = self.index()
                status = "200 OK"
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
        start_response(status, [("Content-Type", "text/html; charset=utf-8"), ("Content-Length", str(len(data)))])
        return [data]

    def index(self) -> str:
        links = []
        for spec in sorted(self.specs.values(), key=lambda s: (s.group, s.title)):
            links.append(f"<li><a href='/d/{spec.id}'>{spec.group} · {spec.title}</a><br><small>{spec.description}</small></li>")

        body = "<h1>dft_local diagnostics</h1><ul>" + "\n".join(links) + "</ul>"
        return render_page("dft_local diagnostics", body)

    def diagnostic_page(self, diagnostic_id: str, raw_inputs: dict[str, str]) -> str:
        spec = self.specs[diagnostic_id]

        try:
            inputs = parse_inputs(spec, raw_inputs)
        except InputParseError as exc:
            return render_page("Input error", f"<h1>Input error</h1><p>{exc}</p>")

        form = self.form(spec, inputs)
        result = spec.compute(self.ctx, inputs)
        body = f"<nav><a href='/'>index</a></nav>{form}{render_result(result)}"
        return render_page(result.title, body)

    @staticmethod
    def form(spec, inputs: dict[str, Any]) -> str:
        if not spec.inputs:
            return ""

        fields = []
        for inp in spec.inputs:
            value = inputs.get(inp.name, inp.default)
            if inp.kind == "bool":
                checked = " checked" if bool(value) else ""
                fields.append(
                    f"<label><input type='checkbox' name='{inp.name}' value='1'{checked}> {inp.label}</label>"
                )
            else:
                fields.append(
                    f"<label>{inp.label}<br><input name='{inp.name}' value='{value}'></label>"
                )
            if inp.help:
                fields.append(f"<small>{inp.help}</small>")

        return "<form method='get'><p>" + "</p><p>".join(fields) + "</p><button>Run</button></form>"


class DiagnosticASGI:
    """Tiny ASGI wrapper around DiagnosticApp.

    This lets uvicorn provide reload/watchfiles while keeping the diagnostic
    app itself dependency-light and easy to test.
    """

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
