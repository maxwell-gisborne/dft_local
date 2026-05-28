from __future__ import annotations

import asyncio

from dft_local.diagnostics.server import DiagnosticASGI


def run_asgi_request(app, path: str = "/", query_string: bytes = b""):
    messages = []

    async def receive():
        return {"type": "http.request", "body": b"", "more_body": False}

    async def send(message):
        messages.append(message)

    scope = {
        "type": "http",
        "method": "GET",
        "path": path,
        "query_string": query_string,
        "headers": [],
    }

    asyncio.run(app(scope, receive, send))

    return messages


def test_asgi_index_response() -> None:
    app = DiagnosticASGI(ctx=None)
    messages = run_asgi_request(app, "/")

    assert messages[0]["type"] == "http.response.start"
    assert messages[0]["status"] == 200
    assert messages[1]["type"] == "http.response.body"
    assert b"dft_local diagnostics" in messages[1]["body"]


def test_asgi_static_overview_response() -> None:
    app = DiagnosticASGI(ctx=None)
    messages = run_asgi_request(app, "/d/transport.boltzmann.overview")

    assert messages[0]["status"] == 200
    assert b"Boltzmann conductivity domain" in messages[1]["body"]


def test_default_data_root_uses_environment(monkeypatch) -> None:
    from dft_local.diagnostics.server import default_data_root

    monkeypatch.setenv("DFT_LOCAL_DATA_ROOT", "some/other/root")

    assert default_data_root() == "some/other/root"
