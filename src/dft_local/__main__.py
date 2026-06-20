"""Command-line helper for the dft_local package."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys


def serve(args: argparse.Namespace) -> int:
    """Run dft_local diagnostic server through uvicorn."""

    env = os.environ.copy()
    env.setdefault("PYTHONPATH", "src")
    env["DFT_LOCAL_DATA_ROOT"] = args.root

    command = [
        sys.executable,
        "-m",
        "uvicorn",
        "dft_local.diagnostics.server:app",
        "--host",
        args.host,
        "--port",
        str(args.port),
    ]

    if args.reload:
        command.extend(
            [
                "--reload",
                "--reload-dir",
                "src/dft_local",
                "--reload-dir",
                "src/dft_local",
            ]
        )

    return subprocess.call(command, env=env)


def export_typst(args: argparse.Namespace) -> int:
    """Export a diagnostic as a static Typst bundle."""

    from dft_local.diagnostics.server import DiagnosticApp, load_default_context

    ctx = None if args.no_context else load_default_context(args.root)
    app = DiagnosticApp(ctx=ctx)

    inputs = {}
    for item in args.input:
        if "=" not in item:
            print(f"invalid input {item!r}; expected name=value", file=sys.stderr)
            return 2
        key, value = item.split("=", 1)
        inputs[key] = value

    if args.out is not None:
        from dft_local.diagnostics.models import parse_inputs
        from dft_local.diagnostics.typst_bundle import export_typst_bundle

        spec = app.specs[args.diagnostic_id]
        parsed_inputs = parse_inputs(spec, inputs)
        result = spec.compute(app.ctx, parsed_inputs)
        out = export_typst_bundle(
            result,
            args.out,
            report_id=args.diagnostic_id.replace(".", "_"),
            title=spec.title,
        )
    else:
        html = app.diagnostic_typst_export_page(args.diagnostic_id, inputs)
        if "Unknown diagnostic" in html:
            print(f"unknown diagnostic: {args.diagnostic_id}", file=sys.stderr)
            return 2
        out = os.path.join("diagnostic_bundles", args.diagnostic_id.replace(".", "_"))

    print(out)
    return 0


def test(args: argparse.Namespace) -> int:
    """Run discovered package tests."""

    from dft_local.testsuite.runner import run_discovered_pytest

    result = run_discovered_pytest(timeout=args.timeout)
    print(result.stdout, end="")
    print(result.stderr, end="", file=sys.stderr)

    return result.returncode


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m dft_local",
        description="Run diagnostics and discovered tests.",
    )

    sub = parser.add_subparsers(dest="command", required=True)

    serve_parser = sub.add_parser("serve", help="run the uvicorn diagnostics server")
    serve_parser.add_argument("--host", default="127.0.0.1")
    serve_parser.add_argument("--port", type=int, default=8765)
    serve_parser.add_argument("--root", default="test_run/run_dir/data")
    serve_parser.add_argument("--reload", action="store_true")
    serve_parser.set_defaults(func=serve)

    export_parser = sub.add_parser("export-typst", help="export a diagnostic as a static Typst bundle")
    export_parser.add_argument("diagnostic_id")
    export_parser.add_argument("--root", default="test_run/run_dir/data")
    export_parser.add_argument("--input", action="append", default=[], metavar="NAME=VALUE")
    export_parser.add_argument("--out", default=None)
    export_parser.add_argument("--no-context", action="store_true")
    export_parser.set_defaults(func=export_typst)

    test_parser = sub.add_parser("test", help="run discovered pytest targets")
    test_parser.add_argument("--timeout", type=float, default=120.0)
    test_parser.set_defaults(func=test)

    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
