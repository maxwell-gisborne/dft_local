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

    test_parser = sub.add_parser("test", help="run discovered pytest targets")
    test_parser.add_argument("--timeout", type=float, default=120.0)
    test_parser.set_defaults(func=test)

    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
