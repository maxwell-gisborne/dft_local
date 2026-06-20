"""Command-line helper for the dft_local package."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any


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


def _parse_input_assignments(items: list[str]) -> dict[str, str]:
    inputs: dict[str, str] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"invalid input {item!r}; expected name=value")
        key, value = item.split("=", 1)
        inputs[key] = value
    return inputs


def _diagnostic_app(root: str, *, include_context: bool):
    from dft_local.diagnostics.server import DiagnosticApp, load_default_context

    ctx = load_default_context(root) if include_context else None
    return DiagnosticApp(ctx=ctx)


def _compute_diagnostic(app: Any, diagnostic_id: str, raw_inputs: dict[str, str]):
    from dft_local.diagnostics.models import parse_inputs

    if diagnostic_id not in app.specs:
        raise KeyError(diagnostic_id)

    spec = app.specs[diagnostic_id]
    parsed_inputs = parse_inputs(spec, raw_inputs)
    result = spec.compute(app.ctx, parsed_inputs)
    return spec, result


def _export_bundle_from_args(
    *,
    diagnostic_id: str,
    out_dir: str | Path,
    root: str,
    inputs: dict[str, str],
    include_context: bool,
    lib_mode: str = "symlink",
    write_root_document: bool = True,
    layout: dict[str, Any] | None = None,
) -> Path:
    from dft_local.diagnostics.typst_bundle import export_typst_bundle

    app = _diagnostic_app(root, include_context=include_context)
    spec, result = _compute_diagnostic(app, diagnostic_id, inputs)

    return export_typst_bundle(
        result,
        out_dir,
        report_id=diagnostic_id.replace(".", "_"),
        title=spec.title,
        diagnostic_id=diagnostic_id,
        data_root=root,
        inputs=inputs,
        include_context=include_context,
        layout=layout,
        lib_mode=lib_mode,  # type: ignore[arg-type]
        write_root_document=write_root_document,
    )


def export_typst(args: argparse.Namespace) -> int:
    """Export a diagnostic as a static Typst bundle."""

    try:
        inputs = _parse_input_assignments(args.input)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    if args.out is not None:
        try:
            out = _export_bundle_from_args(
                diagnostic_id=args.diagnostic_id,
                out_dir=args.out,
                root=args.root,
                inputs=inputs,
                include_context=not args.no_context,
                write_root_document=True,
            )
        except KeyError:
            print(f"unknown diagnostic: {args.diagnostic_id}", file=sys.stderr)
            return 2
    else:
        app = _diagnostic_app(args.root, include_context=not args.no_context)
        html = app.diagnostic_typst_export_page(args.diagnostic_id, inputs)
        if "Unknown diagnostic" in html:
            print(f"unknown diagnostic: {args.diagnostic_id}", file=sys.stderr)
            return 2
        out = Path("diagnostic_bundles") / args.diagnostic_id.replace(".", "_")

    print(out)
    return 0


def bundle(args: argparse.Namespace) -> int:
    """Create or refresh editable Typst diagnostic bundles."""

    argv = list(args.bundle_args)
    if not argv:
        print("expected: bundle new <diagnostic-id> <directory> | bundle <directory> | bundle list", file=sys.stderr)
        return 2

    if argv[0] == "list":
        app = _diagnostic_app(args.root, include_context=not args.no_context)
        for diagnostic_id in sorted(app.specs):
            print(diagnostic_id)
        return 0

    if argv[0] == "new":
        if len(argv) != 3:
            print("expected: bundle new <diagnostic-id> <directory>", file=sys.stderr)
            return 2

        diagnostic_id = argv[1]
        out_dir = Path(argv[2])
        try:
            inputs = _parse_input_assignments(args.input)
        except ValueError as exc:
            print(str(exc), file=sys.stderr)
            return 2

        try:
            out = _export_bundle_from_args(
                diagnostic_id=diagnostic_id,
                out_dir=out_dir,
                root=args.root,
                inputs=inputs,
                include_context=not args.no_context,
                lib_mode=args.lib_mode,
                write_root_document=True,
            )
        except KeyError:
            print(f"unknown diagnostic: {diagnostic_id}", file=sys.stderr)
            return 2

        print("Created Typst diagnostic bundle")
        print(f"Diagnostic: {diagnostic_id}")
        print(f"Bundle:     {out}")
        print(f"Manifest:   {out / 'manifest.json'}")
        print(f"Document:   {out / 'diagnostics.typ'}")
        print(f"Generated:  {out / 'generated'}")
        print("Compile:    typst compile diagnostics.typ diagnostics.pdf")
        return 0

    if len(argv) != 1:
        print("expected: bundle <directory>", file=sys.stderr)
        return 2

    bundle_dir = Path(argv[0])
    manifest_path = bundle_dir / "manifest.json"
    if not manifest_path.exists():
        print(f"bundle manifest not found: {manifest_path}", file=sys.stderr)
        return 2

    manifest = json.loads(manifest_path.read_text())
    diagnostic_id = str(manifest.get("diagnostic_id") or manifest.get("report_id") or "")
    if not diagnostic_id:
        print(f"bundle manifest has no diagnostic_id: {manifest_path}", file=sys.stderr)
        return 2

    raw_inputs = manifest.get("inputs") or {}
    inputs = {str(k): str(v) for k, v in raw_inputs.items()}
    root = args.root if args.root_overridden else str(manifest.get("data_root") or args.root)
    include_context = bool(manifest.get("include_context", not args.no_context))
    lib_mode = str((manifest.get("export") or {}).get("lib_mode") or args.lib_mode)
    layout = manifest.get("layout") if isinstance(manifest.get("layout"), dict) else None

    try:
        out = _export_bundle_from_args(
            diagnostic_id=diagnostic_id,
            out_dir=bundle_dir,
            root=root,
            inputs=inputs,
            include_context=include_context,
            lib_mode=lib_mode,
            write_root_document=bool(args.refresh_root),
            layout=layout,
        )
    except KeyError:
        print(f"unknown diagnostic: {diagnostic_id}", file=sys.stderr)
        return 2

    print("Refreshed Typst diagnostic bundle")
    print(f"Diagnostic: {diagnostic_id}")
    print(f"Bundle:     {out}")
    print(f"Rewritten:  {out / 'generated'}")
    print(f"Manifest:   {out / 'manifest.json'}")
    if args.refresh_root:
        print(f"Document:   {out / 'diagnostics.typ'}")
    else:
        print("Document:   preserved diagnostics.typ")
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

    bundle_parser = sub.add_parser("bundle", help="create or refresh editable Typst diagnostic bundles")
    bundle_parser.add_argument("bundle_args", nargs="*")
    bundle_parser.add_argument("--root", default="test_run/run_dir/data")
    bundle_parser.add_argument("--input", action="append", default=[], metavar="NAME=VALUE")
    bundle_parser.add_argument("--no-context", action="store_true")
    bundle_parser.add_argument("--lib-mode", choices=("symlink", "vendor", "none"), default="symlink")
    bundle_parser.add_argument("--refresh-root", action="store_true")
    bundle_parser.set_defaults(func=bundle, root_overridden=False)

    test_parser = sub.add_parser("test", help="run discovered pytest targets")
    test_parser.add_argument("--timeout", type=float, default=120.0)
    test_parser.set_defaults(func=test)

    args = parser.parse_args(argv)
    if args.command == "bundle":
        # argparse cannot tell whether --root came from the user after parsing.
        # This marker is reserved for a later status/freshness command.
        args.root_overridden = "--root" in (argv or sys.argv[1:])
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
