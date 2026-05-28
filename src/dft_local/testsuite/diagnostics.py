"""Diagnostics for the test suite.

This diagnostic makes tests visible inside the diagnostics server.  It can list
discovered pytest targets and optionally run them in a subprocess.
"""

from __future__ import annotations

from dft_local.diagnostics.models import (
    Card,
    DiagnosticResult,
    DiagnosticSpec,
    InputSpec,
    Table,
    TableRow,
)

from dft_local.testsuite.discovery import load_pytest_files
from dft_local.testsuite.runner import run_pytest_targets


def _target_table(targets: tuple[str, ...]) -> Table:
    return Table(
        id="pytest_targets",
        title="Discovered pytest targets",
        description="Pytest targets exposed by domain modules",
        headers=("index", "target"),
        rows=tuple(
            TableRow((i, target), entity_id=f"pytest-target:{i}")
            for i, target in enumerate(targets)
        ),
        numeric=frozenset({0}),
    )


def _output_table(stdout: str, stderr: str) -> Table:
    rows: list[TableRow] = []

    for stream_name, text in (("stdout", stdout), ("stderr", stderr)):
        if not text:
            rows.append(TableRow((stream_name, "")))
            continue

        for i, line in enumerate(text.splitlines()):
            rows.append(TableRow((stream_name, i + 1, line)))

    return Table(
        id="pytest_output",
        title="Pytest output",
        description="Captured stdout and stderr from the pytest subprocess",
        headers=("stream", "line", "text"),
        rows=tuple(rows),
        numeric=frozenset({1}),
    )


def compute(ctx, inputs: dict[str, object]) -> DiagnosticResult:
    del ctx

    run_tests = bool(inputs["run_tests"])
    timeout = float(inputs["timeout"])

    targets = load_pytest_files()

    cards: list[Card] = [
        Card(
            label="targets",
            value=len(targets),
            status="ok" if targets else "warn",
            help="Number of pytest targets discovered from domain modules",
        )
    ]

    tables: list[Table] = [_target_table(targets)]
    notes: list[str] = [
        "Tests are discovered from domain modules rather than from a central hard-coded list."
    ]

    if run_tests:
        result = run_pytest_targets(
            targets,
            quiet=True,
            timeout=timeout,
        )
        cards.extend(
            [
                Card(
                    label="pytest",
                    value="passed" if result.passed else "failed",
                    status="ok" if result.passed else "bad",
                    help="Result of subprocess pytest run",
                ),
                Card(
                    label="return code",
                    value=result.returncode,
                    status="ok" if result.passed else "bad",
                ),
            ]
        )
        tables.append(_output_table(result.stdout, result.stderr))
        notes.append("Pytest is run in a subprocess so server state is isolated from pytest state.")
    else:
        notes.append("Set run_tests=true to execute the discovered pytest targets.")

    return DiagnosticResult(
        title="Test suite",
        summary="Discovered pytest targets for the domain package package.",
        cards=tuple(cards),
        tables=tuple(tables),
        notes=tuple(notes),
    )


def diagnostics() -> list[DiagnosticSpec]:
    return [
        DiagnosticSpec(
            id="dft_local.testsuite",
            group="dft_local",
            title="Test suite",
            description="List and optionally run tests exposed by domain modules.",
            inputs=(
                InputSpec(
                    name="run_tests",
                    label="run tests",
                    kind="bool",
                    default=False,
                    help="Run discovered pytest targets in a subprocess",
                ),
                InputSpec(
                    name="timeout",
                    label="timeout / s",
                    kind="float",
                    default=120.0,
                    min_value=1.0,
                    max_value=1200.0,
                    help="Subprocess timeout for pytest",
                ),
            ),
            compute=compute,
            tier="expensive",
        )
    ]
