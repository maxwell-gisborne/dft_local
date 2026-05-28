# Diagnostic domains

A diagnostic domain is a package-backed namespace for related diagnostics, tests,
documentation, and business logic.

The diagnostic id hierarchy should match the file hierarchy where practical.

Example file hierarchy:

    src/dft_local/transport/boltzmann/calculation/
      __init__.py
      core.py
      diagnostics.py
      docs.md
      tests.py
      test_core.py
      test_diagnostics.py

Example diagnostic ids:

    transport.boltzmann.calculation.overview
    transport.boltzmann.calculation.conductivity

## What a domain can contain

A domain can contain:

- reusable business logic in `core.py`
- diagnostic page builders in `diagnostics.py`
- local documentation in `docs.md`
- domain test entrypoints in `tests.py`
- unit tests and business-logic tests
- compatibility shims when old import paths need to remain valid

## Creating a new domain

1. Create a package directory matching the intended diagnostic namespace.

2. Put computational logic in `core.py`.

3. Put diagnostic registration in `diagnostics.py`.

4. Add `DiagnosticSpec` ids matching the package path.

Example:

    DiagnosticSpec(
        id="transport.example.calculation.overview",
        title="Example calculation overview",
        group="transport.example.calculation",
        description="Overview of the example calculation.",
        inputs=(),
        compute=compute_example_overview,
    )

5. Add the diagnostics module to `src/dft_local/diagnostics/discovery.py`.

6. Add domain tests.

7. If the domain has a test command, add or update `tests.py` and wire it into
   `src/dft_local/testsuite/discovery.py`.

8. Add or update `docs.md`.

9. Run focused tests, then full tests.

## Moving an existing domain

When moving an existing domain, prefer this sequence:

1. Move the files with `git mv`.
2. Update imports to the new package path.
3. Update diagnostic ids to match the new package path.
4. Update docs and tests.
5. Keep compatibility shims at the old import paths if external code may still import them.
6. Verify the diagnostic index reflects the new hierarchy.
7. Run the full test suite.

Compatibility shim example:

    """Compatibility imports for the moved domain."""

    from dft_local.transport.boltzmann.calculation.core import *  # noqa: F401,F403

## Testing checklist

Run:

    python -m pytest -q
    dft-local test --timeout 120

For diagnostics, also check:

    from dft_local.diagnostics.server import DiagnosticApp, load_default_context

    ctx = load_default_context("test_run/run_dir/data")
    app = DiagnosticApp(ctx=ctx)

    assert "transport.example.calculation.overview" in app.specs
    html = app.diagnostic_page("transport.example.calculation.overview", {})
