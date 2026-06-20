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


## Static Typst bundle export

Diagnostics have two rendering paths:

1. HTML rendering through `render.py` and `server.py`
2. Static Typst bundle export through `typst_bundle.py`

The HTML path is the interactive exploration surface. It may use Web Components,
Datastar model islands, pointer interaction, zoom/pan state, and WebGL-style
viewers.

The Typst path is a static archival/reporting path. It lowers the same
`DiagnosticResult` model into a directory that can be compiled independently.

Bundle layout:

    diagnostic_bundles/<diagnostic_id_with_underscores>/
      diagnostics.typ
      components.typ
      diagnostics.json
      manifest.json
      data/
      assets/
      lib/

Important files:

- `diagnostics.typ` is the standalone report entrypoint.
- `components.typ` contains generated Typst component wrappers.
- `diagnostics.json` contains the whole diagnostic tree.
- `manifest.json` records exported items, provenance, bundle mode, and static
  support status.
- `data/*.json` contains one JSON file per exported block.
- `lib/` points at the reusable Typst helper library.

The default export route is:

    /d-export/<diagnostic_id>

The route writes to:

    diagnostic_bundles/<diagnostic_id_with_underscores>/

### Typst helper library

The reusable Typst code lives at repository root:

    typst-diagnostics-lib/

The exporter normally symlinks this directory into each bundle as `lib/`.
When the project library is not available, the exporter writes a small fallback
library into the bundle instead. This keeps exported bundles useful from an
installed/package context, while keeping normal development on the shared
project library.

Public helper entrypoint:

    lib/mod.typ

Current helper modules:

- `diagnostic.typ` for report figures and notes
- `tables.typ` for static tables
- `plots.typ` for line plots and static viewer summaries

### Static lowering policy

Do not try to make the static Typst output pretend to be the interactive HTML
view. A WebGL or model-island backed component should be lowered to the best
honest static representation available.

Current lowering rules:

- `Graph2D` exports as a line graph through `line-graph(data)`.
- `Table` and `Matrix` export as static tables through `diagnostic-table(data)`.
- `WebGLView(renderer="dos_idos")` exports as two static line graphs:
  weighted DOS and integrated DOS.
- `WebGLView(renderer="region_surface")` exports as a payload summary table plus
  a per-band min/mean/max table for the selected scalar field.
- Unsupported WebGL views export as explicit unsupported placeholders.

Each WebGL lowering keeps the original payload in the JSON file. This preserves
provenance and makes later static renderers possible without changing the
diagnostic producer.

### Adding a new static lowering

When adding a new static renderer:

1. Keep the diagnostic producer unchanged if the existing payload is already
   sufficient.
2. Add a recogniser such as `_is_<renderer>_view(view)` in `typst_bundle.py`.
3. Add a JSON lowering function such as `<renderer>_to_json_data(view)`.
4. Add a branch in `_BundleWriter._collect_webgl`.
5. Add a helper in `typst-diagnostics-lib/`.
6. Mirror the helper in `_write_fallback_typst_lib`.
7. Add a focused bundle test in `test_typst_bundle.py`.
8. If the renderer emits Typst code, cover it in the optional compile smoke test.
9. Run the focused Typst bundle tests, then the diagnostics rendering tests.

Focused tests:

    pytest -q src/dft_local/diagnostics/test_typst_bundle.py

Broader diagnostics check:

    pytest -q \
      src/dft_local/diagnostics/test_typst_bundle.py \
      src/dft_local/diagnostics/test_models_business_logic.py \
      src/dft_local/diagnostics/test_server.py \
      src/dft_local/diagnostics/test_asgi_server.py


