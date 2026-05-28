# Running the diagnostics

The diagnostics live in `dft_local` and are separate from the
old diagnostics package.

## Run the server with uvicorn reload

From the repository root:

    DFT_LOCAL_DATA_ROOT=test_run/run_dir/data \
    uvicorn dft_local.diagnostics.server:app \
      --host 127.0.0.1 \
      --port 8765 \
      --reload \
      --reload-dir src/dft_local \
      --reload-dir src/dft_local

Then open:

    http://127.0.0.1:8765/

Useful diagnostics:

    http://127.0.0.1:8765/d/dft_local.testsuite
    http://127.0.0.1:8765/d/transport.boltzmann.calculation.overview
    transport.boltzmann.ashcroft_comparison.overview
    http://127.0.0.1:8765/d/transport.boltzmann.calculation.conductivity

## Data root

The server reads the data root from the environment variable:

    DFT_LOCAL_DATA_ROOT

If it is not set, the development default is:

    test_run/run_dir/data

## Run discovered tests without the web server

From the repository root:

    PYTHONPATH=src python - <<'PY'
    from dft_local.testsuite.runner import run_discovered_pytest

    result = run_discovered_pytest(timeout=120.0)
    print("returncode:", result.returncode)
    print(result.stdout)
    print(result.stderr)
    PY

The same discovered test list is shown in the `dft_local.testsuite` diagnostic.

## Related docs

    src/dft_local/docs/architecture.md
    src/dft_local/docs/migration_status.md
    src/dft_local/core/docs.md
    src/dft_local/transport/bands/docs.md
    src/dft_local/transport/boltzmann/calculation/docs.md

## Current status

dft_local currently provides:

- explicit diagnostic discovery with no import-time registry side effects
- local diagnostic model copy independent of FastAPI
- minimal ASGI diagnostic server for uvicorn
- subprocess pytest runner
- diagnostic page for discovered tests
- Boltzmann domain overview diagnostic
- real Boltzmann conductivity diagnostic
- bands domain overview diagnostic
- local core numerics, sparse, geometry, kernels, and local-problem modules
- copied Boltzmann physical/business tests
- copied bands continuation/region business tests
- direct core geometry/kernel tests
- discovered suite checkpoint: 168 passed

The old `dft_local` package remains untouched. The migrated transport/domain
logic now lives in `dft_local`; legacy context loading is only used
for diagnostic server compatibility.
