# dft_local architecture

This package is an domain package kept separate from the existing
`dft_local` package.

The main design rule is locality of behaviour:

- mathematical code lives in a domain module
- diagnostics for that mathematical code live next to it
- documentation for the domain lives next to it
- test discovery metadata lives next to it
- system-wide tools discover and run these things

The diagnostics server should not know transport physics, group Fourier
symbols, local paths, regions, or Boltzmann conductivity details.  It should
only know how to render structured diagnostic results.

The test runner should not know the implementation details of each domain. It
should discover test targets supplied by each domain.

## Directory sketch

```text
src/dft_local/
  core/
    shared protocols and discovery helpers

  diagnostics/
    generic diagnostic models, loader, renderer, and server

  testsuite/
    generic pytest discovery and optional server integration

  transport/
    boltzmann/
      core.py
      diagnostics.py
      tests.py
      docs.md
````

## Import direction

Domain code should now prefer local core modules. Legacy imports
should be limited to compatibility boundaries and removed as migration
continues.

Generic infrastructure may import structured model types, but it should not
import domain implementations directly except through discovery.

Preferred direction:

```text
domain module -> dft_local.core as needed
domain diagnostics -> domain core + generic diagnostic models
generic diagnostics server -> discovery -> DiagnosticSpec
```

Avoid:

```text
generic services.py importing every physical thing
import-time registration side effects
large global helper modules with mixed responsibilities
```

## Diagnostic contract

A diagnostic module should expose:

```python
def diagnostics() -> list[DiagnosticSpec]:
    ...
```

No side-effect registration.

A diagnostic compute function should have the shape:

```python
def compute(ctx, inputs: dict[str, object]) -> DiagnosticResult:
    ...
```

The context object is supplied by the server.  During migration it may wrap the
local `dft_local.diagnostics.context.DiagnosticContext`.

## Test contract

A domain test module should expose metadata, not run tests itself:

```python
def pytest_files() -> list[str]:
    ...
```

or simply keep normal pytest files in a discoverable location.  The diagnostics
server may later run selected tests through a safe subprocess wrapper.
