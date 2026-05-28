from dft_local.diagnostics_pannel.models import DiagnosticResult, DiagnosticSpec
from dft_local.diagnostics_pannel.registry import register, get_diagnostic


def test_registry_register_and_get_unique_id():
    spec = DiagnosticSpec(
        id="test.unique.registry",
        group="Tests",
        title="Registry test",
        description="",
        inputs=(),
        compute=lambda ctx, inputs: DiagnosticResult("ok", "ok"),
    )
    register(spec)
    assert get_diagnostic("test.unique.registry") is spec
