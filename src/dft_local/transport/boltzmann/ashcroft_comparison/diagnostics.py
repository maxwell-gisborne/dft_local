from __future__ import annotations

from dft_local.diagnostics.models import Card, DiagnosticResult, DiagnosticSpec


def compute_overview(ctx, inputs: dict[str, object]) -> DiagnosticResult:
    return DiagnosticResult(
        title="Ashcroft comparison",
        summary="Compare local Boltzmann transport calculations with Ashcroft-style reference formulae.",
        cards=(
            Card(
                "status",
                "placeholder",
                "ok",
            ),
        ),
    )


def diagnostics() -> tuple[DiagnosticSpec, ...]:
    return (
        DiagnosticSpec(
            id="transport.boltzmann.ashcroft_comparison.overview",
            title="Ashcroft comparison overview",
            group="transport.boltzmann.ashcroft_comparison",
            description="Overview of Ashcroft-style comparison diagnostics.",
            inputs=(),
            compute=compute_overview,
        ),
    )
