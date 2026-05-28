from __future__ import annotations

import sys
import types

import pytest

from dft_local.core.discovery import DiscoveryError
from dft_local.diagnostics.discovery import load_diagnostics
from dft_local.diagnostics.models import DiagnosticResult, DiagnosticSpec


def dummy_compute(ctx, inputs):
    return DiagnosticResult("dummy", "dummy")


def install_module(name: str, diagnostics) -> None:
    module = types.ModuleType(name)
    module.diagnostics = diagnostics
    sys.modules[name] = module


def spec(id: str) -> DiagnosticSpec:
    return DiagnosticSpec(
        id=id,
        group="test",
        title=id,
        description="",
        inputs=(),
        compute=dummy_compute,
    )


def test_load_diagnostics_loads_specs_without_import_time_registry() -> None:
    install_module("dummy_dft_local_diag_a", lambda: [spec("a")])
    install_module("dummy_dft_local_diag_b", lambda: [spec("b")])

    got = load_diagnostics(("dummy_dft_local_diag_a", "dummy_dft_local_diag_b"))

    assert [s.id for s in got] == ["a", "b"]


def test_load_diagnostics_rejects_duplicate_ids() -> None:
    install_module("dummy_dft_local_diag_dup_a", lambda: [spec("same")])
    install_module("dummy_dft_local_diag_dup_b", lambda: [spec("same")])

    with pytest.raises(DiscoveryError, match="Duplicate diagnostic id"):
        load_diagnostics(("dummy_dft_local_diag_dup_a", "dummy_dft_local_diag_dup_b"))


def test_load_diagnostics_rejects_wrong_item_type() -> None:
    install_module("dummy_dft_local_diag_bad_item", lambda: ["not a spec"])

    with pytest.raises(DiscoveryError, match="expected DiagnosticSpec"):
        load_diagnostics(("dummy_dft_local_diag_bad_item",))
