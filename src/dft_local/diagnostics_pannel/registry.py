"""Small compatibility registry for old diagnostics registry tests."""

from __future__ import annotations

from dft_local.diagnostics.discovery import load_diagnostics

_registered = {spec.id: spec for spec in load_diagnostics()}


def register(spec):
    _registered[spec.id] = spec
    return spec


def get_diagnostic(diagnostic_id: str):
    return _registered[diagnostic_id]


def all_diagnostics():
    return tuple(_registered.values())


def import_builtin_diagnostics():
    return None
