from __future__ import annotations

import sys
import types

from dft_local.testsuite.discovery import load_pytest_files


def install_module(name: str, pytest_files) -> None:
    module = types.ModuleType(name)
    module.pytest_files = pytest_files
    sys.modules[name] = module


def test_load_pytest_files_collects_targets_in_order() -> None:
    install_module("dummy_dft_local_tests_a", lambda: ["a.py", "b.py"])
    install_module("dummy_dft_local_tests_b", lambda: ["c.py"])

    assert load_pytest_files(
        ("dummy_dft_local_tests_a", "dummy_dft_local_tests_b")
    ) == ("a.py", "b.py", "c.py")
