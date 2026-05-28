from __future__ import annotations

from dft_local.testsuite.runner import run_pytest_targets


def test_run_pytest_targets_reports_success_for_known_small_test() -> None:
    result = run_pytest_targets(
        ("src/dft_local/core/test_discovery.py",),
        quiet=True,
        timeout=30.0,
    )

    assert result.passed
    assert result.returncode == 0
    assert "passed" in result.stdout
    assert result.stderr == ""
    assert result.command[:3][-2:] == ("-m", "pytest")
