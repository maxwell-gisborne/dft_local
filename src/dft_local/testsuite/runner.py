"""Subprocess runner for discovered pytest targets.

The diagnostics server should not import and execute pytest in-process.  A
subprocess keeps pytest state, plugins, and imports isolated from the running
server.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import subprocess
import sys

from dft_local.testsuite.discovery import load_pytest_files


@dataclass(frozen=True, slots=True)
class PytestRun:
    """Result of a pytest subprocess run."""

    command: tuple[str, ...]
    returncode: int
    stdout: str
    stderr: str

    @property
    def passed(self) -> bool:
        """Whether pytest exited successfully."""

        return self.returncode == 0


def run_pytest_targets(
    targets: tuple[str, ...],
    *,
    cwd: str | Path = ".",
    quiet: bool = True,
    timeout: float = 120.0,
) -> PytestRun:
    """Run pytest on explicit targets in a subprocess."""

    command = [sys.executable, "-m", "pytest"]
    if quiet:
        command.append("-q")
    command.extend(targets)

    completed = subprocess.run(
        command,
        cwd=Path(cwd),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=timeout,
        check=False,
    )

    return PytestRun(
        command=tuple(command),
        returncode=int(completed.returncode),
        stdout=completed.stdout,
        stderr=completed.stderr,
    )


def run_discovered_pytest(
    *,
    module_names: tuple[str, ...] | None = None,
    cwd: str | Path = ".",
    quiet: bool = True,
    timeout: float = 120.0,
) -> PytestRun:
    """Run pytest targets exposed by discovered domain modules."""

    targets = load_pytest_files() if module_names is None else load_pytest_files(module_names)
    return run_pytest_targets(
        targets,
        cwd=cwd,
        quiet=quiet,
        timeout=timeout,
    )
