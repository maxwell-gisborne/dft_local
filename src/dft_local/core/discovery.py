"""Explicit module discovery for domain-local behaviour.

The old diagnostics panel used import-time registration. This package avoids
that pattern.  A domain module should expose plain functions such as

    diagnostics() -> list[DiagnosticSpec]
    pytest_files() -> list[str]

The system-wide loader imports named modules and asks them for what they
provide.  This keeps behaviour local to the mathematical domain while keeping
the server and test runner generic.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any, Callable, Iterable, TypeVar


T = TypeVar("T")


class DiscoveryError(RuntimeError):
    """Raised when a discovered module has an invalid discovery interface."""


def collect_from_modules(
    module_names: Iterable[str],
    function_name: str,
    *,
    item_type: type[T] | None = None,
) -> tuple[T, ...]:
    """Collect items returned by a named function from many modules.

    Parameters
    ----------
    module_names:
        Import paths to inspect.
    function_name:
        Function each module may expose, for example ``diagnostics`` or
        ``pytest_files``.
    item_type:
        Optional runtime type check for each returned item.

    Returns
    -------
    tuple
        All items returned by all discovered functions, in module order.

    Notes
    -----
    Missing functions are ignored.  This lets a domain module expose tests
    without diagnostics, or diagnostics without test metadata.
    """

    items: list[T] = []

    for module_name in module_names:
        module = import_module(module_name)
        maybe_function = getattr(module, function_name, None)

        if maybe_function is None:
            continue

        if not callable(maybe_function):
            raise DiscoveryError(
                f"{module_name}.{function_name} exists but is not callable"
            )

        produced = maybe_function()

        if produced is None:
            continue

        if not isinstance(produced, (list, tuple)):
            raise DiscoveryError(
                f"{module_name}.{function_name} must return a list or tuple, "
                f"got {type(produced).__name__}"
            )

        for item in produced:
            if item_type is not None and not isinstance(item, item_type):
                raise DiscoveryError(
                    f"{module_name}.{function_name} returned "
                    f"{type(item).__name__}, expected {item_type.__name__}"
                )

            items.append(item)

    return tuple(items)


def require_unique(
    items: Iterable[T],
    key: Callable[[T], Any],
    *,
    name: str,
) -> tuple[T, ...]:
    """Return items after checking that a key is unique for each item."""

    out = tuple(items)
    seen: dict[Any, T] = {}

    for item in out:
        value = key(item)
        if value in seen:
            raise DiscoveryError(f"Duplicate {name}: {value!r}")
        seen[value] = item

    return out
