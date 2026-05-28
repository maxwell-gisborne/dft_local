from __future__ import annotations

import sys
import types

import pytest

from dft_local.core.discovery import (
    DiscoveryError,
    collect_from_modules,
    require_unique,
)


def install_module(name: str, **attrs: object) -> None:
    module = types.ModuleType(name)
    for key, value in attrs.items():
        setattr(module, key, value)
    sys.modules[name] = module


def test_collect_from_modules_ignores_missing_function() -> None:
    install_module("dummy_no_discovery")

    assert collect_from_modules(["dummy_no_discovery"], "diagnostics") == ()


def test_collect_from_modules_collects_items_in_module_order() -> None:
    install_module("dummy_discovery_a", diagnostics=lambda: ["a1", "a2"])
    install_module("dummy_discovery_b", diagnostics=lambda: ["b1"])

    assert collect_from_modules(
        ["dummy_discovery_a", "dummy_discovery_b"],
        "diagnostics",
        item_type=str,
    ) == ("a1", "a2", "b1")


def test_collect_from_modules_rejects_non_callable_discovery_attribute() -> None:
    install_module("dummy_bad_discovery_attr", diagnostics=[])

    with pytest.raises(DiscoveryError, match="not callable"):
        collect_from_modules(["dummy_bad_discovery_attr"], "diagnostics")


def test_collect_from_modules_rejects_non_sequence_return() -> None:
    install_module("dummy_bad_discovery_return", diagnostics=lambda: "not a list")

    with pytest.raises(DiscoveryError, match="list or tuple"):
        collect_from_modules(["dummy_bad_discovery_return"], "diagnostics")


def test_collect_from_modules_rejects_wrong_item_type() -> None:
    install_module("dummy_bad_discovery_item", diagnostics=lambda: [1])

    with pytest.raises(DiscoveryError, match="expected str"):
        collect_from_modules(
            ["dummy_bad_discovery_item"],
            "diagnostics",
            item_type=str,
        )


def test_require_unique_accepts_unique_keys() -> None:
    items = ("a", "bb", "ccc")

    assert require_unique(items, key=len, name="length") == items


def test_require_unique_rejects_duplicate_keys() -> None:
    with pytest.raises(DiscoveryError, match="Duplicate id"):
        require_unique(("a", "b"), key=lambda x: len(x), name="id")
