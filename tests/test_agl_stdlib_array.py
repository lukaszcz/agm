"""Visibility contracts for the ``std/array`` standard-library module."""

from __future__ import annotations

import pytest

from agm.agl.capabilities import HostCapabilities
from agm.agl.scope import AglScopeError
from agm.agl.typecheck import AglTypeError
from tests.agl.module_graph import resolve_and_check_inline_entry


def test_array_methods_are_ambient_but_free_functions_require_an_import() -> None:
    resolve_and_check_inline_entry(
        "let values = [1, 2]\nvalues.map(fn(value: int) => value + 1)\n",
        HostCapabilities(),
    )

    with pytest.raises(AglScopeError):
        resolve_and_check_inline_entry("range(1, 2)\n", HostCapabilities())

    resolve_and_check_inline_entry("import std/array\narray::range(1, 2)\n", HostCapabilities())


def test_map_in_place_preserves_the_receiver_element_type() -> None:
    resolve_and_check_inline_entry(
        "let values = [1, 2]\nvalues.map!(fn(value: int) => value + 1)\n",
        HostCapabilities(),
    )

    with pytest.raises(AglTypeError):
        resolve_and_check_inline_entry(
            "let values = [1, 2]\nvalues.map!(fn(value: int) => value as text)\n",
            HostCapabilities(),
        )
