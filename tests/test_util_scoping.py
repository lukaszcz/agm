"""Context-variable scoping guard."""

from __future__ import annotations

import contextvars

import pytest

from agm.util.scoping import ScopedVar

_VAR: contextvars.ContextVar[int] = contextvars.ContextVar("test_scoped_var", default=0)


def test_scoped_var_restores_the_previous_binding_on_every_path() -> None:
    with ScopedVar(_VAR, 1):
        assert _VAR.get() == 1
        with ScopedVar(_VAR, 2):
            assert _VAR.get() == 2
        assert _VAR.get() == 1

        with pytest.raises(RuntimeError), ScopedVar(_VAR, 3):
            assert _VAR.get() == 3
            raise RuntimeError("boom")
        assert _VAR.get() == 1

    assert _VAR.get() == 0
