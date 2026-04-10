"""Shared test fixtures."""

from __future__ import annotations

from collections.abc import Generator
from typing import Any
from unittest.mock import patch

import pytest


@pytest.fixture()
def mock_execvp() -> Generator[Any, None, None]:
    """Patch ``os.execvp`` so command handlers don't actually exec.

    The mock raises ``SystemExit(0)`` to simulate the exec replacing
    the process, which lets tests assert on the arguments passed.
    """
    def fake_execvp(file: str, args: list[str]) -> None:
        raise SystemExit(0)

    with patch("os.execvp", side_effect=fake_execvp) as m:
        yield m
