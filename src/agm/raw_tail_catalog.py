"""Canonical registry for AgL raw-tail surface spellings.

This pure data leaf is shared by the frontend and the program-name reservation
layer, so raw spellings cannot become valid config-program keys.
"""

from __future__ import annotations

from types import MappingProxyType
from typing import Final, Mapping

RAW_TAIL_BUILTINS: Final[Mapping[str, str]] = MappingProxyType({"exec!": "exec", "ask!": "ask"})
RAW_TAIL_NAMES: Final[frozenset[str]] = frozenset(RAW_TAIL_BUILTINS)
