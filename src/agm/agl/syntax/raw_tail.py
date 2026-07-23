"""Compatibility exports for the canonical raw-tail catalog.

Raw-tail spellings and their ordinary call targets are shared with program-name
reservation, so their registry lives in the pure :mod:`agm.raw_tail_catalog`
data leaf rather than in the syntax package.
"""

from __future__ import annotations

from agm.raw_tail_catalog import RAW_TAIL_BUILTINS as RAW_TAIL_BUILTINS
from agm.raw_tail_catalog import RAW_TAIL_NAMES as RAW_TAIL_NAMES
