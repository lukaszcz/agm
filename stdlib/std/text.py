"""Runtime interpolation companion for ``std/text``.

Externs resolve by attribute name, so re-exporting the shared implementation is
enough — no forwarding wrapper is needed.
"""

from agm.util.interp import interp

__all__ = ["interp"]
