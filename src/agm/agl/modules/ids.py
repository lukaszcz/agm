"""Module identity types for the AgL module system."""

from __future__ import annotations

import re
from dataclasses import dataclass

# Regex for a valid identifier segment: must start with letter or underscore,
# followed by letters, digits, or underscores.
_IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

# Reserved segment used exclusively in the ENTRY_ID sentinel.  The NUL byte
# (\x00) can never appear in a filesystem path segment, so no real .agl file
# can produce a ModuleId with this segment via from_dotted.
_ENTRY_SEGMENT = "\x00entry"


@dataclass(frozen=True, slots=True)
class ModuleId:
    """Immutable identifier for an AgL module.

    Segments form a dotted logical name: ``("foo", "bar", "baz")`` corresponds
    to the path ``foo/bar/baz.agl`` under a root.

    The module-system sentinel :data:`ENTRY_ID` is the only ``ModuleId`` whose
    ``is_entry`` property returns ``True``.  Its reserved segment contains a
    NUL byte and cannot be produced by :meth:`from_dotted`.
    """

    segments: tuple[str, ...]

    # ------------------------------------------------------------------
    # Entry-id discrimination
    # ------------------------------------------------------------------

    @property
    def is_entry(self) -> bool:
        """Return ``True`` if this is the distinguished entry-module sentinel."""
        return _ENTRY_SEGMENT in self.segments

    # ------------------------------------------------------------------
    # String representations
    # ------------------------------------------------------------------

    def dotted(self) -> str:
        """Return the dotted logical name, e.g. ``"foo.bar.baz"``."""
        return ".".join(self.segments)

    def relpath(self) -> str:
        """Return the os-independent relative file path, e.g. ``"foo/bar/baz.agl"``.

        Always uses forward slashes regardless of platform.
        """
        return "/".join(self.segments) + ".agl"

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def from_dotted(cls, s: str) -> "ModuleId":
        """Parse a dotted module name into a :class:`ModuleId`.

        Raises :class:`ValueError` if *s* is empty or any segment is not a
        valid identifier (``[A-Za-z_][A-Za-z0-9_]*``).
        """
        if not s:
            raise ValueError("module id must not be empty")
        segments = s.split(".")
        for seg in segments:
            if not seg:
                raise ValueError(
                    f"module id {s!r} contains an empty segment (check for leading,"
                    " trailing, or consecutive dots)"
                )
            if not _IDENTIFIER_RE.match(seg):
                raise ValueError(
                    f"module id segment {seg!r} is not a valid identifier"
                    " (must match [A-Za-z_][A-Za-z0-9_]*)"
                )
        return cls(segments=tuple(segments))


# ------------------------------------------------------------------
# Sentinels
# ------------------------------------------------------------------

#: Distinguished sentinel representing the entry module (the script passed to
#: ``agm exec`` or supplied via ``-c``).  Its reserved segment contains a NUL
#: byte, so no real ``.agl`` file on disk can produce a colliding ``ModuleId``
#: via :meth:`ModuleId.from_dotted`.  Use ``module_id.is_entry`` to test.
ENTRY_ID: ModuleId = ModuleId(segments=(_ENTRY_SEGMENT,))

# Reserved segment used exclusively in the PRELUDE_ID sentinel.  Contains a NUL
# byte (different from _ENTRY_SEGMENT) so it cannot collide with any real module
# or with ENTRY_ID.
_PRELUDE_SEGMENT = "\x00prelude"

#: Distinguished sentinel representing the built-in prelude / standard library.
#: Used as the ``module_id`` component of :class:`~agm.agl.ir.ids.NominalId`
#: for all built-in exception types (``RecursionError``, ``IndexError``,
#: ``AgentParseError``, etc.) and other prelude nominals that have no source
#: module.  Its reserved segment contains a NUL byte and can never be produced
#: by :meth:`ModuleId.from_dotted`.
PRELUDE_ID: ModuleId = ModuleId(segments=(_PRELUDE_SEGMENT,))

#: Logical module id for the shipped core standard library.
STD_CORE_ID: ModuleId = ModuleId(segments=("std", "core"))
