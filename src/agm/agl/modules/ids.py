"""Module identity types for the AgL module system."""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

# Regex for a valid identifier segment: must start with letter or underscore,
# followed by letters, digits, underscores, or hyphens. AgL names are
# kebab-case, and a module path segment is an AgL name.
_IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_-]*$")

# Reserved segments used exclusively in the ENTRY_ID and RESERVED_ID
# sentinels.  The NUL byte (\x00) can never appear in a filesystem path
# segment, so no real .agl file can produce a ModuleId with either via
# from_path.
_ENTRY_SEGMENT = "\x00entry"
_RESERVED_SEGMENT = "\x00reserved"

ENTRY_DISPLAY = "<entry>"
"""The entry module's user-facing label, standing in for a name it has not got.

Every layer that renders an entry-qualified name for a user recognizes this
label, so it is defined once here rather than spelled at each of them.
"""

RESERVED_DISPLAY = "<builtin>"
"""The label for the host's own reserved identities, which have no module."""


@dataclass(frozen=True, slots=True)
class ModuleId:
    """Immutable identifier for an AgL module.

    Segments form a slash-separated logical path: ``("foo", "bar", "baz")``
    corresponds to ``foo/bar/baz`` and to the file ``foo/bar/baz.agl`` under a
    root.

    The module-system sentinels :data:`ENTRY_ID` and :data:`RESERVED_ID` are
    the only ``ModuleId``s whose ``is_entry`` / ``is_reserved`` properties
    return ``True``.  Their reserved segments contain a NUL byte and cannot be
    produced by :meth:`from_path`.
    """

    segments: tuple[str, ...]

    # ------------------------------------------------------------------
    # Entry-id discrimination
    # ------------------------------------------------------------------

    @property
    def is_entry(self) -> bool:
        """Return ``True`` if this is the distinguished entry-module sentinel."""
        return _ENTRY_SEGMENT in self.segments

    @property
    def is_reserved(self) -> bool:
        """Return ``True`` if this is the host's own reserved-identity sentinel."""
        return _RESERVED_SEGMENT in self.segments

    @property
    def is_standard_library(self) -> bool:
        """Return ``True`` for a module under the ``std`` tree.

        A ``builtin`` declaration in any such module is the *standard*
        declaration of its name; a ``builtin`` declaration anywhere else
        overrides it.
        """
        return self.segments[:1] == ("std",)

    @property
    def owns_standard_builtins(self) -> bool:
        """Return whether a ``builtin`` owned here is the standard one for its name.

        True for every standard-library module and for the reserved sentinel,
        whose identities stand in for a standard declaration nothing loaded.
        """
        return self.is_standard_library or self.is_reserved

    # ------------------------------------------------------------------
    # String representations
    # ------------------------------------------------------------------

    def path_str(self) -> str:
        """Return the slash-separated logical path, e.g. ``"foo/bar/baz"``."""
        return "/".join(self.segments)

    def display(self) -> str:
        """Return a user-facing module label that never exposes sentinel bytes."""
        if self.is_entry:
            return ENTRY_DISPLAY
        if self.is_reserved:
            return RESERVED_DISPLAY
        return self.path_str()

    def synthetic_name_component(self) -> str:
        """Return a Python-identifier-safe component for synthetic host names."""
        if self.is_entry:
            return "entry"
        return "_".join(self.segments)

    def relpath(self) -> str:
        """Return the os-independent relative file path, e.g. ``"foo/bar/baz.agl"``.

        Always uses forward slashes regardless of platform.
        """
        return "/".join(self.segments) + ".agl"

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def from_path(cls, s: str) -> "ModuleId":
        """Parse a slash-separated module path into a :class:`ModuleId`.

        Raises :class:`ValueError` if *s* is empty or any segment is not a
        valid identifier (``[A-Za-z_][A-Za-z0-9_-]*``).
        """
        if not s:
            raise ValueError("module id must not be empty")
        segments = s.split("/")
        for seg in segments:
            if not seg:
                raise ValueError(
                    f"module id {s!r} contains an empty segment (check for leading,"
                    " trailing, or consecutive slashes)"
                )
            if not _IDENTIFIER_RE.match(seg):
                raise ValueError(
                    f"module id segment {seg!r} is not a valid identifier"
                    " (must match [A-Za-z_][A-Za-z0-9_-]*)"
                )
        return cls(segments=tuple(segments))


def spell_scope_path(path: Sequence[str]) -> str:
    """Render a scope or declaration path as the ``A::B::name`` spelling.

    The result is display text — a scope path is carried as its separate
    segments everywhere else and is never recovered by splitting a spelling.
    """
    return "::".join(path)


def spell_declaration(
    module_id: ModuleId, path: Sequence[str], *, local_to: ModuleId | None = None
) -> str:
    """Render a declaration path the way a reader in *local_to* would write it.

    A declaration in the reading module needs no module qualifier, so it is
    spelled by its path alone, as does a reserved host identity, which has no
    module a reader could name; anything else is prefixed with the owning
    module's user-facing label.  Diagnostics that suggest a disambiguating
    spelling use this so the suggestion is one the reader can actually type.
    """
    scoped = spell_scope_path(path)
    if module_id.is_reserved or (local_to is not None and module_id == local_to):
        return scoped
    return f"{module_id.display()}::{scoped}"


def expand_module_wildcard(
    prefix: tuple[str, ...], module_ids: Iterable[ModuleId]
) -> tuple[ModuleId, ...]:
    """Return the loaded modules a wildcard import or export names.

    A wildcard reaches every loaded module whose path starts with *prefix*.
    An entry with no module identity is unreachable — its sentinel segment is
    unspellable — while an entry a package names is an ordinary member of that
    package's module tree and is reached like any other.  The result is ordered
    by logical path so two expansions of the same graph agree.
    """
    return tuple(
        sorted(
            (mid for mid in module_ids if mid.segments[: len(prefix)] == prefix),
            key=ModuleId.path_str,
        )
    )


# ------------------------------------------------------------------
# Sentinels
# ------------------------------------------------------------------

#: Distinguished sentinel representing an entry module with no module identity
#: of its own -- source supplied via ``-c``, a REPL entry, or a file no mounted
#: package owns.  (An entry inside a package keeps that package's declared
#: module id instead.)  Its reserved segment contains a NUL byte, so no real
#: ``.agl`` file on disk can produce a colliding ``ModuleId`` via
#: :meth:`ModuleId.from_path`.  Use ``module_id.is_entry`` to test.
ENTRY_ID: ModuleId = ModuleId(segments=(_ENTRY_SEGMENT,))

#: Distinguished sentinel owning the host's *reserved* nominal identities --
#: the fallback handles, shapes, and descriptors a host-known built-in name
#: carries when no source declares it (notably under ``--no-stdlib``).  They
#: belong to no file, so they carry this sentinel rather than any real
#: module's identity, and they spell bare wherever a name is rendered.
RESERVED_ID: ModuleId = ModuleId(segments=(_RESERVED_SEGMENT,))

#: Logical module id for the shipped standard-library prelude
#: (``packages/stdlib/src/prelude.agl``), the module every other module implicitly
#: imports.  It is the prelude *injection* point only: which module declares a
#: given ``builtin`` is not part of its meaning -- see
#: :attr:`ModuleId.is_standard_library`.
STD_PRELUDE_ID: ModuleId = ModuleId(segments=("std", "prelude"))

#: Logical module id for the shipped engine-settings standard library
#: (``std/config``), which declares the engine keys as ``builtin var`` bindings.
STD_CONFIG_ID: ModuleId = ModuleId(segments=("std", "config"))

#: Logical module id for the ambient process-environment standard library.
STD_ENV_ID: ModuleId = ModuleId(segments=("std", "env"))


def is_std_config_root(module_id: ModuleId, scope_path: Sequence[str]) -> bool:
    """Whether *module_id*/*scope_path* names a root ``std/config`` binding."""
    return module_id == STD_CONFIG_ID and not scope_path
