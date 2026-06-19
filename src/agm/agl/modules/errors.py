"""Module-system errors for the AgL module loader.

All error classes subclass :class:`~agm.agl.diagnostics.AglError` and carry a
``SourceSpan`` where the trigger is an import declaration — enabling the
diagnostics machinery to report the originating file and source location.
"""

from __future__ import annotations

from pathlib import Path

from agm.agl.diagnostics import AglError
from agm.agl.modules.ids import ModuleId
from agm.agl.syntax.spans import SourceSpan


class ModuleNotFound(AglError):
    """A module id could not be found in any searched root.

    ``module_id`` is the id that was not found.  ``searched_roots`` lists every
    root that was searched (sorted, for deterministic diagnostics — the same
    order returned by :meth:`~agm.agl.modules.roots.RootSet.sorted_roots`).
    ``span`` is the :class:`~agm.agl.syntax.spans.SourceSpan` of the
    ``import`` declaration that triggered the lookup, when available.
    """

    def __init__(
        self,
        module_id: ModuleId,
        searched_roots: tuple[Path, ...],
        *,
        span: SourceSpan | None = None,
    ) -> None:
        roots_str = ", ".join(str(r) for r in searched_roots)
        msg = (
            f"module '{module_id.dotted()}' not found; "
            f"searched roots: [{roots_str}]"
        )
        super().__init__(msg, span=span)
        self.module_id = module_id
        self.searched_roots = searched_roots


class AmbiguousModule(AglError):
    """A module id resolves to ≥2 distinct canonical files.

    ``module_id`` is the ambiguous id.  ``candidates`` is a sorted tuple of
    canonical :class:`~pathlib.Path` objects, one per distinct file the id
    resolved to.  ``span`` is the originating import declaration span, when
    available.
    """

    def __init__(
        self,
        module_id: ModuleId,
        candidates: tuple[Path, ...],
        *,
        span: SourceSpan | None = None,
    ) -> None:
        candidates_str = ", ".join(str(c) for c in candidates)
        msg = (
            f"module '{module_id.dotted()}' is ambiguous; "
            f"found in multiple roots: [{candidates_str}]"
        )
        super().__init__(msg, span=span)
        self.module_id = module_id
        self.candidates = candidates


class ModulePrefixNotFound(AglError):
    """A wildcard import prefix (``foo.*``) matched no module.

    ``prefix`` is the tuple of segments that formed the wildcard prefix (e.g.
    ``("foo", "bar")`` for ``import foo.bar.*``).  ``span`` is the originating
    import declaration span, when available.
    """

    def __init__(
        self,
        prefix: tuple[str, ...],
        *,
        span: SourceSpan | None = None,
    ) -> None:
        dotted_prefix = ".".join(prefix)
        msg = f"wildcard prefix '{dotted_prefix}.*' matched no module"
        super().__init__(msg, span=span)
        self.prefix = prefix


class ImportEntryError(AglError):
    """An import declaration resolves to the entry file's canonical identity.

    Importing the entry program is rejected (D9): the entry is a non-importable
    program root.  ``module_id`` is the module id the user attempted to import.
    ``entry_path`` is the canonical path of the entry file.  ``span`` is the
    originating import declaration span, when available.
    """

    def __init__(
        self,
        module_id: ModuleId,
        entry_path: Path,
        *,
        span: SourceSpan | None = None,
    ) -> None:
        msg = (
            f"cannot import '{module_id.dotted()}': "
            f"it resolves to the entry file '{entry_path}'"
        )
        super().__init__(msg, span=span)
        self.module_id = module_id
        self.entry_path = entry_path
