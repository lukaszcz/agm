"""The lexical search order a qualifier follows within one module.

An unanchored qualifier is relative: it is tried against the enclosing scope
region, then outward level by level to the module root. A qualifier anchored to
the current module skips that walk and names the root outright. Both scope
resolution and type lookup follow this rule, and the attribute folder in
:mod:`agm.agl.syntax.module_constants` approximates it, so it lives here as a
pure function over scope paths, below every layer that applies it.
"""

from __future__ import annotations

__all__ = ["enclosing_scope_bases"]


def enclosing_scope_bases(
    scope_path: tuple[str, ...], *, rooted: bool = False
) -> tuple[tuple[str, ...], ...]:
    """Return the bases to prepend to a qualifier, innermost first.

    *scope_path* is the region the reference is written in. *rooted* says the
    qualifier is anchored to the current module, which selects the module root
    alone.
    """
    if rooted:
        return ((),)
    return tuple(scope_path[:end] for end in range(len(scope_path), -1, -1))
