"""AgL semantics — foundation leaf package holding meaning-domain models.

This package is the lowest leaf in the AgL package tree that the rest of the
pipeline can depend on without import cycles.  It contains:

- ``agm.agl.semantics.values`` — the single value home: every runtime value
  type (leaf primitive tags, container/nominal types, IR closures, and the
  per-invocation frame model ``Cell``/``Slot``/``Frame``).
- ``agm.agl.semantics.exceptions`` — ``AglRaise`` (Python control-flow carrier
  for propagating AgL exceptions) and ``make_builtin_exception`` (factory for
  built-in exception values).

Importers should use the submodules directly; this ``__init__`` deliberately
re-exports nothing.
"""
