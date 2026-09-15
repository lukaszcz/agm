"""AgL semantics — shared foundation package holding meaning-domain models.

Frontend and runtime layers share these models; the package itself depends only
on narrow IR identity/location types and its own semantic data. It contains:

- ``agm.agl.semantics.values`` — the single value home: every runtime value
  type (leaf primitive tags, container/nominal types, IR closures, and the
  per-invocation frame model ``Cell``/``Slot``/``Frame``).
- ``agm.agl.semantics.exceptions`` — ``AglRaise`` (Python control-flow carrier
  for propagating AgL exceptions) and ``make_builtin_exception`` (factory for
  built-in exception values).

Importers should use the submodules directly; this ``__init__`` deliberately
re-exports nothing.
"""
