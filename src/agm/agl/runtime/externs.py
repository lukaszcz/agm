"""Extern companion loading, callable resolution, and invocation.

This eval-free runtime module owns the extern registry. It imports a
companion module once, resolves declared callables, and invokes each through
the boundary walkers in :mod:`agm.agl.runtime.boundary`.
"""

from __future__ import annotations

import decimal
import importlib.util
import sys
from collections.abc import Sequence
from pathlib import Path
from types import ModuleType
from typing import Protocol, cast

from agm.agl.diagnostics import AglError
from agm.agl.ir.builtin_nominals import NO_BUILTIN_DECLARATIONS, BuiltinNominals
from agm.agl.ir.contracts import ExternContract
from agm.agl.modules.ids import ModuleId
from agm.agl.runtime.boundary import (
    BoundaryScope,
    BoundaryViolation,
    decode_boundary_value,
    encode_boundary_value,
)
from agm.agl.semantics.cycles import AglCyclicValue, cyclic_value_raise
from agm.agl.semantics.exceptions import AglRaise, make_builtin_exception
from agm.agl.semantics.values import TextValue, Value


class ExternImportError(AglError):
    """A companion module raised while its top-level code was executed.

    Raised by :meth:`ExternRegistry.load_companion`; the pipeline converts it
    into a load-time diagnostic naming the AgL module.
    """

    def __init__(self, module_id: ModuleId, message: str) -> None:
        super().__init__(f"module {module_id.display()!r}: {message}")
        self.module_id = module_id


class ExternResolutionError(AglError):
    """A companion has no callable attribute matching a declared extern name.

    Raised by :meth:`ExternRegistry.resolve`; the pipeline converts it into a
    load-time diagnostic naming the AgL module and the extern function.
    """

    def __init__(self, module_id: ModuleId, name: str) -> None:
        super().__init__(
            f"module {module_id.display()!r} extern {name!r}: companion has no "
            "callable attribute of that name"
        )
        self.module_id = module_id
        self.name = name


# ---------------------------------------------------------------------------
# ExternRegistry
# ---------------------------------------------------------------------------


class ExternCallable(Protocol):
    """A resolved companion callable: positional arguments in, one value out.

    A structural protocol rather than ``Callable[..., object]``: the latter's
    ellipsis argument spec is an implicit ``Any`` under strict typing, so a
    resolved callable threaded through :meth:`ExternRegistry.invoke` would
    trip Any-detection at every call site that merely holds or passes it
    (not just where it is invoked).  A plain ``*args`` signature is exactly
    the shape ``invoke`` needs — positional encoded arguments, one result —
    without that pitfall.
    """

    def __call__(self, *args: object) -> object: ...


class ExternRegistry:
    """Imports extern companions, resolves their callables, and invokes them.

    Two-step resolution mirrors the way the pipeline discovers externs by
    module: :meth:`load_companion` imports a module's companion exactly once
    (cached per canonical path, so re-importing the same file — even for a
    different module id sharing it — is a no-op); :meth:`resolve` then looks
    up one already-loaded companion's callable by name, caching each
    successful lookup by ``(module_id, name)`` so repeated invocations of the
    same extern skip the attribute lookup.  If a module id is later remapped to
    a different companion module, cached callables for that module id are
    discarded.  :meth:`invoke` is the single chokepoint that turns every runtime
    failure crossing the boundary into a catchable ``ExternError``, mirroring
    ``AgentRegistry.dispatch``.
    """

    def __init__(self) -> None:
        self._by_path: dict[Path, ModuleType] = {}
        self._by_module: dict[ModuleId, ModuleType] = {}
        self._resolved: dict[tuple[ModuleId, str], ExternCallable] = {}

    def load_companion(self, module_id: ModuleId, companion_path: Path) -> ModuleType:
        """Import *companion_path* for *module_id*, executing it at most once.

        Registered in ``sys.modules`` under a synthetic name for the duration
        of the import only (no ``sys.path`` manipulation — the companion may
        still import installed packages absolutely).  A companion already
        imported for a different module id under the same canonical path is
        reused without re-running its top-level code.
        """
        canonical = companion_path.resolve()
        cached = self._by_path.get(canonical)
        if cached is not None:
            self._bind_module(module_id, cached)
            return cached

        synthetic_name = (
            f"agm_agl_extern_companion__{module_id.synthetic_name_component()}"
            f"__{len(self._by_path)}"
        )
        spec = importlib.util.spec_from_file_location(synthetic_name, canonical)
        # A ``.py``-suffixed location always resolves to a source-file loader
        # (verified to exist by the loader before this is ever called); this
        # can only be ``None`` for a suffix no loader recognizes.
        assert spec is not None and spec.loader is not None, (
            f"cannot build an import spec for companion {canonical}"
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules[synthetic_name] = module
        try:
            spec.loader.exec_module(module)
        except Exception as exc:
            raise ExternImportError(
                module_id, f"companion {canonical} failed to import: {exc}"
            ) from exc
        finally:
            sys.modules.pop(synthetic_name, None)

        self._by_path[canonical] = module
        self._bind_module(module_id, module)
        return module

    def _bind_module(self, module_id: ModuleId, module: ModuleType) -> None:
        """Bind *module_id* to *module*, dropping stale resolved callables."""
        previous = self._by_module.get(module_id)
        if previous is not None and previous is not module:
            stale_keys = [key for key in self._resolved if key[0] == module_id]
            for key in stale_keys:
                del self._resolved[key]
        self._by_module[module_id] = module

    def resolve(self, module_id: ModuleId, name: str) -> ExternCallable:
        """Return *module_id*'s companion callable named *name*.

        :meth:`load_companion` must have been called for *module_id* first.
        Resolution happens once per ``(module_id, name)`` pair — the effects
        layer calls this on every extern invocation, so a successful lookup
        is cached and returned on subsequent calls without re-consulting the
        companion module.  Failures are load-time (surfaced once, before any
        invocation reaches ``resolve`` again) and are never cached.

        :raises ExternResolutionError: when the companion has no attribute
            named *name*, or that attribute is not callable.
        """
        cache_key = (module_id, name)
        cached = self._resolved.get(cache_key)
        if cached is not None:
            return cached

        module = self._by_module.get(module_id)
        assert module is not None, (
            f"module {module_id.display()!r} has no loaded companion; "
            "load_companion must be called before resolve"
        )
        if not hasattr(module, name):
            raise ExternResolutionError(module_id, name)
        value: object = cast(object, getattr(module, name))
        if not callable(value):
            raise ExternResolutionError(module_id, name)
        self._resolved[cache_key] = value
        return value

    def invoke(
        self,
        function_name: str,
        contract: ExternContract,
        fn: ExternCallable,
        args: Sequence[Value],
        trace_id: str,
        *,
        nominals: BuiltinNominals = NO_BUILTIN_DECLARATIONS,
    ) -> Value:
        """Cross the boundary for one extern call: encode, call, decode.

        Mints a fresh seal token per declared type variable for this call,
        encodes *args* positionally per *contract*, calls *fn*, and strictly
        decodes its result.  All three runtime failure classes — *fn*
        raising, a return-contract violation, and an argument-conversion
        failure — become ``AglRaise(ExternError)`` here, the single
        chokepoint mirroring ``AgentRegistry.dispatch``.  ``python_type`` is
        the raising Python exception's class name, or empty for a contract
        violation.

        A companion repr'ing a sealed handle wrapping a cyclic array or dict
        (``SealedHandle.__repr__``, during *fn*) raises ``AglCyclicValue``;
        that is converted here into the catchable ``CyclicValueError`` rather
        than being folded into ``ExternError``.

        *nominals* resolves the ``ExternError``/``CyclicValueError`` nominal;
        it defaults to the shipped standard library's own identities for a
        caller (e.g. a direct unit test) that invokes without a program.
        """
        scope = BoundaryScope(
            seals={var: object() for var in contract.type_params},
            defs=dict(contract.defs) if contract.defs else None,
        )

        try:
            encoded_args = [
                encode_boundary_value(param.schema, arg, scope)
                for param, arg in zip(contract.params, args, strict=True)
            ]
        except BoundaryViolation as exc:
            raise _extern_error(
                function_name,
                f"argument conversion failed: {exc}",
                trace_id,
                python_type="",
                nominals=nominals,
            ) from exc
        except AglCyclicValue as exc:
            raise cyclic_value_raise(trace_id, nominals=nominals) from exc

        try:
            with decimal.localcontext():
                result = fn(*encoded_args)
        except AglCyclicValue as exc:
            raise cyclic_value_raise(trace_id, nominals=nominals) from exc
        except Exception as exc:
            raise _extern_error(
                function_name,
                str(exc) or type(exc).__name__,
                trace_id,
                python_type=type(exc).__name__,
                nominals=nominals,
            ) from exc

        try:
            return decode_boundary_value(contract.result, result, scope)
        except BoundaryViolation as exc:
            raise _extern_error(
                function_name,
                f"return value violates contract: {exc}",
                trace_id,
                python_type="",
                nominals=nominals,
            ) from exc
        except Exception as exc:
            raise _extern_error(
                function_name,
                f"return value validation failed: {exc}",
                trace_id,
                python_type=type(exc).__name__,
                nominals=nominals,
            ) from exc


def _extern_error(
    function_name: str,
    message: str,
    trace_id: str,
    *,
    python_type: str,
    nominals: BuiltinNominals,
) -> AglRaise:
    """Build the ``AglRaise(ExternError)`` carrier shared by every invoke failure."""
    return AglRaise(
        make_builtin_exception(
            "ExternError",
            message,
            nominals=nominals,
            trace_id=trace_id,
            function=TextValue(function_name),
            python_type=TextValue(python_type),
        )
    )
