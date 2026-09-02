"""Extern companion loading, callable resolution, and invocation.

This eval-free runtime module owns the extern registry. It imports a
companion module once, resolves declared callables, and invokes each through
the value-directed conversion functions in :mod:`agm.agl.runtime.boundary`.
Container views remain usable after a call returns and continue to reflect
their underlying AgL containers.
"""

from __future__ import annotations

import contextvars
import decimal
import importlib.machinery
import importlib.util
import sys
import threading
from collections.abc import Callable, Iterator, Sequence
from contextlib import AbstractContextManager, contextmanager, nullcontext
from pathlib import Path
from types import ModuleType
from typing import Protocol, cast

from agm.agl.diagnostics import AglError
from agm.agl.ir.builtin_nominals import NO_BUILTIN_DECLARATIONS, BuiltinNominals
from agm.agl.ir.ids import NominalId
from agm.agl.ir.program import NominalDescriptor
from agm.agl.modules.ids import ModuleId
from agm.agl.runtime.boundary import (
    AglArrayView,
    AglDictView,
    AglException,
    AglJson,
    BoundaryTypeError,
    BoundaryViolation,
    active_function_encoder,
    decode_boundary_value,
    encode_boundary_value,
    synthesize_nominal_classes,
)
from agm.agl.self_validation import self_validation_enabled
from agm.agl.semantics.cycles import AglCyclicValue, cyclic_value_raise
from agm.agl.semantics.exceptions import AglRaise, make_builtin_exception
from agm.agl.semantics.values import ArrayValue, DictValue, IrClosureValue, TextValue, Value
from agm.core import fs
from agm.util.scoping import ScopedVar

# These companion module attributes are APIs, never synthesized nominal aliases.
_COMPANION_API_NAMES = frozenset({"AglException", "array", "dict", "json", "nominals", "runtime"})


class ExternRuntimeState:
    """Mutable companion state belonging to one interpreter instance."""

    __slots__ = ("_values",)

    def __init__(self) -> None:
        self._values: dict[str, object] = {}

    def get_or_create(self, key: str, factory: Callable[[], object]) -> object:
        """Return this state's value for *key*, creating it once when absent."""
        if key not in self._values:
            self._values[key] = factory()
        return self._values[key]


class _CompanionRuntime:
    """Route companion state to the interpreter active in this call context."""

    __slots__ = ("_active_state", "_detached_state")

    def __init__(self) -> None:
        self._active_state: contextvars.ContextVar[ExternRuntimeState | None] = (
            contextvars.ContextVar("agl_extern_runtime_state", default=None)
        )
        # Direct companion use outside evaluation remains useful to host tests
        # and tools, but evaluation always supplies its interpreter-owned state.
        self._detached_state = ExternRuntimeState()

    def activate(self, state: ExternRuntimeState | None) -> AbstractContextManager[None]:
        """Make *state* visible to a companion for the dynamic call extent."""
        if state is None:
            # Absent evaluator state must not shadow an enclosing activation.
            return nullcontext()
        return ScopedVar(self._active_state, state)

    def state(self, key: str, factory: Callable[[], object]) -> object:
        """Get companion-local state scoped to the active interpreter call."""
        state = self._active_state.get()
        return (state if state is not None else self._detached_state).get_or_create(key, factory)


_COMPANION_RUNTIME = _CompanionRuntime()


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


class _CacheFreeLoader(importlib.machinery.SourceFileLoader):
    """A source loader that never writes a bytecode cache beside its source.

    CPython's source loader caches compiled bytecode in a ``__pycache__``
    directory next to the module it imports.  A companion is imported from
    wherever its AgL module lives, including an installed package's immutable
    store tree, which AGM must never write into: unrecorded files there are
    not part of the package and would strand its removal.  Discarding the
    write leaves the import otherwise identical; an already-present cache is
    still read.
    """

    def set_data(self, path: str, data: object, *, _mode: int = 0o666) -> None:
        """Discard bytecode the loader would otherwise cache next to a companion."""


# ---------------------------------------------------------------------------
# ExternRegistry
# ---------------------------------------------------------------------------


class CallableProxyError(RuntimeError):
    """An AgL callback was invoked outside its active interpreter window."""


class ExternCallWindow:
    """Track one interpreter's active extern calls and their owning thread.

    An interpreter owns this guard rather than its shared
    :class:`ExternRegistry`: multiple interpreters can safely use one
    registry, but a callback belongs to the interpreter that encoded it.
    """

    __slots__ = ("_depth", "_lock", "_thread")

    def __init__(self) -> None:
        self._depth = 0
        self._lock = threading.Lock()
        self._thread: int | None = None

    @contextmanager
    def active(self) -> Iterator[None]:
        """Open a nestable callback window on this interpreter's thread."""
        thread_id = threading.get_ident()
        with self._lock:
            if self._thread is None:
                self._thread = thread_id
            elif self._thread != thread_id:
                raise CallableProxyError(
                    "an interpreter extern call is already active on another thread"
                )
            self._depth += 1
        try:
            yield
        finally:
            with self._lock:
                self._depth -= 1
                if self._depth == 0:
                    self._thread = None

    def require_active(self) -> None:
        """Require this interpreter's active extern call on its owning thread."""
        with self._lock:
            if self._depth != 0 and self._thread == threading.get_ident():
                return
        raise CallableProxyError(
            "AgL callbacks may run only on the owning interpreter thread during an extern call"
        )


class _ClosureInvoker(Protocol):
    """The evaluator-owned execution hook for one crossed AgL closure."""

    def __call__(self, args: tuple[Value, ...]) -> Value: ...


class AglCallableProxy:
    """A Python callable backed by an AgL closure during an extern call.

    The evaluator supplies execution while this runtime-side adapter owns
    Python argument/result conversion and the invocation-window guard.
    """

    __slots__ = ("_arity", "_closure", "_require_active_window", "_invoke")

    def __init__(
        self,
        *,
        arity: int,
        closure: IrClosureValue,
        require_active_window: Callable[[], None],
        invoke: _ClosureInvoker,
    ) -> None:
        self._arity = arity
        self._closure = closure
        self._require_active_window = require_active_window
        self._invoke = invoke

    def __call__(self, *args: object) -> object:
        self._require_active_window()
        if len(args) != self._arity:
            raise TypeError(
                f"AgL callback expected {self._arity} positional arguments, got {len(args)}"
            )
        try:
            values = tuple(decode_boundary_value(arg) for arg in args)
        except BoundaryViolation as exc:
            raise BoundaryTypeError(str(exc)) from exc
        try:
            return encode_boundary_value(self._invoke(values))
        except AglRaise as exc:
            raise AglException(exc.exc) from exc


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
    module: :meth:`load_companion` imports a module's companion once per
    version of the file (cached per canonical path, so re-importing an
    unchanged file — even for a different module id sharing it — is a no-op,
    while an edited one is imported again); :meth:`resolve` then looks
    up one already-loaded companion's callable by name, caching each
    successful lookup by ``(module_id, name)`` so repeated invocations of the
    same extern skip the attribute lookup.  If a module id is later remapped to
    a different companion module, cached callables for that module id are
    discarded.  :meth:`invoke` is the single chokepoint that turns every runtime
    failure crossing the boundary into a catchable ``ExternError``, mirroring
    the agent value dispatcher.
    """

    def __init__(self) -> None:
        self._by_path: dict[Path, tuple[fs.IdentityStamp, ModuleType]] = {}
        self._by_module: dict[ModuleId, ModuleType] = {}
        self._resolved: dict[tuple[ModuleId, str], ExternCallable] = {}
        self._nominal_classes: dict[NominalId, type[object]] = {}
        self._nominal_by_id: dict[NominalId, NominalDescriptor] = {}

    def set_nominals(self, descriptors: dict[NominalId, NominalDescriptor]) -> None:
        """Materialize this program's companion-visible nominal classes, insert-only.

        A nominal identity's layout is fixed at its declaration, so a class
        is synthesized for an identity at most once -- the first call that
        carries it. A companion reference captured before a later
        redeclaration -- a module global, a closure, a default argument --
        therefore stays valid, still denoting the declaration it was
        captured from, rather than following the redeclaration.

        The per-identity descriptor snapshot is refreshed on every call
        regardless, because ``NominalDescriptor.bears_name_path`` -- whether
        this identity is the one its name path currently resolves to -- can
        flip from one call to the next as a later declaration supersedes it;
        :meth:`_agl_module` consults that snapshot to decide which identity a
        companion's bare/dotted nominal lookup resolves to at import time.
        """
        if self_validation_enabled():
            for nominal, descriptor in descriptors.items():
                previous = self._nominal_by_id.get(nominal)
                if previous is not None and previous != descriptor:
                    raise AssertionError(
                        f"nominal {nominal!r} re-registered with a different shape: "
                        f"{previous!r} -> {descriptor!r}"
                    )

        new_ids = descriptors.keys() - self._nominal_classes.keys()
        if new_ids:
            classes = synthesize_nominal_classes(
                (descriptors[nominal] for nominal in new_ids), self._nominal_classes
            )
            self._nominal_classes.update(classes)
        self._nominal_by_id.update(descriptors)

    def _agl_module(self) -> ModuleType:
        """Build the temporary ``agl`` module exposed while importing a companion.

        Only identities that currently bear their own name path
        (``NominalDescriptor.bears_name_path``) are exposed: a superseded
        declaration and a declaration from an unpromoted REPL entry share
        their name path with the declaration that now owns it, so including
        them would make ``leaves`` collide on that path and would leave a
        stale identity reachable by a fresh companion import. Which identity
        wins is therefore decided by the type table's name index (threaded
        through ``bears_name_path``), never by insertion order.

        The host's reserved fallback identities are left out as well: they
        belong to no module, so there is no ``nominals.<path>`` a companion
        could address them by. A companion that needs one of those built-ins
        addresses a real declaration instead, which the module it belongs to
        imports.
        """
        module = ModuleType("agl")
        setattr(module, "array", _array)
        setattr(module, "dict", _dict)
        setattr(module, "json", AglJson)
        setattr(module, "AglException", AglException)
        setattr(module, "runtime", _COMPANION_RUNTIME)
        nominals = ModuleType("agl.nominals")
        setattr(module, "nominals", nominals)
        leaves: dict[tuple[str, ...], type[object]] = {}
        for nominal, cls in self._nominal_classes.items():
            descriptor = self._nominal_by_id[nominal]
            if descriptor.bears_name_path and not descriptor.module_id.is_reserved:
                leaves[_nominal_identity_path(descriptor)] = cls
        _build_nominal_namespace(nominals, leaves)
        names: dict[str, list[type[object]]] = {}
        for cls in leaves.values():
            names.setdefault(cls.__name__, []).append(cls)
        for name, classes in names.items():
            if len(classes) == 1 and name not in _COMPANION_API_NAMES:
                setattr(module, name, classes[0])
        return module

    def load_companion(self, module_id: ModuleId, companion_path: Path) -> ModuleType:
        """Import *companion_path* for *module_id*, executing it once per version.

        Registered in ``sys.modules`` under a synthetic name for the duration
        of the import only (no ``sys.path`` manipulation — the companion may
        still import installed packages absolutely).  A companion already
        imported for a different module id under the same canonical path is
        reused without re-running its top-level code, which is what keeps a
        shared companion's top level from running twice.

        That reuse is conditional on the file's identity stamp, so an edited
        companion is imported again rather than serving the code a long-lived
        process happened to read first.  The fresh import produces a new module
        object, and :meth:`_bind_module` drops the callables resolved from the
        old one; a value already handed out keeps denoting the version it was
        taken from, as a superseded declaration does.
        """
        canonical = companion_path.resolve()
        try:
            stamp = fs.identity_stamp(canonical)
        except OSError as exc:
            # The companion was recorded when the loader saw it and may since
            # have been removed or made unreadable; that reads as a failure to
            # import it, which callers already turn into a diagnostic.
            raise self._import_error(module_id, canonical, exc) from exc
        cached = self._by_path.get(canonical)
        if cached is not None and cached[0] == stamp:
            self._bind_module(module_id, cached[1])
            return cached[1]

        synthetic_name = (
            f"agm_agl_extern_companion__{module_id.synthetic_name_component()}"
            f"__{len(self._by_path)}"
        )
        spec = importlib.util.spec_from_file_location(
            synthetic_name, canonical, loader=_CacheFreeLoader(synthetic_name, str(canonical))
        )
        # A supplied source-file loader always yields a spec; ``None`` would
        # mean the location carries a suffix no loader recognizes.
        assert spec is not None and spec.loader is not None, (
            f"cannot build an import spec for companion {canonical}"
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules[synthetic_name] = module
        previous_agl = sys.modules.get("agl")
        sys.modules["agl"] = self._agl_module()
        try:
            spec.loader.exec_module(module)
        except Exception as exc:
            raise self._import_error(module_id, canonical, exc) from exc
        finally:
            sys.modules.pop(synthetic_name, None)
            if previous_agl is None:
                sys.modules.pop("agl", None)
            else:
                sys.modules["agl"] = previous_agl

        self._by_path[canonical] = (stamp, module)
        self._bind_module(module_id, module)
        return module

    @staticmethod
    def _import_error(module_id: ModuleId, canonical: Path, cause: Exception) -> ExternImportError:
        """Report *cause* as *module_id*'s companion failing to import."""

        return ExternImportError(module_id, f"companion {canonical} failed to import: {cause}")

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
        fn: ExternCallable,
        args: Sequence[Value],
        *,
        nominals: BuiltinNominals = NO_BUILTIN_DECLARATIONS,
        function_encoder: Callable[[IrClosureValue], object] | None = None,
        runtime_state: ExternRuntimeState | None = None,
    ) -> Value:
        """Cross the boundary for one extern call: encode, call, and decode.

        Encodes *args* by their runtime value class, calls *fn*, and decodes
        its result by Python type. An argument or return value with no
        boundary representation, and an ordinary exception raised by *fn*,
        all become catchable ``ExternError`` values; ``python_type`` is empty
        for an unsupported representation and otherwise names the Python
        exception class. A companion repr'ing a cyclic view -- while *fn*
        runs, or while this builds an ``ExternError`` message from an
        exception *fn* raised -- raises ``AglCyclicValue``, which becomes
        ``CyclicValueError`` instead. Retained views remain live after the
        call, except for reads of a function-valued field or element, which
        need the encoder this call publishes.

        *nominals* resolves the ``ExternError``/``CyclicValueError`` nominal;
        it defaults to the shipped standard library's own identities for a
        caller (e.g. a direct unit test) that invokes without a program.
        *runtime_state* is the evaluator-owned companion state activated for
        this call; absent direct callers use detached host state instead.
        *function_encoder* turns an AgL closure into a callable proxy and is
        published for the call's extent, so every closure a companion reaches
        -- through an argument, a retained view, or a nested container --
        encodes through the interpreter it is running under. Like the
        companion runtime state activated here, it is scoped to this call's
        context: a thread the companion spawns sees it only if it runs in a
        copy of that context.
        """
        with active_function_encoder(function_encoder):
            try:
                encoded_args = [encode_boundary_value(arg) for arg in args]
            except BoundaryViolation as exc:
                raise _extern_error(
                    function_name,
                    f"argument cannot cross the boundary: {exc}",
                    python_type="",
                    nominals=nominals,
                ) from exc

            try:
                with _COMPANION_RUNTIME.activate(runtime_state), decimal.localcontext():
                    result = fn(*encoded_args)
            except AglException as exc:
                raise AglRaise(exc.value) from exc
            except AglCyclicValue as exc:
                raise cyclic_value_raise(nominals=nominals) from exc
            except Exception as exc:
                try:
                    message = str(exc) or type(exc).__name__
                except AglCyclicValue as cyclic_exc:
                    raise cyclic_value_raise(nominals=nominals) from cyclic_exc
                raise _extern_error(
                    function_name,
                    message,
                    python_type=type(exc).__name__,
                    nominals=nominals,
                ) from exc

            try:
                return decode_boundary_value(result)
            except BoundaryViolation as exc:
                raise _extern_error(
                    function_name,
                    f"return value cannot cross the boundary: {exc}",
                    python_type="",
                    nominals=nominals,
                ) from exc
            except Exception as exc:
                raise _extern_error(
                    function_name,
                    f"return value validation failed: {exc}",
                    python_type=type(exc).__name__,
                    nominals=nominals,
                ) from exc


def _nominal_identity_path(descriptor: NominalDescriptor) -> tuple[str, ...]:
    """Return a nominal's namespace path, rooted by module (or ``entry``) then scope.

    Only identities that belong to a real module reach here; the reserved
    sentinel has no path and is filtered out by :meth:`Externs._agl_module`.
    """
    if descriptor.module_id.is_entry:
        return ("entry", *descriptor.scope_path, descriptor.declared_name)
    return (
        *descriptor.module_id.segments,
        *descriptor.scope_path,
        descriptor.declared_name,
    )


def _build_nominal_namespace(root: ModuleType, leaves: dict[tuple[str, ...], type[object]]) -> None:
    """Expose every nominal under its identity path in one order-independent pass.

    A path segment that is itself a nominal's own identity path becomes that
    nominal's class rather than a plain namespace module, so a scope and a
    nominal may legally share a name: the class serves as both the leaf
    value and, for anything nested under it, the namespace container.
    """
    containers: dict[tuple[str, ...], object] = {(): root}

    def container_for(path: tuple[str, ...]) -> object:
        cached = containers.get(path)
        if cached is not None:
            return cached
        leaf = leaves.get(path)
        node: object = leaf if leaf is not None else ModuleType(f"{root.__name__}.{'.'.join(path)}")
        setattr(container_for(path[:-1]), path[-1], node)
        containers[path] = node
        return node

    for path in leaves:
        container_for(path)


def _extern_error(
    function_name: str,
    message: str,
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
            fields={
                "function": TextValue(function_name),
                "python-type": TextValue(python_type),
            },
        )
    )


def _array(values: Sequence[object]) -> AglArrayView:
    """Construct the companion representation of a new AgL array."""
    return AglArrayView(ArrayValue([decode_boundary_value(value) for value in values]))


def _dict(values: dict[str, object]) -> AglDictView:
    """Construct the companion representation of a new AgL dict."""
    entries: dict[str, Value] = {}
    for key, value in values.items():
        if not isinstance(key, str):
            raise TypeError("AgL dict keys must be str")
        entries[key] = decode_boundary_value(value)
    return AglDictView(DictValue(entries))
