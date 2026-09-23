"""Extern companion loading, callable resolution, and invocation.

This eval-free runtime module owns the extern registry. It imports a
companion module once, resolves declared callables, and invokes each through
the value-directed conversion functions in :mod:`agm.agl.runtime.boundary`.
Container views remain usable after a call returns and continue to reflect
their underlying AgL containers.
"""

from __future__ import annotations

import contextvars
import dataclasses
import decimal
import functools
import importlib.machinery
import importlib.util
import marshal
import os
import sys
import threading
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import AbstractContextManager, contextmanager, nullcontext
from pathlib import Path
from types import CodeType, ModuleType
from typing import TYPE_CHECKING, Protocol, cast

from agm.agl.artifact_storage import artifact_entry, read_payload, write_payload
from agm.agl.diagnostics import AglError
from agm.agl.ir.builtin_nominals import BuiltinNominals
from agm.agl.ir.contracts import ContractRequest
from agm.agl.ir.ids import ContractId, FunctionId, NominalId
from agm.agl.ir.program import FunctionDescriptor, NominalDescriptor, ValueDescriptors
from agm.agl.modules.ids import ModuleId
from agm.agl.runtime.boundary import (
    AglArrayView,
    AglDictView,
    AglException,
    AglJson,
    BoundaryTypeError,
    BoundaryViolation,
    active_contract_encoder,
    active_descriptors,
    active_function_encoder,
    current_descriptors,
    decode_boundary_value,
    encode_boundary_value,
    synthesize_nominal_classes,
)
from agm.agl.runtime.type_contracts import TypeContract, build_type_contract
from agm.agl.self_validation import self_validation_enabled
from agm.agl.semantics.cycles import AglCyclicValue, cyclic_value_raise
from agm.agl.semantics.exceptions import AglRaise, make_builtin_exception
from agm.agl.semantics.values import (
    ArrayValue,
    ContractValue,
    DictValue,
    IrClosureValue,
    TextValue,
    Value,
)
from agm.core import fs
from agm.core.cleanup import run_cleanup_steps
from agm.util.scoping import ScopedVar
from agm.util.unicode import visible_text

if TYPE_CHECKING:
    from agm.agl.ir.ids import Location
    from agm.agl.runtime.trace import TraceStore

# These companion module attributes are APIs, never synthesized nominal aliases.
_COMPANION_API_NAMES = frozenset(
    {
        "AglException",
        "TypeContract",
        "array",
        "dict",
        "json",
        "nominals",
        "option",
        "option_none",
        "option_some",
        "runtime",
    }
)


class ExternRuntimeState:
    """Mutable companion state belonging to one interpreter instance."""

    __slots__ = ("_entries",)

    def __init__(self) -> None:
        self._entries: dict[str, tuple[object, Callable[[object], None] | None]] = {}

    def get_or_create[T](
        self, key: str, factory: Callable[[], T], close: Callable[[T], None] | None = None
    ) -> T:
        """Return this state's value for *key*, creating it (with its closer) once when absent.

        *close*, when given, is recorded beside the value on this creating
        call and later invoked with that same value by :meth:`close_all`.
        """
        if key not in self._entries:
            value = factory()
            self._entries[key] = (value, cast("Callable[[object], None] | None", close))
        return cast("T", self._entries[key][0])

    def close_all(self) -> None:
        """Close every registered value once, in reverse creation order, then clear the bag.

        Each closer receives the value it was registered for. A second call
        is a no-op. A value registered with no closer is simply dropped.
        """
        entries = list(self._entries.values())
        self._entries.clear()
        steps: list[Callable[[], None]] = [
            functools.partial(close, value)
            for value, close in reversed(entries)
            if close is not None
        ]
        run_cleanup_steps(steps)


@dataclasses.dataclass(frozen=True, slots=True)
class ActiveCall:
    """One extern call's active companion state, trace destination, module, and site.

    Bundled into a single context variable so :meth:`ExternRegistry.invoke`
    activates one object for the call rather than two, and ``runtime.state``
    and ``runtime.trace`` both read it.

    *location* is this call's own site (plain data, not yet resolved).
    *resolve_span* maps it to the attributed (span, owning module) a trace
    record uses -- the interpreter's ``_extern_trace_span``, bound once per
    interpreter and reused across every call, so activating a call never
    allocates a closure. Called only when a companion actually reads the
    span (a trace record is written, or a crossed callback needs it).
    """

    state: ExternRuntimeState
    trace_store: "TraceStore"
    module_id: ModuleId
    location: "Location | None"
    resolve_span: "Callable[[ModuleId, Location | None], tuple[Location | None, ModuleId | None]]"


class _CompanionRuntime:
    """Route companion state and tracing to the interpreter active in this call context."""

    __slots__ = ("_active_call", "_detached_state")

    def __init__(self) -> None:
        self._active_call: contextvars.ContextVar[ActiveCall | None] = contextvars.ContextVar(
            "agl_extern_active_call", default=None
        )
        # Direct companion use outside evaluation remains useful to host tests
        # and tools, but evaluation always supplies its interpreter-owned state.
        self._detached_state = ExternRuntimeState()

    def activate(self, call: ActiveCall | None) -> AbstractContextManager[None]:
        """Make *call* visible to a companion for the dynamic call extent."""
        if call is None:
            # Absent evaluator state must not shadow an enclosing activation.
            return nullcontext()
        return ScopedVar(self._active_call, call)

    def state[T](
        self, key: str, factory: Callable[[], T], close: Callable[[T], None] | None = None
    ) -> T:
        """Get companion-local state scoped to the active interpreter call.

        *close*, given on the call that creates the value, runs once when the
        owning interpreter's run ends (or, for a detached call, when
        :func:`close_detached_state` is called).
        """
        active = self._active_call.get()
        state = active.state if active is not None else self._detached_state
        return state.get_or_create(key, factory, close)

    def tracing(self) -> bool:
        """Whether :meth:`trace` would record: an active call whose store is writing.

        Lets a companion skip building a payload that tracing, off by
        default, would discard.
        """
        active = self._active_call.get()
        return active is not None and active.trace_store.path is not None

    def trace(self, kind: str, payload: dict[str, object]) -> None:
        """Emit a companion trace record tagged with the active call's origin and span.

        A silent no-op outside an active extern call (detached companion use),
        and when tracing is off. The origin (the extern's declaring module's
        display path) is computed only once the store is confirmed to be
        writing, and so is the attributed span and its owning module
        (``site``) -- resolving either walks the active call-site stack.
        """
        active = self._active_call.get()
        if active is not None and active.trace_store.path is not None:
            span, site = active.resolve_span(active.module_id, active.location)
            active.trace_store.companion_record(
                active.module_id.display(),
                kind,
                payload,
                span,
                site.display() if site is not None else None,
            )

    def active_span(self) -> "Location | None":
        """The span of the extern call active in this context, if any.

        Read by ``IrInterpreter._invoke_crossed_closure`` for a companion
        callback that is itself an extern: it has no AgL call site of its
        own, so it inherits the outer call's span instead.
        """
        active = self._active_call.get()
        if active is None:
            return None
        return active.resolve_span(active.module_id, active.location)[0]


_COMPANION_RUNTIME = _CompanionRuntime()


def active_call_span() -> "Location | None":
    """The span of the extern call active in this context, or ``None``."""
    return _COMPANION_RUNTIME.active_span()


def close_detached_state() -> None:
    """Close the fallback state bag used by a companion called outside evaluation.

    For tests and tools only: evaluation always closes its own
    interpreter-owned state at the run boundary instead.
    """
    _COMPANION_RUNTIME._detached_state.close_all()


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


def _optimize_level() -> int:
    """Return the interpreter's ``-O`` level; compiled bytecode differs per level."""
    return sys.flags.optimize


class _CompanionBytecodeLoader(importlib.machinery.SourceFileLoader):
    """A source-authoritative loader caching bytecode outside ``__pycache__``.

    CPython's source loader caches compiled bytecode in a ``__pycache__``
    directory next to the module it imports.  A companion is imported from
    wherever its AgL module lives, including an installed package's immutable
    store tree, which AGM must never write into: unrecorded files there are
    not part of the package and would strand its removal.  This loader instead
    reads and writes the AgL artifact cache, keyed by compiler digest,
    canonical path, optimization level, and the file identity stamp the
    registry already validated before choosing to import, so a stamp-changing
    edit never serves bytecode compiled from the earlier version. An entry is
    written only once that stamp is settled (:func:`agm.core.fs.identity_stamp_is_settled`).
    """

    def __init__(self, fullname: str, path: str, stamp: fs.IdentityStamp) -> None:
        super().__init__(fullname, path)
        self._stamp = stamp

    def get_code(self, fullname: str) -> CodeType:
        """Return *fullname*'s code from the bytecode cache, else compile and cache it."""
        path = Path(self.path)
        key = repr((os.fsencode(path), _optimize_level())).encode()
        entry, identity = artifact_entry(key, "pyc", validator=repr(self._stamp).encode())
        payload = read_payload(entry, identity)
        if payload is not None:
            try:
                cached = cast(object, marshal.loads(payload))
            except (ValueError, EOFError, TypeError):
                cached = None
            if isinstance(cached, CodeType):
                # The registry stamped the companion before constructing this
                # loader. Revalidate immediately before serving its cached
                # code: the file could have changed or vanished while the
                # cache payload was being read.
                try:
                    if fs.identity_stamp(path) == self._stamp:
                        return cached
                except OSError:
                    pass

        source_path = self.get_filename(fullname)
        observed_ns = time.time_ns()
        code = self.source_to_code(self.get_data(source_path), source_path)
        # Publish only if the file still has the registry's stamp and that stamp
        # predates the read by the settle window (git's racy-entry rule); a
        # vanished file just skips the write.
        try:
            current_stamp = fs.identity_stamp(path)
        except OSError:
            return code
        if current_stamp == self._stamp and fs.identity_stamp_is_settled(
            current_stamp, observed_ns
        ):
            write_payload(entry, identity, marshal.dumps(code))
        return code


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
            return encode_boundary_value(self._invoke(values), current_descriptors())
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
        self._function_by_id: dict[FunctionId, FunctionDescriptor] = {}
        # Keyed by request identity; the entry pins its request so the id stays unique.
        self._type_contracts: dict[int, tuple[ContractRequest, TypeContract]] = {}

    def set_nominals(
        self,
        descriptors: dict[NominalId, NominalDescriptor],
        *,
        functions: Mapping[FunctionId, FunctionDescriptor] | None = None,
    ) -> None:
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

        *functions* accumulates alongside *descriptors* into the same program
        descriptor view :meth:`_program_descriptors` publishes for a companion
        import (see :meth:`load_companion`), so a view a companion builds at
        import time renders correctly. Omitted by direct-registry callers that
        never need that view -- companion import without it still succeeds,
        just with no nominal/function spellings recorded yet.
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
        if functions is not None:
            self._function_by_id.update(functions)

    def _program_descriptors(self) -> ValueDescriptors:
        """The program descriptor view accumulated by :meth:`set_nominals`.

        Published for a companion import's extent by :meth:`load_companion`,
        mirroring how :meth:`invoke` publishes the same kind of view for a
        call's extent -- so a view a companion builds at either site keeps a
        real, non-empty descriptor slot rather than the detached one in
        :mod:`agm.agl.runtime.boundary`.
        """
        return ValueDescriptors(nominals=self._nominal_by_id, functions=self._function_by_id)

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
        setattr(module, "TypeContract", TypeContract)
        setattr(module, "runtime", _COMPANION_RUNTIME)
        nominals = ModuleType("agl.nominals")
        setattr(module, "nominals", nominals)
        leaves: dict[tuple[str, ...], type[object]] = {}
        for nominal, cls in self._nominal_classes.items():
            descriptor = self._nominal_by_id[nominal]
            if descriptor.bears_name_path and not descriptor.module_id.is_reserved:
                leaves[_nominal_identity_path(descriptor)] = cls
        option_cls = leaves.get(("std", "option", "Option"))
        if option_cls is not None:
            setattr(module, "option_none", functools.partial(_option_none, option_cls))
            setattr(module, "option_some", functools.partial(_option_some, option_cls))
            setattr(module, "option", functools.partial(_option, option_cls))
        _build_nominal_namespace(nominals, leaves)
        names: dict[str, list[type[object]]] = {}
        for cls in leaves.values():
            names.setdefault(cls.__name__, []).append(cls)
        for name, classes in names.items():
            if len(classes) == 1 and name not in _COMPANION_API_NAMES:
                setattr(module, name, classes[0])
        return module

    def holds_current(self, companion_path: Path) -> bool:
        """Whether :meth:`load_companion` of *companion_path* would reuse an import."""
        canonical = companion_path.resolve()
        cached = self._by_path.get(canonical)
        if cached is None:
            return False
        try:
            return cached[0] == fs.identity_stamp(canonical)
        except OSError:
            return False

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

        Runs the companion's top-level code under :meth:`_program_descriptors`,
        so a view or record it builds at module level -- e.g. ``agl.array``
        over synthesized nominal instances -- renders correctly by ``repr``
        after the import returns, the same guarantee an extern call gets from
        :meth:`invoke`.
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
            synthetic_name,
            canonical,
            loader=_CompanionBytecodeLoader(synthetic_name, str(canonical), stamp),
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
            with active_descriptors(self._program_descriptors()):
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

    def loaded_companion(self, module_id: ModuleId) -> ModuleType | None:
        """The companion module bound to *module_id*, or ``None`` when none is loaded."""
        return self._by_module.get(module_id)

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
        nominals: BuiltinNominals,
        descriptors: ValueDescriptors,
        function_encoder: Callable[[IrClosureValue], object] | None = None,
        active_call: ActiveCall | None = None,
        contracts: Mapping[ContractId, ContractRequest] | None = None,
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

        *nominals* resolves the ``ExternError``/``CyclicValueError`` nominal.
        *descriptors* resolves the nominal/function spellings this call's
        arguments render with -- an unknown-nominal message, or a view's own
        ``repr`` -- and is published for the call's extent so a companion's
        own ``agl.array``/``agl.dict``/nominal construction picks it up too;
        every view or record view this call mints keeps it in its own slot,
        so it keeps rendering correctly after the call returns. Both are
        required: a caller that invokes without a real program (e.g. a direct
        unit test) passes the shipped standard library's own identities and
        an empty descriptor view explicitly, rather than a default silently
        standing in for the wrong program.
        *active_call* bundles the evaluator-owned companion state, trace
        store, module id, and call-site span, activated together for this
        call so ``runtime.state``/``runtime.trace`` inside *fn* reach them;
        absent direct callers use detached host state and a no-op trace
        instead.
        *function_encoder* turns an AgL closure into a callable proxy and is
        published for the call's extent, so every closure a companion reaches
        -- through an argument, a retained view, or a nested container --
        encodes through the interpreter it is running under. Like
        *active_call*, it is scoped to this call's context: a thread the
        companion spawns sees it only if it runs in a copy of that context.
        *contracts* resolves the leading target contracts of a type-directed
        extern call into their cached :class:`TypeContract`, likewise scoped;
        ``None`` (any other call) shadows an enclosing call's resolver.
        """
        with (
            active_function_encoder(function_encoder),
            active_descriptors(descriptors),
            active_contract_encoder(
                None if contracts is None else functools.partial(self._type_contract, contracts)
            ),
        ):
            try:
                encoded_args = [encode_boundary_value(arg, descriptors) for arg in args]
            except BoundaryViolation as exc:
                raise _extern_error(
                    function_name,
                    f"argument cannot cross the boundary: {exc}",
                    python_type="",
                    nominals=nominals,
                ) from exc

            try:
                with (
                    _COMPANION_RUNTIME.activate(active_call),
                    decimal.localcontext(),
                ):
                    result = fn(*encoded_args)
            except AglException as exc:
                raise AglRaise(exc.value) from exc
            except AglCyclicValue as exc:
                raise cyclic_value_raise(nominals=nominals) from exc
            except Exception as exc:
                try:
                    message = visible_text(str(exc)) or type(exc).__name__
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

    def _type_contract(
        self, contracts: Mapping[ContractId, ContractRequest], value: ContractValue
    ) -> TypeContract:
        """Return *value*'s ``TypeContract``, built once per request alongside the classes."""
        request = contracts[value.contract_id]
        cached = self._type_contracts.get(id(request))
        if cached is None:
            cached = (request, build_type_contract(request, self._nominal_classes))
            self._type_contracts[id(request)] = cached
        return cached[1]


def _nominal_identity_path(descriptor: NominalDescriptor) -> tuple[str, ...]:
    """Return a nominal's namespace path, rooted by module then scope.

    A module with no path identity of its own roots at ``entry`` instead.

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
    """Construct the companion representation of a new AgL array.

    Bound as ``agl.array`` for the duration of a companion import (see
    ``ExternRegistry._agl_module``); a companion calls this with no
    descriptors of its own, so the new view's slot takes the active extern
    call's descriptors (or the empty detached view outside any call).
    """
    return AglArrayView(
        ArrayValue([decode_boundary_value(value) for value in values]), current_descriptors()
    )


def _dict(values: dict[str, object]) -> AglDictView:
    """Construct the companion representation of a new AgL dict.

    Bound as ``agl.dict``; see :func:`_array` for where its descriptors come from.
    """
    entries: dict[str, Value] = {}
    for key, value in values.items():
        if not isinstance(key, str):
            raise TypeError("AgL dict keys must be str")
        entries[key] = decode_boundary_value(value)
    return AglDictView(DictValue(entries), current_descriptors())


def _option_none(option_cls: type) -> object:
    """Build ``Option::None`` in the standard-library ``Option`` class.

    Bound as ``agl.option_none`` (see ``ExternRegistry._agl_module``): the one
    shared home for a pattern every stdlib companion otherwise repeats locally.
    """
    return cast(object, getattr(option_cls, "None")())


def _option_some(option_cls: type, value: object) -> object:
    """Build ``Option::Some(value)`` in the standard-library ``Option`` class.

    Bound as ``agl.option_some``; see :func:`_option_none`.
    """
    return cast(object, getattr(option_cls, "Some")(value=value))


def _option(option_cls: type, value: object, present: bool) -> object:
    """Build ``Option::Some(value)`` when *present*, else ``Option::None``.

    Bound as ``agl.option``, for a companion that has already computed both
    the value and whether it exists. *value* is evaluated by the caller, so a
    site whose value is only well-defined when present uses a conditional
    over :func:`_option_some`/:func:`_option_none` instead.
    """
    return _option_some(option_cls, value) if present else _option_none(option_cls)
