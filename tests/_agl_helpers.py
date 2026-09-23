"""Shared helpers for AgL test modules.

Provides a recursive ``node_id`` collector (``all_node_ids``) used by the
seeded parsing and seeded type-checking tests, and ``let_root_capture`` for
extracting the binding IR from an immutable ``let``.

``type_table_for`` is the shared helper for tests that build ad-hoc
``RecordType``/``EnumType`` handles directly (rather than through the real
type builder, which populates the ``TypeTable``): since a handle carries no
field/variant data of its own, it registers every given ``TypeDef`` — one per
ad-hoc nominal type the test constructs, including any nested inside another
one's field/variant templates — into a fresh seeded table, so
``derive_schema``/``build_decode_schema``/``compile_coercion``/etc. resolve
field and variant shapes exactly as specified.

``record_type``/``enum_type`` are convenience factories that build an ad-hoc
handle and its matching ``TypeDef`` together, in one call, for tests that
need both (the handle to pass to the function under test, the ``TypeDef`` to
pass to ``type_table_for``). Each call stamps a fresh, distinct declaration
identity (``next_decl_id``) onto the ``TypeDef`` and derives the returned
handle's ``decl_id`` from that same identity (via ``TypeDef.handle``), so the
pair always names the same declaration and ``TypeTable.register`` (which
requires an identity) always accepts it. A self-referential or
mutually-recursive ad-hoc type reserves its identity up front (``next_decl_id``)
and passes it back in as *decl_id*, so a field embedded in its own body can
name it before the ``TypeDef``/handle pair exists.

``strip_decl_ids`` erases every embedded nominal handle's ``decl_id`` back to
``NO_DECL_ID``, for a test asserting an actual checked/looked-up type's SHAPE
against a hand-written literal that has no way to know the real declaration
identity — most equality assertions against a literal, since identity
participates in ``RecordType``/``EnumType``/``ExceptionType`` equality.

``agent_value`` builds a runtime ``std/prelude::Agent`` enum value directly, for
tests that need one as an expected value, a seeded host setting, or a request
payload without going through source parsing.

``derive_schema``/``build_decode_schema`` return one half of
``derive_schema_and_decode``, the derivation production runs.

``dummy_span`` returns a fixed placeholder ``SourceSpan`` for tests that must
supply one but don't assert on its content.

``program_config_engine_seeds`` partitions a preflighted entry's evaluated
``@config`` into its engine-setting seeds, mirroring the production split
``agm.commands.exec_program`` makes; ``run_inline_command`` uses it so an
inline scenario's ``@config`` engine settings reach the interpreter the same
way the real host applies them.

``run_program``/``shapes_match``/``assert_shape`` and the REPL helpers
``eval_ok``/``read_config_result``/``repl_session``/``unopened_repl_session``/
``record_variant`` are shared by the engine-setting test suites
(``default-agent``, ``default-sandbox``, the generic restamp tests, the REPL
builtin-settings tests): running inline source or a REPL entry, and comparing
or reading back a resulting ``std/config`` engine-setting value structurally
regardless of which nominal table stamped it.
"""

from __future__ import annotations

import dataclasses
import itertools
import os
import stat
from collections.abc import Callable, Mapping
from pathlib import Path

from agm.agl import PipelineDriver
from agm.agl.capabilities import HostCapabilities
from agm.agl.ir.builtin_nominals import NO_BUILTIN_DECLARATIONS
from agm.agl.ir.builtin_vars import is_engine_builtin_var_key
from agm.agl.ir.contracts import DecodePlan
from agm.agl.ir.ids import NominalId
from agm.agl.ir.nodes import IrBind, IrExpr, IrSequence
from agm.agl.ir.program import NominalDescriptor, NominalKind, ValueDescriptors, VariantDescriptor
from agm.agl.ir.reserved_nominals import NO_DECL_ID, require_reserved_nominal_id
from agm.agl.ir.static_keys import StaticBindingKey
from agm.agl.modules.ids import ENTRY_ID, ModuleId
from agm.agl.modules.loader import ModuleGraph
from agm.agl.modules.roots import RootSet
from agm.agl.pipeline import ArgumentPreflight, PreparedProgram, ProgramDiscovery, RunResult
from agm.agl.repl import EntryResult, ReplSession
from agm.agl.runtime.arguments import ProgramArguments
from agm.agl.runtime.engine_config import restamp_engine_setting
from agm.agl.runtime.types import ProgramDeclInfo
from agm.agl.scope.program import resolve_program
from agm.agl.semantics.type_table import (
    BUILTIN_PRELUDE_MEMBER_TYPE_DEFS,
    TypeDef,
    TypeTable,
    create_seeded_type_table,
)
from agm.agl.semantics.types import (
    EnumType,
    ExceptionType,
    RecordType,
    Type,
    TypeVarType,
    free_type_vars,
    transform_type,
)
from agm.agl.semantics.values import BoolValue, RecordValue, TextValue, Value
from agm.agl.syntax import (
    AssignStmt,
    Block,
    BuiltinVarDecl,
    Call,
    EnumDef,
    ExceptionDef,
    ExportDecl,
    FuncDef,
    ImportDecl,
    InfixDecl,
    Item,
    LetDecl,
    RecordDef,
    ScopeRegion,
    TypeAlias,
    UseDecl,
    VarDecl,
)
from agm.agl.syntax.spans import UNKNOWN_SOURCE, SourceSpan
from agm.agl.type_schema import derive_schema_and_decode
from agm.agl.typecheck.env import CheckedModule
from agm.agl.typecheck.program import CheckedProgram, check_program
from agm.agl.zones import ParamZone
from agm.config.context import ConfigContext
from agm.sandbox.prepare import SandboxContext, lazy_sandbox_context

# Declaration identities for ad-hoc test TypeDefs, distinct from real AST node
# ids (which start at 0) and from every reserved identity (<= -2, see
# ir.reserved_nominals) so an ad-hoc type never collides with either.
_decl_ids = itertools.count(900_000)


def derive_schema(typ: Type, type_table: TypeTable) -> dict[str, object]:
    """Return *typ*'s JSON Schema."""
    return derive_schema_and_decode(typ, type_table)[0]


def build_decode_schema(typ: Type, type_table: TypeTable) -> DecodePlan:
    """Return *typ*'s decode plan."""
    return derive_schema_and_decode(typ, type_table)[1]


def dummy_span() -> SourceSpan:
    """Return a fixed placeholder span for tests that need one but don't inspect it."""
    return SourceSpan(1, 1, 1, 1, 0, 0, UNKNOWN_SOURCE)


#: Capability catalog for AgL program/module typechecking tests: shell exec
#: plus every codec kind a member/enum/exception method-resolution test needs
#: to check.
AGL_TEST_CAPS = HostCapabilities(
    supports_shell_exec=True,
    codec_kinds={
        "text": frozenset({"text"}),
        "json": frozenset({"json", "record", "enum", "array", "dict", "int", "decimal", "bool"}),
    },
)


def check_agl_graph(graph: ModuleGraph) -> CheckedProgram:
    """Resolve and typecheck a loaded multi-module graph with :data:`AGL_TEST_CAPS`."""
    return check_program(resolve_program(graph), AGL_TEST_CAPS)


def check_agl_program(
    tmp_path: Path, modules: dict[str, str], *, default_stdlib: bool = True
) -> CheckedProgram:
    """Build and typecheck a multi-module graph with :data:`AGL_TEST_CAPS`."""
    from tests.agl.ir_harness import make_graph_from_files

    graph = make_graph_from_files(tmp_path, modules, default_stdlib=default_stdlib)
    return check_agl_graph(graph)


def checked_module_items(module: CheckedModule) -> tuple[Item, ...]:
    """Return an entry's test-only inline items, unwrapping the synthetic ``main``."""
    items = module.resolved.program.body.items
    if items and isinstance(items[-1], FuncDef) and items[-1].is_synthetic:
        return items[-1].body.items if isinstance(items[-1].body, Block) else (items[-1].body,)
    return items


def final_entry_call(checked: CheckedProgram) -> Call:
    """The entry module's final top-level statement, asserted to be a call expression."""
    call = checked_module_items(checked.modules[ENTRY_ID])[-1]
    assert isinstance(call, Call), f"expected the entry to end in a call, got {call!r}"
    return call


def checked_program_selection_key(checked: CheckedProgram) -> tuple[object, ...]:
    """The declaration identity (module, scope path, name) the entry's final call selected."""
    call = final_entry_call(checked)
    module = checked.modules[ENTRY_ID]
    selection = module.method_selections.get(call.callee.node_id)
    assert selection is not None, "expected the final call to select a method, not a field"
    return selection.declaration_key


def file_program(source: str) -> str:
    """Return file-style source with root statements in an explicit ``main``.

    Test fixtures that model an ``agm exec FILE`` invocation use this helper
    for legacy statement snippets. It keeps module declarations at the static
    root and emits a real ``program def main`` for executable items; an
    already explicit program (or invalid syntax that a test must reject) is
    left unchanged.
    """
    try:
        program = PipelineDriver.parse_entry(source, entry_path=None).program
    except Exception:
        return source
    if program is None or any(
        isinstance(item, FuncDef) and item.is_program for item in program.body.items
    ):
        return source
    static = (
        FuncDef,
        RecordDef,
        EnumDef,
        ExceptionDef,
        TypeAlias,
        BuiltinVarDecl,
        InfixDecl,
        ImportDecl,
        ExportDecl,
        UseDecl,
        ScopeRegion,
    )
    root_items = [item for item in program.body.items if isinstance(item, static)]
    body_items = [item for item in program.body.items if not isinstance(item, static)]
    if not body_items:
        return source

    def text(item: object) -> str:
        span = getattr(item, "span")
        return source[span.start_offset : span.end_offset]

    root = "\n".join(text(item) for item in root_items)
    body_lines = [
        text(item) if isinstance(item, (LetDecl, VarDecl, AssignStmt)) else f"let _ = {text(item)}"
        for item in body_items
    ]
    body = "\n".join(body_lines)
    indented = "\n".join(f"  {line}" if line else line for line in body.splitlines())
    return f"{root + chr(10) if root else ''}program def main() -> unit =\n{indented}\n"


def write_file_program(path: Path, source: str, **kwargs: str) -> None:
    """Write a legacy executable snippet as an explicit file program."""
    path.write_text(file_program(source), **kwargs)


def prepare_inline_command(
    source: str,
    *,
    entry_path: Path | None = None,
    roots: RootSet | None = None,
    default_stdlib: bool = True,
) -> PreparedProgram:
    """Prepare test-only inline source with the ``agm exec -c`` entry transform.

    ``entry_path`` is ``None`` for real inline sources; the corpus passes a path
    for programs whose builtins need a file-backed anchor (``resource``).
    """
    parsed = PipelineDriver.parse_entry(source, entry_path=entry_path, inline_command=True)
    return PipelineDriver.prepare_parsed_entry(
        parsed,
        roots=roots,
        default_stdlib=default_stdlib,
    )


def program_config_engine_seeds(
    argument_preflight: ArgumentPreflight,
) -> dict[StaticBindingKey, Value]:
    """Partition a preflighted entry's ``@config`` into its engine-setting seeds.

    Mirrors ``agm.commands.exec_program``'s own split of
    ``ArgumentPreflight.program_config`` by ``is_engine_builtin_var_key``,
    restamped onto the executable's own nominal identity, so a scenario
    harness feeds an engine setting from ``@config`` (``std/config::trace``,
    ``timeout``, ...) the same way the real ``agm exec`` host does. A module
    parameter's own ``@config`` entries need no such seam: preflight already
    folds them into ``param_seeds``.
    """
    executable = argument_preflight.executable
    assert executable is not None
    return {
        key: restamp_engine_setting(
            key[2],
            value,
            from_table=executable.builtin_nominals,
            to_table=NO_BUILTIN_DECLARATIONS,
        )
        for key, value in argument_preflight.program_config.items()
        if is_engine_builtin_var_key(key)
    }


def run_inline_command(
    runtime: PipelineDriver,
    source: str,
    *,
    roots: RootSet | None = None,
    default_stdlib: bool = True,
    entry_path: Path | None = None,
    **run_kwargs: object,
) -> RunResult:
    """Run test-only inline source through the same entry transform as ``agm exec -c``.

    Routes through :meth:`PipelineDriver.preflight_arguments` for an entry's
    value arguments or its scenario module parameters, and otherwise through
    plain default-program selection.
    """
    prepared = prepare_inline_command(
        source,
        entry_path=entry_path,
        roots=roots,
        default_stdlib=default_stdlib,
    )
    param_values = run_kwargs.pop("param_values", None)
    positional = run_kwargs.pop("positional", None)
    module_params = run_kwargs.pop("module_params", None)
    discovery = runtime.discover_programs(prepared)
    if discovery.compiled is None:
        return runtime.run_prepared(prepared, **run_kwargs)
    entry_programs = [program for program in discovery.programs if program.module.is_entry]
    assert len(entry_programs) <= 1
    entry_program = entry_programs[0] if entry_programs else None

    if entry_program is None:
        assert not param_values and not positional, (
            "scenario supplied 'param_values'/'positional' but the entry program "
            "declares no value parameters to receive them - check the fixture's "
            "program signature"
        )
        assert module_params is None, "scenario supplied module parameters without an entry program"
        return runtime.run_prepared(
            prepared, compiled=discovery.compiled, select_default_program=True, **run_kwargs
        )

    if not entry_program.parameters and module_params is None:
        assert not param_values and not positional, (
            "scenario supplied 'param_values'/'positional' but the entry program "
            "declares no value parameters to receive them - check the fixture's "
            "program signature"
        )
        return runtime.run_prepared(
            prepared, compiled=discovery.compiled, select_default_program=True, **run_kwargs
        )

    argument_preflight = runtime.preflight_arguments(
        prepared,
        entry_program,
        ProgramArguments(
            positional=tuple(positional) if positional else (),
            named=dict(param_values) if param_values else {},
        ),
        compiled=discovery.compiled,
        param_values=module_param_values(discovery, entry_program, module_params),
    )
    if not argument_preflight.result.ok:
        return argument_preflight.result
    assert argument_preflight.executable is not None
    engine_seeds = program_config_engine_seeds(argument_preflight)
    supplied_seeds = run_kwargs.pop("builtin_var_seeds", None)
    if supplied_seeds is not None:
        assert isinstance(supplied_seeds, Mapping)
        engine_seeds.update(supplied_seeds)
    return runtime.run_prepared(
        prepared,
        compiled=discovery.compiled,
        executable=argument_preflight.executable,
        program_symbol=argument_preflight.executable.program_symbols[entry_program.node_id],
        arguments=argument_preflight.arguments,
        param_seeds=argument_preflight.param_seeds,
        builtin_var_seeds=engine_seeds or None,
        **run_kwargs,
    )


def module_param_values(
    discovery: ProgramDiscovery,
    program: ProgramDeclInfo,
    declaration_values: Mapping[str, object] | None,
) -> dict[StaticBindingKey, object]:
    """Translate scenario declaration paths into the selected program's parameter keys."""
    params = tuple(discovery.params_for(program))
    params_by_path = {param.declaration_path: param.key for param in params}
    assert len(params_by_path) == len(params), "fixture has duplicate module parameter paths"

    values: dict[StaticBindingKey, object] = {}
    for declaration_path, value in (declaration_values or {}).items():
        key = params_by_path.get(declaration_path)
        assert key is not None, (
            f"scenario supplied unknown module parameter {declaration_path!r}; "
            "check the fixture declaration path"
        )
        values[key] = value
    return values


def next_decl_id() -> int:
    """Return a fresh declaration identity, distinct across the whole test run."""
    return next(_decl_ids)


def let_root_capture(initializer: IrExpr) -> IrBind:
    """Return the binding IR for an immutable let."""
    if isinstance(initializer, IrBind):
        return initializer
    assert isinstance(initializer, IrSequence)
    root_capture = initializer.items[0]
    assert isinstance(root_capture, IrBind)
    return root_capture


def all_node_ids(obj: object, seen: set[int] | None = None) -> set[int]:
    """Recursively collect every ``node_id`` reachable from *obj*."""
    if seen is None:
        seen = set()
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        nid = getattr(obj, "node_id", None)
        if isinstance(nid, int):
            seen.add(nid)
        for f in dataclasses.fields(obj):
            all_node_ids(getattr(obj, f.name), seen)
    elif isinstance(obj, (tuple, list)):
        for item in obj:
            all_node_ids(item, seen)
    return seen


def strip_decl_ids(t: Type) -> Type:
    """Return *t* with every embedded nominal handle's ``decl_id`` reset to ``NO_DECL_ID``.

    ``RecordType``/``EnumType``/``ExceptionType`` equality includes
    ``decl_id`` (see ``semantics.types``), so an assertion comparing a real
    checked/looked-up type against a hand-written expected literal — which has
    no way to know the real declaration identity — needs both sides reduced to
    the identity-independent shape this compares: name, module, scope path,
    and (recursively, since a type argument may itself nest a nominal
    reference) type arguments.
    """

    def _strip(node: Type) -> Type:
        if isinstance(node, (RecordType, EnumType, ExceptionType)):
            return dataclasses.replace(node, decl_id=NO_DECL_ID)
        return node

    return transform_type(t, _strip)


_ENUM_MEMBER_DEFS: dict[int, tuple[TypeDef, ...]] = {}


def type_table_for(*defs: TypeDef) -> TypeTable:
    """Return a fresh seeded ``TypeTable`` with every given ``TypeDef`` registered.

    Callers building an ad-hoc ``RecordType``/``EnumType`` handle for a test
    (rather than through the real type builder) pass its ``TypeDef`` here —
    including the ``TypeDef`` of any OTHER ad-hoc type nested inside a
    field/variant template (e.g. an outer record embedding an inner one) —
    since a handle carries no shape data of its own for this helper to
    discover automatically.
    """
    table = create_seeded_type_table()
    for typedef in defs:
        register_typedef(table, typedef)
    return table


def register_typedef(table: TypeTable, typedef: TypeDef) -> TypeDef:
    """Register *typedef*, preceded by the member records of an ``enum_typedef``."""
    for member_def in _ENUM_MEMBER_DEFS.get(typedef.decl_node_id, ()):
        table.register(member_def)
    table.register(typedef)
    return typedef


def enum_typedef(
    name: str,
    variants: Mapping[str, Mapping[str, Type]],
    *,
    module_id: ModuleId = ENTRY_ID,
    type_params: tuple[str, ...] = (),
    decl_id: int | None = None,
) -> TypeDef:
    """Build an enum ``TypeDef`` and the member record ``TypeDef``s it names.

    An enum declaration names record declarations, so a hand-built fixture
    needs both. The member declarations are remembered against the enum's
    identity so :func:`type_table_for` and :func:`register_typedef` register
    them alongside it.
    """
    enum_decl_id = next_decl_id() if decl_id is None else decl_id
    member_defs = tuple(
        TypeDef(
            kind="record",
            name=member_name,
            module_id=module_id,
            scope_path=(name,),
            type_params=tuple(
                param
                for param in type_params
                if any(param in free_type_vars(field_type) for field_type in fields.values())
            ),
            fields=tuple(fields.items()),
            field_kinds=(ParamZone.STANDARD,) * len(fields),
            decl_node_id=next_decl_id(),
        )
        for member_name, fields in variants.items()
    )
    _ENUM_MEMBER_DEFS[enum_decl_id] = member_defs
    return TypeDef(
        kind="enum",
        name=name,
        module_id=module_id,
        type_params=type_params,
        members=tuple(
            RecordType(
                name=member.name,
                type_args=tuple(TypeVarType(param) for param in member.type_params),
                module_id=module_id,
                scope_path=(name,),
                decl_id=member.decl_node_id,
            )
            for member in member_defs
        ),
        decl_node_id=enum_decl_id,
    )


def record_type(
    name: str,
    fields: dict[str, Type],
    *,
    type_args: tuple[Type, ...] = (),
    module_id: ModuleId = ENTRY_ID,
    type_params: tuple[str, ...] = (),
    decl_id: int | None = None,
    field_has_default: tuple[bool, ...] | None = None,
) -> tuple[RecordType, TypeDef]:
    """Build an ad-hoc ``RecordType`` handle and its matching ``TypeDef`` together.

    Returns ``(handle, typedef)``; pass ``typedef`` (and the ``TypeDef`` of
    any nested ad-hoc nominal type referenced in *fields*) to
    :func:`type_table_for` so the handle's field shape resolves.

    *decl_id* defaults to a freshly reserved identity; pass one explicitly
    (from :func:`next_decl_id`) for a SELF-referential or mutually-recursive
    ad-hoc type, whose own *fields* must embed a reference carrying this same
    identity before the ``TypeDef``/handle pair exists to read it off of.

    *field_has_default*, one flag per *fields* entry in order, marks which
    fields declare a constant default (omitted: none do) -- the presence
    flag a schema/decode-plan derivation reads (``TypeTable.field_has_default``).
    A default's own VALUE is never carried here; it is resolved only at
    decode time, against a real ``NominalDescriptor`` (see
    ``runtime.convert.decode_value``'s ``default_resolver``).
    """
    typedef = TypeDef(
        kind="record",
        name=name,
        module_id=module_id,
        type_params=type_params,
        fields=tuple(fields.items()),
        field_kinds=(ParamZone.STANDARD,) * len(fields),
        decl_node_id=next_decl_id() if decl_id is None else decl_id,
        field_has_default=field_has_default,
    )
    return typedef.handle(type_args), typedef


def enum_type(
    name: str,
    variants: dict[str, dict[str, Type]],
    *,
    type_args: tuple[Type, ...] = (),
    module_id: ModuleId = ENTRY_ID,
    type_params: tuple[str, ...] = (),
    decl_id: int | None = None,
) -> tuple[EnumType, TypeDef]:
    """Build an ad-hoc ``EnumType`` handle and its matching ``TypeDef`` together.

    See :func:`record_type`.
    """
    typedef = enum_typedef(
        name, variants, module_id=module_id, type_params=type_params, decl_id=decl_id
    )
    return typedef.handle(type_args), typedef


def option_nominal_descriptors(
    option: NominalId, none: NominalId, some: NominalId
) -> dict[NominalId, NominalDescriptor]:
    """Build the ``Option`` enum and its two member-record descriptors for test FFI images."""
    module_id = ModuleId(("std", "option"))
    return {
        option: NominalDescriptor(
            nominal=option,
            module_id=module_id,
            scope_path=(),
            declared_name="Option",
            kind=NominalKind.ENUM,
            variants=(
                VariantDescriptor("Some", ("value",), some),
                VariantDescriptor("None", (), none),
            ),
        ),
        none: NominalDescriptor(
            nominal=none,
            module_id=module_id,
            scope_path=("Option",),
            declared_name="None",
            kind=NominalKind.RECORD,
        ),
        some: NominalDescriptor(
            nominal=some,
            module_id=module_id,
            scope_path=("Option",),
            declared_name="Some",
            kind=NominalKind.RECORD,
            fields=("value",),
        ),
    }


def agent_value(variant: str, **fields: str) -> RecordValue:
    """Build the runtime ``std/prelude::Agent`` enum value for *variant*.

    Each keyword becomes a text-valued field, matching every ``Agent``
    variant's payload shape (``command``, ``model``/``thinking``, etc.); pass
    none for a variant with no payload.
    """
    member_nominal = next(
        (
            NominalId(member.decl_node_id)
            for member in BUILTIN_PRELUDE_MEMBER_TYPE_DEFS.values()
            if member.scope_path == ("Agent",) and member.name == variant
        ),
        NominalId(require_reserved_nominal_id("Agent")),
    )
    return RecordValue(
        nominal=member_nominal,
        fields={name: TextValue(value) for name, value in fields.items()},
    )


def hermetic_config_context() -> ConfigContext:
    """A ``ConfigContext`` scoped to this test's isolated, empty home.

    Reads ``HOME`` from the environment: the autouse ``isolate_host_environment``
    fixture (``conftest.py``) points it at a fresh per-test directory before every
    test runs, so a ``value_driven_agent_factory`` built from this context resolves
    no ``[run.*]`` config and no project directory, regardless of the machine or
    which other tests ran.
    """
    home = Path(os.environ["HOME"])
    return ConfigContext(home=home, proj_dir=None, cwd=home)


def unavailable_sandbox_context() -> SandboxContext:
    """A ``get_sandbox_context`` stand-in for a session backend test that never
    dispatches a sandboxed call.

    Every session test that leaves ``permission_mode``/``sandbox`` at their
    ``SessionOpenRequest``/``SessionAskRequest`` defaults never reaches this
    (``sandbox_run_for`` only calls its ``get_context`` argument when a real
    ``SandboxLimits`` is present), so raising here catches a test that
    silently started exercising sandboxing without a real context.
    """
    raise AssertionError("sandbox context requested unexpectedly")


def session_sandbox_context(home: Path) -> Callable[[], SandboxContext]:
    """A real, lazily-built ``get_sandbox_context`` scoped to *home*.

    For a session backend test that does exercise sandboxing: pair with
    ``write_sandbox_home(home, ...)`` for a resolvable default settings file.
    """
    return lazy_sandbox_context(ConfigContext(home=home, proj_dir=None, cwd=home))


def write_sandbox_home(
    home: Path, *, run_toml: str = "", extra_settings_files: tuple[str, ...] = ()
) -> None:
    """Set up *home* as an AGM home with a default sandbox settings candidate.

    Writes ``[home]/.agm/sandbox/default.json`` so sandbox preparation
    succeeds, *run_toml* as ``[home]/.agm/config.toml`` when given, and an
    empty-object settings file named after each entry in
    *extra_settings_files* (e.g. ``"not-claude"`` for a profile-name probe
    that must never be reached).
    """
    sandbox_dir = home / ".agm" / "sandbox"
    sandbox_dir.mkdir(parents=True, exist_ok=True)
    (sandbox_dir / "default.json").write_text("{}", encoding="utf-8")
    for name in extra_settings_files:
        (sandbox_dir / f"{name}.json").write_text("{}", encoding="utf-8")
    if run_toml:
        (home / ".agm" / "config.toml").write_text(run_toml, encoding="utf-8")


def write_transparent_sandbox_shims(directory: Path, *, log_dir: Path) -> None:
    """Write silent, flag-skipping ``systemd-run``/``srt`` fakes into *directory*.

    Each exec's onward to the command it wraps after touching a marker file
    under *log_dir* first, so a test can confirm the wrap chain actually ran,
    not merely that the wrapped command happened to start anyway. Unlike
    ``TestSandbox``'s diagnostic fakes (which print captured settings/command
    for ``agm run`` assertions), these run silently, so a sandboxed call's
    real stdout stays exactly what the wrapped command printed. The caller
    adds *directory* to the front of ``PATH`` itself.
    """
    directory.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    systemd_run = directory / "systemd-run"
    systemd_run.write_text(
        "#!/bin/bash\n"
        f'touch "{log_dir}/systemd-run"\n'
        "while [[ $# -gt 0 ]]; do\n"
        '  case "$1" in\n'
        "    --user|--scope|-q) shift ;;\n"
        "    -p|--unit) shift 2 ;;\n"
        '    --) shift; exec "$@" ;;\n'
        "    *) shift ;;\n"
        "  esac\n"
        "done\n"
    )
    systemd_run.chmod(systemd_run.stat().st_mode | stat.S_IEXEC)
    srt = directory / "srt"
    srt.write_text(
        "#!/bin/bash\n"
        f'touch "{log_dir}/srt"\n'
        "while [[ $# -gt 0 ]]; do\n"
        '  case "$1" in\n'
        "    --settings) shift 2 ;;\n"
        '    --) shift; exec "$@" ;;\n'
        "    *) shift ;;\n"
        "  esac\n"
        "done\n"
    )
    srt.chmod(srt.stat().st_mode | stat.S_IEXEC)


REPO_STDLIB_ROOT = Path(__file__).resolve().parents[1] / "packages" / "stdlib"


def agl_roots(*paths: Path, include_stdlib: bool = True) -> RootSet:
    """Assemble a search-root set the way a real invocation does.

    Every production caller assembles roots through
    :func:`~agm.agl.modules.roots.assemble_roots`, which designates the
    standard library's path in ``stdlib_roots``.  That designation both mounts
    the library's module tree under ``std`` and tells the loader and the
    process-global caches which modules are library code, so a test that
    merely *searches* the repository standard library without designating it
    compiles a configuration production never runs.
    """
    return RootSet(
        roots=frozenset(paths),
        stdlib_roots=frozenset({REPO_STDLIB_ROOT}) if include_stdlib else frozenset(),
    )


def agl_std_package_roots(*paths: Path) -> RootSet:
    """Assemble roots that also mount the repository standard library as a package.

    A real invocation on a standard-library file mounts the ``std`` package
    that owns it, so the file is compiled under the module id its manifest
    declares rather than anonymously.  A test that reaches such a file without
    mounting its package compiles a configuration production never runs.
    """
    from agm.agl.modules.roots import assemble_roots
    from agm.packages.manifest import load_manifest
    from agm.packages.model import PackageInfo

    std_package = PackageInfo(REPO_STDLIB_ROOT, load_manifest(REPO_STDLIB_ROOT / "package.toml"))
    return assemble_roots(
        invocation_root=None,
        stdlib_root=REPO_STDLIB_ROOT,
        lib_root=None,
        configured=[],
        cli=(str(path) for path in paths),
        cwd=REPO_STDLIB_ROOT,
        package_roots=(std_package,),
    )


def run_program(
    source: str,
    *,
    seed: "dict[str, Value] | None" = None,
    host_settings_policy: object | None = None,
) -> RunResult:
    """Run inline *source* through the shared inline-entry transform.

    Asserts the result is a full :class:`RunResult` (not an argument-preflight
    failure). Shared by the engine-setting test suites (``default-agent``,
    ``default-sandbox``, the generic restamp tests).
    """
    result = run_inline_command(
        PipelineDriver(),
        source,
        roots=agl_roots(),
        builtin_host_settings=seed,
        host_settings_policy=host_settings_policy,
    )
    assert isinstance(result, RunResult)
    return result


def shapes_match(actual: Value, expected: Value) -> bool:
    """Compare two values structurally, ignoring ``RecordValue`` nominal identity.

    A running program's own nominal identity for a builtin/reserved type
    differs from the reserved-fallback identity a host-built expected value
    carries (see :func:`assert_shape`); a nested field (e.g. ``Sandbox``'s
    ``Optional``/``Option``-valued fields) carries its own such identity too.
    Recurses through ``RecordValue`` fields; every other value kind compares
    by its own ``==``.
    """
    if isinstance(expected, RecordValue):
        return (
            isinstance(actual, RecordValue)
            and actual.fields.keys() == expected.fields.keys()
            and all(shapes_match(actual.fields[k], v) for k, v in expected.fields.items())
        )
    return actual == expected


def assert_shape(actual: Value, is_variant: Value, expected: RecordValue) -> None:
    """Verify *actual* is the expected enum member/record with the expected payload.

    *is_variant* is an ``is`` member test run inside the program's own
    source (bound alongside *actual*): the running program loads real stdlib,
    so its own nominal enum/record carries that program's own nominal
    identity, distinct from the reserved-fallback identity *expected* (built
    by a test helper such as ``agent_value``) carries. Only a cast evaluated
    inside that same program can compare identity correctly; fields compare
    structurally instead, via :func:`shapes_match`, since a nested field may
    itself carry a host-known identity (e.g. ``Sandbox``'s ``Optional``/
    ``Option``-valued fields).
    """
    assert is_variant == BoolValue(True)
    assert shapes_match(actual, expected)


def unopened_repl_session(**kwargs: object) -> ReplSession:
    """Build a session over the repository standard library, left unopened.

    For a test that must observe the initial ``std/config`` load from the
    entry that triggers it.
    """
    kwargs.setdefault("stdlib_root", REPO_STDLIB_ROOT)
    return ReplSession(**kwargs)


def repl_session(**kwargs: object) -> ReplSession:
    """Build a session over the repository standard library and open it.

    ``agm.commands.repl`` opens a session before accepting an entry, which
    loads and type-checks the initial library image; going through
    :meth:`ReplSession.open` here exercises that same startup and lets the
    session reuse the process-wide bootstrap image instead of re-checking the
    standard library once per test.
    """
    session = unopened_repl_session(**kwargs)
    session.open()
    return session


def eval_ok(session: ReplSession, text: str) -> EntryResult:
    """Evaluate *text* as a REPL entry, asserting it succeeded."""
    result = session.eval_entry(text)
    assert result.ok, f"entry {text!r} failed: {result.diagnostics} {result.error}"
    return result


def read_config_result(session: ReplSession, key: str) -> EntryResult:
    """Import-and-read ``std/config::key``, returning the full result (value + descriptors)."""
    result = eval_ok(session, f"std/config::{key}")
    assert result.value is not None
    return result


def record_variant(value: Value, descriptors: ValueDescriptors) -> str:
    """Return the terminal member name a ``RecordValue``'s nominal resolves to.

    A ``RecordValue`` carries only its opaque ``NominalId``; its scoped
    display spelling comes from the entry's own descriptor table.
    """
    assert isinstance(value, RecordValue)
    return descriptors.nominals[value.nominal].display_name.rsplit("::", maxsplit=1)[-1]
