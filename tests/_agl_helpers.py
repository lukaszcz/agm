"""Shared helpers for AgL test modules.

Provides a recursive ``node_id`` collector (``all_node_ids``) used by the
seeded parsing and seeded type-checking tests, and ``let_root_capture`` for
extracting the binding IR from a simple or destructuring immutable ``let``.

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

``dummy_span`` returns a fixed placeholder ``SourceSpan`` for tests that must
supply one but don't assert on its content.
"""

from __future__ import annotations

import dataclasses
import itertools
from collections.abc import Mapping
from pathlib import Path

from agm.agl import PipelineDriver
from agm.agl.ir.ids import NominalId
from agm.agl.ir.nodes import IrBind, IrExpr, IrSequence
from agm.agl.ir.program import NominalDescriptor, NominalKind, VariantDescriptor
from agm.agl.ir.reserved_nominals import NO_DECL_ID, require_reserved_nominal_id
from agm.agl.modules.ids import ENTRY_ID, ModuleId
from agm.agl.modules.roots import RootSet
from agm.agl.parser import parse_program_seeded, wrap_inline_program
from agm.agl.pipeline import PreparedProgram, RunResult
from agm.agl.runtime.arguments import ProgramArguments
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
from agm.agl.semantics.values import RecordValue, TextValue
from agm.agl.syntax import (
    AssignStmt,
    BuiltinVarDecl,
    EnumDef,
    ExceptionDef,
    ExportDecl,
    FuncDef,
    ImportDecl,
    InfixDecl,
    LetDecl,
    RecordDef,
    ScopeRegion,
    TypeAlias,
    UseDecl,
    VarDecl,
)
from agm.agl.syntax.nodes import Program
from agm.agl.syntax.spans import UNKNOWN_SOURCE, SourceSpan
from agm.agl.zones import ParamZone

# Declaration identities for ad-hoc test TypeDefs, distinct from real AST node
# ids (which start at 0) and from every reserved identity (<= -2, see
# ir.reserved_nominals) so an ad-hoc type never collides with either.
_decl_ids = itertools.count(900_000)


def dummy_span() -> SourceSpan:
    """Return a fixed placeholder span for tests that need one but don't inspect it."""
    return SourceSpan(1, 1, 1, 1, 0, 0, UNKNOWN_SOURCE)


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


def parse_inline_command(source: str) -> tuple[Program, int, bool]:
    """Parse inline ``agm exec -c`` source and synthesize its entry when needed.

    Returns the executable program, its next node id, and whether a synthetic
    ``main`` was added. Static-root tests must parse source directly instead.
    """
    program, next_node_id = parse_program_seeded(source, start_id=0)
    wrapped, next_node_id = wrap_inline_program(program, next_node_id=next_node_id)
    return wrapped, next_node_id, wrapped is not program


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
    from dataclasses import replace

    parsed = PipelineDriver.parse_entry(source, entry_path=entry_path)
    if parsed.program is not None:
        program, next_node_id = wrap_inline_program(parsed.program, next_node_id=parsed.next_id)
        parsed = replace(parsed, program=program, next_id=next_node_id)
    return PipelineDriver.prepare_parsed_entry(
        parsed,
        roots=roots,
        default_stdlib=default_stdlib,
    )


def run_inline_command(
    runtime: PipelineDriver,
    source: str,
    *,
    roots: RootSet | None = None,
    default_stdlib: bool = True,
    **run_kwargs: object,
) -> RunResult:
    """Run test-only inline source through the same entry transform as ``agm exec -c``.

    Routes through :meth:`PipelineDriver.preflight_arguments` (binding
    ``positional``/``param_values`` as the entry program's own value
    arguments) when the selected entry ``program def`` declares parameters,
    and through plain default-program selection otherwise.
    """
    prepared = prepare_inline_command(
        source,
        roots=roots,
        default_stdlib=default_stdlib,
    )
    param_values = run_kwargs.pop("param_values", None)
    positional = run_kwargs.pop("positional", None)
    discovery = runtime.discover_programs(prepared)
    if discovery.compiled is None:
        return runtime.run_prepared(prepared, **run_kwargs)
    entry_programs = [program for program in discovery.programs if program.module.is_entry]
    assert len(entry_programs) <= 1
    entry_program = entry_programs[0] if entry_programs else None

    if entry_program is None or not entry_program.parameters:
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
    )
    if not argument_preflight.result.ok:
        return argument_preflight.result
    assert argument_preflight.executable is not None
    return runtime.run_prepared(
        prepared,
        compiled=discovery.compiled,
        executable=argument_preflight.executable,
        program_symbol=argument_preflight.executable.program_symbols[entry_program.node_id],
        arguments=argument_preflight.arguments,
        **run_kwargs,
    )


def next_decl_id() -> int:
    """Return a fresh declaration identity, distinct across the whole test run."""
    return next(_decl_ids)


def let_root_capture(initializer: IrExpr) -> IrBind:
    """Return the binding IR for a simple or destructuring immutable let."""
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
) -> tuple[RecordType, TypeDef]:
    """Build an ad-hoc ``RecordType`` handle and its matching ``TypeDef`` together.

    Returns ``(handle, typedef)``; pass ``typedef`` (and the ``TypeDef`` of
    any nested ad-hoc nominal type referenced in *fields*) to
    :func:`type_table_for` so the handle's field shape resolves.

    *decl_id* defaults to a freshly reserved identity; pass one explicitly
    (from :func:`next_decl_id`) for a SELF-referential or mutually-recursive
    ad-hoc type, whose own *fields* must embed a reference carrying this same
    identity before the ``TypeDef``/handle pair exists to read it off of.
    """
    typedef = TypeDef(
        kind="record",
        name=name,
        module_id=module_id,
        type_params=type_params,
        fields=tuple(fields.items()),
        field_kinds=(ParamZone.STANDARD,) * len(fields),
        decl_node_id=next_decl_id() if decl_id is None else decl_id,
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
