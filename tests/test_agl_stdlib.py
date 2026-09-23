from __future__ import annotations

from pathlib import Path

import pytest

from agm.agl import PipelineDriver
from agm.agl.capabilities import HostCapabilities
from agm.agl.modules.ids import ModuleId
from agm.agl.modules.roots import RootSet
from agm.agl.scope import AglScopeError
from agm.agl.scope.program import resolve_program
from agm.agl.scope.symbols import (
    BUILTIN_CALL_NAMES,
    BUILTIN_TYPE_STATICS,
    NON_RESERVED_BUILTIN_CALL_NAMES,
)
from agm.agl.semantics.type_table import create_seeded_type_table
from agm.agl.semantics.types import (
    BUILTIN_EXCEPTIONS,
    BUILTIN_PRELUDE_TYPES,
    COMPATIBILITY_PRELUDE_TYPE_NAMES,
    OPTION_TEXT_TYPE,
    BoolType,
    EnumType,
    IntType,
    RecordType,
    TextType,
    Type,
    TypeVarType,
)
from agm.agl.typecheck.builtins import BuiltinCallChecker
from agm.agl.typecheck.checker import (
    _builtin_function_signature,
    _builtin_function_signature_alternates,
    _signature_matches,
)
from agm.agl.typecheck.env import AglTypeError, FunctionSignature, ParamSpec
from agm.agl.typecheck.program import check_program
from agm.agl.zones import ParamZone
from tests._agl_helpers import agl_roots, run_inline_command
from tests.agl.module_graph import load_graph, resolve_and_check_inline_entry, resolve_inline_entry

_ROOTS = agl_roots()
_CAPS = HostCapabilities()
_STD_DIR = Path(__file__).resolve().parents[1] / "packages" / "stdlib" / "src"
_STD_OPTION = Path(__file__).resolve().parents[1] / "packages" / "stdlib" / "src" / "option.agl"
_STD_OPTIONAL = Path(__file__).resolve().parents[1] / "packages" / "stdlib" / "src" / "optional.agl"


def _check(source: str, *, default_stdlib: bool = True) -> None:
    resolve_and_check_inline_entry(source, _CAPS, default_stdlib=default_stdlib)


def test_core_stdlib_is_bare_by_default() -> None:
    _check("let x: Option[int] = Some(value = 1)\nprint(x)\n")


def test_no_stdlib_disables_default_open_import() -> None:
    with pytest.raises(AglScopeError, match="'Some' is not defined"):
        resolve_inline_entry("let x: Option[int] = Some(value = 1)\nx\n", default_stdlib=False)


def test_no_stdlib_reports_bare_print_as_undefined() -> None:
    """A bare built-in call has no reachable declaration once the standard
    library is switched off — it is an ordinary undefined-name error, not a
    silently-accepted host dispatch."""
    with pytest.raises(AglScopeError, match="'print' is not defined"):
        resolve_inline_entry('print("hi")\n', default_stdlib=False)


def test_no_stdlib_still_allows_explicit_std_prelude_import() -> None:
    _check(
        "import std/prelude::*\nlet x: Option[int] = Some(value = 1)\nx\n",
        default_stdlib=False,
    )


def test_unknown_builtin_function_is_rejected() -> None:
    with pytest.raises(AglTypeError, match="Unknown builtin function 'mystery'"):
        _check("builtin def mystery() -> unit\n()\n")


def test_builtin_function_signature_must_match() -> None:
    with pytest.raises(AglTypeError, match="Builtin function 'print' has an invalid signature"):
        _check("builtin def print(value: text) -> text\n()\n")


def test_stdlib_ask_signature_is_context_inferred_with_optional_arguments() -> None:
    graph = load_graph(
        "program def main() = ()\n", entry_path=None, roots=_ROOTS, default_stdlib=True
    )
    resolved = resolve_program(graph)
    checked = check_program(resolved, _CAPS)
    std_agent = checked.modules[ModuleId.from_path("std/agent")]

    ask_sig = std_agent.function_signatures["ask"]

    assert ask_sig.type_params == ("T",)
    assert ask_sig.result == TypeVarType("T")
    params = ask_sig.params
    assert params[0].name == "prompt" and params[0].type == TextType() and not params[0].has_default
    agent_param = params[1]
    assert agent_param.name == "agent" and agent_param.has_default
    assert isinstance(agent_param.type, EnumType) and agent_param.type.name == "Agent"
    assert params[2].name == "format" and params[2].type == TextType() and params[2].has_default
    assert (
        params[3].name == "strict-json" and params[3].type == BoolType() and params[3].has_default
    )
    policy_param = params[4]
    assert policy_param.name == "on-parse-error"
    assert isinstance(policy_param.type, EnumType)
    assert policy_param.type.name == "ParsePolicy"
    assert policy_param.has_default is True


def test_canonical_builtin_signatures_name_the_shared_prelude_handles() -> None:
    """Every nominal in a canonical builtin contract is the shared prelude
    handle for that name, so it carries the same owning module and declaration
    identity the rest of the pipeline resolves that name to."""
    ask = _builtin_function_signature("ask")
    assert ask is not None
    ask_params = {param.name: param.type for param in ask.params}
    assert ask_params["on-parse-error"] == BUILTIN_PRELUDE_TYPES["ParsePolicy"]
    ask_request = _builtin_function_signature("ask-request")
    assert ask_request is not None
    assert ask_request.result == BUILTIN_PRELUDE_TYPES["AgentRequest"]
    exec_sig = _builtin_function_signature("exec")
    assert exec_sig is not None
    assert exec_sig.result == BUILTIN_PRELUDE_TYPES["ExecResult"]


def test_ask_surfaces_declare_every_named_argument_they_accept() -> None:
    """The declared surface and the accepted named arguments are one surface.

    An ``ask``/``ask-request`` call is checked against the accepted named-argument
    set rather than against the declaration it was resolved from, so only this
    ties the two together; the declaration in turn cannot deviate from the
    canonical signature (:func:`_builtin_function_signature`) without being
    rejected outright.
    """
    for name in ("ask", "ask-request"):
        signature = _builtin_function_signature(name)
        assert signature is not None
        optional = {param.name for param in signature.params if param.has_default}
        assert optional == BuiltinCallChecker._ASK_ALLOWED_NAMED_ARGS


def test_builtin_function_signature_mismatches_are_rejected() -> None:
    cases = [
        "builtin def print[T](value: T, extra: int) -> unit\n()\n",
        "builtin def print[T](item: T) -> unit\n()\n",
        "builtin def ask-request(prompt: text) -> ExecResult\n()\n",
        "builtin def exec(command: int) -> ExecResult\n()\n",
        "builtin def copy(value: int) -> int\n()\n",
        "builtin def shallow-copy[T](value: T) -> unit\n()\n",
    ]
    for source in cases:
        with pytest.raises(AglTypeError, match="Builtin function '.*' has an invalid signature"):
            _check(source)


def _ps(name: str, t: Type, has_default: bool = False) -> ParamSpec:
    return ParamSpec(name=name, type=t, kind=ParamZone.STANDARD, has_default=has_default)


def test_builtin_signature_helpers_cover_negative_paths() -> None:
    table = create_seeded_type_table()
    sig = FunctionSignature(params=(_ps("value", TextType()),), result=TextType())
    assert _builtin_function_signature("unknown") is None
    assert _builtin_function_signature_alternates("unknown") == ()
    assert not _signature_matches(
        sig,
        FunctionSignature(params=(_ps("value", IntType()),), result=TextType()),
        table,
    )
    assert not _signature_matches(
        FunctionSignature(params=(_ps("value", IntType()),), result=TextType()),
        FunctionSignature(
            params=(_ps("value", RecordType(name="R")),),
            result=TextType(),
        ),
        table,
    )
    assert _signature_matches(
        FunctionSignature(
            params=(_ps("value", RecordType(name="R")),),
            result=TextType(),
        ),
        FunctionSignature(
            params=(_ps("value", RecordType(name="R")),),
            result=TextType(),
        ),
        table,
    )
    assert not _signature_matches(
        FunctionSignature(
            params=(_ps("value", EnumType(name="Actual")),),
            result=TextType(),
        ),
        FunctionSignature(
            params=(_ps("value", EnumType(name="Expected")),),
            result=TextType(),
        ),
        table,
    )


def test_standard_library_declares_every_public_builtin() -> None:
    from agm.agl.parser import parse_program
    from agm.agl.syntax.nodes import EnumDef, ExceptionDef, FuncDef, RecordDef

    items = [
        item
        for path in sorted(_STD_DIR.glob("*.agl"))
        for item in parse_program(path.read_text()).body.items
    ]
    records = {item.name for item in items if isinstance(item, RecordDef) and item.is_builtin}
    enums = {item.name for item in items if isinstance(item, EnumDef) and item.is_builtin}
    exceptions = {item.name for item in items if isinstance(item, ExceptionDef) and item.is_builtin}
    builtin_functions = [item for item in items if isinstance(item, FuncDef) and item.is_builtin]
    functions = {item.name for item in builtin_functions if not item.scope_path}
    statics = {
        ("::".join(segment.name for segment in item.scope_path), item.name)
        for item in builtin_functions
        if item.scope_path and (not item.params or item.params[0].name != "self")
    }

    session_nominals = {"SessionTransport", "Session", "SessionStats", "SessionError"}
    assert session_nominals <= records | enums | exceptions
    assert session_nominals <= set(BUILTIN_PRELUDE_TYPES)
    assert records | enums | exceptions == (
        set(BUILTIN_PRELUDE_TYPES) - set(COMPATIBILITY_PRELUDE_TYPE_NAMES)
        | set(BUILTIN_EXCEPTIONS)
        | {OPTION_TEXT_TYPE.name, "Optional"}
    )
    assert exceptions == set(BUILTIN_EXCEPTIONS) | {"SessionError"}
    assert functions == set(BUILTIN_CALL_NAMES) | set(NON_RESERVED_BUILTIN_CALL_NAMES)
    assert statics == {
        ("::".join(owner_path), static_name)
        for owner_path, names in BUILTIN_TYPE_STATICS.items()
        for static_name in names
    }


def test_std_option_declares_the_builtin_option_and_keeps_its_host_identity() -> None:
    from agm.agl.parser import parse_program
    from tests.agl.ir_harness import lower_ir

    program = parse_program(_STD_OPTION.read_text())
    from agm.agl.syntax.nodes import EnumDef

    enums = {
        item.name for item in program.body.items if isinstance(item, EnumDef) and item.is_builtin
    }
    assert enums == {"Option"}

    executable = lower_ir("program def main() -> unit = ()\n", caps=_CAPS)
    option = executable.builtin_nominals.resolve("Option")
    option_descriptor = executable.nominals[option.nominal]
    assert option_descriptor.module_id == ModuleId.from_path("std/option")
    assert {variant.name for variant in option_descriptor.variants} == {"None", "Some"}
    assert all(variant.member in executable.nominals for variant in option_descriptor.variants)


def test_std_optional_declares_the_builtin_optional_and_is_in_the_prelude() -> None:
    from agm.agl.parser import parse_program
    from agm.agl.syntax.nodes import EnumDef

    program = parse_program(_STD_OPTIONAL.read_text())
    enums = {
        item.name for item in program.body.items if isinstance(item, EnumDef) and item.is_builtin
    }

    assert enums == {"Optional"}
    _check("let value: Optional[path] = Default\n()\n")


def test_unknown_builtin_type_is_rejected() -> None:
    with pytest.raises(AglTypeError, match="Unknown builtin type 'Mystery'"):
        _check("builtin record Mystery\n  value: int\n()\n")


def test_builtin_type_alias_target_must_be_text() -> None:
    with pytest.raises(AglTypeError, match="Builtin type 'path' has an invalid definition"):
        _check("builtin type path = int\n()\n", default_stdlib=False)


def test_builtin_type_alias_must_not_be_generic() -> None:
    with pytest.raises(AglTypeError, match="Builtin type 'path' has an invalid definition"):
        _check("builtin type path[T] = text\n()\n", default_stdlib=False)


def test_path_names_text_without_the_standard_library() -> None:
    _check('let p: path = "a"\nlet t: text = p\nlet back: path = t\n()\n', default_stdlib=False)


def test_a_declared_path_type_replaces_the_reserved_text_alias() -> None:
    with pytest.raises(AglTypeError):
        _check('record path(value: int)\nlet p: path = "a"\n()\n', default_stdlib=False)


def test_url_type_alias_is_bare_visible_through_the_prelude() -> None:
    """Unlike ``path``, ``url`` is a plain alias forwarded by an ordinary
    prelude export, not a globally reserved ``builtin type`` name — it needs
    the standard library, and only its type name is bare."""
    _check('let value: url = "https://example.org"\n()\n')


def test_url_type_alias_is_undefined_without_the_standard_library() -> None:
    with pytest.raises(AglTypeError, match="Unknown type 'url'"):
        _check('let value: url = "https://example.org"\n()\n', default_stdlib=False)


def test_url_record_type_is_reachable_only_through_a_qualified_route() -> None:
    with pytest.raises(AglTypeError, match="Unknown type 'Url'"):
        _check("def f(value: Url) -> unit = ()\n()\n")
    _check("import std/url\ndef f(value: url::Url) -> unit = ()\n()\n")


def test_builtin_type_shape_must_match() -> None:
    with pytest.raises(AglTypeError, match="Builtin type 'ExecResult' has an invalid definition"):
        _check("builtin record ExecResult\n  stdout: text\n()\n")


def test_builtin_record_shape_must_match_field_default_presence() -> None:
    """A ``builtin`` declaration adding a default the host contract does not
    have is a structural mismatch, distinct from a field type/name mismatch."""
    with pytest.raises(AglTypeError, match="Builtin type 'SessionStats' has an invalid definition"):
        _check(
            "builtin record SessionStats(\n"
            "  input-tokens: int,\n"
            "  output-tokens: int,\n"
            "  cost: decimal,\n"
            "  context-percent: decimal = 0.0,\n"
            ")\n()\n",
            default_stdlib=False,
        )


def test_std_core_source_builtin_shape_is_not_masked_by_seed(
    tmp_path: Path,
) -> None:
    stdlib_root = tmp_path / "stdlib"
    core_path = stdlib_root / "src" / "prelude.agl"
    core_path.parent.mkdir(parents=True)
    core_path.write_text("builtin enum Agent\n  | AgentCommand(command: int)\n")
    graph = load_graph(
        "program def main() = ()\n",
        entry_path=None,
        roots=RootSet(roots=frozenset(), stdlib_roots=frozenset({stdlib_root})),
        default_stdlib=True,
    )

    with pytest.raises(AglTypeError):
        check_program(resolve_program(graph), _CAPS)


def test_builtin_option_shape_must_match() -> None:
    with pytest.raises(AglTypeError, match="Builtin type 'Option' has an invalid definition"):
        _check("builtin\nenum Option[T] =\n  | None\n  | Some(value: T, extra: int)\n()\n")


def test_builtin_optional_must_reference_option_members() -> None:
    with pytest.raises(AglTypeError, match="Builtin type 'Optional' has an invalid definition"):
        _check(
            "builtin enum Option[T]\n"
            "  | None\n"
            "  | Some(value: T)\n"
            "\n"
            "scope Fake\n"
            "  record None()\n"
            "  record Some[T](value: T)\n"
            "end Fake\n"
            "\n"
            "builtin enum Optional[T]\n"
            "  | Fake::Some[T]\n"
            "  | Fake::None\n"
            "  | Default\n"
            "()\n",
            default_stdlib=False,
        )


def test_builtin_agent_sandbox_must_reference_the_builtin_sandbox_record() -> None:
    with pytest.raises(AglTypeError, match="Builtin type 'AgentSandbox' has an invalid definition"):
        _check(
            "scope Fake\n"
            "  record Sandbox()\n"
            "end Fake\n"
            "\n"
            "builtin enum AgentSandbox\n"
            "  | Disabled\n"
            "  | Native\n"
            "  | Fake::Sandbox\n"
            "()\n"
        )


def test_builtin_exception_shape_must_match() -> None:
    with pytest.raises(AglTypeError, match="Builtin type 'ExecError' has an invalid definition"):
        _check(
            "builtin\n"
            "exception Exception\n"
            "  @arg-named message: text\n"
            "builtin\n"
            "exception ExecError extends Exception\n"
            "  command: text\n"
            "()\n"
        )


def test_exception_base_must_be_exception_type() -> None:
    with pytest.raises(AglTypeError, match="extends unknown exception 'NotAnException'"):
        _check(
            "record NotAnException\n"
            "  message: text\n"
            "exception Bad extends NotAnException\n"
            "  code: int\n"
            "()\n"
        )


def test_implicit_exception_base_bypasses_lexical_shadowing() -> None:
    _check("scope S\n  record Exception()\n  exception Foo()\nend S\n\n()\n")


def test_exception_fields_cannot_duplicate_inherited_fields() -> None:
    with pytest.raises(AglTypeError, match="Duplicate field 'message' in exception 'Bad'"):
        _check("exception Bad extends Exception\n  message: text\n()\n")


def test_exception_in_field_type_is_built_before_record() -> None:
    _check(
        "exception Local extends Exception\n"
        "  code: int\n"
        "record Wrapper\n"
        "  err: Local\n"
        'Wrapper(err = Local(message = "m", code = 1))\n'
    )


def test_exception_extends_concrete_exception_inherits_field_kinds() -> None:
    # Exercises builder.py _build_exception branch: base_registered is not None
    # (concrete exception A extending concrete exception B, both user-defined).
    _check(
        "exception Base extends Exception\n"
        "  code: int\n"
        "exception Derived extends Base\n"
        "  detail: text\n"
        'Derived(message = "m", code = 1, detail = "d")\n'
    )


def test_exception_in_applied_field_type_is_built_before_rejection() -> None:
    with pytest.raises(AglTypeError, match="Type 'Local' does not take type arguments"):
        _check(
            "exception Local extends Exception\n"
            "  code: int\n"
            "record Wrapper\n"
            "  err: Local[int]\n"
            "()\n"
        )


def test_exception_extends_cycle_is_uninhabitable() -> None:
    # A extends B and B extends A: an `extends` cycle gives neither side
    # independent evidence to become inhabited, so both stay uninhabited —
    # the same inhabitation fixpoint that rejects field recursion.
    with pytest.raises(AglTypeError, match="uninhabitable"):
        _check("exception A extends B\n  a: int\nexception B extends A\n  b: int\n()\n")


def test_lowerer_skips_builtin_function_definitions() -> None:
    from tests.agl.ir_harness import lower_ir

    source = "builtin def print[T](value: T) -> unit\nprogram def main() = ()\n"
    lower_ir(source, caps=_CAPS, default_stdlib=False)


def test_copy_and_shallow_copy_source_declared_calls_are_classified() -> None:
    """Runs without the standard library: ``copy``/``shallow-copy`` are
    ``std/prelude``'s own built-in names too, so declaring them again while it
    is loaded would be a duplicate rather than exercising this call-site
    classification."""
    _check(
        "builtin def copy[T](value: T) -> T\n"
        "builtin def shallow-copy[T](value: T) -> T\n"
        "let _ = copy(1)\n"
        "shallow-copy(1)\n",
        default_stdlib=False,
    )


def test_builtin_named_value_call_is_not_classified_as_builtin() -> None:
    _check("enum E\n  | print\nlet x: E = print()\nx\n")


def test_builtin_exception_own_fields_are_standard_zone(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A builtin exception's own fields follow the standard zone, so they
    accept positional arguments in a constructor call."""
    result = run_inline_command(PipelineDriver(), 'print(IndexError(1, 2, message = "m").index)\n')

    assert list(result.diagnostics) == [], " | ".join(d.message for d in result.diagnostics)
    assert result.error is None
    assert capsys.readouterr().out == "1\n"


def test_builtin_exception_inherited_message_stays_named_only() -> None:
    """``Exception``'s ``message`` keeps the named-only zone every builtin
    exception inherits, so it can never be passed positionally."""
    with pytest.raises(AglTypeError, match="named-only"):
        _check('IndexError("m", 1, 2)\n')


def test_source_defined_exception_extends_base_with_message() -> None:
    _check('raise Abort(message = "stop")\n')


def test_builtin_exception_constructor_resolves_without_the_standard_library() -> None:
    """``Abort`` is a host builtin identity available whether or not
    ``std/prelude`` is loaded — the scope resolver seeds its constructor
    candidate ambiently (module id ``std/prelude``) regardless. With the
    standard library switched off, that candidate's owner has no entry in
    the shared whole-program type table (built only from each module's own,
    non-builtin declarations), so resolving it must fall back to the local
    always-seeded builtin environment instead of assuming the whole-program
    pre-pass always populated it."""
    _check('raise Abort(message = "stop")\n', default_stdlib=False)
