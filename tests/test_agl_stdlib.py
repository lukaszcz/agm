from __future__ import annotations

from pathlib import Path

import pytest

from agm.agl.capabilities import HostCapabilities
from agm.agl.lower import lower_module
from agm.agl.modules.ids import ModuleId
from agm.agl.modules.loader import load_graph
from agm.agl.modules.roots import RootSet
from agm.agl.parser import parse_program
from agm.agl.scope import AglScopeError, resolve_module
from agm.agl.scope.program import resolve_program
from agm.agl.scope.symbols import BUILTIN_CALL_NAMES
from agm.agl.semantics.types import (
    BUILTIN_EXCEPTIONS,
    BUILTIN_PRELUDE_TYPES,
    COMPATIBILITY_PRELUDE_TYPE_NAMES,
    AgentType,
    BoolType,
    EnumType,
    IntType,
    RecordType,
    TextType,
    Type,
    TypeVarType,
)
from agm.agl.syntax.nodes import ParamKind
from agm.agl.typecheck import check_module
from agm.agl.typecheck.checker import (
    _builtin_function_signature,
    _builtin_function_signature_alternates,
    _signature_matches,
)
from agm.agl.typecheck.env import AglTypeError, FunctionSignature, ParamSpec
from agm.agl.typecheck.program import check_program

_ROOTS = RootSet(frozenset({Path(__file__).resolve().parents[1] / "stdlib"}))
_CAPS = HostCapabilities()
_STD_CORE = Path(__file__).resolve().parents[1] / "stdlib" / "std" / "core.agl"


def _check(source: str, *, default_stdlib: bool = True) -> None:
    graph = load_graph(source, entry_path=None, roots=_ROOTS, default_stdlib=default_stdlib)
    resolved = resolve_program(graph)
    check_program(resolved, _CAPS)


def test_core_stdlib_is_opened_unqualified_by_default() -> None:
    _check("let x: Option[int] = Some(value = 1)\nprint(x)\n")


def test_no_stdlib_disables_default_open_import() -> None:
    graph = load_graph(
        "let x: Option[int] = Some(value = 1)\nx\n",
        entry_path=None,
        roots=_ROOTS,
        default_stdlib=False,
    )
    with pytest.raises(AglScopeError, match="'Some' is not defined"):
        resolve_program(graph)


def test_no_stdlib_still_allows_explicit_std_core_import() -> None:
    _check(
        "open import std/core\nlet x: Option[int] = Some(value = 1)\nx\n",
        default_stdlib=False,
    )


def test_shipped_stdlib_modules_load_without_the_default_prelude() -> None:
    """A shipped module may not lean on the prelude the user is free to switch off."""
    _check("import std/config\nprint(std/config::runner)\n", default_stdlib=False)


def test_unknown_builtin_function_is_rejected() -> None:
    with pytest.raises(AglTypeError, match="Unknown builtin function 'mystery'"):
        _check("builtin def mystery() -> unit\n()\n")


def test_builtin_function_signature_must_match() -> None:
    with pytest.raises(AglTypeError, match="Builtin function 'print' has an invalid signature"):
        _check("builtin def print(value: text) -> text\n()\n")


def test_stdlib_ask_signature_is_context_inferred_with_optional_arguments() -> None:
    graph = load_graph("()\n", entry_path=None, roots=_ROOTS, default_stdlib=True)
    resolved = resolve_program(graph)
    checked = check_program(resolved, _CAPS)
    std_core = checked.modules[ModuleId.from_path("std/core")]

    ask_sig = std_core.function_signatures["ask"]

    assert ask_sig.type_params == ("T",)
    assert ask_sig.result == TypeVarType("T")
    params = ask_sig.params
    assert params[0].name == "prompt" and params[0].type == TextType() and not params[0].has_default
    assert params[1].name == "agent" and params[1].type == AgentType() and params[1].has_default
    assert params[2].name == "format" and params[2].type == TextType() and params[2].has_default
    assert (
        params[3].name == "strict_json" and params[3].type == BoolType() and params[3].has_default
    )
    p4 = params[4]
    assert p4.name == "on_parse_error"
    assert isinstance(p4.type, EnumType)
    assert p4.type.name == "ParsePolicy"
    assert p4.has_default is True


def test_builtin_function_signature_mismatches_are_rejected() -> None:
    cases = [
        "builtin def print[T](value: T, extra: int) -> unit\n()\n",
        "builtin def print[T](item: T) -> unit\n()\n",
        'builtin def parse_json(value: text = "{}") -> json\n()\n',
        "builtin def ask-request(prompt: text) -> ExecResult\n()\n",
        "builtin def exec(command: int) -> ExecResult\n()\n",
    ]
    for source in cases:
        with pytest.raises(AglTypeError, match="Builtin function '.*' has an invalid signature"):
            _check(source)


def _ps(name: str, t: Type, has_default: bool = False) -> ParamSpec:
    return ParamSpec(name=name, type=t, kind=ParamKind.STANDARD, has_default=has_default)


def test_builtin_signature_helpers_cover_negative_paths() -> None:
    sig = FunctionSignature(params=(_ps("value", TextType()),), result=TextType())
    assert _builtin_function_signature("unknown") is None
    assert _builtin_function_signature_alternates("unknown") == ()
    assert not _signature_matches(
        sig,
        FunctionSignature(params=(_ps("value", IntType()),), result=TextType()),
    )
    assert not _signature_matches(
        FunctionSignature(params=(_ps("value", IntType()),), result=TextType()),
        FunctionSignature(
            params=(_ps("value", RecordType(name="R")),),
            result=TextType(),
        ),
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
    )


def test_std_core_declares_every_public_builtin() -> None:
    from agm.agl.parser import parse_program
    from agm.agl.syntax.nodes import EnumDef, ExceptionDef, FuncDef, RecordDef

    program = parse_program(_STD_CORE.read_text())
    records = {
        item.name for item in program.body.items if isinstance(item, RecordDef) and item.is_builtin
    }
    enums = {
        item.name for item in program.body.items if isinstance(item, EnumDef) and item.is_builtin
    }
    exceptions = {
        item.name
        for item in program.body.items
        if isinstance(item, ExceptionDef) and item.is_builtin
    }
    functions = {
        item.name for item in program.body.items if isinstance(item, FuncDef) and item.is_builtin
    }

    public_prelude = set(BUILTIN_PRELUDE_TYPES) - set(COMPATIBILITY_PRELUDE_TYPE_NAMES)
    assert records | enums == public_prelude
    assert exceptions == set(BUILTIN_EXCEPTIONS)
    assert functions == set(BUILTIN_CALL_NAMES)


def test_unknown_builtin_type_is_rejected() -> None:
    with pytest.raises(AglTypeError, match="Unknown builtin type 'Mystery'"):
        _check("builtin record Mystery\n  value: int\n()\n")


def test_builtin_type_shape_must_match() -> None:
    with pytest.raises(AglTypeError, match="Builtin type 'ExecResult' has an invalid definition"):
        _check("builtin record ExecResult\n  stdout: text\n()\n")


def test_builtin_exception_shape_must_match() -> None:
    with pytest.raises(AglTypeError, match="Builtin type 'ExecError' has an invalid definition"):
        _check(
            "builtin\n"
            "exception Exception\n"
            "  *\n"
            "  message: text\n"
            "  trace_id: text\n"
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


def test_single_module_exception_extends_cycle_is_uninhabitable() -> None:
    source = "exception A extends B\n  a: int\nexception B extends A\n  b: int\n()\n"
    with pytest.raises(AglTypeError, match="uninhabitable"):
        check_module(resolve_module(parse_program(source)), _CAPS)


def test_private_exception_definition_parses_and_checks() -> None:
    _check("private exception Hidden extends Exception\n  code: int\n()\n")


def test_single_module_lowerer_skips_builtin_function_definitions() -> None:
    from tests.agl.ir_harness import _compiled_checked

    source = "builtin def print[T](value: T) -> unit\n()\n"
    checked = check_module(resolve_module(parse_program(source)), _CAPS)
    lower_module(_compiled_checked(checked), source_text=source, source_label="<test>")


def test_source_declared_builtin_function_call_is_classified() -> None:
    _check('builtin def parse_json(value: text) -> json\nparse_json("{}")\n')


def test_builtin_named_value_call_is_not_classified_as_builtin() -> None:
    _check("enum E\n  | print\nlet x: E = print()\nx\n")


def test_source_defined_exception_extends_base_and_trace_id_is_optional() -> None:
    _check('raise Abort(message = "stop")\n')
