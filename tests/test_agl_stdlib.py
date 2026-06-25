from __future__ import annotations

from pathlib import Path

import pytest

from agm.agl.capabilities import HostCapabilities
from agm.agl.lower import lower_program
from agm.agl.modules.loader import load_graph
from agm.agl.modules.roots import RootSet
from agm.agl.parser import parse_program
from agm.agl.scope import AglScopeError, resolve
from agm.agl.scope.graph import resolve_graph
from agm.agl.typecheck import check
from agm.agl.typecheck.checker import (
    _builtin_function_signature,
    _signature_matches,
    _type_shape_matches,
)
from agm.agl.typecheck.env import AglTypeError, FunctionSignature
from agm.agl.typecheck.graph import check_graph
from agm.agl.typecheck.types import IntType, RecordType, TextType

_ROOTS = RootSet(frozenset({Path(__file__).resolve().parents[1] / "stdlib"}))
_CAPS = HostCapabilities()


def _check(source: str, *, default_stdlib: bool = True) -> None:
    graph = load_graph(source, entry_path=None, roots=_ROOTS, default_stdlib=default_stdlib)
    resolved = resolve_graph(graph)
    check_graph(resolved, _CAPS)


def test_core_stdlib_is_opened_unqualified_by_default() -> None:
    _check("let x: Option[int] = Some(value: 1)\nprint(x)\n")


def test_no_stdlib_disables_default_open_import() -> None:
    graph = load_graph(
        "let x: Option[int] = Some(value: 1)\nx\n",
        entry_path=None,
        roots=_ROOTS,
        default_stdlib=False,
    )
    with pytest.raises(AglScopeError, match="'Some' is not defined"):
        resolve_graph(graph)


def test_no_stdlib_still_allows_explicit_std_core_import() -> None:
    _check(
        "import std.core\nlet x: Option[int] = Some(value: 1)\nx\n",
        default_stdlib=False,
    )


def test_unknown_builtin_function_is_rejected() -> None:
    with pytest.raises(AglTypeError, match="Unknown builtin function 'mystery'"):
        _check("builtin def mystery() -> unit\n()\n")


def test_builtin_function_signature_must_match() -> None:
    with pytest.raises(AglTypeError, match="Builtin function 'print' has an invalid signature"):
        _check("builtin def print(value: text) -> text\n()\n")


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


def test_builtin_signature_helpers_cover_negative_paths() -> None:
    sig = FunctionSignature(params=(("value", TextType(), False),), result=TextType())
    assert _builtin_function_signature("unknown") is None
    assert not _signature_matches(
        sig,
        FunctionSignature(params=(("value", IntType(), False),), result=TextType()),
    )
    assert not _signature_matches(
        FunctionSignature(params=(("value", IntType(), False),), result=TextType()),
        FunctionSignature(
            params=(("value", RecordType(name="R", fields={}), False),),
            result=TextType(),
        ),
    )
    assert _signature_matches(
        FunctionSignature(
            params=(("value", RecordType(name="R", fields={}), False),),
            result=TextType(),
        ),
        FunctionSignature(
            params=(("value", RecordType(name="R", fields={}), False),),
            result=TextType(),
        ),
    )
    assert not _type_shape_matches(IntType(), TextType())


def test_unknown_builtin_type_is_rejected() -> None:
    with pytest.raises(AglTypeError, match="Unknown builtin type 'Mystery'"):
        _check("builtin record Mystery\n  value: int\n()\n")


def test_builtin_type_shape_must_match() -> None:
    with pytest.raises(AglTypeError, match="Builtin type 'ExecResult' has an invalid definition"):
        _check("builtin record ExecResult\n  stdout: text\n()\n")


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
        "Wrapper(err: Local(message: \"m\", code: 1))\n"
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


def test_exception_recursion_is_rejected() -> None:
    with pytest.raises(AglTypeError, match="structural type cycle"):
        _check(
            "exception A extends B\n"
            "  a: int\n"
            "exception B extends A\n"
            "  b: int\n"
            "()\n"
        )


def test_single_module_exception_recursion_is_rejected() -> None:
    source = (
        "exception A extends B\n"
        "  a: int\n"
        "exception B extends A\n"
        "  b: int\n"
        "()\n"
    )
    message = "Exception type 'A' is directly or indirectly recursive"
    with pytest.raises(AglTypeError, match=message):
        check(resolve(parse_program(source)), _CAPS)


def test_private_exception_definition_parses_and_checks() -> None:
    _check("private exception Hidden extends Exception\n  code: int\n()\n")


def test_single_module_lowerer_skips_builtin_function_definitions() -> None:
    source = "builtin def print[T](value: T) -> unit\n()\n"
    checked = check(resolve(parse_program(source)), _CAPS)
    lower_program(checked, source_text=source, source_label="<test>")


def test_source_declared_builtin_function_call_is_classified() -> None:
    _check('builtin def parse_json(value: text) -> json\nparse_json("{}")\n')


def test_builtin_named_value_call_is_not_classified_as_builtin() -> None:
    _check("enum E\n  | print\nlet x: E = print()\nx\n")


def test_source_defined_exception_extends_base_and_trace_id_is_optional() -> None:
    _check('raise Abort(message: "stop")\n')
