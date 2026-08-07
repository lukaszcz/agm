"""Tests for rendering a semantic ``Type`` as AgL type-annotation syntax.

``render_type_syntax`` renders a semantic ``Type`` back to the AgL
type-annotation text a program would write for it. Unlike ``Type.__repr__``
(total, debug-only), the renderer is partial and its output must be valid
AgL source — the round-trip tests below are what actually pins that property.
"""

from __future__ import annotations

import pytest

from agm.agl.ir.reserved_nominals import NO_DECL_ID
from agm.agl.modules.ids import ENTRY_ID, STD_CORE_ID, ModuleId
from agm.agl.pipeline import PipelineDriver
from agm.agl.semantics.engine_keys import ENGINE_KEY_NAMES, get_engine_key_type
from agm.agl.semantics.type_syntax import TypeSyntaxError, render_type_syntax
from agm.agl.semantics.types import (
    BUILTIN_PRELUDE_TYPES,
    OPTION_TEXT_TYPE,
    ArrayType,
    BoolType,
    BottomType,
    DecimalType,
    DictType,
    EnumType,
    ExceptionType,
    FunctionType,
    InferenceVarType,
    IntType,
    JsonType,
    RecordType,
    TextType,
    Type,
    TypeVarType,
    UnitType,
)
from agm.agl.syntax import FuncDef
from tests._agl_helpers import strip_decl_ids


@pytest.mark.parametrize(
    ("type_", "expected"),
    (
        (TextType(), "text"),
        (JsonType(), "json"),
        (BoolType(), "bool"),
        (IntType(), "int"),
        (DecimalType(), "decimal"),
        (UnitType(), "unit"),
        (ArrayType(IntType()), "array[int]"),
        (DictType(TextType()), "dict[text, text]"),
        (ArrayType(DictType(BoolType())), "array[dict[text, bool]]"),
        (OPTION_TEXT_TYPE, "std/core::Option[text]"),
        (ArrayType(OPTION_TEXT_TYPE), "array[std/core::Option[text]]"),
        (FunctionType((), UnitType()), "() -> unit"),
        (FunctionType((IntType(),), TextType()), "int -> text"),
        (FunctionType((IntType(), TextType()), BoolType()), "(int, text) -> bool"),
        (
            FunctionType((FunctionType((IntType(),), TextType()),), BoolType()),
            "(int -> text) -> bool",
        ),
    ),
)
def test_render_type_syntax_spellable_forms(type_: Type, expected: str) -> None:
    assert render_type_syntax(type_) == expected


def test_render_type_syntax_entry_owned_record_spells_bare() -> None:
    assert render_type_syntax(RecordType(name="Box")) == "Box"


def test_render_type_syntax_entry_owned_generic_enum_spells_bare_with_args() -> None:
    assert render_type_syntax(EnumType(name="Outcome", type_args=(IntType(), TextType()))) == (
        "Outcome[int, text]"
    )


def test_render_type_syntax_builtin_prelude_enum_spells_bare() -> None:
    assert render_type_syntax(BUILTIN_PRELUDE_TYPES["Agent"]) == "Agent"


def test_render_type_syntax_entry_owned_record_in_a_named_scope_spells_scope_path() -> None:
    assert render_type_syntax(RecordType(name="Token", scope_path=("A",))) == "A::Token"


def test_render_type_syntax_module_qualified_generic_nominal() -> None:
    mylib = ModuleId.from_path("mylib")
    assert (
        render_type_syntax(EnumType(name="Thing", type_args=(IntType(),), module_id=mylib))
        == "mylib::Thing[int]"
    )


def test_render_type_syntax_module_and_scope_qualified_nominal() -> None:
    mylib = ModuleId.from_path("mylib")
    rendered = render_type_syntax(
        EnumType(name="Thing", type_args=(IntType(),), module_id=mylib, scope_path=("Outer",))
    )
    assert rendered == "mylib::Outer::Thing[int]"


def test_render_type_syntax_builtin_exception_spells_bare() -> None:
    assert render_type_syntax(ExceptionType(name="CastError", module_id=STD_CORE_ID)) == (
        "CastError"
    )


def test_render_type_syntax_non_builtin_exception_is_module_qualified() -> None:
    mylib = ModuleId.from_path("mylib")
    assert render_type_syntax(ExceptionType(name="Oops", module_id=mylib)) == "mylib::Oops"


@pytest.mark.parametrize(
    "type_",
    (
        BottomType(),
        TypeVarType("T"),
        InferenceVarType("hint"),
    ),
)
def test_render_type_syntax_raises_for_unspellable_types(type_: Type) -> None:
    with pytest.raises(TypeSyntaxError):
        render_type_syntax(type_)


def test_render_type_syntax_raises_for_unspellable_type_nested_in_a_container() -> None:
    with pytest.raises(TypeSyntaxError):
        render_type_syntax(ArrayType(BottomType()))


def _round_trip(type_: Type, preamble: str = "") -> None:
    """Render *type_*, drive it through the real pipeline as a parameter
    annotation on an otherwise-unused function, and check the checker
    resolves it back to the identical semantic ``Type``.

    A function parameter (rather than a ``param`` declaration) is the
    vehicle because it accepts every type form, including ``unit`` and
    function types that ``param`` rejects as not JSON-decodable.
    *preamble* supplies the declarations a nominal type needs to resolve.
    """
    rendered = render_type_syntax(type_)
    driver = PipelineDriver()
    prepared = driver.prepare_program(f"{preamble}def f(value: {rendered}) -> unit = ()\n")
    discovery = driver.discover_params(prepared)

    assert not discovery.diagnostics, discovery.diagnostics
    checked = discovery.checked
    assert checked is not None
    entry_module = checked.modules[ENTRY_ID]
    func_def = next(
        item for item in entry_module.resolved.program.body.items if isinstance(item, FuncDef)
    )
    resolved_type = entry_module.type_env.get_binding_type(func_def.params[0].node_id)
    assert resolved_type is not None
    # A hand-written nominal literal (no declaration identity attached) can
    # only ever assert the round trip's SHAPE, since it has no way to predict
    # the real declaration identity the checker assigns; a literal that
    # already names a real identity (a built-in prelude/std type) is compared
    # exactly, including identity.
    if isinstance(type_, (RecordType, EnumType, ExceptionType)) and type_.decl_id == NO_DECL_ID:
        resolved_type = strip_decl_ids(resolved_type)
    assert resolved_type == type_


@pytest.mark.parametrize("key_name", sorted(ENGINE_KEY_NAMES))
def test_render_type_syntax_round_trips_every_engine_key_type(key_name: str) -> None:
    key_type = get_engine_key_type(key_name)
    assert key_type is not None
    _round_trip(key_type)


@pytest.mark.parametrize(
    "type_",
    (
        UnitType(),
        ArrayType(IntType()),
        DictType(BoolType()),
        ArrayType(DictType(TextType())),
        OPTION_TEXT_TYPE,
        FunctionType((), UnitType()),
        FunctionType((IntType(),), TextType()),
        FunctionType((IntType(), TextType()), BoolType()),
        # A sole parameter that is a container or a qualified nominal: both
        # are type atoms, so neither needs the parenthesized parameter list.
        FunctionType((ArrayType(IntType()),), TextType()),
        FunctionType((DictType(TextType()),), BoolType()),
        FunctionType((OPTION_TEXT_TYPE,), BoolType()),
        # A function type nested inside a container argument.
        ArrayType(FunctionType((IntType(),), TextType())),
        DictType(FunctionType((IntType(),), TextType())),
        # A function-typed parameter: the sole param is itself a function type.
        FunctionType((FunctionType((IntType(),), TextType()),), BoolType()),
        FunctionType((FunctionType((), UnitType()),), UnitType()),
        # A function-typed parameter alongside another positional parameter.
        FunctionType((FunctionType((IntType(),), TextType()), BoolType()), IntType()),
        # A function-typed result, both bare and through a multi-param head.
        FunctionType((IntType(),), FunctionType((TextType(),), BoolType())),
        FunctionType((IntType(), TextType()), FunctionType((BoolType(),), IntType())),
        FunctionType((), FunctionType((IntType(),), BoolType())),
    ),
)
def test_render_type_syntax_round_trips_containers_and_function_types(type_: Type) -> None:
    _round_trip(type_)


@pytest.mark.parametrize(
    ("type_", "preamble"),
    (
        (RecordType(name="Box"), "record Box(x: int)\n\n"),
        (
            RecordType(name="Token", scope_path=("A",)),
            "scope A\nrecord Token(x: int)\nend A\n\n",
        ),
        (ExceptionType(name="CastError", module_id=STD_CORE_ID), ""),
        (BUILTIN_PRELUDE_TYPES["Agent"], ""),
    ),
)
def test_render_type_syntax_round_trips_nominal_types(type_: Type, preamble: str) -> None:
    _round_trip(type_, preamble)
