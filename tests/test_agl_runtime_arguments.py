"""Tests for ``agm.agl.runtime.arguments``: host program-argument binding.

Covers ``ProgramSignature.fuse`` (pairing an executable's per-parameter
decoders with a declaration's spans/types, by name), ``bind_program_arguments``
(the shared zone binder plus per-parameter decoding, with diagnostics for
every binding/decode failure), and the ``decode_param_value`` boundary it
shares with the rest of the runtime.
"""

from __future__ import annotations

from decimal import Decimal

import pytest

from agm.agl.attributes import ProgramOptionSpec
from agm.agl.capabilities import HostCapabilities
from agm.agl.diagnostics import Diagnostic
from agm.agl.ir.contracts import ParamDecoder
from agm.agl.ir.nodes import UseDefault
from agm.agl.ir.program import IrProgramParam
from agm.agl.ir.reserved_nominals import require_reserved_nominal_id
from agm.agl.modules.ids import RESERVED_ID
from agm.agl.runtime.arguments import (
    OptionSome,
    ProgramArguments,
    ProgramParameter,
    ProgramSignature,
    bind_program_arguments,
    decode_param_value,
)
from agm.agl.runtime.option import some_value
from agm.agl.runtime.types import ProgramParamInfo
from agm.agl.semantics.type_table import TypeTable, create_seeded_type_table
from agm.agl.semantics.types import (
    BUILTIN_PRELUDE_TYPES,
    BoolType,
    DecimalType,
    EnumType,
    IntType,
    JsonType,
    TextType,
)
from agm.agl.semantics.types import Type as AglType
from agm.agl.semantics.values import (
    BoolValue,
    DecimalValue,
    IntValue,
    JsonValue,
    RecordValue,
    TextValue,
)
from agm.agl.syntax.spans import SourceSpan
from agm.agl.type_schema import build_param_decoder
from agm.agl.zones import ParamZone
from tests.agl.module_graph import resolve_and_check_repl_entry

_TABLE = create_seeded_type_table()

# The program span is kept far from every parameter's declaration span, so a
# test that asserts a specific line number actually distinguishes which
# anchor was used.
_PROGRAM_LINE = 99


def _span(line: int) -> SourceSpan:
    return SourceSpan(line, 1, line, 2, 0, 1)


def _decoder(typ: AglType) -> ParamDecoder:
    return build_param_decoder(typ, _TABLE)


def _option_type(inner: AglType) -> EnumType:
    return EnumType(
        name="Option",
        type_args=(inner,),
        module_id=RESERVED_ID,
        decl_id=require_reserved_nominal_id("Option"),
    )


def _record_point_type() -> "tuple[AglType, TypeTable]":
    """A real typechecked ``record Point\\n  x: int`` type, for value-syntax decode tests."""
    checked = resolve_and_check_repl_entry(
        "record Point\n  x: int\nlet p: Point = Point(x = 1)\np",
        HostCapabilities(
            supports_shell_exec=True,
            codec_kinds={
                "text": frozenset({"text"}),
                "json": frozenset(
                    {"json", "record", "enum", "array", "dict", "int", "decimal", "bool"}
                ),
            },
        ),
    )
    last = checked.resolved.program.body.items[-1]
    return checked.node_types[last.node_id], checked.type_env.type_table


def _param_and_info(
    name: str,
    kind: ParamZone,
    typ: AglType,
    *,
    required: bool,
    line: int,
) -> tuple[IrProgramParam, ProgramParamInfo]:
    decoder = _decoder(typ)
    param = IrProgramParam(name=name, kind=kind, required=required, external_decoder=decoder)
    info = ProgramParamInfo(
        name=name,
        kind=kind,
        type=typ,
        has_default=not required,
        span=_span(line),
        cli=ProgramOptionSpec(name=name),
    )
    return param, info


def _signature(*entries: tuple[IrProgramParam, ProgramParamInfo]) -> ProgramSignature:
    return ProgramSignature.fuse(
        params=tuple(e[0] for e in entries),
        infos=tuple(e[1] for e in entries),
        span=_span(_PROGRAM_LINE),
    )


class TestProgramSignatureFuse:
    """``ProgramSignature.fuse`` pairs parameters by name, not by position."""

    def test_pairs_by_name_even_when_orders_differ(self) -> None:
        name_param, name_info = _param_and_info(
            "name", ParamZone.POSITIONAL_ONLY, TextType(), required=True, line=1
        )
        count_param, count_info = _param_and_info(
            "count", ParamZone.NAMED_ONLY, IntType(), required=True, line=2
        )
        # The executable's decoder tuple and the declaration's info tuple
        # arrive in different orders (as they would if built independently) —
        # `fuse` must still pair each decoder with its own parameter's span
        # and type rather than the parameter at the same position.
        signature = ProgramSignature.fuse(
            params=(count_param, name_param),
            infos=(name_info, count_info),
            span=_span(_PROGRAM_LINE),
        )
        by_name = {p.name: p for p in signature.parameters}
        assert by_name["name"].span == name_info.span
        assert by_name["name"].type == TextType()
        assert by_name["count"].span == count_info.span
        assert by_name["count"].type == IntType()

        values, diagnostics = bind_program_arguments(
            signature,
            ProgramArguments(positional=("alice",), named={"count": "not-an-int"}),
        )
        assert values == ()
        assert len(diagnostics) == 1
        # The decode failure for "count" must be anchored at count's own
        # span (line 2), not name's (line 1) — proving the pairing is by
        # name, not position.
        assert diagnostics[0].line == 2

    def test_missing_info_for_a_declared_parameter_is_a_compiler_bug(self) -> None:
        param, _info = _param_and_info(
            "name", ParamZone.POSITIONAL_ONLY, TextType(), required=True, line=1
        )
        with pytest.raises(AssertionError, match="compiler bug"):
            ProgramSignature.fuse(params=(param,), infos=(), span=_span(_PROGRAM_LINE))

    def test_extra_info_absent_from_signature_is_a_compiler_bug(self) -> None:
        _param, info = _param_and_info(
            "name", ParamZone.POSITIONAL_ONLY, TextType(), required=True, line=1
        )
        with pytest.raises(AssertionError, match="compiler bug"):
            ProgramSignature.fuse(params=(), infos=(info,), span=_span(_PROGRAM_LINE))


class TestBindProgramArgumentsSuccess:
    """Successful bindings across zones, sources, and default use."""

    def test_all_supplied_by_name(self) -> None:
        count = _param_and_info("count", ParamZone.NAMED_ONLY, IntType(), required=True, line=1)
        signature = _signature(count)
        values, diagnostics = bind_program_arguments(
            signature, ProgramArguments(positional=(), named={"count": 5})
        )
        assert diagnostics == ()
        assert values == (IntValue(5),)

    def test_positional_only_supplied_positionally(self) -> None:
        name = _param_and_info("name", ParamZone.POSITIONAL_ONLY, TextType(), required=True, line=1)
        signature = _signature(name)
        values, diagnostics = bind_program_arguments(
            signature, ProgramArguments(positional=("alice",), named={})
        )
        assert diagnostics == ()
        assert values == (TextValue("alice"),)

    def test_standard_parameter_supplied_positionally(self) -> None:
        count = _param_and_info("count", ParamZone.STANDARD, IntType(), required=True, line=1)
        signature = _signature(count)
        values, diagnostics = bind_program_arguments(
            signature, ProgramArguments(positional=(3,), named={})
        )
        assert diagnostics == ()
        assert values == (IntValue(3),)

    def test_standard_parameter_supplied_by_name(self) -> None:
        count = _param_and_info("count", ParamZone.STANDARD, IntType(), required=True, line=1)
        signature = _signature(count)
        values, diagnostics = bind_program_arguments(
            signature, ProgramArguments(positional=(), named={"count": 3})
        )
        assert diagnostics == ()
        assert values == (IntValue(3),)

    def test_omitted_defaulted_parameter_yields_use_default(self) -> None:
        flag = _param_and_info("flag", ParamZone.NAMED_ONLY, BoolType(), required=False, line=1)
        signature = _signature(flag)
        values, diagnostics = bind_program_arguments(
            signature, ProgramArguments(positional=(), named={})
        )
        assert diagnostics == ()
        assert values == (UseDefault(param_index=0),)

    def test_multiple_parameters_mixed_sources(self) -> None:
        name = _param_and_info("name", ParamZone.POSITIONAL_ONLY, TextType(), required=True, line=1)
        flag = _param_and_info("flag", ParamZone.NAMED_ONLY, BoolType(), required=False, line=2)
        signature = _signature(name, flag)
        values, diagnostics = bind_program_arguments(
            signature, ProgramArguments(positional=("bob",), named={"flag": True})
        )
        assert diagnostics == ()
        assert values == (TextValue("bob"), BoolValue(True))

    def test_zero_parameter_program_returns_empty_values_and_no_diagnostics(self) -> None:
        signature = _signature()
        values, diagnostics = bind_program_arguments(
            signature, ProgramArguments(positional=(), named={})
        )
        assert values == ()
        assert diagnostics == ()

    def test_explicit_json_null_named_argument_decodes_rather_than_defaulting(self) -> None:
        # A host-supplied `None` (JSON null) is a real argument, not an
        # omitted one: it must decode to `JsonValue(None)`, not fall back to
        # the parameter's default or be reported missing.
        payload = _param_and_info(
            "payload", ParamZone.NAMED_ONLY, JsonType(), required=False, line=1
        )
        signature = _signature(payload)
        values, diagnostics = bind_program_arguments(
            signature, ProgramArguments(positional=(), named={"payload": None})
        )
        assert diagnostics == ()
        assert values == (JsonValue(None),)

    def test_explicit_json_null_satisfies_a_required_parameter(self) -> None:
        payload = _param_and_info(
            "payload", ParamZone.NAMED_ONLY, JsonType(), required=True, line=1
        )
        signature = _signature(payload)
        values, diagnostics = bind_program_arguments(
            signature, ProgramArguments(positional=(), named={"payload": None})
        )
        assert diagnostics == ()
        assert values == (JsonValue(None),)

    def test_explicit_json_null_positional_argument_decodes_rather_than_defaulting(self) -> None:
        payload = _param_and_info("payload", ParamZone.STANDARD, JsonType(), required=False, line=1)
        signature = _signature(payload)
        values, diagnostics = bind_program_arguments(
            signature, ProgramArguments(positional=(None,), named={})
        )
        assert diagnostics == ()
        assert values == (JsonValue(None),)


class TestBindProgramArgumentsBindingErrors:
    """Zone-binding violations become pre-execution diagnostics."""

    def test_named_only_rejected_positionally(self) -> None:
        flag = _param_and_info("flag", ParamZone.NAMED_ONLY, BoolType(), required=True, line=1)
        signature = _signature(flag)
        values, diagnostics = bind_program_arguments(
            signature, ProgramArguments(positional=(True,), named={})
        )
        assert values == ()
        assert len(diagnostics) == 1
        assert isinstance(diagnostics[0], Diagnostic)
        # The remaining-positional-in-named-only-territory violation names no
        # single parameter; anchored at the program's own span (99), not
        # flag's declaration span (1).
        assert diagnostics[0].line == _PROGRAM_LINE
        assert "by name" in diagnostics[0].message

    def test_too_many_positional_reports_count_not_zone(self) -> None:
        name = _param_and_info("name", ParamZone.POSITIONAL_ONLY, TextType(), required=True, line=1)
        signature = _signature(name)
        values, diagnostics = bind_program_arguments(
            signature, ProgramArguments(positional=("alice", "extra"), named={})
        )
        assert values == ()
        assert len(diagnostics) == 1
        assert diagnostics[0].line == _PROGRAM_LINE
        assert "too many" in diagnostics[0].message.lower()

    def test_positional_only_supplied_by_name_names_the_parameter(self) -> None:
        name = _param_and_info("name", ParamZone.POSITIONAL_ONLY, TextType(), required=True, line=5)
        signature = _signature(name)
        values, diagnostics = bind_program_arguments(
            signature, ProgramArguments(positional=(), named={"name": "alice"})
        )
        assert values == ()
        assert len(diagnostics) == 1
        assert "name" in diagnostics[0].message
        assert "positional-only" in diagnostics[0].message
        # Anchored at the parameter's own declaration span, not the program's.
        assert diagnostics[0].line == 5

    def test_unknown_named(self) -> None:
        count = _param_and_info("count", ParamZone.NAMED_ONLY, IntType(), required=True, line=1)
        signature = _signature(count)
        values, diagnostics = bind_program_arguments(
            signature, ProgramArguments(positional=(), named={"bogus": 1, "count": 1})
        )
        assert values == ()
        assert len(diagnostics) == 1
        assert "bogus" in diagnostics[0].message
        # Anchored at the program's span, since no declared parameter is
        # named "bogus".
        assert diagnostics[0].line == _PROGRAM_LINE

    def test_duplicate_positional_and_named(self) -> None:
        name = _param_and_info("name", ParamZone.STANDARD, TextType(), required=True, line=3)
        signature = _signature(name)
        values, diagnostics = bind_program_arguments(
            signature, ProgramArguments(positional=("alice",), named={"name": "bob"})
        )
        assert values == ()
        assert len(diagnostics) == 1
        assert "name" in diagnostics[0].message
        assert diagnostics[0].line == 3

    def test_excess_positional(self) -> None:
        name = _param_and_info("name", ParamZone.POSITIONAL_ONLY, TextType(), required=True, line=1)
        signature = _signature(name)
        values, diagnostics = bind_program_arguments(
            signature, ProgramArguments(positional=("alice", "extra"), named={})
        )
        assert values == ()
        assert len(diagnostics) == 1
        assert diagnostics[0].line == _PROGRAM_LINE


class TestBindProgramArgumentsMissingRequired:
    """Missing-required diagnostics accumulate rather than short-circuiting."""

    def test_single_missing_required(self) -> None:
        count = _param_and_info("count", ParamZone.NAMED_ONLY, IntType(), required=True, line=7)
        signature = _signature(count)
        values, diagnostics = bind_program_arguments(
            signature, ProgramArguments(positional=(), named={})
        )
        assert values == ()
        assert len(diagnostics) == 1
        assert "count" in diagnostics[0].message
        assert diagnostics[0].line == 7

    def test_two_missing_required_both_reported(self) -> None:
        count = _param_and_info("count", ParamZone.NAMED_ONLY, IntType(), required=True, line=7)
        flag = _param_and_info("flag", ParamZone.NAMED_ONLY, BoolType(), required=True, line=8)
        signature = _signature(count, flag)
        values, diagnostics = bind_program_arguments(
            signature, ProgramArguments(positional=(), named={})
        )
        assert values == ()
        assert len(diagnostics) == 2
        assert {d.line for d in diagnostics} == {7, 8}
        messages = " ".join(d.message for d in diagnostics)
        assert "count" in messages
        assert "flag" in messages

    def test_missing_required_and_decode_failure_both_reported(self) -> None:
        count = _param_and_info("count", ParamZone.NAMED_ONLY, IntType(), required=True, line=7)
        flag = _param_and_info("flag", ParamZone.NAMED_ONLY, BoolType(), required=True, line=8)
        signature = _signature(count, flag)
        values, diagnostics = bind_program_arguments(
            signature, ProgramArguments(positional=(), named={"flag": "not-a-bool"})
        )
        assert values == ()
        assert len(diagnostics) == 2
        assert {d.line for d in diagnostics} == {7, 8}


class TestBindProgramArgumentsDecodeFailure:
    """A supplied value that fails to decode against its parameter's type."""

    def test_decode_failure_wrong_json_shape(self) -> None:
        count = _param_and_info("count", ParamZone.NAMED_ONLY, IntType(), required=True, line=4)
        signature = _signature(count)
        values, diagnostics = bind_program_arguments(
            signature, ProgramArguments(positional=(), named={"count": "not-json-int"})
        )
        assert values == ()
        assert len(diagnostics) == 1
        assert "count" in diagnostics[0].message
        assert diagnostics[0].line == 4

    def test_decode_failure_reports_every_failing_parameter(self) -> None:
        count = _param_and_info("count", ParamZone.NAMED_ONLY, IntType(), required=True, line=4)
        flag = _param_and_info("flag", ParamZone.NAMED_ONLY, BoolType(), required=True, line=5)
        signature = _signature(count, flag)
        values, diagnostics = bind_program_arguments(
            signature,
            ProgramArguments(positional=(), named={"count": "nope", "flag": "nope"}),
        )
        assert values == ()
        assert {d.line for d in diagnostics} == {4, 5}


class TestDecodeParamValue:
    """``decode_param_value`` takes ``text`` verbatim, decodes every other type
    through the canonical JSON boundary, and raises on a strict-JSON parse
    failure or a type/shape mismatch."""

    def test_text_verbatim(self) -> None:
        assert decode_param_value(_decoder(TextType()), "hello") == TextValue("hello")

    def test_json_from_string(self) -> None:
        assert decode_param_value(_decoder(IntType()), "5") == IntValue(5)

    def test_json_from_native_value(self) -> None:
        assert decode_param_value(_decoder(IntType()), 5) == IntValue(5)

    @pytest.mark.parametrize(
        ("raw", "case", "fields"),
        [
            ("claude/sonnet-medium", "AgentClaude", {"model": "sonnet", "thinking": "medium"}),
            ("codex/o3-high", "AgentCodex", {"model": "o3", "thinking": "high"}),
            (
                "pi/openai/gpt-5-low",
                "AgentPi",
                {"provider": "openai", "model": "gpt-5", "thinking": "low"},
            ),
            ("worker --flag", "AgentCommand", {"command": "worker --flag"}),
        ],
    )
    def test_agent_text_uses_host_syntax(self, raw: str, case: str, fields: dict[str, str]) -> None:
        value = decode_param_value(_decoder(BUILTIN_PRELUDE_TYPES["Agent"]), raw)

        assert isinstance(value, RecordValue)
        assert value.display_name.rsplit("::", maxsplit=1)[-1] == case
        assert value.fields == {name: TextValue(field) for name, field in fields.items()}

    def test_agent_keeps_canonical_json_shape(self) -> None:
        value = decode_param_value(
            _decoder(BUILTIN_PRELUDE_TYPES["Agent"]),
            '{"$case":"AgentClaude","model":"opus","thinking":"custom"}',
        )

        assert isinstance(value, RecordValue)
        assert value.display_name.rsplit("::", maxsplit=1)[-1] == "AgentClaude"
        assert value.fields == {"model": TextValue("opus"), "thinking": TextValue("custom")}

    def test_rejects_non_json_shaped_native_value(self) -> None:
        with pytest.raises(ValueError, match="JSON-compatible"):
            decode_param_value(_decoder(IntType()), {1, 2, 3})

    def test_is_json_shaped_dict_with_non_str_key_is_false(self) -> None:
        """_is_json_shaped: a dict with non-str keys is not JSON-shaped (covers
        the dict branch of _is_json_shaped).
        """
        from agm.agl.runtime.arguments import _is_json_shaped

        # Dict with non-str key.
        assert _is_json_shaped({1: "a"}) is False
        # Dict with str keys and JSON-shaped values.
        assert _is_json_shaped({"k": 1}) is True

    def test_option_some_decodes_a_native_value(self) -> None:
        value = decode_param_value(_decoder(_option_type(IntType())), OptionSome(5))

        assert value == some_value(IntValue(5))

    def test_option_some_decodes_a_native_float_into_a_decimal(self) -> None:
        """A native (non-string) ``OptionSome`` payload crosses the same
        canonical JSON boundary as a top-level native value, so a native
        Python ``float`` (as a TOML number decodes) widens into ``Decimal``
        exactly as it would outside an ``Option``."""
        value = decode_param_value(_decoder(_option_type(DecimalType())), OptionSome(1.1))

        assert value == some_value(DecimalValue(Decimal("1.1")))

    def test_option_some_rejects_a_non_json_shaped_native_payload(self) -> None:
        with pytest.raises(ValueError):
            decode_param_value(_decoder(_option_type(IntType())), OptionSome({1, 2}))

    def test_option_some_decodes_a_string_through_host_text(self) -> None:
        value = decode_param_value(_decoder(_option_type(IntType())), OptionSome("5"))

        assert value == some_value(IntValue(5))

    def test_option_some_of_text_is_taken_verbatim(self) -> None:
        value = decode_param_value(_decoder(_option_type(TextType())), OptionSome("hello"))

        assert value == some_value(TextValue("hello"))

    def test_option_some_of_agent_uses_host_syntax(self) -> None:
        value = decode_param_value(
            _decoder(_option_type(BUILTIN_PRELUDE_TYPES["Agent"])), OptionSome("codex/o3-high")
        )

        assert isinstance(value, RecordValue)
        payload = value.fields["value"]
        assert isinstance(payload, RecordValue)
        assert payload.display_name.rsplit("::", maxsplit=1)[-1] == "AgentCodex"

    def test_option_some_of_a_record_value_syntax_string(self) -> None:
        typ, table = _record_point_type()
        decoder = build_param_decoder(_option_type(typ), table)

        value = decode_param_value(decoder, OptionSome("Point(x = 1)"))

        assert isinstance(value, RecordValue)
        payload = value.fields["value"]
        assert isinstance(payload, RecordValue)
        assert payload.fields == {"x": IntValue(1)}

    def test_value_syntax_decodes_a_record_string_directly(self) -> None:
        typ, table = _record_point_type()
        decoder = build_param_decoder(typ, table)

        value = decode_param_value(decoder, "Point(x = 1)")

        assert isinstance(value, RecordValue)
        assert value.fields == {"x": IntValue(1)}


def test_program_parameter_dataclass_carries_all_fields() -> None:
    span = _span(1)
    decoder = _decoder(IntType())
    param = ProgramParameter(
        name="count",
        kind=ParamZone.STANDARD,
        type=IntType(),
        has_default=False,
        span=span,
        decoder=decoder,
    )
    assert param.name == "count"
    assert param.kind == ParamZone.STANDARD
    assert param.type == IntType()
    assert param.has_default is False
    assert param.span == span
    assert param.decoder is decoder
