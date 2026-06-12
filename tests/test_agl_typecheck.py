"""Tests for the AgL type-checking pass (Component 5, M1 subset).

All tests drive real AgL source through ``parse_program`` + ``resolve`` +
``check``, asserting on user-visible behavior: raised ``AglTypeError``
diagnostics (message fragment + source line) and type-table / contract-spec
observables via the public ``CheckedProgram`` API.

Tests deliberately do *not* pin internal implementation details.

Note on M1 parser scope
------------------------
The M1 parser supports: let/var/set/input/pass/print/agent-calls and string
templates with interpolation.  Constructs added in later milestones
(record/enum/type-alias, if/case/do/try, operators, lists, dicts) are *not*
yet parseable.  Tests for those are deferred and marked with ``pytest.mark.skip``.
"""

from __future__ import annotations

import pytest

from agm.agl.capabilities import HostCapabilities
from agm.agl.parser import parse_program
from agm.agl.scope import resolve
from agm.agl.syntax.nodes import (
    AgentCall,
    CaseExpr,
    DictLit,
    Expr,
    FieldDef,
    InterpSegment,
    IntLit,
    LetDecl,
    Program,
    SetStmt,
    Stmt,
    StringLit,
    Template,
    VarDecl,
    VarRef,
)
from agm.agl.syntax.spans import SourceSpan
from agm.agl.syntax.types import (
    BoolT,
    DecimalT,
    IntT,
    JsonT,
    ListT,
    NameT,
    TextT,
    TypeExpr,
)
from agm.agl.typecheck import (
    AglTypeError,
    CheckedProgram,
    OutputContractSpec,
    TextType,
    check,
)
from agm.agl.typecheck.types import (
    BoolType,
    DecimalType,
    DictType,
    EnumType,
    ExceptionType,
    IntType,
    JsonType,
    ListType,
    RecordType,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def default_capabilities() -> HostCapabilities:
    """The M1 host capabilities used as the default for these tests.

    Mirrors the catalog ``WorkflowRuntime.run`` builds when a default/fallback
    agent is configured (text codec; default/raw/json/bullets renderers).
    """
    return HostCapabilities(
        agent_names=frozenset(),
        has_fallback_agent=True,
        has_default_agent=True,
        codec_kinds={"text": frozenset({"text"})},
        renderer_names=frozenset({"default", "raw", "json", "bullets"}),
    )


def parse_resolve_check(
    source: str, capabilities: HostCapabilities | None = None
) -> CheckedProgram:
    """Parse, resolve, and type-check *source* with given/default capabilities."""
    if capabilities is None:
        capabilities = default_capabilities()
    return check(resolve(parse_program(source)), capabilities)


def reject_type(
    source: str, capabilities: HostCapabilities | None = None
) -> AglTypeError:
    """Assert that *source* fails type-checking and return the error."""
    if capabilities is None:
        capabilities = default_capabilities()
    with pytest.raises(AglTypeError) as exc_info:
        parse_resolve_check(source, capabilities)
    return exc_info.value


def accept_type(
    source: str, capabilities: HostCapabilities | None = None
) -> CheckedProgram:
    """Assert that *source* type-checks without error and return the result."""
    return parse_resolve_check(source, capabilities)


def diag(err: AglTypeError) -> tuple[int, str]:
    """Return (line, message) from an AglTypeError."""
    d = err.to_diagnostic()
    return d.line, d.message


# ---------------------------------------------------------------------------
# HostCapabilities fixtures
# ---------------------------------------------------------------------------


def caps_no_fallback(*agent_names: str) -> HostCapabilities:
    """Capabilities with no fallback agent; only named agents accepted."""
    return HostCapabilities(
        agent_names=frozenset(agent_names),
        has_fallback_agent=False,
        codec_kinds={"text": frozenset({"text"})},
        renderer_names=frozenset({"default", "raw"}),
    )


def caps_with_shell_exec() -> HostCapabilities:
    """Capabilities that support shell ``exec`` (simulates M4)."""
    return HostCapabilities(
        agent_names=frozenset(),
        has_fallback_agent=True,
        has_default_agent=True,
        supports_shell_exec=True,
        codec_kinds={"text": frozenset({"text"})},
        renderer_names=frozenset({"default", "raw", "json", "bullets"}),
    )


def caps_with_json_codec() -> HostCapabilities:
    """Capabilities that include the JSON codec (simulates M2+)."""
    return HostCapabilities(
        agent_names=frozenset(),
        has_fallback_agent=True,
        codec_kinds={
            "text": frozenset({"text"}),
            "json": frozenset(
                {"json", "record", "enum", "list", "dict", "int", "decimal", "bool"}
            ),
        },
        renderer_names=frozenset({"default", "raw"}),
    )


# ---------------------------------------------------------------------------
# Acceptance: basic M1-parseable programs
# ---------------------------------------------------------------------------


class TestAcceptance:
    def test_let_text_string(self) -> None:
        r = parse_resolve_check('let x = "hello"')
        assert r.resolved.program is not None

    def test_let_int_literal(self) -> None:
        r = parse_resolve_check("let n = 42")
        assert r.resolved.program is not None

    def test_let_bool_literal(self) -> None:
        r = parse_resolve_check("let b = true")
        assert r.resolved.program is not None

    def test_let_decimal_literal(self) -> None:
        r = parse_resolve_check("let d = 3.14")
        assert r.resolved.program is not None

    def test_let_with_text_annotation(self) -> None:
        r = parse_resolve_check('let x: text = "hi"')
        assert r.resolved.program is not None

    def test_let_with_int_annotation(self) -> None:
        r = parse_resolve_check("let n: int = 10")
        assert r.resolved.program is not None

    def test_int_widens_to_decimal(self) -> None:
        # Single coercion: int literal → decimal annotation.
        r = parse_resolve_check("let d: decimal = 3")
        assert r.resolved.program is not None

    def test_null_is_json(self) -> None:
        r = parse_resolve_check("let j: json = null")
        assert r.resolved.program is not None

    def test_print_any_type(self) -> None:
        r = parse_resolve_check("let n = 1\nprint n")
        assert r.resolved.program is not None

    def test_input_defaults_to_text(self) -> None:
        r = parse_resolve_check("input spec\nprint spec")
        assert r.resolved.program is not None

    def test_input_with_annotation(self) -> None:
        r = parse_resolve_check("input count: int")
        assert r.resolved.program is not None

    def test_untyped_agent_call_defaults_to_text(self) -> None:
        r = parse_resolve_check('let x = prompt "Q"')
        from agm.agl.syntax.nodes import LetDecl

        stmt = r.resolved.program.body[0]
        assert isinstance(stmt, LetDecl)
        assert isinstance(stmt.value, AgentCall)
        spec = r.contract_specs[stmt.value.node_id]
        assert isinstance(spec.target_type, TextType)
        assert spec.codec_name == "text"

    def test_typed_agent_call_uses_annotation(self) -> None:
        r = parse_resolve_check('let x: text = prompt "Q"')
        from agm.agl.syntax.nodes import LetDecl

        stmt = r.resolved.program.body[0]
        assert isinstance(stmt, LetDecl)
        assert isinstance(stmt.value, AgentCall)
        spec = r.contract_specs[stmt.value.node_id]
        assert isinstance(spec.target_type, TextType)

    def test_set_uses_binding_type_text(self) -> None:
        r = parse_resolve_check('var x: text = "a"\nset x = "b"')
        assert r.resolved.program is not None

    def test_renderer_default_ok(self) -> None:
        r = parse_resolve_check('let x = "v"\nlet q = prompt "Hi ${x}"')
        assert r.resolved.program is not None

    def test_renderer_raw_ok(self) -> None:
        r = parse_resolve_check('let x = "v"\nlet q = prompt "Hi ${x as raw}"')
        assert r.resolved.program is not None

    def test_exec_call_text_default(self) -> None:
        # F6: exec is accepted only when the host supports shell exec (M4); the
        # default M1 capabilities reject it (covered in TestExecRejection).
        r = parse_resolve_check('let x = exec "ls"', capabilities=caps_with_shell_exec())
        from agm.agl.syntax.nodes import LetDecl

        stmt = r.resolved.program.body[0]
        assert isinstance(stmt, LetDecl)
        assert isinstance(stmt.value, AgentCall)
        spec = r.contract_specs[stmt.value.node_id]
        assert isinstance(spec.target_type, TextType)
        assert spec.codec_name == "text"

    def test_fallback_unknown_agent_ok(self) -> None:
        # default_capabilities has has_fallback_agent=True, so any name ok.
        r = parse_resolve_check('let x = unknown_agent "Q"')
        assert r.resolved.program is not None

    def test_pass_stmt(self) -> None:
        r = parse_resolve_check("pass")
        assert r.resolved.program is not None


# ---------------------------------------------------------------------------
# F6: renderer capability model (kind-restricted custom renderers)
# ---------------------------------------------------------------------------


def caps_with_renderer_kinds(
    name: str, kinds: frozenset[str] | None
) -> HostCapabilities:
    """Default caps plus one custom renderer with a kind descriptor."""
    return HostCapabilities(
        agent_names=frozenset(),
        has_fallback_agent=True,
        has_default_agent=True,
        codec_kinds={"text": frozenset({"text"})},
        renderer_names=frozenset({"default", "raw", "json", "bullets", name}),
        renderer_kinds={name: kinds},
    )


class TestRendererKinds:
    """F6 (plan §9.1): a renderer may restrict the type kinds it accepts."""

    def test_kind_restricted_renderer_accepts_supported_kind(self) -> None:
        # ``listonly`` accepts only ``list`` kind; ``${xs as listonly}`` where
        # ``xs`` is a list typechecks.
        caps = caps_with_renderer_kinds("listonly", frozenset({"list"}))
        r = parse_resolve_check(
            'let xs: list[text] = ["a"]\nlet q = prompt "${xs as listonly}"',
            capabilities=caps,
        )
        assert r.resolved.program is not None

    def test_kind_restricted_renderer_rejects_unsupported_kind(self) -> None:
        # The same renderer rejects a ``text`` operand (unsupported kind).
        caps = caps_with_renderer_kinds("listonly", frozenset({"list"}))
        err = reject_type(
            'let x = "v"\nlet q = prompt "${x as listonly}"',
            capabilities=caps,
        )
        line, msg = diag(err)
        assert "listonly" in msg
        assert "list" in msg  # mentions the supported kinds

    def test_type_agnostic_renderer_accepts_any_kind(self) -> None:
        # ``None`` supported_types → accepts every kind (text and list both ok).
        caps = caps_with_renderer_kinds("anything", None)
        r = parse_resolve_check(
            'let x = "v"\nlet xs: list[text] = ["a"]\n'
            'let q = prompt "${x as anything} ${xs as anything}"',
            capabilities=caps,
        )
        assert r.resolved.program is not None

    def test_builtin_renderers_accept_all_kinds(self) -> None:
        # Built-ins are NOT pinned to any kind (absent from renderer_kinds →
        # type-agnostic), so json/raw/bullets accept a text operand as before.
        r = parse_resolve_check(
            'let x = "v"\n'
            'let a = prompt "${x as raw}"\n'
            'let b = prompt "${x as json}"\n'
            'let c = prompt "${x as bullets}"'
        )
        assert r.resolved.program is not None

    def test_kind_restricted_multi_kind_renderer(self) -> None:
        # A renderer supporting {text, int} accepts both but rejects a list.
        caps = caps_with_renderer_kinds("scalars", frozenset({"text", "int"}))
        ok = parse_resolve_check(
            'let n = 1\nlet q = prompt "${n as scalars}"', capabilities=caps
        )
        assert ok.resolved.program is not None
        err = reject_type(
            'let xs: list[int] = [1]\nlet q = prompt "${xs as scalars}"',
            capabilities=caps,
        )
        _, msg = diag(err)
        assert "scalars" in msg


# ---------------------------------------------------------------------------
# Contract specs
# ---------------------------------------------------------------------------


class TestContractSpecs:
    def test_text_target_text_codec(self) -> None:
        r = parse_resolve_check('let x = prompt "Q"')
        from agm.agl.syntax.nodes import LetDecl

        stmt = r.resolved.program.body[0]
        assert isinstance(stmt, LetDecl)
        assert isinstance(stmt.value, AgentCall)
        spec = r.contract_specs[stmt.value.node_id]
        assert spec == OutputContractSpec(
            target_type=TextType(), codec_name="text", strict_json=None
        )

    def test_exec_call_text_codec(self) -> None:
        # F6: requires a host that supports shell exec (M4).
        r = parse_resolve_check('let x = exec "ls"', capabilities=caps_with_shell_exec())
        from agm.agl.syntax.nodes import LetDecl

        stmt = r.resolved.program.body[0]
        assert isinstance(stmt, LetDecl)
        assert isinstance(stmt.value, AgentCall)
        spec = r.contract_specs[stmt.value.node_id]
        assert spec.codec_name == "text"
        assert spec.strict_json is None

    def test_set_target_type_propagated(self) -> None:
        r = parse_resolve_check('var x: text = "a"\nset x = prompt "Q"')

        set_stmt = r.resolved.program.body[1]
        assert isinstance(set_stmt, SetStmt)
        assert isinstance(set_stmt.value, AgentCall)
        spec = r.contract_specs[set_stmt.value.node_id]
        assert isinstance(spec.target_type, TextType)

    def test_record_annotation_selects_json_codec(self) -> None:
        """§6 expected-type propagation: let r: Review = prompt derives json codec."""
        from agm.agl.syntax.nodes import LetDecl
        from agm.agl.typecheck.types import RecordType

        src = "record Review\n  score: int\n  comment: text\nlet r: Review = prompt \"Q\"\n"
        r = parse_resolve_check(src, capabilities=caps_with_json_codec())
        stmt = r.resolved.program.body[1]
        assert isinstance(stmt, LetDecl)
        assert isinstance(stmt.value, AgentCall)
        spec = r.contract_specs[stmt.value.node_id]
        assert spec.codec_name == "json"
        assert isinstance(spec.target_type, RecordType)
        assert spec.target_type.name == "Review"

    def test_list_annotation_selects_json_codec(self) -> None:
        """§6 expected-type propagation: let xs: list[int] = prompt derives json codec."""
        from agm.agl.syntax.nodes import LetDecl
        from agm.agl.typecheck.types import ListType

        src = "let xs: list[int] = prompt \"Q\"\n"
        r = parse_resolve_check(src, capabilities=caps_with_json_codec())
        stmt = r.resolved.program.body[0]
        assert isinstance(stmt, LetDecl)
        assert isinstance(stmt.value, AgentCall)
        spec = r.contract_specs[stmt.value.node_id]
        assert spec.codec_name == "json"
        assert isinstance(spec.target_type, ListType)

    def test_set_record_target_propagated(self) -> None:
        """§6 expected-type propagation: set with record-typed var derives json codec."""
        from agm.agl.typecheck.types import RecordType

        src = (
            "record Point\n  x: int\n  y: int\n"
            "var p: Point = Point(x: 0, y: 0)\n"
            "set p = prompt \"Q\"\n"
        )
        r = parse_resolve_check(src, capabilities=caps_with_json_codec())
        set_stmt = r.resolved.program.body[2]
        assert isinstance(set_stmt, SetStmt)
        assert isinstance(set_stmt.value, AgentCall)
        spec = r.contract_specs[set_stmt.value.node_id]
        assert spec.codec_name == "json"
        assert isinstance(spec.target_type, RecordType)
        assert spec.target_type.name == "Point"


# ---------------------------------------------------------------------------
# Rejection: null not assignable to text/int/decimal
# ---------------------------------------------------------------------------


class TestNullAssignability:
    def test_null_to_text(self) -> None:
        # matches tests/agl/rejections/type/null_to_text.agl
        err = reject_type("let s: text = null")
        line, msg = diag(err)
        assert line == 1

    def test_null_to_int(self) -> None:
        err = reject_type("let n: int = null")
        line, msg = diag(err)
        assert line == 1

    def test_null_to_decimal(self) -> None:
        err = reject_type("let d: decimal = null")
        line, msg = diag(err)
        assert line == 1

    def test_null_to_bool(self) -> None:
        err = reject_type("let b: bool = null")
        line, msg = diag(err)
        assert line == 1

    def test_null_to_json_ok(self) -> None:
        r = parse_resolve_check("let j: json = null")
        assert r.resolved.program is not None


# ---------------------------------------------------------------------------
# F1: json assignability (is_json_shaped) — design §5.8 rule 3
# ---------------------------------------------------------------------------


class TestJsonAssignability:
    """``json`` accepts JSON-shaped values; records/enums/exceptions do not."""

    def test_json_shaped_scalars(self) -> None:
        from agm.agl.typecheck.types import (
            BoolType,
            DecimalType,
            IntType,
            JsonType,
            TextType,
            is_json_shaped,
        )

        for t in (TextType(), JsonType(), BoolType(), IntType(), DecimalType()):
            assert is_json_shaped(t)

    def test_json_shaped_containers_recurse(self) -> None:
        from agm.agl.typecheck.types import (
            DictType,
            IntType,
            ListType,
            RecordType,
            is_json_shaped,
        )

        assert is_json_shaped(ListType(elem=IntType()))
        assert is_json_shaped(DictType(value=ListType(elem=IntType())))
        # A list of records is NOT json-shaped.
        rec = RecordType(name="R", fields={"n": IntType()})
        assert not is_json_shaped(ListType(elem=rec))
        assert not is_json_shaped(DictType(value=rec))

    def test_record_enum_exception_not_json_shaped(self) -> None:
        from agm.agl.typecheck.types import (
            EnumType,
            ExceptionType,
            IntType,
            RecordType,
            is_json_shaped,
        )

        assert not is_json_shaped(RecordType(name="R", fields={"n": IntType()}))
        assert not is_json_shaped(EnumType(name="E", variants={"V": {}}))
        assert not is_json_shaped(ExceptionType(name="Abort", fields={}))

    def test_is_assignable_json_accepts_shaped(self) -> None:
        from agm.agl.typecheck.types import (
            IntType,
            JsonType,
            ListType,
            is_assignable,
        )

        assert is_assignable(IntType(), JsonType())
        assert is_assignable(ListType(elem=IntType()), JsonType())

    def test_is_assignable_json_rejects_record(self) -> None:
        from agm.agl.typecheck.types import (
            IntType,
            JsonType,
            RecordType,
            is_assignable,
        )

        assert not is_assignable(RecordType(name="R", fields={"n": IntType()}), JsonType())

    def test_list_of_json_accepts_bool_and_null(self) -> None:
        """F2/F1: ``let x: list[json] = [true, null]`` is accepted."""
        r = parse_resolve_check("let x: list[json] = [true, null]")
        assert r.resolved.program is not None

    def test_record_value_not_assignable_to_json_via_source(self) -> None:
        """The ``record_not_json`` rejection keeps passing."""
        err = reject_type("record P\n  n: int\nlet j: json = P(n: 1)")
        line, _ = diag(err)
        assert line == 3


# ---------------------------------------------------------------------------
# F2: list/dict literal expected-type propagation, soundness, and widening
# ---------------------------------------------------------------------------


class TestListDictLiteralChecking:
    def test_list_decimal_widens_int_elements(self) -> None:
        """``let x: list[decimal] = [1, 2.5]`` is accepted (int → decimal)."""
        r = parse_resolve_check("let x: list[decimal] = [1, 2.5]")
        assert r.resolved.program is not None

    def test_list_json_accepts_mixed_json_shaped(self) -> None:
        r = parse_resolve_check("let x: list[json] = [true, null]")
        assert r.resolved.program is not None

    def test_dict_int_rejects_wrong_value(self) -> None:
        """SOUNDNESS: every entry is checked, not just the first."""
        err = reject_type('let m: dict[text, int] = {a: 1, b: "oops"}')
        line, _ = diag(err)
        assert line == 1

    def test_list_int_rejects_later_text_element(self) -> None:
        """SOUNDNESS: a later element that mismatches the target is rejected."""
        err = reject_type('let xs: list[int] = [1, 2, "three"]')
        line, _ = diag(err)
        assert line == 1

    def test_unannotated_list_widens_to_decimal(self) -> None:
        """Without an annotation, ``[1, 2.5]`` infers ``list[decimal]``."""
        from agm.agl.syntax.nodes import LetDecl

        r = parse_resolve_check("let xs = [1, 2.5]")
        stmt = r.resolved.program.body[0]
        assert isinstance(stmt, LetDecl)
        assert r.node_types[stmt.value.node_id] == ListType(elem=DecimalType())

    def test_unannotated_decimal_then_int_stays_decimal(self) -> None:
        """``[2.5, 1]`` keeps ``list[decimal]`` (a later int does not narrow)."""
        from agm.agl.syntax.nodes import LetDecl

        r = parse_resolve_check("let xs = [2.5, 1]")
        stmt = r.resolved.program.body[0]
        assert isinstance(stmt, LetDecl)
        assert r.node_types[stmt.value.node_id] == ListType(elem=DecimalType())

    def test_unannotated_mixed_incompatible_rejected(self) -> None:
        err = reject_type('let xs = [1, "two"]')
        line, _ = diag(err)
        assert line == 1

    def test_dict_json_value_propagation(self) -> None:
        """A json-valued dict accepts heterogeneous JSON-shaped entries."""
        r = parse_resolve_check('let m: dict[text, json] = {a: 1, b: "two", c: null}')
        assert r.resolved.program is not None

    def test_json_dict_literal_with_nested_list_and_null(self) -> None:
        """A ``json`` dict literal accepts nested lists/dicts and null (json_values.agl)."""
        src = (
            "let document: json = {\n"
            '  "quoted": 1,\n'
            "  shorthand: [true, false, null],\n"
            '  nested: {inner: ["x", 2.5]},\n'
            "}"
        )
        r = parse_resolve_check(src)
        assert r.resolved.program is not None


# ---------------------------------------------------------------------------
# Rejection: literal type mismatches
# ---------------------------------------------------------------------------


class TestLiteralTypeMismatches:
    def test_let_int_annotation_string_value(self) -> None:
        # matches tests/agl/rejections/type/let_type_mismatch.agl
        err = reject_type('let n: int = "five"')
        line, msg = diag(err)
        assert line == 1

    def test_let_text_annotation_int_value(self) -> None:
        err = reject_type("let s: text = 42")
        line, msg = diag(err)
        assert line == 1

    def test_let_bool_annotation_string_value(self) -> None:
        err = reject_type('let b: bool = "yes"')
        line, msg = diag(err)
        assert line == 1

    def test_set_type_mismatch(self) -> None:
        # matches tests/agl/rejections/type/set_type_mismatch.agl
        err = reject_type('var n: int = 1\nset n = "two"')
        line, msg = diag(err)
        assert line == 2


# ---------------------------------------------------------------------------
# Rejection: agent capability errors
# ---------------------------------------------------------------------------


class TestAgentCapabilities:
    def test_no_fallback_unknown_agent_error(self) -> None:
        caps = caps_no_fallback()  # no agents, no fallback
        err = reject_type('let x = unknown_agent "Q"', capabilities=caps)
        line, msg = diag(err)
        assert "unknown_agent" in msg

    def test_no_fallback_known_agent_ok(self) -> None:
        caps = caps_no_fallback("my_agent")
        r = parse_resolve_check('let x = my_agent "Q"', capabilities=caps)
        assert r.resolved.program is not None

    def test_prompt_rejected_without_default_or_fallback(self) -> None:
        # F1a: a ``prompt`` call needs a default agent (or a fallback agent).
        # With neither, it is a static error at the call's span.
        caps = caps_no_fallback()  # has_default_agent=False, has_fallback_agent=False
        err = reject_type('let x = prompt "Q"', caps)
        assert "default agent" in str(err).lower()

    def test_prompt_ok_with_default_agent(self) -> None:
        # A configured default agent makes ``prompt`` valid even with no fallback.
        caps = HostCapabilities(
            agent_names=frozenset(),
            has_fallback_agent=False,
            has_default_agent=True,
            codec_kinds={"text": frozenset({"text"})},
            renderer_names=frozenset({"default", "raw"}),
        )
        r = parse_resolve_check('let x = prompt "Q"', capabilities=caps)
        assert r.resolved.program is not None

    def test_prompt_ok_with_fallback_agent(self) -> None:
        # A fallback agent also backs ``prompt``.
        caps = HostCapabilities(
            agent_names=frozenset(),
            has_fallback_agent=True,
            has_default_agent=False,
            codec_kinds={"text": frozenset({"text"})},
            renderer_names=frozenset({"default", "raw"}),
        )
        r = parse_resolve_check('let x = prompt "Q"', capabilities=caps)
        assert r.resolved.program is not None

    def test_exec_rejected_without_shell_exec_support(self) -> None:
        # F6: with the default M1 capabilities (supports_shell_exec=False), an
        # exec call is a static error regardless of agent fallback.
        caps = caps_no_fallback()
        err = reject_type('let x = exec "ls"', capabilities=caps)
        line, msg = diag(err)
        assert line == 1
        assert "exec" in msg

    def test_exec_supported_ok(self) -> None:
        # F6: a host that supports shell exec (M4) accepts exec calls.
        r = parse_resolve_check('let x = exec "ls"', capabilities=caps_with_shell_exec())
        assert r.resolved.program is not None


# ---------------------------------------------------------------------------
# Rejection: codec capability errors (M1 only has "text" codec by default)
# ---------------------------------------------------------------------------


class TestCodecCapabilities:
    def test_non_text_target_no_json_codec(self) -> None:
        # Targeting int with only the text codec is a static error in M1.
        # matches the spirit of tests/agl/rejections/type/*.agl codec checks
        err = reject_type('let n: int = prompt "Q"')
        line, msg = diag(err)
        assert line == 1
        # Error should mention codec or the type.
        assert "codec" in msg.lower() or "int" in msg.lower()

    def test_strict_json_on_text_codec(self) -> None:
        # matches tests/agl/rejections/type/strict_json_on_text_codec.agl
        err = reject_type('let x = prompt[strict_json: true] "Question."')
        line, msg = diag(err)
        assert line == 1
        assert "strict_json" in msg

    def test_non_text_target_with_json_codec_ok(self) -> None:
        # With the JSON codec, non-text targets are fine.
        caps = caps_with_json_codec()
        r = parse_resolve_check('let n: int = prompt "Q"', capabilities=caps)
        assert r.resolved.program is not None

    def test_strict_json_with_json_codec_ok(self) -> None:
        caps = caps_with_json_codec()
        r = parse_resolve_check(
            'let x: json = prompt[strict_json: true] "Question."',
            capabilities=caps,
        )
        from agm.agl.syntax.nodes import LetDecl

        stmt = r.resolved.program.body[0]
        assert isinstance(stmt, LetDecl)
        assert isinstance(stmt.value, AgentCall)
        spec = r.contract_specs[stmt.value.node_id]
        assert spec.strict_json is True


# ---------------------------------------------------------------------------
# Format option (M2): explicit codec selection via [format: name]
# ---------------------------------------------------------------------------


class TestFormatOption:
    def test_format_json_on_record_target_ok(self) -> None:
        """Explicit format: json with a record target selects the json codec."""
        caps = caps_with_json_codec()
        src = 'record P\n  n: int\nlet x: P = prompt[format: json] "Q"\n'
        r = parse_resolve_check(src, capabilities=caps)
        stmt = r.resolved.program.body[1]
        assert isinstance(stmt, LetDecl)
        assert isinstance(stmt.value, AgentCall)
        spec = r.contract_specs[stmt.value.node_id]
        assert spec.codec_name == "json"

    def test_format_json_on_int_target_ok(self) -> None:
        """format: json on an int target is valid (json codec supports int)."""
        caps = caps_with_json_codec()
        r = parse_resolve_check(
            'let n: int = prompt[format: json] "Q"',
            capabilities=caps,
        )
        stmt = r.resolved.program.body[0]
        assert isinstance(stmt, LetDecl)
        assert isinstance(stmt.value, AgentCall)
        spec = r.contract_specs[stmt.value.node_id]
        assert spec.codec_name == "json"

    def test_format_unknown_codec_error(self) -> None:
        """An unknown codec name in format: is a static error."""
        caps = caps_with_json_codec()
        err = reject_type('let x = prompt[format: nope] "Q"', capabilities=caps)
        line, msg = diag(err)
        assert line == 1
        assert "nope" in msg

    def test_format_json_on_text_target_error(self) -> None:
        """format: json on a text target is invalid (json codec doesn't support text)."""
        caps = caps_with_json_codec()
        err = reject_type('let x: text = prompt[format: json] "Q"', capabilities=caps)
        line, msg = diag(err)
        assert line == 1
        assert "json" in msg

    def test_format_overrides_auto_codec_selection(self) -> None:
        """Explicit format: json on text target overrides auto-selection and must error."""
        caps = caps_with_json_codec()
        err = reject_type('let x = prompt[format: json] "Q"', capabilities=caps)
        # text target + json codec → error (json doesn't handle text)
        assert isinstance(err, AglTypeError)


# ---------------------------------------------------------------------------
# Rejection: renderer errors
# ---------------------------------------------------------------------------


class TestRendererErrors:
    def test_unknown_renderer(self) -> None:
        err = reject_type(
            'let x = "v"\nlet q = prompt "Hi ${x as unknown_renderer}"'
        )
        line, msg = diag(err)
        assert "unknown_renderer" in msg

    def test_known_renderer_raw_ok(self) -> None:
        r = parse_resolve_check('let x = "v"\nlet q = prompt "Hi ${x as raw}"')
        assert r.resolved.program is not None

    def test_known_renderer_default_fallback_ok(self) -> None:
        # No explicit renderer specified (default).
        r = parse_resolve_check('let x = "v"\nlet q = prompt "Hi ${x}"')
        assert r.resolved.program is not None

    def test_mixed_kind_dict_literal_in_interpolation_is_json(self) -> None:
        """checker.py:596 — a mixed-kind dict literal inside ``${ … }`` is checked
        with ``expected=json``, so heterogeneous value kinds are accepted.

        ``{kind: "demo", tags: items}`` mixes a ``text`` value with a
        ``list[text]`` value; only the ``json`` expectation (design §5.8 rule 3)
        lets the dict-literal check accept that, recording the segment as ``json``.
        """
        src = (
            'let items = ["a", "b"]\n'
            'let t = "payload: ${ {kind: "demo", tags: items} }"\n'
        )
        r = parse_resolve_check(src)
        # The interpolated dict-literal is checked with ``expected=json``, so its
        # heterogeneous values (text + list[text]) are accepted under the json
        # rule and the node is typed ``dict[text, json]`` — NOT rejected as an
        # inconsistent dict.
        from agm.agl.syntax.nodes import LetDecl

        let_t = r.resolved.program.body[1]
        assert isinstance(let_t, LetDecl)
        tmpl = let_t.value
        assert isinstance(tmpl, Template)
        seg = next(s for s in tmpl.segments if isinstance(s, InterpSegment))
        assert isinstance(seg.expr, DictLit)
        assert r.node_types[seg.expr.node_id] == DictType(value=JsonType())

    def test_mixed_kind_dict_literal_in_interpolation_renders_as_json(self) -> None:
        """Eval companion: the mixed-kind dict interpolates as JSON text."""
        from agm.agl.runtime import WorkflowRuntime

        src = (
            'let items = ["a", "b"]\n'
            'let t = "payload: ${ {kind: "demo", tags: items} }"\n'
        )
        result = WorkflowRuntime().run(src)
        assert result.ok is True
        from agm.agl.eval.values import TextValue

        rendered = result.bindings["t"]
        assert isinstance(rendered, TextValue)
        # The JSON object is rendered inline (keys preserved); no boundary tags.
        assert '"kind"' in rendered.value and '"demo"' in rendered.value
        assert '"tags"' in rendered.value
        assert "<dsl-value" not in rendered.value


# ---------------------------------------------------------------------------
# F1: enum case exhaustiveness warnings
# ---------------------------------------------------------------------------


_ENUM_PRELUDE = "enum R\n  | Pass\n  | Fail\nlet r: R = Pass\n"


def _warnings(checked: CheckedProgram) -> list[tuple[int, str, str]]:
    """Return (line, message, severity) for each checker warning."""
    return [(w.line, w.message, w.severity) for w in checked.warnings]


class TestExhaustivenessWarnings:
    """F1: a non-wildcard enum ``case`` missing variants is a *warning*.

    The program still type-checks (no error raised); the warning names the
    uncovered variants at the case's source line and has severity ``warning``.
    """

    def test_case_stmt_missing_variant_warns(self) -> None:
        checked = accept_type(
            _ENUM_PRELUDE + "case r of\n  | Pass => pass\n"
        )
        warns = _warnings(checked)
        assert len(warns) == 1
        line, msg, severity = warns[0]
        assert severity == "warning"
        # The case statement is on line 5 (after the 4-line prelude).
        assert line == 5
        assert "Fail" in msg
        assert "Pass" not in msg.split("missing")[1]  # Pass is covered, not listed

    def test_case_stmt_wildcard_suppresses_warning(self) -> None:
        checked = accept_type(
            _ENUM_PRELUDE + "case r of\n  | Pass => pass\n  | _ => pass\n"
        )
        assert _warnings(checked) == []

    def test_case_stmt_bare_var_suppresses_warning(self) -> None:
        checked = accept_type(
            _ENUM_PRELUDE + "case r of\n  | Pass => pass\n  | other => pass\n"
        )
        assert _warnings(checked) == []

    def test_case_stmt_all_variants_covered_no_warning(self) -> None:
        checked = accept_type(
            _ENUM_PRELUDE + "case r of\n  | Pass => pass\n  | Fail => pass\n"
        )
        assert _warnings(checked) == []

    def test_non_enum_scrutinee_no_warning(self) -> None:
        # An int scrutinee has no enumerable variant set, so no exhaustiveness
        # analysis (and no warning) is performed.
        checked = accept_type(
            "let n = 1\ncase n of\n  | 0 => pass\n  | 1 => pass\n"
        )
        assert _warnings(checked) == []

    def test_literal_pattern_incompatible_with_enum_scrutinee_rejected(self) -> None:
        # F3/F5: a literal pattern's type must be compatible with the scrutinee's
        # static type.  A text literal against an enum scrutinee can never match,
        # so it is a static error (consistent with rule 4), not a dead arm.
        err = reject_type(
            _ENUM_PRELUDE + 'case r of\n  | Pass => pass\n  | "x" => pass\n'
        )
        line, msg = diag(err)
        assert line == 7

    def test_case_expr_missing_variant_warns(self) -> None:
        checked = accept_type(
            _ENUM_PRELUDE + "let x = case r of\n  | Pass => 1\n"
        )
        warns = _warnings(checked)
        assert len(warns) == 1
        line, msg, severity = warns[0]
        assert severity == "warning"
        assert line == 5
        assert "Fail" in msg

    def test_case_expr_wildcard_suppresses_warning(self) -> None:
        checked = accept_type(
            _ENUM_PRELUDE + "let x = case r of\n  | Pass => 1\n  | _ => 2\n"
        )
        assert _warnings(checked) == []


class TestParsePolicyNoOpWarnings:
    """A parse policy that can never take effect is a *warning* (design §7.2/§7.10).

    Two cases are no-ops:

    * ``on_parse_error: retry[N]`` on an ``exec`` call — exec's stdout is fixed,
      so it never re-runs; the retry is dead.
    * ``on_parse_error`` on a *text* target — a text target never fails parsing,
      so the policy can never fire.

    Both still type-check (the program runs); the warning lands on the call's
    source line and has severity ``warning``.
    """

    def test_retry_on_exec_warns(self) -> None:
        checked = accept_type(
            'let x = exec[on_parse_error: retry[2]] "echo hi"\n',
            caps_with_shell_exec(),
        )
        warns = _warnings(checked)
        assert len(warns) == 1
        line, msg, severity = warns[0]
        assert severity == "warning"
        assert line == 1
        assert "exec" in msg

    def test_abort_on_exec_warns(self) -> None:
        # ``abort`` is also a no-op on exec — there is nothing to abort *vs.* on a
        # successful parse, but more importantly the policy can never alter the
        # single fixed-output parse outcome, so it is flagged for any explicit
        # policy on exec.
        checked = accept_type(
            'let x = exec[on_parse_error: abort] "echo hi"\n',
            caps_with_shell_exec(),
        )
        warns = _warnings(checked)
        assert len(warns) == 1
        assert warns[0][2] == "warning"
        assert "exec" in warns[0][1]

    def test_exec_without_parse_policy_no_warning(self) -> None:
        checked = accept_type(
            'let x = exec "echo hi"\n',
            caps_with_shell_exec(),
        )
        assert _warnings(checked) == []

    def test_on_parse_error_on_text_target_warns(self) -> None:
        # ``blank`` is untyped → defaults to a text target; a text target never
        # fails parsing, so the on_parse_error policy is a no-op (design §7.10).
        checked = accept_type('let blank = prompt[on_parse_error: retry[1]] "Hi"\n')
        warns = _warnings(checked)
        assert len(warns) == 1
        line, msg, severity = warns[0]
        assert severity == "warning"
        assert line == 1
        assert "text" in msg

    def test_text_target_without_parse_policy_no_warning(self) -> None:
        checked = accept_type('let blank = prompt "Hi"\n')
        assert _warnings(checked) == []

    def test_typed_target_with_parse_policy_no_warning(self) -> None:
        # A non-text (int) target legitimately uses the parse policy — no warning.
        checked = accept_type(
            'let n: int = prompt[on_parse_error: retry[1]] "Number"\n',
            caps_with_json_codec(),
        )
        assert _warnings(checked) == []


class TestConstructorPatternVariantMembership:
    """F4: a constructor pattern's variant name must belong to the scrutinee's
    enum (mirroring ``is`` variant-membership checks); a phantom variant is a
    static error, not a silently-dead arm."""

    def test_unknown_variant_pattern_rejected(self) -> None:
        err = reject_type(
            _ENUM_PRELUDE + "case r of\n  | Zed => pass\n  | _ => pass\n"
        )
        line, msg = diag(err)
        assert line == 6
        assert "Zed" in msg

    def test_other_enums_variant_pattern_rejected(self) -> None:
        err = reject_type(
            "enum R\n  | Pass\n  | Fail\n"
            "enum Q\n  | Other\n"
            "let r: R = Pass\n"
            "case r of\n  | Other => pass\n  | _ => pass\n"
        )
        line, msg = diag(err)
        assert line == 8
        assert "Other" in msg

    def test_qualified_wrong_enum_pattern_rejected(self) -> None:
        err = reject_type(
            "enum R\n  | Pass\n  | Fail\n"
            "enum Q\n  | Other\n"
            "let r: R = Pass\n"
            "case r of\n  | Q.Other => pass\n  | _ => pass\n"
        )
        line, msg = diag(err)
        assert line == 8

    def test_payload_pattern_unknown_variant_rejected(self) -> None:
        err = reject_type(
            "enum R\n  | Pass\n  | Fail(reason: text)\n"
            "let r: R = Pass\n"
            "case r of\n  | Zed(reason) => pass\n  | _ => pass\n"
        )
        line, msg = diag(err)
        assert line == 6
        assert "Zed" in msg

    def test_literal_pattern_compatible_with_int_scrutinee_accepted(self) -> None:
        """F5: int literal patterns against an int scrutinee are valid."""
        r = accept_type("let n = 1\ncase n of\n  | 0 => pass\n  | 1 => pass\n  | _ => pass\n")
        assert r.resolved.program is not None

    def test_literal_pattern_decimal_vs_int_scrutinee_accepted(self) -> None:
        """F5: a decimal literal pattern is compatible with an int scrutinee
        (consistent with ``1 = 1.0`` widening)."""
        r = accept_type(
            "let n = 1\ncase n of\n  | 1.0 => pass\n  | _ => pass\n"
        )
        assert r.resolved.program is not None

    def test_literal_pattern_int_vs_text_scrutinee_rejected(self) -> None:
        """F5: an int literal pattern against a text scrutinee is a static error
        (same machinery as F3 rule 4)."""
        err = reject_type('let s = "x"\ncase s of\n  | 1 => pass\n  | _ => pass\n')
        line, msg = diag(err)
        assert line == 3

    def test_literal_pattern_int_vs_json_scrutinee_rejected(self) -> None:
        """F3: a scalar literal pattern against a json scrutinee is a static error
        (json compares only with json)."""
        err = reject_type(
            "let j: json = 1\ncase j of\n  | 5 => pass\n  | _ => pass\n"
        )
        line, msg = diag(err)
        assert line == 3

    def test_phantom_variant_no_longer_suppresses_exhaustiveness(self) -> None:
        """A phantom variant pattern previously counted as covering a variant and
        could suppress the non-exhaustive warning; now it is rejected outright so
        the suppression cannot happen.  (Exhaustiveness accounting only ever sees
        real variants.)"""
        # ``Zed`` is phantom; it must be a hard error rather than counting toward
        # coverage of the genuine ``Fail`` variant.
        err = reject_type(_ENUM_PRELUDE + "case r of\n  | Pass => pass\n  | Zed => pass\n")
        line, msg = diag(err)
        assert "Zed" in msg


# ---------------------------------------------------------------------------
# Type declarations (M2+ parser required for record/enum/type; deferred)
# ---------------------------------------------------------------------------


class TestTypeDeclErrors:
    def test_duplicate_type_name(self) -> None:
        err = reject_type("record P\n  n: int\nrecord P\n  t: text\n")
        line, msg = diag(err)
        assert "P" in msg

    def test_duplicate_record_field(self) -> None:
        err = reject_type("record P\n  n: int\n  n: text\n")
        line, msg = diag(err)
        assert line == 3

    def test_duplicate_enum_variant(self) -> None:
        err = reject_type("enum E\n  | A\n  | A\n")
        line, msg = diag(err)
        assert line == 3
        assert "A" in msg

    def test_unknown_field_type(self) -> None:
        err = reject_type("record P\n  n: Mystery\n")
        line, msg = diag(err)
        assert line == 2
        assert "Mystery" in msg

    def test_recursive_record(self) -> None:
        err = reject_type("record Node\n  next: Node\n")
        line, msg = diag(err)
        assert "Node" in msg

    def test_alias_cycle(self) -> None:
        err = reject_type("type A = B\ntype B = A\n")
        assert isinstance(err, AglTypeError)

    def test_builtin_name_shadow(self) -> None:
        # Cannot redeclare Abort (built-in exception) as a record.
        err = reject_type("record Abort\n  message: text\n")
        line, msg = diag(err)
        assert line == 1
        assert "Abort" in msg


# ---------------------------------------------------------------------------
# Constructors (deferred: record/enum not parseable in M1)
# ---------------------------------------------------------------------------


class TestConstructorErrors:
    def test_ctor_missing_field(self) -> None:
        err = reject_type("record P\n  n: int\n  t: text\nlet p = P(n: 1)\n")
        line, msg = diag(err)
        assert line == 4
        assert "t" in msg

    def test_ctor_unknown_field(self) -> None:
        err = reject_type("record P\n  n: int\nlet p = P(n: 1, extra: 2)\n")
        line, msg = diag(err)
        assert line == 3
        assert "extra" in msg

    def test_ambiguous_constructor(self) -> None:
        err = reject_type(
            "enum Review\n  | Same\n"
            "enum TestResult\n  | Same\n"
            "let x = Same\n"
        )
        line, msg = diag(err)
        assert line == 5
        assert "Same" in msg

    def test_duplicate_pattern_field_rejected(self) -> None:
        """Duplicate field in a constructor pattern is a type error."""
        err = reject_type(
            "enum R\n"
            "  | Fail(issues: list[text])\n"
            "  | Pass\n"
            "let r: R = Pass\n"
            "case r of\n"
            "  | Fail(issues: a, issues: b) => pass\n"
            "  | Pass => pass\n"
        )
        line, msg = diag(err)
        assert line == 6
        assert "issues" in msg


# ---------------------------------------------------------------------------
# Arithmetic / operators (deferred: operators not parseable in M1)
# ---------------------------------------------------------------------------


class TestOperators:
    def test_text_plus_int_error(self) -> None:
        err = reject_type('let x = "a" + 1')
        line, msg = diag(err)
        assert line == 1

    def test_unary_minus_text(self) -> None:
        err = reject_type('let x = -"a"')
        line, msg = diag(err)
        assert line == 1

    def test_ord_on_bool(self) -> None:
        err = reject_type("let x = (true < false)")
        line, msg = diag(err)
        assert line == 1

    def test_eq_type_mismatch(self) -> None:
        err = reject_type('let x = (1 = "one")')
        line, msg = diag(err)
        assert line == 1

    def test_eq_int_decimal_widening_accepted(self) -> None:
        """F3 rule 4: int and decimal compare after widening."""
        r = parse_resolve_check("let x = (1 = 1.0)")
        assert r.resolved.program is not None

    def test_eq_json_vs_json_accepted(self) -> None:
        """F3 rule 4: json = json is fine."""
        r = parse_resolve_check("let a: json = 1\nlet b: json = 2\nlet x = (a = b)")
        assert r.resolved.program is not None

    def test_eq_json_vs_int_rejected(self) -> None:
        """F3 rule 4: json vs non-json is a static error (the bidirectional
        is_assignable previously accepted this because json accepts int)."""
        err = reject_type("let a: json = 1\nlet x = (a = 5)")
        line, msg = diag(err)
        assert line == 2

    def test_eq_int_vs_json_rejected(self) -> None:
        """F3 rule 4: order does not matter — int vs json is still rejected."""
        err = reject_type("let a: json = 1\nlet x = (5 = a)")
        line, msg = diag(err)
        assert line == 2

    def test_lt_json_vs_int_rejected(self) -> None:
        """F3 rule 4: ordering comparison json vs non-json is a static error."""
        err = reject_type("let a: json = 1\nlet x = (a < 5)")
        line, msg = diag(err)
        assert line == 2

    def test_in_bad_rhs(self) -> None:
        err = reject_type("let x = 1 in 2")
        line, msg = diag(err)
        assert line == 1

    def test_is_on_non_enum(self) -> None:
        err = reject_type(
            "enum R\n  | Pass\n"
            "let n = 1\n"
            "let b = n is Pass\n"
        )
        line, msg = diag(err)
        assert line == 4

    def test_is_wrong_variant(self) -> None:
        err = reject_type(
            "enum R\n  | Pass\n"
            "enum Q\n  | Other\n"
            "let r: R = Pass\n"
            "let b = r is Other\n"
        )
        line, msg = diag(err)
        assert line == 6
        assert "Other" in msg


# ---------------------------------------------------------------------------
# List / dict (deferred: not parseable in M1)
# ---------------------------------------------------------------------------


class TestContainersDeferred:
    def test_list_literal(self) -> None:
        r = parse_resolve_check('let xs = ["a", "b"]')
        assert r.resolved.program is not None

    def test_empty_list_no_annotation(self) -> None:
        err = reject_type("let xs = []")
        line, msg = diag(err)
        assert line == 1

    def test_duplicate_dict_key(self) -> None:
        err = reject_type("let m = {a: 1, a: 2}")
        line, msg = diag(err)
        assert line == 1
        assert "a" in msg


# ---------------------------------------------------------------------------
# Record-as-json (deferred)
# ---------------------------------------------------------------------------


class TestRecordNotJson:
    def test_record_not_json(self) -> None:
        err = reject_type(
            "record P\n  n: int\n"
            "let j: json = P(n: 1)\n"
        )
        line, msg = diag(err)
        assert line == 3


# ---------------------------------------------------------------------------
# Exception catch (deferred: try/catch not parseable in M1)
# ---------------------------------------------------------------------------


class TestExceptionCatch:
    def test_wildcard_catch_subtype_field_error(self) -> None:
        err = reject_type(
            "try\n  pass\n"
            "catch _ as e =>\n"
            "  print e.raw\n"
        )
        line, msg = diag(err)
        assert line == 4
        assert "raw" in msg

    def test_raise_exception_base_not_constructible(self) -> None:
        err = reject_type('raise Exception(message: "x")')
        line, msg = diag(err)
        assert line == 1
        assert "Exception" in msg

    def test_abort_constructor_accepted(self) -> None:
        """Concrete exception constructor (Abort) is a valid constructor call."""
        r = accept_type('raise Abort(message: "stop")')
        assert r is not None  # no type error raised

    def test_exception_constructor_missing_field_error(self) -> None:
        """Missing required field in exception constructor is a type error."""
        err = reject_type("raise Abort()")
        line, msg = diag(err)
        assert line == 1
        assert "message" in msg

    def test_exception_constructor_unknown_field_error(self) -> None:
        """Unknown field in exception constructor is a type error."""
        err = reject_type('raise Abort(message: "x", nonexistent: "y")')
        line, msg = diag(err)
        assert line == 1
        assert "nonexistent" in msg

    def test_exception_constructor_type_mismatch(self) -> None:
        """Wrong field type in exception constructor is a type error."""
        err = reject_type("raise Abort(message: 42)")
        line, msg = diag(err)
        assert line == 1


# ---------------------------------------------------------------------------
# Warnings field
# ---------------------------------------------------------------------------


class TestWarnings:
    def test_warnings_field_is_empty_tuple(self) -> None:
        r = parse_resolve_check('let x = "hi"')
        assert isinstance(r.warnings, tuple)
        assert len(r.warnings) == 0

    def test_warnings_on_complex_program(self) -> None:
        r = parse_resolve_check(
            "input spec\n"
            'let x = prompt "Process ${spec}"\n'
            "print x\n"
        )
        assert isinstance(r.warnings, tuple)


# ---------------------------------------------------------------------------
# HostCapabilities dataclass
# ---------------------------------------------------------------------------


class TestHostCapabilities:
    def test_capabilities_immutable(self) -> None:
        caps = HostCapabilities(
            agent_names=frozenset({"a"}),
            has_fallback_agent=False,
            codec_kinds={"text": frozenset({"text"})},
            renderer_names=frozenset({"default"}),
        )
        assert caps.agent_names == frozenset({"a"})
        assert not caps.has_fallback_agent


# ---------------------------------------------------------------------------
# Direct AST construction tests (covers M2/M3/M4 code paths)
#
# These tests construct AST nodes programmatically to exercise type-checking
# code paths for constructs not yet reachable through the M1 parser.
# ---------------------------------------------------------------------------

_TC_NID = 10000


def _tc_nid() -> int:
    global _TC_NID
    _TC_NID += 1
    return _TC_NID


def _tc_sp(line: int = 1) -> SourceSpan:
    return SourceSpan(
        start_line=line, start_col=1, end_line=line, end_col=2,
        start_offset=0, end_offset=1,
    )


def _tc_intlit(v: int = 1) -> IntLit:
    return IntLit(value=v, span=_tc_sp(), node_id=_tc_nid())


def _tc_varref(name: str) -> VarRef:
    return VarRef(name=name, span=_tc_sp(), node_id=_tc_nid())


def _tc_strlit(v: str = "hi") -> StringLit:
    return StringLit(value=v, span=_tc_sp(), node_id=_tc_nid())


def _tc_let(name: str, value: Expr, type_ann: TypeExpr | None = None) -> LetDecl:
    return LetDecl(name=name, type_ann=type_ann, value=value, span=_tc_sp(), node_id=_tc_nid())


def _tc_program(*stmts: Stmt) -> Program:
    return Program(body=tuple(stmts), span=_tc_sp(), node_id=_tc_nid())


def resolve_and_check(
    *stmts: Stmt,
    caps: HostCapabilities | None = None,
) -> CheckedProgram:
    if caps is None:
        caps = default_capabilities()
    prog = _tc_program(*stmts)
    return check(resolve(prog), caps)


def _tc_field(name: str, type_expr: TypeExpr) -> FieldDef:
    return FieldDef(name=name, type_expr=type_expr, span=_tc_sp(), node_id=_tc_nid())


def _tc_int_t() -> IntT:
    return IntT(span=_tc_sp(), node_id=_tc_nid())


def _tc_text_t() -> TextT:
    return TextT(span=_tc_sp(), node_id=_tc_nid())


def _tc_bool_t() -> BoolT:
    return BoolT(span=_tc_sp(), node_id=_tc_nid())


def _tc_decimal_t() -> DecimalT:
    return DecimalT(span=_tc_sp(), node_id=_tc_nid())


def _tc_json_t() -> JsonT:
    return JsonT(span=_tc_sp(), node_id=_tc_nid())


def _tc_name_t(name: str) -> NameT:
    return NameT(name=name, span=_tc_sp(), node_id=_tc_nid())


def _tc_list_t(elem: TypeExpr) -> ListT:
    return ListT(elem=elem, span=_tc_sp(), node_id=_tc_nid())


# ============================================================
# types.py __repr__ coverage
# ============================================================


class TestTypeReprs:
    """Exercise the __repr__ and .kind methods on all type objects."""

    def test_text_repr(self) -> None:
        from agm.agl.typecheck.types import TextType
        assert repr(TextType()) == "text"

    def test_json_repr(self) -> None:
        assert repr(JsonType()) == "json"

    def test_bool_repr(self) -> None:
        assert repr(BoolType()) == "bool"

    def test_int_repr(self) -> None:
        assert repr(IntType()) == "int"

    def test_decimal_repr(self) -> None:
        assert repr(DecimalType()) == "decimal"

    def test_list_repr(self) -> None:
        assert repr(ListType(elem=IntType())) == "list[int]"

    def test_dict_repr(self) -> None:
        assert repr(DictType(value=TextType())) == "dict[text, text]"

    def test_record_repr(self) -> None:
        assert repr(RecordType(name="Point", fields={})) == "Point"

    def test_enum_repr(self) -> None:
        assert repr(EnumType(name="Color", variants={})) == "Color"

    def test_exception_repr(self) -> None:
        assert repr(ExceptionType(name="Abort")) == "Abort"

    def test_kind_properties(self) -> None:
        """All semantic types expose a .kind string property."""
        from agm.agl.typecheck.types import TextType
        assert TextType().kind == "text"
        assert JsonType().kind == "json"
        assert BoolType().kind == "bool"
        assert IntType().kind == "int"
        assert DecimalType().kind == "decimal"
        assert ListType(elem=IntType()).kind == "list"
        assert DictType(value=TextType()).kind == "dict"
        assert RecordType(name="P", fields={}).kind == "record"
        assert EnumType(name="E", variants={}).kind == "enum"
        assert ExceptionType(name="Ex").kind == "exception"


# ============================================================
# env.py TypeEnvironment coverage
# ============================================================


class TestTypeEnvironment:
    """Directly exercise TypeEnvironment methods."""

    def _make_env(self) -> object:
        from agm.agl.typecheck.env import TypeEnvironment
        return TypeEnvironment()

    def test_has_type_builtin(self) -> None:
        from agm.agl.typecheck.env import TypeEnvironment
        env = TypeEnvironment()
        assert env.has_type("AgentCallError")

    def test_has_type_missing(self) -> None:
        from agm.agl.typecheck.env import TypeEnvironment
        env = TypeEnvironment()
        assert not env.has_type("Nonexistent")

    def test_get_type_builtin(self) -> None:
        from agm.agl.typecheck.env import TypeEnvironment
        env = TypeEnvironment()
        t = env.get_type("Exception")
        assert isinstance(t, ExceptionType)

    def test_get_type_missing_returns_none(self) -> None:
        from agm.agl.typecheck.env import TypeEnvironment
        env = TypeEnvironment()
        assert env.get_type("Nope") is None

    def test_register_type_and_retrieve(self) -> None:
        from agm.agl.typecheck.env import TypeEnvironment
        env = TypeEnvironment()
        rec = RecordType(name="Foo", fields={})
        env.register_type("Foo", rec)
        assert env.get_type("Foo") is rec

    def test_register_alias_and_retrieve(self) -> None:
        from agm.agl.typecheck.env import TypeEnvironment
        env = TypeEnvironment()
        alias_target = _tc_text_t()
        env.register_alias("MyText", alias_target)
        assert env.get_alias_target_expr("MyText") is alias_target

    def test_get_alias_target_missing_returns_none(self) -> None:
        from agm.agl.typecheck.env import TypeEnvironment
        env = TypeEnvironment()
        assert env.get_alias_target_expr("DoesNotExist") is None

    def test_get_binding_type(self) -> None:
        from agm.agl.typecheck.env import TypeEnvironment
        env = TypeEnvironment()
        env.set_binding_type(42, IntType())
        assert env.get_binding_type(42) == IntType()

    def test_get_binding_type_missing(self) -> None:
        from agm.agl.typecheck.env import TypeEnvironment
        env = TypeEnvironment()
        assert env.get_binding_type(999) is None

    def test_all_declared_type_names_includes_builtins(self) -> None:
        from agm.agl.typecheck.env import TypeEnvironment
        env = TypeEnvironment()
        names = env.all_declared_type_names()
        assert "Exception" in names
        assert "AgentCallError" in names

    def test_all_declared_type_names_includes_alias(self) -> None:
        from agm.agl.typecheck.env import TypeEnvironment
        env = TypeEnvironment()
        env.register_alias("MyText", _tc_text_t())
        assert "MyText" in env.all_declared_type_names()

    def test_resolve_type_expr_text(self) -> None:
        from agm.agl.typecheck.env import TypeEnvironment
        env = TypeEnvironment()
        t = env.resolve_type_expr(_tc_text_t())
        from agm.agl.typecheck.types import TextType
        assert isinstance(t, TextType)

    def test_resolve_type_expr_json(self) -> None:
        from agm.agl.typecheck.env import TypeEnvironment
        env = TypeEnvironment()
        t = env.resolve_type_expr(_tc_json_t())
        assert isinstance(t, JsonType)

    def test_resolve_type_expr_bool(self) -> None:
        from agm.agl.typecheck.env import TypeEnvironment
        env = TypeEnvironment()
        t = env.resolve_type_expr(_tc_bool_t())
        assert isinstance(t, BoolType)

    def test_resolve_type_expr_decimal(self) -> None:
        from agm.agl.typecheck.env import TypeEnvironment
        env = TypeEnvironment()
        t = env.resolve_type_expr(_tc_decimal_t())
        assert isinstance(t, DecimalType)

    def test_resolve_type_expr_list(self) -> None:
        from agm.agl.typecheck.env import TypeEnvironment
        env = TypeEnvironment()
        t = env.resolve_type_expr(_tc_list_t(_tc_int_t()))
        assert isinstance(t, ListType)
        assert isinstance(t.elem, IntType)

    def test_resolve_type_expr_dict(self) -> None:
        from agm.agl.syntax.types import DictT
        from agm.agl.typecheck.env import TypeEnvironment
        env = TypeEnvironment()
        dict_t = DictT(value=_tc_text_t(), span=_tc_sp(), node_id=_tc_nid())
        t = env.resolve_type_expr(dict_t)
        assert isinstance(t, DictType)

    def test_resolve_type_expr_name_type(self) -> None:
        from agm.agl.typecheck.env import TypeEnvironment
        env = TypeEnvironment()
        env.register_type("Point", RecordType(name="Point", fields={}))
        t = env.resolve_type_expr(_tc_name_t("Point"))
        assert isinstance(t, RecordType)

    def test_resolve_type_expr_alias_chain(self) -> None:
        from agm.agl.typecheck.env import TypeEnvironment
        env = TypeEnvironment()
        env.register_alias("MyInt", _tc_int_t())
        t = env.resolve_type_expr(_tc_name_t("MyInt"))
        assert isinstance(t, IntType)

    def test_resolve_type_expr_alias_cycle_error(self) -> None:
        from agm.agl.typecheck.env import AglTypeError, TypeEnvironment
        env = TypeEnvironment()
        env.register_alias("A", _tc_name_t("B"))
        env.register_alias("B", _tc_name_t("A"))
        with pytest.raises(AglTypeError) as exc:
            env.resolve_type_expr(_tc_name_t("A"))
        assert "cycle" in exc.value.to_diagnostic().message.lower()

    def test_resolve_named_type_unknown_returns_none(self) -> None:
        from agm.agl.typecheck.env import TypeEnvironment
        env = TypeEnvironment()
        assert env.resolve_named_type("Ghost") is None

    def test_resolve_named_type_resolves_enum_through_alias(self) -> None:
        from agm.agl.typecheck.env import TypeEnvironment
        env = TypeEnvironment()
        enum_t = EnumType(name="R", variants={"Pass": {}})
        env.register_type("R", enum_t)
        env.register_alias("Status", _tc_name_t("R"))
        assert env.resolve_named_type("Status") is enum_t

    def test_resolve_named_type_broken_alias_returns_none(self) -> None:
        """An alias chain that fails to resolve yields None (defensive path)."""
        from agm.agl.typecheck.env import TypeEnvironment
        env = TypeEnvironment()
        env.register_alias("Bad", _tc_name_t("Ghost"))
        assert env.resolve_named_type("Bad") is None

    def test_resolve_type_expr_unknown_name_error(self) -> None:
        from agm.agl.typecheck.env import AglTypeError, TypeEnvironment
        env = TypeEnvironment()
        with pytest.raises(AglTypeError) as exc:
            env.resolve_type_expr(_tc_name_t("Ghost"))
        assert "Ghost" in exc.value.to_diagnostic().message

    def test_resolve_type_expr_unknown_expr_error(self) -> None:
        from agm.agl.typecheck.env import AglTypeError, TypeEnvironment
        env = TypeEnvironment()
        with pytest.raises(AglTypeError):
            env.resolve_type_expr(42)


# ============================================================
# TestTypeBuilderViaSource — _TypeBuilder via parse_program
#
# These tests exercise the _TypeBuilder pre-pass (type declarations: records,
# enums, type aliases, forward references, duplicate names, recursion, and
# alias cycles) through real parsed source.
# ============================================================


class TestTypeBuilderViaSource:
    """Verify _TypeBuilder on M2 constructs using real parsed source."""

    def test_record_simple_via_source(self) -> None:
        """A simple record declaration with primitive fields type-checks."""
        r = accept_type("record Point\n  x: int\n  y: int\n")
        assert r.resolved.program is not None

    def test_record_list_field_via_source(self) -> None:
        """A record with a list-typed field type-checks."""
        r = accept_type("record Bag\n  items: list[int]\n")
        assert r.resolved.program is not None

    def test_record_forward_reference_via_source(self) -> None:
        """Forward reference between records declared in any order type-checks."""
        r = accept_type("record A\n  b: B\nrecord B\n  x: int\n")
        assert r.resolved.program is not None

    def test_type_alias_simple_via_source(self) -> None:
        """A simple type alias is accepted."""
        r = accept_type("type MyText = text\n")
        assert r.resolved.program is not None

    def test_type_alias_cycle_via_source(self) -> None:
        """A mutually-recursive type alias cycle is detected."""
        err = reject_type("type A = B\ntype B = A\n")
        assert isinstance(err, AglTypeError)

    def test_duplicate_record_name_via_source(self) -> None:
        """Declaring two records with the same name is an error."""
        err = reject_type("record P\n  x: int\nrecord P\n  y: int\n")
        assert "P" in err.to_diagnostic().message

    def test_record_recursive_via_source(self) -> None:
        """A directly recursive record (field of the same type) is rejected."""
        err = reject_type("record Node\n  next: Node\n")
        assert "Node" in err.to_diagnostic().message

    def test_record_indirect_cycle_via_source(self) -> None:
        """An indirect A->B->A cycle across records is rejected."""
        err = reject_type("record A\n  b: B\nrecord B\n  a: A\n")
        assert isinstance(err, AglTypeError)

    def test_constructor_and_field_access_via_source(self) -> None:
        """A record constructor call followed by field access type-checks."""
        r = accept_type("record P\n  n: int\nlet x = P(n: 1)\nlet y = x.n\n")
        assert r.resolved.program is not None

    def test_enum_simple_via_source(self) -> None:
        """A simple enum type-checks and its variants are registered."""
        r = accept_type("enum Color | Red | Green\nlet c = Color.Red\n")
        assert r.resolved.program is not None

    def test_enum_variant_with_fields_via_source(self) -> None:
        """An enum variant with fields type-checks and is constructible."""
        r = accept_type("enum Result | Ok | Err(msg: text)\nlet e = Err(msg: \"x\")\n")
        assert r.resolved.program is not None

    def test_list_collection_via_source(self) -> None:
        """A list literal type-checks as list[int]."""
        r = accept_type("let xs = [1, 2, 3]\n")
        assert r.resolved.program is not None

    def test_dict_collection_via_source(self) -> None:
        """A dict literal type-checks as dict[json]."""
        r = accept_type('let d = {"k": 1}\n')
        assert r.resolved.program is not None

    def test_enum_referenced_by_two_fields_via_source(self) -> None:
        """Two record fields of the same enum: the enum is built once (early return)."""
        r = accept_type("enum Color | Red\nrecord Pair\n  a: Color\n  b: Color\n")
        assert r.resolved.program is not None

    def test_duplicate_variant_field_via_source(self) -> None:
        """A variant with two fields of the same name is rejected."""
        err = reject_type("enum E | V(x: int, x: text)\n")
        assert "x" in err.to_diagnostic().message

    def test_record_dict_field_via_source(self) -> None:
        """A record field typed as dict[text, T] resolves the value type."""
        r = accept_type("record M\n  data: dict[text, int]\n")
        assert r.resolved.program is not None


# ============================================================
# TestRecursionThroughAlias (F1) — recursion routed through a `type` alias
#
# A record/enum that refers to itself *through an alias* is still a recursive
# nominal type and must be rejected with the design-§5.9 recursive-type
# diagnostic — never silently accepted with a corrupt empty-shell type.
# ============================================================


class TestRecursionThroughAlias:
    """Recursion that hops through a `type` alias must be detected (design §5.9)."""

    @staticmethod
    def _assert_recursive(err: AglTypeError) -> None:
        """The diagnostic must be the recursive-type one, not unknown / alias-cycle."""
        msg = err.to_diagnostic().message
        assert "recursive" in msg.lower()
        # Distinct from the unknown-type diagnostic ...
        assert "Unknown type" not in msg
        # ... and from the pure alias-cycle diagnostic.
        assert "part of a cycle" not in msg

    def test_alias_to_record_then_field_back_through_alias(self) -> None:
        """type T = R + record R { t: T } — R is recursive via T."""
        err = reject_type("type T = R\nrecord R\n  t: T\n")
        self._assert_recursive(err)

    def test_alias_named_differently_to_record(self) -> None:
        """type T = A + record A { t: T } — A is recursive via T."""
        err = reject_type("type T = A\nrecord A\n  t: T\n")
        self._assert_recursive(err)

    def test_alias_to_enum_then_variant_field_back(self) -> None:
        """type T = E + enum E | V(t: T) — E is recursive via T."""
        err = reject_type("type T = E\nenum E | V(t: T)\n")
        self._assert_recursive(err)

    def test_record_list_field_through_alias_back_to_record(self) -> None:
        """record A { xs: list[T] } + type T = A — A is recursive via list[T]."""
        err = reject_type("record A\n  xs: list[T]\ntype T = A\n")
        self._assert_recursive(err)

    def test_multi_hop_alias_chain_back_to_record(self) -> None:
        """type T = U + type U = A + record A { t: T } — recursive via T -> U."""
        err = reject_type("type T = U\ntype U = A\nrecord A\n  t: T\n")
        self._assert_recursive(err)

    def test_record_field_reaches_pure_alias_cycle(self) -> None:
        """A record field through a pure alias cycle (T -> U -> T) stops following
        and surfaces the alias-cycle diagnostic, not an infinite loop."""
        err = reject_type("record R\n  t: T\ntype T = U\ntype U = T\n")
        assert "part of a cycle" in err.to_diagnostic().message


# ============================================================
# checker.py — statement-level paths (source-driven, M3-parseable)
# ============================================================


def _let_value_type(checked: CheckedProgram, index: int) -> object:
    """Return the recorded type of the value expr of the let/var at *index*."""
    decl = checked.resolved.program.body[index]
    assert isinstance(decl, (LetDecl, VarDecl))
    return checked.node_types[decl.value.node_id]


class TestCheckerStatements:
    """Statement-level type-checking paths, driven through real AgL source.

    Every construct here parses under the M3 grammar; the prior AST-built
    variants (``TestCheckerStatementsViaAst``) were retired once if/case/do/try
    and operators all became parseable.
    """

    def test_pass_stmt(self) -> None:
        r = accept_type("pass\n")
        assert r.resolved.program is not None

    def test_expr_stmt(self) -> None:
        r = accept_type("1\n")
        assert r.resolved.program is not None

    def test_raise_non_exception_expr(self) -> None:
        """F1: raising a non-exception value is a static error; the operand must
        have an exception type (design §8.3)."""
        err = reject_type('raise "oops"\n')
        line, msg = diag(err)
        assert line == 1
        assert "exception" in msg.lower()

    def test_raise_non_exception_int(self) -> None:
        """F1: ``raise 5`` is rejected (would crash the host otherwise)."""
        err = reject_type("raise 5\n")
        line, msg = diag(err)
        assert line == 1
        assert "exception" in msg.lower()

    def test_raise_abstract_base_construction_rejected(self) -> None:
        """F2: the CONSTRUCTION of the abstract ``Exception`` base is rejected at
        the constructor level (mirrors tests/agl/rejections fixture)."""
        err = reject_type('raise Exception(message: "x")\n')
        line, msg = diag(err)
        assert line == 1
        assert "Exception" in msg

    def test_rethrow_wildcard_binder_accepted(self) -> None:
        """F2: re-raising a wildcard-caught binder (typed as the abstract
        ``Exception`` base) is legal — it rethrows an existing value."""
        r = accept_type("try\n  pass\ncatch _ as e =>\n  raise e\n")
        assert r.resolved.program is not None

    def test_rethrow_named_exception_accepted(self) -> None:
        """F2: ``catch Exception as e => raise e`` is legal (rethrow)."""
        r = accept_type("try\n  pass\ncatch Exception as e =>\n  raise e\n")
        assert r.resolved.program is not None

    def test_do_until(self) -> None:
        r = accept_type("var n: int = 0\ndo[5]\n  pass\nuntil true\n")
        assert r.resolved.program is not None

    def test_until_condition_must_be_bool(self) -> None:
        """F1: a non-bool ``until`` condition is a static error (the span points
        at the condition)."""
        err = reject_type("var n: int = 0\ndo[2]\n  pass\nuntil 1 + 1\n")
        line, msg = diag(err)
        assert line == 4
        assert "bool" in msg.lower()

    def test_if_condition_must_be_bool(self) -> None:
        """F1: a non-bool ``if`` condition is a static error."""
        err = reject_type('if 5 =>\n  print "x"\n')
        line, msg = diag(err)
        assert line == 1
        assert "bool" in msg.lower()

    def test_elif_condition_must_be_bool(self) -> None:
        """F1: every branch condition must be bool, not only the first."""
        err = reject_type('if true => pass\n| 5 => pass\n')
        line, msg = diag(err)
        assert line == 2
        assert "bool" in msg.lower()

    def test_not_operand_must_be_bool(self) -> None:
        """F1: ``not`` requires a bool operand."""
        err = reject_type("print not 5\n")
        line, msg = diag(err)
        assert line == 1
        assert "bool" in msg.lower()

    def test_if_stmt_with_else(self) -> None:
        r = accept_type("if true => pass | else => pass\n")
        assert r.resolved.program is not None

    def test_if_stmt_without_else(self) -> None:
        r = accept_type("if true => pass\n")
        assert r.resolved.program is not None

    def test_case_stmt_wildcard(self) -> None:
        r = accept_type("let x = 1\ncase x of\n  | _ => pass\n")
        assert r.resolved.program is not None

    def test_case_stmt_var_pattern(self) -> None:
        r = accept_type(
            "enum R\n  | Pass\nlet r: R = Pass\ncase r of\n  | v => pass\n"
        )
        assert r.resolved.program is not None

    def test_case_stmt_literal_pattern(self) -> None:
        r = accept_type("let x = 1\ncase x of\n  | 1 => pass\n  | _ => pass\n")
        assert r.resolved.program is not None

    def test_case_stmt_constructor_pattern_enum(self) -> None:
        r = accept_type(
            "enum R\n  | Fail(reason: text)\n"
            'let r: R = Fail(reason: "x")\n'
            "case r of\n  | Fail(reason: msg) => pass\n"
        )
        assert r.resolved.program is not None

    def test_try_catch_wildcard(self) -> None:
        r = accept_type("try\n  pass\ncatch _ =>\n  pass\n")
        assert r.resolved.program is not None

    def test_try_catch_specific_exception(self) -> None:
        r = accept_type("try\n  pass\ncatch AgentCallError as err =>\n  pass\n")
        assert r.resolved.program is not None

    def test_try_catch_unknown_exception_type(self) -> None:
        err = reject_type("try\n  pass\ncatch GhostException =>\n  pass\n")
        assert "GhostException" in err.to_diagnostic().message

    def test_try_catch_exc_type_is_underscore(self) -> None:
        """``catch _`` is the wildcard handler (abstract ``Exception`` binder)."""
        r = accept_type("try\n  pass\ncatch _ as e =>\n  pass\n")
        assert r.resolved.program is not None

    def test_record_def_and_enum_def_pass_through(self) -> None:
        """Type declarations are handled by the pre-pass; ``_check_stmt`` is a
        pass-through for them."""
        r = accept_type(
            "record Point\n  x: int\n"
            "enum Status\n  | Ok\n"
            "type PAlias = Point\n"
        )
        assert r.resolved.program is not None


# ============================================================
# checker.py — expression type inference (source-driven, M3-parseable)
# ============================================================


class TestCheckerExprs:
    """Expression inference and error paths, driven through real AgL source."""

    def test_null_literal_is_json(self) -> None:
        r = accept_type("let x: json = null\n")
        assert _let_value_type(r, 0) == JsonType()

    def test_decimal_lit(self) -> None:
        r = accept_type("let x = 3.14\n")
        assert _let_value_type(r, 0) == DecimalType()

    def test_bool_lit(self) -> None:
        r = accept_type("let x = true\n")
        assert _let_value_type(r, 0) == BoolType()

    def test_string_lit(self) -> None:
        r = accept_type('let x = "hello"\n')
        assert _let_value_type(r, 0) == TextType()

    def test_unary_not(self) -> None:
        r = accept_type("let b = true\nlet r = not b\n")
        assert _let_value_type(r, 1) == BoolType()

    def test_unary_neg_int(self) -> None:
        r = accept_type("let n = 1\nlet r = -n\n")
        assert _let_value_type(r, 1) == IntType()

    def test_unary_neg_decimal(self) -> None:
        r = accept_type("let n = 1.5\nlet r = -n\n")
        assert _let_value_type(r, 1) == DecimalType()

    def test_unary_neg_wrong_type(self) -> None:
        err = reject_type('let s = "x"\nlet r = -s\n')
        assert "numeric" in err.to_diagnostic().message

    def test_binary_add_int_int(self) -> None:
        r = accept_type("let a = 1\nlet b = 2\nlet r = a + b\n")
        assert _let_value_type(r, 2) == IntType()

    def test_binary_add_int_decimal(self) -> None:
        r = accept_type("let a = 1\nlet b = 1.5\nlet r = a + b\n")
        assert _let_value_type(r, 2) == DecimalType()

    def test_binary_add_text_text(self) -> None:
        r = accept_type('let a = "x"\nlet b = "y"\nlet r = a + b\n')
        assert _let_value_type(r, 2) == TextType()

    def test_binary_add_type_mismatch(self) -> None:
        err = reject_type('let a = "x"\nlet b = 1\nlet r = a + b\n')
        assert "+" in err.to_diagnostic().message

    def test_binary_sub(self) -> None:
        r = accept_type("let a = 5\nlet b = 3\nlet r = a - b\n")
        assert _let_value_type(r, 2) == IntType()

    def test_binary_sub_decimal(self) -> None:
        r = accept_type("let a = 5\nlet b = 3.0\nlet r = a - b\n")
        assert _let_value_type(r, 2) == DecimalType()

    def test_binary_sub_type_mismatch(self) -> None:
        err = reject_type('let a = "x"\nlet b = 1\nlet r = a - b\n')
        assert "-" in err.to_diagnostic().message

    def test_binary_mul(self) -> None:
        r = accept_type("let a = 2\nlet b = 3\nlet r = a * b\n")
        assert _let_value_type(r, 2) == IntType()

    def test_binary_div(self) -> None:
        r = accept_type("let a = 6\nlet b = 2\nlet r = a / b\n")
        assert _let_value_type(r, 2) == DecimalType()

    def test_binary_div_non_numeric_error(self) -> None:
        err = reject_type('let a = "x"\nlet b = 2\nlet r = a / b\n')
        assert "/" in err.to_diagnostic().message

    def test_binary_and(self) -> None:
        r = accept_type("let a = true\nlet b = false\nlet r = a and b\n")
        assert _let_value_type(r, 2) == BoolType()

    def test_binary_or(self) -> None:
        r = accept_type("let a = true\nlet b = false\nlet r = a or b\n")
        assert _let_value_type(r, 2) == BoolType()

    def test_binary_and_non_bool_left_error(self) -> None:
        """F7: ``and``/``or`` require bool operands; a non-bool left operand is a
        static error reported at the left operand's line."""
        err = reject_type("let a = 1\nlet b = true\nlet r = a and b\n")
        d = err.to_diagnostic()
        assert "and" in d.message and "bool" in d.message
        assert d.line == 3

    def test_binary_or_non_bool_right_error(self) -> None:
        err = reject_type("let a = true\nlet b = 2\nlet r = a or b\n")
        d = err.to_diagnostic()
        assert "or" in d.message and "bool" in d.message
        assert d.line == 3

    def test_binary_eq(self) -> None:
        r = accept_type("let a = 1\nlet b = 2\nlet r = a = b\n")
        assert _let_value_type(r, 2) == BoolType()

    def test_binary_eq_type_mismatch(self) -> None:
        err = reject_type('let a = "x"\nlet b = 1\nlet r = a = b\n')
        assert "same type" in err.to_diagnostic().message

    def test_binary_neq(self) -> None:
        r = accept_type("let a = 1\nlet b = 2\nlet r = a != b\n")
        assert _let_value_type(r, 2) == BoolType()

    def test_binary_lt(self) -> None:
        r = accept_type("let a = 1\nlet b = 2\nlet r = a < b\n")
        assert _let_value_type(r, 2) == BoolType()

    def test_binary_le(self) -> None:
        r = accept_type("let a = 1\nlet b = 2\nlet r = a <= b\n")
        assert _let_value_type(r, 2) == BoolType()

    def test_binary_gt(self) -> None:
        r = accept_type("let a = 1\nlet b = 2\nlet r = a > b\n")
        assert _let_value_type(r, 2) == BoolType()

    def test_binary_ge(self) -> None:
        r = accept_type("let a = 1\nlet b = 2\nlet r = a >= b\n")
        assert _let_value_type(r, 2) == BoolType()

    def test_binary_ord_bool_error(self) -> None:
        err = reject_type("let a = true\nlet b = false\nlet r = a < b\n")
        assert "numeric" in err.to_diagnostic().message

    def test_binary_in_text_text(self) -> None:
        r = accept_type('let a = "hi"\nlet b = "hi there"\nlet r = a in b\n')
        assert _let_value_type(r, 2) == BoolType()

    def test_binary_in_elem_list(self) -> None:
        r = accept_type("let lst = [1, 2]\nlet a = 1\nlet r = a in lst\n")
        assert _let_value_type(r, 2) == BoolType()

    def test_binary_in_text_dict(self) -> None:
        r = accept_type('let d = {a: 1}\nlet k = "a"\nlet r = k in d\n')
        assert _let_value_type(r, 2) == BoolType()

    def test_binary_in_elem_list_mismatch(self) -> None:
        err = reject_type('let lst = [1]\nlet a = "x"\nlet r = a in lst\n')
        assert "in" in err.to_diagnostic().message

    def test_binary_in_bad_rhs(self) -> None:
        err = reject_type("let a = 1\nlet b = 2\nlet r = a in b\n")
        assert "in" in err.to_diagnostic().message

    def test_is_test_valid(self) -> None:
        r = accept_type(
            "enum R\n  | Pass\nlet r: R = Pass\nlet b = r is Pass\n"
        )
        # body = [EnumDef, LetDecl(r), LetDecl(b)]; the is-test is the value of
        # the let at index 2.
        assert _let_value_type(r, 2) == BoolType()

    def test_is_test_wrong_variant(self) -> None:
        err = reject_type(
            "enum R\n  | Pass\nlet r: R = Pass\nlet b = r is Other\n"
        )
        assert "Other" in err.to_diagnostic().message

    def test_is_test_non_enum_error(self) -> None:
        err = reject_type("let n = 1\nlet b = n is Pass\n")
        assert "enum" in err.to_diagnostic().message

    def test_field_access_record(self) -> None:
        r = accept_type(
            "record Point\n  x: int\nlet p = Point(x: 1)\nlet x = p.x\n"
        )
        assert _let_value_type(r, 2) == IntType()

    def test_field_access_record_unknown_field(self) -> None:
        err = reject_type(
            "record Point\n  x: int\nlet p = Point(x: 1)\nlet y = p.y\n"
        )
        assert "y" in err.to_diagnostic().message

    def test_field_access_exception_type(self) -> None:
        """Field access on a caught exception binder reads exception fields."""
        r = accept_type(
            "try\n  pass\ncatch AgentCallError as err =>\n  print err.message\n"
        )
        assert r.resolved.program is not None

    def test_field_access_exception_unknown_field(self) -> None:
        err = reject_type(
            "try\n  pass\ncatch AgentCallError as err =>\n  print err.ghost\n"
        )
        assert "ghost" in err.to_diagnostic().message

    def test_field_access_non_record_error(self) -> None:
        err = reject_type("let n = 1\nlet r = n.x\n")
        assert "record" in err.to_diagnostic().message

    # --- Constructor ---

    def test_record_constructor_ok(self) -> None:
        r = accept_type(
            "record Pt\n  x: int\n  y: int\nlet p = Pt(x: 1, y: 2)\n"
        )
        assert isinstance(_let_value_type(r, 1), RecordType)

    def test_record_constructor_missing_field(self) -> None:
        err = reject_type("record Pt\n  x: int\n  y: int\nlet p = Pt(x: 1)\n")
        assert "y" in err.to_diagnostic().message

    def test_record_constructor_unknown_field(self) -> None:
        err = reject_type("record Pt\n  x: int\nlet p = Pt(x: 1, z: 2)\n")
        assert "z" in err.to_diagnostic().message

    def test_enum_constructor_qualified(self) -> None:
        r = accept_type("enum R\n  | Pass\nlet v = R.Pass\n")
        assert isinstance(_let_value_type(r, 1), EnumType)

    def test_enum_constructor_alias_qualified(self) -> None:
        """``type Status = Review`` then ``Status.Pass`` resolves the alias."""
        r = accept_type(
            "enum Review\n  | Pass\ntype Status = Review\nlet v = Status.Pass\n"
        )
        resolved = _let_value_type(r, 2)
        assert isinstance(resolved, EnumType)
        assert resolved.name == "Review"

    def test_enum_constructor_alias_of_alias_qualified(self) -> None:
        r = accept_type(
            "enum Review\n  | Pass\n"
            "type A = Review\ntype B = A\nlet v = B.Pass\n"
        )
        resolved = _let_value_type(r, 3)
        assert isinstance(resolved, EnumType)
        assert resolved.name == "Review"

    def test_enum_constructor_non_enum_alias_qualifier_rejected(self) -> None:
        err = reject_type("type Nums = list[int]\nlet v = Nums.Pass\n")
        assert "Nums" in err.to_diagnostic().message

    def test_enum_constructor_qualified_unknown_enum(self) -> None:
        err = reject_type("let v = Ghost.Pass\n")
        assert "Ghost" in err.to_diagnostic().message

    def test_enum_constructor_qualified_unknown_variant(self) -> None:
        err = reject_type("enum R\n  | Pass\nlet v = R.Fail\n")
        assert "Fail" in err.to_diagnostic().message

    def test_enum_constructor_unqualified_ambiguous(self) -> None:
        err = reject_type(
            "enum E1\n  | Both\nenum E2\n  | Both\nlet v = Both\n"
        )
        assert "ambiguous" in err.to_diagnostic().message.lower()

    def test_enum_constructor_unqualified_with_expected_type_resolves(self) -> None:
        r = accept_type(
            "enum E1\n  | Both\nenum E2\n  | Both\nlet v: E1 = Both\n"
        )
        assert isinstance(_let_value_type(r, 2), EnumType)

    def test_enum_constructor_unqualified_unknown(self) -> None:
        err = reject_type("let v = Ghost\n")
        assert "Ghost" in err.to_diagnostic().message

    def test_variant_constructor_unknown_field(self) -> None:
        err = reject_type("enum R\n  | Pass\nlet v = R.Pass(ghost: 1)\n")
        assert "ghost" in err.to_diagnostic().message

    def test_variant_constructor_missing_field(self) -> None:
        err = reject_type("enum R\n  | Fail(msg: text)\nlet v = R.Fail()\n")
        assert "msg" in err.to_diagnostic().message

    # --- List and Dict literals ---

    def test_list_lit_nonempty(self) -> None:
        r = accept_type("let xs = [1, 2]\n")
        assert _let_value_type(r, 0) == ListType(elem=IntType())

    def test_list_lit_empty_with_annotation(self) -> None:
        r = accept_type("let xs: list[int] = []\n")
        assert _let_value_type(r, 0) == ListType(elem=IntType())

    def test_list_lit_empty_without_annotation_error(self) -> None:
        err = reject_type("let xs = []\n")
        assert "annotation" in err.to_diagnostic().message

    def test_list_lit_inconsistent_types(self) -> None:
        err = reject_type('let xs = [1, "x"]\n')
        assert "inconsistent" in err.to_diagnostic().message

    def test_dict_lit_nonempty(self) -> None:
        r = accept_type("let d = {a: 1}\n")
        assert _let_value_type(r, 0) == DictType(value=IntType())

    def test_dict_lit_empty_with_annotation(self) -> None:
        r = accept_type("let d: dict[text, int] = {}\n")
        assert _let_value_type(r, 0) == DictType(value=IntType())

    def test_dict_lit_empty_without_annotation_error(self) -> None:
        err = reject_type("let d = {}\n")
        assert "annotation" in err.to_diagnostic().message

    def test_dict_lit_duplicate_key(self) -> None:
        err = reject_type("let d = {a: 1, a: 2}\n")
        assert "a" in err.to_diagnostic().message

    def test_dict_lit_two_entries_same_type(self) -> None:
        r = accept_type("let d = {a: 1, b: 2}\n")
        assert _let_value_type(r, 0) == DictType(value=IntType())

    # --- Case expression ---

    def test_case_expr_all_same_type(self) -> None:
        r = accept_type("let x = 1\nlet r = case x of\n  | v => v\n")
        decl = r.resolved.program.body[1]
        assert isinstance(decl, LetDecl)
        assert isinstance(decl.value, CaseExpr)
        assert r.node_types[decl.value.node_id] == IntType()

    def test_case_expr_int_decimal_widening(self) -> None:
        r = accept_type(
            "let x = 1\nlet r = case x of\n  | 1 => 1\n  | _ => 1.5\n"
        )
        decl = r.resolved.program.body[1]
        assert isinstance(decl, LetDecl)
        assert isinstance(decl.value, CaseExpr)
        assert r.node_types[decl.value.node_id] == DecimalType()

    def test_case_expr_type_mismatch(self) -> None:
        err = reject_type(
            'let x = 1\nlet r = case x of\n  | 1 => 1\n  | _ => "x"\n'
        )
        assert "incompatible" in err.to_diagnostic().message

    def test_case_expr_decimal_then_int(self) -> None:
        """branch[0]=decimal, branch[1]=int → stays decimal."""
        r = accept_type(
            "let x = 1\nlet r = case x of\n  | 1 => 1.5\n  | _ => 1\n"
        )
        decl = r.resolved.program.body[1]
        assert isinstance(decl, LetDecl)
        assert isinstance(decl.value, CaseExpr)
        assert r.node_types[decl.value.node_id] == DecimalType()

    def test_case_expr_two_same_type_branches(self) -> None:
        """Two branches of the same type: the second hits the ``continue`` path."""
        r = accept_type(
            "let x = 1\nlet r = case x of\n  | 1 => 1\n  | _ => 2\n"
        )
        decl = r.resolved.program.body[1]
        assert isinstance(decl, LetDecl)
        assert isinstance(decl.value, CaseExpr)
        assert r.node_types[decl.value.node_id] == IntType()

    # --- Template interpolation ---

    def test_interp_segment_with_render(self) -> None:
        """A template interpolation with a known renderer name type-checks."""
        r = accept_type('let x = 1\nlet t = "${x as raw}"\n')
        assert r.resolved.program is not None

    def test_interp_segment_unknown_render(self) -> None:
        err = reject_type('let x = 1\nlet t = "${x as markdown}"\n')
        assert "markdown" in err.to_diagnostic().message

    # --- set stmt with annotated var ---

    def test_set_stmt_type_check(self) -> None:
        r = accept_type("var n: int = 0\nset n = 5\n")
        assert r.resolved.program is not None

    # --- Pattern binding with non-enum subject type ---

    def test_constructor_pattern_non_enum_subject_fieldless(self) -> None:
        """A fieldless constructor pattern on a non-enum subject is rejected."""
        err = reject_type("let x = 1\ncase x of\n  | Pass => pass\n")
        assert isinstance(err, AglTypeError)

    def test_constructor_pattern_non_enum_subject_with_field_binding(self) -> None:
        """A field-binding constructor pattern on a non-enum subject is rejected
        (regression: the bound field var must not be left untyped)."""
        err = reject_type(
            "let x = 1\ncase x of\n  | Cons(f: v) => print v\n"
        )
        assert isinstance(err, AglTypeError)

    def test_constructor_pattern_enum_unknown_field(self) -> None:
        err = reject_type(
            "enum R\n  | Pass\nlet r: R = Pass\n"
            "case r of\n  | Pass(ghost: v) => pass\n"
        )
        assert "ghost" in err.to_diagnostic().message

    def test_enum_constructor_unqualified_single_no_expected(self) -> None:
        r = accept_type("enum E\n  | Sole\nlet v = Sole\n")
        assert isinstance(_let_value_type(r, 1), EnumType)


# ============================================================
# checker.py — remaining AST-built tests (genuinely unparseable in M3)
# ============================================================


class TestCheckerViaAstRemaining:
    """Checker paths not reachable through the M3 grammar.

    Two kinds of construct remain AST-only:

    * An *empty* case expression (``case x of`` with zero branches) — the
      grammar requires at least one branch, so the no-branch result-type
      fallbacks can only be reached by building the AST directly.
    * A constructor call with **duplicate argument names** — the parser
      (``transform``) rejects these as a syntax error *before* type-checking, so
      the checker's defensive duplicate-arg detection is unreachable via source
      and is pinned here as a unit contract.

    Every other former ViaAst test was re-expressed as a source-level test above.
    """

    def test_record_constructor_duplicate_arg(self) -> None:
        """The checker's record-constructor duplicate-arg guard (defensive: the
        parser normally rejects duplicates first)."""
        from agm.agl.syntax.nodes import Constructor, NamedArg, RecordDef

        rec = RecordDef(
            name="Pt",
            fields=(_tc_field("x", _tc_int_t()),),
            span=_tc_sp(), node_id=_tc_nid(),
        )
        ctor = Constructor(
            qualifier=None, name="Pt",
            args=(
                NamedArg(name="x", value=_tc_intlit(1), span=_tc_sp(), node_id=_tc_nid()),
                NamedArg(name="x", value=_tc_intlit(2), span=_tc_sp(), node_id=_tc_nid()),
            ),
            span=_tc_sp(), node_id=_tc_nid(),
        )
        with pytest.raises(AglTypeError) as exc_info:
            resolve_and_check(rec, _tc_let("p", ctor))
        assert "x" in exc_info.value.to_diagnostic().message

    def test_variant_constructor_duplicate_arg(self) -> None:
        """The checker's enum-variant-constructor duplicate-arg guard
        (defensive: the parser normally rejects duplicates first)."""
        from agm.agl.syntax.nodes import (
            Constructor,
            EnumDef,
            NamedArg,
            VariantDef,
        )

        enum_def = EnumDef(
            name="R",
            variants=(
                VariantDef(
                    name="Fail",
                    fields=(_tc_field("msg", _tc_text_t()),),
                    span=_tc_sp(), node_id=_tc_nid(),
                ),
            ),
            span=_tc_sp(), node_id=_tc_nid(),
        )
        ctor = Constructor(
            qualifier="R", name="Fail",
            args=(
                NamedArg(name="msg", value=_tc_strlit("a"), span=_tc_sp(), node_id=_tc_nid()),
                NamedArg(name="msg", value=_tc_strlit("b"), span=_tc_sp(), node_id=_tc_nid()),
            ),
            span=_tc_sp(), node_id=_tc_nid(),
        )
        with pytest.raises(AglTypeError) as exc_info:
            resolve_and_check(enum_def, _tc_let("v", ctor))
        assert "msg" in exc_info.value.to_diagnostic().message

    def test_case_expr_no_branches_with_expected(self) -> None:
        """Empty case expression with an expected type yields that type."""
        let_x = _tc_let("x", _tc_intlit(1))
        case_expr = CaseExpr(
            subject=_tc_varref("x"), branches=(), span=_tc_sp(), node_id=_tc_nid()
        )
        let_r = _tc_let("r", case_expr, type_ann=_tc_int_t())
        r = resolve_and_check(let_x, let_r)
        assert r.node_types[case_expr.node_id] == IntType()

    def test_case_expr_no_branches_no_expected_defaults_text(self) -> None:
        """Empty case expression with no expected type defaults to text."""
        let_x = _tc_let("x", _tc_intlit(1))
        case_expr = CaseExpr(
            subject=_tc_varref("x"), branches=(), span=_tc_sp(), node_id=_tc_nid()
        )
        let_r = _tc_let("r", case_expr)
        r = resolve_and_check(let_x, let_r)
        assert r.node_types[case_expr.node_id] == TextType()

    def test_require_binding_type_none_raises_assertion(self) -> None:
        """Internal invariant: an unset binding type is an AssertionError.

        Directly exercises the ``_require_binding_type`` guard, whose ``None``
        branch is unreachable through normal checking (every reachable binding
        has its type recorded first), so it is verified as a unit contract here.
        """
        from agm.agl.scope.resolver import resolve
        from agm.agl.scope.symbols import BinderKind, BindingRef
        from agm.agl.typecheck.checker import _Checker
        from agm.agl.typecheck.env import TypeEnvironment

        resolved = resolve(_tc_program())
        checker = _Checker(
            env=TypeEnvironment(), resolved=resolved, capabilities=default_capabilities()
        )
        ref = BindingRef(
            name="ghost",
            mutable=False,
            decl_span=_tc_sp(),
            decl_node_id=999999,
            kind=BinderKind.let_binding,
        )
        with pytest.raises(AssertionError):
            checker._require_binding_type(ref)


# ---------------------------------------------------------------------------
# M4b: §8.1 exception-schema conformance (catchability matrix)
# ---------------------------------------------------------------------------


class TestBuiltinExceptionSchemaConformance:
    """§8.1 normative table: every exception type exists with the right fields.

    Each test drives the catchability matrix: (a) the type name is registered,
    (b) ``catch TypeName as e`` type-checks, (c) every documented field is
    accessible via ``e.field``, and (d) no undocumented fields exist.
    """

    def _assert_exception_fields(
        self, exc_name: str, expected_fields: dict[str, type]
    ) -> None:
        """Assert *exc_name* has exactly *expected_fields* (names + AgL type kinds)."""
        from agm.agl.typecheck.types import BUILTIN_EXCEPTIONS, ExceptionType

        exc_type = BUILTIN_EXCEPTIONS.get(exc_name)
        assert exc_type is not None, f"{exc_name} not in BUILTIN_EXCEPTIONS"
        assert isinstance(exc_type, ExceptionType)
        actual = {name: type(t) for name, t in exc_type.fields.items()}
        assert actual == expected_fields, (
            f"{exc_name}: expected fields {expected_fields}, got {actual}"
        )

    def test_exception_base_has_message_and_trace_id_only(self) -> None:
        """§8.1: Exception base has exactly message: text and trace_id: text."""
        from agm.agl.typecheck.types import TextType

        self._assert_exception_fields(
            "Exception",
            {"message": TextType, "trace_id": TextType},
        )

    def test_agent_call_error_fields(self) -> None:
        """§8.1: AgentCallError has agent/cause/metadata + base fields."""
        from agm.agl.typecheck.types import JsonType, TextType

        self._assert_exception_fields(
            "AgentCallError",
            {
                "message": TextType,
                "trace_id": TextType,
                "agent": TextType,
                "cause": TextType,
                "metadata": JsonType,
            },
        )

    def test_agent_parse_error_fields(self) -> None:
        """§8.1: AgentParseError full field set."""
        from agm.agl.typecheck.types import IntType, JsonType, TextType

        self._assert_exception_fields(
            "AgentParseError",
            {
                "message": TextType,
                "trace_id": TextType,
                "agent": TextType,
                "target_type": TextType,
                "expected_schema": JsonType,
                "raw": TextType,
                "normalized_raw": TextType,
                "validation_errors": JsonType,
                "attempts": IntType,
                "metadata": JsonType,
            },
        )

    def test_exec_error_fields(self) -> None:
        """§8.1: ExecError has command/exit_code/stdout/stderr/timed_out."""
        from agm.agl.typecheck.types import BoolType, IntType, TextType

        self._assert_exception_fields(
            "ExecError",
            {
                "message": TextType,
                "trace_id": TextType,
                "command": TextType,
                "exit_code": IntType,
                "stdout": TextType,
                "stderr": TextType,
                "timed_out": BoolType,
            },
        )

    def test_max_iterations_exceeded_fields(self) -> None:
        """§8.1: MaxIterationsExceeded has limit/condition/last_condition_value/metadata."""
        from agm.agl.typecheck.types import BoolType, IntType, JsonType, TextType

        self._assert_exception_fields(
            "MaxIterationsExceeded",
            {
                "message": TextType,
                "trace_id": TextType,
                "limit": IntType,
                "condition": TextType,
                "last_condition_value": BoolType,
                "metadata": JsonType,
            },
        )

    def test_match_error_fields(self) -> None:
        """§8.1: MatchError has scrutinee_type: text, scrutinee: json."""
        from agm.agl.typecheck.types import JsonType, TextType

        self._assert_exception_fields(
            "MatchError",
            {
                "message": TextType,
                "trace_id": TextType,
                "scrutinee_type": TextType,
                "scrutinee": JsonType,
            },
        )

    def test_arithmetic_error_fields(self) -> None:
        """§8.1: ArithmeticError has operation: text."""
        from agm.agl.typecheck.types import TextType

        self._assert_exception_fields(
            "ArithmeticError",
            {
                "message": TextType,
                "trace_id": TextType,
                "operation": TextType,
            },
        )

    def test_undefined_variable_error_fields(self) -> None:
        """§8.1: UndefinedVariableError has name: text."""
        from agm.agl.typecheck.types import TextType

        self._assert_exception_fields(
            "UndefinedVariableError",
            {
                "message": TextType,
                "trace_id": TextType,
                "name": TextType,
            },
        )

    def test_immutable_binding_error_fields(self) -> None:
        """§8.1: ImmutableBindingError has name: text and operation: text."""
        from agm.agl.typecheck.types import TextType

        self._assert_exception_fields(
            "ImmutableBindingError",
            {
                "message": TextType,
                "trace_id": TextType,
                "name": TextType,
                "operation": TextType,
            },
        )

    def test_type_error_fields(self) -> None:
        """§8.1: TypeError has base fields only (no additional fields listed)."""
        from agm.agl.typecheck.types import TextType

        self._assert_exception_fields(
            "TypeError",
            {
                "message": TextType,
                "trace_id": TextType,
            },
        )

    def test_abort_fields(self) -> None:
        """§8.1: Abort has base fields only (message is its sole declared field)."""
        from agm.agl.typecheck.types import TextType

        self._assert_exception_fields(
            "Abort",
            {
                "message": TextType,
                "trace_id": TextType,
            },
        )

    def test_validation_error_not_a_catchable_exception(self) -> None:
        """ValidationError is NOT listed in §8.1 — it is a Python-level record
        inside AgentParseError.validation_errors, not a catchable AgL exception.
        """
        from agm.agl.typecheck.types import BUILTIN_EXCEPTIONS

        assert "ValidationError" not in BUILTIN_EXCEPTIONS

    def test_no_extra_exception_names(self) -> None:
        """§8.1 exact name set — no exceptions beyond the 10 listed (+ base)."""
        from agm.agl.typecheck.types import BUILTIN_EXCEPTIONS

        expected = frozenset({
            "Exception",
            "AgentCallError",
            "AgentParseError",
            "ExecError",
            "MaxIterationsExceeded",
            "MatchError",
            "TypeError",
            "ArithmeticError",
            "UndefinedVariableError",
            "ImmutableBindingError",
            "Abort",
        })
        assert frozenset(BUILTIN_EXCEPTIONS.keys()) == expected

    # --- Catchability: catch TypeName → field access type-checks ---

    def test_catch_agent_call_error_field_agent(self) -> None:
        """§8.1 catchability: AgentCallError.agent is well-typed under catch."""
        r = accept_type(
            "try\n  pass\ncatch AgentCallError as e =>\n  print e.agent\n"
        )
        assert r.resolved.program is not None

    def test_catch_agent_call_error_field_cause(self) -> None:
        r = accept_type(
            "try\n  pass\ncatch AgentCallError as e =>\n  print e.cause\n"
        )
        assert r.resolved.program is not None

    def test_catch_agent_call_error_field_metadata(self) -> None:
        r = accept_type(
            "try\n  pass\ncatch AgentCallError as e =>\n  print e.metadata\n"
        )
        assert r.resolved.program is not None

    def test_catch_agent_parse_error_fields(self) -> None:
        """§8.1: every AgentParseError field is accessible under its catch."""
        for field_name in (
            "agent", "target_type", "expected_schema", "raw",
            "normalized_raw", "validation_errors", "attempts", "metadata",
        ):
            r = accept_type(
                "try\n  pass\n"
                f"catch AgentParseError as e =>\n"
                f"  print e.{field_name}\n"
            )
            assert r.resolved.program is not None, f"field {field_name!r} not accessible"

    def test_catch_undefined_variable_error(self) -> None:
        """§8.1: UndefinedVariableError is catchable with name field."""
        r = accept_type(
            "try\n  pass\ncatch UndefinedVariableError as e =>\n  print e.name\n"
        )
        assert r.resolved.program is not None

    def test_catch_immutable_binding_error_name(self) -> None:
        """§8.1: ImmutableBindingError.name is accessible."""
        r = accept_type(
            "try\n  pass\ncatch ImmutableBindingError as e =>\n  print e.name\n"
        )
        assert r.resolved.program is not None

    def test_catch_immutable_binding_error_operation(self) -> None:
        """§8.1: ImmutableBindingError.operation is accessible."""
        r = accept_type(
            "try\n  pass\ncatch ImmutableBindingError as e =>\n  print e.operation\n"
        )
        assert r.resolved.program is not None

    def test_catch_exception_base_only_base_fields(self) -> None:
        """§8.1: catch Exception as e → only base fields accessible."""
        r = accept_type(
            "try\n  pass\ncatch Exception as e =>\n  print e.message\n"
        )
        assert r.resolved.program is not None

    def test_catch_exception_base_subtype_field_rejected(self) -> None:
        """§8.1: accessing e.raw under catch Exception is a static error."""
        err = reject_type(
            "try\n  pass\ncatch Exception as e =>\n  print e.raw\n"
        )
        assert "raw" in err.to_diagnostic().message

    def test_catch_wildcard_base_fields_accessible(self) -> None:
        """§8.1: catch _ as e → only base fields (message/trace_id) accessible."""
        r = accept_type(
            "try\n  pass\ncatch _ as e =>\n  print e.message\n"
        )
        assert r.resolved.program is not None

    def test_catch_wildcard_subtype_field_rejected(self) -> None:
        """§8.1: accessing subtype field under catch _ is a static error."""
        err = reject_type(
            "try\n  pass\ncatch _ as e =>\n  print e.raw\n"
        )
        assert "raw" in err.to_diagnostic().message
