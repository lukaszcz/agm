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

import decimal

import pytest

from agm.agl.capabilities import HostCapabilities
from agm.agl.parser import parse_program
from agm.agl.scope import resolve
from agm.agl.syntax.nodes import (
    ELSE,
    AgentCall,
    BinaryOp,
    BinOp,
    BoolLit,
    CaseExpr,
    CaseExprBranch,
    CaseStmt,
    CaseStmtBranch,
    CatchClause,
    Constructor,
    DecimalLit,
    DictEntry,
    DictLit,
    EnumDef,
    Expr,
    ExprStmt,
    FieldAccess,
    FieldDef,
    IfBranch,
    IfStmt,
    InterpSegment,
    IntLit,
    IsTest,
    LetDecl,
    ListLit,
    NamedArg,
    NullLit,
    PassStmt,
    PrintStmt,
    Program,
    Raise,
    RecordDef,
    SetStmt,
    Stmt,
    StringLit,
    Template,
    TryCatch,
    TypeAlias,
    UnaryNeg,
    UnaryNot,
    VarDecl,
    VariantDef,
    VarPattern,
    VarRef,
    WildcardPattern,
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
        from agm.agl.syntax.nodes import SetStmt

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
        from agm.agl.syntax.nodes import SetStmt
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


# ---------------------------------------------------------------------------
# Arithmetic / operators (deferred: operators not parseable in M1)
# ---------------------------------------------------------------------------


class TestOperators:
    @pytest.mark.skip(reason="binary + not parseable in M1 — deferred to M3")
    def test_text_plus_int_error(self) -> None:
        err = reject_type('let x = "a" + 1')
        line, msg = diag(err)
        assert line == 1

    @pytest.mark.skip(reason="unary minus not parseable in M1 — deferred to M3")
    def test_unary_minus_text(self) -> None:
        err = reject_type('let x = -"a"')
        line, msg = diag(err)
        assert line == 1

    @pytest.mark.skip(reason="comparison operators not parseable in M1 — deferred to M3")
    def test_ord_on_bool(self) -> None:
        err = reject_type("let x = (true < false)")
        line, msg = diag(err)
        assert line == 1

    @pytest.mark.skip(reason="equality operator not parseable in M1 — deferred to M3")
    def test_eq_type_mismatch(self) -> None:
        err = reject_type('let x = (1 = "one")')
        line, msg = diag(err)
        assert line == 1

    @pytest.mark.skip(reason="'in' operator not parseable in M1 — deferred to M3")
    def test_in_bad_rhs(self) -> None:
        err = reject_type("let x = 1 in 2")
        line, msg = diag(err)
        assert line == 1

    @pytest.mark.skip(reason="'is' test not parseable in M1 — deferred to M3")
    def test_is_on_non_enum(self) -> None:
        err = reject_type(
            "enum R\n  | Pass\n"
            "let n = 1\n"
            "let b = n is Pass\n"
        )
        line, msg = diag(err)
        assert line == 4

    @pytest.mark.skip(reason="'is' test not parseable in M1 — deferred to M3")
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
    @pytest.mark.skip(reason="try-catch not parseable in M1 — deferred to M4")
    def test_wildcard_catch_subtype_field_error(self) -> None:
        err = reject_type(
            "try\n  pass\n"
            "catch _ as e =>\n"
            "  print e.raw\n"
        )
        line, msg = diag(err)
        assert line == 4
        assert "raw" in msg

    @pytest.mark.skip(reason="try-catch not parseable in M1 — deferred to M4")
    def test_raise_exception_base_not_constructible(self) -> None:
        err = reject_type('raise Exception(message: "x")')
        line, msg = diag(err)
        assert line == 1
        assert "Exception" in msg


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


def _tc_boollit(v: bool = True) -> BoolLit:
    return BoolLit(value=v, span=_tc_sp(), node_id=_tc_nid())


def _tc_declit(v: str = "1.5") -> DecimalLit:
    return DecimalLit(value=decimal.Decimal(v), span=_tc_sp(), node_id=_tc_nid())


def _tc_strlit(v: str = "hi") -> StringLit:
    return StringLit(value=v, span=_tc_sp(), node_id=_tc_nid())


def _tc_nulllit() -> NullLit:
    return NullLit(span=_tc_sp(), node_id=_tc_nid())


def _tc_varref(name: str) -> VarRef:
    return VarRef(name=name, span=_tc_sp(), node_id=_tc_nid())


def _tc_let(name: str, value: Expr, type_ann: TypeExpr | None = None) -> LetDecl:
    return LetDecl(name=name, type_ann=type_ann, value=value, span=_tc_sp(), node_id=_tc_nid())


def _tc_var(name: str, value: Expr, type_ann: TypeExpr | None = None) -> VarDecl:
    return VarDecl(name=name, type_ann=type_ann, value=value, span=_tc_sp(), node_id=_tc_nid())


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


def reject_ast(
    *stmts: Stmt,
    caps: HostCapabilities | None = None,
) -> AglTypeError:
    with pytest.raises(AglTypeError) as exc_info:
        resolve_and_check(*stmts, caps=caps)
    return exc_info.value


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
# checker.py — statement-level paths
# ============================================================


class TestCheckerStatementsViaAst:
    """Exercise statement dispatching code paths."""

    def test_pass_stmt(self) -> None:
        stmt = PassStmt(span=_tc_sp(), node_id=_tc_nid())
        r = resolve_and_check(stmt)
        assert r.resolved.program is not None

    def test_expr_stmt(self) -> None:
        stmt = ExprStmt(expr=_tc_intlit(), span=_tc_sp(), node_id=_tc_nid())
        r = resolve_and_check(stmt)
        assert r.resolved.program is not None

    def test_raise_non_exception_expr(self) -> None:
        """Raise with any non-Exception expression is not an error (checker only errors on
        abstract Exception base)."""
        raise_stmt = Raise(exc=_tc_strlit("oops"), span=_tc_sp(), node_id=_tc_nid())
        r = resolve_and_check(raise_stmt)
        assert r.resolved.program is not None

    def test_raise_abstract_base_error(self) -> None:
        """Raise of the abstract Exception base type should fail.
        We construct this by catching a wildcard (giving binder abstract Exception type)
        then re-raising it."""
        # try: pass; catch _ as e => raise e
        # The catch-all gives 'e' the abstract ExceptionType("Exception"),
        # and _check_raise should reject it.
        raise_stmt = Raise(exc=_tc_varref("e"), span=_tc_sp(), node_id=_tc_nid())
        clause = CatchClause(
            exc_type=None, binding="e",
            body=(raise_stmt,),
            span=_tc_sp(), node_id=_tc_nid(),
        )
        try_stmt = TryCatch(
            body=(PassStmt(span=_tc_sp(), node_id=_tc_nid()),),
            handlers=(clause,), span=_tc_sp(), node_id=_tc_nid(),
        )
        err = reject_ast(try_stmt)
        assert "Exception" in err.to_diagnostic().message

    def test_do_until(self) -> None:
        from agm.agl.syntax.nodes import DoUntil
        var_n = VarDecl(
            name="n", type_ann=None, value=_tc_intlit(0), span=_tc_sp(), node_id=_tc_nid()
        )
        do_stmt = DoUntil(
            limit=5,
            body=(PassStmt(span=_tc_sp(), node_id=_tc_nid()),),
            condition=_tc_boollit(True),
            span=_tc_sp(),
            node_id=_tc_nid(),
        )
        r = resolve_and_check(var_n, do_stmt)
        assert r.resolved.program is not None

    def test_if_stmt_with_else(self) -> None:
        branch_if = IfBranch(
            cond=_tc_boollit(True),
            body=(PassStmt(span=_tc_sp(), node_id=_tc_nid()),),
            span=_tc_sp(),
            node_id=_tc_nid(),
        )
        branch_else = IfBranch(
            cond=ELSE,
            body=(PassStmt(span=_tc_sp(), node_id=_tc_nid()),),
            span=_tc_sp(),
            node_id=_tc_nid(),
        )
        if_stmt = IfStmt(branches=(branch_if, branch_else), span=_tc_sp(), node_id=_tc_nid())
        r = resolve_and_check(if_stmt)
        assert r.resolved.program is not None

    def test_if_stmt_without_else(self) -> None:
        branch = IfBranch(
            cond=_tc_boollit(True),
            body=(PassStmt(span=_tc_sp(), node_id=_tc_nid()),),
            span=_tc_sp(),
            node_id=_tc_nid(),
        )
        if_stmt = IfStmt(branches=(branch,), span=_tc_sp(), node_id=_tc_nid())
        r = resolve_and_check(if_stmt)
        assert r.resolved.program is not None

    def test_case_stmt_wildcard(self) -> None:
        let_x = _tc_let("x", _tc_intlit(1))
        branch = CaseStmtBranch(
            pattern=WildcardPattern(span=_tc_sp(), node_id=_tc_nid()),
            body=(PassStmt(span=_tc_sp(), node_id=_tc_nid()),),
            span=_tc_sp(),
            node_id=_tc_nid(),
        )
        case_stmt = CaseStmt(
            subject=_tc_varref("x"), branches=(branch,), span=_tc_sp(), node_id=_tc_nid()
        )
        r = resolve_and_check(let_x, case_stmt)
        assert r.resolved.program is not None

    def test_case_stmt_var_pattern(self) -> None:
        enum_def = EnumDef(
            name="R",
            variants=(VariantDef(name="Pass", fields=(), span=_tc_sp(), node_id=_tc_nid()),),
            span=_tc_sp(),
            node_id=_tc_nid(),
        )
        # Construct R = Pass via constructor
        ctor = Constructor(qualifier=None, name="Pass", args=(), span=_tc_sp(), node_id=_tc_nid())
        let_r = _tc_let("r", ctor, type_ann=_tc_name_t("R"))
        pv = VarPattern(name="v", span=_tc_sp(), node_id=_tc_nid())
        branch = CaseStmtBranch(
            pattern=pv,
            body=(PassStmt(span=_tc_sp(), node_id=_tc_nid()),),
            span=_tc_sp(),
            node_id=_tc_nid(),
        )
        case_stmt = CaseStmt(
            subject=_tc_varref("r"), branches=(branch,), span=_tc_sp(), node_id=_tc_nid()
        )
        r = resolve_and_check(enum_def, let_r, case_stmt)
        assert r.resolved.program is not None

    def test_case_stmt_literal_pattern(self) -> None:
        let_x = _tc_let("x", _tc_intlit(1))
        from agm.agl.syntax.nodes import LiteralPattern
        branch = CaseStmtBranch(
            pattern=LiteralPattern(literal=_tc_intlit(1), span=_tc_sp(), node_id=_tc_nid()),
            body=(PassStmt(span=_tc_sp(), node_id=_tc_nid()),),
            span=_tc_sp(),
            node_id=_tc_nid(),
        )
        case_stmt = CaseStmt(
            subject=_tc_varref("x"), branches=(branch,), span=_tc_sp(), node_id=_tc_nid()
        )
        r = resolve_and_check(let_x, case_stmt)
        assert r.resolved.program is not None

    def test_case_stmt_constructor_pattern_enum(self) -> None:
        from agm.agl.syntax.nodes import ConstructorPattern, PatternField
        enum_def = EnumDef(
            name="R",
            variants=(
                VariantDef(
                    name="Fail",
                    fields=(_tc_field("reason", _tc_text_t()),),
                    span=_tc_sp(),
                    node_id=_tc_nid(),
                ),
            ),
            span=_tc_sp(),
            node_id=_tc_nid(),
        )
        ctor = Constructor(
            qualifier=None,
            name="Fail",
            args=(
                NamedArg(name="reason", value=_tc_strlit("x"), span=_tc_sp(), node_id=_tc_nid()),
            ),
            span=_tc_sp(),
            node_id=_tc_nid(),
        )
        let_r = _tc_let("r", ctor, type_ann=_tc_name_t("R"))
        pv = VarPattern(name="msg", span=_tc_sp(), node_id=_tc_nid())
        pf = PatternField(name="reason", pattern=pv, span=_tc_sp(), node_id=_tc_nid())
        ctor_p = ConstructorPattern(
            qualifier=None, name="Fail", fields=(pf,), span=_tc_sp(), node_id=_tc_nid()
        )
        branch = CaseStmtBranch(
            pattern=ctor_p,
            body=(PassStmt(span=_tc_sp(), node_id=_tc_nid()),),
            span=_tc_sp(),
            node_id=_tc_nid(),
        )
        case_stmt = CaseStmt(
            subject=_tc_varref("r"), branches=(branch,), span=_tc_sp(), node_id=_tc_nid()
        )
        r = resolve_and_check(enum_def, let_r, case_stmt)
        assert r.resolved.program is not None

    def test_try_catch_wildcard(self) -> None:
        clause = CatchClause(
            exc_type=None, binding=None,
            body=(PassStmt(span=_tc_sp(), node_id=_tc_nid()),),
            span=_tc_sp(), node_id=_tc_nid(),
        )
        try_stmt = TryCatch(
            body=(PassStmt(span=_tc_sp(), node_id=_tc_nid()),),
            handlers=(clause,), span=_tc_sp(), node_id=_tc_nid(),
        )
        r = resolve_and_check(try_stmt)
        assert r.resolved.program is not None

    def test_try_catch_specific_exception(self) -> None:
        clause = CatchClause(
            exc_type="AgentCallError", binding="err",
            body=(PassStmt(span=_tc_sp(), node_id=_tc_nid()),),
            span=_tc_sp(), node_id=_tc_nid(),
        )
        try_stmt = TryCatch(
            body=(PassStmt(span=_tc_sp(), node_id=_tc_nid()),),
            handlers=(clause,), span=_tc_sp(), node_id=_tc_nid(),
        )
        r = resolve_and_check(try_stmt)
        assert r.resolved.program is not None

    def test_try_catch_unknown_exception_type(self) -> None:
        clause = CatchClause(
            exc_type="GhostException", binding=None,
            body=(PassStmt(span=_tc_sp(), node_id=_tc_nid()),),
            span=_tc_sp(), node_id=_tc_nid(),
        )
        try_stmt = TryCatch(
            body=(PassStmt(span=_tc_sp(), node_id=_tc_nid()),),
            handlers=(clause,), span=_tc_sp(), node_id=_tc_nid(),
        )
        err = reject_ast(try_stmt)
        assert "GhostException" in err.to_diagnostic().message

    def test_try_catch_exc_type_is_underscore(self) -> None:
        """'_' as exc_type is treated as wildcard (same as None)."""
        clause = CatchClause(
            exc_type="_", binding="e",
            body=(PassStmt(span=_tc_sp(), node_id=_tc_nid()),),
            span=_tc_sp(), node_id=_tc_nid(),
        )
        try_stmt = TryCatch(
            body=(PassStmt(span=_tc_sp(), node_id=_tc_nid()),),
            handlers=(clause,), span=_tc_sp(), node_id=_tc_nid(),
        )
        r = resolve_and_check(try_stmt)
        assert r.resolved.program is not None

    def test_record_def_and_enum_def_pass_through(self) -> None:
        """RecordDef and EnumDef in _check_stmt are pass-through (handled by pre-pass)."""
        rec = RecordDef(
            name="Point", fields=(_tc_field("x", _tc_int_t()),), span=_tc_sp(), node_id=_tc_nid()
        )
        vd = VariantDef(name="Ok", fields=(), span=_tc_sp(), node_id=_tc_nid())
        enum_def = EnumDef(name="Status", variants=(vd,), span=_tc_sp(), node_id=_tc_nid())
        alias = TypeAlias(
            name="PAlias", type_expr=_tc_name_t("Point"), span=_tc_sp(), node_id=_tc_nid()
        )
        r = resolve_and_check(rec, enum_def, alias)
        assert r.resolved.program is not None


# ============================================================
# checker.py — expression type inference
# ============================================================


class TestCheckerExprsViaAst:
    """Exercise expression inference and error paths."""

    def test_null_literal_is_json(self) -> None:
        null_expr = _tc_nulllit()
        let_stmt = _tc_let("x", null_expr, type_ann=_tc_json_t())
        r = resolve_and_check(let_stmt)
        assert r.node_types[null_expr.node_id] == JsonType()

    def test_decimal_lit(self) -> None:
        dec_expr = _tc_declit("3.14")
        let_stmt = _tc_let("x", dec_expr)
        r = resolve_and_check(let_stmt)
        assert r.node_types[dec_expr.node_id] == DecimalType()

    def test_bool_lit(self) -> None:
        bool_expr = _tc_boollit(True)
        let_stmt = _tc_let("x", bool_expr)
        r = resolve_and_check(let_stmt)
        assert r.node_types[bool_expr.node_id] == BoolType()

    def test_string_lit(self) -> None:
        str_expr = _tc_strlit("hello")
        let_stmt = _tc_let("x", str_expr)
        r = resolve_and_check(let_stmt)
        assert r.node_types[str_expr.node_id] == TextType()

    def test_unary_not(self) -> None:
        let_b = _tc_let("b", _tc_boollit(True))
        expr = UnaryNot(operand=_tc_varref("b"), span=_tc_sp(), node_id=_tc_nid())
        let_r = _tc_let("r", expr)
        r = resolve_and_check(let_b, let_r)
        assert r.node_types[let_r.value.node_id] == BoolType()

    def test_unary_neg_int(self) -> None:
        let_n = _tc_let("n", _tc_intlit(1))
        expr = UnaryNeg(operand=_tc_varref("n"), span=_tc_sp(), node_id=_tc_nid())
        let_r = _tc_let("r", expr)
        r = resolve_and_check(let_n, let_r)
        assert r.node_types[let_r.value.node_id] == IntType()

    def test_unary_neg_decimal(self) -> None:
        let_n = _tc_let("n", _tc_declit("1.5"))
        expr = UnaryNeg(operand=_tc_varref("n"), span=_tc_sp(), node_id=_tc_nid())
        let_r = _tc_let("r", expr)
        r = resolve_and_check(let_n, let_r)
        assert r.node_types[let_r.value.node_id] == DecimalType()

    def test_unary_neg_wrong_type(self) -> None:
        let_s = _tc_let("s", _tc_strlit("x"))
        expr = UnaryNeg(operand=_tc_varref("s"), span=_tc_sp(), node_id=_tc_nid())
        err = reject_ast(let_s, _tc_let("r", expr))
        assert "numeric" in err.to_diagnostic().message

    def test_binary_add_int_int(self) -> None:
        let_a = _tc_let("a", _tc_intlit(1))
        let_b = _tc_let("b", _tc_intlit(2))
        binop = BinaryOp(
            op=BinOp.ADD, left=_tc_varref("a"), right=_tc_varref("b"),
            span=_tc_sp(), node_id=_tc_nid(),
        )
        let_r = _tc_let("r", binop)
        r = resolve_and_check(let_a, let_b, let_r)
        assert r.node_types[let_r.value.node_id] == IntType()

    def test_binary_add_int_decimal(self) -> None:
        let_a = _tc_let("a", _tc_intlit(1))
        let_b = _tc_let("b", _tc_declit("1.5"))
        binop = BinaryOp(
            op=BinOp.ADD, left=_tc_varref("a"), right=_tc_varref("b"),
            span=_tc_sp(), node_id=_tc_nid(),
        )
        let_r = _tc_let("r", binop)
        r = resolve_and_check(let_a, let_b, let_r)
        assert r.node_types[let_r.value.node_id] == DecimalType()

    def test_binary_add_text_text(self) -> None:
        let_a = _tc_let("a", _tc_strlit("x"))
        let_b = _tc_let("b", _tc_strlit("y"))
        binop = BinaryOp(
            op=BinOp.ADD, left=_tc_varref("a"), right=_tc_varref("b"),
            span=_tc_sp(), node_id=_tc_nid(),
        )
        let_r = _tc_let("r", binop)
        r = resolve_and_check(let_a, let_b, let_r)
        assert r.node_types[let_r.value.node_id] == TextType()

    def test_binary_add_type_mismatch(self) -> None:
        let_a = _tc_let("a", _tc_strlit("x"))
        let_b = _tc_let("b", _tc_intlit(1))
        binop = BinaryOp(
            op=BinOp.ADD, left=_tc_varref("a"), right=_tc_varref("b"),
            span=_tc_sp(), node_id=_tc_nid(),
        )
        err = reject_ast(let_a, let_b, _tc_let("r", binop))
        assert "+" in err.to_diagnostic().message

    def test_binary_sub(self) -> None:
        let_a = _tc_let("a", _tc_intlit(5))
        let_b = _tc_let("b", _tc_intlit(3))
        binop = BinaryOp(
            op=BinOp.SUB, left=_tc_varref("a"), right=_tc_varref("b"),
            span=_tc_sp(), node_id=_tc_nid(),
        )
        r = resolve_and_check(let_a, let_b, _tc_let("r", binop))
        assert r.node_types[binop.node_id] == IntType()

    def test_binary_sub_decimal(self) -> None:
        let_a = _tc_let("a", _tc_intlit(5))
        let_b = _tc_let("b", _tc_declit("3.0"))
        binop = BinaryOp(
            op=BinOp.SUB, left=_tc_varref("a"), right=_tc_varref("b"),
            span=_tc_sp(), node_id=_tc_nid(),
        )
        r = resolve_and_check(let_a, let_b, _tc_let("r", binop))
        assert r.node_types[binop.node_id] == DecimalType()

    def test_binary_sub_type_mismatch(self) -> None:
        let_a = _tc_let("a", _tc_strlit("x"))
        let_b = _tc_let("b", _tc_intlit(1))
        binop = BinaryOp(
            op=BinOp.SUB, left=_tc_varref("a"), right=_tc_varref("b"),
            span=_tc_sp(), node_id=_tc_nid(),
        )
        err = reject_ast(let_a, let_b, _tc_let("r", binop))
        assert "-" in err.to_diagnostic().message

    def test_binary_mul(self) -> None:
        let_a = _tc_let("a", _tc_intlit(2))
        let_b = _tc_let("b", _tc_intlit(3))
        binop = BinaryOp(
            op=BinOp.MUL, left=_tc_varref("a"), right=_tc_varref("b"),
            span=_tc_sp(), node_id=_tc_nid(),
        )
        r = resolve_and_check(let_a, let_b, _tc_let("r", binop))
        assert r.node_types[binop.node_id] == IntType()

    def test_binary_div(self) -> None:
        let_a = _tc_let("a", _tc_intlit(6))
        let_b = _tc_let("b", _tc_intlit(2))
        binop = BinaryOp(
            op=BinOp.DIV, left=_tc_varref("a"), right=_tc_varref("b"),
            span=_tc_sp(), node_id=_tc_nid(),
        )
        r = resolve_and_check(let_a, let_b, _tc_let("r", binop))
        assert r.node_types[binop.node_id] == DecimalType()

    def test_binary_div_non_numeric_error(self) -> None:
        let_a = _tc_let("a", _tc_strlit("x"))
        let_b = _tc_let("b", _tc_intlit(2))
        binop = BinaryOp(
            op=BinOp.DIV, left=_tc_varref("a"), right=_tc_varref("b"),
            span=_tc_sp(), node_id=_tc_nid(),
        )
        err = reject_ast(let_a, let_b, _tc_let("r", binop))
        assert "/" in err.to_diagnostic().message

    def test_binary_and(self) -> None:
        let_a = _tc_let("a", _tc_boollit(True))
        let_b = _tc_let("b", _tc_boollit(False))
        binop = BinaryOp(
            op=BinOp.AND, left=_tc_varref("a"), right=_tc_varref("b"),
            span=_tc_sp(), node_id=_tc_nid(),
        )
        r = resolve_and_check(let_a, let_b, _tc_let("r", binop))
        assert r.node_types[binop.node_id] == BoolType()

    def test_binary_or(self) -> None:
        let_a = _tc_let("a", _tc_boollit(True))
        let_b = _tc_let("b", _tc_boollit(False))
        binop = BinaryOp(
            op=BinOp.OR, left=_tc_varref("a"), right=_tc_varref("b"),
            span=_tc_sp(), node_id=_tc_nid(),
        )
        r = resolve_and_check(let_a, let_b, _tc_let("r", binop))
        assert r.node_types[binop.node_id] == BoolType()

    def test_binary_and_non_bool_left_error(self) -> None:
        # F7: 'and'/'or' require bool operands; a non-bool left operand is a
        # static error with the span on the offending (left) operand.
        let_a = _tc_let("a", _tc_intlit(1))
        let_b = _tc_let("b", _tc_boollit(True))
        left_ref = _tc_varref("a")
        binop = BinaryOp(
            op=BinOp.AND, left=left_ref, right=_tc_varref("b"),
            span=_tc_sp(line=9), node_id=_tc_nid(),
        )
        err = reject_ast(let_a, let_b, _tc_let("r", binop))
        d = err.to_diagnostic()
        assert "and" in d.message and "bool" in d.message
        # Span points at the left operand, not the whole BinaryOp.
        assert d.line == left_ref.span.start_line

    def test_binary_or_non_bool_right_error(self) -> None:
        # F7: a non-bool right operand is reported at the right operand's span.
        let_a = _tc_let("a", _tc_boollit(True))
        let_b = _tc_let("b", _tc_intlit(2))
        right_ref = _tc_varref("b")
        binop = BinaryOp(
            op=BinOp.OR, left=_tc_varref("a"), right=right_ref,
            span=_tc_sp(line=9), node_id=_tc_nid(),
        )
        err = reject_ast(let_a, let_b, _tc_let("r", binop))
        d = err.to_diagnostic()
        assert "or" in d.message and "bool" in d.message
        assert d.line == right_ref.span.start_line

    def test_binary_eq(self) -> None:
        let_a = _tc_let("a", _tc_intlit(1))
        let_b = _tc_let("b", _tc_intlit(2))
        binop = BinaryOp(
            op=BinOp.EQ, left=_tc_varref("a"), right=_tc_varref("b"),
            span=_tc_sp(), node_id=_tc_nid(),
        )
        r = resolve_and_check(let_a, let_b, _tc_let("r", binop))
        assert r.node_types[binop.node_id] == BoolType()

    def test_binary_eq_type_mismatch(self) -> None:
        let_a = _tc_let("a", _tc_strlit("x"))
        let_b = _tc_let("b", _tc_intlit(1))
        binop = BinaryOp(
            op=BinOp.EQ, left=_tc_varref("a"), right=_tc_varref("b"),
            span=_tc_sp(), node_id=_tc_nid(),
        )
        err = reject_ast(let_a, let_b, _tc_let("r", binop))
        assert "same type" in err.to_diagnostic().message

    def test_binary_neq(self) -> None:
        let_a = _tc_let("a", _tc_intlit(1))
        let_b = _tc_let("b", _tc_intlit(2))
        binop = BinaryOp(
            op=BinOp.NEQ, left=_tc_varref("a"), right=_tc_varref("b"),
            span=_tc_sp(), node_id=_tc_nid(),
        )
        r = resolve_and_check(let_a, let_b, _tc_let("r", binop))
        assert r.node_types[binop.node_id] == BoolType()

    def test_binary_lt(self) -> None:
        let_a = _tc_let("a", _tc_intlit(1))
        let_b = _tc_let("b", _tc_intlit(2))
        binop = BinaryOp(
            op=BinOp.LT, left=_tc_varref("a"), right=_tc_varref("b"),
            span=_tc_sp(), node_id=_tc_nid(),
        )
        r = resolve_and_check(let_a, let_b, _tc_let("r", binop))
        assert r.node_types[binop.node_id] == BoolType()

    def test_binary_le(self) -> None:
        let_a = _tc_let("a", _tc_intlit(1))
        let_b = _tc_let("b", _tc_intlit(2))
        binop = BinaryOp(
            op=BinOp.LE, left=_tc_varref("a"), right=_tc_varref("b"),
            span=_tc_sp(), node_id=_tc_nid(),
        )
        r = resolve_and_check(let_a, let_b, _tc_let("r", binop))
        assert r.node_types[binop.node_id] == BoolType()

    def test_binary_gt(self) -> None:
        let_a = _tc_let("a", _tc_intlit(1))
        let_b = _tc_let("b", _tc_intlit(2))
        binop = BinaryOp(
            op=BinOp.GT, left=_tc_varref("a"), right=_tc_varref("b"),
            span=_tc_sp(), node_id=_tc_nid(),
        )
        r = resolve_and_check(let_a, let_b, _tc_let("r", binop))
        assert r.node_types[binop.node_id] == BoolType()

    def test_binary_ge(self) -> None:
        let_a = _tc_let("a", _tc_intlit(1))
        let_b = _tc_let("b", _tc_intlit(2))
        binop = BinaryOp(
            op=BinOp.GE, left=_tc_varref("a"), right=_tc_varref("b"),
            span=_tc_sp(), node_id=_tc_nid(),
        )
        r = resolve_and_check(let_a, let_b, _tc_let("r", binop))
        assert r.node_types[binop.node_id] == BoolType()

    def test_binary_ord_bool_error(self) -> None:
        let_a = _tc_let("a", _tc_boollit(True))
        let_b = _tc_let("b", _tc_boollit(False))
        binop = BinaryOp(
            op=BinOp.LT, left=_tc_varref("a"), right=_tc_varref("b"),
            span=_tc_sp(), node_id=_tc_nid(),
        )
        err = reject_ast(let_a, let_b, _tc_let("r", binop))
        assert "numeric" in err.to_diagnostic().message

    def test_binary_in_text_text(self) -> None:
        let_a = _tc_let("a", _tc_strlit("hi"))
        let_b = _tc_let("b", _tc_strlit("hi there"))
        binop = BinaryOp(
            op=BinOp.IN, left=_tc_varref("a"), right=_tc_varref("b"),
            span=_tc_sp(), node_id=_tc_nid(),
        )
        r = resolve_and_check(let_a, let_b, _tc_let("r", binop))
        assert r.node_types[binop.node_id] == BoolType()

    def test_binary_in_elem_list(self) -> None:
        let_lst = _tc_let("lst", ListLit(
            elements=(_tc_intlit(1), _tc_intlit(2)),
            span=_tc_sp(), node_id=_tc_nid(),
        ))
        let_a = _tc_let("a", _tc_intlit(1))
        binop = BinaryOp(
            op=BinOp.IN, left=_tc_varref("a"), right=_tc_varref("lst"),
            span=_tc_sp(), node_id=_tc_nid(),
        )
        r = resolve_and_check(let_lst, let_a, _tc_let("r", binop))
        assert r.node_types[binop.node_id] == BoolType()

    def test_binary_in_text_dict(self) -> None:
        key = StringLit(value="a", span=_tc_sp(), node_id=_tc_nid())
        entry = DictEntry(key=key, value=_tc_intlit(1), span=_tc_sp(), node_id=_tc_nid())
        let_d = _tc_let("d", DictLit(entries=(entry,), span=_tc_sp(), node_id=_tc_nid()))
        let_k = _tc_let("k", _tc_strlit("a"))
        binop = BinaryOp(
            op=BinOp.IN, left=_tc_varref("k"), right=_tc_varref("d"),
            span=_tc_sp(), node_id=_tc_nid(),
        )
        r = resolve_and_check(let_d, let_k, _tc_let("r", binop))
        assert r.node_types[binop.node_id] == BoolType()

    def test_binary_in_elem_list_mismatch(self) -> None:
        let_lst = _tc_let("lst", ListLit(
            elements=(_tc_intlit(1),),
            span=_tc_sp(), node_id=_tc_nid(),
        ))
        let_a = _tc_let("a", _tc_strlit("x"))
        binop = BinaryOp(
            op=BinOp.IN, left=_tc_varref("a"), right=_tc_varref("lst"),
            span=_tc_sp(), node_id=_tc_nid(),
        )
        err = reject_ast(let_lst, let_a, _tc_let("r", binop))
        assert "in" in err.to_diagnostic().message

    def test_binary_in_bad_rhs(self) -> None:
        let_a = _tc_let("a", _tc_intlit(1))
        let_b = _tc_let("b", _tc_intlit(2))
        binop = BinaryOp(
            op=BinOp.IN, left=_tc_varref("a"), right=_tc_varref("b"),
            span=_tc_sp(), node_id=_tc_nid(),
        )
        err = reject_ast(let_a, let_b, _tc_let("r", binop))
        assert "in" in err.to_diagnostic().message

    def test_is_test_valid(self) -> None:
        enum_def = EnumDef(
            name="R",
            variants=(VariantDef(name="Pass", fields=(), span=_tc_sp(), node_id=_tc_nid()),),
            span=_tc_sp(), node_id=_tc_nid(),
        )
        ctor = Constructor(qualifier=None, name="Pass", args=(), span=_tc_sp(), node_id=_tc_nid())
        let_r = _tc_let("r", ctor, type_ann=_tc_name_t("R"))
        is_expr = IsTest(
            expr=_tc_varref("r"), qualifier=None, variant="Pass",
            negated=False, span=_tc_sp(), node_id=_tc_nid(),
        )
        r = resolve_and_check(enum_def, let_r, _tc_let("b", is_expr))
        assert r.node_types[is_expr.node_id] == BoolType()

    def test_is_test_wrong_variant(self) -> None:
        enum_def = EnumDef(
            name="R",
            variants=(VariantDef(name="Pass", fields=(), span=_tc_sp(), node_id=_tc_nid()),),
            span=_tc_sp(), node_id=_tc_nid(),
        )
        ctor = Constructor(qualifier=None, name="Pass", args=(), span=_tc_sp(), node_id=_tc_nid())
        let_r = _tc_let("r", ctor, type_ann=_tc_name_t("R"))
        is_expr = IsTest(
            expr=_tc_varref("r"), qualifier=None, variant="Other",
            negated=False, span=_tc_sp(), node_id=_tc_nid(),
        )
        err = reject_ast(enum_def, let_r, _tc_let("b", is_expr))
        assert "Other" in err.to_diagnostic().message

    def test_is_test_non_enum_error(self) -> None:
        let_n = _tc_let("n", _tc_intlit(1))
        is_expr = IsTest(
            expr=_tc_varref("n"), qualifier=None, variant="Pass",
            negated=False, span=_tc_sp(), node_id=_tc_nid(),
        )
        err = reject_ast(let_n, _tc_let("b", is_expr))
        assert "enum" in err.to_diagnostic().message

    def test_field_access_record(self) -> None:
        rec = RecordDef(
            name="Point",
            fields=(_tc_field("x", _tc_int_t()),),
            span=_tc_sp(), node_id=_tc_nid(),
        )
        ctor = Constructor(
            qualifier=None, name="Point",
            args=(NamedArg(name="x", value=_tc_intlit(1), span=_tc_sp(), node_id=_tc_nid()),),
            span=_tc_sp(), node_id=_tc_nid(),
        )
        let_p = _tc_let("p", ctor, type_ann=_tc_name_t("Point"))
        fa = FieldAccess(obj=_tc_varref("p"), field="x", span=_tc_sp(), node_id=_tc_nid())
        r = resolve_and_check(rec, let_p, _tc_let("x", fa))
        assert r.node_types[fa.node_id] == IntType()

    def test_field_access_record_unknown_field(self) -> None:
        rec = RecordDef(
            name="Point", fields=(_tc_field("x", _tc_int_t()),), span=_tc_sp(), node_id=_tc_nid()
        )
        ctor = Constructor(
            qualifier=None, name="Point",
            args=(NamedArg(name="x", value=_tc_intlit(1), span=_tc_sp(), node_id=_tc_nid()),),
            span=_tc_sp(), node_id=_tc_nid(),
        )
        let_p = _tc_let("p", ctor, type_ann=_tc_name_t("Point"))
        fa = FieldAccess(obj=_tc_varref("p"), field="y", span=_tc_sp(), node_id=_tc_nid())
        err = reject_ast(rec, let_p, _tc_let("x", fa))
        assert "y" in err.to_diagnostic().message

    def test_field_access_exception_type(self) -> None:
        """Field access on caught exception binder reads exception fields."""
        clause = CatchClause(
            exc_type="AgentCallError", binding="err",
            body=(
                PrintStmt(
                    value=FieldAccess(
                        obj=_tc_varref("err"), field="message", span=_tc_sp(), node_id=_tc_nid()
                    ),
                    span=_tc_sp(), node_id=_tc_nid(),
                ),
            ),
            span=_tc_sp(), node_id=_tc_nid(),
        )
        try_stmt = TryCatch(
            body=(PassStmt(span=_tc_sp(), node_id=_tc_nid()),),
            handlers=(clause,), span=_tc_sp(), node_id=_tc_nid(),
        )
        r = resolve_and_check(try_stmt)
        assert r.resolved.program is not None

    def test_field_access_exception_unknown_field(self) -> None:
        """Field access on caught exception with non-existent field raises."""
        clause = CatchClause(
            exc_type="AgentCallError", binding="err",
            body=(
                PrintStmt(
                    value=FieldAccess(
                        obj=_tc_varref("err"), field="ghost", span=_tc_sp(), node_id=_tc_nid()
                    ),
                    span=_tc_sp(), node_id=_tc_nid(),
                ),
            ),
            span=_tc_sp(), node_id=_tc_nid(),
        )
        try_stmt = TryCatch(
            body=(PassStmt(span=_tc_sp(), node_id=_tc_nid()),),
            handlers=(clause,), span=_tc_sp(), node_id=_tc_nid(),
        )
        err = reject_ast(try_stmt)
        assert "ghost" in err.to_diagnostic().message

    def test_field_access_non_record_error(self) -> None:
        let_n = _tc_let("n", _tc_intlit(1))
        fa = FieldAccess(obj=_tc_varref("n"), field="x", span=_tc_sp(), node_id=_tc_nid())
        err = reject_ast(let_n, _tc_let("r", fa))
        assert "record" in err.to_diagnostic().message

    # --- Constructor ---

    def test_record_constructor_ok(self) -> None:
        rec = RecordDef(
            name="Pt",
            fields=(_tc_field("x", _tc_int_t()), _tc_field("y", _tc_int_t())),
            span=_tc_sp(), node_id=_tc_nid(),
        )
        ctor = Constructor(
            qualifier=None, name="Pt",
            args=(
                NamedArg(name="x", value=_tc_intlit(1), span=_tc_sp(), node_id=_tc_nid()),
                NamedArg(name="y", value=_tc_intlit(2), span=_tc_sp(), node_id=_tc_nid()),
            ),
            span=_tc_sp(), node_id=_tc_nid(),
        )
        let_p = _tc_let("p", ctor)
        r = resolve_and_check(rec, let_p)
        assert isinstance(r.node_types[ctor.node_id], RecordType)

    def test_record_constructor_missing_field(self) -> None:
        rec = RecordDef(
            name="Pt",
            fields=(_tc_field("x", _tc_int_t()), _tc_field("y", _tc_int_t())),
            span=_tc_sp(), node_id=_tc_nid(),
        )
        ctor = Constructor(
            qualifier=None, name="Pt",
            args=(NamedArg(name="x", value=_tc_intlit(1), span=_tc_sp(), node_id=_tc_nid()),),
            span=_tc_sp(), node_id=_tc_nid(),
        )
        err = reject_ast(rec, _tc_let("p", ctor))
        assert "y" in err.to_diagnostic().message

    def test_record_constructor_unknown_field(self) -> None:
        rec = RecordDef(
            name="Pt",
            fields=(_tc_field("x", _tc_int_t()),),
            span=_tc_sp(), node_id=_tc_nid(),
        )
        ctor = Constructor(
            qualifier=None, name="Pt",
            args=(
                NamedArg(name="x", value=_tc_intlit(1), span=_tc_sp(), node_id=_tc_nid()),
                NamedArg(name="z", value=_tc_intlit(2), span=_tc_sp(), node_id=_tc_nid()),
            ),
            span=_tc_sp(), node_id=_tc_nid(),
        )
        err = reject_ast(rec, _tc_let("p", ctor))
        assert "z" in err.to_diagnostic().message

    def test_record_constructor_duplicate_arg(self) -> None:
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
        err = reject_ast(rec, _tc_let("p", ctor))
        assert "x" in err.to_diagnostic().message

    def test_enum_constructor_qualified(self) -> None:
        enum_def = EnumDef(
            name="R",
            variants=(
                VariantDef(name="Pass", fields=(), span=_tc_sp(), node_id=_tc_nid()),
            ),
            span=_tc_sp(), node_id=_tc_nid(),
        )
        ctor = Constructor(
            qualifier="R", name="Pass", args=(), span=_tc_sp(), node_id=_tc_nid()
        )
        r = resolve_and_check(enum_def, _tc_let("v", ctor))
        assert isinstance(r.node_types[ctor.node_id], EnumType)

    def test_enum_constructor_qualified_unknown_enum(self) -> None:
        ctor = Constructor(
            qualifier="Ghost", name="Pass", args=(), span=_tc_sp(), node_id=_tc_nid()
        )
        err = reject_ast(_tc_let("v", ctor))
        assert "Ghost" in err.to_diagnostic().message

    def test_enum_constructor_qualified_unknown_variant(self) -> None:
        enum_def = EnumDef(
            name="R",
            variants=(VariantDef(name="Pass", fields=(), span=_tc_sp(), node_id=_tc_nid()),),
            span=_tc_sp(), node_id=_tc_nid(),
        )
        ctor = Constructor(
            qualifier="R", name="Fail", args=(), span=_tc_sp(), node_id=_tc_nid()
        )
        err = reject_ast(enum_def, _tc_let("v", ctor))
        assert "Fail" in err.to_diagnostic().message

    def test_enum_constructor_unqualified_ambiguous(self) -> None:
        e1 = EnumDef(
            name="E1",
            variants=(VariantDef(name="Both", fields=(), span=_tc_sp(), node_id=_tc_nid()),),
            span=_tc_sp(), node_id=_tc_nid(),
        )
        e2 = EnumDef(
            name="E2",
            variants=(VariantDef(name="Both", fields=(), span=_tc_sp(), node_id=_tc_nid()),),
            span=_tc_sp(), node_id=_tc_nid(),
        )
        ctor = Constructor(qualifier=None, name="Both", args=(), span=_tc_sp(), node_id=_tc_nid())
        err = reject_ast(e1, e2, _tc_let("v", ctor))
        assert "ambiguous" in err.to_diagnostic().message.lower()

    def test_enum_constructor_unqualified_with_expected_type_resolves(self) -> None:
        """When the expected type is an enum, ambiguity resolution picks the right one."""
        e1 = EnumDef(
            name="E1",
            variants=(VariantDef(name="Both", fields=(), span=_tc_sp(), node_id=_tc_nid()),),
            span=_tc_sp(), node_id=_tc_nid(),
        )
        e2 = EnumDef(
            name="E2",
            variants=(VariantDef(name="Both", fields=(), span=_tc_sp(), node_id=_tc_nid()),),
            span=_tc_sp(), node_id=_tc_nid(),
        )
        ctor = Constructor(qualifier=None, name="Both", args=(), span=_tc_sp(), node_id=_tc_nid())
        let_v = _tc_let("v", ctor, type_ann=_tc_name_t("E1"))
        r = resolve_and_check(e1, e2, let_v)
        assert isinstance(r.node_types[ctor.node_id], EnumType)

    def test_enum_constructor_unqualified_unknown(self) -> None:
        ctor = Constructor(qualifier=None, name="Ghost", args=(), span=_tc_sp(), node_id=_tc_nid())
        err = reject_ast(_tc_let("v", ctor))
        assert "Ghost" in err.to_diagnostic().message

    def test_variant_constructor_duplicate_arg(self) -> None:
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
        err = reject_ast(enum_def, _tc_let("v", ctor))
        assert "msg" in err.to_diagnostic().message

    def test_variant_constructor_unknown_field(self) -> None:
        enum_def = EnumDef(
            name="R",
            variants=(VariantDef(name="Pass", fields=(), span=_tc_sp(), node_id=_tc_nid()),),
            span=_tc_sp(), node_id=_tc_nid(),
        )
        ctor = Constructor(
            qualifier="R", name="Pass",
            args=(NamedArg(name="ghost", value=_tc_intlit(1), span=_tc_sp(), node_id=_tc_nid()),),
            span=_tc_sp(), node_id=_tc_nid(),
        )
        err = reject_ast(enum_def, _tc_let("v", ctor))
        assert "ghost" in err.to_diagnostic().message

    def test_variant_constructor_missing_field(self) -> None:
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
        ctor = Constructor(qualifier="R", name="Fail", args=(), span=_tc_sp(), node_id=_tc_nid())
        err = reject_ast(enum_def, _tc_let("v", ctor))
        assert "msg" in err.to_diagnostic().message

    # --- List and Dict literals ---

    def test_list_lit_nonempty(self) -> None:
        lst_expr = ListLit(
            elements=(_tc_intlit(1), _tc_intlit(2)), span=_tc_sp(), node_id=_tc_nid()
        )
        let_lst = _tc_let("xs", lst_expr)
        r = resolve_and_check(let_lst)
        assert r.node_types[lst_expr.node_id] == ListType(elem=IntType())

    def test_list_lit_empty_with_annotation(self) -> None:
        lst_expr = ListLit(elements=(), span=_tc_sp(), node_id=_tc_nid())
        let_lst = _tc_let("xs", lst_expr, type_ann=_tc_list_t(_tc_int_t()))
        r = resolve_and_check(let_lst)
        assert r.node_types[lst_expr.node_id] == ListType(elem=IntType())

    def test_list_lit_empty_without_annotation_error(self) -> None:
        let_lst = _tc_let("xs", ListLit(elements=(), span=_tc_sp(), node_id=_tc_nid()))
        err = reject_ast(let_lst)
        assert "annotation" in err.to_diagnostic().message

    def test_list_lit_inconsistent_types(self) -> None:
        let_lst = _tc_let("xs", ListLit(
            elements=(_tc_intlit(1), _tc_strlit("x")),
            span=_tc_sp(), node_id=_tc_nid(),
        ))
        err = reject_ast(let_lst)
        assert "inconsistent" in err.to_diagnostic().message

    def test_dict_lit_nonempty(self) -> None:
        key = StringLit(value="a", span=_tc_sp(), node_id=_tc_nid())
        entry = DictEntry(key=key, value=_tc_intlit(1), span=_tc_sp(), node_id=_tc_nid())
        dict_expr = DictLit(entries=(entry,), span=_tc_sp(), node_id=_tc_nid())
        let_d = _tc_let("d", dict_expr)
        r = resolve_and_check(let_d)
        assert r.node_types[dict_expr.node_id] == DictType(value=IntType())

    def test_dict_lit_empty_with_annotation(self) -> None:
        from agm.agl.syntax.types import DictT
        dict_expr = DictLit(entries=(), span=_tc_sp(), node_id=_tc_nid())
        let_d = _tc_let(
            "d", dict_expr,
            type_ann=DictT(value=_tc_int_t(), span=_tc_sp(), node_id=_tc_nid()),
        )
        r = resolve_and_check(let_d)
        assert r.node_types[dict_expr.node_id] == DictType(value=IntType())

    def test_dict_lit_empty_without_annotation_error(self) -> None:
        let_d = _tc_let("d", DictLit(entries=(), span=_tc_sp(), node_id=_tc_nid()))
        err = reject_ast(let_d)
        assert "annotation" in err.to_diagnostic().message

    def test_dict_lit_duplicate_key(self) -> None:
        key1 = StringLit(value="a", span=_tc_sp(), node_id=_tc_nid())
        key2 = StringLit(value="a", span=_tc_sp(), node_id=_tc_nid())
        e1 = DictEntry(key=key1, value=_tc_intlit(1), span=_tc_sp(), node_id=_tc_nid())
        e2 = DictEntry(key=key2, value=_tc_intlit(2), span=_tc_sp(), node_id=_tc_nid())
        let_d = _tc_let("d", DictLit(entries=(e1, e2), span=_tc_sp(), node_id=_tc_nid()))
        err = reject_ast(let_d)
        assert "a" in err.to_diagnostic().message

    # --- Case expression ---

    def test_case_expr_all_same_type(self) -> None:
        let_x = _tc_let("x", _tc_intlit(1))
        vp = VarPattern(name="v", span=_tc_sp(), node_id=_tc_nid())
        branch = CaseExprBranch(
            pattern=vp, body=_tc_varref("v"), span=_tc_sp(), node_id=_tc_nid()
        )
        case_expr = CaseExpr(
            subject=_tc_varref("x"), branches=(branch,), span=_tc_sp(), node_id=_tc_nid()
        )
        let_r = _tc_let("r", case_expr)
        r = resolve_and_check(let_x, let_r)
        assert r.node_types[case_expr.node_id] == IntType()

    def test_case_expr_int_decimal_widening(self) -> None:
        let_x = _tc_let("x", _tc_intlit(1))
        b1 = CaseExprBranch(
            pattern=WildcardPattern(span=_tc_sp(), node_id=_tc_nid()),
            body=_tc_intlit(1),
            span=_tc_sp(), node_id=_tc_nid(),
        )
        b2 = CaseExprBranch(
            pattern=WildcardPattern(span=_tc_sp(), node_id=_tc_nid()),
            body=_tc_declit("1.5"),
            span=_tc_sp(), node_id=_tc_nid(),
        )
        case_expr = CaseExpr(
            subject=_tc_varref("x"), branches=(b1, b2), span=_tc_sp(), node_id=_tc_nid()
        )
        r = resolve_and_check(let_x, _tc_let("r", case_expr))
        assert r.node_types[case_expr.node_id] == DecimalType()

    def test_case_expr_type_mismatch(self) -> None:
        let_x = _tc_let("x", _tc_intlit(1))
        b1 = CaseExprBranch(
            pattern=WildcardPattern(span=_tc_sp(), node_id=_tc_nid()),
            body=_tc_intlit(1),
            span=_tc_sp(), node_id=_tc_nid(),
        )
        b2 = CaseExprBranch(
            pattern=WildcardPattern(span=_tc_sp(), node_id=_tc_nid()),
            body=_tc_strlit("x"),
            span=_tc_sp(), node_id=_tc_nid(),
        )
        case_expr = CaseExpr(
            subject=_tc_varref("x"), branches=(b1, b2), span=_tc_sp(), node_id=_tc_nid()
        )
        err = reject_ast(let_x, _tc_let("r", case_expr))
        assert "incompatible" in err.to_diagnostic().message

    def test_case_expr_no_branches_with_expected(self) -> None:
        """Empty case expression with expected type yields that type."""
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

    def test_case_expr_decimal_then_int(self) -> None:
        """branch[0]=decimal, branch[1]=int → stays decimal (int assignable to decimal)."""
        let_x = _tc_let("x", _tc_intlit(1))
        b1 = CaseExprBranch(
            pattern=WildcardPattern(span=_tc_sp(), node_id=_tc_nid()),
            body=_tc_declit("1.5"),
            span=_tc_sp(), node_id=_tc_nid(),
        )
        b2 = CaseExprBranch(
            pattern=WildcardPattern(span=_tc_sp(), node_id=_tc_nid()),
            body=_tc_intlit(1),
            span=_tc_sp(), node_id=_tc_nid(),
        )
        case_expr = CaseExpr(
            subject=_tc_varref("x"), branches=(b1, b2), span=_tc_sp(), node_id=_tc_nid()
        )
        r = resolve_and_check(let_x, _tc_let("r", case_expr))
        assert r.node_types[case_expr.node_id] == DecimalType()

    # --- Template interpolation ---

    def test_interp_segment_with_render(self) -> None:
        """InterpSegment with a known renderer name passes."""
        let_x = _tc_let("x", _tc_intlit(1))
        seg = InterpSegment(
            expr=_tc_varref("x"),
            render="raw",
            span=_tc_sp(), node_id=_tc_nid(),
        )
        tmpl = Template(segments=(seg,), span=_tc_sp(), node_id=_tc_nid())
        let_t = _tc_let("t", tmpl)
        r = resolve_and_check(let_x, let_t)
        assert r.resolved.program is not None

    def test_interp_segment_unknown_render(self) -> None:
        """InterpSegment with an unknown renderer name raises."""
        let_x = _tc_let("x", _tc_intlit(1))
        seg = InterpSegment(
            expr=_tc_varref("x"),
            render="markdown",
            span=_tc_sp(), node_id=_tc_nid(),
        )
        tmpl = Template(segments=(seg,), span=_tc_sp(), node_id=_tc_nid())
        err = reject_ast(let_x, _tc_let("t", tmpl))
        assert "markdown" in err.to_diagnostic().message

    # --- set stmt with annotated var ---

    def test_set_stmt_type_check(self) -> None:
        var_stmt = _tc_var("n", _tc_intlit(0), type_ann=_tc_int_t())
        set_stmt = SetStmt(target="n", value=_tc_intlit(5), span=_tc_sp(), node_id=_tc_nid())
        r = resolve_and_check(var_stmt, set_stmt)
        assert r.resolved.program is not None

    # --- Pattern binding with non-enum subject type (coverage) ---

    def test_constructor_pattern_non_enum_subject_fieldless(self) -> None:
        """Fieldless ConstructorPattern on a non-enum subject is a static error.

        Patterns match enum variants (design §6.1); a constructor pattern against
        a non-enum subject (here ``int``) must be rejected.
        """
        from agm.agl.syntax.nodes import ConstructorPattern
        let_x = _tc_let("x", _tc_intlit(1))
        ctor_p = ConstructorPattern(
            qualifier=None, name="Pass", fields=(), span=_tc_sp(), node_id=_tc_nid()
        )
        branch = CaseStmtBranch(
            pattern=ctor_p,
            body=(PassStmt(span=_tc_sp(), node_id=_tc_nid()),),
            span=_tc_sp(), node_id=_tc_nid(),
        )
        case_stmt = CaseStmt(
            subject=_tc_varref("x"), branches=(branch,), span=_tc_sp(), node_id=_tc_nid()
        )
        err = reject_ast(let_x, case_stmt)
        assert isinstance(err, AglTypeError)

    def test_constructor_pattern_non_enum_subject_with_field_binding(self) -> None:
        """Field-binding ConstructorPattern on a non-enum subject is rejected.

        Regression for the soundness bug: the resolver binds the pattern field
        variable ``v``, but the checker previously silently skipped binding a type
        for non-enum subjects, so a body reading ``v`` got a ``None`` type. The
        whole branch must instead be rejected with a span-aware type error.
        """
        from agm.agl.syntax.nodes import ConstructorPattern, PatternField
        let_x = _tc_let("x", _tc_intlit(1))
        pv = VarPattern(name="v", span=_tc_sp(), node_id=_tc_nid())
        pf = PatternField(name="f", pattern=pv, span=_tc_sp(), node_id=_tc_nid())
        ctor_p = ConstructorPattern(
            qualifier=None, name="Cons", fields=(pf,), span=_tc_sp(), node_id=_tc_nid()
        )
        branch = CaseStmtBranch(
            pattern=ctor_p,
            body=(PrintStmt(value=_tc_varref("v"), span=_tc_sp(), node_id=_tc_nid()),),
            span=_tc_sp(), node_id=_tc_nid(),
        )
        case_stmt = CaseStmt(
            subject=_tc_varref("x"), branches=(branch,), span=_tc_sp(), node_id=_tc_nid()
        )
        err = reject_ast(let_x, case_stmt)
        assert isinstance(err, AglTypeError)

    def test_require_binding_type_none_raises_assertion(self) -> None:
        """Internal invariant: an unset binding type is an AssertionError, not None.

        Directly exercises the ``_require_binding_type`` guard, whose ``None``
        branch is unreachable through normal checking (every reachable binding
        has its type recorded first), so it is verified as a unit contract here.
        """
        from agm.agl.scope.resolver import resolve
        from agm.agl.scope.symbols import BindingRef
        from agm.agl.typecheck.checker import _Checker
        from agm.agl.typecheck.env import TypeEnvironment

        resolved = resolve(_tc_program())
        checker = _Checker(
            env=TypeEnvironment(), resolved=resolved, capabilities=default_capabilities()
        )
        ref = BindingRef(
            name="ghost", mutable=False, decl_span=_tc_sp(), decl_node_id=999999
        )
        with pytest.raises(AssertionError):
            checker._require_binding_type(ref)

    def test_constructor_pattern_enum_unknown_field(self) -> None:
        """ConstructorPattern with field not in variant raises AglTypeError."""
        from agm.agl.syntax.nodes import ConstructorPattern, PatternField
        enum_def = EnumDef(
            name="R",
            variants=(VariantDef(name="Pass", fields=(), span=_tc_sp(), node_id=_tc_nid()),),
            span=_tc_sp(), node_id=_tc_nid(),
        )
        ctor = Constructor(qualifier=None, name="Pass", args=(), span=_tc_sp(), node_id=_tc_nid())
        let_r = _tc_let("r", ctor, type_ann=_tc_name_t("R"))
        pv = VarPattern(name="v", span=_tc_sp(), node_id=_tc_nid())
        pf = PatternField(name="ghost", pattern=pv, span=_tc_sp(), node_id=_tc_nid())
        ctor_p = ConstructorPattern(
            qualifier=None, name="Pass", fields=(pf,), span=_tc_sp(), node_id=_tc_nid()
        )
        branch = CaseStmtBranch(
            pattern=ctor_p,
            body=(PassStmt(span=_tc_sp(), node_id=_tc_nid()),),
            span=_tc_sp(), node_id=_tc_nid(),
        )
        case_stmt = CaseStmt(
            subject=_tc_varref("r"), branches=(branch,), span=_tc_sp(), node_id=_tc_nid()
        )
        err = reject_ast(enum_def, let_r, case_stmt)
        assert "ghost" in err.to_diagnostic().message

    def test_enum_constructor_unqualified_single_no_expected(self) -> None:
        """Unqualified constructor with single matching enum and no expected type resolves."""
        enum_def = EnumDef(
            name="E",
            variants=(VariantDef(name="Sole", fields=(), span=_tc_sp(), node_id=_tc_nid()),),
            span=_tc_sp(), node_id=_tc_nid(),
        )
        ctor = Constructor(qualifier=None, name="Sole", args=(), span=_tc_sp(), node_id=_tc_nid())
        let_v = _tc_let("v", ctor)
        r = resolve_and_check(enum_def, let_v)
        assert isinstance(r.node_types[ctor.node_id], EnumType)

    def test_dict_lit_two_entries_same_type(self) -> None:
        """Dict literal with two entries uses first entry's type for both."""
        key1 = StringLit(value="a", span=_tc_sp(), node_id=_tc_nid())
        key2 = StringLit(value="b", span=_tc_sp(), node_id=_tc_nid())
        e1 = DictEntry(key=key1, value=_tc_intlit(1), span=_tc_sp(), node_id=_tc_nid())
        e2 = DictEntry(key=key2, value=_tc_intlit(2), span=_tc_sp(), node_id=_tc_nid())
        dict_expr = DictLit(entries=(e1, e2), span=_tc_sp(), node_id=_tc_nid())
        let_d = _tc_let("d", dict_expr)
        r = resolve_and_check(let_d)
        assert r.node_types[dict_expr.node_id] == DictType(value=IntType())

    def test_case_expr_two_same_type_branches(self) -> None:
        """Case expression with two branches of the same type: second hits continue."""
        let_x = _tc_let("x", _tc_intlit(1))
        b1 = CaseExprBranch(
            pattern=WildcardPattern(span=_tc_sp(), node_id=_tc_nid()),
            body=_tc_intlit(1),
            span=_tc_sp(), node_id=_tc_nid(),
        )
        b2 = CaseExprBranch(
            pattern=WildcardPattern(span=_tc_sp(), node_id=_tc_nid()),
            body=_tc_intlit(2),
            span=_tc_sp(), node_id=_tc_nid(),
        )
        case_expr = CaseExpr(
            subject=_tc_varref("x"), branches=(b1, b2), span=_tc_sp(), node_id=_tc_nid()
        )
        r = resolve_and_check(let_x, _tc_let("r", case_expr))
        assert r.node_types[case_expr.node_id] == IntType()
