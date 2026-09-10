"""Lowering and structural validation tests for session IR nodes."""

from __future__ import annotations

from typing import cast

import pytest

from agm.agl.ir.contracts import ContractRequest, ScalarDecode, ScalarKind
from agm.agl.ir.ids import ContractId, Location, SourceId
from agm.agl.ir.nodes import (
    IrBind,
    IrBlock,
    IrConstText,
    IrExpr,
    IrLoad,
    IrMakeClosure,
    IrMakeRecord,
    IrSessionAsk,
    IrSessionDefault,
    IrSessionOp,
    IrSessionOpen,
    IrSessionOpKind,
)
from agm.agl.ir.program import (
    ExecutableModule,
    ExecutableProgram,
    IrFunctionBody,
    SourceFile,
)
from agm.agl.ir.validate import InvalidIrError, validate_ir
from agm.agl.lower.lowerer import _Lowerer
from agm.agl.modules.ids import ENTRY_ID
from agm.agl.semantics.type_table import MethodDef
from agm.agl.semantics.types import (
    BUILTIN_PRELUDE_TYPES,
    FunctionType,
    IntType,
    RecordType,
    UnitType,
)
from tests.agl.ir_harness import inline_main_items, lower_inline_ir

_SRC_ID = SourceId(0)
_LOC = Location(source_id=_SRC_ID, start_offset=0, end_offset=1, start_line=1, start_col=0)


def _contract() -> ContractRequest:
    return ContractRequest(
        codec_name="text",
        strict_json=None,
        json_schema=None,
        decode=None,
        target_type_label="text",
        structured_exec=False,
        format_instructions="",
        is_unit=False,
    )


def _program(
    *initializers: IrExpr, contracts: dict[ContractId, ContractRequest] | None = None
) -> ExecutableProgram:
    return ExecutableProgram(
        entry_module=ENTRY_ID,
        modules={ENTRY_ID: ExecutableModule(module_id=ENTRY_ID, initializers=initializers)},
        symbols={},
        nominals={},
        sources={_SRC_ID: SourceFile(display_name="<test>", normalized_text="x")},
        contracts={} if contracts is None else contracts,
    )


def _main_let_values(source: str) -> dict[str, object]:
    program = lower_inline_ir(source)
    values: dict[str, object] = {}
    for initializer in inline_main_items(program):
        if initializer.__class__.__name__ != "IrBind":
            continue
        symbol = program.symbols[initializer.symbol]
        if symbol.public_name is not None:
            values[symbol.public_name] = initializer.value
    return values


def test_session_open_lowers_omitted_and_explicit_options() -> None:
    values = _main_let_values(
        'let agent = AgentCommand("worker")\n'
        "let omitted = Session::open(agent)\n"
        "let explicit = Session::open(\n"
        "  agent,\n"
        "  transport = Option[SessionTransport]::Some(SessionTransport::Rpc),\n"
        '  name = "review",\n'
        ")\n"
        "()"
    )

    omitted = values["omitted"]
    explicit = values["explicit"]
    assert isinstance(omitted, IrSessionOpen)
    assert isinstance(omitted.agent, IrLoad)
    assert omitted.transport is None
    assert isinstance(omitted.name, IrConstText)
    assert omitted.name.value == ""
    assert isinstance(explicit, IrSessionOpen)
    assert isinstance(explicit.transport, IrMakeRecord)
    assert isinstance(explicit.name, IrConstText)
    assert explicit.name.value == "review"


def test_session_default_lowers_to_its_dedicated_node() -> None:
    values = _main_let_values("let session = Session::default()\n()")

    assert isinstance(values["session"], IrSessionDefault)


def test_session_statics_lower_as_callable_values() -> None:
    program = lower_inline_ir(
        "let open-session: Agent -> Session = Session::open\n"
        "let current-session: () -> Session = Session::default\n"
        "()"
    )
    values = {
        program.symbols[item.symbol].public_name: item.value
        for item in inline_main_items(program)
        if isinstance(item, IrBind) and program.symbols[item.symbol].public_name is not None
    }

    open_closure = values["open-session"]
    default_closure = values["current-session"]
    assert isinstance(open_closure, IrMakeClosure)
    assert isinstance(default_closure, IrMakeClosure)
    open_descriptor = program.functions[open_closure.function_id]
    default_descriptor = program.functions[default_closure.function_id]
    assert isinstance(open_descriptor.impl, IrFunctionBody)
    assert isinstance(default_descriptor.impl, IrFunctionBody)
    assert isinstance(open_descriptor.impl.body, IrSessionOpen)
    assert isinstance(default_descriptor.impl.body, IrSessionDefault)


def test_session_methods_lower_as_bound_and_unbound_values() -> None:
    program = lower_inline_ir(
        "let session = Session::default()\n"
        "let rename: text -> unit = session.set-name\n"
        "let close: Session -> unit = Session::close\n"
        "let fork-bound: () -> Session = session.fork\n"
        "let fork-unbound: Session -> Session = Session::fork\n"
        "let stats-bound: () -> SessionStats = session.stats\n"
        "let stats-unbound: Session -> SessionStats = Session::stats\n"
        'Session::compact(session, instructions = "retain decisions")\n'
        "()"
    )
    values = {
        program.symbols[item.symbol].public_name: item.value
        for item in inline_main_items(program)
        if isinstance(item, IrBind) and program.symbols[item.symbol].public_name is not None
    }

    rename_block = values["rename"]
    close_closure = values["close"]
    assert isinstance(rename_block, IrBlock)
    rename_closure = rename_block.items[-1]
    assert isinstance(rename_closure, IrMakeClosure)
    assert isinstance(close_closure, IrMakeClosure)
    rename_descriptor = program.functions[rename_closure.function_id]
    close_descriptor = program.functions[close_closure.function_id]
    assert isinstance(rename_descriptor.impl, IrFunctionBody)
    assert isinstance(close_descriptor.impl, IrFunctionBody)
    assert isinstance(rename_descriptor.impl.body, IrSessionOp)
    assert rename_descriptor.impl.body.op is IrSessionOpKind.SET_NAME
    assert isinstance(close_descriptor.impl.body, IrSessionOp)
    assert close_descriptor.impl.body.op is IrSessionOpKind.CLOSE


def test_session_ask_lowers_as_a_bound_value() -> None:
    program = lower_inline_ir(
        "let session = Session::default()\nlet query: text -> text = session.ask\n()"
    )
    query = next(
        item.value
        for item in inline_main_items(program)
        if isinstance(item, IrBind) and program.symbols[item.symbol].public_name == "query"
    )
    assert isinstance(query, IrBlock)
    closure = query.items[-1]
    assert isinstance(closure, IrMakeClosure)
    descriptor = program.functions[closure.function_id]
    assert isinstance(descriptor.impl, IrFunctionBody)
    assert isinstance(descriptor.impl.body, IrSessionAsk)


def test_free_ask_lowers_through_the_default_session() -> None:
    values = _main_let_values('let answer: text = ask("question")\n()')

    answer = values["answer"]
    assert isinstance(answer, IrSessionAsk)
    assert isinstance(answer.session, IrSessionDefault)


def test_session_ask_lowers_a_formatted_strict_json_contract_with_retries() -> None:
    program = lower_inline_ir(
        "let session = Session::default()\n"
        'let answer: int = session.ask("How many?", format = "json", strict-json = true, '
        "on-parse-error = Retry(n = 2))\n"
        "()"
    )
    answer = next(
        initializer.value
        for initializer in inline_main_items(program)
        if isinstance(initializer, IrBind)
        and program.symbols[initializer.symbol].public_name == "answer"
    )

    assert isinstance(answer, IrSessionAsk)
    assert answer.contract_id in program.contracts
    assert answer.max_attempts == 3
    assert program.contracts == {
        answer.contract_id: ContractRequest(
            codec_name="json",
            strict_json=True,
            json_schema='{"type": "integer"}',
            decode=ScalarDecode(kind=ScalarKind.INT),
            target_type_label="int",
            structured_exec=False,
            format_instructions=(
                "Return exactly one JSON value conforming to the following JSON Schema.\n"
                "Do not include Markdown, prose, or code fences.\n\n"
                '```json\n{\n  "type": "integer"\n}\n```'
            ),
            target_type_kind="int",
            target_type=IntType(),
        )
    }


def test_session_methods_lower_to_session_nodes() -> None:
    values = _main_let_values(
        "let session = Session::default()\n"
        'let retried: text = session.ask("retry", on-parse-error = Retry(n = 2))\n'
        'let aborted: text = session.ask("abort", on-parse-error = Abort)\n'
        'let defaulted: text = session.ask("default")\n'
        "session.compact()\n"
        'session.compact("retain decisions")\n'
        "session.reset()\n"
        "let forked = session.fork()\n"
        "let stats = session.stats()\n"
        'session.set-name("review")\n'
        "session.close()\n"
        "()"
    )

    asks = [values[name] for name in ("retried", "aborted", "defaulted")]
    assert all(isinstance(ask, IrSessionAsk) for ask in asks)
    assert [ask.max_attempts for ask in asks if isinstance(ask, IrSessionAsk)] == [3, 1, 1]
    assert all(isinstance(ask.session, IrLoad) for ask in asks if isinstance(ask, IrSessionAsk))

    program = lower_inline_ir(
        "let session = Session::default()\n"
        "session.compact()\n"
        'session.compact("retain decisions")\n'
        "session.reset()\n"
        "let forked = session.fork()\n"
        "let stats = session.stats()\n"
        'session.set-name("review")\n'
        "session.close()\n"
        "()"
    )
    operations = [
        initializer
        for initializer in inline_main_items(program)
        if isinstance(initializer, IrSessionOp)
    ]

    assert [operation.op for operation in operations] == [
        "compact",
        "compact",
        "reset",
        "set-name",
        "close",
    ]
    assert operations[0].arg is None
    assert isinstance(operations[1].arg, IrConstText)
    assert operations[1].arg.value == "retain decisions"
    assert operations[2].arg is None
    assert isinstance(operations[3].arg, IrConstText)
    assert operations[3].arg.value == "review"
    assert operations[4].arg is None
    assert isinstance(values["forked"], IrSessionOp)
    assert values["forked"].op == "fork"
    assert isinstance(values["stats"], IrSessionOp)
    assert values["stats"].op == "stats"


def test_canonical_prelude_session_method_is_recognized() -> None:
    """Host-minted Session methods use their canonical nominal identity."""
    session = BUILTIN_PRELUDE_TYPES["Session"]
    assert isinstance(session, RecordType)
    method = MethodDef(
        module_id=ENTRY_ID,
        scope_path=("Unrelated",),
        name="compact",
        decl_node_id=0,
        signature=FunctionType(params=(session,), result=UnitType()),
        receiver_type_param_arity=0,
        is_builtin=True,
    )
    lowerer: _Lowerer = object.__new__(_Lowerer)

    assert lowerer._is_session_builtin_method(method)


def test_session_op_defaults_its_optional_argument_to_none() -> None:
    operation = IrSessionOp(
        location=_LOC,
        session=IrConstText(location=_LOC, value="session"),
        op=IrSessionOpKind.RESET,
    )

    assert operation.arg is None


def test_well_formed_session_nodes_pass_deep_validation() -> None:
    contract_id = ContractId(0)
    session = IrConstText(location=_LOC, value="session")
    program = _program(
        IrSessionOpen(
            location=_LOC,
            agent=IrConstText(location=_LOC, value="agent"),
            transport=IrConstText(location=_LOC, value="transport"),
            name=IrConstText(location=_LOC, value="name"),
        ),
        IrSessionDefault(location=_LOC),
        IrSessionAsk(
            location=_LOC,
            session=session,
            prompt=IrConstText(location=_LOC, value="prompt"),
            contract_id=contract_id,
            max_attempts=1,
        ),
        *(
            IrSessionOp(location=_LOC, session=session, op=op, arg=arg)
            for op, arg in (
                (IrSessionOpKind.COMPACT, None),
                (IrSessionOpKind.COMPACT, IrConstText(location=_LOC, value="instructions")),
                (IrSessionOpKind.RESET, None),
                (IrSessionOpKind.FORK, None),
                (IrSessionOpKind.STATS, None),
                (IrSessionOpKind.SET_NAME, IrConstText(location=_LOC, value="name")),
                (IrSessionOpKind.CLOSE, None),
            )
        ),
        contracts={contract_id: _contract()},
    )

    validate_ir(program, deep=False)
    validate_ir(program)


def test_session_ask_rejects_unknown_contract_and_bad_max_attempts() -> None:
    missing_contract = IrSessionAsk(
        location=_LOC,
        session=IrConstText(location=_LOC, value="session"),
        prompt=IrConstText(location=_LOC, value="prompt"),
        contract_id=ContractId(99),
        max_attempts=1,
    )
    with pytest.raises(InvalidIrError, match="contract_id"):
        validate_ir(_program(missing_contract))

    bad_attempts = IrSessionAsk(
        location=_LOC,
        session=IrConstText(location=_LOC, value="session"),
        prompt=IrConstText(location=_LOC, value="prompt"),
        contract_id=ContractId(0),
        max_attempts=0,
    )
    with pytest.raises(InvalidIrError, match="max_attempts"):
        validate_ir(_program(bad_attempts, contracts={ContractId(0): _contract()}))


def test_session_op_rejects_unknown_tag_and_bad_argument_pairings() -> None:
    session = IrConstText(location=_LOC, value="session")
    with pytest.raises(InvalidIrError, match="operation"):
        validate_ir(
            _program(
                IrSessionOp(
                    location=_LOC,
                    session=session,
                    op=cast(IrSessionOpKind, "unknown"),
                    arg=None,
                )
            )
        )

    for operation in (
        IrSessionOp(location=_LOC, session=session, op=IrSessionOpKind.SET_NAME, arg=None),
        IrSessionOp(
            location=_LOC,
            session=session,
            op=IrSessionOpKind.RESET,
            arg=IrConstText(location=_LOC, value="unexpected"),
        ),
    ):
        with pytest.raises(InvalidIrError, match="argument"):
            validate_ir(_program(operation))
