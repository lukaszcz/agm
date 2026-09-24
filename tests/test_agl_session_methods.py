"""Typechecking tests for ``Session`` built-in methods."""

from __future__ import annotations

import pytest

from agm.agl.capabilities import HostCapabilities
from agm.agl.semantics.types import TextType
from agm.agl.syntax.nodes import Call, LetDecl
from agm.agl.typecheck import AglTypeError, CheckedModule
from tests.agl.module_graph import resolve_and_check_repl_entry

_CAPABILITIES = HostCapabilities(
    supports_shell_exec=True,
    codec_kinds={
        "text": frozenset({"text"}),
        "json": frozenset({"json", "record", "enum", "array", "dict", "int", "decimal", "bool"}),
    },
)


def _check(source: str) -> CheckedModule:
    return resolve_and_check_repl_entry(source, _CAPABILITIES)


def _reject(source: str) -> AglTypeError:
    with pytest.raises(AglTypeError) as raised:
        _check(source)
    return raised.value


def test_session_methods_typecheck_with_declared_result_types() -> None:
    checked = _check(
        "let session = Session::default()\n"
        'session.compact(instructions = "retain decisions")\n'
        "session.reset()\n"
        "let forked: Session = session.fork()\n"
        "let session-stats: SessionStats = session.stats()\n"
        'session.set-name("review")\n'
        "session.close()\n"
        "forked"
    )

    forked = checked.resolved.program.body.items[3]
    session_stats = checked.resolved.program.body.items[4]
    assert isinstance(forked, LetDecl)
    assert isinstance(session_stats, LetDecl)
    session = checked.type_env.type_table.builtin_declaration("Session")
    session_stats_type = checked.type_env.type_table.builtin_declaration("SessionStats")
    assert session is not None
    assert session_stats_type is not None
    assert checked.node_types[forked.value.node_id] == session.handle()
    assert checked.node_types[session_stats.value.node_id] == session_stats_type.handle()


def test_session_ask_infers_targets_like_agent_ask() -> None:
    checked = _check(
        "let session = Session::default()\n"
        'let text-response = session.ask("summarize")\n'
        'let count: int = session.ask("count")\n'
        'let accepted = session.ask::[bool]("approve")\n'
        "accepted"
    )

    text_response = checked.resolved.program.body.items[1]
    assert isinstance(text_response, LetDecl)
    assert checked.node_types[text_response.value.node_id] == TextType()
    assert [site.target_type.kind for site in checked.call_sites] == ["text", "int", "bool"]


def test_unbound_nongeneric_session_method_rejects_type_arguments() -> None:
    _reject("let session = Session::default()\nSession::close::[text](session)")


def test_session_ask_rejects_an_explicit_agent() -> None:
    _reject(
        "let session = Session::default()\n"
        'let agent = AgentCommand("worker")\n'
        'session.ask("summarize", agent = agent)'
    )


def test_session_ask_rejects_an_explicit_sandbox() -> None:
    _reject('let session = Session::default()\nsession.ask("summarize", sandbox = Native)')


def test_agent_receiver_ask_rejects_a_conflicting_explicit_agent() -> None:
    _reject(
        'let worker = AgentCommand("worker")\n'
        'let other = AgentCommand("other")\n'
        'worker.ask("summarize", agent = other)'
    )


def test_session_values_cannot_be_forged_through_record_update() -> None:
    _reject(
        "let session = Session::default()\n"
        'let other = AgentCommand("other")\n'
        "session with agent = other"
    )


def test_session_ask_records_parse_options_and_output_contract_metadata() -> None:
    checked = _check(
        "let session = Session::default()\n"
        'let count: int = session.ask("count", format = "json", strict-json = true, '
        "on-parse-error = Retry(n = 2))\n"
        'let summary: text = session.ask("summarize", format = "text", on-parse-error = Abort)\n'
        "summary"
    )

    count = checked.resolved.program.body.items[1]
    summary = checked.resolved.program.body.items[2]
    assert isinstance(count, LetDecl)
    assert isinstance(summary, LetDecl)
    assert isinstance(count.value, Call)
    assert isinstance(summary.value, Call)
    assert checked.contract_specs[count.value.node_id].target_type.kind == "int"
    assert checked.contract_specs[count.value.node_id].codec_name == "json"
    assert checked.contract_specs[count.value.node_id].strict_json is True
    assert checked.contract_specs[summary.value.node_id].target_type == TextType()
    assert checked.contract_specs[summary.value.node_id].codec_name == "text"
    assert checked.contract_specs[summary.value.node_id].strict_json is None
    assert [(site.callee, site.codec_name, site.parse_policy) for site in checked.call_sites] == [
        ("ask", "json", "retry[2]"),
        ("ask", "text", "abort"),
    ]
    assert len(checked.warnings) == 1


def test_session_ask_infers_a_generic_target_from_a_sibling_argument() -> None:
    checked = _check(
        "def choose[T](value: T, fallback: T) -> T = value\n"
        "let session = Session::default()\n"
        'let count = choose(session.ask("count"), 7)\n'
        "count"
    )

    count = checked.resolved.program.body.items[2]
    assert isinstance(count, LetDecl)
    assert isinstance(count.value, Call)
    ask = count.value.args[0]
    assert isinstance(ask, Call)
    assert checked.node_types[ask.node_id].kind == "int"
    assert checked.contract_specs[ask.node_id].target_type.kind == "int"


def test_session_ask_rejects_a_rigid_generic_target() -> None:
    _reject("def fetch[T](session: Session, prompt: text) -> T = session.ask::[T](prompt)\n()")


@pytest.mark.parametrize(
    "source",
    (
        "let session = Session::default()\nsession.ask()",
        'let session = Session::default()\nsession.ask("one", "two")',
        "let session = Session::default()\nsession.ask(1)",
        "let session = Session::default()\nsession.compact(1)",
        'let session = Session::default()\nsession.compact("one", "two")',
        "let session = Session::default()\nsession.reset(1)",
        "let session = Session::default()\nsession.fork(1)",
        "let session = Session::default()\nsession.stats(1)",
        "let session = Session::default()\nsession.set-name()",
        "let session = Session::default()\nsession.set-name(1)",
        'let session = Session::default()\nsession.set-name("one", "two")',
        "let session = Session::default()\nsession.close(1)",
    ),
)
def test_session_methods_reject_wrong_arities_and_types(source: str) -> None:
    _reject(source)


def test_session_dollar_literal_ask_uses_session_ask_typechecking() -> None:
    checked = _check(
        "let session = Session::default()\n"
        "let count: int = session.ask $ count the completed tasks\n"
        "count"
    )

    count = checked.resolved.program.body.items[1]
    assert isinstance(count, LetDecl)
    assert isinstance(count.value, Call)
    assert checked.call_sites[0].node_id == count.value.node_id
    assert checked.call_sites[0].target_type.kind == "int"


def test_agent_builtin_ask_keeps_its_existing_dispatch() -> None:
    checked = _check(
        'let agent = AgentCommand("worker")\nlet response: text = agent.ask("summarize")\nresponse'
    )

    assert [site.callee for site in checked.call_sites] == ["ask"]


def test_non_session_receivers_do_not_gain_session_methods() -> None:
    _reject('let agent = AgentCommand("worker")\nagent.compact()')


def test_nested_session_constructor_and_regular_method_remain_user_defined() -> None:
    checked = _check(
        "scope User\n"
        "  record Session\n"
        "  def Session::ask(self, prompt: text) -> text = prompt\n"
        "end User\n"
        "\n"
        "let session = User::Session()\n"
        'session.ask("local")'
    )

    assert checked.node_types[checked.resolved.program.body.items[-1].node_id] == TextType()
    assert checked.call_sites == ()


def test_redeclared_session_builtin_method_is_not_a_host_method() -> None:
    error = _reject(
        "scope User\n"
        "  builtin record Session\n"
        "    id: text\n"
        "    agent: Agent\n"
        "    transport: SessionTransport\n"
        "    sandbox: AgentSandbox\n"
        '  builtin def Session::compact(self, instructions: text = "") -> unit\n'
        "end User"
    )

    assert "invalid signature" in str(error).lower()


def test_builtin_method_headers_are_limited_to_registered_receivers() -> None:
    _reject("scope User\n  record Session\n  builtin def Session::unknown(self) -> unit\nend User")
