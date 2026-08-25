"""Behavioral typechecking tests for type-scoped built-in statics."""

from __future__ import annotations

from pathlib import Path

import pytest

from agm.agl import PipelineDriver
from agm.agl.capabilities import HostCapabilities
from agm.agl.modules.roots import RootSet
from agm.agl.scope import AglScopeError
from agm.agl.syntax.nodes import Call, LetDecl
from agm.agl.typecheck import AglTypeError, CheckedModule, check_program
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


def _reject(source: str) -> str:
    with pytest.raises((AglScopeError, AglTypeError)) as raised:
        _check(source)
    return str(raised.value)


def test_session_statics_type_as_session_with_defaults_and_named_arguments() -> None:
    checked = _check(
        'let agent = Agent::AgentCommand(command = "agent")\n'
        "let default_transport = Session::open(agent)\n"
        'let named_transport = Session::open(agent, name = "named")\n'
        "let explicit_transport = Session::open(\n"
        "  agent,\n"
        "  transport = Option[SessionTransport]::Some(SessionTransport::Rpc),\n"
        '  name = "rpc",\n'
        ")\n"
        "let default_session = Session::default()"
    )

    calls = [
        item.value
        for item in checked.resolved.program.body.items
        if isinstance(item, LetDecl) and isinstance(item.value, Call)
    ]
    session = checked.type_env.type_table.builtin_declaration("Session")
    assert session is not None
    session_calls = [call for call in calls if checked.node_types[call.node_id] == session.handle()]
    assert len(session_calls) == 4


@pytest.mark.parametrize(
    "source",
    [
        "Session::default(1)",
        "Session::default::[text]()",
        'Session::open("agent")',
        'Session::open(Agent::AgentCommand(command = "agent"), transport = SessionTransport::Cli)',
        'Session::open(Agent::AgentCommand(command = "agent"), name = 1)',
    ],
)
def test_session_statics_reject_invalid_arguments(source: str) -> None:
    _reject(source)


def test_session_open_rejects_lookalike_option_some_record() -> None:
    _reject(
        "scope Fake\n"
        "record Some[T](foo: T)\n"
        "end Fake\n"
        'let agent = Agent::AgentCommand(command = "agent")\n'
        "Session::open(agent, transport = Fake::Some(SessionTransport::Rpc))"
    )


def test_session_unknown_static_reports_a_static_diagnostic() -> None:
    message = _reject("Session::bogus()")
    assert "unknown static" in message.lower()
    assert "session" in message.lower()


def test_non_prelude_type_cannot_use_builtin_static_syntax() -> None:
    _reject("record SomeUserType\n  name: text\nSomeUserType::open()")


@pytest.mark.parametrize(
    ("session_declaration", "local_constructor", "local_kind"),
    [
        ("enum Session\n  | open", "User::Session::open()", "enum"),
        ("record Session()", "User::Session()", "record"),
    ],
)
def test_nested_user_session_does_not_shadow_prelude_session_static(
    session_declaration: str, local_constructor: str, local_kind: str
) -> None:
    checked = _check(
        f"scope User\n{session_declaration}\nend User\n"
        "let default_session = Session::default()\n"
        f"{local_constructor}"
    )
    calls = [
        item.value
        for item in checked.resolved.program.body.items
        if isinstance(item, LetDecl) and isinstance(item.value, Call)
    ]
    session = checked.type_env.type_table.builtin_declaration("Session")
    assert session is not None
    assert checked.node_types[calls[0].node_id] == session.handle()
    result = checked.resolved.program.body.items[-1]
    assert checked.node_types[result.node_id].kind == "record"


def test_non_prelude_session_static_header_is_not_a_builtin() -> None:
    message = _reject(
        "scope User\nrecord Session()\nbuiltin def Session::open() -> Session\nend User"
    )
    assert "unknown builtin" in message.lower()


def test_replacement_std_core_scope_is_not_the_session_static_owner(tmp_path: Path) -> None:
    """A same-path scope in replacement ``std/core`` cannot impersonate Session."""
    (tmp_path / "std").mkdir()
    (tmp_path / "std" / "core.agl").write_text(
        "scope Session\nbuiltin def default() -> Session\nend Session\n",
        encoding="utf-8",
    )

    driver = PipelineDriver()
    prepared = driver.prepare_program(
        "program def main() -> unit =\n  let session = Session::default()\n  ()\n",
        roots=RootSet(roots=frozenset({tmp_path})),
    )

    assert prepared.resolved is not None
    entry = prepared.resolved.modules[prepared.resolved.entry_id].resolved
    assert entry.builtin_static_calls == {}
    with pytest.raises(AglTypeError, match="Unknown builtin"):
        check_program(prepared.resolved, driver.host_environment().capabilities)


def test_prelude_session_constructor_spelling_is_rejected_as_an_unknown_static(
    tmp_path: Path,
) -> None:
    """A prelude static owner rejects constructor-like value references."""
    (tmp_path / "std").mkdir()
    (tmp_path / "std" / "core.agl").write_text(
        "builtin record Session()\n"
        "builtin def Session::default() -> Session\n"
        "let value = Session::Session\n",
        encoding="utf-8",
    )

    prepared = PipelineDriver().prepare_program(
        "program def main() -> unit =\n  let session = Session::default()\n  ()\n",
        roots=RootSet(roots=frozenset({tmp_path})),
    )

    assert prepared.resolved is None
    assert any(
        "unknown static" in diagnostic.message.lower() for diagnostic in prepared.diagnostics
    )


def test_builtin_static_cannot_be_partially_applied_or_used_as_a_value() -> None:
    partial = _reject("Session::open(?)")
    value = _reject("let open_session = Session::open\nopen_session")
    assert "partial" in partial.lower()
    assert "cannot be used as a value" in value.lower()


def test_static_names_remain_ordinary_identifiers() -> None:
    checked = _check("let open = 1\nlet default = 2\nopen + default")
    result = checked.resolved.program.body.items[-1]
    assert checked.node_types[result.node_id].kind == "int"


def test_enum_constructor_resolution_remains_available() -> None:
    checked = _check("SessionTransport::Cli")
    result = checked.resolved.program.body.items[-1]
    transport = checked.type_env.type_table.builtin_declaration("SessionTransport")
    assert transport is not None
    assert checked.node_types[result.node_id] in checked.type_env.type_table.enum_members(
        transport.handle()
    )
