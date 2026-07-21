"""Cross-entry persistence of ``std/config`` engine settings in the REPL.

AgL exposes the six engine settings as ``builtin var`` declarations in the
``std/config`` stdlib module.  A REPL user reads a setting with
``std/config::KEY`` and writes it with ``std/config::KEY := VALUE`` after an
``import std/config``.  These tests assert that such writes persist across REPL
entries with full parity for all six keys:

* reading a setting in a later entry reflects the most recent earlier write, and
* the runtime-live effects (loop cap, strict-json parsing, shell-exec timeout)
  carry forward to later entries.

Agents are always mocked — no real agent is ever run.
"""

from __future__ import annotations

from pathlib import Path

from agm.agent.defaults import DEFAULT_AGENT_RUNNER
from agm.agl.repl import EntryResult, ReplSession
from agm.agl.runtime.agents import AgentFn
from agm.agl.runtime.host_settings import HostSettingsPolicy
from agm.agl.runtime.request import AgentRequest, AgentResponse
from agm.agl.semantics.values import VOID_VALUE, BoolValue, EnumValue, IntValue, TextValue, Value

_STDLIB_ROOT = Path(__file__).resolve().parents[1] / "stdlib"


class _FencedAgent:
    """A fake ``AgentFn`` returning a fenced-JSON reply (lenient-only)."""

    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, request: AgentRequest) -> AgentResponse:
        self.calls += 1
        return AgentResponse(content="```json\n42\n```")


def _session(
    *,
    default_agent: AgentFn | None = None,
    engine_base: dict[str, Value] | None = None,
    host_settings_policy: HostSettingsPolicy | None = None,
    trace_path: Path | None = None,
) -> ReplSession:
    return ReplSession(
        stdlib_root=_STDLIB_ROOT,
        default_agent=default_agent,
        engine_base=engine_base,
        host_settings_policy=host_settings_policy,
        trace_path=trace_path,
    )


def _ok(session: ReplSession, text: str) -> EntryResult:
    result = session.eval_entry(text)
    assert result.ok, f"entry {text!r} failed: {result.diagnostics} {result.error}"
    return result


def _read(session: ReplSession, key: str) -> Value:
    """Import-and-read *key*, returning the read value of a later-entry read."""
    result = _ok(session, f"std/config::{key}")
    assert result.value is not None
    return result.value


# ---------------------------------------------------------------------------
# Cross-entry persistence for each of the six keys
# ---------------------------------------------------------------------------


class TestCrossEntryPersistence:
    """A write in entry N is visible to a read two entries later (N+2)."""

    def test_max_iters_write_persists_two_entries_later(self) -> None:
        s = _session()
        _ok(s, "import std/config")
        _ok(s, "std/config::max-iters := 7")
        _ok(s, "let unrelated = 1")
        assert _read(s, "max-iters") == IntValue(7)

    def test_strict_json_write_persists_two_entries_later(self) -> None:
        s = _session()
        _ok(s, "import std/config")
        _ok(s, "std/config::strict-json := true")
        _ok(s, "let unrelated = 1")
        assert _read(s, "strict-json") == BoolValue(True)

    def test_timeout_write_persists_two_entries_later(self) -> None:
        s = _session()
        _ok(s, "import std/config")
        _ok(s, 'std/config::timeout := Some("45s")')
        _ok(s, "let unrelated = 1")
        value = _read(s, "timeout")
        assert isinstance(value, EnumValue)
        assert value.variant == "Some"
        assert value.fields["value"] == TextValue("45s")
        # The written timeout is retained as the live shell-exec timeout.
        assert s._shell_exec_timeout == 45.0

    def test_runner_write_persists_two_entries_later(self) -> None:
        s = _session()
        _ok(s, "import std/config")
        _ok(s, 'std/config::runner := "codex"')
        _ok(s, "let unrelated = 1")
        assert _read(s, "runner") == TextValue("codex")

    def test_log_write_persists_two_entries_later(self) -> None:
        s = _session()
        _ok(s, "import std/config")
        _ok(s, "std/config::log := true")
        _ok(s, "let unrelated = 1")
        assert _read(s, "log") == BoolValue(True)

    def test_log_file_write_persists_two_entries_later(self) -> None:
        s = _session()
        _ok(s, "import std/config")
        _ok(s, 'std/config::log-file := Some("trace.jsonl")')
        _ok(s, "let unrelated = 1")
        value = _read(s, "log-file")
        assert isinstance(value, EnumValue)
        assert value.variant == "Some"
        assert value.fields["value"] == TextValue("trace.jsonl")


# ---------------------------------------------------------------------------
# Runtime-live effect carry-forward
# ---------------------------------------------------------------------------


class TestRuntimeLiveEffectCarryForward:
    """The loop cap and strict-json parsing effects apply in later entries."""

    def test_max_iters_write_caps_later_unguarded_loop(self) -> None:
        s = _session()
        _ok(s, "import std/config")
        _ok(s, "std/config::max-iters := 2")
        _ok(s, "let unrelated = 1")
        # An unguarded loop that would run far past the cap must be cut short.
        result = s.eval_entry("var i = 0\ndo\n  i := i + 1\nuntil i >= 1000\ni")
        assert not result.ok
        assert result.error is not None
        assert "MaxIterationsExceeded" in result.error.type_name

    def test_strict_json_write_makes_later_ask_strict(self) -> None:
        agent = _FencedAgent()
        s = _session(default_agent=agent)
        # Confirm the fenced reply parses in the default (lenient) mode.
        r_lenient = _ok(s, 'let a: int = ask """how many"""')
        assert r_lenient.value == VOID_VALUE
        assert dict((name, value) for name, _typ, value in s.bindings())["a"] == IntValue(42)

        _ok(s, "import std/config")
        _ok(s, "std/config::strict-json := true")
        _ok(s, "let unrelated = 1")
        # In strict mode the fenced reply is rejected in a later entry.
        result = s.eval_entry('let b: int = ask """how many"""')
        assert not result.ok
        assert result.error is not None
        assert "AgentParseError" in result.error.type_name


# ---------------------------------------------------------------------------
# Defaults and seeding
# ---------------------------------------------------------------------------


class TestDefaultsAndSeeding:
    """A session that never writes ``std/config`` reads the engine defaults."""

    def test_untouched_session_reads_engine_defaults(self) -> None:
        s = _session()
        _ok(s, "import std/config")
        assert _read(s, "runner") == TextValue(DEFAULT_AGENT_RUNNER)
        assert _read(s, "log") == BoolValue(False)
        log_file = _read(s, "log-file")
        assert isinstance(log_file, EnumValue)
        assert log_file.variant == "None"
        assert _read(s, "strict-json") == BoolValue(False)
        assert _read(s, "max-iters") == IntValue(0)
        timeout = _read(s, "timeout")
        assert isinstance(timeout, EnumValue)
        assert timeout.variant == "None"

    def test_host_timeout_seed_round_trips_without_disabling_live_timeout(self) -> None:
        s = ReplSession(stdlib_root=_STDLIB_ROOT, shell_exec_timeout=0.0000001)
        _ok(s, "import std/config")

        value = _read(s, "timeout")
        assert isinstance(value, EnumValue)
        assert value.fields["value"] == TextValue("0.0000001s")
        _ok(s, "std/config::timeout := std/config::timeout")
        assert s._shell_exec_timeout == 0.0000001

        s.reset()
        value = _ok(s, "import std/config\nstd/config::timeout").value
        assert isinstance(value, EnumValue)
        assert value.fields["value"] == TextValue("0.0000001s")

    def test_host_consumed_seed_reflects_engine_base(self) -> None:
        from agm.agl.runtime.params import build_engine_config_base

        engine_base = build_engine_config_base({"runner": "gpt", "log": True})
        s = _session(engine_base=engine_base)
        _ok(s, "import std/config")
        # A fresh session reads the host-provided base value, not the bare default.
        assert _read(s, "runner") == TextValue("gpt")
        assert _read(s, "log") == BoolValue(True)

    def test_reset_clears_host_consumed_write(self) -> None:
        s = _session()
        _ok(s, "import std/config")
        _ok(s, 'std/config::runner := "codex"')
        assert _read(s, "runner") == TextValue("codex")
        s.reset()
        _ok(s, "import std/config")
        assert _read(s, "runner") == TextValue(DEFAULT_AGENT_RUNNER)


# ---------------------------------------------------------------------------
# Partial-failure discipline
# ---------------------------------------------------------------------------


class TestPartialFailureDiscipline:
    """Setting writes completed before a runtime failure remain persistent."""

    def test_write_before_failure_persists(self) -> None:
        s = _session()
        _ok(s, "import std/config")
        result = s.eval_entry('std/config::runner := "codex"\nlet z: decimal = 1 / 0')
        assert not result.ok
        assert _read(s, "runner") == TextValue("codex")

    def test_runtime_live_write_before_failure_persists(self) -> None:
        s = _session()
        _ok(s, "import std/config")
        result = s.eval_entry("std/config::max-iters := 2\nlet z: decimal = 1 / 0")
        assert not result.ok
        assert _read(s, "max-iters") == IntValue(2)


class TestLiveHostReconfiguration:
    def test_runner_write_rebuilds_default_agent_for_later_entry(self) -> None:
        def old_agent(request: AgentRequest) -> AgentResponse:
            return AgentResponse(content="old")

        def build_runner(command: str) -> AgentFn:
            def rebuilt(request: AgentRequest) -> AgentResponse:
                return AgentResponse(content=command)

            return rebuilt

        policy = HostSettingsPolicy(
            build_runner=build_runner,
            resolve_trace_path=lambda enabled, log_file: None,
        )
        s = _session(default_agent=old_agent, host_settings_policy=policy)
        _ok(s, "import std/config")
        _ok(s, 'std/config::runner := "new-runner"')

        result = _ok(s, 'ask("which")')
        assert result.value == TextValue("new-runner")

        s.reset()
        result = _ok(s, 'ask("which")')
        assert result.value == TextValue(DEFAULT_AGENT_RUNNER)

    def test_log_file_write_repoints_later_repl_entries(self, tmp_path: Path) -> None:
        trace_path = tmp_path / "trace.jsonl"
        policy = HostSettingsPolicy(
            build_runner=lambda command: lambda request: AgentResponse(content=command),
            resolve_trace_path=lambda enabled, log_file: (
                Path(log_file) if enabled or log_file is not None else None
            ),
        )
        s = _session(host_settings_policy=policy)
        _ok(s, "import std/config")
        _ok(s, f'std/config::log-file := Some("{trace_path}")')
        _ok(s, 'print "later"')

        assert trace_path.exists()
        assert '"rendered": "later"' in trace_path.read_text(encoding="utf-8")

    def test_log_false_settles_into_no_log_for_later_entries(self, tmp_path: Path) -> None:
        """A deliberate ``log := false`` keeps later entries untraced."""
        trace_path = tmp_path / "trace.jsonl"
        policy = HostSettingsPolicy(
            build_runner=lambda command: lambda request: AgentResponse(content=command),
            resolve_trace_path=lambda enabled, log_file: (
                (Path(log_file) if log_file is not None else trace_path) if enabled else None
            ),
        )
        s = _session(host_settings_policy=policy, trace_path=trace_path)
        _ok(s, "import std/config")
        _ok(s, 'print "traced"')
        _ok(s, "std/config::log := false")

        result = _ok(s, 'print "untraced"')
        assert result.trace_path is None
        text = trace_path.read_text(encoding="utf-8")
        assert '"rendered": "traced"' in text
        assert "untraced" not in text
