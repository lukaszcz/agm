"""Cross-entry persistence of ``std.config`` engine settings in the REPL.

AgL exposes the six engine settings as ``builtin var`` declarations in the
``std.config`` stdlib module.  A REPL user reads a setting with
``std.config::KEY`` and writes it with ``std.config::KEY := VALUE`` after an
``import std.config``.  These tests assert that such writes persist across REPL
entries with full parity for all six keys:

* reading a setting in a later entry reflects the most recent earlier write, and
* the runtime-live effects (loop cap, strict-json parsing, shell-exec timeout)
  carry forward to later entries.

Agents are always mocked — no real agent is ever run.
"""

from __future__ import annotations

from pathlib import Path

from agm.agl.repl import EntryResult, ReplSession
from agm.agl.runtime.agents import AgentFn
from agm.agl.runtime.request import AgentRequest, AgentResponse
from agm.agl.semantics.values import BoolValue, EnumValue, IntValue, TextValue, Value

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
) -> ReplSession:
    return ReplSession(
        stdlib_root=_STDLIB_ROOT,
        default_agent=default_agent,
        engine_base=engine_base,
    )


def _ok(session: ReplSession, text: str) -> EntryResult:
    result = session.eval_entry(text)
    assert result.ok, f"entry {text!r} failed: {result.diagnostics} {result.error}"
    return result


def _read(session: ReplSession, key: str) -> Value:
    """Import-and-read *key*, returning the read value of a later-entry read."""
    result = _ok(session, f"std.config::{key}")
    assert result.value is not None
    return result.value


# ---------------------------------------------------------------------------
# Cross-entry persistence for each of the six keys
# ---------------------------------------------------------------------------


class TestCrossEntryPersistence:
    """A write in entry N is visible to a read two entries later (N+2)."""

    def test_max_iters_write_persists_two_entries_later(self) -> None:
        s = _session()
        _ok(s, "import std.config")
        _ok(s, "std.config::max-iters := 7")
        _ok(s, "let unrelated = 1")
        assert _read(s, "max-iters") == IntValue(7)

    def test_strict_json_write_persists_two_entries_later(self) -> None:
        s = _session()
        _ok(s, "import std.config")
        _ok(s, "std.config::strict-json := true")
        _ok(s, "let unrelated = 1")
        assert _read(s, "strict-json") == BoolValue(True)

    def test_timeout_write_persists_two_entries_later(self) -> None:
        s = _session()
        _ok(s, "import std.config")
        _ok(s, 'std.config::timeout := Some("45s")')
        _ok(s, "let unrelated = 1")
        value = _read(s, "timeout")
        assert isinstance(value, EnumValue)
        assert value.variant == "Some"
        # The written timeout is retained as the live shell-exec timeout.
        assert s._shell_exec_timeout == 45.0

    def test_runner_write_persists_two_entries_later(self) -> None:
        s = _session()
        _ok(s, "import std.config")
        _ok(s, 'std.config::runner := "codex"')
        _ok(s, "let unrelated = 1")
        assert _read(s, "runner") == TextValue("codex")

    def test_log_write_persists_two_entries_later(self) -> None:
        s = _session()
        _ok(s, "import std.config")
        _ok(s, "std.config::log := true")
        _ok(s, "let unrelated = 1")
        assert _read(s, "log") == BoolValue(True)

    def test_log_file_write_persists_two_entries_later(self) -> None:
        s = _session()
        _ok(s, "import std.config")
        _ok(s, 'std.config::log-file := Some("trace.jsonl")')
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
        _ok(s, "import std.config")
        _ok(s, "std.config::max-iters := 2")
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
        assert r_lenient.value == IntValue(42)

        _ok(s, "import std.config")
        _ok(s, "std.config::strict-json := true")
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
    """A session that never writes ``std.config`` reads the engine defaults."""

    def test_untouched_session_reads_engine_defaults(self) -> None:
        s = _session()
        _ok(s, "import std.config")
        assert _read(s, "runner") == TextValue("claude")
        assert _read(s, "log") == BoolValue(False)
        log_file = _read(s, "log-file")
        assert isinstance(log_file, EnumValue)
        assert log_file.variant == "None"
        assert _read(s, "strict-json") == BoolValue(False)
        assert _read(s, "max-iters") == IntValue(5)
        timeout = _read(s, "timeout")
        assert isinstance(timeout, EnumValue)
        assert timeout.variant == "None"

    def test_host_consumed_seed_reflects_engine_base(self) -> None:
        from agm.agl.runtime.params import build_engine_config_base

        engine_base = build_engine_config_base({"runner": "gpt", "log": True})
        s = _session(engine_base=engine_base)
        _ok(s, "import std.config")
        # A fresh session reads the host-provided base value, not the bare default.
        assert _read(s, "runner") == TextValue("gpt")
        assert _read(s, "log") == BoolValue(True)

    def test_reset_clears_host_consumed_write(self) -> None:
        s = _session()
        _ok(s, "import std.config")
        _ok(s, 'std.config::runner := "codex"')
        assert _read(s, "runner") == TextValue("codex")
        s.reset()
        _ok(s, "import std.config")
        assert _read(s, "runner") == TextValue("claude")


# ---------------------------------------------------------------------------
# Partial-failure discipline
# ---------------------------------------------------------------------------


class TestPartialFailureDiscipline:
    """A host-consumed write in a failed entry must not persist."""

    def test_write_before_failure_does_not_persist(self) -> None:
        s = _session()
        _ok(s, "import std.config")
        # The runner write fires first, then a runtime error aborts the entry.
        result = s.eval_entry('std.config::runner := "codex"\nlet z: decimal = 1 / 0')
        assert not result.ok
        assert _read(s, "runner") == TextValue("claude")
