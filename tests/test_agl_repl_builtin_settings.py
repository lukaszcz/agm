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

from agm.agl.repl import EntryResult, ReplSession
from agm.agl.runtime.agents import AgentFn
from agm.agl.runtime.host_settings import HostSettingsPolicy
from agm.agl.runtime.params import build_engine_config_seeds
from agm.agl.runtime.request import AgentRequest, AgentResponse
from agm.agl.semantics.values import BoolValue, EnumValue, IntValue, TextValue, Value
from agm.agl.setting_overrides import SettingOverride
from tests._agl_helpers import agent_value

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
    agent_dispatcher: AgentFn | None = None,
    engine_base: dict[str, Value] | None = None,
    setting_overrides: dict[str, SettingOverride] | None = None,
    host_settings_policy: HostSettingsPolicy | None = None,
    trace_path: Path | None = None,
) -> ReplSession:
    return ReplSession(
        stdlib_root=_STDLIB_ROOT,
        agent_dispatcher=agent_dispatcher,
        engine_base=engine_base,
        setting_overrides=setting_overrides,
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

    def test_default_agent_write_persists_two_entries_later(self) -> None:
        s = _session()
        _ok(s, "import std/config")
        _ok(s, 'std/config::default-agent := AgentCommand("scripted")')
        _ok(s, "let unrelated = 1")
        value = _read(s, "default-agent")
        assert isinstance(value, EnumValue)
        assert value.variant == "AgentCommand"
        assert value.fields["command"] == TextValue("scripted")


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
        s = _session(agent_dispatcher=agent)
        # Confirm the fenced reply parses in the default (lenient) mode.
        r_lenient = _ok(s, 'let a: int = ask """how many"""')
        assert r_lenient.value == IntValue(42)

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

    def test_explicit_false_strict_json_seed_overrides_and_survives_reset(
        self, tmp_path: Path
    ) -> None:
        stdlib_root = tmp_path / "stdlib"
        config_path = stdlib_root / "std" / "config.agl"
        config_path.parent.mkdir(parents=True)
        config_path.write_text("builtin var strict-json: bool = true\n", encoding="utf-8")
        (config_path.parent / "core.agl").write_text(
            (_STDLIB_ROOT / "std" / "core.agl").read_text(encoding="utf-8"), encoding="utf-8"
        )
        unseeded = ReplSession(stdlib_root=stdlib_root, default_stdlib=False)
        _ok(unseeded, "import std/config")
        assert _read(unseeded, "strict-json") == BoolValue(True)

        s = ReplSession(
            stdlib_root=stdlib_root,
            default_stdlib=False,
            engine_base=build_engine_config_seeds({"strict-json": False}),
        )

        _ok(s, "import std/config")
        assert _read(s, "strict-json") == BoolValue(False)
        _ok(s, "let unrelated = 1")
        assert _read(s, "strict-json") == BoolValue(False)

        _ok(s, "std/config::strict-json := true")
        assert _read(s, "strict-json") == BoolValue(True)

        s.reset()

        _ok(s, "import std/config")
        assert _read(s, "strict-json") == BoolValue(False)

    def test_unwritten_default_agent_declaration_survives_later_entries(self) -> None:
        """An unseeded, unwritten ``default-agent`` keeps its declared value.

        ``std/config`` is linked once, so only the first entry carries its
        ``builtin var`` initializer; later entries must read the value the
        session carried forward rather than losing it or inventing one.
        """
        s = _session()
        _ok(s, "import std/config")
        _ok(s, "let unrelated = 1")

        value = _read(s, "default-agent")

        assert isinstance(value, EnumValue)
        assert value.variant == "AgentClaude"

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


def _host_seeded_session(
    stdlib_root: Path,
    *,
    strict_json: bool | None = None,
    max_iters: int | None = None,
    timeout: str | None = None,
    log: bool | None = None,
    log_file: str | None = None,
    default_agent: str | None = None,
) -> ReplSession:
    """Build a session with explicit host seeds ONLY for the given keys.

    ``max-iters`` is seeded through the ``default_loop_limit`` driver
    argument; ``engine_base["max-iters"]`` is the other channel, exercised by
    ``TestMaxItersEngineBaseSeed`` below. ``ReplSession.__init__`` folds both
    into the same ``_engine_base`` entry.  ``default_agent`` is the raw
    ``AgentCommand`` command text (not AgL source) — a host-seeded ``Value``,
    like ``[exec] runner`` builds in production, rather than the AgL-literal
    ``SettingOverride`` path ``--agent``/``[exec] default-agent`` use.
    """
    raw: dict[str, object] = {}
    if strict_json is not None:
        raw["strict-json"] = strict_json
    if timeout is not None:
        raw["timeout"] = timeout
    if log is not None:
        raw["log"] = log
    if log_file is not None:
        raw["log-file"] = log_file
    engine_base = build_engine_config_seeds(raw)
    if default_agent is not None:
        engine_base["default-agent"] = agent_value("AgentCommand", command=default_agent)
    return ReplSession(
        stdlib_root=stdlib_root,
        default_strict_json=strict_json if strict_json is not None else False,
        default_loop_limit=max_iters,
        engine_base=engine_base,
    )


def _declared_defaults_session(tmp_path: Path) -> ReplSession:
    """Build a session whose ``std/config`` declares a distinct default per key.

    No key is host-seeded, so every key's effective value comes from the
    ``builtin var`` initializer evaluated the first time an entry imports
    ``std/config``.
    """
    stdlib_root = tmp_path / "stdlib"
    config_path = stdlib_root / "std" / "config.agl"
    config_path.parent.mkdir(parents=True)
    config_path.write_text(
        "open import std/core\n"
        'builtin var default-agent: Agent = AgentCommand("declared")\n'
        "builtin var strict-json: bool = true\n"
        "builtin var max-iters: int = 3\n"
        'builtin var timeout: Option[text] = Option[text]::Some("2s")\n'
        "builtin var log: bool = true\n"
        'builtin var log-file: Option[text] = Option[text]::Some("declared.jsonl")\n',
        encoding="utf-8",
    )
    (config_path.parent / "core.agl").write_text(
        (_STDLIB_ROOT / "std" / "core.agl").read_text(encoding="utf-8"), encoding="utf-8"
    )
    return ReplSession(stdlib_root=stdlib_root)


class TestResetHostSeedPrecedence:
    """A host-seeded key's ``:reset`` restores the host seed, discarding a source write.

    ``strict-json`` (including a falsy ``False`` host seed) is pinned in
    :class:`TestDefaultsAndSeeding` above; this class covers the remaining
    five keys.
    """

    def test_max_iters_host_seed_survives_a_source_write(self) -> None:
        s = _host_seeded_session(_STDLIB_ROOT, max_iters=5)
        _ok(s, "import std/config")
        _ok(s, "std/config::max-iters := 9")
        s.reset()
        _ok(s, "import std/config")
        assert _read(s, "max-iters") == IntValue(5)

    def test_timeout_engine_base_host_seed_survives_a_source_write(self) -> None:
        # The host seed arrives through ``engine_base`` (e.g. CLI/config), not
        # through the ``shell_exec_timeout`` driver argument.
        s = _host_seeded_session(_STDLIB_ROOT, timeout="3s")
        _ok(s, "import std/config")
        _ok(s, 'std/config::timeout := Some("99s")')
        s.reset()
        _ok(s, "import std/config")
        value = _read(s, "timeout")
        assert isinstance(value, EnumValue)
        assert value.fields["value"] == TextValue("3s")

    def test_timeout_driver_synthesized_host_seed_survives_a_source_write(self) -> None:
        # No ``engine_base["timeout"]``: the host seed is synthesized from the
        # ``shell_exec_timeout`` driver argument instead.
        s = ReplSession(stdlib_root=_STDLIB_ROOT, shell_exec_timeout=0.0000001)
        _ok(s, "import std/config")
        _ok(s, 'std/config::timeout := Some("99s")')
        s.reset()
        _ok(s, "import std/config")
        value = _read(s, "timeout")
        assert isinstance(value, EnumValue)
        assert value.fields["value"] == TextValue("0.0000001s")
        assert s._shell_exec_timeout == 0.0000001

    def test_log_host_seed_survives_a_source_write(self) -> None:
        s = _host_seeded_session(_STDLIB_ROOT, log=True)
        _ok(s, "import std/config")
        _ok(s, "std/config::log := false")
        s.reset()
        _ok(s, "import std/config")
        assert _read(s, "log") == BoolValue(True)

    def test_log_file_host_seed_survives_a_source_write(self) -> None:
        s = _host_seeded_session(_STDLIB_ROOT, log_file="host.jsonl")
        _ok(s, "import std/config")
        _ok(s, 'std/config::log-file := Some("written.jsonl")')
        s.reset()
        _ok(s, "import std/config")
        value = _read(s, "log-file")
        assert isinstance(value, EnumValue)
        assert value.fields["value"] == TextValue("host.jsonl")

    def test_default_agent_host_seed_survives_a_source_write(self) -> None:
        s = _host_seeded_session(_STDLIB_ROOT, default_agent="host")
        _ok(s, "import std/config")
        _ok(s, 'std/config::default-agent := AgentCommand("written")')
        s.reset()
        _ok(s, "import std/config")
        value = _read(s, "default-agent")
        assert isinstance(value, EnumValue)
        assert value.fields["command"] == TextValue("host")


class TestMaxItersEngineBaseSeed:
    """``engine_base['max-iters']`` is a first-class host seed, like every other key.

    An explicit ``engine_base["max-iters"]`` counts as a host seed exactly
    like a ``default_loop_limit`` argument does, and wins when the two
    disagree -- both fold into the same mapping (``ReplSession.__init__``).
    """

    def test_engine_base_seed_survives_a_source_write_across_reset(self) -> None:
        s = ReplSession(
            stdlib_root=_STDLIB_ROOT,
            engine_base=build_engine_config_seeds({"max-iters": 5}),
        )
        _ok(s, "import std/config")
        _ok(s, "std/config::max-iters := 9")
        s.reset()
        _ok(s, "import std/config")
        assert _read(s, "max-iters") == IntValue(5)

    def test_engine_base_seed_wins_over_the_default_loop_limit_argument(self) -> None:
        # The two seed channels deliberately disagree: an explicit
        # ``engine_base`` entry must win, mirroring every other key's
        # seed-over-driver-argument precedence (``TestResetSeedWinsOverDriverArgument``).
        s = ReplSession(
            stdlib_root=_STDLIB_ROOT,
            default_loop_limit=30,
            engine_base=build_engine_config_seeds({"max-iters": 5}),
        )
        assert s._default_loop_limit == 5
        s.reset()
        assert s._default_loop_limit == 5


class TestTimeoutSeedWinsOverDriverArgument:
    """An ``engine_base['timeout']`` seed wins over a disagreeing ``shell_exec_timeout``.

    ``shell_exec_timeout`` reaches the session twice: once as the typed
    ``engine_base`` seed mapping and once as the raw constructor argument,
    which is only a fallback for an unseeded key. These tests deliberately
    make the two disagree and pin that the session's own live field
    (``_shell_exec_timeout``) follows the seed, not the disagreeing argument.
    """

    def test_engine_base_seed_wins_over_the_shell_exec_timeout_argument(self) -> None:
        s = ReplSession(
            stdlib_root=_STDLIB_ROOT,
            shell_exec_timeout=30.0,
            engine_base=build_engine_config_seeds({"timeout": "5s"}),
        )
        assert s._shell_exec_timeout == 5.0

    def test_engine_base_seed_of_none_wins_over_the_shell_exec_timeout_argument(self) -> None:
        # An explicit empty ``Option`` (a "no timeout" host control) must win
        # over a disagreeing non-``None`` scalar, same as every other case.
        s = ReplSession(
            stdlib_root=_STDLIB_ROOT,
            shell_exec_timeout=30.0,
            engine_base=build_engine_config_seeds({"timeout": None}),
        )
        assert s._shell_exec_timeout is None


class TestSeedGovernsRuntimeEffectOverDriverArgument:
    """A disagreeing seed governs what an entry actually does, not just what it reads.

    ``strict-json`` and ``max-iters`` reach the session through both the typed
    ``engine_base`` seed mapping and a raw constructor argument. The seed wins,
    and these tests observe that where a user does: the JSON parsing mode an
    ``ask`` reply is held to, and the iteration cap an unguarded loop is cut off
    at. Each case first pins the disagreeing argument's own effect on a session
    that has no seed, so the seeded case demonstrably discriminates between the
    two channels.
    """

    def test_lenient_seed_keeps_a_fenced_reply_parseable_despite_a_strict_json_argument(
        self,
    ) -> None:
        # No seed: the argument alone makes the fenced reply unparseable.
        argument_only = ReplSession(
            stdlib_root=_STDLIB_ROOT,
            agent_dispatcher=_FencedAgent(),
            default_strict_json=True,
        )
        strict_result = argument_only.eval_entry('let a: int = ask """how many"""')
        assert not strict_result.ok
        assert strict_result.error is not None
        assert "AgentParseError" in strict_result.error.type_name

        # The same argument, now contradicted by a lenient seed: the reply parses.
        s = ReplSession(
            stdlib_root=_STDLIB_ROOT,
            agent_dispatcher=_FencedAgent(),
            default_strict_json=True,
            engine_base=build_engine_config_seeds({"strict-json": False}),
        )
        assert _ok(s, 'let a: int = ask """how many"""').value == IntValue(42)

    def test_engine_base_cap_stops_a_loop_the_default_loop_limit_argument_would_allow(
        self,
    ) -> None:
        loop = "var i = 0\ndo\n  i := i + 1\nuntil i >= 10\ni"
        # No seed: the argument's cap is loose enough for the loop to finish.
        argument_only = ReplSession(stdlib_root=_STDLIB_ROOT, default_loop_limit=30)
        assert _ok(argument_only, loop).value == IntValue(10)

        # The same argument, now contradicted by a tighter seed: the loop is cut off.
        s = ReplSession(
            stdlib_root=_STDLIB_ROOT,
            default_loop_limit=30,
            engine_base=build_engine_config_seeds({"max-iters": 5}),
        )
        result = s.eval_entry(loop)
        assert not result.ok
        assert result.error is not None
        assert "MaxIterationsExceeded" in result.error.type_name


class TestMaxItersRegisterIsolation:
    """A ``max-iters`` host seed never leaks into the host-consumed settings register.

    ``max-iters`` is a runtime-live key: normalizing it into ``_engine_base``
    must not let it reach ``_persisted_host_settings`` (the register mapping
    fed to the interpreter as ``builtin_host_settings`` for
    log/log-file/default-agent), or the interpreter would receive a
    register value it never reads a max-iters control from.
    """

    def test_default_loop_limit_seed_is_absent_from_the_host_settings_register(self) -> None:
        s = ReplSession(stdlib_root=_STDLIB_ROOT, default_loop_limit=5)
        assert "max-iters" not in s._persisted_host_settings
        s.reset()
        assert "max-iters" not in s._persisted_host_settings

    def test_engine_base_seed_is_absent_from_the_host_settings_register(self) -> None:
        s = ReplSession(
            stdlib_root=_STDLIB_ROOT,
            engine_base=build_engine_config_seeds({"max-iters": 5}),
        )
        assert "max-iters" not in s._persisted_host_settings
        s.reset()
        assert "max-iters" not in s._persisted_host_settings


class TestExplicitZeroLoopLimit:
    """A host ``default_loop_limit=0`` is an explicit "disable the safety valve" control.

    Unlike the *declared* ``max-iters = 0`` default that ``:reset`` collapses
    to ``None`` (``test_reset_keeps_a_declared_zero_max_iters_disabled``
    below), a host ``0`` keeps its exact value across construction and
    ``:reset``: collapsing it would let a declared nonzero default reassert
    itself and silently undo the host's explicit disable.
    """

    def test_zero_survives_construction_and_reset_with_no_declared_default(self) -> None:
        s = ReplSession(stdlib_root=_STDLIB_ROOT, default_loop_limit=0)
        assert s._default_loop_limit == 0
        s.reset()
        assert s._default_loop_limit == 0

    def test_zero_disables_the_loop_cap_despite_a_declared_nonzero_default(
        self, tmp_path: Path
    ) -> None:
        stdlib_root = tmp_path / "stdlib"
        config_path = stdlib_root / "std" / "config.agl"
        config_path.parent.mkdir(parents=True)
        config_path.write_text("builtin var max-iters: int = 3\n", encoding="utf-8")
        (config_path.parent / "core.agl").write_text(
            (_STDLIB_ROOT / "std" / "core.agl").read_text(encoding="utf-8"), encoding="utf-8"
        )
        s = ReplSession(stdlib_root=stdlib_root, default_stdlib=False, default_loop_limit=0)
        _ok(s, "import std/config")
        s.reset()
        assert s._default_loop_limit == 0

        _ok(s, "import std/config")
        # A loop well past the declared cap of 3 must run to completion: the
        # host's explicit 0 disables the safety valve outright rather than
        # falling through to the declared default.
        result = s.eval_entry("var i = 0\ndo\n  i := i + 1\nuntil i >= 10\ni")
        assert result.ok, f"entry failed: {result.diagnostics} {result.error}"
        assert result.value == IntValue(10)


class TestResetDeclaredDefaultPrecedence:
    """An unseeded key's ``:reset`` restores ``std/config``'s declared default.

    Each test writes a value that differs from BOTH the declared default and
    any host-side floor, so a wrong precedence (falling through to a host
    floor instead of the declaration) would be caught.
    """

    def test_strict_json_declared_default_survives_a_source_write(self, tmp_path: Path) -> None:
        s = _declared_defaults_session(tmp_path)
        _ok(s, "import std/config")
        _ok(s, "std/config::strict-json := false")
        s.reset()
        _ok(s, "import std/config")
        assert _read(s, "strict-json") == BoolValue(True)

    def test_max_iters_declared_default_survives_a_source_write(self, tmp_path: Path) -> None:
        s = _declared_defaults_session(tmp_path)
        _ok(s, "import std/config")
        _ok(s, "std/config::max-iters := 99")
        s.reset()
        _ok(s, "import std/config")
        assert _read(s, "max-iters") == IntValue(3)

    def test_timeout_declared_default_survives_a_source_write(self, tmp_path: Path) -> None:
        s = _declared_defaults_session(tmp_path)
        _ok(s, "import std/config")
        _ok(s, 'std/config::timeout := Some("99s")')
        s.reset()
        _ok(s, "import std/config")
        value = _read(s, "timeout")
        assert isinstance(value, EnumValue)
        assert value.fields["value"] == TextValue("2s")

    def test_log_declared_default_survives_a_source_write(self, tmp_path: Path) -> None:
        s = _declared_defaults_session(tmp_path)
        _ok(s, "import std/config")
        _ok(s, "std/config::log := false")
        s.reset()
        _ok(s, "import std/config")
        assert _read(s, "log") == BoolValue(True)

    def test_log_file_declared_default_survives_a_source_write(self, tmp_path: Path) -> None:
        s = _declared_defaults_session(tmp_path)
        _ok(s, "import std/config")
        _ok(s, 'std/config::log-file := Some("written.jsonl")')
        s.reset()
        _ok(s, "import std/config")
        value = _read(s, "log-file")
        assert isinstance(value, EnumValue)
        assert value.fields["value"] == TextValue("declared.jsonl")

    def test_default_agent_declared_default_survives_a_source_write(self, tmp_path: Path) -> None:
        s = _declared_defaults_session(tmp_path)
        _ok(s, "import std/config")
        _ok(s, 'std/config::default-agent := AgentCommand("written")')
        s.reset()
        _ok(s, "import std/config")
        value = _read(s, "default-agent")
        assert isinstance(value, EnumValue)
        assert value.fields["command"] == TextValue("declared")


class TestResetSeedWinsOverDriverArgument:
    """``:reset`` restores the seed value, not the ``ReplSession`` driver argument.

    ``strict-json`` and ``timeout`` reach the session twice: once as an explicit
    seed value and once as a driver argument (``default_strict_json`` /
    ``shell_exec_timeout``) that is only a fallback for an unseeded key.  The
    hosting command derives both from the same resolved configuration, so they
    agree there; these tests deliberately make them disagree to pin which one
    ``:reset`` restores.  The seed must win, so that a reset session's live
    interpreter setting agrees with the register a later entry reads.

    The timeout cases observe ``_shell_exec_timeout`` directly, and immediately
    after the reset: a seeded timeout register is what every following entry is
    seeded from, so it would mask a stale live field on the next entry.
    """

    def test_strict_json_seed_wins_over_the_driver_argument(self) -> None:
        s = ReplSession(
            stdlib_root=_STDLIB_ROOT,
            default_strict_json=True,
            engine_base=build_engine_config_seeds({"strict-json": False}),
        )
        _ok(s, "import std/config")
        assert _read(s, "strict-json") == BoolValue(False)
        _ok(s, "std/config::strict-json := true")
        s.reset()
        _ok(s, "import std/config")
        assert _read(s, "strict-json") == BoolValue(False)

    def test_timeout_seed_wins_over_the_driver_argument(self) -> None:
        s = ReplSession(
            stdlib_root=_STDLIB_ROOT,
            shell_exec_timeout=30.0,
            engine_base=build_engine_config_seeds({"timeout": "5s"}),
        )
        _ok(s, "import std/config")
        _ok(s, 'std/config::timeout := Some("99s")')
        s.reset()
        assert s._shell_exec_timeout == 5.0
        _ok(s, "import std/config")
        value = _read(s, "timeout")
        assert isinstance(value, EnumValue)
        assert value.fields["value"] == TextValue("5s")

    def test_timeout_seeded_as_none_disables_the_timeout_across_reset(self) -> None:
        # An empty ``Option`` is an explicit "no timeout" control, so it must
        # not fall back to the ``shell_exec_timeout`` driver argument.
        s = ReplSession(
            stdlib_root=_STDLIB_ROOT,
            shell_exec_timeout=30.0,
            engine_base=build_engine_config_seeds({"timeout": None}),
        )
        _ok(s, "import std/config")
        assert s._shell_exec_timeout is None
        _ok(s, 'std/config::timeout := Some("99s")')
        s.reset()
        assert s._shell_exec_timeout is None
        _ok(s, "import std/config")
        value = _read(s, "timeout")
        assert isinstance(value, EnumValue)
        assert value.variant == "None"


# ---------------------------------------------------------------------------
# Partial-failure discipline
# ---------------------------------------------------------------------------


class TestPartialFailureDiscipline:
    """Setting writes completed before a runtime failure remain persistent."""

    def test_runtime_live_write_before_failure_persists(self) -> None:
        s = _session()
        _ok(s, "import std/config")
        result = s.eval_entry("std/config::max-iters := 2\nlet z: decimal = 1 / 0")
        assert not result.ok
        assert _read(s, "max-iters") == IntValue(2)


class TestLiveHostReconfiguration:
    def test_log_file_write_repoints_later_repl_entries(self, tmp_path: Path) -> None:
        trace_path = tmp_path / "trace.jsonl"
        policy = HostSettingsPolicy(
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
            resolve_trace_path=lambda enabled, log_file: (
                (Path(log_file) if log_file is not None else trace_path) if enabled else None
            ),
        )
        s = _session(host_settings_policy=policy, trace_path=trace_path)
        _ok(s, "import std/config")
        _ok(s, "std/config::log := true")
        _ok(s, 'print "traced"')
        _ok(s, "std/config::log := false")

        result = _ok(s, 'print "untraced"')
        assert result.trace_path is None
        text = trace_path.read_text(encoding="utf-8")
        assert '"rendered": "traced"' in text
        assert "untraced" not in text


def test_reset_keeps_a_declared_zero_max_iters_disabled(tmp_path: Path) -> None:
    """A zero declaration default remains an unlimited loop setting after reset."""
    stdlib_root = tmp_path / "stdlib"
    config_path = stdlib_root / "std" / "config.agl"
    config_path.parent.mkdir(parents=True)
    config_path.write_text("builtin var max-iters: int = 0\n", encoding="utf-8")
    (config_path.parent / "core.agl").write_text(
        (_STDLIB_ROOT / "std" / "core.agl").read_text(), encoding="utf-8"
    )
    session = ReplSession(stdlib_root=stdlib_root, default_stdlib=False)

    _ok(session, "import std/config")
    _ok(session, "std/config::max-iters := 2")
    session.reset()

    assert session._default_loop_limit is None


def test_reset_preserves_an_explicit_host_loop_limit(tmp_path: Path) -> None:
    """A host loop limit takes precedence over a declaration default after reset."""
    stdlib_root = tmp_path / "stdlib"
    config_path = stdlib_root / "std" / "config.agl"
    config_path.parent.mkdir(parents=True)
    config_path.write_text("builtin var max-iters: int = 0\n", encoding="utf-8")
    (config_path.parent / "core.agl").write_text(
        (_STDLIB_ROOT / "std" / "core.agl").read_text(), encoding="utf-8"
    )
    session = ReplSession(
        stdlib_root=stdlib_root,
        default_stdlib=False,
        default_loop_limit=2,
    )

    _ok(session, "import std/config")
    session.reset()

    assert session._default_loop_limit == 2


def test_reset_uses_declared_live_engine_defaults(tmp_path: Path) -> None:
    """Reset reapplies std/config defaults without a removed runner setting."""
    stdlib_root = tmp_path / "stdlib"
    config_path = stdlib_root / "std" / "config.agl"
    config_path.parent.mkdir(parents=True)
    config_path.write_text(
        "open import std/core\n"
        'builtin var default-agent: Agent = AgentCommand("declared")\n'
        "builtin var strict-json: bool = true\n"
        "builtin var max-iters: int = 3\n"
        'builtin var timeout: Option[text] = Option[text]::Some("2s")\n'
        "builtin var log: bool = false\n"
        "builtin var log-file: Option[text] = Option[text]::None\n"
    )
    (config_path.parent / "core.agl").write_text((_STDLIB_ROOT / "std" / "core.agl").read_text())
    session = ReplSession(stdlib_root=stdlib_root, default_stdlib=False)

    _ok(session, "open import std/core\nimport std/config\nstd/config::strict-json")
    _ok(session, "std/config::strict-json := false\nstd/config::max-iters := 0")
    session.reset()

    assert session._default_strict_json is True
    assert session._default_loop_limit == 3
    assert session._shell_exec_timeout == 2.0


# ---------------------------------------------------------------------------
# Host-supplied AgL setting overrides (--agent / [exec] default-agent)
# ---------------------------------------------------------------------------


class TestSettingOverrideThreading:
    """``setting_overrides`` splices AgL source in as ``std/config``'s default.

    Mirrors ``PipelineDriver.prepare_parsed_entry``'s ``setting_overrides``
    seam (``tests/test_agl_pipeline_setting_overrides.py``), but threaded
    through the REPL's own module-graph load instead of a batch one: the
    override is spliced in the first time an entry loads ``std/config`` (see
    ``EntryPipeline.eval_entry``), so it is resolved, type-checked, and
    constant-checked by that entry's own compilation.
    """

    def test_override_is_observed_on_the_first_entry_that_loads_std_config(self) -> None:
        s = _session(
            setting_overrides={
                "default-agent": SettingOverride(
                    source='AgentCommand("overridden")', origin="--agent"
                )
            }
        )
        _ok(s, "import std/config")
        value = _read(s, "default-agent")
        assert isinstance(value, EnumValue)
        assert value.variant == "AgentCommand"
        assert value.fields["command"] == TextValue("overridden")

    def test_source_write_still_overrides_it_afterward(self) -> None:
        s = _session(
            setting_overrides={
                "default-agent": SettingOverride(
                    source='AgentCommand("overridden")', origin="--agent"
                )
            }
        )
        _ok(s, "import std/config")
        assert _read(s, "default-agent").variant == "AgentCommand"
        _ok(s, 'std/config::default-agent := AgentClaude("haiku", "low")')
        value = _read(s, "default-agent")
        assert isinstance(value, EnumValue)
        assert value.variant == "AgentClaude"
        assert value.fields["model"] == TextValue("haiku")

    def test_malformed_override_fails_only_the_entry_that_triggers_it(self) -> None:
        """A bad literal rejects only the entry that loads ``std/config``, naming the origin.

        The rejection surfaces from that entry's own ordinary type-checking of
        ``std/config``'s builtin var default — the spliced expression's span
        carries the override's origin as its source label, so
        ``format_diagnostic`` names it in the location prefix.  With
        ``default_stdlib=False`` an entry that never imports ``std/config``
        never triggers the splice at all, so the session stays usable for
        those both before and after the rejected attempt.
        """
        from agm.agl.diagnostics import format_diagnostic

        s = ReplSession(
            stdlib_root=_STDLIB_ROOT,
            default_stdlib=False,
            setting_overrides={
                "default-agent": SettingOverride(source='"not-an-agent"', origin="--agent")
            },
        )
        assert _ok(s, "1 + 1").value == IntValue(2)

        result = s.eval_entry("import std/config")
        assert not result.ok
        assert result.diagnostics
        assert any("--agent" in format_diagnostic(diag) for diag in result.diagnostics)

        # Nothing from the rejected entry was promoted or persisted.
        assert _ok(s, "1 + 1").value == IntValue(2)

    def test_unparseable_override_source_fails_only_the_entry_that_triggers_it(self) -> None:
        """An override rejected by the REPL's own splice call, not by typechecking.

        Unlike a type-invalid override (caught later by ``std/config``'s
        ordinary typechecking, see
        ``test_malformed_override_fails_only_the_entry_that_triggers_it``) or
        unparseable command TEXT (caught at interpreter construction),
        unparseable override SOURCE is diagnosed by
        ``_apply_setting_overrides`` itself at the REPL's own splice call site
        (``EntryPipeline.eval_entry``, mirroring
        ``PipelineDriver.prepare_parsed_entry``'s batch splice).
        """
        from agm.agl.diagnostics import format_diagnostic

        s = ReplSession(
            stdlib_root=_STDLIB_ROOT,
            default_stdlib=False,
            setting_overrides={"default-agent": SettingOverride(source="(", origin="--agent")},
        )
        assert _ok(s, "1 + 1").value == IntValue(2)

        result = s.eval_entry("import std/config")
        assert not result.ok
        assert result.diagnostics
        assert any("--agent" in format_diagnostic(diag) for diag in result.diagnostics)

        # Nothing from the rejected entry was promoted or persisted.
        assert _ok(s, "1 + 1").value == IntValue(2)

    def test_unparseable_command_text_rejects_only_the_entry_that_triggers_it(self) -> None:
        """A well-typed ``AgentCommand`` whose text does not shell-split is rejected too.

        Unlike a type-invalid override (caught by ``std/config``'s own
        typechecking, see ``test_malformed_override_fails_only_the_entry_that_triggers_it``),
        an unclosed quote in the command text is a syntactically valid
        constant ``Agent`` expression, so it is only caught when the entry's
        interpreter is constructed and materializes the winning
        ``default-agent`` value -- reported as an ordinary per-entry
        diagnostic (``result.error`` stays ``None``), never a process exit.
        """
        s = ReplSession(
            stdlib_root=_STDLIB_ROOT,
            default_stdlib=False,
            setting_overrides={
                "default-agent": SettingOverride(
                    source='AgentCommand("nonexistent-bin -p \'oops")', origin="--agent"
                )
            },
        )
        assert _ok(s, "1 + 1").value == IntValue(2)

        result = s.eval_entry("import std/config")
        assert not result.ok
        assert result.diagnostics
        assert result.error is None

        # Nothing from the rejected entry was promoted or persisted.
        assert _ok(s, "1 + 1").value == IntValue(2)

    def test_node_ids_stay_disjoint_after_a_spliced_override(self) -> None:
        """The splice's own parsed nodes never collide with later entries' ids.

        A regression guard for threading ``_apply_setting_overrides``'s
        updated next-id back into the REPL's node-id counter: reusing the
        pre-splice id would let a later entry's declaration collide with an
        identity the spliced override expression already claimed.
        """
        s = _session(
            setting_overrides={
                "default-agent": SettingOverride(
                    source='AgentCommand("overridden")', origin="--agent"
                )
            }
        )
        _ok(s, "import std/config")
        _ok(s, "let a = 1")
        _ok(s, "let b = 2")
        result = _ok(s, "a + b")
        assert result.value == IntValue(3)

    def test_reset_reapplies_the_override_after_a_fresh_std_config_load(self) -> None:
        s = _session(
            setting_overrides={
                "default-agent": SettingOverride(
                    source='AgentCommand("overridden")', origin="--agent"
                )
            }
        )
        _ok(s, "import std/config")
        assert _read(s, "default-agent").variant == "AgentCommand"
        _ok(s, 'std/config::default-agent := AgentClaude("haiku", "low")')
        s.reset()

        _ok(s, "import std/config")
        value = _read(s, "default-agent")
        assert isinstance(value, EnumValue)
        assert value.variant == "AgentCommand"
        assert value.fields["command"] == TextValue("overridden")
