"""Cross-entry persistence of ``std/config`` engine settings in the REPL.

AgL exposes the five engine settings as ``builtin var`` declarations in the
``std/config`` stdlib module.  A REPL user reads a setting with
``std/config::KEY`` and writes it with ``std/config::KEY := VALUE`` after an
``import std/config``.  These tests assert that such writes persist across REPL
entries with full parity for all five keys:

* reading a setting in a later entry reflects the most recent earlier write, and
* the runtime-live effects (strict-json parsing, shell-exec timeout)
  carry forward to later entries.

Agents are always mocked — no real agent is ever run.
"""

from __future__ import annotations

from pathlib import Path
from shutil import copyfile

import pytest

from agm.agl.repl import ReplSession
from agm.agl.runtime.engine_config import build_engine_config_seeds
from agm.agl.runtime.host_settings import HostSettingsPolicy
from agm.agl.runtime.request import AgentRequest, AgentResponse
from agm.agl.semantics.values import BoolValue, IntValue, RecordValue, TextValue, Value
from tests._agl_helpers import (
    agent_value,
)
from tests._agl_helpers import (
    eval_ok as _ok,
)
from tests._agl_helpers import (
    read_config_result as _read_result,
)
from tests._agl_helpers import (
    record_variant as _variant,
)
from tests._agl_helpers import (
    repl_session as _session,
)
from tests._agl_helpers import (
    unopened_repl_session as _unopened_session,
)

_STDLIB_ROOT = Path(__file__).resolve().parents[1] / "packages" / "stdlib"


def _copy_core_and_option(directory: Path) -> None:
    """Copy standard-library dependencies required by a custom ``std/config``."""
    for source in (_STDLIB_ROOT / "src").iterdir():
        if source.is_file() and source.name != "config.agl":
            copyfile(source, directory / source.name)
    config_path = directory / "config.agl"
    config = config_path.read_text(encoding="utf-8")
    additions = ""
    if "builtin var timeout" not in config:
        additions += "builtin var timeout: Option[text] = Option[text]::None\n"
    if "builtin var default-agent" not in config:
        additions += 'builtin var default-agent: Agent = AgentCommand("runner")\n'
    if "builtin var default-sandbox" not in config:
        additions += "builtin var default-sandbox: AgentSandbox = Disabled\n"
    if additions:
        # An existing ``import std/prelude::{Option, ...}`` line may not yet
        # name every symbol a freshly added default needs: extend it in place
        # rather than adding a second, conflicting import of the same names.
        if "std/prelude::{Option" in config:
            config = config.replace(
                "std/prelude::{Option, Agent}",
                "std/prelude::{Option, Agent, AgentSandbox}",
            )
            imports = ""
        else:
            imports = "import std/prelude::{Option, Agent, AgentSandbox}\n"
        config_path.write_text(imports + config + additions, encoding="utf-8")


class _FencedAgent:
    """A fake ``AgentFn`` returning a fenced-JSON reply (lenient-only)."""

    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, request: AgentRequest) -> AgentResponse:
        self.calls += 1
        return AgentResponse(content="```json\n42\n```")


def _read(session: ReplSession, key: str) -> Value:
    """Import-and-read *key*, returning the read value of a later-entry read."""
    value = _read_result(session, key).value
    assert value is not None
    return value


def _assert_setting(session: ReplSession, key: str, expected: object) -> None:
    """Assert ``std/config::key`` reads as *expected*.

    A scalar setting compares directly.  An ``Option``/``Agent`` setting is
    given as a ``(variant, field, payload)`` triple, compared against the
    value's terminal variant name (see :func:`_variant`) and that field.
    """
    result = _read_result(session, key)
    if isinstance(expected, tuple):
        variant, field_name, payload = expected
        assert isinstance(result.value, RecordValue)
        assert result.descriptors is not None
        assert _variant(result.value, result.descriptors) == variant
        assert result.value.fields[field_name] == payload
    else:
        assert result.value == expected


# ---------------------------------------------------------------------------
# Cross-entry persistence for each of the five keys
# ---------------------------------------------------------------------------


_PERSISTED_WRITES = [
    pytest.param("strict-json", "true", BoolValue(True), id="strict-json"),
    pytest.param("timeout", 'Some("45s")', ("Some", "value", TextValue("45s")), id="timeout"),
    pytest.param("trace", "true", BoolValue(True), id="trace"),
    pytest.param(
        "trace-file",
        'Some("trace.jsonl")',
        ("Some", "value", TextValue("trace.jsonl")),
        id="trace-file",
    ),
    pytest.param(
        "default-agent",
        'AgentCommand("scripted")',
        ("AgentCommand", "command", TextValue("scripted")),
        id="default-agent",
    ),
]


class TestCrossEntryPersistence:
    """A write in entry N is visible to a read two entries later (N+2)."""

    @pytest.mark.parametrize(("key", "written", "expected"), _PERSISTED_WRITES)
    def test_write_persists_two_entries_later(
        self, key: str, written: str, expected: object
    ) -> None:
        s = _session()
        _ok(s, "import std/config")
        _ok(s, f"std/config::{key} := {written}")
        _ok(s, "let unrelated = 1")
        _assert_setting(s, key, expected)


# ---------------------------------------------------------------------------
# Runtime-live effect carry-forward
# ---------------------------------------------------------------------------


class TestRuntimeLiveEffectCarryForward:
    """The shell-exec timeout and strict-json parsing effects apply in later entries."""

    def test_timeout_write_retains_the_live_shell_exec_timeout(self) -> None:
        s = _session()
        _ok(s, "import std/config")
        _ok(s, 'std/config::timeout := Some("45s")')
        _ok(s, "let unrelated = 1")
        # The written timeout is retained as the live shell-exec timeout.
        assert s._shell_exec_timeout == 45.0

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
        config_path = stdlib_root / "src" / "config.agl"
        config_path.parent.mkdir(parents=True)
        config_path.write_text("builtin var strict-json: bool = true\n", encoding="utf-8")
        _copy_core_and_option(config_path.parent)
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

        result = _read_result(s, "default-agent")

        assert isinstance(result.value, RecordValue)
        assert result.descriptors is not None
        assert _variant(result.value, result.descriptors) == "AgentClaude"

    def test_host_timeout_seed_round_trips_without_disabling_live_timeout(self) -> None:
        s = _session(stdlib_root=_STDLIB_ROOT, shell_exec_timeout=0.0000001)
        _ok(s, "import std/config")

        value = _read(s, "timeout")
        assert isinstance(value, RecordValue)
        assert value.fields["value"] == TextValue("0.0000001s")
        _ok(s, "std/config::timeout := std/config::timeout")
        assert s._shell_exec_timeout == 0.0000001

        s.reset()
        value = _ok(s, "import std/config\nstd/config::timeout").value
        assert isinstance(value, RecordValue)
        assert value.fields["value"] == TextValue("0.0000001s")


def _host_seeded_session(
    stdlib_root: Path,
    *,
    strict_json: bool | None = None,
    timeout: str | None = None,
    trace: bool | None = None,
    trace_file: str | None = None,
    default_agent: str | None = None,
) -> ReplSession:
    """Build a session with explicit host seeds ONLY for the given keys.

    ``default_agent`` is the raw ``AgentCommand`` command text, built into the
    same typed ``Value`` seed ``--default-agent``/``[exec] default-agent``
    decode into via ``cli_support.engine_seeds``.
    """
    raw: dict[str, object] = {}
    if strict_json is not None:
        raw["strict-json"] = strict_json
    if timeout is not None:
        raw["timeout"] = timeout
    if trace is not None:
        raw["trace"] = trace
    if trace_file is not None:
        raw["trace-file"] = trace_file
    engine_base = build_engine_config_seeds(raw)
    if default_agent is not None:
        engine_base["default-agent"] = agent_value("AgentCommand", command=default_agent)
    return _session(
        stdlib_root=stdlib_root,
        default_strict_json=strict_json if strict_json is not None else False,
        engine_base=engine_base,
    )


@pytest.fixture(scope="module")
def declared_defaults_stdlib(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A standard library whose ``std/config`` declares a distinct default per key.

    Built once and never written to afterwards, so every test that opens a
    session over it reads the same declarations and reuses the same checked
    bootstrap image.  Each test still gets its own session, so no session
    state crosses between them.
    """
    stdlib_root = tmp_path_factory.mktemp("declared-defaults") / "stdlib"
    config_path = stdlib_root / "src" / "config.agl"
    config_path.parent.mkdir(parents=True)
    config_path.write_text(
        "import std/prelude::*\n"
        'builtin var default-agent: Agent = AgentCommand("declared")\n'
        "builtin var strict-json: bool = true\n"
        'builtin var timeout: Option[text] = Option[text]::Some("2s")\n'
        "builtin var trace: bool = true\n"
        'builtin var trace-file: Option[text] = Option[text]::Some("declared.jsonl")\n',
        encoding="utf-8",
    )
    _copy_core_and_option(config_path.parent)
    return stdlib_root


_HOST_SEED_PRECEDENCE = [
    pytest.param(
        {"timeout": "3s"},
        "timeout",
        'Some("99s")',
        ("Some", "value", TextValue("3s")),
        id="timeout",
    ),
    pytest.param({"trace": True}, "trace", "false", BoolValue(True), id="trace"),
    pytest.param(
        {"trace_file": "host.jsonl"},
        "trace-file",
        'Some("written.jsonl")',
        ("Some", "value", TextValue("host.jsonl")),
        id="trace-file",
    ),
    pytest.param(
        {"default_agent": "host"},
        "default-agent",
        'AgentCommand("written")',
        ("AgentCommand", "command", TextValue("host")),
        id="default-agent",
    ),
]


class TestResetHostSeedPrecedence:
    """A host-seeded key's ``:reset`` restores the host seed, discarding a source write.

    ``strict-json`` (including a falsy ``False`` host seed) is pinned in
    :class:`TestDefaultsAndSeeding` above; this class covers the remaining
    four keys.  The ``timeout`` case seeds through ``engine_base`` (e.g.
    CLI/config); the driver-argument channel gets its own test below.
    """

    @pytest.mark.parametrize(("seed", "key", "written", "expected"), _HOST_SEED_PRECEDENCE)
    def test_host_seed_survives_a_source_write(
        self, seed: dict[str, object], key: str, written: str, expected: object
    ) -> None:
        s = _host_seeded_session(_STDLIB_ROOT, **seed)
        _ok(s, "import std/config")
        _ok(s, f"std/config::{key} := {written}")
        s.reset()
        _ok(s, "import std/config")
        _assert_setting(s, key, expected)

    def test_timeout_driver_synthesized_host_seed_survives_a_source_write(self) -> None:
        # No ``engine_base["timeout"]``: the host seed is synthesized from the
        # ``shell_exec_timeout`` driver argument instead.
        s = _session(shell_exec_timeout=0.0000001)
        _ok(s, "import std/config")
        _ok(s, 'std/config::timeout := Some("99s")')
        s.reset()
        _ok(s, "import std/config")
        _assert_setting(s, "timeout", ("Some", "value", TextValue("0.0000001s")))
        assert s._shell_exec_timeout == 0.0000001


class TestTimeoutSeedWinsOverDriverArgument:
    """An ``engine_base['timeout']`` seed wins over a disagreeing ``shell_exec_timeout``.

    ``shell_exec_timeout`` reaches the session twice: once as the typed
    ``engine_base`` seed mapping and once as the raw constructor argument,
    which is only a fallback for an unseeded key. These tests deliberately
    make the two disagree and pin that the session's own live field
    (``_shell_exec_timeout``) follows the seed, not the disagreeing argument.
    """

    def test_engine_base_seed_wins_over_the_shell_exec_timeout_argument(self) -> None:
        s = _session(
            stdlib_root=_STDLIB_ROOT,
            shell_exec_timeout=30.0,
            engine_base=build_engine_config_seeds({"timeout": "5s"}),
        )
        assert s._shell_exec_timeout == 5.0

    def test_engine_base_seed_of_none_wins_over_the_shell_exec_timeout_argument(self) -> None:
        # An explicit empty ``Option`` (a "no timeout" host control) must win
        # over a disagreeing non-``None`` scalar, same as every other case.
        s = _session(
            stdlib_root=_STDLIB_ROOT,
            shell_exec_timeout=30.0,
            engine_base=build_engine_config_seeds({"timeout": None}),
        )
        assert s._shell_exec_timeout is None


class TestSeedGovernsRuntimeEffectOverDriverArgument:
    """A disagreeing seed governs what an entry actually does, not just what it reads.

    ``strict-json`` reaches the session through both the typed ``engine_base``
    seed mapping and a raw constructor argument. The seed wins, and this test
    observes that where a user does: the JSON parsing mode an ``ask`` reply is
    held to. It first pins the disagreeing argument's own effect on a session
    that has no seed, so the seeded case demonstrably discriminates between the
    two channels.
    """

    def test_lenient_seed_keeps_a_fenced_reply_parseable_despite_a_strict_json_argument(
        self,
    ) -> None:
        # No seed: the argument alone makes the fenced reply unparseable.
        argument_only = _session(
            stdlib_root=_STDLIB_ROOT,
            agent_dispatcher=_FencedAgent(),
            default_strict_json=True,
        )
        strict_result = argument_only.eval_entry('let a: int = ask """how many"""')
        assert not strict_result.ok
        assert strict_result.error is not None
        assert "AgentParseError" in strict_result.error.type_name

        # The same argument, now contradicted by a lenient seed: the reply parses.
        s = _session(
            stdlib_root=_STDLIB_ROOT,
            agent_dispatcher=_FencedAgent(),
            default_strict_json=True,
            engine_base=build_engine_config_seeds({"strict-json": False}),
        )
        assert _ok(s, 'let a: int = ask """how many"""').value == IntValue(42)


_DECLARED_DEFAULT_PRECEDENCE = [
    pytest.param("strict-json", "false", BoolValue(True), id="strict-json"),
    pytest.param("timeout", 'Some("99s")', ("Some", "value", TextValue("2s")), id="timeout"),
    pytest.param("trace", "false", BoolValue(True), id="trace"),
    pytest.param(
        "trace-file",
        'Some("written.jsonl")',
        ("Some", "value", TextValue("declared.jsonl")),
        id="trace-file",
    ),
    pytest.param(
        "default-agent",
        'AgentCommand("written")',
        ("AgentCommand", "command", TextValue("declared")),
        id="default-agent",
    ),
]


class TestResetDeclaredDefaultPrecedence:
    """An unseeded key's ``:reset`` restores ``std/config``'s declared default.

    Each case writes a value that differs from BOTH the declared default and
    any host-side floor, so a wrong precedence (falling through to a host
    floor instead of the declaration) would be caught.
    """

    @pytest.mark.parametrize(("key", "written", "expected"), _DECLARED_DEFAULT_PRECEDENCE)
    def test_declared_default_survives_a_source_write(
        self, declared_defaults_stdlib: Path, key: str, written: str, expected: object
    ) -> None:
        s = _session(stdlib_root=declared_defaults_stdlib)
        _ok(s, "import std/config")
        _ok(s, f"std/config::{key} := {written}")
        s.reset()
        _ok(s, "import std/config")
        _assert_setting(s, key, expected)


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
        s = _session(
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
        s = _session(
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
        assert isinstance(value, RecordValue)
        assert value.fields["value"] == TextValue("5s")

    def test_timeout_seeded_as_none_disables_the_timeout_across_reset(self) -> None:
        # An empty ``Option`` is an explicit "no timeout" control, so it must
        # not fall back to the ``shell_exec_timeout`` driver argument.
        s = _session(
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
        result = _read_result(s, "timeout")
        assert isinstance(result.value, RecordValue)
        assert result.descriptors is not None
        assert _variant(result.value, result.descriptors) == "None"


class TestResetRestoresMixedSeedOrigins:
    """One ``:reset`` restores every key together, regardless of its origin.

    The per-key tests above (``TestResetHostSeedPrecedence``,
    ``TestResetDeclaredDefaultPrecedence``) each isolate a single key against
    its own dedicated session. This pins the state the seed-map/current-map
    consolidation (folding ``_engine_base`` and ``_declared_engine_defaults``
    into one ``_engine_seed``, and the three persisted-register fields into
    one ``_current``) put most at risk: a single session seeded with an
    explicit CLI/host value for one key (``timeout``) AND a std/config
    declared default LEARNED FROM SOURCE for another (``trace-file``), with both
    then overwritten by a source write, must have one ``:reset`` restore both
    together from the same seed map.
    """

    def test_reset_restores_a_cli_seeded_key_and_a_source_learned_declared_default(
        self, tmp_path: Path
    ) -> None:
        stdlib_root = tmp_path / "stdlib"
        config_path = stdlib_root / "src" / "config.agl"
        config_path.parent.mkdir(parents=True)
        config_path.write_text(
            "import std/prelude::{Option, Agent}\n"
            'builtin var default-agent: Agent = AgentCommand("declared")\n'
            'builtin var timeout: Option[text] = Option[text]::Some("2s")\n'
            'builtin var trace-file: Option[text] = Option[text]::Some("declared.jsonl")\n',
            encoding="utf-8",
        )
        _copy_core_and_option(config_path.parent)
        s = _session(
            stdlib_root=stdlib_root,
            engine_base=build_engine_config_seeds({"timeout": "5s"}),
        )

        _ok(s, "import std/config")
        # trace-file's declared default is now learned from source (recorded
        # into the seed); write over both keys before resetting.
        _ok(s, 'std/config::timeout := Some("9s")')
        _ok(s, 'std/config::trace-file := Some("written.jsonl")')
        s.reset()

        assert s._shell_exec_timeout == 5.0
        _ok(s, "import std/config")
        value = _read(s, "trace-file")
        assert isinstance(value, RecordValue)
        assert value.fields["value"] == TextValue("declared.jsonl")


# ---------------------------------------------------------------------------
# Partial-failure discipline
# ---------------------------------------------------------------------------


class TestPartialFailureDiscipline:
    """Setting writes completed before a runtime failure remain persistent."""

    def test_runtime_live_write_before_failure_persists(self) -> None:
        s = _session()
        _ok(s, "import std/config")
        result = s.eval_entry("std/config::strict-json := true\nlet z: decimal = 1 / 0")
        assert not result.ok
        assert _read(s, "strict-json") == BoolValue(True)


class TestLiveHostReconfiguration:
    def test_trace_file_write_repoints_later_repl_entries(self, tmp_path: Path) -> None:
        trace_path = tmp_path / "trace.jsonl"
        policy = HostSettingsPolicy(
            resolve_trace_path=lambda enabled, trace_file: (
                Path(trace_file) if enabled or trace_file is not None else None
            ),
        )
        s = _session(host_settings_policy=policy)
        _ok(s, "import std/config")
        _ok(s, f'std/config::trace-file := Some("{trace_path}")')
        _ok(s, 'print "later"')

        assert trace_path.exists()
        assert '"rendered": "later"' in trace_path.read_text(encoding="utf-8")

    def test_trace_false_settles_into_no_trace_for_later_entries(self, tmp_path: Path) -> None:
        """A deliberate ``trace := false`` keeps later entries untraced."""
        trace_path = tmp_path / "trace.jsonl"
        policy = HostSettingsPolicy(
            resolve_trace_path=lambda enabled, trace_file: (
                (Path(trace_file) if trace_file is not None else trace_path) if enabled else None
            ),
        )
        s = _session(host_settings_policy=policy, trace_path=trace_path)
        _ok(s, "import std/config")
        _ok(s, "std/config::trace := true")
        _ok(s, 'print "traced"')
        _ok(s, "std/config::trace := false")

        result = _ok(s, 'print "untraced"')
        assert result.trace_path is None
        text = trace_path.read_text(encoding="utf-8")
        assert '"rendered": "traced"' in text
        assert "untraced" not in text


def test_reset_uses_declared_live_engine_defaults(tmp_path: Path) -> None:
    """Reset reapplies std/config defaults without a removed runner setting."""
    stdlib_root = tmp_path / "stdlib"
    config_path = stdlib_root / "src" / "config.agl"
    config_path.parent.mkdir(parents=True)
    config_path.write_text(
        "import std/prelude::*\n"
        'builtin var default-agent: Agent = AgentCommand("declared")\n'
        "builtin var strict-json: bool = true\n"
        'builtin var timeout: Option[text] = Option[text]::Some("2s")\n'
        "builtin var trace: bool = false\n"
        "builtin var trace-file: Option[text] = Option[text]::None\n"
    )
    _copy_core_and_option(config_path.parent)
    session = ReplSession(stdlib_root=stdlib_root, default_stdlib=False)

    _ok(session, "import std/prelude::*\nimport std/config\nstd/config::strict-json")
    _ok(session, 'std/config::strict-json := false\nstd/config::timeout := Some("9s")')
    session.reset()

    assert session._default_strict_json is True
    assert session._shell_exec_timeout == 2.0


# ---------------------------------------------------------------------------
# Host-supplied default-agent seed (--default-agent / [exec] default-agent)
# ---------------------------------------------------------------------------


class TestDefaultAgentSeedThreading:
    """A decoded ``default-agent`` seed behaves like any other typed engine seed.

    ``--default-agent``/``[exec] default-agent`` decode into the same typed
    ``Value`` seed every other engine key uses (see
    ``cli_support.engine_seeds``), so these tests seed ``engine_base``
    directly.
    """

    def test_seed_is_observed_and_persists_across_entries(self) -> None:
        s = _unopened_session(
            engine_base={"default-agent": agent_value("AgentCommand", command="overridden")}
        )
        _ok(s, "import std/config")
        result = _read_result(s, "default-agent")
        assert isinstance(result.value, RecordValue)
        assert result.descriptors is not None
        assert _variant(result.value, result.descriptors) == "AgentCommand"
        assert result.value.fields["command"] == TextValue("overridden")

        _ok(s, "let unrelated = 1")
        value = _read(s, "default-agent")
        assert isinstance(value, RecordValue)
        assert value.fields["command"] == TextValue("overridden")

    def test_source_write_still_overrides_it_afterward(self) -> None:
        s = _unopened_session(
            engine_base={"default-agent": agent_value("AgentCommand", command="overridden")}
        )
        _ok(s, "import std/config")
        initial = _read_result(s, "default-agent")
        assert isinstance(initial.value, RecordValue)
        assert initial.descriptors is not None
        assert initial.descriptors.nominals[initial.value.nominal].display_name == (
            "Agent::AgentCommand"
        )
        _ok(s, 'std/config::default-agent := AgentClaude("haiku", "low")')
        result = _read_result(s, "default-agent")
        assert isinstance(result.value, RecordValue)
        assert result.descriptors is not None
        assert _variant(result.value, result.descriptors) == "AgentClaude"
        assert result.value.fields["model"] == TextValue("haiku")

    def test_malformed_command_text_is_rejected_at_the_first_entry(self) -> None:
        """A well-typed seed whose command text does not shell-split fails eagerly.

        A typed seed carries no source text to type-check -- it is validated
        the moment the first entry's interpreter is constructed, whether or
        not that entry ever imports ``std/config`` (mirrors
        ``tests/test_agl_pipeline_parse_entry.py``'s
        ``TestMalformedAgentCommandAtConstruction``).
        """
        s = ReplSession(
            stdlib_root=_STDLIB_ROOT,
            default_stdlib=False,
            engine_base={
                "default-agent": agent_value("AgentCommand", command="nonexistent-bin -p 'oops")
            },
        )

        result = s.eval_entry("1 + 1")
        assert not result.ok
        assert result.diagnostics
        assert result.error is None
