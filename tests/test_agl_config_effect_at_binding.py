"""Tests for AgL config/engine-setting effect-at-binding (D6, Task 3b).

When the interpreter reaches a ``config KEY = VALUE`` declaration for one of
the three D6 keys (``strict-json``, ``max-iters``, ``timeout``), the engine's
live setting updates from that point forward.  The precedence remains:
CLI > source value > config_base.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from unittest.mock import patch

import pytest

from agm.agl.ir.ids import NominalId
from agm.agl.modules.ids import STD_CORE_ID
from agm.agl.pipeline import PipelineDriver
from agm.agl.runtime.request import AgentRequest
from agm.agl.semantics.values import BoolValue, EnumValue, IntValue, TextValue, Value
from agm.core.process import ProcessCaptureResult


def _run(
    source: str,
    *,
    config_cli: dict[str, Value] | None = None,
    config_base: dict[str, Value] | None = None,
    initial_loop_limit: int = 5,
    initial_strict_json: bool = False,
    initial_timeout: float | None = None,
    default_agent: Callable[[AgentRequest], str] | None = None,
) -> object:
    """Run *source* through prepare + run_prepared."""
    rt = PipelineDriver(
        default_loop_limit=initial_loop_limit,
        default_strict_json=initial_strict_json,
        shell_exec_timeout=initial_timeout,
        default_agent=default_agent,
    )
    prepared = rt.prepare(source)
    return rt.run_prepared(prepared, config_cli=config_cli, config_base=config_base)


def _ok_shell_result(stdout: str = "") -> ProcessCaptureResult:
    """Return a successful ProcessCaptureResult for shell exec mocking."""
    return ProcessCaptureResult(
        returncode=0,
        stdout=stdout,
        stderr="",
        elapsed=0.01,
        timed_out=False,
        spawn_error=None,
        spawn_errno=None,
    )


class _ShellSpy:
    """Callable that records the ``idle_timeout`` passed on each shell call."""

    def __init__(self, stdout: str = "test") -> None:
        self.captured_timeouts: list[float | None] = []
        self._stdout = stdout

    def __call__(
        self,
        cmd: list[str],
        *,
        idle_timeout: float | None = None,
        isolate_process_group: bool = False,
        cwd: Path | None = None,
        env: dict[str, str] | None = None,
        stdin_text: str | None = None,
        interrupt_cleanup_cmd: list[str] | None = None,
        stdout_callback: Callable[[str], None] | None = None,
        stderr_callback: Callable[[str], None] | None = None,
    ) -> ProcessCaptureResult:
        self.captured_timeouts.append(idle_timeout)
        return _ok_shell_result(self._stdout)


# ---------------------------------------------------------------------------
# max-iters: D6 updates _loop_limit at the point of the config binding
# ---------------------------------------------------------------------------


class TestMaxIters:
    def test_loop_after_config_decl_uses_new_limit(self) -> None:
        """config max-iters = 3 before a do-loop raises the effective limit."""
        source = (
            "var i: int = 0\n"
            "config max-iters = 3\n"
            "do\n"
            "  i := i + 1\n"
            "until i >= 2\n"
        )
        # initial_loop_limit=1 is too small; the config binding must raise it to 3.
        result = _run(source, initial_loop_limit=1)
        assert result.ok, f"expected success but got: {result.error!r}"

    def test_initial_limit_applies_before_config_decl(self) -> None:
        """A do-loop before the config max-iters decl uses the initial loop limit."""
        source = (
            "var i: int = 0\n"
            "do\n"
            "  i := i + 1\n"
            "until i >= 2\n"
            "config max-iters = 3\n"
        )
        # initial_loop_limit=1 is not sufficient; config decl comes after, too late.
        result = _run(source, initial_loop_limit=1)
        assert not result.ok
        assert result.error is not None
        assert result.error.type_name == "MaxIterationsExceeded"

    def test_loop_before_config_decl_is_unaffected(self) -> None:
        """A do-loop executing before the config decl uses the initial limit."""
        source = (
            "var i: int = 0\n"
            "do\n"
            "  i := i + 1\n"
            "until i >= 2\n"
            "config max-iters = 3\n"
        )
        # initial_loop_limit=5 is sufficient; the post-loop config decl doesn't matter.
        result = _run(source, initial_loop_limit=5)
        assert result.ok

    def test_cli_beats_source_config_for_loop_limit(self) -> None:
        """CLI config_cli max-iters overrides the source config max-iters = 1."""
        source = (
            "var i: int = 0\n"
            "config max-iters = 1\n"
            "do\n"
            "  i := i + 1\n"
            "until i >= 2\n"
        )
        # CLI says 5; source says 1. CLI must win (applied via D6 at bind time).
        result = _run(source, config_cli={"max-iters": IntValue(5)}, initial_loop_limit=5)
        assert result.ok, f"CLI override should win but got: {result.error!r}"


# ---------------------------------------------------------------------------
# strict-json: D6 updates _strict_json at the point of the config binding
# ---------------------------------------------------------------------------


class TestStrictJson:
    FENCED_RESPONSE = "```json\n5\n```"

    def _stub_agent(self, _request: AgentRequest) -> str:
        return self.FENCED_RESPONSE

    def test_ask_after_config_strict_json_rejects_fenced(self) -> None:
        """config strict-json = true before an ask makes fenced JSON fail."""
        source = (
            "agent spy\n"
            "config strict-json = true\n"
            'let r: int = ask("prompt", agent = spy)\n'
            "print r\n"
        )
        result = _run(source, initial_strict_json=False, default_agent=self._stub_agent)
        assert not result.ok
        assert result.error is not None
        assert result.error.type_name == "AgentParseError"

    def test_ask_before_config_strict_json_uses_lenient(self) -> None:
        """An ask before config strict-json = true uses the initial lenient mode."""
        source = (
            "agent spy\n"
            'let r: int = ask("prompt", agent = spy)\n'
            "config strict-json = true\n"
            "print r\n"
        )
        result = _run(source, initial_strict_json=False, default_agent=self._stub_agent)
        assert result.ok, f"expected success (lenient before decl) but got: {result.error!r}"

    def test_cli_beats_source_config_for_strict_json(self) -> None:
        """CLI config_cli strict-json=false overrides source config strict-json = true."""
        source = (
            "agent spy\n"
            "config strict-json = true\n"
            'let r: int = ask("prompt", agent = spy)\n'
            "print r\n"
        )
        # CLI says False → overrides source True via D6 at bind time → lenient → accepted.
        result = _run(
            source,
            config_cli={"strict-json": BoolValue(False)},
            initial_strict_json=False,
            default_agent=self._stub_agent,
        )
        assert result.ok, f"CLI override should keep lenient but got: {result.error!r}"


# ---------------------------------------------------------------------------
# timeout: D6 updates _shell_exec_timeout at the point of the config binding
# ---------------------------------------------------------------------------


class TestTimeout:
    def test_exec_after_config_timeout_uses_new_timeout(self) -> None:
        """config timeout = "5s" before exec updates the shell_exec_timeout."""
        spy = _ShellSpy(stdout="test")
        source = (
            'let a: text = exec "echo test"\n'
            'config timeout = "5s"\n'
            'let b: text = exec "echo test"\n'
            "print b\n"
        )
        with patch("agm.core.process.run_capture_result", side_effect=spy):
            result = _run(source, initial_timeout=None)

        assert result.ok, f"expected success but got: {result.error!r}"
        assert len(spy.captured_timeouts) == 2
        assert spy.captured_timeouts[0] is None  # before config decl
        assert spy.captured_timeouts[1] == pytest.approx(5.0)  # after config decl

    def test_exec_before_config_timeout_uses_initial_timeout(self) -> None:
        """An exec before config timeout uses the initial shell_exec_timeout."""
        spy = _ShellSpy(stdout="test")
        source = (
            'let a: text = exec "echo test"\n'
            'config timeout = "5s"\n'
            "print a\n"
        )
        with patch("agm.core.process.run_capture_result", side_effect=spy):
            result = _run(source, initial_timeout=10.0)

        assert result.ok
        # Only one exec call happened, before the config decl.
        assert spy.captured_timeouts == [pytest.approx(10.0)]

    def test_cli_beats_source_config_for_timeout(self) -> None:
        """CLI config_cli timeout=None overrides source config timeout = "5s"."""
        from agm.agl.runtime.params import convert_config_value
        from agm.agl.semantics.types import OPTION_TEXT_TYPE

        spy = _ShellSpy(stdout="test")
        source = (
            'config timeout = "5s"\n'
            'let a: text = exec "echo test"\n'
            "print a\n"
        )
        cli_none = convert_config_value("timeout", None, OPTION_TEXT_TYPE)
        with patch("agm.core.process.run_capture_result", side_effect=spy):
            result = _run(
                source,
                config_cli={"timeout": cli_none},
                initial_timeout=None,
            )

        assert result.ok, f"expected success but got: {result.error!r}"
        # CLI None overrides source "5s" → shell_exec_timeout stays None.
        assert spy.captured_timeouts == [None]

    def test_timeout_minutes_suffix_parsed_correctly(self) -> None:
        """config timeout = "2m" sets shell_exec_timeout to 120.0 seconds."""
        spy = _ShellSpy(stdout="test")
        source = (
            'config timeout = "2m"\n'
            'let a: text = exec "echo test"\n'
            "print a\n"
        )
        with patch("agm.core.process.run_capture_result", side_effect=spy):
            result = _run(source, initial_timeout=None)

        assert result.ok, f"expected success but got: {result.error!r}"
        assert spy.captured_timeouts == [pytest.approx(120.0)]

    def test_invalid_timeout_string_raises_agl_error(self) -> None:
        """config timeout = "forever" raises a clean AgL-level ValueError at the decl point.

        The value "forever" is a valid Option[text] (it passes typecheck as
        some("forever")), but parse_timeout raises ValueError at runtime; D6
        converts this to an AglRaise so the program exits with an uncaught
        exception rather than a raw Python traceback.
        """
        source = 'config timeout = "forever"\nprint 1\n'
        result = _run(source, initial_timeout=None)
        assert not result.ok
        assert result.error is not None
        assert result.error.type_name == "ValueError"
        assert result.error.line is not None

    def test_bare_config_timeout_sets_none_from_base(self) -> None:
        """Bare config timeout bound to none from config_base sets shell_exec_timeout to None."""
        from agm.agl.runtime.params import convert_config_value
        from agm.agl.semantics.types import OPTION_TEXT_TYPE

        none_val = convert_config_value("timeout", None, OPTION_TEXT_TYPE)
        spy = _ShellSpy(stdout="test")
        source = (
            "config timeout\n"
            'let a: text = exec "echo test"\n'
            "print a\n"
        )
        with patch("agm.core.process.run_capture_result", side_effect=spy):
            result = _run(
                source,
                config_base={"timeout": none_val},
                initial_timeout=10.0,
            )

        assert result.ok, f"expected success but got: {result.error!r}"
        # After config timeout (none) → _shell_exec_timeout = None.
        assert spy.captured_timeouts == [None]


# ---------------------------------------------------------------------------
# Defensive branch coverage: wrong-typed config_cli values are silently ignored
# ---------------------------------------------------------------------------


def _opt_enum(variant: str, fields: dict[str, Value] | None = None) -> EnumValue:
    """Build an Option EnumValue for use in config_cli."""
    return EnumValue(
        nominal=NominalId(STD_CORE_ID, "Option"),
        display_name="Option",
        variant=variant,
        fields=fields or {},
    )


class TestD6DefensiveBranches:
    """_apply_config_effect silently ignores config_cli values with wrong types.

    These tests exercise defensive isinstance checks in _apply_config_effect
    that guard against host-supplied config_cli entries with incorrect Value
    types.  The type-checker ensures this cannot happen for well-typed programs,
    but a host could supply arbitrary Values.
    """

    def test_wrong_type_strict_json_is_ignored(self) -> None:
        """A non-BoolValue for strict-json in config_cli leaves _strict_json unchanged."""
        source = "config strict-json = true\nprint 1\n"
        result = _run(source, config_cli={"strict-json": IntValue(1)}, initial_strict_json=False)
        assert result.ok  # isinstance(IntValue, BoolValue) → False → no effect

    def test_wrong_type_max_iters_is_ignored(self) -> None:
        """A non-IntValue for max-iters in config_cli leaves _loop_limit unchanged."""
        source = "config max-iters = 5\nprint 1\n"
        result = _run(source, config_cli={"max-iters": BoolValue(True)}, initial_loop_limit=5)
        assert result.ok  # isinstance(BoolValue, IntValue) → False → no effect

    def test_wrong_type_timeout_is_ignored(self) -> None:
        """A non-EnumValue for timeout in config_cli leaves _shell_exec_timeout unchanged."""
        source = "config timeout = \"5s\"\nprint 1\n"
        result = _run(source, config_cli={"timeout": TextValue("5s")}, initial_timeout=None)
        assert result.ok  # isinstance(TextValue, EnumValue) → False → no effect

    def test_timeout_some_with_non_text_inner_is_ignored(self) -> None:
        """A Some(IntValue) timeout in config_cli leaves _shell_exec_timeout unchanged."""
        wrong_some = _opt_enum("Some", {"value": IntValue(42)})
        source = "config timeout = \"5s\"\nprint 1\n"
        result = _run(source, config_cli={"timeout": wrong_some}, initial_timeout=None)
        assert result.ok  # isinstance(IntValue, TextValue) → False → no effect

    def test_timeout_unknown_variant_is_ignored(self) -> None:
        """An Option variant that is neither None nor Some leaves _shell_exec_timeout unchanged."""
        unknown_variant = _opt_enum("Maybe")
        source = "config timeout = \"5s\"\nprint 1\n"
        result = _run(source, config_cli={"timeout": unknown_variant}, initial_timeout=None)
        assert result.ok  # variant != "None" and != "Some" → no effect
