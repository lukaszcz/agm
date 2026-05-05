"""Tests for the idle-timeout watchdog in agm loop and process helpers."""

from __future__ import annotations

import sys
import time
from pathlib import Path

import pytest

from agm.config.general import parse_timeout
from agm.core.process import run_capture, run_subprocess

# ---------------------------------------------------------------------------
# parse_timeout
# ---------------------------------------------------------------------------


class TestParseTimeout:
    def test_plain_integer_is_seconds(self) -> None:
        assert parse_timeout("30") == 30.0

    def test_seconds_suffix(self) -> None:
        assert parse_timeout("30s") == 30.0

    def test_minutes_suffix(self) -> None:
        assert parse_timeout("10m") == 600.0

    def test_hours_suffix(self) -> None:
        assert parse_timeout("2h") == 7200.0

    def test_fractional_value(self) -> None:
        assert parse_timeout("1.5m") == 90.0

    def test_whitespace_is_stripped(self) -> None:
        assert parse_timeout("  5m  ") == 300.0

    def test_invalid_format_raises(self) -> None:
        with pytest.raises(ValueError, match="invalid timeout format"):
            parse_timeout("abc")

    def test_empty_raises(self) -> None:
        with pytest.raises(ValueError, match="invalid timeout format"):
            parse_timeout("")

    def test_zero_value(self) -> None:
        assert parse_timeout("0") == 0.0

    def test_zero_with_suffix(self) -> None:
        assert parse_timeout("0h") == 0.0


# ---------------------------------------------------------------------------
# run_subprocess idle timeout
# ---------------------------------------------------------------------------


class TestIdleTimeout:
    def test_fast_command_completes_within_timeout(self) -> None:
        returncode, stdout, stderr = run_capture(
            [sys.executable, "-c", "print('hello')"],
            idle_timeout=30,
        )
        assert returncode == 0
        assert stdout.strip() == "hello"

    def test_idle_timeout_kills_silent_process(self) -> None:
        """A process that produces no output for the timeout period gets killed."""
        script = (
            "import time\n"
            "time.sleep(60)\n"  # Would hang forever
        )
        start = time.monotonic()
        with pytest.raises(SystemExit) as exc_info:
            run_capture(
                [sys.executable, "-c", script],
                isolate_process_group=True,
                idle_timeout=0.5,
            )
        elapsed = time.monotonic() - start
        assert exc_info.value.code == 124
        # Should be killed quickly, not after 60 seconds
        assert elapsed < 5

    def test_idle_timeout_disabled_by_default(self) -> None:
        """Without idle_timeout, a slow-but-producing process completes normally."""
        script = (
            "import time\n"
            "print('chunk1')\n"
            "time.sleep(0.3)\n"
            "print('chunk2')\n"
        )
        returncode, stdout, stderr = run_capture(
            [sys.executable, "-c", script],
        )
        assert returncode == 0
        assert "chunk1" in stdout
        assert "chunk2" in stdout

    def test_process_that_keeps_producing_output_survives_timeout(self) -> None:
        """A process producing output within the timeout should not be killed."""
        script = (
            "import time, sys\n"
            "for i in range(3):\n"
            "    print(f'chunk {i}')\n"
            "    sys.stdout.flush()\n"
            "    time.sleep(0.2)\n"
        )
        returncode, stdout, stderr = run_capture(
            [sys.executable, "-c", script],
            isolate_process_group=True,
            idle_timeout=1.0,
        )
        assert returncode == 0
        assert "chunk 0" in stdout
        assert "chunk 2" in stdout

    def test_idle_timeout_kills_after_output_stops(self) -> None:
        """Process that outputs then goes silent should be killed after timeout."""
        script = (
            "import time, sys\n"
            "print('initial output')\n"
            "sys.stdout.flush()\n"
            "time.sleep(60)\n"
        )
        start = time.monotonic()
        with pytest.raises(SystemExit) as exc_info:
            run_capture(
                [sys.executable, "-c", script],
                isolate_process_group=True,
                idle_timeout=0.5,
            )
        elapsed = time.monotonic() - start
        assert exc_info.value.code == 124
        assert elapsed < 5

    def test_idle_timeout_with_isolate_process_group(self) -> None:
        """Idle timeout works with isolate_process_group=True."""
        script = (
            "import time\n"
            "time.sleep(60)\n"
        )
        start = time.monotonic()
        with pytest.raises(SystemExit) as exc_info:
            run_capture(
                [sys.executable, "-c", script],
                isolate_process_group=True,
                idle_timeout=0.5,
            )
        elapsed = time.monotonic() - start
        assert exc_info.value.code == 124
        assert elapsed < 5

    def test_idle_timeout_without_isolate_terminates_process(self) -> None:
        """When idle timeout fires without isolate_process_group, _terminate_process is used."""
        script = (
            "import time\n"
            "time.sleep(60)\n"
        )
        start = time.monotonic()
        with pytest.raises(SystemExit) as exc_info:
            run_subprocess(
                [sys.executable, "-c", script],
                capture_output=True,
                isolate_process_group=False,
                idle_timeout=0.5,
            )
        elapsed = time.monotonic() - start
        assert exc_info.value.code == 124
        assert elapsed < 5


# ---------------------------------------------------------------------------
# loop CLI arg parsing --timeout
# ---------------------------------------------------------------------------


class TestLoopTimeoutArg:
    def test_parse_loop_args_timeout_seconds(self) -> None:
        from agm.cli import _parse_loop_args

        args = _parse_loop_args(
            ["--timeout", "30", "--no-selector", "mycmd"],
            command_path=["loop"],
        )
        assert args.timeout == 30.0

    def test_parse_loop_args_timeout_minutes(self) -> None:
        from agm.cli import _parse_loop_args

        args = _parse_loop_args(
            ["--timeout", "30m", "--no-selector", "mycmd"],
            command_path=["loop"],
        )
        assert args.timeout == 1800.0

    def test_parse_loop_args_timeout_hours(self) -> None:
        from agm.cli import _parse_loop_args

        args = _parse_loop_args(
            ["--timeout", "2h", "--no-selector", "mycmd"],
            command_path=["loop"],
        )
        assert args.timeout == 7200.0

    def test_parse_loop_args_timeout_invalid_reports_error(self) -> None:
        from agm.cli import _parse_loop_args

        with pytest.raises(SystemExit):
            _parse_loop_args(
                ["--timeout", "abc", "--no-selector", "mycmd"],
                command_path=["loop"],
            )

    def test_parse_loop_next_args_timeout(self) -> None:
        from agm.cli import _parse_loop_next_args

        args = _parse_loop_next_args(
            ["--timeout", "10m", "mycmd"],
            command_path=["loop", "next"],
        )
        assert args.timeout == 600.0

    def test_timeout_defaults_to_none(self) -> None:
        from agm.cli import _parse_loop_args

        args = _parse_loop_args(
            ["--no-selector", "mycmd"],
            command_path=["loop"],
        )
        assert args.timeout is None


# ---------------------------------------------------------------------------
# config.toml [loop] timeout
# ---------------------------------------------------------------------------


class TestLoopTimeoutConfig:
    def test_load_loop_config_reads_numeric_timeout(self, tmp_path: Path) -> None:
        from agm.config.general import load_loop_config

        home = tmp_path / "home"
        (home / ".agm").mkdir(parents=True)
        (home / ".agm" / "config.toml").write_text('[loop]\ntimeout = 1800\n')

        config = load_loop_config(home=home, proj_dir=None, cwd=tmp_path / "work")
        assert config.timeout == 1800.0

    def test_load_loop_config_reads_string_timeout(self, tmp_path: Path) -> None:
        from agm.config.general import load_loop_config

        home = tmp_path / "home"
        (home / ".agm").mkdir(parents=True)
        (home / ".agm" / "config.toml").write_text('[loop]\ntimeout = "30m"\n')

        config = load_loop_config(home=home, proj_dir=None, cwd=tmp_path / "work")
        assert config.timeout == 1800.0

    def test_load_loop_config_timeout_defaults_to_none(self, tmp_path: Path) -> None:
        from agm.config.general import load_loop_config

        home = tmp_path / "home"
        (home / ".agm").mkdir(parents=True)
        (home / ".agm" / "config.toml").write_text('[loop]\nrunner = "claude -p"\n')

        config = load_loop_config(home=home, proj_dir=None, cwd=tmp_path / "work")
        assert config.timeout is None

    def test_load_loop_config_zero_timeout_is_none(self, tmp_path: Path) -> None:
        from agm.config.general import load_loop_config

        home = tmp_path / "home"
        (home / ".agm").mkdir(parents=True)
        (home / ".agm" / "config.toml").write_text('[loop]\ntimeout = 0\n')

        config = load_loop_config(home=home, proj_dir=None, cwd=tmp_path / "work")
        assert config.timeout is None

    def test_command_specific_timeout_overrides_default(self, tmp_path: Path) -> None:
        from agm.config.general import load_loop_config

        home = tmp_path / "home"
        (home / ".agm").mkdir(parents=True)
        (home / ".agm" / "config.toml").write_text(
            '[loop]\ntimeout = "30m"\n[loop.codex]\ntimeout = "1h"\n'
        )

        config = load_loop_config(
            home=home, proj_dir=None, cwd=tmp_path / "work", command_name="codex",
        )
        assert config.timeout == 3600.0
