"""Tests for the idle-timeout watchdog in process helpers."""

from __future__ import annotations

import sys
import time

import pytest

from agm.config.general import parse_timeout
from agm.core.parse import parse_positive_timeout
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


class TestParsePositiveTimeout:
    def test_positive_duration_is_seconds(self) -> None:
        assert parse_positive_timeout("1.5m") == 90.0

    @pytest.mark.parametrize("value", ["0", "0h", "99999999999999999999", "9" * 400, "soon"])
    def test_zero_unrepresentable_or_malformed_raises(self, value: str) -> None:
        with pytest.raises(ValueError):
            parse_positive_timeout(value)


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

    @pytest.mark.parametrize("isolate_process_group", [True, False])
    def test_idle_timeout_kills_silent_process(self, isolate_process_group: bool) -> None:
        """A process that produces no output for the timeout period gets killed.

        Both the isolated-process-group kill path and the plain terminate path
        must honour the idle timeout.
        """
        script = "import time\ntime.sleep(60)\n"  # Would hang forever
        start = time.monotonic()
        with pytest.raises(SystemExit) as exc_info:
            run_capture(
                [sys.executable, "-c", script],
                isolate_process_group=isolate_process_group,
                idle_timeout=0.5,
            )
        elapsed = time.monotonic() - start
        assert exc_info.value.code == 124
        # Should be killed quickly, not after 60 seconds
        assert elapsed < 5

    def test_idle_timeout_disabled_by_default(self) -> None:
        """Without idle_timeout, a slow-but-producing process completes normally."""
        script = "import time\nprint('chunk1')\ntime.sleep(0.3)\nprint('chunk2')\n"
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
        script = "import time, sys\nprint('initial output')\nsys.stdout.flush()\ntime.sleep(60)\n"
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
        """When idle timeout fires without isolate_process_group, terminate_process is used."""
        script = "import time\ntime.sleep(60)\n"
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
