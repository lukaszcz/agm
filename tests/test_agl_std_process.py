"""Process standard-library operations and controlled process termination."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from agm.agl import PipelineDriver
from agm.agl.modules.roots import RootSet
from tests._agl_helpers import agl_roots

_STDLIB = Path(__file__).resolve().parents[1] / "stdlib"


def _roots() -> RootSet:
    return agl_roots()


def _exit_program(call: str) -> str:
    return f"import std/process\nprogram def main() -> unit = process::exit({call})\n"


@pytest.mark.parametrize((("call", "expected_code")), [("", 0), ("0", 0), ("255", 255)])
def test_process_exit_preserves_the_requested_system_exit_code(
    tmp_path: Path, call: str, expected_code: int
) -> None:
    trace_path = tmp_path / "trace.jsonl"

    with pytest.raises(SystemExit) as raised:
        PipelineDriver().run(_exit_program(call), roots=_roots(), log_file=trace_path)

    assert raised.value.code == expected_code
    records = [json.loads(line) for line in trace_path.read_text(encoding="utf-8").splitlines()]
    assert records[0]["kind"] == "run_start"
    assert records[-1]["kind"] == "run_end"
    assert records[-1]["ok"] is (expected_code == 0)


@pytest.mark.parametrize("code", [-1, 256])
def test_process_exit_rejects_codes_outside_the_portable_range(code: int) -> None:
    result = PipelineDriver().run(_exit_program(str(code)), roots=_roots())

    assert not result.ok
    assert result.error is not None
    assert result.error.type_name == "ExternError"


def test_process_metadata_uses_the_controlled_run_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.chdir(tmp_path)
    source = (
        "import std/process\n"
        "program def main() -> unit =\n"
        "  print(process::cwd())\n"
        "  print(process::pid())\n"
        "  print(process::hostname())\n"
    )

    result = PipelineDriver().run(source, roots=_roots())

    assert result.ok
    cwd, pid, hostname = capsys.readouterr().out.splitlines()
    assert cwd == str(tmp_path)
    assert int(pid) > 0
    assert hostname


@pytest.mark.parametrize((("call", "expected_code")), [("", 0), ("0", 0), ("255", 255)])
def test_process_exit_reaches_the_cli_process_and_finalizes_its_trace(
    tmp_path: Path, env: dict[str, str], call: str, expected_code: int
) -> None:
    program = tmp_path / "exit.agl"
    trace_path = tmp_path / "trace.jsonl"
    program.write_text(_exit_program(call), encoding="utf-8")

    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            "from agm.cli import main; main()",
            "exec",
            "--log-file",
            str(trace_path),
            str(program),
        ],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == expected_code
    records = [json.loads(line) for line in trace_path.read_text(encoding="utf-8").splitlines()]
    assert records[0]["kind"] == "run_start"
    assert records[-1]["kind"] == "run_end"
    assert records[-1]["ok"] is (expected_code == 0)


def test_process_metadata_matches_the_subprocess_identity(
    tmp_path: Path, env: dict[str, str]
) -> None:
    program = tmp_path / "metadata.agl"
    program.write_text(
        "import std/process\n"
        "program def main() -> unit =\n"
        "  print(process::cwd())\n"
        "  print(process::pid())\n"
        "  print(process::hostname())\n",
        encoding="utf-8",
    )

    process = subprocess.Popen(
        [
            sys.executable,
            "-c",
            "from agm.cli import main; main()",
            "exec",
            "--no-log",
            str(program),
        ],
        cwd=tmp_path,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    stdout, stderr = process.communicate()

    assert process.returncode == 0, stderr
    cwd, pid, hostname = stdout.splitlines()
    assert cwd == str(tmp_path)
    assert int(pid) == process.pid
    assert hostname
