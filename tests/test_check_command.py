"""Tests for the `agm check` CLI command.

Covers:
- CLI wires FILE arguments (one or more), -I/--module-path, and --no-stdlib
  into CheckArgs; missing FILE is a usage error.
- A clean program file and a clean library module (no `program def`, the case
  `agm exec --dry-run` rejects) check successfully with no output.
- Syntax errors, type errors, and errors inside an imported module are
  reported as GNU-style diagnostics on stderr with exit code 1.
- Several files are each checked independently, in order, even when an
  earlier one failed; every failing file's diagnostics are reported.
- -I/--module-path roots and --no-stdlib behave exactly as they do for
  `agm exec`.
- A missing/unreadable file is reported and does not stop the remaining
  files from being checked.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from click.testing import CliRunner, Result
from typer.main import get_command

import agm.cli as cli
import agm.commands.check as check_command
from agm.cli_support.args import CheckArgs

# Building the Typer app's Click command tree costs more than the assertions in
# the parser-contract tests below; it is derived from module-level definitions
# only, so it is built once here instead of on every invocation.
_AGM_COMMAND = get_command(cli.app)


@pytest.fixture()
def runner() -> CliRunner:
    return CliRunner()


def invoke(runner: CliRunner, argv: list[str]) -> Result:
    return runner.invoke(_AGM_COMMAND, argv, prog_name="agm", catch_exceptions=False)


@pytest.fixture()
def recorded_runs(monkeypatch: pytest.MonkeyPatch) -> list[object]:
    """Patch ``check.run`` to record its CheckArgs instead of checking."""
    calls: list[object] = []

    def fake_run(args: object) -> None:
        calls.append(args)

    monkeypatch.setattr(check_command, "run", fake_run)
    return calls


class TestCheckArgsParsing:
    """Parser-contract tests: verify CLI flags map to CheckArgs fields."""

    def test_check_single_file_argument(
        self, runner: CliRunner, tmp_path: Path, recorded_runs: list[object]
    ) -> None:
        agl_file = tmp_path / "test.agl"
        agl_file.write_text("def value() -> int = 1\n")

        result = invoke(runner, ["check", str(agl_file)])
        assert result.exit_code == 0

        assert len(recorded_runs) == 1
        args = recorded_runs[0]
        assert isinstance(args, CheckArgs)
        assert args.files == [str(agl_file)]

    def test_check_multiple_file_arguments(
        self, runner: CliRunner, tmp_path: Path, recorded_runs: list[object]
    ) -> None:
        first = tmp_path / "a.agl"
        second = tmp_path / "b.agl"
        first.write_text("def value() -> int = 1\n")
        second.write_text("def other() -> int = 2\n")

        result = invoke(runner, ["check", str(first), str(second)])
        assert result.exit_code == 0

        args = recorded_runs[0]
        assert isinstance(args, CheckArgs)
        assert args.files == [str(first), str(second)]

    def test_check_module_path_flag_repeatable(
        self, runner: CliRunner, tmp_path: Path, recorded_runs: list[object]
    ) -> None:
        agl_file = tmp_path / "test.agl"
        agl_file.write_text("def value() -> int = 1\n")
        dir_a = tmp_path / "a"
        dir_b = tmp_path / "b"
        dir_a.mkdir()
        dir_b.mkdir()

        result = invoke(
            runner,
            ["check", "-I", str(dir_a), "-I", str(dir_b), str(agl_file)],
        )
        assert result.exit_code == 0

        args = recorded_runs[0]
        assert isinstance(args, CheckArgs)
        assert args.module_paths == [str(dir_a), str(dir_b)]

    def test_check_no_stdlib_flag(
        self, runner: CliRunner, tmp_path: Path, recorded_runs: list[object]
    ) -> None:
        agl_file = tmp_path / "test.agl"
        agl_file.write_text("def value() -> int = 1\n")

        result = invoke(runner, ["check", "--no-stdlib", str(agl_file)])
        assert result.exit_code == 0

        args = recorded_runs[0]
        assert isinstance(args, CheckArgs)
        assert args.no_stdlib is True

    def test_check_no_stdlib_defaults_false(
        self, runner: CliRunner, tmp_path: Path, recorded_runs: list[object]
    ) -> None:
        agl_file = tmp_path / "test.agl"
        agl_file.write_text("def value() -> int = 1\n")

        result = invoke(runner, ["check", str(agl_file)])
        assert result.exit_code == 0

        args = recorded_runs[0]
        assert isinstance(args, CheckArgs)
        assert args.no_stdlib is False

    def test_check_missing_file_argument_errors(self, runner: CliRunner) -> None:
        result = invoke(runner, ["check"])

        assert result.exit_code != 0
        assert result.stderr


class TestCheckCommand:
    """Behavior tests: ``check_command.run`` against the real AgL pipeline."""

    def test_clean_program_file_succeeds_silently(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Uses a program with a call site so a regressed call-site printer would fail this."""
        agl_file = tmp_path / "hello.agl"
        agl_file.write_text(
            'let reviewer = AgentCommand("review-runner")\n'
            "program def main() -> unit =\n"
            '  let r = ask("hello", agent = reviewer)\n'
            "  print r\n"
        )

        check_command.run(CheckArgs(files=[str(agl_file)]))

        captured = capsys.readouterr()
        # ``check`` never prints ``exec --dry-run``'s ``call-sites:`` inventory,
        # even though this program has one (the ``ask`` call above).
        assert captured.out == ""
        assert captured.err == ""

    def test_clean_library_module_without_program_def_succeeds(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """A library module with no `program def` is the case `exec --dry-run` rejects."""
        agl_file = tmp_path / "lib.agl"
        agl_file.write_text("def double(n: int) -> int = n * 2\n")

        check_command.run(CheckArgs(files=[str(agl_file)]))

        captured = capsys.readouterr()
        assert captured.out == ""
        assert captured.err == ""

    def test_undefaulted_required_param_does_not_report_missing_param(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """``check`` never param-checks: a required ``param`` with no default is silent.

        Regression test: ``run_prepared(check_only=True)`` used to call
        ``_prepare_ir_params`` before the check-only stop, so this file would
        falsely report ``Missing required param: 'name'`` on every save.
        """
        agl_file = tmp_path / "greet.agl"
        agl_file.write_text("param name: text\nprogram def main() -> unit =\n  print name\n")

        check_command.run(CheckArgs(files=[str(agl_file)]))

        captured = capsys.readouterr()
        assert captured.out == ""
        assert captured.err == ""

    def test_bad_resource_path_reports_diagnostic_and_exits_1(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """A `resource` referencing a nonexistent path is caught during lowering.

        Regression test for the fix above: ``check`` must reach lowering (not
        stop at match compilation) to still catch this, even though it no
        longer param-checks.
        """
        agl_file = tmp_path / "res.agl"
        agl_file.write_text(
            'let prompt = resource("missing.md")\nprogram def main() -> unit =\n  print prompt\n'
        )

        with pytest.raises(SystemExit) as exc_info:
            check_command.run(CheckArgs(files=[str(agl_file)]))
        assert exc_info.value.code == 1

        captured = capsys.readouterr()
        assert "res.agl" in captured.err
        assert "error:" in captured.err

    def test_syntax_error_reports_diagnostic_and_exits_1(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        agl_file = tmp_path / "bad.agl"
        agl_file.write_text("def broken( -> int = 1\n")

        with pytest.raises(SystemExit) as exc_info:
            check_command.run(CheckArgs(files=[str(agl_file)]))
        assert exc_info.value.code == 1

        captured = capsys.readouterr()
        assert "bad.agl" in captured.err
        assert "error:" in captured.err

    def test_type_error_reports_diagnostic_and_exits_1(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        agl_file = tmp_path / "types.agl"
        agl_file.write_text(
            'def add(a: int, b: int) -> int = a + b\ndef bad() -> int = add(1, "two")\n'
        )

        with pytest.raises(SystemExit) as exc_info:
            check_command.run(CheckArgs(files=[str(agl_file)]))
        assert exc_info.value.code == 1

        captured = capsys.readouterr()
        assert "types.agl" in captured.err
        assert "error:" in captured.err

    def test_error_inside_imported_module_reports_that_modules_path(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        (tmp_path / "broken.agl").write_text("def f() -> int = undeclared_name\n")
        entry = tmp_path / "entry.agl"
        entry.write_text("import broken::*\ndef g() -> int = f()\n")

        with pytest.raises(SystemExit) as exc_info:
            check_command.run(CheckArgs(files=[str(entry)]))
        assert exc_info.value.code == 1

        captured = capsys.readouterr()
        assert "broken.agl" in captured.err

    def test_mixed_results_across_several_files_all_reported_exit_1(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        clean = tmp_path / "clean.agl"
        clean.write_text("def value() -> int = 1\n")
        bad_syntax = tmp_path / "bad_syntax.agl"
        bad_syntax.write_text("def broken( -> int = 1\n")
        bad_scope = tmp_path / "bad_scope.agl"
        bad_scope.write_text("def bad() -> int = undeclared_name\n")

        with pytest.raises(SystemExit) as exc_info:
            check_command.run(CheckArgs(files=[str(clean), str(bad_syntax), str(bad_scope)]))
        assert exc_info.value.code == 1

        captured = capsys.readouterr()
        # Every failing file is checked and reported, not just the first.
        assert "bad_syntax.agl" in captured.err
        assert "bad_scope.agl" in captured.err
        assert "clean.agl" not in captured.err

    def test_module_path_flag_resolves_a_library_in_another_directory(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        lib_dir = tmp_path / "lib"
        lib_dir.mkdir()
        (lib_dir / "helper.agl").write_text("def answer() -> int = 42\n")
        work = tmp_path / "work"
        work.mkdir()
        entry = work / "entry.agl"
        entry.write_text("import helper::*\ndef g() -> int = answer()\n")

        # Without -I, the library is unreachable: a static module-not-found error.
        with pytest.raises(SystemExit) as exc_info:
            check_command.run(CheckArgs(files=[str(entry)]))
        assert exc_info.value.code == 1
        captured = capsys.readouterr()
        assert captured.err

        # With -I pointing at the library directory, the entry checks clean.
        check_command.run(CheckArgs(files=[str(entry)], module_paths=[str(lib_dir)]))
        captured = capsys.readouterr()
        assert captured.out == ""
        assert captured.err == ""

    def test_no_stdlib_disables_automatic_std_core_opening(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        agl_file = tmp_path / "opt.agl"
        agl_file.write_text("let x = Some(value = 1)\n")

        # std/prelude is opened automatically by default, so this checks clean.
        check_command.run(CheckArgs(files=[str(agl_file)]))
        captured = capsys.readouterr()
        assert captured.out == ""
        assert captured.err == ""

        # --no-stdlib disables that automatic opening, so `Some` is undefined.
        with pytest.raises(SystemExit) as exc_info:
            check_command.run(CheckArgs(files=[str(agl_file)], no_stdlib=True))
        assert exc_info.value.code == 1
        captured = capsys.readouterr()
        assert "opt.agl" in captured.err

    def test_missing_file_reports_error_and_still_checks_remaining_files(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        missing = tmp_path / "missing.agl"
        clean = tmp_path / "clean.agl"
        clean.write_text("def value() -> int = 1\n")

        with pytest.raises(SystemExit) as exc_info:
            check_command.run(CheckArgs(files=[str(missing), str(clean)]))
        assert exc_info.value.code == 1

        captured = capsys.readouterr()
        assert captured.err
        assert "missing.agl" in captured.err
        # The second, clean file was still checked and produced no diagnostics.
        assert "clean.agl" not in captured.err

    def test_unreadable_file_exits_1(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        unreadable = tmp_path / "secret.agl"
        unreadable.write_text("def value() -> int = 1\n")
        unreadable.chmod(0o000)
        try:
            with pytest.raises(SystemExit) as exc_info:
                check_command.run(CheckArgs(files=[str(unreadable)]))
            assert exc_info.value.code == 1
            captured = capsys.readouterr()
            assert captured.err
            assert "secret.agl" in captured.err
        finally:
            unreadable.chmod(0o644)

    def test_program_def_is_validated_but_never_run(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """check never executes a program: an evaluation-time effect never runs."""
        agl_file = tmp_path / "would_print.agl"
        agl_file.write_text('program def main() -> unit =\n  print "should not print"\n')

        check_command.run(CheckArgs(files=[str(agl_file)]))

        captured = capsys.readouterr()
        assert captured.out == ""

    def test_dry_run_flag_is_accepted_and_has_no_effect(
        self, runner: CliRunner, tmp_path: Path
    ) -> None:
        agl_file = tmp_path / "hello.agl"
        agl_file.write_text('program def main() -> unit =\n  print "hi"\n')

        result = invoke(runner, ["check", "--dry-run", str(agl_file)])

        assert result.exit_code == 0

    def test_warning_only_file_exits_0_and_prints_warning(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """A TAB-indented line is a genuine warning-severity advisory, not an error."""
        agl_file = tmp_path / "tabby.agl"
        agl_file.write_bytes(b'program def main() -> unit =\n\tprint "hi"\n')

        # A warning-only file does not raise: it exits 0.
        check_command.run(CheckArgs(files=[str(agl_file)]))

        captured = capsys.readouterr()
        assert captured.out == ""
        assert "tabby.agl" in captured.err
        assert "warning:" in captured.err

    def test_invalid_module_root_configuration_reports_error_and_continues(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Inject the failure at the lower-level ``effective_exec_roots`` (in
        # ``agm.cli_support.exec_roots``) so ``check.py``'s own call to the
        # ``_or_none`` wrapper exercises the real error-reporting path,
        # instead of the test reimplementing it.
        from agm.cli_support import exec_roots as exec_roots_module
        from agm.config.module_roots import StdlibResolutionError

        bad = tmp_path / "bad.agl"
        clean = tmp_path / "clean.agl"
        bad.write_text("def value() -> int = 1\n")
        clean.write_text("def other() -> int = 2\n")

        original = exec_roots_module.effective_exec_roots

        def flaky(
            *,
            entry_path: Path | None,
            module_paths: list[str],
            cwd: Path,
            home: Path,
            proj_dir: Path | None,
        ) -> exec_roots_module.ExecRoots:
            if entry_path == bad:
                raise StdlibResolutionError("corrupt active std package")
            return original(
                entry_path=entry_path,
                module_paths=module_paths,
                cwd=cwd,
                home=home,
                proj_dir=proj_dir,
            )

        monkeypatch.setattr(exec_roots_module, "effective_exec_roots", flaky)

        with pytest.raises(SystemExit) as exc_info:
            check_command.run(CheckArgs(files=[str(bad), str(clean)]))
        assert exc_info.value.code == 1

        captured = capsys.readouterr()
        assert captured.err
        # The second, clean file was still checked and produced no diagnostics.
        assert "clean.agl" not in captured.err
