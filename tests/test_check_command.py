"""Tests for the `agm check` CLI command.

Covers:
- CLI wires FILE arguments (one or more), -I/--module-path, and --no-stdlib
  into CheckArgs; missing FILE is a usage error.
- A clean program file, a clean library module (no `program def`, the case
  `agm exec --dry-run` rejects), and a module with no items at all check
  successfully with no output.
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

    @pytest.mark.parametrize("source", ["", "# a placeholder module\n"])
    def test_module_without_items_succeeds(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str], source: str
    ) -> None:
        """A module holding no items at all is legal and checks clean."""
        agl_file = tmp_path / "blank.agl"
        agl_file.write_text(source)

        check_command.run(CheckArgs(files=[str(agl_file)]))

        captured = capsys.readouterr()
        assert captured.out == ""
        assert captured.err == ""

    def test_undefaulted_required_program_parameter_does_not_report_missing_argument(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """``check`` never argument-checks: a required parameter with no default is silent.

        ``check`` statically validates a program without selecting it or binding
        any arguments, so a required ``program def`` parameter with no default
        never reports a missing-argument diagnostic on every save.
        """
        agl_file = tmp_path / "greet.agl"
        agl_file.write_text("program def main(name: text) -> unit =\n  print name\n")

        check_command.run(CheckArgs(files=[str(agl_file)]))

        captured = capsys.readouterr()
        assert captured.out == ""
        assert captured.err == ""

    def test_bad_resource_path_reports_diagnostic_and_exits_1(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """A `resource` referencing a nonexistent path is caught during lowering.

        Regression test for the fix above: ``check`` must reach lowering (not
        stop at match compilation) to still catch this, even though it never
        validates a program's arguments.
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

    def test_undecodable_program_parameter_type_reports_diagnostic_and_exits_1(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """A ``program def`` value parameter must decode from a host-supplied argument."""
        agl_file = tmp_path / "types.agl"
        agl_file.write_text("program def main(p: unit) -> unit = ()\n")

        with pytest.raises(SystemExit) as exc_info:
            check_command.run(CheckArgs(files=[str(agl_file)]))
        assert exc_info.value.code == 1

        captured = capsys.readouterr()
        assert "types.agl" in captured.err
        assert "error:" in captured.err
        assert "unit" in captured.err

    def test_error_inside_imported_module_reports_that_modules_path(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        (tmp_path / "broken.agl").write_text("def f() -> int = undeclared-name\n")
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
        bad_scope.write_text("def bad() -> int = undeclared-name\n")

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


def test_check_searches_the_development_std_checkout_holding_the_entry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A standard-library module is checked against its own checkout, not the store."""
    import semver

    from agm.packages.activation import ActivationIndex, ActivePackage, write_activation_index
    from agm.version import AGM_VERSION

    home = tmp_path / "agm-home"
    monkeypatch.delenv("AGM_STDLIB", raising=False)
    monkeypatch.setenv("AGM_HOME", str(home))
    monkeypatch.chdir(tmp_path)
    store_root = home / "packages" / "std" / AGM_VERSION
    (store_root / "src").mkdir(parents=True)
    (store_root / "package.toml").write_text(
        f'[package]\nname = "std"\nversion = "{AGM_VERSION}"\n', encoding="utf-8"
    )
    (store_root / "src" / "prelude.agl").write_text("def unused() -> int = 0\n", encoding="utf-8")
    write_activation_index(
        ActivationIndex({"std": ActivePackage(semver.Version.parse(AGM_VERSION))}),
        home=home,
        env={"AGM_HOME": str(home)},
    )
    checkout = tmp_path / "checkout"
    (checkout / "src").mkdir(parents=True)
    (checkout / "package.toml").write_text(
        '[package]\nname = "std"\nversion = "9.9.9"\n', encoding="utf-8"
    )
    entry = checkout / "src" / "agent.agl"
    entry.write_text("def value() -> int = 1\n", encoding="utf-8")

    # The checkout holds no ``std/prelude``, so the check fails against it; the
    # activated store tree — which does — was never searched.
    with pytest.raises(SystemExit):
        check_command.run(CheckArgs(files=[str(entry)]))

    # The entry's own path is stripped first, so the checkout is proven present
    # as a searched root rather than as the name of the file being checked.
    searched = capsys.readouterr().err.replace(str(entry.resolve()), "").replace(str(entry), "")
    assert str(checkout.resolve()) in searched
    assert str(store_root.resolve()) not in searched


_CONSTRUCTOR_FORMS = """\
record Point
  x: int
  y: int

record Box[T]
  value: T

enum Shape
  | Circle(r: int)
  | Square(s: int)

enum Opt[T]
  | Nothing
  | Just(v: T)

type PointAlias = Point

type BoxAlias[T] = Box[T]

def point-ctor() -> (int, int) -> Point = Point

def box-ctor() -> (int) -> Box[int] = Box

def circle-ctor() -> (int) -> Shape = Circle

def just-ctor() -> (int) -> Opt[int] = Just

def alias-point() -> PointAlias = PointAlias(x = 1, y = 2)

def alias-box() -> BoxAlias[int] = BoxAlias(value = 2)

def made() -> Shape = Square(s = 3)
"""

_ENUM_ALIAS_AS_VALUE = """\
enum Opt[T]
  | Nothing
  | Just(v: T)

type OptAlias[T] = Opt[T]

def enum-alias() -> int =
  let f = OptAlias
  1
"""


class TestPackageOwnedEntry:
    """A checked file that a package owns is compiled under its module identity."""

    _REPO_STDLIB = Path(__file__).resolve().parents[1] / "packages" / "stdlib"

    def _isolated_home(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Point the AGM home and cwd at *tmp_path*, away from the developer's own."""
        home = tmp_path / "agm-home"
        home.mkdir()
        monkeypatch.setenv("AGM_HOME", str(home))
        monkeypatch.chdir(tmp_path)

    def _development_package(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, name: str
    ) -> Path:
        """Lay out an empty development package *name* and isolate the AGM home."""
        self._isolated_home(tmp_path, monkeypatch)
        root = tmp_path / name
        (root / "src").mkdir(parents=True)
        (root / "package.toml").write_text(
            f'[package]\nname = "{name}"\nversion = "1.0.0"\n', encoding="utf-8"
        )
        return root

    def test_mutually_importing_package_modules_check_from_either_file(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Neither file is anonymous, so each may import the other back."""
        root = self._development_package(tmp_path, monkeypatch, "duo")
        first = root / "src" / "a.agl"
        second = root / "src" / "b.agl"
        first.write_text(
            "import duo/b\ndef fa() -> int = duo/b::fb()\n",
            encoding="utf-8",
        )
        second.write_text(
            "import duo/a\ndef fb() -> int = 2\ndef gb() -> int = duo/a::fa()\n",
            encoding="utf-8",
        )

        check_command.run(CheckArgs(files=[str(first), str(second)], no_stdlib=True))

        captured = capsys.readouterr()
        assert captured.out == ""
        assert captured.err == ""

    def test_wildcard_import_reaches_the_package_owned_entry(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """A named entry is an ordinary member of its package's module tree."""
        root = self._development_package(tmp_path, monkeypatch, "duo")
        first = root / "src" / "a.agl"
        second = root / "src" / "b.agl"
        first.write_text("import duo/b\ndef fa() -> int = 1\n", encoding="utf-8")
        second.write_text("import duo/*\ndef gb() -> int = duo/a::fa()\n", encoding="utf-8")

        check_command.run(CheckArgs(files=[str(first)], no_stdlib=True))

        captured = capsys.readouterr()
        assert captured.out == ""
        assert captured.err == ""

    def test_standard_library_module_checked_directly_accepts_its_builtins(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """``builtin var`` is a standard-library privilege the entry must keep."""
        self._isolated_home(tmp_path, monkeypatch)

        check_command.run(CheckArgs(files=[str(self._REPO_STDLIB / "src" / "config.agl")]))

        captured = capsys.readouterr()
        assert captured.out == ""
        assert captured.err == ""

    def test_standard_library_prelude_checks_without_importing_itself(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        self._isolated_home(tmp_path, monkeypatch)

        check_command.run(CheckArgs(files=[str(self._REPO_STDLIB / "src" / "prelude.agl")]))

        captured = capsys.readouterr()
        assert captured.out == ""
        assert captured.err == ""

    def test_builtin_var_outside_the_standard_library_is_still_rejected(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        root = self._development_package(tmp_path, monkeypatch, "mine")
        module = root / "src" / "settings.agl"
        module.write_text("builtin var log: bool = false\n", encoding="utf-8")

        with pytest.raises(SystemExit) as exc_info:
            check_command.run(CheckArgs(files=[str(module)], no_stdlib=True))

        assert exc_info.value.code == 1
        assert capsys.readouterr().err

    def test_package_owned_module_may_declare_a_builtin_named_function(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """A named module's members live in its own qualified namespace, so a
        built-in spelling is free there whether the file is the checked entry
        or one of its library imports."""
        root = self._development_package(tmp_path, monkeypatch, "kit")
        owner = root / "src" / "files.agl"
        user = root / "src" / "user.agl"
        owner.write_text("def copy(n: int) -> int = n\n", encoding="utf-8")
        user.write_text(
            "import kit/files\ndef twice(n: int) -> int = kit/files::copy(n)\n",
            encoding="utf-8",
        )

        check_command.run(CheckArgs(files=[str(owner)], no_stdlib=True))

        assert capsys.readouterr().err == ""

        check_command.run(CheckArgs(files=[str(user)], no_stdlib=True))

        assert capsys.readouterr().err == ""

    def test_loose_entry_file_may_not_declare_a_builtin_named_function(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """A file no package owns has only a bare namespace to declare into."""
        home = tmp_path / "agm-home"
        home.mkdir()
        monkeypatch.setenv("AGM_HOME", str(home))
        monkeypatch.chdir(tmp_path)
        loose = tmp_path / "loose.agl"
        loose.write_text("def copy(n: int) -> int = n\n", encoding="utf-8")

        with pytest.raises(SystemExit) as exc_info:
            check_command.run(CheckArgs(files=[str(loose)], no_stdlib=True))

        assert exc_info.value.code == 1
        assert capsys.readouterr().err

    def _check_ok(self, path: Path, capsys: pytest.CaptureFixture[str]) -> bool:
        """Return whether checking *path* alone reported no error."""
        try:
            check_command.run(CheckArgs(files=[str(path)], no_stdlib=True))
        except SystemExit:
            return False
        finally:
            capsys.readouterr()
        return True

    @pytest.mark.parametrize(
        "source, accepted", [(_CONSTRUCTOR_FORMS, True), (_ENUM_ALIAS_AS_VALUE, False)]
    )
    def test_own_constructors_read_alike_as_entry_and_as_import(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
        source: str,
        accepted: bool,
    ) -> None:
        """A package module's own constructors resolve against its own name.

        Record and enum constructors, bare and generic, as values and as call
        callees, are accepted or rejected identically whether the host named
        the file or an import reached it.
        """
        root = self._development_package(tmp_path, monkeypatch, "kit")
        owner = root / "src" / "forms.agl"
        owner.write_text(source, encoding="utf-8")
        consumer = root / "src" / "user.agl"
        consumer.write_text("import kit/forms\n", encoding="utf-8")

        assert self._check_ok(owner, capsys) is accepted
        assert self._check_ok(consumer, capsys) is accepted
