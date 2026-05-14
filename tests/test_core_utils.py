"""Comprehensive tests for agm.core.fs, agm.core.dotenv, and agm.core.dry_run."""

from __future__ import annotations

import os
import shlex
import stat
from collections.abc import Generator
from pathlib import Path

import pytest

import agm.core.dry_run as dry_run
from agm.core.dotenv import set_dotenv_value
from agm.core.fs import (
    access,
    append_text,
    chmod,
    exists,
    is_dir,
    is_empty_dir,
    is_file,
    iterdir,
    mkdir,
    read_text,
    rglob,
    rmdir,
    rmtree,
    unlink,
    write_text,
)
from agm.core.fs import (
    stat as fs_stat,
)
from agm.core.path import display_path

# ---------------------------------------------------------------------------
# Fixture: always reset dry-run state after each test
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def reset_dry_run() -> Generator[None, None, None]:
    """Ensure dry-run state is reset to False after every test."""
    dry_run.set_enabled(False)
    yield
    dry_run.set_enabled(False)


# ===========================================================================
# agm.core.dry_run
# ===========================================================================


class TestDryRunState:
    def test_enabled_is_false_by_default(self) -> None:
        assert dry_run.enabled() is False

    def test_set_enabled_true(self) -> None:
        dry_run.set_enabled(True)
        assert dry_run.enabled() is True

    def test_set_enabled_false_resets(self) -> None:
        dry_run.set_enabled(True)
        dry_run.set_enabled(False)
        assert dry_run.enabled() is False


class TestFormatCommand:
    def test_simple_command(self) -> None:
        assert dry_run.format_command(["git", "status"]) == "git status"

    def test_quotes_parts_with_spaces(self) -> None:
        result = dry_run.format_command(["echo", "hello world"])
        assert result == "echo 'hello world'"

    def test_quotes_parts_with_special_chars(self) -> None:
        result = dry_run.format_command(["sh", "-c", "echo $VAR"])
        assert result == "sh -c 'echo $VAR'"

    def test_single_element(self) -> None:
        assert dry_run.format_command(["ls"]) == "ls"

    def test_empty_list(self) -> None:
        assert dry_run.format_command([]) == ""


class TestFormatCommandWithCwd:
    def test_no_cwd_returns_plain_command(self) -> None:
        result = dry_run.format_command_with_cwd(["git", "pull"])
        assert result == "git pull"

    def test_with_cwd_wraps_in_cd(self, tmp_path: Path) -> None:
        result = dry_run.format_command_with_cwd(["make"], cwd=tmp_path)
        assert result == f"(cd {tmp_path} && make)"

    def test_with_cwd_path_containing_spaces_is_shell_quoted(self, tmp_path: Path) -> None:
        # Use a fixed path with a known space so the assertion is deterministic.
        spaced = Path("/tmp/my project/src")
        result = dry_run.format_command_with_cwd(["make"], cwd=spaced)
        quoted = shlex.quote(str(spaced))
        assert result == f"(cd {quoted} && make)"

    def test_with_cwd_none_is_same_as_omitting_cwd(self) -> None:
        cmd = ["cargo", "build"]
        assert dry_run.format_command_with_cwd(cmd, cwd=None) == dry_run.format_command(cmd)


class TestPrintFunctions:
    def test_print_command_no_cwd(self, capsys: pytest.CaptureFixture[str]) -> None:
        dry_run.print_command(["git", "fetch"])
        captured = capsys.readouterr()
        assert captured.out == "dry-run: git fetch\n"

    def test_print_command_with_cwd(
        self, capsys: pytest.CaptureFixture[str], tmp_path: Path
    ) -> None:
        dry_run.print_command(["make"], cwd=tmp_path)
        captured = capsys.readouterr()
        assert captured.out == f"dry-run: (cd {tmp_path} && make)\n"

    def test_print_operation(self, capsys: pytest.CaptureFixture[str]) -> None:
        dry_run.print_operation("mkdir", "/some/path")
        captured = capsys.readouterr()
        assert captured.out == "dry-run: agm mkdir /some/path\n"

    def test_print_configuration(self, capsys: pytest.CaptureFixture[str]) -> None:
        dry_run.print_configuration("worktree")
        captured = capsys.readouterr()
        assert captured.out == "dry-run: worktree configuration\n"

    def test_print_detail(self, capsys: pytest.CaptureFixture[str]) -> None:
        dry_run.print_detail("branch", "main")
        captured = capsys.readouterr()
        assert captured.out == "dry-run:   branch: main\n"

    def test_print_labeled_command_no_cwd(self, capsys: pytest.CaptureFixture[str]) -> None:
        dry_run.print_labeled_command("setup", ["uv", "sync"])
        captured = capsys.readouterr()
        assert captured.out == "dry-run: command [setup]: uv sync\n"

    def test_print_labeled_command_with_cwd(
        self, capsys: pytest.CaptureFixture[str], tmp_path: Path
    ) -> None:
        dry_run.print_labeled_command("build", ["make"], cwd=tmp_path)
        captured = capsys.readouterr()
        assert captured.out == f"dry-run: command [build]: (cd {tmp_path} && make)\n"


# ===========================================================================
# agm.core.fs — read-only wrappers
# ===========================================================================


class TestFsReadOnly:
    def test_exists_true_for_file(self, tmp_path: Path) -> None:
        f = tmp_path / "a.txt"
        f.write_text("hi", encoding="utf-8")
        assert exists(f) is True

    def test_exists_false_for_missing(self, tmp_path: Path) -> None:
        assert exists(tmp_path / "ghost.txt") is False

    def test_exists_true_for_directory(self, tmp_path: Path) -> None:
        assert exists(tmp_path) is True

    def test_is_file_true(self, tmp_path: Path) -> None:
        f = tmp_path / "file.txt"
        f.write_text("x", encoding="utf-8")
        assert is_file(f) is True

    def test_is_file_false_for_directory(self, tmp_path: Path) -> None:
        assert is_file(tmp_path) is False

    def test_is_file_false_for_missing(self, tmp_path: Path) -> None:
        assert is_file(tmp_path / "nope") is False

    def test_is_dir_true(self, tmp_path: Path) -> None:
        assert is_dir(tmp_path) is True

    def test_is_dir_false_for_file(self, tmp_path: Path) -> None:
        f = tmp_path / "file.txt"
        f.write_text("x", encoding="utf-8")
        assert is_dir(f) is False

    def test_is_dir_false_for_missing(self, tmp_path: Path) -> None:
        assert is_dir(tmp_path / "no_dir") is False

    def test_read_text_returns_content(self, tmp_path: Path) -> None:
        f = tmp_path / "hello.txt"
        f.write_text("hello world", encoding="utf-8")
        assert read_text(f) == "hello world"

    def test_read_text_respects_encoding(self, tmp_path: Path) -> None:
        f = tmp_path / "latin.txt"
        f.write_bytes("caf\xe9".encode("latin-1"))
        assert read_text(f, encoding="latin-1") == "café"

    def test_stat_returns_size_and_mtime(self, tmp_path: Path) -> None:
        f = tmp_path / "s.txt"
        f.write_text("data", encoding="utf-8")
        result = fs_stat(f)
        assert result.st_size > 0
        assert result.st_mtime > 0

    def test_iterdir_lists_children(self, tmp_path: Path) -> None:
        (tmp_path / "a").write_text("", encoding="utf-8")
        (tmp_path / "b").write_text("", encoding="utf-8")
        children = iterdir(tmp_path)
        assert sorted(p.name for p in children) == ["a", "b"]

    def test_iterdir_empty_directory(self, tmp_path: Path) -> None:
        d = tmp_path / "empty"
        d.mkdir()
        assert iterdir(d) == []

    def test_rglob_finds_nested_files(self, tmp_path: Path) -> None:
        sub = tmp_path / "sub"
        sub.mkdir()
        (sub / "x.py").write_text("", encoding="utf-8")
        (tmp_path / "y.py").write_text("", encoding="utf-8")
        (tmp_path / "z.txt").write_text("", encoding="utf-8")
        results = rglob(tmp_path, "*.py")
        names = {p.name for p in results}
        assert names == {"x.py", "y.py"}

    def test_rglob_no_matches(self, tmp_path: Path) -> None:
        assert rglob(tmp_path, "*.rs") == []

    def test_is_empty_dir_true(self, tmp_path: Path) -> None:
        d = tmp_path / "empty"
        d.mkdir()
        assert is_empty_dir(d) is True

    def test_is_empty_dir_false(self, tmp_path: Path) -> None:
        (tmp_path / "file.txt").write_text("x", encoding="utf-8")
        assert is_empty_dir(tmp_path) is False

    def test_access_readable(self, tmp_path: Path) -> None:
        f = tmp_path / "r.txt"
        f.write_text("data", encoding="utf-8")
        assert access(f, os.R_OK) is True

    def test_access_non_existent(self, tmp_path: Path) -> None:
        assert access(tmp_path / "ghost", os.R_OK) is False


# ===========================================================================
# agm.core.fs — write operations (normal mode)
# ===========================================================================


class TestFsWriteNormal:
    def test_mkdir_creates_directory(self, tmp_path: Path) -> None:
        d = tmp_path / "new_dir"
        mkdir(d)
        assert d.is_dir()

    def test_mkdir_parents(self, tmp_path: Path) -> None:
        d = tmp_path / "a" / "b" / "c"
        mkdir(d, parents=True)
        assert d.is_dir()

    def test_mkdir_exist_ok(self, tmp_path: Path) -> None:
        mkdir(tmp_path, exist_ok=True)  # should not raise

    def test_write_text_creates_file(self, tmp_path: Path) -> None:
        f = tmp_path / "out.txt"
        write_text(f, "content")
        assert f.read_text(encoding="utf-8") == "content"

    def test_write_text_overwrites(self, tmp_path: Path) -> None:
        f = tmp_path / "out.txt"
        f.write_text("old", encoding="utf-8")
        write_text(f, "new")
        assert f.read_text(encoding="utf-8") == "new"

    def test_chmod_changes_mode(self, tmp_path: Path) -> None:
        f = tmp_path / "script.sh"
        f.write_text("#!/bin/sh\n", encoding="utf-8")
        chmod(f, 0o755)
        mode = f.stat().st_mode & 0o777
        assert mode == 0o755

    def test_append_text_appends_content(self, tmp_path: Path) -> None:
        f = tmp_path / "log.txt"
        f.write_text("line1\n", encoding="utf-8")
        append_text(f, "line2\n")
        assert f.read_text(encoding="utf-8") == "line1\nline2\n"

    def test_append_text_creates_file_if_missing(self, tmp_path: Path) -> None:
        f = tmp_path / "new.txt"
        append_text(f, "hello")
        assert f.read_text(encoding="utf-8") == "hello"

    def test_rmtree_removes_directory_tree(self, tmp_path: Path) -> None:
        d = tmp_path / "tree"
        (d / "sub").mkdir(parents=True)
        (d / "sub" / "file.txt").write_text("x", encoding="utf-8")
        rmtree(d)
        assert not d.exists()

    def test_rmdir_removes_empty_directory(self, tmp_path: Path) -> None:
        d = tmp_path / "empty"
        d.mkdir()
        rmdir(d)
        assert not d.exists()

    def test_unlink_removes_file(self, tmp_path: Path) -> None:
        f = tmp_path / "del.txt"
        f.write_text("bye", encoding="utf-8")
        unlink(f)
        assert not f.exists()

    def test_unlink_missing_ok(self, tmp_path: Path) -> None:
        unlink(tmp_path / "ghost.txt", missing_ok=True)  # should not raise

    def test_unlink_missing_raises_by_default(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            unlink(tmp_path / "ghost.txt")


# ===========================================================================
# agm.core.fs — write operations (dry-run mode)
# ===========================================================================


class TestFsWriteDryRun:
    def test_mkdir_dry_run_prints_and_skips(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        d = tmp_path / "new_dir"
        dry_run.set_enabled(True)
        mkdir(d)
        assert not d.exists()
        captured = capsys.readouterr()
        assert "dry-run: agm mkdir" in captured.out
        assert str(d) in captured.out

    def test_write_text_dry_run_prints_and_skips(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        f = tmp_path / "out.txt"
        dry_run.set_enabled(True)
        write_text(f, "should not write")
        assert not f.exists()
        captured = capsys.readouterr()
        assert "dry-run: agm write-file" in captured.out
        assert str(f) in captured.out

    def test_chmod_dry_run_prints_and_skips(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        f = tmp_path / "script.sh"
        f.write_text("#!/bin/sh\n", encoding="utf-8")
        original_mode = f.stat().st_mode & 0o777
        dry_run.set_enabled(True)
        chmod(f, 0o755)
        assert (f.stat().st_mode & 0o777) == original_mode
        captured = capsys.readouterr()
        assert "dry-run: agm chmod" in captured.out
        assert "0o755" in captured.out
        assert str(f) in captured.out

    def test_append_text_dry_run_prints_and_skips(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        f = tmp_path / "log.txt"
        f.write_text("original\n", encoding="utf-8")
        dry_run.set_enabled(True)
        append_text(f, "extra")
        assert f.read_text(encoding="utf-8") == "original\n"
        captured = capsys.readouterr()
        assert "dry-run: agm append-file" in captured.out
        assert str(f) in captured.out

    def test_rmtree_dry_run_prints_and_skips(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        d = tmp_path / "tree"
        d.mkdir()
        (d / "file.txt").write_text("x", encoding="utf-8")
        dry_run.set_enabled(True)
        rmtree(d)
        assert d.exists()
        captured = capsys.readouterr()
        assert "dry-run: agm remove-tree" in captured.out
        assert str(d) in captured.out

    def test_rmdir_dry_run_prints_and_skips(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        d = tmp_path / "empty"
        d.mkdir()
        dry_run.set_enabled(True)
        rmdir(d)
        assert d.exists()
        captured = capsys.readouterr()
        assert "dry-run: agm rmdir" in captured.out
        assert str(d) in captured.out

    def test_unlink_dry_run_prints_and_skips(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        f = tmp_path / "keep.txt"
        f.write_text("keep me", encoding="utf-8")
        dry_run.set_enabled(True)
        unlink(f)
        assert f.exists()
        captured = capsys.readouterr()
        assert "dry-run: agm unlink" in captured.out
        assert str(f) in captured.out


# ===========================================================================
# agm.core.dotenv
# ===========================================================================


class TestSetDotenvValue:
    def test_creates_new_file_with_key(self, tmp_path: Path) -> None:
        env_file = tmp_path / ".env"
        set_dotenv_value(env_file, "FOO", "bar")
        assert env_file.read_text(encoding="utf-8") == "FOO=bar\n"

    def test_creates_parent_directories(self, tmp_path: Path) -> None:
        env_file = tmp_path / "a" / "b" / ".env"
        set_dotenv_value(env_file, "KEY", "val")
        assert env_file.exists()
        assert "KEY=val" in env_file.read_text(encoding="utf-8")

    def test_appends_new_key_to_existing_file(self, tmp_path: Path) -> None:
        env_file = tmp_path / ".env"
        env_file.write_text("EXISTING=yes\n", encoding="utf-8")
        set_dotenv_value(env_file, "NEW", "value")
        content = env_file.read_text(encoding="utf-8")
        assert "EXISTING=yes\n" in content
        assert "NEW=value\n" in content

    def test_replaces_existing_key(self, tmp_path: Path) -> None:
        env_file = tmp_path / ".env"
        env_file.write_text("FOO=old\nBAR=keep\n", encoding="utf-8")
        set_dotenv_value(env_file, "FOO", "new")
        content = env_file.read_text(encoding="utf-8")
        assert "FOO=new\n" in content
        assert "FOO=old" not in content
        assert "BAR=keep\n" in content

    def test_replaces_key_with_export_prefix(self, tmp_path: Path) -> None:
        env_file = tmp_path / ".env"
        env_file.write_text("export FOO=old\nBAR=keep\n", encoding="utf-8")
        set_dotenv_value(env_file, "FOO", "updated")
        content = env_file.read_text(encoding="utf-8")
        assert "FOO=updated\n" in content
        assert "export FOO=old" not in content
        assert "BAR=keep\n" in content

    def test_replaces_only_first_occurrence_of_duplicate_key(self, tmp_path: Path) -> None:
        env_file = tmp_path / ".env"
        env_file.write_text("FOO=first\nFOO=second\n", encoding="utf-8")
        set_dotenv_value(env_file, "FOO", "once")
        content = env_file.read_text(encoding="utf-8")
        assert content.count("FOO=") == 1
        assert "FOO=once\n" in content

    def test_ensures_trailing_newline_before_append(self, tmp_path: Path) -> None:
        env_file = tmp_path / ".env"
        env_file.write_bytes(b"NOEOL=yes")  # no trailing newline
        set_dotenv_value(env_file, "ADDED", "val")
        content = env_file.read_text(encoding="utf-8")
        lines = content.splitlines()
        assert "NOEOL=yes" in lines
        assert "ADDED=val" in lines

    def test_key_not_partially_matched(self, tmp_path: Path) -> None:
        env_file = tmp_path / ".env"
        env_file.write_text("FOOBAR=oops\n", encoding="utf-8")
        set_dotenv_value(env_file, "FOO", "new")
        content = env_file.read_text(encoding="utf-8")
        assert "FOOBAR=oops\n" in content
        assert "FOO=new\n" in content

    def test_replaces_key_with_spaces_around_equals(self, tmp_path: Path) -> None:
        env_file = tmp_path / ".env"
        env_file.write_text("FOO =old\n", encoding="utf-8")
        set_dotenv_value(env_file, "FOO", "trimmed")
        content = env_file.read_text(encoding="utf-8")
        assert "FOO=trimmed\n" in content
        assert "FOO =old" not in content

    def test_dry_run_does_not_write_file(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        env_file = tmp_path / ".env"
        dry_run.set_enabled(True)
        set_dotenv_value(env_file, "KEY", "value")
        assert not env_file.exists()
        captured = capsys.readouterr()
        assert "dry-run" in captured.out

    def test_value_with_special_characters(self, tmp_path: Path) -> None:
        env_file = tmp_path / ".env"
        set_dotenv_value(env_file, "URL", "https://example.com/path?q=1&r=2")
        content = env_file.read_text(encoding="utf-8")
        assert "URL=https://example.com/path?q=1&r=2\n" in content

    def test_empty_file_gets_key_appended(self, tmp_path: Path) -> None:
        env_file = tmp_path / ".env"
        env_file.write_text("", encoding="utf-8")
        set_dotenv_value(env_file, "EMPTY_FILE_KEY", "yes")
        content = env_file.read_text(encoding="utf-8")
        assert "EMPTY_FILE_KEY=yes\n" in content

    def test_preserves_comments_and_blank_lines(self, tmp_path: Path) -> None:
        env_file = tmp_path / ".env"
        env_file.write_text("# comment\n\nFOO=bar\n", encoding="utf-8")
        set_dotenv_value(env_file, "NEW", "val")
        content = env_file.read_text(encoding="utf-8")
        assert "# comment\n" in content
        assert "\n" in content
        assert "FOO=bar\n" in content
        assert "NEW=val\n" in content

    def test_stat_returns_correct_size(self, tmp_path: Path) -> None:
        """fs.stat reports the right byte-size for a file."""
        f = tmp_path / "measured.txt"
        text = "hello\n"
        f.write_text(text, encoding="utf-8")
        result = fs_stat(f)
        assert result.st_size == len(text.encode("utf-8"))

    def test_access_write_permission(self, tmp_path: Path) -> None:
        f = tmp_path / "writable.txt"
        f.write_text("data", encoding="utf-8")
        assert access(f, os.W_OK) is True

    def test_access_execute_permission_on_script(self, tmp_path: Path) -> None:
        f = tmp_path / "run.sh"
        f.write_text("#!/bin/sh\n", encoding="utf-8")
        f.chmod(stat.S_IRWXU)
        assert access(f, os.X_OK) is True


# ===========================================================================
# agm.core.path — display_path
# ===========================================================================


class TestDisplayPath:
    def test_path_under_cwd_returns_relative(self, tmp_path: Path) -> None:
        path = tmp_path / "sub" / "file.log"
        assert display_path(path, cwd=tmp_path) == str(Path("sub") / "file.log")

    def test_path_equals_cwd_returns_dot(self, tmp_path: Path) -> None:
        assert display_path(tmp_path, cwd=tmp_path) == "."

    def test_path_outside_cwd_returns_absolute(self, tmp_path: Path) -> None:
        other = Path("/tmp/other/place.log")
        assert display_path(other, cwd=tmp_path) == str(other)

    def test_uses_path_cwd_by_default(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.chdir(tmp_path)
        child = tmp_path / "a.log"
        assert display_path(child) == "a.log"

    def test_nested_directory_under_cwd(self, tmp_path: Path) -> None:
        path = tmp_path / ".agent-files" / "loop-20250101-120000.log"
        result = display_path(path, cwd=tmp_path)
        assert result == str(Path(".agent-files") / "loop-20250101-120000.log")
