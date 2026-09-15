"""Shared test fixtures."""

from __future__ import annotations

import os
import shutil
import signal
from collections.abc import Callable, Generator
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace

import pytest

from agm.agl.self_validation import self_validation_enabled, set_self_validation_enabled
from agm.core import dry_run
from tests import _command_coverage
from tests._durations import (
    pytest_runtest_protocol,
    pytest_sessionfinish,
    pytest_testnodedown,
)

# Re-exported so pytest picks the per-test cost accounting up as conftest hooks.
# Registering the module with ``-p`` instead would break every invocation that
# does not put the repository root on ``sys.path`` (plain ``uv run pytest``).
__all__ = ["pytest_runtest_protocol", "pytest_sessionfinish", "pytest_testnodedown"]

# Enable AgL's optional invariant self-checks — match-compilation self-checks and
# IR structural validation — for the whole test suite.  They are disabled in
# normal execution (zero production cost); turning them on here makes every case
# compiled and every program lowered anywhere in the suite double as an invariant
# oracle.  Individual tests may disable them to exercise the production path (see
# the ``self_validation_disabled`` fixture).
set_self_validation_enabled(True)


def _detach_from_controlling_terminal() -> None:
    """Put this process in a new session so it has no controlling terminal.

    Tests spawn many external processes — interactive shells through the
    workspace-shell wrapper (``bash -i`` / ``zsh -i``), the fake ``tmux`` shell
    script, runner/selector scripts, ``git``.  Each of these opens ``/dev/tty``
    on startup.  When the suite runs in a real terminal and one of those
    processes reads the controlling terminal while *not* in the foreground
    process group, the kernel raises ``SIGTTIN`` and suspends the whole
    ``just check`` job — the intermittent ``zsh: suspended (tty input)`` hang.

    ``os.setsid`` drops the controlling terminal entirely, so ``/dev/tty`` opens
    fail with ``ENXIO`` (exactly as under CI / an agent harness) and ``SIGTTIN``
    can never fire.  This makes the suite behave identically regardless of how
    it is launched and removes a whole class of concurrency-sensitive hangs.

    ``setsid`` fails for a process-group leader; the project always runs tests
    via ``uv run`` (so pytest is a child, never the leader), but we degrade
    gracefully if that ever changes.
    """
    try:
        os.setsid()
    except OSError:
        # Already a session/group leader — leave the disposition unchanged.
        return


def pytest_configure(config: pytest.Config) -> None:
    """Detach test-running processes from the controlling terminal.

    Runs once per process.  Under ``-n auto`` only the xdist *workers* execute
    tests, so the controller stays attached (it owns terminal reporting); each
    worker detaches.  Without xdist the single process runs tests and detaches.

    Also registers the e2e command-coverage gate as a plugin in its own right.
    It declares ``pytest_sessionfinish``/``pytest_testnodedown`` hooks of its
    own, which a conftest re-export (the mechanism used above for the duration
    hooks) cannot express twice under one name.
    """
    config.pluginmanager.register(_command_coverage, "agm_command_coverage")
    is_xdist_worker = hasattr(config, "workerinput")
    xdist_active = getattr(config.option, "dist", "no") != "no"
    if is_xdist_worker or not xdist_active:
        _detach_from_controlling_terminal()


@pytest.fixture(scope="session", autouse=True)
def isolated_compiler_cache(
    tmp_path_factory: pytest.TempPathFactory,
) -> Generator[None, None, None]:
    """Keep compiler artifacts and inherited child-process caches disposable."""
    with pytest.MonkeyPatch.context() as patch:
        patch.setenv("XDG_CACHE_HOME", str(tmp_path_factory.mktemp("compiler-cache")))
        yield


@pytest.fixture()
def self_validation_disabled() -> Generator[None, None, None]:
    """Run the body with AgL's optional self-checks off, as in normal execution.

    The suite enables them globally; tests that pin the production path — where a
    compile or lowering is trusted without being re-verified — take this fixture.
    """
    previous = self_validation_enabled()
    set_self_validation_enabled(False)
    try:
        yield
    finally:
        set_self_validation_enabled(previous)


@pytest.fixture()
def default_sigint() -> Generator[None, None, None]:
    """Ensure SIGINT uses Python's default handler for the duration of a test.

    When the test suite is launched as a background process (e.g. by a CI runner
    or agent harness, or with a trailing ``&``), the shell sets SIGINT to
    ``SIG_IGN`` and child processes — pytest-xdist workers and any subprocess
    they spawn — inherit it.  Interrupt tests must restore the default
    disposition so SIGINT is actually delivered as ``KeyboardInterrupt`` and so
    spawned ``agm`` processes reset to the default handler on ``exec`` instead of
    silently ignoring the signal.
    """

    previous = signal.signal(signal.SIGINT, signal.default_int_handler)
    try:
        yield
    finally:
        signal.signal(signal.SIGINT, previous)


@pytest.fixture(autouse=True)
def reset_dry_run_state() -> Generator[None, None, None]:
    dry_run.set_enabled(False)
    yield
    dry_run.set_enabled(False)


@pytest.fixture(autouse=True)
def clear_project_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("PROJ_DIR", raising=False)
    monkeypatch.delenv("REPO_DIR", raising=False)


@pytest.fixture(autouse=True)
def clear_workspace_shell_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Strip agm workspace-shell control variables inherited from the host.

    When the suite is run from inside an agm workspace shell, the process
    inherits ``AGM_REAL_SHELL``/``AGM_WORKSPACE_SHELL`` and friends.  Any test
    that spawns the shell wrapper with ``{**os.environ}`` would then pick up the
    host's real shell instead of the one it set up, so the wrapper would launch
    e.g. zsh in place of the requested bash.  Remove them so tests behave the
    same whether or not the suite itself runs inside a workspace shell.
    """

    for name in list(os.environ):
        if name.startswith("AGM_"):
            monkeypatch.delenv(name, raising=False)


_REPO_STDLIB_ROOT = Path(__file__).resolve().parent.parent / "packages" / "stdlib"


@pytest.fixture(autouse=True)
def pin_agm_stdlib_to_repo(
    clear_workspace_shell_env: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Pin the AgL stdlib to the in-repo tree so tests never read installed files.

    ``clear_workspace_shell_env`` strips every ``AGM_*`` variable first; this
    runs afterwards and points ``AGM_STDLIB`` at ``<repo>/packages/stdlib``.  Every
    ``agm exec``/``agm repl`` invocation — in-process or as a subprocess that
    inherits the environment — then resolves the standard library from the
    current source tree, independent of whatever ``~/.agm`` happens to contain.
    """
    monkeypatch.setenv("AGM_STDLIB", str(_REPO_STDLIB_ROOT))


@pytest.fixture(autouse=True)
def detach_installed_agm_prefix(monkeypatch: pytest.MonkeyPatch) -> None:
    """Hide any AGM installed at the test runner's own prefix from the suite.

    ``agm_home_dir`` falls back to ``<installation prefix>/.agm`` whenever that
    prefix holds a package activation index, and the prefix is derived from
    ``sys.argv[0]``.  Launched as ``uv run pytest`` that prefix is the project's
    ``.venv``, so a developer who had run ``uv run agm pkg install`` would make
    the suite read — and write — a real installed package store instead of its
    own temporary one.  ``AGM_HOME`` cannot prevent this, because most tests
    pass an explicit ``env`` mapping that never sees the process environment.

    Report no installation prefix instead, so an installed tree is invisible
    regardless of how the suite was launched.  Tests that exercise the fallback
    take the ``installed_agm_prefix`` fixture.
    """
    monkeypatch.setattr("agm.config.general.agm_installation_prefix", lambda: None)


@pytest.fixture()
def installed_agm_prefix(monkeypatch: pytest.MonkeyPatch) -> Callable[[Path], None]:
    """Opt out of ``detach_installed_agm_prefix`` for one test.

    Call the returned function with a staged temporary prefix to restore the
    installed-prefix fallback for the rest of the test.  This is the documented
    escape hatch for the tests that assert the fallback itself; it points at a
    temporary directory, never at a real installation.
    """

    def pin(prefix: Path) -> None:
        monkeypatch.setattr("agm.config.general.agm_installation_prefix", lambda: prefix)

    return pin


@pytest.fixture()
def env(tmp_path: Path) -> dict[str, str]:
    """Environment dict with git identity and isolated HOME."""
    e = os.environ.copy()
    e["GIT_AUTHOR_NAME"] = "Test"
    e["GIT_AUTHOR_EMAIL"] = "test@test.com"
    e["GIT_COMMITTER_NAME"] = "Test"
    e["GIT_COMMITTER_EMAIL"] = "test@test.com"
    e["GIT_CONFIG_NOSYSTEM"] = "1"
    e["SHELL"] = shutil.which("bash") or "/bin/sh"
    e.pop("PROJ_DIR", None)
    e.pop("REPO_DIR", None)
    e.pop("TMUX", None)
    e.pop("TMUX_PANE", None)
    fake_home = tmp_path / "home"
    fake_home.mkdir()
    e["HOME"] = str(fake_home)
    return e


@dataclass(slots=True)
class FakeAgentTransport:
    """Records dispatched agent runs and stubs their transport result.

    Every dispatched ``(prompt, runner argv)`` pair is recorded on ``calls``.
    Queue responses built with :meth:`success` / :meth:`failure` via
    :meth:`queue` to script what consecutive dispatches return, in order.  A
    transport that was never scripted answers every dispatch with a placeholder
    success; once scripted, a dispatch past the last queued response fails the
    test rather than inventing one.
    """

    calls: list[tuple[str, list[str]]] = field(default_factory=list)
    _responses: list[object] = field(default_factory=list)
    _scripted: bool = False

    def queue(self, *responses: object) -> None:
        """Append *responses* to the FIFO consumed one-per-dispatch."""
        self._scripted = True
        self._responses.extend(responses)

    @staticmethod
    def success(stdout: str = "ok") -> SimpleNamespace:
        """Build a successful transport result carrying *stdout*."""
        return SimpleNamespace(
            spawn_error=None, timed_out=False, returncode=0, stdout=stdout, stderr="", elapsed=0.0
        )

    @staticmethod
    def failure(
        *,
        spawn_error: str | None = None,
        timed_out: bool = False,
        returncode: int | None = None,
        stdout: str = "",
        stderr: str = "",
        elapsed: float = 0.0,
    ) -> SimpleNamespace:
        """Build a failing transport result: spawn error, timeout, or bad exit code."""
        return SimpleNamespace(
            spawn_error=spawn_error,
            timed_out=timed_out,
            returncode=returncode,
            stdout=stdout,
            stderr=stderr,
            elapsed=elapsed,
        )

    def _next_response(self) -> object:
        if self._responses:
            return self._responses.pop(0)
        if self._scripted:
            raise AssertionError("agent dispatched more times than there were queued responses")
        return self.success()


@pytest.fixture()
def fake_agent_transport(monkeypatch: pytest.MonkeyPatch) -> FakeAgentTransport:
    """Stub the ``agm.agent.runner`` prepare/run seam that every agent dispatch uses.

    Replaces ``prepare_rendered_prompt_run`` and ``run_prepared_prompt_result``
    so no real subprocess is spawned, regardless of whether the dispatch is
    driven through the ``agm exec`` CLI, ``exec_command.run``, or a bare
    ``PipelineDriver`` — all three route through the same seam. See
    :class:`FakeAgentTransport` for recording/queuing behavior.
    """
    from agm.agent.runner import PreparedPromptRun

    transport = FakeAgentTransport()

    def prepare(prompt: str, *, runner: list[str], **_: object) -> PreparedPromptRun:
        transport.calls.append((prompt, runner))
        return PreparedPromptRun(
            command=runner, effective_file=Path("/dev/null"), env={}, temp_files=[]
        )

    monkeypatch.setattr("agm.agent.runner.prepare_rendered_prompt_run", prepare)
    monkeypatch.setattr(
        "agm.agent.runner.run_prepared_prompt_result",
        lambda _prepared, **_: transport._next_response(),
    )
    return transport


# pytest names each test's ``tmp_path`` directory after the test itself, so an
# absolute path embedded in a message can supply the very word an assertion is
# looking for — ``assert "ambiguous" in message`` has passed on the strength of
# a ``..._ambiguous_module_fails0`` path component alone, with the production
# wording deleted.  Setting AGM_TEST_NEUTRAL_TMP_PATH=1 hands out neutrally
# named directories instead, so any assertion resting on its own test's name
# fails.  Off by default: test-named temp directories are worth keeping when
# reading a failure.
if os.environ.get("AGM_TEST_NEUTRAL_TMP_PATH") == "1":

    @pytest.fixture
    def tmp_path(tmp_path_factory: pytest.TempPathFactory) -> Path:
        """Temp directory whose name cannot leak the test name into assertions."""
        return tmp_path_factory.mktemp("t")
