"""An enforced gate for "100% command coverage in e2e tests".

Line coverage cannot express this guarantee.  A command's implementation can be
fully covered by in-process unit tests while the command itself is never once
driven through the real ``agm`` binary, so a broken argument container, a
missing registration or a crash in CLI wiring ships unnoticed.  What has to be
checked is the *command surface*: every leaf command AGM registers must be
invoked, as a subprocess, by at least one e2e test.

Both halves of that comparison are derived, never written down:

* The command set comes from walking the live Typer registry of
  :data:`agm.cli.app` — its ``registered_commands`` and ``registered_groups``,
  recursively.  Adding a subcommand anywhere in the tree therefore adds a gate
  obligation with no test-side bookkeeping.  Walking the Typer objects rather
  than the Click tree also keeps package-registered commands (resolved
  dynamically at invocation time) out of the set: they are not AGM's own
  surface.
* The invoked set is recorded at runtime by ``tests.test_e2e.run_agm``, the one
  helper every e2e test uses to spawn the binary.  Recording the *actual* argv
  means a command counts as covered only when it really ran — not when a call
  merely appears in the source of a skipped, dead or dynamically-parameterized
  test.

Group spellings that are pure aliases (``wsp`` for ``workspace``, ``wt`` for
``worktree`` — see ``agm.command_catalog.COMMAND_ALIASES``) are folded onto
their canonical group before comparison.  Distinctly registered leaves such as
``config cp`` and ``config copy`` are *not* folded: they are separate callbacks
and each needs its own end-to-end proof.

Aggregation across ``pytest-xdist`` workers reuses the pattern established by
:mod:`tests._durations`: each worker ships its recorded set up through
``config.workeroutput`` at session finish and the controller merges the shipped
sets in ``pytest_testnodedown`` before judging.  The module is registered as a
plugin by ``tests/conftest.py`` so its hooks live beside — not on top of — the
duration hooks that share those names.

The gate only fires on a full-suite run; see :func:`gate_enabled`.
"""

from __future__ import annotations

import os
from collections.abc import Iterable, Sequence
from pathlib import Path

import pytest
import typer
from typer.main import get_command_name

from agm.cli import app
from agm.command_catalog import COMMAND_ALIASES

# Forces the gate on ("1") or off ("0"), overriding full-suite detection.
GATE_ENV = "AGM_TEST_COMMAND_COVERAGE"

_WORKEROUTPUT_KEY = "agm_command_coverage"

# Leaf commands that cannot be driven end-to-end, with the reason each is out.
EXCLUDED_COMMANDS: dict[tuple[str, ...], str] = {}

# Canonical leaf paths invoked through ``run_agm`` in this process.
_invoked: set[tuple[str, ...]] = set()


# ---------------------------------------------------------------------------
# The command surface, read off the live Typer registry
# ---------------------------------------------------------------------------


def _command_name(info: typer.models.CommandInfo) -> str:
    """Return the CLI name Typer registers *info* under."""
    if info.name:
        return info.name
    assert info.callback is not None, "a Typer command has either a name or a callback"
    return get_command_name(info.callback.__name__)


def group_alias_map(root: typer.Typer = app) -> dict[str, str]:
    """Return ``{alias group name: canonical group name}`` for *root*'s groups.

    An alias registers the *same* sub-application under a second name, so the
    canonical spelling is found by identity: the non-alias name that carries the
    same ``Typer`` instance.  Which names are aliases is AGM's own declaration
    (``agm.command_catalog.COMMAND_ALIASES``), not a guess.
    """
    canonical: dict[int, str] = {
        id(group.typer_instance): group.name or ""
        for group in root.registered_groups
        if group.name not in COMMAND_ALIASES
    }
    return {
        group.name: canonical[id(group.typer_instance)]
        for group in root.registered_groups
        if group.name in COMMAND_ALIASES and id(group.typer_instance) in canonical
    }


def _walk(node: typer.Typer, prefix: tuple[str, ...]) -> Iterable[tuple[str, ...]]:
    """Yield every leaf-command path reachable from the Typer app *node*."""
    for info in node.registered_commands:
        yield (*prefix, _command_name(info))
    for group in node.registered_groups:
        if group.typer_instance is not None:
            yield from _walk(group.typer_instance, (*prefix, group.name or ""))


def leaf_commands(root: typer.Typer = app) -> frozenset[tuple[str, ...]]:
    """Return every canonical leaf-command path AGM registers."""
    aliases = group_alias_map(root)
    return frozenset((aliases.get(path[0], path[0]), *path[1:]) for path in _walk(root, ()) if path)


# ---------------------------------------------------------------------------
# Recording what an e2e invocation actually ran
# ---------------------------------------------------------------------------


def _command_tree(root: typer.Typer = app) -> dict[str, object]:
    """Return the command tree as nested dicts; ``None`` marks a leaf."""
    tree: dict[str, object] = {}
    for info in root.registered_commands:
        tree[_command_name(info)] = None
    for group in root.registered_groups:
        if group.typer_instance is not None:
            tree[group.name or ""] = _command_tree(group.typer_instance)
    return tree


def resolve_leaf_path(args: Sequence[str], root: typer.Typer = app) -> tuple[str, ...] | None:
    """Return the canonical leaf command *args* invokes, or ``None``.

    Options (and, by construction, AGM's global options — all of them flags) are
    skipped; the walk descends through group names and stops at the first token
    that is not one.  Argv that names no registered leaf — a package-registered
    command, a bare group, a usage error — resolves to ``None``.
    """
    aliases = group_alias_map(root)
    node = _command_tree(root)
    path: list[str] = []
    for token in args:
        if token.startswith("-"):
            continue
        if token not in node:
            return None
        path.append(token)
        child = node[token]
        if child is None:
            return (aliases.get(path[0], path[0]), *path[1:])
        node = child
    return None


def record_invocation(args: Sequence[str]) -> None:
    """Record the leaf command *args* invokes, if it names one."""
    path = resolve_leaf_path(args)
    if path is not None:
        _invoked.add(path)


def recorded_commands() -> frozenset[tuple[str, ...]]:
    """Return the leaf commands recorded in this process so far."""
    return frozenset(_invoked)


# ---------------------------------------------------------------------------
# Gate scope
# ---------------------------------------------------------------------------


def gate_enabled(config: pytest.Config) -> bool:
    """Return whether this session is a full-suite run the gate may judge.

    Running a single file must stay silent: an e2e-free session would otherwise
    report the entire command surface as uncovered.  A session qualifies when
    every path it was pointed at contains the whole ``tests`` directory and no
    selection option (``-k``, ``-m``, ``--deselect``, ``--lf``/``--ff``) narrows
    what runs.  ``AGM_TEST_COMMAND_COVERAGE`` overrides the detection in either
    direction, which is how the gate is exercised on a subset.
    """
    override = os.environ.get(GATE_ENV)
    if override:
        return override != "0"
    option = config.option
    if option.collectonly:
        return False
    if option.keyword or option.markexpr or option.deselect:
        return False
    if getattr(option, "lf", False) or getattr(option, "failedfirst", False):
        return False
    tests_dir = Path(__file__).resolve().parent
    invocation_dir = Path(config.invocation_params.dir)
    args = [str(arg).split("::")[0] for arg in config.args]
    if not args:
        return False
    return all(
        (target := (invocation_dir / arg).resolve()) == tests_dir or target in tests_dir.parents
        for arg in args
    )


# ---------------------------------------------------------------------------
# pytest hooks
# ---------------------------------------------------------------------------


def pytest_sessionfinish(session: pytest.Session) -> None:
    """Ship a worker's recorded commands upwards, or judge them on the controller."""
    workeroutput = getattr(session.config, "workeroutput", None)
    if workeroutput is not None:
        workeroutput[_WORKEROUTPUT_KEY] = sorted(" ".join(path) for path in _invoked)
        return
    _enforce(session)


def pytest_testnodedown(node: object, error: object) -> None:
    """Merge one finished xdist worker's recorded commands on the controller."""
    output = getattr(node, "workeroutput", None) or {}
    _invoked.update(tuple(entry.split(" ")) for entry in output.get(_WORKEROUTPUT_KEY, ()))


def _enforce(session: pytest.Session) -> None:
    """Fail the session listing every leaf command no e2e test invoked."""
    if not gate_enabled(session.config):
        return
    if session.testsfailed or session.shouldstop or session.shouldfail:
        # An incomplete run proves nothing about which commands were reachable.
        return
    missing = sorted(leaf_commands() - _invoked - set(EXCLUDED_COMMANDS))
    if not missing:
        return
    reporter = session.config.pluginmanager.get_plugin("terminalreporter")
    if reporter is not None:
        reporter.write_sep("=", "commands with no e2e coverage")
        for path in missing:
            reporter.write_line("  agm " + " ".join(path))
        reporter.write_line(f"{len(missing)} command(s) were never invoked through the agm binary")
    session.exitstatus = pytest.ExitCode.TESTS_FAILED
