"""Tests for AGM command layout and enforced e2e command coverage.

The key test here is ``test_e2e_command_coverage``, which walks the live Typer
command tree, canonicalizes aliases, and asserts that every resulting leaf
command is exercised by at least one ``run_agm([...])`` invocation in the e2e
test files.  Adding a new leaf command without a matching e2e invocation (or a
correct alias entry) causes the test to fail — making the "100% e2e command
coverage" guarantee enforced rather than aspirational.
"""

from __future__ import annotations

import ast
from collections import defaultdict
from pathlib import Path
from typing import cast

from click.core import Command, Context, Group
from typer.main import Typer as MainTyper
from typer.main import get_command

from agm.cli import app

# ---------------------------------------------------------------------------
# Command-level spelling alias map
# ---------------------------------------------------------------------------

# Maps alias leaf-path → canonical leaf-path for commands that are registered
# under two names with identical behaviour.  After group-alias canonicalization
# (wsp→workspace, wt→worktree), every KEY here must still be a live leaf path
# in the Click tree — the guard in test_e2e_command_coverage enforces this so
# the map cannot silently rot when commands change.
COMMAND_ALIASES: dict[tuple[str, ...], tuple[str, ...]] = {
    # "config cp" is the abbreviated form; "config copy" is canonical.
    ("config", "cp"): ("config", "copy"),
    # "dep rm" is abbreviated; "dep remove" is canonical.
    ("dep", "rm"): ("dep", "remove"),
    # "worktree rm" is abbreviated; "worktree remove" is canonical.
    ("worktree", "rm"): ("worktree", "remove"),
}


# ---------------------------------------------------------------------------
# Click-tree walkers
# ---------------------------------------------------------------------------


def _collect_leaf_paths(cmd: Command, path: tuple[str, ...] = ()) -> list[tuple[str, ...]]:
    """Recursively collect every leaf-command path reachable from *cmd*.

    Groups are detected by the presence of ``list_commands`` and
    ``get_command`` methods (both ``click.Group`` and ``click.MultiCommand``
    carry them).  ``cast`` narrows the type without triggering mypy's
    ``disallow_any_expr`` rule, which fires on any ``isinstance`` check
    against a click type that has ``**kwargs: Any`` in its constructor.
    """
    ctx = Context(cmd)
    if hasattr(cmd, "list_commands") and hasattr(cmd, "get_command"):
        grp = cast(Group, cmd)
        result: list[tuple[str, ...]] = []
        for name in grp.list_commands(ctx):
            subcmd = grp.get_command(ctx, name)
            if subcmd is not None:
                result.extend(_collect_leaf_paths(subcmd, path + (name,)))
        return result
    return [path]


def _group_signature(group: Group) -> frozenset[tuple[str, ...]]:
    """Return the full recursive leaf-structure of *group* as its alias signature.

    Two child groups are aliases (the same Typer sub-app registered under two
    names) iff they expose the identical set of descendant leaf sub-paths.  We
    compare the whole recursive structure rather than just immediate child names
    so two genuinely-distinct groups that happened to share top-level names but
    differ deeper are not falsely merged.
    """
    return frozenset(_collect_leaf_paths(group))


def _detect_group_aliases(root: Group) -> dict[str, str]:
    """Return ``{alias_group_name: canonical_group_name}`` for *root*'s children.

    Detection: child groups with identical recursive leaf-structure were built
    from the same Typer sub-app and are therefore aliases of one another.  The
    canonical name is the alphabetically-first among the alias set — a
    deterministic, otherwise-arbitrary representative; canonicalization is
    applied consistently to both the leaf set and the scanned invocations, so
    the gate is correct regardless of which spelling wins.
    """
    sigs: dict[frozenset[tuple[str, ...]], list[str]] = defaultdict(list)
    ctx = Context(root)
    for name in root.list_commands(ctx):
        cmd = root.get_command(ctx, name)
        if cmd is not None and hasattr(cmd, "list_commands") and hasattr(cmd, "get_command"):
            grp = cast(Group, cmd)
            sigs[_group_signature(grp)].append(name)

    alias_to_canonical: dict[str, str] = {}
    for names in sigs.values():
        if len(names) > 1:
            canonical = min(names)  # deterministic, otherwise-arbitrary representative
            for name in names:
                if name != canonical:
                    alias_to_canonical[name] = canonical
    return alias_to_canonical


def _canonicalize_path(
    path: tuple[str, ...],
    *,
    group_aliases: dict[str, str],
    command_aliases: dict[tuple[str, ...], tuple[str, ...]],
) -> tuple[str, ...]:
    """Apply group-alias then command-alias canonicalization to *path*."""
    # Step 1: replace alias group prefix with its canonical group name.
    if path and path[0] in group_aliases:
        path = (group_aliases[path[0]],) + path[1:]
    # Step 2: replace a whole-path alias with its canonical path.
    return command_aliases.get(path, path)


def _canonicalize_paths(
    paths: list[tuple[str, ...]],
    *,
    group_aliases: dict[str, str],
    command_aliases: dict[tuple[str, ...], tuple[str, ...]],
) -> set[tuple[str, ...]]:
    """Return the set of canonical paths for all entries in *paths*."""
    return {
        _canonicalize_path(p, group_aliases=group_aliases, command_aliases=command_aliases)
        for p in paths
    }


# ---------------------------------------------------------------------------
# e2e scanner
# ---------------------------------------------------------------------------


def _extract_covered_canonical_paths(
    filepath: Path,
    raw_leaf_set: frozenset[tuple[str, ...]],
    *,
    group_aliases: dict[str, str],
    command_aliases: dict[tuple[str, ...], tuple[str, ...]],
) -> set[tuple[str, ...]]:
    """Return canonical leaf paths covered by ``run_agm([...])`` calls in *filepath*.

    Parses the file with the ``ast`` module, finds every ``run_agm([...])``
    call whose first argument is a list of string literals, and does a
    longest-prefix match (depth 2 then depth 1) against *raw_leaf_set* to
    extract the command path.  The matched path is then canonicalized.
    """
    source = filepath.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(filepath))

    covered: set[tuple[str, ...]] = set()
    for node in ast.walk(tree):
        if not (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "run_agm"
            and node.args
            and isinstance(node.args[0], ast.List)
        ):
            continue
        tokens: list[str] = []
        for elt in node.args[0].elts:
            if isinstance(elt, ast.Constant) and isinstance(elt.value, str):
                tokens.append(elt.value)
            else:
                break  # stop at the first non-string-literal element
        if not tokens:
            continue
        # Longest-prefix match against the raw leaf set (depth 2, then 1).
        matched: tuple[str, ...] | None = None
        for depth in (2, 1):
            prefix = tuple(tokens[:depth])
            if prefix in raw_leaf_set:
                matched = prefix
                break
        if matched is None:
            continue
        covered.add(
            _canonicalize_path(
                matched, group_aliases=group_aliases, command_aliases=command_aliases
            )
        )
    return covered


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_command_subpackages_match_cli_command_groups() -> None:
    """commands/ sub-packages must mirror CLI command groups (repository guideline)."""
    commands_dir = Path(__file__).resolve().parents[1] / "src" / "agm" / "commands"
    subpackages = {
        path.name
        for path in commands_dir.iterdir()
        if path.is_dir() and not path.name.startswith("__")
    }

    assert subpackages == {"config", "dep", "loop", "sync", "tmux", "workspace", "worktree"}


def test_e2e_command_coverage() -> None:
    """Every live leaf command must be exercised by at least one run_agm([...]) call.

    Alias canonicalization:
      - Group aliases (``wsp/*`` → ``workspace/*``, ``wt/*`` → ``worktree/*``) are
        detected automatically: two child groups at the same level with identical
        child command name sets are aliases; the alphabetically-first name is
        canonical.
      - Command-level spelling aliases (``config cp`` → ``config copy``, etc.) are
        declared in ``COMMAND_ALIASES``; the guard assertion below ensures that map
        cannot silently rot when commands change.

    NOTE: top-level ``open``/``close`` are distinct entry points from
    ``workspace open``/``workspace close`` — both must be covered.
    """
    # cast: the local stub types typer.Typer differently from typer.main.Typer,
    # but they are the same class at runtime.
    root_cmd: Command = get_command(cast(MainTyper, app))
    assert hasattr(root_cmd, "list_commands") and hasattr(root_cmd, "get_command"), (
        "agm CLI root must be a Click group"
    )
    root = cast(Group, root_cmd)

    # 1. Collect every raw leaf path from the live Click tree.
    all_raw_paths = _collect_leaf_paths(root)
    raw_leaf_set: frozenset[tuple[str, ...]] = frozenset(all_raw_paths)

    # 2. Detect group aliases (e.g. wsp ↔ workspace, wt ↔ worktree).
    group_aliases = _detect_group_aliases(root)

    # 3. Compute the group-canonicalized leaf set (command aliases not yet applied).
    group_canonical_paths = _canonicalize_paths(
        all_raw_paths, group_aliases=group_aliases, command_aliases={}
    )

    # 4. Guard: every COMMAND_ALIASES key and value must be a real live leaf path
    #    in the group-canonicalized set so the map cannot silently rot.
    for alias_path, canonical_path in COMMAND_ALIASES.items():
        assert alias_path in group_canonical_paths, (
            f"COMMAND_ALIASES key {alias_path!r} is not a live leaf path after group "
            "canonicalization; remove or update the entry"
        )
        assert canonical_path in group_canonical_paths, (
            f"COMMAND_ALIASES value {canonical_path!r} is not a live leaf path after group "
            "canonicalization; remove or update the entry"
        )

    # 5. Compute the fully canonical leaf set (both alias steps applied).
    canonical_leaf_set = _canonicalize_paths(
        all_raw_paths, group_aliases=group_aliases, command_aliases=COMMAND_ALIASES
    )

    # 6. Scan e2e test files for run_agm([...]) invocations.
    tests_dir = Path(__file__).resolve().parent
    e2e_files = [
        tests_dir / "test_e2e.py",
        tests_dir / "test_agl_e2e.py",
    ]
    covered: set[tuple[str, ...]] = set()
    for e2e_file in e2e_files:
        if e2e_file.exists():
            covered |= _extract_covered_canonical_paths(
                e2e_file,
                raw_leaf_set,
                group_aliases=group_aliases,
                command_aliases=COMMAND_ALIASES,
            )

    # 7. Assert every canonical leaf is covered.
    missing = canonical_leaf_set - covered
    assert not missing, (
        "The following commands lack e2e coverage (no run_agm([...]) invocation found):\n"
        + "\n".join(f"  agm {' '.join(p)}" for p in sorted(missing))
    )
