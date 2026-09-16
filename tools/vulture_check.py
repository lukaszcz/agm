"""Report dead code in AGM's Python sources with vulture.

Vulture cannot see names that frameworks invoke by name. Lark calls
``AstBuilder`` methods by grammar rule and alias, so every rule and alias in the
grammar is whitelisted on each run, and ``EXTERNALLY_USED`` lists the rest.
Whitelisted names count as used wherever they occur in the scanned trees.
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
GRAMMAR = REPO_ROOT / "src/agm/agl/grammar/agl.lark"
SCANNED_PATHS = ("src/agm", "packages/stdlib/src", "tools")

EXTERNALLY_USED = (
    # pickle hooks
    "persistent_id",
    "reducer_override",
    "persistent_load",
    "find_class",
    # lark, prompt_toolkit, importlib, and click hooks
    "lex",
    "lex_document",
    "get_completions",
    "get_code",
    "value_from_envvar",
    "add_to_parser",
    "collect_usage_pieces",
    "format_options",
    # lazy module exports and protocol members
    "__getattr__",
    "__lt__",
    "__call__",
    # named only in string annotations
    "_AglNominalShape",
    # _DirectoryIdentity fields, read by equality
    "device",
    "inode",
    # test hooks
    "clear_retained_artifacts",
    "clear_parsed_module_cache",
    "set_self_validation_enabled",
)


def grammar_callback_names(grammar: str) -> set[str]:
    """Return every rule (``name:``, ``?name:``, ``!name:``) and alias (``-> name``) name."""
    names: set[str] = set()
    for line in grammar.splitlines():
        if line.lstrip().startswith("//"):
            continue
        head, colon, _ = line.partition(":")
        # A definition may carry a priority (``name.2``) or template parameters (``name{x}``).
        candidates = [head.lstrip("?!").split("{")[0].split(".")[0].rstrip()] if colon else []
        candidates += [alias.split()[0] for alias in line.split("->")[1:] if alias.split()]
        names.update(name for name in candidates if name.isidentifier() and name.islower())
    return names


def main() -> int:
    names = grammar_callback_names(GRAMMAR.read_text()) | set(EXTERNALLY_USED)
    with tempfile.TemporaryDirectory() as tmp:
        whitelist = Path(tmp) / "whitelist.py"
        whitelist.write_text("".join(f"_.{name}\n" for name in sorted(names)))
        command = [sys.executable, "-m", "vulture", *SCANNED_PATHS, str(whitelist)]
        return subprocess.run(command, cwd=REPO_ROOT, check=False).returncode


if __name__ == "__main__":
    sys.exit(main())
