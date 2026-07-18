"""Tests for `extern def` syntax, AST, and scope resolution.

Covers everything from source text to a resolved AST for `extern def`:
- lexer: `extern` is a fully reserved keyword.
- grammar/transformer: signature forms, `private extern def`, rejected forms
  (missing return type, body, combination with `builtin`).
- scope: extern participates in the top-level function pre-pass exactly like
  an ordinary `def` (mutual recursion, export maps, reserved-name guard,
  root-only placement).
- placement: `extern def` requires a file-backed module (`origin_path`).

NO typecheck, lowering, or runtime behavior is exercised here — externs are
not executable yet.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from agm.agl.lexer import tokenize
from agm.agl.modules.loader import build_repl_graph, load_graph
from agm.agl.modules.roots import RootSet
from agm.agl.parser import AglSyntaxError, parse_program, parse_program_seeded
from agm.agl.scope import AglScopeError, resolve_module
from agm.agl.scope.program import resolve_program
from agm.agl.syntax.nodes import FuncDef
from tests.agl.ir_harness import make_graph_from_files, write_companion_file

_STDLIB_ROOT = Path(__file__).resolve().parents[1] / "stdlib"


def first(source: str) -> object:
    """Parse *source* and return its first top-level item."""
    return parse_program(source.strip()).body.items[0]


def parse_and_resolve(source: str, *, origin_path: Path | None = None) -> object:
    """Parse *source* and run scope resolution, threading *origin_path*."""
    return resolve_module(parse_program(source.strip()), origin_path=origin_path)


def reject_scope(source: str, *, origin_path: Path | None = None) -> AglScopeError:
    """Assert that *source* fails scope resolution and return the error."""
    with pytest.raises(AglScopeError) as exc_info:
        parse_and_resolve(source, origin_path=origin_path)
    return exc_info.value


# ---------------------------------------------------------------------------
# Lexer
# ---------------------------------------------------------------------------


class TestLexer:
    def test_extern_lexes_as_keyword_not_name(self) -> None:
        result = [(t.type, str(t)) for t in tokenize("extern def f")]
        assert result[0] == ("extern", "extern")
        assert ("NAME", "extern") not in result

    def test_extern_is_reserved_cannot_be_let_bound(self) -> None:
        with pytest.raises(AglSyntaxError):
            parse_program("let extern = 1")


# ---------------------------------------------------------------------------
# Grammar / transformer
# ---------------------------------------------------------------------------


class TestGrammarAndTransformer:
    def test_minimal_extern_def(self) -> None:
        fd = first("extern def f(x: int) -> int")
        assert isinstance(fd, FuncDef)
        assert fd.name == "f"
        assert fd.is_extern is True
        assert fd.is_builtin is False
        assert fd.body is None
        assert fd.return_type is not None
        assert len(fd.params) == 1

    def test_extern_def_with_type_params(self) -> None:
        fd = first("extern def reverse[T](xs: list[T]) -> list[T]")
        assert isinstance(fd, FuncDef)
        assert fd.type_params == ("T",)
        assert fd.is_extern is True

    def test_extern_def_with_zones_and_defaults(self) -> None:
        fd = first("extern def f(a: int, /, b: int, @named, c: int = 1) -> int")
        assert isinstance(fd, FuncDef)
        kinds = [p.kind.value for p in fd.params]
        assert kinds == ["positional_only", "standard", "named_only"]
        assert fd.params[2].default is not None

    def test_extern_def_with_named_only_star_zone(self) -> None:
        fd = first("extern def f(a: int, *, b: int) -> int")
        assert isinstance(fd, FuncDef)
        assert [p.kind.value for p in fd.params] == ["standard", "named_only"]

    def test_private_extern_def(self) -> None:
        fd = first("private extern def f(x: int) -> int")
        assert isinstance(fd, FuncDef)
        assert fd.is_private is True
        assert fd.is_extern is True
        assert fd.body is None

    def test_extern_modifier_on_its_own_line(self) -> None:
        fd = first("extern\ndef f(x: int) -> int")
        assert isinstance(fd, FuncDef)
        assert fd.is_extern is True

    def test_missing_return_type_rejected(self) -> None:
        with pytest.raises(AglSyntaxError):
            parse_program("extern def f(x: int)")

    def test_body_rejected(self) -> None:
        with pytest.raises(AglSyntaxError):
            parse_program("extern def f(x: int) -> int = x")

    def test_extern_combined_with_builtin_rejected(self) -> None:
        with pytest.raises(AglSyntaxError):
            parse_program("extern builtin def f(x: int) -> int")
        with pytest.raises(AglSyntaxError):
            parse_program("builtin extern def f(x: int) -> int")

    def test_private_func_def_still_carries_is_extern_false(self) -> None:
        # Regression: the shared private-wrap helper must not accidentally
        # flip is_extern on ordinary private defs.
        fd = first("private def f(x: int) -> int = x")
        assert isinstance(fd, FuncDef)
        assert fd.is_extern is False
        assert fd.is_private is True


# ---------------------------------------------------------------------------
# Scope
# ---------------------------------------------------------------------------


class TestScope:
    _PATH = Path("/virtual/mod.agl")

    def test_extern_collected_and_callable(self) -> None:
        resolved = parse_and_resolve("extern def f(x: int) -> int", origin_path=self._PATH)
        assert "f" in resolved.declared_functions

    def test_extern_forward_reference_mutual_recursion(self) -> None:
        source = "def f(x: int) -> int = helper(x)\nextern def helper(x: int) -> int\n"
        resolved = parse_and_resolve(source, origin_path=self._PATH)
        assert set(resolved.declared_functions) == {"f", "helper"}

    def test_extern_cannot_reuse_reserved_builtin_name(self) -> None:
        err = reject_scope("extern def print(x: int) -> int", origin_path=self._PATH)
        msg = str(err).lower()
        assert "built-in" in msg or "reserved" in msg

    def test_non_root_extern_def_rejected(self) -> None:
        source = "def f() -> int =\n  extern def g(x: int) -> int\n  g(1)\n"
        err = reject_scope(source, origin_path=self._PATH)
        assert "root" in str(err).lower()

    def test_extern_exported_from_module_graph(self, tmp_path: Path) -> None:
        write_companion_file(tmp_path / "root", "lib.mod", "def f(x):\n    return x\n")
        graph = make_graph_from_files(
            tmp_path,
            {
                "entry": "import lib.mod\nlib.mod::f(1)",
                "lib.mod": "extern def f(x: int) -> int",
            },
        )
        resolved = resolve_program(graph)
        assert any(name == "f" for (_mid, name) in resolved.all_public_funcs)

    def test_private_extern_not_exported(self, tmp_path: Path) -> None:
        # ``lib.mod`` is never imported by the entry, so it is never reached by
        # the loader's BFS and needs no companion file on disk.
        graph = make_graph_from_files(
            tmp_path,
            {
                "entry": "()",
                "lib.mod": "private extern def secret(x: int) -> int",
            },
        )
        resolved = resolve_program(graph)
        assert all(name != "secret" for (_mid, name) in resolved.all_public_funcs)


# ---------------------------------------------------------------------------
# Placement: extern def requires a file-backed module
# ---------------------------------------------------------------------------


class TestPlacement:
    def test_resolve_with_no_origin_path_rejects_extern(self) -> None:
        err = reject_scope("extern def f(x: int) -> int")
        assert "file-backed" in str(err).lower() or "extern" in str(err).lower()

    def test_resolve_with_origin_path_accepts_extern(self) -> None:
        resolved = parse_and_resolve(
            "extern def f(x: int) -> int", origin_path=Path("/virtual/mod.agl")
        )
        assert "f" in resolved.declared_functions

    def test_graph_resolution_of_file_backed_entry_accepts_extern(self, tmp_path: Path) -> None:
        entry_path = tmp_path / "entry.agl"
        (tmp_path / "entry.py").write_text("def f(x):\n    return x\n")
        graph = load_graph(
            "extern def f(x: int) -> int\n()",
            entry_path=entry_path,
            roots=RootSet(roots=frozenset()),
            default_stdlib=False,
        )
        resolved = resolve_program(graph)
        assert "f" in resolved.modules[graph.entry_id].resolved.declared_functions

    def test_graph_resolution_of_inline_entry_rejects_extern(self) -> None:
        graph = load_graph(
            "extern def f(x: int) -> int\n()",
            entry_path=None,
            roots=RootSet(roots=frozenset()),
            default_stdlib=False,
        )
        with pytest.raises(AglScopeError):
            resolve_program(graph)

    def test_repl_graph_entry_rejects_extern(self) -> None:
        program, next_id = parse_program_seeded("extern def f(x: int) -> int", start_id=0)
        graph, _next_id, _newly_loaded = build_repl_graph(
            program,
            next_id,
            path=None,
            cached={},
            roots=RootSet(roots=frozenset({_STDLIB_ROOT})),
        )
        with pytest.raises(AglScopeError):
            resolve_program(graph)

    def test_extern_in_library_module_accepts(self, tmp_path: Path) -> None:
        # A library module loaded from disk always carries a real path.
        write_companion_file(tmp_path / "root", "lib.mod", "def f(x):\n    return x\n")
        graph = make_graph_from_files(
            tmp_path,
            {
                "entry": "import lib.mod\nlib.mod::f(1)",
                "lib.mod": "extern def f(x: int) -> int",
            },
        )
        resolved = resolve_program(graph)
        assert resolved is not None
