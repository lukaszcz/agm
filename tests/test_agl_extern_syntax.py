"""Tests for `extern def` syntax, AST, and scope resolution.

Covers everything from source text to a resolved AST for `extern def`:
- lexer: `extern` is a fully reserved keyword.
- grammar/transformer: signature forms, `private extern def`, rejected forms
  (missing return type, body, combination with `builtin`).
- scope: extern participates in the top-level function pre-pass exactly like
  an ordinary `def` (mutual recursion, export maps, reserved-name guard,
  root-only placement).
- companion names: `@extern-name` supplies the Python companion name and the
  declared name is used verbatim without it; the effective name must be a
  non-keyword Python identifier, and no two externs of one module may map to
  the same companion name.
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
from agm.agl.scope import AglScopeError, ModuleResolution
from agm.agl.scope.program import resolve_program
from agm.agl.syntax.nodes import FuncDef
from agm.agl.syntax.visitor import walk
from tests.agl.ir_harness import make_graph_from_files, write_companion_file
from tests.agl.module_graph import resolve_program_ast

_STDLIB_ROOT = Path(__file__).resolve().parents[1] / "stdlib"
_SCOPE_REJECTIONS_DIR = Path(__file__).resolve().parent / "agl" / "rejections" / "scope"


def first(source: str) -> object:
    """Parse *source* and return its first top-level item."""
    return parse_program(source.strip()).body.items[0]


def parse_and_resolve(source: str, *, origin_path: Path | None = None) -> object:
    """Parse *source* and run scope resolution, threading *origin_path*.

    Uses ``resolve_program_ast``'s hand-built single-module graph rather than
    a real loaded one: ``TestPlacement``/``TestScope`` use this with a
    virtual, non-existent *origin_path* to test the scope resolver's own
    file-backed-origin rule in isolation from the module loader's separate
    (already graph-tested, see the ``TestScope``/``TestPlacement`` cases
    below that call ``make_graph_from_files``/``load_graph`` directly)
    missing-companion-file check. Building a real loaded graph here would hit
    that loader check first and mask the one this helper means to test.
    """
    return resolve_program_ast(parse_program(source.strip()), origin_path=origin_path)


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
        fd = first("extern def reverse[T](xs: array[T]) -> array[T]")
        assert isinstance(fd, FuncDef)
        assert fd.type_params == ("T",)
        assert fd.is_extern is True

    def test_extern_def_with_zone_attributes_and_defaults(self) -> None:
        fd = first("extern def f(@arg-pos a: int, b: int, @arg-named c: int = 1) -> int")
        assert isinstance(fd, FuncDef)
        assert [tuple(attribute.name for attribute in p.attributes) for p in fd.params] == [
            ("arg-pos",),
            (),
            ("arg-named",),
        ]
        assert fd.params[2].default is not None

    def test_extern_name_attribute_on_the_line_above(self) -> None:
        fd = first('@extern-name("first_option")\nextern def first?(xs: array[int]) -> int')
        assert isinstance(fd, FuncDef)
        assert fd.is_extern is True
        assert fd.name == "first?"
        assert [attribute.name for attribute in fd.attributes] == ["extern-name"]

    def test_extern_name_attribute_on_the_same_line(self) -> None:
        fd = first('@extern-name("helper") extern def do-it!(x: int) -> int')
        assert isinstance(fd, FuncDef)
        assert fd.is_extern is True
        assert [attribute.name for attribute in fd.attributes] == ["extern-name"]

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
        write_companion_file(tmp_path / "root", "lib/mod", "def f(x):\n    return x\n")
        graph = make_graph_from_files(
            tmp_path,
            {
                "entry": "import lib/mod::*\nlib/mod::f(1)",
                "lib/mod": "extern def f(x: int) -> int",
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
                "lib/mod": "private extern def secret(x: int) -> int",
            },
        )
        resolved = resolve_program(graph)
        assert all(name != "secret" for (_mid, name) in resolved.all_public_funcs)


# ---------------------------------------------------------------------------
# Companion names
# ---------------------------------------------------------------------------


class TestCompanionName:
    """``@extern-name`` and the Python-identifier rule scope applies to it."""

    _PATH = Path("/virtual/mod.agl")

    def _companion_names(self, source: str) -> dict[str, str]:
        """Return every extern's declared name mapped to its companion name."""
        resolved = parse_and_resolve(source, origin_path=self._PATH)
        assert isinstance(resolved, ModuleResolution)
        externs: list[FuncDef] = []

        def visit(node: object) -> None:
            if isinstance(node, FuncDef) and node.is_extern:
                externs.append(node)

        walk(resolved.program, visit)
        return {fd.name: resolved.attributes.extern_names[fd.node_id] for fd in externs}

    def test_an_unattributed_extern_uses_its_declared_name(self) -> None:
        assert self._companion_names("extern def helper(x: int) -> int") == {"helper": "helper"}

    def test_the_attribute_supplies_the_companion_name(self) -> None:
        names = self._companion_names(
            '@extern-name("first_option")\nextern def first?(x: int) -> int'
        )
        assert names == {"first?": "first_option"}

    def test_the_companion_name_is_found_past_an_unrelated_attribute(self) -> None:
        names = self._companion_names(
            '@doc("the first element")\n@extern-name("first_option")\n'
            "extern def first?(x: int) -> int"
        )
        assert names == {"first?": "first_option"}

    @pytest.mark.parametrize("name", ["do-it!", "valid?", "my-func"])
    def test_a_non_python_identifier_needs_the_attribute(self, name: str) -> None:
        err = reject_scope(f"extern def {name}(x: int) -> int", origin_path=self._PATH)
        assert "identifier" in str(err).lower()
        assert "extern-name" in str(err)

    @pytest.mark.parametrize("name", ["class", "import", "lambda", "global"])
    def test_a_python_keyword_needs_the_attribute(self, name: str) -> None:
        err = reject_scope(f"extern def {name}(x: int) -> int", origin_path=self._PATH)
        assert "keyword" in str(err).lower()
        assert "extern-name" in str(err)

    def test_the_attribute_rescues_a_name_python_cannot_define(self) -> None:
        names = self._companion_names(
            '@extern-name("do_it")\nextern def do-it!(x: int) -> int',
        )
        assert names == {"do-it!": "do_it"}

    def test_the_supplied_name_is_itself_checked(self) -> None:
        err = reject_scope(
            '@extern-name("do-it")\nextern def helper(x: int) -> int', origin_path=self._PATH
        )
        assert "identifier" in str(err).lower()

    def test_a_python_soft_keyword_name_is_accepted(self) -> None:
        assert self._companion_names("extern def match(x: int) -> int") == {"match": "match"}

    def test_a_dunder_style_name_is_accepted(self) -> None:
        assert self._companion_names("extern def __init__(x: int) -> int") == {
            "__init__": "__init__"
        }

    def test_the_attribute_is_rejected_on_an_ordinary_def(self) -> None:
        err = reject_scope('@extern-name("helper")\ndef helper() -> int = 1')
        assert "extern-name" in str(err)

    def test_scoped_externs_may_share_a_declared_name(self) -> None:
        source = (
            "scope A\n"
            '  @extern-name("a_size")\n'
            "  extern def size(x: int) -> int\n"
            "end A\n"
            "\n"
            "scope B\n"
            '  @extern-name("b_size")\n'
            "  extern def size(x: int) -> int\n"
            "end B\n"
        )
        resolved = parse_and_resolve(source, origin_path=self._PATH)
        assert isinstance(resolved, ModuleResolution)
        assert sorted(resolved.attributes.extern_names.values()) == ["a_size", "b_size"]

    def test_bad_python_name_rejection_fixture(self) -> None:
        # ``tests/test_agl_e2e.py`` resolves every rejection fixture inline, so
        # an ``extern def`` in one is caught by the file-backing rule before the
        # rule the fixture demonstrates. Giving it a file-backed origin here
        # exercises the intended failure.
        source = (_SCOPE_REJECTIONS_DIR / "extern_bad_python_name.agl").read_text(encoding="utf-8")
        err = reject_scope(source, origin_path=self._PATH)
        assert "identifier" in str(err).lower()

    def test_scoped_externs_sharing_a_companion_name_are_rejected(self) -> None:
        source = (
            "scope A\n"
            '  @extern-name("shared")\n'
            "  extern def first(x: int) -> int\n"
            "end A\n"
            "\n"
            "scope B\n"
            '  @extern-name("shared")\n'
            "  extern def second(x: int) -> int\n"
            "end B\n"
        )
        err = reject_scope(source, origin_path=self._PATH)
        assert "shared" in str(err)
        assert "first" in str(err)
        assert "second" in str(err)

    def test_root_externs_sharing_a_companion_name_are_rejected(self) -> None:
        source = (
            '@extern-name("shared")\n'
            "extern def alpha(x: int) -> int\n"
            '@extern-name("shared")\n'
            "extern def beta(x: int) -> int\n"
        )
        err = reject_scope(source, origin_path=self._PATH)
        assert "shared" in str(err)
        assert "alpha" in str(err)
        assert "beta" in str(err)

    def test_a_root_and_a_scoped_extern_may_not_share_a_companion_name(self) -> None:
        source = (
            "extern def shared(x: int) -> int\n"
            "\n"
            "scope A\n"
            '  @extern-name("shared")\n'
            "  extern def helper(x: int) -> int\n"
            "end A\n"
        )
        err = reject_scope(source, origin_path=self._PATH)
        assert "shared" in str(err)
        assert "helper" in str(err)

    @pytest.mark.parametrize("argument", ["", "1"], ids=("empty", "non-text"))
    def test_a_malformed_attribute_is_reported_before_a_collision(self, argument: str) -> None:
        source = (
            "scope A\n"
            f"  @extern-name({argument})\n"
            "  extern def f(x: int) -> int\n"
            "end A\n"
            "\n"
            "scope B\n"
            "  extern def f(x: int) -> int\n"
            "end B\n"
        )
        err = reject_scope(source, origin_path=self._PATH)
        assert "extern-name" in str(err)
        assert "companion" not in str(err)


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
            "extern def f(x: int) -> int",
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
        write_companion_file(tmp_path / "root", "lib/mod", "def f(x):\n    return x\n")
        graph = make_graph_from_files(
            tmp_path,
            {
                "entry": "import lib/mod::*\nlib/mod::f(1)",
                "lib/mod": "extern def f(x: int) -> int",
            },
        )
        resolved = resolve_program(graph)
        assert resolved is not None
