"""Parity tests for qualified constructor owners in expressions and patterns."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import pytest

from agm.agl.modules.ids import ModuleId
from agm.agl.scope import AglScopeError
from agm.agl.scope.imports import (
    ImportEnv,
    QualResolutionFound,
    QualResolutionMissingMember,
    SingleTarget,
    build_import_env,
    resolve_qualified,
)
from agm.agl.scope.program import resolve_program
from agm.agl.scope.symbols import (
    TypeArgumentsError,
    UnknownMemberError,
    UnknownQualifierError,
)
from agm.agl.syntax.nodes import ImportDecl, ImportItem, QualifierChain, QualifierSegment
from agm.agl.syntax.spans import UNKNOWN_SOURCE, SourceSpan
from agm.agl.typecheck import AglTypeError
from agm.agl.typecheck.program import check_program
from tests.agl.ir_harness import base_caps, make_graph_from_files
from tests.agl.module_graph import build_inline_entry_graph, resolve_inline_entry
from tests.agl.qualifier_support import file_verdict, graph_verdict

Outcome = Literal["accepted", "scope", "typecheck"]


def _import_env(
    module_path: str,
    public_atoms: tuple[str | tuple[str, ...], ...],
    *,
    tail: tuple[ImportItem, ...] | None = None,
    hidden: tuple[ImportItem, ...] = (),
) -> ImportEnv:
    span = SourceSpan(1, 1, 1, 1, 0, 0, UNKNOWN_SOURCE)
    module = ModuleId.from_path(module_path)
    decl = ImportDecl(
        module_path=module.segments,
        wildcard=False,
        alias=None,
        tail=tail,
        hidden=hidden,
        span=span,
        node_id=1,
    )
    return build_import_env(
        (decl,),
        {decl.node_id: SingleTarget(module)},
        {module: {atom: (module, atom) for atom in public_atoms}},
        {module: {}},
    )


def _item(name: str) -> ImportItem:
    span = SourceSpan(1, 1, 1, 1, 0, 0, UNKNOWN_SOURCE)
    return ImportItem(name, None, span, 2)


def _qualifier(*segments: str, member: str = "") -> QualifierChain:
    span = SourceSpan(1, 1, 1, 1, 0, 0, UNKNOWN_SOURCE)
    return QualifierChain(
        anchor=None,
        segments=tuple(
            QualifierSegment(name=name, type_args=None, span=span, node_id=index)
            for index, name in enumerate(segments)
        ),
        member=member,
        span=span,
        node_id=1,
    )


def _module_outcome(source: str) -> Outcome:
    """A phase-accurate verdict: which phase, if any, first rejects *source*.

    *source* is a tiny inline snippet; its graph is built through
    :func:`~tests.agl.module_graph.build_inline_entry_graph` and classified
    by :func:`~tests.agl.qualifier_support.graph_verdict`.
    """
    graph, _import_node_id = build_inline_entry_graph(source)
    phase, _cls, _span, _identity = graph_verdict(graph)
    if phase == "matchcompile":
        raise AssertionError(f"unexpected matchcompile phase for {source!r}")
    return phase


def _program_outcome(tmp_path: Path, modules: dict[str, str]) -> Outcome:
    """A phase-accurate verdict: which call raised, not which class."""
    phase, _cls, _span, _identity = file_verdict(tmp_path, modules)
    if phase == "matchcompile":
        raise AssertionError(f"unexpected matchcompile phase for {modules!r}")
    return phase


@pytest.mark.parametrize(
    ("expression", "pattern", "expression_outcome", "pattern_outcome"),
    [
        (
            "enum Color\n  | Red\n::Color::Red",
            "enum Color\n  | Red\nlet value: Color = Color::Red\ncase value of | ::Color::Red => 1",
            "accepted",
            "accepted",
        ),
        (
            "enum Color\n  | Red\nNope::Red",
            "enum Color\n  | Red\nlet value: Color = Color::Red\n"
            "case value of | Nope::Red => 1 | _ => 2",
            "scope",
            "scope",
        ),
        (
            "enum Color\n  | Red\n::Missing::Red",
            (
                "enum Color\n  | Red\nlet value: Color = Color::Red\n"
                "case value of | ::Missing::Red => 1 | _ => 2"
            ),
            "scope",
            "scope",
        ),
        (
            "enum Color\n  | Red\nColor::Gone",
            (
                "enum Color\n  | Red\nlet value: Color = Color::Red\n"
                "case value of | Color::Gone => 1 | _ => 2"
            ),
            "scope",
            "scope",
        ),
    ],
)
def test_expression_and_pattern_qualifier_verdicts_remain_in_parity(
    expression: str,
    pattern: str,
    expression_outcome: Outcome,
    pattern_outcome: Outcome,
) -> None:
    assert _module_outcome(expression) == expression_outcome
    assert _module_outcome(pattern) == pattern_outcome


def test_own_type_beats_module_route_in_both_positions(tmp_path: Path) -> None:
    expression = {
        "entry": "import pkg/Foo\nenum Foo\n  | local\nFoo::local",
        "pkg/Foo": "def local() -> int = 1",
    }
    pattern = {
        "entry": (
            "import pkg/Foo\n"
            "enum Foo\n"
            "  | local\n"
            "let value: Foo = local\n"
            "case value of | Foo::local => 1"
        ),
        "pkg/Foo": "def local() -> int = 1",
    }

    assert _program_outcome(tmp_path / "expression", expression) == "accepted"
    assert _program_outcome(tmp_path / "pattern", pattern) == "accepted"


@pytest.mark.parametrize(
    ("annotation", "member"),
    (
        ("types::Color", "enum Color = Red | Green"),
        ("types::Box[int]", "enum Box[T] = Box(value: T)"),
    ),
    ids=("bare-owner", "applied-owner"),
)
def test_type_parameter_shadowing_a_real_module_route_is_rejected(
    tmp_path: Path, annotation: str, member: str
) -> None:
    """A type parameter's name can coincide with an actually-imported module's.

    ``def f[types](x: types::Color)`` names the type parameter, not the
    ``types`` module, exactly as a type parameter shadowing a purely local
    name does -- the module import makes no difference to the verdict.
    """
    modules = {
        "entry": f"import types\ndef f[types](x: {annotation}) -> int = 1",
        "types": member,
    }

    assert _program_outcome(tmp_path, modules) == "scope"


@pytest.mark.parametrize(
    "entry",
    (
        "import one/types\ndef f[one](x: one/types::Color) -> int = 1",
        "import one/types\ndef f[one](x: int) -> int =\n  let c = one/types::Color::Red\n  1",
        "import one/types\ndef f[one](x: int) = one/types::Color::Red",
    ),
    ids=("annotation", "value", "one-liner"),
)
def test_type_parameter_never_shadows_a_slash_module_route(tmp_path: Path, entry: str) -> None:
    """A ``/``-containing qualifier segment always names a module route.

    ``one/types::Color`` spells the imported ``one/types`` module's route, not
    the type parameter ``one`` -- a route segment can only ever coincide with a
    type parameter's bare name, never with its own ``/``-joined spelling, so
    the type-parameter shadowing rule must not reject it.
    """
    modules = {
        "entry": entry,
        "one/types": "enum Color\n  | Red",
    }

    assert _program_outcome(tmp_path, modules) == "accepted"


def _needle_span(entry: str, locate: str, span_len: int) -> tuple[int, int, int]:
    """Compute (start_line, start_col, end_col) for *locate*'s last occurrence in *entry*.

    The reported span covers only *locate*'s first *span_len* characters --
    a scope row's shadowed segment is shorter than the qualifier locating it.
    """
    offset = entry.rindex(locate)
    prefix = entry[:offset]
    line = prefix.count("\n") + 1
    col = offset - prefix.rfind("\n")
    return line, col, col + span_len


@pytest.mark.parametrize(
    ("entry", "phase", "locate"),
    (
        (
            "import types\ndef f[types](x: int) = fn(y: types::Color) => 1",
            "scope",
            "types::Color",
        ),
        (
            "import types\ndef S::f[types](x: int) = fn(y: types::Color) => 1",
            "scope",
            "types::Color",
        ),
        (
            "import types\n\nscope S\n  def f[types](x: int) = fn(y: types::Color) => 1\nend S",
            "scope",
            "types::Color",
        ),
        (
            "import types\ndef f[types](x: int) -> int = types::Color::Red",
            "scope",
            "types::Color::Red",
        ),
        (
            "import types\nrecord Box\n  v: int\n"
            "def Box::m[types](self) = fn(y: types::Color) => 1",
            "scope",
            "types::Color",
        ),
        (
            "import lib\ndef f(x: lib::SlotA[int]) -> int = "
            "case x of | lib::SlotA[text]::FilledA(value) => 1 | _ => 0",
            "typecheck",
            "lib::SlotA[text]::FilledA(value)",
        ),
        (
            "import lib\nprogram def main(x: lib::SlotA[int]) -> unit = "
            "print(case x of | lib::SlotA[text]::FilledA(value) => 1 | _ => 0)",
            "typecheck",
            "lib::SlotA[text]::FilledA(value)",
        ),
    ),
    ids=(
        "plain-def",
        "scoped-shorthand-def",
        "scope-region-def",
        "value-position-return",
        "method-def",
        "plain-def-case",
        "program-def-case",
    ),
)
def test_a_one_liner_def_body_validates_its_qualifier_chains(
    tmp_path: Path, entry: str, phase: Outcome, locate: str
) -> None:
    """A ``def`` whose body is a bare expression, never a ``Block``, still
    validates every qualifier chain reachable only through that body: a
    lambda parameter's type-parameter shadowing, a bare value chain, and an
    ill-typed constructor pattern -- across a plain, scoped-shorthand,
    scope-region, method, and ``program`` ``def``.

    A scope-phase row's precise class is :class:`UnknownQualifierError`, at
    the shadowed qualifier's span; a typecheck-phase row's is
    :class:`AglTypeError`, at the whole rejected pattern's span.
    """
    modules = {
        "entry": entry,
        "types": "enum Color\n  | Red\n  | Green",
        "lib": "enum SlotA[T]\n  | FilledA(value: T)\n  | EmptyA",
    }
    graph = make_graph_from_files(tmp_path, modules)
    expected_line, expected_start_col, expected_end_col = _needle_span(entry, locate, len(locate))

    if phase == "scope":
        with pytest.raises(UnknownQualifierError) as scope_excinfo:
            resolve_program(graph)
        span = scope_excinfo.value.span
    else:
        resolved = resolve_program(graph)
        with pytest.raises(AglTypeError) as type_excinfo:
            check_program(resolved, base_caps())
        span = type_excinfo.value.span

    assert span is not None
    assert (span.start_line, span.start_col, span.end_col) == (
        expected_line,
        expected_start_col,
        expected_end_col,
    )


_NESTED_SCOPE_MODULE = (
    "scope S\n  enum Color | Blue | Red\n\n  scope P\n    record Q\n  end P\nend S\n"
)


_REGION_PREFIX = "scope S\n  enum Color | Blue | Red\n\n  scope P\n    record Q\n  end P\n\n"


@pytest.mark.parametrize("binder", ("let", "var"))
@pytest.mark.parametrize(
    ("annotation", "value", "error", "needle"),
    (
        ("P[int]::Q", "P::Q", TypeArgumentsError, "P[int]"),
        ("Color::Nope", "Color::Blue", UnknownMemberError, "Color::Nope"),
    ),
    ids=("type-args-on-scope-segment", "unknown-enum-member"),
)
def test_scoped_binder_shorthand_matches_its_region_form(
    binder: str, annotation: str, value: str, error: type[AglScopeError], needle: str
) -> None:
    """A root ``let S::x`` or ``var S::x`` binder validates its annotation inside ``S``,
    exactly as ``scope S ... let x ... end S`` does: same phase, same precise class, and
    a span over the same rejected sub-expression -- not the whole annotation or binder.
    """
    shorthand = f"{_NESTED_SCOPE_MODULE}{binder} S::x: {annotation} = {value}\n"
    region = f"{_REGION_PREFIX}  {binder} x: {annotation} = {value}\nend S\n"
    for source in (shorthand, region):
        with pytest.raises(AglScopeError) as excinfo:
            resolve_inline_entry(source)
        assert type(excinfo.value) is error
        span = excinfo.value.span
        assert span is not None
        assert (span.start_line, span.start_col, span.end_col) == _needle_span(
            source, needle, len(needle)
        )


def test_import_tail_keeps_an_unselected_qualified_owner_reachable() -> None:
    module = ModuleId.from_path("Pal")
    env = _import_env("Pal", ("public", ("Secret", "hidden")), tail=(_item("public"),))

    assert resolve_qualified(env, ("Pal",), ("Secret", "hidden")) == QualResolutionFound(
        module, (module, ("Secret", "hidden"))
    )


def test_explicit_owner_matching_the_route_segment_is_rejected(tmp_path: Path) -> None:
    """A route segment sharing its spelling with a bogus explicit owner is rejected.

    ``pal::pal::Red`` spells the module route ``pal`` and an explicit (wrong)
    owner ``pal`` before ``Red``. The route segment happens to share its
    spelling with the explicit owner, but that coincidence must not make the
    route silently contribute the member itself as though no owner had been
    spelled — the qualifier is a bogus owner and must be rejected.
    """
    modules = {
        "entry": (
            "import pal\nlet value: pal::Color = pal::Color::Red\n"
            "case value of | pal::pal::Red => 1 | _ => 2"
        ),
        "pal": "enum Color\n  | Red",
    }

    assert _program_outcome(tmp_path, modules) == "scope"


def test_correctly_spelled_module_and_owner_route_still_resolves(tmp_path: Path) -> None:
    """The correct spelling ``pal::Color::Red`` keeps resolving after the fix."""
    modules = {
        "entry": (
            "import pal\nlet value: pal::Color = pal::Color::Red\n"
            "case value of | pal::Color::Red => 1 | _ => 2"
        ),
        "pal": "enum Color\n  | Red\n  | Green",
    }

    assert _program_outcome(tmp_path, modules) == "accepted"


def test_import_hiding_removes_a_qualified_owner_at_the_route_seam() -> None:
    env = _import_env("Pal", ("public", ("Secret", "hidden")), hidden=(_item("Secret"),))

    assert isinstance(
        resolve_qualified(env, ("Pal",), ("Secret", "hidden")), QualResolutionMissingMember
    )
