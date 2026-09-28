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
from agm.agl.scope.symbols import AmbiguousQualificationError
from agm.agl.syntax.nodes import ImportDecl, ImportItem, QualifierChain, QualifierSegment
from agm.agl.syntax.spans import UNKNOWN_SOURCE, SourceSpan

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
    """A phase-accurate verdict: which of the two calls raised, not which class.

    Resolves *source* once for its own scope-phase verdict, then separately
    for the full type-checked verdict -- a second, independent parse and
    resolve of the same tiny inline snippet, since :func:`resolve_and_check_inline_entry`
    does not expose its intermediate resolution for reuse. The second call
    re-resolves scope deterministically over the same source the first call
    already resolved without raising, so it cannot raise ``AglScopeError``
    either; only ``AglTypeError`` is a genuine typecheck-phase verdict.
    """
    from agm.agl.typecheck import AglTypeError
    from tests.agl.ir_harness import base_caps
    from tests.agl.module_graph import resolve_and_check_inline_entry, resolve_inline_entry

    try:
        resolve_inline_entry(source)
    except AglScopeError:
        return "scope"
    try:
        resolve_and_check_inline_entry(source, base_caps())
    except AglTypeError:
        return "typecheck"
    return "accepted"


def _program_outcome(tmp_path: Path, modules: dict[str, str]) -> Outcome:
    """A phase-accurate verdict: which call raised, not which class.

    Reuses scope's own resolution for the type-check call, so nothing here
    re-resolves the program: ``check_program`` never raises ``AglScopeError``
    itself (only ``AglTypeError``, on the first static type violation).
    """
    from agm.agl.typecheck import AglTypeError
    from agm.agl.typecheck.program import check_program
    from tests.agl.ir_harness import base_caps, make_graph_from_files

    graph = make_graph_from_files(tmp_path, modules)
    try:
        resolved = resolve_program(graph)
    except AglScopeError:
        return "scope"
    try:
        check_program(resolved, base_caps())
    except AglTypeError:
        return "typecheck"
    return "accepted"


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


def test_type_name_and_module_route_clash_stays_rejected_in_both_positions(
    tmp_path: Path,
) -> None:
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

    assert _program_outcome(tmp_path / "expression", expression) == "scope"
    assert _program_outcome(tmp_path / "pattern", pattern) == "scope"


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
    ),
    ids=("annotation", "value"),
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


@pytest.mark.parametrize(
    ("entry", "outcome"),
    (
        (
            "import types\ndef f[types](x: int) = fn(y: types::Color) => 1",
            "scope",
        ),
        (
            "import types\ndef S::f[types](x: int) = fn(y: types::Color) => 1",
            "scope",
        ),
        (
            "import types\n\nscope S\n  def f[types](x: int) = fn(y: types::Color) => 1\nend S",
            "scope",
        ),
        (
            "import types\ndef f[types](x: int) -> int = types::Color::Red",
            "scope",
        ),
        (
            "import lib\ndef f(x: lib::SlotA[int]) -> int = "
            "case x of | lib::SlotA[text]::FilledA(value) => 1 | _ => 0",
            "typecheck",
        ),
        (
            "import lib\nprogram def main(x: lib::SlotA[int]) -> unit = "
            "print(case x of | lib::SlotA[text]::FilledA(value) => 1 | _ => 0)",
            "typecheck",
        ),
    ),
    ids=(
        "plain-def",
        "scoped-shorthand-def",
        "scope-region-def",
        "value-position-return",
        "plain-def-case",
        "program-def-case",
    ),
)
def test_a_one_liner_def_body_still_validates_its_qualifier_chains(
    tmp_path: Path, entry: str, outcome: Outcome
) -> None:
    """A ``def`` whose body is a bare expression, never a ``Block``, still
    validates every qualifier chain reachable only through that body: a
    lambda parameter's type-parameter shadowing, a bare value chain, and an
    ill-typed constructor pattern -- across a plain, scoped-shorthand,
    scope-region, and ``program`` ``def``.
    """
    modules = {
        "entry": entry,
        "types": "enum Color\n  | Red\n  | Green",
        "lib": "enum SlotA[T]\n  | FilledA(value: T)\n  | EmptyA",
    }

    assert _program_outcome(tmp_path, modules) == outcome


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
        "pal": "enum Color\n  | Red",
    }

    assert _program_outcome(tmp_path, modules) == "accepted"


def test_ambiguous_imported_owner_is_rejected_by_scope(tmp_path: Path) -> None:
    modules = {
        "entry": "import one/types\nimport two/types\nlet value = types::Color::Red\nvalue",
        "one/types": "enum Color\n  | Red",
        "two/types": "enum Color\n  | Red",
    }

    assert _program_outcome(tmp_path, modules) == "scope"


def test_import_hiding_removes_a_qualified_owner_at_the_route_seam() -> None:
    env = _import_env("Pal", ("public", ("Secret", "hidden")), hidden=(_item("Secret"),))

    assert isinstance(
        resolve_qualified(env, ("Pal",), ("Secret", "hidden")), QualResolutionMissingMember
    )


def test_ambiguous_qualified_owner_is_rejected_by_scope_in_an_is_test(
    tmp_path: Path,
) -> None:
    modules = {
        "entry": (
            "import one/types\n"
            "import two/types\n"
            "enum Local\n"
            "  | ok\n"
            "let value: Local = Local::ok\n"
            "value is types::Color::Red"
        ),
        "one/types": "enum Color\n  | Red",
        "two/types": "enum Color\n  | Red",
    }

    assert _program_outcome(tmp_path, modules) == "scope"


def test_bare_owner_ambiguity_resolves_by_full_member_path_across_positions(
    tmp_path: Path,
) -> None:
    """A bare owner ambiguous alone still resolves once its member path is unique.

    Both ``m`` and ``n`` wildcard-import a ``Color`` enum; both declare
    ``Red``, only ``m`` declares ``Green``. ``Color`` alone stays ambiguous,
    but ``Color::Red`` is rejected everywhere it is written while
    ``Color::Green`` is accepted everywhere, because scope resolves the full
    ``owner::member`` spelling rather than the bare owner name in isolation.
    """
    modules = {
        "m": "enum Color\n  | Red\n  | Green",
        "n": "enum Color\n  | Red\n  | Blue",
    }

    def program(entry: str) -> dict[str, str]:
        return {"entry": f"import m::*\nimport n::*\n{entry}", **modules}

    assert _program_outcome(tmp_path / "value-ambiguous", program("Color::Red")) == "scope"
    assert _program_outcome(tmp_path / "value-unique", program("Color::Green")) == "accepted"
    assert (
        _program_outcome(tmp_path / "annotation-ambiguous", program("fn(x: Color::Red) => 1"))
        == "scope"
    )
    assert (
        _program_outcome(tmp_path / "annotation-unique", program("fn(x: Color::Green) => 1"))
        == "accepted"
    )
    assert (
        _program_outcome(tmp_path / "alias-ambiguous", program("type CC = Color::Red")) == "scope"
    )
    assert (
        _program_outcome(tmp_path / "alias-unique", program("type CC = Color::Green")) == "accepted"
    )
    assert (
        _program_outcome(
            tmp_path / "is-ambiguous", program("let v: m::Color = Color::Green\nv is Color::Red")
        )
        == "scope"
    )
    assert (
        _program_outcome(
            tmp_path / "is-unique", program("let v: m::Color = Color::Green\nv is Color::Green")
        )
        == "accepted"
    )
    assert (
        _program_outcome(
            tmp_path / "pattern-ambiguous",
            program("let v: m::Color = Color::Green\ncase v of | Color::Red => 1 | _ => 2"),
        )
        == "scope"
    )
    assert (
        _program_outcome(
            tmp_path / "pattern-unique",
            program("let v: m::Color = Color::Green\ncase v of | Color::Green => 1 | _ => 2"),
        )
        == "accepted"
    )


def test_applied_generic_owner_ambiguity_resolves_by_full_member_path(tmp_path: Path) -> None:
    """An applied generic owner (``Box[int]::Member``) resolves like any other owner spelling.

    Both ``gm`` and ``gn`` wildcard-import a generic ``Box`` enum; both
    declare ``Full``, only ``gm`` declares ``Empty``. ``Box[int]::Full`` stays
    ambiguous, while ``Box[int]::Empty`` resolves through ``gm`` alone in both
    a type annotation and a value position.
    """
    modules = {
        "gm": "enum Box[T]\n  | Full(value: T)\n  | Empty",
        "gn": "enum Box[T]\n  | Full(value: T)\n  | Other",
    }

    def program(entry: str) -> dict[str, str]:
        return {"entry": f"import gm::*\nimport gn::*\n{entry}", **modules}

    assert (
        _program_outcome(tmp_path / "annotation-ambiguous", program("fn(x: Box[int]::Full) => 1"))
        == "scope"
    )
    assert (
        _program_outcome(tmp_path / "annotation-unique", program("fn(x: Box[int]::Empty) => 1"))
        == "accepted"
    )
    assert _program_outcome(tmp_path / "value-ambiguous", program("Box[int]::Full(1)")) == "scope"
    assert _program_outcome(tmp_path / "value-unique", program("Box[int]::Empty")) == "accepted"


def _ambiguity_span(tmp_path: Path, modules: dict[str, str]) -> SourceSpan:
    from agm.agl.typecheck.program import check_program
    from tests.agl.ir_harness import base_caps, make_graph_from_files

    graph = make_graph_from_files(tmp_path, modules)
    with pytest.raises(AmbiguousQualificationError) as excinfo:
        check_program(resolve_program(graph), base_caps())
    span = excinfo.value.span
    assert span is not None
    return span


@pytest.mark.parametrize(
    ("position", "entry"),
    (
        ("annotation", "fn(x: Color::Red) => 1"),
        ("is", "let v: m::Color = Color::Green\nv is Color::Red"),
        ("pattern", "let v: m::Color = Color::Green\ncase v of | Color::Red => 1 | _ => 2"),
    ),
)
def test_ambiguous_qualifier_span_is_the_chain_alone(
    tmp_path: Path, position: str, entry: str
) -> None:
    """An ambiguity's span is the qualified chain itself, never the whole
    annotation, ``is`` expression, or pattern it appears in."""
    modules = {
        "m": "enum Color\n  | Red\n  | Green",
        "n": "enum Color\n  | Red\n  | Blue",
    }
    span = _ambiguity_span(
        tmp_path / position, {"entry": f"import m::*\nimport n::*\n{entry}", **modules}
    )
    line = entry.count("\n") + 3
    start_col = entry.splitlines()[-1].index("Color::Red") + 1
    assert (span.start_line, span.start_col, span.end_col) == (line, start_col, start_col + 10)
